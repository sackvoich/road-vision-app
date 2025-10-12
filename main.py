# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import asyncio
import json
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked
import logging
import torch

# --- Настройка логгирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Инициализация FastAPI ---
app = FastAPI(title="Road Object Processor")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Загрузка моделей ---
# Поместите ваши модели в папку 'models'
# OSX - macos
# WIN - windows
# LIN - linux
TARGET_OS = 'OSX' 

try:
    if TARGET_OS == 'OSX':
        logger.info(f"Target OS is {TARGET_OS}")
        detection_model = YOLO('./models/traffic_signs_detection_model.mlpackage', task='detect')
        segmentation_model = YOLO('./models/zebra_segmentation_model.mlpackage', task='segment')
        logger.info("Models loaded successfully.")
    elif TARGET_OS == 'WIN':
        import torch
        logger.info(f"Target OS is {TARGET_OS}")
        detection_model = YOLO('./models/traffic_signs_detection_model.pt', task='detect')
        detection_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        segmentation_model = YOLO('./models/zebra_segmentation_model.pt', task='segment')
        segmentation_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Models loaded successfully.")
    else:
        logger.info(f"Target OS is {TARGET_OS}")
        logger.error("Current os is not supported")
        raise NotImplementedError
except Exception as e:
    logger.error(f"Error loading models: {e}")
    # Если модели не загрузились, приложение не сможет работать
    detection_model = None
    segmentation_model = None

# --- Глобальные переменные для ограничения FPS ---
last_processed_time = 0
MIN_INTERVAL = 0.2  # 5 FPS максимум

# --- Менеджер WebSocket соединений ---
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("Client disconnected.")

    async def send_json(self, data: dict, websocket: WebSocket):
        await websocket.send_json(data)

manager = ConnectionManager()


# --- Функции обработки для каждой модели (СИНХРОННЫЕ) ---
@ThreadingLocked()
def run_detection(frame: np.ndarray) -> list:
    """Запускает модель детекции и форматирует результат."""
    detections = []
    if detection_model is None:
        return detections
        
    results = detection_model(frame, verbose=False, conf=0.1)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = detection_model.names.get(cls, str(cls))
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
    return detections

@ThreadingLocked()
def run_segmentation(frame: np.ndarray) -> list:
    """Запускает модель сегментации и форматирует результат."""
    segmentations = []
    if segmentation_model is None:
        return segmentations
        
    results = segmentation_model(frame, verbose=False, conf=0.1)
    for result in results:
        masks = result.masks
        if masks is not None and masks.xy is not None:
            for i, mask_points in enumerate(masks.xy):
                # Для сегментации также можно получить класс и уверенность, если они есть
                cls = int(result.boxes[i].cls[0].cpu().numpy())
                conf = float(result.boxes[i].conf[0].cpu().numpy())
                class_name = segmentation_model.names.get(cls, str(cls))
                
                segmentations.append({
                    'class': class_name,
                    'confidence': conf,
                    'points': mask_points.tolist() # [[x1, y1], [x2, y2], ...]
                })
    return segmentations


# --- Главная функция обработки (АСИНХРОННАЯ) ---
async def process_frame_parallel(frame_data: str) -> dict:
    """Декодирует кадр и запускает обе модели в параллельных потоках."""
    try:
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}

        # Запускаем обе синхронные функции параллельно в отдельных потоках
        detection_task = asyncio.to_thread(run_detection, frame)
        segmentation_task = asyncio.to_thread(run_segmentation, frame)

        # Ожидаем завершения обеих задач
        detection_results, segmentation_results = await asyncio.gather(
            detection_task,
            segmentation_task
        )
        
        return {
            'detections': detection_results,
            'segmentations': segmentation_results,
            'timestamp': asyncio.get_event_loop().time()
        }
    
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        return {'detections': [], 'segmentations': [], 'error': str(e)}

# --- Image Processing Endpoint ---
@app.post("/api/image")
async def process_single_image(file: UploadFile = File(...)):
    """Принимает изображение, обрабатывает его и возвращает JSON с результатами."""
    try:
        # Проверяем тип данных
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # Читаем данные
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received.")

        # Кодируем в base64, чтобы переиспользовать существующую функцию
        base64_encoded_data = base64.b64encode(contents).decode('utf-8')
        
        # Используем ту же логику обработки, что и для WebSocket
        result = await process_frame_parallel(base64_encoded_data)
        
        # Убираем timestamp, он не нужен для одиночного изображения
        result.pop('timestamp', None)
        
        return result
    except HTTPException as e:
        # Просто перебрасываем HTTPException, чтобы FastAPI мог его обработать
        raise e
    except Exception as e:
        logger.error(f"Error processing single image: {e}")
        # Для всех остальных ошибок возвращаем 500
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- WebSocket Endpoint ---
@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global last_processed_time
    
    try:
        while True:
            data = await websocket.receive_text()
            
            current_time = asyncio.get_event_loop().time()
            if current_time - last_processed_time < MIN_INTERVAL:
                # Пропускаем кадр для соблюдения FPS
                continue 
            
            last_processed_time = current_time
            
            # Асинхронно обрабатываем кадр обеими моделями
            result = await process_frame_parallel(data)
            
            # Отправляем объединенные результаты
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# --- Статический HTML для фронтенда ---
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Stream Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .video-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; justify-content: center; }
            video, canvas { width: 320px; height: 240px; border: 2px solid #ccc; border-radius: 4px; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; background-color: #007bff; color: white; }
            button:disabled { background-color: #cccccc; }
            .controls { margin-bottom: 20px; text-align: center; }
            .results { background: #f5f5f5; padding: 15px; border-radius: 5px; max-height: 200px; overflow-y: auto; }
            h1, h3 { text-align: center; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📹 Live Stream Processor</h1>
            <p style="text-align: center;">Want to process a single file? <a href="/upload">Go to the Image Upload page</a>.</p>
            
            <div class="controls">
                <button id="startBtn">▶️ Start Streaming</button>
                <button id="stopBtn" disabled>⏹️ Stop Streaming</button>
                <div id="status" style="margin-top: 10px;">Status: Ready</div>
            </div>

            <div class="video-container">
                <div>
                    <h3>Input</h3>
                    <video id="video" autoplay muted playsinline></video>
                </div>
                <div>
                    <h3>Processed Output</h3>
                    <canvas id="canvas" width="640" height="480"></canvas>
                </div>
            </div>

            <div class="results">
                <h3>Detection Results:</h3>
                <pre id="results"></pre>
            </div>
        </div>

        <script src="/static/app.js"></script>
    </body>
    </html>
    """)



@app.get("/upload")
async def get_upload_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .video-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; justify-content: center; }
            canvas { border: 2px solid #ccc; border-radius: 4px; max-width: 100%; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; background-color: #007bff; color: white; }
            button:disabled { background-color: #cccccc; }
            .controls { margin-bottom: 20px; text-align: center; }
            .results { background: #f5f5f5; padding: 15px; border-radius: 5px; max-height: 200px; overflow-y: auto; }
            h1, h3 { text-align: center; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🖼️ Single Image Processor</h1>
            <p style="text-align: center;">Want to process a live video stream? <a href="/">Go to the Live Stream page</a>.</p>

            <div class="controls">
                <input type="file" id="uploadInput" accept="image/*" style="margin-bottom: 10px;"/>
                <br>
                <button id="processBtn" disabled>⚙️ Process Image</button>
                <button id="downloadBtn" disabled>💾 Download Image</button>
                <div id="status" style="margin-top: 10px;">Status: Ready</div>
            </div>

            <div class="video-container">
                <div>
                    <h3>Processed Output</h3>
                    <canvas id="canvas" width="640" height="480"></canvas>
                </div>
            </div>

            <div class="results">
                <h3>Detection Results:</h3>
                <pre id="results"></pre>
            </div>
        </div>

        <script src="/static/upload.js"></script>
    </body>
    </html>
    """)

# Монтируем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    if detection_model is None or segmentation_model is None:
        logger.error("Could not start server because one or more models failed to load.")
    else:
        uvicorn.run(app, host="localhost", port=8000, log_level="info")