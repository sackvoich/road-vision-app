# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import asyncio
import json
from ultralytics import YOLO
import logging

app = FastAPI(title="Simple Video Processor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка моделей
detection_model = YOLO('./models/traffic_signs_detection_model.pt')
segmentation_model = YOLO('./models/zebra_segmentation_model.pt')

# Глобальные переменные для ограничения FPS
last_processed_time = 0
min_interval = 0.2  # 5 FPS максимум

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, data: dict, websocket: WebSocket):
        await websocket.send_json(data)

manager = ConnectionManager()

def process_frame(frame_data: str) -> dict:
    """Обработка кадра двумя нейросетями: детекция знаков и сегментация зебр"""
    try:
        # Декодируем base64 в numpy array
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode image"}

        results = {}

        # --- Детекция дорожных знаков ---
        det_results = detection_model(frame, verbose=False)
        detections = []
        for result in det_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = detection_model.names[cls] if cls in detection_model.names else str(cls)
                    detections.append({
                        'class': class_name,
                        'class_id': cls,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        results['traffic_signs'] = detections

        # --- Сегментация пешеходных переходов (зебр) ---
        seg_results = segmentation_model(frame, verbose=False)
        zebra_crossings = []
        for result in seg_results:
            if result.masks is not None:
                for mask, box, conf, cls in zip(result.masks.data, result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    confidence = conf.cpu().numpy()
                    class_id = int(cls.cpu().numpy())
                    class_name = segmentation_model.names[class_id] if class_id in segmentation_model.names else str(class_id)
                    
                    zebra_crossings.append({
                        'class': class_name,
                        'class_id': class_id,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    })
        results['zebra_crossings'] = zebra_crossings

        # Общая статистика
        results['timestamp'] = asyncio.get_event_loop().time()
        results['total_detections'] = len(detections) + len(zebra_crossings)

        return results

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return {
            'traffic_signs': [],
            'zebra_crossings': [],
            'error': str(e)
        }

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global last_processed_time
    try:
        while True:
            # Получаем кадр от клиента
            data = await websocket.receive_text()
            # Ограничение FPS - пропускаем кадры если слишком часто
            current_time = asyncio.get_event_loop().time()
            if current_time - last_processed_time < min_interval:
                await websocket.send_json({
                    "status": "skipped",
                    "message": "Frame rate limited"
                })
                continue
            last_processed_time = current_time

            # Обработка кадра
            result = process_frame(data)
            # Отправка результатов обратно
            await manager.send_json(result, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Статический файл для фронтенда
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Stream Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .video-container { display: flex; gap: 20px; margin-bottom: 20px; }
            video, canvas { width: 300px; height: 225px; border: 2px solid #ccc; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .controls { margin-bottom: 20px; }
            .results { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📹 Video Stream Processor</h1>
            <div class="controls">
                <button id="startBtn">▶️ Start Streaming</button>
                <button id="stopBtn" disabled>⏹️ Stop Streaming</button>
                <span id="status">Status: Ready</span>
            </div>
            <div class="video-container">
                <div>
                    <h3>Live Camera</h3>
                    <video id="video" autoplay muted playsinline></video>
                </div>
                <div>
                    <h3>Processed</h3>
                    <canvas id="canvas"></canvas>
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

# Монтируем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")