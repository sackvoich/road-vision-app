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

# Модель детекции
model = YOLO('./models/traffic_signs_detection_model.pt')  # Автоматически скачает маленькую модель

# Глобальные переменные для ограничения FPS
last_processed_time = 0
min_interval = 0.2  # 5 FPS максимум (изменяй по необходимости)

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
    """Обработка кадра нейросетью"""
    try:
        # Декодируем base64 в numpy array
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        # Детекция объектов
        results = model(frame, verbose=False)
        
        # Форматирование результатов
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Получаем имя класса
                    class_name = model.names[cls] if cls in model.names else str(cls)
                    
                    detections.append({
                        'class': class_name,
                        'class_id': cls,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return {
            'detections': detections,
            'timestamp': asyncio.get_event_loop().time(),
            'objects_count': len(detections)
        }
    
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return {'detections': [], 'error': str(e)}

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
                # Слишком рано, пропускаем кадр
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