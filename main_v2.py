from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import time
import json
from ultralytics import YOLO
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Video Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка моделей с fallback
def load_model(model_path, fallback_model='yolov8n.pt'):
    try:
        model = YOLO(model_path)
        logger.info(f"✅ Model loaded: {model_path}")
        logger.info(f"Classes: {model.names}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load {model_path}: {e}")
        logger.info(f"Using fallback: {fallback_model}")
        return YOLO(fallback_model)

# Загружаем модели
traffic_model = load_model('./models/traffic_signs_detection_model.pt')
zebra_model = load_model('./models/zebra_segmentation_model.pt')

# Глобальные переменные для ограничения FPS
last_processed_time = 0

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, data: dict, websocket: WebSocket):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending data: {e}")

manager = ConnectionManager()

def process_frame_simple(frame_data: str, enabled_models: dict, scale_factor: float = 1.0) -> dict:
    """Простая обработка кадра"""
    try:
        # Декодируем base64
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        logger.info(f"Frame shape: {frame.shape}, scale: {scale_factor}")
        
        # Применяем scaling
        if scale_factor != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            frame = cv2.resize(frame, (new_w, new_h))
            logger.info(f"Resized to: {frame.shape}")

        # Обработка выбранными моделями
        all_detections = []
        
        # Traffic signs model
        if enabled_models.get('traffic', False):
            try:
                results = traffic_model(frame, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            class_name = traffic_model.names.get(cls, str(cls))
                            
                            # Масштабируем bbox обратно к оригинальному размеру
                            if scale_factor != 1.0:
                                x1, y1, x2, y2 = [coord / scale_factor for coord in [x1, y1, x2, y2]]
                            
                            all_detections.append({
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'model': 'traffic'
                            })
                logger.info(f"Traffic model found {len([d for d in all_detections if d['model'] == 'traffic'])} objects")
            except Exception as e:
                logger.error(f"Error in traffic model: {e}")

        # Zebra crossing model
        if enabled_models.get('zebra', False):
            try:
                results = zebra_model(frame, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            class_name = zebra_model.names.get(cls, str(cls))
                            
                            if scale_factor != 1.0:
                                x1, y1, x2, y2 = [coord / scale_factor for coord in [x1, y1, x2, y2]]
                            
                            all_detections.append({
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'model': 'zebra'
                            })
                logger.info(f"Zebra model found {len([d for d in all_detections if d['model'] == 'zebra'])} objects")
            except Exception as e:
                logger.error(f"Error in zebra model: {e}")

        return {
            'detections': all_detections,
            'objects_count': len(all_detections),
            'timestamp': time.time(),
            'scale_factor': scale_factor,
            'frame_size': frame.shape[:2] if scale_factor == 1.0 else [int(frame.shape[1] / scale_factor), int(frame.shape[0] / scale_factor)]
        }
        
    except Exception as e:
        logger.error(f"Error in process_frame_simple: {e}")
        return {'detections': [], 'error': str(e)}

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global last_processed_time
    
    try:
        while True:
            # Ждем данные от клиента
            data = await websocket.receive_text()
            
            # Парсим JSON
            try:
                request = json.loads(data)
                frame_data = request['frame']
                settings = request.get('settings', {})
            except Exception as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_json({'error': 'Invalid JSON format'})
                continue

            # Получаем настройки
            enabled_models = settings.get('models', {'traffic': True, 'zebra': False})
            fps_limit = max(1, min(30, settings.get('fps_limit', 5)))  # Ограничиваем 1-30 FPS
            scale_factor = max(0.1, min(1.0, settings.get('scale_factor', 1.0)))  # Ограничиваем 0.1-1.0

            # Ограничение FPS
            current_time = time.time()
            min_interval = 1.0 / fps_limit
            
            if current_time - last_processed_time < min_interval:
                await websocket.send_json({
                    'status': 'skipped', 
                    'message': f'FPS limit: {fps_limit}'
                })
                continue
                
            last_processed_time = current_time

            # Обрабатываем кадр
            result = process_frame_simple(frame_data, enabled_models, scale_factor)
            
            # Отправляем результат
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Video Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                background: #f0f2f5; 
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                opacity: 0.9;
                font-size: 1.1em;
            }
            .controls {
                padding: 25px;
                background: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
            }
            .control-group {
                margin-bottom: 20px;
            }
            .control-group h3 {
                color: #495057;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            .model-selector {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
            .model-option {
                flex: 1;
                min-width: 200px;
            }
            .model-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                border: 2px solid #e9ecef;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
            }
            .model-card:hover {
                border-color: #667eea;
                transform: translateY(-2px);
            }
            .model-card.active {
                border-color: #667eea;
                background: #f0f4ff;
            }
            .model-card h4 {
                color: #495057;
                margin-bottom: 10px;
            }
            .model-card .status {
                font-size: 0.9em;
                padding: 5px 10px;
                border-radius: 15px;
                background: #28a745;
                color: white;
                display: inline-block;
            }
            .slider-group {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 15px;
            }
            .slider-group label {
                display: block;
                margin-bottom: 10px;
                color: #495057;
                font-weight: 600;
            }
            input[type="range"] {
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: #e9ecef;
                outline: none;
            }
            .value-display {
                text-align: center;
                font-size: 1.2em;
                color: #667eea;
                font-weight: bold;
                margin-top: 10px;
            }
            .buttons {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 20px;
            }
            button {
                padding: 15px 30px;
                font-size: 1.1em;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
            }
            #startBtn {
                background: #28a745;
                color: white;
            }
            #startBtn:hover:not(:disabled) {
                background: #218838;
                transform: translateY(-2px);
            }
            #stopBtn {
                background: #dc3545;
                color: white;
            }
            #stopBtn:hover:not(:disabled) {
                background: #c82333;
                transform: translateY(-2px);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none !important;
            }
            .video-container {
                display: flex;
                gap: 30px;
                padding: 30px;
                flex-wrap: wrap;
                justify-content: center;
            }
            .video-box {
                flex: 1;
                min-width: 300px;
                text-align: center;
            }
            .video-box h3 {
                color: #495057;
                margin-bottom: 15px;
                font-size: 1.3em;
            }
            video, canvas {
                width: 100%;
                max-width: 500px;
                height: 300px;
                border: 3px solid #e9ecef;
                border-radius: 10px;
                background: #000;
            }
            .status-bar {
                padding: 15px;
                text-align: center;
                font-weight: 600;
                font-size: 1.1em;
                border-radius: 8px;
                margin: 20px;
            }
            .status-ready { background: #d4edda; color: #155724; }
            .status-streaming { background: #cce5ff; color: #004085; }
            .status-error { background: #f8d7da; color: #721c24; }
            .results {
                background: #f8f9fa;
                padding: 25px;
                margin: 20px;
                border-radius: 10px;
            }
            .results h3 {
                color: #495057;
                margin-bottom: 15px;
            }
            #results {
                background: white;
                padding: 20px;
                border-radius: 8px;
                max-height: 250px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                border: 1px solid #e9ecef;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 15px;
                padding: 20px;
                background: #f8f9fa;
                margin: 20px;
                border-radius: 10px;
            }
            .stat-item {
                text-align: center;
                padding: 15px;
                background: white;
                border-radius: 8px;
                flex: 1;
                min-width: 150px;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                display: block;
            }
            .stat-label {
                color: #6c757d;
                font-size: 0.9em;
            }
            @media (max-width: 768px) {
                .video-container { flex-direction: column; }
                .model-selector { flex-direction: column; }
                .container { margin: 10px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Advanced Video Detection Processor</h1>
                <p>Real-time object detection with customizable models and settings</p>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <h3>🔧 Model Selection</h3>
                    <div class="model-selector">
                        <div class="model-option">
                            <div class="model-card active" id="trafficCard">
                                <h4>🚦 Traffic Signs</h4>
                                <div class="status">Active</div>
                            </div>
                        </div>
                        <div class="model-option">
                            <div class="model-card" id="zebraCard">
                                <h4>🦓 Zebra Crossings</h4>
                                <div class="status">Inactive</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="control-group">
                    <h3>⚙️ Processing Settings</h3>
                    <div class="slider-group">
                        <label for="fpsLimit">🎮 FPS Limit: <span id="fpsValue">5</span></label>
                        <input type="range" id="fpsLimit" min="1" max="30" value="5">
                    </div>
                    
                    <div class="slider-group">
                        <label for="scaleFactor">📐 Image Scale: <span id="scaleValue">100</span>%</label>
                        <input type="range" id="scaleFactor" min="10" max="100" value="100">
                    </div>
                </div>
                
                <div class="buttons">
                    <button id="startBtn">▶️ Start Streaming</button>
                    <button id="stopBtn" disabled>⏹️ Stop Streaming</button>
                </div>
            </div>
            
            <div id="status" class="status-bar status-ready">
                Status: Ready to start streaming
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-value" id="objectCount">0</span>
                    <span class="stat-label">Objects Detected</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value" id="fpsDisplay">0</span>
                    <span class="stat-label">Current FPS</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value" id="processingTime">0ms</span>
                    <span class="stat-label">Processing Time</span>
                </div>
            </div>
            
            <div class="video-container">
                <div class="video-box">
                    <h3>📷 Live Camera Feed</h3>
                    <video id="video" autoplay muted playsinline></video>
                </div>
                <div class="video-box">
                    <h3>🔍 Processed Result</h3>
                    <canvas id="canvas"></canvas>
                </div>
            </div>

            <div class="results">
                <h3>📊 Detection Results</h3>
                <pre id="results">Waiting for data...</pre>
            </div>
        </div>

        <script src="/static/app_v2.js"></script>
    </body>
    </html>
    """)

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")