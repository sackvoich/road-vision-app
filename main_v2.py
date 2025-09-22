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
from concurrent.futures import ThreadPoolExecutor
import time

app = FastAPI(title="Advanced Video Processor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models
try:
    traffic_model = YOLO('./models/traffic_signs_detection_model.pt')
    zebra_model = YOLO('./models/zebra_segmentation_model.pt')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # Fallback to standard models if custom models fail
    traffic_model = YOLO('yolov8n.pt')
    zebra_model = YOLO('yolov8n.pt')

# Thread pool for running models
executor = ThreadPoolExecutor(max_workers=4)

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, data: dict, websocket: WebSocket):
        try:
            await websocket.send_json(data)
        except Exception as e:
            print(f"Error sending data: {e}")

manager = ConnectionManager()

def process_frame_with_model(model, frame, model_name):
    """Process frame with a specific model"""
    try:
        results = model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_name = model.names[cls] if cls in model.names else str(cls)
                    
                    detections.append({
                        'class': class_name,
                        'class_id': cls,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'model': model_name
                    })
        
        return detections
    except Exception as e:
        logging.error(f"Error processing frame with {model_name}: {e}")
        return []

def process_frame(frame_data: str, enabled_models: dict, scale_factor: float = 1.0) -> dict:
    """Process frame with selected models"""
    try:
        # Decode base64 to numpy array
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        original_height, original_width = frame.shape[:2]
        
        # Apply scaling if needed
        if scale_factor != 1.0:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Process with enabled models
        all_detections = []
        
        if enabled_models.get('traffic', False):
            traffic_detections = process_frame_with_model(traffic_model, frame, 'traffic')
            all_detections.extend(traffic_detections)
        
        if enabled_models.get('zebra', False):
            zebra_detections = process_frame_with_model(zebra_model, frame, 'zebra')
            all_detections.extend(zebra_detections)
        
        # Scale bounding boxes back to original size if scaling was applied
        if scale_factor != 1.0:
            for detection in all_detections:
                detection['bbox'] = [
                    detection['bbox'][0] / scale_factor,
                    detection['bbox'][1] / scale_factor,
                    detection['bbox'][2] / scale_factor,
                    detection['bbox'][3] / scale_factor
                ]
        
        return {
            'detections': all_detections,
            'timestamp': time.time(),
            'objects_count': len(all_detections),
            'scale_factor': scale_factor,
            'original_size': [original_width, original_height]
        }
    
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return {'detections': [], 'error': str(e)}

async def process_frame_async(frame_data: str, enabled_models: dict, scale_factor: float):
    """Async wrapper for frame processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        process_frame, 
        frame_data, 
        enabled_models, 
        scale_factor
    )

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue
            
            # Extract frame and settings
            frame_data = request.get('frame', '')
            settings = request.get('settings', {})
            
            if not frame_data:
                await websocket.send_json({"error": "No frame data"})
                continue
            
            # Get settings
            enabled_models = settings.get('models', {'traffic': True, 'zebra': False})
            scale_factor = settings.get('scale_factor', 1.0)
            
            # Validate scale factor
            scale_factor = max(0.1, min(1.0, scale_factor))  # Clamp between 0.1 and 1.0
            
            # Process frame asynchronously
            result = await process_frame_async(frame_data, enabled_models, scale_factor)
            
            # Send results back
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Static file for frontend
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Video Stream Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: #333;
            }
            .controls-section {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .model-controls, .settings-controls {
                flex: 1;
                min-width: 300px;
            }
            .control-group {
                margin-bottom: 15px;
            }
            .control-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #555;
            }
            .checkbox-group {
                display: flex;
                gap: 15px;
            }
            .checkbox-group label {
                display: flex;
                align-items: center;
                gap: 5px;
                font-weight: normal;
            }
            input[type="range"] {
                width: 100%;
            }
            .value-display {
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }
            .buttons {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s;
            }
            #startBtn {
                background: #28a745;
                color: white;
            }
            #stopBtn {
                background: #dc3545;
                color: white;
            }
            button:hover {
                opacity: 0.9;
                transform: translateY(-2px);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .video-container {
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
                justify-content: center;
                margin-bottom: 30px;
            }
            .video-box {
                text-align: center;
            }
            .video-box h3 {
                margin-bottom: 10px;
                color: #333;
            }
            video, canvas {
                width: 400px;
                height: 300px;
                border: 2px solid #ddd;
                border-radius: 8px;
                background: #000;
            }
            .status-bar {
                text-align: center;
                padding: 10px;
                margin: 20px 0;
                border-radius: 6px;
                font-weight: 600;
            }
            .status-ready {
                background: #d4edda;
                color: #155724;
            }
            .status-streaming {
                background: #cce5ff;
                color: #004085;
            }
            .status-error {
                background: #f8d7da;
                color: #721c24;
            }
            .results {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .results h3 {
                margin-top: 0;
                color: #333;
            }
            #results {
                background: #fff;
                padding: 15px;
                border-radius: 6px;
                max-height: 200px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 14px;
            }
            .detection-count {
                font-weight: bold;
                font-size: 18px;
                color: #007bff;
                text-align: center;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📹 Advanced Video Stream Processor</h1>
                <p>Real-time detection of traffic signs and zebra crossings</p>
            </div>
            
            <div class="controls-section">
                <div class="model-controls">
                    <h3>Model Selection</h3>
                    <div class="control-group">
                        <div class="checkbox-group">
                            <label>
                                <input type="checkbox" id="trafficModel" checked>
                                Traffic Signs Detection
                            </label>
                            <label>
                                <input type="checkbox" id="zebraModel">
                                Zebra Crossing Detection
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="settings-controls">
                    <h3>Processing Settings</h3>
                    <div class="control-group">
                        <label for="fpsLimit">FPS Limit: <span id="fpsValue">5</span></label>
                        <input type="range" id="fpsLimit" min="1" max="30" value="5">
                    </div>
                    
                    <div class="control-group">
                        <label for="scaleFactor">Image Scale: <span id="scaleValue">100</span>%</label>
                        <input type="range" id="scaleFactor" min="10" max="100" value="100">
                    </div>
                </div>
            </div>
            
            <div class="buttons">
                <button id="startBtn">▶️ Start Streaming</button>
                <button id="stopBtn" disabled>⏹️ Stop Streaming</button>
            </div>
            
            <div id="status" class="status-bar status-ready">
                Status: Ready
            </div>
            
            <div class="detection-count">
                Objects Detected: <span id="objectCount">0</span>
            </div>
            
            <div class="video-container">
                <div class="video-box">
                    <h3>Live Camera</h3>
                    <video id="video" autoplay muted playsinline></video>
                </div>
                <div class="video-box">
                    <h3>Processed Output</h3>
                    <canvas id="canvas"></canvas>
                </div>
            </div>

            <div class="results">
                <h3>Detection Results:</h3>
                <pre id="results"></pre>
            </div>
        </div>

        <script src="/static/app_v2.js"></script>
    </body>
    </html>
    """)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")