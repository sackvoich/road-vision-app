from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import asyncio
import logging
from PIL import Image
import coremltools as ct
from typing import List, Dict, Tuple, Optional

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Core ML Video Processor")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Core ML Models ---
detection_model = None
segmentation_model = None

try:
    logger.info("Loading Core ML models...")
    detection_model = ct.models.MLModel(
        './models/traffic_signs_detection_model.mlpackage',
        compute_units=ct.ComputeUnit.ALL
    )
    segmentation_model = ct.models.MLModel(
        './models/zebra_segmentation_model.mlpackage',
        compute_units=ct.ComputeUnit.ALL
    )
    logger.info("Core ML models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Core ML models: {e}")

# --- Global Variables ---
MIN_FRAME_INTERVAL = 0.1  # 10 FPS max
last_processed_time = 0

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_json(self, data: dict, websocket: WebSocket):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending data: {e}")

manager = ConnectionManager()

# --- Image Preprocessing ---
def letterbox_image(image: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> Tuple[Image.Image, float, Tuple[int, int], Tuple[int, int]]:
    """
    Resize image with padding (letterboxing) to maintain aspect ratio.
    Returns: (padded_image, scale_factor, padding, original_size)
    """
    original_w, original_h = image.size
    target_w, target_h = target_size
    
    # Calculate scale to fit image within target size
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # Resize image
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Create padded image
    padded = Image.new("RGB", target_size, (114, 114, 114))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))
    
    return padded, scale, (pad_x, pad_y), (original_w, original_h)

# --- Detection Model Processing ---
def process_detection_output(output: dict, scale: float, padding: Tuple[int, int], 
                           original_size: Tuple[int, int]) -> List[Dict]:
    """
    Process detection model output with NMS already applied.
    Coordinates are relative (0-1 range) in format [x_center, y_center, width, height]
    """
    detections = []
    
    coordinates = output.get('coordinates')
    confidences = output.get('confidence')
    
    if coordinates is None or confidences is None:
        logger.warning("Missing detection outputs")
        return []
    
    # Get dimensions
    num_boxes = coordinates.shape[0]
    model_size = 640  # Model input size
    pad_x, pad_y = padding
    orig_w, orig_h = original_size
    
    for i in range(num_boxes):
        # Get confidence scores for all classes
        class_confidences = confidences[i]
        class_id = np.argmax(class_confidences)
        confidence = float(class_confidences[class_id])
        
        # Skip low confidence detections
        if confidence < 0.25:
            continue
        
        # Get box coordinates (relative to model input size)
        cx_rel, cy_rel, w_rel, h_rel = coordinates[i]
        
        # Convert relative coordinates to pixel coordinates in model space
        cx = cx_rel * model_size
        cy = cy_rel * model_size
        w = w_rel * model_size
        h = h_rel * model_size
        
        # Convert center coordinates to corner coordinates
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h
        
        # Remove padding offset
        x1 -= pad_x
        y1 -= pad_y
        x2 -= pad_x
        y2 -= pad_y
        
        # Scale back to original image dimensions
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        # Clip to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        detections.append({
            'class': f'Sign_{class_id}',
            'confidence': confidence,
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
    
    return detections

# --- Segmentation Model Processing ---
def process_segmentation_output(output: dict, scale: float, padding: Tuple[int, int],
                              original_size: Tuple[int, int], conf_threshold: float = 0.3,
                              nms_threshold: float = 0.45) -> List[Dict]:
    """
    Process raw segmentation output and apply NMS.
    """
    segmentations = []
    
    raw_predictions = output.get('var_1368')
    proto_masks = output.get('p')
    
    if raw_predictions is None or proto_masks is None:
        logger.warning("Missing segmentation outputs")
        return []
    
    # Shape: [1, 37, 8400] -> [8400, 37]
    predictions = raw_predictions[0].T
    masks = proto_masks[0]  # [32, 160, 160]
    
    model_size = 640
    pad_x, pad_y = padding
    orig_w, orig_h = original_size
    
    # Extract boxes and masks
    boxes = []
    scores = []
    mask_coeffs = []
    
    for pred in predictions:
        # First 4 values are box coordinates
        cx, cy, w, h = pred[:4]
        # 5th value is confidence
        score = pred[4]
        # Remaining 32 values are mask coefficients
        mask_coeff = pred[5:37]
        
        if score < conf_threshold:
            continue
        
        # Convert to corner format for NMS
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        boxes.append([x1, y1, int(w), int(h)])
        scores.append(float(score))
        mask_coeffs.append(mask_coeff)
    
    if not boxes:
        return []
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []
    
    # Process selected detections
    indices = indices.flatten() if hasattr(indices, 'flatten') else indices
    
    for idx in indices:
        # Get mask coefficients
        coeff = mask_coeffs[idx]
        
        # Generate mask from prototypes
        mask = np.zeros((160, 160), dtype=np.float32)
        for j in range(32):
            mask += coeff[j] * masks[j]
        
        # Apply sigmoid
        mask = 1 / (1 + np.exp(-mask))
        
        # Get box in model coordinates
        x1, y1, w, h = boxes[idx]
        cx = x1 + w/2
        cy = y1 + h/2
        
        # Scale mask to box size
        if w > 0 and h > 0:
            resized_mask = cv2.resize(mask, (w, h))
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                main_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(main_contour, True)
                simplified = cv2.approxPolyDP(main_contour, epsilon, True)
                
                # Transform points to original image coordinates
                points = []
                for point in simplified:
                    px = point[0][0] + x1  # Add box offset
                    py = point[0][1] + y1
                    
                    # Remove padding
                    px = (px - pad_x) / scale
                    py = (py - pad_y) / scale
                    
                    # Clip to bounds
                    px = max(0, min(px, orig_w))
                    py = max(0, min(py, orig_h))
                    
                    points.append([float(px), float(py)])
                
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    segmentations.append({
                        'class': 'Zebra',
                        'confidence': scores[idx],
                        'points': points
                    })
    
    return segmentations

# --- Main Frame Processing ---
async def process_frame(frame_data: str) -> dict:
    """Process a single frame through both models."""
    try:
        # Decode image
        image_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_cv2 is None:
            return {"error": "Failed to decode image"}
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Get original dimensions
        original_size = frame_pil.size
        
        # Preprocess image
        padded_image, scale, padding, _ = letterbox_image(frame_pil)
        
        # Run models in parallel
        async def run_detection():
            return await asyncio.to_thread(
                detection_model.predict,
                {'image': padded_image, 'confidenceThreshold': 0.25, 'iouThreshold': 0.7}
            )
        
        async def run_segmentation():
            return await asyncio.to_thread(
                segmentation_model.predict,
                {'image': padded_image}
            )
        
        # Execute both models
        detection_task = run_detection()
        segmentation_task = run_segmentation()
        
        det_output, seg_output = await asyncio.gather(detection_task, segmentation_task)
        
        # Process outputs
        detections = process_detection_output(det_output, scale, padding, original_size)
        segmentations = process_segmentation_output(seg_output, scale, padding, original_size)
        
        return {
            'detections': detections,
            'segmentations': segmentations,
            'timestamp': asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)
        return {'error': str(e), 'detections': [], 'segmentations': []}

# --- WebSocket Endpoint ---
@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global last_processed_time
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            
            # Rate limiting
            current_time = asyncio.get_event_loop().time()
            if current_time - last_processed_time < MIN_FRAME_INTERVAL:
                continue
            
            last_processed_time = current_time
            
            # Process frame
            result = await process_frame(data)
            
            # Send results
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# --- Static HTML Frontend ---
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Core ML Video Stream Processor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            box-shadow: none;
        }
        #status {
            text-align: center;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 10px;
            font-weight: 500;
            color: #666;
        }
        .video-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .video-box {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .video-box h3 {
            padding: 15px;
            background: #f8f9fa;
            margin: 0;
            color: #495057;
            font-size: 1.2em;
        }
        video, canvas {
            width: 100%;
            height: auto;
            display: block;
            background: #000;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
        }
        .stat-label {
            color: #868e96;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #495057;
            font-size: 1.8em;
            font-weight: 700;
        }
        .results {
            background: #2d3436;
            color: #dfe6e9;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Monaco', 'Menlo', monospace;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.9em;
        }
        .results h3 {
            color: #74b9ff;
            margin-bottom: 10px;
        }
        pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚦 Core ML Traffic Analysis</h1>
        
        <div class="controls">
            <button id="startBtn">▶️ Start Streaming</button>
            <button id="stopBtn" disabled>⏹️ Stop Streaming</button>
        </div>
        
        <div id="status">Status: Initializing camera...</div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Signs Detected</div>
                <div class="stat-value" id="signCount">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Zebras Detected</div>
                <div class="stat-value" id="zebraCount">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Processing Time</div>
                <div class="stat-value" id="processTime">0ms</div>
            </div>
        </div>
        
        <div class="video-container">
            <div class="video-box">
                <h3>📷 Live Camera Feed</h3>
                <video id="video" autoplay muted playsinline></video>
            </div>
            <div class="video-box">
                <h3>🤖 AI Processed</h3>
                <canvas id="canvas"></canvas>
            </div>
        </div>
        
        <div class="results">
            <h3>📊 Detection Results:</h3>
            <pre id="results">Waiting for data...</pre>
        </div>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>
    """)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    if detection_model is None or segmentation_model is None:
        logger.error("Cannot start server: Models failed to load")
        exit(1)
    
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")