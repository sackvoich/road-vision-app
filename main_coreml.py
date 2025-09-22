# main.py
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

# --- 1. Настройка ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Core ML Video Processor")

# CORS (на всякий случай)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 2. Загрузка моделей Core ML ---
try:
    logger.info("Loading Core ML models...")
    # Модель знаков с NMS
    detection_model = ct.models.MLModel('./models/traffic_signs_detection_model.mlpackage', compute_units=ct.ComputeUnit.ALL)
    # Модель сегментации без NMS
    segmentation_model = ct.models.MLModel('./models/zebra_segmentation_model.mlpackage', compute_units=ct.ComputeUnit.ALL)
    logger.info("Core ML models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Core ML models: {e}")
    detection_model = None
    segmentation_model = None

# --- 3. Глобальные переменные и менеджер WebSocket ---
last_processed_time = 0
MIN_INTERVAL = 0.1  # 10 FPS

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

# --- 4. Функции предобработки и парсинга ---

def preprocess_image_for_coreml(img: Image.Image, target_size=(640, 640)) -> tuple:
    """Изменяет размер изображения с сохранением пропорций и добавляет поля (letterboxing)."""
    original_w, original_h = img.size
    target_w, target_h = target_size
    ratio = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * ratio), int(original_h * ratio)
    
    resized_img = img.resize((new_w, new_h), Image.LANCZOS)
    padded_img = Image.new("RGB", target_size, (114, 114, 114))
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    
    padded_img.paste(resized_img, (pad_x, pad_y))
    return padded_img, ratio, (pad_x, pad_y)

def parse_detection_with_nms(results: dict) -> list:
    """Парсит вывод модели детекции со встроенным NMS."""
    detections = []
    boxes = results.get('coordinates')
    confidences = results.get('confidence')

    if boxes is None or confidences is None:
        logger.warning("Detection model output is missing 'coordinates' or 'confidence'.")
        return []

    for i, box in enumerate(boxes):
        class_id = np.argmax(confidences[i])
        confidence = confidences[i][class_id]
        
        # Координаты от модели: [center_x, center_y, width, height]
        cx, cy, w, h = box
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = x1 + w, y1 + h

        detections.append({
            'class': f"Sign_{class_id}",
            'confidence': float(confidence),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
    return detections

def parse_raw_segmentation(results: dict, conf_threshold=0.3, nms_threshold=0.45) -> list:
    """Парсит сырой вывод модели сегментации (без NMS)."""
    raw_predictions = results.get('var_1368')
    proto_masks = results.get('p')

    if raw_predictions is None or proto_masks is None:
        logger.warning("Segmentation model output is missing raw tensors.")
        return []

    raw_predictions, proto_masks = raw_predictions[0].T, proto_masks[0]
    boxes, scores, class_ids, mask_coeffs = [], [], [], []
    
    for row in raw_predictions:
        cx, cy, w, h, score, mask_coeff = row[0], row[1], row[2], row[3], row[4], row[5:]
        if score < conf_threshold:
            continue
        
        boxes.append([int(cx - w/2), int(cy - h/2), int(w), int(h)])
        scores.append(float(score))
        class_ids.append(0)
        mask_coeffs.append(mask_coeff)

    if not boxes: return []
        
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    final_segmentations = []
    mask_h, mask_w = proto_masks.shape[1:]
    
    for i in indices:
        coeff = mask_coeffs[i]
        final_mask = np.zeros((mask_h, mask_w), dtype=np.float32)
        for j in range(coeff.shape[0]):
             final_mask += coeff[j] * proto_masks[j]
        
        final_mask = 1 / (1 + np.exp(-final_mask))
        x1, y1, w, h = boxes[i]
        resized_mask = cv2.resize(final_mask, (w, h))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: continue
        main_contour = max(contours, key=cv2.contourArea).squeeze(1)
        main_contour = main_contour + np.array([x1, y1])
        
        final_segmentations.append({
            'class': 'Zebra',
            'confidence': scores[i],
            'points': main_contour.tolist()
        })
        
    return final_segmentations

# --- 5. Главная функция обработки кадра ---
async def process_frame_parallel_coreml(frame_data: str) -> dict:
    try:
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_cv2 is None: return {"error": "Failed to decode image"}
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        padded_image, scale, (pad_x, pad_y) = preprocess_image_for_coreml(frame_pil)

        def predict_detection():
            return detection_model.predict({'image': padded_image, 'confidenceThreshold': 0.25})
        def predict_segmentation():
            return segmentation_model.predict({'image': padded_image})
        
        det_results_task = asyncio.to_thread(predict_detection)
        seg_results_task = asyncio.to_thread(predict_segmentation)
        raw_det_results, raw_seg_results = await asyncio.gather(det_results_task, seg_results_task)

        det_parse_task = asyncio.to_thread(parse_detection_with_nms, raw_det_results)
        seg_parse_task = asyncio.to_thread(parse_raw_segmentation, raw_seg_results)
        detection_results, segmentation_results = await asyncio.gather(det_parse_task, seg_parse_task)
        
        for res in detection_results + segmentation_results:
            if 'bbox' in res:
                box = res['bbox']
                x1, y1, x2, y2 = box[0]-pad_x, box[1]-pad_y, box[2]-pad_x, box[3]-pad_y
                res['bbox'] = [x1/scale, y1/scale, x2/scale, y2/scale]
            if 'points' in res:
                points = np.array(res['points']) - np.array([pad_x, pad_y])
                res['points'] = (points / scale).tolist()

        return {
            'detections': detection_results,
            'segmentations': segmentation_results,
            'timestamp': asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Error in Core ML processing: {e}", exc_info=True)
        return {'detections': [], 'segmentations': [], 'error': str(e)}

# --- 6. WebSocket Endpoint ---
@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global last_processed_time
    try:
        while True:
            data = await websocket.receive_text()
            current_time = asyncio.get_event_loop().time()
            if current_time - last_processed_time < MIN_INTERVAL:
                continue
            last_processed_time = current_time
            
            result = await process_frame_parallel_coreml(data)
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# --- 7. Статический HTML для фронтенда ---
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Stream Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .video-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; justify-content: center; }
            video, canvas { width: 100%; max-width: 640px; height: auto; border: 2px solid #ccc; border-radius: 4px; }
            .video-box { flex: 1; min-width: 320px; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; background-color: #007bff; color: white; }
            button:disabled { background-color: #cccccc; }
            .controls { margin-bottom: 20px; text-align: center; }
            .results { background: #2b2b2b; color: #f1f1f1; padding: 15px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-family: "Courier New", Courier, monospace;}
            h1, h3 { text-align: center; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📹 Multi-Model Core ML Processor</h1>
            <div class="controls">
                <button id="startBtn">▶️ Start Streaming</button>
                <button id="stopBtn" disabled>⏹️ Stop Streaming</button>
                <div id="status" style="margin-top: 10px;">Status: Ready</div>
            </div>
            <div class="video-container">
                <div class="video-box">
                    <h3>Live Camera</h3>
                    <video id="video" autoplay muted playsinline></video>
                </div>
                <div class="video-box">
                    <h3>Processed</h3>
                    <canvas id="canvas"></canvas>
                </div>
            </div>
            <div class="results">
                <h3>Detection Results:</h3>
                <pre id="results"></pre>
            </div>
        </div>
        <script src="/static/app_coreml.js"></script>
    </body>
    </html>
    """)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 8. Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    if detection_model is None or segmentation_model is None:
        logger.error("Could not start server because one or more models failed to load.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")