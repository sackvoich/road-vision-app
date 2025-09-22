# main_coreml.py
# Для запуска с поддержкой CoreML требуется macOS и установка зависимостей:
# pip install coremltools
#
# На других платформах будет использоваться имитация работы CoreML моделей

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import asyncio
import json
import logging
from PIL import Image

# Попытка импортировать coremltools, с fallback на имитацию
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    logging.warning("coremltools not available. Using mock implementation.")
    COREML_AVAILABLE = False
    
    # Создание имитации модуля coremltools для разработки вне macOS
    class MockMLModel:
        def __init__(self, path):
            self.path = path
            logging.info(f"Mock loading model from {path}")
        
        def predict(self, input_dict):
            # Возвращаем фиктивные результаты для тестирования
            logging.info("Mock prediction")
            # Фиктивные результаты детекции
            if 'traffic_sign' in self.path:
                return {
                    'confidence': np.array([0.9, 0.7]),
                    'coordinates': np.array([[0.3, 0.4, 0.1, 0.1], [0.7, 0.6, 0.05, 0.05]])
                }
            # Фиктивные результаты сегментации
            else:
                return {
                    'output': np.random.rand(1, 640, 640, 1).astype(np.float32)
                }
        
        def get_spec(self):
            class MockSpec:
                class MockDescription:
                    def __init__(self):
                        self.input = "Mock input description"
                def __init__(self):
                    self.description = self.MockDescription()
            return MockSpec()
    
    class ct:
        class models:
            MLModel = MockMLModel

app = FastAPI(title="CoreML Video Processor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка моделей CoreML
try:
    traffic_signs_model = ct.models.MLModel('./models/traffic_signs_detection_model.mlpackage')
    zebra_model = ct.models.MLModel('./models/zebra_segmentation_model.mlpackage')
    
    # Получение информации о моделях для логирования
    traffic_signs_spec = traffic_signs_model.get_spec()
    zebra_spec = zebra_model.get_spec()
    
    logging.info(f"Traffic signs model loaded. Input description: {traffic_signs_spec.description.input}")
    logging.info(f"Zebra segmentation model loaded. Input description: {zebra_spec.description.input}")
    
except Exception as e:
    logging.error(f"Failed to load CoreML models: {e}")
    raise

# Глобальные переменные для ограничения FPS
last_processed_time = 0
# Минимальный интервал между обработками кадров (в секундах)
# Для 5 FPS установите 0.2, для 10 FPS - 0.1, для 2 FPS - 0.5
min_interval = 0.05  # 5 FPS максимум (изменяй по необходимости)

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

def preprocess_image_for_coreml(frame, target_size=(640, 640)):
    """
    Предобработка изображения для подачи в CoreML модель
    
    Args:
        frame: Входное изображение (numpy array)
        target_size: Целевой размер изображения (ширина, высота)
    
    Returns:
        Предобработанное изображение в формате PIL Image
    """
    # Изменяем размер изображения до размера, ожидаемого моделью
    resized_frame = cv2.resize(frame, target_size)
    
    # Конвертируем BGR (OpenCV) в RGB (PIL)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Конвертируем numpy array в PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    return pil_image

def process_frame_with_traffic_signs_model(frame_data: str) -> dict:
    """Обработка кадра моделью детекции знаков"""
    try:
        # Декодируем base64 в numpy array
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        # Предобработка для CoreML модели
        input_image = preprocess_image_for_coreml(frame)
        
        # Создание словаря входных данных для модели
        input_dict = {"image": input_image}
        
        # Детекция объектов с помощью CoreML модели
        results = traffic_signs_model.predict(input_dict)
        
        # Форматирование результатов
        detections = []
        
        # Для модели YOLO в CoreML выходной формат обычно:
        # [batch, num_classes + 5, num_predictions] где 5 это (x, y, w, h, confidence)
        # В нашем случае: [1, 159, 8400]
        if 'var_1932' in results:
            output = results['var_1932']
            if isinstance(output, np.ndarray) and len(output.shape) == 3:
                # Извлекаем данные детекции
                predictions = output[0]  # Убираем batch dimension
                
                # Применение порога уверенности
                threshold = 0.5
                
                # Обработка предсказаний
                # predictions.shape = [159, 8400]
                # Первые 4 элемента в каждом столбце - координаты (x, y, w, h)
                # Пятый элемент - уверенность объекта
                # Остальные - уверенности классов
                
                for i in range(predictions.shape[1]):
                    # Извлечение уверенности объекта
                    obj_confidence = predictions[4, i]
                    
                    if obj_confidence > threshold:
                        # Извлечение координат
                        x, y, w, h = predictions[0:4, i]
                        
                        # Извлечение уверенности классов
                        class_scores = predictions[5:, i]
                        
                        # Получение класса с максимальной уверенностью
                        class_id = np.argmax(class_scores)
                        class_confidence = class_scores[class_id]
                        
                        # Общая уверенность = уверенность объекта * уверенность класса
                        confidence = obj_confidence * class_confidence
                        
                        if confidence > threshold:
                            # Преобразование из формата (center_x, center_y, width, height) в (x1, y1, x2, y2)
                            # Координаты нормализованы (0-1), преобразуем в пиксели оригинального изображения
                            x1 = max(0, (x - w/2) * frame.shape[1])
                            y1 = max(0, (y - h/2) * frame.shape[0])
                            x2 = min(frame.shape[1], (x + w/2) * frame.shape[1])
                            y2 = min(frame.shape[0], (y + h/2) * frame.shape[0])
                            
                            detections.append({
                                'class': f'class_{class_id}',  # Временное имя класса
                                'class_id': int(class_id),
                                'confidence': float(confidence),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)]
                            })
        
        return {
            'detections': detections,
            'timestamp': asyncio.get_event_loop().time(),
            'objects_count': len(detections)
        }
    
    except Exception as e:
        logging.error(f"Error processing frame with traffic signs model: {e}")
        return {'detections': [], 'error': str(e)}

def process_frame_with_zebra_model(frame_data: str) -> dict:
    """Обработка кадра моделью сегментации зебр"""
    try:
        # Декодируем base64 в numpy array
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        # Предобработка для CoreML модели
        input_image = preprocess_image_for_coreml(frame)
        
        # Создание словаря входных данных для модели
        input_dict = {"image": input_image}
        
        # Сегментация с помощью CoreML модели
        results = zebra_model.predict(input_dict)
        
        # Форматирование результатов
        segments = []
        
        # Для сегментационной модели у нас есть два выхода:
        # 1. 'var_1368': [1, 37, 8400] - вероятно, детекции (похоже на YOLO формат)
        # 2. 'p': [1, 32, 160, 160] - вероятно, сегментационная маска
        
        # Обработка сегментационной маски
        if 'p' in results:
            output = results['p']
            if isinstance(output, np.ndarray) and len(output.shape) == 4:
                # Форма: [batch, channels, height, width]
                # Извлекаем маску (предполагаем, что канал 0 содержит информацию о зебрах)
                mask = output[0, 0, :, :]  # Форма: [160, 160]
                
                # Применяем порог для бинаризации маски
                threshold = 0.5
                binary_mask = (mask > threshold).astype(np.uint8)
                
                # Проверяем, есть ли пиксели сегментации
                if np.sum(binary_mask) > 0:
                    segments.append({
                        'type': 'zebra_crossing',
                        'confidence': float(np.mean(mask)),  # Средняя вероятность
                        'mask_shape': binary_mask.shape
                    })
        
        # Если сегментационная маска не дала результатов, пробуем детекции
        if not segments and 'var_1368' in results:
            output = results['var_1368']
            if isinstance(output, np.ndarray) and len(output.shape) == 3:
                # Извлекаем данные детекции
                predictions = output[0]  # Убираем batch dimension
                
                # Применение порога уверенности
                threshold = 0.5
                
                # Обработка предсказаний
                # predictions.shape = [37, 8400]
                # Первые 4 элемента в каждом столбце - координаты (x, y, w, h)
                # Пятый элемент - уверенность объекта
                # Остальные - уверенности классов
                
                for i in range(predictions.shape[1]):
                    # Извлечение уверенности объекта
                    obj_confidence = predictions[4, i]
                    
                    if obj_confidence > threshold:
                        # Извлечение координат
                        x, y, w, h = predictions[0:4, i]
                        
                        # Извлечение уверенности классов
                        class_scores = predictions[5:, i]
                        
                        # Получение класса с максимальной уверенностью
                        class_id = np.argmax(class_scores)
                        class_confidence = class_scores[class_id]
                        
                        # Общая уверенность = уверенность объекта * уверенность класса
                        confidence = obj_confidence * class_confidence
                        
                        if confidence > threshold:
                            # Преобразование из формата (center_x, center_y, width, height) в (x1, y1, x2, y2)
                            # Координаты нормализованы (0-1), преобразуем в пиксели оригинального изображения
                            x1 = max(0, (x - w/2) * frame.shape[1])
                            y1 = max(0, (y - h/2) * frame.shape[0])
                            x2 = min(frame.shape[1], (x + w/2) * frame.shape[1])
                            y2 = min(frame.shape[0], (y + h/2) * frame.shape[0])
                            
                            segments.append({
                                'type': 'zebra_crossing',
                                'confidence': float(confidence),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)]
                            })
        
        return {
            'segments': segments,
            'timestamp': asyncio.get_event_loop().time(),
            'segments_count': len(segments)
        }
    
    except Exception as e:
        logging.error(f"Error processing frame with zebra model: {e}")
        return {'segments': [], 'error': str(e)}

@app.websocket("/ws/video/traffic_signs")
async def websocket_traffic_signs_endpoint(websocket: WebSocket):
    """WebSocket эндпоинт для обработки видео потока моделью детекции знаков"""
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
            
            # Обработка кадра моделью детекции знаков
            result = process_frame_with_traffic_signs_model(data)
            
            # Отправка результатов обратно
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from traffic signs endpoint")
    except Exception as e:
        logging.error(f"WebSocket error in traffic signs endpoint: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/video/zebra")
async def websocket_zebra_endpoint(websocket: WebSocket):
    """WebSocket эндпоинт для обработки видео потока моделью сегментации зебр"""
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
            
            # Обработка кадра моделью сегментации зебр
            result = process_frame_with_zebra_model(data)
            
            # Отправка результатов обратно
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from zebra endpoint")
    except Exception as e:
        logging.error(f"WebSocket error in zebra endpoint: {e}")
        manager.disconnect(websocket)

# Статический файл для фронтенда
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CoreML Video Stream Processor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .video-container { display: flex; gap: 20px; margin-bottom: 20px; }
            video, canvas { width: 300px; height: 225px; border: 2px solid #ccc; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .controls { margin-bottom: 20px; }
            .results { background: #f5f5f5; padding: 15px; border-radius: 5px; }
            .model-selector { margin: 15px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📹 CoreML Video Stream Processor</h1>
            
            <div class="model-selector">
                <label>Выберите модель:</label>
                <select id="modelSelector">
                    <option value="traffic_signs">Детекция дорожных знаков</option>
                    <option value="zebra">Сегментация зебр</option>
                </select>
            </div>
            
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
                <h3>Processing Results:</h3>
                <pre id="results"></pre>
            </div>
        </div>

        <script>
        class VideoProcessor {
            constructor() {
                this.video = document.getElementById('video');
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
                this.resultsElement = document.getElementById('results');
                this.statusElement = document.getElementById('status');
                this.ws = null;
                this.isStreaming = false;
                // Ограничиваем FPS на клиенте тоже (должен быть согласован с сервером)
                // Для 5 FPS установите 5, для 10 FPS - 10, для 2 FPS - 2
                this.fps = 5; // Изменяйте это значение в соответствии с настройками сервера
                
                this.setupEventListeners();
                this.initializeCamera();
            }

            setupEventListeners() {
                document.getElementById('startBtn').addEventListener('click', () => this.startStreaming());
                document.getElementById('stopBtn').addEventListener('click', () => this.stopStreaming());
            }

            async initializeCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { 
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            facingMode: 'environment' // Используем заднюю камеру
                        },
                        audio: false
                    });
                    this.video.srcObject = stream;
                    this.updateStatus('Camera ready');
                } catch (error) {
                    this.updateStatus('Camera error: ' + error.message);
                    console.error('Camera error:', error);
                }
            }

            updateStatus(message) {
                this.statusElement.textContent = 'Status: ' + message;
            }

            startStreaming() {
                if (this.isStreaming) return;

                // Получаем выбранный тип модели
                const modelType = document.getElementById('modelSelector').value;
                let wsUrl;
                
                if (modelType === 'traffic_signs') {
                    wsUrl = `ws://${window.location.host}/ws/video/traffic_signs`;
                } else {
                    wsUrl = `ws://${window.location.host}/ws/video/zebra`;
                }

                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.isStreaming = true;
                    this.updateStatus('Streaming with ' + modelType + ' model...');
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    this.sendFrames();
                };

                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.displayResults(data);
                    this.drawResults(data);
                };

                this.ws.onerror = (error) => {
                    this.updateStatus('WebSocket error');
                    console.error('WebSocket error:', error);
                };

                this.ws.onclose = () => {
                    this.isStreaming = false;
                    this.updateStatus('Disconnected');
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                };
            }

            stopStreaming() {
                if (this.ws) {
                    this.ws.close();
                }
                this.isStreaming = false;
                this.updateStatus('Stopped');
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            }

            sendFrames() {
                if (!this.isStreaming) return;

                // Рисуем текущий кадр на canvas
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                
                // Конвертируем в base64
                const imageData = this.canvas.toDataURL('image/jpeg', 0.7);
                const base64Data = imageData.split(',')[1];

                // Отправляем на сервер
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(base64Data);
                }

                // Ограничиваем FPS
                setTimeout(() => this.sendFrames(), 1000 / this.fps);
            }

            displayResults(data) {
                this.resultsElement.textContent = JSON.stringify(data, null, 2);
            }

            drawResults(data) {
                if (data.detections) {
                    this.drawDetections(data.detections);
                } else if (data.segments) {
                    this.drawSegments(data.segments);
                }
            }

            drawDetections(detections) {
                if (!detections) return;

                // Очищаем canvas
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Рисуем оригинальное изображение
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                
                // Рисуем bounding boxes
                detections.forEach(det => {
                    const [x1, y1, x2, y2] = det.bbox;
                    
                    // Рисуем прямоугольник
                    this.ctx.strokeStyle = '#00ff00';
                    this.ctx.lineWidth = 2;
                    this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    
                    // Подпись
                    this.ctx.fillStyle = '#00ff00';
                    this.ctx.font = '12px Arial';
                    this.ctx.fillText(
                        `${det.class} (${(det.confidence * 100).toFixed(1)}%)`,
                        x1,
                        y1 - 5
                    );
                });
            }

            drawSegments(segments) {
                if (!segments) return;

                // Очищаем canvas
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Рисуем оригинальное изображение
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                
                // Рисуем маски сегментации (упрощенное отображение)
                segments.forEach(segment => {
                    // В реальной реализации здесь будет наложение маски сегментации
                    this.ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
                    this.ctx.fillRect(50, 50, 200, 100); // Пример области сегментации зебры
                    
                    // Подпись
                    this.ctx.fillStyle = '#ff0000';
                    this.ctx.font = '12px Arial';
                    this.ctx.fillText(
                        `Zebra Crossing`,
                        50,
                        45
                    );
                });
            }
        }

        // Инициализация при загрузке страницы
        document.addEventListener('DOMContentLoaded', () => {
            new VideoProcessor();
        });
        </script>
    </body>
    </html>
    """)

# Монтируем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")