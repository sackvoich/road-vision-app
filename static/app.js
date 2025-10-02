// static/app.js
class VideoProcessor {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultsElement = document.getElementById('results');
        this.statusElement = document.getElementById('status');
        this.ws = null;
        this.isStreaming = false;
        this.fps = 5; // Ограничиваем FPS на клиенте для отправки
        this.lastResults = null; // Храним последние результаты
        
        this.setupEventListeners();
        this.initializeCamera();
        this.renderLoop(); // Запускаем цикл отрисовки
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
            // Убедимся, что размеры canvas соответствуют видео
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            };
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

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/video`);
        
        this.ws.onopen = () => {
            this.isStreaming = true;
            this.updateStatus('Streaming...');
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            this.sendFrames();
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status === 'skipped') return;
            
            // Просто обновляем данные и выводим текст
            this.lastResults = data;
            this.displayResults(data);
        };

        this.ws.onerror = (error) => {
            this.updateStatus('WebSocket error');
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            this.isStreaming = false;
            this.lastResults = null; // Очищаем результаты при отключении
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
    }

    sendFrames() {
        if (!this.isStreaming || this.video.videoWidth === 0) return;

        // Используем временный canvas, чтобы не мешать отрисовке
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.video.videoWidth;
        tempCanvas.height = this.video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(this.video, 0, 0, tempCanvas.width, tempCanvas.height);
        
        const imageData = tempCanvas.toDataURL('image/jpeg', 0.7);
        const base64Data = imageData.split(',')[1];

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(base64Data);
        }

        setTimeout(() => this.sendFrames(), 1000 / this.fps);
    }

    renderLoop() {
        // Рисуем видеопоток на canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Поверх рисуем последние полученные результаты
        if (this.lastResults) {
            if (this.lastResults.detections && this.lastResults.detections.length > 0) {
                this.drawDetections(this.lastResults.detections);
            }
            if (this.lastResults.segmentations && this.lastResults.segmentations.length > 0) {
                this.drawSegmentations(this.lastResults.segmentations);
            }
        }

        // Зацикливаем
        requestAnimationFrame(() => this.renderLoop());
    }

    displayResults(data) {
        // Убираем лишнюю информацию для чистоты вывода
        const simplifiedData = {
            detections: data.detections,
            segmentations: data.segmentations
        };
        this.resultsElement.textContent = JSON.stringify(simplifiedData, null, 2);
    }

    drawDetections(detections) {
        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            const label = `${det.class} (${(det.confidence * 100).toFixed(1)}%)`;

            // Рисуем прямоугольник
            this.ctx.strokeStyle = '#00FF00'; // Зеленый
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Рисуем фон для текста
            this.ctx.fillStyle = '#00FF00';
            const textWidth = this.ctx.measureText(label).width;
            this.ctx.fillRect(x1, y1 - 15, textWidth + 10, 15);
            
            // Рисуем текст
            this.ctx.fillStyle = '#000000';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(label, x1 + 5, y1 - 5);
        });
    }

    drawSegmentations(segmentations) {
        segmentations.forEach(seg => {
            const points = seg.points;
            if (points.length < 2) return;

            this.ctx.fillStyle = 'rgba(255, 0, 255, 0.4)'; // Розовый полупрозрачный
            this.ctx.strokeStyle = '#FF00FF'; // Розовый
            this.ctx.lineWidth = 2;

            this.ctx.beginPath();
            this.ctx.moveTo(points[0][0], points[0][1]);
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i][0], points[i][1]);
            }
            this.ctx.closePath();
            
            this.ctx.fill();
            this.ctx.stroke();
            
            // Подпись для сегментации
            const label = `${seg.class} (${(seg.confidence * 100).toFixed(1)}%)`;
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(label, points[0][0], points[0][1] - 5);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VideoProcessor();
});