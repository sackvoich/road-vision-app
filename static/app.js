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
        this.fps = 5; // Ограничиваем FPS на клиенте
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
                    facingMode: 'environment'
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
        this.ws = new WebSocket(`ws://${window.location.host}/ws/video`);
        this.ws.onopen = () => {
            this.isStreaming = true;
            this.updateStatus('Streaming...');
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            this.sendFrames();
        };
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.displayResults(data);
            this.drawDetections(data);
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

    drawDetections(data) {
        if (!data) return;

        // Рисуем оригинальное изображение
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Рисуем дорожные знаки (зеленый)
        if (data.traffic_signs && data.traffic_signs.length > 0) {
            data.traffic_signs.forEach(det => {
                const [x1, y1, x2, y2] = det.bbox;
                this.ctx.strokeStyle = '#00ff00'; // Зеленый для знаков
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                this.ctx.fillStyle = '#00ff00';
                this.ctx.font = '12px Arial';
                this.ctx.fillText(
                    `${det.class} (${(det.confidence * 100).toFixed(1)}%)`,
                    x1,
                    y1 - 5
                );
            });
        }

        // Рисуем пешеходные переходы (синий)
        if (data.zebra_crossings && data.zebra_crossings.length > 0) {
            data.zebra_crossings.forEach(det => {
                const [x1, y1, x2, y2] = det.bbox;
                this.ctx.strokeStyle = '#0000ff'; // Синий для зебр
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                this.ctx.fillStyle = '#0000ff';
                this.ctx.font = '12px Arial';
                this.ctx.fillText(
                    `${det.class} (${(det.confidence * 100).toFixed(1)}%)`,
                    x1,
                    y1 - 5
                );
            });
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    new VideoProcessor();
});