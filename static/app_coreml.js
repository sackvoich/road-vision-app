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
        this.fps = 10; // Можно поднять FPS для CoreML

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
            // ВАЖНО: Устанавливаем размер canvas, когда видео готово
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                console.log(`Canvas size set to: ${this.canvas.width}x${this.canvas.height}`);
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
            // --- УЛУЧШЕННАЯ ОТЛАДКА ---
            console.log("Received raw data:", event.data); 
            const data = JSON.parse(event.data);
            
            if (data.error) {
                console.error("Backend Error:", data.error);
                return;
            }

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
        if (this.ws) this.ws.close();
        this.isStreaming = false;
        this.updateStatus('Stopped');
    }

    sendFrames() {
        if (!this.isStreaming || this.video.paused || this.video.ended) {
            this.stopStreaming();
            return;
        }
        
        // Убедимся, что canvas имеет правильный размер
        if (this.canvas.width !== this.video.videoWidth) {
             this.canvas.width = this.video.videoWidth;
             this.canvas.height = this.video.videoHeight;
        }
        
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(imageData.split(',')[1]);
        }
        setTimeout(() => this.sendFrames(), 1000 / this.fps);
    }

    displayResults(data) {
        this.resultsElement.textContent = JSON.stringify(data, null, 2);
    }
    
    drawResults(data) {
        // Очищаем и рисуем кадр с камеры
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Рисуем детекции (знаки)
        if (data.detections && Array.isArray(data.detections)) {
            this.drawDetections(data.detections);
        }
        
        // Рисуем сегментации (зебры)
        if (data.segmentations && Array.isArray(data.segmentations)) {
            this.drawSegmentations(data.segmentations);
        }
    }

    drawDetections(detections) {
        detections.forEach(det => {
            // Более надежная проверка, что bbox существует и является массивом
            if (!det.bbox || !Array.isArray(det.bbox) || det.bbox.length !== 4) {
                console.warn("Skipping detection with invalid bbox:", det);
                return;
            }

            const [x1, y1, x2, y2] = det.bbox;
            const label = `${det.class} (${(det.confidence * 100).toFixed(0)}%)`;
            
            this.ctx.strokeStyle = '#00FF00'; // Ярко-зеленый
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            this.ctx.fillStyle = '#00FF00';
            this.ctx.font = '14px Arial';
            this.ctx.fillText(label, x1, y1 > 10 ? y1 - 5 : 10);
        });
    }

    drawSegmentations(segmentations) {
        segmentations.forEach(seg => {
            // Более надежная проверка
            if (!seg.points || !Array.isArray(seg.points) || seg.points.length < 3) {
                 console.warn("Skipping segmentation with invalid points:", seg);
                 return;
            }
            const points = seg.points;

            this.ctx.fillStyle = 'rgba(138, 43, 226, 0.45)'; // Полупрозрачный фиолетовый
            this.ctx.strokeStyle = '#8A2BE2'; // Фиолетовый
            this.ctx.lineWidth = 2;

            this.ctx.beginPath();
            this.ctx.moveTo(points[0][0], points[0][1]);
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i][0], points[i][1]);
            }
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.stroke();
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VideoProcessor();
});