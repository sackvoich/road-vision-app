// static/app_v2.js
class AdvancedVideoProcessor {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultsElement = document.getElementById('results');
        this.statusElement = document.getElementById('status');
        this.objectCountElement = document.getElementById('objectCount');
        
        this.trafficModelCheckbox = document.getElementById('trafficModel');
        this.zebraModelCheckbox = document.getElementById('zebraModel');
        this.fpsSlider = document.getElementById('fpsLimit');
        this.fpsValue = document.getElementById('fpsValue');
        this.scaleSlider = document.getElementById('scaleFactor');
        this.scaleValue = document.getElementById('scaleValue');
        
        this.ws = null;
        this.isStreaming = false;
        this.lastFrameTime = 0;
        
        // Установка размеров canvas
        this.canvas.width = 640;
        this.canvas.height = 480;
        
        this.setupEventListeners();
        this.initializeCamera();
        this.setupControlListeners();
    }

    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startStreaming());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopStreaming());
    }

    setupControlListeners() {
        this.fpsSlider.addEventListener('input', () => {
            this.fpsValue.textContent = this.fpsSlider.value;
        });
        
        this.scaleSlider.addEventListener('input', () => {
            this.scaleValue.textContent = this.scaleSlider.value;
        });
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
            
            // Ждем загрузки видео
            this.video.onloadedmetadata = () => {
                this.updateStatus('Camera ready', 'ready');
            };
            
        } catch (error) {
            this.updateStatus('Camera error: ' + error.message, 'error');
            console.error('Camera error:', error);
        }
    }

    updateStatus(message, type = 'ready') {
        this.statusElement.textContent = 'Status: ' + message;
        this.statusElement.className = 'status-bar status-' + type;
    }

    updateObjectCount(count) {
        this.objectCountElement.textContent = count;
    }

    startStreaming() {
        if (this.isStreaming) return;

        const host = window.location.hostname || 'localhost';
        const isSecure = window.location.protocol === 'https:';
        const wsProtocol = isSecure ? 'wss:' : 'ws:';
        
        this.ws = new WebSocket(`${wsProtocol}//${host}:8000/ws/video`);
        
        this.ws.onopen = () => {
            this.isStreaming = true;
            this.updateStatus('Streaming...', 'streaming');
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            this.sendFrames();
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.displayResults(data);
                this.drawDetections(data);
                this.updateObjectCount(data.objects_count || 0);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };

        this.ws.onerror = (error) => {
            this.updateStatus('WebSocket error', 'error');
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            this.isStreaming = false;
            this.updateStatus('Disconnected', 'ready');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        };
    }

    stopStreaming() {
        if (this.ws) {
            this.ws.close();
        }
        this.isStreaming = false;
        this.updateStatus('Stopped', 'ready');
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }

    sendFrames() {
        if (!this.isStreaming || !this.video.videoWidth) return;

        const fpsLimit = parseInt(this.fpsSlider.value);
        const interval = 1000 / fpsLimit;
        const now = performance.now();
        
        if (now - this.lastFrameTime >= interval) {
            this.processAndSendFrame();
            this.lastFrameTime = now;
        }

        requestAnimationFrame(() => this.sendFrames());
    }

    processAndSendFrame() {
        try {
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.7);
            const base64Data = imageData.split(',')[1];

            const settings = {
                models: {
                    traffic: this.trafficModelCheckbox.checked,
                    zebra: this.zebraModelCheckbox.checked
                },
                fps_limit: parseInt(this.fpsSlider.value),
                scale_factor: parseInt(this.scaleSlider.value) / 100
            };

            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    frame: base64Data,
                    settings: settings
                }));
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
    }

    displayResults(data) {
        this.resultsElement.textContent = JSON.stringify(data, null, 2);
    }

    drawDetections(data) {
        const detections = data.detections || [];
        
        // Clear and redraw original image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            
            let color = '#00ff00'; // Default green
            if (det.model === 'traffic') color = '#ff0000'; // Red for traffic
            if (det.model === 'zebra') color = '#0000ff'; // Blue for zebra
            
            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw label background
            const label = `${det.class} (${(det.confidence * 100).toFixed(1)}%)`;
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.fillRect(x1 - 2, y1 - 20, textWidth + 10, 20);
            
            // Draw label text
            this.ctx.fillStyle = color;
            this.ctx.font = '14px Arial';
            this.ctx.fillText(label, x1, y1 - 5);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new AdvancedVideoProcessor();
});