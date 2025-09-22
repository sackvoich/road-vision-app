// static/app_v2.js
class AdvancedVideoProcessor {
    constructor() {
        // DOM Elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultsElement = document.getElementById('results');
        this.statusElement = document.getElementById('status');
        this.objectCountElement = document.getElementById('objectCount');
        
        // Control elements
        this.trafficModelCheckbox = document.getElementById('trafficModel');
        this.zebraModelCheckbox = document.getElementById('zebraModel');
        this.fpsSlider = document.getElementById('fpsLimit');
        this.fpsValue = document.getElementById('fpsValue');
        this.scaleSlider = document.getElementById('scaleFactor');
        this.scaleValue = document.getElementById('scaleValue');
        
        // WebSocket and state
        this.ws = null;
        this.isStreaming = false;
        this.lastFrameTime = 0;
        
        // Bind events and initialize
        this.setupEventListeners();
        this.initializeCamera();
        this.setupControlListeners();
    }

    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startStreaming());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopStreaming());
    }

    setupControlListeners() {
        // FPS slider
        this.fpsSlider.addEventListener('input', () => {
            this.fpsValue.textContent = this.fpsSlider.value;
        });
        
        // Scale slider
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
                    facingMode: 'environment' // Use rear camera
                },
                audio: false
            });
            this.video.srcObject = stream;
            this.updateStatus('Camera ready', 'ready');
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

        this.ws = new WebSocket(`ws://${window.location.host}/ws/video`);
        
        this.ws.onopen = () => {
            this.isStreaming = true;
            this.updateStatus('Streaming...', 'streaming');
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            this.sendFrames();
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.displayResults(data);
            this.drawDetections(data);
            this.updateObjectCount(data.objects_count || 0);
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
        if (!this.isStreaming) return;

        // Respect FPS limit
        const now = performance.now();
        const fpsLimit = parseInt(this.fpsSlider.value);
        const interval = 1000 / fpsLimit;
        
        if (now - this.lastFrameTime >= interval) {
            this.processAndSendFrame();
            this.lastFrameTime = now;
        }

        // Continue sending frames
        requestAnimationFrame(() => this.sendFrames());
    }

    processAndSendFrame() {
        // Draw current frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to base64
        const imageData = this.canvas.toDataURL('image/jpeg', 0.7);
        const base64Data = imageData.split(',')[1];

        // Get current settings
        const settings = {
            models: {
                traffic: this.trafficModelCheckbox.checked,
                zebra: this.zebraModelCheckbox.checked
            },
            fps_limit: parseInt(this.fpsSlider.value),
            scale_factor: parseInt(this.scaleSlider.value) / 100
        };

        // Send to server
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                frame: base64Data,
                settings: settings
            }));
        }
    }

    displayResults(data) {
        this.resultsElement.textContent = JSON.stringify(data, null, 2);
    }

    drawDetections(data) {
        const detections = data.detections || [];
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw original image
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw bounding boxes
        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            
            // Different colors for different models
            let color;
            if (det.model === 'traffic') {
                color = '#00ff00'; // Green for traffic signs
            } else if (det.model === 'zebra') {
                color = '#0000ff'; // Blue for zebra crossings
            } else {
                color = '#ff0000'; // Red for others
            }
            
            // Draw rectangle
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw label
            const label = `${det.class} (${(det.confidence * 100).toFixed(1)}%)`;
            this.ctx.fillStyle = color;
            this.ctx.font = '14px Arial';
            this.ctx.fillText(label, x1, y1 - 5);
        });
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AdvancedVideoProcessor();
});