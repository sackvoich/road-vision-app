// static/app_v2.js
class AdvancedVideoProcessor {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.initializeCamera();
        
        this.ws = null;
        this.isStreaming = false;
        this.frameTimes = [];
        this.lastProcessedTime = 0;
        
        // Set canvas dimensions
        this.canvas.width = 640;
        this.canvas.height = 480;
    }

    initializeElements() {
        // Video elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // UI elements
        this.resultsElement = document.getElementById('results');
        this.statusElement = document.getElementById('status');
        this.objectCountElement = document.getElementById('objectCount');
        this.fpsDisplayElement = document.getElementById('fpsDisplay');
        this.processingTimeElement = document.getElementById('processingTime');
        
        // Control elements
        this.trafficCard = document.getElementById('trafficCard');
        this.zebraCard = document.getElementById('zebraCard');
        this.fpsSlider = document.getElementById('fpsLimit');
        this.fpsValue = document.getElementById('fpsValue');
        this.scaleSlider = document.getElementById('scaleFactor');
        this.scaleValue = document.getElementById('scaleValue');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        // State
        this.enabledModels = {
            traffic: true,
            zebra: false
        };
    }

    setupEventListeners() {
        // Model selection
        this.trafficCard.addEventListener('click', () => this.toggleModel('traffic'));
        this.zebraCard.addEventListener('click', () => this.toggleModel('zebra'));
        
        // Sliders
        this.fpsSlider.addEventListener('input', () => {
            this.fpsValue.textContent = this.fpsSlider.value;
        });
        
        this.scaleSlider.addEventListener('input', () => {
            this.scaleValue.textContent = this.scaleSlider.value;
        });
        
        // Buttons
        this.startBtn.addEventListener('click', () => this.startStreaming());
        this.stopBtn.addEventListener('click', () => this.stopStreaming());
    }

    toggleModel(modelName) {
        this.enabledModels[modelName] = !this.enabledModels[modelName];
        
        const card = modelName === 'traffic' ? this.trafficCard : this.zebraCard;
        const status = card.querySelector('.status');
        
        if (this.enabledModels[modelName]) {
            card.classList.add('active');
            status.textContent = 'Active';
            status.style.background = '#28a745';
        } else {
            card.classList.remove('active');
            status.textContent = 'Inactive';
            status.style.background = '#6c757d';
        }
        
        console.log('Enabled models:', this.enabledModels);
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
        this.statusElement.className = `status-bar status-${type}`;
    }

    updateStats(objectsCount, processingTime) {
        this.objectCountElement.textContent = objectsCount;
        this.processingTimeElement.textContent = processingTime > 0 ? 
            `${(processingTime * 1000).toFixed(1)}ms` : '0ms';
        
        // Calculate FPS
        const now = performance.now();
        this.frameTimes.push(now);
        
        // Keep only last 10 frames for FPS calculation
        if (this.frameTimes.length > 10) {
            this.frameTimes.shift();
        }
        
        if (this.frameTimes.length > 1) {
            const fps = 1000 / ((now - this.frameTimes[0]) / (this.frameTimes.length - 1));
            this.fpsDisplayElement.textContent = fps.toFixed(1);
        }
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
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.sendFrames();
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleServerResponse(data);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };

        this.ws.onerror = (error) => {
            this.updateStatus('Connection error', 'error');
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            this.isStreaming = false;
            this.updateStatus('Disconnected', 'ready');
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
        };
    }

    stopStreaming() {
        if (this.ws) {
            this.ws.close();
        }
        this.isStreaming = false;
        this.updateStatus('Stopped', 'ready');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
    }

    sendFrames() {
        if (!this.isStreaming || !this.video.videoWidth) return;

        try {
            // Draw frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert to base64
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            const base64Data = imageData.split(',')[1];

            // Prepare settings
            const settings = {
                models: this.enabledModels,
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
        } catch (error) {
            console.error('Error sending frame:', error);
        }

        // Continue streaming
        if (this.isStreaming) {
            setTimeout(() => this.sendFrames(), 1000 / parseInt(this.fpsSlider.value));
        }
    }

    handleServerResponse(data) {
        // Display raw results
        this.resultsElement.textContent = JSON.stringify(data, null, 2);
        
        // Update statistics
        this.updateStats(data.objects_count || 0, 0);
        
        // Draw detections if any
        if (data.detections && data.detections.length > 0) {
            this.drawDetections(data.detections);
        } else {
            // Just redraw the original frame if no detections
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        }
    }

    drawDetections(detections) {
        // Clear and redraw original image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            
            // Different colors for different models
            let color = '#FF0000'; // Red for traffic
            if (det.model === 'zebra') color = '#0000FF'; // Blue for zebra
            
            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw label background
            const label = `${det.class} (${(det.confidence * 100).toFixed(1)}%)`;
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.fillRect(x1 - 2, y1 - 25, textWidth + 10, 20);
            
            // Draw label
            this.ctx.fillStyle = color;
            this.ctx.font = '14px Arial';
            this.ctx.fillText(label, x1, y1 - 8);
        });
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AdvancedVideoProcessor();
});