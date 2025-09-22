/ static/app.js

class VideoProcessor {
    constructor() {
        // DOM elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultsElement = document.getElementById('results');
        this.statusElement = document.getElementById('status');
        
        // Stats elements
        this.fpsElement = document.getElementById('fps');
        this.signCountElement = document.getElementById('signCount');
        this.zebraCountElement = document.getElementById('zebraCount');
        this.processTimeElement = document.getElementById('processTime');
        
        // WebSocket and streaming state
        this.ws = null;
        this.isStreaming = false;
        this.frameRate = 10; // Target FPS
        
        // Performance tracking
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        this.lastProcessTime = 0;
        
        // Class colors for consistent visualization
        this.classColors = {
            'Zebra': { stroke: '#9b59b6', fill: 'rgba(155, 89, 182, 0.3)' },
            'default': { stroke: '#2ecc71', fill: 'rgba(46, 204, 113, 0.3)' }
        };
        
        this.initialize();
    }
    
    async initialize() {
        this.setupEventListeners();
        await this.initializeCamera();
    }
    
    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startStreaming());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopStreaming());
    }
    
    async initializeCamera() {
        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment' // Use back camera on mobile
                },
                audio: false
            });
            
            this.video.srcObject = stream;
            
            // Set canvas size when video metadata is loaded
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.updateStatus('Camera ready - Click Start to begin');
                console.log(`Video dimensions: ${this.video.videoWidth}x${this.video.videoHeight}`);
            };
            
        } catch (error) {
            this.updateStatus(`Camera error: ${error.message}`);
            console.error('Camera initialization failed:', error);
        }
    }
    
    updateStatus(message) {
        this.statusElement.textContent = `Status: ${message}`;
    }
    
    updateStats(data) {
        // Update FPS
        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFpsUpdate > 1000) {
            const fps = Math.round(this.frameCount * 1000 / (now - this.lastFpsUpdate));
            this.fpsElement.textContent = fps;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
        
        // Update detection counts
        const signCount = data.detections ? data.detections.length : 0;
        const zebraCount = data.segmentations ? data.segmentations.length : 0;
        this.signCountElement.textContent = signCount;
        this.zebraCountElement.textContent = zebraCount;
        
        // Update processing time
        if (data.timestamp) {
            const processTime = Date.now() - this.lastProcessTime;
            this.processTimeElement.textContent = `${processTime}ms`;
        }
    }
    
    startStreaming() {
        if (this.isStreaming) return;
        
        // Determine WebSocket protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/video`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isStreaming = true;
            this.updateStatus('Connected - Streaming...');
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            // Start sending frames
            this.sendFrames();
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.error) {
                    console.error('Backend error:', data.error);
                    this.updateStatus(`Error: ${data.error}`);
                    return;
                }
                
                // Update stats
                this.updateStats(data);
                
                // Display JSON results
                this.displayResults(data);
                
                // Draw visualization
                this.drawResults(data);
                
            } catch (error) {
                console.error('Error processing message:', error);
            }
        };
        
        this.ws.onerror = (error) => {
            this.updateStatus('WebSocket error - Check connection');
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            this.isStreaming = false;
            this.updateStatus('Disconnected');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            // Reset stats
            this.fpsElement.textContent = '0';
            this.signCountElement.textContent = '0';
            this.zebraCountElement.textContent = '0';
            this.processTimeElement.textContent = '0ms';
        };
    }
    
    stopStreaming() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.close();
        }
        this.isStreaming = false;
        this.updateStatus('Stopped');
    }
    
    sendFrames() {
        if (!this.isStreaming || this.video.paused || this.video.ended) {
            return;
        }
        
        // Ensure canvas matches video dimensions
        if (this.canvas.width !== this.video.videoWidth || 
            this.canvas.height !== this.video.videoHeight) {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
        }
        
        // Draw current video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to base64 JPEG
        const quality = 0.8;
        const imageData = this.canvas.toDataURL('image/jpeg', quality);
        const base64Data = imageData.split(',')[1];
        
        // Send if WebSocket is open
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(base64Data);
            this.lastProcessTime = Date.now();
        }
        
        // Schedule next frame
        setTimeout(() => this.sendFrames(), 1000 / this.frameRate);
    }
    
    displayResults(data) {
        // Format JSON for display
        const formatted = JSON.stringify(data, null, 2);
        this.resultsElement.textContent = formatted;
    }
    
    drawResults(data) {
        // Clear canvas and redraw video frame
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw detections (traffic signs)
        if (data.detections && Array.isArray(data.detections)) {
            this.drawDetections(data.detections);
        }
        
        // Draw segmentations (zebra crossings)
        if (data.segmentations && Array.isArray(data.segmentations)) {
            this.drawSegmentations(data.segmentations);
        }
    }
    
    drawDetections(detections) {
        detections.forEach(det => {
            // Validate bbox
            if (!det.bbox || !Array.isArray(det.bbox) || det.bbox.length !== 4) {
                console.warn('Invalid bbox:', det);
                return;
            }
            
            const [x1, y1, x2, y2] = det.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Skip invalid boxes
            if (width <= 0 || height <= 0) {
                return;
            }
            
            // Get color for this class
            const colors = this.classColors[det.class] || this.classColors.default;
            
            // Draw bounding box
            this.ctx.strokeStyle = colors.stroke;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, width, height);
            
            // Draw label with background
            const label = `${det.class} (${(det.confidence * 100).toFixed(1)}%)`;
            this.ctx.font = '14px sans-serif';
            const textMetrics = this.ctx.measureText(label);
            const textWidth = textMetrics.width;
            const textHeight = 14; // Approximate height
            
            const labelY = y1 > textHeight + 4 ? y1 - textHeight - 4 : y1 + 4;
            
            // Background
            this.ctx.fillStyle = colors.stroke;
            this.ctx.fillRect(x1, labelY - textHeight / 2 - 2, textWidth + 8, textHeight + 4);
            
            // Text
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(label, x1 + 4, labelY + textHeight / 2);
        });
    }
    
    drawSegmentations(segmentations) {
        segmentations.forEach(seg => {
            // Validate points
            if (!seg.points || !Array.isArray(seg.points) || seg.points.length < 3) {
                console.warn('Invalid segmentation points:', seg);
                return;
            }
            
            const points = seg.points;
            const colors = this.classColors[seg.class] || this.classColors.default;
            
            this.ctx.beginPath();
            this.ctx.moveTo(points[0][0], points[0][1]);
            
            // Draw lines to connect all points
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i][0], points[i][1]);
            }
            
            this.ctx.closePath();
            
            // Apply styles and draw the polygon
            this.ctx.fillStyle = colors.fill;
            this.ctx.strokeStyle = colors.stroke;
            this.ctx.lineWidth = 2;
            
            this.ctx.fill();
            this.ctx.stroke();
        });
    }
}

// --- Initialization ---
// Create an instance of the processor class once the DOM is fully loaded.
document.addEventListener('DOMContentLoaded', () => {
    new VideoProcessor();
});