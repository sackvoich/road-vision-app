
// static/upload.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('uploadInput');
    const processBtn = document.getElementById('processBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const resultsElement = document.getElementById('results');
    const statusElement = document.getElementById('status');

    let lastResults = null;
    let uploadedImage = null;

    uploadInput.addEventListener('change', () => {
        processBtn.disabled = uploadInput.files.length === 0;
        downloadBtn.disabled = true;
        lastResults = null;
        resultsElement.textContent = '';

        if (uploadInput.files.length > 0) {
            const file = uploadInput.files[0];
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                uploadedImage = img;
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                statusElement.textContent = 'Status: Image loaded. Ready to process.';
            }
            img.onerror = () => {
                statusElement.textContent = 'Status: Failed to load image.';
            }
        }
    });

    processBtn.addEventListener('click', async () => {
        const file = uploadInput.files[0];
        if (!file) return;

        statusElement.textContent = 'Status: Uploading and processing...';
        processBtn.disabled = true;
        downloadBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

            const data = await response.json();
            if (data.error) throw new Error(data.error);
            
            lastResults = data;
            
            redrawCanvas();
            
            downloadBtn.disabled = false;
            statusElement.textContent = 'Status: Processing complete.';
            resultsElement.textContent = JSON.stringify({
                detections: data.detections,
                segmentations: data.segmentations
            }, null, 2);

        } catch (error) {
            statusElement.textContent = `Status: Error: ${error.message}`;
            console.error('Processing error:', error);
        } finally {
            processBtn.disabled = false;
        }
    });

    downloadBtn.addEventListener('click', () => {
        if (!lastResults) return;
        const link = document.createElement('a');
        link.download = 'processed-image.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    });

    function redrawCanvas() {
        if (!uploadedImage) return;
        // Ensure canvas size is correct
        if (canvas.width !== uploadedImage.width || canvas.height !== uploadedImage.height) {
            canvas.width = uploadedImage.width;
            canvas.height = uploadedImage.height;
        }
        ctx.drawImage(uploadedImage, 0, 0, canvas.width, canvas.height);
        if (lastResults) {
            drawDetections(lastResults.detections || []);
            drawSegmentations(lastResults.segmentations || []);
        }
    }
    
    function drawDetections(detections) {
        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            const label = `${det.class} (${(det.confidence * 100).toFixed(1)}%)`;

            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            ctx.fillStyle = '#00FF00';
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x1, y1 - 15, textWidth + 10, 15);
            
            ctx.fillStyle = '#000000';
            ctx.font = '12px Arial';
            ctx.fillText(label, x1 + 5, y1 - 5);
        });
    }

    function drawSegmentations(segmentations) {
        segmentations.forEach(seg => {
            const points = seg.points;
            if (points.length < 2) return;

            ctx.fillStyle = 'rgba(255, 0, 255, 0.4)';
            ctx.strokeStyle = '#FF00FF';
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(points[0][0], points[0][1]);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i][0], points[i][1]);
            }
            ctx.closePath();
            
            ctx.fill();
            ctx.stroke();
            
            const label = `${seg.class} (${(seg.confidence * 100).toFixed(1)}%)`;
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '12px Arial';
            ctx.fillText(label, points[0][0], points[0][1] - 5);
        });
    }
});
