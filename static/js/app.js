(() => {
  const els = {
    video: document.getElementById('camera'),
    canvas: document.getElementById('frameCanvas'),
    btnSnap: document.getElementById('btnSnap'),
    btnStart: document.getElementById('btnStartStream'),
    btnStop: document.getElementById('btnStopStream'),
    fileInput: document.getElementById('fileInput'),
    btnUpload: document.getElementById('btnUpload'),
    results: document.getElementById('resultsPanel'),
    loader: document.getElementById('loader'),
    message: document.getElementById('message'),
    streamStatus: document.getElementById('streamStatus')
  };

  const ctx = els.canvas.getContext('2d');
  let streamOk = false;
  let ws = null;
  let streaming = false;
  const clientFps = 5;
  let frameTimer = null;

  function showLoader(show) {
    els.loader.classList.toggle('hidden', !show);
  }
  function showMessage(text, type = 'info') {
    if (!text) { els.message.classList.add('hidden'); return; }
    els.message.textContent = text;
    els.message.classList.remove('hidden');
  }
  function setResults(obj) {
    try { els.results.textContent = JSON.stringify(obj, null, 2); }
    catch { els.results.textContent = String(obj); }
  }

  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false
      });
      els.video.srcObject = stream;
      await els.video.play();
      streamOk = true;
      resizeCanvasToVideo();
    } catch (err) {
      streamOk = false;
      showMessage('Нет доступа к камере. Вы можете загрузить изображение для обработки.');
    }
  }

  function resizeCanvasToVideo() {
    const w = els.video.videoWidth || 640;
    const h = els.video.videoHeight || 480;
    els.canvas.width = w;
    els.canvas.height = h;
  }

  function drawOverlay(results) {
    if (!results) return;
    if (results.detections && Array.isArray(results.detections)) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.font = '12px Inter, Arial';
      results.detections.forEach(d => {
        const [x1, y1, x2, y2] = d.bbox || [];
        if ([x1,y1,x2,y2].some(v => typeof v !== 'number')) return;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const label = `${d.class ?? ''} ${(d.confidence*100||0).toFixed(1)}%`;
        const tw = ctx.measureText(label).width + 8;
        const th = 16;
        ctx.fillStyle = 'rgba(34,197,94,0.85)';
        ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
        ctx.fillStyle = '#0f172a';
        ctx.fillText(label, x1 + 4, Math.max(10, y1 - 4));
      });
    }
    if (results.segmentations && Array.isArray(results.segmentations)) {
      ctx.fillStyle = 'rgba(59,130,246,0.25)';
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.font = '12px Inter, Arial';
      results.segmentations.forEach(s => {
        const pts = s.points || [];
        if (pts.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        const label = `${s.class ?? ''} ${(s.confidence*100||0).toFixed(1)}%`;
        ctx.fillStyle = '#e2e8f0';
        ctx.fillText(label, pts[0][0], Math.max(10, pts[0][1] - 4));
      });
    }
  }

  function drawFromVideo() {
    if (!streamOk) return;
    try { ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height); } catch {}
  }

  function startStreaming() {
    if (streaming) return;
    streaming = true;
    els.btnStart.disabled = true;
    els.btnStop.disabled = false;
    els.streamStatus.textContent = 'Стрим включен';
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/video`);
    ws.onopen = () => {
      scheduleSendFrame();
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.status === 'skipped') return;
        setResults(data);
        drawFromVideo();
        drawOverlay(data);
      } catch {}
    };
    ws.onerror = () => {
      stopStreaming();
      showMessage('Ошибка WebSocket во время стрима');
    };
    ws.onclose = () => {
      streaming = false;
      els.btnStart.disabled = false;
      els.btnStop.disabled = true;
      els.streamStatus.textContent = 'Стрим остановлен';
      if (frameTimer) { clearTimeout(frameTimer); frameTimer = null; }
    };
  }

  function stopStreaming() {
    streaming = false;
    if (frameTimer) { clearTimeout(frameTimer); frameTimer = null; }
    try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch {}
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    els.streamStatus.textContent = 'Стрим остановлен';
  }

  function scheduleSendFrame() {
    if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (els.video.videoWidth > 0) {
      const tmp = document.createElement('canvas');
      tmp.width = els.video.videoWidth; tmp.height = els.video.videoHeight;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(els.video, 0, 0, tmp.width, tmp.height);
      const dataUrl = tmp.toDataURL('image/jpeg', 0.7);
      const base64Data = dataUrl.split(',')[1];
      ws.send(base64Data);
    }
    frameTimer = setTimeout(scheduleSendFrame, 1000 / clientFps);
  }
  
  // *** ИЗМЕНЁННАЯ ФУНКЦИЯ ***
  async function sendFileForProcessing(file) {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/image', {
          method: 'POST',
          body: formData
      });

      if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Server error: ${response.status} - ${errorText}`);
      }
      return await response.json();
  }

  // *** ИЗМЕНЁННАЯ ФУНКЦИЯ ***
  async function snapAndSend() {
    if (!streamOk) {
      showMessage('Камера не активна.');
      return;
    }
    drawFromVideo();
    showLoader(true);
    showMessage('');
    try {
      const blob = await new Promise(res => els.canvas.toBlob(res, 'image/jpeg', 0.9));
      const result = await sendFileForProcessing(new File([blob], "snapshot.jpg", {type: "image/jpeg"}));
      
      setResults(result);
      drawFromVideo(); // Перерисовываем исходный кадр
      drawOverlay(result); // Рисуем результаты поверх
    } catch (e) {
      console.error(e);
      showMessage(`Ошибка обработки кадра: ${e.message}`);
    } finally {
      showLoader(false);
    }
  }

  // *** ИЗМЕНЁННАЯ ФУНКЦИЯ ***
  async function uploadSelected() {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (!file) { showMessage('Выберите изображение.'); return; }
    showLoader(true); showMessage('');
    try {
      // Отобразим выбранное изображение в canvas
      const url = URL.createObjectURL(file);
      const img = await loadImage(url);
      els.canvas.width = img.width; els.canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);

      // Отправим файл и получим результат
      const result = await sendFileForProcessing(file);
      setResults(result);

      // Перерисовываем исходное изображение и рисуем оверлей
      ctx.drawImage(img, 0, 0); 
      drawOverlay(result);
    } catch (e) {
      console.error(e);
      showMessage(`Ошибка отправки изображения: ${e.message}`);
    } finally {
      showLoader(false);
      els.fileInput.value = '';
    }
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  // Events
  window.addEventListener('resize', resizeCanvasToVideo);
  els.btnSnap.addEventListener('click', snapAndSend);
  els.btnStart.addEventListener('click', startStreaming);
  els.btnStop.addEventListener('click', stopStreaming);
  els.btnUpload.addEventListener('click', uploadSelected);

  // Kickoff
  initCamera();
})();