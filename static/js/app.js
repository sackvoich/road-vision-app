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
    streamStatus: document.getElementById('streamStatus'),
    dropOverlay: document.getElementById('dropOverlay')
  };

const signDecoder = {
    '1_1': '1.1 "Железнодорожный переезд со шлагбаумом"',
    '1_2': '1.2 "Железнодорожный переезд без шлагбаума"',
    '1_5': '1.5 "Пересечение с трамвайной линией"',
    '1_6': '1.6 "Пересечение равнозначных дорог"',
    '1_7': '1.7 "Пересечение с круговым движением"',
    '1_8': '1.8 "Светофорное регулирование"',
    '1_10': '1.10 "Выезд на набережную"',
    '1_11': '1.11 "Опасный поворот"',
    '1_11_1': '1.11.1 "Опасный поворот направо"',
    '1_12': '1.12 "Опасные повороты"',
    '1_12_2': '1.12.2 "Опасные повороты с первым поворотом налево"',
    '1_13': '1.13 "Крутой спуск"',
    '1_14': '1.14 "Крутой подъем"',
    '1_15': '1.15 "Скользкая дорога"',
    '1_16': '1.16 "Неровная дорога"',
    '1_17': '1.17 "Искусственная неровность"',
    '1_18': '1.18 "Выброс гравия"',
    '1_19': '1.19 "Опасная обочина"',
    '1_20': '1.20 "Сужение дороги"',
    '1_20_2': '1.20.2 "Сужение дороги слева"',
    '1_20_3': '1.20.3 "Сужение дороги с обеих сторон"',
    '1_21': '1.21 "Двустороннее движение"',
    '1_22': '1.22 "Пешеходный переход"',
    '1_23': '1.23 "Дети"',
    '1_25': '1.25 "Дорожные работы"',
    '1_26': '1.26 "Перегон скота"',
    '1_27': '1.27 "Дикие животные"',
    '1_30': '1.30 "Прочие опасности"',
    '1_31': '1.31 "Направление поворота"',
    '1_33': '1.33 "Затор"',
    '2_1': '2.1 "Главная дорога"',
    '2_2': '2.2 "Конец главной дороги"',
    '2_3': '2.3 "Пересечение со второстепенной дорогой"',
    '2_3_2': '2.3.2 "Примыкание второстепенной дороги справа"',
    '2_3_3': '2.3.3 "Примыкание второстепенной дороги слева"',
    '2_3_4': '2.3.4 "Примыкание второстепенной дороги справа"',
    '2_3_5': '2.3.5 "Примыкание второстепенной дороги слева"',
    '2_3_6': '2.3.6 "Примыкание второстепенной дороги справа"',
    '2_4': '2.4 "Уступите дорогу"',
    '2_5': '2.5 "Движение без остановки запрещено"',
    '2_6': '2.6 "Преимущество встречного движения"',
    '2_7': '2.7 "Преимущество перед встречным движением"',
    '3_1': '3.1 "Въезд запрещен"',
    '3_2': '3.2 "Движение запрещено"',
    '3_4': '3.4 "Движение грузовых автомобилей запрещено"',
    '3_4_1': '3.4.1 "Движение прицепов запрещено"',
    '3_6': '3.6 "Движение тракторов запрещено"',
    '3_10': '3.10 "Движение пешеходов запрещено"',
    '3_11': '3.11 "Ограничение массы"',
    '3_12': '3.12 "Ограничение массы, приходящейся на ось"',
    '3_13': '3.13 "Ограничение высоты"',
    '3_14': '3.14 "Ограничение ширины"',
    '3_16': '3.16 "Ограничение минимальной дистанции"',
    '3_18': '3.18 "Поворот направо запрещен"',
    '3_18_2': '3.18.2 "Поворот налево запрещен"',
    '3_19': '3.19 "Разворот запрещен"',
    '3_20': '3.20 "Обгон запрещен"',
    '3_21': '3.21 "Конец зоны запрещения обгона"',
    '3_24': '3.24 "Ограничение максимальной скорости"',
    '3_25': '3.25 "Конец зоны ограничения максимальной скорости"',
    '3_27': '3.27 "Остановка запрещена"',
    '3_28': '3.28 "Стоянка запрещена"',
    '3_29': '3.29 "Стоянка запрещена по нечетным числам месяца"',
    '3_30': '3.30 "Стоянка запрещена по четным числам месяца"',
    '3_31': '3.31 "Конец зоны всех ограничений"',
    '3_32': '3.32 "Движение транспортных средств с опасными грузами запрещено"',
    '3_33': '3.33 "Движение транспортных средств с взрывчатыми и легковоспламеняющимися грузами запрещено"',
    '4_1_1': '4.1.1 "Движение прямо"',
    '4_1_2': '4.1.2 "Движение направо"',
    '4_1_2_1': '4.1.2.1 "Направление движения транспортных средств с опасными грузами"',
    '4_1_2_2': '4.1.2.2 "Направление движения транспортных средств с опасными грузами"',
    '4_1_3': '4.1.3 "Движение налево"',
    '4_1_4': '4.1.4 "Движение прямо или направо"',
    '4_1_5': '4.1.5 "Движение прямо или налево"',
    '4_1_6': '4.1.6 "Движение направо или налево"',
    '4_2_1': '4.2.1 "Объезд препятствия справа"',
    '4_2_2': '4.2.2 "Объезд препятствия слева"',
    '4_2_3': '4.2.3 "Объезд препятствия справа или слева"',
    '4_3': '4.3 "Круговое движение"',
    '4_5': '4.5 "Велосипедная дорожка"',
    '4_8_2': '4.8.2 "Направление движения транспортных средств с опасными грузами"',
    '4_8_3': '4.8.3 "Направление движения транспортных средств с опасными грузами"',
    '5_3': '5.3 "Дорога для автомобилей"',
    '5_4': '5.4 "Конец дороги для автомобилей"',
    '5_5': '5.5 "Дорога с односторонним движением"',
    '5_6': '5.6 "Конец дороги с односторонним движением"',
    '5_8': '5.8 "Реверсивное движение"',
    '5_11': '5.11 "Дорога с полосой для маршрутных ТС"',
    '5_12': '5.12 "Конец дороги с полосой для маршрутных ТС"',
    '5_14': '5.14 "Полоса для маршрутных ТС"',
    '5_15_1': '5.15.1 "Направления движения по полосам"',
    '5_15_2': '5.15.2 "Направления движения по полосе"',
    '5_15_2_2': '5.15.2.2 "Направления движения по полосам"',
    '5_15_3': '5.15.3 "Начало полосы"',
    '5_15_5': '5.15.5 "Конец полосы"',
    '5_15_7': '5.15.7 "Направление движения по полосам"',
    '5_16': '5.16 "Место остановки автобуса и/или троллейбуса"',
    '5_17': '5.17 "Место остановки трамвая"',
    '5_18': '5.18 "Место стоянки легковых такси"',
    '5_19_1': '5.19.1 "Пешеходный переход"',
    '5_20': '5.20 "Искусственная неровность"',
    '5_21': '5.21 "Жилая зона"',
    '5_22': '5.22 "Конец жилой зоны"',
    '6_2': '6.2 "Рекомендуемая скорость"',
    '6_3_1': '6.3.1 "Место для разворота"',
    '6_4': '6.4 "Парковка (парковочное место)"',
    '6_6': '6.6 "Подземный пешеходный переход"',
    '6_7': '6.7 "Надземный пешеходный переход"',
    '6_8_1': '6.8.1 "Тупик"',
    '6_8_2': '6.8.2 "Тупик"',
    '6_8_3': '6.8.3 "Тупик"',
    '6_15_1': '6.15.1 "Направление движения для грузовых автомобилей"',
    '6_15_2': '6.15.2 "Направление движения для грузовых автомобилей"',
    '6_15_3': '6.15.3 "Направление движения для грузовых автомобилей"',
    '6_16': '6.16 "Стоп-линия"',
    '7_1': '7.1 "Расстояние до объекта"',
    '7_2': '7.2 "Зона действия"',
    '7_3': '7.3 "Направления действия"',
    '7_4': '7.4 "Вид транспортного средства"',
    '7_5': '7.5 "Субботние, воскресные и праздничные дни"',
    '7_6': '7.6 "Рабочие дни"',
    '7_7': '7.7 "Дни недели"',
    '7_11': '7.11 "Ограничение разрешенной максимальной массы"',
    '7_12': '7.12 "Опасная обочина"',
    '7_14': '7.14 "Полоса движения"',
    '7_15': '7.15 "Слепые пешеходы"',
    '7_18': '7.18 "Кроме инвалидов"',
    '8_1_1': '8.1.1 "Расстояние до объекта"',
    '8_1_3': '8.1.3 "Расстояние до объекта"',
    '8_1_4': '8.1.4 "Расстояние до объекта"',
    '8_2_1': '8.2.1 "Зона действия"',
    '8_2_2': '8.2.2 "Зона действия"',
    '8_2_3': '8.2.3 "Зона действия"',
    '8_2_4': '8.2.4 "Зона действия"',
    '8_3_1': '8.3.1 "Направления действия"',
    '8_3_2': '8.3.2 "Направления действия"',
    '8_3_3': '8.3.3 "Направления действия"',
    '8_4_1': '8.4.1 "Вид транспортного средства"',
    '8_4_3': '8.4.3 "Вид транспортного средства"',
    '8_4_4': '8.4.4 "Вид транспортного средства"',
    '8_5_2': '8.5.2 "Способ постановки ТС на стоянку"',
    '8_5_4': '8.5.4 "Способ постановки ТС на стоянку"',
    '8_6_2': '8.6.2 "Способ постановки ТС на стоянку"',
    '8_6_4': '8.6.4 "Способ постановки ТС на стоянку"',
    '8_8': '8.8 "Платные услуги"',
    '8_13': '8.13 "Направление главной дороги"',
    '8_13_1': '8.13.1 "Направление главной дороги"',
    '8_14': '8.14 "Полоса движения"',
    '8_15': '8.15 "Слепые пешеходы"',
    '8_16': '8.16 "Влажное покрытие"',
    '8_17': '8.17 "Инвалиды"',
    '8_18': '8.18 "Кроме инвалидов"',
    '8_23': '8.23 "Фотовидеофиксация"',
    'zebra': 'Пешеходный переход (зебра)' 
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
    let resultsText = "--- Результаты анализа ---\n";

    if (obj.detections && obj.detections.length > 0) {
      resultsText += "\nОбнаруженные знаки:\n";
      obj.detections.forEach(d => {
        const signName = signDecoder[d.class] || `Неизвестный знак: ${d.class}`;
        const confidence = (d.confidence * 100).toFixed(1);
        resultsText += `- ${signName} (уверенность: ${confidence}%)\n`;
      });
    } else {
      resultsText += "\nДорожные знаки не обнаружены.\n";
    }

    if (obj.segmentations && obj.segmentations.length > 0) {
      resultsText += "\nОбнаружена разметка:\n";
      obj.segmentations.forEach(s => {
          const segmentationName = signDecoder[s.class] || s.class;
          const confidence = (s.confidence * 100).toFixed(1);
          resultsText += `- ${segmentationName} (уверенность: ${confidence}%)\n`;
      });
    } else {
        resultsText += "\nРазметка 'зебра' не обнаружена.\n";
    }
    
    els.results.textContent = resultsText;
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
        const readableName = signDecoder[d.class] || d.class;
        const label = `${readableName} ${(d.confidence*100||0).toFixed(1)}%`;
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
        const readableName = signDecoder[s.class] || s.class;
        const label = `${readableName} ${(s.confidence*100||0).toFixed(1)}%`;
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

  async function handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        showMessage('Эй, мудила! Это не картинка. Нужен файл изображения.');
        return;
    }
    showLoader(true); 
    showMessage('');
    try {
      const url = URL.createObjectURL(file);
      const img = await loadImage(url);
      els.canvas.width = img.width; 
      els.canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);

      const result = await sendFileForProcessing(file);
      setResults(result);

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

  async function uploadSelected() {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (!file) { 
      showMessage('Сначала выбери файл, гений.'); 
      return; 
    }
    await handleFileUpload(file);
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

  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.dropOverlay.classList.add('is-dragover');
  });
  
  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.relatedTarget) {
      els.dropOverlay.classList.remove('is-dragover');
    }
  });

  window.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.dropOverlay.classList.remove('is-dragover');

    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      await handleFileUpload(file);
    }
  });
  // Kickoff
  initCamera();
})();