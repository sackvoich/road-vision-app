(() => {
  // Простые улучшения UX: мягкая прокрутка по якорям и подсветка активного раздела
  const links = document.querySelectorAll('a.anchor[href^="#"]');
  for (const a of links) {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href').slice(1);
      const el = document.getElementById(id);
      if (el) {
        e.preventDefault();
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        el.classList.add('highlight');
        setTimeout(() => el.classList.remove('highlight'), 800);
      }
    });
  }
})();


