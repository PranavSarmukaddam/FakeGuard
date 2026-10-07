/* ==========================================================================
   FakeGuard — Client Interactions (main.js)
   Clean, responsive UI interactions for the News Verification Platform
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function () {

  // ── NAVBAR SHADOW ON SCROLL ─────────────────────────────────────────
  const navbar = document.getElementById('navbar');
  if (navbar) {
    window.addEventListener('scroll', function () {
      if (window.scrollY > 15) {
        navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.35)';
      } else {
        navbar.style.boxShadow = 'none';
      }
    }, { passive: true });
  }

  // ── MOBILE NAV TOGGLE ──────────────────────────────────────────────
  const navToggle = document.getElementById('navToggle');
  const navLinks  = document.getElementById('navLinks');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function () {
      navLinks.classList.toggle('open');
    });
    navLinks.querySelectorAll('.nav-link').forEach(function (link) {
      link.addEventListener('click', function () {
        navLinks.classList.remove('open');
      });
    });
  }

  // ── ANIMATE BARS (CONFIDENCE, SIMILARITY, MINI-BARS) ────────────────
  const bars = document.querySelectorAll('.conf-bar, .sim-bar, .mini-bar');
  bars.forEach(function (bar) {
    const target = bar.style.width;
    bar.style.width = '0%';
    requestAnimationFrame(function () {
      setTimeout(function () {
        bar.style.transition = 'width 0.9s cubic-bezier(0.16, 1, 0.3, 1)';
        bar.style.width = target;
      }, 100);
    });
  });

  // ── AUTO-DISMISS FLASH NOTIFICATIONS ───────────────────────────────
  const flashes = document.querySelectorAll('.flash');
  flashes.forEach(function (flash, idx) {
    setTimeout(function () {
      flash.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      flash.style.opacity = '0';
      flash.style.transform = 'translateY(-6px)';
      setTimeout(function () { flash.remove(); }, 400);
    }, 4500 + idx * 500);
  });

});
