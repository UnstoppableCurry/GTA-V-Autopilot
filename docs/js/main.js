/**
 * Minimal enhancements: external video fallback notice & focus management.
 */
(function () {
  'use strict';

  // Mark external video links for analytics-free tracking in UI only
  document.querySelectorAll('[data-external-video]').forEach(function (link) {
    link.setAttribute('rel', 'noopener noreferrer');
  });

  // Pause other videos when one starts playing (accessibility / bandwidth)
  var videos = document.querySelectorAll('video');
  videos.forEach(function (video) {
    video.addEventListener('play', function () {
      videos.forEach(function (other) {
        if (other !== video && !other.paused) {
          other.pause();
        }
      });
    });
  });
})();
