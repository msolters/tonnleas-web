/*
 * JigRipper Mini — embed it on any page.
 *
 *   <div class="jigripper-mini"></div>
 *   <script src="https://www.jigripper.com/mini/embed.js" async></script>
 *
 * Each .jigripper-mini element becomes the widget: a search-box-sized listen
 * ring that names the tune being played. Until someone taps it, it is only this
 * script's stand-in — a drawing of the widget, with nothing fetched from
 * jigripper.com but this file. The tap swaps in the real widget (an iframe of
 * jigripper.com/?mini=1, i.e. the JigRipper app itself) and starts the one-time
 * model download; a second tap, inside the widget, starts listening.
 *
 * Options, as attributes on the element:
 *   data-theme="light" | "dark"   default: picked from the page's background
 *
 * Sites whose Content-Security-Policy blocks outside scripts can host a copy of
 * this file themselves, or use the iframe on its own (it needs no script):
 *   <iframe src="https://www.jigripper.com/?mini=1&embed=1" allow="microphone"
 *           title="JigRipper Mini" style="width:100%;max-width:536px;height:92px;border:0"></iframe>
 *
 * Nothing you play leaves the device and nothing is recorded; the widget sets no
 * cookies on your site and sends nothing to it.
 */
(function () {
  'use strict';

  var script = document.currentScript;
  var ORIGIN = script && script.src ? new URL(script.src).origin : 'https://www.jigripper.com';
  var WIDTH = 536, HEIGHT = 92;   // the widget: a 520 px card + 8 px margin all round

  var THEMES = {
    dark: { surface: 'rgba(20,28,25,0.92)', border: 'rgba(143,191,160,0.16)', text: '#E8E4DA',
      text2: '#8E9C91', text3: '#5F6F63', silver: '#8FBFA0', ring: 'rgba(60,140,50,0.75)' },
    light: { surface: 'rgba(255,253,248,0.94)', border: 'rgba(40,64,50,0.16)', text: '#1E2A23',
      text2: '#56645B', text3: '#7D887F', silver: '#3F7357', ring: 'rgba(44,112,40,0.8)' },
  };

  /** Light unless the page behind the element is dark. */
  function pickTheme(el) {
    var want = el.getAttribute('data-theme');
    if (want === 'light' || want === 'dark') return want;
    for (var e = el; e && e.nodeType === 1; e = e.parentElement) {
      var m = /rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?/.exec(getComputedStyle(e).backgroundColor);
      if (m && (m[4] === undefined || +m[4] > 0.5)) {
        return (0.2126 * m[1] + 0.7152 * m[2] + 0.0722 * m[3]) / 255 < 0.5 ? 'dark' : 'light';
      }
    }
    return 'light';
  }

  /** The idle listen ring, drawn once (the knot's maths at rest). */
  function ringSvg(color) {
    var size = 36, view = 80, c = view / 2, outer = size * 0.74, inner = size * 0.38 * 0.55;
    var mid = (inner + outer) / 2, amp = ((outer - inner) / 2) * 0.12, paths = '';
    for (var s = 0; s < 3; s++) {
      var d = '';
      for (var i = 0; i <= 160; i++) {
        var th = (i / 160) * Math.PI * 2;
        var r = mid + amp * Math.sin(4 * th + (s * 2 * Math.PI) / 3);
        d += (i ? 'L' : 'M') + (c + r * Math.cos(th)).toFixed(1) + ',' + (c + r * Math.sin(th)).toFixed(1);
      }
      paths += '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.3" stroke-linecap="round"/>';
    }
    return '<svg width="80" height="80" viewBox="0 0 80 80" aria-hidden="true" style="position:absolute;left:-8px;top:-8px">' + paths + '</svg>';
  }

  function mount(el) {
    if (el.getAttribute('data-jigripper-mounted')) return;
    el.setAttribute('data-jigripper-mounted', '1');
    var theme = pickTheme(el), t = THEMES[theme];
    var font = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

    var box = document.createElement('div');
    box.style.cssText = 'position:relative;width:100%;max-width:' + WIDTH + 'px;height:' + HEIGHT + 'px';

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.setAttribute('aria-label', 'JigRipper Mini: identify a tune. Loads a one-time download of about 155 MB.');
    btn.style.cssText = 'all:unset;box-sizing:border-box;position:absolute;inset:8px;display:flex;align-items:center;'
      + 'padding:6px 21px 6px 6px;border-radius:21px;border:1px solid ' + t.border + ';background:' + t.surface + ';'
      + 'cursor:pointer;font-family:' + font + ';-webkit-tap-highlight-color:transparent';
    btn.innerHTML =
      '<span style="position:relative;flex:0 0 64px;width:64px;height:64px;margin-right:8px">' + ringSvg(t.ring) + '</span>'
      + '<span style="flex:1;min-width:0;line-height:1.32;white-space:nowrap;overflow:hidden;text-align:left">'
      + '<span style="display:block;font-size:16px;color:' + t.text + ';overflow:hidden;text-overflow:ellipsis">Identify a tune</span>'
      + '<span style="display:block;font-size:13px;color:' + t.text2 + ';margin-top:1px;overflow:hidden;text-overflow:ellipsis">Tap the ring, then play a few bars</span>'
      + '<span style="display:block;font-size:10.5px;letter-spacing:0.07em;text-transform:uppercase;color:' + t.text3 + ';margin-top:3px;overflow:hidden;text-overflow:ellipsis">'
      + '<span style="text-transform:none;font-size:11px;font-weight:500;letter-spacing:0.02em;text-decoration:underline;text-decoration-color:' + t.silver + '73;color:' + t.silver + '">JigRipper</span>'
      + ' &nbsp;·&nbsp; One-time download, about 155 MB</span>'
      + '</span>';

    btn.addEventListener('click', function () {
      var frame = document.createElement('iframe');
      frame.src = ORIGIN + '/?mini=1&embed=1&autostart=1' + (theme === 'light' ? '&theme=light' : '');
      frame.title = 'JigRipper Mini';
      frame.allow = 'microphone; autoplay';
      frame.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;border:0;opacity:0;transition:opacity .25s;'
        + 'color-scheme:normal;background:transparent';
      // The stand-in stays until the widget has drawn, so there is no blank flash.
      frame.addEventListener('load', function () {
        setTimeout(function () { frame.style.opacity = '1'; btn.style.visibility = 'hidden'; }, 150);
      });
      btn.disabled = true;
      btn.style.cursor = 'default';
      box.appendChild(frame);
    }, { once: true });

    box.appendChild(btn);
    el.textContent = '';
    el.appendChild(box);
  }

  function mountAll() {
    var els = document.querySelectorAll('.jigripper-mini, [data-jigripper-mini]');
    for (var i = 0; i < els.length; i++) mount(els[i]);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll);
  else mountAll();
})();
