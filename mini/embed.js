/*! JigRipper Mini embed.js — Copyright (c) 2026 Séisionnach LLC
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This licence covers this file only. The widget it opens (jigripper.com/?mini=1),
 * the tune model and the rest of JigRipper are not part of it: they are used
 * under JigRipper's terms of use, https://www.jigripper.com/terms/
 */
/*
 * JigRipper Mini — embed it on any page.
 *
 *   <div class="jigripper-mini"></div>
 *   <script src="https://www.jigripper.com/mini/embed.js" async></script>
 *
 * Each .jigripper-mini element becomes the widget: a search-box-sized listen
 * ring that names the tune being played. This script only makes room for it and
 * frames jigripper.com/mini/embed/ there: a small placeholder card, a few KB,
 * that loads nothing else. Tapping the card swaps in the real widget (the
 * JigRipper app itself, in mini mode), which downloads the tune model once and
 * then starts listening: one tap does it all (a few browsers ask for a second
 * tap on the ring before they allow audio). Everything you see, the
 * card included, comes from jigripper.com, so it is always the current version.
 *
 * Options, as attributes on the element:
 *   data-theme="light" | "dark"   default: picked from the page's background
 *   data-session-links="new-tab" | "same-tab"
 *                                 where "Open on The Session ↗" opens. Default
 *                                 new-tab, so visitors keep your page; same-tab
 *                                 takes your page itself to the tune.
 *
 * The script sets up every element that is on the page when it loads. Pages that
 * add one later (single-page apps, content loaded on demand) call:
 *   JigRipperMini.mount()          set up any new .jigripper-mini elements
 *   JigRipperMini.mount(element)   set up that element, or the ones inside it
 *   JigRipperMini.open(element)    open a widget as if its card were tapped
 *                                  (default: the first one on the page). It
 *                                  downloads, then listens, though a browser
 *                                  may want a tap on the ring first when nobody
 *                                  has tapped the page.
 * If you call these before the script has loaded, wait for its load event:
 *   <script src="…/mini/embed.js" async onload="JigRipperMini.mount()"></script>
 *
 * Sites whose Content-Security-Policy blocks outside scripts can host a copy of
 * this file themselves (the licence above allows it), or use the iframe on its
 * own (it needs no script):
 *   <iframe src="https://www.jigripper.com/mini/embed/" allow="microphone; autoplay"
 *           title="JigRipper Mini" style="width:100%;max-width:536px;height:92px;border:0"></iframe>
 * (Options on the iframe's URL: ?theme=light, and links=same for same-tab Session links,
 * e.g. /mini/embed/?theme=light&links=same.)
 *
 * Nothing you play leaves the device and nothing is recorded; the widget sets no
 * cookies on your site and sends nothing to it.
 */
(function () {
  'use strict';

  // Where the widget comes from. A copy of this file hosted on someone else's
  // site must still open jigripper.com, so the script's own address is used only
  // when it IS JigRipper's: the live site, or our own dev servers (the tailnet,
  // localhost:8081/8082) so the mini-docs page there shows the dev widget.
  var script = document.currentScript;
  var ORIGIN = 'https://www.jigripper.com';
  try {
    var u = new URL(script.src);
    if (/^(www\.)?jigripper\.com$/.test(u.hostname) || /\.ts\.net$/.test(u.hostname)
        || (/^(localhost|127\.0\.0\.1)$/.test(u.hostname) && /^808[12]$/.test(u.port))) ORIGIN = u.origin;
  } catch (e) { /* inline or unknown: keep jigripper.com */ }
  var WIDTH = 536, HEIGHT = 92;   // the widget: a 520 px card + 8 px margin all round

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

  function mount(el) {
    if (el.getAttribute('data-jigripper-mounted')) return;
    el.setAttribute('data-jigripper-mounted', '1');
    var box = document.createElement('div');
    box.style.cssText = 'position:relative;width:100%;max-width:' + WIDTH + 'px;height:' + HEIGHT + 'px';
    var frame = document.createElement('iframe');
    frame.src = ORIGIN + '/mini/embed/?theme=' + pickTheme(el)
      + (el.getAttribute('data-session-links') === 'same-tab' ? '&links=same' : '');
    frame.title = 'JigRipper Mini';
    frame.allow = 'microphone; autoplay';
    frame.setAttribute('data-jigripper-frame', '');
    // color-scheme pinned to light on BOTH sides (here and the card's :root):
    // when the frame and the page inside it disagree, Chrome paints an opaque
    // backdrop behind it. "normal" here followed the host, so a page declaring
    // "light dark", seen by someone who prefers dark, got a white box (found on
    // thesession.org, 2026-10-11).
    frame.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;border:0;'
      + 'color-scheme:light;background:transparent';
    frame.addEventListener('load', function () { frame.setAttribute('data-loaded', ''); }, { once: true });
    box.appendChild(frame);
    el.textContent = '';
    el.appendChild(box);
  }

  var SELECTOR = '.jigripper-mini, [data-jigripper-mini]';

  /** Set up `root` itself if it is a widget element, and any inside it. */
  function mountAll(root) {
    root = root && root.nodeType === 1 ? root : document;
    if (root !== document && root.matches && root.matches(SELECTOR)) mount(root);
    var els = root.querySelectorAll(SELECTOR);
    for (var i = 0; i < els.length; i++) mount(els[i]);
  }

  /** Open a widget as if its card were tapped (default: the first). The card is
   *  jigripper.com's page, so this asks it to, once it has loaded. */
  function open(el) {
    var target = el && el.nodeType === 1 ? el : document.querySelector(SELECTOR);
    if (!target) return false;
    mountAll(target);
    var frame = target.querySelector('iframe[data-jigripper-frame]');
    if (!frame) return false;
    var send = function () { frame.contentWindow.postMessage({ jigripperMini: 'open' }, ORIGIN); };
    if (frame.hasAttribute('data-loaded')) send(); else frame.addEventListener('load', send, { once: true });
    return true;
  }

  // Loaded twice (a page that includes the script per widget)? Keep one API.
  var api = window.JigRipperMini || {};
  api.mount = function (root) { mountAll(root); };
  api.open = open;
  window.JigRipperMini = api;

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { mountAll(); });
  else mountAll();
})();
