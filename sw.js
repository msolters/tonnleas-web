/* JigRipper service worker: the web app, and JigRipper Mini framed by another
 * site, open and listen with no network once they have been used online.
 *
 * Written into the site as /sw.js by scripts/web-release.mjs at every release,
 * from mobile/scripts/offline/sw.template.js; BUILD and PRECACHE are filled in
 * there. Never edit sw.js on the site. Registered by the app once it is ready
 * (src/offline/offline-worker.web.ts).
 *
 *  - PRECACHE: every file the app needs to reach Listening, kept when this
 *    release's worker installs. BUILD hashes their contents, so any change to
 *    any of them makes a new sw.js, and browsers fetch the new set.
 *  - Pages: network first, so an online visitor always gets the current
 *    release; the kept copy when the network fails or stalls PAGE_TIMEOUT_MS.
 *  - Hashed files (the bundle, assets/) and ?v= files: the kept copy first.
 *    Their address changes whenever their bytes do.
 *  - Other files of ours: network first, kept copy offline.
 *  - Models (.onnx, .part): never touched. The workers keep them in their own
 *    store (jigripper-model-*), as they did before this worker existed.
 *  - Other sites (Cloudflare's analytics, thesession.org): never touched.
 *
 * To switch it off, release with `web-release.mjs --sw-off`
 * (scripts/offline/sw-off.js).
 */
const BUILD = '4ba75a9293b5ec69';
const SHELL_PREFIX = 'jigripper-shell-';
const SHELL = SHELL_PREFIX + BUILD;
const PRECACHE = [
  "/",
  "/mini/embed/",
  "/_expo/static/js/web/index-17cf23ff186342682b8c5df6d181b619.js",
  "/favicon.ico",
  "/1006/canonical_redirects.json",
  "/1006/label_map.json",
  "/1006/tune_index.json",
  "/1006/tunes_canonical.json",
  "/quad_index.bin",
  "/setting_interval_index.bin",
  "/dsp_core.js",
  "/dsp_core.wasm",
  "/markov.js",
  "/markov.wasm",
  "/hcqt_fold12.js",
  "/hcqt_fold12.wasm?v=v1",
  "/ort.min.js",
  "/ort-wasm-simd.wasm?v=v1",
  "/dsp-worker.js?v=v33",
  "/presence-worker.js?v=v33",
  "/basic-pitch-worker-onnx.js?v=v33",
  "/foote-worker.js?v=v33",
  "/audio-processor.js?v=v33",
  "/mini/",
  "/help/",
  "/privacy/",
  "/privacy.html",
  "/terms/",
  "/support.html",
  "/help/img/listen-ring.png?v=1c8c70d0fd",
  "/help/img/context-buttons.png?v=e5c3b270ac",
  "/help/img/river.jpg?v=8aa9ef759a",
  "/help/img/sheet-fullscreen.jpg?v=363be0e500",
  "/help/img/sheet-keys-settings.png?v=a79c54e51c",
  "/help/img/sheet-paper.jpg?v=76083ef13e",
  "/help/img/sheet-buttons-fullscreen.png?v=32ee343ff5",
  "/help/img/sheet-buttons.png?v=acc4f66949",
  "/help/img/abc-panel.png?v=dd69d9e879",
  "/help/img/thesession-ipad.jpg?v=167e63a35b",
  "/help/img/thesession-tab.png?v=06ab612bc8",
  "/help/img/hold-clasp.png?v=5d3712137a",
  "/help/img/search-results.jpg?v=e6b23c5806",
  "/help/img/edit-sessions-list.png?v=9e312088e6",
  "/help/img/edit-session-drawer.png?v=cabb94db5a",
  "/help/img/edit-tune-menu.png?v=896779bc47",
  "/help/img/edit-tune-replace.png?v=9fd82d71c0",
  "/help/img/edit-tune-menu-sets.png?v=51ba51f3fb",
  "/help/img/edit-sets-list.png?v=ed3ab080f2",
  "/help/img/edit-set-delete.png?v=a02765585c",
  "/help/img/edit-set-detail.png?v=db896124be",
  "/help/img/edit-set-rename.png?v=84f2a89f72",
  "/help/img/edit-set-bar.png?v=fb2504793f",
  "/help/img/watch-apple-tune.png?v=73a13f3ada",
  "/help/img/watch-smart-stack.png?v=7f01f7b14a",
  "/help/img/watch-unlock-iphone.png?v=51fbcb002e",
  "/help/img/watch-wear-tile.png?v=42c842c492",
  "/help/img/mini.png?v=98451cab35",
  "/assets/node_modules/@react-navigation/elements/lib/module/assets/back-icon.35ba0eaec5a4f5ed12ca16fabeae451d.png",
  "/assets/node_modules/@react-navigation/elements/lib/module/assets/back-icon-mask.0a328cd9c1afd0afe8e3b1ec5165b1b4.png",
  "/assets/node_modules/@react-navigation/elements/lib/module/assets/clear-icon.c94f6478e7ae0cdd9f15de1fcb9e5e55.png",
  "/assets/node_modules/@react-navigation/elements/lib/module/assets/close-icon.808e1b1b9b53114ec2838071a7e6daa7.png",
  "/assets/node_modules/@react-navigation/elements/lib/module/assets/search-icon.286d67d3f74808a60a78d3ebf1a5fb57.png",
  "/assets/node_modules/expo-router/assets/error.d1ea1496f9057eb392d5bbf3732a61b7.png",
  "/assets/node_modules/expo-router/assets/file.19eeb73b9593a38f8e9f418337fc7d10.png",
  "/assets/node_modules/expo-router/assets/pkg.ab19f4cbc543357183a20571f68380a3.png",
  "/assets/node_modules/expo-router/assets/forward.d8b800c443b8972542883e0b9de2bdc6.png",
  "/assets/node_modules/expo-router/assets/sitemap.412dd9275b6b48ad28f5e3d81bb1f626.png",
  "/assets/node_modules/expo-router/assets/arrow_down.017bc6ba3fc25503e5eb5e53826d48a8.png",
  "/assets/node_modules/expo-router/assets/unmatched.20e71bdf79e3a97bf55fd9e164041578.png"
];
const PAGE_TIMEOUT_MS = 4000;
const MODEL = /\.onnx(\.part\d+)?$/;
const HASHED = /^\/(_expo\/static|assets)\//;
// No answer, or the server failing: use the kept copy. Firefox's navigation
// preload does not reject when the connection is refused, it RESOLVES with a
// network-error response (status 0), so a rejection alone is not enough.
// A redirect is an answer: a page's arrives as an opaque redirect, status 0
// too (/mini-docs → /mini-docs/), and goes to the browser to follow.
const failed = (res) => !res || res.type === 'error' || (res.status === 0 && res.type !== 'opaqueredirect') || res.status >= 500;

// A page is kept by its path: /?mini=1&embed=1… and / are the same index.html.
const keyOf = (url, isPage) => {
  const u = new URL(url, self.location.href);
  return (isPage || u.pathname.endsWith('/') || u.pathname.endsWith('.html')) ? u.origin + u.pathname : u.href;
};
// This release's cache, opened once per start of the worker and only if it
// still exists. A newer release deletes it when it takes over; a request this
// worker is still finishing then writes into the detached cache, which nobody
// reads, instead of creating it again by name.
let shellP = null;
const shell = () => shellP || (shellP = caches.has(SHELL).then((has) => (has ? caches.open(SHELL) : null)));
const keep = async (key, res) => {
  if (!res || !res.ok || res.type !== 'basic') return;
  try { const c = await shell(); if (c) await c.put(key, res); } catch (e) { /* storage full: just not kept */ }
};
// Only this release's copies: never another cache's (an older release, the model store).
const kept = async (key) => { const c = await shell(); return c ? c.match(key, { ignoreVary: true }) : undefined; };
// The same file as the kept copy? Then there is nothing to write (the catalog is 22 MB).
const sameFile = (a, b) => {
  const e = a.headers.get('etag');
  if (e) return e === b.headers.get('etag');
  const m = a.headers.get('last-modified');
  return !!m && m === b.headers.get('last-modified') && a.headers.get('content-length') === b.headers.get('content-length');
};

self.addEventListener('install', (event) => {
  // Take over as soon as this release's files are kept. Every rule below goes
  // by address (hashed bundle, ?v= files) or to the network first, so a page of
  // the last release keeps working under this one. Waiting instead never ends
  // while a tab or a Mini frame stays open: a refresh opens the new frames
  // before the old ones close.
  self.skipWaiting();
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL);
    shellP = Promise.resolve(cache);
    // no-cache = a conditional request: a file the browser already holds costs
    // a 304, and one it holds from an older release is replaced. Any failure
    // fails the install; the browser tries again at the next visit.
    await Promise.all(PRECACHE.map(async (u) => {
      const res = await fetch(new Request(u, { cache: 'no-cache' }));
      if (!res.ok) throw new Error(`precache ${u}: HTTP ${res.status}`);
      await cache.put(keyOf(u), res);
    }));
    await Promise.race([followTyped(), new Promise((r) => setTimeout(r, 5000))]);
  })());
});

// The site is www.jigripper.com; jigripper.com redirects there. With no network
// the browser can follow only a redirect it has seen, so a typed jigripper.com/mini
// that it had never seen showed the dinosaur (2026-10-25). Following the typed
// addresses here once is enough for Chrome; Firefox does not follow a kept
// redirect offline. Best effort: failures are ignored, and the install waits 5 s at most.
const TYPED = ['/', '/mini', '/mini/', '/help', '/help/', '/mini-docs', '/mini-docs/'];
async function followTyped() {
  const { protocol, hostname } = self.location;
  if (!hostname.startsWith('www.')) return;
  const bare = `${protocol}//${hostname.slice(4)}`;
  await Promise.all(TYPED.map((p) => fetch(bare + p, { mode: 'no-cors', credentials: 'include' }).catch(() => {})));
}

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith(SHELL_PREFIX) && k !== SHELL) await caches.delete(k);
    if (self.registration.navigationPreload) await self.registration.navigationPreload.enable().catch(() => {});
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || MODEL.test(url.pathname)) return;
  // Cloudflare's own paths (the analytics beacon, challenges), except the
  // email decoder it adds to pages that show an address.
  if (url.pathname.startsWith('/cdn-cgi/') && !url.pathname.startsWith('/cdn-cgi/scripts/')) return;
  if (req.mode === 'navigate') return event.respondWith(page(event));
  if (HASHED.test(url.pathname) || url.searchParams.has('v')) return event.respondWith(keptFirst(event));
  event.respondWith(networkFirst(event));
});

async function page(event) {
  const req = event.request;
  const net = (async () => {
    const pre = await event.preloadResponse;
    const res = (pre && !failed(pre)) ? pre : await fetch(req);
    if (failed(res)) throw new Error('network: ' + res.status);
    if (res.ok) event.waitUntil(keep(keyOf(req.url, true), res.clone()));
    return res;
  })();
  const timeout = new Promise((r) => setTimeout(r, PAGE_TIMEOUT_MS, 'timeout'));
  try {
    const first = await Promise.race([net, timeout]);
    if (first !== 'timeout') return first;
    return (await keptPage(req.url)) || await net;
  } catch (e) {
    const copy = await keptPage(req.url);
    if (copy) return copy;
    throw e;
  }
}

// GitHub Pages answers /help with a redirect to /help/; offline, so do we.
async function keptPage(url) {
  const key = keyOf(url, true);
  const copy = await kept(key);
  if (copy || key.endsWith('/') || !(await kept(key + '/'))) return copy;
  return Response.redirect(key + '/' + new URL(url).search, 301);
}

async function keptFirst(event) {
  const req = event.request;
  const copy = await kept(req.url);
  if (copy) return copy;
  const res = await fetch(req);
  if (res.ok) event.waitUntil(keep(req.url, res.clone()));
  return res;
}

async function networkFirst(event) {
  const req = event.request;
  const res = await fetch(req).catch(() => null);
  if (failed(res)) return (await kept(req.url)) || res || Response.error();
  const copy = await kept(req.url);
  if (!copy || !sameFile(res, copy)) event.waitUntil(keep(req.url, res.clone()));
  return res;
}
