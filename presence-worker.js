/**
 * Instrument Presence Web Worker — InstrumentSeparator-v3-big presence head
 *
 * Pipeline:
 *   raw audio chunk (22050 Hz mono, ~2 s)
 *     → STFT (n_fft=1024, hop=256, hann, center=true)
 *     → magnitude  (1, T, 513)
 *     → ONNX instrument_presence → presence (1, 10) sigmoid probabilities
 *
 * Class order (fixed by the model):
 *   [fiddle, flute, whistle, concertina, accordion, pipes,
 *    plucked, piano, harp, percussion]
 *
 * The head was trained with NO negative audio — on speech/noise/silence it
 * hallucinates confidently (silence reads concertina 0.96). The host must
 * only consult it when the audio is already judged to be music.
 *
 * Protocol mirrors melody-separator-worker.js:
 *   Receives:
 *     { type:'init',    baseUrl:string, modelUrl?:string }
 *     { type:'process', id:number, samples:Float32Array (22050Hz mono) }
 *   Sends:
 *     { type:'model-loaded' }
 *     { type:'model-error', error:string }
 *     { type:'result',  id:number, presence:Float32Array (10) }
 */

// ── Worker → log relay ────────────────────────────────────────────────────
// Worker threads can't use the main-thread console-relay, so their logs never
// reach /tmp/app-console.log. POST straight to the same relay endpoint (a
// CORS-simple text POST — no preflight). Dev-only; silent no-op if it's offline.
// DEV only: the worker has no __DEV__, so gate on being served from the dev server
// (localhost). In production (jigripper.com) wlog is a COMPLETE no-op — no console
// spam (efficiency) and no localhost fetch (which trips Chrome's Local Network Access
// prompt — correctness).
var WORKER_DEV = (function () {
  try { var h = (self.location && self.location.hostname) || ''; return h === 'localhost' || h === '127.0.0.1'; }
  catch (_) { return false; }
})();
function wlog(m) {
  if (!WORKER_DEV) return;
  try { console.log('[presence-worker]', m); } catch (_) {}
  try {
    fetch('http://localhost:8124/log', {
      method: 'POST',
      body: new Date().toISOString() + ' [log] [presence-worker] ' + m,
      keepalive: true,
    }).catch(function () {});
  } catch (_) {}
}

// ── STFT params (must match the model's training) ──
var SEP_N_FFT = 1024;
var SEP_HOP = 256;
var SEP_BINS = SEP_N_FFT / 2 + 1; // 513

// ── FFT ──
var _twRe = null, _twIm = null, _fftSize = 0;
function initFFT(n) {
  if (_fftSize === n) return;
  _fftSize = n;
  var half = n >> 1;
  _twRe = new Float32Array(half);
  _twIm = new Float32Array(half);
  for (var k = 0; k < half; k++) {
    var ang = -2 * Math.PI * k / n;
    _twRe[k] = Math.cos(ang);
    _twIm[k] = Math.sin(ang);
  }
}
function fft(re, im, n) {
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      var tr = re[i]; re[i] = re[j]; re[j] = tr;
      var ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (var len = 2; len <= n; len *= 2) {
    var halfLen = len >> 1;
    var step = n / len;
    for (var i2 = 0; i2 < n; i2 += len) {
      for (var k2 = 0; k2 < halfLen; k2++) {
        var idx = k2 * step;
        var rA = re[i2 + k2 + halfLen], iA = im[i2 + k2 + halfLen];
        var tRe = rA * _twRe[idx] - iA * _twIm[idx];
        var tIm = rA * _twIm[idx] + iA * _twRe[idx];
        re[i2 + k2 + halfLen] = re[i2 + k2] - tRe;
        im[i2 + k2 + halfLen] = im[i2 + k2] - tIm;
        re[i2 + k2] += tRe;
        im[i2 + k2] += tIm;
      }
    }
  }
}

var _hann = null;
function getHann() {
  if (_hann && _hann.length === SEP_N_FFT) return _hann;
  _hann = new Float32Array(SEP_N_FFT);
  for (var i = 0; i < SEP_N_FFT; i++) _hann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / SEP_N_FFT));
  return _hann;
}

// center=true STFT, magnitude only: reflect-pad by N_FFT/2 each side.
// Returns frame-major mag[f * SEP_BINS + b].
function stftMag(samples) {
  initFFT(SEP_N_FFT);
  var win = getHann();
  var PAD = SEP_N_FFT >> 1;
  var padded = new Float32Array(samples.length + 2 * PAD);
  padded.set(samples, PAD);
  for (var i = 0; i < PAD; i++) {
    var l = Math.min(i + 1, samples.length - 1);
    padded[PAD - 1 - i] = l >= 0 ? samples[l] : 0;
    var r = samples.length - 1 - (i + 1);
    padded[padded.length - PAD + i] = r >= 0 ? samples[r] : 0;
  }
  var nFrames = Math.floor((padded.length - SEP_N_FFT) / SEP_HOP) + 1;
  var mag = new Float32Array(nFrames * SEP_BINS);
  var bRe = new Float32Array(SEP_N_FFT);
  var bIm = new Float32Array(SEP_N_FFT);
  for (var f = 0; f < nFrames; f++) {
    var off = f * SEP_HOP;
    for (var k = 0; k < SEP_N_FFT; k++) {
      bRe[k] = padded[off + k] * win[k];
      bIm[k] = 0;
    }
    fft(bRe, bIm, SEP_N_FFT);
    var dst = f * SEP_BINS;
    for (var b = 0; b < SEP_BINS; b++) {
      var r2 = bRe[b], i2 = bIm[b];
      mag[dst + b] = Math.sqrt(r2 * r2 + i2 * i2);
    }
  }
  return { mag: mag, nFrames: nFrames };
}

// ── ONNX runtime ──
var ort = null;
var session = null;

self.onmessage = async function(e) {
  var msg = e.data;
  if (msg.type === 'init') {
    try {
      importScripts(msg.baseUrl + '/ort.min.js');
      ort = self.ort;
      ort.env.wasm.numThreads = 1;
      // Cache-bust tokens come from the bundle (ASSET_VERSIONS) via init. Empty →
      // no query → identical to the un-versioned behaviour.
      var wasmV = msg.wasmVersion ? ('?v=' + msg.wasmVersion) : '';
      var modelV = msg.modelVersion ? ('?v=' + msg.modelVersion) : '';
      // ORT string wasmPaths is a bare prefix (no room for ?v=), so version via the
      // object form (ORT 1.18 indexes wasmPaths[filename] when it's an object).
      ort.env.wasm.wasmPaths = wasmV
        ? {
            'ort-wasm-simd-threaded.wasm': msg.baseUrl + '/ort-wasm-simd-threaded.wasm' + wasmV,
            'ort-wasm-simd.wasm': msg.baseUrl + '/ort-wasm-simd.wasm' + wasmV,
          }
        : (msg.baseUrl + '/');
      // Do NOT use cache:'no-store' — that re-downloads the whole 38MB model on EVERY
      // load (crippling on a public host). Normal caching + the versioned URL gives an
      // instant repeat-load and still guarantees a fresh fetch after a model swap
      // (the earlier stale/partial-cache corruption was a same-URL swap; ?v= fixes
      // that without re-downloading every time).
      var modelUrl = msg.modelUrl || (msg.baseUrl + '/instrument_presence.onnx' + modelV);
      wlog('init: ort loaded, fetching ' + modelUrl);
      var resp = await fetch(modelUrl);
      var buf = await resp.arrayBuffer();
      wlog('model fetched ' + buf.byteLength + ' bytes; creating session…');
      // graphOptimizationLevel 'all' hangs onnxruntime-web 1.18.0 on the v2-core
      // CNN presence graph (fuses Conv/BN into ops the WASM build can't finalize)
      // — the load never resolves → fresh=0 → no suppression. 'basic' loads fine.
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
      });
      wlog('session created OK (inputs=' + (session.inputNames||[]) + ') — ready');
      self.postMessage({ type: 'model-loaded' });
    } catch (err) {
      wlog('model LOAD failed: ' + ((err && err.message) || String(err)));
      self.postMessage({ type: 'model-error', error: (err && err.message) || String(err) });
    }
    return;
  }
  if (msg.type === 'process') {
    if (!session) {
      self.postMessage({ type: 'result', id: msg.id, presence: null });
      return;
    }
    try {
      var stft = stftMag(msg.samples);
      var inputTensor = new ort.Tensor('float32', stft.mag, [1, stft.nFrames, SEP_BINS]);
      // Bind by the session's ACTUAL input name — the 10-class head used "mag",
      // the [1,11] noise head uses "magnitude". Dynamic keeps both working.
      var inName = (session.inputNames && session.inputNames[0]) || 'magnitude';
      var feeds = {}; feeds[inName] = inputTensor;
      var outMap = await session.run(feeds);
      var out = outMap.presence || outMap[Object.keys(outMap)[0]];
      var presence = new Float32Array(out.data);
      if (!self._loggedFirstRun) { self._loggedFirstRun = true; wlog('first inference OK, presence[' + presence.length + '] noise=' + presence[10]); }
      self.postMessage({ type: 'result', id: msg.id, presence: presence }, [presence.buffer]);
    } catch (err) {
      wlog('run failed: ' + ((err && err.message) || String(err)));
      self.postMessage({ type: 'result', id: msg.id, presence: null, error: (err && err.message) || String(err) });
    }
  }
};
