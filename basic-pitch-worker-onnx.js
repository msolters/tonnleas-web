/**
 * basic-pitch-worker-onnx.js — WEB Basic Pitch via onnxruntime-web (ORT 1.18, same ort.min.js the
 * #362 worker uses). The A/B alternative to the TF.js worker (basic-pitch-worker.js), enabled by
 * ?bponnx=1. The whole point is PARITY: it runs the SAME tf2onnx-converted graph (basic_pitch.onnx)
 * that the NATIVE iOS/Android path runs (src/search/basic-pitch-session.ts), with byte-identical
 * windowing — so web and native produce the same note transcription, killing the BP engine
 * divergence (TF.js vs ONNX).
 *
 * Message contract is identical to the TF.js worker so the host (basic-pitch-session.web.ts) is
 * engine-agnostic:
 *   in  { type:'init', modelUrl, wasmPathPrefix, dev }   → { type:'ready' }
 *   in  { type:'transcribe', id, audio:ArrayBuffer }      → { type:'result', id, frames, onsets, contours }
 */
/* eslint-disable */

// ── windowing constants (MIRROR src/search/basic-pitch-session.ts EXACTLY) ──
const AUDIO_SAMPLE_RATE = 22050;
const FFT_HOP = 256;
const ANNOTATIONS_FPS = Math.floor(AUDIO_SAMPLE_RATE / FFT_HOP);
const AUDIO_WINDOW_LENGTH_SECONDS = 2;
const AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH_SECONDS - FFT_HOP; // 43844
const N_OVERLAPPING_FRAMES = 30;
const N_OVERLAP_OVER_2 = Math.floor(N_OVERLAPPING_FRAMES / 2);                     // 15
const OVERLAP_LENGTH_FRAMES = N_OVERLAPPING_FRAMES * FFT_HOP;                      // 7680
const HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LENGTH_FRAMES;                          // 36164
const FRAMES_PER_WINDOW = 172;   // model output frames per AUDIO_N_SAMPLES window
const ONNX_FRAMES = 'note', ONNX_ONSETS = 'onset', ONNX_CONTOURS = 'contour';

let _session = null;
let _inputName = null;
let _dev = false;
function _dlog() { if (_dev) { try { console.log.apply(console, arguments); } catch (e) {} } }

// unwrapWindow — MIRROR of native: trim N_OVERLAP_OVER_2 frames off each end, return K-wide rows.
function unwrapWindow(buf, K) {
  const startFrame = N_OVERLAP_OVER_2;
  const endFrame = FRAMES_PER_WINDOW - N_OVERLAP_OVER_2;
  const rows = [];
  for (let t = startFrame; t < endFrame; t++) {
    const row = new Array(K);
    const off = t * K;
    for (let k = 0; k < K; k++) row[k] = buf[off + k];
    rows.push(row);
  }
  return rows;
}

async function transcribe(audio) {
  if (!_session) throw new Error('BP-ONNX not loaded');
  // Pad with OVERLAP_LENGTH_FRAMES/2 zeros up front (mirrors native + the TF.js worker).
  const PAD = Math.floor(OVERLAP_LENGTH_FRAMES / 2);
  const padded = new Float32Array(audio.length + PAD);
  padded.set(audio, PAD);
  const totalLen = padded.length;
  const nWindows = totalLen <= AUDIO_N_SAMPLES
    ? 1
    : Math.ceil((totalLen - AUDIO_N_SAMPLES) / HOP_SIZE) + 1;
  const nOutputFramesOriginal = Math.floor(audio.length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE));

  const framesOut = [], onsetsOut = [], contoursOut = [];
  let calculatedFrames = 0;
  const windowBuf = new Float32Array(AUDIO_N_SAMPLES);
  for (let w = 0; w < nWindows; w++) {
    if (calculatedFrames >= nOutputFramesOriginal) break;
    const start = w * HOP_SIZE;
    windowBuf.fill(0);
    const copyLen = Math.min(AUDIO_N_SAMPLES, totalLen - start);
    if (copyLen > 0) windowBuf.set(padded.subarray(start, start + copyLen));

    const input = new self.ort.Tensor('float32', windowBuf, [1, AUDIO_N_SAMPLES, 1]);
    const out = await _session.run({ [_inputName]: input });
    const frames   = unwrapWindow(out[ONNX_FRAMES].data, 88);
    const onsets   = unwrapWindow(out[ONNX_ONSETS].data, 88);
    const contours = unwrapWindow(out[ONNX_CONTOURS].data, 264);

    let kept = frames.length;
    if (calculatedFrames + kept > nOutputFramesOriginal) kept = nOutputFramesOriginal - calculatedFrames;
    for (let i = 0; i < kept; i++) { framesOut.push(frames[i]); onsetsOut.push(onsets[i]); contoursOut.push(contours[i]); }
    calculatedFrames += kept;
  }
  return { frames: framesOut, onsets: onsetsOut, contours: contoursOut };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      _dev = !!msg.dev;
      const ortBase = msg.ortBase;   // origin where ort.min.js + ort-wasm-simd.wasm live (= baseUrl + '/')
      importScripts(ortBase + 'ort.min.js');
      // Version the ORT wasm for the CDN/browser cache. String wasmPaths is a bare
      // prefix (no room for ?v=), so use the object form (ORT 1.18 indexes it by
      // filename). Empty token → keep the plain prefix (un-versioned, as before).
      const _wv = msg.wasmVersion ? ('?v=' + msg.wasmVersion) : '';
      self.ort.env.wasm.wasmPaths = _wv
        ? {
            'ort-wasm-simd-threaded.wasm': ortBase + 'ort-wasm-simd-threaded.wasm' + _wv,
            'ort-wasm-simd.wasm': ortBase + 'ort-wasm-simd.wasm' + _wv,
          }
        : ortBase;
      self.ort.env.wasm.numThreads = 1;   // single-thread (no SharedArrayBuffer on iOS Safari)
      self.ort.env.wasm.simd = true;
      const t0 = Date.now();
      _session = await self.ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      _inputName = _session.inputNames[0];
      _dlog('[bp-onnx] ready input=' + _inputName + ' outputs=' + _session.outputNames.join(',') + ' in ' + (Date.now() - t0) + 'ms');
      self.postMessage({ type: 'ready' });
      return;
    }
    if (msg.type === 'transcribe') {
      const audio = new Float32Array(msg.audio);
      const t0 = Date.now();
      const result = await transcribe(audio);
      _dlog('[bp-onnx] transcribe ' + (audio.length / AUDIO_SAMPLE_RATE).toFixed(1) + 's → ' + result.frames.length + ' frames in ' + (Date.now() - t0) + 'ms');
      self.postMessage({ type: 'result', id: msg.id, frames: result.frames, onsets: result.onsets, contours: result.contours });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg && msg.id, message: (err && err.message) || String(err) });
  }
};
