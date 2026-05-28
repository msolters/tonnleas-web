/* eslint-disable no-restricted-globals, no-undef */
/**
 * Basic Pitch transcription worker.
 *
 * Hosts TF.js + the BP graph model off the main thread. Receives audio
 * via postMessage, runs the BP windowed inference loop, returns the
 * raw [frames, onsets, contours] arrays. Main thread does the cheap
 * post-processing (outputToNotesPoly, addPitchBendsToNoteEvents, etc.)
 * which is pure JS and doesn't need to live in the worker.
 *
 * The inference loop is a faithful port of @spotify/basic-pitch's
 * `evaluateModel` — we can't import the npm package directly because
 * it uses ESM imports of `@tensorflow/tfjs` that don't resolve under
 * the classic `importScripts` loader. Re-implementing the ~30 lines
 * of glue is simpler than wiring a bundler step.
 *
 * Protocol:
 *   in : { type: 'init',      modelUrl, wasmPathPrefix }
 *        { type: 'transcribe', id, audio: Float32Array }
 *   out: { type: 'ready' }
 *        { type: 'result',    id, frames, onsets, contours }   (transferable arrays)
 *        { type: 'error',     id, message }
 *
 * Audio is sent as the underlying ArrayBuffer for zero-copy transfer.
 */

// Relative paths so the worker resolves tfjs against ITS OWN URL, not
// the origin root — gh-pages serves under /tonnleas-web/ so an absolute
// `/tfjs/…` would 404. Same import order, just origin-agnostic.
importScripts('tfjs/tf.min.js');
importScripts('tfjs/tf-backend-wasm.min.js');

const OUTPUT_TO_TENSOR_NAME = {
  contours: 'Identity',
  onsets:   'Identity_2',
  frames:   'Identity_1',
};
const AUDIO_SAMPLE_RATE = 22050;
const FFT_HOP = 256;
const ANNOTATIONS_FPS = Math.floor(AUDIO_SAMPLE_RATE / FFT_HOP);
const AUDIO_WINDOW_LENGTH_SECONDS = 2;
const AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH_SECONDS - FFT_HOP;
const N_OVERLAPPING_FRAMES = 30;
const N_OVERLAP_OVER_2 = Math.floor(N_OVERLAPPING_FRAMES / 2);
const OVERLAP_LENGTH_FRAMES = N_OVERLAPPING_FRAMES * FFT_HOP;
const HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LENGTH_FRAMES;

let modelPromise = null;
let backendReady = null;

async function ensureBackend(wasmPathPrefix) {
  if (backendReady) return backendReady;
  backendReady = (async () => {
    try {
      tf.wasm.setWasmPaths(wasmPathPrefix);
    } catch (e) {
      // older WASM backend may use a different API path
      try { tf.setWasmPaths(wasmPathPrefix); } catch (_) {}
    }
    await tf.setBackend('wasm');
    await tf.ready();
    if (_dev) console.log('[bp-worker] tf backend =', tf.getBackend());
  })();
  return backendReady;
}

// Dev flag — set by the main thread on init. False in prod gh-pages
// builds so the worker's diagnostic logs stay off the user's console.
let _dev = false;

function loadModel(modelUrl) {
  if (modelPromise) return modelPromise;
  modelPromise = tf.loadGraphModel(modelUrl);
  return modelPromise;
}

function prepareData(audio) {
  const wavSamples = tf.concat1d([
    tf.zeros([Math.floor(OVERLAP_LENGTH_FRAMES / 2)], 'float32'),
    tf.tensor(audio),
  ]);
  const reshaped = tf.expandDims(
    tf.signal.frame(wavSamples, AUDIO_N_SAMPLES, HOP_SIZE, true, 0), -1,
  );
  wavSamples.dispose();
  return [reshaped, audio.length];
}

function unwrapOutput(result) {
  const cropped = result.slice(
    [0, N_OVERLAP_OVER_2, 0],
    [-1, result.shape[1] - 2 * N_OVERLAP_OVER_2, -1],
  );
  const out = cropped.reshape([cropped.shape[0] * cropped.shape[1], cropped.shape[2]]);
  cropped.dispose();
  return out;
}

async function transcribe(audio) {
  const model = await modelPromise;
  const [reshapedInput, audioOriginalLength] = prepareData(audio);
  const nOutputFramesOriginal = Math.floor(
    audioOriginalLength * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE),
  );
  /** @type {number[][][]} */
  const framesOut = [];
  const onsetsOut = [];
  const contoursOut = [];
  let calculatedFrames = 0;
  const batches = reshapedInput.shape[0];
  for (let i = 0; i < batches; i++) {
    const singleBatch = tf.slice(reshapedInput, i, 1);
    const [resultingFrames, resultingOnsets, resultingContours] = model.execute(
      singleBatch,
      [OUTPUT_TO_TENSOR_NAME.frames, OUTPUT_TO_TENSOR_NAME.onsets, OUTPUT_TO_TENSOR_NAME.contours],
    );
    let uFrames   = unwrapOutput(resultingFrames);
    let uOnsets   = unwrapOutput(resultingOnsets);
    let uContours = unwrapOutput(resultingContours);
    resultingFrames.dispose();
    resultingOnsets.dispose();
    resultingContours.dispose();
    singleBatch.dispose();
    const tmpFrames = uFrames.shape[0];
    if (calculatedFrames >= nOutputFramesOriginal) {
      uFrames.dispose(); uOnsets.dispose(); uContours.dispose();
      continue;
    }
    if (tmpFrames + calculatedFrames >= nOutputFramesOriginal) {
      const framesToKeep = nOutputFramesOriginal - calculatedFrames;
      const sliced = uFrames.slice([0, 0], [framesToKeep, -1]);
      uFrames.dispose(); uFrames = sliced;
      const sOn = uOnsets.slice([0, 0], [framesToKeep, -1]);
      uOnsets.dispose(); uOnsets = sOn;
      const sCo = uContours.slice([0, 0], [framesToKeep, -1]);
      uContours.dispose(); uContours = sCo;
    }
    calculatedFrames += tmpFrames;
    // Pull tensors into JS arrays (Float32Array per row) before dispose.
    const fArr = await uFrames.array();
    const oArr = await uOnsets.array();
    const cArr = await uContours.array();
    uFrames.dispose(); uOnsets.dispose(); uContours.dispose();
    for (const r of fArr) framesOut.push(r);
    for (const r of oArr) onsetsOut.push(r);
    for (const r of cArr) contoursOut.push(r);
  }
  reshapedInput.dispose();
  return { frames: framesOut, onsets: onsetsOut, contours: contoursOut };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      _dev = !!msg.dev;
      await ensureBackend(msg.wasmPathPrefix);
      await loadModel(msg.modelUrl);
      self.postMessage({ type: 'ready' });
      return;
    }
    if (msg.type === 'transcribe') {
      // msg.audio is an ArrayBuffer (transferred). Wrap as Float32Array.
      const audio = new Float32Array(msg.audio);
      const result = await transcribe(audio);
      self.postMessage({ type: 'result', id: msg.id, ...result });
      return;
    }
  } catch (err) {
    self.postMessage({
      type: 'error',
      id: msg && msg.id,
      message: (err && err.message) || String(err),
    });
  }
};
