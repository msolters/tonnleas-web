/* Model #362 DSP + inference Web Worker.
 *
 * Lives entirely off the main thread:
 *   1. WASM HCQT-fold12 front-end (hcqt_fold12.js — single-source-of-truth C
 *      compiled to wasm-simd128). Resampling included in WASM.
 *   2. onnxruntime-web in this same worker runs model_nokeycanon_fp16.onnx
 *      (362-lr768 FP16, factored low-rank head — 131 MB, R1=58/73). INT8
 *      QDQ (63 MB, R1=60/73) is offline-better and beat FP16 on the
 *      headless eval harness (fake-audio playback of a captured session
 *      WAV) — but LIVE phone-mic-in-room audio has reverb / AGC behavior
 *      / mic-response artifacts the fake-mic path doesn't reproduce, and
 *      INT8's flatter softmax produces confident wrong commits on that
 *      path (e.g. "Silver Spire" mis-IDed mid-Wise-Maid). The stale-lock
 *      decay still helps, but doesn't close the gap on live audio. The
 *      Markov decay in useTuneIdentifier is kept (load-bearing if anyone
 *      ever toggles INT8 via ?model362=int8). ORT-Web 1.18 is pinned for
 *      iOS WebKit; 1.19+ needs SharedArrayBuffer that Safari withholds.
 *      The .wasm execution provider keeps ONNX off the main thread too.
 *   3. KeyCanon rotation applied to each (12, 344) window in-place before
 *      inference (banker's round, bit-exact port of keycanon_reference.py).
 *   4. Per-window softmax → mean across windows → per-tune MAX over class
 *      indices via label_map → canonical-redirect.
 *
 * Audio in is buffered: the worker keeps a rolling buffer of up to
 * MAX_BUF_SEC seconds of native-rate samples. Each 'process' call appends
 * a new chunk and re-runs the pipeline over the current buffer (the same
 * trade-off the dashboard's IncrementalHcqtCache uses — small accuracy
 * hit at the trailing edge for sub-realtime latency). No JS in the audio
 * path; resampling, HPSS, CQT, consensus, fold12 all happen inside WASM.
 *
 * Message protocol (all messages from main → worker):
 *   { type: 'init',    baseUrl: string, assetsBase: string, inputSr: number }
 *     baseUrl     — origin (or '') under which hcqt_fold12.js + ort.min.js live.
 *     assetsBase  — origin under which model_nokeycanon_fp16.onnx + json catalogs live.
 *     inputSr     — capture sample rate (e.g. 44100). Hot-path values pre-sized for this.
 *   { type: 'process', samples: ArrayBuffer<Float32>, replaceBuffer?: bool }
 *     replaceBuffer true → discard rolling buffer, start fresh.
 *   { type: 'reset' }   — clear rolling buffer.
 *
 * Messages back (worker → main):
 *   { type: 'ready' }                              once everything's loaded
 *   { type: 'init-error', error }
 *   { type: 'result', topK, dspMs, infMs, nWindows, bufferSec }
 *   { type: 'process-error', error }
 */

'use strict';

const MAX_BUF_SEC = 8;                 // rolling buffer cap (native SR)
const TARGET_SR = 22050;
const N_CHROMA = 12;
const WINDOW_FRAMES = 344;
const TENSOR_SIZE = N_CHROMA * WINDOW_FRAMES;  // 4128 floats / window
const MAX_WINDOWS = 64;                // 8 s / 0.5 s/window stride at 50% overlap
const TOP_K = 5;
const TWO_PI = 2 * Math.PI;

let _baseUrl = '';
let _assetsBase = '';
let _inputSr = 0;
let _wasm = null;
let _ort = null;
let _session = null;
let _inputName = null;
let _outputName = null;
let _labelMap = null;
let _tuneIndex = null;
let _redirects = null;

/* Pre-allocated WASM heap pointers (reused across cycles, freed only on reset). */
let _nativeBufPtr = 0;
let _nativeBufCap = 0;
let _resampledPtr = 0;
let _resampledCap = 0;
let _tensorsPtr = 0;
let _tensorsBytes = MAX_WINDOWS * TENSOR_SIZE * 4;

/* Rolling buffer of native-SR audio (Float32). Lives on the JS heap; we
 * memcpy it into the WASM heap each cycle. Avoids any per-cycle malloc
 * in WASM. */
let _ring = null;
let _ringLen = 0;

function ensureRing() {
    const cap = MAX_BUF_SEC * _inputSr;
    if (!_ring || _ring.length !== cap) {
        _ring = new Float32Array(cap);
        _ringLen = 0;
    }
}

function appendSamples(samples) {
    ensureRing();
    const cap = _ring.length;
    const incoming = samples.length;
    if (incoming >= cap) {
        // Take the most recent `cap` samples and replace the ring entirely.
        _ring.set(samples.subarray(incoming - cap));
        _ringLen = cap;
        return;
    }
    if (_ringLen + incoming <= cap) {
        _ring.set(samples, _ringLen);
        _ringLen += incoming;
        return;
    }
    // Shift older samples to make room.
    const keep = cap - incoming;
    _ring.copyWithin(0, _ringLen - keep, _ringLen);
    _ring.set(samples, keep);
    _ringLen = cap;
}

/* ── KeyCanon (bit-exact port of keycanon_reference.py / scripts/hcqt-ref/keycanon.js) ── */

function bankerRound(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return (f & 1) === 0 ? f : f + 1;
}

function phaseMag(C, T, m) {
    let re = 0, im = 0;
    for (let k = 0; k < 12; k++) {
        const c = Math.cos(-TWO_PI * m * k / 12);
        const s = Math.sin(-TWO_PI * m * k / 12);
        const base = k * T;
        let rowSum = 0;
        for (let t = 0; t < T; t++) rowSum += C[base + t];
        re += c * rowSum;
        im += s * rowSum;
    }
    return { phi: Math.atan2(im, re), mag: Math.sqrt(re * re + im * im + 1e-8) };
}

function applyKeycanonInPlace(C, T) {
    const m1 = phaseMag(C, T, 1);
    const phiEff = m1.mag > 1e-4 ? m1.phi : phaseMag(C, T, 5).phi / 5;
    const kCanon = bankerRound(-phiEff * (12 / TWO_PI));
    const shift = (((kCanon % 12) + 12) % 12);
    if (shift === 0) return;
    const tmp = new Float32Array(C.length);
    for (let k = 0; k < 12; k++) {
        const src = ((k + shift) % 12) * T;
        const dst = k * T;
        for (let t = 0; t < T; t++) tmp[dst + t] = C[src + t];
    }
    C.set(tmp);
}

/* ── Initialization ─────────────────────────────────────────────────── */

async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} → ${r.status}`);
    return r.json();
}

async function init({ baseUrl, assetsBase, inputSr, modelFile }) {
    _baseUrl = baseUrl || '';
    _assetsBase = assetsBase || '';
    _inputSr = inputSr | 0;
    if (_inputSr <= 0) throw new Error(`bad inputSr ${inputSr}`);

    importScripts(`${_baseUrl}/hcqt_fold12.js`);
    importScripts(`${_baseUrl}/ort.min.js`);

    _wasm = await self.createHcqtFold12Module({
        locateFile: (p) => `${_baseUrl}/${p}`,
    });

    self.ort.env.wasm.wasmPaths = `${_baseUrl}/`;
    // Multi-threaded WASM needs SharedArrayBuffer, which only exists when
    // the page is cross-origin isolated (COOP/COEP). The coi-serviceworker
    // shim provides that on GitHub Pages. When it's active we use up to 4
    // threads (clamped to hardware) for a 2-4x faster #362 inference;
    // otherwise we fall back to single-thread — graceful degradation, no
    // crash if SAB is unavailable.
    const _canThread =
        (typeof self.crossOriginIsolated === 'undefined' || self.crossOriginIsolated) &&
        typeof SharedArrayBuffer !== 'undefined';
    self.ort.env.wasm.numThreads = _canThread
        ? Math.min(4, (self.navigator && self.navigator.hardwareConcurrency) || 1)
        : 1;
    if (typeof console !== 'undefined') {
        console.log('[#362 worker] wasm threads=' + self.ort.env.wasm.numThreads +
            ' (crossOriginIsolated=' + (self.crossOriginIsolated === true) + ')');
    }

    // Fetch the model ourselves so we can stream-track download progress
    // and post 'model-progress' messages to the main thread for a real
    // progress bar (otherwise ORT.InferenceSession.create downloads
    // opaquely with no events). The model file is ~63 MB so this matters.
    const modelUrl = `${_assetsBase}/${modelFile || 'model_nokeycanon_int8_qdq.onnx'}`;
    console.log(`[#362 worker] init.modelFile=${modelFile ?? '(undefined)'} → fetching ${modelUrl}`);
    const resp = await fetch(modelUrl, { credentials: 'same-origin' });
    if (!resp.ok) throw new Error(`fetch ${modelUrl} → ${resp.status}`);
    // Sanity gate: Metro / dev server might 200-OK with HTML on a path
    // it doesn't recognize as a static file, which then trips ORT with
    // a confusing "protobuf parsing failed". Fail FAST with the real URL.
    const ct = resp.headers.get('content-type') ?? '';
    if (ct.includes('text/html') || ct.includes('application/json')) {
        throw new Error(`fetch ${modelUrl} returned content-type "${ct}" — not an ONNX model (likely a 404/dev-server fallback)`);
    }
    const total = +(resp.headers.get('content-length') ?? '0');
    const reader = resp.body?.getReader();
    if (!reader) throw new Error('no body stream from model fetch');
    const chunks = [];
    let loaded = 0;
    self.postMessage({ type: 'model-progress', loaded: 0, total });
    let lastPost = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        // Throttle progress events to ~30 Hz max.
        const now = Date.now();
        if (now - lastPost > 33) {
            self.postMessage({ type: 'model-progress', loaded, total });
            lastPost = now;
        }
    }
    self.postMessage({ type: 'model-progress', loaded, total: total || loaded });
    // After download, ORT spends ~1-3 s building the session (parsing the
    // graph, initializing wasm execution provider, allocating tensors).
    // No event surface for that — fire a 'model-warming' so the splash
    // screen can switch the status line during the otherwise-silent gap.
    self.postMessage({ type: 'model-warming' });
    // Concatenate into one ArrayBuffer for ORT.
    const buf = new Uint8Array(loaded);
    {
        let off = 0;
        for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
    }
    _session = await self.ort.InferenceSession.create(
        buf,
        { executionProviders: ['wasm'], graphOptimizationLevel: 'all' },
    );
    _inputName  = _session.inputNames[0];
    _outputName = _session.outputNames[0];

    const [lm, ti, cr] = await Promise.all([
        fetchJson(`${_assetsBase}/label_map.json`),
        fetchJson(`${_assetsBase}/tune_index.json`),
        fetchJson(`${_assetsBase}/canonical_redirects.json`),
    ]);
    _labelMap = lm; _tuneIndex = ti; _redirects = cr;

    /* Pre-allocate WASM scratch sized for MAX_BUF_SEC at inputSr (worst case). */
    _nativeBufCap = MAX_BUF_SEC * _inputSr;
    _nativeBufPtr = _wasm._malloc(_nativeBufCap * 4);
    _resampledCap = _wasm._hcqt_fold12_resample_max_len(_inputSr, _nativeBufCap);
    _resampledPtr = _wasm._malloc(_resampledCap * 4);
    _tensorsPtr   = _wasm._malloc(_tensorsBytes);

    ensureRing();
}

/* ── Per-cycle inference ────────────────────────────────────────────── */

const _tmpProbs = new Float32Array(28491);

// Temperature for softening per-cycle predictions before they hit the
// Markov posterior. #362's cosine-ArcFace head is ×64-scaled — vanilla
// softmax peaks at 50-90 % on top-1 every cycle, including warmup
// mis-picks. The Markov update then yanks the posterior toward the
// (potentially wrong) top each cycle, producing rapid jumping before
// the chain has enough evidence to settle. v6's 3-way ensemble blend
// tempered this naturally; for #362 we apply an explicit softening:
// effectively divide logits by SOFTMAX_T so the per-cycle distribution
// is gentler. Doesn't change the argmax — just gives the Markov chain
// room to integrate evidence over a few cycles instead of overreacting
// to each one.
//
// Tuning history:
//   T=4 → top-1 5-15 %, obsQuality EMA pinned under 0.12 display gate
//   T=2 → top-1 25-50 % in lab tests, but in real-room conditions the
//         per-tune-MAX often dipped into the 5-10 % range, leaving the
//         engine's CONFIDENCE_FLOOR=0.03 gate one bad cycle away from
//         flipping to "noise" and abandoning the lock
//   T=1.5 (current) → top-1 closer to 30-60 %, leaves comfortable
//         headroom above CONFIDENCE_FLOOR so the lock survives the
//         occasional weak cycle that's typical of mid-phrase audio
//         (sustained note, breath, tail of a roll, etc.)
const SOFTMAX_T = 1.5;

function softmaxInPlace(arr) {
    const invT = 1 / SOFTMAX_T;
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        arr[i] *= invT;
        if (arr[i] > max) max = arr[i];
    }
    let sum = 0;
    for (let i = 0; i < arr.length; i++) { arr[i] = Math.exp(arr[i] - max); sum += arr[i]; }
    const inv = 1 / sum;
    for (let i = 0; i < arr.length; i++) arr[i] *= inv;
}

/* Pre-computed dense ordering of canonical tune_ids (ascending). Both the
 * worker's tuneProbs array and the engine's dense-Markov index use this
 * same ordering — main thread reproduces it from tune_index.json.
 *
 * Also pre-builds `_classToDense[class_idx] = dense_tune_idx`, so the
 * per-tune-MAX hot loop is a single 28491-pass with no Map lookups. */
let _denseTuneIds = null;       // Int32Array, length n_tunes (~22410)
let _classToDense = null;       // Int32Array, length n_classes (~28491)
let _nTunes = 0;
let _nClasses = 0;

function buildDenseMaps() {
    if (_denseTuneIds) return;
    const tuneIds = Object.keys(_tuneIndex).map(Number).sort((a, b) => a - b);
    _nTunes = tuneIds.length;
    _denseTuneIds = new Int32Array(tuneIds);
    const tidToDense = new Map();
    for (let i = 0; i < _nTunes; i++) tidToDense.set(tuneIds[i], i);

    const classKeys = Object.keys(_labelMap);
    _nClasses = classKeys.length;
    _classToDense = new Int32Array(_nClasses);
    for (const cStr of classKeys) {
        const ci = +cStr;
        const tid = +_labelMap[cStr];
        const dense = tidToDense.get(tid);
        _classToDense[ci] = dense === undefined ? -1 : dense;
    }
}

/* Compute per-tune-MAX probabilities (dense over canonical tune_ids).
 * Hot loop, called every cycle — no allocations beyond the output. */
function aggregatePerTuneMax(meanProbs, outTuneProbs) {
    outTuneProbs.fill(0);
    for (let ci = 0; ci < _nClasses; ci++) {
        const d = _classToDense[ci];
        if (d < 0) continue;
        const p = meanProbs[ci];
        if (p > outTuneProbs[d]) outTuneProbs[d] = p;
    }
}

/* Walk the dense-per-tune array to pick out the top-K for display.
 * Cheap (~22k pass with no allocations beyond the result). */
function topKFromTuneProbs(tuneProbs, topK) {
    const out = [];
    const minScores = new Float32Array(topK);
    const minIds = new Int32Array(topK);
    let filled = 0;
    let worstAt = 0;
    let worstVal = -1;
    for (let i = 0; i < _nTunes; i++) {
        const v = tuneProbs[i];
        if (filled < topK) {
            minScores[filled] = v;
            minIds[filled] = i;
            filled++;
            if (filled === topK) {
                // Recompute worst slot
                worstVal = minScores[0]; worstAt = 0;
                for (let k = 1; k < topK; k++) if (minScores[k] < worstVal) { worstVal = minScores[k]; worstAt = k; }
            }
            continue;
        }
        if (v > worstVal) {
            minScores[worstAt] = v;
            minIds[worstAt] = i;
            worstVal = minScores[0]; worstAt = 0;
            for (let k = 1; k < topK; k++) if (minScores[k] < worstVal) { worstVal = minScores[k]; worstAt = k; }
        }
    }
    // Sort descending and resolve to canonical metadata
    const idxs = [];
    for (let k = 0; k < filled; k++) idxs.push(k);
    idxs.sort((a, b) => minScores[b] - minScores[a]);
    for (const k of idxs) {
        const tid = _denseTuneIds[minIds[k]];
        const canon = _redirects[String(tid)] ?? tid;
        const meta = _tuneIndex[String(canon)] ?? _tuneIndex[String(tid)] ?? {};
        out.push({ tuneId: canon, score: minScores[k], name: meta.name ?? '?', type: meta.type ?? '?' });
    }
    return out;
}

async function processBuffer() {
    if (_ringLen <= 0) return null;

    // Copy ring contents into the WASM native buffer.
    _wasm.HEAPF32.set(_ring.subarray(0, _ringLen), _nativeBufPtr >> 2);

    const t0 = self.performance ? self.performance.now() : Date.now();
    const nWindows = _wasm._hcqt_fold12_362_native(
        _inputSr,
        _nativeBufPtr, _ringLen,
        _resampledPtr, _resampledCap,
        _tensorsPtr,   MAX_WINDOWS,
    );
    const dspMs = (self.performance ? self.performance.now() : Date.now()) - t0;
    if (nWindows <= 0) return { topK: [], dspMs, infMs: 0, nWindows: 0 };

    // Slice tensors out of the WASM heap (one copy — KeyCanon rotates in-place).
    const tensors = new Float32Array(
        _wasm.HEAPF32.buffer, _tensorsPtr, nWindows * TENSOR_SIZE
    ).slice();

    // PRE-KeyCanon summary chroma (12-bin, averaged over all windows × all
    // frames). Used by the engine's audio-key estimator — KeyCanon rotates
    // the chroma away from the recording's actual key, so we capture this
    // BEFORE applying KeyCanon. Worker emits in G-based fold12 order
    // (bin 0 = G); pipeline.web.ts rotates G→C for the engine's C-based
    // profiles.
    const chromaSummary = new Float32Array(N_CHROMA);
    for (let w = 0; w < nWindows; w++) {
        const wbase = w * TENSOR_SIZE;
        for (let c = 0; c < N_CHROMA; c++) {
            const cbase = wbase + c * WINDOW_FRAMES;
            let s = 0;
            for (let f = 0; f < WINDOW_FRAMES; f++) s += tensors[cbase + f];
            chromaSummary[c] += s;
        }
    }
    {
        const inv = 1 / (nWindows * WINDOW_FRAMES);
        for (let c = 0; c < N_CHROMA; c++) chromaSummary[c] *= inv;
    }

    for (let w = 0; w < nWindows; w++) {
        applyKeycanonInPlace(
            tensors.subarray(w * TENSOR_SIZE, (w + 1) * TENSOR_SIZE),
            WINDOW_FRAMES,
        );
    }

    const t1 = self.performance ? self.performance.now() : Date.now();
    const inputTensor = new self.ort.Tensor('float32', tensors, [nWindows, N_CHROMA, WINDOW_FRAMES]);
    const out = await _session.run({ [_inputName]: inputTensor });
    const infMs = (self.performance ? self.performance.now() : Date.now()) - t1;
    const logits = out[_outputName].data;
    const nClasses = logits.length / nWindows;

    /* ── A/B-probe stats (compression-campaign diagnostic) ──
     * Computed once per cycle, ~28k×nWindows passes (cheap). Tells us
     * whether INT8 vs FP16 differ in (a) raw logit magnitude — calibration
     * bug suspect, (b) per-window softmax peakedness — flatness/margin,
     * (c) per-window argmax agreement — jitter at the source.
     * Log line is parseable JSON for offline diff. */
    let logitAbsSum = 0;
    let logitMaxSum = 0;
    let logitMinSum = 0;
    let winTop1Sum = 0;
    let winTop2Sum = 0;
    let winEntropySum = 0;
    const winArgmaxes = new Int32Array(nWindows);

    /* Per-window softmax → mean across windows + per-window stats */
    const meanProbs = new Float32Array(nClasses);
    for (let w = 0; w < nWindows; w++) {
        const base = w * nClasses;
        let lmin = Infinity, lmax = -Infinity, lAbsSum = 0;
        for (let c = 0; c < nClasses; c++) {
            const v = logits[base + c];
            _tmpProbs[c] = v;
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
            lAbsSum += v < 0 ? -v : v;
        }
        logitAbsSum += lAbsSum / nClasses;
        logitMinSum += lmin;
        logitMaxSum += lmax;
        softmaxInPlace(_tmpProbs);
        /* Top-1 / Top-2 / argmax / entropy of this window's softmax */
        let t1p = -1, t2p = -1, t1i = -1, ent = 0;
        for (let c = 0; c < nClasses; c++) {
            const p = _tmpProbs[c];
            if (p > t1p) { t2p = t1p; t1p = p; t1i = c; }
            else if (p > t2p) { t2p = p; }
            if (p > 1e-12) ent -= p * Math.log(p);
        }
        winTop1Sum += t1p;
        winTop2Sum += t2p;
        winEntropySum += ent;
        winArgmaxes[w] = t1i;
        for (let c = 0; c < nClasses; c++) meanProbs[c] += _tmpProbs[c];
    }
    const invNW = 1 / nWindows;
    for (let c = 0; c < nClasses; c++) meanProbs[c] *= invNW;

    buildDenseMaps();
    const tuneProbs = new Float32Array(_nTunes);
    aggregatePerTuneMax(meanProbs, tuneProbs);
    const topK = topKFromTuneProbs(tuneProbs, TOP_K);

    /* tuneProbs top-1 / top-2 margin — the value that actually feeds
     * the downstream lock decision. */
    let tp1 = -1, tp2 = -1;
    for (let i = 0; i < _nTunes; i++) {
        const v = tuneProbs[i];
        if (v > tp1) { tp2 = tp1; tp1 = v; }
        else if (v > tp2) { tp2 = v; }
    }
    /* Distinct argmaxes across windows = jitter. 1 = perfectly stable. */
    const seen = new Set();
    for (let w = 0; w < nWindows; w++) seen.add(winArgmaxes[w]);
    const jitter = seen.size;

    const stats = {
        nW: nWindows,
        logitAbsMean: +(logitAbsSum * invNW).toFixed(4),
        logitMin: +(logitMinSum * invNW).toFixed(3),
        logitMax: +(logitMaxSum * invNW).toFixed(3),
        winTop1: +(winTop1Sum * invNW).toFixed(4),
        winTop2: +(winTop2Sum * invNW).toFixed(4),
        winMargin: +((winTop1Sum - winTop2Sum) * invNW).toFixed(4),
        winEntropy: +(winEntropySum * invNW).toFixed(3),
        winArgmaxJitter: jitter,
        tuneTop1: +tp1.toFixed(4),
        tuneTop2: +tp2.toFixed(4),
        tuneMargin: +(tp1 - tp2).toFixed(4),
    };
    console.log('[ab-probe] ' + JSON.stringify(stats));

    return { topK, tuneProbs, chromaSummary, dspMs, infMs, nWindows };
}

/* ── Message dispatch ────────────────────────────────────────────────── */

self.onmessage = async (e) => {
    const msg = e.data || {};
    try {
        if (msg.type === 'init') {
            await init(msg);
            self.postMessage({ type: 'ready' });
            return;
        }
        if (msg.type === 'reset') {
            _ringLen = 0;
            self.postMessage({ type: 'reset-ok' });
            return;
        }
        if (msg.type === 'process') {
            if (!_wasm || !_session) throw new Error('not initialized');
            const samples = new Float32Array(msg.samples);
            if (msg.replaceBuffer) _ringLen = 0;
            appendSamples(samples);
            const result = await processBuffer();
            if (!result) { self.postMessage({ type: 'result', topK: [], dspMs: 0, infMs: 0, nWindows: 0, bufferSec: 0 }); return; }
            // Transfer the dense per-tune-probs + chroma summary ArrayBuffers
            // to avoid copies on the wire — the worker drops its references,
            // the main thread receives fresh Float32Arrays of the same data.
            const tuneProbsBuf = result.tuneProbs ? result.tuneProbs.buffer : null;
            const chromaBuf    = result.chromaSummary ? result.chromaSummary.buffer : null;
            const transfer = [];
            if (tuneProbsBuf) transfer.push(tuneProbsBuf);
            if (chromaBuf)    transfer.push(chromaBuf);
            self.postMessage(
                {
                    type: 'result',
                    topK: result.topK,
                    tuneProbs: tuneProbsBuf,
                    chromaSummary: chromaBuf,
                    dspMs: result.dspMs,
                    infMs: result.infMs,
                    nWindows: result.nWindows,
                    bufferSec: _ringLen / _inputSr,
                },
                transfer.length > 0 ? transfer : undefined,
            );
            return;
        }
        self.postMessage({ type: 'unknown', received: msg.type });
    } catch (err) {
        self.postMessage({
            type: msg.type === 'init' ? 'init-error' : 'process-error',
            error: err?.message ?? String(err),
            stack: err?.stack,
        });
    }
};
