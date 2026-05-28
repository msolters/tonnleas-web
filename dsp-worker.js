/**
 * DSP + Inference Web Worker — Tonnléas
 *
 * Self-contained DSP pipeline AND ONNX inference running off the main thread.
 * The main thread is completely free for animations, audio capture, and UI.
 *
 * Protocol:
 *   Receives: { type: 'init', chromaFB: ArrayBuffer, baseUrl: string, modelUrl: string }
 *             { type: 'process', id: number, samples: ArrayBuffer, cycle: number }
 *   Sends:    { type: 'ready' }
 *             { type: 'model-loaded' }
 *             { type: 'model-error', error: string }
 *             { type: 'result', id, chroma, rawEnergy, nFrames, ensembleAvg, nClasses, tempo }
 */

// ══════════════════════════════════════════════════════════
// Constants (must match src/constants.ts)
// ══════════════════════════════════════════════════════════
var SAMPLE_RATE = 22050;
var N_FFT = 2048;
var HOP_LENGTH = 512;
var N_CHROMA = 12;
var WINDOW_FRAMES = 344;
var HOP_FRAMES = 86;
var SOFTMAX_TEMP = 0.15;
var MEDIAN_WIDTH = 9;
var PEAK_THRESHOLD = 0.15;
var HPSS_KERNEL = 31;
var MELODY_FREQ_LO = 250;
var MELODY_FREQ_HI = 3500;
var DRONE_WINDOW = 172;

// Ensemble weights
var WEIGHT_STD = 0.50;
var WEIGHT_FG = 0.20;
var WEIGHT_MEL = 0.30;
var WEIGHT_STD_2WAY = 0.50;
var WEIGHT_MEL_2WAY = 0.50;

// ══════════════════════════════════════════════════════════
// FFT — Radix-2 Cooley-Tukey with pre-computed twiddle factors
// ══════════════════════════════════════════════════════════
var NUM_STAGES = Math.log2(N_FFT); // 11 for N=2048
var twiddleRe = new Array(NUM_STAGES);
var twiddleIm = new Array(NUM_STAGES);

for (var s = 0; s < NUM_STAGES; s++) {
  var len = 1 << (s + 1);
  var half = len >> 1;
  twiddleRe[s] = new Float64Array(half);
  twiddleIm[s] = new Float64Array(half);
  var angle = -2 * Math.PI / len;
  for (var j = 0; j < half; j++) {
    twiddleRe[s][j] = Math.cos(angle * j);
    twiddleIm[s][j] = Math.sin(angle * j);
  }
}

function fft(re, im) {
  var n = re.length;
  // Bit-reversal permutation
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      var tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  // Butterfly stages
  for (var s = 0; s < NUM_STAGES; s++) {
    var len = 1 << (s + 1);
    var half = len >> 1;
    var twRe = twiddleRe[s];
    var twIm = twiddleIm[s];
    for (var i = 0; i < n; i += len) {
      for (var j = 0; j < half; j++) {
        var k = i + j + half;
        var ij = i + j;
        var cRe = twRe[j];
        var cIm = twIm[j];
        var tRe = cRe * re[k] - cIm * im[k];
        var tIm = cRe * im[k] + cIm * re[k];
        re[k] = re[ij] - tRe;
        im[k] = im[ij] - tIm;
        re[ij] += tRe;
        im[ij] += tIm;
      }
    }
  }
}

// ══════════════════════════════════════════════════════════
// STFT — Synchronous (no yields, no cancellation)
// ══════════════════════════════════════════════════════════
var hannWindow = null;
var PAD = N_FFT >> 1;
var _paddedBuf = null;
var _reBuf = null;
var _imBuf = null;

function initHannWindow() {
  hannWindow = new Float32Array(N_FFT);
  for (var i = 0; i < N_FFT; i++) {
    hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
  }
}

function computeSTFT(samples) {
  if (!hannWindow) initHannWindow();
  var n = samples.length;
  var nBins = (N_FFT >> 1) + 1;
  var paddedLen = n + 2 * PAD;

  if (!_paddedBuf || _paddedBuf.length < paddedLen) {
    _paddedBuf = new Float32Array(paddedLen);
  }
  var padded = _paddedBuf;
  padded.fill(0, 0, PAD);
  padded.set(samples, PAD);
  padded.fill(0, PAD + n, paddedLen);

  var nFrames = Math.floor((paddedLen - N_FFT) / HOP_LENGTH) + 1;
  if (nFrames <= 0) return null;

  var mag = new Float32Array(nBins * nFrames);

  if (!_reBuf || _reBuf.length < N_FFT) {
    _reBuf = new Float32Array(N_FFT);
    _imBuf = new Float32Array(N_FFT);
  }
  var re = _reBuf;
  var im = _imBuf;
  var hw = hannWindow;

  for (var f = 0; f < nFrames; f++) {
    var offset = f * HOP_LENGTH;
    for (var i = 0; i < N_FFT; i++) {
      re[i] = padded[offset + i] * hw[i];
    }
    im.fill(0, 0, N_FFT);
    fft(re, im);
    for (var b = 0; b < nBins; b++) {
      mag[b * nFrames + f] = re[b] * re[b] + im[b] * im[b];
    }
  }

  return { mag: mag, nFrames: nFrames, nBins: nBins };
}

// ══════════════════════════════════════════════════════════
// HPSS — Synchronous (no async, no yields)
// ══════════════════════════════════════════════════════════
function quickselect(buf, left, right, k) {
  while (left < right) {
    var mid = (left + right) >> 1;
    if (buf[mid] < buf[left]) { var t = buf[left]; buf[left] = buf[mid]; buf[mid] = t; }
    if (buf[right] < buf[left]) { var t = buf[left]; buf[left] = buf[right]; buf[right] = t; }
    if (buf[right] < buf[mid]) { var t = buf[mid]; buf[mid] = buf[right]; buf[right] = t; }
    var pivot = buf[mid];
    var i = left;
    var j = right;
    while (true) {
      while (buf[i] < pivot) i++;
      while (buf[j] > pivot) j--;
      if (i >= j) break;
      var tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp;
      i++;
      j--;
    }
    if (j < k) left = j + 1;
    else right = j;
  }
  return buf[k];
}

var _medianBuf = new Float32Array(HPSS_KERNEL);

function median1d(arr, len, kernel, out) {
  var half = kernel >> 1;
  if (_medianBuf.length < kernel + 1) _medianBuf = new Float32Array(kernel + 1);
  var buf = _medianBuf;
  for (var i = 0; i < len; i++) {
    var start = Math.max(0, i - half);
    var end = Math.min(len - 1, i + half);
    var count = end - start + 1;
    for (var j = 0; j < count; j++) buf[j] = arr[start + j];
    var medianIdx = count >> 1;
    out[i] = quickselect(buf, 0, count - 1, medianIdx);
  }
  return out;
}

function hpss(mag, nFrames, nBins) {
  // Harmonic: median along time for each frequency bin
  var harmonic = new Float32Array(nBins * nFrames);
  var medOutH = new Float32Array(Math.max(nFrames, nBins));
  for (var b = 0; b < nBins; b++) {
    var row = mag.subarray(b * nFrames, b * nFrames + nFrames);
    median1d(row, nFrames, HPSS_KERNEL, medOutH);
    harmonic.set(medOutH.subarray(0, nFrames), b * nFrames);
  }

  // Percussive: median along frequency for each time frame
  var percussive = new Float32Array(nBins * nFrames);
  var col = new Float32Array(nBins);
  var medOutP = new Float32Array(nBins);
  for (var f = 0; f < nFrames; f++) {
    for (var b = 0; b < nBins; b++) col[b] = mag[b * nFrames + f];
    median1d(col, nBins, HPSS_KERNEL, medOutP);
    for (var b = 0; b < nBins; b++) percussive[b * nFrames + f] = medOutP[b];
  }

  // Soft mask: H_mask = H^2 / (H^2 + P^2 + eps)
  var harmonicMasked = new Float32Array(nBins * nFrames);
  var eps = 1e-10;
  for (var i = 0; i < harmonicMasked.length; i++) {
    var h2 = harmonic[i] * harmonic[i];
    var p2 = percussive[i] * percussive[i];
    harmonicMasked[i] = mag[i] * h2 / (h2 + p2 + eps);
  }
  return harmonicMasked;
}

// ══════════════════════════════════════════════════════════
// Normalize — medianFilter, peakNormalize, softmaxNormalize
// ══════════════════════════════════════════════════════════
var _mfOut = null;
var _mfRow = null;
var _mfMedOut = null;

function medianFilter(chroma, nFrames) {
  var len = chroma.length;
  if (!_mfOut || _mfOut.length < len) _mfOut = new Float32Array(len);
  if (!_mfRow || _mfRow.length < nFrames) _mfRow = new Float32Array(nFrames);
  if (!_mfMedOut || _mfMedOut.length < nFrames) _mfMedOut = new Float32Array(nFrames);
  var out = _mfOut;
  var row = _mfRow;
  var medOut = _mfMedOut;

  for (var c = 0; c < N_CHROMA; c++) {
    var base = c * nFrames;
    for (var f = 0; f < nFrames; f++) row[f] = chroma[base + f];
    median1d(row, nFrames, MEDIAN_WIDTH, medOut);
    out.set(medOut.subarray(0, nFrames), base);
  }
  return new Float32Array(out.subarray(0, len));
}

function peakNormalize(chroma, nFrames) {
  for (var f = 0; f < nFrames; f++) {
    var max = 1e-10;
    for (var c = 0; c < N_CHROMA; c++) {
      var v = chroma[c * nFrames + f];
      if (v > max) max = v;
    }
    for (var c = 0; c < N_CHROMA; c++) {
      var idx = c * nFrames + f;
      chroma[idx] /= max;
      if (chroma[idx] < PEAK_THRESHOLD) chroma[idx] *= 0.1;
    }
  }
}

function softmaxNormalize(chroma, nFrames) {
  var out = new Float32Array(chroma.length);
  for (var f = 0; f < nFrames; f++) {
    var max = -Infinity;
    for (var c = 0; c < N_CHROMA; c++) {
      var v = chroma[c * nFrames + f] / SOFTMAX_TEMP;
      if (v > max) max = v;
    }
    var sum = 0;
    for (var c = 0; c < N_CHROMA; c++) {
      var idx = c * nFrames + f;
      var e = Math.exp(chroma[idx] / SOFTMAX_TEMP - max);
      out[idx] = e;
      sum += e;
    }
    for (var c = 0; c < N_CHROMA; c++) {
      out[c * nFrames + f] /= sum;
    }
  }
  return out;
}

// ══════════════════════════════════════════════════════════
// Chromagram — specToChroma, processStandard, removeDrone,
//              processForeground, processMelodyOnly
// ══════════════════════════════════════════════════════════
var chromaFB = null;
var chromaFB_melody = null;

// Pre-allocated buffers for melody-only extraction
var _melLongBuf = null;
var _melDiffBuf = null;
var _melOnsetBuf = null;
var _melGateBuf = null;

// Pre-allocated buffers for removeDrone
var _droneRow = null;
var _droneMedOut = null;

function initFilterBanks(fb) {
  chromaFB = fb;
  var nBins = fb.length / N_CHROMA;
  var minBin = Math.round(MELODY_FREQ_LO * N_FFT / SAMPLE_RATE);
  var maxBin = Math.round(MELODY_FREQ_HI * N_FFT / SAMPLE_RATE);
  chromaFB_melody = new Float32Array(fb.length);
  for (var c = 0; c < N_CHROMA; c++) {
    for (var b = minBin; b <= maxBin && b < nBins; b++) {
      chromaFB_melody[c * nBins + b] = fb[c * nBins + b];
    }
  }
}

// ══════════════════════════════════════════════════════════
// CQT Chromagram — matches librosa.feature.chroma_cqt
// Uses Goertzel algorithm for efficient single-frequency DFT
// ══════════════════════════════════════════════════════════

var CQT_N_OCTAVES = 7;
var CQT_BINS_PER_OCTAVE = 36;  // 3 per semitone, matching librosa chroma_cqt default
var CQT_BINS = CQT_N_OCTAVES * CQT_BINS_PER_OCTAVE;  // 252 bins
var CQT_FMIN = 32.7032;  // C1
var CQT_Q = 1.0 / (Math.pow(2, 1.0 / CQT_BINS_PER_OCTAVE) - 1);  // Q for 36 bins/octave ≈ 51.9

// Pre-compute CQT bin frequencies and window lengths
var cqtFreqs = new Float64Array(CQT_BINS);
var cqtWinLens = new Int32Array(CQT_BINS);

for (var k = 0; k < CQT_BINS; k++) {
  cqtFreqs[k] = CQT_FMIN * Math.pow(2, k / CQT_BINS_PER_OCTAVE);
  cqtWinLens[k] = Math.ceil(CQT_Q * SAMPLE_RATE / cqtFreqs[k]);
}

// Pre-compute Hann windows for each CQT bin
var cqtWindows = new Array(CQT_BINS);
for (var k = 0; k < CQT_BINS; k++) {
  var N = cqtWinLens[k];
  cqtWindows[k] = new Float32Array(N);
  for (var i = 0; i < N; i++) {
    cqtWindows[k][i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N));
  }
}

/**
 * Compute CQT-based chromagram from raw audio samples.
 * Matches librosa.feature.chroma_cqt(y, sr=22050, hop_length=512, n_chroma=12).
 *
 * @param samples Float32Array of audio samples (mono, 22050 Hz)
 * @returns { chroma: Float32Array(12 * nFrames), nFrames: number }
 */
// Pre-compute per-bin normalization: 2 / (N_k * sqrt(N_k))
var cqtNormFactors = new Float64Array(CQT_BINS);
for (var k = 0; k < CQT_BINS; k++) {
  cqtNormFactors[k] = 2.0 / (cqtWinLens[k] * Math.sqrt(cqtWinLens[k]));
}

// Pre-compute sin/cos tables per CQT bin
// Only for bins where N_k <= 4096 (higher freq bins with short windows).
// For low-freq bins with very long windows, compute on the fly to save memory.
var CQT_PRECOMPUTE_LIMIT = 4096;
var cqtCosTable = new Array(CQT_BINS);
var cqtSinTable = new Array(CQT_BINS);
var cqtOmega = new Float64Array(CQT_BINS);
for (var k = 0; k < CQT_BINS; k++) {
  var N_k = cqtWinLens[k];
  cqtOmega[k] = 2 * Math.PI * cqtFreqs[k] / SAMPLE_RATE;
  if (N_k <= CQT_PRECOMPUTE_LIMIT) {
    cqtCosTable[k] = new Float32Array(N_k);
    cqtSinTable[k] = new Float32Array(N_k);
    for (var i = 0; i < N_k; i++) {
      var phase = cqtOmega[k] * i;
      cqtCosTable[k][i] = Math.cos(phase);
      cqtSinTable[k][i] = Math.sin(phase);
    }
  } else {
    cqtCosTable[k] = null;
    cqtSinTable[k] = null;
  }
}

/**
 * Hybrid CQT: uses STFT projection for low-freq bins (long windows),
 * direct DFT for high-freq bins (short windows).
 *
 * For low-freq bins where N_k > N_FFT, we project the STFT onto the
 * CQT kernel in the frequency domain (O(nBins) per bin per frame).
 * For high-freq bins where N_k <= N_FFT, direct DFT is fast enough.
 *
 * This matches librosa's approach (STFT-based CQT) for speed.
 */

// Pre-compute frequency-domain CQT kernels for low-freq bins
// These project STFT bins onto CQT frequencies
var cqtStftKernelRe = new Array(CQT_BINS);
var cqtStftKernelIm = new Array(CQT_BINS);
var cqtUseStft = new Uint8Array(CQT_BINS); // 1 = use STFT projection

var nBinsStft = (N_FFT >> 1) + 1;
for (var k = 0; k < CQT_BINS; k++) {
  if (cqtWinLens[k] > N_FFT) {
    // Low-freq bin: use STFT projection
    // Build the kernel in frequency domain: FFT of the windowed exponential
    // For a bin at frequency f_k, the STFT bin nearest to f_k gets most energy
    cqtUseStft[k] = 1;

    // Simple approach: weighted sum of nearby STFT bins
    // The CQT frequency f_k maps to STFT bin f_k * N_FFT / SR
    var stftBin = cqtFreqs[k] * N_FFT / SAMPLE_RATE;
    var bLow = Math.max(0, Math.floor(stftBin) - 2);
    var bHigh = Math.min(nBinsStft - 1, Math.ceil(stftBin) + 2);

    cqtStftKernelRe[k] = new Float32Array(nBinsStft);
    cqtStftKernelIm[k] = null; // magnitude-only projection

    // Triangular window centered on the CQT frequency
    for (var b = bLow; b <= bHigh; b++) {
      var bFreq = b * SAMPLE_RATE / N_FFT;
      var dist = Math.abs(bFreq - cqtFreqs[k]) / (cqtFreqs[k] / (CQT_Q * 0.5));
      if (dist < 1) {
        cqtStftKernelRe[k][b] = 1 - dist; // triangular weight
      }
    }
  } else {
    cqtUseStft[k] = 0;
  }
}

function computeCQTChroma(samples, stftMag, stftNFrames, stftNBins) {
  var n = samples.length;
  var nFrames = Math.floor(n / HOP_LENGTH);
  if (nFrames <= 0) return null;

  // Use min of STFT frames and computed frames
  if (stftMag && stftNFrames < nFrames) nFrames = stftNFrames;

  var chroma = new Float32Array(N_CHROMA * nFrames);

  for (var f = 0; f < nFrames; f++) {
    var center = f * HOP_LENGTH + (HOP_LENGTH >> 1);

    for (var k = 0; k < CQT_BINS; k++) {
      var magnitude;

      if (cqtUseStft[k] && stftMag) {
        // Low-freq: project from STFT magnitude
        var kernel = cqtStftKernelRe[k];
        var sum = 0;
        for (var b = 0; b < stftNBins; b++) {
          if (kernel[b] > 0) {
            // STFT mag is power (re²+im²), take sqrt for magnitude
            sum += kernel[b] * Math.sqrt(stftMag[b * stftNFrames + f]);
          }
        }
        magnitude = sum * cqtNormFactors[k];
      } else {
        // High-freq: direct DFT
        var N_k = cqtWinLens[k];
        var halfN = N_k >> 1;
        var start = center - halfN;
        var win = cqtWindows[k];
        var cosT = cqtCosTable[k];
        var sinT = cqtSinTable[k];
        var omega = cqtOmega[k];

        var re = 0, im = 0;
        if (cosT) {
          for (var i = 0; i < N_k; i++) {
            var idx = start + i;
            var sample = (idx >= 0 && idx < n) ? samples[idx] * win[i] : 0;
            re += sample * cosT[i];
            im -= sample * sinT[i];
          }
        } else {
          for (var i = 0; i < N_k; i++) {
            var idx = start + i;
            var sample = (idx >= 0 && idx < n) ? samples[idx] * win[i] : 0;
            var phase = omega * i;
            re += sample * Math.cos(phase);
            im -= sample * Math.sin(phase);
          }
        }
        magnitude = Math.sqrt(re * re + im * im) * cqtNormFactors[k];
      }

      var chromaBin = Math.floor((k % CQT_BINS_PER_OCTAVE) * N_CHROMA / CQT_BINS_PER_OCTAVE);
      chroma[chromaBin * nFrames + f] += magnitude;
    }
  }

  return { chroma: chroma, nFrames: nFrames };
}

/**
 * Full CQT chromagram pipeline matching CLAUDE.md Step 1.1:
 * HPSS → CQT chroma → median filter → peak normalize → soft threshold
 *
 * @param samples raw audio (already HPSS-filtered harmonic component)
 * @returns { chroma, nFrames } or null
 */
function processCQTChroma(samples) {
  var result = computeCQTChroma(samples);
  if (!result) return null;
  var chroma = result.chroma;
  var nFrames = result.nFrames;

  // Median filter (1, 9) along time axis
  chroma = medianFilter(chroma, nFrames);

  // Peak normalize per frame
  peakNormalize(chroma, nFrames);

  return { chroma: chroma, nFrames: nFrames };
}

function specToChroma(spec, nFrames, nBins, fb) {
  var filterBank = fb || chromaFB;
  var chroma = new Float32Array(N_CHROMA * nFrames);
  for (var f = 0; f < nFrames; f++) {
    for (var c = 0; c < N_CHROMA; c++) {
      var sum = 0;
      var cBase = c * nBins;
      for (var b = 0; b < nBins; b++) {
        sum += filterBank[cBase + b] * spec[b * nFrames + f];
      }
      chroma[c * nFrames + f] = sum;
    }
  }
  return chroma;
}

function processStandard(mag, nFrames, nBins) {
  var chroma = specToChroma(mag, nFrames, nBins);
  var filtered = medianFilter(chroma, nFrames);
  // rawEnergy: per-frame normalized copy for chromagram display.
  // Without this, web mic gain differences make the chromagram too dark.
  var rawEnergy = new Float32Array(filtered);
  for (var f = 0; f < nFrames; f++) {
    var mx = 1e-10;
    for (var c = 0; c < N_CHROMA; c++) {
      var v = rawEnergy[c * nFrames + f];
      if (v > mx) mx = v;
    }
    var inv = 1 / mx;
    for (var c = 0; c < N_CHROMA; c++) {
      rawEnergy[c * nFrames + f] *= inv;
    }
  }
  peakNormalize(filtered, nFrames);
  return { chroma: filtered, rawEnergy: rawEnergy };
}

function removeDrone(chroma, nFrames) {
  var out = new Float32Array(chroma.length);
  if (!_droneRow || _droneRow.length < nFrames) _droneRow = new Float32Array(nFrames);
  if (!_droneMedOut || _droneMedOut.length < nFrames) _droneMedOut = new Float32Array(nFrames);
  var row = _droneRow;
  var medOut = _droneMedOut;

  for (var c = 0; c < N_CHROMA; c++) {
    var base = c * nFrames;
    for (var f = 0; f < nFrames; f++) row[f] = chroma[base + f];
    median1d(row, nFrames, DRONE_WINDOW, medOut);
    for (var f = 0; f < nFrames; f++) {
      out[base + f] = Math.max(0, chroma[base + f] - medOut[f]);
    }
  }
  return out;
}

/**
 * Chromagram matching the interval model training pipeline (CLAUDE.md Step 1.1):
 *   HPSS harmonic → chroma (STFT-based, approximating CQT) →
 *   median filter (1,9) → peak normalize → soft threshold (0.15 × 0.1)
 *
 * Key differences from processForeground:
 *   - Uses STANDARD filter bank (not melody-range restricted)
 *   - NO drone removal (training pipeline doesn't do this)
 *   - Soft threshold matches training exactly
 */
/**
 * Interval model chroma pipeline (CLAUDE.md Step 1.1):
 *   Raw audio → CQT chroma → median filter → peak normalize → soft threshold
 *
 * Uses direct CQT (not STFT-based) with librosa-matching normalization.
 * HPSS is skipped — the model has an internal melody gate.
 */
/**
 * Interval model chromagram — matches ref_chromagram.js exactly:
 *   STFT → HPSS soft mask → log-frequency chroma mapping → median → peak norm → threshold
 *
 * Key: does NOT use chroma_fb.json. Uses simple log2(f/C1) % 12 mapping
 * with energy summation (squared magnitude → sqrt).
 */
// ── Incremental interval chroma cache ──
// Stores the raw (pre-median, pre-normalize) chroma for the whole session.
// Each cycle, only compute new frames from the HPSS spectrogram.
var _ivlRawCache = null;      // Float32Array [12 × cachedFrames] — raw energy chroma
var _ivlCacheFrames = 0;      // how many frames are cached
var _ivlProcessedCache = null; // Float32Array — fully processed (median + norm + thresh)
var _ivlProcessedFrames = 0;

function resetIntervalChromaCache() {
  _ivlRawCache = null;
  _ivlCacheFrames = 0;
  _ivlProcessedCache = null;
  _ivlProcessedFrames = 0;
}

/**
 * Incremental interval chroma: only computes new frames.
 * Returns the full session chroma (processed).
 */
function processIntervalChromaIncremental(harmonicSpec, nFrames, nBins) {
  var C1 = 440 * Math.pow(2, -4.75);

  // How many new frames to compute?
  var newStart = _ivlCacheFrames;
  var newCount = nFrames - newStart;

  if (newCount <= 0 && _ivlProcessedCache && _ivlProcessedFrames === nFrames) {
    // Nothing new — return cached result
    return { chroma: _ivlProcessedCache, nFrames: nFrames };
  }

  // Grow the raw cache if needed
  if (!_ivlRawCache || _ivlRawCache.length < N_CHROMA * nFrames) {
    var newCache = new Float32Array(N_CHROMA * Math.max(nFrames, 1024));
    if (_ivlRawCache) {
      // Copy existing data (shift to new layout since nFrames changed)
      // Raw cache is [12 × oldFrames], need to re-layout to [12 × newCapacity]
      // Actually, harmonicSpec changes layout each cycle (bin-major with current nFrames).
      // We need to recompute from scratch when nFrames changes total layout.
      // For simplicity, just recompute all when buffer grows.
    }
    _ivlRawCache = newCache;
    newStart = 0;
    newCount = nFrames;
  }

  // Compute raw chroma for new frames
  var rawCache = _ivlRawCache;
  var capacity = Math.floor(rawCache.length / N_CHROMA);

  for (var f = newStart; f < nFrames; f++) {
    // Zero the chroma bins for this frame
    for (var c = 0; c < N_CHROMA; c++) rawCache[c * capacity + f] = 0;

    for (var k = 1; k < nBins; k++) {
      var freq = k * SAMPLE_RATE / N_FFT;
      if (freq < 60 || freq > 5000) continue;
      var pitchClass = Math.round(12 * Math.log2(freq / C1)) % 12;
      pitchClass = ((pitchClass % 12) + 12) % 12;
      var mag = harmonicSpec[k * nFrames + f];
      rawCache[pitchClass * capacity + f] += mag * mag;
    }
    for (var c = 0; c < N_CHROMA; c++) {
      rawCache[c * capacity + f] = Math.sqrt(rawCache[c * capacity + f]);
    }
  }
  _ivlCacheFrames = nFrames;

  // Build compact output [12 × nFrames] from the capacity-sized cache
  var chroma = new Float32Array(N_CHROMA * nFrames);
  for (var c = 0; c < N_CHROMA; c++) {
    chroma.set(rawCache.subarray(c * capacity, c * capacity + nFrames), c * nFrames);
  }

  // Apply median filter + peak normalize + soft threshold
  chroma = medianFilter(chroma, nFrames);
  peakNormalize(chroma, nFrames);
  for (var i = 0; i < chroma.length; i++) {
    if (chroma[i] < 0.15) chroma[i] *= 0.1;
  }

  _ivlProcessedCache = chroma;
  _ivlProcessedFrames = nFrames;

  return { chroma: chroma, nFrames: nFrames };
}

// Legacy non-cached version (kept for reference)
function processIntervalChroma(harmonicSpec, nFrames, nBins) {
  // harmonicSpec is HPSS-filtered magnitude spectrogram [nBins × nFrames]
  var C1 = 440 * Math.pow(2, -4.75); // ~32.7 Hz
  var chroma = new Float32Array(N_CHROMA * nFrames);

  // Map each STFT bin to a chroma class via log2(f/C1)
  for (var f = 0; f < nFrames; f++) {
    for (var k = 1; k < nBins; k++) {
      var freq = k * SAMPLE_RATE / N_FFT;
      if (freq < 60 || freq > 5000) continue; // musical range only

      var pitchClass = Math.round(12 * Math.log2(freq / C1)) % 12;
      pitchClass = ((pitchClass % 12) + 12) % 12; // ensure positive

      // Energy summation (squared magnitude)
      var mag = harmonicSpec[k * nFrames + f];
      chroma[pitchClass * nFrames + f] += mag * mag;
    }
    // Square root for magnitude-like values
    for (var c = 0; c < N_CHROMA; c++) {
      chroma[c * nFrames + f] = Math.sqrt(chroma[c * nFrames + f]);
    }
  }

  // Median filter (9) along time
  chroma = medianFilter(chroma, nFrames);

  // Peak normalize per frame
  peakNormalize(chroma, nFrames);

  // Soft threshold: bins below 0.15 *= 0.1
  for (var i = 0; i < chroma.length; i++) {
    if (chroma[i] < 0.15) chroma[i] *= 0.1;
  }

  return { chroma: chroma, nFrames: nFrames };
}

function processForeground(mag, nFrames, nBins) {
  var harmonicSpec = hpss(mag, nFrames, nBins);
  var chroma = specToChroma(harmonicSpec, nFrames, nBins, chromaFB_melody);
  var deDroned = removeDrone(chroma, nFrames);
  var filtered = medianFilter(deDroned, nFrames);
  peakNormalize(filtered, nFrames);
  return filtered;
}

function processMelodyOnly(stdChroma, mag, nFrames, nBins) {
  var len = N_CHROMA * nFrames;

  // Step 1: Multi-resolution subtraction
  var longFrames = (nFrames + 3) >> 2;
  if (!_melLongBuf || _melLongBuf.length < N_CHROMA * longFrames) {
    _melLongBuf = new Float32Array(N_CHROMA * longFrames);
  }
  var longChroma = _melLongBuf;

  // Downsample: average groups of 4
  for (var c = 0; c < N_CHROMA; c++) {
    var srcRow = c * nFrames;
    var dstRow = c * longFrames;
    for (var lf = 0; lf < longFrames; lf++) {
      var f0 = lf * 4;
      var f1 = Math.min(f0 + 4, nFrames);
      var sum = 0;
      for (var f = f0; f < f1; f++) sum += stdChroma[srcRow + f];
      longChroma[dstRow + lf] = sum / (f1 - f0);
    }
  }

  // Subtraction + upsample in one pass
  var melody = new Float32Array(len);
  for (var c = 0; c < N_CHROMA; c++) {
    var srcRow = c * nFrames;
    var longRow = c * longFrames;
    for (var f = 0; f < nFrames; f++) {
      var lPos = f / 4;
      var li = lPos | 0;
      var lj = Math.min(li + 1, longFrames - 1);
      var frac = lPos - li;
      var longVal = longChroma[longRow + li] * (1 - frac) + longChroma[longRow + lj] * frac;
      var diff = stdChroma[srcRow + f] - 0.8 * longVal;
      melody[srcRow + f] = diff > 0 ? diff : 0;
    }
  }

  // Step 2: Temporal derivative blending
  if (!_melDiffBuf || _melDiffBuf.length < len) {
    _melDiffBuf = new Float32Array(len);
  }
  var chromaDiff = _melDiffBuf;

  var diffMax = 0;
  for (var c = 0; c < N_CHROMA; c++) {
    var row = c * nFrames;
    chromaDiff[row] = 0;
    for (var f = 1; f < nFrames; f++) {
      var d = stdChroma[row + f] - stdChroma[row + f - 1];
      if (d < 0) d = -d;
      chromaDiff[row + f] = d;
      if (d > diffMax) diffMax = d;
    }
  }

  if (diffMax > 0) {
    var invDiffMax = 1 / diffMax;
    for (var c = 0; c < N_CHROMA; c++) {
      var row = c * nFrames;
      for (var f = 0; f < nFrames; f++) {
        var idx = row + f;
        melody[idx] = 0.6 * melody[idx]
          + 0.4 * stdChroma[idx] * chromaDiff[idx] * invDiffMax;
      }
    }
  }

  // Step 3: Onset-weighted gating
  if (!_melOnsetBuf || _melOnsetBuf.length < nFrames) {
    _melOnsetBuf = new Float32Array(nFrames);
  }
  if (!_melGateBuf || _melGateBuf.length < nFrames) {
    _melGateBuf = new Float32Array(nFrames);
  }
  var onset = _melOnsetBuf;
  var gate = _melGateBuf;

  var onsetMax = 0;
  onset[0] = 0;
  for (var f = 1; f < nFrames; f++) {
    var flux = 0;
    for (var b = 0; b < nBins; b++) {
      var d = mag[b * nFrames + f] - mag[b * nFrames + f - 1];
      if (d > 0) flux += d;
    }
    onset[f] = flux;
    if (flux > onsetMax) onsetMax = flux;
  }

  if (onsetMax > 0) {
    var invOnsetMax = 1 / onsetMax;
    for (var f = 0; f < nFrames; f++) onset[f] *= invOnsetMax;
  }

  for (var f = 0; f < nFrames; f++) {
    var mx = 0;
    var end = Math.min(f + 6, nFrames);
    for (var j = f; j < end; j++) {
      if (onset[j] > mx) mx = onset[j];
    }
    gate[f] = mx < 0.1 ? 0.1 : mx;
  }

  for (var c = 0; c < N_CHROMA; c++) {
    var row = c * nFrames;
    for (var f = 0; f < nFrames; f++) {
      melody[row + f] *= gate[f];
    }
  }

  // Step 4: Chord penalty
  for (var f = 0; f < nFrames; f++) {
    var fMax = 0;
    for (var c = 0; c < N_CHROMA; c++) {
      var v = melody[c * nFrames + f];
      if (v > fMax) fMax = v;
    }
    if (fMax < 1e-10) continue;

    var active = 0;
    var thresh = fMax * 0.25;
    for (var c = 0; c < N_CHROMA; c++) {
      if (melody[c * nFrames + f] > thresh) active++;
    }

    var penalty = 1.0 - (active - 2) * 0.4;
    if (penalty > 1) penalty = 1;
    if (penalty < 0.2) penalty = 0.2;

    for (var c = 0; c < N_CHROMA; c++) {
      melody[c * nFrames + f] *= penalty;
    }
  }

  // Final: median filter + peak normalize
  var filtered = medianFilter(melody, nFrames);
  peakNormalize(filtered, nFrames);
  return filtered;
}

// ══════════════════════════════════════════════════════════
// Prepare Model Inputs
// ══════════════════════════════════════════════════════════
var TENSOR_SIZE = 2 * N_CHROMA * WINDOW_FRAMES;
var CH1_OFFSET = N_CHROMA * WINDOW_FRAMES;

function prepareModelInputs(chroma, nFrames) {
  var chromaSoft = softmaxNormalize(chroma, nFrames);
  var tensors = [];

  if (nFrames < WINDOW_FRAMES) {
    var data = new Float32Array(TENSOR_SIZE);
    for (var c = 0; c < N_CHROMA; c++) {
      var srcOff = c * nFrames;
      var dstOff = c * WINDOW_FRAMES;
      data.set(chromaSoft.subarray(srcOff, srcOff + nFrames), dstOff);
      data[CH1_OFFSET + dstOff] = 0;
      for (var f = 1; f < nFrames; f++) {
        data[CH1_OFFSET + dstOff + f] = data[dstOff + f] - data[dstOff + f - 1];
      }
    }
    tensors.push(data);
  } else {
    for (var start = 0; start <= nFrames - WINDOW_FRAMES; start += HOP_FRAMES) {
      var data = new Float32Array(TENSOR_SIZE);
      for (var c = 0; c < N_CHROMA; c++) {
        var srcOff = c * nFrames + start;
        var dstOff = c * WINDOW_FRAMES;
        data.set(chromaSoft.subarray(srcOff, srcOff + WINDOW_FRAMES), dstOff);
        data[CH1_OFFSET + dstOff] = 0;
        for (var f = 1; f < WINDOW_FRAMES; f++) {
          data[CH1_OFFSET + dstOff + f] = data[dstOff + f] - data[dstOff + f - 1];
        }
      }
      tensors.push(data);
    }
  }

  return tensors;
}

/**
 * Prepare model inputs for fold12 model (d768-9L).
 * Input: 12-bin chroma → cube sharpen → positive diff → 343-frame windows.
 * Output shape per tensor: (12, 343) = 12 × 343 float32.
 */
var FOLD12_WINDOW = 344;
var FOLD12_HOP = 172;
var FOLD12_TENSOR_SIZE = 12 * 344;  // no diff — model takes raw sharpened fold12

/**
 * Prepare fold12 model inputs from 12-bin HCQT consensus-cleaned chroma.
 *
 * The HCQT WASM already does: CQT h1,h2,h3 → consensus → blur → mask f1 → fold 36→12.
 * This function just windows the cleaned chroma into 344-frame chunks.
 *
 * @param chroma12 Float32Array [12 × nFrames] — HCQT cleaned chroma
 * @param nFrames number of frames
 */
function prepareFold12Inputs(chroma12, nFrames) {
  var tensors = [];

  if (nFrames < FOLD12_WINDOW) {
    var padded = new Float32Array(12 * FOLD12_WINDOW);
    for (var c = 0; c < 12; c++) {
      padded.set(chroma12.subarray(c * nFrames, c * nFrames + nFrames), c * FOLD12_WINDOW);
    }
    chroma12 = padded;
    nFrames = FOLD12_WINDOW;
  }

  for (var start = 0; start <= nFrames - FOLD12_WINDOW; start += FOLD12_HOP) {
    var data = new Float32Array(FOLD12_TENSOR_SIZE);
    for (var c = 0; c < 12; c++) {
      data.set(
        chroma12.subarray(c * nFrames + start, c * nFrames + start + FOLD12_WINDOW),
        c * FOLD12_WINDOW
      );
    }
    tensors.push(data);
  }

  return tensors;
}

// ══════════════════════════════════════════════════════════
// Tempo Estimation
// ══════════════════════════════════════════════════════════
var TEMPO_FRAME_LEN = 1024;
var TEMPO_HOP_LEN = 512;
var TEMPO_NBINS = (TEMPO_FRAME_LEN >> 1) + 1;
var _tempoHann = null;
var _tempoRe = null;
var _tempoIm = null;
var _tempoPrevMag = null;

function ensureTempoBuffers() {
  if (!_tempoHann) {
    _tempoHann = new Float32Array(TEMPO_FRAME_LEN);
    for (var i = 0; i < TEMPO_FRAME_LEN; i++) {
      _tempoHann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / TEMPO_FRAME_LEN));
    }
    _tempoRe = new Float32Array(TEMPO_FRAME_LEN);
    _tempoIm = new Float32Array(TEMPO_FRAME_LEN);
    _tempoPrevMag = new Float32Array(TEMPO_NBINS);
  }
}

function estimateTempo(samples) {
  ensureTempoBuffers();
  var frameLen = TEMPO_FRAME_LEN;
  var hopLen = TEMPO_HOP_LEN;
  var nFrames = Math.floor((samples.length - frameLen) / hopLen) + 1;
  if (nFrames < 10) return null;

  var re = _tempoRe;
  var im = _tempoIm;
  var nBins = TEMPO_NBINS;
  var prevMag = _tempoPrevMag;
  prevMag.fill(0);
  var flux = new Float32Array(nFrames);
  var onsetHann = _tempoHann;

  for (var f = 0; f < nFrames; f++) {
    var offset = f * hopLen;
    for (var i = 0; i < frameLen; i++) {
      re[i] = (offset + i < samples.length) ? samples[offset + i] * onsetHann[i] : 0;
      im[i] = 0;
    }
    fft(re, im);

    var fluxSum = 0;
    for (var b = 0; b < nBins; b++) {
      var mag = Math.sqrt(re[b] * re[b] + im[b] * im[b]);
      var diff = mag - prevMag[b];
      if (diff > 0) fluxSum += diff;
      prevMag[b] = mag;
    }
    flux[f] = fluxSum;
  }

  var maxFlux = 0;
  for (var i = 0; i < nFrames; i++) if (flux[i] > maxFlux) maxFlux = flux[i];
  if (maxFlux < 1e-10) return null;
  for (var i = 0; i < nFrames; i++) flux[i] /= maxFlux;

  // Threshold lowered from 0.15 → 0.06 — Irish trad sessions are usually
  // unaccompanied (fiddle / flute / whistle / pipes) so onset spectral-flux
  // peaks are modest compared to drum-driven music.  Vocal removal also
  // strips some attack transients; even on raw samples (which we now use)
  // a bowed note at uniform dynamics doesn't produce huge flux changes.
  var threshold = 0.06;
  // Also relax local-max from 5-frame to 3-frame so closely spaced notes
  // (sixteenths in a reel: ~0.13 s apart) don't get filtered out.
  var onsets = [];
  for (var i = 1; i < nFrames - 1; i++) {
    if (flux[i] > threshold &&
      flux[i] > flux[i - 1] &&
      flux[i] >= flux[i + 1]) {
      onsets.push(i * hopLen / SAMPLE_RATE);
    }
  }

  if (onsets.length < 4) {
    console.log('[Worker] tempo: only ' + onsets.length + ' onsets passed threshold (need 4+) — flux maxFlux=' + maxFlux.toFixed(4));
    return null;
  }

  var iois = [];
  for (var i = 1; i < onsets.length; i++) {
    var dt = onsets[i] - onsets[i - 1];
    if (dt > 0.08 && dt < 1.5) iois.push(dt);
  }
  if (iois.length < 3) return null;

  var maxLag = Math.min(nFrames, Math.floor(2.0 * SAMPLE_RATE / hopLen));
  var minLag = Math.floor(0.2 * SAMPLE_RATE / hopLen);
  var bestLag = minLag;
  var bestCorr = -Infinity;

  for (var lag = minLag; lag < maxLag && lag < nFrames; lag++) {
    var corr = 0;
    var count = 0;
    for (var i = 0; i < nFrames - lag; i++) {
      corr += flux[i] * flux[i + lag];
      count++;
    }
    corr /= Math.max(count, 1);
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }

  var beatPeriod = bestLag * hopLen / SAMPLE_RATE;
  var bpm = 60.0 / beatPeriod;

  var adjustedBpm = bpm;
  if (adjustedBpm < 60) adjustedBpm *= 2;
  if (adjustedBpm < 60) adjustedBpm *= 2;
  if (adjustedBpm > 250) adjustedBpm /= 2;
  if (adjustedBpm > 250) adjustedBpm /= 2;

  return Math.round(adjustedBpm);
}

// ══════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════
// ONNX Inference (runs inside the worker — main thread is free)
// ══════════════════════════════════════════════════════════
var _onnxSession = null;
var _onnxReady = false;
var _softmaxPool = null;

async function inferWindows(windowTensors) {
  if (!_onnxSession) return [];
  var allProbs = [];
  for (var w = 0; w < windowTensors.length; w++) {
    var input = new self.ort.Tensor('float32', windowTensors[w], [1, 2 * N_CHROMA, WINDOW_FRAMES]);
    var output = await _onnxSession.run({ input: input });
    var logits = output.output.data;
    var nC = logits.length;
    if (!_softmaxPool || _softmaxPool.length !== nC) _softmaxPool = new Float32Array(nC);
    var maxL = -Infinity;
    for (var i = 0; i < nC; i++) if (logits[i] > maxL) maxL = logits[i];
    var sum = 0;
    for (var i = 0; i < nC; i++) {
      _softmaxPool[i] = Math.exp(logits[i] - maxL);
      sum += _softmaxPool[i];
    }
    var invSum = 1 / sum;
    var probs = new Float32Array(nC);
    for (var i = 0; i < nC; i++) probs[i] = _softmaxPool[i] * invSum;
    allProbs.push(probs);
  }
  return allProbs;
}

function averageProbs(allProbs) {
  var n = allProbs.length;
  var nC = allProbs[0].length;
  var avg = new Float32Array(nC);
  for (var p = 0; p < n; p++)
    for (var i = 0; i < nC; i++) avg[i] += allProbs[p][i];
  for (var i = 0; i < nC; i++) avg[i] /= n;
  return avg;
}

function findMax(arr) {
  var mx = 0;
  for (var i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
  return mx || 1;
}

async function runEnsemble(tensorsStd, tensorsFg, tensorsMel) {
  if (!_onnxReady) return { avg: new Float32Array(0), nClasses: 0 };
  var hasStd = tensorsStd.length > 0;
  var hasFg = tensorsFg.length > 0;
  var hasMel = tensorsMel.length > 0;

  var probsStd = hasStd ? await inferWindows(tensorsStd) : [];
  var probsFg = hasFg ? await inferWindows(tensorsFg) : [];
  var probsMel = hasMel ? await inferWindows(tensorsMel) : [];

  var avgStd = hasStd && probsStd.length > 0 ? averageProbs(probsStd) : null;
  var avgFg = hasFg && probsFg.length > 0 ? averageProbs(probsFg) : null;
  var avgMel = hasMel && probsMel.length > 0 ? averageProbs(probsMel) : null;

  var ref = avgStd || avgFg || avgMel;
  if (!ref) return { avg: new Float32Array(0), nClasses: 0 };
  var nClasses = ref.length;

  var maxStd = avgStd ? findMax(avgStd) : 1;
  var maxFg = avgFg ? findMax(avgFg) : 1;
  var maxMel = avgMel ? findMax(avgMel) : 1;

  var wStd, wFg, wMel;
  if (hasStd && hasFg && hasMel) { wStd = WEIGHT_STD; wFg = WEIGHT_FG; wMel = WEIGHT_MEL; }
  else if (hasStd && hasMel) { wStd = WEIGHT_STD_2WAY; wFg = 0; wMel = WEIGHT_MEL_2WAY; }
  else if (hasStd && hasFg) { wStd = 0.60; wFg = 0.40; wMel = 0; }
  else if (hasStd) { wStd = 1; wFg = 0; wMel = 0; }
  else if (hasFg) { wStd = 0; wFg = 1; wMel = 0; }
  else { wStd = 0; wFg = 0; wMel = 1; }

  var avg = new Float32Array(nClasses);
  for (var i = 0; i < nClasses; i++) {
    var v = 0;
    if (avgStd) v += wStd * (avgStd[i] / maxStd);
    if (avgFg) v += wFg * (avgFg[i] / maxFg);
    if (avgMel) v += wMel * (avgMel[i] / maxMel);
    avg[i] = v;
  }
  return { avg: avg, nClasses: nClasses };
}

async function loadOnnxModel(baseUrl, modelUrl) {
  try {
    // Load ort runtime
    importScripts(baseUrl + '/ort.min.js');
    // Multi-threaded WASM requires SharedArrayBuffer (cross-origin
    // isolation via COOP/COEP — supplied by coi-serviceworker on GitHub
    // Pages). Use up to 4 threads when available, else single-thread.
    // Degrades gracefully: no SAB → numThreads = 1, no crash.
    var _canThread =
      (typeof self.crossOriginIsolated === 'undefined' || self.crossOriginIsolated) &&
      typeof SharedArrayBuffer !== 'undefined';
    self.ort.env.wasm.numThreads = _canThread
      ? Math.min(4, (self.navigator && self.navigator.hardwareConcurrency) || 1)
      : 1;
    self.ort.env.wasm.wasmPaths = baseUrl + '/';
    // Limit WASM proxy to reduce memory
    self.ort.env.wasm.proxy = false;

    // Create session
    _onnxSession = await self.ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      enableMemPattern: true,     // reuse memory allocation patterns
      enableCpuMemArena: true,    // arena allocator reduces fragmentation
      interOpNumThreads: 1,
      intraOpNumThreads: 1,
    });
    _onnxReady = true;
    // Log memory after model load for debugging iOS crashes
    if (typeof performance !== 'undefined' && performance.memory) {
      var mem = performance.memory;
      console.log('[DSP Worker] Model loaded. JS heap: ' +
        Math.round(mem.usedJSHeapSize / 1048576) + 'MB / ' +
        Math.round(mem.jsHeapSizeLimit / 1048576) + 'MB');
    }
    self.postMessage({ type: 'model-loaded' });
  } catch (err) {
    self.postMessage({ type: 'model-error', error: err.message || String(err) });
  }
}

// ══════════════════════════════════════════════════════════
// Class→Tune aggregation (moved here to avoid 107KB/cycle transfer)
// ══════════════════════════════════════════════════════════
var _classToDense = null;  // Int32Array: classIdx → dense tune index
var _numDenseTunes = 0;
var _OBS_TOP_K = 100;

// Aggregate class probs → per-tune dense probs, return top-K
function aggregateInWorker(classProbs) {
  if (!_classToDense || _numDenseTunes === 0) return null;
  var n = _numDenseTunes;
  // Accumulate per-tune
  var tuneProbs = new Float32Array(n);
  for (var c = 0; c < classProbs.length; c++) {
    var di = _classToDense[c];
    if (di >= 0) tuneProbs[di] += classProbs[c];
  }
  // Find top-K
  var topK = Math.min(_OBS_TOP_K, n);
  var topIdx = new Int32Array(topK);
  var topVal = new Float32Array(topK);
  topIdx.fill(-1);
  var count = 0, minVal = 0, minPos = 0;
  for (var i = 0; i < n; i++) {
    var v = tuneProbs[i];
    if (v <= 0) continue;
    if (count < topK) {
      topIdx[count] = i;
      topVal[count] = v;
      count++;
      if (count === topK) {
        minVal = topVal[0]; minPos = 0;
        for (var j = 1; j < topK; j++) { if (topVal[j] < minVal) { minVal = topVal[j]; minPos = j; } }
      }
    } else if (v > minVal) {
      topIdx[minPos] = i;
      topVal[minPos] = v;
      minVal = topVal[0]; minPos = 0;
      for (var j = 1; j < topK; j++) { if (topVal[j] < minVal) { minVal = topVal[j]; minPos = j; } }
    }
  }
  return { topIdx: topIdx, topVal: topVal, topCount: count, tuneProbs: tuneProbs };
}

// ══════════════════════════════════════════════════════════
// Worker Message Handler
// ══════════════════════════════════════════════════════════
var _lastHpssTime = 0;

self.onmessage = async function(e) {
  var type = e.data.type;
  var id = e.data.id;

  if (type === 'init') {
    var fb = new Float32Array(e.data.chromaFB);
    // Accept class→tune mapping for in-worker aggregation
    if (e.data.classToDense) {
      _classToDense = new Int32Array(e.data.classToDense);
      _numDenseTunes = e.data.numDenseTunes || 0;
      _OBS_TOP_K = e.data.obsTopK || 100;
    }
    initFilterBanks(fb);
    initHannWindow();
    if (e.data.baseUrl !== undefined) self._baseUrl = e.data.baseUrl;
    self.postMessage({ type: 'ready' });
    // Load ONNX model if URLs provided (eager mode — native builds)
    if (e.data.baseUrl !== undefined && e.data.modelUrl) {
      loadOnnxModel(e.data.baseUrl, e.data.modelUrl);
    }
    return;
  }

  if (type === 'load-model') {
    // Deferred model load (web — triggered on first audio)
    if (!_onnxReady && e.data.baseUrl !== undefined && e.data.modelUrl) {
      loadOnnxModel(e.data.baseUrl, e.data.modelUrl);
    }
    return;
  }

  // ── HCQT melody extraction (WASM + cache) ──
  // Loaded lazily on first process call. Cache stores computed chroma
  // indexed by frame count — only new frames get processed.
  if (!self._hcqtLoading && !self._hcqtModule) {
    self._hcqtLoading = true;
    self._hcqtChromaCache = null;
    self._hcqtCachedFrames = 0;
    self._hcqtNovelty = 0;
    // Persistent key profile state (survives across calls)
    self._keyProfileEmaPtr = null;
    self._keyProfileWarmPtr = null;
    importScripts((self._baseUrl || '') + '/hcqt_melody.js');
    if (typeof createHCQTModule === 'function') {
      createHCQTModule({
        locateFile: function(path) {
          return (self._baseUrl || '') + '/' + path;
        }
      }).then(function(mod) {
        self._hcqtModule = mod;
        self._hcqtLoading = false;
        console.log('[Worker] HCQT WASM module loaded');
      }).catch(function(err) {
        self._hcqtLoading = false;
        console.warn('[Worker] HCQT WASM load failed:', err);
      });
    } else {
      self._hcqtLoading = false;
    }
  }

  if (type === 'process') {
    var samples = new Float32Array(e.data.samples);
    // Optional raw (pre-vocal-removal) samples used solely for tempo
    // estimation.  Vocal removal strips drums / percussive transients which
    // is what the spectral-flux onset detector relies on, so without this
    // tempo would be null on every cycle.
    var samplesRawForTempo = e.data.samplesRawForTempo
      ? new Float32Array(e.data.samplesRawForTempo)
      : null;
    var cycle = e.data.cycle;
    var doForeground = _lastHpssTime < 800 || cycle % 3 === 2;

    // ── HCQT 3-way ensemble: all from CQT, no STFT needed ──
    var nFrames = 0;
    var hcqtFrames = 0;
    var ensemble = { avg: new Float32Array(0), nClasses: 0 };

    if (self._hcqtModule) {
      var hcqtT0 = Date.now();
      var mod = self._hcqtModule;
      var nSamples = samples.length;
      var maxFrames = Math.floor(nSamples / HOP_LENGTH);

      // Allocate: 3 chroma outputs (12×T each) + guide (36×T) + masked (36×T)
      // Allocate persistent key profile state (once, reused across calls)
      if (!self._keyProfileEmaPtr) {
        self._keyProfileEmaPtr = mod._malloc(12 * 4);  // 12 floats
        self._keyProfileWarmPtr = mod._malloc(4);       // 1 int
        // Zero-initialize
        for (var ki = 0; ki < 12; ki++) mod.HEAPF32[(self._keyProfileEmaPtr >> 2) + ki] = 0;
        mod.HEAP32[self._keyProfileWarmPtr >> 2] = 0;
      }

      var stdPtr     = mod._malloc(12 * maxFrames * 4);
      var fgPtr      = mod._malloc(12 * maxFrames * 4);
      var melPtr     = mod._malloc(12 * maxFrames * 4);
      var guidePtr   = mod._malloc(36 * maxFrames * 4);
      var maskedPtr  = mod._malloc(36 * maxFrames * 4);
      var noveltyPtr = mod._malloc(4);  // 1 float
      var samplesPtr = mod._malloc(nSamples * 4);

      mod.HEAPF32.set(samples, samplesPtr >> 2);

      // Single WASM call: 3 ensemble chromas + debug outputs + novelty
      var actualFrames = mod._hcqt_melody(
        samplesPtr, nSamples, stdPtr, fgPtr, melPtr, guidePtr, maskedPtr,
        self._keyProfileEmaPtr, self._keyProfileWarmPtr, noveltyPtr
      );
      nFrames = actualFrames;
      hcqtFrames = actualFrames;

      // Read all outputs
      var chromaStd = new Float32Array(mod.HEAPF32.buffer, stdPtr,    12 * nFrames).slice();
      var chromaFg  = new Float32Array(mod.HEAPF32.buffer, fgPtr,     12 * nFrames).slice();
      var chromaMel = new Float32Array(mod.HEAPF32.buffer, melPtr,    12 * nFrames).slice();
      self._hcqtGuideCache  = new Float32Array(mod.HEAPF32.buffer, guidePtr,  36 * nFrames).slice();
      self._hcqtMaskedCache = new Float32Array(mod.HEAPF32.buffer, maskedPtr, 36 * nFrames).slice();
      self._hcqtChromaCache = chromaStd;
      self._hcqtNovelty = mod.HEAPF32[noveltyPtr >> 2];

      mod._free(samplesPtr);
      mod._free(stdPtr);
      mod._free(fgPtr);
      mod._free(melPtr);
      mod._free(guidePtr);
      mod._free(maskedPtr);
      mod._free(noveltyPtr);
      // NOTE: keyProfileEmaPtr and keyProfileWarmPtr are persistent — NOT freed

      var noveltyStr = self._hcqtNovelty > 0.01 ? ' novelty=' + self._hcqtNovelty.toFixed(3) : '';
      console.log('[Worker] HCQT 3-way: ' + nFrames + ' frames in ' + (Date.now() - hcqtT0) + 'ms' + noveltyStr);

      // 3-way ensemble from CQT using fold12 preprocessing (sharpen→diff→rectify)
      // Use the 12-bin HCQT consensus-cleaned chroma with cube sharpen
      var tensorsStd = prepareModelInputs(chromaStd, nFrames);
      var tensorsFg  = prepareModelInputs(chromaFg, nFrames);
      var tensorsMel = prepareModelInputs(chromaMel, nFrames);

      if (_onnxReady) {
        ensemble = await runEnsemble(tensorsStd, tensorsFg, tensorsMel);
      }
    } else {
      // HCQT not loaded yet — can't do fold12 preprocessing without 36-bin data
      // Just compute basic chroma for display, skip inference
      nFrames = Math.floor(samples.length / HOP_LENGTH);
      console.log('[Worker] HCQT not ready, skipping inference');
    }

    // Tempo on raw samples when available — denoised audio has no drums.
    var tempo = estimateTempo(samplesRawForTempo || samples);
    if (tempo === null && samplesRawForTempo) {
      // Diagnostic: confirm we tried raw and it still came back null
      console.log('[Worker] tempo: null even on raw samples (len=' + samplesRawForTempo.length + ')');
    } else if (tempo !== null) {
      console.log('[Worker] tempo: ' + tempo + ' bpm (source=' + (samplesRawForTempo ? 'raw' : 'cleaned') + ')');
    }

    // Transfer results
    var fullChroma  = new Float32Array(self._hcqtChromaCache || new Float32Array(12));
    var fullRawEnergy = new Float32Array(fullChroma);
    var guide  = self._hcqtGuideCache  ? new Float32Array(self._hcqtGuideCache)  : null;
    var masked = self._hcqtMaskedCache ? new Float32Array(self._hcqtMaskedCache) : null;

    var transferList = [fullChroma.buffer, fullRawEnergy.buffer];
    if (guide)  transferList.push(guide.buffer);
    if (masked) transferList.push(masked.buffer);

    self.postMessage({
      type: 'result',
      id: id,
      chroma: fullChroma,
      chromaFrames: hcqtFrames || nFrames,
      rawEnergy: fullRawEnergy,
      hcqtChroma: null,
      hcqtGuide: guide,
      hcqtMasked: masked,
      hcqtFrames: hcqtFrames,
      nFrames: nFrames,
      nClasses: ensemble.nClasses,
      ensembleAvg: ensemble.avg,
      tempo: tempo,
      keyNovelty: self._hcqtNovelty || 0,
    }, transferList);
    return;
  }
};
