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
var WEIGHT_STD = 0.40;
var WEIGHT_FG = 0.25;
var WEIGHT_MEL = 0.35;
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

  var threshold = 0.15;
  var onsets = [];
  for (var i = 2; i < nFrames - 2; i++) {
    if (flux[i] > threshold &&
      flux[i] > flux[i - 1] && flux[i] > flux[i - 2] &&
      flux[i] >= flux[i + 1] && flux[i] >= flux[i + 2]) {
      onsets.push(i * hopLen / SAMPLE_RATE);
    }
  }

  if (onsets.length < 4) return null;

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
    var input = new self.ort.Tensor('float32', windowTensors[w], [1, 2, N_CHROMA, WINDOW_FRAMES]);
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
    self.ort.env.wasm.numThreads = 1;
    self.ort.env.wasm.wasmPaths = baseUrl + '/';

    // Create session
    _onnxSession = await self.ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    _onnxReady = true;
    self.postMessage({ type: 'model-loaded' });
  } catch (err) {
    self.postMessage({ type: 'model-error', error: err.message || String(err) });
  }
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
    initFilterBanks(fb);
    initHannWindow();
    self.postMessage({ type: 'ready' });
    // Load ONNX model if URLs provided
    if (e.data.baseUrl !== undefined && e.data.modelUrl) {
      loadOnnxModel(e.data.baseUrl, e.data.modelUrl);
    }
    return;
  }

  if (type === 'process') {
    var samples = new Float32Array(e.data.samples);
    var cycle = e.data.cycle;
    var doForeground = _lastHpssTime < 800 || cycle % 3 === 2;

    var stft = computeSTFT(samples);
    if (!stft) {
      self.postMessage({ type: 'result', id: id, error: 'STFT failed' });
      return;
    }
    var mag = stft.mag;
    var nFrames = stft.nFrames;
    var nBins = stft.nBins;

    var stdResult = processStandard(mag, nFrames, nBins);
    var melChroma = processMelodyOnly(stdResult.chroma, mag, nFrames, nBins);

    var tensorsFg = [];
    if (doForeground) {
      var _hpssT0 = Date.now();
      var fgChroma = processForeground(mag, nFrames, nBins);
      _lastHpssTime = Date.now() - _hpssT0;
      tensorsFg = prepareModelInputs(fgChroma, nFrames);
    }

    var tensorsStd = prepareModelInputs(stdResult.chroma, nFrames);
    var tensorsMel = prepareModelInputs(melChroma, nFrames);
    var tempo = !doForeground ? estimateTempo(samples) : null;

    // Run inference in the worker (no main thread blocking)
    var ensemble = { avg: new Float32Array(0), nClasses: 0 };
    if (_onnxReady) {
      ensemble = await runEnsemble(tensorsStd, tensorsFg, tensorsMel);
    }

    // Transfer chroma, rawEnergy, and ensemble avg
    var transferList = [stdResult.chroma.buffer, stdResult.rawEnergy.buffer];
    if (ensemble.avg.buffer.byteLength > 0) transferList.push(ensemble.avg.buffer);

    self.postMessage({
      type: 'result',
      id: id,
      chroma: stdResult.chroma,
      rawEnergy: stdResult.rawEnergy,
      nFrames: nFrames,
      ensembleAvg: ensemble.avg,
      nClasses: ensemble.nClasses,
      tempo: tempo,
    }, transferList);
    return;
  }
};
