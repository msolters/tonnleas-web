/**
 * Vocal Removal Web Worker — MDX-Net speech/vocal suppression
 *
 * Runs a KUIELab MDX-Net ONNX model to separate vocals from instrumentals.
 * Accepts raw 44.1kHz mono audio, returns cleaned (instrumental) audio.
 *
 * The model operates on STFT spectrograms:
 *   Input:  [1, 4, 2048, 512] — stereo complex STFT (real+imag × L+R)
 *   Output: [1, 4, 2048, 512] — vocal spectrogram (to be subtracted)
 *
 * For mono input, we replicate across both stereo channels.
 *
 * Protocol:
 *   Receives: { type: 'init', baseUrl: string, modelUrl: string }
 *             { type: 'process', id: number, samples: Float32Array (44.1kHz mono) }
 *   Sends:    { type: 'ready' }
 *             { type: 'model-loaded' }
 *             { type: 'model-error', error: string }
 *             { type: 'result', id: number, cleaned: Float32Array (44.1kHz mono) }
 */

// ── MDX-Net STFT Parameters ──
var MDX_SR = 44100;
var MDX_N_FFT = 6144;
var MDX_HOP = 1024;
var MDX_DIM_F = 2048;  // frequency bins kept (from n_fft/2+1 = 3073)
var MDX_DIM_T = 512;   // time frames per chunk
var MDX_CHUNK_SAMPLES = MDX_HOP * (MDX_DIM_T - 1); // 523264 samples ≈ 11.86s

// ── FFT helpers ──
// Radix-2 Cooley-Tukey FFT (in-place, iterative)
var _twiddleRe = null;
var _twiddleIm = null;
var _fftSize = 0;

function initFFT(n) {
  if (_fftSize === n) return;
  _fftSize = n;
  var half = n / 2;
  _twiddleRe = new Float32Array(half);
  _twiddleIm = new Float32Array(half);
  for (var k = 0; k < half; k++) {
    var angle = -2 * Math.PI * k / n;
    _twiddleRe[k] = Math.cos(angle);
    _twiddleIm[k] = Math.sin(angle);
  }
}

function fft(re, im, n) {
  // Bit-reversal permutation
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      var tr = re[i]; re[i] = re[j]; re[j] = tr;
      var ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  // Butterfly
  for (var len = 2; len <= n; len *= 2) {
    var halfLen = len / 2;
    var step = n / len;
    for (var i = 0; i < n; i += len) {
      for (var k = 0; k < halfLen; k++) {
        var idx = k * step;
        var tRe = re[i + k + halfLen] * _twiddleRe[idx] - im[i + k + halfLen] * _twiddleIm[idx];
        var tIm = re[i + k + halfLen] * _twiddleIm[idx] + im[i + k + halfLen] * _twiddleRe[idx];
        re[i + k + halfLen] = re[i + k] - tRe;
        im[i + k + halfLen] = im[i + k] - tIm;
        re[i + k] += tRe;
        im[i + k] += tIm;
      }
    }
  }
}

function ifft(re, im, n) {
  // Conjugate
  for (var i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im, n);
  // Conjugate + scale
  var inv = 1 / n;
  for (var i = 0; i < n; i++) { re[i] *= inv; im[i] = -im[i] * inv; }
}

// ── Hann window ──
var _hannWindow = null;
function getHannWindow(n) {
  if (_hannWindow && _hannWindow.length === n) return _hannWindow;
  _hannWindow = new Float32Array(n);
  for (var i = 0; i < n; i++) {
    _hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / n));
  }
  return _hannWindow;
}

// ── STFT (mono → complex spectrogram) ──
// Returns { re: Float32Array[nBins * nFrames], im: Float32Array[nBins * nFrames], nFrames }
// Layout: bin-major (re[bin * nFrames + frame])
function computeSTFT(samples, nfft, hop) {
  initFFT(nfft);
  var win = getHannWindow(nfft);
  var nBins = nfft / 2 + 1;
  var nFrames = Math.floor((samples.length - nfft) / hop) + 1;
  if (nFrames < 1) nFrames = 1;

  var re = new Float32Array(nBins * nFrames);
  var im = new Float32Array(nBins * nFrames);
  var fftRe = new Float32Array(nfft);
  var fftIm = new Float32Array(nfft);

  for (var f = 0; f < nFrames; f++) {
    var offset = f * hop;
    // Window + load
    for (var i = 0; i < nfft; i++) {
      var idx = offset + i;
      fftRe[i] = idx < samples.length ? samples[idx] * win[i] : 0;
      fftIm[i] = 0;
    }
    fft(fftRe, fftIm, nfft);
    // Store bin-major
    for (var b = 0; b < nBins; b++) {
      re[b * nFrames + f] = fftRe[b];
      im[b * nFrames + f] = fftIm[b];
    }
  }

  return { re: re, im: im, nFrames: nFrames, nBins: nBins };
}

// ── iSTFT (complex spectrogram → mono audio) ──
function computeISTFT(re, im, nBins, nFrames, nfft, hop) {
  initFFT(nfft);
  var win = getHannWindow(nfft);
  var outLen = (nFrames - 1) * hop + nfft;
  var output = new Float32Array(outLen);
  var windowSum = new Float32Array(outLen);
  var fftRe = new Float32Array(nfft);
  var fftIm = new Float32Array(nfft);

  for (var f = 0; f < nFrames; f++) {
    // Load bins (positive frequencies)
    for (var b = 0; b < nBins; b++) {
      fftRe[b] = re[b * nFrames + f];
      fftIm[b] = im[b * nFrames + f];
    }
    // Mirror negative frequencies (conjugate symmetry for real signal)
    for (var b = 1; b < nfft / 2; b++) {
      fftRe[nfft - b] = fftRe[b];
      fftIm[nfft - b] = -fftIm[b];
    }

    ifft(fftRe, fftIm, nfft);

    // Overlap-add with window
    var offset = f * hop;
    for (var i = 0; i < nfft; i++) {
      output[offset + i] += fftRe[i] * win[i];
      windowSum[offset + i] += win[i] * win[i];
    }
  }

  // Normalize by window sum (COLA condition)
  for (var i = 0; i < outLen; i++) {
    if (windowSum[i] > 1e-8) output[i] /= windowSum[i];
  }

  return output;
}

// ── Build model input tensor from STFT ──
// Input shape: [1, 4, DIM_F, DIM_T]
// Channels: [L_real, L_imag, R_real, R_imag] — for mono, replicate L=R
function buildModelInput(stftRe, stftIm, nBins, nFrames) {
  var dimF = MDX_DIM_F;
  var dimT = MDX_DIM_T;
  var tensor = new Float32Array(1 * 4 * dimF * dimT);

  // Crop to dimF bins, pad/slice to dimT frames
  var usedFrames = Math.min(nFrames, dimT);
  var usedBins = Math.min(nBins, dimF);

  for (var b = 0; b < usedBins; b++) {
    for (var f = 0; f < usedFrames; f++) {
      var srcIdx = b * nFrames + f;
      var re = stftRe[srcIdx];
      var im = stftIm[srcIdx];
      // Channel 0: L real, Channel 1: L imag
      // Channel 2: R real, Channel 3: R imag (= L for mono)
      var dstBase = b * dimT + f;
      tensor[0 * dimF * dimT + dstBase] = re;   // ch0: L_re
      tensor[1 * dimF * dimT + dstBase] = im;   // ch1: L_im
      tensor[2 * dimF * dimT + dstBase] = re;   // ch2: R_re (=L)
      tensor[3 * dimF * dimT + dstBase] = im;   // ch3: R_im (=L)
    }
  }

  return tensor;
}

// ── Extract vocal mask from model output and compute instrumental STFT ──
// Model outputs the vocal spectrogram. Subtract from original to get instrumental.
function applyVocalRemoval(origRe, origIm, vocalTensor, nBins, nFrames) {
  var dimF = MDX_DIM_F;
  var dimT = MDX_DIM_T;
  var usedFrames = Math.min(nFrames, dimT);
  var usedBins = Math.min(nBins, dimF);

  // Output: copy of original STFT with vocals subtracted
  var outRe = new Float32Array(origRe.length);
  var outIm = new Float32Array(origIm.length);
  outRe.set(origRe);
  outIm.set(origIm);

  for (var b = 0; b < usedBins; b++) {
    for (var f = 0; f < usedFrames; f++) {
      var srcIdx = b * nFrames + f;
      var dstBase = b * dimT + f;
      // Average L and R channels from model output (they're the same for mono input)
      var vocRe = (vocalTensor[0 * dimF * dimT + dstBase] + vocalTensor[2 * dimF * dimT + dstBase]) * 0.5;
      var vocIm = (vocalTensor[1 * dimF * dimT + dstBase] + vocalTensor[3 * dimF * dimT + dstBase]) * 0.5;
      // Subtract vocal from original
      outRe[srcIdx] -= vocRe;
      outIm[srcIdx] -= vocIm;
    }
  }

  return { re: outRe, im: outIm };
}

// ── ONNX Runtime ──
var ort = null;
var session = null;

self.onmessage = async function(e) {
  var msg = e.data;

  if (msg.type === 'init') {
    try {
      // Load ONNX Runtime
      importScripts(msg.baseUrl + '/ort.min.js');
      ort = self.ort;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = msg.baseUrl + '/';

      self.postMessage({ type: 'ready' });

      // Load model
      var modelUrl = msg.modelUrl || (msg.baseUrl + '/kuielab_a_vocals.onnx');
      var resp = await fetch(modelUrl);
      var buf = await resp.arrayBuffer();
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      self.postMessage({ type: 'model-loaded' });
    } catch (err) {
      self.postMessage({ type: 'model-error', error: err.message || String(err) });
    }
    return;
  }

  if (msg.type === 'process') {
    if (!session) {
      self.postMessage({ type: 'result', id: msg.id, cleaned: msg.samples });
      return;
    }

    try {
      var samples = msg.samples instanceof Float32Array ? msg.samples : new Float32Array(msg.samples);

      // 1. STFT
      var stft = computeSTFT(samples, MDX_N_FFT, MDX_HOP);

      // 2. Build input tensor
      var inputData = buildModelInput(stft.re, stft.im, stft.nBins, stft.nFrames);
      var inputTensor = new ort.Tensor('float32', inputData, [1, 4, MDX_DIM_F, MDX_DIM_T]);

      // 3. Inference — get vocal spectrogram
      var results = await session.run({ input: inputTensor });
      var outputName = session.outputNames[0];
      var vocalTensor = results[outputName].data;

      // 4. Subtract vocals from original STFT
      var instrumental = applyVocalRemoval(stft.re, stft.im, vocalTensor, stft.nBins, stft.nFrames);

      // 5. iSTFT → cleaned audio
      var cleaned = computeISTFT(instrumental.re, instrumental.im, stft.nBins, stft.nFrames, MDX_N_FFT, MDX_HOP);

      // Trim to original length
      if (cleaned.length > samples.length) {
        cleaned = cleaned.subarray(0, samples.length);
      }

      self.postMessage(
        { type: 'result', id: msg.id, cleaned: cleaned },
        [cleaned.buffer]
      );
    } catch (err) {
      console.error('[VocalRemoval] Error:', err);
      // On error, pass through original audio
      self.postMessage({ type: 'result', id: msg.id, cleaned: msg.samples });
    }
    return;
  }
};
