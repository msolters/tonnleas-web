/**
 * Melody Separator Web Worker — InstrumentSeparator-v3-big melody extractor
 *
 * Pipeline:
 *   raw audio (22050 Hz mono)
 *     → STFT (n_fft=1024, hop=256, hann, center=true)
 *     → magnitude  (1, T, 513)
 *     → ONNX melody_extractor → cleaned magnitude (same shape)
 *     → iSTFT with original phase
 *     → cleaned audio (same length, 22050 Hz mono)
 *
 * Protocol mirrors vocal-removal-worker.js so the host integration is
 * the same shape:
 *   Receives:
 *     { type:'init',    baseUrl:string, modelUrl?:string }
 *     { type:'process', id:number, samples:Float32Array (22050Hz mono) }
 *   Sends:
 *     { type:'ready' }
 *     { type:'model-loaded' }
 *     { type:'model-error', error:string }
 *     { type:'result',  id:number, cleaned:Float32Array (22050Hz mono) }
 */

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
function ifft(re, im, n) {
  for (var i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im, n);
  var inv = 1 / n;
  for (var i2 = 0; i2 < n; i2++) { re[i2] *= inv; im[i2] = -im[i2] * inv; }
}

var _hann = null;
function getHann() {
  if (_hann && _hann.length === SEP_N_FFT) return _hann;
  _hann = new Float32Array(SEP_N_FFT);
  for (var i = 0; i < SEP_N_FFT; i++) _hann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / SEP_N_FFT));
  return _hann;
}

// center=true STFT: reflect-pad by N_FFT/2 each side, frame stride hop.
// Returns frame-major arrays: mag/re/im[f * SEP_BINS + b].
function stftCenter(samples) {
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
  var re = new Float32Array(nFrames * SEP_BINS);
  var im = new Float32Array(nFrames * SEP_BINS);
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
      var r = bRe[b], i2 = bIm[b];
      mag[dst + b] = Math.sqrt(r * r + i2 * i2);
      re[dst + b] = r;
      im[dst + b] = i2;
    }
  }
  return { mag: mag, re: re, im: im, nFrames: nFrames };
}

// iSTFT — combine cleaned magnitudes with original phase from re/im,
// hann-window overlap-add, normalise by Σ w².
function istftCenter(mag, origRe, origIm, nFrames, outLen) {
  initFFT(SEP_N_FFT);
  var win = getHann();
  var PAD = SEP_N_FFT >> 1;
  var totalLen = outLen + 2 * PAD;
  var out = new Float32Array(totalLen);
  var wsum = new Float32Array(totalLen);
  var bRe = new Float32Array(SEP_N_FFT);
  var bIm = new Float32Array(SEP_N_FFT);
  for (var f = 0; f < nFrames; f++) {
    var src = f * SEP_BINS;
    for (var b = 0; b < SEP_BINS; b++) {
      var r = origRe[src + b], i2 = origIm[src + b];
      var oldMag = Math.sqrt(r * r + i2 * i2);
      var m = mag[src + b];
      if (oldMag > 1e-9) {
        var sc = m / oldMag;
        bRe[b] = r * sc;
        bIm[b] = i2 * sc;
      } else {
        bRe[b] = m;
        bIm[b] = 0;
      }
    }
    for (var b2 = 1; b2 < SEP_BINS - 1; b2++) {
      bRe[SEP_N_FFT - b2] = bRe[b2];
      bIm[SEP_N_FFT - b2] = -bIm[b2];
    }
    ifft(bRe, bIm, SEP_N_FFT);
    var off = f * SEP_HOP;
    for (var k = 0; k < SEP_N_FFT; k++) {
      out[off + k] += bRe[k] * win[k];
      wsum[off + k] += win[k] * win[k];
    }
  }
  for (var i3 = 0; i3 < totalLen; i3++) {
    if (wsum[i3] > 1e-6) out[i3] /= wsum[i3];
  }
  return out.subarray(PAD, PAD + outLen);
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
      // ?v= cache-bust tokens (from ASSET_VERSIONS via init). Empty → un-versioned.
      var wasmV = msg.wasmVersion ? ('?v=' + msg.wasmVersion) : '';
      var modelV = msg.modelVersion ? ('?v=' + msg.modelVersion) : '';
      ort.env.wasm.wasmPaths = wasmV
        ? {
            'ort-wasm-simd-threaded.wasm': msg.baseUrl + '/ort-wasm-simd-threaded.wasm' + wasmV,
            'ort-wasm-simd.wasm': msg.baseUrl + '/ort-wasm-simd.wasm' + wasmV,
          }
        : (msg.baseUrl + '/');
      var modelUrl = msg.modelUrl || (msg.baseUrl + '/melody_extractor.onnx' + modelV);
      var resp = await fetch(modelUrl);
      var buf = await resp.arrayBuffer();
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      self.postMessage({ type: 'model-loaded' });
    } catch (err) {
      self.postMessage({ type: 'model-error', error: (err && err.message) || String(err) });
    }
    return;
  }
  if (msg.type === 'process') {
    if (!session) {
      self.postMessage({ type: 'result', id: msg.id, cleaned: msg.samples });
      return;
    }
    try {
      var t0 = Date.now();
      var samples = msg.samples;
      var stft = stftCenter(samples);
      var tStft = Date.now() - t0;

      // BYPASS MODE: skip the model, run STFT→iSTFT identity to isolate
      // whether the iSTFT itself is what's degrading the audio.  If
      // identification works correctly with bypass=true, the iSTFT is
      // fine and the model output is the suspect.  If it's still wrong,
      // the iSTFT is the culprit.
      var BYPASS = false;
      if (BYPASS) {
        var cleanedRoundTrip = istftCenter(stft.mag, stft.re, stft.im, stft.nFrames, samples.length);
        if (msg.id < 5 || msg.id % 10 === 0) {
          var rmsIn = 0, rmsOut = 0;
          for (var k = 0; k < samples.length; k++) rmsIn += samples[k] * samples[k];
          for (var k2 = 0; k2 < cleanedRoundTrip.length; k2++) rmsOut += cleanedRoundTrip[k2] * cleanedRoundTrip[k2];
          rmsIn = Math.sqrt(rmsIn / samples.length);
          rmsOut = Math.sqrt(rmsOut / cleanedRoundTrip.length);
          console.log('[Sep][BYPASS] id=' + msg.id + ' frames=' + stft.nFrames
            + ' rms_in=' + rmsIn.toFixed(4) + ' rms_out=' + rmsOut.toFixed(4)
            + ' ratio=' + (rmsIn > 0 ? (rmsOut / rmsIn).toFixed(4) : 'n/a'));
        }
        self.postMessage({ type: 'result', id: msg.id, cleaned: cleanedRoundTrip }, [cleanedRoundTrip.buffer]);
        return;
      }

      var t1 = Date.now();
      var inputTensor = new ort.Tensor('float32', stft.mag, [1, stft.nFrames, SEP_BINS]);
      var outMap = await session.run({ magnitude: inputTensor });
      // Diagnostic: log every output key + its first/last few values + dims
      // so we can verify which tensor is the cleaned magnitude.
      if (msg.id < 3) {
        var info = [];
        for (var k in outMap) {
          var t = outMap[k];
          var d = t && t.data;
          info.push(k + ' [' + (t && t.dims ? t.dims.join(',') : '?') + '] '
            + 'first=' + (d ? Array.prototype.slice.call(d, 0, 3).map(function(x){return x.toFixed(3);}).join(',') : '?')
            + ' sum=' + (d ? Array.prototype.reduce.call(d, function(a, b){return a + b;}, 0).toFixed(2) : '?'));
        }
        console.log('[Sep][outputs id=' + msg.id + '] ' + info.join(' || '));
      }
      var cleanedMag = outMap.melody_magnitude
        ? outMap.melody_magnitude.data
        : outMap[Object.keys(outMap)[0]].data;
      var tSep = Date.now() - t1;

      // ── Adaptive blend ──
      // The v3-big separator was trained on noisy pub mixes and over-
      // suppresses pure-melody inputs (it can knock 25-50 % off the
      // loudest harmonic peaks of a solo whistle).  We don't want to
      // pay that fidelity tax on clean audio.  So blend cleaned ←→ raw
      // by a coefficient driven by SPECTRAL FLATNESS of the input:
      //   • flatness near 0  → peaky spectrum (single instrument /
      //     pure melody) → trust raw, lean lightly on the cleaner
      //   • flatness near 1  → uniform spectrum (noise / chatter / mixed
      //     bleed) → trust the cleaner, lean lightly on raw
      // Flatness = geometric_mean / arithmetic_mean per frame, averaged.
      var flatSum = 0, flatCount = 0;
      for (var ff = 0; ff < stft.nFrames; ff++) {
        var fOff = ff * SEP_BINS;
        var logS = 0, arS = 0;
        for (var fb = 0; fb < SEP_BINS; fb++) {
          var v = stft.mag[fOff + fb];
          if (v < 1e-9) v = 1e-9;
          logS += Math.log(v);
          arS += v;
        }
        var gm = Math.exp(logS / SEP_BINS);
        var am = arS / SEP_BINS;
        if (am > 1e-9) {
          flatSum += gm / am;
          flatCount++;
        }
      }
      var flatness = flatCount > 0 ? flatSum / flatCount : 0;
      // Map flatness → blend (cleaned weight).  Solo melody (flute,
      // whistle, fiddle) ships a flatness around 0.10-0.16 — at those
      // levels we want NO cleaning, because even tiny model-induced
      // chroma shifts are misidentifying the tune.  Ramp doesn't kick
      // in until flatness clears 0.20:
      //   flatness ≤ 0.20  → blend 0.00 (pure raw — bypass model)
      //   flatness ≥ 0.50  → blend 0.85 (mostly cleaned — noisy mix)
      // The model only earns its keep on input that's actually noisy.
      var BLEND;
      if (typeof self.__SEP_BLEND__ === 'number') {
        BLEND = self.__SEP_BLEND__; // manual override
      } else {
        var t = (flatness - 0.20) / (0.50 - 0.20);
        BLEND = Math.max(0, Math.min(0.85, t * 0.85));
      }
      // Bypass entirely when blend is zero — saves the per-bin loop
      // and gives a guaranteed identity reconstruction.
      var blendedMag;
      if (BLEND <= 0) {
        blendedMag = stft.mag;
      } else {
        // Per-bin suppression floor: never let any single bin drop
        // below MIN_KEEP × its input magnitude.  Without this, the
        // model zaps strong harmonic peaks (391 → 0) which trashes
        // chroma.  MIN_KEEP=0.7 means peaks lose at most 30 % to the
        // model.  Quiet noise bins are LOW magnitude so 0.7 × small
        // is still small → noise removal still works for them.
        var MIN_KEEP = (typeof self.__SEP_MIN_KEEP__ === 'number') ? self.__SEP_MIN_KEEP__ : 0.7;
        blendedMag = new Float32Array(cleanedMag.length);
        var rawWeight = 1 - BLEND;
        for (var bi = 0; bi < cleanedMag.length; bi++) {
          var rawV = stft.mag[bi];
          var cleanedV = cleanedMag[bi];
          var floor = MIN_KEEP * rawV;
          var clamped = cleanedV > floor ? cleanedV : floor;
          blendedMag[bi] = BLEND * clamped + rawWeight * rawV;
        }
      }
      var t2 = Date.now();
      var cleaned = istftCenter(blendedMag, stft.re, stft.im, stft.nFrames, samples.length);
      var tIstft = Date.now() - t2;

      // Diagnostics — energy in vs out, plus precise per-bin diff stats.
      if (msg.id < 5 || msg.id % 10 === 0) {
        var rmsIn = 0, rmsOut = 0;
        for (var k = 0; k < samples.length; k++) rmsIn += samples[k] * samples[k];
        for (var k2 = 0; k2 < cleaned.length; k2++) rmsOut += cleaned[k2] * cleaned[k2];
        rmsIn = Math.sqrt(rmsIn / samples.length);
        rmsOut = Math.sqrt(rmsOut / cleaned.length);
        // Bin-wise comparison: how often does output ≠ input?  How big are
        // the per-bin perturbations on average?  Where do they live?
        var nDiff = 0, totalAbsDiff = 0, maxDiff = 0, sumIn = 0, sumOut = 0;
        var lowBandDiff = 0, lowBandIn = 0;
        var midBandDiff = 0, midBandIn = 0;
        var highBandDiff = 0, highBandIn = 0;
        var nOverInput = 0; // how often output > input (cleaning shouldn't add energy)
        var maxFrame = 0, maxBin = 0, maxIn = 0, maxOut = 0;
        // Top-5 outlier bins
        var topAbs = [0, 0, 0, 0, 0];
        var topMeta = [null, null, null, null, null];
        for (var f = 0; f < stft.nFrames; f++) {
          var off = f * SEP_BINS;
          for (var b = 0; b < SEP_BINS; b++) {
            var iv = stft.mag[off + b];
            var ov = cleanedMag[off + b];
            sumIn += iv;
            sumOut += ov;
            var diff = ov - iv;
            var absDiff = Math.abs(diff);
            totalAbsDiff += absDiff;
            if (absDiff > 1e-6) nDiff++;
            if (diff > 0) nOverInput++;
            if (absDiff > maxDiff) {
              maxDiff = absDiff;
              maxFrame = f; maxBin = b; maxIn = iv; maxOut = ov;
            }
            if (absDiff > topAbs[4]) {
              // insertion sort into top-5
              var ins = 4;
              while (ins > 0 && topAbs[ins - 1] < absDiff) {
                topAbs[ins] = topAbs[ins - 1];
                topMeta[ins] = topMeta[ins - 1];
                ins--;
              }
              topAbs[ins] = absDiff;
              topMeta[ins] = { f: f, b: b, iv: iv, ov: ov };
            }
            if (b < 30) { lowBandDiff += absDiff; lowBandIn += iv; }
            else if (b < 100) { midBandDiff += absDiff; midBandIn += iv; }
            else { highBandDiff += absDiff; highBandIn += iv; }
          }
        }
        var totalCells = stft.nFrames * SEP_BINS;
        console.log('[Sep] id=' + msg.id + ' samp_io=' + samples.length + '/' + cleaned.length
          + ' frames=' + stft.nFrames + ' bins=' + SEP_BINS
          + ' flatness=' + flatness.toFixed(3) + ' blend=' + BLEND.toFixed(2)
          + ' (cleaned ' + (BLEND * 100).toFixed(0) + '% / raw ' + ((1 - BLEND) * 100).toFixed(0) + '%)'
          + ' rms_io=' + rmsIn.toFixed(4) + '/' + rmsOut.toFixed(4)
          + ' mag_io=' + (sumIn / totalCells).toFixed(4) + '/' + (sumOut / totalCells).toFixed(4));
        console.log('  diff: n=' + nDiff + '/' + totalCells + ' (' + (100 * nDiff / totalCells).toFixed(1) + '%)'
          + ' avg=' + (totalAbsDiff / totalCells).toFixed(5)
          + ' max=' + maxDiff.toFixed(4)
          + ' nOverInput=' + nOverInput + ' (' + (100 * nOverInput / totalCells).toFixed(1) + '%)');
        console.log('  per-band suppression%: low=' + (100 * lowBandDiff / Math.max(lowBandIn, 1e-9)).toFixed(1)
          + ' mid(whistle)=' + (100 * midBandDiff / Math.max(midBandIn, 1e-9)).toFixed(1)
          + ' high=' + (100 * highBandDiff / Math.max(highBandIn, 1e-9)).toFixed(1));
        console.log('  worst: f=' + maxFrame + ' b=' + maxBin + ' (~' + Math.round(maxBin * 22050 / SEP_N_FFT) + 'Hz)'
          + ' in=' + maxIn.toFixed(4) + ' out=' + maxOut.toFixed(4));
        console.log('  top5 outliers: ' + topMeta.map(function(m){
          return m ? ('f=' + m.f + ' b=' + m.b + ' (' + Math.round(m.b * 22050 / SEP_N_FFT) + 'Hz) ' + m.iv.toFixed(2) + '→' + m.ov.toFixed(2)) : '-';
        }).join(' | '));
      }

      self.postMessage({ type: 'result', id: msg.id, cleaned: cleaned }, [cleaned.buffer]);
    } catch (err) {
      self.postMessage({ type: 'result', id: msg.id, cleaned: msg.samples, error: (err && err.message) || String(err) });
    }
  }
};

self.postMessage({ type: 'ready' });
