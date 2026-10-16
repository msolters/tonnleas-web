/**
 * Sherpa-ONNX Voice Recognition Worker
 *
 * Runs streaming speech recognition with hotword boosting for tune names.
 * Uses the 20M zipformer transducer model via WASM.
 */

// ── Dev-only logging ──
// A worker has no __DEV__ and is served as written, so it gates on the host
// that served it: a dev machine logs, the public site prints nothing. Mirrors
// WORKER_DEV in presence-worker.js and isDevHost() in src/utils/console-relay.ts.
var WORKER_DEV = (function () {
  try {
    var h = ((self.location && self.location.hostname) || '').toLowerCase().replace(/^\[|\]$/g, '');
    if (h === 'localhost' || h === '127.0.0.1' || h === '::1') return true;
    if (h.slice(-6) === '.local' || h.slice(-7) === '.ts.net') return true;
    var m = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(h);
    if (!m) return false;
    var a = +m[1], b = +m[2];
    return a === 10 || a === 127 || (a === 192 && b === 168) ||
           (a === 172 && b >= 16 && b <= 31) || (a === 169 && b === 254);
  } catch (_) { return false; }
})();
function devLog() { if (WORKER_DEV) console.log.apply(console, arguments); }


var recognizer = null;
var recognizerStream = null;
var baseUrl = '';
var pendingHotwords = '';
// ?v= cache-bust tokens (from ASSET_VERSIONS via init) — Cloudflare edge-caches
// .onnx/.wasm by extension, so both the model files and the wasm need a token.
var _modelV = '';   // sherpa-models/*.onnx
var _wasmV = '';    // sherpa-onnx-wasm-main-asr.wasm

// ── Module configuration — must be set BEFORE importing the glue JS ──
var Module = {
  // Skip the .data file — we load model files manually
  noInitialRun: true,
  // The runtime's own stdout (console.log by default) is dev-only too.
  print: devLog,

  locateFile: function(path) {
    return baseUrl + '/' + path + (path.slice(-5) === '.wasm' ? _wasmV : '');
  },

  setStatus: function(status) {
    if (status) {
      self.postMessage({ type: 'loading', status: status });
    }
  },


  onRuntimeInitialized: function() {
    devLog('[SherpaWorker] WASM runtime initialized');
    // Now Module.FS is available — load model files
    loadModelsAndInit();
  }
};

async function loadModelsAndInit() {
  try {
    // Use Emscripten's FS_createDataFile to write files to the virtual filesystem.
    // Module.FS is not directly exported, but FS_createDataFile is.
    var createFile = Module['FS_createDataFile'];
    if (!createFile) {
      throw new Error('Emscripten FS_createDataFile not available');
    }

    var models = [
      { url: baseUrl + '/sherpa-models/encoder-epoch-99-avg-1.int8.onnx' + _modelV, name: 'encoder.onnx' },
      { url: baseUrl + '/sherpa-models/decoder-epoch-99-avg-1.onnx' + _modelV, name: 'decoder.onnx' },
      { url: baseUrl + '/sherpa-models/joiner-epoch-99-avg-1.int8.onnx' + _modelV, name: 'joiner.onnx' },
      { url: baseUrl + '/sherpa-models/tokens.txt', name: 'tokens.txt' },
    ];

    for (var i = 0; i < models.length; i++) {
      var m = models[i];
      self.postMessage({ type: 'loading', status: 'Downloading ' + m.name + '...' });
      var resp = await fetch(m.url);
      if (!resp.ok) throw new Error('Failed to fetch ' + m.name + ': ' + resp.status);
      var buf = await resp.arrayBuffer();
      var data = new Uint8Array(buf);
      // FS_createDataFile(parent, name, data, canRead, canWrite)
      createFile('/', m.name, data, true, true);
      devLog('[SherpaWorker] Loaded ' + m.name + ' (' + (data.length / 1024 / 1024).toFixed(1) + 'MB)');
    }

    self.postMessage({ type: 'loading', status: 'Creating recognizer...' });

    // Use modified_beam_search for better quality, but skip hotwords
    // for now (the 20M model's tokenizer can't handle arbitrary words
    // without a bpe.vocab file). Our post-processing fuzzy matcher
    // handles tune name matching instead.
    var config = {
      featConfig: { sampleRate: 16000, featureDim: 80 },
      modelConfig: {
        transducer: {
          encoder: '/encoder.onnx',
          decoder: '/decoder.onnx',
          joiner: '/joiner.onnx',
        },
        tokens: '/tokens.txt',
        numThreads: 1,
        provider: 'cpu',
        modelType: '',
        debug: 0,
      },
      decodingMethod: 'modified_beam_search',
      maxActivePaths: 4,
      enableEndpoint: 1,
      rule1MinTrailingSilence: 4.0,   // wait 4s of silence before ending (was 2.4)
      rule2MinTrailingSilence: 2.0,   // wait 2s even mid-phrase (was 1.2)
      rule3MinUtteranceLength: 5,     // min 5 chars before accepting (was 20 frames)
    };

    recognizer = createOnlineRecognizer(Module, config);
    devLog('[SherpaWorker] Recognizer created');
    self.postMessage({ type: 'ready' });

  } catch (err) {
    console.error('[SherpaWorker] Init failed:', err);
    self.postMessage({ type: 'error', message: err.message || String(err) });
  }
}

self.onmessage = function(e) {
  var type = e.data.type;

  if (type === 'init') {
    baseUrl = e.data.baseUrl || '';
    pendingHotwords = e.data.hotwords || '';
    _modelV = e.data.modelVersion ? ('?v=' + e.data.modelVersion) : '';
    _wasmV = e.data.wasmVersion ? ('?v=' + e.data.wasmVersion) : '';

    // Update Module.locateFile with the actual baseUrl (versioning the .wasm)
    Module.locateFile = function(path) {
      return baseUrl + '/' + path + (path.slice(-5) === '.wasm' ? _wasmV : '');
    };

    self.postMessage({ type: 'loading', status: 'Loading WASM...' });

    // Load the sherpa-onnx JS API wrapper
    importScripts(baseUrl + '/sherpa-onnx-asr.js');

    // Prevent the Emscripten glue from trying to download the .data file.
    // The glue checks Module.expectedDataFileDownloads — set it to 0 BEFORE import.
    // Also define a no-op loadPackage to prevent the file unpacker from running.
    Module.expectedDataFileDownloads = 0;
    Module.monitorRunDependencies = function() {};

    // Load the Emscripten glue — this triggers Module.onRuntimeInitialized
    importScripts(baseUrl + '/sherpa-onnx-wasm-main-asr.js');
  }

  else if (type === 'audio') {
    if (!recognizer) return;

    var samples = new Float32Array(e.data.samples);

    if (!recognizerStream) {
      recognizerStream = recognizer.createStream();
    }

    recognizerStream.acceptWaveform(16000, samples);

    while (recognizer.isReady(recognizerStream)) {
      recognizer.decode(recognizerStream);
    }

    var isEndpoint = recognizer.isEndpoint(recognizerStream);
    var result = recognizer.getResult(recognizerStream).text;

    if (result.length > 0) {
      // Only send results with at least 2 words or 6+ chars —
      // skip fragments like "for", "the", "a"
      var wordCount = result.trim().split(/\s+/).length;
      if (isEndpoint && (wordCount >= 2 || result.trim().length >= 6)) {
        self.postMessage({ type: 'result', text: result, isFinal: true });
      } else if (!isEndpoint && wordCount >= 2) {
        // Send partials only if they have substance
        self.postMessage({ type: 'result', text: result, isFinal: false });
      }
    }

    if (isEndpoint) {
      recognizer.reset(recognizerStream);
    }
  }

  else if (type === 'stop') {
    if (recognizerStream && recognizer) {
      recognizer.reset(recognizerStream);
    }
  }
};
