// AudioWorklet processor: captures raw PCM samples, resamples if needed,
// and forwards to main thread at the target sample rate.
class PCMProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = options.processorOptions || {};
        this.targetRate = opts.targetRate || 22050;
        this.nativeRate = opts.nativeRate || sampleRate;
        this.ratio = this.nativeRate / this.targetRate;
        this.needsResample = Math.abs(this.ratio - 1) > 0.01;
        this.srcPos = 0; // fractional position in source stream
        this.stopped = false;

        // Listen for stop signal from main thread
        this.port.onmessage = (e) => {
            if (e.data === 'stop') this.stopped = true;
        };
    }

    process(inputs) {
        if (this.stopped) return false; // signal AudioWorklet to stop

        const input = inputs[0];
        if (input.length === 0) return true;
        const samples = input[0];
        if (!samples || samples.length === 0) return true;

        if (!this.needsResample) {
            this.port.postMessage(new Float32Array(samples));
            return true;
        }

        // Linear interpolation downsampling
        // ratio = nativeRate / targetRate (e.g. 48000/44100 ≈ 1.088)
        // For each output sample, advance by `ratio` in the input
        const ratio = this.ratio;
        // Clamp srcPos to non-negative — can go slightly negative due to
        // fractional position tracking across 128-sample blocks
        if (this.srcPos < 0) this.srcPos = 0;

        const outLen = Math.floor((samples.length - this.srcPos) / ratio);
        if (outLen <= 0) {
            this.srcPos -= samples.length;
            return true;
        }

        const out = new Float32Array(outLen);
        let pos = this.srcPos;
        for (let i = 0; i < outLen; i++) {
            const idx = Math.floor(pos);
            const frac = pos - idx;
            // Clamp index to valid range
            const safeIdx = Math.max(0, Math.min(idx, samples.length - 1));
            if (safeIdx + 1 < samples.length) {
                out[i] = samples[safeIdx] * (1 - frac) + samples[safeIdx + 1] * frac;
            } else {
                out[i] = samples[safeIdx];
            }
            pos += ratio;
        }
        // Carry over fractional position for next block
        this.srcPos = pos - samples.length;

        this.port.postMessage(out);
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
