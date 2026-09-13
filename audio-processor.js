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
        // Per-channel smoothed RMS for the active-channel downmix (see process).
        this.chanRms = null;
        this.mono = null;

        // Listen for stop signal from main thread
        this.port.onmessage = (e) => {
            if (e.data === 'stop') this.stopped = true;
        };
    }

    // Active-channel downmix. A multi-input USB interface (Scarlett: every
    // physical input is a channel, the mic on ONE of them) must not be averaged
    // 1/N — that buries the live channel ~25 dB under the silent ones, which is
    // exactly the "flat zero input" the native library had before the same
    // rule was patched into it. A channel counts as active when its smoothed
    // RMS is within 20 dB (>= 10%) of the loudest channel; the mono output is
    // the mean of the active channels only, so a single mic comes through at
    // full level and a stereo pair still averages like before.
    downmix(input) {
        const n = input.length;
        const len = input[0].length;
        if (!this.chanRms || this.chanRms.length !== n) this.chanRms = new Float32Array(n);
        if (!this.mono || this.mono.length !== len) this.mono = new Float32Array(len);
        const rms = this.chanRms;
        let max = 0;
        for (let c = 0; c < n; c++) {
            const ch = input[c];
            let acc = 0;
            for (let i = 0; i < len; i++) acc += ch[i] * ch[i];
            // EMA over ~10 blocks (~27 ms at 48 kHz) so a channel doesn't flap
            // in and out of the mix between 128-sample blocks.
            rms[c] = rms[c] * 0.9 + Math.sqrt(acc / len) * 0.1;
            if (rms[c] > max) max = rms[c];
        }
        const threshold = Math.max(max * 0.1, 1e-6);
        const out = this.mono;
        out.fill(0);
        let active = 0;
        for (let c = 0; c < n; c++) {
            if (rms[c] < threshold) continue;
            active++;
            const ch = input[c];
            for (let i = 0; i < len; i++) out[i] += ch[i];
        }
        if (active > 1) {
            const inv = 1 / active;
            for (let i = 0; i < len; i++) out[i] *= inv;
        }
        return out;
    }

    process(inputs) {
        if (this.stopped) return false; // signal AudioWorklet to stop

        const input = inputs[0];
        if (input.length === 0) return true;
        const samples = input.length === 1 ? input[0] : this.downmix(input);
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
