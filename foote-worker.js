/**
 * Foote novelty worker — runs the self-similarity / checkerboard / peak-pick
 * math off the main thread so analysis-region scans don't block the UI or DSP.
 *
 * NATIVE PARITY: src/search/foote-analyze.ts is a line-for-line port of analyze()
 * below (React Native has no Worker, so it runs the same math inline). Any change
 * to the algorithm here MUST be mirrored there (and vice-versa) — the two share no
 * code because this file is a static asset that can't import bundled modules.
 *
 * Protocol:
 *   IN  { type: 'analyze', scanId, frames: Float32Array (N*12), timestamps: Float64Array(N),
 *         segments: [{tuneId,startTime,endTime}], windowEndMs }
 *   OUT { type: 'result',  scanId, boundaries, suggestions, gapAbsorptions }
 *
 * frames is bin-major per-frame: frames[f*12 + b].
 */

const KERNEL_HALF = 8;
const NOVELTY_THRESHOLD = 0.15;
const BOUNDARY_TOLERANCE_MS = 2000;
const SIMILARITY_MERGE_THRESH = 0.85;
const MIN_PEAK_DISTANCE = 4;

function cosineSim12(frames, i, j) {
  let dot = 0, magA = 0, magB = 0;
  const a0 = i * 12, b0 = j * 12;
  for (let k = 0; k < 12; k++) {
    const a = frames[a0 + k], b = frames[b0 + k];
    dot += a * b;
    magA += a * a;
    magB += b * b;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 1e-9 ? dot / denom : 0;
}

// Melody-emphasized frame cosine for STRUCTURAL matching. Squares each bin so the
// dominant (melodic) pitch dominates the comparison and quieter ACCOMPANIMENT bins
// contribute less — an in-worker approximation of melody isolation (the clean
// HCQT-consensus melody chroma is native-only, absent on web). Blunts the
// accompaniment-change confound: a reharmonization mid-tune shifts the quiet bins,
// which now barely move the structural affinity.
function cosineSim12Mel(frames, i, j) {
  let dot = 0, magA = 0, magB = 0;
  const a0 = i * 12, b0 = j * 12;
  for (let k = 0; k < 12; k++) {
    const a = frames[a0 + k] * frames[a0 + k], b = frames[b0 + k] * frames[b0 + k];
    dot += a * b;
    magA += a * a;
    magB += b * b;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 1e-9 ? dot / denom : 0;
}

function cosineSimVec(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 1e-9 ? dot / denom : 0;
}

function buildSimilarityMatrix(frames, N) {
  const S = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    S[i * N + i] = 1;
    for (let j = i + 1; j < N; j++) {
      const sim = cosineSim12(frames, i, j);
      S[i * N + j] = sim;
      S[j * N + i] = sim;
    }
  }
  return S;
}

function computeNovelty(S, N) {
  const novelty = new Float32Array(N);
  const K = KERNEL_HALF;
  for (let t = K; t < N - K; t++) {
    let sum = 0, count = 0;
    for (let i = -K; i < K; i++) {
      for (let j = -K; j < K; j++) {
        const row = t + i, col = t + j;
        if (row < 0 || row >= N || col < 0 || col >= N) continue;
        const sign = ((i < 0) === (j < 0)) ? 1 : -1;
        sum += sign * S[row * N + col];
        count++;
      }
    }
    novelty[t] = count > 0 ? Math.max(0, sum / count) : 0;
  }
  return novelty;
}

function pickPeaks(novelty) {
  const peaks = [];
  const N = novelty.length;
  for (let i = MIN_PEAK_DISTANCE; i < N - MIN_PEAK_DISTANCE; i++) {
    if (novelty[i] < NOVELTY_THRESHOLD) continue;
    let isMax = true;
    for (let d = 1; d <= MIN_PEAK_DISTANCE; d++) {
      if (novelty[i - d] >= novelty[i] || novelty[i + d] >= novelty[i]) { isMax = false; break; }
    }
    if (isMax) peaks.push(i);
  }
  return peaks;
}

function regionChromaProfile(frames, N, start, end) {
  const profile = new Float32Array(12);
  const count = Math.min(end, N) - start;
  if (count <= 0) return profile;
  for (let i = start; i < end && i < N; i++) {
    const base = i * 12;
    for (let b = 0; b < 12; b++) profile[b] += frames[base + b];
  }
  let max = 0;
  for (let b = 0; b < 12; b++) {
    profile[b] /= count;
    if (profile[b] > max) max = profile[b];
  }
  if (max > 0) for (let b = 0; b < 12; b++) profile[b] /= max;
  return profile;
}

// ── Amplitude-envelope seams ───────────────────────────────────────────────
// Chroma novelty is BLIND to a same-key tune change (e.g. two Dorian reels):
// the harmonic distribution barely shifts at the join, so no novelty peak and
// the pre-side never looks "dissimilar". But a tune change in a set almost
// always shows in the WAVEFORM — a brief articulation dip (breath/phrase gap)
// at the seam. We smooth the per-frame energy and flag local minima that dip
// well below the surrounding level as candidate boundaries. These supplement
// the chroma peaks in the START-EXTEND search below.
function findEnergySeams(energies, N) {
  if (!energies || N < 7) return [];
  // Smooth with a small moving average to suppress per-frame jitter.
  const sm = new Float32Array(N);
  const H = 2; // ±2 frames ≈ ±200 ms
  for (let i = 0; i < N; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - H); j <= Math.min(N - 1, i + H); j++) { s += energies[j]; c++; }
    sm[i] = s / c;
  }
  const seams = [];
  const LOCAL = 12;             // ±~1.2 s window for the local reference level
  const DIP_FRAC = 0.6;         // seam must dip below 0.6× the local median
  for (let i = H + 1; i < N - H - 1; i++) {
    // Local minimum of the smoothed envelope.
    if (sm[i] > sm[i - 1] || sm[i] > sm[i + 1]) continue;
    // Local reference = median of the surrounding window.
    const lo = Math.max(0, i - LOCAL), hi = Math.min(N, i + LOCAL + 1);
    const win = Array.prototype.slice.call(sm.subarray(lo, hi)).sort((a, b) => a - b);
    const med = win[win.length >> 1];
    if (med > 0 && sm[i] < DIP_FRAC * med) {
      // Depth-weighted strength in [0,1] — deeper dip = stronger seam.
      const strength = Math.min(1, (med - sm[i]) / med);
      seams.push({ idx: i, strength });
    }
  }
  return seams;
}

// Wide-window region-divergence boundary finder. KEEP IN SYNC with
// src/search/foote-analyze.ts. Local novelty (KERNEL_HALF ≈ 0.8 s) is blind to
// a tune change where a good chunk of each tune must be in view; this compares
// MEAN chroma profiles of REGION_MS-wide spans on each side of every candidate
// split and flags local minima below REGION_DIVERGE_SIM as boundaries.
const REGION_FINDER_ON = true;     // additive boundary source; flip to disable
const REGION_MS = 8000;            // profile span on each side of a candidate split
const REGION_STEP_MS = 1000;       // slide step
const REGION_DIVERGE_SIM = 0.80;   // cross-sim at a local min below this = a boundary
const REGION_MIN_GAP_MS = 6000;    // min spacing between region boundaries
function findRegionBoundaries(frames, N, timestamps) {
  const out = [];
  if (!REGION_FINDER_ON || N < 6) return out;
  const span = timestamps[N - 1] - timestamps[0];
  if (span <= 0) return out;
  const msPerFrame = span / (N - 1);
  const rf = Math.round(REGION_MS / msPerFrame);
  const stepF = Math.max(1, Math.round(REGION_STEP_MS / msPerFrame));
  if (rf < 4 || N < rf * 2 + 2) return out;
  const idxs = [];
  const sims = [];
  for (let i = rf; i <= N - rf; i += stepF) {
    const left = regionChromaProfile(frames, N, i - rf, i);
    const right = regionChromaProfile(frames, N, i, i + rf);
    idxs.push(i);
    sims.push(cosineSimVec(left, right));
  }
  let lastTs = -Infinity;
  for (let k = 1; k < sims.length - 1; k++) {
    const s = sims[k];
    if (s >= REGION_DIVERGE_SIM) continue;
    if (s > sims[k - 1] || s > sims[k + 1]) continue;   // local minimum only
    const ts = timestamps[idxs[k]];
    if (ts - lastTs < REGION_MIN_GAP_MS) continue;       // space boundaries out
    out.push({ idx: idxs[k], strength: Math.min(1, (REGION_DIVERGE_SIM - s) / REGION_DIVERGE_SIM) });
    lastTs = ts;
  }
  return out;
}

function analyze(frames, timestamps, segments, windowEndMs, energies) {
  const N = timestamps.length;
  const result = { boundaries: [], suggestions: [], gapAbsorptions: [], trimSuggestions: [], extendSuggestions: [], boundaryMoves: [], xgDiag: [], xgBuild: 'struct-v5-mel' };
  if (N < KERNEL_HALF * 2 + 2) return result;

  const S = buildSimilarityMatrix(frames, N);
  const novelty = computeNovelty(S, N);
  const peakIndices = pickPeaks(novelty);
  result.boundaries = peakIndices.map(i => timestamps[i]);
  const energySeams = findEnergySeams(energies, N);
  const regionDivBounds = findRegionBoundaries(frames, N, timestamps);

  // ── Per-segment Foote-driven boundary refinement ───────────────────────────
  // For every segment overlapping the scan window:
  //   • compute the segment's interior chroma profile (mean of frames inside),
  //   • use the global novelty peaks as candidate boundaries,
  //   • emit a TAIL-TRIM suggestion at the latest peak past the midpoint where
  //     the chroma RIGHT AFTER the peak no longer matches the profile,
  //   • emit a START-EXTEND suggestion at the earliest peak before the segment
  //     start whose chroma RIGHT AFTER it (toward the segment) DOES match the
  //     profile — i.e. pre-lock audio that belongs to the same tune.
  // Both checks use cosineSim against the profile, so silence (zero chroma),
  // chatter (diffuse chroma), and a different tune (different harmonic
  // distribution) all fail and the boundary is honoured.
  const SIM_MATCH = 0.60;     // post-side cosine sim against MAX-POOLED profile
                              // — only used in conjunction with the pre-side
                              // dissimilarity check below, so a moderate
                              // threshold is safe and lets legit same-tune
                              // single-chord frames (which score ~0.65-0.75)
                              // through.
  const SIM_DISSIM = 0.45;    // pre-side must drop BELOW this to count as a
                              // real boundary — silence (sim ~0), noise
                              // (sim ~0.3), and a different-key tune all
                              // qualify; same-tune chord continuation does
                              // NOT (it'd still match the profile).
  const POST_PEAK_FRAMES = 8; // ~800 ms at ~10 fps cache density
  const MAX_EXTEND_MS = 20000;
  const MIN_TRIM_MS = 1000;
  const MIN_EXTEND_MS = 1000;

  // Max-pooled chroma profile = the union of tonal centers in [s, e).
  // Captures every chord the tune visits — a single-chord post-peak frame is
  // a subset of these peaks, so cosine sim stays high even when the tune
  // moves through I/IV/V.  Mean-pooling smears them and yields per-frame sims
  // ~0.5-0.7 which falls below any usable match threshold.
  function maxPooledProfile(framesArr, totalN, startIdx, endIdx) {
    const out = new Float32Array(12);
    for (let i = startIdx; i < endIdx && i < totalN; i++) {
      const base = i * 12;
      for (let b = 0; b < 12; b++) {
        const v = framesArr[base + b];
        if (v > out[b]) out[b] = v;
      }
    }
    return out;
  }

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    let segStartIdx = -1, segEndIdx = -1;
    for (let i = 0; i < N; i++) {
      if (segStartIdx < 0 && timestamps[i] >= seg.startTime) segStartIdx = i;
      if (timestamps[i] >= seg.endTime) { segEndIdx = i; break; }
    }
    if (segEndIdx < 0) segEndIdx = N;
    if (segStartIdx < 0 || segEndIdx - segStartIdx < 4) continue;

    // Build the segment's max-pooled chroma profile.
    const profile = maxPooledProfile(frames, N, segStartIdx, segEndIdx);

    // Tail trim: latest peak past midpoint whose post-peak chroma is
    // DISSIMILAR (cosine < SIM_DISSIM against the max-pooled profile).
    const segMidTs = (seg.startTime + seg.endTime) / 2;
    let bestTrimTs = -1, bestTrimConf = 0;
    for (const p of peakIndices) {
      const pTs = timestamps[p];
      if (pTs < segMidTs) continue;
      if (pTs > seg.endTime + 5_000) break;
      const postEnd = Math.min(N, p + POST_PEAK_FRAMES);
      if (postEnd - p < 3) continue;
      const postProfile = maxPooledProfile(frames, N, p, postEnd);
      const postSim = cosineSimVec(postProfile, profile);
      if (postSim < SIM_DISSIM) {
        const conf = Math.min(1, (SIM_DISSIM - postSim) / SIM_DISSIM * 0.5 + 0.5);
        if (pTs > bestTrimTs) { bestTrimTs = pTs; bestTrimConf = conf; }
      }
    }
    if (bestTrimTs > 0 && seg.endTime - bestTrimTs >= MIN_TRIM_MS) {
      result.trimSuggestions.push({
        segmentIdx: si,
        currentTuneId: seg.tuneId,
        newEndMs: bestTrimTs,
        confidence: bestTrimConf,
      });
    }

    // Start extend: earliest candidate boundary in
    // [seg.startTime - MAX_EXTEND_MS, seg.startTime) that LOOKS LIKE A REAL
    // ONSET — post-peak side matches this tune AND the pre-peak side belongs
    // to something else. Candidates come from TWO sources:
    //   1. chroma novelty peaks — fire when the pre-side is absolutely
    //      dissimilar (preSim < SIM_DISSIM): a different-key tune / silence.
    //   2. amplitude energy seams — fire when chroma is blind (a SAME-KEY tune
    //      change has no novelty peak and the pre-side stays chroma-similar).
    //      The waveform dip marks the join; we still require the new tune to
    //      fit the post-side BETTER than the pre-side (relative margin), so a
    //      mid-tune phrase gap (both sides match equally) is rejected.
    const REL_MARGIN = 0.08;    // post must beat pre by this for an energy seam
    const extendFloor = Math.max(0, seg.startTime - MAX_EXTEND_MS);
    // Merge both candidate sources, ordered by frame index (chronological).
    const extendCands = [];
    for (const p of peakIndices) extendCands.push({ idx: p, seam: 0 });
    for (const s of energySeams) extendCands.push({ idx: s.idx, seam: s.strength });
    for (const r of regionDivBounds) extendCands.push({ idx: r.idx, seam: r.strength });
    extendCands.sort((a, b) => a.idx - b.idx);
    let bestExtendTs = -1, bestExtendConf = 0;
    for (const cand of extendCands) {
      const p = cand.idx;
      const pTs = timestamps[p];
      if (pTs < extendFloor) continue;
      if (pTs >= seg.startTime) break;
      const preStart = Math.max(0, p - POST_PEAK_FRAMES);
      const postEnd = Math.min(N, p + POST_PEAK_FRAMES);
      if (p - preStart < 3 || postEnd - p < 3) continue;
      const preProfile = maxPooledProfile(frames, N, preStart, p);
      const postProfile = maxPooledProfile(frames, N, p, postEnd);
      const preSim = cosineSimVec(preProfile, profile);
      const postSim = cosineSimVec(postProfile, profile);
      if (postSim < SIM_MATCH) continue;
      let conf = 0;
      if (preSim < SIM_DISSIM) {
        // Chroma boundary (original, strong) signal.
        conf = Math.min(1,
          (postSim - SIM_MATCH) / (1 - SIM_MATCH) * 0.4
          + (SIM_DISSIM - preSim) / SIM_DISSIM * 0.4
          + 0.2,
        );
      } else if (cand.seam > 0 && (postSim - preSim) >= REL_MARGIN) {
        // Same-key seam: chroma can't separate the tunes absolutely, but the
        // waveform dip + relative chroma preference pin the join.
        const relNorm = Math.min(1, (postSim - preSim) / 0.25);
        conf = Math.min(1, 0.45 + 0.30 * relNorm + 0.25 * cand.seam);
      } else {
        continue;
      }
      if (bestExtendTs < 0) { bestExtendTs = pTs; bestExtendConf = conf; }
    }
    if (bestExtendTs > 0 && seg.startTime - bestExtendTs >= MIN_EXTEND_MS) {
      result.extendSuggestions.push({
        segmentIdx: si,
        currentTuneId: seg.tuneId,
        newStartMs: bestExtendTs,
        confidence: bestExtendConf,
      });
    }
  }

  // ── Same-key boundary CROSSOVER refinement ──────────────────────────────────
  // The trim/extend passes above are gated on ABSOLUTE chroma dissimilarity, so
  // they can't move a boundary between two tunes that share a key: the profiles
  // stay similar and preSim/postSim never fall below SIM_DISSIM. But we KNOW the
  // adjacent segments are DIFFERENT tunes — we don't need to prove a boundary
  // exists, only find WHERE it is. Compare each side's MEAN-pooled chroma template
  // (mean-pool keeps the note-EMPHASIS that differs between two same-key tunes;
  // the max-pool used above flattens it) and place the boundary at the RELATIVE
  // crossover — where a sliding window flips from matching A to matching B — then
  // emit a trim(A)+extend(B) pair (the same machinery the absolute passes use).
  // This snaps the boundary back from the ~8 s-late lock toward the real onset.
  // Conservative: needs a clear, SUSTAINED A→B flip, else it leaves the boundary
  // alone (do-no-harm). KEEP IN SYNC with src/search/foote-analyze.ts.
  {
    const XG_GUARD_MS = 8000;     // exclude A's late tail (really B) from A's template
    const XG_TMPL_MS = 8000;      // template span per side
    const XG_WIN_MS = 2000;       // sliding local-match window
    const XG_STEP_MS = 500;
    const XG_MARGIN = 0.015;      // min |simB − simA| to call a window A- or B-leaning
    const XG_SUSTAIN_MS = 2000;   // the B-run before the boundary must last this long
    const XG_MIN_MOVE_MS = 2000;  // only move the boundary if the crossover is ≥ this earlier
    // Cap the reach at ~the real lock latency (window-fill ≈ 8 s + dwell). A
    // legit late-lock correction is within this; anything larger is the crossover
    // over-extending across an intervening tune when the previous segment is a
    // long mis-lock (the Malbay/White-Petticoat swallow). Bounded damage.
    const XG_MAX_MOVE_MS = 18000;
    // A is a trustworthy template only if it's SELF-CONSISTENT — its first half
    // and second half agree. On PEAKY chroma a real tune's own parts diverge more
    // (live: Malbay halves 0.497), so 0.82 wrongly skipped legit corrections. 0.40
    // still catches a mis-lock spanning two DIFFERENT tunes (which diverges further).
    const XG_A_CONSISTENCY = 0.40;
    // Sustained A-preference needed to STOP the walk. One ambiguous window (the 2s
    // window straddling the turnover, or a part-variation blip inside A) must not
    // halt short of the real turnover — require this many consecutive A-leaning
    // windows (× STEP = 1.5 s) before deciding we're solidly back in A's body.
    const XG_A_STOP_RUN = 3;
    // ── STRUCTURAL crossover (Step 2) ── When the two templates share the whole
    // pitch-class set (relative modes, e.g. E Aeolian ↔ G Ionian: tmplSim ≈ 0.96),
    // content is blind. Switch to MELODIC-RECURRENCE matching: for each window frame,
    // its best cosine match into A's vs B's recent region. A window whose phrases
    // recur in B leans B even with an identical histogram. Frame-level → captures the
    // note SEQUENCE the mean-pool throws away.
    const XG_STRUCT_TRIGGER = 0.85;   // tmplSim ≥ this ⇒ content can't separate ⇒ go structural
    const XG_STRUCT_STRIDE = 3;       // frame subsample (cost control)
    const XG_STRUCT_MARGIN = 0.012;   // calibrated from live dRange: B-lean maxes ~0.018-0.032 (A-region shared pitches inflate affinityA), noise floor ~0.007
    const XG_STRUCT_REGION_MS = 10000;// how much of each side's recent audio to match against
    const idxAtOrAfter = (ms) => { let lo = 0, hi = N; while (lo < hi) { const mid = (lo + hi) >>> 1; if (timestamps[mid] < ms) lo = mid + 1; else hi = mid; } return lo; };  // timestamps ascending → binary search (was O(N) linear)
    const meanTs = (aMs, bMs) => {
      const s = idxAtOrAfter(aMs), e = idxAtOrAfter(bMs);
      return (e - s >= 3) ? regionChromaProfile(frames, N, s, e) : null;
    };
    for (let si = 1; si < segments.length; si++) {
      const A = segments[si - 1], B = segments[si];
      if (A.tuneId === B.tuneId) continue;                 // only real tune changes
      const bound = B.startTime;
      // Skip mis-locks: if A's own first half vs second half diverge, A spans more
      // than one tune (the 8137 case that swallowed Malbay + White Petticoat) — its
      // template is untrustworthy and B would "win" across the buried tunes. A legit
      // single tune stays self-similar throughout, even played several times, so a
      // genuinely LONG tune is NOT skipped (that was the over-blunt duration guard).
      const aMid = (A.startTime + A.endTime) / 2;
      const aH1 = meanTs(A.startTime, aMid);
      const aH2 = meanTs(aMid, A.endTime);
      const aHalves = (aH1 && aH2) ? cosineSimVec(aH1, aH2) : 1;
      if (aHalves < XG_A_CONSISTENCY) {
        result.xgDiag.push(`${A.tuneId}->${B.tuneId} SKIP consistency halves=${aHalves.toFixed(3)}`);
        continue;
      }
      const tmplAend = bound - XG_GUARD_MS;
      if (tmplAend - A.startTime < 3000) continue;         // A too short to template cleanly
      const tmplA = meanTs(A.startTime, Math.min(tmplAend, A.startTime + XG_TMPL_MS));
      const tmplB = meanTs(bound, Math.min(B.endTime, bound + XG_TMPL_MS));
      if (!tmplA || !tmplB) continue;
      const tmplSim = cosineSimVec(tmplA, tmplB);
      if (tmplSim > 0.995) continue;    // templates identical → nothing to split on
      const floor = Math.max(A.startTime + XG_WIN_MS, bound - XG_MAX_MOVE_MS);
      let cross = -1, foundA = false, aRun = 0;
      if (tmplSim >= XG_STRUCT_TRIGGER) {
        // ── STRUCTURAL path: relative modes / shared pitch content ──────────────
        // Content is blind (same histogram). Walk back comparing each window's MELODIC
        // RECURRENCE into A's recent region vs B's — the note SEQUENCE, not the pool.
        const aRs = idxAtOrAfter(Math.max(A.startTime, bound - XG_GUARD_MS - XG_STRUCT_REGION_MS));
        const aRe = idxAtOrAfter(bound - XG_GUARD_MS);
        const bRs = idxAtOrAfter(bound);
        const bRe = idxAtOrAfter(Math.min(B.endTime, bound + XG_STRUCT_REGION_MS));
        let minDiff = 0, maxDiff = 0;
        if (aRe - aRs >= 4 && bRe - bRs >= 4) {
          // Precompute each frame's BEST cosine into the (FIXED) A- and B-regions ONCE.
          // The window slides at 75% overlap, so per-window structAffinity recomputed
          // this ~4×; now each window is just a mean of precomputed values. Filled on a
          // stride grid anchored at spanS; the window mean reads the same grid.
          const spanS = idxAtOrAfter(floor), spanE = idxAtOrAfter(bound);
          const bestA = new Float32Array(spanE), bestB = new Float32Array(spanE);
          for (let i = spanS; i < spanE; i += XG_STRUCT_STRIDE) {
            let bA = 0, bB = 0;
            for (let j = aRs; j < aRe; j += XG_STRUCT_STRIDE) { const s = cosineSim12Mel(frames, i, j); if (s > bA) bA = s; }
            for (let j = bRs; j < bRe; j += XG_STRUCT_STRIDE) { const s = cosineSim12Mel(frames, i, j); if (s > bB) bB = s; }
            bestA[i] = bA; bestB[i] = bB;
          }
          const meanBest = (best, wS, wE) => {
            let sum = 0, cnt = 0;
            for (let i = spanS + Math.ceil((wS - spanS) / XG_STRUCT_STRIDE) * XG_STRUCT_STRIDE; i < wE; i += XG_STRUCT_STRIDE) { sum += best[i]; cnt++; }
            return cnt > 0 ? sum / cnt : 0;
          };
          for (let w = bound - XG_WIN_MS; w >= floor; w -= XG_STEP_MS) {
            const wS = idxAtOrAfter(w), wE = idxAtOrAfter(w + XG_WIN_MS);
            if (wE - wS < 3) break;
            const diff = meanBest(bestB, wS, wE) - meanBest(bestA, wS, wE);
            if (diff < minDiff) minDiff = diff; if (diff > maxDiff) maxDiff = diff;
            if (diff < -XG_STRUCT_MARGIN) {
              if (++aRun >= XG_A_STOP_RUN) { foundA = true; break; }
            } else {
              aRun = 0;
              if (diff > XG_STRUCT_MARGIN) cross = w;
            }
          }
        }
        result.xgDiag.push(`${A.tuneId}->${B.tuneId} STRUCT tmplSim=${tmplSim.toFixed(3)} dRange=[${minDiff.toFixed(3)},${maxDiff.toFixed(3)}] cross=${cross<0?'none':((bound-cross)/1000).toFixed(1)+'s'} foundA=${foundA?1:0}`);
      } else {
        // ── CONTENT path: the "C-row" discriminative reweighting ────────────────
        // Weight each pitch class by how much the two templates DISAGREE there, so the
        // walk is decided by the rows that DISTINGUISH the tunes (e.g. C♯ vs C♮).
        let magA = 0, magB = 0;
        for (let k = 0; k < 12; k++) { magA += tmplA[k]*tmplA[k]; magB += tmplB[k]*tmplB[k]; }
        magA = Math.sqrt(magA) || 1; magB = Math.sqrt(magB) || 1;
        const dw = new Float32Array(12), wtA = new Float32Array(12), wtB = new Float32Array(12);
        let sumW = 0;
        for (let k = 0; k < 12; k++) {
          dw[k] = Math.abs(tmplB[k]/magB - tmplA[k]/magA);
          sumW += dw[k];
          wtA[k] = tmplA[k]*dw[k]; wtB[k] = tmplB[k]*dw[k];
        }
        const wwin = new Float32Array(12);
        for (let w = bound - XG_WIN_MS; w >= floor; w -= XG_STEP_MS) {
          const win = meanTs(w, w + XG_WIN_MS);
          if (!win) break;
          for (let k = 0; k < 12; k++) wwin[k] = win[k]*dw[k];
          const diff = cosineSimVec(wwin, wtB) - cosineSimVec(wwin, wtA);
          if (diff < -XG_MARGIN) {
            if (++aRun >= XG_A_STOP_RUN) { foundA = true; break; }
          } else {
            aRun = 0;
            if (diff > XG_MARGIN) cross = w;
          }
        }
        result.xgDiag.push(`${A.tuneId}->${B.tuneId} tmplSim=${tmplSim.toFixed(3)} sumW=${sumW.toFixed(2)} cross=${cross<0?'none':((bound-cross)/1000).toFixed(1)+'s'} foundA=${foundA?1:0}`);
      }
      if (cross < 0) continue;
      // Require foundA (walked back INTO A) on BOTH paths. Without it, a two-part tune
      // whose ENDING doesn't match its FIRST-8s template makes the walk confidently
      // prefer B all the way to the cap (foundA=0) — and the next tune eats the tail
      // (live: Galtee's end over-eaten via 724->35 cross=18s foundA=0). The now-working
      // very-late boundaries all resolve via foundA=1 + the boundary window (Step 3),
      // so this closes the over-reach without losing them.
      if (!foundA) continue;
      if (bound - cross < XG_SUSTAIN_MS || bound - cross < XG_MIN_MOVE_MS) continue;
      // Directly move the A→B boundary to `cross` (steal A's tail + nulls in
      // [cross, bound) → B). The scanner looks up B's metadata from tuneId.
      result.boundaryMoves.push({ tuneId: B.tuneId, prevTuneId: A.tuneId, boundaryMs: cross, upperMs: bound, confidence: 0.7 });
    }
  }

  if (segments.length < 2) return result;

  const regionBounds = [0, ...peakIndices, N];
  const regionProfiles = [];
  for (let r = 0; r < regionBounds.length - 1; r++) {
    regionProfiles.push(regionChromaProfile(frames, N, regionBounds[r], regionBounds[r + 1]));
  }

  function findRegion(t) {
    for (let r = 0; r < regionBounds.length - 1; r++) {
      const rStart = timestamps[regionBounds[r]];
      const rEnd = timestamps[Math.min(regionBounds[r + 1], N - 1)];
      if (t >= rStart && t <= rEnd) return r;
    }
    return -1;
  }

  // Pre-compute each segment's chroma profile from frames within its time bounds.
  const segProfiles = new Array(segments.length);
  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    let ssf = -1, sef = -1;
    for (let i = 0; i < N; i++) {
      if (ssf < 0 && timestamps[i] >= seg.startTime) ssf = i;
      if (timestamps[i] >= seg.endTime) { sef = i; break; }
    }
    if (sef < 0) sef = N;
    segProfiles[si] = (ssf >= 0 && sef > ssf) ? regionChromaProfile(frames, N, ssf, sef) : null;
  }

  // Reassignment strategy — walk every segment, look at a window of nearby
  // segments (both indices and absolute time), find the longest one whose
  // chroma profile is highly similar.  If the longest similar match is
  // substantially longer than the current segment AND no novelty boundary
  // sits between them, suggest relabeling the weak segment to the strong
  // one's tuneId.  This catches chains like
  //     A_brief → B_brief → C_brief → D_correct (D long, others short)
  // because each of A/B/C compares against the whole window — D wins as the
  // longest profile-similar segment within reach, and they all converge.
  const NEIGHBOR_RADIUS_SEGS = 4;        // search ±4 segment indices either side
  const NEIGHBOR_RADIUS_MS = 60_000;     // and ±60 s wall time
  const LENGTH_DOMINANCE_RATIO = 1.8;    // anchor must be ≥ 1.8× the weak segment
  // Strong flicker-swallow ratio: a same-key relabel this dominant is the model's
  // own sustained verdict correcting a brief flicker (e.g. wrong-guess → Morrison's),
  // so it may cross the consumer's same-key veto. KEEP IN SYNC with foote-analyze.ts.
  const SAMEKEY_OVERRIDE_RATIO = 2.5;
  // A relabel only ever CORRECTS a brief mislabel flicker — never a SUSTAINED
  // identification. Chroma self-similarity is blind to same-key tune changes
  // (Cooley's vs Gravel Walks — both D reels — high cosine sim, NO boundary
  // between), so without this cap both the bookend override and the length-
  // dominance path would absorb a full neighbouring tune that merely shares a
  // key. A real trad tune runs ≥ ~20 s; an ID flicker is a few seconds.
  // KEEP IN SYNC with src/search/foote-analyze.ts.
  const RELABEL_MAX_MIDDLE_MS = 14_000;

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    const segDur = seg.endTime - seg.startTime;
    const myProfile = segProfiles[si];
    if (!myProfile) continue;
    if (segDur > RELABEL_MAX_MIDDLE_MS) continue;  // sustained = a real tune; leave it

    // ── BOOKEND override ─────────────────────────────────────────────────────
    // The classic consolidation signature: the SAME tune reappears on BOTH
    // sides of a short, low-confidence segment (Cooley's · Tam Lin · Cooley's).
    // When the bracketed segment is acoustically self-similar to those identical
    // bookends — and no real Foote boundary separates it from either flank — the
    // whole span is almost certainly ONE tune. Relabel to the bookend tune,
    // OVERRIDING the length-dominance heuristic below (which would otherwise pull
    // a brief segment into a long unrelated neighbour like Farewell to Ireland).
    // The decision rests on the chroma self-similarity, not segment duration.
    // KEEP IN SYNC with src/search/foote-analyze.ts.
    {
      let beforeNi = -1, afterNi = -1;
      let beforeGap = Infinity, afterGap = Infinity;
      for (let ni = 0; ni < segments.length; ni++) {
        if (ni === si) continue;
        const nb = segments[ni];
        if (nb.tuneId === seg.tuneId) continue;
        if (nb.endTime <= seg.startTime) {
          const g = seg.startTime - nb.endTime;
          if (g < beforeGap) { beforeGap = g; beforeNi = ni; }
        } else if (nb.startTime >= seg.endTime) {
          const g = nb.startTime - seg.endTime;
          if (g < afterGap) { afterGap = g; afterNi = ni; }
        }
      }
      if (beforeNi >= 0 && afterNi >= 0
          && segments[beforeNi].tuneId === segments[afterNi].tuneId
          && beforeGap <= NEIGHBOR_RADIUS_MS && afterGap <= NEIGHBOR_RADIUS_MS) {
        const pB = segProfiles[beforeNi];
        const pA = segProfiles[afterNi];
        if (pB && pA) {
          const simB = cosineSimVec(myProfile, pB);
          const simA = cosineSimVec(myProfile, pA);
          if (Math.min(simB, simA) > SIMILARITY_MERGE_THRESH) {
            const spanLo = Math.min(seg.startTime, segments[beforeNi].endTime);
            const spanHi = Math.max(seg.endTime, segments[afterNi].startTime);
            let hardBoundary = false;
            for (const b of result.boundaries) {
              if (b > spanLo + BOUNDARY_TOLERANCE_MS && b < spanHi - BOUNDARY_TOLERANCE_MS) {
                hardBoundary = true; break;
              }
            }
            if (!hardBoundary) {
              // Floor 0.70 (> the consumer's 0.65 apply-gate): the bookend
              // pattern is strong structural evidence on its own, so even a
              // borderline-similar bracketed segment clears the gate.
              const bookSim = Math.min(simB, simA);
              const confidence = Math.min(1,
                (bookSim - SIMILARITY_MERGE_THRESH) / (1 - SIMILARITY_MERGE_THRESH) * 0.3 + 0.7);
              if (!result.suggestions.some(s => s.segmentIdx === si)) {
                result.suggestions.push({
                  segmentIdx: si,
                  currentTuneId: seg.tuneId,
                  suggestedTuneId: segments[beforeNi].tuneId,
                  confidence,
                  bookend: true,
                });
              }
              continue;
            }
          }
        }
      }
    }

    let bestNi = -1;
    let bestSim = SIMILARITY_MERGE_THRESH;  // must beat the merge threshold
    let bestDur = segDur;

    const lo = Math.max(0, si - NEIGHBOR_RADIUS_SEGS);
    const hi = Math.min(segments.length - 1, si + NEIGHBOR_RADIUS_SEGS);
    for (let ni = lo; ni <= hi; ni++) {
      if (ni === si) continue;
      const neighbor = segments[ni];
      if (neighbor.tuneId === seg.tuneId) continue;
      // Time-distance gate — skip neighbors that are too far in wall time
      const segMid = (seg.startTime + seg.endTime) / 2;
      const nMid = (neighbor.startTime + neighbor.endTime) / 2;
      if (Math.abs(nMid - segMid) > NEIGHBOR_RADIUS_MS) continue;

      const nProfile = segProfiles[ni];
      if (!nProfile) continue;
      const sim = cosineSimVec(myProfile, nProfile);
      if (sim <= bestSim) continue;
      const nDur = neighbor.endTime - neighbor.startTime;
      // Prefer the longer of the equally-similar candidates so chains converge
      // toward the dominant correct ID.
      if (nDur < bestDur) continue;

      // Skip if a real novelty boundary sits between us and them — that's
      // legitimate texture change, not a misidentification.
      const lo2 = Math.min(seg.startTime, neighbor.startTime);
      const hi2 = Math.max(seg.endTime, neighbor.endTime);
      let boundaryBetween = false;
      for (const b of result.boundaries) {
        if (b > lo2 + BOUNDARY_TOLERANCE_MS && b < hi2 - BOUNDARY_TOLERANCE_MS) {
          // Check that the boundary actually separates us from them (not interior)
          const onMySide = (b > seg.endTime) !== (b > neighbor.startTime);
          if (onMySide) { boundaryBetween = true; break; }
        }
      }
      if (boundaryBetween) continue;

      bestNi = ni;
      bestSim = sim;
      bestDur = nDur;
    }

    if (bestNi < 0) continue;
    if (bestDur < segDur * LENGTH_DOMINANCE_RATIO) continue;  // anchor must dominate

    const confidence = Math.min(1,
      (bestSim - SIMILARITY_MERGE_THRESH) / (1 - SIMILARITY_MERGE_THRESH) * 0.5 + 0.5);
    if (!result.suggestions.some(s => s.segmentIdx === si)) {
      result.suggestions.push({
        segmentIdx: si,
        currentTuneId: seg.tuneId,
        suggestedTuneId: segments[bestNi].tuneId,
        confidence,
        strongDominance: bestDur >= segDur * SAMEKEY_OVERRIDE_RATIO,
      });
    }
  }

  // Gap absorption — for unidentified regions between segments.  Operate on a
  // sorted copy so original index references remain stable.
  const sorted = segments.map((s, idx) => ({ ...s, _origIdx: idx })).sort((a, b) => a.startTime - b.startTime);
  for (let si = 0; si < sorted.length; si++) {
    const seg = sorted[si];
    const nextSeg = sorted[si + 1];
    const gapStart = seg.endTime;
    const gapEnd = nextSeg ? nextSeg.startTime : windowEndMs;
    const gapDur = gapEnd - gapStart;
    // Upper bound 8 s (was 60 s): a gap as long as a real tune must NOT be
    // silently absorbed into a chroma-similar neighbour — chroma can't tell a
    // same-key NEIGHBOURING tune from a continuation, so anything beyond a brief
    // transition is more likely a tune the model never locked. KEEP IN SYNC
    // with src/search/foote-analyze.ts.
    if (gapDur < 2000 || gapDur > 8000) continue;

    let gsf = -1, gef = -1;
    for (let i = 0; i < N; i++) {
      if (gsf < 0 && timestamps[i] >= gapStart) gsf = i;
      if (timestamps[i] >= gapEnd) { gef = i; break; }
    }
    if (gef < 0) gef = N;
    if (gsf < 0 || gef <= gsf) continue;

    const gapProfile = regionChromaProfile(frames, N, gsf, gef);
    let gapEnergy = 0;
    for (let b = 0; b < 12; b++) gapEnergy += gapProfile[b];
    if (gapEnergy < 0.5) continue;

    let ssf = -1, sef = -1;
    for (let i = 0; i < N; i++) {
      if (ssf < 0 && timestamps[i] >= seg.startTime) ssf = i;
      if (timestamps[i] >= seg.endTime) { sef = i; break; }
    }
    if (sef < 0) sef = N;
    if (ssf >= 0 && sef > ssf) {
      const segProfile = regionChromaProfile(frames, N, ssf, sef);
      const sim = cosineSimVec(gapProfile, segProfile);
      if (sim > SIMILARITY_MERGE_THRESH) {
        const confidence = Math.min(1, (sim - SIMILARITY_MERGE_THRESH) / (1 - SIMILARITY_MERGE_THRESH) * 0.5 + 0.5);
        result.gapAbsorptions.push({
          gapStartMs: gapStart, gapEndMs: gapEnd,
          absorberSegmentIdx: seg._origIdx, absorberTuneId: seg.tuneId, confidence,
        });
        continue;
      }
    }

    if (nextSeg) {
      let nsf = -1, nef = -1;
      for (let i = 0; i < N; i++) {
        if (nsf < 0 && timestamps[i] >= nextSeg.startTime) nsf = i;
        if (timestamps[i] >= nextSeg.endTime) { nef = i; break; }
      }
      if (nef < 0) nef = N;
      if (nsf >= 0 && nef > nsf) {
        const nextProfile = regionChromaProfile(frames, N, nsf, nef);
        const sim = cosineSimVec(gapProfile, nextProfile);
        if (sim > SIMILARITY_MERGE_THRESH) {
          const confidence = Math.min(1, (sim - SIMILARITY_MERGE_THRESH) / (1 - SIMILARITY_MERGE_THRESH) * 0.5 + 0.5);
          result.gapAbsorptions.push({
            gapStartMs: gapStart, gapEndMs: gapEnd,
            absorberSegmentIdx: nextSeg._origIdx, absorberTuneId: nextSeg.tuneId, confidence,
          });
        }
      }
    }
  }

  return result;
}

self.onmessage = function (e) {
  const m = e.data;
  if (!m || m.type !== 'analyze') return;
  try {
    const r = analyze(m.frames, m.timestamps, m.segments || [], m.windowEndMs || 0, m.energies || null);
    self.postMessage({ type: 'result', scanId: m.scanId, ...r });
  } catch (err) {
    self.postMessage({ type: 'error', scanId: m.scanId, error: String(err && err.message || err) });
  }
};
