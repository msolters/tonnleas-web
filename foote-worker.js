/**
 * Foote novelty worker — runs the self-similarity / checkerboard / peak-pick
 * math off the main thread so analysis-region scans don't block the UI or DSP.
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

function analyze(frames, timestamps, segments, windowEndMs) {
  const N = timestamps.length;
  const result = { boundaries: [], suggestions: [], gapAbsorptions: [], trimSuggestions: [], extendSuggestions: [] };
  if (N < KERNEL_HALF * 2 + 2) return result;

  const S = buildSimilarityMatrix(frames, N);
  const novelty = computeNovelty(S, N);
  const peakIndices = pickPeaks(novelty);
  result.boundaries = peakIndices.map(i => timestamps[i]);

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

    // Start extend: earliest peak in [seg.startTime - MAX_EXTEND_MS, seg.startTime)
    // that LOOKS LIKE A REAL BOUNDARY — pre-peak side dissimilar to the
    // segment AND post-peak side matches it.  This is the structural
    // signature of "different content → this tune starts here" in the self-
    // similarity matrix.  A mid-tune chord change has BOTH sides matching
    // the profile (post passes but pre passes too) so it's skipped, leaving
    // us free to look further back for the real onset.
    const extendFloor = Math.max(0, seg.startTime - MAX_EXTEND_MS);
    let bestExtendTs = -1, bestExtendConf = 0;
    for (const p of peakIndices) {
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
      if (postSim >= SIM_MATCH && preSim < SIM_DISSIM) {
        const conf = Math.min(1,
          (postSim - SIM_MATCH) / (1 - SIM_MATCH) * 0.4
          + (SIM_DISSIM - preSim) / SIM_DISSIM * 0.4
          + 0.2,
        );
        if (bestExtendTs < 0) { bestExtendTs = pTs; bestExtendConf = conf; }
      }
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

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    const segDur = seg.endTime - seg.startTime;
    const myProfile = segProfiles[si];
    if (!myProfile) continue;

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
    // Gaps up to 60 s are absorbable — the lock can take 20-30 s after a
    // silence reset, and we want Foote to still merge that gap into the tune
    // that picks up on either side.
    if (gapDur < 2000 || gapDur > 60000) continue;

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
    const r = analyze(m.frames, m.timestamps, m.segments || [], m.windowEndMs || 0);
    self.postMessage({ type: 'result', scanId: m.scanId, ...r });
  } catch (err) {
    self.postMessage({ type: 'error', scanId: m.scanId, error: String(err && err.message || err) });
  }
};
