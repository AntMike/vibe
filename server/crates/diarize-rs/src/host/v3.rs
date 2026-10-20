//! The host half of Sortformer v3 (Nemotron-3-Diarization): offline chunking,
//! the FIFO / Arrival-Order Speaker Cache, and 10 ms post-processing.
//!
//! A port of HF transformers 5.18 `Nemotron3DiarizationForAudioFrameClassification.forward`
//! (offline mode) and `Nemotron3DiarizationSpeakerCache` (Apache-2.0), which is
//! what `baseline/probs/standup_*` was recorded with. It is kept apart from the
//! v2 state machine in [`super::aosc`]/[`super::cache`] rather than folded into
//! it, because the two differ in almost every detail that matters for parity:
//!
//! * the cache stores feature-stacking embeddings and the model predicts at
//!   10 ms, so cache scores come from 80 ms average-pooled probabilities;
//! * cache probabilities are re-estimated every step until the first
//!   compression, then frozen (v2 seeds them once);
//! * the score's complement term clamps `1 - p`, not `1 - max(p, thr)`;
//! * popped frames get `scores_boost_latest`, and disabled slots get the
//!   trained silence embedding instead of a running mean;
//! * the last encoder frame may be padding, masked as a key and scored zero.
//!
//! Geometry: the cache length and scoring come from the GGUF KVs. The chunk
//! length, look-ahead, FIFO length and update period do *not*: the
//! `sortformer.streaming.*` KVs (264 / 0 / 264 / 0) record the training
//! configuration, and HF's offline forward uses its own `chunk_length=340,
//! chunk_right_context=40, fifo_length=40, speaker_cache_update_period=300`
//! (`config.json`), which no GGUF key carries. Same situation, and same
//! answer, as v2's inference geometry in `lib.rs`.

use ndarray::Array2;

use super::segment::{binarize, median_filter};
use super::{
    to_mono, Backend, ChunkRequest, DiarizationConfig, HostError, HostResult, MelFrontend, SpeakerSegment, HOP_LENGTH, N_MELS,
    SAMPLE_RATE,
};
use crate::sf_weights::Scoring;

/// HF offline chunking, in encoder frames (see the module docs).
pub const V3_CHUNK_LEN: usize = 340;
pub const V3_RIGHT_CONTEXT: usize = 40;
pub const V3_FIFO_LEN: usize = 40;
pub const V3_UPDATE_PERIOD: usize = 300;

/// v2's post-processing was tuned on 80 ms frames; its median window is kept
/// as a duration, not a frame count, when the model runs at 10 ms.
const V2_FRAME_DURATION: f32 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V3Geometry {
    pub chunk_len: usize,
    pub right_context: usize,
    pub fifo_len: usize,
    pub update_period: usize,
    pub spkcache_len: usize,
}

/// Everything the v3 host needs to know about the model.
#[derive(Debug, Clone)]
pub struct V3Model {
    pub num_speakers: usize,
    pub emb_dim: usize,
    /// Mel frames per encoder frame; also the head's upsampling factor.
    pub subsampling: usize,
    pub geom: V3Geometry,
    pub scoring: Scoring,
    pub silence: Vec<f32>,
}

pub struct V3Host {
    config: DiarizationConfig,
    mel: MelFrontend,
    model: V3Model,
}

impl V3Host {
    pub fn new(mel: MelFrontend, model: V3Model, config: DiarizationConfig) -> Self {
        Self { config, mel, model }
    }

    fn frame_duration(&self) -> f32 {
        HOP_LENGTH as f32 / SAMPLE_RATE as f32
    }

    pub fn diarize(
        &mut self,
        backend: &mut dyn Backend,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> HostResult<Vec<SpeakerSegment>> {
        let mono = to_mono(&audio, sample_rate, channels)?;
        let preds = self.probs(backend, &mono)?;

        let mut config = self.config.clone();
        let window = (config.median_window as f32 * V2_FRAME_DURATION / self.frame_duration()).round() as usize;
        config.median_window = window | 1;
        let filtered = if config.median_window > 1 { median_filter(&config, &preds) } else { preds };

        let n_audio_samples = mono.len() as u64;
        let mut segments = binarize(&config, &filtered, self.frame_duration());
        for seg in &mut segments {
            seg.end = seg.end.min(n_audio_samples);
        }
        segments.retain(|s| s.end > s.start);
        Ok(segments)
    }

    pub fn diarize_raw(
        &mut self,
        backend: &mut dyn Backend,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> HostResult<(Vec<f32>, usize)> {
        let mono = to_mono(audio, sample_rate, channels)?;
        let preds = self.probs(backend, &mono)?;
        let frames = preds.shape()[0];
        Ok((preds.into_raw_vec_and_offset().0, frames))
    }

    /// Sigmoid probabilities `[T_mel, n_spk]`, one row per 10 ms mel frame
    /// (HF's `logits.sigmoid()`, including its trailing padding frame).
    fn probs(&self, backend: &mut dyn Backend, audio: &[f32]) -> HostResult<Array2<f32>> {
        let m = &self.model;
        let (n_spk, d, factor) = (m.num_speakers, m.emb_dim, m.subsampling);
        let features = self.mel.extract_mel_features(audio)?;
        let total = features.shape()[1];
        // HF counts `floor(L / hop)` valid frames of the `L / hop + 1` that
        // the centred STFT produces, and zeroes the rest.
        let valid = audio.len() / HOP_LENGTH;
        if valid == 0 {
            return Ok(Array2::zeros((total, n_spk)));
        }
        let n_embeds = total.div_ceil(factor);

        let mut state = CacheState::default();
        let mut out = Vec::with_capacity(n_embeds * factor * n_spk);
        for start in (0..n_embeds).step_by(m.geom.chunk_len) {
            let end = (start + m.geom.chunk_len).min(n_embeds);
            let num_chunk = end - start;
            let win_end = (end + m.geom.right_context).min(n_embeds);
            let win_frames = (win_end - start) * factor;
            let first = start * factor;

            let mut mel = vec![0.0f32; N_MELS * win_frames];
            for f in 0..win_frames.min(valid.saturating_sub(first)) {
                for (bin, value) in features.slice(ndarray::s![0, first + f, ..]).iter().enumerate() {
                    mel[bin * win_frames + f] = *value;
                }
            }
            let valid_in_win = valid.saturating_sub(first).min(win_frames);

            let forward = backend
                .run_chunk(ChunkRequest {
                    mel: &mel,
                    mel_frames: win_frames,
                    current_len: valid_in_win,
                    spkcache: &state.embs,
                    spkcache_frames: state.n_cache,
                    fifo: &state.fifo,
                    fifo_frames: state.n_fifo,
                })
                .map_err(HostError::Backend)?;
            forward.validate(d, n_spk)?;

            let n_cached = state.n_cache + state.n_fifo;
            let n_frames = forward.prediction_frames / factor;
            out.extend_from_slice(&forward.predictions[n_cached * factor * n_spk..(n_cached + num_chunk) * factor * n_spk]);

            // Average-pool the 10 ms sigmoids back to encoder frames; padding
            // frames score zero (HF `_pool_probs`).
            let n_valid = n_cached + valid_in_win.div_ceil(factor);
            let mut pooled = vec![0.0f32; n_frames * n_spk];
            for t in 0..n_valid.min(n_frames) {
                for s in 0..n_spk {
                    let sum: f32 = (0..factor).map(|j| forward.predictions[(t * factor + j) * n_spk + s]).sum();
                    pooled[t * n_spk + s] = sum / factor as f32;
                }
            }
            state.update(m, &forward.embeddings[..num_chunk * d], &pooled, num_chunk);
        }
        out.truncate(total * n_spk);
        Array2::from_shape_vec((total, n_spk), out).map_err(|e| HostError::Audio(format!("failed to reshape preds: {e}")))
    }
}

/// HF `Nemotron3DiarizationSpeakerCache`, row-major flat buffers.
#[derive(Default)]
struct CacheState {
    embs: Vec<f32>,
    probs: Vec<f32>,
    n_cache: usize,
    fifo: Vec<f32>,
    n_fifo: usize,
    compressed: bool,
}

impl CacheState {
    /// Push the chunk into the FIFO; on overflow move the oldest frames into
    /// the speaker cache, compressing it when it outgrows `spkcache_len`.
    /// `probs` is the pooled `[n_cache + n_fifo + T_window, n_spk]` of this step.
    fn update(&mut self, m: &V3Model, chunk_embs: &[f32], probs: &[f32], num_chunk: usize) {
        let (d, s) = (m.emb_dim, m.num_speakers);
        self.fifo.extend_from_slice(chunk_embs);
        let n = self.n_fifo + num_chunk;
        self.n_fifo = n;
        if n <= m.geom.fifo_len {
            return;
        }
        let popped = m.geom.update_period.max(n - m.geom.fifo_len).min(n);

        let fifo_probs = &probs[self.n_cache * s..(self.n_cache + popped) * s];
        // An uncompressed cache still holds frames in arrival order whose
        // probabilities this step re-estimated; a compressed one only has the
        // probabilities stored when it was compressed.
        if !self.compressed {
            self.probs = probs[..self.n_cache * s].to_vec();
        }
        self.probs.extend_from_slice(fifo_probs);
        self.embs.extend_from_slice(&self.fifo[..popped * d]);
        self.fifo.drain(..popped * d);
        self.n_fifo = n - popped;
        self.n_cache += popped;

        if self.n_cache > m.geom.spkcache_len {
            let (embs, probs) = compress(m, &self.embs, &self.probs, self.n_cache);
            self.embs = embs;
            self.probs = probs;
            self.n_cache = m.geom.spkcache_len;
            self.compressed = true;
        }
    }
}

/// HF `_get_frame_scores` plus the latest-frames boost, `[n_frames, n_spk]`
/// row-major. `-inf` marks a frame a speaker may not keep.
fn frame_scores(m: &V3Model, probs: &[f32], n_frames: usize, min_pos: usize) -> Vec<f32> {
    let s = m.num_speakers;
    let sc = &m.scoring;
    let thr = sc.pred_score_threshold;
    let mut scores = vec![f32::NEG_INFINITY; n_frames * s];
    for t in 0..n_frames {
        let row = &probs[t * s..(t + 1) * s];
        let sum_c: f32 = row.iter().map(|&p| (1.0 - p).max(thr).ln()).sum();
        for (k, &p) in row.iter().enumerate() {
            if p > 0.5 {
                scores[t * s + k] = p.max(thr).ln() - (1.0 - p).max(thr).ln() + sum_c - 0.5f32.ln();
            }
        }
    }
    for k in 0..s {
        let positives = (0..n_frames).filter(|&t| scores[t * s + k] > 0.0).count();
        if positives >= min_pos {
            for t in 0..n_frames {
                if scores[t * s + k] <= 0.0 {
                    scores[t * s + k] = f32::NEG_INFINITY;
                }
            }
        }
    }
    // Frames past the cache capacity are the ones just popped from the FIFO.
    for value in scores.iter_mut().skip(m.geom.spkcache_len * s) {
        *value += sc.scores_boost_latest;
    }
    scores
}

/// Indices of the `k` largest values, ties broken by ascending index.
///
/// ponytail: torch's CPU `topk` picks among exact ties in whatever order its
/// STL's `nth_element` leaves them (MSVC on the reference's Windows wheel),
/// and ties are common: every frame where one speaker's sigmoid saturates to
/// 1.0 and the rest to 0.0 scores exactly `-ln 0.5`. Lowest index first
/// differs from it on those ties only. Measured on standup_600_720 (f32): max
/// |dprob| 0.017 on Vulkan, 0.039 on CPU, against 0.0007 with an emulation of
/// MSVC's `nth_element`; port that emulation if CPU parity matters.
fn topk(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    order.truncate(k);
    order
}

/// HF `_compress`: keep the `spkcache_len` best (frame, speaker) pairs, grouped
/// by speaker and in arrival order within a speaker; disabled slots become
/// the learned silence embedding with zero probabilities.
fn compress(m: &V3Model, embs: &[f32], probs: &[f32], n_frames: usize) -> (Vec<f32>, Vec<f32>) {
    let (d, s, len) = (m.emb_dim, m.num_speakers, m.geom.spkcache_len);
    let sc = &m.scoring;
    let budget = (len / s).saturating_sub(sc.sil_frames_per_spk) as f32;
    let min_pos = (budget * sc.min_pos_scores_rate).floor() as usize;
    let strong = (budget * sc.strong_boost_rate).floor() as usize;
    let weak = (budget * sc.weak_boost_rate).floor() as usize;

    let mut scores = frame_scores(m, probs, n_frames, min_pos);
    for (count, boost) in [(strong, -2.0 * 0.5f32.ln()), (weak, -0.5f32.ln())] {
        for k in 0..s {
            let column: Vec<f32> = (0..n_frames).map(|t| scores[t * s + k]).collect();
            for t in topk(&column, count) {
                scores[t * s + k] += boost;
            }
        }
    }

    // Speaker-major flat scores over `n_frames + sil` rows, the silence rows
    // at +inf so they are always kept.
    let rows = n_frames + sc.sil_frames_per_spk;
    let flat_scores: Vec<f32> = (0..s * rows)
        .map(|i| {
            let (k, t) = (i / rows, i % rows);
            if t < n_frames {
                scores[t * s + k]
            } else {
                f32::INFINITY
            }
        })
        .collect();
    let sentinel = s * rows;
    let mut picked: Vec<usize> = topk(&flat_scores, len)
        .into_iter()
        .map(|i| if flat_scores[i] == f32::NEG_INFINITY { sentinel } else { i })
        .collect();
    picked.sort_unstable();

    let mut new_embs = Vec::with_capacity(len * d);
    let mut new_probs = Vec::with_capacity(len * s);
    for i in picked {
        let frame = if i == sentinel { n_frames } else { (i % rows).min(n_frames) };
        if frame == n_frames {
            new_embs.extend_from_slice(&m.silence);
            new_probs.extend(std::iter::repeat_n(0.0, s));
        } else {
            new_embs.extend_from_slice(&embs[frame * d..(frame + 1) * d]);
            new_probs.extend_from_slice(&probs[frame * s..(frame + 1) * s]);
        }
    }
    (new_embs, new_probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(spkcache_len: usize, boost: f32) -> V3Model {
        V3Model {
            num_speakers: 2,
            emb_dim: 1,
            subsampling: 8,
            geom: V3Geometry {
                chunk_len: 4,
                right_context: 0,
                fifo_len: 0,
                update_period: 4,
                spkcache_len,
            },
            scoring: Scoring {
                sil_frames_per_spk: 1,
                scores_boost_latest: boost,
                ..Scoring::default()
            },
            silence: vec![-1.0],
        }
    }

    #[test]
    fn latest_frames_score_boost_applies_past_the_cache_capacity_only() {
        let m = model(2, 0.05);
        // Four identical frames where speaker 0 is confidently active.
        let probs = [0.9f32, 0.1].repeat(4);
        let plain = frame_scores(&model(2, 0.0), &probs, 4, 0);
        let boosted = frame_scores(&m, &probs, 4, 0);
        for t in 0..4 {
            let delta = boosted[t * 2] - plain[t * 2];
            let want = if t >= 2 { 0.05 } else { 0.0 };
            assert!((delta - want).abs() < 1e-6, "frame {t}: delta {delta}");
            // Silent speaker 1 stays disabled whatever the boost.
            assert_eq!(boosted[t * 2 + 1], f32::NEG_INFINITY);
        }
    }

    #[test]
    fn latest_boost_decides_ties_in_favour_of_popped_frames() {
        // Cache of 4 (2 speakers x (1 + 1 silence)), 6 identical frames: with
        // the boost the two newest frames win speaker 0's single real slot
        // pair; without it the oldest do.
        let probs = [0.9f32, 0.1].repeat(6);
        let embs: Vec<f32> = (0..6).map(|t| t as f32).collect();
        let (kept, _) = compress(&model(4, 0.05), &embs, &probs, 6);
        let (kept_plain, _) = compress(&model(4, 0.0), &embs, &probs, 6);
        assert_eq!(kept.len(), 4);
        assert!(kept.contains(&5.0) && kept.contains(&4.0), "{kept:?}");
        assert!(kept_plain.contains(&0.0), "{kept_plain:?}");
        assert!(kept.contains(&-1.0), "silence slot filled with the learned embedding");
    }
}
