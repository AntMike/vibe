//! The Sortformer v3 (Nemotron-3-Diarization) graph: one graph per chunk.
//!
//! ```text
//! host     mel [n_mels, M] -> stack 8 frames -> [n_mels * 8, M / 8]
//! graph    proj (no bias)                      -> chunk embeddings [512, T]  (cached by AOSC)
//!          concat [spkcache | fifo | chunk]    -> [512, T_concat]
//!          embed_norm -> 31 pre-LN RoPE blocks -> final_norm
//!          encoder_proj 512 -> 192
//!          sub-pixel conv k3 192 -> 192 * 8, reshape -> [192, 8 * T_concat]   (10 ms frames)
//!          relu -> Linear -> relu -> Linear -> sigmoid -> [n_spk, 8 * T_concat]
//! ```
//!
//! Ported from the V3 branch of NeMo-Speech.cpp's `SortformerGraph::build_graph`
//! and `rope_transformer.cpp` (Apache-2.0), checked against HF transformers'
//! `modeling_nemotron3_diarization.py`, which is the reference the baseline
//! probabilities were recorded with. Unlike v2 there is no xscale, no rel-pos
//! table and no conv module; positions are plain RoPE indices that restart at
//! every chunk (HF: "positions restart at every chunk").
//!
//! The RoPE pairing is NeoX-style (`rotate_half`: dim `i` pairs with
//! `i + head_dim / 2`), so the graph uses `GGML_ROPE_TYPE_NEOX`; the
//! interleaved default would pair `2i` with `2i + 1` and silently scramble
//! every attention score.

use crate::sf_graph::ChunkOutput;
use crate::sf_ops::{layer_norm, sys, Context, Tensor};
use crate::sf_runtime::{Graph, Runtime};
use crate::sf_transformer::diar_head;
use crate::sf_weights::{RopeBlockView, SfError, SfResult, SortformerWeights};

/// HF's `max_position_embeddings`; only YaRN scaling reads it, which is off.
const ROPE_CTX_ORIG: i32 = 5000;

/// `mul_mat` with `GGML_PREC_F32`. On the Vulkan backend (RTX 2070) the default
/// precision moved v3 probabilities by up to 0.02 against HF over 31 blocks;
/// with this it is 0.005. CPU ignores the flag.
unsafe fn mm(ctx: Context, w: Tensor, x: Tensor) -> Tensor {
    let y = sys::ggml_mul_mat(ctx, w, x);
    sys::ggml_mul_mat_set_prec(y, sys::ggml_prec_GGML_PREC_F32);
    y
}

/// `y = W x + b` with [`mm`].
unsafe fn linear(ctx: Context, w: Tensor, x: Tensor, bias: Tensor) -> Tensor {
    sys::ggml_add(ctx, mm(ctx, w, x), bias)
}

/// Feature stacking (HF `Nemotron3DiarizationFeatureStacking`): every
/// `factor` consecutive mel frames become one `n_mels * factor` vector, frame
/// slower than mel bin, and the last incomplete group is zero-padded.
///
/// `mel` is mel-major (`mel[m * frames + f]`, the seam's layout); the result is
/// row-major `[groups, n_mels * factor]`, i.e. ggml `[n_mels * factor, groups]`.
/// Done on the host because it is a pure re-indexing and is easier to test
/// here than as a graph reshape.
pub(crate) fn stack_frames(mel: &[f32], frames: usize, n_mels: usize, factor: usize) -> (Vec<f32>, usize) {
    let groups = frames.div_ceil(factor);
    let width = n_mels * factor;
    let mut out = vec![0.0f32; groups * width];
    for m in 0..n_mels {
        for f in 0..frames {
            out[(f / factor) * width + (f % factor) * n_mels + m] = mel[m * frames + f];
        }
    }
    (out, groups)
}

/// One pre-LN encoder block: `x + Attn(LN(x))`, then `x + FFN(LN(x))` with an
/// exact (erf) GELU — HF's `"gelu"`, not the tanh approximation `ggml_gelu`
/// computes.
///
/// `mask` is null or an f32 `[T, T]` additive key mask (0 / -inf).
unsafe fn rope_block(
    ctx: Context,
    x: Tensor,
    positions: Tensor,
    mask: Tensor,
    b: &RopeBlockView,
    d_model: i64,
    heads: i64,
    rope_base: f32,
) -> Tensor {
    let time = (*x).ne[1];
    let head_dim = d_model / heads;

    let xn = layer_norm(ctx, x, b.norm1_w, b.norm1_b);
    let qkv = mm(ctx, b.qkv_w, xn);
    let part = |index: i64| {
        let view = sys::ggml_view_3d(
            ctx,
            qkv,
            head_dim,
            heads,
            time,
            head_dim as usize * (*qkv).nb[0],
            (*qkv).nb[1],
            index as usize * d_model as usize * (*qkv).nb[0],
        );
        sys::ggml_cont(ctx, view)
    };
    let rope = |t: Tensor| {
        sys::ggml_rope_ext(
            ctx,
            t,
            positions,
            std::ptr::null_mut(),
            head_dim as i32,
            sys::GGML_ROPE_TYPE_NEOX as i32,
            ROPE_CTX_ORIG,
            rope_base,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
    };
    // [head_dim, heads, T] -> [head_dim, T, heads].
    let q = sys::ggml_cont(ctx, sys::ggml_permute(ctx, rope(part(0)), 0, 2, 1, 3));
    let k = sys::ggml_cont(ctx, sys::ggml_permute(ctx, rope(part(1)), 0, 2, 1, 3));
    let v = sys::ggml_cont(ctx, sys::ggml_permute(ctx, part(2), 0, 2, 1, 3));

    // ponytail: manual mul_mat + soft_max on every backend. Concats here stay
    // under ~700 frames, so flash attention would buy little; switch if the
    // chunk geometry ever grows.
    let kq = mm(ctx, k, q);
    let kq = sys::ggml_soft_max_ext(ctx, kq, mask, 1.0 / (head_dim as f32).sqrt(), 0.0);
    let vt = sys::ggml_cont(ctx, sys::ggml_permute(ctx, v, 1, 0, 2, 3));
    let o = mm(ctx, vt, kq);
    let o = sys::ggml_cont(ctx, sys::ggml_permute(ctx, o, 0, 2, 1, 3));
    let o = sys::ggml_reshape_2d(ctx, o, d_model, time);
    let x = sys::ggml_add(ctx, x, linear(ctx, b.out_w, o, b.out_b));

    let h = layer_norm(ctx, x, b.norm2_w, b.norm2_b);
    let h = linear(ctx, b.ff_in_w, h, b.ff_in_b);
    let h = sys::ggml_gelu_erf(ctx, h);
    sys::ggml_add(ctx, x, linear(ctx, b.ff_out_w, h, b.ff_out_b))
}

/// Runs one v3 chunk.
///
/// `mel` is the chunk window, mel-major, `mel_frames` long; frames at or past
/// `valid_mel_frames` are padding. An encoder frame is valid when its first
/// mel frame is (HF: `attention_mask[:, ::subsampling_factor]`); invalid ones
/// stay in the sequence as queries but are masked out as keys, exactly as
/// HF's bidirectional mask does. `cache` is the `[spkcache | fifo]` prefix,
/// row-major `[n_cached, d_model]`.
///
/// Returns the chunk's embeddings (every encoder frame of the window,
/// look-ahead included) and sigmoid probabilities at the upsampled rate over
/// the whole concat, row-major `[(n_cached + T) * upsample, n_spk]`.
pub fn run_chunk_v3(
    model: &SortformerWeights,
    mel: &[f32],
    mel_frames: usize,
    valid_mel_frames: usize,
    cache: &[f32],
) -> SfResult<ChunkOutput> {
    let hp = model.hparams();
    let v3 = model.v3().ok_or(SfError::Ggml("run_chunk_v3 on a v2 model"))?;
    let d_model = hp.enc_d_model as usize;
    let factor = hp.enc_subsampling_factor as usize;
    let n_mels = hp.fe_num_mels as usize;
    let up = hp.upsample_factor as usize;

    let (stacked, t_new) = stack_frames(mel, mel_frames, n_mels, factor);
    let n_cached = cache.len() / d_model;
    let t_concat = n_cached + t_new;
    let n_valid = n_cached + valid_mel_frames.div_ceil(factor).min(t_new);

    // RoPE wants I32 positions; the runtime uploads f32 slices, so carry the
    // integers' bit patterns through unchanged.
    let positions: Vec<f32> = (0..t_concat).map(|i| f32::from_bits(i as u32)).collect();
    let mask: Vec<f32> = (0..t_concat * t_concat)
        .map(|i| if i % t_concat < n_valid { 0.0 } else { f32::NEG_INFINITY })
        .collect();

    let runtime = Runtime::new(model.backend())?;
    unsafe {
        let mut graph = Graph::new()?;
        let ctx = graph.ctx;
        let f32_ty = sys::ggml_type_GGML_TYPE_F32;

        let stacked_in = sys::ggml_new_tensor_2d(ctx, f32_ty, (n_mels * factor) as i64, t_new as i64);
        let emb = mm(ctx, v3.stack_proj_w, stacked_in);
        let cache_in = if n_cached > 0 {
            sys::ggml_new_tensor_2d(ctx, f32_ty, d_model as i64, n_cached as i64)
        } else {
            std::ptr::null_mut()
        };
        let mut x = if n_cached > 0 { sys::ggml_concat(ctx, cache_in, emb, 1) } else { emb };

        let pos_in = sys::ggml_new_tensor_1d(ctx, sys::ggml_type_GGML_TYPE_I32, t_concat as i64);
        let mask_in = if n_valid < t_concat {
            sys::ggml_new_tensor_2d(ctx, f32_ty, t_concat as i64, t_concat as i64)
        } else {
            std::ptr::null_mut()
        };

        x = layer_norm(ctx, x, v3.embed_norm_w, v3.embed_norm_b);
        for block in model.rope_blocks() {
            x = rope_block(ctx, x, pos_in, mask_in, block, hp.enc_d_model, hp.enc_heads, hp.enc_rope_base);
        }
        x = layer_norm(ctx, x, v3.final_norm_w, v3.final_norm_b);

        let head = model.head();
        let p = linear(ctx, head.enc_proj_w, x, head.enc_proj_b);
        // Sub-pixel conv as im2col + one matmul: im2col's `[k + 3 * c_in]`
        // column order is the weight's own memory order. im2col runs in F32 so
        // the activations are not rounded to F16 on the way in (`ggml_conv_1d`
        // would), and the F16 weight is widened for the same reason.
        let tf_d = hp.tf_d_model;
        let pt = sys::ggml_cont(ctx, sys::ggml_transpose(ctx, p));
        let pt = sys::ggml_reshape_3d(ctx, pt, t_concat as i64, tf_d, 1);
        let cols = sys::ggml_im2col(ctx, v3.upsample_w, pt, 1, 0, 1, 0, 1, 0, false, f32_ty);
        let cols = sys::ggml_reshape_2d(ctx, cols, 3 * tf_d, t_concat as i64);
        let w = if (*v3.upsample_w).type_ == f32_ty {
            v3.upsample_w
        } else {
            sys::ggml_cast(ctx, v3.upsample_w, f32_ty)
        };
        let w = sys::ggml_reshape_2d(ctx, w, 3 * tf_d, tf_d * up as i64);
        let hi = sys::ggml_add(ctx, mm(ctx, w, cols), v3.upsample_b);
        // [tf_d * up, T] -> [tf_d, up * T]: channel `j * tf_d + c` of frame `t`
        // is channel `c` of 10 ms frame `t * up + j` (HF's reshape).
        let hi = sys::ggml_reshape_2d(ctx, hi, tf_d, up as i64 * t_concat as i64);
        let preds = diar_head(ctx, hi, head);

        graph.output(emb);
        graph.output(preds);

        let mut inputs: Vec<(Tensor, &[f32])> = vec![(stacked_in, &stacked), (pos_in, &positions)];
        if n_cached > 0 {
            inputs.push((cache_in, cache));
        }
        if !mask_in.is_null() {
            inputs.push((mask_in, &mask));
        }
        runtime.execute(&mut graph, &inputs)?;

        Ok(ChunkOutput {
            embeddings: graph.read(emb),
            embedding_frames: t_new,
            predictions: graph.read(preds),
            prediction_frames: t_concat * up,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacking_groups_frames_mel_fastest_and_zero_pads_the_tail() {
        // 2 mel bins x 5 frames, value = 10 * frame + bin; factor 4 -> 2 groups.
        let (n_mels, frames, factor) = (2, 5, 4);
        let mut mel = vec![0.0f32; n_mels * frames];
        for m in 0..n_mels {
            for f in 0..frames {
                mel[m * frames + f] = (10 * f + m) as f32 + 1.0;
            }
        }
        let (stacked, groups) = stack_frames(&mel, frames, n_mels, factor);
        assert_eq!(groups, 2);
        assert_eq!(&stacked[..8], &[1.0, 2.0, 11.0, 12.0, 21.0, 22.0, 31.0, 32.0]);
        // Frame 4 opens group 1; frames 5..7 are the zero padding.
        assert_eq!(&stacked[8..], &[41.0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn stacking_by_eight_matches_the_model_width() {
        let (n_mels, frames) = (128, 17);
        let mel: Vec<f32> = (0..n_mels * frames).map(|i| i as f32).collect();
        let (stacked, groups) = stack_frames(&mel, frames, n_mels, 8);
        assert_eq!(groups, 3);
        assert_eq!(stacked.len(), 3 * 1024);
        // Group 2 holds frame 16 then 7 zero frames.
        let g2 = &stacked[2 * 1024..];
        assert_eq!(g2[5], mel[5 * frames + 16]);
        assert!(g2[128..].iter().all(|&v| v == 0.0));
    }
}
