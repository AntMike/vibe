//! Canary's Transformer decoder, decoded greedily: cross-attention keys and
//! values are projected once per chunk, then each token runs through its own
//! graph, which reads the self-attention cache and appends to it.

use std::ptr;

use super::Model;
use crate::encoder::EncoderWeights;
use crate::ops::{add_bias, layer_norm, Tensor};
use crate::runtime::{accelerated_runtime, CpuRuntime, Graph};
use crate::{sys, Error, Result};

const F32: sys::ggml_type = sys::ggml_type_GGML_TYPE_F32;
const FLOAT: usize = std::mem::size_of::<f32>();

/// Every decoder layer's attention keys and values, kept on the backend
/// across the per-token graphs. Layer `l`'s self-attention rows start at
/// `l * positions * hidden`, its cross-attention rows at `l * frames * hidden`.
struct Cache {
    ctx: *mut sys::ggml_context,
    buffer: sys::ggml_backend_buffer_t,
    self_k: Tensor,
    self_v: Tensor,
    cross_k: Tensor,
    cross_v: Tensor,
    positions: usize,
    frames: usize,
}

impl Cache {
    unsafe fn new(backend: sys::ggml_backend_t, layers: usize, hidden: usize, positions: usize, frames: usize) -> Result<Self> {
        let ctx = sys::ggml_init(sys::ggml_init_params {
            mem_size: sys::ggml_tensor_overhead() * 4,
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        });
        if ctx.is_null() {
            return Err(Error::Ggml("decoder cache context"));
        }
        let self_len = (layers * positions * hidden) as i64;
        let cross_len = (layers * frames * hidden) as i64;
        let mut cache = Self {
            ctx,
            buffer: ptr::null_mut(),
            self_k: sys::ggml_new_tensor_1d(ctx, F32, self_len),
            self_v: sys::ggml_new_tensor_1d(ctx, F32, self_len),
            cross_k: sys::ggml_new_tensor_1d(ctx, F32, cross_len),
            cross_v: sys::ggml_new_tensor_1d(ctx, F32, cross_len),
            positions,
            frames,
        };
        cache.buffer = sys::ggml_backend_alloc_ctx_tensors(ctx, backend);
        if cache.buffer.is_null() {
            return Err(Error::Ggml("decoder cache buffer"));
        }
        Ok(cache)
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                sys::ggml_backend_buffer_free(self.buffer);
            }
            sys::ggml_free(self.ctx);
        }
    }
}

/// Decodes one chunk's encoder output (`[hidden, frames]`) after `prompt`,
/// returning the ids it generated, end of text left out.
pub(super) fn greedy(
    model: &Model,
    encoded: &[f32],
    frames: usize,
    prompt: &[u32],
    should_abort: &mut dyn FnMut() -> bool,
) -> Result<Vec<u32>> {
    let info = &model.info;
    let positions = info.decoder_max_position as usize;
    unsafe {
        let runtime = accelerated_runtime(model.backend())?;
        let cache = Cache::new(
            runtime.backend(),
            info.decoder_layers as usize,
            info.decoder_dimension as usize,
            positions,
            frames,
        )?;
        fill_cross(model, &runtime, &cache, encoded)?;
        // ponytail: the prompt goes in one token per graph (ten graphs); one
        // batched prompt pass needs a causal mask, add it if this shows up.
        let mut logits = Vec::new();
        for (position, &token) in prompt.iter().enumerate() {
            logits = step(model, &runtime, &cache, token, position)?;
        }
        let mut ids = Vec::new();
        for position in prompt.len()..positions {
            let next = argmax(&logits);
            if next == model.prompt.end {
                return Ok(ids);
            }
            ids.push(next);
            if let Some(block) = repeating_block(&ids) {
                trim_repeats(&mut ids, block);
                eprintln!(
                    "[canary][decoder] stopped a {block}-token block repeating; kept {} tokens",
                    ids.len()
                );
                return Ok(ids);
            }
            if should_abort() {
                return Err(Error::Aborted);
            }
            logits = step(model, &runtime, &cache, next, position)?;
        }
        eprintln!("[canary][decoder] filled the {positions}-token context before end of text");
        Ok(ids)
    }
}

unsafe fn weight(model: &Model, name: &str) -> Tensor {
    model.tensor(name).expect("validated tensor catalog")
}

/// Projects the encoder output into every layer's cross-attention keys and values.
unsafe fn fill_cross(model: &Model, runtime: &CpuRuntime, cache: &Cache, encoded: &[f32]) -> Result<()> {
    let hidden = model.info.decoder_dimension as usize;
    let mut graph = Graph::new()?;
    let ctx = graph.ctx;
    let input = sys::ggml_new_tensor_2d(ctx, F32, hidden as i64, cache.frames as i64);
    for layer in 0..model.info.decoder_layers as usize {
        for (name, target) in [("k", cache.cross_k), ("v", cache.cross_v)] {
            let prefix = format!("dec.layer.{layer}.cross_attn.{name}");
            let projected = add_bias(
                ctx,
                sys::ggml_mul_mat(ctx, weight(model, &format!("{prefix}.weight")), input),
                weight(model, &format!("{prefix}.bias")),
            );
            let slot = sys::ggml_view_1d(
                ctx,
                target,
                (cache.frames * hidden) as i64,
                layer * cache.frames * hidden * FLOAT,
            );
            sys::ggml_build_forward_expand(graph.graph, sys::ggml_cpy(ctx, projected, slot));
        }
    }
    runtime.execute(&mut graph, &[(input, encoded)])
}

/// Runs `token` at `position` through the decoder, appends its keys and values
/// to the cache, and returns the logits for the token after it.
unsafe fn step(model: &Model, runtime: &CpuRuntime, cache: &Cache, token: u32, position: usize) -> Result<Vec<f32>> {
    let info = &model.info;
    let hidden = info.decoder_dimension as usize;
    let heads = info.decoder_heads as usize;
    let mut graph = Graph::new()?;
    let ctx = graph.ctx;
    let w = |name: &str| weight(model, name);
    // Ids go in as F32 (exact below 2^24) and are cast for get_rows, as in
    // Parakeet's decoder.
    let token_in = sys::ggml_new_tensor_1d(ctx, F32, 1);
    let position_in = sys::ggml_new_tensor_1d(ctx, F32, 1);
    let row =
        |table: &str, index: Tensor| sys::ggml_get_rows(ctx, w(table), sys::ggml_cast(ctx, index, sys::ggml_type_GGML_TYPE_I32));
    let mut x = sys::ggml_add(
        ctx,
        row("dec.embed.token.weight", token_in),
        row("dec.embed.pos_enc", position_in),
    );
    x = layer_norm(ctx, x, w("dec.embed.norm.weight"), w("dec.embed.norm.bias"));
    for layer in 0..info.decoder_layers as usize {
        let name = |part: &str, kind: &str| format!("dec.layer.{layer}.{part}.{kind}");
        let linear = |part: &str, x: Tensor| {
            add_bias(
                ctx,
                sys::ggml_mul_mat(ctx, w(&name(part, "weight")), x),
                w(&name(part, "bias")),
            )
        };
        let norm = |part: &str, x: Tensor| layer_norm(ctx, x, w(&name(part, "weight")), w(&name(part, "bias")));

        // Self-attention over every position so far, this one included. The
        // cache writes go into the graph before the reads that depend on them.
        let h = norm("norm1", x);
        let offset = layer * cache.positions * hidden;
        for (part, target) in [("self_attn.k", cache.self_k), ("self_attn.v", cache.self_v)] {
            let slot = sys::ggml_view_1d(ctx, target, hidden as i64, (offset + position * hidden) * FLOAT);
            sys::ggml_build_forward_expand(graph.graph, sys::ggml_cpy(ctx, linear(part, h), slot));
        }
        let attended = attend(
            ctx,
            linear("self_attn.q", h),
            (cache.self_k, cache.self_v),
            offset,
            position + 1,
            heads,
        );
        x = sys::ggml_add(ctx, x, linear("self_attn.o", attended));

        let h = norm("norm2", x);
        let attended = attend(
            ctx,
            linear("cross_attn.q", h),
            (cache.cross_k, cache.cross_v),
            layer * cache.frames * hidden,
            cache.frames,
            heads,
        );
        x = sys::ggml_add(ctx, x, linear("cross_attn.o", attended));

        let up = linear("ffn.up", norm("norm3", x));
        let up = match info.decoder_activation.as_str() {
            "relu" => sys::ggml_relu(ctx, up),
            _ => sys::ggml_silu(ctx, up),
        };
        x = sys::ggml_add(ctx, x, linear("ffn.down", up));
    }
    x = layer_norm(ctx, x, w("dec.norm.weight"), w("dec.norm.bias"));
    let logits = add_bias(ctx, sys::ggml_mul_mat(ctx, w("dec.head.weight"), x), w("dec.head.bias"));
    graph.output(logits);
    runtime.execute(&mut graph, &[(token_in, &[token as f32]), (position_in, &[position as f32])])?;
    let mut values = vec![0.0f32; (*logits).ne[0] as usize];
    sys::ggml_backend_tensor_get(logits, values.as_mut_ptr().cast(), 0, values.len() * FLOAT);
    Ok(values)
}

/// Multi-head attention of one query (`[hidden, 1]`) over `length` cached
/// rows of keys and values starting at element `offset`.
unsafe fn attend(
    ctx: *mut sys::ggml_context,
    query: Tensor,
    (keys, values): (Tensor, Tensor),
    offset: usize,
    length: usize,
    heads: usize,
) -> Tensor {
    let hidden = (*query).ne[0] as usize;
    let head = hidden / heads;
    let view = |cache: Tensor| {
        sys::ggml_view_3d(
            ctx,
            cache,
            head as i64,
            length as i64,
            heads as i64,
            hidden * FLOAT,
            head * FLOAT,
            offset * FLOAT,
        )
    };
    let query = sys::ggml_permute(
        ctx,
        sys::ggml_reshape_3d(ctx, query, head as i64, heads as i64, 1),
        0,
        2,
        1,
        3,
    );
    let scores = sys::ggml_mul_mat(ctx, view(keys), query);
    let weights = sys::ggml_soft_max_ext(ctx, scores, ptr::null_mut(), 1.0 / (head as f32).sqrt(), 0.0);
    let values = sys::ggml_cont(ctx, sys::ggml_permute(ctx, view(values), 1, 0, 2, 3));
    let out = sys::ggml_cont(
        ctx,
        sys::ggml_permute(ctx, sys::ggml_mul_mat(ctx, values, weights), 0, 2, 1, 3),
    );
    sys::ggml_reshape_2d(ctx, out, hidden as i64, 1)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(id, _)| id as u32)
}

// Greedy decoding has no way out once a phrase's likeliest continuation is the
// phrase itself. transcribe.cpp's stop bar: a block repeating at the tail needs
// `COPIES` copies spanning at least `MIN_TOKENS`, or, once the span passes
// `MAX_TOKENS`, at least `MIN_COPIES`. A 1-token block needs 64 copies, so
// emphatic repetition survives.
const MAX_BLOCK: usize = 128;
const COPIES: usize = 8;
const MIN_TOKENS: usize = 64;
const MAX_TOKENS: usize = 192;
const MIN_COPIES: usize = 4;

/// Length of the block looping at the end of `ids`, if one is.
fn repeating_block(ids: &[u32]) -> Option<usize> {
    let n = ids.len();
    (1..=MAX_BLOCK).find(|&block| {
        let span = (COPIES * block).clamp(MIN_TOKENS, MAX_TOKENS);
        let span = block * MIN_COPIES.max(span.div_ceil(block));
        span <= n && (n - span + block..n).all(|i| ids[i] == ids[i - block])
    })
}

/// Drops every copy of the tail block but the first.
fn trim_repeats(ids: &mut Vec<u32>, block: usize) {
    while ids.len() >= 2 * block && ids[ids.len() - block..] == ids[ids.len() - 2 * block..ids.len() - block] {
        ids.truncate(ids.len() - block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_a_looping_phrase_and_keeps_one_copy() {
        let mut ids = vec![9, 8];
        for _ in 0..7 {
            ids.extend([1, 2, 3]);
        }
        // 7 copies of a 3-token block span 21 < 64 tokens: not yet a loop.
        assert_eq!(repeating_block(&ids), None);
        for _ in 0..15 {
            ids.extend([1, 2, 3]);
        }
        let block = repeating_block(&ids).unwrap();
        assert_eq!(block, 3);
        trim_repeats(&mut ids, block);
        assert_eq!(ids, [9, 8, 1, 2, 3]);
    }

    #[test]
    fn a_repeated_word_needs_64_copies() {
        assert_eq!(repeating_block(&[5; 63]), None);
        assert_eq!(repeating_block(&[5; 64]), Some(1));
    }
}
