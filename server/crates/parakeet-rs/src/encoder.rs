use crate::ops::{add_conv_bias, attention, conv_2d_depthwise, conv_module, layer_norm, macaron, Context, Tensor};
use crate::runtime::Graph;
use crate::{sys, MelFeatures, Result};

/// What the FastConformer encoder reads from a loaded model. Parakeet and
/// Canary ship the same encoder under the same `enc.*` tensor names; Canary's
/// also carries biases, which Parakeet's GGUFs leave out.
pub(crate) trait EncoderWeights {
    fn tensor(&self, name: &str) -> Option<Tensor>;
    fn encoder_shape(&self) -> EncoderShape;
    /// The GPU backend holding the weights, or `None` when they stay on the CPU.
    fn backend(&self) -> Option<sys::ggml_backend_t>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EncoderShape {
    pub layers: i32,
    pub dimension: i32,
    pub heads: i32,
    pub conv_kernel: i32,
    pub xscaling: bool,
}

unsafe fn symmetric_pad(ctx: Context, mut x: Tensor) -> Tensor {
    let ty = (*x).type_;
    let left = sys::ggml_fill(
        ctx,
        sys::ggml_new_tensor_4d(ctx, ty, 1, (*x).ne[1], (*x).ne[2], (*x).ne[3]),
        0.0,
    );
    x = sys::ggml_concat(ctx, left, x, 0);
    let right = sys::ggml_fill(
        ctx,
        sys::ggml_new_tensor_4d(ctx, ty, 1, (*x).ne[1], (*x).ne[2], (*x).ne[3]),
        0.0,
    );
    x = sys::ggml_concat(ctx, x, right, 0);
    let top = sys::ggml_fill(
        ctx,
        sys::ggml_new_tensor_4d(ctx, ty, (*x).ne[0], 1, (*x).ne[2], (*x).ne[3]),
        0.0,
    );
    x = sys::ggml_concat(ctx, top, x, 1);
    let bottom = sys::ggml_fill(
        ctx,
        sys::ggml_new_tensor_4d(ctx, ty, (*x).ne[0], 1, (*x).ne[2], (*x).ne[3]),
        0.0,
    );
    sys::ggml_concat(ctx, x, bottom, 1)
}

unsafe fn weight(model: &impl EncoderWeights, name: &str) -> Tensor {
    model.tensor(name).expect("validated tensor catalog")
}

/// A bias the model may not ship: null when it is absent.
fn optional(model: &impl EncoderWeights, name: &str) -> Tensor {
    model.tensor(name).unwrap_or(std::ptr::null_mut())
}

pub(crate) unsafe fn build_pre_encode(ctx: Context, model: &impl EncoderWeights, mel: Tensor) -> Tensor {
    let mut x = sys::ggml_cont(ctx, sys::ggml_permute(ctx, mel, 1, 0, 2, 3));
    x = symmetric_pad(ctx, x);
    x = sys::ggml_conv_2d(ctx, weight(model, "enc.pre_encode.conv.0.weight"), x, 2, 2, 0, 0, 1, 1);
    x = sys::ggml_relu(ctx, add_conv_bias(ctx, x, weight(model, "enc.pre_encode.conv.0.bias")));

    x = symmetric_pad(ctx, x);
    x = conv_2d_depthwise(ctx, weight(model, "enc.pre_encode.conv.2.weight"), x, 2, 2, 0, 0);
    x = add_conv_bias(ctx, x, weight(model, "enc.pre_encode.conv.2.bias"));
    x = sys::ggml_conv_2d(ctx, weight(model, "enc.pre_encode.conv.3.weight"), x, 1, 1, 0, 0, 1, 1);
    x = sys::ggml_relu(ctx, add_conv_bias(ctx, x, weight(model, "enc.pre_encode.conv.3.bias")));

    x = symmetric_pad(ctx, x);
    x = conv_2d_depthwise(ctx, weight(model, "enc.pre_encode.conv.5.weight"), x, 2, 2, 0, 0);
    x = add_conv_bias(ctx, x, weight(model, "enc.pre_encode.conv.5.bias"));
    x = sys::ggml_conv_2d(ctx, weight(model, "enc.pre_encode.conv.6.weight"), x, 1, 1, 0, 0, 1, 1);
    x = sys::ggml_relu(ctx, add_conv_bias(ctx, x, weight(model, "enc.pre_encode.conv.6.bias")));

    let freq = (*x).ne[0];
    let time = (*x).ne[1];
    let channels = (*x).ne[2];
    let batch = (*x).ne[3];
    x = sys::ggml_cont(ctx, sys::ggml_permute(ctx, x, 0, 2, 1, 3));
    x = sys::ggml_reshape_3d(ctx, x, freq * channels, time, batch);
    x = sys::ggml_mul_mat(ctx, weight(model, "enc.pre_encode.out.weight"), x);
    let bias = weight(model, "enc.pre_encode.out.bias");
    let bias = sys::ggml_reshape_4d(ctx, bias, (*bias).ne[0], 1, 1, 1);
    sys::ggml_add(ctx, x, bias)
}

unsafe fn block(
    ctx: Context,
    model: &impl EncoderWeights,
    mut x: Tensor,
    pos: Tensor,
    mask: Tensor,
    layer: i32,
    promote_pointwise: bool,
) -> Tensor {
    let n = |suffix: &str| format!("enc.blocks.{layer}.{suffix}");
    let w = |suffix: &str| weight(model, &n(suffix));
    let b = |suffix: &str| optional(model, &n(suffix));
    let shape = model.encoder_shape();
    x = macaron(
        ctx,
        x,
        w("norm_ff1.weight"),
        w("norm_ff1.bias"),
        w("ff1.linear1.weight"),
        b("ff1.linear1.bias"),
        w("ff1.linear2.weight"),
        b("ff1.linear2.bias"),
    );
    let norm = layer_norm(ctx, x, w("norm_attn.weight"), w("norm_attn.bias"));
    let attn = attention(
        ctx,
        norm,
        pos,
        mask,
        (w("attn.linear_q.weight"), b("attn.linear_q.bias")),
        (w("attn.linear_k.weight"), b("attn.linear_k.bias")),
        (w("attn.linear_v.weight"), b("attn.linear_v.bias")),
        (w("attn.linear_out.weight"), b("attn.linear_out.bias")),
        w("attn.linear_pos.weight"),
        w("attn.pos_bias_u"),
        w("attn.pos_bias_v"),
        shape.dimension as i64,
        shape.heads as i64,
    );
    x = sys::ggml_add(ctx, x, attn);
    let norm = layer_norm(ctx, x, w("norm_conv.weight"), w("norm_conv.bias"));
    let conv = conv_module(
        ctx,
        norm,
        (w("conv.pointwise1.weight"), b("conv.pointwise1.bias")),
        (w("conv.depthwise.weight"), b("conv.depthwise.bias")),
        w("conv.bn.weight"),
        w("conv.bn.bias"),
        w("conv.bn.running_mean"),
        w("conv.bn.running_var"),
        (w("conv.pointwise2.weight"), b("conv.pointwise2.bias")),
        shape.dimension as i64,
        shape.conv_kernel as i64,
        promote_pointwise,
    );
    x = sys::ggml_add(ctx, x, conv);
    x = macaron(
        ctx,
        x,
        w("norm_ff2.weight"),
        w("norm_ff2.bias"),
        w("ff2.linear1.weight"),
        b("ff2.linear1.bias"),
        w("ff2.linear2.weight"),
        b("ff2.linear2.bias"),
    );
    layer_norm(ctx, x, w("norm_out.weight"), w("norm_out.bias"))
}

pub(crate) struct EncoderBuild {
    pub output: Tensor,
    pub position: Tensor,
    pub mask: Tensor,
}

pub(crate) unsafe fn build_encoder(
    ctx: Context,
    model: &impl EncoderWeights,
    mel: Tensor,
    promote_pointwise: bool,
) -> EncoderBuild {
    let shape = model.encoder_shape();
    let mut x = build_pre_encode(ctx, model, mel);
    if shape.xscaling {
        x = sys::ggml_scale(ctx, x, (shape.dimension as f32).sqrt());
    }
    let time = (*x).ne[1];
    let dimension = shape.dimension as i64;
    let position = sys::ggml_new_tensor_2d(ctx, sys::ggml_type_GGML_TYPE_F32, dimension, 2 * time - 1);
    let mask = sys::ggml_new_tensor_4d(ctx, sys::ggml_type_GGML_TYPE_F32, time, time, 1, 1);
    for layer in 0..shape.layers {
        eprintln!("[parakeet][graph] build_encoder_layer={}/{}", layer + 1, shape.layers);
        x = block(ctx, model, x, position, mask, layer, promote_pointwise);
    }
    EncoderBuild {
        output: x,
        position,
        mask,
    }
}

/// Runs the whole encoder over `features`: `[dimension, frames, 1]` on the host.
pub(crate) fn encode(model: &impl EncoderWeights, features: &MelFeatures) -> Result<(Vec<f32>, [i64; 3])> {
    unsafe {
        let runtime = crate::runtime::accelerated_runtime(model.backend())?;
        let mut graph = Graph::new()?;
        let input = sys::ggml_new_tensor_4d(
            graph.ctx,
            sys::ggml_type_GGML_TYPE_F32,
            features.num_frames as i64,
            features.num_mels as i64,
            1,
            1,
        );
        let built = build_encoder(graph.ctx, model, input, model.backend().is_none());
        graph.output(built.output);
        let time = (*built.output).ne[1] as usize;
        let positions = positional_embedding(model.encoder_shape().dimension as usize, time);
        let mask = full_mask(time);
        runtime.execute(
            &mut graph,
            &[(input, &features.data), (built.position, &positions), (built.mask, &mask)],
        )?;
        let shape = [(*built.output).ne[0], (*built.output).ne[1], (*built.output).ne[2]];
        let count = shape.iter().product::<i64>() as usize;
        let mut data = vec![0.0f32; count];
        sys::ggml_backend_tensor_get(built.output, data.as_mut_ptr().cast(), 0, data.len() * 4);
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / data.len().max(1) as f32;
        let rms = (data.iter().map(|value| value * value).sum::<f32>() / data.len().max(1) as f32).sqrt();
        eprintln!("[parakeet][encoder] shape={shape:?} min={min:.6} max={max:.6} mean={mean:.6} rms={rms:.6}");
        Ok((data, shape))
    }
}

pub(crate) fn positional_embedding(dimension: usize, time: usize) -> Vec<f32> {
    let length = 2 * time - 1;
    let zero = (length - 1) / 2;
    let mut result = vec![0.0; length * dimension];
    for i in 0..length {
        let position = (zero as isize - i as isize) as f32;
        for k in 0..dimension / 2 {
            let div = ((2 * k) as f32 * (-10000.0f32.ln() / dimension as f32)).exp();
            result[i * dimension + 2 * k] = (position * div).sin();
            result[i * dimension + 2 * k + 1] = (position * div).cos();
        }
    }
    result
}

pub(crate) fn full_mask(time: usize) -> Vec<f32> {
    vec![0.0; time * time]
}
