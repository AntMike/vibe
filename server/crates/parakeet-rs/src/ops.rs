//! Unsafe graph-building primitives for the FastConformer port.

use crate::sys;

pub(crate) type Tensor = *mut sys::ggml_tensor;
pub(crate) type Context = *mut sys::ggml_context;

pub(crate) unsafe fn layer_norm(ctx: Context, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor {
    let mut y = sys::ggml_norm(ctx, x, 1.0e-5);
    y = sys::ggml_mul(ctx, y, gamma);
    if !beta.is_null() {
        y = sys::ggml_add(ctx, y, beta);
    }
    y
}

/// `x + bias`, or `x` when the model ships no bias (null).
pub(crate) unsafe fn add_bias(ctx: Context, x: Tensor, bias: Tensor) -> Tensor {
    if bias.is_null() {
        x
    } else {
        sys::ggml_add(ctx, x, bias)
    }
}

pub(crate) unsafe fn feed_forward(ctx: Context, x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor {
    let y = add_bias(ctx, sys::ggml_mul_mat(ctx, w1, x), b1);
    let y = sys::ggml_silu(ctx, y);
    add_bias(ctx, sys::ggml_mul_mat(ctx, w2, y), b2)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn macaron(
    ctx: Context,
    x: Tensor,
    norm_w: Tensor,
    norm_b: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
) -> Tensor {
    let y = layer_norm(ctx, x, norm_w, norm_b);
    let y = feed_forward(ctx, y, w1, b1, w2, b2);
    let y = sys::ggml_scale(ctx, y, 0.5);
    sys::ggml_add(ctx, x, y)
}

pub(crate) unsafe fn rel_shift(ctx: Context, x: Tensor) -> Tensor {
    let ne = (*x).ne;
    let pos = ne[0];
    let tq = ne[1];
    let heads = ne[2];
    let batch = ne[3];
    let template = sys::ggml_new_tensor_4d(ctx, sys::ggml_type_GGML_TYPE_F32, 1, tq, heads, batch);
    let zeros = sys::ggml_fill(ctx, template, 0.0);
    let mut y = sys::ggml_concat(ctx, zeros, x, 0);
    y = sys::ggml_reshape_4d(ctx, y, tq, pos + 1, heads, batch);
    y = sys::ggml_view_4d(ctx, y, tq, pos, heads, batch, (*y).nb[1], (*y).nb[2], (*y).nb[3], (*y).nb[1]);
    y = sys::ggml_cont(ctx, y);
    sys::ggml_reshape_4d(ctx, y, pos, tq, heads, batch)
}

pub(crate) unsafe fn add_conv_bias(ctx: Context, output: Tensor, bias: Tensor) -> Tensor {
    if bias.is_null() {
        return output;
    }
    let bias = sys::ggml_reshape_4d(ctx, bias, 1, 1, (*bias).ne[0], 1);
    sys::ggml_add(ctx, output, bias)
}

pub(crate) unsafe fn conv_2d_depthwise(
    ctx: Context,
    kernel: Tensor,
    data: Tensor,
    stride0: i32,
    stride1: i32,
    pad0: i32,
    pad1: i32,
) -> Tensor {
    let mut a = sys::ggml_reshape_4d(
        ctx,
        kernel,
        (*kernel).ne[0],
        (*kernel).ne[1],
        1,
        (*kernel).ne[2] * (*kernel).ne[3],
    );
    let data4 = sys::ggml_reshape_4d(ctx, data, (*data).ne[0], (*data).ne[1], 1, (*data).ne[2] * (*data).ne[3]);
    let cols = sys::ggml_im2col(ctx, a, data4, stride0, stride1, pad0, pad1, 1, 1, true, (*kernel).type_);
    let b = sys::ggml_reshape_4d(
        ctx,
        cols,
        (*cols).ne[0],
        (*cols).ne[2] * (*cols).ne[1],
        (*data).ne[2],
        (*data).ne[3],
    );
    a = sys::ggml_reshape_4d(ctx, a, (*a).ne[0] * (*a).ne[1], (*a).ne[2], (*a).ne[3], 1);
    let out = sys::ggml_mul_mat(ctx, a, b);
    sys::ggml_reshape_4d(ctx, out, (*cols).ne[1], (*cols).ne[2], (*data).ne[2], (*data).ne[3])
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attention(
    ctx: Context,
    x: Tensor,
    pos: Tensor,
    mask: Tensor,
    (q_w, q_b): (Tensor, Tensor),
    (k_w, k_b): (Tensor, Tensor),
    (v_w, v_b): (Tensor, Tensor),
    (out_w, out_b): (Tensor, Tensor),
    pos_w: Tensor,
    bias_u: Tensor,
    bias_v: Tensor,
    d_model: i64,
    heads: i64,
) -> Tensor {
    let time = (*x).ne[1];
    let batch = (*x).ne[2];
    let head = d_model / heads;
    let scale = 1.0 / (head as f32).sqrt();
    let mut q = add_bias(ctx, sys::ggml_mul_mat(ctx, q_w, x), q_b);
    let mut k = add_bias(ctx, sys::ggml_mul_mat(ctx, k_w, x), k_b);
    let mut v = add_bias(ctx, sys::ggml_mul_mat(ctx, v_w, x), v_b);
    let mut p = sys::ggml_mul_mat(ctx, pos_w, pos);
    q = sys::ggml_reshape_4d(ctx, q, head, heads, time, batch);
    let mut qu = sys::ggml_add(ctx, q, bias_u);
    let mut qv = sys::ggml_add(ctx, q, bias_v);
    qu = sys::ggml_permute(ctx, qu, 0, 2, 1, 3);
    qv = sys::ggml_cont(ctx, sys::ggml_permute(ctx, qv, 0, 2, 1, 3));
    k = sys::ggml_permute(ctx, sys::ggml_reshape_4d(ctx, k, head, heads, time, batch), 0, 2, 1, 3);
    v = sys::ggml_permute(ctx, sys::ggml_reshape_4d(ctx, v, head, heads, time, batch), 0, 2, 1, 3);
    p = sys::ggml_cont(
        ctx,
        sys::ggml_permute(ctx, sys::ggml_reshape_4d(ctx, p, head, heads, 2 * time - 1, 1), 0, 2, 1, 3),
    );
    let mut relative = rel_shift(ctx, sys::ggml_mul_mat(ctx, p, qv));
    relative = sys::ggml_view_4d(
        ctx,
        relative,
        time,
        time,
        heads,
        batch,
        (*relative).nb[1],
        (*relative).nb[2],
        (*relative).nb[3],
        0,
    );
    relative = sys::ggml_add(ctx, relative, mask);
    relative = sys::ggml_cast(
        ctx,
        sys::ggml_scale(ctx, sys::ggml_cont(ctx, relative), scale),
        sys::ggml_type_GGML_TYPE_F16,
    );
    k = sys::ggml_cast(ctx, k, sys::ggml_type_GGML_TYPE_F16);
    v = sys::ggml_cast(ctx, v, sys::ggml_type_GGML_TYPE_F16);
    let mut output = sys::ggml_flash_attn_ext(ctx, qu, k, v, relative, scale, 0.0, 0.0);
    output = sys::ggml_reshape_3d(ctx, output, d_model, time, batch);
    add_bias(ctx, sys::ggml_mul_mat(ctx, out_w, output), out_b)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn conv_module(
    ctx: Context,
    x: Tensor,
    (pw1_w, pw1_b): (Tensor, Tensor),
    (dw_w, dw_b): (Tensor, Tensor),
    norm_w: Tensor,
    norm_b: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    (pw2_w, pw2_b): (Tensor, Tensor),
    d_model: i64,
    kernel: i64,
    promote_pointwise: bool,
) -> Tensor {
    let time = (*x).ne[1];
    let batch = (*x).ne[2];
    let pw1_w = if promote_pointwise {
        sys::ggml_cast(ctx, pw1_w, sys::ggml_type_GGML_TYPE_F32)
    } else {
        pw1_w
    };
    let pw2_w = if promote_pointwise {
        sys::ggml_cast(ctx, pw2_w, sys::ggml_type_GGML_TYPE_F32)
    } else {
        pw2_w
    };
    let pw1 = sys::ggml_reshape_2d(ctx, pw1_w, d_model, 2 * d_model);
    let mut y = add_bias(ctx, sys::ggml_mul_mat(ctx, pw1, x), pw1_b);
    let gate = sys::ggml_view_3d(ctx, y, d_model, time, batch, (*y).nb[1], (*y).nb[2], 0);
    let value = sys::ggml_view_3d(
        ctx,
        y,
        d_model,
        time,
        batch,
        (*y).nb[1],
        (*y).nb[2],
        d_model as usize * sys::ggml_element_size(y),
    );
    // Some GGML GPU backends reject the standalone SIGMOID op. Express
    // sigmoid(x) = 1 / (1 + exp(-x)) using universally supported primitives.
    let value = sys::ggml_cont(ctx, value);
    let neg = sys::ggml_scale(ctx, value, -1.0);
    let exp = sys::ggml_exp(ctx, neg);
    let ones = sys::ggml_fill(ctx, sys::ggml_dup_tensor(ctx, exp), 1.0);
    let sigmoid = sys::ggml_div(ctx, ones, sys::ggml_add(ctx, ones, exp));
    y = sys::ggml_mul(ctx, gate, sigmoid);
    y = sys::ggml_cont(ctx, sys::ggml_permute(ctx, y, 1, 0, 2, 3));
    let pad = (kernel - 1) / 2;
    let left = sys::ggml_fill(ctx, sys::ggml_new_tensor_4d(ctx, (*y).type_, pad, d_model, batch, 1), 0.0);
    let right = sys::ggml_fill(ctx, sys::ggml_new_tensor_4d(ctx, (*y).type_, pad, d_model, batch, 1), 0.0);
    y = sys::ggml_concat(ctx, sys::ggml_concat(ctx, left, y, 0), right, 0);
    let kernel4 = sys::ggml_reshape_4d(ctx, dw_w, kernel, 1, 1, d_model);
    let data4 = sys::ggml_reshape_4d(ctx, y, (*y).ne[0], 1, d_model, batch);
    y = conv_2d_depthwise(ctx, kernel4, data4, 1, 1, 0, 0);
    y = sys::ggml_reshape_3d(ctx, y, (*y).ne[0], d_model, batch);
    y = add_bias(ctx, sys::ggml_cont(ctx, sys::ggml_permute(ctx, y, 1, 0, 2, 3)), dw_b);
    let epsilon = sys::ggml_fill(ctx, sys::ggml_dup_tensor(ctx, running_var), 1.0e-5);
    let scale = sys::ggml_div(ctx, norm_w, sys::ggml_sqrt(ctx, sys::ggml_add(ctx, running_var, epsilon)));
    y = sys::ggml_add(ctx, sys::ggml_mul(ctx, sys::ggml_sub(ctx, y, running_mean), scale), norm_b);
    y = sys::ggml_silu(ctx, y);
    add_bias(
        ctx,
        sys::ggml_mul_mat(ctx, sys::ggml_reshape_2d(ctx, pw2_w, d_model, d_model), y),
        pw2_b,
    )
}
