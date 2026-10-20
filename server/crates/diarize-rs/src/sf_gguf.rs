//! Reading a Sortformer GGUF file: the hyper-parameter KVs, the borrowed
//! tensor views each sub-module needs, and the catalog lookup that fills them.
//!
//! Nothing here owns a context or touches a backend — [`crate::sf_weights`]
//! does the owning, the GPU upload and the BatchNorm fusion, and calls into
//! this module while it builds a `SortformerWeights`.
//!
//! Names and KV keys follow `plans/transcribe.cpp/scripts/convert-sortformer.py`:
//!
//! ```text
//! enc.pre_encode.*        DwStridingSubsampling stem (x8)
//! encoder.layers.{i}.*    17 NEST FastConformer blocks, d_model 512
//! encoder_proj.*          Linear 512 -> 192
//! transformer.layers.{i}.* 18 post-LN Transformer blocks, d_model 192
//! head.first_hidden_to_hidden.* head hidden layer
//! head.single_hidden_to_spks.* head output (192 -> 4)
//! ```
//!
//! v3 (`sortformer.version = "v3"`) replaces the stem and the conformer and
//! transformer stacks with `encoder.pre_encode.proj` (feature stacking),
//! `encoder.{embed,final}_norm`, `encoder.layers.{i}.{norm1,attn.w_qkv,
//! attn.out_proj,norm2,ffn.net.0,ffn.net.3}` and `subpixel_upsample`, plus
//! `learnable_sil_emb`; the head and `encoder_proj` are shared.

use std::ffi::{CStr, CString};
use std::ptr;

use crate::host::{CHUNK_LEN, FIFO_LEN, RIGHT_CONTEXT, SPKCACHE_LEN};
use crate::sf_ops::{sys, Tensor};
use crate::sf_weights::{SfError, SfResult};

/// Everything the graph builders need to know about geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerHParams {
    pub max_speakers: i64,
    /// Samples per diarization frame: `hop_length * subsampling_factor`.
    pub frame_hop: i64,

    pub enc_layers: i64,
    pub enc_d_model: i64,
    pub enc_heads: i64,
    pub enc_d_ff: i64,
    pub enc_conv_kernel: i64,
    pub enc_subsampling_factor: i64,
    pub enc_subsampling_channels: i64,
    pub enc_feat_in: i64,
    pub enc_conv_norm_type: String,

    pub tf_layers: i64,
    pub tf_d_model: i64,
    pub tf_heads: i64,
    pub tf_d_ff: i64,
    pub tf_activation: String,
    pub tf_pre_ln: bool,

    pub fe_num_mels: i64,
    pub fe_sample_rate: i64,
    pub fe_n_fft: i64,
    pub fe_win_length: i64,
    pub fe_hop_length: i64,

    pub stream_chunk_len: i64,
    pub stream_spkcache_len: i64,
    pub stream_fifo_len: i64,
    pub stream_spkcache_update_period: i64,
    pub stream_right_context: i64,

    /// `sortformer.version`: `"v2"` (conformer + transformer, 80 ms frames)
    /// or `"v3"` (Nemotron-3-Diarization: RoPE transformer, 10 ms frames).
    /// Files written before v3 existed carry no such key and are v2.
    pub version: String,
    /// Head upsampling back to the mel frame rate; 1 for v2, 8 for v3.
    pub upsample_factor: i64,
    /// v3 fills disabled cache slots with a trained `learnable_sil_emb`
    /// instead of v2's running mean of silent frames.
    pub learnable_silence: bool,
    pub enc_rope_base: f32,
    pub scoring: Scoring,
}

impl SortformerHParams {
    pub fn is_v3(&self) -> bool {
        self.version == "v3"
    }
}

/// AOSC cache-compression parameters (`sortformer.scoring.*`). Absent keys
/// fall back to the v2 values the host was ported with.
#[derive(Debug, Clone, PartialEq)]
pub struct Scoring {
    pub sil_frames_per_spk: usize,
    pub pred_score_threshold: f32,
    /// Added to the scores of the frames just popped from the FIFO, so that on
    /// a tie the cache prefers recent material. v2's host never applied it.
    pub scores_boost_latest: f32,
    pub sil_threshold: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
}

impl Default for Scoring {
    fn default() -> Self {
        use crate::host::{
            MIN_POS_SCORES_RATE, PRED_SCORE_THRESHOLD, SIL_THRESHOLD, SPKCACHE_SIL_FRAMES_PER_SPK, STRONG_BOOST_RATE,
            WEAK_BOOST_RATE,
        };
        Self {
            sil_frames_per_spk: SPKCACHE_SIL_FRAMES_PER_SPK,
            pred_score_threshold: PRED_SCORE_THRESHOLD,
            scores_boost_latest: 0.0,
            sil_threshold: SIL_THRESHOLD,
            strong_boost_rate: STRONG_BOOST_RATE,
            weak_boost_rate: WEAK_BOOST_RATE,
            min_pos_scores_rate: MIN_POS_SCORES_RATE,
        }
    }
}

/// One v3 encoder block: pre-LN self-attention with RoPE and a fused,
/// bias-free QKV projection, then a pre-LN GELU feed-forward.
#[derive(Clone, Copy)]
pub(crate) struct RopeBlockView {
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    /// `[d, 3d]`: q, k, v stacked along the output axis in that order.
    pub qkv_w: Tensor,
    pub out_w: Tensor,
    pub out_b: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    pub ff_in_w: Tensor,
    pub ff_in_b: Tensor,
    pub ff_out_w: Tensor,
    pub ff_out_b: Tensor,
}

/// The v3 tensors outside the encoder blocks (the head is shared with v2).
#[derive(Clone, Copy)]
pub(crate) struct V3View {
    /// Feature stacking: `[n_mels * 8, d]`, no bias.
    pub stack_proj_w: Tensor,
    pub embed_norm_w: Tensor,
    pub embed_norm_b: Tensor,
    pub final_norm_w: Tensor,
    pub final_norm_b: Tensor,
    /// Sub-pixel conv, ggml `[3, tf_d, tf_d * upsample]`.
    pub upsample_w: Tensor,
    pub upsample_b: Tensor,
}

/// Borrowed pointers to one FastConformer block's tensors. Mirrors
/// `conformer::BlockView`; a null bias means "skip the add".
#[derive(Clone, Copy)]
pub(crate) struct ConformerBlockView {
    pub norm_ff1_w: Tensor,
    pub norm_ff1_b: Tensor,
    pub ff1_lin1_w: Tensor,
    pub ff1_lin1_b: Tensor,
    pub ff1_lin2_w: Tensor,
    pub ff1_lin2_b: Tensor,

    pub norm_attn_w: Tensor,
    pub norm_attn_b: Tensor,
    pub attn_q_w: Tensor,
    pub attn_q_b: Tensor,
    pub attn_k_w: Tensor,
    pub attn_k_b: Tensor,
    pub attn_v_w: Tensor,
    pub attn_v_b: Tensor,
    pub attn_out_w: Tensor,
    pub attn_out_b: Tensor,
    pub attn_pos_w: Tensor,
    pub attn_pos_u: Tensor,
    pub attn_pos_v: Tensor,

    pub norm_conv_w: Tensor,
    pub norm_conv_b: Tensor,
    pub conv_pw1_w: Tensor,
    pub conv_pw1_b: Tensor,
    pub conv_dw_w: Tensor,
    pub conv_dw_b: Tensor,
    pub conv_pw2_w: Tensor,
    pub conv_pw2_b: Tensor,
    /// Raw BatchNorm parameters, only read during load-time fusion.
    pub conv_bn_w: Tensor,
    pub conv_bn_b: Tensor,
    pub conv_bn_rm: Tensor,
    pub conv_bn_rv: Tensor,
    /// Fused `gamma / sqrt(var + eps)` and `beta - mean * scale`, filled by
    /// [`crate::sf_weights::fuse_batch_norm`]. These, not the raw four above, are what the graph
    /// consumes.
    pub conv_bn_fused_scale: Tensor,
    pub conv_bn_fused_bias: Tensor,

    pub norm_ff2_w: Tensor,
    pub norm_ff2_b: Tensor,
    pub ff2_lin1_w: Tensor,
    pub ff2_lin1_b: Tensor,
    pub ff2_lin2_w: Tensor,
    pub ff2_lin2_b: Tensor,

    pub norm_out_w: Tensor,
    pub norm_out_b: Tensor,
}

/// The pre_encode (DwStridingSubsampling) stem.
#[derive(Clone, Copy)]
pub(crate) struct PreEncodeView {
    pub conv0_w: Tensor,
    pub conv0_b: Tensor,
    pub conv2_w: Tensor,
    pub conv2_b: Tensor,
    pub conv3_w: Tensor,
    pub conv3_b: Tensor,
    pub conv5_w: Tensor,
    pub conv5_b: Tensor,
    pub conv6_w: Tensor,
    pub conv6_b: Tensor,
    pub out_w: Tensor,
    pub out_b: Tensor,
}

/// One post-LN Transformer encoder block. Every linear carries a bias.
#[derive(Clone, Copy)]
pub(crate) struct TransformerBlockView {
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    pub attn_q_w: Tensor,
    pub attn_q_b: Tensor,
    pub attn_k_w: Tensor,
    pub attn_k_b: Tensor,
    pub attn_v_w: Tensor,
    pub attn_v_b: Tensor,
    pub attn_o_w: Tensor,
    pub attn_o_b: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    pub ff_in_w: Tensor,
    pub ff_in_b: Tensor,
    pub ff_out_w: Tensor,
    pub ff_out_b: Tensor,
}

#[derive(Clone, Copy)]
pub(crate) struct DiarHeadView {
    pub enc_proj_w: Tensor,
    pub enc_proj_b: Tensor,
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    pub single_spk_head_w: Tensor,
    pub single_spk_head_b: Tensor,
}
// ---- tensor catalog ---------------------------------------------------- //

pub(crate) unsafe fn tensor(ctx: *mut sys::ggml_context, name: &str) -> SfResult<Tensor> {
    let c_name = CString::new(name).map_err(|_| SfError::InvalidPath)?;
    let tensor = sys::ggml_get_tensor(ctx, c_name.as_ptr());
    if tensor.is_null() {
        return Err(SfError::MissingTensor(name.to_owned()));
    }
    Ok(tensor)
}

/// Look a tensor up and check its two fastest dims. Pass `-1` to skip a dim.
unsafe fn checked(ctx: *mut sys::ggml_context, name: &str, ne0: i64, ne1: i64) -> SfResult<Tensor> {
    let t = tensor(ctx, name)?;
    let have = [(*t).ne[0], (*t).ne[1]];
    if (ne0 >= 0 && have[0] != ne0) || (ne1 >= 0 && have[1] != ne1) {
        return Err(SfError::TensorShape {
            name: name.to_owned(),
            have,
            want: [ne0, ne1],
        });
    }
    Ok(t)
}

pub(crate) unsafe fn resolve_pre_encode(ctx: *mut sys::ggml_context) -> SfResult<PreEncodeView> {
    Ok(PreEncodeView {
        conv0_w: tensor(ctx, "encoder.pre_encode.conv.0.weight")?,
        conv0_b: tensor(ctx, "encoder.pre_encode.conv.0.bias")?,
        conv2_w: tensor(ctx, "encoder.pre_encode.conv.2.weight")?,
        conv2_b: tensor(ctx, "encoder.pre_encode.conv.2.bias")?,
        conv3_w: tensor(ctx, "encoder.pre_encode.conv.3.weight")?,
        conv3_b: tensor(ctx, "encoder.pre_encode.conv.3.bias")?,
        conv5_w: tensor(ctx, "encoder.pre_encode.conv.5.weight")?,
        conv5_b: tensor(ctx, "encoder.pre_encode.conv.5.bias")?,
        conv6_w: tensor(ctx, "encoder.pre_encode.conv.6.weight")?,
        conv6_b: tensor(ctx, "encoder.pre_encode.conv.6.bias")?,
        out_w: tensor(ctx, "encoder.pre_encode.out.weight")?,
        out_b: tensor(ctx, "encoder.pre_encode.out.bias")?,
    })
}

pub(crate) unsafe fn resolve_conformer_block(
    ctx: *mut sys::ggml_context,
    layer: i64,
    hp: &SortformerHParams,
) -> SfResult<ConformerBlockView> {
    let d = hp.enc_d_model;
    let dff = hp.enc_d_ff;
    let name = |suffix: &str| format!("encoder.layers.{layer}.{suffix}");
    let get = |suffix: &str| tensor(ctx, &name(suffix));
    let get2 = |suffix: &str, ne0: i64, ne1: i64| checked(ctx, &name(suffix), ne0, ne1);
    Ok(ConformerBlockView {
        norm_ff1_w: get2("norm_feed_forward1.weight", d, -1)?,
        norm_ff1_b: get("norm_feed_forward1.bias")?,
        ff1_lin1_w: get2("feed_forward1.linear1.weight", d, dff)?,
        ff1_lin1_b: get("feed_forward1.linear1.bias")?,
        ff1_lin2_w: get2("feed_forward1.linear2.weight", dff, d)?,
        ff1_lin2_b: get("feed_forward1.linear2.bias")?,

        norm_attn_w: get("norm_self_att.weight")?,
        norm_attn_b: get("norm_self_att.bias")?,
        attn_q_w: get2("self_attn.linear_q.weight", d, d)?,
        attn_q_b: get("self_attn.linear_q.bias")?,
        attn_k_w: get2("self_attn.linear_k.weight", d, d)?,
        attn_k_b: get("self_attn.linear_k.bias")?,
        attn_v_w: get2("self_attn.linear_v.weight", d, d)?,
        attn_v_b: get("self_attn.linear_v.bias")?,
        attn_out_w: get2("self_attn.linear_out.weight", d, d)?,
        attn_out_b: get("self_attn.linear_out.bias")?,
        attn_pos_w: get2("self_attn.linear_pos.weight", d, d)?,
        attn_pos_u: get("self_attn.pos_bias_u")?,
        attn_pos_v: get("self_attn.pos_bias_v")?,

        norm_conv_w: get("norm_conv.weight")?,
        norm_conv_b: get("norm_conv.bias")?,
        conv_pw1_w: get("conv.pointwise_conv1.weight")?,
        conv_pw1_b: get("conv.pointwise_conv1.bias")?,
        conv_dw_w: get("conv.depthwise_conv.weight")?,
        conv_dw_b: get("conv.depthwise_conv.bias")?,
        conv_pw2_w: get("conv.pointwise_conv2.weight")?,
        conv_pw2_b: get("conv.pointwise_conv2.bias")?,
        conv_bn_w: get("conv.batch_norm.weight")?,
        conv_bn_b: get("conv.batch_norm.bias")?,
        conv_bn_rm: get("conv.batch_norm.running_mean")?,
        conv_bn_rv: get("conv.batch_norm.running_var")?,
        conv_bn_fused_scale: ptr::null_mut(),
        conv_bn_fused_bias: ptr::null_mut(),

        norm_ff2_w: get("norm_feed_forward2.weight")?,
        norm_ff2_b: get("norm_feed_forward2.bias")?,
        ff2_lin1_w: get2("feed_forward2.linear1.weight", d, dff)?,
        ff2_lin1_b: get("feed_forward2.linear1.bias")?,
        ff2_lin2_w: get2("feed_forward2.linear2.weight", dff, d)?,
        ff2_lin2_b: get("feed_forward2.linear2.bias")?,

        norm_out_w: get("norm_out.weight")?,
        norm_out_b: get("norm_out.bias")?,
    })
}

pub(crate) unsafe fn resolve_transformer_block(
    ctx: *mut sys::ggml_context,
    layer: i64,
    hp: &SortformerHParams,
) -> SfResult<TransformerBlockView> {
    let d = hp.tf_d_model;
    let dff = hp.tf_d_ff;
    let name = |suffix: &str| format!("transformer.layers.{layer}.{suffix}");
    let get = |suffix: &str| tensor(ctx, &name(suffix));
    let get2 = |suffix: &str, ne0: i64, ne1: i64| checked(ctx, &name(suffix), ne0, ne1);
    Ok(TransformerBlockView {
        norm1_w: get2("layer_norm_1.weight", d, -1)?,
        norm1_b: get("layer_norm_1.bias")?,
        attn_q_w: get2("first_sub_layer.query_net.weight", d, d)?,
        attn_q_b: get("first_sub_layer.query_net.bias")?,
        attn_k_w: get2("first_sub_layer.key_net.weight", d, d)?,
        attn_k_b: get("first_sub_layer.key_net.bias")?,
        attn_v_w: get2("first_sub_layer.value_net.weight", d, d)?,
        attn_v_b: get("first_sub_layer.value_net.bias")?,
        attn_o_w: get2("first_sub_layer.out_projection.weight", d, d)?,
        attn_o_b: get("first_sub_layer.out_projection.bias")?,
        norm2_w: get("layer_norm_2.weight")?,
        norm2_b: get("layer_norm_2.bias")?,
        ff_in_w: get2("second_sub_layer.dense_in.weight", d, dff)?,
        ff_in_b: get("second_sub_layer.dense_in.bias")?,
        ff_out_w: get2("second_sub_layer.dense_out.weight", dff, d)?,
        ff_out_b: get("second_sub_layer.dense_out.bias")?,
    })
}

pub(crate) unsafe fn resolve_rope_block(ctx: *mut sys::ggml_context, layer: i64, hp: &SortformerHParams) -> SfResult<RopeBlockView> {
    let d = hp.enc_d_model;
    let dff = hp.enc_d_ff;
    let name = |suffix: &str| format!("encoder.layers.{layer}.{suffix}");
    let get = |suffix: &str| tensor(ctx, &name(suffix));
    let get2 = |suffix: &str, ne0: i64, ne1: i64| checked(ctx, &name(suffix), ne0, ne1);
    Ok(RopeBlockView {
        norm1_w: get2("norm1.weight", d, -1)?,
        norm1_b: get("norm1.bias")?,
        qkv_w: get2("attn.w_qkv.weight", d, 3 * d)?,
        out_w: get2("attn.out_proj.weight", d, d)?,
        out_b: get("attn.out_proj.bias")?,
        norm2_w: get2("norm2.weight", d, -1)?,
        norm2_b: get("norm2.bias")?,
        ff_in_w: get2("ffn.net.0.weight", d, dff)?,
        ff_in_b: get("ffn.net.0.bias")?,
        ff_out_w: get2("ffn.net.3.weight", dff, d)?,
        ff_out_b: get("ffn.net.3.bias")?,
    })
}

pub(crate) unsafe fn resolve_v3(ctx: *mut sys::ggml_context, hp: &SortformerHParams) -> SfResult<V3View> {
    let d = hp.enc_d_model;
    let tf = hp.tf_d_model;
    Ok(V3View {
        stack_proj_w: checked(ctx, "encoder.pre_encode.proj.weight", hp.fe_num_mels * hp.enc_subsampling_factor, d)?,
        embed_norm_w: checked(ctx, "encoder.embed_norm.weight", d, -1)?,
        embed_norm_b: checked(ctx, "encoder.embed_norm.bias", d, -1)?,
        final_norm_w: checked(ctx, "encoder.final_norm.weight", d, -1)?,
        final_norm_b: checked(ctx, "encoder.final_norm.bias", d, -1)?,
        upsample_w: checked(ctx, "subpixel_upsample.weight", 3, tf)?,
        upsample_b: checked(ctx, "subpixel_upsample.bias", tf * hp.upsample_factor, -1)?,
    })
}

pub(crate) unsafe fn resolve_head(ctx: *mut sys::ggml_context, hp: &SortformerHParams) -> SfResult<DiarHeadView> {
    let d = hp.tf_d_model;
    let spk = hp.max_speakers;
    Ok(DiarHeadView {
        enc_proj_w: checked(ctx, "encoder_proj.weight", hp.enc_d_model, d)?,
        enc_proj_b: checked(ctx, "encoder_proj.bias", d, -1)?,
        fc1_w: checked(ctx, "head.first_hidden_to_hidden.weight", d, d)?,
        fc1_b: checked(ctx, "head.first_hidden_to_hidden.bias", d, -1)?,
        single_spk_head_w: checked(ctx, "head.single_hidden_to_spks.weight", d, spk)?,
        single_spk_head_b: checked(ctx, "head.single_hidden_to_spks.bias", spk, -1)?,
    })
}
// ---- metadata ---------------------------------------------------------- //

unsafe fn find_key(ctx: *const sys::gguf_context, name: &str) -> Option<i64> {
    let name = CString::new(name).ok()?;
    let id = sys::gguf_find_key(ctx, name.as_ptr());
    (id >= 0).then_some(id)
}

unsafe fn u32_kv(ctx: *const sys::gguf_context, name: &str) -> SfResult<i64> {
    find_key(ctx, name)
        .map(|id| sys::gguf_get_val_u32(ctx, id) as i64)
        .ok_or_else(|| SfError::MissingMetadata(name.to_owned()))
}

unsafe fn f32_kv(ctx: *const sys::gguf_context, name: &str) -> SfResult<f32> {
    let id = find_key(ctx, name).ok_or_else(|| SfError::MissingMetadata(name.to_string()))?;
    Ok(sys::gguf_get_val_f32(ctx, id))
}

unsafe fn opt_u32(ctx: *const sys::gguf_context, name: &str) -> Option<i64> {
    find_key(ctx, name).map(|id| sys::gguf_get_val_u32(ctx, id) as i64)
}

unsafe fn opt_f32(ctx: *const sys::gguf_context, name: &str) -> Option<f32> {
    find_key(ctx, name).map(|id| sys::gguf_get_val_f32(ctx, id))
}

unsafe fn opt_bool(ctx: *const sys::gguf_context, name: &str) -> Option<bool> {
    find_key(ctx, name).map(|id| sys::gguf_get_val_bool(ctx, id))
}

unsafe fn opt_string(ctx: *const sys::gguf_context, name: &str) -> Option<String> {
    find_key(ctx, name).and_then(|_| string_kv(ctx, name).ok())
}

pub(crate) unsafe fn string_kv(ctx: *const sys::gguf_context, name: &str) -> SfResult<String> {
    let id = find_key(ctx, name).ok_or_else(|| SfError::MissingMetadata(name.to_owned()))?;
    let value = sys::gguf_get_val_str(ctx, id);
    if value.is_null() {
        return Err(SfError::MissingMetadata(name.to_owned()));
    }
    Ok(CStr::from_ptr(value).to_string_lossy().into_owned())
}

pub(crate) unsafe fn read_hparams(ctx: *const sys::gguf_context) -> SfResult<SortformerHParams> {
    // NVIDIA's converter stores the frontend geometry the way NeMo's config does —
    // window sizes in seconds — and derives the rest at load time. Do the same
    // rounding it does rather than inventing keys the file does not carry.
    let sample_rate = u32_kv(ctx, "sortformer.preprocessor.sample_rate")?;
    let win_length = (f32_kv(ctx, "sortformer.preprocessor.window_size")? * sample_rate as f32).round() as i64;
    let hop_length = (f32_kv(ctx, "sortformer.preprocessor.window_stride")? * sample_rate as f32).round() as i64;
    let subsampling = u32_kv(ctx, "sortformer.encoder.subsampling_factor")?;

    Ok(SortformerHParams {
        max_speakers: u32_kv(ctx, "sortformer.num_speakers")?,
        // One encoder frame spans `hop * subsampling` input samples: 160 * 8 = 1280,
        // i.e. 80 ms at 16 kHz.
        frame_hop: hop_length * subsampling,

        enc_layers: u32_kv(ctx, "sortformer.encoder.n_layers")?,
        enc_d_model: u32_kv(ctx, "sortformer.encoder.d_model")?,
        enc_heads: u32_kv(ctx, "sortformer.encoder.n_heads")?,
        enc_d_ff: u32_kv(ctx, "sortformer.encoder.d_ff")?,
        enc_conv_kernel: u32_kv(ctx, "sortformer.encoder.conv_kernel_size")?,
        enc_subsampling_factor: subsampling,
        enc_subsampling_channels: u32_kv(ctx, "sortformer.encoder.subsampling_conv_channels")?,
        enc_feat_in: u32_kv(ctx, "sortformer.encoder.feat_in")?,
        enc_conv_norm_type: string_kv(ctx, "sortformer.encoder.conv_norm")?,

        tf_layers: u32_kv(ctx, "sortformer.transformer.n_layers")?,
        tf_d_model: u32_kv(ctx, "sortformer.transformer.hidden_size")?,
        tf_heads: u32_kv(ctx, "sortformer.transformer.n_heads")?,
        tf_d_ff: u32_kv(ctx, "sortformer.transformer.inner_size")?,
        // Not a stored key: NVIDIA's converter refuses any checkpoint whose
        // `hidden_act` is not relu, so relu is the only value that can reach here.
        tf_activation: "relu".to_string(),
        tf_pre_ln: find_key(ctx, "sortformer.transformer.pre_ln")
            .map(|id| sys::gguf_get_val_bool(ctx, id))
            .unwrap_or(false),

        fe_num_mels: u32_kv(ctx, "sortformer.preprocessor.features")?,
        fe_sample_rate: sample_rate,
        fe_n_fft: u32_kv(ctx, "sortformer.preprocessor.n_fft")?,
        fe_win_length: win_length,
        fe_hop_length: hop_length,

        // Training-time provenance only (188/0/188 for v2, 264/0/264 for v3).
        // Neither runtime streams with these; the host picks its own inference
        // geometry (see `lib.rs` and `host::v3`). Absent keys read as the v2
        // inference geometry the host was ported with.
        stream_chunk_len: opt_u32(ctx, "sortformer.streaming.chunk_len").unwrap_or(CHUNK_LEN as i64),
        stream_spkcache_len: opt_u32(ctx, "sortformer.streaming.spkcache_len").unwrap_or(SPKCACHE_LEN as i64),
        stream_fifo_len: opt_u32(ctx, "sortformer.streaming.fifo_len").unwrap_or(FIFO_LEN as i64),
        stream_spkcache_update_period: opt_u32(ctx, "sortformer.streaming.spkcache_update_period")
            .unwrap_or(CHUNK_LEN as i64),
        stream_right_context: opt_u32(ctx, "sortformer.streaming.chunk_right_context").unwrap_or(RIGHT_CONTEXT as i64),

        version: opt_string(ctx, "sortformer.version").unwrap_or_else(|| "v2".to_owned()),
        upsample_factor: opt_u32(ctx, "sortformer.upsample_factor").unwrap_or(1),
        learnable_silence: opt_bool(ctx, "sortformer.learnable_silence").unwrap_or(false),
        enc_rope_base: opt_f32(ctx, "sortformer.encoder.rope_base").unwrap_or(10000.0),
        scoring: {
            let d = Scoring::default();
            let f = |key: &str, fallback: f32| opt_f32(ctx, &format!("sortformer.scoring.{key}")).unwrap_or(fallback);
            Scoring {
                sil_frames_per_spk: opt_u32(ctx, "sortformer.scoring.spkcache_sil_frames_per_spk")
                    .map_or(d.sil_frames_per_spk, |v| v as usize),
                pred_score_threshold: f("pred_score_threshold", d.pred_score_threshold),
                scores_boost_latest: f("scores_boost_latest", d.scores_boost_latest),
                sil_threshold: f("sil_threshold", d.sil_threshold),
                strong_boost_rate: f("strong_boost_rate", d.strong_boost_rate),
                weak_boost_rate: f("weak_boost_rate", d.weak_boost_rate),
                min_pos_scores_rate: f("min_pos_scores_rate", d.min_pos_scores_rate),
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn set_u32(ctx: *mut sys::gguf_context, key: &str, value: u32) {
        let key = CString::new(key).unwrap();
        sys::gguf_set_val_u32(ctx, key.as_ptr(), value);
    }

    unsafe fn set_f32(ctx: *mut sys::gguf_context, key: &str, value: f32) {
        let key = CString::new(key).unwrap();
        sys::gguf_set_val_f32(ctx, key.as_ptr(), value);
    }

    unsafe fn set_str(ctx: *mut sys::gguf_context, key: &str, value: &str) {
        let (key, value) = (CString::new(key).unwrap(), CString::new(value).unwrap());
        sys::gguf_set_val_str(ctx, key.as_ptr(), value.as_ptr());
    }

    /// Only the keys `read_hparams` cannot do without, with v2's values.
    unsafe fn mandatory_keys() -> *mut sys::gguf_context {
        let ctx = sys::gguf_init_empty();
        set_str(ctx, "general.architecture", "sortformer");
        for (key, value) in [
            ("sortformer.preprocessor.sample_rate", 16000),
            ("sortformer.encoder.subsampling_factor", 8),
            ("sortformer.num_speakers", 4),
            ("sortformer.encoder.n_layers", 17),
            ("sortformer.encoder.d_model", 512),
            ("sortformer.encoder.n_heads", 8),
            ("sortformer.encoder.d_ff", 2048),
            ("sortformer.encoder.conv_kernel_size", 9),
            ("sortformer.encoder.subsampling_conv_channels", 256),
            ("sortformer.encoder.feat_in", 128),
            ("sortformer.transformer.n_layers", 18),
            ("sortformer.transformer.hidden_size", 192),
            ("sortformer.transformer.n_heads", 8),
            ("sortformer.transformer.inner_size", 768),
            ("sortformer.preprocessor.features", 128),
            ("sortformer.preprocessor.n_fft", 512),
        ] {
            set_u32(ctx, key, value);
        }
        set_f32(ctx, "sortformer.preprocessor.window_size", 0.025);
        set_f32(ctx, "sortformer.preprocessor.window_stride", 0.01);
        set_str(ctx, "sortformer.encoder.conv_norm", "batch_norm");
        ctx
    }

    #[test]
    fn absent_optional_keys_read_as_v2_defaults() {
        unsafe {
            let ctx = mandatory_keys();
            let hp = read_hparams(ctx).unwrap();
            sys::gguf_free(ctx);
            assert_eq!(hp.version, "v2");
            assert!(!hp.is_v3());
            assert_eq!(hp.upsample_factor, 1);
            assert!(!hp.learnable_silence);
            assert_eq!(hp.scoring, Scoring::default());
            assert_eq!(hp.stream_chunk_len, CHUNK_LEN as i64);
            assert_eq!(hp.stream_spkcache_len, SPKCACHE_LEN as i64);
            assert_eq!(hp.stream_fifo_len, FIFO_LEN as i64);
            assert_eq!(hp.stream_right_context, RIGHT_CONTEXT as i64);
            assert_eq!((hp.fe_win_length, hp.fe_hop_length, hp.frame_hop), (400, 160, 1280));
        }
    }

    #[test]
    fn v3_keys_are_read_from_the_file() {
        unsafe {
            let ctx = mandatory_keys();
            set_str(ctx, "sortformer.version", "v3");
            set_u32(ctx, "sortformer.upsample_factor", 8);
            set_f32(ctx, "sortformer.scoring.scores_boost_latest", 0.05);
            set_u32(ctx, "sortformer.scoring.spkcache_sil_frames_per_spk", 1);
            set_f32(ctx, "sortformer.encoder.rope_base", 5000.0);
            set_u32(ctx, "sortformer.streaming.spkcache_len", 264);
            let hp = read_hparams(ctx).unwrap();
            sys::gguf_free(ctx);
            assert!(hp.is_v3());
            assert_eq!(hp.upsample_factor, 8);
            assert_eq!(hp.scoring.scores_boost_latest, 0.05);
            assert_eq!(hp.scoring.sil_frames_per_spk, 1);
            // Keys left out still fall back individually.
            assert_eq!(hp.scoring.pred_score_threshold, Scoring::default().pred_score_threshold);
            assert_eq!(hp.enc_rope_base, 5000.0);
            assert_eq!(hp.stream_spkcache_len, 264);
            assert_eq!(hp.stream_chunk_len, CHUNK_LEN as i64);
        }
    }
}
