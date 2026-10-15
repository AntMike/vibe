//! NVIDIA Canary (canary-1b-v2): Parakeet's FastConformer encoder, with
//! biases, feeding an autoregressive Transformer decoder that a short prompt
//! tells which language it hears and which to write. Ported from
//! transcribe.cpp's `src/arch/canary`.

mod decoder;

use std::path::Path;
use std::time::Instant;

use crate::encoder::{EncoderShape, EncoderWeights};
use crate::load::{find_key, free_gguf, lookup_tensor, open_gguf, optional_bool, string, strings, u32_value};
use crate::model::DeviceWeights;
use crate::{sys, Error, MelConfig, MelFrontend, Result, Tokenizer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub architecture: String,
    pub variant: String,
    pub languages: Vec<String>,
    /// Translations the model was trained on, as `source>target`. Empty
    /// allows every pair.
    pub translation_pairs: Vec<String>,
    pub encoder_layers: i32,
    pub encoder_dimension: i32,
    pub encoder_heads: i32,
    pub encoder_conv_kernel: i32,
    pub decoder_layers: i32,
    pub decoder_dimension: i32,
    pub decoder_heads: i32,
    pub decoder_max_position: i32,
    /// `relu`, or `silu`/`swish`.
    pub decoder_activation: String,
}

/// What one VAD chunk said. Canary has no token timestamps, so a segment
/// spans its whole chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub start_sample: usize,
    pub end_sample: usize,
    pub text: String,
}

/// The control tokens of the canary2 prompt.
struct Prompt {
    space: Option<u32>,
    context: u32,
    transcript: u32,
    emotion: u32,
    pnc: u32,
    no_itn: u32,
    no_timestamp: u32,
    no_diarize: u32,
    end: u32,
    languages: Vec<(String, u32)>,
}

pub struct Model {
    gguf: *mut sys::gguf_context,
    weights: *mut sys::ggml_context,
    info: ModelInfo,
    tokenizer: Tokenizer,
    prompt: Prompt,
    device: Option<DeviceWeights>,
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { free_gguf(self.gguf, self.weights) };
    }
}

unsafe impl Send for Model {}

impl Model {
    /// Reads the metadata of a Canary GGUF without loading its weights. Any
    /// other architecture is an error.
    pub fn metadata(path: impl AsRef<Path>) -> Result<ModelInfo> {
        let (gguf, weights) = open_gguf(path.as_ref(), true)?;
        let info = unsafe { read_info(gguf) };
        unsafe { free_gguf(gguf, weights) };
        info
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let started = Instant::now();
        let (gguf, weights) = open_gguf(path.as_ref(), false)?;
        let read = unsafe {
            read_info(gguf).and_then(|info| {
                let tokenizer = Tokenizer::load_pieces(gguf)?;
                let prompt = read_prompt(gguf, &info, &tokenizer)?;
                Ok((info, tokenizer, prompt))
            })
        };
        let (info, tokenizer, prompt) = match read {
            Ok(read) => read,
            Err(error) => {
                unsafe { free_gguf(gguf, weights) };
                return Err(error);
            }
        };
        let mut model = Self {
            gguf,
            weights,
            info,
            tokenizer,
            prompt,
            device: None,
        };
        model.validate_tensor_catalog()?;
        model.device = unsafe { DeviceWeights::load(model.weights)? };
        eprintln!(
            "[canary][load] done elapsed_ms={} variant={} backend={}",
            started.elapsed().as_millis(),
            model.info.variant,
            if model.device.is_some() { "gpu" } else { "cpu" }
        );
        Ok(model)
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Transcribes `samples` (16 kHz mono) chunk by chunk as the VAD cuts it,
    /// heard as `source` and written in `target`: the same language
    /// transcribes, another translates.
    #[allow(clippy::too_many_arguments)]
    pub fn transcribe_with(
        &self,
        vad: &mut vad_rs::Vad,
        samples: &[f32],
        source: &str,
        target: &str,
        mut should_abort: impl FnMut() -> bool,
        mut on_segment: impl FnMut(&Segment),
        mut on_progress: impl FnMut(i32),
    ) -> Result<Vec<Segment>> {
        let prompt = self.prompt_ids(source, target)?;
        let ranges = vad.segments(samples).map_err(|error| Error::Vad(error.to_string()))?;
        let total = ranges
            .iter()
            .map(|range| range.end_sample - range.start_sample)
            .sum::<usize>()
            .max(1);
        let mut done = 0;
        on_progress(0);
        let mut segments = Vec::new();
        for range in ranges {
            if should_abort() {
                return Err(Error::Aborted);
            }
            let text = self.transcribe_chunk(&samples[range.start_sample..range.end_sample], &prompt, &mut should_abort)?;
            if !text.is_empty() {
                let segment = Segment {
                    start_sample: range.start_sample,
                    end_sample: range.end_sample,
                    text,
                };
                on_segment(&segment);
                segments.push(segment);
            }
            done += range.end_sample - range.start_sample;
            on_progress((done * 100 / total) as i32);
        }
        Ok(segments)
    }

    fn transcribe_chunk(&self, samples: &[f32], prompt: &[u32], should_abort: &mut dyn FnMut() -> bool) -> Result<String> {
        let started = Instant::now();
        let features = MelFrontend::new(MelConfig::default())?.compute(samples)?;
        let (encoded, shape) = crate::encoder::encode(self, &features)?;
        let ids = decoder::greedy(self, &encoded, shape[1] as usize, prompt, should_abort)?;
        eprintln!(
            "[canary][chunk] samples={} frames={} tokens={} elapsed_ms={}",
            samples.len(),
            shape[1],
            ids.len(),
            started.elapsed().as_millis()
        );
        Ok(self.text(&ids))
    }

    /// canary2's prompt for one direction: the empty decoder context, then
    /// emotion, source and target language, punctuation on, and inverse text
    /// normalization, timestamps and diarization off.
    fn prompt_ids(&self, source: &str, target: &str) -> Result<Vec<u32>> {
        let language = |code: &str| {
            self.prompt
                .languages
                .iter()
                .find(|(candidate, _)| candidate == code)
                .map(|&(_, id)| id)
                .ok_or_else(|| Error::UnsupportedLanguage(code.to_owned()))
        };
        let (source_id, target_id) = (language(source)?, language(target)?);
        let pair = format!("{source}>{target}");
        if source != target && !self.info.translation_pairs.is_empty() && !self.info.translation_pairs.contains(&pair) {
            return Err(Error::UnsupportedLanguage(pair));
        }
        let p = &self.prompt;
        Ok(p.space
            .into_iter()
            .chain([
                p.context,
                p.transcript,
                p.emotion,
                source_id,
                target_id,
                p.pnc,
                p.no_itn,
                p.no_timestamp,
                p.no_diarize,
            ])
            .collect())
    }

    /// The transcript without control tokens or `<unk>`, which the vocabulary
    /// emits for characters it lacks.
    fn text(&self, ids: &[u32]) -> String {
        let kept: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| {
                id != self.tokenizer.unknown_id()
                    && self
                        .tokenizer
                        .piece(id)
                        .is_some_and(|piece| !(piece.starts_with("<|") && piece.ends_with("|>")))
            })
            .collect();
        self.tokenizer.decode(&kept).trim().to_owned()
    }

    fn tensor(&self, name: &str) -> Option<*mut sys::ggml_tensor> {
        match &self.device {
            Some(device) => lookup_tensor(device.ctx, name),
            None => lookup_tensor(self.weights, name),
        }
    }

    fn validate_tensor_catalog(&self) -> Result<()> {
        let mut names: Vec<String> = [
            "enc.pre_encode.out.weight",
            "enc.pre_encode.out.bias",
            "dec.embed.token.weight",
            "dec.embed.pos_enc",
            "dec.embed.norm.weight",
            "dec.embed.norm.bias",
            "dec.norm.weight",
            "dec.norm.bias",
            "dec.head.weight",
            "dec.head.bias",
        ]
        .map(String::from)
        .into();
        for conv in [0, 2, 3, 5, 6] {
            for kind in ["weight", "bias"] {
                names.push(format!("enc.pre_encode.conv.{conv}.{kind}"));
            }
        }
        for layer in 0..self.info.encoder_layers {
            for suffix in ENCODER_BLOCK {
                names.push(format!("enc.blocks.{layer}.{suffix}"));
            }
        }
        for layer in 0..self.info.decoder_layers {
            for part in DECODER_BLOCK {
                for kind in ["weight", "bias"] {
                    names.push(format!("dec.layer.{layer}.{part}.{kind}"));
                }
            }
        }
        names.into_iter().try_for_each(|name| {
            lookup_tensor(self.weights, &name)
                .map(|_| ())
                .ok_or(Error::MissingTensor(name))
        })
    }
}

/// The encoder tensors the graph cannot run without; its biases are optional.
const ENCODER_BLOCK: &[&str] = &[
    "norm_ff1.weight",
    "norm_ff1.bias",
    "ff1.linear1.weight",
    "ff1.linear2.weight",
    "norm_attn.weight",
    "norm_attn.bias",
    "attn.linear_q.weight",
    "attn.linear_k.weight",
    "attn.linear_v.weight",
    "attn.linear_out.weight",
    "attn.linear_pos.weight",
    "attn.pos_bias_u",
    "attn.pos_bias_v",
    "norm_conv.weight",
    "norm_conv.bias",
    "conv.pointwise1.weight",
    "conv.depthwise.weight",
    "conv.pointwise2.weight",
    "conv.bn.weight",
    "conv.bn.bias",
    "conv.bn.running_mean",
    "conv.bn.running_var",
    "norm_ff2.weight",
    "norm_ff2.bias",
    "ff2.linear1.weight",
    "ff2.linear2.weight",
    "norm_out.weight",
    "norm_out.bias",
];

/// Each ships a `.weight` and a `.bias`.
const DECODER_BLOCK: &[&str] = &[
    "norm1",
    "norm2",
    "norm3",
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.up",
    "ffn.down",
];

impl EncoderWeights for Model {
    fn tensor(&self, name: &str) -> Option<*mut sys::ggml_tensor> {
        Model::tensor(self, name)
    }

    fn encoder_shape(&self) -> EncoderShape {
        EncoderShape {
            layers: self.info.encoder_layers,
            dimension: self.info.encoder_dimension,
            heads: self.info.encoder_heads,
            conv_kernel: self.info.encoder_conv_kernel,
            // transcribe.cpp's port, checked against NeMo, never scales Canary's input.
            xscaling: false,
        }
    }

    fn backend(&self) -> Option<sys::ggml_backend_t> {
        self.device.as_ref().map(|device| device.backend)
    }
}

fn invalid(key: &'static str, message: impl Into<String>) -> Error {
    Error::InvalidMetadata {
        key,
        message: message.into(),
    }
}

unsafe fn read_info(ctx: *const sys::gguf_context) -> Result<ModelInfo> {
    let architecture = string(ctx, "general.architecture")?;
    if architecture != "canary" {
        return Err(Error::UnsupportedArchitecture(architecture));
    }
    let format = string(ctx, "stt.canary.tokenizer.prompt_format")?;
    if format != "canary2" {
        return Err(invalid(
            "stt.canary.tokenizer.prompt_format",
            format!("only the canary2 prompt is ported, not {format:?}"),
        ));
    }
    if optional_bool(ctx, "stt.canary.decoder.encoder_decoder_proj").unwrap_or(false) {
        return Err(invalid(
            "stt.canary.decoder.encoder_decoder_proj",
            "the encoder-to-decoder projection (canary-180m-flash) is not ported",
        ));
    }
    if u32_value(ctx, "stt.canary.encoder.subsampling_factor")? != 8 || u32_value(ctx, "stt.frontend.num_mels")? != 128 {
        return Err(invalid(
            "stt.canary.encoder.subsampling_factor",
            "only 8x subsampling of 128 mel bins is ported",
        ));
    }
    let activation = string(ctx, "stt.canary.decoder.activation")?;
    if !matches!(activation.as_str(), "relu" | "silu" | "swish") {
        return Err(invalid(
            "stt.canary.decoder.activation",
            format!("unsupported activation {activation:?}"),
        ));
    }
    let info = ModelInfo {
        architecture,
        variant: string(ctx, "stt.variant")?,
        languages: strings(ctx, "general.languages")?,
        translation_pairs: match find_key(ctx, "stt.translation.pairs") {
            Some(_) => strings(ctx, "stt.translation.pairs")?,
            None => Vec::new(),
        },
        encoder_layers: u32_value(ctx, "stt.canary.encoder.n_layers")? as i32,
        encoder_dimension: u32_value(ctx, "stt.canary.encoder.d_model")? as i32,
        encoder_heads: u32_value(ctx, "stt.canary.encoder.n_heads")? as i32,
        encoder_conv_kernel: u32_value(ctx, "stt.canary.encoder.conv_kernel")? as i32,
        decoder_layers: u32_value(ctx, "stt.canary.decoder.n_layers")? as i32,
        decoder_dimension: u32_value(ctx, "stt.canary.decoder.d_model")? as i32,
        decoder_heads: u32_value(ctx, "stt.canary.decoder.n_heads")? as i32,
        decoder_max_position: u32_value(ctx, "stt.canary.decoder.max_position")? as i32,
        decoder_activation: activation,
    };
    // Cross-attention reads the encoder output as is.
    if info.encoder_dimension != info.decoder_dimension {
        return Err(invalid(
            "stt.canary.decoder.d_model",
            "the decoder must be as wide as the encoder",
        ));
    }
    Ok(info)
}

unsafe fn read_prompt(ctx: *const sys::gguf_context, info: &ModelInfo, tokenizer: &Tokenizer) -> Result<Prompt> {
    let piece = |piece: &str| tokenizer.find(piece).ok_or(Error::MissingMetadata("tokenizer.ggml.tokens"));
    let languages = info
        .languages
        .iter()
        .map(|code| {
            find_key(ctx, &format!("stt.canary.special.lang.{code}_id"))
                .map(|id| sys::gguf_get_val_u32(ctx, id))
                .or_else(|| tokenizer.find(&format!("<|{code}|>")))
                .map(|id| (code.clone(), id))
                .ok_or_else(|| Error::UnsupportedLanguage(code.clone()))
        })
        .collect::<Result<_>>()?;
    Ok(Prompt {
        // A single-SentencePiece vocabulary (canary-1b-v2) prefixes the prompt
        // NeMo tokenizes with one `▁`, its dummy prefix.
        space: match optional_bool(ctx, "stt.canary.tokenizer.single_sp").unwrap_or(false) {
            true => Some(piece("\u{2581}")?),
            false => None,
        },
        context: u32_value(ctx, "stt.canary.special.startofcontext_id")?,
        transcript: u32_value(ctx, "stt.canary.special.startoftranscript_id")?,
        emotion: piece("<|emo:undefined|>")?,
        pnc: u32_value(ctx, "stt.canary.special.pnc_id")?,
        no_itn: u32_value(ctx, "stt.canary.special.noitn_id")?,
        no_timestamp: u32_value(ctx, "stt.canary.special.notimestamp_id")?,
        no_diarize: u32_value(ctx, "stt.canary.special.nodiarize_id")?,
        end: u32_value(ctx, "stt.canary.special.endoftext_id")?,
        languages,
    })
}
