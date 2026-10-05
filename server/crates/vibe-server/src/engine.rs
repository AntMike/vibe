use anyhow::{bail, Context as _};
use serde::Serialize;
use whisper_rs::{ContextOptions, Segment, StreamCallbacks, TranscribeOptions, TranscribeResult, Window};

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct EngineCapabilities {
    pub engine: String,
    pub requires_vad: bool,
    pub languages: Vec<String>,
    pub language_detection: bool,
    pub streaming: bool,
    pub translation: bool,
    pub timestamps: bool,
    pub text_prompts: bool,
}

// One engine lives at a time, so the size gap between variants costs nothing.
#[allow(clippy::large_enum_variant)]
pub enum Engine {
    Whisper(whisper_rs::Context),
    Nemotron {
        model: Box<nemotron_rs::Model>,
        vad: Option<(String, vad_rs::Vad)>,
    },
    Parakeet {
        model: Box<parakeet_rs::Model>,
        vad: Option<(String, vad_rs::Vad)>,
    },
}

impl Engine {
    /// Whether `options.windows` is honoured: each window transcribed on its own.
    pub fn transcribes_windows(&self, options: &TranscribeOptions) -> bool {
        match self {
            // Stable timestamps already cut the audio on VAD and ignores windows.
            Self::Whisper(_) => !options.stable_timestamps,
            Self::Parakeet { .. } => true,
            Self::Nemotron { .. } => false,
        }
    }

    pub fn requires_vad(&self) -> bool {
        matches!(self, Self::Nemotron { .. } | Self::Parakeet { .. })
    }

    pub fn load(path: &str, options: ContextOptions) -> anyhow::Result<Self> {
        if path.ends_with(".gguf") {
            if let Ok(info) = parakeet_rs::Model::metadata(path) {
                if info.architecture == "parakeet" && info.variant.contains("v3") {
                    return Ok(Self::Parakeet {
                        model: Box::new(parakeet_rs::Model::load(path)?),
                        vad: None,
                    });
                }
            }
            // A GGUF file is never a whisper model, so report why the GGUF
            // engines rejected it instead of handing it to whisper.cpp.
            let model = nemotron_rs::Model::load(path).context("GGUF file could not be loaded as Parakeet or Nemotron")?;
            return Ok(Self::Nemotron {
                model: Box::new(model),
                vad: None,
            });
        }
        Ok(Self::Whisper(whisper_rs::Context::new(path, options)?))
    }

    pub fn transcribe(&mut self, samples: &[f32], options: TranscribeOptions) -> anyhow::Result<TranscribeResult> {
        match self {
            Self::Whisper(context) => context.transcribe(samples, options).map_err(Into::into),
            Self::Nemotron { model, vad } => {
                if options.translate {
                    bail!("Nemotron does not support translation");
                }
                if options.prompt.is_some() {
                    bail!("Nemotron does not support text prompts");
                }
                let language = if options.detect_language {
                    "auto"
                } else {
                    options.language.as_deref().unwrap_or("en-US")
                };
                let vad_model_path = options
                    .vad_model_path
                    .as_deref()
                    .context("vad_model_path is required for Nemotron")?;
                if vad.as_ref().is_none_or(|(path, _)| path != vad_model_path) {
                    *vad = Some((
                        vad_model_path.to_string(),
                        vad_rs::Vad::new(vad_model_path, vad_rs::Options::default())?,
                    ));
                }
                let result = model
                    .transcribe(&mut vad.as_mut().unwrap().1, samples, language)
                    .context("Nemotron inference failed")?;
                Ok(TranscribeResult {
                    segments: result
                        .segments
                        .into_iter()
                        .filter(|segment| !segment.text.is_empty())
                        .map(|segment| Segment {
                            start: segment.tokens.first().map_or(0, |token| token.frame as i64 * 8),
                            end: segment.tokens.last().map_or(0, |token| (token.frame as i64 + 1) * 8),
                            text: segment.text,
                            no_speech_prob: 0.0,
                        })
                        .collect(),
                })
            }
            Self::Parakeet { .. } => self.transcribe_stream(samples, options, StreamCallbacks::default()),
        }
    }

    pub fn transcribe_stream(
        &mut self,
        samples: &[f32],
        options: TranscribeOptions,
        callbacks: StreamCallbacks<'_>,
    ) -> anyhow::Result<TranscribeResult> {
        match self {
            Self::Whisper(context) => context.transcribe_stream(samples, options, callbacks).map_err(Into::into),
            Self::Nemotron { model, vad } => {
                if options.translate {
                    bail!("Nemotron does not support translation");
                }
                if options.prompt.is_some() {
                    bail!("Nemotron does not support text prompts");
                }
                let language = if options.detect_language {
                    "auto"
                } else {
                    options.language.as_deref().unwrap_or("en-US")
                };
                let vad_model_path = options
                    .vad_model_path
                    .as_deref()
                    .context("vad_model_path is required for Nemotron")?;
                if vad.as_ref().is_none_or(|(path, _)| path != vad_model_path) {
                    *vad = Some((
                        vad_model_path.to_string(),
                        vad_rs::Vad::new(vad_model_path, vad_rs::Options::default())?,
                    ));
                }
                let StreamCallbacks {
                    mut on_progress,
                    mut on_segment,
                    mut should_abort,
                } = callbacks;
                let result = model.transcribe_with(
                    &mut vad.as_mut().unwrap().1,
                    samples,
                    language,
                    || should_abort.as_mut().is_some_and(|callback| callback()),
                    |transcription| {
                        if let Some(callback) = on_segment.as_mut() {
                            if let Some(segment) = nemotron_segment(transcription) {
                                callback(segment);
                            }
                        }
                    },
                    |progress| {
                        if let Some(callback) = on_progress.as_mut() {
                            callback(progress);
                        }
                    },
                )?;
                Ok(TranscribeResult {
                    segments: result.segments.iter().filter_map(nemotron_segment).collect(),
                })
            }
            Self::Parakeet { model, vad } => {
                validate_parakeet_options(&options)?;
                let language = if options.detect_language {
                    "auto"
                } else {
                    options.language.as_deref().unwrap_or("en")
                };
                let vad = parakeet_vad(vad, options.vad_model_path.as_deref())?;
                let StreamCallbacks {
                    mut on_progress,
                    mut on_segment,
                    mut should_abort,
                } = callbacks;
                // Each window (a speaker turn) is cut on VAD and decoded on its
                // own, so no sentence runs across two speakers.
                let windows = if options.windows.is_empty() {
                    vec![Window {
                        start_sample: 0,
                        end_sample: samples.len(),
                        group: 0,
                    }]
                } else {
                    options.windows.clone()
                };
                let total = windows
                    .iter()
                    .map(|window| window.end_sample.saturating_sub(window.start_sample))
                    .sum::<usize>()
                    .max(1);
                let mut done = 0;
                let mut segments = Vec::new();
                for window in windows {
                    let end = window.end_sample.min(samples.len());
                    let start = window.start_sample.min(end);
                    let len = end - start;
                    // The model drops words that start right at the edge of its
                    // audio, so it hears the end of the previous window too and
                    // keeps only the words timed inside this one. The window
                    // itself ends at a quiet cut: audio that stops mid-word makes
                    // the model drop the whole last sentence.
                    let context_start = start.saturating_sub(PARAKEET_LEAD_IN);
                    let in_window =
                        |token: &parakeet_rs::Token| start == 0 || context_start + token.frame * PARAKEET_FRAME_SAMPLES >= start;
                    let offset_cs = (context_start / 160) as i64;
                    let tokenizer = model.tokenizer();
                    model
                        .transcribe_with(
                            vad,
                            &samples[context_start..end],
                            language,
                            || should_abort.as_mut().is_some_and(|callback| callback()),
                            |transcription| {
                                let tokens: Vec<_> = transcription
                                    .tokens
                                    .iter()
                                    .filter(|token| in_window(token))
                                    .cloned()
                                    .collect();
                                let kept = if tokens.len() == transcription.tokens.len() {
                                    parakeet_segment(transcription)
                                } else {
                                    let ids: Vec<_> = tokens.iter().map(|token| token.id).collect();
                                    parakeet_segment(&parakeet_rs::Transcription {
                                        text: tokenizer.decode_clean(&ids),
                                        tokens,
                                    })
                                };
                                if let Some(mut segment) = kept {
                                    segment.start += offset_cs;
                                    segment.end += offset_cs;
                                    if let Some(callback) = on_segment.as_mut() {
                                        callback(segment.clone());
                                    }
                                    segments.push(segment);
                                }
                            },
                            |progress| {
                                if let Some(callback) = on_progress.as_mut() {
                                    callback(((done + len * progress as usize / 100) * 100 / total) as i32);
                                }
                            },
                        )
                        .context("Parakeet inference failed")?;
                    done += len;
                }
                Ok(TranscribeResult { segments })
            }
        }
    }

    pub fn capabilities(&self) -> EngineCapabilities {
        match self {
            Self::Whisper(_) => whisper_capabilities(),
            Self::Nemotron { model, .. } => EngineCapabilities {
                engine: "nemotron".to_string(),
                requires_vad: true,
                languages: model.info().languages.clone(),
                language_detection: model.info().language_detection,
                streaming: false,
                translation: false,
                timestamps: true,
                text_prompts: false,
            },
            Self::Parakeet { model, .. } => EngineCapabilities {
                engine: "parakeet".to_string(),
                requires_vad: true,
                languages: model.info().languages.clone(),
                language_detection: model.info().language_detection,
                streaming: false,
                translation: false,
                timestamps: true,
                text_prompts: false,
            },
        }
    }
}

pub fn whisper_capabilities() -> EngineCapabilities {
    EngineCapabilities {
        engine: "whisper".to_string(),
        requires_vad: false,
        languages: whisper_rs::supported_languages(),
        language_detection: true,
        streaming: true,
        translation: true,
        timestamps: true,
        text_prompts: true,
    }
}

fn nemotron_segment(transcription: &nemotron_rs::Transcription) -> Option<Segment> {
    (!transcription.text.is_empty()).then(|| Segment {
        start: transcription.tokens.first().map_or(0, |token| token.frame as i64 * 8),
        end: transcription.tokens.last().map_or(0, |token| (token.frame as i64 + 1) * 8),
        text: transcription.text.clone(),
        no_speech_prob: 0.0,
    })
}

/// Half a second (in samples) of the previous window decoded before each Parakeet window.
const PARAKEET_LEAD_IN: usize = 8_000;
/// Samples per Parakeet encoder frame (80 ms), the unit of token timestamps.
const PARAKEET_FRAME_SAMPLES: usize = 1_280;

fn validate_parakeet_options(options: &TranscribeOptions) -> anyhow::Result<()> {
    if options.translate {
        bail!("Parakeet does not support translation");
    }
    if options.prompt.is_some() {
        bail!("Parakeet does not support text prompts");
    }
    Ok(())
}

fn parakeet_vad<'a>(cached: &'a mut Option<(String, vad_rs::Vad)>, path: Option<&str>) -> anyhow::Result<&'a mut vad_rs::Vad> {
    let path = path.context("vad_model_path is required for Parakeet")?;
    if cached.as_ref().is_none_or(|(cached_path, _)| cached_path != path) {
        *cached = Some((path.to_owned(), vad_rs::Vad::new(path, vad_rs::Options::default())?));
    }
    Ok(&mut cached.as_mut().expect("VAD initialized").1)
}

fn parakeet_segment(transcription: &parakeet_rs::Transcription) -> Option<Segment> {
    (!transcription.text.is_empty()).then(|| Segment {
        start: transcription.tokens.first().map_or(0, |token| token.frame as i64 * 8),
        end: transcription
            .tokens
            .last()
            .map_or(0, |token| (token.frame + token.duration_frames.max(1)) as i64 * 8),
        text: transcription.text.clone(),
        no_speech_prob: 0.0,
    })
}
