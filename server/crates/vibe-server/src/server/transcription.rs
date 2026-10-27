use std::collections::HashMap;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use whisper_rs::{TranscribeOptions, Window};

use crate::audio;
use crate::cli::AppConfig;
use crate::server::diarization;
use crate::server::form::FormValues;
use crate::server::stream::stream_transcription;
use crate::server::unload_timeout::ModelLease;
use crate::server::{error, format, TextResponse};

pub(super) struct TranscriptionRequest {
    pub file: Vec<u8>,
    pub form: HashMap<String, String>,
}

pub(super) async fn transcribe(
    config: AppConfig,
    mut model: ModelLease,
    request: TranscriptionRequest,
) -> Result<Response, Response> {
    let TranscriptionRequest { file, form } = request;
    let (windows, diarize_model, enhance_audio, stable_timestamps, vad_model_path, stream, response_format) = {
        let values = FormValues::new(&form);
        (
            values.string("windows"),
            values.string("diarize_model"),
            values.bool("enhance_audio"),
            values.bool("stable_timestamps"),
            values.string("vad_model"),
            values.bool("stream"),
            values.str_or("response_format", "json").to_string(),
        )
    };
    let audio_options = audio::ReadOptions {
        enhance_audio,
        verbose: config.verbose(),
    };
    let samples = audio::read_bytes_with_options(file, audio_options).map_err(|err| {
        error(
            StatusCode::BAD_REQUEST,
            "invalid_audio",
            &format!("invalid audio file: {err}"),
        )
    })?;

    let mut diar_segments = diarize_model
        .as_deref()
        .map_or_else(Vec::new, |model_path| diarization::diarize(model_path, &samples));

    if stable_timestamps && vad_model_path.is_none() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "'vad_model' is required when 'stable_timestamps' is true",
        ));
    }

    let ctx = model
        .ctx
        .as_mut()
        .ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "no_model", "no model loaded"))?;

    if ctx.requires_vad() && vad_model_path.is_none() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "'vad_model' is required for this transcription engine",
        ));
    }

    let mut opts = build_options(&form, config.verbose(), stable_timestamps, vad_model_path);
    if let Some(unknown) = opts
        .languages
        .iter()
        .find(|lang| !whisper_rs::supported_languages().contains(lang))
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &format!("unknown language in 'languages': {unknown}"),
        ));
    }
    // Asked-for stretches: only those are decoded, the way one speaker's lines
    // are redone in another language. They stand in for the speaker turns.
    if let Some(windows) = windows {
        if !ctx.transcribes_windows(&opts) {
            return Err(error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "'windows' is not supported by this engine or with 'stable_timestamps'",
            ));
        }
        opts.windows = parse_windows(&windows, samples.len()).ok_or_else(|| {
            error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "'windows' must be comma-separated start-end pairs in seconds",
            )
        })?;
        diar_segments.clear();
    // Each speaker turn is transcribed on its own, so a line never spans two
    // speakers and each turn is decoded in its own language. Speakers are then
    // matched against the chunks decoded rather than the raw turns.
    } else if ctx.transcribes_windows(&opts) && !diar_segments.is_empty() {
        diar_segments = diarization::speaker_chunks(&diar_segments, &samples);
        opts.windows = diar_segments.iter().map(window).collect();
    }

    if stream {
        return Ok(stream_transcription(model, samples, opts, diar_segments));
    }

    let result = ctx.transcribe(&samples, opts).map_err(|err| {
        tracing::error!(samples = samples.len(), "transcription failed: {err:#}");
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            super::engine_error_code(&err),
            &format!("transcription failed: {err}"),
        )
    })?;

    Ok(format_response(&response_format, &result, &diar_segments))
}

pub(super) fn build_options(
    form: &HashMap<String, String>,
    verbose: bool,
    stable_timestamps: bool,
    vad_model_path: Option<String>,
) -> TranscribeOptions {
    let values = FormValues::new(form);
    TranscribeOptions {
        language: values.string("language"),
        detect_language: values.bool("detect_language"),
        translate: values.bool("translate"),
        threads: values.i32("n_threads"),
        prompt: values.string("prompt"),
        carry_prompt: values.bool("carry_prompt"),
        verbose,
        temperature: values.f32("temperature"),
        max_text_ctx: values.i32("max_text_ctx"),
        word_timestamps: values.bool("word_timestamps"),
        max_segment_len: values.i32("max_segment_len"),
        sampling_greedy: form.get("sampling_strategy").is_none_or(|strategy| strategy != "beam_search"),
        best_of: values.i32("best_of"),
        beam_size: values.i32("beam_size"),
        stable_timestamps,
        vad_model_path,
        languages: comma_list(values.string("languages")),
        windows: Vec::new(),
        draft_runs: values.i32("draft_runs"),
        hotwords: comma_list(values.string("hotwords")),
        raw_languages: values.bool("raw_languages"),
        entropy_thold: values.opt_f32("entropy_thold"),
        logprob_thold: values.opt_f32("logprob_thold"),
        no_speech_thold: values.opt_f32("no_speech_thold"),
        group_turns: values.bool("group_turns"),
        context_tokens: values.i32("context_tokens"),
        turn_loudness: values.opt_f32("turn_loudness"),
    }
}

fn comma_list(value: Option<String>) -> Vec<String> {
    value
        .map(|value| {
            value
                .split(',')
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

const SAMPLE_RATE: f64 = 16_000.0;

/// `"1.5-4,10-12.25"` (seconds) as windows of one group, clipped to the audio;
/// None when a pair does not parse or runs backwards.
fn parse_windows(value: &str, len: usize) -> Option<Vec<Window>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (start, end) = pair.split_once('-')?;
            let (start, end): (f64, f64) = (start.trim().parse().ok()?, end.trim().parse().ok()?);
            (start >= 0.0 && end > start).then(|| Window {
                start_sample: ((start * SAMPLE_RATE) as usize).min(len),
                end_sample: ((end * SAMPLE_RATE).ceil() as usize).min(len),
                group: 0,
            })
        })
        .collect()
}

fn window(chunk: &diarization::Segment) -> Window {
    Window {
        start_sample: (chunk.start * SAMPLE_RATE) as usize,
        end_sample: (chunk.end * SAMPLE_RATE).ceil() as usize,
        group: chunk.speaker_id,
    }
}

fn format_response(
    response_format: &str,
    result: &whisper_rs::TranscribeResult,
    diar_segments: &[diarization::Segment],
) -> Response {
    match response_format {
        "verbose_json" => Json(format::build_verbose_json(result, diar_segments)).into_response(),
        "text" => ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], result.text()).into_response(),
        "srt" => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            format::format_srt(&result.segments),
        )
            .into_response(),
        "vtt" => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            format::format_vtt(&result.segments),
        )
            .into_response(),
        _ => Json(TextResponse { text: result.text() }).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_parse_as_seconds_and_clip_to_the_audio() {
        let spans = |windows: Vec<Window>| windows.iter().map(|w| (w.start_sample, w.end_sample)).collect::<Vec<_>>();
        let len = 20 * 16_000;
        assert_eq!(
            parse_windows("1.5-4, 10-25", len).map(spans),
            Some(vec![(24_000, 64_000), (160_000, len)])
        );
        assert!(parse_windows("4-1", len).is_none());
        assert!(parse_windows("1.5", len).is_none());
        assert!(parse_windows("a-b", len).is_none());
    }
}
