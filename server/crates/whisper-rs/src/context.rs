//! The historical whisper-rs API surface (`Context`, `set_verbose`), now
//! implemented on the pure-Rust engine instead of whisper.cpp FFI. Option
//! mapping and callback semantics are kept from the previous implementation
//! so vibe-server is unaffected.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use ggml_rs_sys as ffi;

use crate::{
    model_file, ContextOptions, Error, FullCallbacks, FullParams, FullSegment, Result, SamplingStrategy, Segment,
    StreamCallbacks, TranscribeOptions, TranscribeResult, Whisper, Window, SAMPLE_RATE,
};

pub struct Context {
    whisper: Whisper,
}

impl Context {
    pub fn new(model_path: impl AsRef<Path>, options: ContextOptions) -> Result<Self> {
        let path = model_path.as_ref();
        install_abort_handler();

        // A malformed header should be reported cleanly rather than surfacing
        // as an engine load failure, so the file is checked first.
        let size = model_file::validate(path)?;
        model_file::check_available_memory(path, size)?;

        let mut engine_options = crate::WhisperOptions {
            use_gpu: !options.no_gpu && vulkan_available(),
            ..Default::default()
        };
        if options.gpu_device >= 0 {
            engine_options.gpu_device = options.gpu_device;
        }

        Ok(Self {
            whisper: Whisper::with_options(path, engine_options)?,
        })
    }

    pub fn transcribe(&mut self, samples: &[f32], options: TranscribeOptions) -> Result<TranscribeResult> {
        self.transcribe_stream(samples, options, StreamCallbacks::default())
    }

    pub fn transcribe_stream(
        &mut self,
        samples: &[f32],
        mut options: TranscribeOptions,
        mut callbacks: StreamCallbacks<'_>,
    ) -> Result<TranscribeResult> {
        if samples.is_empty() {
            return Err(Error::NoSamples);
        }
        set_verbose(options.verbose);

        if options.stable_timestamps {
            return transcribe_stable_timestamps(&mut self.whisper, samples, options, callbacks);
        }
        // detect_language alone asks for the language and no segments, which
        // the single-pass path already answers.
        if !options.detect_language {
            // Whisper hears silence as speech: it invents a line for it ("Thanks
            // for watching!") and then carries that line into the speech that
            // follows, losing it. With a VAD at hand it only ever decodes speech.
            if let Some(vad_model_path) = options.vad_model_path.as_deref() {
                let mut vad = vad_rs::Vad::new(vad_model_path, vad_rs::Options::stable_timestamps())
                    .map_err(|error| Error::Message(error.to_string()))?;
                let speech = vad.segments(samples).map_err(|error| Error::Message(error.to_string()))?;
                options.windows = speech_windows(&options.windows, &speech, samples.len());
                if options.windows.is_empty() {
                    if let Some(on_progress) = callbacks.on_progress.as_mut() {
                        on_progress(100);
                    }
                    return Ok(TranscribeResult::default());
                }
            }
            if !options.windows.is_empty() {
                return transcribe_windows(&mut self.whisper, samples, options, callbacks);
            }
        }

        let params = full_params(&options);

        let mut engine_callbacks = FullCallbacks::default();
        if let Some(on_progress) = callbacks.on_progress.as_mut() {
            engine_callbacks.on_progress = Some(Box::new(&mut **on_progress));
        }
        if let Some(on_segment) = callbacks.on_segment.as_mut() {
            engine_callbacks.on_new_segment = Some(Box::new(|segment: &FullSegment| on_segment(convert(segment))));
        }
        if let Some(should_abort) = callbacks.should_abort.as_mut() {
            engine_callbacks.should_abort = Some(Box::new(&mut **should_abort));
        }

        let segments = self.whisper.full_stream(&params, samples, &mut engine_callbacks)?;
        drop(engine_callbacks);

        Ok(TranscribeResult {
            segments: segments.iter().map(convert).collect(),
        })
    }
}

fn convert(segment: &FullSegment) -> Segment {
    Segment {
        start: segment.t0,
        end: segment.t1,
        text: segment.text.clone(),
        no_speech_prob: segment.no_speech_prob,
    }
}

/// The stable-timestamps path: VAD the audio, transcribe each speech segment
/// on its own 30-second-free window and shift the timestamps back. Ported
/// from the previous `stable.rs`.
fn transcribe_stable_timestamps(
    whisper: &mut Whisper,
    samples: &[f32],
    options: TranscribeOptions,
    mut callbacks: StreamCallbacks<'_>,
) -> Result<TranscribeResult> {
    let vad_model_path = options.vad_model_path.as_deref().ok_or(Error::MissingVadModel)?;
    let mut vad = vad_rs::Vad::new(vad_model_path, vad_rs::Options::stable_timestamps())
        .map_err(|error| Error::Message(error.to_string()))?;
    let vad_segments = vad.segments(samples).map_err(|error| Error::Message(error.to_string()))?;
    if vad_segments.is_empty() {
        if let Some(on_progress) = callbacks.on_progress.as_mut() {
            on_progress(100);
        }
        return Ok(TranscribeResult::default());
    }

    let params = full_params(&options);
    let n_vad_segments = vad_segments.len();
    let mut result = TranscribeResult {
        segments: Vec::with_capacity(n_vad_segments),
    };

    for (index, vad_segment) in vad_segments.into_iter().enumerate() {
        if let Some(should_abort) = callbacks.should_abort.as_mut() {
            if should_abort() {
                return Err(Error::Aborted);
            }
        }

        let t0cs = vad_segment.start_centiseconds();
        let window = &samples[vad_segment.start_sample..vad_segment.end_sample];

        let decoded = {
            let mut engine_callbacks = FullCallbacks::default();
            if let Some(should_abort) = callbacks.should_abort.as_mut() {
                engine_callbacks.should_abort = Some(Box::new(&mut **should_abort));
            }
            whisper.full_stream(&params, window, &mut engine_callbacks)?
        };

        for segment in &decoded {
            let mut segment = convert(segment);
            segment.start += t0cs;
            segment.end += t0cs;
            if let Some(on_segment) = callbacks.on_segment.as_mut() {
                on_segment(segment.clone());
            }
            result.segments.push(segment);
        }

        if let Some(on_progress) = callbacks.on_progress.as_mut() {
            on_progress(((index + 1) * 100 / n_vad_segments) as i32);
        }
    }

    Ok(result)
}

/// Utterances are joined into one window up to this long, Whisper's own step;
/// a single longer utterance stays whole, Whisper slides over it itself.
const MAX_WINDOW_SAMPLES: usize = 30 * SAMPLE_RATE;
/// A pause longer than this ends a window, so a long silence is never decoded.
const MAX_GAP_SAMPLES: usize = 2 * SAMPLE_RATE;

/// The speech in each window (the whole recording when there are none), as
/// windows of at most 30 seconds with no long pause inside, in order.
fn speech_windows(windows: &[Window], speech: &[vad_rs::SpeechSegment], len: usize) -> Vec<Window> {
    let whole = [Window {
        start_sample: 0,
        end_sample: len,
        group: 0,
    }];
    let windows = if windows.is_empty() { &whole[..] } else { windows };
    let mut out: Vec<Window> = Vec::new();
    // ponytail: n*m scan; both lists are sorted and short (hundreds at most).
    for window in windows {
        for segment in speech {
            let start = segment.start_sample.max(window.start_sample);
            let end = segment.end_sample.min(window.end_sample);
            if end <= start {
                continue;
            }
            match out.last_mut() {
                Some(last)
                    if last.group == window.group
                        && start >= last.end_sample
                        && start - last.end_sample <= MAX_GAP_SAMPLES
                        && end - last.start_sample <= MAX_WINDOW_SAMPLES =>
                {
                    last.end_sample = end;
                }
                _ => out.push(Window {
                    start_sample: start,
                    end_sample: end,
                    group: window.group,
                }),
            }
        }
    }
    out
}

/// Share of the progress bar spent detecting window languages before decoding.
const DETECT_PROGRESS: usize = 20;

/// The windowed path: transcribe each window (a speaker turn) on its own and
/// shift its timestamps back. With an auto language, each window gets its own
/// detected language, so a recording that switches languages mid-way is not
/// decoded entirely in the language of its first 30 seconds.
fn transcribe_windows(
    whisper: &mut Whisper,
    samples: &[f32],
    options: TranscribeOptions,
    mut callbacks: StreamCallbacks<'_>,
) -> Result<TranscribeResult> {
    let windows: Vec<Window> = options
        .windows
        .iter()
        .map(|window| Window {
            start_sample: window.start_sample.min(samples.len()),
            end_sample: window.end_sample.min(samples.len()),
            group: window.group,
        })
        .filter(|window| window.end_sample > window.start_sample)
        .collect();
    let mut params = full_params(&options);
    let auto = params
        .language
        .as_deref()
        .is_none_or(|lang| lang.is_empty() || lang == "auto");
    let n_windows = windows.len().max(1);

    let languages: Vec<Option<String>> = if !auto {
        vec![params.language.clone(); windows.len()]
    } else if !whisper.is_multilingual() {
        vec![Some("en".to_string()); windows.len()]
    } else {
        let mut detections = Vec::with_capacity(windows.len());
        for (index, window) in windows.iter().enumerate() {
            if callbacks.should_abort.as_mut().is_some_and(|should_abort| should_abort()) {
                return Err(Error::Aborted);
            }
            let (lang, prob) = whisper.detect_language(
                &samples[window.start_sample..window.end_sample],
                params.n_threads,
                &params.languages,
            )?;
            let secs = (window.end_sample - window.start_sample) as f32 / SAMPLE_RATE as f32;
            tracing::debug!(
                start = window.start_sample as f32 / SAMPLE_RATE as f32,
                secs,
                group = window.group,
                lang = Whisper::lang_str(lang),
                prob,
                "window language"
            );
            detections.push(Detection {
                group: window.group,
                secs,
                lang,
                prob,
            });
            if let Some(on_progress) = callbacks.on_progress.as_mut() {
                on_progress(((index + 1) * DETECT_PROGRESS / n_windows) as i32);
            }
        }
        resolve_languages(&detections)
            .into_iter()
            .map(|lang| Whisper::lang_str(lang).map(str::to_string))
            .collect()
    };

    let progress_start = if auto { DETECT_PROGRESS } else { 0 };
    let mut result = TranscribeResult::default();
    for (index, (window, language)) in windows.iter().zip(languages).enumerate() {
        if callbacks.should_abort.as_mut().is_some_and(|should_abort| should_abort()) {
            return Err(Error::Aborted);
        }
        params.language = language;
        let decoded = {
            let mut engine_callbacks = FullCallbacks::default();
            if let Some(should_abort) = callbacks.should_abort.as_mut() {
                engine_callbacks.should_abort = Some(Box::new(&mut **should_abort));
            }
            whisper.full_stream(
                &params,
                &samples[window.start_sample..window.end_sample],
                &mut engine_callbacks,
            )?
        };

        let t0cs = (window.start_sample / (SAMPLE_RATE / 100)) as i64;
        for segment in &decoded {
            let mut segment = convert(segment);
            segment.start += t0cs;
            segment.end += t0cs;
            if let Some(on_segment) = callbacks.on_segment.as_mut() {
                on_segment(segment.clone());
            }
            result.segments.push(segment);
        }

        if let Some(on_progress) = callbacks.on_progress.as_mut() {
            on_progress((progress_start + (index + 1) * (100 - progress_start) / n_windows) as i32);
        }
    }
    Ok(result)
}

/// A window shorter than this is too little audio to trust its own language guess.
const MIN_CONFIDENT_SECS: f32 = 1.5;
/// Below this probability a window's own language guess is not trusted either.
const MIN_CONFIDENT_PROB: f32 = 0.5;

struct Detection {
    group: usize,
    secs: f32,
    lang: i32,
    prob: f32,
}

/// Each window's language: its own guess when that is confident, otherwise the
/// language its group (speaker) speaks longest, then the recording's, then its
/// own guess after all. Short replies ("yeah", "ok") are where detection fails.
fn resolve_languages(detections: &[Detection]) -> Vec<i32> {
    let confident = |detection: &Detection| detection.secs >= MIN_CONFIDENT_SECS && detection.prob >= MIN_CONFIDENT_PROB;
    let mut by_group: BTreeMap<usize, BTreeMap<i32, f32>> = BTreeMap::new();
    let mut overall: BTreeMap<i32, f32> = BTreeMap::new();
    for detection in detections.iter().filter(|detection| confident(detection)) {
        *by_group
            .entry(detection.group)
            .or_default()
            .entry(detection.lang)
            .or_default() += detection.secs;
        *overall.entry(detection.lang).or_default() += detection.secs;
    }
    let longest = |totals: &BTreeMap<i32, f32>| totals.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(lang, _)| *lang);
    let family_pick = |lang: i32| {
        let family = CONFUSED_FAMILIES
            .iter()
            .find(|family| Whisper::lang_str(lang).is_some_and(|name| family.contains(&name)))?;
        let in_family: BTreeMap<i32, f32> = overall
            .iter()
            .filter(|(lang, _)| Whisper::lang_str(**lang).is_some_and(|name| family.contains(&name)))
            .map(|(lang, secs)| (*lang, *secs))
            .collect();
        longest(&in_family)
    };
    detections
        .iter()
        .map(|detection| {
            let lang = if confident(detection) {
                detection.lang
            } else {
                by_group
                    .get(&detection.group)
                    .and_then(longest)
                    .or_else(|| longest(&overall))
                    .unwrap_or(detection.lang)
            };
            family_pick(lang).unwrap_or(lang)
        })
        .collect()
}

/// Languages Whisper cannot tell apart turn by turn: it labels Ukrainian turns
/// Russian at 0.99 confidence, so neither length nor probability separates a
/// real switch from a misdetection. A recording settles on whichever member of
/// the family it speaks most and decodes every such turn in it.
/// ponytail: one family per recording, so a lone Russian speaker in a Ukrainian
/// call is decoded as Ukrainian; split per speaker if that ever matters.
const CONFUSED_FAMILIES: &[&[&str]] = &[&["uk", "ru", "be"]];

/// Maps the historical `TranscribeOptions` onto the engine's `FullParams`,
/// mirroring the previous `full_params` over `whisper_full_default_params`.
fn full_params(options: &TranscribeOptions) -> FullParams {
    let mut params = FullParams {
        strategy: if !options.sampling_greedy && options.beam_size > 0 {
            SamplingStrategy::BeamSearch
        } else {
            SamplingStrategy::Greedy
        },
        ..FullParams::default()
    };
    params.print_special = options.verbose;
    params.detect_language = options.detect_language;
    params.translate = options.translate;
    params.token_timestamps = options.word_timestamps;
    // Segments are only wrapped when max_len is set, so split_on_word is
    // inert on its own; it just makes the wrap land on word boundaries.
    params.split_on_word = options.word_timestamps;

    if options.threads > 0 {
        params.n_threads = options.threads;
    }
    if options.max_text_ctx > 0 {
        params.n_max_text_ctx = options.max_text_ctx;
    }
    if options.max_segment_len > 0 {
        params.max_len = options.max_segment_len;
    }
    if options.temperature > 0.0 {
        params.temperature = options.temperature;
    }
    // Older settings and API/CLI callers can exceed the decoder pool. Cap both
    // values: beam search also uses best_of during temperature fallback.
    if options.best_of > 0 {
        params.greedy_best_of = options.best_of.min(crate::state::MAX_DECODERS as i32);
    }
    if options.beam_size > 0 {
        params.beam_size = options.beam_size.min(crate::state::MAX_DECODERS as i32);
    }

    params.language = options.language.clone();
    params.languages = options.languages.clone();
    params.initial_prompt = options.prompt.clone();
    params.carry_initial_prompt = options.carry_prompt;
    params
}

/// Whether GGML's informational logging is forwarded to stderr.
/// Warnings and errors are always forwarded, whatever this is set to.
static VERBOSE: AtomicBool = AtomicBool::new(false);

pub fn set_verbose(verbose: bool) {
    VERBOSE.store(verbose, Ordering::Relaxed);
    install_abort_handler();
    unsafe { ffi::ggml_log_set(Some(ggml_log_callback), std::ptr::null_mut()) };
}

/// Registers the ggml abort hook so a `GGML_ASSERT` message reaches stderr
/// before the process dies. ggml calls `abort()` right after the hook
/// returns, so this only records the message.
pub(crate) fn install_abort_handler() {
    static INSTALLED: Once = Once::new();

    INSTALLED.call_once(|| {
        unsafe { ffi::ggml_set_abort_callback(Some(ggml_abort_callback)) };
    });
}

extern "C" fn ggml_log_callback(level: ffi::ggml_log_level, text: *const c_char, _user_data: *mut c_void) {
    if text.is_null() || !should_log(level) {
        return;
    }
    eprint!("{}", unsafe { CStr::from_ptr(text) }.to_string_lossy());
    let _ = std::io::stderr().flush();
}

/// Warnings and errors always get through; everything else needs `--verbose`.
/// A `CONT` line continues whichever line was logged before it, so it follows
/// the same decision.
fn should_log(level: ffi::ggml_log_level) -> bool {
    static LAST_LOGGED: AtomicBool = AtomicBool::new(false);

    let logged = match level {
        ffi::ggml_log_level_GGML_LOG_LEVEL_WARN | ffi::ggml_log_level_GGML_LOG_LEVEL_ERROR => true,
        ffi::ggml_log_level_GGML_LOG_LEVEL_CONT => LAST_LOGGED.load(Ordering::Relaxed),
        _ => VERBOSE.load(Ordering::Relaxed),
    };
    if level != ffi::ggml_log_level_GGML_LOG_LEVEL_CONT {
        LAST_LOGGED.store(logged, Ordering::Relaxed);
    }
    logged
}

/// GPU gating carried over from the previous implementation: Vulkan is only
/// attempted when the runtime library is present (Windows), and always
/// considered available elsewhere (macOS uses Metal).
#[cfg(windows)]
fn vulkan_available() -> bool {
    use windows_sys::Win32::Foundation::FreeLibrary;
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryA;

    let handle = unsafe { LoadLibraryA(c"vulkan-1.dll".as_ptr().cast()) };
    if handle.is_null() {
        return false;
    }
    unsafe { FreeLibrary(handle) };
    true
}

#[cfg(not(windows))]
fn vulkan_available() -> bool {
    true
}

extern "C" fn ggml_abort_callback(message: *const c_char) {
    let message = if message.is_null() {
        "(no message)".to_string()
    } else {
        unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
    };
    // ggml aborts as soon as this returns, so the message has to be flushed now.
    eprintln!("ggml fatal error: {}", message.trim_end());
    let _ = std::io::stderr().flush();
    tracing::error!(target: "whisper_rs", "ggml fatal error: {}", message.trim_end());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_options_are_capped_for_both_sampling_strategies() {
        for sampling_greedy in [true, false] {
            // Include a hidden oversized best_of with a valid visible beam_size.
            for (best_of, beam_size, expected_best_of, expected_beam_size) in [
                (10, 10, 8, 8),
                (9, 5, 8, 5),
                (5, 9, 5, 8),
                (8, 8, 8, 8),
                (1, 3, 1, 3),
                (i32::MAX, i32::MAX, 8, 8),
            ] {
                let params = full_params(&TranscribeOptions {
                    sampling_greedy,
                    best_of,
                    beam_size,
                    ..Default::default()
                });
                assert_eq!(params.greedy_best_of, expected_best_of);
                assert_eq!(params.beam_size, expected_beam_size);
                assert_eq!(
                    params.strategy,
                    if sampling_greedy {
                        SamplingStrategy::Greedy
                    } else {
                        SamplingStrategy::BeamSearch
                    }
                );
            }
        }
    }

    #[test]
    fn speech_windows_keep_speech_only_and_split_on_long_pauses() {
        let secs = |s: usize| s * SAMPLE_RATE;
        let speech = |start, end| vad_rs::SpeechSegment {
            start_sample: secs(start),
            end_sample: secs(end),
        };
        let window = |start, end, group| Window {
            start_sample: secs(start),
            end_sample: secs(end),
            group,
        };
        // Silence for 28 s, then speech with a short pause, a long pause, and a 40 s utterance.
        let heard = [speech(28, 30), speech(31, 35), speech(40, 60), speech(60, 100)];
        assert_eq!(
            speech_windows(&[], &heard, secs(100)),
            vec![window(28, 35, 0), window(40, 60, 0), window(60, 100, 0)]
        );
        // Speaker turns cut the speech; silence inside a turn is dropped.
        assert_eq!(
            speech_windows(&[window(0, 33, 1), window(33, 100, 2)], &heard, secs(100)),
            vec![window(28, 33, 1), window(33, 35, 2), window(40, 60, 2), window(60, 100, 2)]
        );
        assert!(speech_windows(&[], &[], secs(10)).is_empty());
    }

    #[test]
    fn unsure_windows_fall_back_to_their_speaker_then_the_recording() {
        let (en, uk, de) = (0, 1, 2);
        let detection = |group, secs, lang, prob| Detection { group, secs, lang, prob };
        let langs = resolve_languages(&[
            detection(0, 10.0, en, 0.9),
            detection(1, 8.0, uk, 0.95),
            detection(0, 6.0, uk, 0.9), // speaker 0 switches language: trusted
            detection(1, 0.8, de, 0.9), // too short: speaker 1's language
            detection(0, 4.0, de, 0.3), // unsure: speaker 0's longest language
            detection(2, 0.5, de, 0.9), // unknown speaker: the recording's longest (uk, 14s)
        ]);
        assert_eq!(langs, vec![en, uk, uk, uk, en, uk]);
        assert_eq!(resolve_languages(&[detection(0, 0.5, de, 0.2)]), vec![de]);
    }

    #[test]
    fn confusable_languages_follow_the_recording() {
        let id = |name| crate::lang::lang_id(name).unwrap() as i32;
        let (en, uk, ru) = (id("en"), id("uk"), id("ru"));
        let detection = |group, secs, lang, prob| Detection { group, secs, lang, prob };
        let langs = resolve_languages(&[
            detection(0, 70.0, uk, 0.9),
            detection(1, 5.9, ru, 0.99), // confident, but uk/ru is not trusted per turn
            detection(1, 20.0, en, 0.9), // English is told apart reliably: kept
            detection(2, 18.0, ru, 0.95),
        ]);
        assert_eq!(langs, vec![uk, uk, en, uk]);
        // A Russian recording stays Russian.
        assert_eq!(resolve_languages(&[detection(0, 30.0, ru, 0.9), detection(0, 3.0, uk, 0.8)]), vec![ru, ru]);
    }

    #[test]
    fn non_positive_decoder_options_keep_engine_defaults() {
        let defaults = FullParams::default();
        for value in [0, -1] {
            let params = full_params(&TranscribeOptions {
                best_of: value,
                beam_size: value,
                ..Default::default()
            });
            assert_eq!(params.greedy_best_of, defaults.greedy_best_of);
            assert_eq!(params.beam_size, defaults.beam_size);
        }
    }
}
