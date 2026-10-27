use crate::Segment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextOptions {
    pub gpu_device: i32,
    pub no_gpu: bool,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            gpu_device: -1,
            no_gpu: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub detect_language: bool,
    pub translate: bool,
    pub threads: i32,
    pub prompt: Option<String>,
    /// Repeat `prompt` before every 30-second window instead of only the first,
    /// so a glossary keeps steering the spelling through the whole file.
    pub carry_prompt: bool,
    pub verbose: bool,
    pub temperature: f32,
    pub max_text_ctx: i32,
    pub word_timestamps: bool,
    pub max_segment_len: i32,
    pub sampling_greedy: bool,
    pub best_of: i32,
    pub beam_size: i32,
    pub stable_timestamps: bool,
    pub vad_model_path: Option<String>,
    /// Candidates for language auto-detection; empty means every language.
    pub languages: Vec<String>,
    /// When set, each window is transcribed on its own, with its own language
    /// when the language is auto. Audio outside every window is skipped.
    pub windows: Vec<Window>,
    /// Parakeet only: how many times each window is decoded (1..=8). Runs after
    /// the first hear a slightly shifted or sped-up copy, and every stretch they
    /// hear differently keeps the surest variant.
    pub draft_runs: i32,
    /// Known terms the decoder leans toward spelling out (Parakeet and Whisper).
    pub hotwords: Vec<String>,
    /// With `detect_language`: each window's own detection as heard, None when
    /// it is too short or unsure, instead of resolved against its speaker and
    /// the recording. Shows who the detector hears in which language.
    pub raw_languages: bool,
    /// Decoder fallback thresholds; None keeps whisper's own (entropy 2.4,
    /// log-prob -1.0, no-speech 0.6). Server callers only, for tuning.
    pub entropy_thold: Option<f32>,
    pub logprob_thold: Option<f32>,
    pub no_speech_thold: Option<f32>,
    /// Decode a speaker's back-to-back windows as one, so a turn cut by pauses
    /// keeps its own context. Eval knob, off by default.
    pub group_turns: bool,
    /// Prompt each window with up to this many tokens (capped at 224) of text
    /// already decoded from the same speaker or language; 0 is off.
    pub context_tokens: i32,
    /// Bring each window to this RMS in dBFS before decoding, so quiet and loud
    /// speakers reach the model alike; None leaves the audio as it is.
    pub turn_loudness: Option<f32>,
}

/// A slice of the input transcribed on its own, such as one speaker turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start_sample: usize,
    pub end_sample: usize,
    /// Windows of one group (one speaker) share a fallback language for when
    /// a window is too short or unclear to detect its own.
    pub group: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            detect_language: false,
            translate: false,
            threads: 0,
            prompt: None,
            carry_prompt: false,
            verbose: false,
            temperature: 0.0,
            max_text_ctx: 0,
            word_timestamps: false,
            max_segment_len: 0,
            sampling_greedy: true,
            best_of: 0,
            beam_size: 0,
            stable_timestamps: false,
            vad_model_path: None,
            languages: Vec::new(),
            windows: Vec::new(),
            draft_runs: 1,
            hotwords: Vec::new(),
            raw_languages: false,
            entropy_thold: None,
            logprob_thold: None,
            no_speech_thold: None,
            group_turns: false,
            context_tokens: 0,
            turn_loudness: None,
        }
    }
}

#[derive(Default)]
pub struct StreamCallbacks<'a> {
    pub on_progress: Option<Box<dyn FnMut(i32) + Send + 'a>>,
    pub on_segment: Option<Box<dyn FnMut(Segment) + Send + 'a>>,
    pub should_abort: Option<Box<dyn FnMut() -> bool + Send + 'a>>,
}
