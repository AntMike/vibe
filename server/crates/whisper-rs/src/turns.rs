//! Per-turn decoding knobs for the windowed path, all off by default: merging a
//! speaker's back-to-back windows, prompting a window with text decoded before
//! it, and levelling each window's loudness.

use crate::Window;

/// Whisper's own limit on prompt tokens (half its 448-token text context).
const MAX_CONTEXT_TOKENS: usize = 224;

/// Windows by start, with each run of one group's windows in one language
/// (nobody else starting in between) joined into one spanning first start to
/// last end. `languages[i]` is window `i`'s detected language; the merged
/// windows come back with theirs.
pub(crate) fn merge_turns(windows: &[Window], languages: &[Option<String>]) -> (Vec<Window>, Vec<Option<String>>) {
    let mut sorted: Vec<(Window, Option<String>)> = windows.iter().cloned().zip(languages.iter().cloned()).collect();
    sorted.sort_by_key(|(window, _)| window.start_sample);
    let mut out: Vec<(Window, Option<String>)> = Vec::new();
    for (window, language) in sorted {
        match out.last_mut() {
            Some((last, last_language)) if last.group == window.group && *last_language == language => {
                last.end_sample = last.end_sample.max(window.end_sample)
            }
            _ => out.push((window, language)),
        }
    }
    out.into_iter().unzip()
}

/// Text one window decoded to, kept to prompt the windows after it.
pub(crate) struct Decoded {
    pub start_sample: usize,
    pub group: usize,
    pub language: Option<String>,
    pub text: String,
}

/// The prompt for `window`: `prompt` first, then the newest text from windows
/// that start before it and share its speaker or language, in time order, cut
/// from the old end to fit `max_tokens`. Windows decode out of time order, so
/// "before" is by start, not by when it was decoded.
pub(crate) fn context_prompt(
    decoded: &[Decoded],
    window: &Window,
    language: &Option<String>,
    prompt: Option<&str>,
    max_tokens: i32,
    count: impl Fn(&str) -> usize,
) -> Option<String> {
    let max_tokens = (max_tokens.max(0) as usize).min(MAX_CONTEXT_TOKENS);
    let mut budget = max_tokens.saturating_sub(prompt.map_or(0, &count));
    let mut earlier: Vec<&Decoded> = decoded
        .iter()
        .filter(|d| d.start_sample < window.start_sample && (d.group == window.group || &d.language == language))
        .collect();
    earlier.sort_by_key(|d| std::cmp::Reverse(d.start_sample));

    let mut picked: Vec<String> = Vec::new();
    for d in earlier {
        let text = d.text.trim();
        let n = count(text);
        if n <= budget {
            budget -= n;
            picked.push(text.to_string());
            continue;
        }
        // ponytail: trims whole words from the front, may leave a token or two unused.
        let words: Vec<&str> = text.split_whitespace().collect();
        if let Some(tail) = (1..words.len()).map(|from| words[from..].join(" ")).find(|tail| count(tail) <= budget) {
            picked.push(tail);
        }
        break;
    }
    picked.retain(|text| !text.is_empty());
    picked.reverse();
    let parts: Vec<&str> = prompt.into_iter().chain(picked.iter().map(String::as_str)).collect();
    (!parts.is_empty()).then(|| parts.join(" "))
}

/// A copy of `samples` scaled to `target_dbfs` RMS and clamped to [-1, 1];
/// silence is returned as it is, there is nothing to scale.
// ponytail: plain RMS, not K-weighted LUFS; upgrade to LUFS if the sweep shows loudness matters.
pub(crate) fn levelled(samples: &[f32], target_dbfs: f32) -> Vec<f32> {
    let rms = (samples.iter().map(|&s| s as f64 * s as f64).sum::<f64>() / samples.len().max(1) as f64).sqrt();
    if rms < 1e-6 {
        return samples.to_vec();
    }
    let gain = (10f64.powf(target_dbfs as f64 / 20.0) / rms) as f32;
    samples.iter().map(|&s| (s * gain).clamp(-1.0, 1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start: usize, end: usize, group: usize) -> Window {
        Window {
            start_sample: start,
            end_sample: end,
            group,
        }
    }

    fn decoded(start: usize, group: usize, language: &str, text: &str) -> Decoded {
        Decoded {
            start_sample: start,
            group,
            language: Some(language.into()),
            text: text.into(),
        }
    }

    fn words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    #[test]
    fn merge_joins_a_speakers_run_until_someone_else_starts() {
        let windows = [window(30, 40, 0), window(0, 10, 0), window(12, 20, 0), window(45, 50, 1), window(55, 60, 0)];
        let uk = Some("uk".to_string());
        let (merged, languages) = merge_turns(&windows, &vec![uk.clone(); 5]);
        assert_eq!(merged, vec![window(0, 40, 0), window(45, 50, 1), window(55, 60, 0)]);
        assert_eq!(languages, vec![uk.clone(), uk.clone(), uk]);
    }

    #[test]
    fn merge_keeps_a_speakers_windows_apart_when_the_language_differs() {
        let windows = [window(20, 30, 0), window(0, 10, 0), window(40, 50, 0)];
        let languages = [Some("ru".to_string()), Some("uk".to_string()), Some("uk".to_string())];
        let (merged, merged_languages) = merge_turns(&windows, &languages);
        assert_eq!(merged, vec![window(0, 10, 0), window(20, 30, 0), window(40, 50, 0)]);
        assert_eq!(merged_languages, vec![Some("uk".into()), Some("ru".into()), Some("uk".into())]);
    }

    #[test]
    fn prompt_takes_earlier_text_of_the_same_speaker_or_language() {
        let turns = [
            decoded(300, 0, "en", "later"),
            decoded(100, 1, "uk", "same speaker"),
            decoded(0, 2, "en", "same language"),
            decoded(50, 2, "de", "other"),
        ];
        let lang = Some("uk".to_string());
        let prompt = context_prompt(&turns, &window(200, 250, 1), &Some("en".into()), None, 50, words);
        assert_eq!(prompt.as_deref(), Some("same language same speaker"));
        let prompt = context_prompt(&turns, &window(200, 250, 1), &lang, Some("Names"), 50, words);
        assert_eq!(prompt.as_deref(), Some("Names same speaker"));
    }

    #[test]
    fn prompt_keeps_the_newest_text_within_the_token_cap() {
        let turns = [decoded(0, 0, "en", "one two three"), decoded(10, 0, "en", "four five")];
        let prompt = context_prompt(&turns, &window(20, 30, 0), &None, Some("p"), 4, words);
        assert_eq!(prompt.as_deref(), Some("p three four five"));
        let long = [decoded(0, 0, "en", &"w ".repeat(500))];
        let prompt = context_prompt(&long, &window(20, 30, 0), &None, None, 1000, words).unwrap();
        assert_eq!(words(&prompt), MAX_CONTEXT_TOKENS);
    }

    #[test]
    fn levelling_reaches_the_target_rms_and_leaves_silence() {
        let sine: Vec<f32> = (0..16000).map(|i| 0.01 * (i as f32 * 0.05).sin()).collect();
        let out = levelled(&sine, -20.0);
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!((20.0 * rms.log10() + 20.0).abs() < 0.5);
        let silence = vec![0.0f32; 100];
        assert_eq!(levelled(&silence, -20.0), silence);
    }
}
