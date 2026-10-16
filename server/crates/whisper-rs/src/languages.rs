//! Which language each speech window is decoded in: its own detection,
//! resolved against its speaker and the whole recording.

use std::collections::BTreeMap;

use crate::{Error, FullParams, Result, StreamCallbacks, Whisper, Window, SAMPLE_RATE};

/// Share of the progress bar spent detecting window languages before decoding.
pub(crate) const DETECT_PROGRESS: usize = 20;

pub(crate) fn auto_language(params: &FullParams) -> bool {
    params
        .language
        .as_deref()
        .is_none_or(|lang| lang.is_empty() || lang == "auto")
}

/// The language each window is decoded in: the one asked for, English on an
/// English-only model, otherwise each window's own detection, resolved against
/// its speaker and the recording. Detection fills the first `DETECT_PROGRESS`.
pub(crate) fn window_languages(
    whisper: &mut Whisper,
    samples: &[f32],
    windows: &[Window],
    params: &FullParams,
    callbacks: &mut StreamCallbacks<'_>,
) -> Result<Vec<Option<String>>> {
    if !auto_language(params) {
        return Ok(vec![params.language.clone(); windows.len()]);
    }
    if !whisper.is_multilingual() {
        return Ok(vec![Some("en".to_string()); windows.len()]);
    }
    let n_windows = windows.len().max(1);
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
    Ok(resolve_languages(&detections)
        .into_iter()
        .map(|lang| Whisper::lang_str(lang).map(str::to_string))
        .collect())
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            resolve_languages(&[detection(0, 30.0, ru, 0.9), detection(0, 3.0, uk, 0.8)]),
            vec![ru, ru]
        );
    }
}
