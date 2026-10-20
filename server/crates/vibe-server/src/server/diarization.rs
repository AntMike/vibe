#[cfg(feature = "diarize")]
pub use diarize_rs::Segment;

#[cfg(not(feature = "diarize"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker_id: usize,
}

#[cfg(feature = "diarize")]
pub fn diarize(model_path: &str, samples: &[f32]) -> Vec<Segment> {
    match diarize_rs::Diarizer::new(model_path).and_then(|mut diarizer| diarizer.diarize(samples, 16_000, 1)) {
        Ok(segments) => segments,
        Err(err) => {
            tracing::warn!("diarization failed, skipping speakers: {err}");
            Vec::new()
        }
    }
}

#[cfg(not(feature = "diarize"))]
pub fn diarize(model_path: &str, _samples: &[f32]) -> Vec<Segment> {
    tracing::warn!("diarization support is not enabled, skipping model: {model_path}");
    Vec::new()
}

/// Least room (seconds) either side of a speaker change to look for a quiet cut.
const MIN_CUT_SEARCH_SECS: f64 = 0.25;
/// Frame (samples, 20 ms at 16 kHz) whose energy picks the quietest cut.
const CUT_FRAME: usize = 320;
/// A speaker's turns this close (seconds) to the next one join it in one window.
const MERGE_GAP_SECS: f64 = 0.5;
/// A window overlapping another speaker's stays whole from this long (seconds): a short reply
/// over someone, rather than an "mm-hm" that stays inside the speaker it interrupts.
// ponytail: calibration knob; lower it if short replies still land on the other speaker.
const MIN_NESTED_SECS: f64 = 0.7;
/// Two speakers' windows overlapping by no more than this (seconds) are a handover, not
/// crosstalk: the diarizer pads every turn (0.23 s before, 0.08 s after), so back-to-back turns
/// overlap by about that much.
const HANDOVER_SECS: f64 = 0.35;

/// Speaker turns as per-speaker windows that cover the whole recording, to transcribe one at
/// a time. A speaker's turns less than `MERGE_GAP_SECS` apart merge into one window. Windows of
/// different speakers may overlap, each kept whole, unless the shorter is under
/// `MIN_NESTED_SECS`: its overlapping part is then left to the other. Audio no window covers
/// (the diarizer misses speech, seconds of it at times) is split at its quietest moment between
/// the windows either side, and the first and last windows reach the ends of the audio, since
/// audio outside every window would never be transcribed.
// ponytail: overlapping windows may transcribe the same words twice; dedup is the upgrade.
pub fn speaker_chunks(turns: &[Segment], samples: &[f32]) -> Vec<Segment> {
    let mut sorted = turns.to_vec();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut windows: Vec<Segment> = Vec::new();
    for turn in sorted {
        match windows.iter_mut().rfind(|window| window.speaker_id == turn.speaker_id) {
            Some(last) if turn.start - last.end < MERGE_GAP_SECS => last.end = last.end.max(turn.end),
            _ => windows.push(turn),
        }
    }

    // A handover is cut at its quietest moment, as a gap is, rather than kept as crosstalk.
    for i in 0..windows.len() {
        for j in i + 1..windows.len() {
            let (a, b) = (&windows[i], &windows[j]);
            if b.start >= a.end {
                break;
            }
            if a.speaker_id == b.speaker_id || b.end <= a.end || a.end - b.start > HANDOVER_SECS {
                continue;
            }
            let cut = quietest_point(samples, b.start, a.end).clamp(b.start, a.end);
            windows[i].end = cut;
            windows[j].start = cut;
        }
    }

    let whole = windows.clone();
    for (index, window) in windows.iter_mut().enumerate() {
        let secs = whole[index].end - whole[index].start;
        if secs >= MIN_NESTED_SECS {
            continue;
        }
        for other in &whole {
            // `other` is longer, so it cannot sit strictly inside `window`: the overlap
            // is a prefix, a suffix or all of it.
            let overlaps = other.start < window.end && other.end > window.start;
            if other.speaker_id == window.speaker_id || other.end - other.start <= secs || !overlaps {
                continue;
            }
            if other.start <= window.start {
                window.start = other.end;
            } else {
                window.end = other.start;
            }
        }
    }
    windows.retain(|window| window.end > window.start);
    windows.sort_by(|a, b| a.start.total_cmp(&b.start));

    let mut covered: Vec<(f64, f64)> = Vec::new();
    for window in &windows {
        match covered.last_mut() {
            Some(last) if window.start <= last.1 => last.1 = last.1.max(window.end),
            _ => covered.push((window.start, window.end)),
        }
    }
    for pair in covered.windows(2) {
        let (end, start) = (pair[0].1, pair[1].0);
        let floor = windows
            .iter()
            .filter(|window| window.end == end)
            .map(|window| window.start)
            .fold(0.0, f64::max);
        let ceiling = windows
            .iter()
            .filter(|window| window.start == start)
            .map(|window| window.end)
            .fold(f64::INFINITY, f64::min);
        let mid = (end + start) / 2.0;
        let lo = (mid - MIN_CUT_SEARCH_SECS).min(end).max(floor);
        let hi = (mid + MIN_CUT_SEARCH_SECS).max(start).min(ceiling);
        let cut = quietest_point(samples, lo, hi);
        for window in &mut windows {
            if window.end == end {
                window.end = cut;
            }
            if window.start == start {
                window.start = cut;
            }
        }
    }
    if let (Some(first), Some(last)) = (covered.first(), covered.last()) {
        for window in &mut windows {
            if window.start == first.0 {
                window.start = 0.0;
            }
            if window.end == last.1 {
                window.end = samples.len() as f64 / 16_000.0;
            }
        }
    }
    windows.retain(|window| window.end > window.start);
    windows.sort_by(|a, b| a.start.total_cmp(&b.start));
    windows
}

/// The centre (seconds) of the lowest-energy frame between `lo` and `hi`.
fn quietest_point(samples: &[f32], lo: f64, hi: f64) -> f64 {
    let first = (lo * 16_000.0) as usize;
    let last = ((hi * 16_000.0) as usize).min(samples.len());
    (first..last.saturating_sub(CUT_FRAME).max(first + 1))
        .step_by(CUT_FRAME)
        .filter(|&start| start + CUT_FRAME <= samples.len())
        .min_by(|&a, &b| energy(&samples[a..a + CUT_FRAME]).total_cmp(&energy(&samples[b..b + CUT_FRAME])))
        .map_or((lo + hi) / 2.0, |start| (start + CUT_FRAME / 2) as f64 / 16_000.0)
}

fn energy(frame: &[f32]) -> f32 {
    frame.iter().map(|sample| sample * sample).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: f64, end: f64, speaker_id: usize) -> Segment {
        Segment { start, end, speaker_id }
    }

    fn spans(chunks: &[Segment]) -> Vec<(f64, f64, usize)> {
        chunks
            .iter()
            .map(|chunk| (chunk.start, chunk.end, chunk.speaker_id))
            .collect()
    }

    /// Loud audio of `secs` seconds, silent between each `(start, end)`.
    fn audio(secs: f64, quiet: &[(f64, f64)]) -> Vec<f32> {
        (0..(secs * 16_000.0) as usize)
            .map(|index| {
                let t = index as f64 / 16_000.0;
                if quiet.iter().any(|&(start, end)| t >= start && t < end) {
                    0.0
                } else {
                    0.5
                }
            })
            .collect()
    }

    #[test]
    fn windows_cover_the_audio_and_cut_where_it_is_quiet() {
        let samples = audio(26.0, &[(16.0, 16.1)]);
        let chunks = speaker_chunks(
            &[
                turn(10.0, 12.0, 1), // overlaps the turn before it: both kept whole
                turn(0.5, 4.0, 0),
                turn(4.2, 10.5, 0),  // same speaker, close: same window
                turn(11.0, 11.5, 0), // nested in speaker 1's turn: left to it
                turn(20.0, 25.0, 0), // the speech the diarizer missed at 13-20 is still covered
            ],
            &samples,
        );
        assert_eq!(spans(&chunks), vec![(0.0, 10.5, 0), (10.0, 16.01, 1), (16.01, 26.0, 0)]);
        assert!(speaker_chunks(&[], &samples).is_empty());
    }

    #[test]
    fn a_reply_over_another_turn_keeps_both_whole() {
        let samples = audio(10.0, &[]);
        let chunks = speaker_chunks(&[turn(0.0, 10.0, 0), turn(4.0, 8.0, 1)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 10.0, 0), (4.0, 8.0, 1)]);
    }

    #[test]
    fn a_handover_is_cut_where_it_is_quiet_not_kept_as_crosstalk() {
        let samples = audio(9.0, &[(5.1, 5.15)]);
        let chunks = speaker_chunks(&[turn(0.0, 5.2, 0), turn(5.0, 9.0, 1)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 5.11, 0), (5.11, 9.0, 1)]);
    }

    #[test]
    fn a_blip_inside_another_turn_stays_in_it() {
        let samples = audio(10.0, &[]);
        let chunks = speaker_chunks(&[turn(0.0, 10.0, 0), turn(5.0, 5.5, 1)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 10.0, 0)]);
    }

    #[test]
    fn a_blip_on_a_turn_edge_is_trimmed_to_the_part_outside() {
        let samples = audio(10.0, &[]);
        let chunks = speaker_chunks(&[turn(0.0, 6.0, 0), turn(5.5, 6.1, 1)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 6.0, 0), (6.0, 10.0, 1)]);
    }

    #[test]
    fn audio_no_turn_covers_is_split_at_the_quietest_point() {
        let samples = audio(14.0, &[(6.4, 6.5)]);
        let chunks = speaker_chunks(&[turn(1.0, 5.0, 0), turn(8.0, 12.0, 1)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 6.41, 0), (6.41, 14.0, 1)]);
    }

    #[test]
    fn one_speakers_turns_merge_when_close() {
        let samples = audio(5.0, &[(2.5, 2.6)]);
        let chunks = speaker_chunks(&[turn(0.0, 2.0, 0), turn(2.3, 4.0, 0)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 5.0, 0)]);
        let chunks = speaker_chunks(&[turn(0.0, 2.0, 0), turn(3.0, 4.0, 0)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 2.51, 0), (2.51, 5.0, 0)]);
    }
}
