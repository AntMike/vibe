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
/// A turn inside another speaker's gets its own chunk from this long (seconds): a short reply
/// over someone, rather than an "mm-hm" that would only cut their sentence in two.
// ponytail: calibration knob; lower it if short replies still land on the other speaker.
const MIN_NESTED_SECS: f64 = 0.7;

/// Speaker turns as chunks that tile the whole recording, to transcribe one at
/// a time. A speaker's consecutive turns merge. A turn nested in another
/// speaker's splits it in three when long enough, else is left to that one. Each speaker change is cut at the quietest
/// moment between the two turns, and the first and last chunks reach the ends
/// of the audio: the diarizer misses speech (seconds of it, at times), and
/// audio outside every chunk would never be transcribed.
pub fn speaker_chunks(turns: &[Segment], samples: &[f32]) -> Vec<Segment> {
    let mut sorted = turns.to_vec();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut chunks: Vec<Segment> = Vec::new();
    for turn in sorted {
        if let Some(last) = chunks.last_mut() {
            if turn.end <= last.end {
                let long_enough = |secs: f64| secs >= MIN_NESTED_SECS;
                if turn.speaker_id != last.speaker_id
                    && long_enough(turn.end - turn.start)
                    && long_enough(turn.start - last.start)
                {
                    let rest = Segment {
                        start: turn.end,
                        ..last.clone()
                    };
                    last.end = turn.start;
                    chunks.push(turn);
                    if long_enough(rest.end - rest.start) {
                        chunks.push(rest);
                    }
                }
                continue;
            }
            if turn.speaker_id == last.speaker_id {
                last.end = turn.end;
                continue;
            }
        }
        chunks.push(turn);
    }
    for index in 1..chunks.len() {
        let (end, start) = (chunks[index - 1].end, chunks[index].start);
        let mid = (end + start) / 2.0;
        let lo = end.min(start).min(mid - MIN_CUT_SEARCH_SECS).max(chunks[index - 1].start);
        let hi = end.max(start).max(mid + MIN_CUT_SEARCH_SECS).min(chunks[index].end);
        let cut = quietest_point(samples, lo, hi);
        chunks[index - 1].end = cut;
        chunks[index].start = cut;
    }
    if let Some(first) = chunks.first_mut() {
        first.start = 0.0;
    }
    if let Some(last) = chunks.last_mut() {
        last.end = samples.len() as f64 / 16_000.0;
    }
    chunks.retain(|chunk| chunk.end > chunk.start);
    chunks
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
    fn chunks_tile_the_audio_and_cut_where_it_is_quiet() {
        let samples = audio(26.0, &[(10.38, 10.5), (16.0, 16.1)]);
        let chunks = speaker_chunks(
            &[
                turn(10.0, 12.0, 1), // overlaps the turn before it
                turn(0.5, 4.0, 0),
                turn(5.0, 10.5, 0),  // same speaker: same chunk, gap or not
                turn(11.0, 11.5, 0), // nested in speaker 1's turn: left to it
                turn(20.0, 25.0, 0), // the speech the diarizer missed at 13-20 is still covered
            ],
            &samples,
        );
        assert_eq!(spans(&chunks), vec![(0.0, 10.39, 0), (10.39, 16.01, 1), (16.01, 26.0, 0)]);
        assert!(speaker_chunks(&[], &samples).is_empty());
    }

    #[test]
    fn a_reply_inside_another_turn_gets_its_own_chunk() {
        let samples = audio(20.0, &[(4.9, 5.0), (6.0, 6.1)]);
        let chunks = speaker_chunks(&[turn(0.0, 20.0, 0), turn(5.0, 6.0, 1)], &samples);
        assert_eq!(spans(&chunks), vec![(0.0, 4.92, 0), (4.92, 6.02, 1), (6.02, 20.0, 0)]);
    }
}
