//! Who was talking, from the loudness of the two capture tracks: your mic and the call audio.
//! Speech on only the mic is you; the call app's active-speaker ring lags and lingers, so its
//! turns for you are replaced by the mic's, and cut out of everyone else's.

use serde::Serialize;

/// A stretch of the recording during which `name` was talking. Seconds from the start of the recording.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub name: String,
}

const FRAME_SECS: f64 = 0.1;
/// Quietest level that still counts as speech (about -50 dBFS), whatever the noise floor.
const MIN_SPEECH: f32 = 0.003;
/// Speech sits at least this many times (15 dB) above the track's noise floor.
// ponytail: calibration knobs; tune here if typing or a loud room reads as talking.
const ABOVE_FLOOR: f32 = 5.6;
/// Pauses up to this many frames stay inside one stretch of talk.
const MAX_PAUSE_FRAMES: usize = 3;
/// Stretches shorter than this many frames are clicks, not talk.
const MIN_TALK_FRAMES: usize = 3;

/// RMS loudness of one track in 100 ms frames, fed from its audio callback.
pub struct Loudness {
    frame_len: usize,
    sum: f64,
    count: usize,
    frames: Vec<f32>,
}

impl Loudness {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            frame_len: ((sample_rate as f64 * FRAME_SECS) as usize * channels as usize).max(1),
            sum: 0.0,
            count: 0,
            frames: Vec::new(),
        }
    }

    pub fn push(&mut self, samples: impl IntoIterator<Item = f32>) {
        for sample in samples {
            self.sum += (sample * sample) as f64;
            self.count += 1;
            if self.count == self.frame_len {
                self.frames.push((self.sum / self.count as f64).sqrt() as f32);
                self.sum = 0.0;
                self.count = 0;
            }
        }
    }

    pub fn frames(&self) -> &[f32] {
        &self.frames
    }
}

/// Which frames of a track hold speech: well above its own noise floor.
fn speaking(frames: &[f32]) -> Vec<bool> {
    let mut sorted = frames.to_vec();
    sorted.sort_by(f32::total_cmp);
    let floor = sorted.get(sorted.len() / 10).copied().unwrap_or(0.0);
    let threshold = (floor * ABOVE_FLOOR).max(MIN_SPEECH);
    frames.iter().map(|&rms| rms > threshold).collect()
}

/// Runs of `true` as (start, end) seconds, bridging short pauses and dropping clicks.
fn stretches(flags: &[bool]) -> Vec<(f64, f64)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (index, _) in flags.iter().enumerate().filter(|(_, on)| **on) {
        match runs.last_mut() {
            Some(run) if index - run.1 <= MAX_PAUSE_FRAMES => run.1 = index + 1,
            _ => runs.push((index, index + 1)),
        }
    }
    runs.into_iter()
        .filter(|(start, end)| end - start >= MIN_TALK_FRAMES)
        .map(|(start, end)| (start as f64 * FRAME_SECS, end as f64 * FRAME_SECS))
        .collect()
}

fn overlap(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.1.min(b.1) - a.0.max(b.0)).max(0.0)
}

/// `turn` minus every span in `cuts` (sorted), as the pieces left over.
fn cut_out(turn: &SpeakerTurn, cuts: &[(f64, f64)]) -> Vec<SpeakerTurn> {
    let mut pieces = Vec::new();
    let mut start = turn.start;
    for &(cut_start, cut_end) in cuts.iter().filter(|&&cut| overlap(cut, (turn.start, turn.end)) > 0.0) {
        if cut_start > start {
            pieces.push(SpeakerTurn {
                start,
                end: cut_start,
                name: turn.name.clone(),
            });
        }
        start = start.max(cut_end);
    }
    if turn.end > start {
        pieces.push(SpeakerTurn {
            start,
            end: turn.end,
            name: turn.name.clone(),
        });
    }
    pieces
}

/// The call app's turns, corrected with what the tracks heard. You are whoever the call showed
/// talking most while only the mic had speech; your turns become exactly those stretches, and
/// they are cut out of everyone else's. Unchanged when there are no turns to name you from.
pub fn refine_turns(turns: Vec<SpeakerTurn>, mic: &[f32], call: &[f32]) -> Vec<SpeakerTurn> {
    let (mic, call) = (speaking(mic), speaking(call));
    let only_mic: Vec<bool> = (0..mic.len())
        .map(|i| mic[i] && !call.get(i).copied().unwrap_or(false))
        .collect();
    let mine = stretches(&only_mic);

    let mut by_name: Vec<(&str, f64)> = Vec::new();
    for turn in &turns {
        let seconds: f64 = mine.iter().map(|&span| overlap(span, (turn.start, turn.end))).sum();
        match by_name.iter_mut().find(|(name, _)| *name == turn.name) {
            Some(entry) => entry.1 += seconds,
            None => by_name.push((&turn.name, seconds)),
        }
    }
    let total: f64 = by_name.iter().map(|(_, seconds)| seconds).sum();
    let Some((me, seconds)) = by_name.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)) else {
        return turns;
    };
    if seconds <= total / 2.0 {
        return turns;
    }
    let me = me.to_string();

    let mut refined: Vec<SpeakerTurn> = turns
        .iter()
        .filter(|turn| turn.name != me)
        .flat_map(|turn| cut_out(turn, &mine))
        .chain(mine.iter().map(|&(start, end)| SpeakerTurn {
            start,
            end,
            name: me.clone(),
        }))
        .collect();
    refined.sort_by(|a, b| a.start.total_cmp(&b.start));
    refined
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: f64, end: f64, name: &str) -> SpeakerTurn {
        SpeakerTurn {
            start,
            end,
            name: name.into(),
        }
    }

    /// 100 ms frames: loud (0.1) inside each (start, end) second span, a quiet floor (0.0005) elsewhere.
    fn track(secs: f64, loud: &[(f64, f64)]) -> Vec<f32> {
        (0..(secs / FRAME_SECS).round() as usize)
            .map(|i| {
                let t = (i as f64 + 0.5) * FRAME_SECS;
                if loud.iter().any(|&(s, e)| t >= s && t < e) {
                    0.1
                } else {
                    0.0005
                }
            })
            .collect()
    }

    #[test]
    fn loudness_averages_whole_frames() {
        let mut loudness = Loudness::new(10, 2); // 2 samples per frame
        loudness.push([0.5, -0.5, 1.0]);
        assert_eq!(loudness.frames(), &[0.5]);
    }

    #[test]
    fn mic_only_speech_becomes_your_turns_and_leaves_the_others() {
        // You talk 0-5 s and 12-13 s; Serhii talks 5-12 s on the call. Slack's ring lags a second
        // and lingers: it shows you until 6 and Serhii until 13.
        let mic = track(15.0, &[(0.0, 5.0), (12.0, 13.0)]);
        let call = track(15.0, &[(5.0, 12.0)]);
        let turns = vec![turn(0.5, 6.0, "Mike"), turn(6.0, 13.0, "Serhii")];
        let refined = refine_turns(turns, &mic, &call);
        let rounded: Vec<(f64, f64, &str)> = refined
            .iter()
            .map(|t| {
                (
                    (t.start * 10.0).round() / 10.0,
                    (t.end * 10.0).round() / 10.0,
                    t.name.as_str(),
                )
            })
            .collect();
        assert_eq!(rounded, vec![(0.0, 5.0, "Mike"), (6.0, 12.0, "Serhii"), (12.0, 13.0, "Mike")]);
    }

    #[test]
    fn unchanged_without_turns_or_without_mic_speech() {
        let mic = track(5.0, &[(0.0, 5.0)]);
        let quiet = track(5.0, &[]);
        assert!(refine_turns(vec![], &mic, &quiet).is_empty());
        let turns = vec![turn(0.0, 5.0, "Serhii")];
        assert_eq!(refine_turns(turns.clone(), &quiet, &mic), turns);
    }
}
