//! Several Parakeet decodes of one window merged into one. Run 0 hears the
//! audio as is and keeps its lines and timing; the other runs hear it nudged
//! by a fraction of an encoder frame or played a little faster or slower, and
//! wherever a run heard a stretch differently, the surer variant is kept.

use std::borrow::Cow;

/// How a run's copy of the audio differs from the original.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Perturbation {
    /// Samples of silence put before the audio.
    shift: usize,
    /// Playback speed: the copy's sample `k` is the original's `k * speed`.
    speed: f64,
}

/// Silence before the audio, in samples: 40, 20, 60 and 10 ms, all inside one 80 ms frame.
const SHIFTS: [usize; 4] = [640, 320, 960, 160];
const SPEEDS: [f64; 4] = [1.04, 0.96, 1.06, 0.94];

/// A stretch one run heard and the other did not is kept only when its words
/// score above this, so a lone unsure word neither gets added nor survives.
// ponytail: fixed threshold; tune against real drafts if insertions or drops look off.
const EMPTY_SCORE: f32 = 0.5;

impl Perturbation {
    /// Run 0 is the audio as is; later runs alternate a shift and a speed change.
    pub fn for_run(run: usize) -> Self {
        match run {
            0 => Self { shift: 0, speed: 1.0 },
            run if run % 2 == 1 => Self {
                shift: SHIFTS[(run / 2) % SHIFTS.len()],
                speed: 1.0,
            },
            run => Self {
                shift: 0,
                speed: SPEEDS[(run / 2 - 1) % SPEEDS.len()],
            },
        }
    }

    pub fn apply<'a>(&self, samples: &'a [f32]) -> Cow<'a, [f32]> {
        if self.shift == 0 && self.speed == 1.0 {
            return Cow::Borrowed(samples);
        }
        let mut out = vec![0.0; self.shift];
        if !samples.is_empty() {
            // Linear interpolation between the two nearest original samples.
            let len = ((samples.len() - 1) as f64 / self.speed) as usize + 1;
            out.extend((0..len).map(|k| {
                let at = k as f64 * self.speed;
                let index = at as usize;
                let a = samples[index];
                let b = samples.get(index + 1).copied().unwrap_or(a);
                a + (b - a) * (at - index as f64) as f32
            }));
        }
        Cow::Owned(out)
    }

    /// Where a sample of the perturbed copy sits in the original audio.
    pub fn original(&self, position: usize) -> usize {
        (position.saturating_sub(self.shift) as f64 * self.speed).round() as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    /// Start on the original timeline, in centiseconds.
    pub time: i64,
    /// The least sure of its tokens.
    pub confidence: f32,
}

/// One of run 0's lines with its words.
pub struct Line {
    pub start: i64,
    pub end: i64,
    pub words: Vec<Word>,
}

/// Tokens grouped into words: a piece starting with '▁' starts a new word.
pub fn words(
    tokens: &[parakeet_rs::Token],
    tokenizer: &parakeet_rs::Tokenizer,
    time: impl Fn(&parakeet_rs::Token) -> i64,
) -> Vec<Word> {
    let mut groups: Vec<&[parakeet_rs::Token]> = Vec::new();
    let mut start = 0;
    for index in 1..=tokens.len() {
        let starts_word = tokens
            .get(index)
            .is_none_or(|token| tokenizer.piece(token.id).is_some_and(|piece| piece.starts_with('▁')));
        if starts_word {
            groups.push(&tokens[start..index]);
            start = index;
        }
    }
    groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .filter_map(|group| {
            let ids: Vec<_> = group.iter().map(|token| token.id).collect();
            let text = tokenizer.decode_clean(&ids);
            (!text.is_empty()).then(|| Word {
                text,
                time: time(&group[0]),
                confidence: group.iter().map(|token| token.prob).fold(1.0, f32::min),
            })
        })
        .collect()
}

/// Run 0's lines with the other runs merged in, one word list per line. Lines
/// without a real pause between them are aligned as one stretch, so a word a
/// run timed a frame off at a line boundary still lines up with its twin. Runs
/// are merged one at a time, so ties always keep what run 0 heard.
// ponytail: the alignment table is words × words per stretch; band it by time
// if a stretch without a half-second pause ever runs to thousands of words.
pub fn combine(lines: &[Line], runs: &[Vec<Word>]) -> Vec<Vec<Word>> {
    let mut combined = vec![Vec::new(); lines.len()];
    let mut first = 0;
    while first < lines.len() {
        let mut last = first;
        while last + 1 < lines.len() && lines[last + 1].start - lines[last].end < PAUSE {
            last += 1;
        }
        let from = if first == 0 {
            i64::MIN
        } else {
            (lines[first - 1].end + lines[first].start) / 2
        };
        let to = lines
            .get(last + 1)
            .map_or(i64::MAX, |next| (lines[last].end + next.start) / 2);
        let base = (first..=last)
            .flat_map(|index| lines[index].words.iter().map(move |word| (index, word.clone())))
            .collect();
        let merged = runs.iter().fold(base, |current, run| {
            let heard: Vec<_> = run.iter().filter(|word| word.time >= from && word.time < to).collect();
            merge(current, &heard, lines, first)
        });
        for (index, word) in merged {
            combined[index].push(word);
        }
        first = last + 1;
    }
    combined
}

/// Centiseconds of silence between two lines that makes them separate stretches.
const PAUSE: i64 = 50;

fn key(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn score<'a>(words: impl Iterator<Item = &'a Word>) -> f32 {
    let (sum, count) = words.fold((0.0, 0), |(sum, count), word| (sum + word.confidence, count + 1));
    if count == 0 {
        EMPTY_SCORE
    } else {
        sum / count as f32
    }
}

/// The longest common subsequence of the two word lists keeps the words both
/// heard; each stretch between those is a disagreement won by the higher score.
/// Words carry the index of the line they belong to.
fn merge(current: Vec<(usize, Word)>, other: &[&Word], lines: &[Line], first_line: usize) -> Vec<(usize, Word)> {
    let a: Vec<_> = current.iter().map(|(_, word)| key(&word.text)).collect();
    let b: Vec<_> = other.iter().map(|word| key(&word.text)).collect();
    let mut length = vec![vec![0u32; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            length[i][j] = if a[i] == b[j] {
                length[i + 1][j + 1] + 1
            } else {
                length[i + 1][j].max(length[i][j + 1])
            };
        }
    }
    let mut merged = Vec::with_capacity(current.len());
    let (mut i, mut j) = (0, 0);
    let (mut block_i, mut block_j) = (0, 0);
    loop {
        let matched = i < a.len() && j < b.len() && a[i] == b[j];
        if matched || i == a.len() || j == b.len() {
            if !matched {
                (i, j) = (a.len(), b.len());
            }
            let theirs = &other[block_j..j];
            if score(theirs.iter().copied()) > score(current[block_i..i].iter().map(|(_, word)| word)) {
                // A replacement joins the line of the first word it replaces;
                // an addition between two lines joins the one it is timed in.
                let line = match (
                    block_i.checked_sub(1).map(|k| current[k].0),
                    current.get(block_i).map(|(line, _)| *line),
                ) {
                    (_, Some(next)) if block_i < i => next,
                    (Some(previous), Some(next)) if previous != next && theirs[0].time < lines[next].start => previous,
                    (_, Some(next)) => next,
                    (Some(previous), None) => previous,
                    (None, None) => first_line,
                };
                merged.extend(theirs.iter().map(|&word| (line, word.clone())));
            } else {
                merged.extend_from_slice(&current[block_i..i]);
            }
            if !matched {
                return merged;
            }
            merged.push(current[i].clone());
            (i, j) = (i + 1, j + 1);
            (block_i, block_j) = (i, j);
        } else if length[i + 1][j] >= length[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str, start: i64, confidence: f32) -> Vec<Word> {
        text.split(' ')
            .enumerate()
            .map(|(index, text)| Word {
                text: text.to_string(),
                time: start + index as i64 * 10,
                confidence,
            })
            .collect()
    }

    fn text(words: &[Word]) -> String {
        words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn surer_variant_wins_each_disagreement() {
        let mut first = words("the cat sat on a mat.", 0, 0.9);
        first[1].confidence = 0.4; // "cat" was a guess
        let lines = [
            Line {
                start: 0,
                end: 60,
                words: first.clone(),
            },
            Line {
                start: 60,
                end: 80,
                words: words("It purred.", 60, 0.9),
            },
        ];
        let mut run = words("the hat sat on the mat. It purred.", 0, 0.6);
        run[4].confidence = 0.3; // "the" less sure than run 0's "a"
        run[6].time = 55; // "It" timed a frame early, before the line break

        let combined = combine(&lines, &[run]);
        assert_eq!(text(&combined[0]), "the hat sat on a mat.");
        assert_eq!(text(&combined[1]), "It purred.");
        // A single run (draft_runs = 1) is left exactly as it was.
        assert_eq!(combine(&lines, &[])[0], first);
        assert_eq!(Perturbation::for_run(0).original(12_345), 12_345);
    }
}
