//! Term boosting for greedy decoding: tokens that spell out a known term get a
//! bonus on their logit, so "Moonblu" comes out as the known "Мунблум". Matching
//! goes by characters, ignoring case, so any tokenization of a term counts and
//! an inflected ending simply follows it.

use std::collections::HashMap;

use crate::Tokenizer;

/// Logit bonus spread over a whole term, each token getting its share by the letters it spells.
// ponytail: fixed weight; tune against real drafts if terms show up where nobody said them.
const WEIGHT: f32 = 3.0;

#[derive(Debug, Default)]
pub struct Boost {
    blank_id: u32,
    /// Per term, per character position: each token that continues the term there, and where it leaves it.
    steps: Vec<Vec<Vec<(u32, usize)>>>,
}

impl Boost {
    pub fn new(tokenizer: &Tokenizer, terms: &[String]) -> Self {
        let pieces: Vec<Option<Vec<char>>> = (0..tokenizer.len() as u32)
            .map(|id| {
                tokenizer
                    .piece(id)
                    .filter(|piece| {
                        id != tokenizer.blank_id() && !piece.is_empty() && !(piece.starts_with('<') && piece.ends_with('>'))
                    })
                    .map(|piece| piece.replace('▁', " ").to_lowercase().chars().collect())
            })
            .collect();
        let steps = terms
            .iter()
            .map(|term| term.trim())
            .filter(|term| !term.is_empty())
            .map(|term| {
                // The leading space makes a term start only where a word does.
                let term: Vec<char> = format!(" {term}").to_lowercase().chars().collect();
                (0..term.len())
                    .map(|at| {
                        pieces
                            .iter()
                            .enumerate()
                            .filter_map(|(id, piece)| {
                                let piece = piece.as_ref()?;
                                let rest = &term[at..];
                                if rest.starts_with(piece) {
                                    Some((id as u32, at + piece.len()))
                                } else if at > 1 && piece.starts_with(rest) {
                                    // Finishes the term and runs into its ending: "Мунбл" + "уму".
                                    Some((id as u32, term.len()))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            blank_id: tokenizer.blank_id(),
            steps,
        }
    }

    /// Raises the tokens that start a term or continue one in `partial`; a token
    /// that fits several terms gets the largest bonus, not their sum.
    pub(crate) fn apply(&self, partial: &[(usize, usize)], logits: &mut [f32]) {
        // A term starts only where the model hears some token: there it may be the
        // word it misheard, but over blank it would be made up ("там" for Тамара).
        let hears = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .is_some_and(|(id, _)| id as u32 != self.blank_id);
        let starts = self.starts().filter(|_| hears);
        let mut bonus = HashMap::<u32, f32>::new();
        for (term, at) in starts.chain(partial.iter().copied()) {
            let steps = &self.steps[term];
            let letters = steps.len() - 1;
            for &(id, to) in &steps[at] {
                // A bare word-start "▁" spells nothing of the term.
                let spelled = (at..to).filter(|&index| index > 0).count();
                let value = WEIGHT * spelled as f32 / letters as f32;
                let entry = bonus.entry(id).or_default();
                *entry = entry.max(value);
            }
        }
        for (id, value) in bonus {
            if let Some(logit) = logits.get_mut(id as usize) {
                *logit += value;
            }
        }
    }

    /// Moves `partial` on to the terms still being spelled once `token` is
    /// emitted; true when `token` spells one to its end.
    pub(crate) fn advance(&self, partial: &mut Vec<(usize, usize)>, token: u32) -> bool {
        let mut next = Vec::new();
        let mut finished = false;
        for (term, at) in self.starts().chain(partial.iter().copied()) {
            let steps = &self.steps[term];
            if let Some(&(_, to)) = steps[at].iter().find(|(id, _)| *id == token) {
                if to == steps.len() {
                    finished = true;
                } else if !next.contains(&(term, to)) {
                    next.push((term, to));
                }
            }
        }
        *partial = next;
        finished
    }

    fn starts(&self) -> impl Iterator<Item = (usize, usize)> {
        (0..self.steps.len()).map(|term| (term, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boosts_the_tokens_that_spell_a_term() {
        let tokenizer = Tokenizer::from_pieces(&["▁", "▁Fig", "ma", "▁big", "fig", "<blank>", "▁f", "mas"], 5);
        let boost = Boost::new(&tokenizer, &["figma".into()]);
        let mut partial = Vec::new();

        // Where the model hears blank, no term starts.
        let mut logits = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        boost.apply(&partial, &mut logits);
        assert_eq!(logits, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // Where it hears "▁big", "▁Fig" and "▁f" are raised over it.
        let mut logits = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        boost.apply(&partial, &mut logits);
        assert_eq!(logits, [0.0, 1.8, 0.0, 1.0, 0.0, 0.0, 0.6, 0.0]);

        // Through "▁" then "fig" as well as through "▁Fig".
        assert!(!boost.advance(&mut partial, 0));
        assert!(!boost.advance(&mut partial, 4));
        assert_eq!(partial, [(0, 4)]);
        let mut logits = [0.0; 8];
        boost.apply(&partial, &mut logits);
        assert_eq!(logits[2], 1.2);
        assert_eq!(logits[7], 1.2, "an ending may run past the term");

        assert!(boost.advance(&mut partial, 2), "the term is complete");
        assert!(partial.is_empty());
        assert!(!boost.advance(&mut partial, 3));
        assert!(partial.is_empty(), "a word that isn't a term starts nothing");
    }
}
