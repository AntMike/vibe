//! Term boosting in the decoder, ported from parakeet-rs: tokens that spell out
//! a known term get a bonus on their logit. Whisper's vocab is byte-level BPE, so
//! matching goes by bytes, ignoring case where a token is whole UTF-8. Beam
//! search clones sequences freely, so nothing is tracked per decoder: the word
//! being spelled is read back off the sequence at every step.

use std::collections::HashMap;

use crate::vocab::{Token, Vocab};
use crate::TokenData;

/// Logit bonus spread over a whole term, each token getting its share by the bytes it spells.
// ponytail: fixed weight; Parakeet's 3.0 made Whisper worse on the Oct 15 pieces (scripts/asr-eval).
const WEIGHT: f32 = 1.5;

/// Per byte position of a term: each token that continues the term there, and where it leaves it.
type Steps = Vec<Vec<(Token, usize)>>;

pub struct Boost {
    /// Per term: its lowercased bytes with a leading space, and its steps.
    terms: Vec<(Vec<u8>, Steps)>,
}

impl Boost {
    pub fn new(vocab: &Vocab, terms: &[String]) -> Self {
        let pieces: Vec<Vec<u8>> = (0..vocab.token_eot).map(|id| lower(vocab.token_bytes(id))).collect();
        let terms = terms
            .iter()
            .map(|term| term.trim())
            .filter(|term| !term.is_empty())
            .map(|term| {
                // The leading space makes a term start only where a word does.
                let term = format!(" {term}").to_lowercase().into_bytes();
                let steps = (0..term.len())
                    .map(|at| {
                        let rest = &term[at..];
                        pieces
                            .iter()
                            .enumerate()
                            .filter(|(_, piece)| !piece.is_empty())
                            .filter_map(|(id, piece)| {
                                if rest.starts_with(piece) {
                                    Some((id as Token, at + piece.len()))
                                } else if at > 1 && piece.starts_with(rest) {
                                    // Finishes the term and runs into its ending: "Мунбл" + "уму".
                                    Some((id as Token, term.len()))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .collect();
                (term, steps)
            })
            .collect();
        Self { terms }
    }

    /// Raises the tokens that start a term, or continue the one the word at the
    /// end of `tokens` is spelling; a token that fits several terms gets the
    /// largest bonus, not their sum.
    pub(crate) fn apply(&self, tokens: &[TokenData], vocab: &Vocab, logits: &mut [f32]) {
        // Where the model would rather stamp time or end, it hears silence or
        // is done: a term there would be made up.
        let hears = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .is_some_and(|(id, _)| (id as Token) < vocab.token_eot);
        if !hears {
            return;
        }
        // The word being spelled: text tokens back to the one that starts it.
        let mut tail = Vec::new();
        for token in tokens.iter().rev().take_while(|token| token.id < vocab.token_eot) {
            let piece = lower(vocab.token_bytes(token.id));
            tail.splice(0..0, piece.iter().copied());
            if piece.starts_with(b" ") {
                break;
            }
        }
        let mut bonus = HashMap::<Token, f32>::new();
        for (term, steps) in &self.terms {
            let partial = (!tail.is_empty() && tail.len() < term.len() && term.starts_with(&tail)).then_some(tail.len());
            for at in std::iter::once(0).chain(partial) {
                for &(id, to) in &steps[at] {
                    // A bare word-start " " spells nothing of the term.
                    let value = WEIGHT * (to - at.max(1)) as f32 / (term.len() - 1) as f32;
                    let entry = bonus.entry(id).or_default();
                    *entry = entry.max(value);
                }
            }
        }
        for (id, value) in bonus {
            if let Some(logit) = logits.get_mut(id as usize) {
                *logit += value;
            }
        }
    }
}

/// A token's bytes lowercased, or as they are when they split a character.
// ponytail: an uppercase letter split across tokens stays uppercase and misses the term.
fn lower(bytes: &[u8]) -> Vec<u8> {
    std::str::from_utf8(bytes).map_or_else(|_| bytes.to_vec(), |text| text.to_lowercase().into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boosts_the_tokens_that_spell_a_term() {
        let words = [" ", " Fig", "ma", " big", "fig", " f", "mas", "[_EOT_]", "[_BEG_]"];
        let (eot, beg) = (7, 8);
        let vocab = Vocab {
            n_vocab: words.len() as i32,
            token_to_id: HashMap::new(),
            id_to_token: words.iter().map(|word| word.as_bytes().to_vec()).collect(),
            token_eot: eot,
            token_sot: eot,
            token_translate: eot,
            token_transcribe: eot,
            token_solm: eot,
            token_prev: eot,
            token_nosp: eot,
            token_not: eot,
            token_beg: beg,
        };
        let boost = Boost::new(&vocab, &["figma".into()]);
        let sequence = |ids: &[Token]| -> Vec<TokenData> {
            ids.iter()
                .map(|&id| TokenData {
                    id,
                    ..TokenData::empty()
                })
                .collect()
        };

        // Where the model would stamp time, no term starts.
        let mut logits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        boost.apply(&[], &vocab, &mut logits);
        assert_eq!(logits, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        // Where it hears " big", " Fig" and " f" are raised over it.
        let mut logits = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        boost.apply(&sequence(&[beg]), &vocab, &mut logits);
        assert_eq!(logits, [0.0, 1.8, 0.0, 1.0, 0.0, 0.6, 0.0, 0.0, 0.0]);

        // After " " then "fig" (the " big" before it is another word), the term
        // goes on, and a new one may still start.
        let mut logits = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        boost.apply(&sequence(&[3, 0, 4]), &vocab, &mut logits);
        assert_eq!(logits[2], 1.2);
        assert_eq!(logits[6], 1.2, "an ending may run past the term");
        assert_eq!(logits[1], 1.8);

        // A finished term is not continued.
        let mut logits = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        boost.apply(&sequence(&[1, 2]), &vocab, &mut logits);
        assert_eq!(logits, [0.0, 1.8, 0.0, 1.0, 0.0, 0.6, 0.0, 0.0, 0.0]);
    }
}
