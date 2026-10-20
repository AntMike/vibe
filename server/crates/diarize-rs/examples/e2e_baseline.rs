//! End-to-end validation: WAV in, probabilities and segments out, diffed
//! against the reference recorded in `baseline/` (ONNX for v2, HF transformers
//! for v3). Run from `server/`:
//!
//!     cargo run -p diarize-rs --release --example e2e_baseline -- \
//!         baseline/models/Nemotron-3-Diarization.f32.gguf standup_200_260 standup_60_120
//!
//! Each clip reads `baseline/wav/<clip>.wav` (the two v2 clips keep their
//! historical wav paths), `baseline/probs/<clip>.{f32,json}` and
//! `baseline/segments/<clip>.json`; a clip whose fixtures are missing is
//! skipped with a message. With no clips given it runs the v2 pair. The speaker count comes from the model
//! (the width of what `diarize_raw` returns), the frame length from the
//! baseline's `frame_ms` (80 ms when absent, as in the v2 recordings).
//!
//! This runs the whole path: audio -> mel (with the GGUF's trained filterbank)
//! -> chunking -> the ggml graph -> the AOSC speaker cache -> post-processing.
//! Every chunk past the first runs with a non-empty cache.
//!
//! Two caveats on what a difference here means:
//!
//! * The mel is not expected to be bit-identical to the baseline's: this path
//!   uses the filterbank shipped in the GGUF as `preprocessor.fb`.
//! * v2's `segments/` depends on the CallHome thresholds, so small probability
//!   differences near a threshold can move a boundary by a frame or two without
//!   meaning the graph disagrees. v3's were made by HF's `extract_speaker_dict`
//!   (plain `p > 0.5`), so for v3 ours are derived the same way from our probs.
//!
//! Exits non-zero when a clip's frame count is off by more than 8 frames, the
//! >0.5 decisions agree on fewer than 99.5% of cells, the set of speakers that
//! are ever active differs, a segment's speaker or the segment count differs,
//! or no clip had its fixtures.

use std::fs;

use diarize_rs::Diarizer;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The text right after `"key":` in a small json file.
fn json_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(&format!("\"{key}\""))?;
    let body = &text[at + key.len() + 2..];
    Some(body[body.find(':')? + 1..].trim_start())
}

fn read_shape(path: &str) -> Vec<usize> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let body = json_value(&text, "shape").expect("shape key");
    let close = body.find(']').expect("shape array end");
    body[1..close]
        .split(',')
        .map(|p| p.trim().parse::<usize>().expect("shape entry"))
        .collect()
}

/// `frame_ms` from a probs json, if recorded.
fn read_frame_ms(path: &str) -> Option<f64> {
    let text = fs::read_to_string(path).ok()?;
    let body = json_value(&text, "frame_ms")?;
    let end = body.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(body.len());
    body[..end].parse().ok()
}

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "{path}: expected 16 kHz");
    assert_eq!(spec.channels, 1, "{path}: expected mono");
    // The baseline dumpers loaded these as `i16 / 32768.0`; match it exactly.
    reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

/// Segments as recorded in `baseline/segments/<clip>.json`.
fn read_baseline_segments(path: &str) -> Vec<(f64, f64, usize)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Objects inside the `"segments"` array; keys are emitted alphabetically
    // (`end`, `speaker_id`, `start`), so scan by key rather than by position.
    let list = &text[text.find("\"segments\"").expect("segments key")..];
    let list = &list[..list.find(']').expect("segments array end")];
    let field = |obj: &str, key: &str| -> f64 {
        let body = json_value(obj, key).unwrap_or_else(|| panic!("{key} in {obj}"));
        let end = body
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
            .unwrap_or(body.len());
        body[..end].trim().parse().expect("number")
    };
    let mut out = Vec::new();
    let mut rest = list;
    while let Some(open) = rest.find('{') {
        let close = rest[open..].find('}').expect("object end") + open;
        let obj = &rest[open..close];
        out.push((field(obj, "start"), field(obj, "end"), field(obj, "speaker_id") as usize));
        rest = &rest[close + 1..];
    }
    out
}

struct Diff {
    max: f32,
    mean: f64,
    p50: f32,
    p95: f32,
    p99: f32,
    agree: f64,
    worst_row: usize,
}

fn compare(a: &[f32], b: &[f32], cols: usize) -> Diff {
    assert_eq!(a.len(), b.len());
    let mut diffs = Vec::with_capacity(a.len());
    let mut max = 0.0f32;
    let mut worst_row = 0;
    let mut sum = 0.0f64;
    let mut agree = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        diffs.push(d);
        sum += d as f64;
        if d > max {
            max = d;
            worst_row = i / cols;
        }
        if (x > 0.5) == (y > 0.5) {
            agree += 1;
        }
    }
    diffs.sort_by(|p, q| p.partial_cmp(q).unwrap());
    Diff {
        max,
        mean: sum / a.len() as f64,
        p50: diffs[diffs.len() / 2],
        p95: diffs[diffs.len() * 95 / 100],
        p99: diffs[diffs.len() * 99 / 100],
        agree: 100.0 * agree as f64 / a.len() as f64,
        worst_row,
    }
}

/// HF `extract_speaker_dict`: per-speaker runs of `p > 0.5`, sorted by
/// (start, speaker), times rounded to 10 ms.
fn threshold_segments(probs: &[f32], n_spk: usize, frame_s: f64) -> Vec<(f64, f64, usize)> {
    let frames = probs.len() / n_spk;
    let round = |f: usize| (f as f64 * frame_s * 100.0).round() / 100.0;
    let mut out = Vec::new();
    for spk in 0..n_spk {
        let mut open = None;
        for t in 0..=frames {
            let active = t < frames && probs[t * n_spk + spk] > 0.5;
            match (active, open) {
                (true, None) => open = Some(t),
                (false, Some(start)) => {
                    out.push((round(start), round(t), spk));
                    open = None;
                }
                _ => {}
            }
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.2.cmp(&b.2)));
    out
}

fn active_speakers(probs: &[f32], n_spk: usize) -> Vec<usize> {
    (0..n_spk)
        .filter(|&s| probs.chunks_exact(n_spk).any(|row| row[s] > 0.5))
        .collect()
}

/// Where a clip's audio lives: the v2 clips predate `baseline/wav/`.
fn wav_path(clip: &str) -> String {
    match clip {
        "sortformer-2spk-mix" => "plans/transcribe.cpp/samples/sortformer-2spk-mix.wav".to_string(),
        "6_speakers" => "testdata/6_speakers.wav".to_string(),
        _ => format!("baseline/wav/{clip}.wav"),
    }
}

/// `None` when the clip's fixtures are missing (skipped), else pass/fail.
fn run_clip(model: &str, clip: &str) -> Option<bool> {
    println!("\n================ {clip} ================");
    let wav = wav_path(clip);
    let fixtures = [
        wav.clone(),
        format!("baseline/probs/{clip}.f32"),
        format!("baseline/segments/{clip}.json"),
    ];
    if let Some(missing) = fixtures.iter().find(|path| !std::path::Path::new(path).exists()) {
        println!("SKIP: {missing} not found");
        return None;
    }
    let samples = read_wav(&wav);
    println!("audio: {} samples ({:.3} s)", samples.len(), samples.len() as f64 / 16_000.0);

    let mut diarizer = Diarizer::new(model).expect("load model");

    let started = std::time::Instant::now();
    let (probs, frames) = diarizer.diarize_raw(&samples, 16_000, 1).expect("diarize_raw");
    let n_spk = probs.len() / frames.max(1);
    println!(
        "diarize_raw: {frames} frames x {n_spk} speakers in {:.2} s",
        started.elapsed().as_secs_f64()
    );

    // ---- probabilities vs baseline ----
    let base_shape = read_shape(&format!("baseline/probs/{clip}.json"));
    let frame_s = read_frame_ms(&format!("baseline/probs/{clip}.json")).unwrap_or(80.0) / 1000.0;
    let base = read_f32(&format!("baseline/probs/{clip}.f32"));
    println!("baseline probs shape {base_shape:?}, ours [{frames}, {n_spk}]");
    let mut ok = true;
    let base_frames = base.len() / n_spk;
    if base_frames.abs_diff(frames) > 8 {
        println!("!! frame-count mismatch: baseline {base_frames} frames, ours {frames}");
        ok = false;
    } else if base_frames != frames {
        println!("frame count differs by {} (within tolerance)", base_frames.abs_diff(frames));
    }
    let n = base.len().min(probs.len());
    let d = compare(&probs[..n], &base[..n], n_spk);
    println!(
        "probs: max={:.6} (row {}) mean={:.6} p50={:.7} p95={:.7} p99={:.7} agree@0.5={:.3}%",
        d.max, d.worst_row, d.mean, d.p50, d.p95, d.p99, d.agree
    );
    let peak = |v: &[f32]| v.iter().fold(0.0f32, |a, &b| a.max(b));
    println!("max prob: ours {:.4}, baseline {:.4}", peak(&probs), peak(&base));

    // Per-10 s breakdown: where, if anywhere, does it start to drift?
    let block = (10.0 / frame_s).round() as usize;
    println!("per-10s max abs diff:");
    let mut row = 0;
    while row < n / n_spk {
        let hi = (row + block).min(n / n_spk);
        let c = compare(&probs[row * n_spk..hi * n_spk], &base[row * n_spk..hi * n_spk], n_spk);
        println!(
            "  rows {row:>5}..{hi:<5}  max={:.6} mean={:.6} agree@0.5={:.3}%",
            c.max, c.mean, c.agree
        );
        row = hi;
    }
    if d.agree < 99.5 {
        ok = false;
    }
    let (ours_active, base_active) = (active_speakers(&probs, n_spk), active_speakers(&base, n_spk));
    println!("speakers ever > 0.5: ours {ours_active:?}, baseline {base_active:?}");
    if ours_active != base_active {
        ok = false;
    }

    // ---- segments vs baseline ----
    let post = diarizer.diarize(&samples, 16_000, 1).expect("diarize");
    let mut speakers: Vec<usize> = post.iter().map(|s| s.speaker_id).collect();
    speakers.sort_unstable();
    speakers.dedup();
    println!("diarize (post-processed): {} segments, speakers {speakers:?}", post.len());
    let segments: Vec<(f64, f64, usize)> = if (frame_s - 0.08).abs() < 1e-9 {
        post.into_iter().map(|s| (s.start, s.end, s.speaker_id)).collect()
    } else {
        threshold_segments(&probs, n_spk, frame_s)
    };
    let baseline_segments = read_baseline_segments(&format!("baseline/segments/{clip}.json"));
    println!("\nsegments: ours {} / baseline {}", segments.len(), baseline_segments.len());
    let rows = segments.len().max(baseline_segments.len());
    let mut worst_edge = 0.0f64;
    for i in 0..rows {
        match (segments.get(i), baseline_segments.get(i)) {
            (Some(a), Some(b)) => {
                let (dstart, dend) = (a.0 - b.0, a.1 - b.1);
                worst_edge = worst_edge.max(dstart.abs()).max(dend.abs());
                let flag = if a.2 == b.2 { " " } else { "!" };
                if a.2 != b.2 {
                    ok = false;
                }
                println!(
                    "{flag} {i:>3}: spk{} {:7.3}-{:7.3}  |  spk{} {:7.3}-{:7.3}   (d {dstart:+.3}/{dend:+.3})",
                    a.2, a.0, a.1, b.2, b.0, b.1
                );
            }
            (Some(a), None) => {
                println!("+ {i:>3}: spk{} {:7.3}-{:7.3}  |  (none)", a.2, a.0, a.1);
                ok = false;
            }
            (None, Some(b)) => {
                println!("- {i:>3}: (none)                 |  spk{} {:7.3}-{:7.3}", b.2, b.0, b.1);
                ok = false;
            }
            (None, None) => {}
        }
    }
    println!("worst segment edge shift: {worst_edge:.3} s");

    Some(ok)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: e2e_baseline <model.gguf> [clip...]");
    let mut clips: Vec<String> = args.collect();
    if clips.is_empty() {
        clips = vec!["sortformer-2spk-mix".to_string(), "6_speakers".to_string()];
    }

    let (mut all_ok, mut ran) = (true, 0);
    for clip in &clips {
        if let Some(ok) = run_clip(&model, clip) {
            all_ok &= ok;
            ran += 1;
        }
    }
    if ran == 0 {
        println!("\nno clip had its fixtures");
        all_ok = false;
    }

    println!("\n{}", if all_ok { "OK" } else { "DIVERGENCE (see above)" });
    if !all_ok {
        std::process::exit(1);
    }
}
