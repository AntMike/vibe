//! Time `Diarizer::diarize` on one 16 kHz WAV and summarise what it found.
//!
//!     cargo run -p diarize-rs --release --example diarize_file -- <model.gguf> <audio.wav>

fn main() {
    let mut args = std::env::args().skip(1);
    let (model, wav) = match (args.next(), args.next()) {
        (Some(model), Some(wav)) => (model, wav),
        _ => panic!("usage: diarize_file <model.gguf> <audio.wav>"),
    };
    let mut reader = hound::WavReader::open(&wav).unwrap_or_else(|e| panic!("open {wav}: {e}"));
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();

    let started = std::time::Instant::now();
    let mut diarizer = diarize_rs::Diarizer::new(&model).expect("load model");
    let loaded = started.elapsed().as_secs_f64();
    let segments = diarizer
        .diarize(&samples, spec.sample_rate, spec.channels)
        .expect("diarize");
    let total = started.elapsed().as_secs_f64();

    let mut speakers: Vec<usize> = segments.iter().map(|s| s.speaker_id).collect();
    speakers.sort_unstable();
    speakers.dedup();
    let audio_s = samples.len() as f64 / spec.channels as f64 / spec.sample_rate as f64;
    println!("audio {audio_s:.1} s, load {loaded:.2} s, load + diarize {total:.2} s");
    println!("{} segments, {} speakers {speakers:?}", segments.len(), speakers.len());
}
