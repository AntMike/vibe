//! Speaker diarization with NVIDIA Sortformer.
//!
//! Everything runs in-process against a GGUF exported from the original NeMo
//! checkpoint (see `scripts/README.md`): the `sf_*` modules are the ggml graph,
//! `host/` is the mel frontend plus the AOSC speaker-cache state machine, and
//! `sf_backend` is the seam between them. There is no ONNX Runtime here.
//!
//! Two model generations load through the same [`Diarizer`]: Sortformer v2
//! (4 speakers, 80 ms frames; `sf_graph` + `host::aosc`) and v3,
//! Nemotron-3-Diarization (8 speakers, 10 ms frames; `sf_graph_v3` +
//! `host::v3`), told apart by the GGUF's `sortformer.version` key.

mod host;
mod sf_backend;
mod sf_gguf;
mod sf_graph;
mod sf_graph_v3;
mod sf_ops;
mod sf_runtime;
mod sf_transformer;
pub mod sf_weights;

pub use sf_graph::{run_chunk, run_chunk_stages, run_chunk_valid, ChunkOutput, Stage};

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use host::{DiarizationConfig, MelFrontend, SortformerHost, StreamGeometry, V3Geometry, V3Host, V3Model};
use sf_backend::GgmlBackend;
use sf_weights::SortformerWeights;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to initialize diarizer from {path}: {source}")]
    Init {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("diarization failed: {0}")]
    Diarize(Box<dyn std::error::Error + Send + Sync>),
}

pub struct Diarizer {
    host: Host,
    backend: GgmlBackend,
}

/// v2 and v3 share the graph seam but not the state machine (see `host::v3`).
enum Host {
    V2(SortformerHost),
    V3(V3Host),
}

impl Diarizer {
    /// Load a Sortformer GGUF.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        let init = |source: Box<dyn std::error::Error + Send + Sync>| Error::Init {
            path: path.to_path_buf(),
            source,
        };

        let weights = SortformerWeights::load(path).map_err(|e| init(Box::new(e)))?;

        // Use the filterbank the checkpoint ships rather than recomputing one.
        // NeMo built `preprocessor.fb` with torchaudio; librosa's triangles and
        // Slaney normalisation differ in the last few ulps, and the log at the
        // end of the frontend keeps those differences visible.
        let filterbank = weights.mel_filterbank().map_err(|e| init(Box::new(e)))?;
        let mel = MelFrontend::from_trained_filterbank(&filterbank).map_err(|e| init(Box::new(e)))?;

        let hp = weights.hparams();
        if hp.is_v3() {
            // The host pools the head's output back to encoder frames by the
            // subsampling factor; that only inverts the upsampling if they agree.
            if hp.upsample_factor != hp.enc_subsampling_factor {
                return Err(init(
                    format!(
                        "v3 upsample_factor {} != subsampling_factor {}",
                        hp.upsample_factor, hp.enc_subsampling_factor
                    )
                    .into(),
                ));
            }
            let silence = weights
                .silence_embedding()
                .map_err(|e| init(Box::new(e)))?
                .ok_or_else(|| init("v3 model without learnable_sil_emb".into()))?;
            let model = V3Model {
                num_speakers: hp.max_speakers as usize,
                emb_dim: hp.enc_d_model as usize,
                subsampling: hp.enc_subsampling_factor as usize,
                geom: V3Geometry {
                    chunk_len: host::V3_CHUNK_LEN,
                    right_context: host::V3_RIGHT_CONTEXT,
                    fifo_len: host::V3_FIFO_LEN,
                    update_period: host::V3_UPDATE_PERIOD,
                    spkcache_len: hp.stream_spkcache_len as usize,
                },
                scoring: hp.scoring.clone(),
                silence,
            };
            // HF's extractor windows with a symmetric Hann; v2's ONNX-era
            // frontend used librosa's periodic one.
            let host = V3Host::new(mel.with_symmetric_window(), model, DiarizationConfig::callhome());
            return Ok(Self {
                host: Host::V3(host),
                backend: GgmlBackend::new(weights),
            });
        }

        // Streaming geometry: these are the *inference-time* values, and they
        // are deliberately NOT read from the GGUF's `sortformer.streaming.*`
        // KVs. Those record the training-time configuration (chunk 188 / fifo 0
        // / spkcache 188) and would change the shape of every chunk. The ONNX
        // export carried the inference values in `metadata_props`
        // (chunk_len=124, fifo_len=124, spkcache_len=188), which is what the
        // recorded baseline in `baseline/` was produced with, and what
        // NVIDIA's published streaming configuration for v2 uses.
        // `right_context = 1` appears in neither file — it is a host-side
        // assumption inherited from parakeet-rs, reproduced here on purpose.
        let geom = StreamGeometry {
            chunk_len: 124,
            fifo_len: 124,
            spkcache_len: 188,
            right_context: 1,
        };

        Ok(Self {
            host: Host::V2(SortformerHost::new(mel, geom, DiarizationConfig::callhome())),
            backend: GgmlBackend::new(weights),
        })
    }

    pub fn diarize(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<Segment>> {
        let speaker_segments = match &mut self.host {
            Host::V2(host) => host.diarize(&mut self.backend, samples.to_vec(), sample_rate, channels),
            Host::V3(host) => host.diarize(&mut self.backend, samples.to_vec(), sample_rate, channels),
        }
        .map_err(|source| Error::Diarize(Box::new(source)))?;

        Ok(speaker_segments
            .into_iter()
            .map(|segment| Segment {
                start: segment.start as f64 / 16_000.0,
                end: segment.end as f64 / 16_000.0,
                speaker_id: segment.speaker_id,
            })
            .collect())
    }

    /// The raw `[T, n_spk]` speaker-probability matrix (`[T, 4]` at 80 ms for
    /// v2, `[T, 8]` at 10 ms for v3), before median filtering and
    /// binarisation. This is the artifact the `baseline/probs/` oracle records,
    /// so validation compares against it rather than against segments.
    #[doc(hidden)]
    pub fn diarize_raw(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(Vec<f32>, usize)> {
        match &mut self.host {
            Host::V2(host) => host.diarize_raw(&mut self.backend, samples, sample_rate, channels),
            Host::V3(host) => host.diarize_raw(&mut self.backend, samples, sample_rate, channels),
        }
        .map_err(|source| Error::Diarize(Box::new(source)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker_id: usize,
}
