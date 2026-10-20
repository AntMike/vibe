# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers==5.18.0", "numpy", "soundfile", "librosa"]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""Offline Nemotron-3-Diarization reference (HF transformers 5.18.0, CPU, f32).

Usage: dump_v3_reference.py <clip.wav> <name>
Writes (relative to server/baseline/): probs/<name>.f32 + .json ({"shape": [T, 8]}),
segments/<name>.json ({"segments": [{"end","speaker_id","start"}]}), e2e_baseline format.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

wav, name = sys.argv[1], sys.argv[2]
out = Path(__file__).resolve().parents[3] / "baseline"
MODEL = "nvidia/Nemotron-3-Diarization"

with wave.open(wav, "rb") as r:
    assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2
    audio = np.frombuffer(r.readframes(r.getnframes()), "<i2").astype(np.float32) / 32768.0

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForAudioFrameClassification.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
inputs = processor(audio, sampling_rate=16000)
with torch.no_grad():
    logits = model(**inputs).logits  # (1, T, 8)
probs = logits.sigmoid()[0].numpy().astype("<f4")
T, n = probs.shape

(out / "probs").mkdir(parents=True, exist_ok=True)
(out / "segments").mkdir(parents=True, exist_ok=True)
probs.tofile(out / "probs" / f"{name}.f32")
(out / "probs" / f"{name}.json").write_text(json.dumps({"shape": [T, n], "dtype": "float32", "frame_ms": 10}))
segs = processor.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]
(out / "segments" / f"{name}.json").write_text(json.dumps(
    {"segments": [{"end": s["End"], "speaker_id": s["Speaker"], "start": s["Start"]} for s in segs]}, indent=1))
print(f"shape [{T}, {n}] min {probs.min():.4g} max {probs.max():.4g} segments {len(segs)}")
