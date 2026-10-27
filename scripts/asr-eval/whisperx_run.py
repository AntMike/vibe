# /// script
# requires-python = ">=3.12"
# dependencies = ["whisperx", "torch", "torchaudio"]
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# torchaudio = { index = "pytorch-cu126" }
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Transcribe every .wav in a folder with whisperx large-v3 (language auto-detected), one .txt per wav.

Caches go to F:\\Documents\\Vibe\\_models\\hf, never C:.

Example:
    uv run scripts/asr-eval/whisperx_run.py "F:/Documents/Vibe/_eval/2026-10-20/pieces" "F:/Documents/Vibe/_eval/2026-10-20/out/whisperx"
"""

import os

CACHE = r"F:\Documents\Vibe\_models\hf"
for var in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[var] = CACHE  # before torch/whisperx import

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--json", action="store_true", help="also write <piece>.json with segments (start/end in seconds, text)")
    args = ap.parse_args()

    import whisperx

    model = whisperx.load_model("large-v3", args.device, compute_type=args.compute_type, language=None, download_root=CACHE)
    args.out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(args.folder.glob("*.wav"))
    t0 = time.time()
    for wav in wavs:
        result = model.transcribe(whisperx.load_audio(str(wav)), batch_size=8)
        text = "\n".join(s["text"].strip() for s in result["segments"])
        (args.out / f"{wav.stem}.txt").write_text(text + "\n", encoding="utf-8")
        if args.json:
            segs = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]
            (args.out / f"{wav.stem}.json").write_text(json.dumps({"text": text, "segments": segs}, ensure_ascii=False), encoding="utf-8")
    print(f"whisperx: {len(wavs)} pieces -> {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
