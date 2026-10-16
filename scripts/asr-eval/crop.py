# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Crop evenly spaced pieces out of a call recording for ASR evaluation.

Skips the first --skip seconds, splits the rest into --pieces equal slots and cuts
--seconds from the start of each slot with ffmpeg (no re-encode). Writes
piece-01.wav, piece-02.wav, ... and pieces.json ([{"name", "start", "seconds"}]).

Example:
    uv run scripts/asr-eval/crop.py "F:/Documents/Vibe/Record-.../audio.wav" --pieces 4
"""

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg") or next((str(p) for p in (REPO / "desktop/src-tauri").rglob("ffmpeg*.exe")), None)
    if not found:
        sys.exit("ffmpeg not found on PATH or under desktop/src-tauri")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--skip", type=int, default=60, help="seconds to leave out at the start")
    ap.add_argument("--out", type=Path, help="default: <recording dir>/eval/pieces")
    args = ap.parse_args()

    with wave.open(str(args.recording)) as w:
        duration = w.getnframes() / w.getframerate()
    slot = (duration - args.skip) / args.pieces
    if slot < args.seconds:
        sys.exit(f"{duration:.0f} s is too short for {args.pieces} x {args.seconds} s after skipping {args.skip} s")

    out = args.out or args.recording.parent / "eval" / "pieces"
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    pieces = []
    for i in range(args.pieces):
        name = f"piece-{i + 1:02d}"
        start = int(args.skip + i * slot)
        cmd = [ffmpeg, "-v", "error", "-y", "-ss", str(start), "-t", str(args.seconds), "-i", str(args.recording), "-c", "copy", str(out / f"{name}.wav")]
        subprocess.run(cmd, check=True)
        pieces.append({"name": name, "start": start, "seconds": args.seconds})
        print(f"{name}: {start} s .. {start + args.seconds} s")
    (out / "pieces.json").write_text(json.dumps(pieces, indent=2), encoding="utf-8")
    print(f"wrote {len(pieces)} pieces to {out}")


if __name__ == "__main__":
    main()
