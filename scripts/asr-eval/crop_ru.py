# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Cut 4 x 120 s Russian-dominated windows from the Oct 27 call into <eval>/pieces-ru.

Slides over the app transcript's segments (centiseconds); a window qualifies when ru segments of
speakers 4 and 1 are >= 70% of its speech time. Writes <id>.wav (16 kHz mono), <id>.app.txt (the
engine text in the span) and pieces.json. --dry-run only prints the chosen windows.

Example:
    uv run scripts/asr-eval/crop_ru.py --dry-run
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from crop import find_ffmpeg  # noqa: E402

CALL = r"F:\Documents\Vibe\2026-10-27-17-20-*"
EVAL = Path(r"F:\Documents\Vibe\_eval\2026-10-20")
RU_SPEAKERS = {1, 4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--min-share", type=float, default=0.7)
    ap.add_argument("--out", type=Path, default=EVAL / "pieces-ru")
    args = ap.parse_args()

    call = Path(glob.glob(CALL)[0])
    segs = json.loads((call / "transcript.vibe.json").read_text(encoding="utf-8"))["segments"]

    def share(lo: int) -> tuple[float, float]:
        """(ru share of speech seconds, speech seconds) in [lo, lo+seconds)."""
        hi, ru, total = lo + args.seconds, 0.0, 0.0
        for s in segs:
            t = max(0, min(s["stop"] / 100, hi) - max(s["start"] / 100, lo))
            total += t
            if s.get("language") == "ru" and s.get("speaker") in RU_SPEAKERS:
                ru += t
        return (ru / total if total else 0.0), total

    end = int(segs[-1]["stop"] / 100) - args.seconds
    cands = sorted(((*share(lo), lo) for lo in range(0, end, 10)), key=lambda c: -c[1])  # most speech first
    chosen: list[tuple[float, float, int]] = []
    for c in cands:
        if c[0] >= args.min_share and all(abs(c[2] - o[2]) >= args.seconds for o in chosen):
            chosen.append(c)
        if len(chosen) == args.count:
            break
    if len(chosen) < args.count:
        sys.exit(f"only {len(chosen)} windows with ru share >= {args.min_share}")
    chosen.sort(key=lambda c: c[2])

    pieces = []
    for i, (sh, speech, lo) in enumerate(chosen, 1):
        name = f"ru-{i:02d}"
        print(f"{name}: {lo} s .. {lo + args.seconds} s  ru share {sh:.2f}  speech {speech:.0f} s")
        pieces.append({"name": name, "start": lo, "seconds": args.seconds, "source": call.name})
        if args.dry_run:
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-ss", str(lo), "-t", str(args.seconds), "-i", str(call / "audio.wav"), "-ar", "16000", "-ac", "1", str(args.out / f"{name}.wav")], check=True)
        text = " ".join(s["text"].strip() for s in segs if lo <= (s["start"] + s["stop"]) / 200 <= lo + args.seconds)
        (args.out / f"{name}.app.txt").write_text(text + "\n", encoding="utf-8")
    if not args.dry_run:
        (args.out / "pieces.json").write_text(json.dumps(pieces, indent=2), encoding="utf-8")
        print(f"wrote {len(pieces)} pieces to {args.out}")


if __name__ == "__main__":
    main()
