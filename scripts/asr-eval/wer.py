# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jiwer==4.0.0",
# ]
# ///
"""Score ASR outputs against the proofread references of the cropped pieces (WER and CER).

The second argument is either an output dir from run.py (one row per <config> subdir) or the
app's transcript.vibe.json (segments whose midpoint falls inside each piece, scored as "app").
Only pieces with a <piece>.reference.txt are scored. Prints a markdown table sorted by WER.

Example:
    uv run scripts/asr-eval/wer.py "F:/Documents/Vibe/Record-.../eval/pieces"
    uv run scripts/asr-eval/wer.py "F:/Documents/Vibe/Record-.../eval/pieces" "F:/Documents/Vibe/Record-.../transcript.vibe.json"
"""

import argparse
import json
import re
import sys
from pathlib import Path

import jiwer


def normalize(text: str, reference: bool = False) -> str:
    text = text.lower()
    if reference:
        text = text.replace("[?]", " ")  # unresolved words are not scored
    text = re.sub(r"['’ʼ‘`]", "", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def app_texts(transcript: Path, pieces: list[dict]) -> dict[str, dict[str, str]]:
    """The app's segments (start/stop in centiseconds) whose midpoint lies inside each piece."""
    segments = json.loads(transcript.read_text(encoding="utf-8"))["segments"]
    texts = {}
    for piece in pieces:
        lo, hi = piece["start"], piece["start"] + piece["seconds"]
        texts[piece["name"]] = " ".join(s["text"].strip() for s in segments if lo <= (s["start"] + s["stop"]) / 200 <= hi)
    return {"app": texts}


def dir_texts(out: Path, names: list[str]) -> dict[str, dict[str, str]]:
    return {
        config.name: {n: (config / f"{n}.txt").read_text(encoding="utf-8") for n in names if (config / f"{n}.txt").exists()}
        for config in sorted(p for p in out.iterdir() if p.is_dir())
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pieces", type=Path)
    ap.add_argument("target", type=Path, nargs="?", help="run.py output dir (default <pieces>/../out) or a .vibe.json")
    args = ap.parse_args()

    pieces = json.loads((args.pieces / "pieces.json").read_text(encoding="utf-8"))
    refs = {}
    for piece in pieces:
        ref = args.pieces / f"{piece['name']}.reference.txt"
        if ref.exists() and (text := normalize(ref.read_text(encoding="utf-8"), reference=True)):
            refs[piece["name"]] = text
    if not refs:
        sys.exit(f"no non-empty *.reference.txt in {args.pieces}; run reference.py and proofread first")

    target = args.target or args.pieces.parent / "out"
    if target.suffix == ".json":
        configs = app_texts(target, [p for p in pieces if p["name"] in refs])
    else:
        configs = dir_texts(target, list(refs))

    rows = []
    for config, texts in configs.items():
        names = [n for n in refs if n in texts]  # ponytail: a missing piece is left out of that config's overall
        if not names:
            continue
        hyps = [normalize(texts[n]) for n in names]
        per_piece = {n: jiwer.wer(refs[n], h) for n, h in zip(names, hyps)}
        overall = jiwer.wer([refs[n] for n in names], hyps), jiwer.cer([refs[n] for n in names], hyps)
        rows.append((config, per_piece, *overall))

    rows.sort(key=lambda r: r[2])
    print("| config | " + " | ".join(refs) + " | WER | CER |")
    print("|---" * (len(refs) + 3) + "|")
    for config, per_piece, wer, cer in rows:
        cells = [f"{per_piece[n]:.1%}" if n in per_piece else "-" for n in refs]
        print(f"| {config} | " + " | ".join(cells) + f" | {wer:.1%} | {cer:.1%} |")


if __name__ == "__main__":
    main()
