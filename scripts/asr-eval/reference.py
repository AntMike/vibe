# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Draft a gold reference transcript for each cropped piece from several engines' outputs.

For every piece, sends the engines' texts (<out>/<config>/<piece>.txt) and the vocabulary's
known terms to an AI CLI (claude by default, or codex) as text on stdin, and writes
<pieces>/<piece>.reference.txt. Only these ~60 s text snippets reach the AI, never audio.
Proofread every reference by ear before trusting WER numbers.

Example:
    uv run scripts/asr-eval/reference.py "F:/Documents/Vibe/Record-.../eval/pieces" --ai claude
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VOCABULARY = Path(r"F:\Documents\Vibe\vocabulary.json")
AI = {
    "claude": ["claude", "-p", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence"],
    "codex": ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--color", "never", "-"],
}
INSTRUCTIONS = (
    "Several automatic transcripts of the same 60-second piece of a Ukrainian team call with English "
    "product/tech terms. Write the single most likely verbatim transcript: keep the speaker's language and "
    "script, Ukrainian endings on borrowed words in Cyrillic, brand names in Latin; use the known terms list "
    "for spellings; mark a word you cannot resolve as `[?]`; no commentary."
)


def known_families(path: Path) -> list[list[str]]:
    """Pinned terms, then terms seen in 2+ recordings, each with its other spellings; blocked ones left out."""
    vocab = json.loads(path.read_text(encoding="utf-8"))
    terms, blocked = vocab.get("terms", {}), {t.lower() for t in vocab.get("blocked", [])}
    known = [t for t, v in terms.items() if v.get("seen", 0) >= 2 and t not in vocab.get("pinned", [])]
    known.sort(key=lambda t: (-terms[t]["seen"], -terms[t].get("last", 0)))
    return [[t, *terms.get(t, {}).get("forms", [])] for t in [*vocab.get("pinned", []), *known] if t.lower() not in blocked][:80]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pieces", type=Path)
    ap.add_argument("--out-dir", type=Path, help="engine outputs; default: <pieces>/../out")
    ap.add_argument("--ai", choices=AI, default="claude")
    ap.add_argument("--vocabulary", type=Path, default=VOCABULARY)
    ap.add_argument("--force", action="store_true", help="redo pieces that already have a reference")
    args = ap.parse_args()

    out = args.out_dir or args.pieces.parent / "out"
    exe = shutil.which(AI[args.ai][0]) or sys.exit(f"{args.ai} CLI not found on PATH")
    families = known_families(args.vocabulary) if args.vocabulary.exists() else []
    terms = "\n".join(f"- {f[0]}" + (f" ({', '.join(f[1:])})" if len(f) > 1 else "") for f in families) or "(none)"

    for piece in json.loads((args.pieces / "pieces.json").read_text(encoding="utf-8")):
        name = piece["name"]
        target = args.pieces / f"{name}.reference.txt"
        if target.exists() and not args.force:
            print(f"{name}: reference exists, skipped (--force to redo)")
            continue
        drafts = sorted(out.glob(f"*/{name}.txt"))
        if not drafts:
            print(f"{name}: no engine output under {out}, skipped", file=sys.stderr)
            continue
        # ponytail: drafts are numbered, not named, so the AI does not favour an engine by name.
        texts = "\n\n".join(f"### Transcript {i}\n{p.read_text(encoding='utf-8').strip()}" for i, p in enumerate(drafts, 1))
        prompt = f"{INSTRUCTIONS}\n\nKnown terms (spelling, then other forms):\n{terms}\n\n{texts}\n"
        done = subprocess.run([exe, *AI[args.ai][1:]], input=prompt, capture_output=True, encoding="utf-8", errors="replace")
        if done.returncode or not done.stdout.strip():
            print(f"{name}: {args.ai} failed ({done.returncode})\n{done.stderr[-800:]}", file=sys.stderr)
            continue
        target.write_text(done.stdout.strip() + "\n", encoding="utf-8")
        print(f"{name}: reference from {len(drafts)} transcripts -> {target}")

    print("\nProofread every *.reference.txt by ear against its piece-NN.wav before trusting WER.")


if __name__ == "__main__":
    main()
