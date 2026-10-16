# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Transcribe every cropped piece with one or all ASR configs through `vibe-server transcribe`.

Saves the printed text to <out>/<config>/<piece>.txt and prints the time each config took.
A config that fails (say, a flag the server binary does not know yet) is reported and skipped.

Example:
    uv run scripts/asr-eval/run.py "F:/Documents/Vibe/Record-.../eval/pieces" --config v3
    uv run scripts/asr-eval/run.py "F:/Documents/Vibe/Record-.../eval/pieces" --grid --hotwords
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ["LOCALAPPDATA"]) / "github.com.thewh1teagle.vibe"
VAD = MODELS / "ggml-silero-v6.2.0.bin"
VOCABULARY = Path(r"F:\Documents\Vibe\vocabulary.json")

V3, TURBO = "ggml-large-v3.bin", "ggml-large-v3-turbo.bin"
# Whisper runs the way the app runs it: speech windows from the VAD model, beam 8, temperature 0.
BEAM = ["--beam-size", "8"]
CONFIGS = {
    "v3": {"model": V3, "args": BEAM, "prefilter": None},
    "v3-uk-en": {"model": V3, "args": [*BEAM, "--languages", "uk,en"], "prefilter": None},
    "v3-beam5": {"model": V3, "args": ["--beam-size", "5"], "prefilter": None},
    "turbo": {"model": TURBO, "args": BEAM, "prefilter": None},
    "turbo-uk-en": {"model": TURBO, "args": [*BEAM, "--languages", "uk,en"], "prefilter": None},
    "v3-loudnorm": {"model": V3, "args": BEAM, "prefilter": "loudnorm"},
    "parakeet": {"model": "parakeet-tdt-0.6b-v3-Q4_K_M.gguf", "args": [], "prefilter": None},
    "canary-uk": {"model": "canary-1b-v2-Q4_K_M.gguf", "args": ["--language", "uk"], "prefilter": None},
}


def find_server() -> Path:
    for build in ("release", "debug"):
        exe = REPO / "server/target" / build / "vibe-server.exe"
        if exe.exists():
            return exe
    sys.exit("vibe-server.exe not built: cargo build -p vibe-server (in server/), or pass --server")


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg") or next((str(p) for p in (REPO / "desktop/src-tauri").rglob("ffmpeg*.exe")), None)
    if not found:
        sys.exit("ffmpeg not found on PATH or under desktop/src-tauri")
    return found


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
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--config", choices=CONFIGS)
    pick.add_argument("--grid", action="store_true", help="run every config")
    ap.add_argument("--server", type=Path, help="vibe-server.exe; default: server/target/{release,debug}")
    ap.add_argument("--out", type=Path, help="default: <pieces>/../out")
    ap.add_argument("--hotwords", action="store_true", help="pass the vocabulary's known terms as --hotwords")
    ap.add_argument("--vocabulary", type=Path, default=VOCABULARY)
    ap.add_argument("--prompt", help="initial prompt for whisper")
    args = ap.parse_args()

    server = args.server or find_server()
    out = args.out or args.pieces.parent / "out"
    pieces = json.loads((args.pieces / "pieces.json").read_text(encoding="utf-8"))
    extra = ["--prompt", args.prompt] if args.prompt else []
    if args.hotwords:
        extra += ["--hotwords", ",".join(w for family in known_families(args.vocabulary) for w in family)]

    for name in CONFIGS if args.grid else [args.config]:
        config = CONFIGS[name]
        vad = ["--vad-model", str(VAD)]
        (out / name).mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                for piece in pieces:
                    wav = args.pieces / f"{piece['name']}.wav"
                    if config["prefilter"] == "loudnorm":
                        normed = Path(tmp) / wav.name
                        cmd = [find_ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-af", "loudnorm", "-ar", "16000", "-ac", "1", str(normed)]
                        subprocess.run(cmd, check=True)
                        wav = normed
                    cmd = [str(server), "transcribe", str(MODELS / config["model"]), str(wav), *vad, *config["args"], *extra]
                    done = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True)
                    (out / name / f"{piece['name']}.txt").write_text(done.stdout.strip() + "\n", encoding="utf-8")
            except subprocess.CalledProcessError as err:
                print(f"{name}: failed ({err.returncode})\n{(err.stderr or '')[-800:]}", file=sys.stderr)
                continue
        print(f"{name}: {time.perf_counter() - began:.1f} s for {len(pieces)} pieces")


if __name__ == "__main__":
    main()
