# /// script
# requires-python = ">=3.10"
# dependencies = ["pyannote.metrics", "pyannote.core", "requests"]
# ///
"""Score the server's speaker diarization against the Slack call-speaker data saved with recordings.

WEAK REFERENCE: Slack's active-speaker indicator lags and misses short turns, so absolute numbers
mean little; only compare models against each other.

Reference = `callSpeakers` turns (by name) in F:/Documents/Vibe/<project>/transcript.vibe.json;
hypothesis = server segments (by speaker id). DER uses collar 1.0 (+-0.5 s), overlap skipped.
Server responses are cached in F:/Documents/Vibe/_app/eval/diar/<model-stem>/<project>.json (--fresh to redo).

STORAGE: nothing may go to C:. Run uv itself with:
    export UV_CACHE_DIR="F:/Documents/Vibe/_app/cache/uv" UV_PYTHON_INSTALL_DIR="F:/Documents/Vibe/_app/cache/uv-python" \
           HF_HOME="F:/Documents/Vibe/_app/cache/hf" PIP_CACHE_DIR="F:/Documents/Vibe/_app/cache/pip" \
           TMP="F:/Documents/Vibe/_app/cache/tmp" TEMP="F:/Documents/Vibe/_app/cache/tmp" PYTHONIOENCODING=utf-8
    uv run scripts/diar-eval/der.py --models a.gguf,b.gguf --limit 5
"""

import os

CACHE = "F:/Documents/Vibe/_app/cache"
for key, sub in (("UV_CACHE_DIR", "uv"), ("UV_PYTHON_INSTALL_DIR", "uv-python"), ("HF_HOME", "hf"),
                 ("PIP_CACHE_DIR", "pip"), ("TMP", "tmp"), ("TEMP", "tmp")):
    os.environ.setdefault(key, f"{CACHE}/{sub}")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
RECORDINGS = Path("F:/Documents/Vibe")
EVAL = RECORDINGS / "_app/eval/diar"
MODELS = Path(os.environ["LOCALAPPDATA"]) / "github.com.thewh1teagle.vibe"
DEFAULT_DIAR = "diar_streaming_sortformer_4spk-v2.q8_0.gguf"


def find_server() -> Path:
    for build in ("release", "debug"):
        exe = REPO / "server/target" / build / "vibe-server.exe"
        if exe.exists():
            return exe
    sys.exit("vibe-server.exe not built (cargo build -p vibe-server in server/)")


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([str(find_server()), "serve", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if requests.get(f"{base}/health", timeout=1).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        sys.exit("server did not become ready")
    try:
        r = requests.post(f"{base}/v1/models/load", json={"path": str(MODELS / "ggml-large-v3-turbo.bin")}, timeout=600)
        r.raise_for_status()
    except Exception:
        proc.kill()
        raise
    return proc


def diarize(port: int, audio: Path, model: Path) -> dict:
    # Stream the open handle: Cyrillic paths break curl on Windows.
    with audio.open("rb") as fh:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/audio/transcriptions",
            files={"file": ("audio" + audio.suffix, fh)},
            data={"response_format": "verbose_json", "language": "uk",
                  "vad_model": str(MODELS / "ggml-silero-v6.2.0.bin"), "diarize_model": str(model)},
            timeout=3600)
    r.raise_for_status()
    return r.json()


def annotation(turns) -> Annotation:
    ann = Annotation()
    for start, end, label in turns:
        if end > start:
            ann[Segment(start, end)] = label
    return ann


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=DEFAULT_DIAR, help="comma-separated diarization .gguf (name in models dir or path)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-names", type=int, default=0)
    ap.add_argument("--port", type=int, default=3187)
    ap.add_argument("--fresh", action="store_true", help="ignore cached responses")
    args = ap.parse_args()

    models = [Path(m) if Path(m).is_absolute() else MODELS / m for m in args.models.split(",")]
    recs = []
    for path in sorted(RECORDINGS.glob("*/transcript.vibe.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        turns = data.get("callSpeakers") or []
        audio = next(path.parent.glob("audio.*"), None)
        if turns and audio and len({t["name"] for t in turns}) >= args.min_names:
            recs.append((path.parent.name, audio, turns))
    recs = recs[: args.limit] if args.limit else recs

    def cached(name: str, model: Path) -> Path:
        return EVAL / model.stem / f"{name}.json"

    need = any(args.fresh or not cached(n, m).exists() for n, _, _ in recs for m in models)
    proc = start_server(args.port) if need else None
    der = [DiarizationErrorRate(collar=1.0, skip_overlap=True) for _ in models]
    pur = [DiarizationPurity(collar=1.0, skip_overlap=True) for _ in models]
    print("NOTE: reference is weak (Slack's indicator lags and misses short turns); compare models, not absolutes.")
    print(f"{'recording':44} {'names':>5} " + " | ".join(f"{m.stem[:26]:^26}" for m in models))
    print(f"{'':44} {'':>5} " + " | ".join(f"{'DER':>8} {'cnt':>6} {'purity':>8}" for _ in models))
    try:
        for name, audio, turns in recs:
            ref = annotation((t["start"], t["end"], t["name"]) for t in turns)
            names = len(ref.labels())
            cells = []
            for i, model in enumerate(models):
                cache = cached(name, model)
                if args.fresh or not cache.exists():
                    began = time.perf_counter()
                    resp = diarize(args.port, audio, model)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
                    print(f"  [{name[:30]}] {model.stem}: {time.perf_counter() - began:.0f} s", file=sys.stderr)
                resp = json.loads(cache.read_text(encoding="utf-8"))
                hyp = annotation((s["start"], s["end"], s["speaker"]) for s in resp.get("segments", []) if s.get("speaker") is not None)
                cells.append(f"{der[i](ref, hyp):8.1%} {abs(len(hyp.labels()) - names):6d} {pur[i](ref, hyp):8.1%}")
            print(f"{name[:44]:44} {names:5d} " + " | ".join(cells))
    finally:
        if proc:
            proc.kill()
    # abs(metric) is the accumulated rate: total-duration-weighted.
    if recs:
        print(f"{'OVERALL (duration-weighted)':44} {'':>5} " + " | ".join(f"{abs(d):8.1%} {'':6} {abs(p):8.1%}" for d, p in zip(der, pur)))


if __name__ == "__main__":
    main()
