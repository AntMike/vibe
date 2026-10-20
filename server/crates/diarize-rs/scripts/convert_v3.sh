#!/usr/bin/env bash
# Build the f32 GGUF with NVIDIA's unmodified NeMo-Speech.cpp converter.
set -euo pipefail
export UV_CACHE_DIR="F:/Documents/Vibe/_app/cache/uv" UV_PYTHON_INSTALL_DIR="F:/Documents/Vibe/_app/cache/uv-python" HF_HOME="F:/Documents/Vibe/_app/cache/hf" PIP_CACHE_DIR="F:/Documents/Vibe/_app/cache/pip" TORCH_HOME="F:/Documents/Vibe/_app/cache/torch" NUMBA_CACHE_DIR="F:/Documents/Vibe/_app/cache/numba" TMP="F:/Documents/Vibe/_app/cache/tmp" TEMP="F:/Documents/Vibe/_app/cache/tmp"
cd "$(dirname "$0")/../../.."   # server/
SRC=baseline/NeMo-Speech.cpp
[ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/NVIDIA/NeMo-Speech.cpp.git "$SRC"
git -C "$SRC" rev-parse HEAD
OUT="$PWD/baseline/models/Nemotron-3-Diarization.f32.gguf"
NEMO="$PWD/baseline/models/Nemotron-3-Diarization.nemo"
cd "$SRC"
uv run --python 3.11 --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  --with-requirements requirements.txt \
  python convert_model.py "$NEMO" --architecture diarization --outtype f32 -o "$OUT"
sha256sum "$OUT"
