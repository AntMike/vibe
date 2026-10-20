#!/usr/bin/env bash
# Fetch Nemotron-3-Diarization: NVIDIA q8_0 GGUF -> app model dir, .nemo -> baseline/models.
set -euo pipefail
export UV_CACHE_DIR="F:/Documents/Vibe/_app/cache/uv" UV_PYTHON_INSTALL_DIR="F:/Documents/Vibe/_app/cache/uv-python" HF_HOME="F:/Documents/Vibe/_app/cache/hf" PIP_CACHE_DIR="F:/Documents/Vibe/_app/cache/pip" TORCH_HOME="F:/Documents/Vibe/_app/cache/torch" NUMBA_CACHE_DIR="F:/Documents/Vibe/_app/cache/numba" TMP="F:/Documents/Vibe/_app/cache/tmp" TEMP="F:/Documents/Vibe/_app/cache/tmp"
cd "$(dirname "$0")/../../.."   # server/
URL=https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main
Q8=F:/Documents/Vibe/_app/data/Nemotron-3-Diarization.q8_0.gguf
NEMO=baseline/models/Nemotron-3-Diarization.nemo
mkdir -p baseline/models
get() { # file size dest
  [ "$(stat -c %s "$3" 2>/dev/null || echo 0)" = "$2" ] || curl -fL --retry 3 -o "$3" "$URL/$1"
  [ "$(stat -c %s "$3")" = "$2" ] || { echo "bad size: $3" >&2; exit 1; }
}
get Nemotron-3-Diarization.q8_0.gguf 107012128 "$Q8"
get Nemotron-3-Diarization.nemo 198676480 "$NEMO"
sha256sum "$Q8" "$NEMO"
