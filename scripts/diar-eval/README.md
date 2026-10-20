# diar-eval

Scores `vibe-server` diarization against the Slack `callSpeakers` data saved in
`F:/Documents/Vibe/<project>/transcript.vibe.json`. The reference is weak (Slack's indicator lags
and misses short turns), so only compare models against each other.

Nothing may be written to C:. Export these before `uv run`:

```
export UV_CACHE_DIR="F:/Documents/Vibe/_app/cache/uv" UV_PYTHON_INSTALL_DIR="F:/Documents/Vibe/_app/cache/uv-python" HF_HOME="F:/Documents/Vibe/_app/cache/hf" PIP_CACHE_DIR="F:/Documents/Vibe/_app/cache/pip" TMP="F:/Documents/Vibe/_app/cache/tmp" TEMP="F:/Documents/Vibe/_app/cache/tmp" PYTHONIOENCODING=utf-8
uv run scripts/diar-eval/der.py --models old.gguf,new.gguf --min-names 9 --limit 5
```

Uses `server/target/{release,debug}/vibe-server.exe` as found. Responses are cached in
`F:/Documents/Vibe/_app/eval/diar/<model-stem>/<project>.json`; `--fresh` redoes them.
