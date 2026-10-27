# Convert BuzzASR/ukrainian to ggml

Everything goes on F:, nothing on C:. Needs git, uv. Set `UV_CACHE_DIR` to a folder on F: first if `--with torch` would otherwise fill the uv cache on C:.

```bash
export HF_HOME=/f/Documents/Vibe/_models/hf
cd /f/Documents/Vibe/_models

# 1. the HF model (transformers format)
uv run --with huggingface_hub hf download BuzzASR/ukrainian --local-dir buzz-ukrainian-hf

# 2. whisper.cpp (the converter) and the OpenAI whisper repo (assets the converter reads)
git clone --depth 1 https://github.com/ggml-org/whisper.cpp whisper.cpp
git clone --depth 1 https://github.com/openai/whisper whisper-openai

# 3. convert: args are <model dir> <openai whisper repo> <output dir>
uv run --with torch --with transformers --with numpy \
  whisper.cpp/models/convert-h5-to-ggml.py buzz-ukrainian-hf whisper-openai .

# 4. name it the way asr-context_001.py expects
mv ggml-model.bin ggml-buzz-uk.bin
```

Result: `F:\Documents\Vibe\_models\ggml-buzz-uk.bin`; the `buzz-uk` row skips while it is missing.
If the converter complains about a missing tokenizer or config, copy `vocab.json`, `added_tokens.json` and `config.json` from the HF folder into the model dir it reads.
