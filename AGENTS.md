# duMonde Fine-Tune Pipeline

Single-directory repo (no packages, no test framework, no linter). Each script is standalone.

## Workflow order (dependency chain)

```
prepare_training.py  →  train_sft.py  →  test_inference.py  →  export_gguf.py  →  serve.sh / modal
```

## Key commands

```bash
# Train (merges adapter to /workspace/tuned_model automatically)
python train_sft.py --model unsloth/gemma-4-E4B-it --data /workspace/training-sft.jsonl

# Test
python test_inference.py --model-path /workspace/tuned_model

# Export GGUF (creates isolated venv for llama.cpp)
python export_gguf.py

# Serve
bash serve.sh                              # llama.cpp, port 8080
python serve_unsloth.py                    # FastAPI, port 8081
```

## Architecture gotchas

- `train_sft.py` line 83: `save_pretrained_merged()` is required — bare `save_pretrained()` writes only the LoRA adapter (160 MB), which inference/export scripts fail to load silently. The merge produces a self-contained model (~8 GB). The adapter is also kept at `/workspace/tuned_model/adapter` for re-merging at other precisions.
- `export_gguf.py` creates a separate venv `/workspace/llama-env` for llama.cpp Python deps to avoid conflicts with Unsloth's torch build.
- Scripts hardcode `/workspace/` paths (vast.ai convention). Training output goes to `/workspace/tuned_model`, GGUF exports to `/workspace/gemma4-bws-*.gguf`.
- `BNB_CUDA_VERSION=124` and `TORCHDYNAMO_DISABLE=1` are set automatically in `train_sft.py` and `test_inference.py` via `os.environ.setdefault`.

## Backend selection (which trainer to use)

- `train_sft.py` — Unsloth path. Only works for architectures Unsloth supports
  (Gemma, Llama, Mistral, Phi, Qwen, DeepSeek, etc.). Faster, lower VRAM.
- `train_sft_peft.py` — TRL + PEFT path. Works for any HuggingFace model,
  including custom-code architectures (Nemotron-H, Mamba hybrids, etc.).
  Needs `trust_remote_code=True`. Slower, more VRAM, supports `--qlora`.

**Pick by architecture:** if the model card shows tags like `nemotron_h`,
`mamba`, `custom_code`, or ships a non-standard `modeling_*.py`, use
`train_sft_peft.py`. Otherwise use `train_sft.py`.

**Test/serve after a PEFT training:** `test_inference.py` and `serve_unsloth.py`
are Unsloth-only. For a PEFT-trained model, run `export_gguf.py` and use
`serve.sh` (llama.cpp + Q8 GGUF) for both testing and serving.

## Data format

Standard HuggingFace chat JSONL with `messages` array:
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`prepare_training.py` enriches this: reads a screener CSV, matches interviews to demographics, injects taxonomy-aware system prompts, and can fetch intersection scaffolding from the duMonde admin API.

## Serving / deployment

- `serve.sh`: serves Q8 GGUF via `llama.cpp` on port 8080
- `serve_unsloth.py`: OpenAI-compatible FastAPI endpoint on port 8081 (env `MODEL_PATH`, `PORT`)
- `modal_app.py` mentioned in README but **not in this repo** — lives elsewhere

## Notes

- No package.json, no lint/typecheck — `pip install` via `install.sh` only
- `llama.cpp` cloned to `/workspace/llama.cpp` by `install.sh` or on-demand by `export_gguf.py`
- Training tested on Gemma-4 4B with RTX 4090 (24 GB VRAM)
- JSONL and GGUF files are gitignored (`.gitignore` blocks `*.jsonl` and `*.gguf`; example files use `.example.jsonl` extension)
