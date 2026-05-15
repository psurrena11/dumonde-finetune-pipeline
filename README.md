# vast — Gemma-4 Fine-Tune Pipeline

Fine-tune Gemma-4 on vast.ai, export GGUF, serve via llama.cpp or Unsloth API.

## Instance

- Template: **PyTorch**
- GPU: **RTX 4090 (24GB VRAM)**

## Scripts

| File | Purpose |
|---|---|
| `train_sft.py` | SFT fine-tune via Unsloth + LoRA |
| `export_gguf.py` | Export adapter to Q4 and Q8 GGUF |
| `test_inference.py` | Quick inference test via Unsloth (no llama.cpp needed) |
| `serve.sh` | Serve Q8 GGUF via llama.cpp on port 8080 |
| `serve_unsloth.py` | OpenAI-compatible API via FastAPI + Unsloth (no llama.cpp) |
| `install.sh` | Install all Python deps on a fresh instance |

## Setup (fresh instance)

Generate SSH key and add to GitHub before cloning:
```bash
ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# Paste output into GitHub → Settings → SSH Keys
```

Then clone and install:
```bash
git clone git@github.com:psurrena11/vast.git /workspace/vast
cd /workspace/vast
bash install.sh
```

Upload training data:
```bash
# from local machine
scp -P <port> path/to/interviews-sft.jsonl root@<host>:/workspace/
```

## Workflow

### 1. Train
```bash
python /workspace/vast/train_sft.py
# Output: /workspace/tuned_model
```
> Note: `BNB_CUDA_VERSION=130` is set inside the script — no prefix needed.

### 2. Test
```bash
python /workspace/vast/test_inference.py
```

### 3. Export GGUF
```bash
BNB_CUDA_VERSION=130 python /workspace/vast/export_gguf.py
# Output: /workspace/gemma4-bws-Q4_K_M.gguf (local)
#         /workspace/gemma4-bws-Q8_0.gguf   (server)
```

### 4. Pull Q4 locally (LM Studio)
```bash
scp -P <port> root@<host>:/workspace/gemma4-bws-Q4_K_M.gguf ~/Downloads/
```
Load in LM Studio, chat template: Gemma.

### 5a. Serve via llama.cpp
Build llama.cpp first (see `install.sh` comment), then:
```bash
bash /workspace/vast/serve.sh
```

### 5b. Serve via Unsloth (no llama.cpp build needed)
```bash
python /workspace/vast/serve_unsloth.py
# API at http://<host>:<mapped-port>/v1/chat/completions
```

## Deploy to Modal.com (production)

```bash
pip install modal
modal setup
modal volume create gemma4-bws-weights
modal volume put gemma4-bws-weights ~/Downloads/gemma-4-e4b-it.Q8_0.gguf /gemma4-bws-Q8_0.gguf
modal deploy vast/modal_serve.py
```

Set `GEMMA4_API_URL` in `.env` to the returned Modal URL.

## Test prompt

```
Background: US, lean-left, Northeast, Catholic, Millennial.

What feels most urgent to you politically right now?
```

Expected: first-person voice, grounded in that background. No "as an AI" responses.
