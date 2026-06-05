# duMonde LLM Fine-Tune Pipeline

SFT fine-tune any HuggingFace model on vast.ai using Unsloth + LoRA, export to GGUF, and serve via llama.cpp or a FastAPI endpoint.

Tested on Gemma-4 4B with an RTX 4090 (24GB VRAM).

## Scripts

| File | Purpose |
|---|---|
| `install.sh` | Install Unsloth deps and build llama.cpp on a fresh instance |
| `install_peft.sh` | Install PEFT deps in isolated venv (`/workspace/peft-env`) |
| `train_sft.py` | SFT fine-tune via Unsloth + LoRA |
| `train_sft_peft.py` | SFT fine-tune via TRL + PEFT (any HF model, including custom-code) |

| `export_gguf.py` | Merge adapter and export to Q4/Q8 GGUF |
| `test_inference.py` | Quick inference test via Unsloth |
| `serve.sh` | Serve Q8 GGUF via llama.cpp on port 8080 |
| `serve_unsloth.py` | OpenAI-compatible API via FastAPI + Unsloth |
| `modal_app.py` | Deploy the GGUF as an OpenAI-compatible API on Modal (serverless GPU) |

## Requirements

- vast.ai instance with PyTorch template, GPU with 24GB+ VRAM
- Training data as JSONL with `messages` field (standard chat format)
- HuggingFace account (for model downloads)

### PEFT backend (`train_sft_peft.py`)

Without Unsloth's memory optimizations, `train_sft_peft.py` needs significantly more VRAM. On an RTX 4090 (24GB), a 4B model fits comfortably with LoRA (rank 16). For larger models (7B+, 4B with high rank), use `--qlora` and/or upgrade to an A6000 (48GB) or A100 (80GB).

## Setup

Clone and install (builds llama.cpp — takes ~10 min):
```bash
git clone https://github.com/psurrena11/dumonde-finetune-pipeline.git /workspace/vast
bash /workspace/vast/install.sh
source ~/.bashrc
```

If using `train_sft_peft.py`, install the PEFT venv separately (avoids Unsloth dependency conflicts):
```bash
bash /workspace/vast/install_peft.sh
source /workspace/peft-env/bin/activate
```

Upload training data:
```bash
scp -P <port> path/to/training-sft.jsonl root@<host>:/workspace/
```

## Workflow

### 1. Train
```bash
python /workspace/vast/train_sft.py [flags]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `unsloth/gemma-4-E4B-it` | HuggingFace model ID |
| `--data` | `/workspace/training-sft.jsonl` | Path to training JSONL |
| `--rank` | `16` | LoRA rank (r) — higher = more capacity, more VRAM |
| `--alpha` | `16` | LoRA alpha — higher = stronger training data influence |
| `--no-thinking` | off | Disable thinking mode in chat template |

Output: `/workspace/tuned_model` (LoRA adapter, ~160MB)

> **Custom-code or non-Unsloth architectures (e.g. Nemotron-H, Mamba hybrids)?**
> Use `train_sft_peft.py` instead. Same flags as `train_sft.py` plus
> `--qlora` for 4-bit training. After training, test and serve via the GGUF
> path (sections 3 + 5) — `test_inference.py` and `serve_unsloth.py` are
> Unsloth-only.

### 2. Test
```bash
python /workspace/vast/test_inference.py [flags]
```

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `"What's on your mind lately?"` | Prompt to test with |
| `--no-thinking` | off | Disable thinking mode |

### 3. Export GGUF
```bash
python /workspace/vast/export_gguf.py
```
Output:
- `/workspace/gemma4-bws-Q4_K_M.gguf` — for local use (LM Studio, Ollama)
- `/workspace/gemma4-bws-Q8_0.gguf` — for API serving

### 4. Pull locally
```bash
scp -P <port> root@<host>:/workspace/gemma4-bws-Q4_K_M.gguf ~/Downloads/
```
Load in LM Studio or Ollama. Chat template: Gemma.

### 5. Serve on vast.ai

Via llama.cpp:
```bash
bash /workspace/vast/serve.sh
```

Via Unsloth (OpenAI-compatible, no llama.cpp needed):
```bash
python /workspace/vast/serve_unsloth.py
# API at http://<host>:<mapped-port>/v1/chat/completions
```

## Training data format

Standard HuggingFace chat JSONL — one example per line:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

See `training-sft.example.jsonl` for sample rows. Rename to `training-sft.jsonl` and replace with your data before uploading.

## Run locally with Ollama

Install Ollama, then create a model from your exported GGUF:

```bash
cp Modelfile.example Modelfile
# Edit Modelfile: replace 'your-model-Q4_K_M.gguf' with your actual file path
ollama create my-model -f Modelfile
ollama run my-model
```

The `Modelfile.example` includes the Gemma chat template and sensible default parameters. If you trained on a different model family, update the `TEMPLATE` block to match its chat format.

## Deploy on Modal (serverless GPU)

`modal_app.py` serves the Q4 GGUF through Ollama on a Modal GPU container (L4),
behind an OpenAI-compatible, bearer-authenticated endpoint. The model scales to
zero and cold-starts on demand. Consumed by the A/B testing tool in the
`beforeicallmyparents` repo.

The finetune emits **first-person persona prose**, not structured JSON. It works
on the A/B **Q&A tab** only — the Today/Conversation/URL/Text tabs expect a
briefing JSON schema and will error for this model.

### One-time setup

Install + authenticate Modal (local machine):
```bash
pip install modal
modal setup
```

Upload the GGUF to a Modal Volume (run from the dir holding the file). Use the
Q8_0 (API quality) — `modal_app.py` serves `gemma4-bws-Q8_0.gguf`:
```bash
modal volume create dumonde-models
modal volume put dumonde-models gemma4-bws-Q8_0.gguf /gemma4-bws-Q8_0.gguf
```

Create the shared API key as a Modal Secret (prints the value — save it):
```bash
KEY=$(openssl rand -hex 24); echo "API KEY: $KEY"; modal secret create dumonde-api-key API_KEY=$KEY
```

### Deploy
```bash
modal deploy modal_app.py
```
Prints the endpoint URL (also visible via `modal app list` or the dashboard):
```
https://<username>--dumonde-server-api.modal.run
```

### Test
First chat call cold-starts a GPU and imports the GGUF into the Volume (~30–60s, once, cached in the Volume); later cold starts just reload the cached model.
```bash
URL="https://<username>--dumonde-server-api.modal.run"
KEY="<your API_KEY>"

curl -s $URL/health

curl -s $URL/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"whats it like on the block today?"}]}'

modal app logs dumonde   # live logs
```

### Wire up the A/B tool (`beforeicallmyparents`)

The A/B harness calls the endpoint via the OpenAI SDK (`gemma4` model key). Set
these env vars (local `.env` and the Render service):

| Env var | Value |
|---|---|
| `GEMMA4_API_URL` | the endpoint base **with `/v1`**: `https://<username>--dumonde-server-api.modal.run/v1` |
| `GEMMA4_MODEL_ID` | `dumonde` (must match the Ollama model name in `modal_app.py`) |
| `GEMMA4_API_KEY` | the real value stored in the `dumonde-api-key` Modal Secret (the endpoint enforces bearer auth — a placeholder will 401) |

The SDK posts to `${GEMMA4_API_URL}/chat/completions` with `Authorization: Bearer ${GEMMA4_API_KEY}`. Use the **Q&A tab** to exercise it.

### Lifecycle / cost

- **Idle:** scales to zero. `scaledown_window` keeps a container warm for that many seconds after the last request (max 3600). At idle = zero, cost is $0.
- **Stop entirely:** `modal app stop dumonde`. Bring back with `modal deploy modal_app.py`.
- **Apply config/code changes:** edits to `modal_app.py` only go live on the next `modal deploy` — Modal does not watch the file, and `git commit` does not deploy.
- An L4 bills ~$0.80/hr while warm; lower `scaledown_window` (e.g. `300`) for bursty use, raise it to stay warm through a working session.

The backend sends `Authorization: Bearer ${DUMONDE_MODEL_API_KEY}`. The key lives only in the Modal Secret and Render env — never in git.

### Persona

The Modelfile baked into `modal_app.py` sets a default persona via `SYSTEM`. Callers override it per request by including a `system` message in the chat payload — useful for duMonde's taxonomy-driven voices.

## Notes

- `export_gguf.py` creates an isolated Python venv for llama.cpp to avoid dependency conflicts with the training environment
- `BNB_CUDA_VERSION` (124) and `TORCHDYNAMO_DISABLE` are set automatically in `train_sft.py` to match the CUDA 12.4 torch wheel installed by `install.sh`
- llama.cpp is cloned and built by `install.sh` — no manual setup needed

## Verification

Sanity-check the trainer files after pulling or editing. No tests in this repo —
just syntax + CLI surface checks (both read-only, no GPU needed).

```bash
# Syntax
python -c "import ast; ast.parse(open('train_sft.py').read())"
python -c "import ast; ast.parse(open('train_sft_peft.py').read())"

# CLI surfaces
python train_sft.py --help
python train_sft_peft.py --help

# Confirm common flags match across backends
diff <(python train_sft.py --help) <(python train_sft_peft.py --help) | head
```

If `train_sft.py --help` shows `--model`, `--data`, `--rank`, `--alpha` and so
does `train_sft_peft.py --help`, the two backends are CLI-compatible and either
one can be swapped in by changing only the script name.
