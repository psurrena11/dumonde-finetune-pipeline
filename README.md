# LLM Fine-Tune Pipeline for vast.ai

SFT fine-tune any HuggingFace model on vast.ai using Unsloth + LoRA, export to GGUF, and serve via llama.cpp or a FastAPI endpoint.

Tested on Gemma-4 4B with an RTX 4090 (24GB VRAM).

## Scripts

| File | Purpose |
|---|---|
| `install.sh` | Install all deps and build llama.cpp on a fresh instance |
| `train_sft.py` | SFT fine-tune via Unsloth + LoRA |
| `export_gguf.py` | Merge adapter and export to Q4/Q8 GGUF |
| `test_inference.py` | Quick inference test via Unsloth |
| `serve.sh` | Serve Q8 GGUF via llama.cpp on port 8080 |
| `serve_unsloth.py` | OpenAI-compatible API via FastAPI + Unsloth |

## Requirements

- vast.ai instance with PyTorch template, GPU with 24GB+ VRAM
- Training data as JSONL with `messages` field (standard chat format)
- HuggingFace account (for model downloads)

## Setup

Generate SSH key and add to GitHub before cloning:
```bash
ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# Paste into GitHub → Settings → SSH Keys
```

Clone and install (builds llama.cpp — takes ~10 min):
```bash
git clone git@github.com:psurrena11/llm-finetune-pipelin.git /workspace/vast
bash /workspace/vast/install.sh
source ~/.bashrc
```

Upload training data:
```bash
scp -P <port> path/to/training-sft.jsonl root@<host>:/workspace/
```

## Workflow

### 1. Train
```bash
python /workspace/vast/train_sft.py
# Custom model:    python /workspace/vast/train_sft.py --model mistralai/Mistral-7B-Instruct-v0.3
# Custom data:     python /workspace/vast/train_sft.py --data /workspace/mydata.jsonl
# Disable thinking: python /workspace/vast/train_sft.py --no-thinking
```
Thinking mode is enabled by default. Use `--no-thinking` to disable it in the chat template.

Output: `/workspace/tuned_model` (LoRA adapter, ~160MB)

### 2. Test
```bash
python /workspace/vast/test_inference.py
# Custom prompt:    python /workspace/vast/test_inference.py --prompt "Your prompt here"
# Disable thinking: python /workspace/vast/test_inference.py --no-thinking
```

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

## Notes

- `export_gguf.py` creates an isolated Python venv for llama.cpp to avoid dependency conflicts with the training environment
- `BNB_CUDA_VERSION` and `TORCHDYNAMO_DISABLE` are set automatically in `train_sft.py` for compatibility with vast.ai CUDA 13.x instances
- llama.cpp is cloned and built by `install.sh` — no manual setup needed
