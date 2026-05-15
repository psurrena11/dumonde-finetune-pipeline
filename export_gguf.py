import os
import subprocess
import sys
from unsloth import FastLanguageModel

MERGED_DIR  = "/workspace/gemma4-bws"
LLAMA_DIR   = "/workspace/llama.cpp"
VENV_DIR    = "/workspace/llama-env"
VENV_PY     = f"{VENV_DIR}/bin/python"
VENV_PIP    = f"{VENV_DIR}/bin/pip"
OUT_Q4      = "/workspace/gemma4-bws-Q4_K_M.gguf"
OUT_Q8      = "/workspace/gemma4-bws-Q8_0.gguf"

# Step 1: merge adapter into base model and save as safetensors
print("=== Merging adapter into base model ===")
model, tokenizer = FastLanguageModel.from_pretrained("/workspace/tuned_model", load_in_4bit=False)
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {MERGED_DIR}")

# Step 2: clone llama.cpp if not present
if not os.path.exists(LLAMA_DIR):
    print("=== Cloning llama.cpp ===")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMA_DIR], check=True)

# Step 3: set up isolated venv for llama.cpp deps
if not os.path.exists(VENV_PY):
    print("=== Creating llama.cpp venv ===")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    subprocess.run([VENV_PIP, "install", "-q", "-r", f"{LLAMA_DIR}/requirements.txt"], check=True)

# Step 4: convert to GGUF
convert = f"{LLAMA_DIR}/convert_hf_to_gguf.py"
print("=== Converting to Q4_K_M ===")
subprocess.run([VENV_PY, convert, MERGED_DIR, "--outfile", OUT_Q4, "--outtype", "q4_k_m"], check=True)

print("=== Converting to Q8_0 ===")
subprocess.run([VENV_PY, convert, MERGED_DIR, "--outfile", OUT_Q8, "--outtype", "q8_0"], check=True)

print("\nDone.")
print(f"  Local: {OUT_Q4}")
print(f"  API:   {OUT_Q8}")
