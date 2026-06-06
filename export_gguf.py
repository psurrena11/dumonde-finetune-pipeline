import os
import subprocess
import sys

TUNED_DIR    = "/workspace/tuned_model"
LLAMA_DIR    = "/workspace/llama.cpp"
VENV_DIR     = "/workspace/llama-env"
VENV_PY      = f"{VENV_DIR}/bin/python"
VENV_PIP     = f"{VENV_DIR}/bin/pip"
QUANTIZE_BIN = f"{LLAMA_DIR}/build/bin/llama-quantize"
OUT_F16      = "/workspace/tuned-f16.gguf"
OUT_Q4       = "/workspace/tuned-Q4_K_M.gguf"
OUT_Q8       = "/workspace/tuned-Q8_0.gguf"

# Step 1: merge if using Unsloth (adapter is a separate checkpoint)
if os.path.exists(os.path.join(TUNED_DIR, "adapter")):
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        pass  # PEFT path — already merged by train script
    else:
        print("=== Merging adapter (Unsloth) ===")
        model, tokenizer = FastLanguageModel.from_pretrained(TUNED_DIR, load_in_4bit=False)
        model.save_pretrained_merged(TUNED_DIR, tokenizer, save_method="merged_16bit")

# Step 2: clone llama.cpp if not present, then build
if not os.path.exists(LLAMA_DIR):
    print("=== Cloning llama.cpp ===")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMA_DIR], check=True)

if not os.path.exists(QUANTIZE_BIN):
    print("=== Building llama.cpp ===")
    subprocess.run(["cmake", "-B", "build", "-DGGML_CUDA=ON"], cwd=LLAMA_DIR, check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count())], cwd=LLAMA_DIR, check=True)

# Step 3: set up isolated venv for llama.cpp Python deps
if not os.path.exists(VENV_PY):
    print("=== Creating llama.cpp venv ===")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    subprocess.run([VENV_PIP, "install", "-q", "numpy", "sentencepiece", "gguf", "transformers"], check=True)
    subprocess.run([VENV_PIP, "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)

# Step 4: convert to f16 GGUF (base for quantization)
convert = f"{LLAMA_DIR}/convert_hf_to_gguf.py"
print("=== Converting to f16 GGUF ===")
subprocess.run([VENV_PY, convert, TUNED_DIR, "--outfile", OUT_F16, "--outtype", "f16"], check=True)

# Step 5: quantize to Q8 and Q4 using llama-quantize binary
print("=== Quantizing to Q8_0 ===")
subprocess.run([QUANTIZE_BIN, OUT_F16, OUT_Q8, "Q8_0"], check=True)

print("=== Quantizing to Q4_K_M ===")
subprocess.run([QUANTIZE_BIN, OUT_F16, OUT_Q4, "Q4_K_M"], check=True)

print("\nDone.")
print(f"  Local: {OUT_Q4}")
print(f"  API:   {OUT_Q8}")
