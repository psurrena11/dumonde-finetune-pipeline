#!/bin/bash
set -e

echo "=== Creating isolated PEFT venv ==="
python3 -m venv /workspace/peft-env
source /workspace/peft-env/bin/activate

echo "=== Installing PyTorch (CUDA 12.4) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing PEFT training deps ==="
pip install transformers trl peft datasets bitsandbytes accelerate safetensors

echo ""
echo "=== Done ==="
echo "Activate with: source /workspace/peft-env/bin/activate"
echo "Train with:   python /workspace/vast/train_sft_peft.py --model <model-id> --qlora"
