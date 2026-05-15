#!/bin/bash
set -e

echo "=== Installing Python deps ==="
pip install --upgrade pip

# Unsloth + CUDA deps (vast.ai PyTorch template already has CUDA/torch)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -U unsloth unsloth-zoo
pip install trl transformers datasets
pip install -U torchao

# For serve_unsloth.py
pip install fastapi uvicorn pydantic

echo ""
echo "=== Installing CLI tools ==="
apt-get install -y neovim bat eza btop

cat >> ~/.bashrc << 'EOF'

alias vim="nvim"
alias v="nvim"
alias top="btop"
alias cat="bat"
alias ..="cd .."
EOF

source ~/.bashrc

echo ""
echo "=== Optional: build llama.cpp (needed for serve.sh) ==="
echo "Run manually if needed:"
echo "  apt-get install -y cmake && cd /workspace && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build -DLLAMA_CUDA=ON && cmake --build build --config Release -j$(nproc)"
echo ""
echo "=== Done ==="
