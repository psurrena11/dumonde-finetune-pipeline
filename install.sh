#!/bin/bash
set -e

echo "=== Installing Python deps ==="
pip install --upgrade pip

# Unsloth + CUDA deps (vast.ai PyTorch template already has CUDA/torch)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -U unsloth unsloth-zoo
pip install trl transformers datasets
pip uninstall torchao -y || true

# Force reinstall GPU torch — cu124 required for torchao/torch.int1 support
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install torchvision --index-url https://download.pytorch.org/whl/cu124

# For serve_unsloth.py
pip install fastapi uvicorn pydantic

echo ""
echo "=== Building llama.cpp (needed for GGUF export) ==="
apt-get install -y cmake
if [ ! -d "/workspace/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
fi
cd /workspace/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd /workspace

echo ""
echo "=== Installing CLI tools ==="
apt-get install -y neovim bat eza btop || apt-get install -y neovim batcat eza btop
# some distros package bat as batcat — symlink it
if command -v batcat &>/dev/null && ! command -v bat &>/dev/null; then
    ln -sf "$(which batcat)" /usr/local/bin/bat
fi

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
echo "=== Done — run 'source ~/.bashrc' to activate aliases in this shell ==="
