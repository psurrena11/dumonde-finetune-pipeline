#!/bin/bash
/workspace/llama.cpp/build/bin/llama-server \
    -m /workspace/gemma4-bws-Q8_0.gguf \
    --port 8080 \
    --host 0.0.0.0
