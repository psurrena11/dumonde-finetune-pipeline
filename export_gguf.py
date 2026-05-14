from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "/workspace/tuned_model",
    load_in_4bit=True,
)

# Q4 for local testing (LM Studio)
model.save_pretrained_gguf("/workspace/gemma4-bws", tokenizer, quantization_method="q4_k_m")

# Q8 for API serving
model.save_pretrained_gguf("/workspace/gemma4-bws", tokenizer, quantization_method="q8_0")

print("Done.")
print("  Local: /workspace/gemma4-bws-Q4_K_M.gguf")
print("  API:   /workspace/gemma4-bws-Q8_0.gguf")
