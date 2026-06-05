import os
import argparse
import json
import glob

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

parser = argparse.ArgumentParser(description="SFT fine-tune via TRL + PEFT (any HF model, including custom-code)")
parser.add_argument("--model", default="empero-ai/openNemo-9B-abliterated",
                    help="HuggingFace model ID (must be trust_remote_code-compatible)")
parser.add_argument("--data", default="/workspace/training-sft.jsonl", help="Path to training JSONL")
parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA (saves ~12 GB VRAM on a 9B model)")
parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
args = parser.parse_args()

FINAL_DIR = "/workspace/tuned_model"
CHECKPOINT_DIR = "/workspace/checkpoints"
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Loading model: {args.model}")
quant_cfg = None
if args.qlora:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_cfg,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_cfg = LoraConfig(
    r=args.rank,
    lora_alpha=args.alpha,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

examples = []
with open(args.data) as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))

def format_example(ex):
    text = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = Dataset.from_list([format_example(ex) for ex in examples])

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    max_seq_length=args.max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        optim="adamw_8bit",
        output_dir=CHECKPOINT_DIR,
        report_to="none",
        logging_steps=10,
        save_steps=100,
    ),
)
trainer.train()

print("Merging LoRA adapter into base model...")
merged = model.merge_and_unload()
merged.save_pretrained(FINAL_DIR, safe_serialization=True)
tokenizer.save_pretrained(FINAL_DIR)

model.save_pretrained(os.path.join(FINAL_DIR, "adapter"))
tokenizer.save_pretrained(os.path.join(FINAL_DIR, "adapter"))

weights = glob.glob(os.path.join(FINAL_DIR, "*.safetensors"))
if not weights:
    raise RuntimeError(f"No model weights written to {FINAL_DIR} - save failed")

print(f"Done. Merged model saved to {FINAL_DIR} ({len(weights)} shard(s)); "
      f"adapter at {FINAL_DIR}/adapter; checkpoints in {CHECKPOINT_DIR}")
