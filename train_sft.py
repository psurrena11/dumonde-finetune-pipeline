import os
os.environ.setdefault("BNB_CUDA_VERSION", "124")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="unsloth/gemma-4-E4B-it", help="HuggingFace model ID")
parser.add_argument("--data", default="/workspace/training-sft.jsonl", help="Path to training JSONL")
parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode in chat template")
args = parser.parse_args()
enable_thinking = not args.no_thinking

FINAL_DIR = "/workspace/tuned_model"
CHECKPOINT_DIR = "/workspace/checkpoints"
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Loading model: {args.model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    args.model,
    load_in_4bit=False,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=args.rank,
    lora_alpha=args.alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
)

# Load JSONL and apply Gemma instruct chat template
examples = []
with open(args.data, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))

def format_example(ex):
    text = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return {"text": text}

dataset = Dataset.from_list([format_example(ex) for ex in examples])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=5,
        output_dir=CHECKPOINT_DIR,
        report_to="none",
        logging_steps=10,
        save_steps=100,
    ),
)

trainer.train()

# Save a standalone, directly-loadable model at FINAL_DIR. A bare
# model.save_pretrained() writes only the LoRA adapter, which inference/serve/
# export then fail to pick up, leaving you serving the base model. Merge to
# 16bit so /workspace/tuned_model is self-contained.
model.save_pretrained_merged(FINAL_DIR, tokenizer, save_method="merged_16bit")

# Also keep the lightweight adapter alongside for re-merging at other precisions.
model.save_pretrained(os.path.join(FINAL_DIR, "adapter"))
tokenizer.save_pretrained(os.path.join(FINAL_DIR, "adapter"))

# Fail loudly if the merged weights did not land.
import glob
weights = glob.glob(os.path.join(FINAL_DIR, "*.safetensors"))
if not weights:
    raise RuntimeError(f"No model weights written to {FINAL_DIR} - save failed")

print(f"Done. Merged model saved to {FINAL_DIR} ({len(weights)} shard(s)); "
      f"adapter at {FINAL_DIR}/adapter; checkpoints in {CHECKPOINT_DIR}")
