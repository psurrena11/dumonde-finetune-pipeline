from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="unsloth/gemma-4-E4B-it", help="HuggingFace model ID")
args = parser.parse_args()

os.makedirs("/workspace/tuned_model", exist_ok=True)

print(f"Loading model: {args.model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    args.model,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
)

# Load JSONL and apply Gemma instruct chat template
examples = []
with open("/workspace/interviews-sft.jsonl", "r") as f:
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
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=5,
        output_dir="/workspace/tuned_model",
        report_to="none",
        logging_steps=10,
        save_steps=100,
    ),
)

trainer.train()

model.save_pretrained("/workspace/tuned_model")
tokenizer.save_pretrained("/workspace/tuned_model")

print("Done. Adapter saved to /workspace/tuned_model")
