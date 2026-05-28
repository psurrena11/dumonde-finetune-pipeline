import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from unsloth import FastLanguageModel
import argparse
import torch

DEFAULT_SYSTEM = (
    "You are a very liberal, Black / African American, Catholic, Gen Z, "
    "from the Northeast, Middle class background, Bachelor's degree education. "
    "Answer in your own authentic voice: specific, grounded in your actual "
    "experience, and honest about where you contradict yourself. Do not sound "
    "like a policy document or a survey response."
)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="/workspace/tuned_model",
                    help="Model/adapter dir (use a checkpoint-* dir if top-level save lacks the adapter)")
parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
parser.add_argument("--prompt", default="What's on your mind lately?")
parser.add_argument("--system", default=DEFAULT_SYSTEM,
                    help="Persona system prompt the model was trained to condition on")
parser.add_argument("--greedy", action="store_true", help="Greedy decode instead of sampling")
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(args.model_path, load_in_4bit=False)
FastLanguageModel.for_inference(model)

messages = []
if args.system:
    messages.append({'role': 'system', 'content': args.system})
messages.append({'role': 'user', 'content': args.prompt})

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=not args.no_thinking,
)
inputs = tokenizer(text=[text], return_tensors='pt').to('cuda')

gen_kwargs = dict(max_new_tokens=300)
if not args.greedy:
    gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.95)

outputs = model.generate(**inputs, **gen_kwargs)

n_in = inputs["input_ids"].shape[1]
new_tokens = outputs[0][n_in:]
print(f"\n[input tokens: {n_in} | generated tokens: {len(new_tokens)}]\n")
print("=== RAW (with special tokens) ===")
print(tokenizer.decode(new_tokens, skip_special_tokens=False))
print("\n=== ANSWER ===")
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
