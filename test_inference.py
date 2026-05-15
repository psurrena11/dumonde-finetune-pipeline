from unsloth import FastLanguageModel
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
parser.add_argument("--prompt", default="Background: US, lean-left, Northeast, Catholic, Millennial.\n\nWhat feels most urgent to you politically right now?")
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained('/workspace/tuned_model', load_in_4bit=False)
FastLanguageModel.for_inference(model)

text = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': args.prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=not args.no_thinking,
)
inputs = tokenizer(text=[text], return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
