from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained('/workspace/tuned_model', load_in_4bit=False)
FastLanguageModel.for_inference(model)

prompt = "Background: US, lean-left, Northeast, Catholic, Millennial.\n\nWhat feels most urgent to you politically right now?"

text = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': prompt}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text=[text], return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
