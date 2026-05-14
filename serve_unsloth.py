from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import os

os.environ["UNSLOTH_USE_MODELSCOPE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from unsloth import FastLanguageModel

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/tuned_model")
PORT = int(os.getenv("PORT", "8081"))

model, tokenizer = FastLanguageModel.from_pretrained(MODEL_PATH, load_in_4bit=True)
FastLanguageModel.for_inference(model)

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 300

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt from the response
    response = decoded[len(text):].strip() if decoded.startswith(text) else decoded.split("model\n", 1)[-1].strip()
    return {
        "choices": [{"message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
        "model": MODEL_PATH,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
