from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

class PromptRequest(BaseModel):
    prompt: str

model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
model_dir = f"./crypto_llm_{model_name.replace('/', '_')}"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

app = FastAPI()

@app.post("/generate")
def generate(prompt_request: PromptRequest):
    inputs = tokenizer(prompt_request.prompt, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=500,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": output_text}
