import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model(model_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=200):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate output
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Strip the prompt from the output
    generated_ids = output_ids[0][input_ids.shape[-1]:]  # remove input prompt
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

if __name__ == "__main__":
    model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
    model_dir = f"./crypto_llm_{model_name.replace('/', '_')}"

    print(f"🤖 Dara is ready using `{model_name}`.")
    print("💬 Ask me anything about crypto (type 'exit' to quit).")

    tokenizer, model = load_model(model_dir)
    
    while True:
        prompt = input("\n🗣️ You: ")
        if prompt.strip().lower() == 'exit':
            print("👋 Dara signing off.")
            break
        response = generate_response(prompt, tokenizer, model)
        print("\n🧠 Dara:\n", response, "\n")
