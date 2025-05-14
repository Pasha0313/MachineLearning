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

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=200,  
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(test_prompts):
    model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
    model_dir = f"./crypto_llm_{model_name.replace('/', '_')}"
    tokenizer, model = load_model(model_dir)

    print(f"📊 Evaluating model `{model_name}` on test prompts...\n")
    results = []

    for i, prompt in enumerate(test_prompts, start=1):
        response = generate_response(prompt, tokenizer, model)
        print(f"\n🧠 Prompt {i}: {prompt}")
        print(f"🔮 Response: {response}\n")
        results.append((prompt, response))

    return results

if __name__ == "__main__":
    test_prompts = [
        "Bitcoin price dropped suddenly because",
        "Ethereum upgrades to proof-of-stake and",
        "The SEC files a lawsuit against",
        "Cardano's adoption rate is growing as",
        "Solana hits a new high after"
    ]

    evaluate_model(test_prompts)
