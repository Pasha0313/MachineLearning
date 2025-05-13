import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model(model_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=200):
    inputs = tokenizer(prompt, return_tensors='pt')

    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        num_return_sequences=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(test_prompts, model_dir, save_path=None):
    tokenizer, model = load_model(model_dir)
    results = []

    print(f"📊 Evaluating model from {model_dir}...\n")
    for i, prompt in enumerate(test_prompts, start=1):
        print(f"🧠 Prompt {i}: {prompt}")
        response = generate_response(prompt, tokenizer, model)
        print(f"🔮 Response:\n{response}\n")
        results.append((prompt, response))

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for prompt, response in results:
                f.write(f"Prompt: {prompt}\nResponse: {response}\n\n")
        print(f"📁 Evaluation results saved to: {save_path}")

    return results

if __name__ == "__main__":
    # Load model name from environment or fallback
    model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
    model_dir = f"./crypto_llm_{model_name.replace('/', '_')}"

    # Define test prompts
    test_prompts = [
        "Bitcoin price surged after",
        "Ethereum will likely outperform Bitcoin if",
        "The SEC announced new regulation on",
        "Solana’s outage impacted",
        "Cardano is seeing increased adoption due to"
    ]

    # Run evaluation and save results
    evaluate_model(test_prompts, model_dir, save_path="evaluation_results.txt")
