import os
import torch
import nltk
from nltk.tokenize import word_tokenize
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bert_score import score as bert_score

# Download NLTK tokenizer
nltk.download('punkt')

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
        max_new_tokens=max_length,  
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    return torch.exp(loss).item()

def compute_bleu(reference, hypothesis):
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    return nltk.translate.bleu_score.sentence_bleu([ref_tokens], hyp_tokens)

def evaluate_model(test_prompts, ground_truths=None):
    model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
    model_dir = f"./crypto_llm_{model_name.replace('/', '_')}"
    tokenizer, model = load_model(model_dir)

    print(f"📊 Evaluating model `{model_name}` on test prompts...\n")
    results = []

    for i, prompt in enumerate(test_prompts, start=1):
        response = generate_response(prompt, tokenizer, model)
        ppl = compute_perplexity(response, tokenizer, model)

        ref = ground_truths[i - 1] if ground_truths and i <= len(ground_truths) else ""
        bleu = compute_bleu(ref, response) if ref else 0.0

        print(f"\n🧠 Prompt {i}: {prompt}")
        print(f"🔮 Response: {response}")
        print(f"📉 Perplexity: {ppl:.2f} | 🟦 BLEU: {bleu:.4f}")

        results.append((prompt, response, ppl, ref, bleu))

    # Compute BERTScore (batch-based)
    if ground_truths:
        preds = [r[1] for r in results]
        refs = [r[3] for r in results]
        P, R, F1 = bert_score(preds, refs, lang="en", verbose=True)
        for i, (p, r, f) in enumerate(zip(P, R, F1)):
            results[i] += (f.item(),)
            print(f"🧠 Prompt {i+1} BERTScore F1: {f:.4f}")
    else:
        for i in range(len(results)):
            results[i] += (None,)  # Placeholder for BERTScore

    # Save to file
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        for res in results:
            prompt, response, ppl, ref, bleu, bert = res
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Perplexity: {ppl:.2f}\nBLEU: {bleu:.4f}\n")
            f.write(f"BERTScore F1: {bert:.4f}\n\n" if bert is not None else "\n")

    print("\n✅ Evaluation complete. Results saved to 'evaluation_results.txt'")
    return results

if __name__ == "__main__":
    test_prompts = [
        "Bitcoin price dropped suddenly because",
        "Ethereum upgrades to proof-of-stake and",
        "The SEC files a lawsuit against",
        "Cardano's adoption rate is growing as",
        "Solana hits a new high after"
    ]

    # Optional: Provide ground truth completions for reference
    references = [
        "a large number of long positions were liquidated due to leverage.",
        "it aims to reduce energy consumption and increase scalability.",
        "a major crypto firm violated existing securities laws.",
        "new partnerships and DApp launches are contributing.",
        "positive sentiment followed a major network upgrade."
    ]

    evaluate_model(test_prompts, references)
