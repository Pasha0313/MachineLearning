import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Show only fatal TF messages
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# Get model name from environment or fallback
model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
print(f"🧠 Using model: {model_name}")

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def fine_tune_model(dataset_path, model_output_dir):
    print("📄 Loading dataset using streaming-friendly Datasets library...")

    # Load line-by-line text file
    dataset = load_dataset("text", data_files={"train": dataset_path})["train"]
    print(f"📦 Raw samples loaded: {len(dataset)}")

    # Tokenize and filter out empty lines
    def tokenize_function(example):
        texts = [t for t in example["text"] if isinstance(t, str) and t.strip()]
        return tokenizer(texts, truncation=True, max_length=128)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    ).filter(lambda x: len(x["input_ids"]) > 0)

    print(f"✅ Valid tokenized samples: {len(tokenized_dataset)}")
    if len(tokenized_dataset) == 0:
        raise RuntimeError("❌ No usable data after tokenization. Check your input file.")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("🚀 Starting training...")
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
        logging_first_step=True,
        dataloader_num_workers=6
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"✅ Fine-tuned model saved to {model_output_dir}")

if __name__ == "__main__":
    model_output_dir = f"./crypto_llm_{model_name.replace('/', '_')}"
    fine_tune_model("cleaned_all_articles.txt", model_output_dir)
