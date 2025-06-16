import os
import random
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load model and tokenizer
model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
print(f"🧠 Using model: {model_name}")

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set reproducibility seed
set_seed(42)

def fine_tune_model(dataset_path, model_output_dir):
    print("📄 Loading dataset ...")
    raw_dataset = load_dataset("text", data_files={"data": dataset_path})["data"]
    print(f"📦 Raw samples loaded: {len(raw_dataset)}")

    # Shuffle and split dataset
    raw_dataset = raw_dataset.shuffle(seed=42)
    split_dataset = raw_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    tokenized_train = tokenized_train.filter(lambda x: len(x["input_ids"]) > 0)
    tokenized_val = tokenized_val.filter(lambda x: len(x["input_ids"]) > 0)

    print(f"✅ Train samples: {len(tokenized_train)}, Validation samples: {len(tokenized_val)}")

    if len(tokenized_train) == 0:
        raise RuntimeError("❌ No usable training data.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting fine-tuning ...")
    trainer.train()
    print("✅ Training complete")

    print("💾 Saving model ...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"✅ Model saved to `{model_output_dir}`")

if __name__ == "__main__":
    output_dir = f"./crypto_llm_{model_name.replace('/', '_')}"
    fine_tune_model("cleaned_all_articles.txt", output_dir)
