from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import os

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL only

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Get model name from environment or fallback
model_name = os.getenv("FINE_TUNE_MODEL", "distilgpt2")
print(f"🧠 Using model: {model_name}")

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def fine_tune_model(dataset_path, model_output_dir):
    print("📂 Loading and flattening dataset...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()

    merged_path = "merged_dataset.txt"
    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write(text.replace("\n", " "))  # Flatten line breaks if needed

    print("📄 Preparing dataset...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=merged_path,
        block_size=128
    )

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
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"✅ Fine-tuned model saved to {model_output_dir}")

if __name__ == "__main__":
    # ✅ Now use the dynamic folder
    model_output_dir = f"./crypto_llm_{model_name.replace('/', '_')}"
    fine_tune_model('cleaned_all_articles.txt', model_output_dir)
