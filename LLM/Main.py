import subprocess
import sys
import os
from Import_Config import *

# ✅ Load config and environment
config = load_config()
model_name, auto_run, live_output, user_input = get_control_settings(config)
os.environ["FINE_TUNE_MODEL"] = model_name

sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Define pipeline steps
steps = [
    ("Scrape Cointelegraph", "scraper_cointelegraph.py"),
    ("Scrape CoinDesk", "scraper_coindesk.py"),
    ("Scrape Cryptoslate", "scraper_cryptoslate.py"),
    ("Scrape Decrypt", "scraper_decrypt.py"),
    ("Data Preprocessing", "Data_Preprocessing_II.py"),
    ("Convert JSON to TXT", "Convert_JSON_to_TXT.py"),
    ("Fine-tune LLM", "Fine_Tune_LLM_III.py"),
    ("Model Inference", "Model_Inference.py"),
    ("Evaluate Model (optional)", "Model_Evaluation_II.py"),
    ("Interactive Chat (Dara)", "Prompt_Generator.py")
]

# ✅ Show available steps
print("🧩 Available steps:")
for idx, (name, _) in enumerate(steps, start=1):
    print(f"{idx}. {name}")

# ✅ Handle user selection
if auto_run:
    user_input = "all"

if user_input.strip().lower() == "all":
    selected_steps = set(range(1, len(steps) + 1))
else:
    selected_steps = {
        int(s.strip()) for s in user_input.split(",") if s.strip().isdigit()
    }

# ✅ Run the selected steps
for idx, (name, script) in enumerate(steps, start=1):
    if idx not in selected_steps:
        continue

    print(f"\n🚀 Starting step {idx}: {name}")
    try:
        if live_output:
            subprocess.run([sys.executable, script], check=True)
        else:
            result = subprocess.run(
                [sys.executable, script],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            print(f"✅ {name} completed successfully.\nOutput:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {name}:\n{e.stderr}")
        break
