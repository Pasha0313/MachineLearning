import pandas as pd
import subprocess
import re
import time
import os

# ========= Configuration =========
llama_exe = r"C:\Data\Job\UK\Zencargo\LLAMA-cpu-x64\llama-cli.exe"
model_path = r"C:\Data\Job\UK\Zencargo\LLAMA-cpu-x64\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
prompt_file = r"C:\Data\Job\UK\Zencargo\prompts_for_llm.csv"
output_file = r"C:\Data\Job\UK\Zencargo\llm_enriched_output.csv"
error_log = r"C:\Data\Job\UK\Zencargo\llm_failures_log.csv"

# Limit number of rows for testing (set to None for full run)
TEST_LIMIT = 10
#TEST_LIMIT = None
# ========= HS Code Validator =========
def validate_hs_code(text: str) -> str | None:
    """
    Extracts the first valid 10-digit HS code from model output.
    """
    match = re.search(r"\b\d{10}\b", text)
    return match.group(0) if match else None

# ========= Load Prompts =========
if not os.path.exists(prompt_file):
    raise FileNotFoundError(f"❌ Prompt file not found: {prompt_file}")

df = pd.read_csv(prompt_file)
if TEST_LIMIT:
    df = df.head(TEST_LIMIT)

df["llm_response"] = ""
df["parsed_hs_code"] = ""

invalid_rows = []

# ========= Run LLM Inference =========
for i, row in df.iterrows():
    prompt = row["prompt"].strip()
    sku = row["sku_code"]

    # Ensure prompt is wrapped properly (failsafe)
    if not prompt.startswith("<s>[INST]"):
        prompt = f"<s>[INST] {prompt} [/INST]"

    print(f"🔍 [{i+1}/{len(df)}] Prompting for SKU {sku}")

    try:
        result = subprocess.run([
            llama_exe,
            "-m", model_path,
            "-p", prompt,
            "--n-predict", "96",
            "--temp", "0.7",           # more creative sampling
            "--top-p", "0.9",          # nucleus sampling
            "--repeat-penalty", "1.2"  # reduce repetition
        ], capture_output=True, text=True, timeout=30)

        output = result.stdout.strip()
        hs_code = validate_hs_code(output)

        # Store results
        df.at[i, "llm_response"] = output
        df.at[i, "parsed_hs_code"] = hs_code if hs_code else "❌ Invalid"

        print(f"📤 Raw Output:\n{output}\n{'-'*40}")
        if hs_code:
            print(f"✅ Parsed HS Code: {hs_code}")
        else:
            print("❌ No valid 10-digit code returned")
            invalid_rows.append({
                "sku_code": sku,
                "prompt": prompt,
                "llm_output": output
            })

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for SKU {sku}")
        df.at[i, "llm_response"] = "⏰ Timeout"
        df.at[i, "parsed_hs_code"] = "❌ Invalid"
        invalid_rows.append({
            "sku_code": sku,
            "prompt": prompt,
            "llm_output": "⏰ Timeout"
        })

    time.sleep(1)

# ========= Save Results =========
df.to_csv(output_file, index=False)
print(f"\n🎯 All results saved to: {output_file}")

if invalid_rows:
    pd.DataFrame(invalid_rows).to_csv(error_log, index=False)
    print(f"⚠️ {len(invalid_rows)} invalid entries logged to: {error_log}")
else:
    print("✅ No invalid entries detected.")
