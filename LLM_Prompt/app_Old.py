import random
import streamlit as st
import pandas as pd
import subprocess
import re
import os

# File paths
BASE_DIR = "C:/Data/Job/UK/Zencargo"
EXCEL_FILE = os.path.join(BASE_DIR, "Automating HS Code validation.xlsx")
PROMPT_CSV = os.path.join(BASE_DIR, "prompts_for_llm.csv")
ENRICHED_CSV = os.path.join(BASE_DIR, "llm_enriched_output.csv")
FINAL_CSV = os.path.join(BASE_DIR, "final_validated_output.csv")
LLAMA_CLI = os.path.join(BASE_DIR, "LLAMA-cpu-x64", "llama-cli.exe")
MODEL_PATH = os.path.join(BASE_DIR, "LLAMA-cpu-x64", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# --- Utility Functions ---
def validate_hs_code(code):
    if pd.isna(code):
        return False
    cleaned = re.sub(r"[\s\-.]", "", str(code))
    return cleaned.isdigit() and len(cleaned) == 10

def extract_hs_code(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group(0) if match else None

# --- Add this example pool above create_prompt() ---
example_pool = [
    ("Men’s cotton shirt (Material: Cotton; Use: Clothing; Dimensions: Size M)", "6205200000"),
    ("Bluetooth speaker (Material: Plastic and metal; Use: Electronics; Dimensions: 10x5x5 cm)", "8518210000"),
    ("Toy car (Material: Plastic; Use: Children's toy; Dimensions: 15x7x6 cm)", "9503007000"),
    ("Aluminum laptop stand (Material: Aluminum; Use: Office; Dimensions: 30x25x5 cm)", "7616999099"),
    ("Ceramic mug (Material: Ceramic; Use: Drinkware; Dimensions: 300ml)", "6912002310"),
    ("Winter gloves (Material: Wool; Use: Cold weather; Dimensions: Size L)", "6116930000"),
    ("Notebook computer (Material: Plastic and metal; Use: Computing; Dimensions: 13-inch)", "8471300000"),
    ("Wooden chair (Material: Wood; Use: Seating furniture; Dimensions: 80x45x45 cm)", "9401610000"),
    ("Leather wallet (Material: Leather; Use: Personal item; Dimensions: 10x8x1 cm)", "4202310000"),
]


def create_prompt(row):
    sku = row["sku_code"]
    description = row.get("product_description", "general item")
    material = row.get("product_material", "Unknown")
    use = row.get("product_use", "General use")
    dimensions = row.get("product_dimensions", "Unspecified")

    full_description = f"{sku} - {description} (Material: {material}; Use: {use}; Dimensions: {dimensions})"

    examples = random.sample(example_pool, 2)
    example_str = "\n\n".join(
        f"Product: {desc}\nHS Code: {code}" for desc, code in examples
    )

    instruction = (
        "You are a customs classification assistant.\n\n"
        "Your task is to assign a valid 10-digit UK HS code (Harmonized System Code) for the given product.\n\n"
        "Respond with only the 10-digit numeric HS code. Do not include explanations or extra text.\n\n"
        f"Examples:\n{example_str}\n\n"
        f"Product: {full_description}\n"
        "HS Code:"
    )
    return f"<s>[INST] {instruction.strip()} [/INST]>"

# --- Phase 1 ---
def run_phase_1():
    df = pd.read_excel(EXCEL_FILE)
    df["hs_code"] = df["hs_code"].astype(str)
    df["hs_code_valid"] = df["hs_code"].apply(validate_hs_code)
    df["needs_enrichment"] = ~df["hs_code_valid"]

    if "product_description" not in df.columns:
        df["product_description"] = "general item"

    rows = df[df["needs_enrichment"]].copy()

    # Add metadata simulation before prompt creation
    def simulate_metadata(sku):
        sku_str = str(sku)
        digits = [char for char in reversed(sku_str) if char.isdigit()]
        if digits:
            endings = int(digits[0])
        else:
            endings = random.randint(0, 9)

        categories = [
            ("Plastic bottle", "Plastic", "Food storage", "20x10x10 cm"),
            ("Men’s shirt", "Cotton", "Clothing", "Size M"),
            ("Bluetooth speaker", "Plastic and metal", "Electronics", "10x5x5 cm"),
            ("Toy car", "Plastic", "Children's toy", "15x7x6 cm"),
            ("Laptop stand", "Aluminum", "Office use", "30x25x5 cm"),
            ("Leather wallet", "Leather", "Personal accessory", "10x8x1 cm"),
            ("LED lamp", "Plastic and glass", "Desk lighting", "40 cm height"),
            ("Ceramic mug", "Ceramic", "Drinkware", "300ml"),
            ("Winter gloves", "Wool", "Cold weather", "Size L"),
            ("Notebook", "Paper", "Stationery", "A5 size"),
        ]
        return categories[endings % len(categories)]

    # Apply simulated metadata to each row
    rows["product_description"], rows["product_material"], \
    rows["product_use"], rows["product_dimensions"] = zip(
        *rows["sku_code"].apply(simulate_metadata)
    )
    rows["prompt"] = rows.apply(create_prompt, axis=1)
    rows[["sku_code", "product_description", "prompt"]].to_csv(PROMPT_CSV, index=False)
    return rows

# --- Phase 2 ---
def run_phase_2(prompt_df, max_rows=None):
    prompt_df = prompt_df.copy()
    prompt_df["llm_response"] = ""
    prompt_df["parsed_hs_code"] = ""

    if max_rows:
        prompt_df = prompt_df.head(max_rows)

    for i, row in prompt_df.iterrows():
        sku = row["sku_code"] if "sku_code" in row else f"Row {i+1}"
        st.write(f"🔍 [{i+1}/{len(prompt_df)}] Prompting for SKU {sku}...")

        prompt = row["prompt"]
        result = subprocess.run([
            LLAMA_CLI,
            "-m", MODEL_PATH,
            "-p", prompt,
            "--n-predict", "128",
            "--temp", "0.7",
            "--top-p", "0.9",
            "--repeat-penalty", "1.2"
        ], capture_output=True, text=True)
        output = result.stdout.strip()
        parsed = extract_hs_code(output)

        prompt_df.at[i, "llm_response"] = output
        prompt_df.at[i, "parsed_hs_code"] = parsed if parsed else "❌ Invalid"

        if parsed:
            st.success(f"✅ Parsed HS Code: {parsed}")
        else:
            st.error("❌ No valid 10-digit code returned")

    prompt_df.to_csv(ENRICHED_CSV, index=False)
    st.write(f"💾 Saved enriched results to: {ENRICHED_CSV}")
    return prompt_df

# --- Phase 3 ---
def run_phase_3(enriched_df):
    enriched_df["approved_hs_code"] = enriched_df["parsed_hs_code"].apply(
        lambda code: code if validate_hs_code(code) else ""
    )
    enriched_df.to_csv(FINAL_CSV, index=False)
    st.success("✅ Phase 3 completed: HS codes validated and saved.")
    return enriched_df

# --- Phase 4 (Review UI) ---
def run_phase_4_ui(df):
    st.subheader("🔍 Review & Edit HS Code Suggestions")

    for idx, row in df.iterrows():
        st.markdown(f"**SKU:** `{row['sku_code']}`")
        st.markdown(f"**Description:** {row['product_description']}")
        st.markdown(f"**LLM Suggestion:** `{row['parsed_hs_code']}`")

        new_code = st.text_input(f"Edit HS code for SKU {row['sku_code']}", value=row['parsed_hs_code'], key=f"code_{idx}")
        df.at[idx, "approved_hs_code"] = new_code.strip()

        st.markdown("---")

    df.to_csv(FINAL_CSV, index=False)
    st.success("📝 All updates saved to final output file.")

# --- UI Layout ---
if st.button("❌ Exit App"):
    st.warning("✅ App exited. Close browser tab or refresh to restart.")
    st.stop()

st.title("📦 HS Code Validator & Enricher")

# --- Run Full Pipeline Button ---
if st.button("▶️ Run Full Pipeline"):
    st.info("Phase 1: Preparing prompts...")
    prompts = run_phase_1()
    st.session_state["prompts"] = prompts
    st.success(f"{len(prompts)} prompts generated.")
    st.session_state["phase_1_done"] = True

# --- Run Phase 2 only if Phase 1 is done ---
if st.session_state.get("phase_1_done", False):
    st.info("Phase 2: Running LLM inference...")

    max_rows_input = st.number_input(
        "How many rows would you like to process in Phase 2? (Leave 0 to run all)",
        min_value=0,
        max_value=len(st.session_state["prompts"]),
        value=10,
        step=1,
        key="phase2_limit"
    )
    max_rows = None if max_rows_input == 0 else max_rows_input

    if st.button("▶️ Run Phase 2"):
        enriched = run_phase_2(st.session_state["prompts"], max_rows=max_rows)
        st.session_state["enriched"] = enriched
        st.session_state["phase_2_done"] = True
        st.success("HS Code enrichment complete.")

# --- Phase 3 ---
if st.session_state.get("phase_2_done", False):
    st.info("Phase 3: Validating and finalizing...")
    final = run_phase_3(st.session_state["enriched"])
    st.session_state["final"] = final

    st.download_button("📥 Download Final Results", final.to_csv(index=False), "final_validated_output.csv")

    with st.expander("🔎 View Final Table"):
        st.dataframe(final[["sku_code", "product_description", "approved_hs_code"]])

# --- Review Mode ---
if st.checkbox("🧾 Launch Manual Review UI (Phase 4)"):
    try:
        enriched_df = pd.read_csv(ENRICHED_CSV)
        run_phase_4_ui(enriched_df)
    except Exception as e:
        st.error(f"❌ Error loading data for review: {e}")
