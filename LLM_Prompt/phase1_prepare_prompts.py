import pandas as pd
import re
import random
import os
import streamlit as st

# ========= Step 1: HS Code Validator =========
def validate_hs_code(code: str) -> bool:
    if pd.isna(code) or str(code).strip() == "":
        return False
    cleaned = re.sub(r"[\s\-.]", "", str(code))
    return cleaned.isdigit() and len(cleaned) == 10

# ========= Step 2: Load Excel File =========
file_path = r"C:\Data\Job\UK\Zencargo\Automating HS Code validation.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

xls = pd.ExcelFile(file_path)
df = xls.parse(xls.sheet_names[0])
print(f"✅ Loaded sheet: {xls.sheet_names[0]} with {len(df)} records")

# ========= Step 3: Validate Existing HS Codes =========
df["hs_code"] = df["hs_code"].astype(str)
df["hs_code_valid"] = df["hs_code"].apply(validate_hs_code)
df["needs_enrichment"] = ~df["hs_code_valid"]

# ========= Step 4: Ensure Description =========
mock_descriptions = [
    "Men’s cotton shirt", "Plastic bottle for food storage", "Wooden chair",
    "Smartphone with camera", "Bluetooth headphones", "Leather wallet",
    "Stainless steel fork", "Aluminum cooking pot", "Children’s toy set", "Notebook computer",
    "Wireless router", "Ceramic plate", "LED desk lamp", "Winter jacket", "Canvas backpack"
]

def clean_description(desc):
    if pd.isna(desc) or str(desc).strip() == "":
        return None
    return str(desc).replace("\n", " ").replace('"', "").strip()

if "product_description" not in df.columns or df["product_description"].isna().all():
    st.warning("⚠️ Missing or empty 'product_description' column. Using simulated descriptions.")
    df["product_description"] = [
        random.choice(mock_descriptions) for _ in range(len(df))
    ]
else:
    df["product_description"] = df["product_description"].apply(clean_description)
    df["product_description"].fillna("Generic consumer product", inplace=True)

# ========= Step 5: Filter & Deduplicate =========
rows_to_enrich = df[df["needs_enrichment"]].copy()

# Deduplicate by both SKU and description to avoid dropping variants
rows_to_enrich.drop_duplicates(subset=["sku_code", "product_description"], inplace=True)
rows_to_enrich.reset_index(drop=True, inplace=True)

# ========= Step 6: Simulated Metadata + Prompt Construction =========
# Simulate metadata based on SKU pattern (deterministic by last digit)
def simulate_metadata(sku):
    sku_str = str(sku)
    digits = [char for char in reversed(sku_str) if char.isdigit()]
    if digits:
        endings = int(digits[0])
    else:
        endings = random.randint(0, 9)  # fallback if no digit found

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
rows_to_enrich["product_description"], rows_to_enrich["product_material"], \
rows_to_enrich["product_use"], rows_to_enrich["product_dimensions"] = zip(
    *rows_to_enrich["sku_code"].apply(simulate_metadata)
)

# Pool of diverse examples
example_pool = [
    ("Plastic bottle for food storage (Material: Plastic; Use: Food storage; Dimensions: 20x10x10 cm)", "3923309000"),
    ("Men's cotton shirt (Material: Cotton; Use: Clothing; Dimensions: Size M)", "6205200000"),
    ("Bluetooth speaker (Material: Plastic and metal; Use: Electronics; Dimensions: 10x5x5 cm)", "8518210000"),
    ("Toy car (Material: Plastic; Use: Children's toy; Dimensions: 15x7x6 cm)", "9503007000"),
    ("Aluminum laptop stand (Material: Aluminum; Use: Office; Dimensions: 30x25x5 cm)", "7616999099"),
    ("Ceramic mug (Material: Ceramic; Use: Drinkware; Dimensions: 300ml)", "6912002310"),
]

def create_prompt(row):
    sku = row["sku_code"]
    description = row["product_description"]
    material = row["product_material"]
    use = row["product_use"]
    dimensions = row["product_dimensions"]

    # Construct product detail
    full_description = f"{sku} - {description} (Material: {material}; Use: {use}; Dimensions: {dimensions})"

    # Randomly pick two distinct examples from the pool
    examples = random.sample(example_pool, 2)
    example_str = "\n\n".join(
        f"Product: {desc}\nHS Code: {code}" for desc, code in examples
    )

    # Final instruction and prompt
    instruction = (
        "You are a customs classification assistant.\n\n"
        "Your task is to assign a valid 10-digit UK HS code (Harmonized System Code) for the given product.\n\n"
        "Respond with only the 10-digit numeric HS code. Do not include explanations or extra text.\n\n"
        f"Examples:\n{example_str}\n\n"
        f"Product: {full_description}\n"
        "HS Code:"
    )

    return f"<s>[INST] {instruction.strip()} [/INST]>"

# Apply prompt generation
rows_to_enrich["prompt"] = rows_to_enrich.apply(create_prompt, axis=1)

# Optional: preview first few prompts
for i in range(min(5, len(rows_to_enrich))):
    print(f"🧾 Prompt {i+1} for SKU {rows_to_enrich.iloc[i]['sku_code']}:\n{rows_to_enrich.iloc[i]['prompt']}\n")

# ========= Step 7: Save Outputs =========
output_dir = r"C:\Data\Job\UK\Zencargo"
os.makedirs(output_dir, exist_ok=True)

prompts_path = os.path.join(output_dir, "prompts_for_llm.csv")
report_path = os.path.join(output_dir, "invalid_hs_codes_report.csv")

rows_to_enrich[["sku_code", "product_description", "prompt"]].to_csv(prompts_path, index=False)
rows_to_enrich.to_csv(report_path, index=False)

print(f"✅ Saved {len(rows_to_enrich)} prompts to: {prompts_path}")
print(f"📄 Full report saved to: {report_path}")
