import pandas as pd
import os
import re

# ========= File Paths =========
input_file = r"C:\Data\Job\UK\Zencargo\llm_enriched_output.csv"
output_file = r"C:\Data\Job\UK\Zencargo\phase3_final_enriched_output.csv"
fallback_log = r"C:\Data\Job\UK\Zencargo\fallback_rule_applied.csv"

# ========= Load Data =========
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ LLM output not found: {input_file}")

df = pd.read_csv(input_file)
df["fallback_hs_code"] = ""
df["final_hs_code"] = ""

# ========= Define Rule-Based Mapping =========
keyword_to_hs = {
    "shirt": "6205200000",
    "bottle": "3923309000",
    "fork": "8215991000",
    "wallet": "4202310000",
    "toy": "9503007000",
    "chair": "9401610000",
    "smartphone": "8517120000",
    "phone": "8517120000",
    "headphones": "8518300000",
    "computer": "8471300000",
    "notebook": "8471300000",
    "cooking pot": "7615100000",
    "leather": "4202310000"
}

def apply_rule_based_fallback(description: str) -> str:
    desc_lower = description.lower()
    for keyword, hs in keyword_to_hs.items():
        if keyword in desc_lower:
            return hs
    return "❌ No Match"

# ========= Apply Rules to Invalid Rows =========
fallback_rows = df[df["parsed_hs_code"] == "❌ Invalid"].copy()
fallback_rows["fallback_hs_code"] = fallback_rows["product_description"].apply(apply_rule_based_fallback)

# ========= Merge Final HS Code =========
for idx, row in df.iterrows():
    if row["parsed_hs_code"] != "❌ Invalid":
        df.at[idx, "final_hs_code"] = row["parsed_hs_code"]
    else:
        df.at[idx, "final_hs_code"] = fallback_rows.loc[row.name, "fallback_hs_code"]

# ========= Save Outputs =========
df.to_csv(output_file, index=False)
fallback_rows.to_csv(fallback_log, index=False)

print(f"✅ Final file saved to: {output_file}")
print(f"📄 Fallback log saved to: {fallback_log}")
print(f"🧠 {len(fallback_rows[fallback_rows['fallback_hs_code'] != '❌ No Match'])} rows enriched via rules.")
