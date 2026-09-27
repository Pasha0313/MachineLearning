import json
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = re.sub(r'\[[^\]]*\]', '', text)  # remove bracket content
    return text.strip()

def preprocess_and_flatten(input_files, output_txt_file):
    with open(output_txt_file, 'w', encoding='utf-8') as out:
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            for article in articles:
                cleaned_content = clean_text(article.get('content', ''))
                if cleaned_content:
                    out.write(cleaned_content + '\n')  # one article per line

if __name__ == "__main__":
    input_files = [
        "cointelegraph_articles.json",
        "coindesk_markets_articles.json",
        "decrypt_articles.json",
        "cryptoslate_articles.json"
    ]
    preprocess_and_flatten(input_files, "cleaned_all_articles.txt")
