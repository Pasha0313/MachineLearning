import json
import re
import glob

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = re.sub(r'\[[^\]]*\]', '', text)  # remove bracket content
    return text.strip()

def preprocess_and_merge(input_files, output_file):
    all_articles = []
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        for article in articles:
            cleaned_content = clean_text(article['content'])
            all_articles.append({
                'title': article['title'],
                'content': cleaned_content,
                'url': article['url']
            })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    input_files = [
    "cointelegraph_articles.json",
    "coindesk_markets_articles.json",
    "decrypt_articles.json",
    "cryptoslate_articles.json"
    ]
    preprocess_and_merge(input_files, "cleaned_all_articles.json")
