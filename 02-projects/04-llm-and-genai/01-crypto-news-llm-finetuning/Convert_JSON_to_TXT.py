import json

def convert_to_txt(json_path, txt_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    with open(txt_path, 'w', encoding='utf-8') as f:
        for article in articles:
            title = article.get('title', '').strip()
            content = article.get('content', '').strip()
            if title and content:
                f.write(f"{title}\n{content}\n\n")

if __name__ == "__main__":
    convert_to_txt("cleaned_all_articles.json", "cleaned_all_articles.txt")
