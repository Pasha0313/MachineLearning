import json
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://decrypt.co"

def get_article_links(load_clicks=5):
    links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("🌐 Loading Decrypt News page...")
            page.goto(BASE_URL + "/news", timeout=60000)
            page.wait_for_timeout(5000)  # Allow JS to render

            for i in range(load_clicks):
                try:
                    load_more = page.locator('button:has-text("Load More")')
                    if load_more.is_visible():
                        load_more.click()
                        print(f"🔁 Clicked 'Load More' ({i+1}/{load_clicks})")
                        time.sleep(2)
                    else:
                        print("⚠️ No more 'Load More' button found.")
                        break
                except Exception as e:
                    print(f"⚠️ Load more error: {e}")
                    break

            # ✅ Match URLs like https://decrypt.co/229360/article-title
            article_links = page.eval_on_selector_all(
                'a[href^="/"]',
                '''elements => elements
                    .map(e => e.href)
                    .filter(href => /^https:\\/\\/decrypt\\.co\\/\\d+\\//.test(href))'''
            )

            links = list(set(article_links))  # remove duplicates
            print(f"✅ Found {len(links)} article links.")

        except Exception as e:
            print(f"❌ Error during scraping: {e}")
        finally:
            browser.close()

    return links


def get_article_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=60000)

            # Use a more targeted h1 or fallback
            title = page.locator("h1").first.inner_text(timeout=10000)

            # Try several possible article content containers
            paragraphs = page.locator('article p').all_inner_texts()
            if not paragraphs:
                paragraphs = page.locator('div[class*="article-body"] p').all_inner_texts()
            if not paragraphs:
                paragraphs = page.locator('main p').all_inner_texts()

            content = "\n".join(paragraphs)

            browser.close()

            if not title.strip() or not content.strip():
                print(f"⚠️ Empty content at {url}")
                return None

            return {
                "title": title.strip(),
                "content": content.strip(),
                "url": url
            }

    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None


def scrape_articles():
    links = get_article_links()
    articles = []
    for i, url in enumerate(links):
        print(f"🔎 [{i+1}/{len(links)}] Scraping: {url}")
        article = get_article_content(url)
        if article:
            articles.append(article)
        time.sleep(1)
    print(f"✅ Successfully scraped {len(articles)} articles.")
    return articles

if __name__ == "__main__":
    articles = scrape_articles()
    with open("decrypt_articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
