import json
import time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

BASE_URL = "https://cryptoslate.com"
NEWS_URL = f"{BASE_URL}/news/"

def get_article_links(load_clicks=5):
    links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print(f"🌐 Navigating to {NEWS_URL}")
            page.goto(NEWS_URL, timeout=60000)
            page.wait_for_timeout(3000)

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
                    print(f"⚠️ Could not click 'Load More': {e}")
                    break

            raw_links = page.eval_on_selector_all(
                'a[href^="https://cryptoslate.com/"]',
                'elements => elements.map(e => e.href)'
            )

            for url in set(raw_links):
                path = urlparse(url).path.strip("/")
                if (
                    len(path.split("/")) == 1 and
                    not any(x in path for x in [
                        "coins", "cryptos", "category", "people", "companies", "privacy", "terms",
                        "directory", "about", "contact", "glossary", "news/page", "tag", "author",
                        "press", "faq", "feed", "products", "reports", "project", "media",
                        "insights", "top-news", "market-reports", "alpha", "disclaimers", "advertising"
                    ])
                ):
                    links.append(url)

            print(f"✅ Found {len(links)} filtered article links.")

        except Exception as e:
            print(f"❌ Error during scraping: {e}")
        finally:
            browser.close()

    return links

def get_article_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)

            page.wait_for_selector("h1", timeout=10000)
            title = page.locator("h1").first.inner_text()

            paragraphs = page.locator("article p").all_inner_texts()
            content = "\n".join(paragraphs)

            try:
                pub_date = page.locator("time").first.get_attribute("datetime")
            except:
                pub_date = None

            browser.close()

            if not title.strip() or not content.strip():
                print(f"⚠️ Empty content at {url}")
                return None

            return {
                "title": title.strip(),
                "content": content.strip(),
                "url": url,
                "published": pub_date
            }

    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

def scrape_articles():
    links = get_article_links(load_clicks=5)
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
    with open("cryptoslate_articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
