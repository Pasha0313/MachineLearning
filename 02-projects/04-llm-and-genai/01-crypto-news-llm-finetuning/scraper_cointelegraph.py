import json
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://cointelegraph.com"

def get_article_links(load_clicks=5):
    links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(BASE_URL + "/tags/markets", timeout=60000)
            page.wait_for_selector('a[href^="/news/"]', timeout=10000)

            for i in range(load_clicks):
                try:
                    load_more = page.locator('button:has-text("Load more")')
                    if load_more.is_visible():
                        load_more.click()
                        print(f"🔁 Clicked 'Load more' ({i+1}/{load_clicks})")
                        time.sleep(2)  # allow time for new content
                    else:
                        print("⚠️ No more 'Load more' button found.")
                        break
                except Exception as e:
                    print(f"⚠️ Could not click 'Load more': {e}")
                    break

            article_links = page.query_selector_all('a[href^="/news/"]')

            for link in article_links:
                href = link.get_attribute("href")
                if href and href.startswith("/news/") and href.count("/") == 2:
                    full_url = BASE_URL + href
                    if full_url not in links:
                        links.append(full_url)

            print(f"✅ Found {len(links)} article links after clicking {load_clicks} times.")

        except Exception as e:
            print(f"❌ Error during scraping: {e}")
        browser.close()
    return links


def get_article_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=60000)
            page.wait_for_selector("article", timeout=10000)

            title = page.locator("h1").inner_text()
            paragraphs = page.locator("article p").all_inner_texts()
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
    with open("cointelegraph_articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
