import json
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.coindesk.com"

def get_article_links(load_clicks=5):
    links = []
    filtered_links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=100)  # Visual + debugging
        page = browser.new_page()

        try:
            print(f"🌐 Navigating to {BASE_URL}/markets ...")
            response = page.goto(BASE_URL + "/markets", timeout=90000)
            if not response or not response.ok:
                raise Exception("Page failed to load.")

            print("⏳ Waiting for articles to appear ...")
            page.wait_for_timeout(6000)  # Let JavaScript content fully load

            for i in range(load_clicks):
                try:
                    load_more = page.locator('button:has-text("Load More")')
                    if load_more.is_visible():
                        load_more.click()
                        print(f"🔁 Clicked 'Load More' ({i+1}/{load_clicks})")
                        time.sleep(3)
                    else:
                        print("⚠️ No 'Load More' button found.")
                        break
                except Exception as e:
                    print(f"⚠️ Load more failed: {e}")
                    break

            print("🔎 Collecting article links ...")
            article_links = page.eval_on_selector_all(
                'a[href^="/markets/"]',
                'elements => elements.map(e => e.href)'
            )

            for url in article_links:
                if "/markets/" in url and url.count("/") > 4 and url not in filtered_links:
                    filtered_links.append(url)

            print(f"✅ Found {len(filtered_links)} article links.")

        except Exception as e:
            print(f"❌ Error during scraping: {e}")

        finally:
            browser.close()

    return filtered_links

def get_article_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)  # Set to True when stable
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=60000)

            # Wait for title to ensure page is fully loaded
            page.wait_for_selector("h1", timeout=10000)
            title = page.locator("h1").inner_text()

            # First try known structure
            paragraphs = page.locator('div[class*="at-content-wrapper"] p').all_inner_texts()
            if not paragraphs:
                # Fallback: try any main content
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
    with open("coindesk_markets_articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
