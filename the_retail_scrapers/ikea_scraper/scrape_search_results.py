from playwright.async_api import Page, TimeoutError

CARD_SELECTOR = "div[data-testid='plp-product-card']"
LOAD_MORE_SELECTOR = "button.plp-pagination__button"


async def scrape_search_results(page: Page, query: str) -> list:
    url = f"https://www.ikea.com/us/en/search/?q={query.replace(' ', '%20')}"
    await page.goto(url, timeout=60000)

    # 1) Load at least the first batch of cards
    try:
        await page.wait_for_selector(CARD_SELECTOR, timeout=30000)
    except TimeoutError:
        print(f"No results found for '{query}' (no cards matched {CARD_SELECTOR})")
        return []

    # 2) Keep clicking "Show more" until it disappears
    print("Loading all result pages...")
    while True:
        try:
            btn = await page.wait_for_selector(LOAD_MORE_SELECTOR, timeout=2000)
        except TimeoutError:
            # No button found within timeout → no more pages
            break

        if not btn:
            break

        try:
            await btn.scroll_into_view_if_needed()
            await btn.click()
            # give IKEA time to load the next batch
            await page.wait_for_timeout(1500)
        except Exception as e:
            print("Error clicking Show more:", e)
            break

    # 3) Now grab ALL product cards that have been loaded
    items = await page.query_selector_all(CARD_SELECTOR)
    print("DEBUG: found", len(items), "items on page (after Show more)")

    products = []
    for item in items:
        try:
            link = await item.query_selector("a.plp-product__image-link")
            image = await item.query_selector("img.plp-product__image")
            name = await image.get_attribute("alt") if image else None

            if not (name and image and link):
                continue

            img_src = await image.get_attribute("src")
            href = await link.get_attribute("href")
            if href and not href.startswith("http"):
                href = f"https://www.ikea.com{href}"

            products.append({
                "name": name.strip(),
                "image": img_src,
                "url": href,
                "gallery_images": [],
                "price": "N/A",
                "description": "N/A",
                "dimensions": {},
                "rating": "N/A",
                "rating_count": "0",
                "color": "unknown"
            })
        except Exception:
            continue

    print(f"Found {len(products)} products")
    return products

