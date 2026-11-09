import asyncio
from typing import List
semaphore = asyncio.Semaphore(9)  # Adjust this for performance tuning

async def scrape_product_pages(context, product: dict) -> dict:
    async with semaphore:
        try:
            page = await context.new_page()
            await page.goto(product['url'], timeout=30000)
            type_el = await page.query_selector("span.pip-price-module__description")
            if type_el:
                product["type_line"] = (await type_el.inner_text()).strip()

            imgs = await page.query_selector_all("img[src*='/products/']")
            gallery: List[str] = []
            for img in imgs:
                src = await img.get_attribute("src")
                if not src:
                    continue
                if src not in gallery:
                    gallery.append(src)
            if gallery:
                product["image"] = gallery[0]
                product["gallery_images"] = gallery
            else:
                # keep whatever was already there
                product.setdefault("gallery_images", [])

            # Price
            price_el = await page.query_selector("span.pip-price__integer")
            if price_el:
                product["price"] = (await price_el.inner_text()).strip()

            # Description
            desc_el = await page.query_selector("p.pip-product-summary__description")
            if desc_el:
                product["description"] = await desc_el.inner_text()

            # Dimensions
            dimensions = {}
            dim_items = await page.query_selector_all("ul.pip-product-dimensions__dimensions-container li")
            for item in dim_items:
                try:
                    label_el = await item.query_selector("span.pip-product-dimensions__measurement-name")
                    full_text = await item.inner_text()
                    if label_el:
                        label = await label_el.inner_text()
                        value = full_text.replace(label, "").strip()
                        dimensions[label.strip(": ")] = value
                except:
                    continue
            product["dimensions"] = dimensions if dimensions else "N/A"

            # Rating
            rating_el = await page.query_selector("span.pip-rating__stars")
            if rating_el:
                aria = await rating_el.get_attribute("aria-label")
                if aria and "Review:" in aria:
                    try:
                        product["rating"] = aria.split("Review: ")[1].split(" out of")[0].strip()
                    except:
                        product["rating"] = "N/A"

            # Rating count
            count_el = await page.query_selector("span.pip-rating__label")
            if count_el:
                text = await count_el.inner_text()
                product["rating_count"] = text.strip("()").strip()

            # Color detection
            name_lower = product["name"].lower()
            if "brown" in name_lower:
                product["color"] = "brown"
            elif "gray" in name_lower:
                product["color"] = "gray"
            elif "black" in name_lower:
                product["color"] = "black"
            elif "white" in name_lower:
                product["color"] = "white"
            else:
                product["color"] = "unknown"

            await page.close()
            return product

        except Exception as e:
            print(f"Error with {product['name']}: {e}")
            return product
    print("SCRAPING PRODUCT PAGE:", product.get("name"), product.get("url"))


