import asyncio
import json
import os
import re

from playwright.async_api import async_playwright
from ikea_scraper import scrape_search_results, scrape_product_pages

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def price_to_cents(price_str):
    if not price_str or price_str == "N/A":
        return None
    m = re.search(r"([0-9][0-9.,]*)", str(price_str))
    if not m:
        return None
    return int(float(m.group(1).replace(",", "")) * 100)

def normalize_for_db(p, query):
    price_cents = price_to_cents(p.get("price"))
    if not price_cents:
        return None

    raw_type_line = p.get("type_line") or ""
    parts = [s.strip() for s in raw_type_line.split(",") if s.strip()]
    type_part = parts[0] if parts else None           
    color_from_type = parts[1] if len(parts) > 1 else None
    url = p.get("url")
    external_id = url 

    
    if type_part:
        tl = type_part.lower()
        textile_words = [
            "slipcover",
            "cover",
            "cushion",
            "pad",
            "throw",
            "blanket",
            "quilt",
            "duvet",
        ]
        if any(w in tl for w in textile_words):
            return None

    primary_category = None
    secondary_category = None

    if type_part:
        words = type_part.lower().split()
        last = words[-1]

        if last in ["sofa", "loveseat", "sectional", "chaise"]:
            primary_category = "sofa"

        elif last in ["table", "nightstand", "console"]:
            primary_category = "table"

        elif last in ["desk", "workstation", "vanity"]:
            primary_category = "desk"

        elif last in ["lamp", "sconce", "pendant"]:
            primary_category = "lamp"

        elif last in ["chair", "armchair"]:
            primary_category = "chair"

        # bed-related – remember last = "frame" here for "bed frame"
        elif last in ["bed", "frame", "headboard"]:
            primary_category = "bed"

        # storage pieces: dressers, wardrobes, cabinets, shelves, etc.
        elif last in ["dresser", "sideboard", "cabinet", "wardrobe",
                    "bookcase", "bookshelf", "shelf", "shelving"]:
            primary_category = "storage"

        # media units / tv stands
        elif last in ["stand", "bench"] and "tv" in type_part.lower():
            primary_category = "media_unit"

        # ottoman / stools / benches that aren't tv benches
        elif last in ["ottoman", "stool", "bench"]:
            primary_category = "extra_seating"

        # rugs / mats
        elif last in ["rug", "mat"]:
            primary_category = "rug"

        # mirrors
        elif last in ["mirror"]:
            primary_category = "mirror"

        secondary_category = type_part  

    rating_raw = p.get("rating")
    try:
        rating = float(rating_raw) if rating_raw not in (None, "", "N/A") else None
    except ValueError:
        rating = None
    color_value = color_from_type or p.get("color")
    return {
        "domain": "home",
        "name": p.get("name"),
        "brand": "IKEA",
        "raw_name": p.get("name"),
        "raw_description": p.get("description"),
        "source": "ikea",
        "external_id": external_id,          
        "tag_version": 0,
        "primary_category": primary_category,
        "secondary_category": secondary_category or query,
        "price_cents": price_cents,
        "rating": rating,
        "discount_percent": None,
        "style_tags": [],
        "persona_tags": [],
        "use_tags": [],
        "general_tags": [],
        "attributes": {
            "thumbnail": p.get("image"),
            "gallery_images": p.get("gallery_images") or [],
            "dimensions": p.get("dimensions") or {},
            "color": color_value,
            "url": url,
            "raw_type_line": raw_type_line,
            "raw_rating_count": p.get("rating_count"),
        },
        "in_stock": True,
    }



async def main():
    query = input("What piece of furniture are you looking for? ").strip()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        products = await scrape_search_results(page, query)
        await page.close()

        if not products:
            print("No products found.")
            await browser.close()
            return

        tasks = [scrape_product_pages(context, p) for p in products]
        final_products = await asyncio.gather(*tasks)

        print(f"Got {len(final_products)} products, writing file...")
        with open("ikea_gallery_scrape.json", "w") as f:
            json.dump(final_products, f, indent=2)

        print(f"\n{len(final_products)} products saved to ikea_gallery_scrape.json")

        rows = []
        for p in final_products:
            row = normalize_for_db(p, query)
            if row:
                rows.append(row)

        # ----- NEW: dedupe by (source, external_id) -----
        if rows:
            unique = {}
            for row in rows:
                key = (row["source"], row["external_id"])
                unique[key] = row   

            deduped_rows = list(unique.values())
            num_original = len(rows)
            num_unique = len(deduped_rows)
            num_dupes = num_original - num_unique

            print(f"After dedupe: {num_unique} unique products "
                  f"(skipped {num_dupes} duplicate entries)")

            if deduped_rows:
                print(f"Inserting {len(deduped_rows)} products into Supabase (upsert)...")
                supabase.table("products").upsert(
                    deduped_rows,
                    on_conflict="source,external_id"
                ).execute()
                print("Supabase upsert complete.")
            else:
                print("No valid rows to insert after dedupe.")
        else:
            print("No valid rows to insert (missing/invalid prices).")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())


