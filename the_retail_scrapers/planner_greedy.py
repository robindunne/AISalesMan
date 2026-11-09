# planner_greedy.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from vibe_encoder import vibe_similarity


@dataclass
class BundleItem:
    slot_name: str
    product_id: str
    product: Dict[str, Any]
    price_cents: int
    area_sqft: float


@dataclass
class BundleResult:
    room_type: str
    room_index: int
    area_sqft: float
    budget_cents: Optional[int]

    slots_requested: Dict[str, int]
    slots_filled: Dict[str, int]
    slots_unfilled: Dict[str, int]

    chosen_items: List[BundleItem]
    total_cost_cents: int
    total_area_sqft: float
    max_area_sqft: Optional[float]


# Map slot names to coarse primary_category in products
SLOT_TO_CATEGORY: Dict[str, str] = {
    # bedroom
    "bed": "bed",
    "dresser": "storage",
    "desk_or_vanity": "desk",
    "lamp_standing": "lamp",
    "lamp_small": "lamp",
    "nightstand": "table",
    "accent_chair": "chair",
    "mirror": "mirror",

    # living room
    "sofa": "sofa",
    "coffee_table": "table",
    "media_unit": "storage",
    "accent_chair_big": "chair",
    "side_table": "table",
    "rug": "rug",
    "storage": "storage",

    # bathroom
    "sconce": "lamp",
    "storage_small": "storage",
    "bath_mat": "rug",
    "hamper": "laundry",

    # kitchen
    "dining_table": "table",
    "dining_chair": "chair",
    "bar_stool": "chair",
    "sideboard": "storage",
    "kitchen_rug": "rug",

    # office
    "office_chair": "chair",
}


# Area ranges by slot so we do not pick a 2 sqft "dining table" or similar
# Units are square feet, based on attributes["area_sqft"]
SLOT_AREA_RANGE: Dict[str, tuple] = {
    "dining_table": (8.0, 40.0),
    "coffee_table": (4.0, 20.0),
    "side_table": (1.0, 10.0),
    "nightstand": (1.0, 10.0),
    "sofa": (15.0, 60.0),
    "sofa": (5.0, 80.0),           # allow smaller / mis-estimated sofas
    "accent_chair": (5.0, 25.0),
    "accent_chair_big": (8.0, 30.0),
    "office_chair": (5.0, 20.0),
    "bar_stool": (3.0, 15.0),
    "lamp_standing": (1.0, 6.0),
    "lamp_small": (0.5, 3.0),
    # relax lamp area so missing/approximate area_sqft does not kill them
    "lamp_standing": (0.1, 30.0),
    "lamp_small": (0.1, 30.0),
    "rug": (10.0, 80.0),
    "kitchen_rug": (6.0, 40.0),
    "bath_mat": (3.0, 10.0),
}



# Minimum prices by slot, in cents.
# This filters out tiny cheap items that are not realistic for that slot.
MIN_PRICE_BY_SLOT: Dict[str, int] = {
    "sofa": 10000,           # 100 dollars
    "bed": 8000,             # 80 dollars
    "dining_table": 5000,    # 50 dollars
    "coffee_table": 2000,    # 20 dollars
    "side_table": 1500,      # 15 dollars
    "nightstand": 1500,
    "media_unit": 3000,
    "storage": 2000,
    "storage_small": 2000,
    "hamper": 2000,
    "office_chair": 3000,
    "accent_chair": 2000,
    "accent_chair_big": 4000,
    "bar_stool": 3000,
    "dining_chair": 2000,
    "rug": 1500,
    "kitchen_rug": 1500,
    "bath_mat": 500,
    "lamp_standing": 0,
    "lamp_small": 0,
    "sconce": 500,
}


def get_price_cents(row: Dict[str, Any]) -> Optional[int]:
    """
    Normalize price to cents.
    Supports either price_cents or price (numeric dollars).
    """
    if "price_cents" in row and row["price_cents"] is not None:
        return int(row["price_cents"])
    if "price" in row and row["price"] is not None:
        try:
            return int(round(float(row["price"]) * 100))
        except (TypeError, ValueError):
            return None
    return None

def _name_and_secondary(row: Dict[str, Any]) -> tuple[str, str]:
    type_line = (
        row.get("raw_type_line")
        or row.get("raw_name")
        or row.get("name")
        or ""
    ).lower()
    sec = (row.get("secondary_category") or "").lower()
    return type_line, sec

def get_area_sqft_from_row(row: Dict[str, Any]) -> float:
    """
    Pull area_sqft from attributes jsonb if present.
    Otherwise default to a small but nonzero footprint so we do not undercount.
    """
    attrs = row.get("attributes") or {}
    area = attrs.get("area_sqft")
    if isinstance(area, (int, float)):
        return float(area)
    return 10.0


def get_color_from_row(row: Dict[str, Any]) -> Optional[str]:
    attrs = row.get("attributes") or {}
    color = attrs.get("color")
    if not isinstance(color, str):
        return None
    # take first word as a rough hue, lowercased
    return color.split()[0].lower()


def is_candidate_valid_for_slot(
    slot_name: str,
    row: Dict[str, Any],
    price_cents: int,
    area_sqft: float,
) -> bool:
    """
    Slot specific filtering using secondary_category, raw_type_line, price, and area.
    This is where we prevent:
      - dish mats as rugs
      - tealights as sconces
      - tiny side tables as dining tables
    """
   # min_price = MIN_PRICE_BY_SLOT.get(slot_name, 0)
    #if price_cents < min_price:
     #   return False

    # Area constraints if we have a range
    if slot_name in SLOT_AREA_RANGE:
        min_a, max_a = SLOT_AREA_RANGE[slot_name]
        if not (min_a <= area_sqft <= max_a):
            return False

    sec = (row.get("secondary_category") or "").lower()
    type_line = (
        row.get("raw_type_line")
        or row.get("raw_name")
        or row.get("name")
        or ""
    ).lower()

    # Rugs: avoid bath mats and dish drying mats unless slot explicitly wants them
    if slot_name in {"rug", "kitchen_rug"}:
        if any(bad in type_line for bad in ["dish", "drying", "bath"]) or any(
            bad in sec for bad in ["dish", "drying", "bath"]
        ):
            return False

    # Bathroom mat: rugs that look like mats
    if slot_name == "bath_mat":
        type_line, sec = _name_and_secondary(row)

        # Strong match: secondary category mentions mat
        if "bath mat" in sec or "shower mat" in sec or "mat" in sec:
            return True

        # Fallback: name/type mentions bath/shower mat
        if any(kw in type_line for kw in ["bath mat", "bathmat", "shower mat"]):
            return True

        # Very loose fallback: any small rug is okay
        return True  # small footprint rug

    # Hamper: storage that looks like laundry/basket
    if slot_name == "hamper":
        type_line, sec = _name_and_secondary(row)

        # Strong match: laundry-related
        if any(
            kw in sec or kw in type_line
            for kw in ["laundry hamper", "laundry basket", "laundry bag", "hamper"]
        ):
            return True

        # Fallback: generic basket-like storage
        if any(kw in sec or kw in type_line for kw in ["basket", "bin", "bag"]):
            return True

        # Otherwise don’t use it as a hamper
        return False

    # Standing lamp: prefer floor lamps, reject desk or clip lamps
    if slot_name == "lamp_standing":
        has_floor_hint = any(
            kw in type_line or kw in sec
            for kw in ["floor lamp", "floor uplight", "floor", "standing lamp"]
        )
        if not has_floor_hint:
            return False
    # Handle lamp-related slots using secondary_category
    if slot_name in {"lamp_standing", "lamp_small", "sconce"}:
        type_line, sec = _name_and_secondary(row)

        # never treat tealights as lamps/sconces
        if "tealight" in type_line or "tealight" in sec:
            return False

        # Sconce: wall-mounted lamps only
        if slot_name == "sconce":
            return any(
                kw in sec or kw in type_line
                for kw in ["wall lamp", "wall/reading lamp", "sconce"]
            )

        # Standing lamp: floor-style lamps, not table/desk/bedside
        if slot_name == "lamp_standing":
            # reject clear table/desk lamps
            if any(
                kw in sec or kw in type_line
                for kw in ["table lamp", "work lamp", "desk lamp", "bedside lamp"]
            ):
                return False

            return any(
                kw in sec or kw in type_line
                for kw in ["floor lamp", "floor/reading lamp", "uplight"]
            )

        # Small lamp: table / desk / bedside lamps, not floor
        if slot_name == "lamp_small":
            # reject floor-style lamps
            if any(
                kw in sec or kw in type_line
                for kw in ["floor lamp", "floor/reading lamp", "uplight"]
            ):
                return False

            # accept table/desk/bedside/reading lamps
            return any(
                kw in sec or kw in type_line
                for kw in [
                    "table lamp",
                    "desk lamp",
                    "work lamp",
                    "bedside lamp",
                    "reading lamp",
                ]
            )
    # Small lamp: table or desk lamps, no floor lamps
    if slot_name == "lamp_small":
        if any(kw in type_line or kw in sec for kw in ["floor lamp", "floor", "standing"]):
            return False
        # If we have hints of table or desk, that is ideal,
        # but we do not require it so we do not over filter.
        # You can tighten this later if your secondary_category is richer.

    # Sconce: wall mounted lights only, no tealights
    if slot_name == "sconce":
        if "tealight" in type_line or "tealight" in sec:
            return False
        is_wall_light = any(
            kw in type_line or kw in sec
            for kw in ["wall lamp", "wall light", "sconce"]
        )
        if not is_wall_light:
            return False

    # Dining table: avoid side tables or coffee tables
    if slot_name == "dining_table":
        if not any(
            kw in type_line or kw in sec
            for kw in ["dining table", "extendable dining", "drop leaf table"]
        ):
            return False
        if any(kw in type_line for kw in ["side table", "coffee table"]):
            return False

    # Coffee table: should mention coffee table
    if slot_name == "coffee_table":
        if "coffee table" not in type_line and "coffee" not in sec:
            return False

    # Side table and nightstand
    if slot_name in {"side_table", "nightstand"}:
        if not any(
            kw in type_line
            for kw in ["side table", "nightstand", "bedside table"]
        ):
            return False

    # Office chair: avoid obvious dining chairs if we can
    if slot_name == "office_chair":
        if not any(
            kw in type_line or kw in sec
            for kw in ["office chair", "swivel chair", "desk chair", "work chair"]
        ):
            return False

    # Bar stool: needs to be some kind of stool
    if slot_name == "bar_stool":
        if "stool" not in type_line and "stool" not in sec:
            return False

    return True


def fetch_candidates_for_slot(
    supabase,
    slot_name: str,
    limit: int = 80,
) -> List[Dict[str, Any]]:
    """
    Fetch product candidates for a given slot based on primary_category.

    Step 1: apply strict filters (type_line / secondary_category + price + area).
    Step 2: if that yields nothing, fall back to a looser filter:
        - still enforce min price for the slot
        - still enforce area range for the slot (if defined)
        - but skip subtype heuristics so we at least fill the slot.
    """
    category = SLOT_TO_CATEGORY.get(slot_name)
    if not category:
        return []

    query = (
        supabase.table("products")
        .select("*")
        .eq("primary_category", category)
        .limit(limit)
    )

    try:
        query = query.order("price_cents", desc=False)
    except Exception:
        try:
            query = query.order("price", desc=False)
        except Exception:
            pass

    resp = query.execute()
    data = resp.data or []

    strict_filtered: List[Dict[str, Any]] = []
    loose_pool: List[Dict[str, Any]] = []

    # Base constraints shared by strict and loose:
    min_price_for_slot = MIN_PRICE_BY_SLOT.get(slot_name, 0)
    has_area_range = slot_name in SLOT_AREA_RANGE
    if has_area_range:
        min_a, max_a = SLOT_AREA_RANGE[slot_name]
    else:
        min_a, max_a = (0.0, float("inf"))

    for row in data:
        price = get_price_cents(row)
        if price is None or price < min_price_for_slot:
            continue

        area = get_area_sqft_from_row(row)
        if not (min_a <= area <= max_a):
            continue

        # Everything that passes base constraints is in the loose pool
        loose_pool.append(row)

        # Only some pass strict subtype heuristics
        if is_candidate_valid_for_slot(slot_name, row, price, area):
            strict_filtered.append(row)

    # Prefer strict subset when available
    if strict_filtered:
        return strict_filtered

    # Fallback: still reasonable (category + min price + area),
    # but not over-filtered. Better to fill than lie about type.
    return loose_pool


def candidate_sort_key(
    row: Dict[str, Any],
    slot_name: str,
    room_plan: Dict[str, Any],
) -> int:
    """
    Primary key is price, but we shift it down (bonus) when:
      - persona_tags match the room persona
      - style_tags overlap room target_style_tags
      - color matches any of the room target colors
      - product description has high transformer vibe similarity to room vibe_text
    Lower score is better.
    """
    price = get_price_cents(row)
    if price is None:
        price = 10**9

    score = price

    # 1) Persona bonus
    persona = room_plan.get("persona")
    if persona:
        tags = row.get("persona_tags") or []
        if isinstance(tags, str):
            tags = tags.split(",") if tags else []
        if isinstance(tags, list) and persona in tags:
            # treat persona match like making item cheaper
            score -= 2000  # about 20 dollars

    # 2) Style tag overlap bonus (optional but nice if you pass target_style_tags)
    target_style_tags = room_plan.get("target_style_tags") or []
    if isinstance(target_style_tags, str):
        target_style_tags = target_style_tags.split(",") if target_style_tags else []
    if isinstance(target_style_tags, list) and target_style_tags:
        prod_style_tags = row.get("style_tags") or []
        if isinstance(prod_style_tags, str):
            prod_style_tags = prod_style_tags.split(",") if prod_style_tags else []
        overlap = len(set(prod_style_tags) & set(target_style_tags))
        if overlap > 0:
            # each overlapping style tag gives a bonus
            score -= 500 * overlap

    # 3) Color bonus
    target_colors = room_plan.get("target_colors") or []
    if isinstance(target_colors, list) and target_colors:
        target_colors_set = {c.lower() for c in target_colors}
        color = get_color_from_row(row)
        if color and any(c in color for c in target_colors_set):
            score -= 500  # small nudge

    # 4) Transformer vibe similarity bonus
    vibe_text = room_plan.get("vibe_text")
    if vibe_text:
        prod_text = f"{row.get('raw_name', '')} {row.get('raw_description', '')}"
        sim = vibe_similarity(vibe_text, prod_text)  # around 0 to 1
        # higher sim -> lower score (better)
        score -= int(2000 * sim)

    return score


def plan_room_greedy(
    supabase,
    room_plan: Dict[str, Any],
    max_area_fraction: float = 0.7,
) -> BundleResult:
    """
    Simple greedy planner for a single room.

    Strategy:
      - Fill required slots first, then optional slots.
      - For each slot, pick the best candidate that:
          - has not been used yet
          - keeps us under budget_cents (if provided)
          - keeps us under max_area_fraction * room_area_sqft
        "Best" is:
          - lowest effective price
          - but with small bonuses for persona and color matches.
    """
    room_type: str = room_plan["room_type"]
    room_index: int = room_plan["room_index"]
    room_area_sqft: float = float(room_plan.get("area_sqft", 200.0))      
    budget_cents: Optional[int] = room_plan.get("budget_cents")
    slots: Dict[str, int] = dict(room_plan["slots_requested"])
    meta: Dict[str, Dict[str, Any]] = room_plan.get("meta", {})

    max_area_sqft: Optional[float] = None
    remaining_budget: Optional[int] = budget_cents
    remaining_area: Optional[float] = None

    chosen_items: List[BundleItem] = []
    used_product_ids: set = set()
    slots_filled: Dict[str, int] = {k: 0 for k in slots.keys()}

    # Cache candidates per slot so we do not hit Supabase repeatedly
    candidate_cache: Dict[str, List[Dict[str, Any]]] = {}

    def get_candidates(slot: str) -> List[Dict[str, Any]]:
        if slot not in candidate_cache:
            candidate_cache[slot] = fetch_candidates_for_slot(supabase, slot)
        return candidate_cache[slot]

    # Fill required slots first, then optional
    for required_flag in (True, False):
        for slot_name, needed_count in slots.items():
            slot_meta = meta.get(slot_name, {})
            is_required = bool(slot_meta.get("required", False))
            if is_required != required_flag:
                continue

            if needed_count <= 0:
                continue

            if slot_name not in SLOT_TO_CATEGORY:
                # We do not yet know how to map this slot into categories
                continue

            candidates = get_candidates(slot_name)
            if not candidates:
                continue

            candidates_sorted = sorted(
                candidates,
                key=lambda row: candidate_sort_key(row, slot_name, room_plan),
            )

            while slots_filled[slot_name] < needed_count:
                picked = None

                for row in candidates_sorted:
                    pid = row.get("id")
                    if pid in used_product_ids:
                        continue

                    price = get_price_cents(row)
                    if price is None:
                        continue

                    area = get_area_sqft_from_row(row)

                    # Check budget constraint
                    if remaining_budget is not None and price > remaining_budget:
                        continue

                    # Check area constraint
                    #if remaining_area is not None and area > remaining_area:
                    #    continue

                    picked = (row, pid, price, area)
                    break

                if picked is None:
                    # Could not fit any candidate under constraints
                    break

                row, pid, price, area = picked
                used_product_ids.add(pid)
                slots_filled[slot_name] += 1

                chosen_items.append(
                    BundleItem(
                        slot_name=slot_name,
                        product_id=str(pid),
                        product=row,
                        price_cents=price,
                        area_sqft=area,
                    )
                )

                if remaining_budget is not None:
                    remaining_budget -= price
                if remaining_area is not None:
                    remaining_area -= area

    # Compute unfilled counts
    slots_unfilled: Dict[str, int] = {}
    for slot_name, needed in slots.items():
        filled = slots_filled.get(slot_name, 0)
        remaining = needed - filled
        if remaining > 0:
            slots_unfilled[slot_name] = remaining

    total_cost_cents = sum(item.price_cents for item in chosen_items)
    total_area_sqft = sum(item.area_sqft for item in chosen_items)

    return BundleResult(
        room_type=room_type,
        room_index=room_index,
        area_sqft=room_area_sqft,
        budget_cents=budget_cents,
        slots_requested=slots,
        slots_filled=slots_filled,
        slots_unfilled=slots_unfilled,
        chosen_items=chosen_items,
        total_cost_cents=total_cost_cents,
        total_area_sqft=total_area_sqft,
        max_area_sqft=max_area_sqft,
    )
