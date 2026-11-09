# planner_cli.py

from typing import Dict, List, Optional, Any

from planner_config import SlotProposal, propose_slots


def ask_int(prompt: str, default: int = 0) -> int:
    """
    Ask the user for an integer value with a default.
    Hitting enter returns the default.
    """
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number.")


def collect_room_counts() -> Dict[str, int]:
    """
    Ask the user how many of each room type they have.
    Returns a dict like:
        {"bedroom": 2, "living_room": 1, ...}
    """
    print("Tell me about your home layout.")
    bedrooms = ask_int("How many bedrooms do you have", default=1)
    living_rooms = ask_int("How many living rooms do you have", default=1)
    offices = ask_int("How many offices or workspaces do you have", default=0)
    bathrooms = ask_int("How many bathrooms do you have", default=1)
    kitchens = ask_int("How many kitchens do you have", default=1)

    layout = {
        "bedroom": bedrooms,
        "living_room": living_rooms,
        "office": offices,
        "bathroom": bathrooms,
        "kitchen": kitchens,
    }

    # Strip any zero counts so we do not iterate over empty room types later.
    layout = {k: v for k, v in layout.items() if v > 0}

    print("\nHome layout:")
    for room_type, count in layout.items():
        print(f"  - {room_type}: {count}")

    print()
    return layout


def infer_size_tier(layout: Dict[str, int]) -> str:
    """
    Infer a rough size tier for the whole home based on total room count.
    More rooms means larger assumed square footage.
    """
    total_rooms = sum(layout.values())
    if total_rooms <= 3:
        return "small"
    if total_rooms <= 6:
        return "medium"
    return "large"


def infer_room_area_sqft(layout: Dict[str, int], room_type: str) -> int:
    """
    Infer an approximate room area in square feet based on:
      - overall size tier from total room count
      - room type

    We bias toward slightly larger rooms so that more soft slots are included.
    Users can always remove items later.
    """
    tier = infer_size_tier(layout)

    base_areas = {
        "small": {
            "bedroom": 130,
            "living_room": 180,
            "office": 110,
            "bathroom": 45,
            "kitchen": 140,
        },
        "medium": {
            "bedroom": 160,
            "living_room": 220,
            "office": 130,
            "bathroom": 55,
            "kitchen": 170,
        },
        "large": {
            "bedroom": 190,
            "living_room": 260,
            "office": 150,
            "bathroom": 65,
            "kitchen": 200,
        },
    }

    tier_map = base_areas.get(tier, base_areas["medium"])
    # Default fallback if a room type is missing.
    return tier_map.get(room_type, 150)


def ask_persona(room_type: str) -> str:
    """
    Ask for a persona string for this room.
    """
    print(
        "Personas could be things like: "
        "elderly_man, young_bachelor, family, kid_room, default."
    )
    raw = input(
        f"Persona for this {room_type} "
        "(press enter for default): "
    ).strip()
    return raw or "default"


def ask_room_budget_cents(room_type: str, index: int) -> Optional[int]:
    """
    Ask the user for a budget for this specific room, in dollars.
    Returns budget in cents, or None if zero or not set.
    """
    prompt = (
        f"Budget for {room_type} #{index} in dollars "
        "(0 means no specific per room budget)"
    )
    dollars = ask_int(prompt, default=0)
    if dollars <= 0:
        return None
    return dollars * 100


def collect_room_details(layout: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    For each room instance, ask for:
      - persona
      - family_size (for kitchens)
      - optional budget per room

    Room size in square feet is inferred automatically from the overall layout,
    and then passed into propose_slots. We err on the side of assuming larger
    rooms so that more items are proposed. Users can always remove items later.

    Returns a list of room plan dicts, one per room instance.
    """
    room_plans: List[Dict[str, Any]] = []

    print("Now we will configure each room.\n")
    home_size_tier = infer_size_tier(layout)
    print(f"Inferred home size tier: {home_size_tier}\n")

    for room_type, count in layout.items():
        for idx in range(1, count + 1):
            print("=" * 60)
            print(f"Configure {room_type} #{idx}")
            print("=" * 60)

            # Infer room area instead of asking the user.
            area_sqft = infer_room_area_sqft(layout, room_type)
            print(f"(Assuming {area_sqft} sqft for this {room_type} based on home size.)")

            # Persona
            persona = ask_persona(room_type)

            # Family size only matters for kitchens in the current template logic
            family_size: Optional[int] = None
            if room_type == "kitchen":
                family_size = ask_int(
                    "How many people will regularly eat in this kitchen "
                    "(family size)",
                    default=2,
                )
            # Optional per room budget
            budget_cents = ask_room_budget_cents(room_type, idx)

            # 🔥 NEW: free-form vibe text for this room
            vibe_text = input(
                "Describe the vibe / mood / style for this room "
                "(e.g. 'small calm cozy study space with warm lighting but not cluttered'):\n> "
            ).strip()

            # 🔥 NEW: optional explicit style tags (can be empty, transformer still works)
            style_keywords = input(
                "Optional: style keywords (neutral, modern, cozy, boho, etc.), "
                "comma-separated. Leave blank to skip:\n> "
            ).strip()
            if style_keywords:
                target_style_tags = [t.strip() for t in style_keywords.split(",") if t.strip()]
            else:
                target_style_tags = []

            # Call the slot proposal logic
            proposal: SlotProposal = propose_slots(
                room_type=room_type,
                room_area_sqft=area_sqft,
                persona=persona,
                family_size=family_size,
            )

            # Print proposed slots for this room
            print("\nProposed slots (we lean toward more items, you can remove later):")
            for slot_name, count_val in proposal.slots.items():
                meta = proposal.meta.get(slot_name, {})
                required = meta.get("required", False)
                required_label = "required" if required else "optional"
                print(f"  - {slot_name}: {count_val} ({required_label})")

            print()

            room_plan = {
                "room_type": room_type,
                "index": idx,
                "area_sqft": area_sqft,
                "persona": persona,
                "family_size": family_size,
                "budget_cents": budget_cents,
                "slots": proposal.slots,
                "meta": {
                    k: dict(v) if not isinstance(v, dict) else v
                    for k, v in proposal.meta.items()
                },
                "vibe_text": vibe_text,
                "target_style_tags": target_style_tags,
            }
            room_plans.append(room_plan)


    print("\nSummary of proposed slots per room:\n")
    for plan in room_plans:
        rt = plan["room_type"]
        idx = plan["index"]
        print(
            f"{rt} #{idx} "
            f"(assumed area {plan['area_sqft']} sqft, persona {plan['persona']}):"
        )
        for slot_name, count_val in plan["slots"].items():
            meta = plan["meta"].get(slot_name, {})
            required = meta.get("required", False)
            required_label = "required" if required else "optional"
            print(f"  - {slot_name}: {count_val} ({required_label})")
        print()

    return room_plans


def main() -> None:
    print("AI Home Planner CLI")
    print("===================")
    print("We will collect your room layout and propose item slots.\n")

    layout = collect_room_counts()
    room_plans = collect_room_details(layout)

    # This is where the planner will hook in next:
    # We will pass room_plans into a bundle planner that talks to Supabase.
    print(
        "Next step will be to run a planner that picks real products from "
        "your Supabase products table to fill these slots under a budget."
    )
if __name__ == "__main__":
    main()
