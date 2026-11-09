#!/usr/bin/env python3
"""
planner_demo.py

Runs Greedy + A* planners with vibe embedding.
"""

from dotenv import load_dotenv
from supabase import create_client
from vibe_search_core import load_vibe_model, get_vibe_embedding
from planner_astar import plan_room_astar
from planner_greedy import plan_room_greedy  # your existing greedy
import numpy as np
import os
import time
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Room type schema (slots + hard/soft requirements)
# ---------------------------------------------------------------------------

DEFAULT_ROOM_SLOTS: Dict[str, Dict[str, Any]] = {
    "bedroom": {
        "slots": {
            "bed": 1,
            "dresser": 1,
            "desk_or_vanity": 1,
            "lamp_standing": 1,
            "lamp_small": 1,
            "nightstand": 1,
            "accent_chair": 1,
            "mirror": 1,
        },
        "meta": {
            "bed": {"required": True},
            "dresser": {"required": True},
            "desk_or_vanity": {"required": False},
            "lamp_standing": {"required": False},
            "lamp_small": {"required": False},
            "nightstand": {"required": True},
            "accent_chair": {"required": False},
            "mirror": {"required": False},
        },
    },

    "living_room": {
        "slots": {
            "sofa": 1,
            "coffee_table": 1,
            "media_unit": 1,
            "lamp_standing": 1,
            "lamp_small": 1,
            "accent_chair_big": 1,
            "side_table": 1,
            "rug": 1,
            "storage": 1,
            "mirror": 1,
        },
        "meta": {
            "sofa": {"required": True},
            "coffee_table": {"required": True},
            "media_unit": {"required": True},
            "lamp_standing": {"required": False},
            "lamp_small": {"required": False},
            "accent_chair_big": {"required": False},
            "side_table": {"required": False},
            "rug": {"required": False},
            "storage": {"required": False},
            "mirror": {"required": False},
        },
    },

    "bathroom": {
        "slots": {
            "mirror": 1,
            "sconce": 2,
            "storage_small": 1,
            "bath_mat": 1,
            "hamper": 1,
        },
        "meta": {
            "mirror": {"required": True},
            "sconce": {"required": True},
            "storage_small": {"required": False},
            "bath_mat": {"required": False},
            "hamper": {"required": False},
        },
    },

    "kitchen": {
        "slots": {
            "dining_table": 1,
            "dining_chair": 4,   # or 6 for bigger kitchens
            "bar_stool": 4,      # if there is an island, else 0
            "sideboard": 1,
            "kitchen_rug": 1,
        },
        "meta": {
            "dining_table": {"required": True},
            "dining_chair": {"required": True},
            "bar_stool": {"required": False},
            "sideboard": {"required": False},
            "kitchen_rug": {"required": False},
        },
    },

    "office": {
        "slots": {
            "desk_or_vanity": 1,
            "office_chair": 1,
            "lamp_small": 1,
            "storage": 1,
            "lamp_standing": 1,
            "side_table": 1,
            "accent_chair": 1,
            "rug": 1,
        },
        "meta": {
            "desk_or_vanity": {"required": True},
            "office_chair": {"required": True},
            "lamp_small": {"required": False},
            "storage": {"required": False},
            "lamp_standing": {"required": False},
            "side_table": {"required": False},
            "accent_chair": {"required": False},
            "rug": {"required": False},
        },
    },
}
STYLE_TAG_VOCAB: list[str] = [
    "cozy",
    "modern",
    "neutral",
    "dark",
    "boho",
    "minimal",
    "rustic",
    "color_pop",
]
PERSONA_TAG_VOCAB: list[str] = [
    "cozy_homebody",
    "family",
    "premium",
    "budget",
    "student",
    "kids",
    "young_professional",
    "minimalist",
]
def get_supabase():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)

_vibe_model = None
_style_tag_embs = None
_persona_tag_embs = None

def _get_vibe_model():
    global _vibe_model
    if _vibe_model is None:
        _vibe_model = load_vibe_model()
    return _vibe_model


def _get_style_tag_embeddings(model):
    """
    Precompute embeddings for each style tag in our vocab.
    """
    global _style_tag_embs
    if _style_tag_embs is None:
        embs = []
        for tag in STYLE_TAG_VOCAB:
            embs.append(get_vibe_embedding(model, tag))
        _style_tag_embs = np.stack(embs, axis=0)  # shape: (V, D)
    return _style_tag_embs


def extract_style_tags(vibe_text: str) -> list[str]:
    """
    Map free-text user description to a small set of style tags
    from STYLE_TAG_VOCAB using the vibe embedding model.
    """
    text = vibe_text.strip()
    if not text:
        return []

    model = _get_vibe_model()

    # embed the user sentence
    text_emb = get_vibe_embedding(model, text)  # assumed L2-normalized

    # embed the known style tags (cached)
    tag_embs = _get_style_tag_embeddings(model)  # (V, D)

    # cosine similarity because everything is normalized
    sims = tag_embs @ text_emb  # (V,)

    # pick top-k tags with a minimum similarity threshold
    top_k = 3
    min_sim = 0.25

    idxs = np.argsort(-sims)[:top_k]

    chosen_tags: list[str] = []
    for i in idxs:
        if sims[i] >= min_sim:
            chosen_tags.append(STYLE_TAG_VOCAB[i])

    return chosen_tags
def _get_persona_tag_embeddings(model):
    global _persona_tag_embs
    if _persona_tag_embs is None:
        embs = []
        for tag in PERSONA_TAG_VOCAB:
            embs.append(get_vibe_embedding(model, tag))
        _persona_tag_embs = np.stack(embs, axis=0)
    return _persona_tag_embs


def extract_persona_tag(vibe_text: str) -> str | None:
    """
    Map the full user description into ONE persona tag from PERSONA_TAG_VOCAB,
    or None if nothing matches confidently.
    """
    text = vibe_text.strip()
    if not text:
        return None

    model = _get_vibe_model()

    text_emb = get_vibe_embedding(model, text)
    persona_embs = _get_persona_tag_embeddings(model)

    sims = persona_embs @ text_emb  

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    min_sim = 0.25
    if best_sim < min_sim:
        return None

    return PERSONA_TAG_VOCAB[best_idx]

def main():
    supabase = get_supabase()
    vibe_text = input("Describe the home: ").strip()

    user_style_tags = extract_style_tags(vibe_text)
    user_persona = extract_persona_tag(vibe_text)

    print("[DEBUG] user_style_tags extracted from sentence:", user_style_tags)
    print("[DEBUG] user_persona extracted from sentence:", user_persona)
    budget_input = input("Total furniture budget for this home (in dollars, e.g. 2500): ").strip()
    house_budget_cents = None
    if budget_input:
        try:
            house_budget_cents = int(float(budget_input) * 100)
        except ValueError:
            print("Could not parse budget, proceeding without budget.")
            house_budget_cents = None
    def ask_count(label: str) -> int:
        raw = input(f"How many {label}? (0 for none): ").strip()
        if not raw:
            return 0
        try:
            return int(raw)
        except ValueError:
            print(f"Could not parse '{raw}' as a number, treating as 0.")
            return 0

    num_bedrooms = ask_count("bedrooms")
    num_living = ask_count("living rooms")
    num_bath = ask_count("bathrooms")
    num_kitchen = ask_count("kitchens")
    num_office = ask_count("offices")

    rooms: List[Dict[str, Any]] = []

    for i in range(1, num_bedrooms + 1):
        rooms.append({"room_type": "bedroom", "room_index": i})

    for i in range(1, num_living + 1):
        rooms.append({"room_type": "living_room", "room_index": i})

    for i in range(1, num_bath + 1):
        rooms.append({"room_type": "bathroom", "room_index": i})

    for i in range(1, num_kitchen + 1):
        rooms.append({"room_type": "kitchen", "room_index": i})

    for i in range(1, num_office + 1):
        rooms.append({"room_type": "office", "room_index": i})

    if not rooms:
        rooms = [
            {"room_type": "bedroom", "room_index": 1},
            {"room_type": "living_room", "room_index": 1},
        ]

    for room_plan in rooms:
        room_type = room_plan["room_type"]

        schema = DEFAULT_ROOM_SLOTS.get(room_type)
        if schema is None:
            continue

        room_plan["slots_requested"] = schema["slots"]
        room_plan["slot_meta"] = schema["meta"]

        room_plan["target_style_tags"] = user_style_tags
        room_plan["persona"] = user_persona
        room_plan["house_budget_cents"] = house_budget_cents

        print("\n====================================================")
        print(f"Greedy: {room_plan['room_type']} #{room_plan['room_index']}")
        t0 = time.time()
        g = plan_room_greedy(supabase, room_plan)
        t1 = time.time()
        print(f"Greedy planning time: {t1 - t0:.4f}s")
        print("Chosen items (Greedy):")
        for item in g.chosen_items:
            product = item.product
            name = product.get("name", "<no name>")
            price = item.price_cents / 100.0
            print(f"  - [{item.slot_name}] {name}   ${price:.2f}")

        print("\n----------------------------------------------------")
        print(f"A*: {room_plan['room_type']} #{room_plan['room_index']}")
        t0 = time.time()
        a = plan_room_astar(supabase, room_plan)
        t1 = time.time()
        print(f"A* planning time: {t1 - t0:.4f}s")

        print("Chosen items (A*):")
        for item in a.chosen_items:
            slot = item.slot_name
            prod = item.product
            score = prod.get("_score") or prod.get("_match_score") or 0.0
            print(f"  - [{slot}] {prod.get('name')}   score={score:.3f}")

if __name__ == "__main__":
    main()