# planner_config.py

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class SlotProposal:
    slots: Dict[str, int]
    meta: Dict[str, Dict[str, Any]]


ROOM_TEMPLATES: Dict[str, Dict[str, Dict[str, int]]] = {
    "bedroom": {
        "hard": {
            "bed": 1,
            "dresser": 1,
            "desk_or_vanity": 1,
            "lamp_standing": 1,
            "lamp_small": 1,
            "nightstand": 1,
        },
        "soft": {
            "accent_chair": 1,
            "mirror": 1,
        },
    },

    "living_room": {
        "hard": {
            "sofa": 1,
            "coffee_table": 1,
            "media_unit": 1,
            "lamp_standing": 1,
            "lamp_small": 1,
        },
        "soft": {
            "accent_chair_big": 1,
            "side_table": 1,
            "rug": 1,
            "storage": 1,
            "mirror": 1,
        },
    },

    "bathroom": {
        "hard": {
            "mirror": 1,
            "sconce": 2,
        },
        "soft": {
            "storage_small": 1,
            "bath_mat": 1,
            "hamper": 1,
        },
    },

    "kitchen": {
        "hard": {
            "dining_table": 1,
            # dining_chair is added dynamically
        },
        "soft": {
            "bar_stool": 4,
            "sideboard": 1,
            "kitchen_rug": 1,
        },
    },

    "office": {
        "hard": {
            "desk_or_vanity": 1,
            "office_chair": 1,
            "lamp_small": 1,
            "storage": 1,
        },
        "soft": {
            "lamp_standing": 1,
            "side_table": 1,
            "accent_chair": 1,
            "rug": 1,
        },
    },
}


def _include_bedroom_soft(slot_name: str, room_area_sqft: float) -> bool:
    if slot_name == "accent_chair":
        return room_area_sqft >= 120.0
    return True


def _include_living_room_soft(slot_name: str, room_area_sqft: float) -> bool:
    if slot_name == "accent_chair_big":
        return room_area_sqft >= 140.0
    return True


def _include_bathroom_soft(slot_name: str, room_area_sqft: float) -> bool:
    if slot_name == "storage_small":
        return room_area_sqft >= 50.0
    return True


def _include_soft_slot(room_type: str, slot_name: str, room_area_sqft: float) -> bool:
    if room_type == "bedroom":
        return _include_bedroom_soft(slot_name, room_area_sqft)
    if room_type == "living_room":
        return _include_living_room_soft(slot_name, room_area_sqft)
    if room_type == "bathroom":
        return _include_bathroom_soft(slot_name, room_area_sqft)
    # kitchen and office soft slots always allowed for now
    return True


def _compute_kitchen_dining_chairs(
    room_area_sqft: float,
    family_size: Optional[int],
) -> int:
    # chair rule:
    # area < 100: N
    # 100-149: N + 1
    # 150-199: N + 2
    # 200+: N + 4
    if family_size is None or family_size <= 0:
        family_size = 2
    N = family_size

    if room_area_sqft < 100:
        return N
    if room_area_sqft < 150:
        return N + 1
    if room_area_sqft < 200:
        return N + 2
    return N + 4


def propose_slots(
    room_type: str,
    room_area_sqft: float,
    persona: str = "default",
    family_size: Optional[int] = None,
) -> SlotProposal:
    if room_type not in ROOM_TEMPLATES:
        raise ValueError(f"Unknown room_type: {room_type}")

    tmpl = ROOM_TEMPLATES[room_type]
    hard = tmpl.get("hard", {})
    soft = tmpl.get("soft", {})

    slots: Dict[str, int] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    # hard slots
    for slot_name, count in hard.items():
        if count <= 0:
            continue
        slots[slot_name] = slots.get(slot_name, 0) + count
        meta[slot_name] = {
            "required": True,
            "base_count": count,
            "source": "template_hard",
        }

    # soft slots
    for slot_name, count in soft.items():
        if count <= 0:
            continue
        if not _include_soft_slot(room_type, slot_name, room_area_sqft):
            continue
        slots[slot_name] = slots.get(slot_name, 0) + count
        meta[slot_name] = {
            "required": False,
            "base_count": count,
            "source": "template_soft",
        }

    # kitchen chairs
    if room_type == "kitchen":
        chair_count = _compute_kitchen_dining_chairs(room_area_sqft, family_size)
        if chair_count > 0:
            slots["dining_chair"] = chair_count
            meta["dining_chair"] = {
                "required": True,
                "base_count": chair_count,
                "source": "kitchen_dining_rule",
                "family_size": family_size,
            }

    # tag with persona
    for slot_name in slots.keys():
        meta.setdefault(slot_name, {})
        meta[slot_name].setdefault("persona", persona)

    return SlotProposal(slots=slots, meta=meta)
