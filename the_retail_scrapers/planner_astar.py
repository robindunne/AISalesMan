from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import heapq

from planner_greedy import (
    BundleItem,
    BundleResult,
    get_price_cents,
    get_area_sqft_from_row,
    fetch_candidates_for_slot,
)

@dataclass(order=True)
class SearchState:
    """
    Represents a partial room being built.
    Ordered by `priority` (negative potential so heapq acts as a max-heap),
    then by `tiebreaker` to avoid comparison errors when priorities tie.
    """
    priority: float
    tiebreaker: int = field(compare=True)
    idx: int = field(compare=False)                 # how many slots have been filled
    total_score: float = field(compare=False)       # accumulated match score
    total_cost_cents: int = field(compare=False)
    total_area_sqft: float = field(compare=False)
    chosen_items: List[BundleItem] = field(default_factory=list, compare=False)


def _normalize_tags(value: Any) -> List[str]:
    """
    Ensure we always treat tags as a list[str].
    Supports:
      - None
      - list[str]
      - comma-separated string "a,b,c"
    """
    if value is None:
        return []
    if isinstance(value, list):
        # filter out empties
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [s.strip() for s in value.split(",") if s.strip()]
    # fallback
    return [str(value).strip()]


def _compute_style_score(
    product_row: Dict[str, Any],
    user_style_tags: List[str],
) -> float:
    """
    style_score ∈ [0,1]

    If user_style_tags are provided:
      style_score = (# of user style tags present in product) / (# user style tags)
    If no user_style_tags → style_score = 1.0 (no style preference).
    """
    user_styles = [s.lower() for s in user_style_tags if s]
    if not user_styles:
        return 1.0

    product_styles = _normalize_tags(product_row.get("style_tags"))
    product_styles_l = {s.lower() for s in product_styles}

    overlap = len({s for s in user_styles if s in product_styles_l})
    return overlap / float(len(user_styles))


def _compute_persona_score(
    product_row: Dict[str, Any],
    user_persona: Optional[str],
) -> float:
    """
    persona_score ∈ [0,1]

    If user_persona is given:
      persona_score = 1.0 if user_persona in product_persona_tags else 0.0
    If no user_persona → persona_score = 1.0 (no persona preference).
    """
    persona = (user_persona or "").strip().lower()
    if not persona:
        return 1.0

    product_personas = _normalize_tags(product_row.get("persona_tags"))
    product_personas_l = {p.lower() for p in product_personas}

    return 1.0 if persona in product_personas_l else 0.0


def _compute_price_area_score(
    product_row: Dict[str, Any],
    room_budget_cents: Optional[int],
    max_area_sqft: Optional[float],
) -> float:
    """
    price_area_score ∈ [0,1]

    - Rewards items that use a small fraction of the room budget and area cap.
    - Punishes items that consume a large fraction of budget and/or area.
    """

    price_cents = get_price_cents(product_row)
    area_sqft = get_area_sqft_from_row(product_row)

    # --- Budget part ---
    if not room_budget_cents or room_budget_cents <= 0 or price_cents <= 0:
        budget_score = 1.0
    else:
        frac_budget = price_cents / float(room_budget_cents)
        if frac_budget <= 0.25:
            budget_score = 1.0
        elif frac_budget >= 1.0:
            budget_score = 0.0
        else:
            # linear decay from 1 at 0.25 → 0 at 1.0
            budget_score = 1.0 - (frac_budget - 0.25) / (1.0 - 0.25)

    # --- Area part ---
    if not max_area_sqft or max_area_sqft <= 0 or area_sqft <= 0:
        area_score = 1.0
    else:
        frac_area = area_sqft / float(max_area_sqft)
        if frac_area <= 0.25:
            area_score = 1.0
        elif frac_area >= 1.0:
            area_score = 0.0
        else:
            # linear decay from 1 at 0.25 → 0 at 1.0
            area_score = 1.0 - (frac_area - 0.25) / (1.0 - 0.25)

    # Combine budget + area
    return 0.5 * budget_score + 0.5 * area_score


def _compute_overall_match_score(
    product_row: Dict[str, Any],
    user_style_tags: List[str],
    user_persona: Optional[str],
    room_budget_cents: Optional[int],
    max_area_sqft: Optional[float],
    w_style: float = 0.50,
    w_persona: float = 0.35,
    w_price: float = 0.15,
) -> float:
    """
    Total heuristic match H ∈ [0,1]:

        H = w_style * style_score +
            w_persona * persona_score +
            w_price * price_area_score
    """

    total_w = w_style + w_persona + w_price
    if total_w <= 0:
        return 0.0

    style_score = _compute_style_score(product_row, user_style_tags)
    persona_score = _compute_persona_score(product_row, user_persona)
    price_area_score = _compute_price_area_score(
        product_row,
        room_budget_cents,
        max_area_sqft,
    )

    H = (
        w_style * style_score +
        w_persona * persona_score +
        w_price * price_area_score
    ) / total_w

    if H < 0.0:
        H = 0.0
    if H > 1.0:
        H = 1.0

    return H


# ---------------------------------------------------------------------------
# A*-style planner: global search over slot sequence with branch-and-bound
# ---------------------------------------------------------------------------

def plan_room_astar(
    supabase: Any,
    room_plan: Dict[str, Any],
    max_area_fraction: float = 0.7,
) -> BundleResult:
    """
    "A*" planner that differs from greedy by:
      - Searching globally over the entire slot sequence, not slot-by-slot.
      - Using a heuristic based on (style, persona, price + area).
      - Using budget/area as hard constraints.
      - Doing branch-and-bound using an optimistic upper bound on total score.

    Signature matches planner_demo:
        plan_room_astar(supabase, room_plan) -> BundleResult
    """

    room_index: int = int(room_plan.get("room_index", 1))
    room_type: str = room_plan.get("room_type", "unknown")
    room_area_sqft: float = float(room_plan.get("area_sqft", 0.0))

    slots: Dict[str, int] = dict(room_plan.get("slots_requested") or {})

    # Budget per room (in cents). 0 or None means "no specific per-room budget".
    budget_cents: Optional[int] = room_plan.get("budget_cents") or None
    # house_budget_cents = room_plan.get("house_budget_cents")  # handled at higher level, if needed

    # Area cap as a HARD constraint only
    max_area_sqft: Optional[float] = (
        room_area_sqft * max_area_fraction if room_area_sqft > 0 else None
    )

    # User preferences:
    user_style_tags: List[str] = _normalize_tags(
        room_plan.get("target_style_tags")
    )
    user_persona: Optional[str] = room_plan.get("persona")

    # ------------------------
    # 1) Pre-fetch and score candidates per slot
    # ------------------------
    candidates_by_slot: Dict[str, List[Dict[str, Any]]] = {}

    for slot_name in slots.keys():
        candidates = fetch_candidates_for_slot(supabase, slot_name)
        print(f"[DEBUG] Slot {slot_name}: fetched {len(candidates)} candidates")

        scored: List[Dict[str, Any]] = []
        for row in candidates:
            H = _compute_overall_match_score(
                row,
                user_style_tags=user_style_tags,
                user_persona=user_persona,
                room_budget_cents=budget_cents,
                max_area_sqft=max_area_sqft,
            )

            # Allow H == 0 items so slots remain fillable even with weak matches
            if H < 0:
                continue

            row_with_score = dict(row)
            row_with_score["_match_score"] = H
            scored.append(row_with_score)

        if not scored:
            print(f"[DEBUG] ⚠️ No *scored* candidates for slot '{slot_name}'")
        candidates_by_slot[slot_name] = scored

    # ------------------------
    # 2) Build slot sequence (only slots that have at least one candidate)
    # ------------------------
    slot_sequence: List[str] = []
    for slot_name, count in slots.items():
        if count <= 0:
            continue
        if not candidates_by_slot.get(slot_name):
            # We'll report this later as unfilled, but don't include it in search
            continue
        slot_sequence.extend([slot_name] * count)

    num_slots = len(slot_sequence)

    if num_slots == 0:
        # No fillable slots → return empty result, mark everything as unfilled.
        slots_filled_zero = {slot: 0 for slot in slots.keys()}
        return BundleResult(
            room_type=room_type,
            room_index=room_index,
            area_sqft=room_area_sqft,
            budget_cents=budget_cents,
            slots_requested=slots,
            slots_filled=slots_filled_zero,
            slots_unfilled=dict(slots),
            chosen_items=[],
            total_cost_cents=0,
            total_area_sqft=0.0,
            max_area_sqft=max_area_sqft,
        )

    # ------------------------
    # 3) Precompute best possible score per slot instance
    # ------------------------
    best_score_per_slot_instance: List[float] = []
    for slot_name in slot_sequence:
        cands = candidates_by_slot.get(slot_name, [])
        if cands:
            best = max((c.get("_match_score", 0.0) for c in cands), default=0.0)
        else:
            best = 0.0
        best_score_per_slot_instance.append(best)

    def remaining_upper_bound(idx: int) -> float:
        """
        Sum of best possible scores for remaining slot instances (optimistic upper bound).
        """
        if idx >= num_slots:
            return 0.0
        return float(sum(best_score_per_slot_instance[idx:]))

    # ------------------------
    # 4) A*-style search with branch-and-bound
    # ------------------------
    open_heap: List[SearchState] = []
    tiebreaker_counter = 0

    # Initial empty state
    initial_potential = remaining_upper_bound(0)
    heapq.heappush(
        open_heap,
        SearchState(
            priority=-initial_potential,
            tiebreaker=tiebreaker_counter,
            idx=0,
            total_score=0.0,
            total_cost_cents=0,
            total_area_sqft=0.0,
            chosen_items=[],
        ),
    )
    tiebreaker_counter += 1

    expansions = 0
    max_expansions = 5000  # Prevent runaway search
    best_complete_state: Optional[SearchState] = None

    while open_heap and expansions < max_expansions:
        state = heapq.heappop(open_heap)
        expansions += 1

        idx = state.idx

        # Prune by constraints
        if budget_cents is not None and state.total_cost_cents > budget_cents:
            continue
        if max_area_sqft is not None and state.total_area_sqft > max_area_sqft:
            continue

        # If this is a complete assignment (all slots considered)
        if idx >= num_slots:
            # Update best complete if this one is better
            if best_complete_state is None or state.total_score > best_complete_state.total_score:
                best_complete_state = state
            # Do not expand further from here
            continue

        # Branch-and-bound: if even the optimistic upper bound from here
        # can't beat the best complete solution, prune.
        if best_complete_state is not None:
            potential_max = state.total_score + remaining_upper_bound(idx)
            if potential_max <= best_complete_state.total_score:
                continue

        slot_name = slot_sequence[idx]
        candidates = candidates_by_slot.get(slot_name, [])

        # If somehow no candidates here, this path is dead.
        if not candidates:
            continue

        # Try each candidate for this slot
        for row in candidates:
            price_cents = get_price_cents(row)
            area_sqft = get_area_sqft_from_row(row)

            new_cost = state.total_cost_cents + price_cents
            new_area = state.total_area_sqft + area_sqft

            # Check constraints BEFORE creating new state
            if budget_cents is not None and new_cost > budget_cents:
                continue
            if max_area_sqft is not None and new_area > max_area_sqft:
                continue

            match_score = float(row.get("_match_score", 0.0))
            new_total_score = state.total_score + match_score
            next_idx = idx + 1

            potential_total = new_total_score + remaining_upper_bound(next_idx)

            new_item = BundleItem(
                slot_name=slot_name,
                product_id=str(row.get("id")),
                product=row,
                price_cents=price_cents,
                area_sqft=area_sqft,
            )
            new_chosen = state.chosen_items + [new_item]

            heapq.heappush(
                open_heap,
                SearchState(
                    priority=-potential_total,
                    tiebreaker=tiebreaker_counter,
                    idx=next_idx,
                    total_score=new_total_score,
                    total_cost_cents=new_cost,
                    total_area_sqft=new_area,
                    chosen_items=new_chosen,
                ),
            )
            tiebreaker_counter += 1

    # ------------------------
    # 5) Pick final state (complete if possible, else best partial)
    # ------------------------
    if best_complete_state is not None:
        final_state = best_complete_state
    else:
        if not open_heap:
            # Completely failed – return empty result
            slots_filled_zero = {slot: 0 for slot in slots.keys()}
            return BundleResult(
                room_type=room_type,
                room_index=room_index,
                area_sqft=room_area_sqft,
                budget_cents=budget_cents,
                slots_requested=slots,
                slots_filled=slots_filled_zero,
                slots_unfilled=dict(slots),
                chosen_items=[],
                total_cost_cents=0,
                total_area_sqft=0.0,
                max_area_sqft=max_area_sqft,
            )

        # Use best partial state by actual score
        best_partial_state = max(open_heap, key=lambda s: s.total_score)
        final_state = best_partial_state

    # ------------------------
    # 6) Convert final state to BundleResult
    # ------------------------
    slots_filled_counts: Dict[str, int] = {slot: 0 for slot in slots.keys()}
    for item in final_state.chosen_items:
        if item.slot_name in slots_filled_counts:
            slots_filled_counts[item.slot_name] += 1

    slots_unfilled: Dict[str, int] = {}
    for slot_name, required_count in slots.items():
        filled = slots_filled_counts.get(slot_name, 0)
        if filled < required_count:
            slots_unfilled[slot_name] = required_count - filled

    return BundleResult(
        room_type=room_type,
        room_index=room_index,
        area_sqft=room_area_sqft,
        budget_cents=budget_cents,
        slots_requested=slots,
        slots_filled=slots_filled_counts,
        slots_unfilled=slots_unfilled,
        chosen_items=final_state.chosen_items,
        total_cost_cents=final_state.total_cost_cents,
        total_area_sqft=final_state.total_area_sqft,
        max_area_sqft=max_area_sqft,
    )
