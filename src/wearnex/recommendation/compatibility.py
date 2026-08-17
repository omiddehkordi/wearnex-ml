"""Rule-based category pairing used until a learned compatibility model exists.

`recommend_similar` in the engine (same embedding neighborhood) answers
"more like this". `recommend_complementary` answers "what goes with
this" — a different question that plain visual similarity can't answer,
since a similar-looking pair of jeans isn't an outfit suggestion. This
starts as a hand-authored category graph; swap `complementary_categories`
for a learned outfit-compatibility model (e.g. trained on co-purchase or
curated-outfit data) without changing the engine's call site.
"""

from __future__ import annotations

_PAIRINGS: dict[str, list[str]] = {
    "t_shirt": ["jeans", "shorts", "pants", "jacket"],
    "shirt": ["pants", "jeans", "skirt"],
    "sweater": ["jeans", "pants", "skirt", "coat"],
    "hoodie": ["jeans", "pants", "shorts"],
    "jacket": ["t_shirt", "shirt", "jeans", "pants"],
    "coat": ["sweater", "dress", "pants"],
    "dress": ["jacket", "coat", "shoes", "bag"],
    "skirt": ["shirt", "sweater", "t_shirt"],
    "shorts": ["t_shirt", "hoodie"],
    "pants": ["shirt", "t_shirt", "sweater", "jacket"],
    "jeans": ["t_shirt", "shirt", "sweater", "jacket", "hoodie"],
    "shoes": ["dress", "jeans", "pants", "skirt", "shorts"],
    "bag": ["dress", "jeans", "pants", "skirt"],
    "hat": ["t_shirt", "jacket", "hoodie"],
    "accessory": ["dress", "shirt", "sweater"],
}


def complementary_categories(category: str) -> list[str]:
    """Categories that are conventionally worn/paired with `category`."""
    return _PAIRINGS.get(category, [])
