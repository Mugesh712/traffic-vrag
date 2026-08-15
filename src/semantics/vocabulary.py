"""Controlled attribute vocabularies, shared by M5 (extraction) and M6 (voting).

Each vocabulary maps a canonical value -> the surface forms that count as a
match. Longer, more specific phrases are listed first so "pickup truck" wins
over a bare "truck".

Two consumers, two uses:
  * M5 scans a free-text caption for any surface form (`match_vocab`).
  * M6 normalizes an already-extracted value to its canonical form
    (`normalize_value`). That is idempotent against M5's own output, but it is
    not redundant: a swapped-in BLIP-2 / InternVL2 backend would emit raw
    strings, and this is the layer that keeps the value space closed.
"""
from __future__ import annotations

import re

COLOR_VOCAB: dict[str, list[str]] = {
    "white": ["white", "off-white", "off white", "cream"],
    "black": ["black"],
    "gray": ["gray", "grey", "charcoal"],
    "silver": ["silver"],
    "red": ["red", "maroon", "crimson"],
    "blue": ["blue", "navy"],
    "green": ["green"],
    "yellow": ["yellow"],
    "orange": ["orange"],
    "brown": ["brown", "tan", "beige"],
    "gold": ["gold", "golden"],
    "purple": ["purple", "violet"],
    "pink": ["pink"],
}

VEHICLE_TYPE_VOCAB: dict[str, list[str]] = {
    "pickup": ["pickup truck", "pickup"],
    "suv": ["suv", "sport utility vehicle"],
    "minivan": ["minivan", "mini van"],
    "van": ["van"],
    "sedan": ["sedan", "saloon"],
    "hatchback": ["hatchback"],
    "coupe": ["coupe"],
    "convertible": ["convertible"],
    "wagon": ["station wagon", "wagon"],
    "taxi": ["taxi", "cab"],
    "police": ["police car", "police vehicle"],
    "ambulance": ["ambulance"],
    "bus": ["bus", "coach"],
    "truck": ["truck", "lorry"],
    "motorcycle": ["motorcycle", "motorbike"],
    "scooter": ["scooter", "moped"],
    "bicycle": ["bicycle", "bike"],
}

MAKE_VOCAB: dict[str, list[str]] = {
    "toyota": ["toyota"],
    "honda": ["honda"],
    "ford": ["ford"],
    "chevrolet": ["chevrolet", "chevy"],
    "bmw": ["bmw"],
    "mercedes-benz": ["mercedes-benz", "mercedes benz", "mercedes"],
    "audi": ["audi"],
    "tesla": ["tesla"],
    "nissan": ["nissan"],
    "hyundai": ["hyundai"],
    "kia": ["kia"],
    "volkswagen": ["volkswagen", "vw"],
    "mazda": ["mazda"],
    "subaru": ["subaru"],
    "jeep": ["jeep"],
    "dodge": ["dodge"],
    "lexus": ["lexus"],
    "volvo": ["volvo"],
    "porsche": ["porsche"],
    "land rover": ["land rover", "range rover"],
    "jaguar": ["jaguar"],
    "suzuki": ["suzuki"],
    "renault": ["renault"],
    "tata": ["tata"],
    "mahindra": ["mahindra"],
}

DIRECTION_VOCAB: dict[str, list[str]] = {
    "away_from_camera": ["driving away", "moving away", "facing away", "back of the"],
    "toward_camera": ["facing the camera", "facing forward", "coming toward", "approaching", "front of the"],
    "left": ["moving to the left", "heading left", "traveling left", "traveling to the left"],
    "right": ["moving to the right", "heading right", "traveling right", "traveling to the right"],
    "stationary": ["parked", "standing still", "stationary"],
}

# `model` has no vocabulary: a generic captioning VLM does not name specific
# models, and inventing a lookup table would manufacture false precision.
# It stays free-text (lowercased) until a badge-OCR or stronger VLM lands.
ATTRIBUTE_VOCABS: dict[str, dict[str, list[str]]] = {
    "color": COLOR_VOCAB,
    "vehicle_type": VEHICLE_TYPE_VOCAB,
    "make": MAKE_VOCAB,
    "direction": DIRECTION_VOCAB,
}

# Attributes carried end to end, in output order.
ATTRIBUTES: tuple[str, ...] = ("color", "vehicle_type", "make", "model", "direction")


def match_vocab(text: str, vocab: dict[str, list[str]], allow_plural: bool = False) -> str | None:
    """First canonical value whose surface form appears in `text`, or None.

    `allow_plural` also accepts a trailing "s"/"es". Off by default because
    M5 reads VLM captions, which describe one crop and say "a white sedan";
    M12 reads user questions, which say "show me white sedans" and would
    otherwise match nothing at all.
    """
    lowered = text.lower()
    suffix = r"(?:e?s)?" if allow_plural else ""
    for canonical, surface_forms in vocab.items():
        for phrase in surface_forms:
            if re.search(rf"\b{re.escape(phrase)}{suffix}\b", lowered):
                return canonical
    return None


def normalize_value(attribute: str, value: str | None) -> str | None:
    """Map a raw attribute value onto its canonical form.

    Returns None for an unrecognized value rather than passing it through:
    an unmatched string would widen the value space that M7's semantic gate
    and the knowledge graph rely on being closed.
    """
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None

    vocab = ATTRIBUTE_VOCABS.get(attribute)
    if vocab is None:  # free-text attribute (`model`)
        return cleaned
    if cleaned in vocab:  # already canonical — the common case for M5 output
        return cleaned
    return match_vocab(cleaned, vocab)
