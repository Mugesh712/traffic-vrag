"""M12, step 1 — Query understanding: question -> structured QueryIntent.

WHY RULES BY DEFAULT, NOT AN LLM. The roadmap specifies an LLM here, and this
deviates deliberately. The intent's value space is already *closed*: a colour
must be one of M5's canonical colours, an event must be one of M9's six types,
a class one of the detector's six. An LLM's real strength is open-ended
understanding, but here it would mostly be asked to produce values from a
fixed list -- and any value outside that list is unqueryable anyway, since the
knowledge graph and vector metadata only ever store canonical ones.

A vocabulary-driven parser is instead deterministic (so M16's retrieval
ablation is reproducible), free, and needs no running model server for tests.
This is the same "rules first, LLM later" call the roadmap itself makes for
M9's event detection.

The LLM backend slots in behind `parse_intent`'s signature when M13 brings
Ollama; it is registered but not implemented, exactly like M5's BLIP-2 stub.
"""
from __future__ import annotations

import re

from src.semantics.vocabulary import (
    COLOR_VOCAB,
    DIRECTION_VOCAB,
    MAKE_VOCAB,
    VEHICLE_TYPE_VOCAB,
    match_vocab,
)
from src.utils.config import PipelineSettings, get_settings
from src.utils.schemas import QueryIntent, QuestionType, TimeRange, TrafficClass

# Surface forms -> canonical M9 event type. Longer phrases first within a
# type, and LANE_CHANGE before TURN so "changed lanes" is not read as a turn.
EVENT_PHRASES: dict[str, list[str]] = {
    "LANE_CHANGE": ["changed lanes", "change lanes", "lane change", "lane-change", "switched lanes"],
    "OVERTAKE": ["overtook", "overtake", "overtaking", "overtaken", "passed", "passing", "cut in front"],
    "PARK": ["parked", "parking", "park"],
    "STOP": ["stopped", "stopping", "stops", "stop", "halted", "waiting", "idle"],
    "TURN": ["turned", "turning", "turns", "turn"],
    "CROSSES": ["crossed", "crossing", "crosses", "cross", "entered", "went through"],
}

# Event types that relate two objects, reported separately as `relations`.
PAIRWISE_EVENTS = {"OVERTAKE"}

CLASS_PHRASES: dict[str, list[str]] = {
    "car": ["car", "cars", "sedan", "vehicle", "vehicles"],
    "truck": ["truck", "trucks", "lorry", "lorries"],
    "bus": ["bus", "buses", "coach"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "bike"],
    "bicycle": ["bicycle", "bicycles", "cyclist"],
    "person": ["person", "people", "pedestrian", "pedestrians"],
}

COUNTING_MARKERS = ["how many", "count", "number of", "total number", "how much"]
COUNTERFACTUAL_MARKERS = ["what if", "would have", "had the", "if the", "instead of", "could have"]
FORECAST_MARKERS = ["will ", "going to", "predict", "next", "about to", "likely to", "expect"]

_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)


class IntentParseError(RuntimeError):
    pass


def _classify_question(text: str) -> QuestionType:
    """Counting is checked first: "how many cars would have stopped" is a
    counting question with counterfactual phrasing, and answering it with a
    top-K similarity list would be the wrong shape entirely."""
    if any(marker in text for marker in COUNTING_MARKERS):
        return "counting"
    if any(marker in text for marker in COUNTERFACTUAL_MARKERS):
        return "counterfactual"
    if any(marker in text for marker in FORECAST_MARKERS):
        return "forecast"
    return "factual"


def _normalize_time(hour: str, minute: str | None, second: str | None, meridiem: str | None) -> str:
    h = int(hour)
    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
    return f"{h:02d}:{int(minute or 0):02d}:{int(second or 0):02d}"


def _find_times(text: str) -> list[tuple[int, str]]:
    """(position, "HH:MM:SS") for every clock time in the text.

    A bare integer is only a time when it carries a minute field or a
    meridiem -- otherwise "how many 3 cars" or "top 5" would parse as times.
    """
    found = []
    for match in _TIME_RE.finditer(text):
        hour, minute, second, meridiem = match.groups()
        if minute is None and meridiem is None:
            continue
        if int(hour) > 23:
            continue
        found.append((match.start(), _normalize_time(hour, minute, second, meridiem)))
    return found


def _parse_time_range(text: str) -> TimeRange:
    times = _find_times(text)
    if not times:
        return TimeRange()

    # "between X and Y" takes both; otherwise the nearest preceding keyword
    # decides which bound a time sets.
    if "between" in text and len(times) >= 2:
        return TimeRange(after=times[0][1], before=times[1][1])

    time_range = TimeRange()
    for position, value in times:
        preceding = text[:position]
        if re.search(r"\b(before|until|till|prior to|earlier than)\b[^.]*$", preceding):
            time_range.before = value
        elif re.search(r"\b(after|since|from|later than)\b[^.]*$", preceding):
            time_range.after = value
        elif re.search(r"\b(at|around)\b[^.]*$", preceding):
            # A point in time becomes a tight window rather than an exact
            # equality, which would almost never match a real timestamp.
            time_range.after = value
            time_range.before = value
    return time_range


def _find_phrases(text: str, phrase_map: dict[str, list[str]]) -> list[str]:
    """Plural-tolerant, for the same reason as match_vocab(allow_plural=True):
    questions say "which trucks", captions say "a truck"."""
    found = []
    for canonical, phrases in phrase_map.items():
        if any(re.search(rf"\b{re.escape(p)}(?:e?s)?\b", text) for p in phrases):
            found.append(canonical)
    return found


_BODY_TYPES_IMPLYING_CAR = (
    "sedan", "suv", "hatchback", "coupe", "convertible", "wagon", "taxi", "police",
)


def _extract_descriptors(text: str) -> tuple[dict[str, str], list[str]]:
    """Attributes and object classes named in one span of text."""
    attributes: dict[str, str] = {}
    for name, vocab in (
        ("color", COLOR_VOCAB),
        ("vehicle_type", VEHICLE_TYPE_VOCAB),
        ("make", MAKE_VOCAB),
        ("direction", DIRECTION_VOCAB),
    ):
        value = match_vocab(text, vocab, allow_plural=True)
        if value is not None:
            attributes[name] = value

    classes = _find_phrases(text, CLASS_PHRASES)
    # A specific body type already implies a car; don't let the generic word
    # "vehicle" then drag every other class in alongside it.
    if attributes.get("vehicle_type") in _BODY_TYPES_IMPLYING_CAR:
        classes = [c for c in classes if c == "car"] or ["car"]

    # Words like "truck" and "bus" name both a detector class and a VLM body
    # type. Constrain on the class, which the detector always assigns, rather
    # than on vehicle_type, which only exists when the VLM confirmed one --
    # otherwise "blue truck" filters on a property that is usually null and
    # matches nothing. Caught by a live query returning zero results.
    if attributes.get("vehicle_type") in TrafficClass.__args__:
        classes = sorted(set(classes) | {attributes.pop("vehicle_type")})

    return attributes, [c for c in classes if c in TrafficClass.__args__]


def _split_at_relation(text: str, relations: list[str]) -> tuple[str, str]:
    """Split a relational question at its verb: everything before describes
    the subject, everything after describes the counterpart.

    "which white car overtook a blue truck" -> ("which white car", "a blue truck")
    Without this, "truck" is read as the subject's own vehicle_type and the
    query goes looking for a white truck that overtook something.
    """
    earliest = None
    for relation in relations:
        for phrase in EVENT_PHRASES.get(relation, []):
            match = re.search(rf"\b{re.escape(phrase)}\b", text)
            if match and (earliest is None or match.end() < earliest):
                earliest = match.end()
    if earliest is None:
        return text, ""
    return text[:earliest], text[earliest:]


def parse_intent_rules(question: str, known_regions: list[str] | None = None) -> QueryIntent:
    text = question.lower().strip()

    event_types = _find_phrases(text, EVENT_PHRASES)
    relations = [e for e in event_types if e in PAIRWISE_EVENTS]

    subject_text, counterpart_text = _split_at_relation(text, relations)
    target_attributes, classes = _extract_descriptors(subject_text)
    counterpart_attributes, counterpart_classes = (
        _extract_descriptors(counterpart_text) if counterpart_text.strip() else ({}, [])
    )

    regions = [r for r in (known_regions or []) if r.lower().replace("_", " ") in text or r.lower() in text]

    return QueryIntent(
        question=question,
        question_type=_classify_question(text),
        target_attributes=target_attributes,
        object_classes=classes,
        event_types=event_types,
        relations=relations,
        region_ids=regions,
        time_range=_parse_time_range(text),
        counterpart_attributes=counterpart_attributes,
        counterpart_classes=counterpart_classes,
    )


def parse_intent(
    question: str,
    settings: PipelineSettings | None = None,
    known_regions: list[str] | None = None,
) -> QueryIntent:
    settings = settings or get_settings()
    backend = settings.retrieval.intent_backend
    if backend == "rules":
        return parse_intent_rules(question, known_regions)
    if backend == "llm":
        raise NotImplementedError(
            "LLM intent parsing arrives with M13's Ollama backend; "
            "use intent_backend: rules (configs/pipeline.yaml)."
        )
    raise IntentParseError(f"Unknown intent backend: {backend}")
