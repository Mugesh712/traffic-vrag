"""Tests for M12's rule-based query understanding."""
from __future__ import annotations

import pytest

from src.retrieval.intent import parse_intent_rules


def intent(question: str, regions=None):
    return parse_intent_rules(question, known_regions=regions)


# --- question type ---------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which white sedans were present?", "factual"),
        ("How many trucks stopped?", "counting"),
        ("What is the number of buses?", "counting"),
        ("What if the white car had stopped earlier?", "counterfactual"),
        ("Which vehicle will turn next?", "forecast"),
    ],
)
def test_question_type_classification(question, expected):
    assert intent(question).question_type == expected


@pytest.mark.parametrize("question", [
    "What was the licence plate number of the red car?",
    "What is the serial number of the bus?",
    "What is the model number of that van?",
])
def test_identification_questions_are_not_mistaken_for_counting(question):
    """Found by the M16 QA benchmark: the bare marker "number of" fires inside
    "plate number of", routing an unanswerable identification question to the
    deterministic counting path -- which asserts a count instead of declining,
    and scored the system BELOW the baselines on abstention."""
    assert intent(question).question_type == "factual"


@pytest.mark.parametrize("question", [
    "How many cars are there?",
    "What is the number of cars?",
    "Total number of vehicles?",
    "How often did a car stop?",
])
def test_genuine_counting_questions_still_classify_as_counting(question):
    assert intent(question).question_type == "counting"


def test_markers_match_on_word_boundaries():
    """"count" must not fire inside "account"/"discount"."""
    assert intent("Which car has the largest discount sticker?").question_type == "factual"


def test_counting_wins_over_counterfactual_phrasing():
    """"How many would have stopped" is still a counting question -- answering
    it with a top-K similarity list would be the wrong answer shape."""
    assert intent("How many cars would have stopped?").question_type == "counting"


# --- attributes ------------------------------------------------------------


def test_attributes_parse_into_canonical_vocabulary_values():
    parsed = intent("show me the white toyota sedan")
    assert parsed.target_attributes == {
        "color": "white", "vehicle_type": "sedan", "make": "toyota",
    }


def test_attribute_synonyms_normalize():
    parsed = intent("find the grey lorry")
    assert parsed.target_attributes["color"] == "gray"
    # "lorry" is a truck, which is a detector class -- so it constrains the
    # class, not the VLM-only vehicle_type. See the class-word test below.
    assert parsed.object_classes == ["truck"]


def test_no_attributes_when_question_names_none():
    assert intent("what happened?").target_attributes == {}


# --- classes ---------------------------------------------------------------


def test_pedestrian_maps_to_person_class():
    assert intent("did any pedestrians cross?").object_classes == ["person"]


@pytest.mark.parametrize("word,plural", [
    ("truck", "trucks"), ("bus", "buses"), ("motorcycle", "motorcycles"), ("bicycle", "bicycles"),
])
def test_class_words_constrain_class_not_the_vlm_body_type(word, plural):
    """"truck" names both a detector class and a VLM body type. The detector
    always assigns a class; vehicle_type only exists when the VLM confirmed
    one, so constraining on it matches nothing -- a live query for "blue
    trucks" returned zero before this."""
    parsed = intent(f"show me blue {plural}")
    assert "vehicle_type" not in parsed.target_attributes
    assert word in parsed.object_classes
    assert parsed.target_attributes["color"] == "blue"


def test_body_types_with_no_matching_class_still_constrain_vehicle_type():
    """"sedan" is only ever a body type, so it must survive as one."""
    parsed = intent("show me white sedans")
    assert parsed.target_attributes["vehicle_type"] == "sedan"
    assert parsed.object_classes == ["car"]


def test_specific_body_type_narrows_class_to_car():
    """"sedan" implies a car; the generic word "vehicle" must not then drag
    every other class in alongside it."""
    parsed = intent("show me the white sedan vehicle")
    assert parsed.object_classes == ["car"]


# --- events ----------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("which car overtook another?", "OVERTAKE"),
        ("which car passed the truck?", "OVERTAKE"),
        ("what stopped at the junction?", "STOP"),
        ("which vehicle parked?", "PARK"),
        ("who turned left?", "TURN"),
        ("which car crossed the line?", "CROSSES"),
    ],
)
def test_event_phrases_map_to_canonical_types(question, expected):
    assert expected in intent(question).event_types


def test_lane_change_is_not_read_as_a_turn():
    """"changed lanes" contains no turn word, but a careless phrase list would
    still let TURN match first."""
    parsed = intent("which car changed lanes?")
    assert parsed.event_types == ["LANE_CHANGE"]


def test_counterpart_of_a_relation_is_not_read_as_the_subject():
    """"which white car overtook a blue truck" describes TWO objects. Folding
    the truck into the subject's attributes would search for a white truck --
    caught by running the question against the real stores and getting zero
    results."""
    parsed = intent("which white car overtook a blue truck?")
    # Subject: white, a car. Critically, NOT blue and NOT a truck.
    assert parsed.target_attributes == {"color": "white"}
    assert parsed.object_classes == ["car"]
    # Counterpart: blue, a truck (constrained as a class, per the rule above).
    assert parsed.counterpart_attributes["color"] == "blue"
    assert parsed.counterpart_classes == ["truck"]


def test_non_relational_question_has_no_counterpart():
    parsed = intent("which white car stopped?")
    assert parsed.counterpart_attributes == {}
    assert parsed.counterpart_classes == []


def test_pairwise_events_are_reported_as_relations():
    parsed = intent("which white car overtook a truck?")
    assert parsed.relations == ["OVERTAKE"]
    assert intent("which car stopped?").relations == []


# --- time ------------------------------------------------------------------


def test_before_sets_only_the_upper_bound():
    parsed = intent("which cars were present before 12:05?")
    assert parsed.time_range.before == "12:05:00"
    assert parsed.time_range.after is None


def test_after_sets_only_the_lower_bound():
    parsed = intent("what happened after 11:30?")
    assert parsed.time_range.after == "11:30:00"
    assert parsed.time_range.before is None


def test_meridiem_converts_to_twenty_four_hour():
    assert intent("before 12:05 pm").time_range.before == "12:05:00"
    assert intent("before 1:05 pm").time_range.before == "13:05:00"
    assert intent("after 12:30 am").time_range.after == "00:30:00"


def test_between_sets_both_bounds():
    parsed = intent("what happened between 11:00 and 11:30?")
    assert parsed.time_range.after == "11:00:00"
    assert parsed.time_range.before == "11:30:00"


def test_at_a_time_becomes_a_tight_window_not_an_equality():
    """An exact timestamp equality would essentially never match real data."""
    parsed = intent("what was happening at 11:04?")
    assert parsed.time_range.after == "11:04:00"
    assert parsed.time_range.before == "11:04:00"


def test_bare_integers_are_not_parsed_as_times():
    """"how many 3 cars" and "top 5" must not become clock times."""
    assert intent("how many cars stopped in the top 5?").time_range.is_empty()


def test_impossible_hours_are_rejected():
    assert intent("show me object 99:99").time_range.is_empty()


# --- regions ---------------------------------------------------------------


def test_known_regions_are_matched_including_underscored_names():
    parsed = intent("which cars crossed the stop line?", regions=["stop_line", "crosswalk"])
    assert parsed.region_ids == ["stop_line"]


def test_unknown_regions_are_not_invented():
    assert intent("which cars crossed the finish line?", regions=["stop_line"]).region_ids == []


# --- structured constraints ------------------------------------------------


def test_has_structured_constraints_detects_a_purely_semantic_question():
    assert intent("show me something unusual").has_structured_constraints() is False
    assert intent("show me white cars").has_structured_constraints() is True
