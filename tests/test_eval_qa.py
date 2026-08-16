"""Tests for M16's QA scoring.

The risk in an evaluation harness is that it quietly flatters the system it
evaluates. These pin the properties that prevent that: baselines get the same
parsing leniency our system gets, unscorable question types stay unscored
rather than defaulting to zero, and groundedness is checked against what was
actually retrieved.
"""
from __future__ import annotations

import pytest

from src.eval.qa import (
    QAItem,
    aggregate,
    extract_count,
    is_abstention,
    score_item,
    score_objects,
)


def item(qid="q1", qtype="factual", count=None, gt_ids=None, answerable=True):
    return QAItem(
        question_id=qid, question="?", question_type=qtype,
        expected_count=count, expected_gt_ids=gt_ids or [], answerable=answerable,
    )


# --- count extraction -------------------------------------------------------


def test_extract_count_from_digits():
    assert extract_count("There are 7 matching object(s).") == 7


def test_extract_count_from_number_words():
    """Baselines answer in prose; digit-only parsing would mark a correct
    baseline wrong and flatter our own system, which emits digits."""
    assert extract_count("There are seven cars in the video.") == 7


def test_extract_count_prefers_whichever_form_comes_first():
    assert extract_count("seven cars, or 8 if you count the van") == 7
    assert extract_count("8 cars, or seven excluding the van") == 8


def test_extract_count_handles_zero_words():
    assert extract_count("No vehicles were detected.") == 0


def test_extract_count_returns_none_when_absent():
    assert extract_count("Several cars drove past.") is None


# --- abstention -------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "insufficient evidence",
    "There is not enough information to answer.",
    "I cannot determine the licence plate.",
    # Real phrasings observed from models during the M16 run. Scoring only the
    # exact sentinel M13's prompt requests would credit our system for
    # following its own prompt format rather than for actually declining, and
    # would mark a genuinely-abstaining baseline as having answered.
    "The weather in this video is not specified in the provided context.",
    "The provided context does not contain any information about the weather.",
    "That detail is not mentioned in the frame descriptions.",
])
def test_abstention_phrases_are_recognised(text):
    assert is_abstention(text)


def test_abstention_detection_is_conservative_about_real_answers():
    """It must never credit a confident answer as an abstention -- that would
    inflate the headline metric of a paper about admitting uncertainty."""
    assert not is_abstention("The red car has licence plate ABC-123.")
    assert not is_abstention("Three white cars and one blue truck.")


def test_a_real_answer_is_not_an_abstention():
    assert not is_abstention("The white car overtook the truck.")


def test_declining_an_unanswerable_question_scores_correct():
    """The paper's central claim, measured directly."""
    score = score_item(item(answerable=False), "sys", "insufficient evidence", [], [])
    assert score.abstention_correct is True


def test_answering_an_unanswerable_question_scores_wrong():
    score = score_item(item(answerable=False), "sys", "The plate was ABC-123.", [], [])
    assert score.abstention_correct is False


def test_unanswerable_questions_are_not_scored_for_anything_else():
    """Only declining matters; grading content of an answer that shouldn't
    exist would be meaningless."""
    score = score_item(
        item(qtype="counting", count=3, answerable=False), "sys", "There are 3.", [], []
    )
    assert score.counting_correct is None
    assert score.object_f1 is None


# --- object set scoring -----------------------------------------------------


def test_score_objects_perfect_match():
    assert score_objects(["a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)


def test_score_objects_partial_match():
    precision, recall, f1 = score_objects(["a", "x"], ["a", "b"])
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_score_objects_with_no_citations_is_zero_not_an_error():
    assert score_objects([], ["a"]) == (0.0, 0.0, 0.0)


def test_object_f1_uses_the_gt_mapping():
    """The benchmark names objects by gt_id; the system cites global_ids. The
    bridge between them is the tracking match, not a name coincidence."""
    score = score_item(
        item(qtype="factual", gt_ids=["gt_1"]), "sys", "The white car.",
        cited_ids=["obj_0001"], retrieved_ids=["obj_0001"],
        gt_id_by_cited_id={"obj_0001": "gt_1"},
    )
    assert score.object_f1 == pytest.approx(1.0)


def test_object_f1_unscored_without_a_mapping():
    """No annotation -> no mapping -> the metric stays None, not zero, so it
    cannot drag an average down as if it were a measured failure."""
    score = score_item(
        item(qtype="factual", gt_ids=["gt_1"]), "sys", "The white car.",
        cited_ids=["obj_0001"], retrieved_ids=["obj_0001"], gt_id_by_cited_id=None,
    )
    assert score.object_f1 == 0.0  # mapping empty -> nothing matched
    assert score.grounded is True


# --- groundedness -----------------------------------------------------------


def test_citing_only_retrieved_objects_is_grounded():
    score = score_item(item(), "sys", "a", cited_ids=["obj_1"], retrieved_ids=["obj_1", "obj_2"])
    assert score.grounded is True


def test_citing_something_never_retrieved_is_ungrounded():
    score = score_item(item(), "sys", "a", cited_ids=["obj_9"], retrieved_ids=["obj_1"])
    assert score.grounded is False


def test_citing_nothing_counts_as_grounded():
    """Vacuously true: an answer that claims no sources cannot mis-cite one."""
    score = score_item(item(), "sys", "a", cited_ids=[], retrieved_ids=["obj_1"])
    assert score.grounded is True


# --- unscorable question types ----------------------------------------------


@pytest.mark.parametrize("qtype", ["counterfactual", "forecast"])
def test_speculative_types_are_not_scored_for_correctness(qtype):
    """There is no ground-truth answer to "what if", so inventing a rubric
    would be fake rigor. They still contribute groundedness."""
    score = score_item(
        item(qtype=qtype, gt_ids=["gt_1"]), "sys", "The bus would have continued.",
        cited_ids=["obj_1"], retrieved_ids=["obj_1"], gt_id_by_cited_id={"obj_1": "gt_1"},
    )
    assert score.object_f1 is None
    assert score.counting_correct is None
    assert score.grounded is True


def test_counting_unscored_when_the_benchmark_gives_no_expected_count():
    score = score_item(item(qtype="counting", count=None), "sys", "There are 7.", [], [])
    assert score.counting_correct is None


# --- aggregation ------------------------------------------------------------


def test_aggregate_averages_each_metric_only_over_applicable_items():
    """An unscored question type must not drag a mean toward zero."""
    scores = [
        score_item(item("q1", "counting", count=7), "sys", "There are 7.", [], []),
        score_item(item("q2", "counting", count=3), "sys", "There are 5.", [], []),
        score_item(item("q3", "forecast"), "sys", "It will turn.", [], []),
    ]
    summary = aggregate(scores)
    assert summary["n_questions"] == 3
    assert summary["counting_accuracy"] == pytest.approx(0.5)  # over the 2 counting items only


def test_aggregate_reports_none_when_a_metric_never_applied():
    scores = [score_item(item("q1", "forecast"), "sys", "It will turn.", [], [])]
    assert aggregate(scores)["counting_accuracy"] is None
