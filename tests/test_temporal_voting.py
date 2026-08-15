"""Tests for M6 confidence-weighted temporal voting."""
from __future__ import annotations

import pytest

from src.semantics.temporal_voting import _evidence_weight, _vote_on_attribute
from src.utils.config import get_settings
from src.utils.schemas import FrameAttributes


def obs(
    color: str | None = None,
    *,
    crop_quality: float = 1.0,
    occlusion: float = 0.0,
    from_retry: bool = False,
    track_id: str = "1",
    frame_id: str = "f0",
    **fields,
) -> FrameAttributes:
    return FrameAttributes(
        track_id=track_id,
        frame_id=frame_id,
        color=color,
        crop_quality=crop_quality,
        occlusion=occlusion,
        from_retry=from_retry,
        **fields,
    )


def vote(observations, attribute="color"):
    distribution, _ = _vote_on_attribute(attribute, observations, get_settings())
    return distribution


# --- weighting ------------------------------------------------------------


def test_one_clear_crop_outvotes_several_poor_ones():
    """The whole point of weighting: a legible crop beats a numerical majority
    of tiny, blurry, half-occluded ones."""
    observations = [
        obs("white", crop_quality=1.0),
        obs("silver", crop_quality=0.05),
        obs("silver", crop_quality=0.05),
        obs("silver", crop_quality=0.05),
    ]
    result = vote(observations)
    assert result.winner == "white"
    assert result.n_votes == 4
    assert result.distribution["white"] > result.distribution["silver"]


def test_raw_majority_wins_when_quality_is_equal():
    observations = [obs("white"), obs("silver"), obs("silver")]
    result = vote(observations)
    assert result.winner == "silver"
    assert result.distribution["silver"] == pytest.approx(2 / 3)


def test_occlusion_discounts_a_vote():
    clear = _evidence_weight(obs("white", occlusion=0.0), get_settings())
    hidden = _evidence_weight(obs("white", occlusion=0.75), get_settings())
    assert hidden == pytest.approx(clear * 0.25)


def test_retry_derived_value_carries_less_weight():
    settings = get_settings()
    primary = _evidence_weight(obs("white"), settings)
    retried = _evidence_weight(obs("white", from_retry=True), settings)
    assert retried == pytest.approx(primary * settings.voting.retry_confidence)


def test_zero_quality_crops_degrade_to_plain_counting():
    """All-zero weights must not produce a division by zero or an empty winner."""
    observations = [obs("white", crop_quality=0.0), obs("white", crop_quality=0.0), obs("silver", crop_quality=0.0)]
    result = vote(observations)
    assert result.winner == "white"
    assert result.distribution["white"] == pytest.approx(2 / 3)


# --- distribution ---------------------------------------------------------


def test_distribution_sums_to_one_and_is_ordered_by_share():
    observations = [obs("white")] * 7 + [obs("silver")] * 2 + [obs("gray")]
    result = vote(observations)
    assert sum(result.distribution.values()) == pytest.approx(1.0)
    assert list(result.distribution) == ["white", "silver", "gray"]
    assert result.distribution["white"] == pytest.approx(0.7)


def test_ties_are_broken_deterministically_by_value_name():
    observations = [obs("silver"), obs("gray")]
    first = vote(observations)
    second = vote(list(reversed(observations)))
    assert first.winner == second.winner == "gray"  # alphabetical, not input order


# --- uncertainty ----------------------------------------------------------


def test_no_evidence_yields_no_winner():
    result = vote([obs(None), obs(None)])
    assert result.winner is None
    assert result.uncertain is True
    assert result.uncertain_reason == "no_evidence"
    assert result.n_votes == 0


def test_too_few_observations_is_uncertain_even_when_unanimous():
    result = vote([obs("white"), obs("white")])  # min_evidence_count is 3
    assert result.winner == "white"
    assert result.uncertain is True
    assert result.uncertain_reason == "insufficient_evidence"
    assert result.margin == pytest.approx(1.0)


def test_narrow_margin_is_uncertain():
    observations = [obs("white")] * 5 + [obs("silver")] * 5
    result = vote(observations)
    # Dead heat, so the alphabetical tie-break decides — but the point is that
    # the result is flagged uncertain rather than presented as a finding.
    assert result.winner == "silver"
    assert result.uncertain is True
    assert result.uncertain_reason == "low_margin"
    assert result.margin == pytest.approx(0.0)


def test_margin_just_under_threshold_is_uncertain():
    # 11 vs 9 -> margin 0.10, below the 0.15 threshold.
    observations = [obs("white")] * 11 + [obs("silver")] * 9
    result = vote(observations)
    assert result.winner == "white"
    assert result.margin == pytest.approx(0.1)
    assert result.uncertain_reason == "low_margin"


def test_clear_winner_with_enough_evidence_is_certain():
    observations = [obs("white")] * 5 + [obs("silver")]
    result = vote(observations)
    assert result.winner == "white"
    assert result.uncertain is False
    assert result.uncertain_reason is None


# --- normalization --------------------------------------------------------


def test_non_canonical_values_are_normalized_and_counted():
    """A backend emitting raw strings still lands in the closed vocabulary."""
    observations = [obs("off-white"), obs("cream"), obs("white")]
    distribution, n_normalized = _vote_on_attribute("color", observations, get_settings())
    assert distribution.winner == "white"
    assert distribution.distribution == {"white": 1.0}
    assert n_normalized == 2


def test_values_outside_the_vocabulary_are_not_evidence():
    observations = [obs("white"), obs("chartreuse"), obs("white")]
    result = vote(observations)
    assert result.n_votes == 2  # "chartreuse" discarded, not passed through
    assert result.distribution == {"white": 1.0}


def test_free_text_model_attribute_bypasses_vocabulary():
    observations = [
        obs(model="Corolla"), obs(model="corolla"), obs(model="Corolla"),
    ]
    result = vote(observations, attribute="model")
    assert result.winner == "corolla"  # lowercased, but not vocabulary-filtered
    assert result.n_votes == 3
