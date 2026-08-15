"""Unit tests for M8's scoring and clip-level aggregation."""
from __future__ import annotations

import numpy as np
import pytest

from src.semantics.best_shot_confirmation import (
    COARSE_ATTRIBUTES,
    EXCLUDED_ATTRIBUTES,
    FINE_ATTRIBUTES,
    _aggregate_clip_level,
    _frontality,
    _parse_crop_path,
    _score_crop,
)
from src.utils.config import get_settings


def textured(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A crop with real high-frequency content, so Laplacian variance is nonzero."""
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


# --- crop scoring ---------------------------------------------------------


def test_bigger_crop_scores_higher():
    settings = get_settings()
    assert _score_crop(textured(128, 128), 0.0, settings) > _score_crop(
        textured(24, 24), 0.0, settings
    )


def test_blurry_crop_scores_lower_than_sharp_one():
    settings = get_settings()
    sharp = textured(128, 128)
    blurry = cv2_blur(sharp)
    assert _score_crop(blurry, 0.0, settings) < _score_crop(sharp, 0.0, settings)


def cv2_blur(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(image, (15, 15), 0)


def test_occlusion_discounts_the_score():
    settings = get_settings()
    crop = textured(128, 128)
    assert _score_crop(crop, 0.75, settings) == pytest.approx(
        _score_crop(crop, 0.0, settings) * 0.25
    )


def test_degenerate_crop_scores_zero():
    assert _score_crop(np.zeros((1, 1, 3), dtype=np.uint8), 0.0, get_settings()) == 0.0


# --- viewpoint proxy ------------------------------------------------------


def test_frontality_prefers_square_crops_over_wide_ones():
    settings = get_settings()
    frontal = _frontality(np.zeros((100, 110, 3), dtype=np.uint8), settings)  # aspect 1.1
    side_on = _frontality(np.zeros((100, 250, 3), dtype=np.uint8), settings)  # aspect 2.5
    assert frontal == pytest.approx(1.0)
    assert side_on == pytest.approx(0.0)
    middle = _frontality(np.zeros((100, 170, 3), dtype=np.uint8), settings)  # aspect 1.7
    assert 0.0 < middle < 1.0


def test_viewpoint_weight_zero_disables_the_preference(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.confirmation, "viewpoint_weight", 0.0)
    square = textured(100, 100, seed=1)
    wide = textured(100, 100, seed=1)  # same pixels, so only viewpoint could differ
    assert _score_crop(square, 0.0, settings) == _score_crop(wide, 0.0, settings)


# --- crop path parsing ----------------------------------------------------


def test_parse_crop_path_recovers_identifiers():
    assert _parse_crop_path("data/crops/clip_000/7/frame_000123.jpg") == (
        "clip_000",
        "7",
        "frame_000123",
    )


def test_parse_crop_path_rejects_unexpected_shape():
    assert _parse_crop_path("frame.jpg") is None


# --- clip-level aggregation ----------------------------------------------


def test_aggregation_weights_sightings_by_evidence():
    """A sighting backed by 20 frames must outweigh one backed by 3."""
    value, confidence, support = _aggregate_clip_level(
        "color", [("white", False, 20), ("silver", False, 3)]
    )
    assert value == "white"
    assert support == 23
    assert confidence == pytest.approx(20 / 23)


def test_aggregation_ignores_uncertain_sightings():
    value, confidence, support = _aggregate_clip_level(
        "color", [("white", False, 5), ("blue", True, 40)]
    )
    assert value == "white"  # the 40-vote sighting was uncertain, so it abstains
    assert support == 5
    assert confidence == pytest.approx(1.0)


def test_aggregation_returns_nothing_when_all_uncertain():
    assert _aggregate_clip_level("make", [("toyota", True, 9), (None, True, 0)]) == (None, 0.0, 0)


def test_aggregation_tie_breaks_deterministically():
    first = _aggregate_clip_level("color", [("white", False, 5), ("silver", False, 5)])
    second = _aggregate_clip_level("color", [("silver", False, 5), ("white", False, 5)])
    assert first == second
    assert first[0] == "silver"  # alphabetical, not input order


# --- the attribute split --------------------------------------------------


def test_attribute_classes_are_disjoint_and_direction_is_excluded():
    assert set(FINE_ATTRIBUTES).isdisjoint(COARSE_ATTRIBUTES)
    assert "direction" in EXCLUDED_ATTRIBUTES
    assert "direction" not in FINE_ATTRIBUTES + COARSE_ATTRIBUTES
