"""Unit tests for M5's deterministic parts: caption -> field extraction,
frame sampling, confidence weighting, and the content-addressed cache.
Does not load a VLM backend.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.semantics.vlm_extractor import (
    _cache_path,
    _caption_to_fields,
    _confidence,
    _crop_hash,
    _load_cached_fields,
    _save_cached_fields,
    _select_sample_indices,
)
from src.semantics.vocabulary import COLOR_VOCAB, VEHICLE_TYPE_VOCAB, match_vocab


def test_match_vocab_prefers_longer_phrase_first():
    # "pickup truck" must win over a bare "truck" match for the same caption.
    assert match_vocab("a red pickup truck on the road", VEHICLE_TYPE_VOCAB) == "pickup"


def test_match_vocab_word_boundary_avoids_false_substring_match():
    # "tan" must not match inside "instant" or similar.
    assert match_vocab("an instant classic car", COLOR_VOCAB) is None
    assert match_vocab("a tan sedan", COLOR_VOCAB) == "brown"


def test_caption_to_fields_extracts_multiple_attributes():
    caption = "A white Toyota sedan is parked on the side of the street."
    fields, n_found = _caption_to_fields(caption)
    assert fields["color"] == "white"
    assert fields["vehicle_type"] == "sedan"
    assert fields["make"] == "toyota"
    assert fields["direction"] == "stationary"
    assert fields["model"] is None
    assert n_found == 4


def test_caption_to_fields_empty_on_no_matches():
    fields, n_found = _caption_to_fields("An indescribable object of unknown provenance.")
    assert n_found == 0
    assert all(v is None for v in fields.values())


def test_confidence_matches_weighted_sum():
    fields = {"color": "white", "vehicle_type": "sedan", "make": None, "direction": None, "model": None}
    assert _confidence(fields) == pytest.approx(0.4 + 0.3)


def test_confidence_zero_when_nothing_found():
    fields = {"color": None, "vehicle_type": None, "make": None, "direction": None, "model": None}
    assert _confidence(fields) == 0.0


@pytest.mark.parametrize(
    "n_available,n_samples,expected_len",
    [(3, 8, 3), (8, 8, 8), (20, 8, 8), (1, 8, 1)],
)
def test_select_sample_indices_spans_track_lifetime(n_available, n_samples, expected_len):
    indices = _select_sample_indices(n_available, n_samples)
    assert len(indices) == expected_len
    assert indices == sorted(set(indices))  # deduped, ascending
    assert indices[0] == 0
    if n_available > 1:
        assert indices[-1] == n_available - 1  # last frame always represented


def test_crop_hash_is_stable_and_content_sensitive():
    crop_a = np.zeros((10, 10, 3), dtype=np.uint8)
    crop_a_copy = np.zeros((10, 10, 3), dtype=np.uint8)
    crop_b = np.ones((10, 10, 3), dtype=np.uint8)

    assert _crop_hash(crop_a, "<CAPTION>") == _crop_hash(crop_a_copy, "<CAPTION>")
    assert _crop_hash(crop_a, "<CAPTION>") != _crop_hash(crop_b, "<CAPTION>")


def test_crop_hash_separates_tasks():
    """M8 asks for a high-detail caption of crops M5 may already have cached
    under the standard task; without the task in the key it would silently be
    served the wrong, less detailed result."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    assert _crop_hash(crop, "<DETAILED_CAPTION>") != _crop_hash(crop, "<MORE_DETAILED_CAPTION>")


def test_cache_round_trip(tmp_path):
    crop_hash = "deadbeef" * 8
    fields = {"color": "white", "vehicle_type": "sedan", "make": None, "direction": None, "model": None}

    assert _load_cached_fields(tmp_path, crop_hash) is None
    _save_cached_fields(tmp_path, crop_hash, fields, caption="a white sedan", from_retry=True)
    # Provenance must survive the cache, or a rerun would silently upgrade a
    # retry-derived value to full weight in M6.
    assert _load_cached_fields(tmp_path, crop_hash) == (fields, True)

    # Sharded by first 2 hex chars, so a flat cache dir never gets huge.
    assert _cache_path(tmp_path, crop_hash).parent.name == crop_hash[:2]
