"""End-to-end tests for M8's reconciliation, with a fake VLM backend.

Each case drives one cell of the reconciliation matrix through the real
confirm_video() path. Fixtures are written under the real data/ tree and
cleaned up unconditionally.
"""
from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pytest

import src.semantics.vlm_extractor as vlm_extractor
from src.semantics.best_shot_confirmation import confirm_video
from src.utils.config import get_settings

VIDEO_ID = "video_test_m8"
CLIP_ID = "clip_test_m8"
GLOBAL_ID = "obj_0001"


class FakeBackend:
    """Returns one fixed caption for every crop."""

    def __init__(self, caption: str):
        self.caption_text = caption
        self.tasks_seen: list[str] = []

    def caption(self, crops_bgr, task):
        self.tasks_seen.append(task)
        return [self.caption_text] * len(crops_bgr)


@pytest.fixture
def settings():
    cfg = get_settings()
    yield cfg
    outputs = cfg.resolve_path(cfg.paths.outputs_dir)
    for sub, name in (
        ("master_object_index", f"{VIDEO_ID}.json"),
        ("global_objects_final", f"{VIDEO_ID}.json"),
        ("tracks_associated", f"{CLIP_ID}.json"),
        ("attributes_canonical", f"{CLIP_ID}.json"),
        ("attributes_raw", f"{CLIP_ID}.json"),
    ):
        (outputs / sub / name).unlink(missing_ok=True)
    shutil.rmtree(cfg.resolve_path(cfg.paths.crops_dir) / CLIP_ID, ignore_errors=True)
    shutil.rmtree(cfg.resolve_path(cfg.vlm.cache_dir), ignore_errors=True)


def write_fixture(settings, clip_level: dict[str, tuple[str | None, bool]], n_crops: int = 3):
    """clip_level maps attribute -> (winner, uncertain) as M6 would have voted."""
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    crops_dir = settings.resolve_path(settings.paths.crops_dir) / CLIP_ID / "1"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    crop_paths = []
    for i in range(n_crops):
        frame_id = f"frame_{i:06d}"
        path = crops_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(path), (rng.random((120, 130, 3)) * 255).astype(np.uint8))
        crop_paths.append(f"data/crops/{CLIP_ID}/1/{frame_id}.jpg")

    for sub in ("master_object_index", "tracks_associated", "attributes_canonical", "attributes_raw"):
        (outputs / sub).mkdir(parents=True, exist_ok=True)

    (outputs / "master_object_index" / f"{VIDEO_ID}.json").write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "objects": [
                    {
                        "global_id": GLOBAL_ID,
                        "class": "car",
                        "sightings": [
                            {
                                "clip_id": CLIP_ID, "track_id": "1",
                                "first_seen": "2026-08-15T10:00:00",
                                "last_seen": "2026-08-15T10:00:05", "gate_scores": {},
                            }
                        ],
                    }
                ],
                "rejected_links": [], "suppressed_links": [],
            }
        )
    )

    (outputs / "tracks_associated" / f"{CLIP_ID}.json").write_text(
        json.dumps(
            {
                "clip_id": CLIP_ID,
                "tracks": [
                    {
                        "track_id": "1", "class": "car",
                        "frames": [f"frame_{i:06d}" for i in range(n_crops)],
                        "bboxes": [[0.0, 0.0, 130.0, 120.0]] * n_crops,
                        "centers": [[65.0, 60.0]] * n_crops,
                        "velocity": [[0.0, 0.0]] * n_crops,
                        "dominant_direction_deg": None, "embedding": [1.0, 0.0],
                        "best_shot_crops": crop_paths,
                        "best_shot_scores": [1.0] * n_crops,
                    }
                ],
                "merge_log": [], "rejected_links": [], "suppressed_links": [],
            }
        )
    )

    (outputs / "attributes_canonical" / f"{CLIP_ID}.json").write_text(
        json.dumps(
            {
                "clip_id": CLIP_ID,
                "tracks": [
                    {
                        "track_id": "1",
                        "votes": [
                            {
                                "attribute": attribute, "winner": winner,
                                "distribution": {winner: 1.0} if winner else {},
                                "uncertain": uncertain, "n_votes": 0 if winner is None else 8,
                                "margin": 1.0, "uncertain_reason": None,
                            }
                            for attribute, (winner, uncertain) in clip_level.items()
                        ],
                    }
                ],
            }
        )
    )

    (outputs / "attributes_raw" / f"{CLIP_ID}.json").write_text(
        json.dumps({"clip_id": CLIP_ID, "attributes": []})
    )
    return crop_paths


def run(settings, monkeypatch, caption: str, clip_level: dict, n_crops: int = 3):
    write_fixture(settings, clip_level, n_crops)
    fake = FakeBackend(caption)
    monkeypatch.setattr(vlm_extractor, "_load_backend", lambda s: fake)
    index = confirm_video(VIDEO_ID, settings=settings)
    return index, fake


def attribute_of(index, name: str):
    return next(a for a in index.objects[0].attributes if a.attribute == name)


def outcome_of(index, name: str):
    entries = [e for e in index.correction_log if e.attribute == name]
    return entries[0].outcome if entries else None


# --- the reconciliation matrix -------------------------------------------


def test_agreement_confirms_and_boosts_confidence(settings, monkeypatch):
    index, _ = run(
        settings, monkeypatch, "a white sedan",
        {"color": ("white", False), "vehicle_type": ("sedan", False)},
    )
    colour = attribute_of(index, "color")
    assert colour.value == "white"
    assert colour.source == "agreed"
    assert outcome_of(index, "color") == "confirmed"
    # 1.0 clip confidence + boost, clamped.
    assert colour.confidence == pytest.approx(1.0)


def test_best_shot_fills_an_uncertain_clip_level_answer(settings, monkeypatch):
    """M8's most valuable case given how much Florence-2 leaves unanswered."""
    index, _ = run(
        settings, monkeypatch, "a white toyota sedan",
        {"color": ("white", False), "make": (None, True)},
    )
    make = attribute_of(index, "make")
    assert make.value == "toyota"
    assert make.source == "best_shot"
    assert make.uncertain is False
    assert outcome_of(index, "make") == "filled"


def test_best_shot_overturns_clip_level_on_a_fine_attribute(settings, monkeypatch):
    """make/model need pixels, so the sharpest view wins a disagreement."""
    index, _ = run(
        settings, monkeypatch, "a white honda sedan",
        {"color": ("white", False), "make": ("toyota", False)},
    )
    make = attribute_of(index, "make")
    assert make.value == "honda"
    assert make.source == "best_shot"
    assert outcome_of(index, "make") == "corrected"


def test_clip_level_overrules_best_shot_on_a_coarse_attribute(settings, monkeypatch):
    """colour is robust across frames; one crop can be fooled by glare."""
    index, _ = run(
        settings, monkeypatch, "a blue sedan",
        {"color": ("white", False), "vehicle_type": ("sedan", False)},
    )
    colour = attribute_of(index, "color")
    assert colour.value == "white"
    assert colour.source == "clip_voting"
    assert outcome_of(index, "color") == "retained"
    # Disagreement costs confidence even though clip-level won.
    assert colour.confidence == pytest.approx(1.0 * (1 - settings.confirmation.disagreement_penalty))


def test_direction_is_never_reconciled(settings, monkeypatch):
    """State, not identity: a single frame shows orientation, not travel."""
    index, _ = run(
        settings, monkeypatch, "a white sedan driving away from the camera",
        {"color": ("white", False), "direction": ("left", False)},
    )
    direction = attribute_of(index, "direction")
    assert direction.value == "left"  # M6's trajectory-derived answer survives
    assert direction.source == "clip_voting"
    assert outcome_of(index, "direction") is None  # not in the correction log


def test_silent_best_shot_leaves_clip_level_untouched(settings, monkeypatch):
    index, _ = run(
        settings, monkeypatch, "an indescribable object",
        {"color": ("white", False)},
    )
    colour = attribute_of(index, "color")
    assert colour.value == "white"
    assert colour.source == "clip_voting"
    assert outcome_of(index, "color") is None


# --- mechanics ------------------------------------------------------------


def test_uses_the_high_detail_task_not_the_standard_one(settings, monkeypatch):
    _, fake = run(settings, monkeypatch, "a white sedan", {"color": ("white", False)})
    _, high_detail = vlm_extractor.BACKEND_TASKS["florence2"]
    assert set(fake.tasks_seen) == {high_detail}


def test_reads_at_most_top_k_crops(settings, monkeypatch):
    index, _ = run(
        settings, monkeypatch, "a white sedan", {"color": ("white", False)}, n_crops=7
    )
    assert len(index.objects[0].best_shot_crops) == settings.confirmation.top_k


def test_correction_log_separates_filled_from_corrected(settings, monkeypatch):
    """The paper reports these as different claims, so they must not be merged."""
    index, _ = run(
        settings, monkeypatch, "a white honda sedan",
        {"color": ("white", False), "make": ("toyota", False), "model": (None, True)},
    )
    outcomes = {e.attribute: e.outcome for e in index.correction_log}
    assert outcomes["make"] == "corrected"  # overturned a confident answer
    assert outcomes["color"] == "confirmed"
    assert "model" not in outcomes  # nothing to fill it with
