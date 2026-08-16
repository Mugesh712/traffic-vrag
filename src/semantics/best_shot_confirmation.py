"""M8 — Global best-shot VLM confirmation (Contribution #2, Part 2).

Takes each global object from M7, picks the best crops of it across ALL its
clips, re-reads them with a high-detail prompt, and reconciles the answer
against M6's clip-level canonical attributes.

WHY A SECOND VLM PASS AT ALL. M5/M6 see a track through many mediocre crops
and vote. That is the right instrument for coarse attributes and the wrong one
for fine ones: no amount of voting over 40-pixel-wide crops can read a badge.
M8 spends its budget in the opposite way -- a handful of the sharpest, largest,
least-occluded views of the object, at the highest detail setting.

RECONCILIATION IS ASYMMETRIC, and this is the contribution:

    agree                       -> confirm, boost confidence
    clip-level uncertain        -> adopt the best-shot value      (filled)
    disagree on make / model    -> best shot wins                 (corrected)
    disagree on colour / type   -> clip-level voting wins         (retained)
    direction                   -> never touched

Coarse attributes are robust and *gain* from many votes -- one crop can be
fooled by shadow or headlight glare, so a majority across frames is better
evidence than any single view. Fine attributes need pixels, so voting over
mostly-poor crops just averages noise and the one crisp crop is the only view
that could ever resolve a make. Evidence quantity wins for coarse, evidence
quality wins for fine.

`direction` is excluded on the same identity/state grounds as M7: a single
frame shows instantaneous orientation, not direction of travel, which M3
already derives from the whole trajectory.

The correction log separates `filled` (answered where M6 was uncertain) from
`corrected` (overturned a confident answer). They are different claims and
inflating one total with the other would overstate the result.

Output: data/outputs/global_objects_final/<video_id>.json (FinalObjectIndex)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.semantics.temporal_voting import _vote_on_attribute
from src.semantics.vlm_extractor import caption_crops_to_fields, get_tasks
from src.semantics.vocabulary import ATTRIBUTES
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import (
    ClipAssociatedTracks,
    ClipCanonicalAttributes,
    ClipRawAttributes,
    CorrectionLogEntry,
    FinalAttribute,
    FinalObjectIndex,
    FrameAttributes,
    GlobalObjectFinal,
    MasterObjectIndex,
)

logger = get_logger(__name__)

# Fine-grained attributes need resolution, so the best shot decides them.
FINE_ATTRIBUTES = ("make", "model")
# Coarse attributes are robust across frames, so clip-level voting decides them.
COARSE_ATTRIBUTES = ("color", "vehicle_type")
# State, not identity — never reconciled from a single frame. See module docstring.
EXCLUDED_ATTRIBUTES = ("direction",)


class BestShotConfirmationError(RuntimeError):
    pass


@dataclass
class _CropCandidate:
    path: str
    clip_id: str
    track_id: str
    frame_id: str
    score: float
    image: np.ndarray


def _frontality(crop: np.ndarray, settings: PipelineSettings) -> float:
    """1.0 for a frontal-looking crop, 0.0 for a side-on one.

    A proxy from aspect ratio, not a viewpoint classifier: a car seen head-on
    is roughly as wide as it is tall, while a side view is much wider. Good
    enough to bias crop choice toward views where a badge could be legible,
    which is what M8 is actually trying to read.
    """
    h, w = crop.shape[:2]
    if h <= 0:
        return 0.0
    aspect = w / h
    cfg = settings.confirmation
    span = cfg.side_aspect_ratio - cfg.frontal_aspect_ratio
    if span <= 0:
        return 1.0
    return float(np.clip((cfg.side_aspect_ratio - aspect) / span, 0.0, 1.0))


def _score_crop(crop: np.ndarray, occlusion: float, settings: PipelineSettings) -> float:
    cfg = settings.confirmation
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    resolution = min(1.0, float(np.sqrt(h * w)) / cfg.quality_reference_size_px)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / cfg.quality_reference_sharpness)
    # Blend rather than multiply, so viewpoint_weight=0 disables the term.
    viewpoint = (1.0 - cfg.viewpoint_weight) + cfg.viewpoint_weight * _frontality(crop, settings)
    return resolution * sharpness * (1.0 - occlusion) * viewpoint


def _parse_crop_path(path: str) -> tuple[str, str, str] | None:
    """data/crops/<clip_id>/<track_id>/<frame_id>.jpg -> (clip, track, frame)."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return None
    return parts[-3], parts[-2], parts[-1].rsplit(".", 1)[0]


def _aggregate_clip_level(
    attribute: str, per_sighting: list[tuple[str | None, bool, int]]
) -> tuple[str | None, float, int]:
    """Collapse M6's per-clip answers for one global object into one view.

    Support is summed vote counts, so a sighting backed by 20 frames outweighs
    one backed by 3. Uncertain sightings contribute nothing rather than voting
    with a value the system does not stand behind.

    Returns (value, confidence, total_support).
    """
    support: dict[str, int] = {}
    for value, uncertain, n_votes in per_sighting:
        if value is None or uncertain:
            continue
        support[value] = support.get(value, 0) + max(n_votes, 1)

    if not support:
        return None, 0.0, 0

    total = sum(support.values())
    value = sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return value, support[value] / total, total


def confirm_video(video_id: str, settings: PipelineSettings | None = None) -> FinalObjectIndex:
    settings = settings or get_settings()
    outputs_dir = settings.resolve_path(settings.paths.outputs_dir)
    cfg = settings.confirmation

    index_path = outputs_dir / "master_object_index" / f"{video_id}.json"
    if not index_path.exists():
        raise BestShotConfirmationError(
            f"No master object index at {index_path}; run `link` first."
        )
    master = MasterObjectIndex.model_validate_json(index_path.read_text())

    # Per-clip inputs, loaded once: M4 for the crop shortlist, M6 for the
    # clip-level answers, M5 for the occlusion measured at crop time.
    tracks_by_clip: dict[str, dict[str, object]] = {}
    canonical_by_clip: dict[str, dict[str, dict[str, tuple[str | None, bool, int]]]] = {}
    occlusion_by_key: dict[tuple[str, str, str], float] = {}

    clip_ids = sorted({s.clip_id for o in master.objects for s in o.sightings})
    for clip_id in clip_ids:
        tracks_path = outputs_dir / "tracks_associated" / f"{clip_id}.json"
        if tracks_path.exists():
            clip_tracks = ClipAssociatedTracks.model_validate_json(tracks_path.read_text())
            tracks_by_clip[clip_id] = {t.track_id: t for t in clip_tracks.tracks}

        canonical_path = outputs_dir / "attributes_canonical" / f"{clip_id}.json"
        if canonical_path.exists():
            canonical = ClipCanonicalAttributes.model_validate_json(canonical_path.read_text())
            canonical_by_clip[clip_id] = {
                t.track_id: {v.attribute: (v.winner, v.uncertain, v.n_votes) for v in t.votes}
                for t in canonical.tracks
            }

        raw_path = outputs_dir / "attributes_raw" / f"{clip_id}.json"
        if raw_path.exists():
            raw = ClipRawAttributes.model_validate_json(raw_path.read_text())
            for obs in raw.attributes:
                occlusion_by_key[(clip_id, obs.track_id, obs.frame_id)] = obs.occlusion

    standard_task, high_detail_task = get_tasks(settings)

    final_objects: list[GlobalObjectFinal] = []
    correction_log: list[CorrectionLogEntry] = []
    n_with_best_shot = 0

    for obj in master.objects:
        # --- 1. Best shots across ALL clips of this object ------------------
        candidates: list[_CropCandidate] = []
        # Ablation: with M8 disabled, no crops are gathered and no VLM pass
        # runs, so every attribute falls through to M6's clip-level answer --
        # exactly the pre-M8 baseline the ablation needs to compare against.
        for sighting in obj.sightings if cfg.enabled else []:
            track = tracks_by_clip.get(sighting.clip_id, {}).get(sighting.track_id)
            if track is None:
                continue
            for crop_path in track.best_shot_crops:
                parsed = _parse_crop_path(crop_path)
                if parsed is None:
                    continue
                clip_id, track_id, frame_id = parsed
                image = cv2.imread(str(settings.resolve_path(crop_path)))
                if image is None:
                    logger.warning("confirm_video: missing crop %s, skipping", crop_path)
                    continue
                occlusion = occlusion_by_key.get((clip_id, track_id, frame_id), 0.0)
                candidates.append(
                    _CropCandidate(
                        path=crop_path,
                        clip_id=clip_id,
                        track_id=track_id,
                        frame_id=frame_id,
                        score=_score_crop(image, occlusion, settings),
                        image=image,
                    )
                )

        # Ties broken by path so the selection is reproducible.
        candidates.sort(key=lambda c: (-c.score, c.path))
        top = candidates[: cfg.top_k]
        if top:
            n_with_best_shot += 1

        # --- 2. High-detail VLM pass on the best shots ----------------------
        best_shot_votes: dict[str, tuple[str | None, bool]] = {}
        if top:
            # No retry task: this is already the most detailed caption available.
            results = caption_crops_to_fields(
                [c.image for c in top], settings, task=high_detail_task, retry_task=None
            )
            observations = [
                FrameAttributes(
                    track_id=obj.global_id,
                    frame_id=candidate.path,
                    **fields,
                    # The crop score is the vote weight; occlusion is already
                    # folded into it, so it must not be applied twice.
                    crop_quality=max(candidate.score, 0.0),
                    occlusion=0.0,
                    from_retry=from_retry,
                )
                for candidate, (fields, from_retry) in zip(top, results)
            ]
            for attribute in ATTRIBUTES:
                vote, _ = _vote_on_attribute(
                    attribute,
                    observations,
                    settings,
                    min_evidence_count=cfg.min_evidence_count,
                    margin_threshold=cfg.confidence_margin_threshold,
                )
                best_shot_votes[attribute] = (vote.winner, vote.uncertain)

        # --- 3. Reconcile against M6 ----------------------------------------
        attributes: list[FinalAttribute] = []
        for attribute in ATTRIBUTES:
            per_sighting = [
                canonical_by_clip.get(s.clip_id, {})
                .get(s.track_id, {})
                .get(attribute, (None, True, 0))
                for s in obj.sightings
            ]
            clip_value, clip_confidence, _ = _aggregate_clip_level(attribute, per_sighting)

            best_value, best_uncertain = best_shot_votes.get(attribute, (None, True))
            if best_uncertain:
                best_value = None

            if attribute in EXCLUDED_ATTRIBUTES:
                outcome, value, confidence, source = (
                    "unconfirmed",
                    clip_value,
                    clip_confidence,
                    "clip_voting" if clip_value else "unconfirmed",
                )
            elif best_value is None:
                outcome, value, confidence, source = (
                    "unconfirmed",
                    clip_value,
                    clip_confidence,
                    "clip_voting" if clip_value else "unconfirmed",
                )
            elif clip_value is None:
                outcome, value, confidence, source = ("filled", best_value, 1.0, "best_shot")
            elif clip_value == best_value:
                outcome = "confirmed"
                value = clip_value
                confidence = min(1.0, clip_confidence + cfg.confidence_boost)
                source = "agreed"
            elif attribute in FINE_ATTRIBUTES:
                outcome, value, confidence, source = ("corrected", best_value, 1.0, "best_shot")
            else:
                outcome = "retained"
                value = clip_value
                confidence = clip_confidence * (1.0 - cfg.disagreement_penalty)
                source = "clip_voting"

            attributes.append(
                FinalAttribute(
                    attribute=attribute,
                    value=value,
                    confidence=round(confidence, 6),
                    source=source,
                    uncertain=value is None,
                )
            )

            # Log anything the best-shot pass actually had a say in.
            if outcome in ("confirmed", "filled", "corrected", "retained"):
                correction_log.append(
                    CorrectionLogEntry(
                        global_id=obj.global_id,
                        attribute=attribute,
                        clip_level_value=clip_value,
                        best_shot_value=best_value,
                        final_value=value,
                        outcome=outcome,
                    )
                )

        final_objects.append(
            GlobalObjectFinal(
                global_id=obj.global_id,
                **{"class": obj.cls},
                sightings=obj.sightings,
                attributes=attributes,
                best_shot_crops=[c.path for c in top],
            )
        )

    final_index = FinalObjectIndex(
        video_id=video_id, objects=final_objects, correction_log=correction_log
    )

    output_dir = outputs_dir / "global_objects_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(final_index.model_dump_json(indent=2, by_alias=True))

    counts = {
        outcome: sum(1 for e in correction_log if e.outcome == outcome)
        for outcome in ("confirmed", "filled", "corrected", "retained")
    }
    logger.info(
        "confirm_video: video_id=%s %d objects (%d with best shots) — "
        "%d confirmed, %d filled, %d corrected, %d retained -> %s",
        video_id,
        len(final_objects),
        n_with_best_shot,
        counts["confirmed"],
        counts["filled"],
        counts["corrected"],
        counts["retained"],
        output_path,
    )

    return final_index
