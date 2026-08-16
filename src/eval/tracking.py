"""M16 — Tracking / identity metrics: MOTA, IDF1, ID switches, fragmentation.

Standard CLEAR-MOT and IDF1 definitions, implemented directly rather than
pulled from a library, so the exact matching rules are visible and testable.

MOTA = 1 - (FN + FP + IDSW) / |GT|
    Per-frame greedy-optimal IoU matching of predictions to ground truth.
    Unmatched GT boxes are false negatives, unmatched predictions are false
    positives, and a matched GT whose predicted id differs from the id it was
    matched to last time is an ID switch. MOTA is deliberately unbounded below:
    a tracker emitting many false positives can score negative, which is
    correct and worth reporting honestly rather than clamping to zero.

IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
    Identity-level, NOT frame-level. It asks "over the whole sequence, how well
    does one predicted trajectory cover one true trajectory", by solving a
    single global assignment between GT ids and predicted ids that maximises
    total overlap. That global step is why IDF1 punishes an identity that
    fragments across several predicted ids, while MOTA -- which only ever looks
    one frame back -- charges a fragmented identity just one ID switch.
    Reporting both is the point: they disagree in exactly the way M4 and M7
    are designed to fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_IOU_THRESHOLD = 0.5


@dataclass
class TrackingMetrics:
    mota: float
    idf1: float
    id_switches: int
    fragmentations: int
    false_positives: int
    false_negatives: int
    true_positives: int
    gt_box_count: int
    pred_box_count: int
    gt_track_count: int
    pred_track_count: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "MOTA": round(self.mota, 4),
            "IDF1": round(self.idf1, 4),
            "IDSW": self.id_switches,
            "Frag": self.fragmentations,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "GT tracks": self.gt_track_count,
            "Pred tracks": self.pred_track_count,
        }


@dataclass
class FrameBoxes:
    """One frame's boxes, keyed by identity."""

    frame_id: str
    boxes: dict[str, list[float]] = field(default_factory=dict)


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_frame(
    gt: dict[str, list[float]], pred: dict[str, list[float]], iou_threshold: float
) -> list[tuple[str, str]]:
    """Optimal IoU matching for one frame, as (gt_id, pred_id) pairs.

    Hungarian rather than greedy: greedy can strand a pair that would have
    matched, which would inflate both FP and FN on crowded frames.
    """
    if not gt or not pred:
        return []
    gt_ids, pred_ids = sorted(gt), sorted(pred)
    cost = np.zeros((len(gt_ids), len(pred_ids)))
    for i, g in enumerate(gt_ids):
        for j, p in enumerate(pred_ids):
            cost[i, j] = -iou(gt[g], pred[p])

    rows, cols = linear_sum_assignment(cost)
    return [
        (gt_ids[r], pred_ids[c])
        for r, c in zip(rows, cols)
        if -cost[r, c] >= iou_threshold
    ]


def _idf1(
    gt_frames: list[FrameBoxes], pred_frames: list[FrameBoxes], iou_threshold: float
) -> tuple[float, int, int, int]:
    """Global identity assignment. Returns (idf1, idtp, idfp, idfn)."""
    gt_ids = sorted({i for f in gt_frames for i in f.boxes})
    pred_ids = sorted({i for f in pred_frames for i in f.boxes})
    gt_counts = {i: sum(1 for f in gt_frames for k in f.boxes if k == i) for i in gt_ids}
    pred_counts = {i: sum(1 for f in pred_frames for k in f.boxes if k == i) for i in pred_ids}
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())

    if not gt_ids or not pred_ids:
        return 0.0, 0, total_pred, total_gt

    pred_by_frame = {f.frame_id: f.boxes for f in pred_frames}
    overlap = np.zeros((len(gt_ids), len(pred_ids)))
    for frame in gt_frames:
        preds = pred_by_frame.get(frame.frame_id, {})
        for gi, g in enumerate(gt_ids):
            if g not in frame.boxes:
                continue
            for pj, p in enumerate(pred_ids):
                if p in preds and iou(frame.boxes[g], preds[p]) >= iou_threshold:
                    overlap[gi, pj] += 1

    rows, cols = linear_sum_assignment(-overlap)
    idtp = int(sum(overlap[r, c] for r, c in zip(rows, cols)))
    idfp = total_pred - idtp
    idfn = total_gt - idtp
    denominator = 2 * idtp + idfp + idfn
    return (2 * idtp / denominator if denominator else 0.0), idtp, idfp, idfn


def evaluate_tracking(
    gt_frames: list[FrameBoxes],
    pred_frames: list[FrameBoxes],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> TrackingMetrics:
    """Both sequences must cover the same frame_ids, in the same order."""
    pred_by_frame = {f.frame_id: f.boxes for f in pred_frames}

    tp = fp = fn = idsw = 0
    last_match: dict[str, str] = {}  # gt_id -> pred_id it matched to last time
    seen_since_gap: dict[str, bool] = {}
    fragmentations = 0
    gt_box_count = pred_box_count = 0

    for frame in gt_frames:
        preds = pred_by_frame.get(frame.frame_id, {})
        gt_box_count += len(frame.boxes)
        pred_box_count += len(preds)

        matches = match_frame(frame.boxes, preds, iou_threshold)
        matched_gt = {g for g, _ in matches}
        matched_pred = {p for _, p in matches}

        tp += len(matches)
        fn += len(frame.boxes) - len(matched_gt)
        fp += len(preds) - len(matched_pred)

        for gt_id, pred_id in matches:
            if gt_id in last_match and last_match[gt_id] != pred_id:
                idsw += 1
            # Fragmentation: this identity was tracked, then lost, now tracked
            # again -- counted once per resumption, independent of whether the
            # id changed (an identity can fragment and still be re-assigned the
            # same id, which is a recovery MOTA's IDSW count would never show).
            if seen_since_gap.get(gt_id) is False:
                fragmentations += 1
            last_match[gt_id] = pred_id
            seen_since_gap[gt_id] = True

        for gt_id in frame.boxes:
            if gt_id not in matched_gt and seen_since_gap.get(gt_id):
                seen_since_gap[gt_id] = False

    idf1, _, _, _ = _idf1(gt_frames, pred_frames, iou_threshold)
    mota = 1.0 - (fn + fp + idsw) / gt_box_count if gt_box_count else 0.0

    return TrackingMetrics(
        mota=mota,
        idf1=idf1,
        id_switches=idsw,
        fragmentations=fragmentations,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        gt_box_count=gt_box_count,
        pred_box_count=pred_box_count,
        gt_track_count=len({i for f in gt_frames for i in f.boxes}),
        pred_track_count=len({i for f in pred_frames for i in f.boxes}),
    )
