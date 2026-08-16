"""M16 — Attribute metrics.

Two different questions, only one of which needs ground truth:

ACCURACY (needs GT) -- is the value right? Scored per attribute against human
annotation. An `uncertain` prediction is scored as an abstention, tracked
separately from a wrong answer, because a system that says "I don't know" has
not made the same mistake as one that confidently says "blue" about a white
car. Reporting only a single accuracy number would erase exactly the
distinction this project exists to make, so coverage (how often it answers) is
always reported alongside precision-when-answering.

CONSISTENCY (no GT needed) -- does one object keep one value? A global object
seen across several clips should not be white in one and blue in another. This
is measurable without any annotation because it is an internal-agreement
property, which makes it the one attribute metric available on unannotated
footage. It is necessary but not sufficient: a system that always answers
"white" is perfectly consistent and useless, so consistency is only meaningful
read together with accuracy or coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.ground_truth import GroundTruth
from src.semantics.vocabulary import ATTRIBUTES
from src.utils.schemas import ClipCanonicalAttributes, FinalObjectIndex

# `direction` is state, not identity (see M7/M8) -- an object legitimately
# changes it, so it is excluded from identity-attribute scoring.
SCORED_ATTRIBUTES = tuple(a for a in ATTRIBUTES if a != "direction")


@dataclass
class AttributeAccuracy:
    attribute: str
    correct: int = 0
    wrong: int = 0
    abstained: int = 0  # predicted, but flagged uncertain
    missing_gt: int = 0  # no annotation to score against

    @property
    def answered(self) -> int:
        return self.correct + self.wrong

    @property
    def precision_when_answering(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def coverage(self) -> float:
        total = self.answered + self.abstained
        return self.answered / total if total else 0.0

    def as_row(self) -> dict:
        return {
            "Attribute": self.attribute,
            "Correct": self.correct,
            "Wrong": self.wrong,
            "Abstained": self.abstained,
            "Precision": round(self.precision_when_answering, 4),
            "Coverage": round(self.coverage, 4),
        }


@dataclass
class ConsistencyResult:
    attribute: str
    consistent_objects: int = 0
    inconsistent_objects: int = 0
    single_sighting_objects: int = 0  # trivially consistent, excluded from the rate

    @property
    def rate(self) -> float:
        total = self.consistent_objects + self.inconsistent_objects
        return self.consistent_objects / total if total else 0.0

    def as_row(self) -> dict:
        return {
            "Attribute": self.attribute,
            "Consistent": self.consistent_objects,
            "Inconsistent": self.inconsistent_objects,
            "Rate": round(self.rate, 4),
            "Single-sighting (excl.)": self.single_sighting_objects,
        }


def evaluate_attribute_accuracy(
    final: FinalObjectIndex,
    ground_truth: GroundTruth,
    gt_id_by_global_id: dict[str, str],
) -> list[AttributeAccuracy]:
    """Score M8's final attributes against annotation.

    `gt_id_by_global_id` comes from the tracking match -- accuracy is only
    meaningful for objects that were correctly identified in the first place.
    """
    gt_attributes = ground_truth.attributes_by_id()
    results = {a: AttributeAccuracy(attribute=a) for a in SCORED_ATTRIBUTES}

    for obj in final.objects:
        gt_id = gt_id_by_global_id.get(obj.global_id)
        truth = gt_attributes.get(gt_id) if gt_id else None
        predicted = {a.attribute: a for a in obj.attributes}

        for attribute in SCORED_ATTRIBUTES:
            result = results[attribute]
            expected = getattr(truth, attribute, None) if truth else None
            if expected is None:
                result.missing_gt += 1
                continue
            prediction = predicted.get(attribute)
            if prediction is None or prediction.value is None or prediction.uncertain:
                result.abstained += 1
            elif prediction.value == expected:
                result.correct += 1
            else:
                result.wrong += 1

    return [results[a] for a in SCORED_ATTRIBUTES]


def evaluate_attribute_consistency(
    final: FinalObjectIndex,
    canonical_by_clip: dict[str, ClipCanonicalAttributes],
) -> list[ConsistencyResult]:
    """Does each global object hold one value per attribute across its clips?

    Compares the per-clip winners M6 produced for each of the object's
    sightings. Objects seen in only one clip cannot disagree with themselves,
    so they are counted separately rather than inflating the rate -- on
    single-clip footage that means the rate is legitimately undefined, and
    saying so is better than reporting a meaningless 100%.
    """
    results = {a: ConsistencyResult(attribute=a) for a in SCORED_ATTRIBUTES}

    winners_by_clip_track: dict[tuple[str, str], dict[str, tuple[str | None, bool]]] = {}
    for clip_id, canonical in canonical_by_clip.items():
        for track in canonical.tracks:
            winners_by_clip_track[(clip_id, track.track_id)] = {
                v.attribute: (v.winner, v.uncertain) for v in track.votes
            }

    for obj in final.objects:
        for attribute in SCORED_ATTRIBUTES:
            values = []
            for sighting in obj.sightings:
                per_track = winners_by_clip_track.get((sighting.clip_id, sighting.track_id), {})
                winner, uncertain = per_track.get(attribute, (None, True))
                # An uncertain sighting abstains rather than disagreeing --
                # the same permissive treatment M7's semantic gate gives it.
                if winner is not None and not uncertain:
                    values.append(winner)

            result = results[attribute]
            if len(values) < 2:
                result.single_sighting_objects += 1
            elif len(set(values)) == 1:
                result.consistent_objects += 1
            else:
                result.inconsistent_objects += 1

    return [results[a] for a in SCORED_ATTRIBUTES]


@dataclass
class CorrectionSummary:
    """M8's own correction log, aggregated. Needs no ground truth -- it counts
    what the best-shot pass *did*, not whether it was right."""

    confirmed: int = 0
    filled: int = 0
    corrected: int = 0
    retained: int = 0
    per_attribute: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_rows(self) -> list[dict]:
        return [
            {"Outcome": "Confirmed", "Count": self.confirmed,
             "Meaning": "best shot agreed with clip-level voting"},
            {"Outcome": "Filled", "Count": self.filled,
             "Meaning": "best shot answered where voting was uncertain"},
            {"Outcome": "Corrected", "Count": self.corrected,
             "Meaning": "best shot overturned a confident clip-level answer"},
            {"Outcome": "Retained", "Count": self.retained,
             "Meaning": "clip-level voting overruled a disagreeing best shot"},
        ]


def summarize_corrections(final: FinalObjectIndex) -> CorrectionSummary:
    summary = CorrectionSummary()
    for entry in final.correction_log:
        if hasattr(summary, entry.outcome):
            setattr(summary, entry.outcome, getattr(summary, entry.outcome) + 1)
        per_attr = summary.per_attribute.setdefault(entry.attribute, {})
        per_attr[entry.outcome] = per_attr.get(entry.outcome, 0) + 1
    return summary


def uncertainty_rate(canonical_by_clip: dict[str, ClipCanonicalAttributes]) -> dict[str, float]:
    """How often M6 declines to commit, per attribute. The paper's thesis is
    that admitting uncertainty beats confident hallucination, so the rate at
    which the system actually does so is a headline number, not a footnote."""
    totals: dict[str, int] = {}
    uncertain: dict[str, int] = {}
    for canonical in canonical_by_clip.values():
        for track in canonical.tracks:
            for vote in track.votes:
                totals[vote.attribute] = totals.get(vote.attribute, 0) + 1
                if vote.uncertain:
                    uncertain[vote.attribute] = uncertain.get(vote.attribute, 0) + 1
    return {
        attribute: round(uncertain.get(attribute, 0) / count, 4)
        for attribute, count in sorted(totals.items())
    }
