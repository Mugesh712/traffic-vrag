"""M6 — Clip-level temporal aggregation (Contribution #2, Part 1).

Collapses M5's many per-frame attribute guesses for one track into a single
canonical value per attribute, by confidence-weighted majority voting, and
refuses to guess when the evidence does not support a winner.

EVIDENCE WEIGHTING. Each observation votes with

    weight = attribute_confidence x crop_quality x (1 - occlusion)

  * attribute_confidence is provenance-based: 1.0 for a value the primary
    caption produced, `retry_confidence` (< 1) for one recovered only by the
    speculative retry caption. M5's `confidence` field is deliberately NOT
    used here — it scores caption *coverage* (how many fields were found),
    so weighting a colour vote by it would let a caption that happened to
    also name a make outvote one that saw the colour just as clearly.
  * crop_quality is size x sharpness, measured in M5 where the crop existed.
  * occlusion is the fraction of the box covered by another track's box.

Only relative weights within one track's votes matter, so the absolute scale
of these factors is unimportant.

ADMITTING UNCERTAINTY. A winner is marked uncertain when it beats the
runner-up by less than `confidence_margin_threshold`, or when fewer than
`min_evidence_count` observations supplied any value. Saying "uncertain"
rather than guessing is the point: it is what lets M7's semantic gate treat
an unknown attribute as compatible with anything instead of blocking a
correct link on a hallucinated value.

DETERMINISM. Ties are broken by value name, so the same input always yields
the same canonical output.

Output: data/outputs/attributes_canonical/<clip_id>.json (ClipCanonicalAttributes)
"""
from __future__ import annotations

from collections import defaultdict

from src.semantics.vocabulary import ATTRIBUTES, normalize_value
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import (
    CanonicalAttributes,
    ClipCanonicalAttributes,
    ClipRawAttributes,
    FrameAttributes,
    VoteDistribution,
)

logger = get_logger(__name__)

# Floor so an observation with a zero-scoring crop still carries a vote.
# Without it a track whose crops are all tiny or blurry would total zero
# weight and have no winner at all; with it, voting degrades gracefully to
# plain counting.
_MIN_WEIGHT = 1e-6


class TemporalVotingError(RuntimeError):
    pass


def _evidence_weight(obs: FrameAttributes, settings: PipelineSettings) -> float:
    attribute_confidence = settings.voting.retry_confidence if obs.from_retry else 1.0
    weight = attribute_confidence * obs.crop_quality * (1.0 - obs.occlusion)
    return max(weight, _MIN_WEIGHT)


def _vote_on_attribute(
    attribute: str,
    observations: list[FrameAttributes],
    settings: PipelineSettings,
    *,
    min_evidence_count: int | None = None,
    margin_threshold: float | None = None,
) -> tuple[VoteDistribution, int]:
    """Returns the vote distribution plus how many raw values normalization changed.

    The two thresholds default to the M6 config but are overridable, because
    M8 votes over only its three best shots: M6's "at least 3 observations"
    rule encodes "many frames agreed", which is the wrong question to ask of a
    deliberately tiny, deliberately high-quality sample.
    """
    if min_evidence_count is None:
        min_evidence_count = settings.voting.min_evidence_count
    if margin_threshold is None:
        margin_threshold = settings.voting.confidence_margin_threshold
    tally: dict[str, float] = defaultdict(float)
    n_votes = 0
    n_normalized = 0

    for obs in observations:
        raw = getattr(obs, attribute)
        if raw is None:
            continue
        value = normalize_value(attribute, raw)
        if value is None:  # outside the closed vocabulary — not evidence
            continue
        if value != raw:
            n_normalized += 1
        tally[value] += _evidence_weight(obs, settings)
        n_votes += 1

    if not tally:
        return (
            VoteDistribution(
                attribute=attribute,
                winner=None,
                distribution={},
                uncertain=True,
                n_votes=0,
                margin=0.0,
                uncertain_reason="no_evidence",
            ),
            n_normalized,
        )

    total = sum(tally.values())
    # Sorted by descending share, ties broken by value name for determinism.
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    distribution = {value: round(weight / total, 6) for value, weight in ranked}

    shares = list(distribution.values())
    margin = shares[0] - (shares[1] if len(shares) > 1 else 0.0)
    winner = ranked[0][0]

    uncertain_reason = None
    if n_votes < min_evidence_count:
        uncertain_reason = "insufficient_evidence"
    elif margin < margin_threshold:
        uncertain_reason = "low_margin"

    return (
        VoteDistribution(
            attribute=attribute,
            winner=winner,
            distribution=distribution,
            uncertain=uncertain_reason is not None,
            n_votes=n_votes,
            margin=round(margin, 6),
            uncertain_reason=uncertain_reason,
        ),
        n_normalized,
    )


def vote_clip_attributes(
    clip_id: str, settings: PipelineSettings | None = None
) -> ClipCanonicalAttributes:
    settings = settings or get_settings()

    raw_path = settings.resolve_path(settings.paths.outputs_dir) / "attributes_raw" / f"{clip_id}.json"
    if not raw_path.exists():
        raise TemporalVotingError(
            f"No raw attributes found for {clip_id} at {raw_path}; run `attribute` first."
        )
    raw_attributes = ClipRawAttributes.model_validate_json(raw_path.read_text())

    by_track: dict[str, list[FrameAttributes]] = defaultdict(list)
    for obs in raw_attributes.attributes:
        by_track[obs.track_id].append(obs)

    canonical_tracks: list[CanonicalAttributes] = []
    total_normalized = 0
    n_uncertain = 0

    for track_id in sorted(by_track):
        votes = []
        for attribute in ATTRIBUTES:
            vote, n_normalized = _vote_on_attribute(attribute, by_track[track_id], settings)
            votes.append(vote)
            total_normalized += n_normalized
            n_uncertain += int(vote.uncertain)
        canonical_tracks.append(CanonicalAttributes(track_id=track_id, votes=votes))

    canonical = ClipCanonicalAttributes(clip_id=clip_id, tracks=canonical_tracks)

    output_dir = settings.resolve_path(settings.paths.outputs_dir) / "attributes_canonical"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_id}.json"
    output_path.write_text(canonical.model_dump_json(indent=2))

    n_attributes = len(canonical_tracks) * len(ATTRIBUTES)
    logger.info(
        "vote_clip_attributes: clip_id=%s %d tracks, %d/%d attributes uncertain, "
        "%d values normalized -> %s",
        clip_id,
        len(canonical_tracks),
        n_uncertain,
        n_attributes,
        total_normalized,
        output_path,
    )

    return canonical
