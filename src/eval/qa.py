"""M16 — QA benchmark: schema, scoring, and the system-under-test runner.

HOW FREE-TEXT ANSWERS ARE SCORED, AND WHY NOT WITH AN LLM JUDGE. Grading
generated prose with another LLM is nondeterministic, unauditable, and -- when
the judge is the same family of model being judged -- circular. Instead each
question type is scored on whatever about it is *objectively checkable*:

  counting     exact integer match. A count is either right or wrong.
  factual      set overlap (P/R/F1) between the objects the answer CITED and
  temporal     the objects the annotation says it should have. This scores
               grounding rather than phrasing, so "the white sedan" and
               "a pale car" are not penalised differently for prose.
  counterfactual  NOT scored for correctness -- deliberately. There is no
  forecast        ground-truth answer to "what if the bus hadn't stopped",
                  and inventing a rubric would be fake rigor. They are scored
                  only on GROUNDEDNESS (did every citation resolve to a real
                  retrieved object?) and abstention behaviour, which are the
                  properties that actually matter for a system whose thesis is
                  "don't assert what you can't support".

ABSTENTION IS A FIRST-CLASS SCORE. A benchmark item may be marked
`answerable: false` -- e.g. "what was the licence plate?" when no plate is
readable. Answering it at all is a failure; saying "insufficient evidence" is
the correct response. This measures the paper's central claim directly, and it
is the one QA metric that needs no annotation of objects at all.

GROUNDEDNESS NEEDS NO GROUND TRUTH. Every cited object id is checked against
what retrieval actually returned for that question. A citation naming
something never retrieved is a hallucinated reference regardless of whether
the underlying claim happens to be true.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

QuestionType = Literal["factual", "counting", "temporal", "counterfactual", "forecast"]

# Types with an objectively checkable answer. The other two are scored on
# groundedness/abstention only -- see the module docstring.
CORRECTNESS_SCORED = ("factual", "counting", "temporal")

UNREVIEWED_MARKER = "REMOVE_THIS_KEY_ONCE_REVIEWED"

_NUMBER_WORDS = {
    "zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

# Phrasing-based abstention detection, applied identically to every system.
# The list covers the ways a model actually declines, not just the exact
# sentinel M13's prompt asks for -- a baseline that says "not specified in the
# provided context" has abstained just as surely as one that says the magic
# words, and scoring only the sentinel would credit our system for following
# its own prompt format rather than for declining.
# KNOWN LIMITATION: this under-counts novel phrasings. It errs toward scoring
# an abstention as an answer (a miss), never toward crediting an answer as an
# abstention, so it is conservative in the direction that matters.
_ABSTENTION_PHRASES = (
    "insufficient evidence", "not enough information", "cannot determine",
    "can't determine", "cannot answer", "can't answer", "unable to determine",
    "no information", "not possible to", "not specified", "not mentioned",
    "does not contain", "does not specify", "does not mention", "not provided",
    "no evidence", "not available in the",
)


class QABenchmarkError(RuntimeError):
    pass


class QAItem(BaseModel):
    question_id: str
    question: str
    question_type: QuestionType
    # Ground truth. Which fields matter depends on question_type.
    expected_count: Optional[int] = None
    expected_gt_ids: list[str] = Field(default_factory=list)
    answerable: bool = True
    notes: str = ""


class QABenchmark(BaseModel):
    video_id: str
    items: list[QAItem] = Field(default_factory=list)


@dataclass
class QAScore:
    question_id: str
    question_type: str
    system: str
    answer: str
    cited_ids: list[str] = field(default_factory=list)
    # None means "not applicable to this question type / not scorable".
    counting_correct: Optional[bool] = None
    object_f1: Optional[float] = None
    object_precision: Optional[float] = None
    object_recall: Optional[float] = None
    abstention_correct: Optional[bool] = None
    grounded: Optional[bool] = None

    def as_row(self) -> dict:
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, bool):
                return "yes" if v else "no"
            return round(v, 3)

        return {
            "Question": self.question_id,
            "Type": self.question_type,
            "Counting": fmt(self.counting_correct),
            "Object F1": fmt(self.object_f1),
            "Abstention": fmt(self.abstention_correct),
            "Grounded": fmt(self.grounded),
        }


def benchmark_path(video_id: str, settings: PipelineSettings) -> Path:
    return settings.resolve_path("data/qa_benchmark") / f"{video_id}.json"


def load_benchmark(video_id: str, settings: PipelineSettings | None = None) -> QABenchmark | None:
    """None when no benchmark exists, so the harness still runs its other tables."""
    settings = settings or get_settings()
    path = benchmark_path(video_id, settings)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if UNREVIEWED_MARKER in raw:
        raise QABenchmarkError(
            f"{path} is still an unreviewed template: it contains "
            f"'{UNREVIEWED_MARKER}'. Write real questions and expected answers, "
            "then delete that key."
        )
    return QABenchmark.model_validate(raw)


def write_benchmark_template(
    video_id: str, settings: PipelineSettings | None = None, overwrite: bool = False
) -> Path:
    """Emit a starter benchmark covering all five question types.

    The questions are illustrative and the expected answers are deliberately
    left null -- only a human who has watched the footage can fill them in,
    and a template with pre-filled "expected" values would just be the
    system's own output wearing a ground-truth costume.
    """
    settings = settings or get_settings()
    path = benchmark_path(video_id, settings)
    if path.exists() and not overwrite:
        raise QABenchmarkError(f"{path} already exists; pass overwrite=True to replace it.")

    items = [
        {"question_id": "q1", "question": "How many cars are in the video?",
         "question_type": "counting", "expected_count": None, "expected_gt_ids": [],
         "answerable": True, "notes": "Fill expected_count by watching the clip."},
        {"question_id": "q2", "question": "Which vehicles are white?",
         "question_type": "factual", "expected_count": None, "expected_gt_ids": [],
         "answerable": True, "notes": "List the gt_ids from data/ground_truth/."},
        {"question_id": "q3", "question": "Which vehicles appeared before 08:30:15?",
         "question_type": "temporal", "expected_count": None, "expected_gt_ids": [],
         "answerable": True, "notes": "Adjust the time to suit the clip."},
        {"question_id": "q4", "question": "What would have happened if the first car had not turned?",
         "question_type": "counterfactual", "expected_count": None, "expected_gt_ids": [],
         "answerable": True, "notes": "Scored on groundedness only, not correctness."},
        {"question_id": "q5", "question": "Which vehicle is most likely to leave the frame next?",
         "question_type": "forecast", "expected_count": None, "expected_gt_ids": [],
         "answerable": True, "notes": "Scored on groundedness only, not correctness."},
        {"question_id": "q6", "question": "What was the licence plate number of the red car?",
         "question_type": "factual", "expected_count": None, "expected_gt_ids": [],
         "answerable": False,
         "notes": "Unanswerable on purpose: the correct response is to decline."},
    ]

    template = {
        UNREVIEWED_MARKER: (
            "Starter QA benchmark. Replace these with questions about YOUR footage and "
            "fill in the expected answers by watching the clip, then delete this key. "
            "Expected values must come from a human, not from the system's output."
        ),
        "video_id": video_id,
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2))
    logger.info("write_benchmark_template: wrote %d starter question(s) -> %s", len(items), path)
    return path


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def extract_count(answer: str) -> Optional[int]:
    """First integer in the answer, accepting digits or number words.

    Baselines answer in prose ("there are seven cars"), so digit-only parsing
    would mark a correct baseline wrong and flatter our own system, which
    emits digits. Whichever form appears first in the text wins.
    """
    lowered = answer.lower()
    digit = re.search(r"\b(\d+)\b", lowered)
    word = re.search(rf"\b({'|'.join(_NUMBER_WORDS)})\b", lowered)
    if digit and (not word or digit.start() <= word.start()):
        return int(digit.group(1))
    if word:
        return _NUMBER_WORDS[word.group(1)]
    return None


def is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _ABSTENTION_PHRASES)


def score_objects(cited_gt_ids: list[str], expected_gt_ids: list[str]) -> tuple[float, float, float]:
    """(precision, recall, f1) over cited-vs-expected object sets."""
    cited, expected = set(cited_gt_ids), set(expected_gt_ids)
    if not expected:
        return 0.0, 0.0, 0.0
    if not cited:
        return 0.0, 0.0, 0.0
    hits = len(cited & expected)
    precision = hits / len(cited)
    recall = hits / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_item(
    item: QAItem,
    system: str,
    answer: str,
    cited_ids: list[str],
    retrieved_ids: list[str],
    gt_id_by_cited_id: dict[str, str] | None = None,
) -> QAScore:
    """Score one answer. Metrics that do not apply are left None, never 0."""
    score = QAScore(
        question_id=item.question_id, question_type=item.question_type,
        system=system, answer=answer, cited_ids=list(cited_ids),
    )

    # Groundedness: needs no ground truth at all.
    score.grounded = all(cited in set(retrieved_ids) for cited in cited_ids) if cited_ids else True

    if not item.answerable:
        # The only correct behaviour is to decline, so nothing else is scored.
        score.abstention_correct = is_abstention(answer)
        return score
    score.abstention_correct = not is_abstention(answer)

    if item.question_type == "counting" and item.expected_count is not None:
        score.counting_correct = extract_count(answer) == item.expected_count

    if item.question_type in CORRECTNESS_SCORED and item.expected_gt_ids:
        mapped = [gt_id_by_cited_id.get(c) for c in cited_ids] if gt_id_by_cited_id else []
        precision, recall, f1 = score_objects([m for m in mapped if m], item.expected_gt_ids)
        score.object_precision, score.object_recall, score.object_f1 = precision, recall, f1

    return score


def aggregate(scores: list[QAScore]) -> dict[str, object]:
    """Per-system summary. Each metric averages only over the items where it
    applies, so an unscored question type never drags a mean toward zero."""
    def mean(values):
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_questions": len(scores),
        "counting_accuracy": mean([1.0 if s.counting_correct else 0.0
                                   for s in scores if s.counting_correct is not None]),
        "object_f1": mean([s.object_f1 for s in scores]),
        "abstention_accuracy": mean([1.0 if s.abstention_correct else 0.0
                                     for s in scores if s.abstention_correct is not None]),
        "grounded_rate": mean([1.0 if s.grounded else 0.0
                               for s in scores if s.grounded is not None]),
    }
