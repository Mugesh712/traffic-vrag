"""M13 — LLM reasoning & explainable answer generation.

Turns M12's RetrievalResult into a grounded, cited answer: question -> context
-> LLM -> validated citations -> structured AnswerResult.

COUNTING QUESTIONS NEVER REACH THE LLM. M12 already computes an authoritative
count from the graph (COUNT(*), not a top-K similarity list). Asking an LLM to
count items in a text block is exactly where LLMs hallucinate, and there is a
correct deterministic answer sitting right there -- so counting questions are
answered by formatting `result.count`/`count_breakdown` directly.

EMPTY RESULTS NEVER REACH THE LLM EITHER. If retrieval found nothing, the
answer is "insufficient evidence" by construction, not by trusting the model
to follow that instruction. A prompt rule is a request; a code path that never
calls the model on empty context is a guarantee.

CITATIONS ARE VALIDATED, NOT TRUSTED. The prompt requires every claim to cite
"[global_id @ HH:MM:SS]". After generation, every citation is checked against
the objects actually in context. A citation naming an object outside that set
is a hallucinated reference and is recorded in `unsupported_citations` --
never silently dropped, so a fabricated citation is visible rather than
laundered into a clean-looking answer. `supporting_object_ids` is built only
from citations that passed this check, not from the full retrieved set: it is
a claim about what the answer actually leans on, not a dump of what was found.

THE REASONING TRACE IS PIPELINE-GENERATED, NOT SELF-REPORTED. Asking the LLM
to explain its own reasoning would trust an unverified claim about an
unverified claim. The trace instead records what the retrieval and validation
code actually did -- deterministic, testable, and not a second hallucination
surface.

KG_SUBGRAPH IS BUILT FROM THE VALIDATED ANSWER, NOT THE FULL RETRIEVED SET.
Only cited objects and their events become nodes, so the subgraph explains
this specific answer rather than dumping everything M12 happened to retrieve.
Built purely from RetrievalResult's already-assembled context, with no second
Neo4j query.

Pluggable backends, same pattern as M5: Ollama is implemented; other backends
are registered but stubbed until needed.
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import (
    AnswerResult,
    KGEdge,
    KGNode,
    KGSubgraph,
    RetrievalResult,
    RetrievedObject,
    TimestampSpan,
)

logger = get_logger(__name__)

INSUFFICIENT_EVIDENCE = "insufficient evidence"

# "[obj_0001 @ 11:03:20]" -- the exact citation shape the prompt demands.
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_]+)\s*@\s*(\d{1,2}:\d{2}:\d{2})\]")

SYSTEM_PROMPT = f"""You are answering questions about a traffic surveillance video, using ONLY the CONTEXT provided below. You have no other knowledge of this video.

Rules:
1. Answer ONLY from the CONTEXT. Never use outside knowledge or guess.
2. Every factual claim must cite the object it comes from, using the object's ACTUAL id and ACTUAL timestamp copied from the CONTEXT -- never the literal words "global_id" or "HH:MM:SS".
3. If the CONTEXT does not contain enough information to answer, respond with exactly: {INSUFFICIENT_EVIDENCE}
4. Do not invent objects, attributes, or events that are not in the CONTEXT.
5. Be concise -- a few sentences, not a report.

Example. Given this context:
- obj_0007 (car):
    attributes: color=red (confidence 0.90)
    event: STOP (as subject) [obj_0007 @ 09:15:00] to 09:15:30

And the question "did any car stop?", a correctly formatted answer is:
The red car [obj_0007 @ 09:15:00] stopped."""


class AnswerGeneratorError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Pluggable LLM backends
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    def generate(self, prompt: str, temperature: float) -> str: ...


class OllamaBackend:
    """Local models via Ollama (Qwen2.5, Llama 3.1, Mistral, Phi-4, ...) --
    any model pulled into the local Ollama server works, since the model name
    is just a config string."""

    def __init__(self, model: str, host: str, timeout_sec: float):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str, temperature: float) -> str:
        import requests

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AnswerGeneratorError(
                f"Ollama request failed ({exc.__class__.__name__}): {exc}. "
                f"Is `ollama serve` running with model '{self.model}' pulled?"
            ) from exc
        return response.json()["response"].strip()


class _UnimplementedBackend:
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError(
            f"LLM backend '{self.name}' is not implemented yet; use 'ollama' "
            "(configs/pipeline.yaml: answer.backend)."
        )


_BACKEND_CACHE: dict[str, Any] = {}


def _load_backend(settings: PipelineSettings) -> LLMBackend:
    name = settings.answer.backend
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]

    if name == "ollama":
        backend: LLMBackend = OllamaBackend(
            model=settings.answer.model, host=settings.answer.host, timeout_sec=settings.answer.timeout_sec
        )
    elif name in ("openai", "anthropic"):
        backend = _UnimplementedBackend(name)
    else:
        raise AnswerGeneratorError(f"Unknown answer backend: {name}")

    _BACKEND_CACHE[name] = backend
    return backend


# ---------------------------------------------------------------------------
# Context assembly (pure)
# ---------------------------------------------------------------------------


def _format_object_context(obj: RetrievedObject, max_events: int) -> tuple[str, set[str]]:
    """Returns (context_lines, valid_times) -- the exact set of timestamps
    this block shows for `obj`. Computed together, in one pass, so "what the
    model saw" and "what it's allowed to cite" can never drift apart into two
    separately-maintained lists that quietly disagree.
    """
    lines = [f"- {obj.global_id} ({obj.cls}):"]
    valid_times: set[str] = set()

    confirmed = [
        f"{a['type']}={a['value']} (confidence {a.get('confidence', 0):.2f})"
        for a in obj.attributes
        if a.get("winner") and not a.get("uncertain")
    ]
    if confirmed:
        lines.append(f"    attributes: {', '.join(confirmed)}")

    if obj.clips:
        lines.append(f"    seen in clips: {', '.join(obj.clips)}")

    for event in obj.events[:max_events]:
        role = event.get("role", "subject")
        start, end = _time_of(event.get("start_time")), _time_of(event.get("end_time"))
        valid_times.add(start)
        valid_times.add(end)
        lines.append(f"    event: {event.get('type')} (as {role}) [{obj.global_id} @ {start}] to {end}")

    return "\n".join(lines), valid_times


def _time_of(iso: str | None) -> str:
    if not iso:
        return "unknown"
    time_part = iso.split("T")[-1] if "T" in iso else iso
    return time_part.split(".")[0]


def build_context(
    result: RetrievalResult, settings: PipelineSettings
) -> tuple[str, dict[str, set[str]]]:
    """Bounded, structured context block, plus exactly which (global_id, time)
    citations it makes valid. Every fact here is one the model is allowed to
    cite; nothing outside it should appear as a claim."""
    objects = result.objects[: settings.answer.max_context_objects]
    if not objects:
        return "", {}
    blocks = []
    valid_times: dict[str, set[str]] = {}
    for o in objects:
        block, times = _format_object_context(o, settings.answer.max_events_per_object)
        blocks.append(block)
        valid_times[o.global_id] = times
    return "\n".join(blocks), valid_times


def build_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"


# ---------------------------------------------------------------------------
# Citation validation (pure)
# ---------------------------------------------------------------------------


def extract_citations(answer_text: str) -> list[tuple[str, str]]:
    """[(global_id, "HH:MM:SS"), ...] in the order they appear, duplicates kept
    (a repeated citation is not an error)."""
    return _CITATION_RE.findall(answer_text)


def validate_citations(
    citations: list[tuple[str, str]], objects: list[RetrievedObject], valid_times: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    """Split citations into (supported_object_ids, unsupported_citation_strings).

    Order-preserved, deduped. A citation is supported only when BOTH the
    object id was in context AND the exact timestamp was one that context
    actually showed for it. Checking the id alone is not enough: a model can
    (and, observed live, does) name a real object next to a fabricated
    timestamp that appears nowhere in the context -- e.g. inventing "00:00:00"
    for an object whose only real timestamp was "11:01:00". An object with no
    events in context has no valid timestamp to cite at all, so any citation
    of it is correctly rejected.
    """
    known_ids = {o.global_id for o in objects}
    supported: list[str] = []
    unsupported: list[str] = []
    for global_id, time in citations:
        if global_id in known_ids and time in valid_times.get(global_id, set()):
            if global_id not in supported:
                supported.append(global_id)
        else:
            tag = f"[{global_id} @ {time}]"
            if tag not in unsupported:
                unsupported.append(tag)
    return supported, unsupported


def _resolve_timestamps(
    citations: list[tuple[str, str]], objects_by_id: dict[str, RetrievedObject]
) -> list[TimestampSpan]:
    """A citation's timestamp becomes an event span when it matches a known
    event start time for that object, otherwise a single point in time."""
    spans: list[TimestampSpan] = []
    seen: set[tuple[str, str]] = set()
    for global_id, time in citations:
        obj = objects_by_id.get(global_id)
        if obj is None or (global_id, time) in seen:
            continue
        seen.add((global_id, time))

        matched_event = next(
            (e for e in obj.events if _time_of(e.get("start_time")) == time), None
        )
        if matched_event:
            spans.append(
                TimestampSpan(start=time, end=_time_of(matched_event.get("end_time")))
            )
        else:
            spans.append(TimestampSpan(start=time, end=time))
    return spans


def _evidence_frames_for(
    supported_ids: list[str], objects_by_id: dict[str, RetrievedObject]
) -> list[str]:
    frames: list[str] = []
    for global_id in supported_ids:
        for frame in objects_by_id[global_id].evidence_frames:
            if frame not in frames:
                frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# KG subgraph (pure) — built from the validated answer, not the full retrieval
# ---------------------------------------------------------------------------


def build_kg_subgraph(
    supported_ids: list[str], objects_by_id: dict[str, RetrievedObject]
) -> KGSubgraph:
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []
    seen_events: set[str] = set()

    for global_id in supported_ids:
        obj = objects_by_id[global_id]
        nodes.append(
            KGNode(
                id=global_id,
                label="TrafficObject",
                properties={
                    "class": obj.cls,
                    **{a["type"]: a["value"] for a in obj.attributes if a.get("winner")},
                },
            )
        )
        for event in obj.events:
            event_id = event.get("event_id")
            if not event_id:
                continue
            if event_id not in seen_events:
                seen_events.add(event_id)
                nodes.append(
                    KGNode(
                        id=event_id,
                        label="Event",
                        properties={
                            "type": event.get("type"),
                            "start_time": event.get("start_time"),
                            "end_time": event.get("end_time"),
                            "confidence": event.get("confidence"),
                        },
                    )
                )
            edges.append(
                KGEdge(
                    source=event_id,
                    target=global_id,
                    type="INVOLVES",
                    properties={"role": event.get("role", "subject")},
                )
            )

    return KGSubgraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Deterministic paths: counting and empty results
# ---------------------------------------------------------------------------


def _format_count_answer(result: RetrievalResult) -> str:
    if not result.count_breakdown:
        return f"There are {result.count} matching object(s)."
    parts = [
        f"{row['n']} {row.get('color', '')} {row.get('vehicle_type') or row.get('class', '')}".split()
        for row in result.count_breakdown
    ]
    described = ", ".join(" ".join(p) for p in parts)
    return f"There are {result.count} matching object(s): {described}."


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_answer(
    result: RetrievalResult, settings: PipelineSettings | None = None
) -> AnswerResult:
    settings = settings or get_settings()

    if result.intent.question_type == "counting" and result.count is not None:
        return AnswerResult(
            answer=_format_count_answer(result),
            status="counting",
            reasoning_trace=(
                f"Counting question answered directly from the graph "
                f"(count={result.count}); the LLM was not called."
            ),
        )

    if not result.objects:
        return AnswerResult(
            answer=INSUFFICIENT_EVIDENCE,
            status="insufficient_evidence",
            reasoning_trace="Retrieval returned no objects; the LLM was not called.",
        )

    objects_by_id = {o.global_id: o for o in result.objects}
    context, valid_times = build_context(result, settings)
    prompt = build_prompt(result.question, context)

    backend = _load_backend(settings)
    raw_answer = backend.generate(prompt, settings.answer.temperature)

    if INSUFFICIENT_EVIDENCE in raw_answer.lower():
        return AnswerResult(
            answer=raw_answer,
            status="insufficient_evidence",
            reasoning_trace=(
                f"Retrieved {len(result.objects)} candidate object(s); "
                "the model judged the context insufficient to answer."
            ),
        )

    citations = extract_citations(raw_answer)
    supported_ids, unsupported = validate_citations(citations, result.objects, valid_times)

    trace = (
        f"Retrieved {len(result.objects)} candidate object(s), gave the model "
        f"{len(objects_by_id)} in context. It made {len(citations)} citation(s), "
        f"{len(supported_ids)} object(s) supported."
    )
    if unsupported:
        trace += f" {len(unsupported)} citation(s) named an object outside the context: {unsupported}."
        logger.warning("generate_answer: hallucinated citation(s): %s", unsupported)

    return AnswerResult(
        answer=raw_answer,
        supporting_object_ids=supported_ids,
        timestamps=_resolve_timestamps(citations, objects_by_id),
        evidence_frames=_evidence_frames_for(supported_ids, objects_by_id),
        kg_subgraph=build_kg_subgraph(supported_ids, objects_by_id),
        reasoning_trace=trace,
        status="answered",
        unsupported_citations=unsupported,
    )
