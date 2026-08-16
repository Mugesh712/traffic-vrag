"""M16 — Baselines to compare the full system against.

Both baselines answer the same benchmark questions from the same frames, with
the same VLM and the same LLM, so any difference is attributable to the
pipeline in between rather than to model choice.

(a) FrameCaptionBaseline -- "Video-LLM end-to-end" stand-in.
    Uniformly samples frames, captions each, concatenates them into one
    transcript, and asks the LLM the question directly.

    NAMING HONESTY: this is NOT a true Video-LLM. A real one (Video-LLaVA,
    VideoChat, Qwen2-VL) ingests frames jointly with temporal attention rather
    than reading independent per-frame captions, and would likely do better on
    motion and interaction questions. This approximates that baseline with the
    tooling actually available here, and the paper must describe it as
    "frame-sampled captioning + LLM", not as a Video-LLM. The class is
    pluggable precisely so a real Video-LLM can be dropped in for the final
    numbers.

(b) CaptionRAGBaseline -- caption-based retrieval-augmented generation.
    Captions frames, embeds each caption, retrieves the top-k most similar to
    the question, and answers from those. This is the honest "what if we
    skipped the whole pipeline" comparison: no detection, no tracking, no
    global identity, no knowledge graph, no uncertainty -- just text over
    frames. It is the baseline that shows what tracking and identity actually
    buy, because it is exactly the system minus all of them.

Neither baseline can cite object ids, since neither has any notion of an
object identity. Their citation lists are therefore empty, which is reported
rather than papered over: it means object-F1 is structurally zero for them,
and that fact *is* the result -- an ungrounded system cannot point at what it
is talking about.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.retrieval.answer_generator import INSUFFICIENT_EVIDENCE, _load_backend as _load_llm
from src.semantics.vlm_extractor import caption_images, get_tasks
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import VideoManifest

logger = get_logger(__name__)

BASELINE_PROMPT = f"""You are answering a question about a traffic surveillance video, using ONLY the frame descriptions below. You have no other knowledge of this video.

Rules:
1. Answer ONLY from the descriptions. Never use outside knowledge or guess.
2. If the descriptions do not contain enough information, respond with exactly: {INSUFFICIENT_EVIDENCE}
3. Be concise."""


@dataclass
class BaselineAnswer:
    answer: str
    cited_ids: list[str]  # always empty: baselines have no object identities
    n_context_items: int


def _sample_frames(
    video_id: str, settings: PipelineSettings, max_frames: int
) -> list[tuple[str, np.ndarray]]:
    """(frame_id, image) evenly spread across the video's sampled frames."""
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    manifest_path = outputs / "ingest" / f"{video_id}_manifest.json"
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())

    records = [(f.frame_id, f.frame_path) for clip in manifest.clips for f in clip.frames]
    if not records:
        return []
    if len(records) > max_frames:
        step = np.linspace(0, len(records) - 1, max_frames)
        records = [records[int(round(i))] for i in sorted(set(int(round(i)) for i in step))]

    frames = []
    for frame_id, frame_path in records:
        image = cv2.imread(str(settings.resolve_path(frame_path)))
        if image is not None:
            frames.append((frame_id, image))
    return frames


def _time_of(frame_id: str, video_id: str, settings: PipelineSettings, cache: dict) -> str:
    if not cache:
        outputs = settings.resolve_path(settings.paths.outputs_dir)
        manifest = VideoManifest.model_validate_json(
            (outputs / "ingest" / f"{video_id}_manifest.json").read_text()
        )
        for clip in manifest.clips:
            for f in clip.frames:
                cache[f.frame_id] = f.wallclock_time.split("T")[-1].split(".")[0]
    return cache.get(frame_id, "unknown")


class FrameCaptionBaseline:
    """Approximates an end-to-end Video-LLM (see module docstring)."""

    name = "frame_caption"

    def __init__(self, settings: PipelineSettings, max_frames: int = 16):
        self.settings = settings
        self.max_frames = max_frames
        self._captions: list[tuple[str, str]] | None = None
        self._time_cache: dict = {}

    def _ensure_captions(self, video_id: str) -> list[tuple[str, str]]:
        if self._captions is None:
            frames = _sample_frames(video_id, self.settings, self.max_frames)
            _, high_detail = get_tasks(self.settings)
            texts = caption_images([img for _, img in frames], self.settings, task=high_detail)
            self._captions = [(fid, text) for (fid, _), text in zip(frames, texts)]
            logger.info("%s: captioned %d frame(s)", self.name, len(self._captions))
        return self._captions

    def answer(self, video_id: str, question: str) -> BaselineAnswer:
        captions = self._ensure_captions(video_id)
        if not captions:
            return BaselineAnswer(INSUFFICIENT_EVIDENCE, [], 0)

        lines = [
            f"[{_time_of(fid, video_id, self.settings, self._time_cache)}] {text}"
            for fid, text in captions
        ]
        prompt = (
            f"{BASELINE_PROMPT}\n\nFRAME DESCRIPTIONS:\n"
            + "\n".join(lines)
            + f"\n\nQUESTION: {question}\n\nANSWER:"
        )
        llm = _load_llm(self.settings)
        return BaselineAnswer(
            llm.generate(prompt, self.settings.answer.temperature), [], len(captions)
        )


class CaptionRAGBaseline:
    """Retrieval over per-frame captions, with no pipeline structure at all."""

    name = "caption_rag"

    def __init__(self, settings: PipelineSettings, max_frames: int = 64, top_k: int = 8):
        self.settings = settings
        self.max_frames = max_frames
        self.top_k = top_k
        self._collection = None
        self._time_cache: dict = {}

    def _ensure_index(self, video_id: str):
        if self._collection is not None:
            return self._collection

        import chromadb

        frames = _sample_frames(video_id, self.settings, self.max_frames)
        _, high_detail = get_tasks(self.settings)
        texts = caption_images([img for _, img in frames], self.settings, task=high_detail)

        client = chromadb.EphemeralClient()  # baseline index is disposable
        collection = client.create_collection(f"baseline_{video_id}")
        if frames:
            collection.add(
                ids=[fid for fid, _ in frames],
                documents=[
                    f"[{_time_of(fid, video_id, self.settings, self._time_cache)}] {text}"
                    for (fid, _), text in zip(frames, texts)
                ],
            )
        self._collection = collection
        logger.info("%s: indexed %d frame caption(s)", self.name, len(frames))
        return collection

    def answer(self, video_id: str, question: str) -> BaselineAnswer:
        collection = self._ensure_index(video_id)
        if collection.count() == 0:
            return BaselineAnswer(INSUFFICIENT_EVIDENCE, [], 0)

        results = collection.query(
            query_texts=[question], n_results=min(self.top_k, collection.count())
        )
        documents = results["documents"][0]
        prompt = (
            f"{BASELINE_PROMPT}\n\nFRAME DESCRIPTIONS:\n"
            + "\n".join(documents)
            + f"\n\nQUESTION: {question}\n\nANSWER:"
        )
        llm = _load_llm(self.settings)
        return BaselineAnswer(
            llm.generate(prompt, self.settings.answer.temperature), [], len(documents)
        )


def build_baselines(settings: PipelineSettings | None = None) -> list:
    settings = settings or get_settings()
    return [FrameCaptionBaseline(settings), CaptionRAGBaseline(settings)]
