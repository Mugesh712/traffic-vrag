"""M16 — Evaluation orchestration.

Loads a processed video's outputs, computes every metric that its available
inputs support, and emits LaTeX + text tables.

WHAT RUNS WITHOUT GROUND TRUTH: attribute consistency, M8's correction
breakdown, M6's uncertainty rate, and the structural pipeline-stage summary
(how many tracks M3 produced, how many M4 merged, how many objects M7 linked).
These are real measurements of what the system did.

WHAT REQUIRES GROUND TRUTH: MOTA, IDF1, ID switches, fragmentation, and
attribute accuracy. Without annotation these are undefined, and the harness
says so in the table rather than emitting zeros.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.eval.attributes import (
    evaluate_attribute_accuracy,
    evaluate_attribute_consistency,
    summarize_corrections,
    uncertainty_rate,
)
from src.eval.ground_truth import GroundTruth, load_ground_truth
from src.eval.report import Table, render_document, render_text
from src.eval.tracking import FrameBoxes, evaluate_tracking, match_frame
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import (
    ClipAssociatedTracks,
    ClipCanonicalAttributes,
    ClipTracks,
    FinalObjectIndex,
    MasterObjectIndex,
    VideoManifest,
)

logger = get_logger(__name__)

NO_GT_REASON = (
    "no ground-truth annotation for this video. Generate a template with "
    "`evaluate <video_id> --write-template`, correct it by hand, then re-run."
)


class EvaluationError(RuntimeError):
    pass


@dataclass
class PipelineOutputs:
    video_id: str
    manifest: VideoManifest
    tracks: dict[str, ClipTracks]
    associated: dict[str, ClipAssociatedTracks]
    canonical: dict[str, ClipCanonicalAttributes]
    master: MasterObjectIndex | None
    final: FinalObjectIndex | None


def load_outputs(video_id: str, settings: PipelineSettings) -> PipelineOutputs:
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    manifest_path = outputs / "ingest" / f"{video_id}_manifest.json"
    if not manifest_path.exists():
        raise EvaluationError(f"No ingest manifest at {manifest_path}; run the pipeline first.")
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())

    tracks: dict[str, ClipTracks] = {}
    associated: dict[str, ClipAssociatedTracks] = {}
    canonical: dict[str, ClipCanonicalAttributes] = {}
    for clip in manifest.clips:
        for sub, model, sink in (
            ("tracks", ClipTracks, tracks),
            ("tracks_associated", ClipAssociatedTracks, associated),
            ("attributes_canonical", ClipCanonicalAttributes, canonical),
        ):
            path = outputs / sub / f"{clip.clip_id}.json"
            if path.exists():
                sink[clip.clip_id] = model.model_validate_json(path.read_text())

    master_path = outputs / "master_object_index" / f"{video_id}.json"
    final_path = outputs / "global_objects_final" / f"{video_id}.json"
    return PipelineOutputs(
        video_id=video_id,
        manifest=manifest,
        tracks=tracks,
        associated=associated,
        canonical=canonical,
        master=MasterObjectIndex.model_validate_json(master_path.read_text()) if master_path.exists() else None,
        final=FinalObjectIndex.model_validate_json(final_path.read_text()) if final_path.exists() else None,
    )


def _frames_from_clip_tracks(
    clip_tracks, manifest: VideoManifest, id_prefix: str = ""
) -> list[FrameBoxes]:
    """Flatten per-track boxes into per-frame boxes keyed by track id."""
    ordered = [f.frame_id for clip in manifest.clips for f in clip.frames]
    by_frame: dict[str, dict[str, list[float]]] = {f: {} for f in ordered}
    for clip_id, data in clip_tracks.items():
        for track in data.tracks:
            key = f"{id_prefix}{clip_id}:{track.track_id}"
            for frame_id, bbox in zip(track.frames, track.bboxes):
                if frame_id in by_frame:
                    by_frame[frame_id][key] = list(bbox)
    return [FrameBoxes(frame_id=f, boxes=by_frame[f]) for f in ordered]


def _frames_from_global_objects(
    master: MasterObjectIndex, associated: dict[str, ClipAssociatedTracks], manifest: VideoManifest
) -> list[FrameBoxes]:
    """Boxes keyed by GLOBAL id -- so an object correctly linked across clips
    counts as one identity, which is precisely what M7 is meant to achieve and
    what IDF1 rewards."""
    tracks_by_key = {
        (clip_id, t.track_id): t for clip_id, data in associated.items() for t in data.tracks
    }
    ordered = [f.frame_id for clip in manifest.clips for f in clip.frames]
    by_frame: dict[str, dict[str, list[float]]] = {f: {} for f in ordered}
    for obj in master.objects:
        for sighting in obj.sightings:
            track = tracks_by_key.get((sighting.clip_id, sighting.track_id))
            if track is None:
                continue
            for frame_id, bbox in zip(track.frames, track.bboxes):
                if frame_id in by_frame:
                    by_frame[frame_id][obj.global_id] = list(bbox)
    return [FrameBoxes(frame_id=f, boxes=by_frame[f]) for f in ordered]


def _gt_frames(ground_truth: GroundTruth, manifest: VideoManifest) -> list[FrameBoxes]:
    ordered = [f.frame_id for clip in manifest.clips for f in clip.frames]
    return [FrameBoxes(frame_id=f, boxes=ground_truth.boxes_for_frame(f)) for f in ordered]


def _match_globals_to_gt(
    gt_frames: list[FrameBoxes], pred_frames: list[FrameBoxes], iou_threshold: float = 0.5
) -> dict[str, str]:
    """global_id -> gt_id, by which pairing co-occurs most often."""
    votes: dict[tuple[str, str], int] = {}
    pred_by_frame = {f.frame_id: f.boxes for f in pred_frames}
    for frame in gt_frames:
        for gt_id, pred_id in match_frame(frame.boxes, pred_by_frame.get(frame.frame_id, {}), iou_threshold):
            votes[(pred_id, gt_id)] = votes.get((pred_id, gt_id), 0) + 1

    best: dict[str, tuple[str, int]] = {}
    for (pred_id, gt_id), count in sorted(votes.items()):
        if pred_id not in best or count > best[pred_id][1]:
            best[pred_id] = (gt_id, count)
    return {pred_id: gt_id for pred_id, (gt_id, _) in best.items()}


def build_tables(
    outputs: PipelineOutputs, ground_truth: GroundTruth | None, settings: PipelineSettings
) -> list[Table]:
    tables: list[Table] = []

    # --- Stage summary (no GT needed) ---------------------------------------
    n_raw = sum(len(t.tracks) for t in outputs.tracks.values())
    n_assoc = sum(len(a.tracks) for a in outputs.associated.values())
    n_merges = sum(len(a.merge_log) for a in outputs.associated.values())
    n_global = len(outputs.master.objects) if outputs.master else 0
    n_multi = sum(1 for o in outputs.master.objects if len(o.sightings) > 1) if outputs.master else 0
    tables.append(
        Table(
            caption="Pipeline stage summary",
            label="tab:stages",
            columns=["Stage", "Identities", "Note"],
            rows=[
                {"Stage": "M3 ByteTrack", "Identities": n_raw, "Note": "raw per-clip tracks"},
                {"Stage": "M4 association", "Identities": n_assoc, "Note": f"{n_merges} fragment merges"},
                {"Stage": "M7 global linking", "Identities": n_global,
                 "Note": f"{n_multi} span multiple clips"},
            ],
        )
    )

    # --- Tracking (needs GT) ------------------------------------------------
    if ground_truth is None:
        tables.append(
            Table(caption="Tracking and identity metrics", label="tab:tracking",
                  columns=["Variant", "MOTA", "IDF1", "IDSW", "Frag"], rows=[],
                  unavailable_reason=NO_GT_REASON)
        )
        gt_id_by_global: dict[str, str] = {}
    else:
        gt_frames = _gt_frames(ground_truth, outputs.manifest)
        rows = []
        variants = [
            ("Raw ByteTrack (M3)", _frames_from_clip_tracks(outputs.tracks, outputs.manifest)),
            ("+ M4 association", _frames_from_clip_tracks(outputs.associated, outputs.manifest)),
        ]
        if outputs.master:
            variants.append(
                ("+ M7 global linking",
                 _frames_from_global_objects(outputs.master, outputs.associated, outputs.manifest))
            )
        for name, pred_frames in variants:
            metrics = evaluate_tracking(gt_frames, pred_frames)
            rows.append({"Variant": name, **metrics.as_row()})
        tables.append(
            Table(caption="Tracking and identity metrics", label="tab:tracking",
                  columns=["Variant", "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN"], rows=rows)
        )
        gt_id_by_global = (
            _match_globals_to_gt(
                gt_frames, _frames_from_global_objects(outputs.master, outputs.associated, outputs.manifest)
            )
            if outputs.master else {}
        )

    # --- Attribute accuracy (needs GT) --------------------------------------
    if ground_truth is None or outputs.final is None:
        tables.append(
            Table(caption="Attribute accuracy", label="tab:attr-accuracy",
                  columns=["Attribute", "Correct", "Wrong", "Abstained", "Precision", "Coverage"],
                  rows=[],
                  unavailable_reason=NO_GT_REASON if ground_truth is None
                  else "M8 output missing; run `confirm` first.")
        )
    else:
        accuracy = evaluate_attribute_accuracy(outputs.final, ground_truth, gt_id_by_global)
        # Ground truth can exist for BOXES while carrying no attribute labels --
        # MOT-format benchmarks annotate tracks only. Every attribute is then
        # unscorable, and rendering that as 0.0 precision would state a
        # measurement ("it got none right") where the truth is "nothing was
        # scored". Same principle as a missing-GT table, one level finer.
        scorable = sum(r.answered + r.abstained for r in accuracy)
        tables.append(
            Table(caption="Attribute accuracy (abstentions counted separately from errors)",
                  label="tab:attr-accuracy",
                  columns=["Attribute", "Correct", "Wrong", "Abstained", "Precision", "Coverage"],
                  rows=[r.as_row() for r in accuracy] if scorable else [],
                  unavailable_reason=(
                      "the ground truth annotates boxes but no attribute values "
                      "(MOT-format benchmarks label tracks only). Fill in colour/type "
                      "in data/ground_truth/ to score these."
                  ) if not scorable else None)
        )

    # --- Attribute consistency (no GT needed) -------------------------------
    if outputs.final is not None:
        consistency = evaluate_attribute_consistency(outputs.final, outputs.canonical)
        measurable = sum(c.consistent_objects + c.inconsistent_objects for c in consistency)
        tables.append(
            Table(
                caption="Attribute consistency across sightings (no ground truth required)",
                label="tab:attr-consistency",
                columns=["Attribute", "Consistent", "Inconsistent", "Rate", "Single-sighting (excl.)"],
                rows=[c.as_row() for c in consistency],
                unavailable_reason=(
                    "every object was seen in only one clip, so cross-sighting consistency "
                    "is undefined for this video. Needs footage where objects persist across "
                    "clip boundaries."
                ) if measurable == 0 else None,
            )
        )

        summary = summarize_corrections(outputs.final)
        tables.append(
            Table(caption="M8 best-shot confirmation outcomes (no ground truth required)",
                  label="tab:m8-corrections",
                  columns=["Outcome", "Count", "Meaning"], rows=summary.as_rows())
        )

    # --- Uncertainty rate (no GT needed) ------------------------------------
    if outputs.canonical:
        rates = uncertainty_rate(outputs.canonical)
        tables.append(
            Table(caption="Rate at which M6 declines to commit (no ground truth required)",
                  label="tab:uncertainty",
                  columns=["Attribute", "Uncertain rate"],
                  rows=[{"Attribute": a, "Uncertain rate": r} for a, r in rates.items()])
        )

    return tables


def _gt_map_for_qa(outputs: PipelineOutputs, ground_truth: GroundTruth | None) -> dict[str, str]:
    """global_id -> gt_id, or empty when there is no annotation to map to.

    Object-F1 needs this bridge: the benchmark names objects by gt_id (what a
    human annotated), while the system cites global_ids (what it inferred).
    Without annotation the mapping is empty and object-F1 stays unscored.
    """
    if ground_truth is None or outputs.master is None:
        return {}
    return _match_globals_to_gt(
        _gt_frames(ground_truth, outputs.manifest),
        _frames_from_global_objects(outputs.master, outputs.associated, outputs.manifest),
    )


def run_qa_benchmark(
    video_id: str,
    benchmark,
    settings: PipelineSettings,
    gt_id_by_global_id: dict[str, str],
    include_baselines: bool = True,
) -> list[Table]:
    """Run every benchmark question through our system and each baseline.

    Imported lazily so the rest of the harness never requires a running LLM.
    """
    from src.eval.baselines import build_baselines
    from src.eval.qa import aggregate, score_item
    from src.retrieval.answer_generator import generate_answer
    from src.retrieval.hybrid_retriever import hybrid_retrieve

    per_system: dict[str, list] = {}
    failures: list[str] = []

    for item in benchmark.items:
        try:
            retrieval = hybrid_retrieve(item.question, video_id, settings=settings)
            answer = generate_answer(retrieval, settings=settings)
            per_system.setdefault("traffic_vrag", []).append(
                score_item(
                    item, "traffic_vrag", answer.answer, answer.supporting_object_ids,
                    [o.global_id for o in retrieval.objects], gt_id_by_global_id,
                )
            )
        except Exception as exc:
            failures.append(f"traffic_vrag/{item.question_id}: {exc.__class__.__name__}: {exc}")

    if include_baselines:
        for baseline in build_baselines(settings):
            for item in benchmark.items:
                try:
                    result = baseline.answer(video_id, item.question)
                    per_system.setdefault(baseline.name, []).append(
                        score_item(item, baseline.name, result.answer, result.cited_ids, [], None)
                    )
                except Exception as exc:
                    failures.append(f"{baseline.name}/{item.question_id}: {exc.__class__.__name__}: {exc}")

    tables: list[Table] = []
    if per_system:
        rows = []
        for system, scores in sorted(per_system.items()):
            summary = aggregate(scores)
            rows.append({
                "System": system,
                "Questions": summary["n_questions"],
                "Counting acc.": summary["counting_accuracy"] if summary["counting_accuracy"] is not None else "-",
                "Object F1": summary["object_f1"] if summary["object_f1"] is not None else "-",
                "Abstention acc.": summary["abstention_accuracy"] if summary["abstention_accuracy"] is not None else "-",
                "Grounded": summary["grounded_rate"] if summary["grounded_rate"] is not None else "-",
            })
        tables.append(
            Table(
                caption=(
                    "QA accuracy against baselines. Counterfactual and forecast questions "
                    "are excluded from correctness scoring by design (no ground-truth answer "
                    "exists); they contribute only to groundedness and abstention."
                ),
                label="tab:qa",
                columns=["System", "Questions", "Counting acc.", "Object F1",
                         "Abstention acc.", "Grounded"],
                rows=rows,
            )
        )

        detail_rows = [s.as_row() | {"System": system}
                       for system, scores in sorted(per_system.items()) for s in scores]
        tables.append(
            Table(caption="QA per-question detail", label="tab:qa-detail",
                  columns=["System", "Question", "Type", "Counting", "Object F1",
                           "Abstention", "Grounded"],
                  rows=detail_rows)
        )

    if failures:
        logger.warning("run_qa_benchmark: %d question(s) failed: %s", len(failures), failures[:5])
        tables.append(
            Table(caption="QA runs that failed", label="tab:qa-failures",
                  columns=["Failure"], rows=[{"Failure": f} for f in failures])
        )
    return tables


def evaluate_video(
    video_id: str,
    settings: PipelineSettings | None = None,
    run_qa: bool = False,
    include_baselines: bool = True,
) -> tuple[list[Table], Path]:
    settings = settings or get_settings()
    outputs = load_outputs(video_id, settings)
    ground_truth = load_ground_truth(video_id, settings)
    if ground_truth is None:
        logger.warning(
            "evaluate_video: no ground truth for %s -- accuracy tables will be "
            "reported as unavailable, not zero", video_id,
        )

    tables = build_tables(outputs, ground_truth, settings)

    from src.eval.qa import load_benchmark

    benchmark = load_benchmark(video_id, settings)
    if benchmark is None:
        tables.append(
            Table(caption="QA accuracy against baselines", label="tab:qa",
                  columns=["System", "Questions", "Counting acc.", "Object F1",
                           "Abstention acc.", "Grounded"],
                  rows=[],
                  unavailable_reason=(
                      "no QA benchmark for this video. Create one with "
                      "`evaluate <video_id> --write-qa-template`, fill in the expected "
                      "answers by hand, then re-run with --run-qa."
                  ))
        )
    elif not run_qa:
        tables.append(
            Table(caption="QA accuracy against baselines", label="tab:qa",
                  columns=["System", "Questions", "Counting acc.", "Object F1",
                           "Abstention acc.", "Grounded"],
                  rows=[],
                  unavailable_reason=(
                      f"benchmark of {len(benchmark.items)} question(s) found but not run. "
                      "Re-run with --run-qa (needs a live LLM and, for the full system, "
                      "Neo4j + the vector store)."
                  ))
        )
    else:
        gt_map = _gt_map_for_qa(outputs, ground_truth)
        tables.extend(run_qa_benchmark(video_id, benchmark, settings, gt_map, include_baselines))

    report_dir = settings.resolve_path(settings.paths.outputs_dir) / "eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    latex_path = report_dir / f"{video_id}_tables.tex"
    latex_path.write_text(render_document(tables, video_id))
    (report_dir / f"{video_id}_metrics.json").write_text(
        json.dumps(
            {t.label: {"caption": t.caption, "rows": t.rows,
                       "unavailable_reason": t.unavailable_reason} for t in tables},
            indent=2,
        )
    )
    logger.info("evaluate_video: video_id=%s wrote %d table(s) -> %s", video_id, len(tables), latex_path)
    return tables, latex_path
