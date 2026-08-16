"""M16 — Import benchmark annotations into the harness's ground-truth format.

Turns a public dataset's labels into data/ground_truth/<video_id>.json, so the
tracking tables (MOTA/IDF1/IDSW) become computable without drawing a single box
by hand. Supports the two formats that matter here:

  mot     MOT16/17/20-style gt.txt, the lingua franca of tracking benchmarks:
          frame,id,x,y,w,h,conf,class,visibility  (1-indexed frames, xywh)
  detrac  UA-DETRAC's native XML, which additionally carries vehicle_type per
          target -- so it seeds attribute ground truth as well as boxes.

THE CENTRAL CORRECTNESS ISSUE: FRAME ALIGNMENT. Benchmarks annotate EVERY
video frame; this pipeline samples a subset (M1's frame_sample_interval_sec)
and names frames by their global video index. Importing every annotation would
hand the evaluator ground-truth boxes on frames the system was never shown, and
each one would be scored as a miss -- MOTA would collapse to a large negative
number that measures the sampling rate, not the tracker. So the importer
intersects annotations with the frames actually in M1's manifest, and reports
how many it dropped rather than doing it silently.

Frame indices are also rebased: MOT counts from 1, this pipeline from 0.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.eval.ground_truth import ground_truth_path
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import VideoManifest

logger = get_logger(__name__)

# UA-DETRAC labels vehicles by body type. Our schema separates the detector
# CLASS (what YOLO emits) from the VLM's vehicle_type attribute, so each label
# populates both, at the right granularity.
DETRAC_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "car": ("car", None),        # no finer body type asserted
    "van": ("car", "van"),
    "bus": ("bus", "bus"),
    "truck": ("truck", "truck"),
    "others": ("car", None),
}

# MOT class ids that correspond to something this pipeline detects. MOT17's
# convention: 1=pedestrian, 3=car, 4=van, 5=truck, 6=bus. Everything else
# (distractors, sitting people, occluders) is deliberately dropped -- importing
# a class the detector can never emit would create guaranteed false negatives.
MOT_CLASS_MAP: dict[int, str] = {1: "person", 3: "car", 4: "car", 5: "truck", 6: "bus"}


class GroundTruthImportError(RuntimeError):
    pass


@dataclass
class ImportReport:
    video_id: str
    source: Path
    n_tracks: int = 0
    n_boxes_kept: int = 0
    n_boxes_dropped_unsampled: int = 0
    n_boxes_dropped_class: int = 0
    n_attributes: int = 0
    sampled_frames: int = 0
    annotated_frames: int = 0
    dropped_track_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        pct = (
            100 * self.n_boxes_kept / (self.n_boxes_kept + self.n_boxes_dropped_unsampled)
            if (self.n_boxes_kept + self.n_boxes_dropped_unsampled)
            else 0.0
        )
        lines = [
            f"Imported {self.n_tracks} track(s), {self.n_boxes_kept} box(es) "
            f"-> {ground_truth_path(self.video_id, get_settings())}",
            f"  kept {pct:.1f}% of annotated boxes: the pipeline sampled "
            f"{self.sampled_frames} of {self.annotated_frames} annotated frame(s)",
        ]
        if self.n_boxes_dropped_unsampled:
            lines.append(
                f"  dropped {self.n_boxes_dropped_unsampled} box(es) on frames the "
                "pipeline never sampled (scoring them would be a guaranteed miss)"
            )
        if self.n_boxes_dropped_class:
            lines.append(
                f"  dropped {self.n_boxes_dropped_class} box(es) of classes this "
                "pipeline does not detect"
            )
        if self.dropped_track_ids:
            lines.append(
                f"  {len(self.dropped_track_ids)} track(s) vanished entirely "
                "(never visible on a sampled frame)"
            )
        if self.n_attributes:
            lines.append(f"  seeded {self.n_attributes} attribute annotation(s) from the source")
        return "\n".join(lines)


def _sampled_frame_index(manifest: VideoManifest) -> dict[int, str]:
    """global 0-based video frame number -> the pipeline's frame_id.

    M1 names sampled frames frame_%06d using the GLOBAL frame index, so the
    number embedded in the id is exactly the index a benchmark counts in.
    """
    index: dict[int, str] = {}
    for clip in manifest.clips:
        for frame in clip.frames:
            match = re.search(r"(\d+)$", frame.frame_id)
            if match:
                index[int(match.group(1))] = frame.frame_id
    return index


def _load_manifest(video_id: str, settings: PipelineSettings) -> VideoManifest:
    path = settings.resolve_path(settings.paths.outputs_dir) / "ingest" / f"{video_id}_manifest.json"
    if not path.exists():
        raise GroundTruthImportError(
            f"No ingest manifest at {path}. Run `ingest` first -- the importer has to "
            "know which frames the pipeline actually sampled."
        )
    return VideoManifest.model_validate_json(path.read_text())


def _parse_mot(source: Path) -> tuple[list[dict], dict[str, dict]]:
    """MOT gt.txt -> (rows, attributes). Rows carry 1-indexed frame numbers."""
    rows: list[dict] = []
    for line_no, line in enumerate(source.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise GroundTruthImportError(
                f"{source}:{line_no}: expected at least 6 comma-separated fields "
                f"(frame,id,x,y,w,h), got {len(parts)}"
            )
        try:
            frame, track_id = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = (float(p) for p in parts[2:6])
        except ValueError as exc:
            raise GroundTruthImportError(f"{source}:{line_no}: {exc}") from exc

        # MOT marks ignored/negative rows with conf=0; they are not objects.
        if len(parts) >= 7 and parts[6] not in ("", "-1") and float(parts[6]) == 0:
            continue

        class_id = int(float(parts[7])) if len(parts) >= 8 and parts[7] not in ("", "-1") else 3
        rows.append({
            "frame": frame, "track_id": track_id,
            "box": [x, y, x + w, y + h],  # xywh (top-left) -> xyxy
            "class_id": class_id,
        })
    return rows, {}


def _parse_detrac(source: Path) -> tuple[list[dict], dict[str, dict]]:
    """UA-DETRAC XML -> (rows, attributes). Frame numbers are 1-indexed."""
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise GroundTruthImportError(f"{source}: not valid XML ({exc})") from exc

    rows: list[dict] = []
    attributes: dict[str, dict] = {}
    for frame_el in root.iter("frame"):
        try:
            frame_num = int(frame_el.get("num"))
        except (TypeError, ValueError) as exc:
            raise GroundTruthImportError(f"{source}: <frame> missing a numeric num=") from exc

        for target in frame_el.iter("target"):
            target_id = target.get("id")
            box_el = target.find("box")
            if target_id is None or box_el is None:
                continue
            left = float(box_el.get("left", 0))
            top = float(box_el.get("top", 0))
            width = float(box_el.get("width", 0))
            height = float(box_el.get("height", 0))

            attr_el = target.find("attribute")
            vehicle_type = (attr_el.get("vehicle_type") if attr_el is not None else None) or "car"
            cls, body_type = DETRAC_TYPE_MAP.get(vehicle_type.lower(), ("car", None))

            rows.append({
                "frame": frame_num, "track_id": int(target_id),
                "box": [left, top, left + width, top + height],
                "class_name": cls,
            })
            if body_type:
                attributes[f"gt_{target_id}"] = {"vehicle_type": body_type}
    return rows, attributes


PARSERS = {"mot": _parse_mot, "detrac": _parse_detrac}


def import_ground_truth(
    video_id: str,
    source: str | Path,
    fmt: str,
    settings: PipelineSettings | None = None,
    frame_offset: int = -1,
    overwrite: bool = False,
) -> ImportReport:
    """Convert `source` into data/ground_truth/<video_id>.json.

    `frame_offset` is added to the source's frame numbers to reach this
    pipeline's 0-based global index; the default -1 converts the 1-indexed
    convention both supported formats use.
    """
    settings = settings or get_settings()
    source = Path(source)
    if not source.exists():
        raise GroundTruthImportError(f"No annotation file at {source}")
    if fmt not in PARSERS:
        raise GroundTruthImportError(f"Unknown format {fmt!r}; expected one of {sorted(PARSERS)}")

    destination = ground_truth_path(video_id, settings)
    if destination.exists() and not overwrite:
        raise GroundTruthImportError(
            f"{destination} already exists; pass overwrite=True to replace it."
        )

    manifest = _load_manifest(video_id, settings)
    sampled = _sampled_frame_index(manifest)
    rows, source_attributes = PARSERS[fmt](source)

    report = ImportReport(
        video_id=video_id, source=source,
        sampled_frames=len(sampled),
        annotated_frames=len({r["frame"] for r in rows}),
    )

    boxes_by_track: dict[str, dict[str, list[float]]] = defaultdict(dict)
    classes: dict[str, str] = {}

    for row in rows:
        cls = row.get("class_name") or MOT_CLASS_MAP.get(row.get("class_id", 3))
        if cls is None:
            report.n_boxes_dropped_class += 1
            continue

        frame_id = sampled.get(row["frame"] + frame_offset)
        if frame_id is None:
            # Annotated, but on a frame the pipeline never saw.
            report.n_boxes_dropped_unsampled += 1
            continue

        gt_id = f"gt_{row['track_id']}"
        boxes_by_track[gt_id][frame_id] = [round(v, 2) for v in row["box"]]
        classes.setdefault(gt_id, cls)
        report.n_boxes_kept += 1

    all_track_ids = {f"gt_{r['track_id']}" for r in rows}
    report.dropped_track_ids = sorted(all_track_ids - set(boxes_by_track))
    report.n_tracks = len(boxes_by_track)

    attributes = []
    for gt_id in sorted(boxes_by_track):
        entry = {"gt_id": gt_id, "color": None, "vehicle_type": None, "make": None, "model": None}
        entry.update(source_attributes.get(gt_id, {}))
        if entry["vehicle_type"]:
            report.n_attributes += 1
        attributes.append(entry)

    payload = {
        "video_id": video_id,
        "tracks": [
            {"gt_id": gt_id, "class": classes[gt_id], "boxes": boxes_by_track[gt_id]}
            for gt_id in sorted(boxes_by_track)
        ],
        "attributes": attributes,
    }

    # No unreviewed marker: unlike the hand-annotation template, this is real
    # human-made ground truth from a published benchmark, so it is immediately
    # usable. Colour/make are still null and need a human if those tables matter.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))

    if report.n_tracks == 0:
        logger.warning(
            "import_ground_truth: no annotation survived frame alignment. Check that "
            "--frame-offset matches the source's indexing convention (currently %d).",
            frame_offset,
        )
    logger.info("import_ground_truth: %s", report.summary().replace("\n", " | "))
    return report
