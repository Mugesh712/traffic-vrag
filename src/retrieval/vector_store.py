"""M11 — Vector store & object timeline embeddings.

Turns each global object and each event into a natural-language sentence,
embeds it, and stores it in ChromaDB alongside structured metadata so M12 can
combine semantic search with exact metadata filters (e.g. color=white AND
event_type=OVERTAKE).

STRUCTURE, same split as M10: `build_object_summary()` / `build_event_summary()`
/ the metadata builders are pure functions from pipeline JSON to plain values,
testable without a vector store. `populate_vector_store()` is the only part
that touches ChromaDB.

TWO COLLECTIONS. Object timelines summarize a whole object's story across
every clip it appeared in ("white sedan ... overtook a blue truck ... stopped
..."); events are embedded individually so a query about one specific moment
("who stopped near the intersection") is not diluted by the rest of an
object's timeline sitting in the same vector.

WHY BOOLEAN FLAGS FOR EVENT TYPES, NOT A LIST FIELD. An object's metadata
needs to answer "did this object ever OVERTAKE", but Chroma's stable metadata
model is flat scalars (str/int/float/bool) -- there is no reliable "value is
in this list" filter across versions. Storing `event_OVERTAKE: true` per known
event type turns that into a plain boolean equality filter, at the cost of
one column per event type, which is a small, fixed set (six).

IDEMPOTENCY: every record for a video_id is deleted before upserting the
current run, mirroring M10's scoped delete-then-insert -- otherwise an object
that a rerun no longer produces (e.g. after a threshold change) would be left
behind as a stale, unreachable-by-rerun vector.

Output: data/outputs/vector_store/ (Chroma's on-disk persistent store)
"""
from __future__ import annotations

from collections import defaultdict

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import EventLog, FinalObjectIndex, GlobalObjectFinal

logger = get_logger(__name__)

EVENT_TYPES = ("OVERTAKE", "STOP", "PARK", "TURN", "LANE_CHANGE", "CROSSES")


class VectorStoreError(RuntimeError):
    pass


def get_chroma_client(settings: PipelineSettings):
    """The one place a Chroma client is constructed.

    Embedded and server modes expose the same collection API, so every caller
    is mode-agnostic -- which is what lets docker-compose run a real chromadb
    service without the pipeline code knowing or caring.
    """
    import chromadb

    mode = settings.vector_store.mode
    if mode == "embedded":
        return chromadb.PersistentClient(
            path=str(settings.resolve_path(settings.vector_store.persist_dir))
        )
    if mode == "http":
        return chromadb.HttpClient(
            host=settings.vector_store.host, port=settings.vector_store.port
        )
    raise VectorStoreError(
        f"Unknown vector_store.mode: {mode!r} (expected 'embedded' or 'http')"
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _attr(obj: GlobalObjectFinal, name: str) -> str | None:
    """A confirmed (not-uncertain) attribute value, or None."""
    for a in obj.attributes:
        if a.attribute == name and not a.uncertain and a.value:
            return a.value
    return None


def _descriptor(obj: GlobalObjectFinal) -> str:
    """"White Toyota sedan" -- as specific as confirmed attributes allow,
    falling back to the coarse detector class when nothing else is known."""
    parts = []
    color = _attr(obj, "color")
    make = _attr(obj, "make")
    if color:
        parts.append(color.capitalize())
    if make:
        parts.append(make.capitalize())
    parts.append(_attr(obj, "vehicle_type") or obj.cls)
    return " ".join(parts)


def _format_time(iso: str) -> str:
    """"2026-08-15T10:00:01.04" -> "10:00:01". Drops date and fractional
    seconds; a timeline summary reads naturally as clock time, not a
    timestamp."""
    time_part = iso.split("T")[-1] if "T" in iso else iso
    return time_part.split(".")[0]


def _first_last_seen(obj: GlobalObjectFinal) -> tuple[str | None, str | None]:
    if not obj.sightings:
        return None, None
    return (
        min(s.first_seen for s in obj.sightings),
        max(s.last_seen for s in obj.sightings),
    )


def _event_phrase(event, objects_by_id: dict[str, GlobalObjectFinal]) -> str | None:
    """One clause describing `event` from its subject's point of view, or None
    for an event type with nothing natural to say (kept out of the sentence
    rather than forcing an awkward generic phrase)."""
    if event.type == "OVERTAKE":
        other = objects_by_id.get(event.object_id) if event.object_id else None
        other_desc = _descriptor(other) if other else "another vehicle"
        return f"overtook {_article(other_desc)}{other_desc.lower()}"
    if event.type == "STOP":
        return "stopped"
    if event.type == "PARK":
        return "parked"
    if event.type == "TURN":
        direction = event.metadata.get("turn_direction")
        return f"turned {direction}" if direction else "turned"
    if event.type == "LANE_CHANGE":
        return "changed lanes"
    if event.type == "CROSSES":
        region = event.metadata.get("region_id")
        return f"crossed {region}" if region else "crossed a marked line"
    return None


def _article(descriptor: str) -> str:
    return "an " if descriptor[:1].lower() in "aeiou" else "a "


def _was_overtaken_phrase(event, objects_by_id: dict[str, GlobalObjectFinal]) -> str:
    other = objects_by_id.get(event.subject_id)
    other_desc = _descriptor(other) if other else "another vehicle"
    return f"was overtaken by {_article(other_desc)}{other_desc.lower()}"


def build_object_summary(
    obj: GlobalObjectFinal,
    events: list,
    objects_by_id: dict[str, GlobalObjectFinal],
) -> str:
    """"White Toyota sedan, first seen 11:02:10 heading east, overtook a blue
    truck at 11:04:20, stopped at 11:05:42." -- the descriptor, then every
    event this object participated in, chronologically."""
    first_seen, last_seen = _first_last_seen(obj)
    sentence = [_descriptor(obj)]

    if first_seen:
        clause = f"first seen {_format_time(first_seen)}"
        direction = _attr(obj, "direction")
        if direction:
            clause += f" heading {direction}"
        sentence.append(clause)

    involved = sorted(
        (e for e in events if e.subject_id == obj.global_id or e.object_id == obj.global_id),
        key=lambda e: e.start_time,
    )
    for event in involved:
        if event.subject_id == obj.global_id:
            phrase = _event_phrase(event, objects_by_id)
        else:
            phrase = _was_overtaken_phrase(event, objects_by_id) if event.type == "OVERTAKE" else None
        if phrase:
            sentence.append(f"{phrase} at {_format_time(event.start_time)}")

    if not involved and last_seen and last_seen != first_seen:
        sentence[-1] = f"seen from {_format_time(first_seen)} to {_format_time(last_seen)}"

    return ", ".join(sentence) + "."


def build_object_metadata(video_id: str, obj: GlobalObjectFinal, events: list) -> dict:
    first_seen, last_seen = _first_last_seen(obj)

    # Subject-role only. event_OVERTAKE=true must mean "this object did the
    # overtaking", not "this object was involved in an overtake in any role" --
    # the roadmap's filter example ("color=white AND event_type=OVERTAKE") is
    # asking "which white vehicles overtook something", and a role-blind flag
    # would also match the vehicle that got overtaken. Caught by querying the
    # real store: filtering color=blue AND event_OVERTAKE=true wrongly
    # returned the blue truck that was passed, not an overtaking blue vehicle.
    subject_types = {e.type for e in events if e.subject_id == obj.global_id}
    any_role_events = sum(
        1 for e in events if e.subject_id == obj.global_id or e.object_id == obj.global_id
    )

    metadata = {
        "video_id": video_id,
        "global_id": obj.global_id,
        "class": obj.cls,
        "color": _attr(obj, "color") or "unknown",
        "make": _attr(obj, "make") or "unknown",
        "vehicle_type": _attr(obj, "vehicle_type") or "unknown",
        "direction": _attr(obj, "direction") or "unknown",
        "first_seen": first_seen or "",
        "last_seen": last_seen or "",
        "n_clips": len(obj.sightings),
        "n_events": any_role_events,
    }
    for event_type in EVENT_TYPES:
        metadata[f"event_{event_type}"] = event_type in subject_types
    return metadata


def build_event_summary(event, objects_by_id: dict[str, GlobalObjectFinal]) -> str:
    """One event, described on its own -- so a query about a single moment
    is not diluted by the rest of an object's timeline."""
    subject = objects_by_id.get(event.subject_id)
    subject_desc = _descriptor(subject) if subject else "A vehicle"
    time = _format_time(event.start_time)

    if event.type == "OVERTAKE":
        other = objects_by_id.get(event.object_id) if event.object_id else None
        other_desc = _descriptor(other) if other else "another vehicle"
        return f"{subject_desc} overtook {_article(other_desc)}{other_desc.lower()} at {time}."
    if event.type in ("STOP", "PARK"):
        verb = "stopped" if event.type == "STOP" else "parked"
        return f"{subject_desc} {verb} at {time}."
    if event.type == "TURN":
        direction = event.metadata.get("turn_direction")
        clause = f"turned {direction}" if direction else "turned"
        return f"{subject_desc} {clause} at {time}."
    if event.type == "LANE_CHANGE":
        return f"{subject_desc} changed lanes at {time}."
    if event.type == "CROSSES":
        region = event.metadata.get("region_id")
        clause = f"crossed {region}" if region else "crossed a marked line"
        return f"{subject_desc} {clause} at {time}."
    return f"{subject_desc} was involved in a {event.type.lower()} event at {time}."


def build_event_metadata(video_id: str, event, objects_by_id: dict[str, GlobalObjectFinal]) -> dict:
    subject = objects_by_id.get(event.subject_id)
    return {
        "video_id": video_id,
        "event_id": event.event_id,
        "type": event.type,
        "subject_id": event.subject_id,
        "object_id": event.object_id or "",
        "subject_color": (_attr(subject, "color") if subject else None) or "unknown",
        "subject_class": subject.cls if subject else "unknown",
        "start_time": event.start_time,
        "end_time": event.end_time,
        "confidence": event.confidence,
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def populate_vector_store(video_id: str, settings: PipelineSettings | None = None):
    settings = settings or get_settings()
    outputs = settings.resolve_path(settings.paths.outputs_dir)

    final_path = outputs / "global_objects_final" / f"{video_id}.json"
    if not final_path.exists():
        raise VectorStoreError(f"No confirmed objects at {final_path}; run `confirm` first.")
    final_index = FinalObjectIndex.model_validate_json(final_path.read_text())
    objects_by_id = {o.global_id: o for o in final_index.objects}

    events_path = outputs / "events" / f"{video_id}.json"
    events = []
    if events_path.exists():
        events = EventLog.model_validate_json(events_path.read_text()).events
    else:
        logger.info("populate_vector_store: no events for %s; timelines will have no event clauses", video_id)

    object_ids, object_docs, object_metas = [], [], []
    for obj in final_index.objects:
        object_ids.append(obj.global_id)
        object_docs.append(build_object_summary(obj, events, objects_by_id))
        object_metas.append(build_object_metadata(video_id, obj, events))

    event_ids, event_docs, event_metas = [], [], []
    for event in events:
        event_ids.append(event.event_id)
        event_docs.append(build_event_summary(event, objects_by_id))
        event_metas.append(build_event_metadata(video_id, event, objects_by_id))

    client = get_chroma_client(settings)

    object_collection = client.get_or_create_collection(settings.vector_store.object_collection)
    object_collection.delete(where={"video_id": video_id})
    if object_ids:
        object_collection.upsert(ids=object_ids, documents=object_docs, metadatas=object_metas)

    event_collection = client.get_or_create_collection(settings.vector_store.event_collection)
    event_collection.delete(where={"video_id": video_id})
    if event_ids:
        event_collection.upsert(ids=event_ids, documents=event_docs, metadatas=event_metas)

    logger.info(
        "populate_vector_store: video_id=%s embedded %d objects, %d events -> %s",
        video_id,
        len(object_ids),
        len(event_ids),
        settings.vector_store.persist_dir,
    )

    return object_ids, event_ids
