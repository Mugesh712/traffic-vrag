import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, frameUrl, getObject, listObjects } from "../lib/api";
import { AttributeRow, ProvenanceLegend } from "./AttributeRow";
import { HoverPreview } from "./HoverPreview";
import { ObjectPreviewCard } from "./ObjectPreviewCard";
import type { ObjectDetail, ObjectSummary } from "../lib/types";

/* Roster: a list to scan, a record to read.
 *
 * Master-detail rather than a grid of cards. An analyst works one object at a
 * time -- crop, attributes, when it was on camera -- and a grid forces every
 * record to be small enough to tile, which makes all of them too small to
 * actually judge. */

interface Props {
  jobId: string;
  ready: boolean;
  selectedId: string | null;
  onSelect: (globalId: string | null) => void;
}

export function ObjectExplorer({ jobId, ready, selectedId, onSelect }: Props) {
  const [objects, setObjects] = useState<ObjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [classFilter, setClassFilter] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    listObjects(jobId)
      .then((data) => !cancelled && setObjects(data))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Could not load objects."));
    return () => {
      cancelled = true;
    };
  }, [jobId, ready]);

  const counts = useMemo(() => {
    const c = new Map<string, number>();
    for (const o of objects ?? []) c.set(o.class, (c.get(o.class) ?? 0) + 1);
    return [...c.entries()].sort((a, b) => b[1] - a[1]);
  }, [objects]);

  const visible = useMemo(
    () => (objects ?? []).filter((o) => !classFilter || o.class === classFilter),
    [objects, classFilter],
  );

  if (!ready) return <Muted>Objects appear once the confirm stage finishes.</Muted>;
  if (error)
    return (
      <p className="rounded-sm border border-flag/40 bg-flag-dim px-3 py-2 font-mono text-xs text-flag">{error}</p>
    );
  if (objects === null) return <Muted>Loading roster…</Muted>;
  if (objects.length === 0) return <Muted>No objects were detected in this video.</Muted>;

  // Looked up against the FULL roster, not `visible` -- a citation click
  // (obj chip, inline citation) must land on the object it names even if a
  // stale class filter would otherwise exclude it. Falling back to some
  // other object here would silently show the wrong record with no
  // indication anything was substituted.
  const selected = selectedId
    ? (objects.find((o) => o.global_id === selectedId) ?? null)
    : (visible[0] ?? null);
  const selectedIsMissing = selectedId !== null && selected === null;

  return (
    <div className="grid gap-5 md:grid-cols-[210px_1fr]">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-1.5">
          <FilterPill active={classFilter === null} onClick={() => setClassFilter(null)}>
            all {objects.length}
          </FilterPill>
          {counts.map(([cls, n]) => (
            <FilterPill key={cls} active={classFilter === cls} onClick={() => setClassFilter(cls)}>
              {cls} {n}
            </FilterPill>
          ))}
        </div>

        <ul className="flex max-h-[62vh] flex-col overflow-y-auto md:max-h-[70vh]">
          {visible.map((o) => {
            const isSel = selected?.global_id === o.global_id;
            const confirmed = o.attributes.filter((a) => !a.uncertain && a.value).length;
            return (
              <li key={o.global_id} className="relative block">
                <HoverPreview
                  className="relative block"
                  render={() => <RosterPreview jobId={jobId} summary={o} />}
                >
                  <button
                    type="button"
                    onClick={() => onSelect(o.global_id)}
                    aria-current={isSel}
                    className={`interactive flex w-full items-center gap-2 border-l-2 py-1.5 pl-2.5 pr-2 text-left ${
                      isSel
                        ? "border-annotate bg-annotate-dim"
                        : "border-transparent hover:border-hairline-lit hover:bg-console"
                    }`}
                  >
                    <span className="font-mono text-[12px] tabular text-ink-data">{o.global_id}</span>
                    <span className="ml-auto font-mono text-[11px] text-ink-3">{o.class}</span>
                    {/* How much of this record is actually established. */}
                    <span
                      className="font-mono text-[11px] tabular text-ink-3"
                      title={`${confirmed} of ${o.attributes.length} attributes confirmed`}
                    >
                      {confirmed}/{o.attributes.length}
                    </span>
                  </button>
                </HoverPreview>
              </li>
            );
          })}
        </ul>
      </div>

      {selected ? (
        <ObjectRecord jobId={jobId} summary={selected} />
      ) : selectedIsMissing ? (
        <Muted>{selectedId} isn't in this video's roster.</Muted>
      ) : (
        <Muted>Nothing selected.</Muted>
      )}
    </div>
  );
}

/** The roster row's hover popover fetches its own detail lazily -- only when
 * actually hovered, via the same GET the record page already uses on
 * selection. Attributes render instantly from the summary already in memory;
 * the crop fades in once the fetch resolves. */
function RosterPreview({ jobId, summary }: { jobId: string; summary: ObjectSummary }) {
  const [detail, setDetail] = useState<ObjectDetail | null>(null);

  useEffect(() => {
    let cancelled = false;
    getObject(jobId, summary.global_id)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [jobId, summary.global_id]);

  return (
    <ObjectPreviewCard
      globalId={summary.global_id}
      label={summary.class}
      cropUrl={detail ? (detail.best_shot_crops[0] ?? null) : undefined}
      cropLoading={!detail}
      attributes={summary.attributes}
    />
  );
}

function ObjectRecord({ jobId, summary }: { jobId: string; summary: ObjectSummary }) {
  const [detail, setDetail] = useState<ObjectDetail | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    getObject(jobId, summary.global_id)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [jobId, summary.global_id]);

  return (
    <div className="panel-in flex min-w-0 flex-col gap-4 rounded-sm border border-hairline bg-console p-4 shadow-[0_1px_0_rgba(255,255,255,0.03)_inset,0_12px_28px_-16px_rgba(0,0,0,0.55)]">
      <div className="flex items-baseline gap-2.5">
        <h2 className="font-mono text-lg tabular text-ink">{summary.global_id}</h2>
        <span className="font-display text-[15px] uppercase tracking-[0.12em] text-sodium">{summary.class}</span>
      </div>

      <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
        {detail && detail.best_shot_crops.length > 0 && (
          <div className="flex gap-1.5 sm:flex-col">
            {detail.best_shot_crops.slice(0, 3).map((path) => (
              <a key={path} href={frameUrl(path)} target="_blank" rel="noreferrer" title={path}>
                <img
                  src={frameUrl(path)}
                  alt={`Best-shot crop of ${summary.global_id}`}
                  className="scanlines interactive h-20 w-20 rounded-sm border border-hairline object-cover hover:border-annotate"
                  onError={(e) => {
                    (e.currentTarget.parentElement as HTMLElement).style.display = "none";
                  }}
                />
              </a>
            ))}
          </div>
        )}

        <div className="flex min-w-0 flex-col gap-1.5">
          {summary.attributes.map((a) => (
            <AttributeRow key={a.attribute} attr={a} />
          ))}
          <div className="mt-1">
            <ProvenanceLegend />
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <span className="font-display text-[13px] uppercase tracking-[0.12em] text-ink-3">Sightings</span>
        {detail ? (
          detail.timeline.length > 0 ? (
            <ScrubbableTimeline
              timeline={detail.timeline}
              bestShotCrops={detail.best_shot_crops}
              globalId={summary.global_id}
            />
          ) : (
            <Muted>No sightings recorded.</Muted>
          )
        ) : (
          <Muted>Loading…</Muted>
        )}
      </div>
    </div>
  );
}

/** When this object was on camera, per clip, with a draggable playhead over
 * the whole observed range. Dragging swaps which best-shot crop is "in
 * focus" above the track -- proportionally across the range, since crops
 * carry no per-frame timestamp, so the label says "illustrative" rather than
 * claiming a precision the data doesn't have. The clip/time readout beside
 * it IS exact: it is just whichever real sighting segment the playhead
 * currently sits over. */
function ScrubbableTimeline({
  timeline,
  bestShotCrops,
  globalId,
}: {
  timeline: ObjectDetail["timeline"];
  bestShotCrops: string[];
  globalId: string;
}) {
  const t = (iso: string) => new Date(iso).getTime();
  const min = Math.min(...timeline.map((s) => t(s.first_seen)));
  const max = Math.max(...timeline.map((s) => t(s.last_seen)));
  const span = Math.max(max - min, 1);
  const clock = (iso: string) => iso.split("T")[1]?.slice(0, 8) ?? iso;

  const trackRef = useRef<HTMLDivElement>(null);
  const [frac, setFrac] = useState(0);
  const [dragging, setDragging] = useState(false);

  const updateFromClientX = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const f = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
    setFrac(Math.min(1, Math.max(0, f)));
  };

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => updateFromClientX(e.clientX);
    const onUp = () => setDragging(false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging]);

  const scrubTime = min + frac * span;
  const currentSighting =
    timeline.find((s) => scrubTime >= t(s.first_seen) && scrubTime <= t(s.last_seen)) ??
    [...timeline].sort((a, b) => Math.abs(t(a.first_seen) - scrubTime) - Math.abs(t(b.first_seen) - scrubTime))[0];
  const cropIndex = bestShotCrops.length > 0 ? Math.min(bestShotCrops.length - 1, Math.floor(frac * bestShotCrops.length)) : -1;
  const focusedCrop = cropIndex >= 0 ? bestShotCrops[cropIndex] : null;

  const minSighting = timeline.find((s) => t(s.first_seen) === min);
  const maxSighting = timeline.find((s) => t(s.last_seen) === max);

  return (
    <div className="flex flex-col gap-2.5">
      {focusedCrop && (
        <div className="flex items-center gap-3">
          <img
            src={frameUrl(focusedCrop)}
            alt={`Representative crop of ${globalId} near this point in its timeline`}
            className="scanlines h-20 w-20 flex-none rounded-sm border border-hairline object-cover"
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
          <div className="flex flex-col gap-0.5 font-mono text-[11px] text-ink-3">
            <span>
              illustrative crop {cropIndex + 1}/{bestShotCrops.length} — not exactly this instant
            </span>
            {currentSighting && (
              <span className="tabular text-ink-data">
                {currentSighting.clip_id} · {clock(currentSighting.first_seen)}–{clock(currentSighting.last_seen)}
              </span>
            )}
          </div>
        </div>
      )}

      <div
        ref={trackRef}
        className="relative h-8 cursor-ew-resize touch-none select-none rounded-sm border border-hairline bg-well"
        onPointerDown={(e) => {
          setDragging(true);
          updateFromClientX(e.clientX);
        }}
        role="slider"
        aria-label={`Scrub ${globalId}'s observed timeline`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(frac * 100)}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setFrac((f) => Math.max(0, f - 0.05));
          if (e.key === "ArrowRight") setFrac((f) => Math.min(1, f + 0.05));
        }}
      >
        {timeline.map((s, i) => {
          const a = t(s.first_seen);
          const b = t(s.last_seen);
          return (
            <div
              key={`${s.clip_id}-${s.track_id}-${i}`}
              title={`${s.clip_id}: ${clock(s.first_seen)} – ${clock(s.last_seen)}`}
              className="absolute top-1/2 h-2 -translate-y-1/2 rounded-[2px] bg-sodium/70"
              style={{
                left: `${((a - min) / span) * 100}%`,
                width: `${Math.max(((b - a) / span) * 100, 2)}%`,
              }}
            />
          );
        })}
        <div
          className={`absolute top-0 bottom-0 w-[2px] bg-annotate ${dragging ? "" : "transition-[left] duration-150"}`}
          style={{ left: `${frac * 100}%` }}
        >
          <div className="absolute -top-1 left-1/2 h-2.5 w-2.5 -translate-x-1/2 rotate-45 rounded-[1px] bg-annotate" />
        </div>
      </div>

      <div className="flex justify-between font-mono text-[11px] tabular text-ink-3">
        <span>{minSighting ? clock(minSighting.first_seen) : ""}</span>
        <span>{maxSighting ? clock(maxSighting.last_seen) : ""}</span>
      </div>
    </div>
  );
}

function FilterPill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`interactive rounded-sm border px-1.5 py-0.5 font-mono text-[11px] ${
        active
          ? "border-annotate bg-annotate-dim text-annotate"
          : "border-hairline text-ink-3 hover:border-hairline-lit hover:text-ink-2"
      }`}
    >
      {children}
    </button>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return <p className="font-mono text-xs text-ink-3">{children}</p>;
}
