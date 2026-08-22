import { useEffect, useMemo, useState } from "react";
import { ApiError, frameUrl, getObject, listObjects } from "../lib/api";
import { AttributeRow, ProvenanceLegend } from "./AttributeRow";
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

  const selected = visible.find((o) => o.global_id === selectedId) ?? visible[0] ?? null;

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
              <li key={o.global_id}>
                <button
                  type="button"
                  onClick={() => onSelect(o.global_id)}
                  aria-current={isSel}
                  className={`flex w-full items-center gap-2 border-l-2 py-1.5 pl-2.5 pr-2 text-left transition-colors ${
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
              </li>
            );
          })}
        </ul>
      </div>

      {selected ? <ObjectRecord jobId={jobId} summary={selected} /> : <Muted>Nothing selected.</Muted>}
    </div>
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
    <div className="flex min-w-0 flex-col gap-4 rounded-sm border border-hairline bg-console p-4">
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
                  className="h-20 w-20 rounded-sm border border-hairline object-cover transition-colors hover:border-annotate"
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
            <SightingStrip timeline={detail.timeline} />
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

/** When this object was on camera, per clip. Bars are drawn against the
 * object's own full observed range, so a short sighting reads as short. */
function SightingStrip({ timeline }: { timeline: ObjectDetail["timeline"] }) {
  const t = (iso: string) => new Date(iso).getTime();
  const min = Math.min(...timeline.map((s) => t(s.first_seen)));
  const max = Math.max(...timeline.map((s) => t(s.last_seen)));
  const span = Math.max(max - min, 1);
  const clock = (iso: string) => iso.split("T")[1]?.slice(0, 8) ?? iso;

  return (
    <ul className="flex flex-col gap-1">
      {timeline.map((s, i) => {
        const a = t(s.first_seen);
        const b = t(s.last_seen);
        return (
          <li key={`${s.clip_id}-${s.track_id}-${i}`} className="flex items-center gap-2">
            <span className="w-20 flex-none font-mono text-[11px] text-ink-3">{s.clip_id}</span>
            <span className="relative h-3 flex-1 rounded-[2px] border border-hairline bg-well">
              <span
                className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-[2px] bg-sodium"
                style={{
                  left: `${((a - min) / span) * 100}%`,
                  width: `${Math.max(((b - a) / span) * 100, 2)}%`,
                }}
                title={`${clock(s.first_seen)} – ${clock(s.last_seen)} · track ${s.track_id}`}
              />
            </span>
            <span className="flex-none font-mono text-[11px] tabular text-ink-3">
              {clock(s.first_seen)}→{clock(s.last_seen).slice(-2)}
            </span>
          </li>
        );
      })}
    </ul>
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
      className={`rounded-sm border px-1.5 py-0.5 font-mono text-[11px] transition-colors ${
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
