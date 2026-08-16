import { useEffect, useState } from "react";
import { ApiError, frameUrl, getObject, listObjects } from "../lib/api";
import { StatusPill } from "./StatusPill";
import type { ObjectDetail, ObjectSummary } from "../lib/types";

interface Props {
  jobId: string;
  ready: boolean;
  selectedId: string | null;
  onSelect: (globalId: string | null) => void;
}

export function ObjectExplorer({ jobId, ready, selectedId, onSelect }: Props) {
  const [objects, setObjects] = useState<ObjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  if (!ready) {
    return <p className="text-sm text-ink-faint">Objects appear here once the confirm stage finishes.</p>;
  }
  if (error) {
    return <p className="rounded-md border border-flag/30 bg-flag/10 px-3 py-2 text-sm text-flag">{error}</p>;
  }
  if (objects === null) {
    return <p className="font-mono text-xs text-ink-faint">Loading objects…</p>;
  }
  if (objects.length === 0) {
    return <p className="text-sm text-ink-faint">No objects were detected in this video.</p>;
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {objects.map((obj) => (
        <ObjectCard
          key={obj.global_id}
          jobId={jobId}
          summary={obj}
          expanded={selectedId === obj.global_id}
          onToggle={() => onSelect(selectedId === obj.global_id ? null : obj.global_id)}
        />
      ))}
    </div>
  );
}

function ObjectCard({
  jobId,
  summary,
  expanded,
  onToggle,
}: {
  jobId: string;
  summary: ObjectSummary;
  expanded: boolean;
  onToggle: () => void;
}) {
  const [detail, setDetail] = useState<ObjectDetail | null>(null);

  useEffect(() => {
    if (expanded && !detail) {
      getObject(jobId, summary.global_id).then(setDetail).catch(() => {});
    }
  }, [expanded, detail, jobId, summary.global_id]);

  const confirmed = summary.attributes.filter((a) => !a.uncertain && a.value);

  return (
    <div
      className={`flex flex-col gap-2.5 rounded-lg border p-3 transition-colors ${
        expanded ? "border-accent bg-accent-soft" : "border-border bg-surface hover:border-ink-faint"
      }`}
    >
      <button onClick={onToggle} className="flex items-center justify-between gap-2 text-left">
        <span className="font-mono text-sm text-ink">{summary.global_id}</span>
        <StatusPill tone="neutral">{summary.class}</StatusPill>
      </button>

      <div className="flex flex-col gap-1">
        {confirmed.length === 0 && <span className="text-xs text-ink-faint">No confirmed attributes</span>}
        {confirmed.map((attr) => (
          <div key={attr.attribute} className="flex items-center justify-between gap-2 text-xs">
            <span className="text-ink-faint">{attr.attribute}</span>
            <span className="flex items-center gap-1.5 font-mono text-ink">
              {attr.value}
              <span className="text-ink-faint">{(attr.confidence * 100).toFixed(0)}%</span>
            </span>
          </div>
        ))}
      </div>

      {expanded && detail && (
        <div className="mt-1 flex flex-col gap-2 border-t border-border pt-2.5">
          {detail.best_shot_crops.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {detail.best_shot_crops.map((path) => (
                <img
                  key={path}
                  src={frameUrl(path)}
                  alt=""
                  className="h-14 w-14 rounded border border-border object-cover"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              ))}
            </div>
          )}
          <ul className="flex flex-col gap-0.5 font-mono text-[11px] text-ink-faint">
            {detail.timeline.map((sighting, i) => (
              <li key={i}>
                {sighting.clip_id}: {sighting.first_seen.split("T")[1]?.slice(0, 8)} –{" "}
                {sighting.last_seen.split("T")[1]?.slice(0, 8)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
