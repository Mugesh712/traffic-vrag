import type { TimestampSpan } from "../lib/types";

interface Props {
  spans: TimestampSpan[];
}

// A horizontal strip of the cited moments, positioned along the observed
// time range -- not a generic list, since relative position is the thing
// this view adds over just reading the citations in the answer text.
export function TimelineStrip({ spans }: Props) {
  if (spans.length === 0) return null;

  const toSeconds = (t: string) => {
    const [h, m, s] = t.split(":").map(Number);
    return h * 3600 + m * 60 + s;
  };
  const starts = spans.map((s) => toSeconds(s.start));
  const ends = spans.map((s) => toSeconds(s.end));
  const min = Math.min(...starts);
  const max = Math.max(...ends, min + 1);
  const pct = (t: number) => `${((t - min) / (max - min)) * 100}%`;

  return (
    <div className="flex flex-col gap-2">
      <span className="font-mono text-xs uppercase tracking-wide text-ink-faint">Timeline</span>
      <div className="relative h-8 rounded-md border border-border bg-surface-sunken">
        {spans.map((span, i) => {
          const s = toSeconds(span.start);
          const e = toSeconds(span.end);
          const width = Math.max(((e - s) / (max - min)) * 100, 1.5);
          return (
            <div
              key={i}
              title={`${span.start} – ${span.end}`}
              className="absolute top-1/2 h-2.5 -translate-y-1/2 rounded-sm bg-accent/70"
              style={{ left: pct(s), width: `${width}%` }}
            />
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[11px] text-ink-faint">
        <span>{spans.find((s) => toSeconds(s.start) === min)?.start}</span>
        <span>{spans.find((s) => toSeconds(s.end) === max)?.end}</span>
      </div>
    </div>
  );
}
