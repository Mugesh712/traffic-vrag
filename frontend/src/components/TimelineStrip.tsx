import type { TimestampSpan } from "../lib/types";

/* Cited moments placed along the observed range. Relative position is the
 * whole point -- reading the citations in the answer text already gives the
 * times, so this only earns its space by showing how they sit against each
 * other. Ends are labelled directly rather than via an axis: two numbers beat
 * a ruler at this size. */
export function TimelineStrip({ spans }: { spans: TimestampSpan[] }) {
  if (spans.length === 0) return null;

  const toSeconds = (t: string) => {
    const [h, m, s] = t.split(":").map(Number);
    return h * 3600 + m * 60 + s;
  };
  const starts = spans.map((s) => toSeconds(s.start));
  const ends = spans.map((s) => toSeconds(s.end));
  const min = Math.min(...starts);
  const max = Math.max(...ends, min + 1);
  const span = max - min;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="font-display text-[13px] uppercase tracking-[0.12em] text-ink-3">Cited span</span>
      <div className="relative h-6 rounded-sm border border-hairline bg-well">
        {spans.map((s, i) => {
          const a = toSeconds(s.start);
          const b = toSeconds(s.end);
          return (
            <div
              key={`${s.start}-${s.end}-${i}`}
              title={`${s.start} – ${s.end}`}
              className="absolute top-1/2 h-2 -translate-y-1/2 rounded-[2px] bg-sodium"
              style={{
                left: `${((a - min) / span) * 100}%`,
                // Floor the width so an instantaneous citation is still a mark
                // rather than a zero-width sliver.
                width: `${Math.max(((b - a) / span) * 100, 2)}%`,
              }}
            />
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[11px] tabular text-ink-3">
        <span>{spans.find((s) => toSeconds(s.start) === min)?.start}</span>
        <span>{spans.find((s) => toSeconds(s.end) === max)?.end}</span>
      </div>
    </div>
  );
}
