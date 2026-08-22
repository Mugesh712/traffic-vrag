import { frameUrl } from "../lib/api";

/* The frames the answer actually rests on, grouped by the object they show.
 * Kept large enough to judge -- an analyst is checking whether the crop
 * supports the claim, and a thumbnail too small to read is decoration.
 *
 * Grouped rather than one undifferentiated strip: a flat pool of photos
 * forces the reader to match each one back to a citation by eye, which is
 * exactly the "which vehicle is this" problem the gallery exists to solve. */
export function EvidenceGallery({
  framesByObject,
  onSelectObject,
}: {
  framesByObject: Record<string, string[]>;
  onSelectObject: (id: string) => void;
}) {
  const entries = Object.entries(framesByObject);
  if (entries.length === 0) return null;

  return (
    <div className="flex flex-col gap-2.5">
      <span className="font-display text-[13px] uppercase tracking-[0.12em] text-ink-3">
        Evidence · {entries.reduce((n, [, frames]) => n + frames.length, 0)}
      </span>
      {entries.map(([globalId, paths]) => (
        <div key={globalId} className="flex flex-col gap-1">
          <button
            type="button"
            onClick={() => onSelectObject(globalId)}
            className="w-fit font-mono text-[11px] tabular text-ink-2 transition-colors hover:text-annotate"
          >
            {globalId}
          </button>
          <div className="flex flex-wrap gap-1.5">
            {paths.map((path) => (
              <a
                key={path}
                href={frameUrl(path)}
                target="_blank"
                rel="noreferrer"
                title={path}
                className="group relative block"
              >
                <img
                  src={frameUrl(path)}
                  alt={`${globalId}, evidence frame ${path.split("/").pop()}`}
                  loading="lazy"
                  className="h-[72px] w-[104px] rounded-sm border border-hairline object-cover transition-colors group-hover:border-annotate"
                  onError={(e) => {
                    (e.currentTarget.parentElement as HTMLElement).style.display = "none";
                  }}
                />
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
