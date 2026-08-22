import { frameUrl } from "../lib/api";

/* The frames the answer actually rests on. Kept large enough to judge -- an
 * analyst is checking whether the crop supports the claim, and a thumbnail
 * too small to read is decoration. */
export function EvidenceGallery({ paths }: { paths: string[] }) {
  if (paths.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <span className="font-display text-[13px] uppercase tracking-[0.12em] text-ink-3">
        Evidence · {paths.length}
      </span>
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
              alt={`Evidence frame ${path.split("/").pop()}`}
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
  );
}
