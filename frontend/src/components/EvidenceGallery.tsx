import { frameUrl } from "../lib/api";

interface Props {
  paths: string[];
}

export function EvidenceGallery({ paths }: Props) {
  if (paths.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      <span className="font-mono text-xs uppercase tracking-wide text-ink-faint">Evidence frames</span>
      <div className="flex flex-wrap gap-2">
        {paths.map((path) => (
          <a key={path} href={frameUrl(path)} target="_blank" rel="noreferrer" title={path}>
            <img
              src={frameUrl(path)}
              alt=""
              loading="lazy"
              className="h-20 w-28 rounded-md border border-border object-cover transition-colors hover:border-accent"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
          </a>
        ))}
      </div>
    </div>
  );
}
