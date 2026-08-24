import { frameUrl } from "../lib/api";
import { AttributeRow } from "./AttributeRow";
import type { ObjectAttribute } from "../lib/types";

/* The one preview card, reused everywhere a tracked object needs a
 * hover-close look: KG nodes, answer chips, roster rows. Two data shapes feed
 * it, because the two contexts genuinely know different things --
 *
 *   attributes   -- the Roster's full ObjectAttribute[] (confidence, source,
 *                    uncertain), rendered through the same AttributeRow used
 *                    on the record page, so the confidence language is one
 *                    system rather than reinvented smaller here.
 *   properties    -- the KG subgraph's flat {key: value} winner-only map. No
 *                    confidence travels with it, so it is shown as plain
 *                    instrument-data pairs rather than faked into a
 *                    confidence bar it cannot honestly support.
 *
 * cropUrl is undefined when the caller never checked (no popover crop
 * section at all), null when it checked and there genuinely is none.
 */

interface Props {
  globalId: string;
  label?: string;
  cropUrl?: string | null;
  cropLoading?: boolean;
  attributes?: ObjectAttribute[];
  properties?: Record<string, unknown>;
}

export function ObjectPreviewCard({ globalId, label, cropUrl, cropLoading, attributes, properties }: Props) {
  return (
    <div className="glass panel-in w-60 max-w-[80vw] rounded-sm p-2.5">
      <div className="mb-2 flex items-baseline gap-2">
        <span className="font-mono text-[12px] tabular text-ink-data">{globalId}</span>
        {label && <span className="ml-auto font-mono text-[10px] uppercase tracking-wide text-ink-3">{label}</span>}
      </div>

      {cropUrl !== undefined && (
        <div className="mb-2">
          {cropUrl ? (
            <img
              src={frameUrl(cropUrl)}
              alt={`Best-shot crop of ${globalId}`}
              className="scanlines h-24 w-full rounded-[2px] border border-hairline object-cover"
              onError={(e) => {
                (e.currentTarget.parentElement as HTMLElement).style.display = "none";
              }}
            />
          ) : cropLoading ? (
            <div className="h-24 w-full animate-pulse rounded-[2px] bg-console-2" />
          ) : (
            <div className="flex h-24 w-full items-center justify-center rounded-[2px] border border-dashed border-hairline/60 font-mono text-[10px] text-ink-3">
              no frame on file
            </div>
          )}
        </div>
      )}

      {attributes && attributes.length > 0 && (
        <div className="flex flex-col gap-1">
          {attributes.map((a) => (
            <AttributeRow key={a.attribute} attr={a} />
          ))}
        </div>
      )}

      {properties && Object.keys(properties).length > 0 && (
        <div className="flex flex-col gap-0.5 font-mono text-[11px]">
          {Object.entries(properties)
            .filter(([, v]) => v != null)
            .map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-2">
                <span className="uppercase tracking-wide text-ink-3">{k.replace("_", " ")}</span>
                <span className="tabular text-ink-data">{String(v)}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
