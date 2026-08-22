import type { ObjectAttribute } from "../lib/types";

/* THE PROVENANCE CHIP -- the signature element.
 *
 * One row carries three real fields at once, so trustworthiness is read at
 * the same glance as the value rather than looked up separately:
 *
 *   uncertain   -> hatched vs solid ground   (texture, survives greyscale)
 *   confidence  -> fill along the lower edge (magnitude)
 *   source      -> one glyph + its label     (provenance)
 *
 * An uncertain attribute is SHOWN, not dropped. Most objects here have no
 * confirmed make or model; hiding those rows would make the panel look
 * tidier and quietly overstate what the pipeline actually established.
 * "Declined to answer" is a result, and it gets drawn like one.
 */

const SOURCE_GLYPH: Record<string, { glyph: string; title: string }> = {
  agreed: { glyph: "◆", title: "best shot agreed with clip-level voting" },
  best_shot: { glyph: "▲", title: "answered by the high-detail best-shot re-read" },
  clip_voting: { glyph: "●", title: "from confidence-weighted clip-level voting" },
  unconfirmed: { glyph: "○", title: "no source could confirm a value" },
};

export function AttributeRow({ attr }: { attr: ObjectAttribute }) {
  const source = SOURCE_GLYPH[attr.source] ?? SOURCE_GLYPH.unconfirmed;
  const hasValue = !attr.uncertain && attr.value;

  return (
    <div
      className={`relative overflow-hidden rounded-sm border ${
        hasValue ? "border-hairline bg-console-2" : "hatch border-hairline/60 bg-console"
      }`}
    >
      <div className="flex items-baseline gap-2 px-2.5 py-1.5">
        <span className="font-display text-[13px] uppercase tracking-wide text-ink-3">
          {attr.attribute.replace("_", " ")}
        </span>
        <span
          className={`ml-auto font-mono text-[13px] tabular ${
            hasValue ? "text-ink-data" : "italic text-ink-3"
          }`}
        >
          {hasValue ? attr.value : "unconfirmed"}
        </span>
        <span
          className={`w-3 text-center text-[11px] ${hasValue ? "text-sodium" : "text-ink-3"}`}
          title={`${attr.source} — ${source.title}`}
          aria-label={`source: ${attr.source}`}
        >
          {source.glyph}
        </span>
      </div>

      {/* Confidence as fill. Absent entirely when there is nothing claimed --
          a 0%-wide bar would still read as a measurement of something. */}
      {hasValue && (
        <div className="h-[3px] w-full bg-well/60">
          <div
            className="h-full bg-sodium"
            style={{ width: `${Math.round(attr.confidence * 100)}%` }}
            role="img"
            aria-label={`confidence ${(attr.confidence * 100).toFixed(0)} percent`}
            title={`confidence ${(attr.confidence * 100).toFixed(0)}%`}
          />
        </div>
      )}
    </div>
  );
}

/** Legend. The glyphs are compact enough to need decoding once; without this
 * the provenance channel is decorative. */
export function ProvenanceLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ink-3">
      {Object.entries(SOURCE_GLYPH).map(([key, { glyph, title }]) => (
        <span key={key} title={title} className="flex items-center gap-1">
          <span className="text-sodium">{glyph}</span>
          {key}
        </span>
      ))}
      <span className="flex items-center gap-1">
        <span className="hatch inline-block h-3 w-5 rounded-[2px] border border-hairline/60" />
        uncertain
      </span>
    </div>
  );
}
