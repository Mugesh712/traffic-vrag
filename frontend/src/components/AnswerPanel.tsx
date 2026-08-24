import { useMemo } from "react";
import { ObjectChip } from "./ObjectChip";
import { TimelineStrip } from "./TimelineStrip";
import { EvidenceGallery } from "./EvidenceGallery";
import { KGSubgraph, KGLegend } from "./KGSubgraph";
import { HoverPreview } from "./HoverPreview";
import { ObjectPreviewCard } from "./ObjectPreviewCard";
import { frameUrl } from "../lib/api";
import type { KGNode, QueryResponse } from "../lib/types";

/* The claim on the left, its proof on the right.
 *
 * Evidence sits BESIDE the answer rather than under it, because checking a
 * claim against its frames is the actual task here -- putting the proof below
 * the fold makes reading the sentence the default and verifying it the
 * effort. On narrow screens it stacks, claim first. */

interface Props {
  response: QueryResponse;
  onSelectObject: (globalId: string) => void;
}

/** Renders inline [obj_0004 @ 18:15:17] citations as marks rather than raw
 * text. The bracket form is what makes the answer checkable; leaving it as
 * plain prose buries the one part an analyst is meant to follow.
 *
 * Each citation also carries a thumbnail of the object it names, when the
 * retrieval turned one up, and a hover preview with its full winning
 * attributes -- built from the KG subgraph's own nodes (already in memory
 * from this response), so this costs no extra request. "Which car is
 * obj_0004" used to mean scanning the evidence strip below for a matching
 * id; now it answers without leaving the sentence. */
function CitedAnswer({
  text,
  framesByObject,
  nodesById,
  onSelectObject,
}: {
  text: string;
  framesByObject: Record<string, string[]>;
  nodesById: Map<string, KGNode>;
  onSelectObject: (id: string) => void;
}) {
  const parts = text.split(/(\[[A-Za-z0-9_]+\s*@\s*\d{1,2}:\d{2}:\d{2}\])/g);
  return (
    <p className="text-[15px] leading-[1.65] text-ink">
      {parts.map((part, i) => {
        const m = part.match(/^\[([A-Za-z0-9_]+)\s*@\s*(\d{1,2}:\d{2}:\d{2})\]$/);
        if (!m) return <span key={i}>{part}</span>;
        const id = m[1];
        const thumb = framesByObject[id]?.[0];
        const node = nodesById.get(id);
        return (
          <HoverPreview
            key={i}
            render={() => (
              <ObjectPreviewCard
                globalId={id}
                label={(node?.properties.class as string) ?? undefined}
                cropUrl={thumb ?? null}
                properties={node?.properties}
              />
            )}
          >
            <button
              type="button"
              onClick={() => onSelectObject(id)}
              title={`${id} at ${m[2]} — open in roster`}
              className="interactive mx-0.5 inline-flex items-baseline gap-1 rounded-sm border-b border-dotted border-sodium/60 px-0.5 font-mono text-[13px] tabular text-ink-data hover:border-annotate hover:text-annotate"
            >
              {thumb && (
                <img
                  src={frameUrl(thumb)}
                  alt=""
                  aria-hidden
                  loading="lazy"
                  className="h-4 w-6 translate-y-[3px] rounded-[1px] border border-hairline object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
              )}
              {id}
              <span className="text-ink-3">{m[2]}</span>
            </button>
          </HoverPreview>
        );
      })}
    </p>
  );
}

const STATUS_LABEL: Record<QueryResponse["status"], string> = {
  answered: "answered",
  counting: "counted from graph",
  insufficient_evidence: "declined",
};

export function AnswerPanel({ response, onSelectObject }: Props) {
  const declined = response.status === "insufficient_evidence";

  // Built once per response, reused by every citation/chip hover preview --
  // the KG subgraph already carries each cited object's winning attributes,
  // so no extra request is needed to enrich a hover.
  const nodesById = useMemo(() => new Map(response.kg_subgraph.nodes.map((n) => [n.id, n])), [response]);

  return (
    <article className="grid gap-5 rounded-sm border border-hairline bg-console p-4 shadow-[0_1px_0_rgba(255,255,255,0.03)_inset,0_12px_28px_-16px_rgba(0,0,0,0.55)] lg:grid-cols-[1fr_minmax(280px,42%)]">
      {/* -- claim -- */}
      <div className="flex min-w-0 flex-col gap-3.5">
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider ${
              declined ? "text-ink-2" : "text-sodium"
            }`}
          >
            <span aria-hidden>{declined ? "○" : "▣"}</span>
            {STATUS_LABEL[response.status]}
          </span>
          {response.count !== null && (
            <span className="font-mono text-[11px] tabular text-ink-3">count {response.count}</span>
          )}
        </div>

        <CitedAnswer
          text={response.answer}
          framesByObject={response.evidence_frames_by_object}
          nodesById={nodesById}
          onSelectObject={onSelectObject}
        />

        {response.supporting_object_ids.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-display text-[13px] uppercase tracking-[0.12em] text-ink-3">Supported</span>
            {response.supporting_object_ids.map((id) => {
              const node = nodesById.get(id);
              const thumb = response.evidence_frames_by_object[id]?.[0];
              return (
                <HoverPreview
                  key={id}
                  render={() => (
                    <ObjectPreviewCard
                      globalId={id}
                      label={(node?.properties.class as string) ?? undefined}
                      cropUrl={thumb ?? null}
                      properties={node?.properties}
                    />
                  )}
                >
                  <ObjectChip globalId={id} onClick={() => onSelectObject(id)} />
                </HoverPreview>
              );
            })}
          </div>
        )}

        {/* Kept prominent, not tucked away: this is the system reporting its
            own hallucination. */}
        {response.unsupported_citations.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 rounded-sm border border-flag/40 bg-flag-dim px-2.5 py-2">
            <span className="font-display text-[13px] uppercase tracking-[0.12em] text-flag">
              Unverified
            </span>
            {response.unsupported_citations.map((tag) => (
              <ObjectChip key={tag} globalId={tag} unsupported />
            ))}
          </div>
        )}

        {response.timestamps.length > 0 && <TimelineStrip spans={response.timestamps} />}

        {response.retrieval_warnings.length > 0 && (
          <div className="rounded-sm border border-flag/30 bg-flag-dim px-2.5 py-2 font-mono text-[11px] leading-relaxed text-flag">
            {response.retrieval_warnings.map((w, i) => (
              <p key={i}>{w}</p>
            ))}
          </div>
        )}

        <details className="group">
          <summary className="cursor-pointer list-none font-mono text-[11px] uppercase tracking-wider text-ink-3 hover:text-ink-2">
            <span aria-hidden className="mr-1 inline-block transition-transform group-open:rotate-90">
              ▸
            </span>
            Trace
          </summary>
          <p className="mt-1.5 pl-3 font-mono text-[11px] leading-relaxed text-ink-3">
            {response.reasoning_trace}
          </p>
        </details>
      </div>

      {/* -- proof -- */}
      <div className="flex min-w-0 flex-col gap-4 lg:border-l lg:border-hairline lg:pl-5">
        {response.kg_subgraph.nodes.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="font-display text-[13px] uppercase tracking-[0.12em] text-ink-3">Subgraph</span>
            <KGSubgraph data={response.kg_subgraph} evidenceFramesByObject={response.evidence_frames_by_object} />
            <KGLegend />
          </div>
        )}
        <EvidenceGallery framesByObject={response.evidence_frames_by_object} onSelectObject={onSelectObject} />
      </div>
    </article>
  );
}
