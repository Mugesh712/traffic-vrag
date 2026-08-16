import { StatusPill } from "./StatusPill";
import { ObjectChip } from "./ObjectChip";
import { TimelineStrip } from "./TimelineStrip";
import { EvidenceGallery } from "./EvidenceGallery";
import { KGSubgraph } from "./KGSubgraph";
import type { QueryResponse } from "../lib/types";

interface Props {
  response: QueryResponse;
  onSelectObject: (globalId: string) => void;
}

const STATUS_TONE = {
  answered: "done",
  counting: "done",
  insufficient_evidence: "pending",
} as const;

export function AnswerPanel({ response, onSelectObject }: Props) {
  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[15px] leading-relaxed text-ink">{response.answer}</p>
        <StatusPill tone={STATUS_TONE[response.status]}>{response.status.replace("_", " ")}</StatusPill>
      </div>

      {response.supporting_object_ids.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 font-mono text-xs uppercase tracking-wide text-ink-faint">Sources</span>
          {response.supporting_object_ids.map((id) => (
            <ObjectChip key={id} globalId={id} onClick={() => onSelectObject(id)} />
          ))}
        </div>
      )}

      {response.unsupported_citations.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 font-mono text-xs uppercase tracking-wide text-flag">Unverified citations</span>
          {response.unsupported_citations.map((tag) => (
            <ObjectChip key={tag} globalId={tag} unsupported />
          ))}
        </div>
      )}

      {response.timestamps.length > 0 && <TimelineStrip spans={response.timestamps} />}

      <EvidenceGallery paths={response.evidence_frames} />

      {response.kg_subgraph.nodes.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="font-mono text-xs uppercase tracking-wide text-ink-faint">Knowledge graph</span>
          <KGSubgraph data={response.kg_subgraph} />
        </div>
      )}

      {response.retrieval_warnings.length > 0 && (
        <div className="rounded-md border border-pending/30 bg-pending/10 px-3 py-2 text-xs text-pending">
          {response.retrieval_warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
        </div>
      )}

      <details className="text-xs text-ink-faint">
        <summary className="cursor-pointer font-mono uppercase tracking-wide hover:text-ink-soft">
          Reasoning trace
        </summary>
        <p className="mt-1.5 leading-relaxed">{response.reasoning_trace}</p>
      </details>
    </div>
  );
}
