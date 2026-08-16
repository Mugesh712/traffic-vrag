import { STAGES, type JobStatus, type Stage } from "../lib/types";

const STAGE_LABELS: Record<Stage, string> = {
  ingest: "Ingest",
  detect: "Detect",
  track: "Track",
  associate: "Associate",
  attribute: "Attribute (VLM)",
  vote: "Vote",
  link: "Link",
  confirm: "Confirm",
  events: "Events",
  build_kg: "Knowledge graph",
  index: "Vector index",
};

interface Props {
  status: JobStatus;
}

export function JobProgress({ status }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="font-mono text-xs uppercase tracking-wide text-ink-faint">Pipeline</span>
          <span className="font-mono text-xs text-ink-soft">{status.progress_percent.toFixed(0)}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-sunken">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              status.status === "failed" ? "bg-flag" : "bg-accent"
            }`}
            style={{ width: `${status.progress_percent}%` }}
          />
        </div>
      </div>

      <ol className="flex flex-col gap-1">
        {STAGES.map((stage) => {
          const done = status.stages_completed.includes(stage);
          const active = status.current_stage === stage && status.status === "running";
          const failed = status.status === "failed" && active;
          return (
            <li key={stage} className="flex items-center gap-2.5 py-0.5 text-sm">
              <StageIcon done={done} active={active} failed={failed} />
              <span className={done || active ? "text-ink" : "text-ink-faint"}>{STAGE_LABELS[stage]}</span>
              {active && status.stage_detail && (
                <span className="truncate font-mono text-xs text-ink-faint">{status.stage_detail}</span>
              )}
            </li>
          );
        })}
      </ol>

      {status.status === "failed" && (
        <div className="rounded-md border border-flag/30 bg-flag/10 px-3 py-2 text-xs text-flag">
          <p className="mb-0.5 font-mono uppercase tracking-wide">Failed at {status.current_stage}</p>
          <p className="text-ink">{status.error}</p>
        </div>
      )}
    </div>
  );
}

function StageIcon({ done, active, failed }: { done: boolean; active: boolean; failed: boolean }) {
  if (failed) {
    return <span className="flex h-4 w-4 flex-none items-center justify-center rounded-full bg-flag text-[10px] text-bg">!</span>;
  }
  if (done) {
    return (
      <span className="flex h-4 w-4 flex-none items-center justify-center rounded-full bg-done text-bg">
        <svg width="9" height="9" viewBox="0 0 12 12" fill="none">
          <path d="M2 6l3 3 5-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    );
  }
  if (active) {
    return <span className="h-4 w-4 flex-none animate-pulse rounded-full border-2 border-accent" />;
  }
  return <span className="h-4 w-4 flex-none rounded-full border-2 border-border" />;
}
