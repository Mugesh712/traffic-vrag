import { STAGES, type JobStatus, type Stage } from "../lib/types";

/* A chain of custody, not a progress bar.
 *
 * All eleven stages are listed from the first frame, numbered, so an analyst
 * can see the whole procedure and which step produced what. A single 0-100%
 * bar compresses that into one number and hides the thing actually worth
 * knowing when a run is slow or wrong: WHICH stage it is in. The percentage
 * is kept, but demoted to a caption.
 */

const STAGE_LABELS: Record<Stage, string> = {
  ingest: "Ingest",
  detect: "Detect",
  track: "Track",
  associate: "Associate",
  attribute: "Attribute",
  vote: "Vote",
  link: "Link",
  confirm: "Confirm",
  events: "Events",
  build_kg: "Graph",
  index: "Index",
};

// What each stage contributes, shown while it runs so a long wait is legible
// rather than mysterious -- M5/M8 run a VLM per crop and dominate the clock.
const STAGE_NOTE: Record<Stage, string> = {
  ingest: "video → clips → sampled frames",
  detect: "YOLO, traffic classes only",
  track: "ByteTrack + ReID per clip",
  associate: "repairs fragmented tracks",
  attribute: "VLM reads each crop — slowest stage",
  vote: "confidence-weighted canonical values",
  link: "identity across clips",
  confirm: "high-detail re-read of best shots",
  events: "stop, park, turn, overtake…",
  build_kg: "loads Neo4j",
  index: "embeds timelines for retrieval",
};

export function JobProgress({ status }: { status: JobStatus }) {
  const failedStage = status.status === "failed" ? status.current_stage : null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-[15px] font-600 uppercase tracking-[0.12em] text-ink-2">
          Chain of custody
        </h2>
        <span className="font-mono text-[11px] tabular text-ink-3">
          {status.stages_completed.length}/{status.total_stages}
        </span>
      </div>

      <ol className="flex flex-col">
        {STAGES.map((stage, i) => {
          const done = status.stages_completed.includes(stage);
          const active = status.current_stage === stage && status.status === "running";
          const failed = failedStage === stage;

          return (
            <li
              key={stage}
              className={`flex items-start gap-2.5 border-l-2 py-1.5 pl-2.5 ${
                failed
                  ? "border-flag"
                  : active
                    ? "border-annotate"
                    : done
                      ? "border-sodium/50"
                      : "border-hairline"
              }`}
            >
              <span className="mt-px w-5 flex-none font-mono text-[11px] tabular text-ink-3">
                {String(i + 1).padStart(2, "0")}
              </span>

              <span
                className={`mt-0.5 w-3 flex-none text-center text-[11px] ${
                  failed ? "text-flag" : active ? "text-annotate" : done ? "text-sodium" : "text-ink-3"
                }`}
                aria-hidden
              >
                {failed ? "×" : done ? "✓" : active ? "▶" : "·"}
              </span>

              <span className="min-w-0 flex-1">
                <span
                  className={`block font-display text-[15px] uppercase tracking-wide ${
                    done || active ? "text-ink" : "text-ink-3"
                  }`}
                >
                  {STAGE_LABELS[stage]}
                  <span className="sr-only">
                    {failed ? " failed" : done ? " complete" : active ? " running" : " pending"}
                  </span>
                </span>

                {active && (
                  <span className="block font-mono text-[11px] leading-snug text-annotate">
                    {status.stage_detail ?? STAGE_NOTE[stage]}
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ol>

      {status.status === "failed" && status.error && (
        <p className="rounded-sm border border-flag/40 bg-flag-dim px-2.5 py-2 font-mono text-[11px] leading-relaxed text-flag">
          {status.error}
        </p>
      )}
    </div>
  );
}
