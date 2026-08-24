import { STAGES, type JobStatus, type Stage } from "../lib/types";

/* A chain of custody, not a progress bar.
 *
 * All eleven stages are listed from the first frame, numbered, so an analyst
 * can see the whole procedure and which step produced what. A single 0-100%
 * bar compresses that into one number and hides the thing actually worth
 * knowing when a run is slow or wrong: WHICH stage it is in. The percentage
 * is kept, but demoted to a caption.
 *
 * Each stage also gets its own icon -- eleven different kinds of work
 * (detecting, re-identifying, voting, re-reading...) drawn as eleven
 * different marks, in the same thin-stroke geometric language the KG
 * subgraph's shapes use, rather than one generic spinner standing in for
 * all of it. The GLOW is reserved for the stage actually running right now
 * -- the one place in this whole app where "live" is literally true. */

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

          const colorClass = failed ? "text-flag" : active ? "text-annotate" : done ? "text-sodium" : "text-ink-3";

          return (
            <li
              key={stage}
              className={`flex items-start gap-2.5 border-l-2 py-1.5 pl-2.5 transition-colors ${
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

              <span className={`relative mt-0.5 w-4 flex-none ${colorClass} ${active ? "glow-breathe" : ""}`}>
                <StageIcon stage={stage} />
                {(done || failed) && (
                  <span
                    aria-hidden
                    className={`absolute -right-1.5 -top-1.5 flex h-3 w-3 items-center justify-center rounded-full border border-well bg-well text-[8px] leading-none ${
                      failed ? "text-flag" : "text-sodium"
                    }`}
                  >
                    {failed ? "×" : "✓"}
                  </span>
                )}
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

/** One small geometric mark per stage, sharing a 15x15 viewBox and a single
 * thin currentColor stroke so the set reads as one system. Filled shapes
 * (the arrow-head, the bolt) use currentColor too, staying monochrome with
 * whatever status colour the row is already in -- no new hues introduced
 * just for iconography. */
function StageIcon({ stage }: { stage: Stage }) {
  const common = {
    width: 15,
    height: 15,
    viewBox: "0 0 15 15",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.4,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  switch (stage) {
    case "ingest":
      return (
        <svg {...common}>
          <path d="M7.5 1.5v7M4.7 5.8l2.8 2.7 2.8-2.7" />
          <path d="M2 12.5h11" />
        </svg>
      );
    case "detect":
      return (
        <svg {...common}>
          <circle cx="7.5" cy="7.5" r="3.6" />
          <path d="M7.5 1v2M7.5 12v2M1 7.5h2M12 7.5h2" />
        </svg>
      );
    case "track":
      return (
        <svg {...common}>
          <circle cx="12" cy="3.5" r="1.3" fill="currentColor" stroke="none" />
          <path d="M10 5 7.3 7.5M5.3 9.5 2.5 12" strokeDasharray="2.2 2.2" />
        </svg>
      );
    case "associate":
      return (
        <svg {...common}>
          <path d="M2.5 3 7.5 7.5M12.5 3 7.5 7.5M7.5 7.5v5.5" />
        </svg>
      );
    case "attribute":
      return (
        <svg {...common}>
          <path d="M1.3 7.5S4 3.3 7.5 3.3 13.7 7.5 13.7 7.5 11 11.7 7.5 11.7 1.3 7.5 1.3 7.5Z" />
          <circle cx="7.5" cy="7.5" r="1.5" fill="currentColor" stroke="none" />
        </svg>
      );
    case "vote":
      return (
        <svg {...common}>
          <path d="M2.5 4.2h4.2M2.5 7.5h7.8M2.5 10.8h5.8" />
        </svg>
      );
    case "link":
      return (
        <svg {...common}>
          <circle cx="5.6" cy="7.5" r="3" />
          <circle cx="9.4" cy="7.5" r="3" />
        </svg>
      );
    case "confirm":
      return (
        <svg {...common}>
          <circle cx="6.3" cy="6.3" r="4" />
          <path d="M9.2 9.2 13 13" />
        </svg>
      );
    case "events":
      return (
        <svg {...common}>
          <path d="M8.6 1.3 3.6 8.5h3.3L6 13.5l5.4-7.4H8.1z" fill="currentColor" stroke="none" />
        </svg>
      );
    case "build_kg":
      return (
        <svg {...common}>
          <path d="M5 10 7.2 5M9 5 11 9.2" strokeWidth="1" />
          <rect x="1.8" y="9.6" width="3.4" height="3.4" rx="0.6" fill="currentColor" stroke="none" />
          <rect x="6" y="1.8" width="3" height="3" transform="rotate(45 7.5 3.3)" fill="currentColor" stroke="none" />
          <circle cx="11.7" cy="10.3" r="1.8" fill="currentColor" stroke="none" />
        </svg>
      );
    case "index":
      return (
        <svg {...common}>
          <path d="M7.5 1.8 13 4.6 7.5 7.4 2 4.6z" />
          <path d="M2.3 8 7.5 10.6 12.7 8" />
          <path d="M2.3 11.1 7.5 13.7 12.7 11.1" />
        </svg>
      );
  }
}
