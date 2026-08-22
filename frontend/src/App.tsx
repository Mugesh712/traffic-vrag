import { useState } from "react";
import { UploadPanel } from "./components/UploadPanel";
import { JobProgress } from "./components/JobProgress";
import { ChatPanel } from "./components/ChatPanel";
import { ObjectExplorer } from "./components/ObjectExplorer";
import { useJobStatus } from "./lib/useJobStatus";

type Tab = "ask" | "objects";

/** The job lives in the URL so a finding can be sent to someone else. A
 * result an analyst cannot link to is a result they have to describe from
 * memory. */
function readJobFromUrl(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("job");
}

function App() {
  const [jobId, setJobIdState] = useState<string | null>(readJobFromUrl);
  const [tab, setTab] = useState<Tab>("ask");
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const { status, error: pollError } = useJobStatus(jobId);

  const setJobId = (next: string | null) => {
    setJobIdState(next);
    const url = new URL(window.location.href);
    if (next) url.searchParams.set("job", next);
    else url.searchParams.delete("job");
    window.history.replaceState({}, "", url);
  };

  // Gates match the API's own gates exactly (see main.py's /query and
  // /jobs/{id}/objects), so the UI never offers an action the backend would
  // reject -- "confirm" unlocks objects, "index" unlocks querying.
  const objectsReady = !!status?.stages_completed.includes("confirm");
  const queryReady = !!status?.stages_completed.includes("index");

  const running = status?.status === "running" || status?.status === "queued";

  return (
    <div className="min-h-screen bg-well">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-hairline px-4 py-2.5 sm:px-5">
        <span className="font-display text-lg font-600 uppercase tracking-[0.16em] text-ink">
          Traffic<span className="text-sodium">·</span>VRAG
        </span>
        <span className="hidden font-mono text-[11px] uppercase tracking-wider text-ink-3 sm:inline">
          video reasoning
        </span>

        {jobId && (
          <>
            <span className="ml-auto flex items-center gap-1.5 font-mono text-[11px] text-ink-3">
              <span
                aria-hidden
                className={`inline-block h-1.5 w-1.5 rounded-full ${
                  status?.status === "failed"
                    ? "bg-flag"
                    : running
                      ? "animate-pulse bg-annotate"
                      : "bg-sodium"
                }`}
              />
              {status?.status ?? "connecting"}
            </span>
            <span className="font-mono text-[11px] tabular text-ink-3">{jobId.slice(0, 8)}</span>
            <button
              onClick={() => {
                setJobId(null);
                setSelectedObject(null);
              }}
              className="font-mono text-[11px] uppercase tracking-wider text-ink-3 transition-colors hover:text-annotate"
            >
              new
            </button>
          </>
        )}
      </header>

      {!jobId ? (
        <UploadPanel onUploaded={setJobId} />
      ) : (
        <div className="mx-auto flex max-w-[1400px] flex-col gap-5 px-4 py-5 lg:flex-row sm:px-5">
          <aside className="w-full flex-none lg:w-[230px]">
            <div className="lg:sticky lg:top-5">
              {status ? (
                <JobProgress status={status} />
              ) : (
                <p className="font-mono text-xs text-ink-3">Connecting…</p>
              )}
              {pollError && (
                <p role="alert" className="mt-2 font-mono text-[11px] text-flag">
                  {pollError}
                </p>
              )}
            </div>
          </aside>

          <main className="min-w-0 flex-1">
            <nav className="mb-4 flex gap-4 border-b border-hairline" aria-label="Views">
              <TabButton active={tab === "ask"} onClick={() => setTab("ask")}>
                Interrogate
              </TabButton>
              <TabButton active={tab === "objects"} onClick={() => setTab("objects")}>
                Roster
              </TabButton>
            </nav>

            {tab === "ask" ? (
              <ChatPanel
                jobId={jobId}
                ready={queryReady}
                onSelectObject={(id) => {
                  setSelectedObject(id);
                  setTab("objects");
                }}
              />
            ) : (
              <ObjectExplorer
                jobId={jobId}
                ready={objectsReady}
                selectedId={selectedObject}
                onSelect={setSelectedObject}
              />
            )}
          </main>
        </div>
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={`-mb-px border-b-2 pb-2 font-display text-[17px] uppercase tracking-[0.1em] transition-colors ${
        active ? "border-sodium text-ink" : "border-transparent text-ink-3 hover:text-ink-2"
      }`}
    >
      {children}
    </button>
  );
}

export default App;
