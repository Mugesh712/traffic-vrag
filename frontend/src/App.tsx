import { useState } from "react";
import { UploadPanel } from "./components/UploadPanel";
import { JobProgress } from "./components/JobProgress";
import { ChatPanel } from "./components/ChatPanel";
import { ObjectExplorer } from "./components/ObjectExplorer";
import { useJobStatus } from "./lib/useJobStatus";

type Tab = "ask" | "objects";

function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("ask");
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const { status, error: pollError } = useJobStatus(jobId);

  // Gates match the API's own gates exactly (see main.py's /query and
  // /jobs/{id}/objects), so the UI never offers an action the backend would
  // reject -- "confirm" unlocks objects, "index" unlocks querying.
  const objectsReady = !!status?.stages_completed.includes("confirm");
  const queryReady = !!status?.stages_completed.includes("index");

  return (
    <div className="min-h-screen bg-bg">
      <header className="flex items-center justify-between border-b border-border px-6 py-3.5">
        <div className="flex items-baseline gap-2.5">
          <span className="font-mono text-xs uppercase tracking-wider text-accent">Traffic-VRAG</span>
          <span className="text-sm text-ink-faint">Semantically-Corrected Video Reasoning</span>
        </div>
        {jobId && (
          <button
            onClick={() => {
              setJobId(null);
              setSelectedObject(null);
            }}
            className="font-mono text-xs text-ink-faint transition-colors hover:text-ink"
          >
            New upload
          </button>
        )}
      </header>

      {!jobId ? (
        <UploadPanel onUploaded={setJobId} />
      ) : (
        <div className="mx-auto flex max-w-7xl gap-6 px-6 py-6">
          <aside className="w-64 flex-none">
            <div className="sticky top-6 flex flex-col gap-4">
              <div>
                <span className="font-mono text-xs text-ink-faint">job {jobId.slice(0, 8)}</span>
              </div>
              {status ? (
                <JobProgress status={status} />
              ) : (
                <p className="font-mono text-xs text-ink-faint">Connecting…</p>
              )}
              {pollError && <p className="text-xs text-flag">{pollError}</p>}
            </div>
          </aside>

          <main className="min-w-0 flex-1">
            <nav className="mb-4 flex gap-1 border-b border-border">
              <TabButton active={tab === "ask"} onClick={() => setTab("ask")}>
                Ask
              </TabButton>
              <TabButton active={tab === "objects"} onClick={() => setTab("objects")}>
                Objects
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
      className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
        active ? "border-accent text-ink" : "border-transparent text-ink-faint hover:text-ink-soft"
      }`}
    >
      {children}
    </button>
  );
}

export default App;
