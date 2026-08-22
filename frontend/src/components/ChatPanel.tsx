import { useState } from "react";
import { AnswerPanel } from "./AnswerPanel";
import { ApiError, askQuestion } from "../lib/api";
import type { ChatTurn } from "../lib/types";

// crypto.randomUUID() exists only in secure contexts (HTTPS or localhost), so
// it is undefined when the demo is served over plain HTTP from a remote host --
// and calling it there throws before the question is ever sent. These ids are
// only React list keys, so a non-cryptographic fallback is fine.
function turnId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `turn-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

interface Props {
  jobId: string;
  ready: boolean;
  onSelectObject: (globalId: string) => void;
}

/* Examples chosen to show the two paths that exist -- one answered from the
 * graph without the LLM, one that needs retrieval and reasoning -- and one
 * the system should decline, because declining is a result this project
 * cares about demonstrating. */
const EXAMPLES = [
  "How many trucks are in the video?",
  "Which vehicles overtook another vehicle?",
  "What is the licence plate of the red car?",
];

export function ChatPanel({ jobId, ready, onSelectObject }: Props) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");

  async function submit(question: string) {
    if (!question.trim() || !ready) return;
    const id = turnId();
    setTurns((prev) => [...prev, { id, question, response: null, error: null, pending: true }]);
    setInput("");
    try {
      const response = await askQuestion(jobId, question);
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, response, pending: false } : t)));
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Query failed. Is the API server running?";
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: message, pending: false } : t)));
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {turns.length === 0 && (
        <div className="rounded-sm border border-dashed border-hairline p-4">
          <p className="mb-2.5 font-mono text-[11px] uppercase tracking-wider text-ink-3">
            {ready ? "Interrogate the footage" : "Indexing — you can type ahead"}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLES.map((q) => (
              <button
                key={q}
                onClick={() => submit(q)}
                disabled={!ready}
                className="rounded-sm border border-hairline bg-console px-2 py-1 text-left text-xs text-ink-2 transition-colors hover:border-annotate hover:text-annotate disabled:cursor-not-allowed disabled:opacity-40"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-4">
        {turns.map((turn) => (
          <div key={turn.id} className="flex flex-col gap-2">
            <p className="flex items-baseline gap-2 text-[15px] text-ink">
              <span aria-hidden className="font-mono text-[11px] text-sodium">
                Q
              </span>
              {turn.question}
            </p>
            {turn.pending && (
              <p className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-ink-3">
                <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-annotate" />
                retrieving and reasoning…
              </p>
            )}
            {turn.error && (
              <p
                role="alert"
                className="rounded-sm border border-flag/40 bg-flag-dim px-2.5 py-2 font-mono text-xs text-flag"
              >
                {turn.error}
              </p>
            )}
            {turn.response && <AnswerPanel response={turn.response} onSelectObject={onSelectObject} />}
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(input);
        }}
        className="sticky bottom-0 flex gap-2 bg-well py-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={ready ? "Ask about the footage…" : "Waiting for indexing…"}
          disabled={!ready}
          aria-label="Question"
          className="min-w-0 flex-1 rounded-sm border border-hairline bg-console px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-annotate focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!ready || !input.trim()}
          className="rounded-sm bg-annotate px-4 py-2 font-display text-[15px] uppercase tracking-[0.1em] text-well transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
        >
          Ask
        </button>
      </form>
    </div>
  );
}
