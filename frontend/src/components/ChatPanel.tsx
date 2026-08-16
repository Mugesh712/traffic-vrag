import { useState } from "react";
import { AnswerPanel } from "./AnswerPanel";
import { ApiError, askQuestion } from "../lib/api";
import type { ChatTurn } from "../lib/types";

interface Props {
  jobId: string;
  ready: boolean;
  onSelectObject: (globalId: string) => void;
}

const EXAMPLES = [
  "How many cars are in the video?",
  "Which vehicles overtook another vehicle?",
  "What happened before 12:05?",
];

export function ChatPanel({ jobId, ready, onSelectObject }: Props) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");

  async function submit(question: string) {
    if (!question.trim() || !ready) return;
    const id = crypto.randomUUID();
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
        <div className="rounded-lg border border-dashed border-border p-4">
          <p className="mb-2 text-sm text-ink-soft">
            {ready ? "Ask a question about the video:" : "Indexing isn't finished yet — you can type ahead."}
          </p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((q) => (
              <button
                key={q}
                onClick={() => submit(q)}
                disabled={!ready}
                className="rounded-md border border-border bg-surface px-2.5 py-1 text-xs text-ink-soft transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
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
            <p className="self-end rounded-lg rounded-br-sm bg-accent-soft px-3 py-1.5 text-sm text-ink">
              {turn.question}
            </p>
            {turn.pending && (
              <p className="font-mono text-xs text-ink-faint">Retrieving and reasoning…</p>
            )}
            {turn.error && (
              <p className="rounded-md border border-flag/30 bg-flag/10 px-3 py-2 text-sm text-flag">{turn.error}</p>
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
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={ready ? "Ask about the video…" : "Waiting for indexing to finish…"}
          disabled={!ready}
          className="flex-1 rounded-md border border-border bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!ready || !input.trim()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-bg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Ask
        </button>
      </form>
    </div>
  );
}
