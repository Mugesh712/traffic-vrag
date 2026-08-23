import { useEffect, useState } from "react";
import { ApiError, askQuestion } from "./api";
import type { ChatTurn } from "./types";

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

/** Owned at the App level, not inside ChatPanel: a citation click switches to
 * the Roster tab by design, and if the conversation lived in ChatPanel's own
 * state, that switch would unmount it and wipe the very answer the analyst
 * just clicked into -- turning the app's own citation-to-roster flow into a
 * way to lose your place. */
export function useChatTurns(jobId: string | null) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);

  useEffect(() => {
    setTurns([]);
  }, [jobId]);

  async function submit(question: string) {
    if (!question.trim() || !jobId) return;
    const id = turnId();
    setTurns((prev) => [...prev, { id, question, response: null, error: null, pending: true }]);
    try {
      const response = await askQuestion(jobId, question);
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, response, pending: false } : t)));
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Query failed. Is the API server running?";
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: message, pending: false } : t)));
    }
  }

  return { turns, submit };
}
