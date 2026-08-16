import { useEffect, useRef, useState } from "react";
import { getJobStatus } from "./api";
import type { JobStatus } from "./types";

const POLL_INTERVAL_MS = 2000;

/** Polls GET /jobs/{id}/status until the job reaches a terminal state
 * (completed/failed), then stops -- no point polling a job that can no
 * longer change. */
export function useJobStatus(jobId: string | null) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!jobId) {
      setStatus(null);
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const next = await getJobStatus(jobId!);
        if (cancelled) return;
        setStatus(next);
        setError(null);
        if (next.status === "queued" || next.status === "running") {
          timer.current = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch {
        if (cancelled) return;
        setError("Lost contact with the API server.");
        timer.current = setTimeout(poll, POLL_INTERVAL_MS);
      }
    }
    poll();

    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [jobId]);

  return { status, error };
}
