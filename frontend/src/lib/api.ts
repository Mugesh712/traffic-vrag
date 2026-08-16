import type { JobStatus, ObjectDetail, ObjectSummary, QueryResponse } from "./types";

// Matches src/utils/config.py's ApiConfig default (port 8000) and the CORS
// origins it allows (5173, this dev server's default port).
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail ?? response.statusText);
  }
  return response.json();
}

export function frameUrl(path: string): string {
  return `${API_BASE}/frames/${path}`;
}

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  return handle(response);
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/status`);
  return handle(response);
}

export async function askQuestion(jobId: string, question: string): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, question }),
  });
  return handle(response);
}

export async function listObjects(jobId: string): Promise<ObjectSummary[]> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/objects`);
  return handle(response);
}

export async function getObject(jobId: string, globalId: string): Promise<ObjectDetail> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/objects/${globalId}`);
  return handle(response);
}
