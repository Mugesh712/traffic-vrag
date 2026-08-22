// Mirrors src/api/main.py's response shapes and src/api/job_store.py's
// STAGES exactly. If the backend's shape changes, this is the file to update
// alongside it -- there is no schema generation step, so the two are kept in
// sync by hand.

export const STAGES = [
  "ingest", "detect", "track", "associate", "attribute", "vote",
  "link", "confirm", "events", "build_kg", "index",
] as const;

export type Stage = (typeof STAGES)[number];

export interface JobStatus {
  job_id: string;
  video_id: string | null;
  status: "queued" | "running" | "completed" | "failed";
  current_stage: Stage | null;
  stage_detail: string | null;
  stages_completed: Stage[];
  total_stages: number;
  progress_percent: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface TimestampSpan {
  start: string;
  end: string;
}

export interface KGNode {
  id: string;
  label: string;
  properties: Record<string, unknown>;
}

export interface KGEdge {
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
}

export interface KGSubgraph {
  nodes: KGNode[];
  edges: KGEdge[];
}

export interface QueryResponse {
  answer: string;
  status: "answered" | "insufficient_evidence" | "counting";
  supporting_object_ids: string[];
  timestamps: TimestampSpan[];
  evidence_frames_by_object: Record<string, string[]>;
  kg_subgraph: KGSubgraph;
  reasoning_trace: string;
  unsupported_citations: string[];
  retrieval_warnings: string[];
  count: number | null;
}

export interface ObjectAttribute {
  attribute: string;
  value: string | null;
  confidence: number;
  source: string;
  uncertain: boolean;
}

export interface ObjectSummary {
  global_id: string;
  class: string;
  attributes: ObjectAttribute[];
}

export interface ObjectSighting {
  clip_id: string;
  track_id: string;
  first_seen: string;
  last_seen: string;
  gate_scores: Record<string, number>;
}

export interface ObjectDetail extends ObjectSummary {
  timeline: ObjectSighting[];
  best_shot_crops: string[];
}

export interface ChatTurn {
  id: string;
  question: string;
  response: QueryResponse | null;
  error: string | null;
  pending: boolean;
}
