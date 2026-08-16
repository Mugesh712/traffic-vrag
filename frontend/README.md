# Traffic-VRAG frontend (M15)

React demo UI for the pipeline: upload a video, watch per-stage progress,
ask questions, and browse detected objects.

## Run

```
npm install
npm run dev
```

Requires the API server running (`python -m src.cli serve` from the project
root, default `http://localhost:8000`). Override the API base URL with
`VITE_API_BASE_URL` if it's running elsewhere.

## Layout

- `src/lib/types.ts` — TypeScript mirrors of `src/api/main.py`'s response
  shapes. No schema generation step; kept in sync by hand when the API changes.
- `src/lib/api.ts` — typed fetch wrappers, one per endpoint.
- `src/lib/useJobStatus.ts` — polls job status until it reaches a terminal state.
- `src/components/` — one component per piece of the answer panel (chips,
  timeline, evidence gallery, KG subgraph) plus the upload/progress/explorer
  views. `App.tsx` composes them and gates each panel on the same
  `stages_completed` values the API itself gates on (`confirm` for objects,
  `index` for querying), so the UI never offers an action the backend would
  reject with a 409.
