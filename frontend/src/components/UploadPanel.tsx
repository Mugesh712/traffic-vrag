import { useCallback, useRef, useState } from "react";
import { ApiError, uploadVideo } from "../lib/api";
import { STAGES } from "../lib/types";

/* Intake. The drop target sits beside the procedure it will trigger, so the
 * eleven stages are visible before anything is uploaded -- an analyst can see
 * what the tool is going to do to their footage rather than discovering it a
 * stage at a time. */

export function UploadPanel({ onUploaded }: { onUploaded: (jobId: string) => void }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("video/")) {
        setError(`"${file.name}" doesn't look like a video file.`);
        return;
      }
      setError(null);
      setUploading(true);
      try {
        const { job_id } = await uploadVideo(file);
        onUploaded(job_id);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Upload failed. Is the API server running?");
      } finally {
        setUploading(false);
      }
    },
    [onUploaded],
  );

  return (
    <div className="mx-auto grid max-w-5xl gap-8 px-5 py-12 md:grid-cols-[1.35fr_1fr] md:py-20">
      <div className="flex flex-col gap-3">
        <button
          type="button"
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) submit(file);
          }}
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className={`flex w-full cursor-pointer flex-col items-center gap-3 rounded-sm border-2 border-dashed px-8 py-16 text-center transition-colors ${
            dragging
              ? "border-annotate bg-annotate-dim"
              : "border-hairline bg-console hover:border-hairline-lit"
          }`}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden className="text-ink-3">
            <path
              d="M12 16V4m0 0L7 9m5-5 5 5M4 20h16"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="font-display text-xl uppercase tracking-[0.08em] text-ink">
            {uploading ? "Uploading…" : "Drop footage"}
          </span>
          <span className="font-mono text-[11px] text-ink-3">MP4 · MOV · AVI</span>
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            disabled={uploading}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) submit(file);
            }}
          />
        </button>

        {error && (
          <p
            role="alert"
            className="rounded-sm border border-flag/40 bg-flag-dim px-3 py-2 font-mono text-xs text-flag"
          >
            {error}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-2.5">
        <h2 className="font-display text-[15px] font-600 uppercase tracking-[0.12em] text-ink-2">
          On upload
        </h2>
        <ol className="grid grid-cols-2 gap-x-4 gap-y-1 md:grid-cols-1">
          {STAGES.map((s, i) => (
            <li key={s} className="flex items-baseline gap-2 font-mono text-[11px] text-ink-3">
              <span className="tabular">{String(i + 1).padStart(2, "0")}</span>
              <span className="uppercase tracking-wide">{s.replace("_", " ")}</span>
            </li>
          ))}
        </ol>
        <p className="mt-1 max-w-[34ch] text-xs leading-relaxed text-ink-3">
          Attribute and confirm run a vision-language model over every crop. On a
          machine without a GPU they dominate the clock.
        </p>
      </div>
    </div>
  );
}
