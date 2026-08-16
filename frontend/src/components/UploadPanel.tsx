import { useCallback, useRef, useState } from "react";
import { ApiError, uploadVideo } from "../lib/api";

interface Props {
  onUploaded: (jobId: string) => void;
}

export function UploadPanel({ onUploaded }: Props) {
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
    <div className="mx-auto flex max-w-xl flex-col items-center gap-4 py-24">
      <div
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
        className={`flex w-full cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed px-10 py-16 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-border bg-surface hover:border-ink-faint"
        }`}
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" className="text-ink-faint">
          <path
            d="M12 16V4m0 0L7 9m5-5 5 5M5 20h14"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="text-sm font-medium text-ink">
          {uploading ? "Uploading…" : "Drop a traffic video here, or click to browse"}
        </p>
        <p className="text-xs text-ink-faint">MP4, MOV, AVI — the full pipeline runs automatically after upload</p>
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
      </div>
      {error && (
        <p className="w-full rounded-md border border-flag/30 bg-flag/10 px-3 py-2 text-sm text-flag">{error}</p>
      )}
    </div>
  );
}
