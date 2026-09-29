import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { ApiError, uploadVideo } from "../lib/api";
import bgUrl from "../assets/landing/bg.jpg";
import { LANDING_TEXT, type LandingText } from "./landingText";

/* The landing page is a pixel-faithful build of the requested design.
 *
 * HOW IT IS BUILT. The design reference was sliced the way a designer's mockup
 * is: `bg.jpg` is the reference itself with only its TEXT removed (inpainted),
 * so every panel, border, gradient, icon, tile, illustration and the hero
 * photograph is identical to the design. All text is then re-rendered as real
 * HTML on top (landingText.ts: fonts, sizes, tracking and baselines measured
 * from the reference), so it stays crisp, selectable and accessible. Finally,
 * transparent hit-areas sit over every control so the page actually works.
 *
 * The whole thing is one fixed artboard (the design's own 1881x1184 canvas)
 * scaled to the window width with CSS `zoom`, which re-lays text at the
 * zoomed size -- so it looks identical at any window width, and text never
 * goes soft the way a transform: scale() would make it.
 */

const W = 1881;
const H = 1184;

// Measured regions of the controls on the artboard (see the reference).
const NAV_ROWS = [
  { label: "Home", top: 124, current: true },
  { label: "Upload & Analyze", top: 198 },
  { label: "Query", top: 270 },
  { label: "Results", top: 335 },
  { label: "Object Database", top: 405 },
  { label: "History", top: 473 },
  { label: "Settings", top: 542 },
];
const DROPZONE = { left: 268, top: 436, width: 1009, height: 346, radius: 10 };
const BUTTON = { left: 612, top: 672, width: 321, height: 63, radius: 11 };
const QA_ROWS = [
  { top: 488, height: 89, label: "Upload & Analyze Video" },
  { top: 591, height: 92, label: "Ask a Question" },
  { top: 697, height: 92, label: "View Results" },
];
const PILLS = [
  { left: 265, width: 383, label: "How many white Toyota cars were seen between 10:00 and 10:15?" },
  { left: 664, width: 383, label: "Track the red truck across the entire video" },
  { left: 1063, width: 366, label: "What was the sequence of events before the accident?" },
  { left: 1444, width: 402, label: "Find all vehicles that changed lanes near the intersection" },
];

/* Scale factor that fits the artboard to the window's width. */
function useArtboardZoom() {
  const [zoom, setZoom] = useState(() =>
    typeof window === "undefined" ? 1 : document.documentElement.clientWidth / W,
  );
  useEffect(() => {
    const onResize = () => setZoom(document.documentElement.clientWidth / W);
    window.addEventListener("resize", onResize);
    onResize();
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return zoom;
}

export function UploadPanel({ onUploaded }: { onUploaded: (jobId: string) => void }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const zoom = useArtboardZoom();

  const openPicker = useCallback(() => {
    if (!uploading) inputRef.current?.click();
  }, [uploading]);

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
    <div className="min-h-screen overflow-x-hidden bg-[#030e1b]">
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) submit(file);
        }}
      />

      <div
        className="relative"
        style={{
          width: W,
          height: H,
          zoom,
          backgroundImage: `url(${bgUrl})`,
          backgroundSize: "100% 100%",
        }}
      >
        <h1 className="sr-only">Traffic-VRAG — From Traffic Videos to Actionable Insights</h1>

        {/* ---------------------------------------------------------- text */}
        {LANDING_TEXT.map((t) => (
          <Text key={t.id} t={t} uploading={uploading} />
        ))}

        {/* ------------------------------------------------------- sidebar */}
        <nav aria-label="Main">
          {NAV_ROWS.map((n) => (
            <button
              key={n.label}
              type="button"
              aria-label={n.label}
              aria-current={n.current ? "page" : undefined}
              onClick={n.current ? undefined : openPicker}
              className="hit absolute"
              style={{ left: 7, top: n.top, width: 207, height: 60, borderRadius: 9 }}
            />
          ))}
        </nav>

        {/* ------------------------------------------------------ dropzone */}
        <div
          role="button"
          tabIndex={0}
          aria-label="Drop your video here, or choose a file"
          onClick={openPicker}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              openPicker();
            }
          }}
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
          className={`hit-soft absolute cursor-pointer ${dragging ? "is-dragging" : ""}`}
          style={{
            left: DROPZONE.left,
            top: DROPZONE.top,
            width: DROPZONE.width,
            height: DROPZONE.height,
            borderRadius: DROPZONE.radius,
          }}
        />

        <button
          type="button"
          aria-label="Choose file"
          onClick={openPicker}
          disabled={uploading}
          className="hit-bright absolute"
          style={{
            left: BUTTON.left,
            top: BUTTON.top,
            width: BUTTON.width,
            height: BUTTON.height,
            borderRadius: BUTTON.radius,
          }}
        />

        {error && (
          <p
            role="alert"
            className="absolute text-center font-['Inter'] text-[15px] font-500 text-[#ff8a8f]"
            style={{ left: DROPZONE.left, width: DROPZONE.width, top: 764 }}
          >
            {error}
          </p>
        )}

        {/* ------------------------------------------------- quick actions */}
        {QA_ROWS.map((r) => (
          <button
            key={r.label}
            type="button"
            aria-label={r.label}
            onClick={openPicker}
            className="hit absolute"
            style={{ left: 1338, top: r.top, width: 511, height: r.height, borderRadius: 10 }}
          />
        ))}

        {/* ------------------------------------------------ sample queries */}
        {PILLS.map((p) => (
          <button
            key={p.label}
            type="button"
            aria-label={`${p.label} — upload a video to ask this`}
            onClick={openPicker}
            className="hit absolute"
            style={{ left: p.left, top: 1100, width: p.width, height: 61, borderRadius: 10 }}
          />
        ))}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- text */

const GRADIENT =
  "linear-gradient(90deg, #5494f0 0%, #6485ef 20%, #7f77f1 40%, #906fef 50%, #a966c8 62%, #b46bc6 72%, #af62b5 100%)";

function Text({ t, uploading }: { t: LandingText; uploading: boolean }) {
  const base: CSSProperties = {
    position: "absolute",
    left: t.left,
    top: t.top,
    fontFamily: t.fam === "mono" ? '"IBM Plex Mono", ui-monospace, monospace' : '"Inter", ui-sans-serif, system-ui, sans-serif',
    fontWeight: t.wt,
    fontSize: t.size,
    letterSpacing: t.ls,
    lineHeight: 1,
    color: t.color,
    whiteSpace: "nowrap",
    pointerEvents: "none",
  };

  if (t.id === "h2") {
    return (
      <span
        aria-hidden
        style={{ ...base, backgroundImage: GRADIENT, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}
      >
        {t.text}
      </span>
    );
  }

  if (t.id === "logo") {
    return (
      <span aria-hidden style={base}>
        TRAFFIC<span style={{ color: "#f7c6ec" }}>·</span>VRAG
      </span>
    );
  }

  // Centred on the original's centre, so "Uploading…" sits where the
  // heading was rather than hanging off its left edge.
  if (t.id === "drop") {
    const centre = 772;
    return (
      <span aria-live="polite" style={{ ...base, left: centre - 300, width: 600, textAlign: "center" }}>
        {uploading ? "Uploading…" : t.text}
      </span>
    );
  }

  return (
    <span aria-hidden={t.id.startsWith("nav_") ? true : undefined} style={base}>
      {t.text}
    </span>
  );
}
