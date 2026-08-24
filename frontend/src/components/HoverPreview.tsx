import { useRef, useState } from "react";

/* Generic hover-preview anchor: wraps a trigger, opens `render()` beside it
 * on hover or keyboard focus, closes on a short delay so moving the pointer
 * from the trigger INTO the popover doesn't flicker it shut.
 *
 * Deliberately hover/focus-only, not click-to-toggle: several triggers this
 * wraps (chips, roster rows) already do something on click (navigate to the
 * roster). Adding a second meaning to the same tap would make touch users
 * guess which one fires. Touch users keep the existing tap-to-navigate
 * behaviour unchanged; the preview is a bonus for a pointer device, not a
 * replacement interaction. */

export function HoverPreview({
  children,
  render,
  className = "relative inline-block",
  align = "left",
}: {
  children: React.ReactNode;
  render: () => React.ReactNode;
  className?: string;
  align?: "left" | "center";
}) {
  const [open, setOpen] = useState(false);
  const closeTimer = useRef<number | null>(null);

  const cancelClose = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };
  const scheduleClose = () => {
    cancelClose();
    closeTimer.current = window.setTimeout(() => setOpen(false), 140);
  };

  return (
    <span
      className={className}
      onMouseEnter={() => {
        cancelClose();
        setOpen(true);
      }}
      onMouseLeave={scheduleClose}
      onFocus={() => setOpen(true)}
      onBlur={scheduleClose}
    >
      {children}
      {open && (
        <span
          className={`pointer-events-none absolute top-full z-30 mt-1.5 block ${
            align === "center" ? "left-1/2 -translate-x-1/2" : "left-0"
          }`}
        >
          <span
            className="pointer-events-auto block"
            onMouseEnter={cancelClose}
            onMouseLeave={scheduleClose}
          >
            {render()}
          </span>
        </span>
      )}
    </span>
  );
}
