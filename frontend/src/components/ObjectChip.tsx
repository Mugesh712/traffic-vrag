interface Props {
  globalId: string;
  unsupported?: boolean;
  onClick?: () => void;
}

/* Unsupported citations -- names the model produced that validate_citations()
 * rejected -- render struck through and flagged, never dropped. Same
 * principle that keeps them in the API response: a fabricated reference
 * should be visible, not laundered into a clean-looking answer. This is the
 * only place in the interface allowed to use the flag hue besides an outright
 * failure. */
export function ObjectChip({ globalId, unsupported, onClick }: Props) {
  if (unsupported) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-sm border border-flag/50 bg-flag-dim px-1.5 py-0.5 font-mono text-[12px] text-flag line-through decoration-flag/70"
        title="Cited by the model but absent from the retrieved context"
      >
        {globalId}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center rounded-sm border border-hairline bg-console-2 px-1.5 py-0.5 font-mono text-[12px] text-ink-data transition-colors hover:border-annotate hover:text-annotate"
    >
      {globalId}
    </button>
  );
}
