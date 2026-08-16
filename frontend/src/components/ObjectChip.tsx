interface Props {
  globalId: string;
  unsupported?: boolean;
  onClick?: () => void;
}

// Unsupported citations (a name the model cited that validate_citations
// rejected, see M13) render visibly different, never silently dropped --
// the same principle that keeps them in the API response at all.
export function ObjectChip({ globalId, unsupported, onClick }: Props) {
  if (unsupported) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-md border border-flag/40 bg-flag/10 px-2 py-1 font-mono text-xs text-flag line-through decoration-flag/60"
        title="Cited by the model but not found in the retrieved context"
      >
        {globalId}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 font-mono text-xs text-ink transition-colors hover:border-accent hover:text-accent"
    >
      {globalId}
    </button>
  );
}
