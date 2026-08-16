interface Props {
  tone: "done" | "pending" | "flag" | "neutral";
  children: React.ReactNode;
}

const TONE_CLASSES: Record<Props["tone"], string> = {
  done: "text-done bg-done/10 border-done/30",
  pending: "text-pending bg-pending/10 border-pending/30",
  flag: "text-flag bg-flag/10 border-flag/30",
  neutral: "text-ink-soft bg-white/5 border-border",
};

export function StatusPill({ tone, children }: Props) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-mono font-medium uppercase tracking-wide ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
