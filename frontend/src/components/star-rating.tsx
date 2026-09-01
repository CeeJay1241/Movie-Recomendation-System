export function StarRating({ rating, max = 5 }: { rating: number; max?: number }) {
  const full = Math.round(rating);
  const clamped = Math.max(0, Math.min(max, full));

  return (
    <span className="inline-flex items-center gap-1 text-sm">
      <span className="text-amber-400">
        {"★".repeat(clamped)}
        <span className="text-[var(--border)]">{"★".repeat(max - clamped)}</span>
      </span>
      <span className="text-[var(--muted)] font-mono text-xs">{rating.toFixed(2)}</span>
    </span>
  );
}

export function RatingBar({ rating, max = 5 }: { rating: number; max?: number }) {
  const pct = (rating / max) * 100;
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 bg-[var(--border)] rounded-full overflow-hidden">
        <div
          className="h-full bg-[var(--accent)] rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-[var(--muted)] w-10 text-right">
        {rating.toFixed(2)}
      </span>
    </div>
  );
}
