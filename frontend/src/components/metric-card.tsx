export function MetricCard({
  label,
  value,
  subtitle,
}: {
  label: string;
  value: string;
  subtitle?: string;
}) {
  return (
    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
      <p className="text-xs text-[var(--muted)] uppercase tracking-wide mb-1">
        {label}
      </p>
      <p className="text-2xl font-semibold tabular-nums">{value}</p>
      {subtitle && (
        <p className="text-xs text-[var(--muted)] mt-1">{subtitle}</p>
      )}
    </div>
  );
}
