export function GenreBadge({ genre }: { genre: string }) {
  return (
    <span className="inline-block px-2 py-0.5 rounded-full bg-[var(--accent)]/15 text-[var(--accent-light)] text-xs font-medium">
      {genre}
    </span>
  );
}

export function GenreBadges({ genres }: { genres: string }) {
  const list = genres.split("|").filter(Boolean);
  return (
    <div className="flex flex-wrap gap-1">
      {list.map((g) => (
        <GenreBadge key={g} genre={g} />
      ))}
    </div>
  );
}
