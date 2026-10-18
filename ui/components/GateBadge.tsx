// Kapı rozeti (Faz 3 tasarımı §4.4): alarmlardan ayrı, her karenin gate / applied alanından.
import { gateBadge } from "@/lib/alarms";
import type { Health } from "@/lib/client";

export function GateBadge({ health }: { health: Health | null }) {
  const b = health ? gateBadge(health) : null;
  if (!b) return null;
  const color = b.level === "applied" ? "var(--ir)" : "var(--muted)";
  return (
    <span
      className="rounded border px-2 py-0.5 text-xs font-medium"
      style={{ borderColor: color, color }}
      title={health?.reason || undefined}
    >
      Kapı: {b.text}
    </span>
  );
}
