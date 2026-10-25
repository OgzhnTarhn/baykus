"use client";

// Zaman çizelgesi (Faz 4 §5.8): dizinin kare ekseni üzerinde olay işaretleri (türüne göre biçim,
// renk yalnız destekleyici), alarm bantları (yasak bölge kırmızı, bekleme sınırı turuncu; açık bant
// geçerli kareye kadar uzanır) ve geçerli kare imleci. İşaretler klavyeyle odaklanabilir düğmelerdir;
// Enter ya da tıklama olayı seçer. Süzgeçler tür (çoklu), bölge ve iz numarasıyla daraltır.
import type { ReactNode } from "react";

import type { Marker, MarkerShape, TimelineFilter } from "@/lib/timeline";
import { ALARM_TEXT_ZONE, EVENT_TEXT, text } from "@/lib/texts";

const HEIGHT = 56;
const AXIS_Y = 40;
const BAND_Y = 2;
const BAND_H = 14;
const MARKER_R = 3;

const BAND_COLOR: Record<string, string> = {
  intrusion: "var(--alarm)",
  dwell_exceeded: "var(--ir)",
};

// İşaret rengi türü destekler; ayırt edici asıl sinyal şekildir (renk körlüğü / gri ekran).
const MARKER_COLOR: Record<string, string> = {
  enter: "var(--ok)",
  exit: "var(--muted)",
  exit_lost: "var(--alarm)",
  stopped: "var(--warn)",
  line_cross: "var(--rgb)",
  alarm_start: "var(--alarm)",
  alarm_ack: "var(--warn)",
  alarm_clear: "var(--ok)",
};

function colorOf(kind: string): string {
  return MARKER_COLOR[kind] ?? (kind.startsWith("map_") ? "var(--rgb)" : "var(--muted)");
}

// Tek bir işaret biçimi; `r` yarı boydur (6 px işaret için 3). Seçili işaret aynı biçimin beyaz,
// biraz büyük kopyası arkaya çizilerek çerçevelenir.
function glyph(shape: MarkerShape, x: number, y: number, r: number, color: string): ReactNode {
  switch (shape) {
    case "up":
      return <polygon points={`${x},${y - r} ${x - r},${y + r} ${x + r},${y + r}`} fill={color} />;
    case "down":
      return <polygon points={`${x},${y + r} ${x - r},${y - r} ${x + r},${y - r}`} fill={color} />;
    case "cross":
      return (
        <path
          d={`M${x - r},${y - r}L${x + r},${y + r}M${x - r},${y + r}L${x + r},${y - r}`}
          stroke={color}
          strokeWidth={1.5}
          fill="none"
        />
      );
    case "square":
      return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} fill={color} />;
    case "diamond":
      return <polygon points={`${x},${y - r} ${x + r},${y} ${x},${y + r} ${x - r},${y}`} fill={color} />;
    case "bar":
      return <rect x={x - 1} y={y - r - 1} width={2} height={2 * r + 2} fill={color} />;
    case "circle":
      return <circle cx={x} cy={y} r={r} fill={color} />;
    case "dot":
      return <circle cx={x} cy={y} r={Math.max(1, r / 2)} fill={color} />;
  }
}

export function TimelineView({
  markers,
  bands,
  frame,
  frameCount,
  firstFrame,
  selectedSeq,
  onPick,
  width,
}: {
  markers: Marker[];
  bands: { alarm_id: string; from: number; to: number | null; kind: string }[];
  frame: number;
  frameCount: number;
  firstFrame: number;
  selectedSeq: number | null;
  onPick(seq: number): void;
  width: number;
}) {
  const w = Math.max(1, width);
  const span = Math.max(1, frameCount - 1);
  const xOf = (f: number) => ((f - firstFrame) / span) * w;
  const cursorX = xOf(frame);

  return (
    <svg
      className="h-14 w-full"
      height={HEIGHT}
      width="100%"
      viewBox={`0 0 ${w} ${HEIGHT}`}
      role="group"
      aria-label="Olay zaman çizelgesi"
    >
      {bands.map((b) => {
        const x1 = xOf(b.from);
        const x2 = xOf(b.to ?? frame);
        const left = Math.min(x1, x2);
        const bw = Math.max(1, Math.abs(x2 - x1));
        const color = BAND_COLOR[b.kind] ?? "var(--muted)";
        return (
          <g key={b.alarm_id}>
            <rect
              x={left}
              y={BAND_Y}
              width={bw}
              height={BAND_H}
              fill={color}
              fillOpacity={0.22}
              stroke={color}
              strokeOpacity={0.7}
            />
            <text x={left + 3} y={BAND_Y + 10} fontSize={8} fill={color} pointerEvents="none">
              {text(ALARM_TEXT_ZONE, b.kind)}
            </text>
          </g>
        );
      })}

      <line x1={0} y1={AXIS_Y} x2={w} y2={AXIS_Y} stroke="var(--line)" />

      {markers.map((m) => {
        const selected = m.seq === selectedSeq;
        const color = colorOf(m.kind);
        const r = selected ? MARKER_R * 1.5 : MARKER_R;
        return (
          <g
            key={m.seq}
            role="button"
            tabIndex={0}
            aria-label={`${text(EVENT_TEXT, m.kind)}, kare ${m.frame}`}
            aria-pressed={selected}
            className="cursor-pointer"
            onClick={() => onPick(m.seq)}
            onKeyDown={(e) => {
              if (e.key !== "Enter" && e.key !== " ") return;
              e.preventDefault();
              onPick(m.seq);
            }}
          >
            {selected && glyph(m.shape, m.x, AXIS_Y, r + 1.5, "#fff")}
            {glyph(m.shape, m.x, AXIS_Y, r, color)}
          </g>
        );
      })}

      <line
        x1={cursorX}
        y1={0}
        x2={cursorX}
        y2={HEIGHT}
        stroke="var(--rgb)"
        strokeWidth={1.5}
        pointerEvents="none"
      />
    </svg>
  );
}

export function TimelineFilters({
  kinds,
  zones,
  value,
  onChange,
}: {
  kinds: string[];
  zones: { id: string; name: string }[];
  value: TimelineFilter;
  onChange(f: TimelineFilter): void;
}) {
  // Boş `kinds` "hepsi" demektir (lib/timeline.ts); kutular bu yüzden hepsi işaretli görünür.
  const active = value.kinds && value.kinds.length ? value.kinds : kinds;
  const toggle = (kind: string) => {
    const next = active.includes(kind) ? active.filter((k) => k !== kind) : [...active, kind];
    onChange({ ...value, kinds: next.length === 0 || next.length === kinds.length ? undefined : next });
  };

  return (
    <div className="flex flex-col gap-2 text-sm">
      <fieldset className="flex flex-wrap gap-x-4 gap-y-1">
        <legend className="mb-1 text-xs text-muted">Olay türü</legend>
        {kinds.map((kind) => (
          <label key={kind} className="flex items-center gap-2">
            <input type="checkbox" checked={active.includes(kind)} onChange={() => toggle(kind)} />
            {text(EVENT_TEXT, kind)}
          </label>
        ))}
      </fieldset>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className="flex items-center gap-2">
          <span className="text-xs text-muted">Bölge</span>
          <select
            className="rounded border border-line bg-panel-2 px-2 py-1 text-sm"
            value={value.zone ?? ""}
            onChange={(e) => onChange({ ...value, zone: e.target.value || undefined })}
          >
            <option value="">Bütün bölgeler</option>
            {zones.map((z) => (
              <option key={z.id} value={z.id}>
                {z.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-xs text-muted">İz no</span>
          <input
            type="number"
            min={0}
            className="num w-20 rounded border border-line bg-panel-2 px-2 py-1 text-sm"
            value={value.track ?? ""}
            placeholder="hepsi"
            onChange={(e) => {
              const s = e.target.value;
              const n = Number(s);
              onChange({ ...value, track: s === "" || !Number.isFinite(n) ? undefined : n });
            }}
          />
        </label>
      </div>
    </div>
  );
}
