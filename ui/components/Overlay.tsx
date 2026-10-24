// Kutular, sınıf, kalibre skor ve iz numarası (FR-U2). Kutular kare koordinatında (motor yanıtı);
// SVG viewBox kare boyutu olduğu için görüntüyle birlikte ölçeklenir.
import type { Detection } from "@/lib/client";
import { boxStyle } from "@/lib/motion";

export function classColor(cls: string): string {
  return `var(--cls-${cls}, var(--text))`;
}

export function label(d: Detection): string {
  const id = d.track_id === null ? "" : ` #${d.track_id}`;
  return `${d.cls} ${d.score.toFixed(2)}${id}`;
}

export function Overlay({
  detections,
  width,
  height,
}: {
  detections: Detection[];
  width: number;
  height: number;
}) {
  const font = Math.max(10, Math.round(height / 36));
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden
    >
      {detections.map((d, i) => {
        const { x1, y1, x2, y2 } = d.box;
        const color = classColor(d.cls);
        const style = boxStyle(d.motion);
        const text = style.tag ? `${style.tag} ${label(d)}` : label(d);
        const ty = y1 - 3 < font ? y2 + font : y1 - 3;
        return (
          <g key={d.track_id ?? `i${i}`}>
            <rect
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill="none"
              stroke={color}
              strokeWidth={style.width}
              strokeDasharray={style.dash ?? undefined}
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={x1}
              y={ty}
              fontSize={font}
              fill={color}
              stroke="#000"
              strokeWidth={3}
              paintOrder="stroke"
              className="num"
            >
              {text}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
