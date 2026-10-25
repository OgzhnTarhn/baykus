"use client";

// Bölge editörü (Faz 4 tasarımı §5.8): araç düğmeleri ve harita üzerindeki düzenleme katmanı.
// Çizim ve köşe düzenleme yalnız taslakta (ZoneDraft) yapılır; motora gönderim çağıranın işidir
// (onChange → PUT). Bu dosya motora doğrudan istek atmaz.
import {
  useEffect,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

import type { Tool, ZoneDraft } from "@/lib/editor";
import { ZONE_KIND_TEXT } from "@/lib/texts";

const TOOLS: Tool[] = ["area", "restricted", "dwell", "line"];

const POLYGON_HELP = "Tıklayarak nokta ekle; çift tıklama ya da Enter bitirir, Esc iptal.";
const LINE_HELP = "İki nokta: A tarafı soldadır.";

const VERTEX_R = 5;
const MIDPOINT_R = 3;

// Araç düğmeleri: etkin araca tekrar tıklama aracı kapatır (null).
export function ZoneToolbar({
  tool,
  onTool,
  disabled = false,
}: {
  tool: Tool | null;
  onTool(t: Tool | null): void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-1.5">
        {TOOLS.map((t) => (
          <button
            key={t}
            type="button"
            aria-pressed={tool === t}
            disabled={disabled}
            className={`rounded border px-2 py-1 text-sm disabled:opacity-40 ${
              tool === t ? "border-rgb text-rgb" : "border-line text-muted"
            }`}
            onClick={() => onTool(tool === t ? null : t)}
          >
            {ZONE_KIND_TEXT[t]}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted">{tool === "line" ? LINE_HELP : POLYGON_HELP}</p>
    </div>
  );
}

// İşaretçi olayını SVG kullanıcı birimine çevirir (katman MapPanel'in kaplama svg'sinin içindedir).
function localPoint(e: ReactPointerEvent<SVGElement>): [number, number] | null {
  const svg = e.currentTarget.ownerSVGElement;
  if (svg === null) return null;
  const ctm = svg.getScreenCTM();
  if (ctm === null) return null;
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
  return [p.x, p.y];
}

// Harita kaplamasının svg çocukları: açık çizim, seçili bölgenin köşe ve orta nokta tutamaçları.
export function ZoneEditorLayer({
  draft,
  version,
  selectedZone,
  api,
  onChange,
}: {
  draft: ZoneDraft;
  version: number;
  selectedZone: string | null;
  api: {
    toScreen(p: [number, number]): [number, number];
    toMap(p: [number, number]): [number, number];
    scale: number;
  };
  onChange(): void;
}) {
  void version; // yalnız yeniden çizim tetiği
  const [drag, setDrag] = useState<{ index: number; p: [number, number] } | null>(null);
  // Esc motora gitmez (küme değişmedi); açık çizim ekrandan düşsün diye yerel sayaç artar.
  const [epoch, setEpoch] = useState(0);

  const drawing = draft.drawing();
  const open = drawing.tool !== null;

  const found = selectedZone === null ? undefined : draft.defs().find((z) => z.id === selectedZone);
  const zone = found !== undefined && !draft.isSuspended(found.id) ? found : undefined;

  // Enter bitirir (bölge döndüyse gönderilir), Esc iptal eder; dinleyici yalnız çizim açıkken kurulur.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Enter") {
        if (draft.finish() !== null) onChange();
        return;
      }
      if (e.key === "Escape") {
        draft.cancel();
        setEpoch((n) => n + 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, draft, onChange]);

  // Sürükleme yalnız sol tuşla başlar; sağ tık silmeye ayrılmıştır.
  const onVertexDown = (e: ReactPointerEvent<SVGCircleElement>, index: number): void => {
    e.stopPropagation();
    if (e.button !== 0 || zone === undefined) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag({ index, p: zone.points[index] });
  };

  const onVertexMove = (e: ReactPointerEvent<SVGCircleElement>, index: number): void => {
    if (drag === null || drag.index !== index || zone === undefined) return;
    const local = localPoint(e);
    if (local === null) return;
    const p = api.toMap(local);
    draft.moveVertex(zone.id, index, p);
    setDrag({ index, p });
  };

  // Sürükleme bitti: değişiklik motora gönderilir (onChange → PUT).
  const onVertexUp = (e: ReactPointerEvent<SVGCircleElement>): void => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (drag === null) return;
    setDrag(null);
    onChange();
  };

  const onVertexCancel = (): void => {
    setDrag(null);
  };

  // Köşe silme: çift tıklama ya da sağ tıklama (tarayıcı menüsü engellenir).
  const onVertexDelete = (e: ReactMouseEvent<SVGCircleElement>, index: number): void => {
    e.preventDefault();
    e.stopPropagation();
    if (zone === undefined) return;
    draft.deleteVertex(zone.id, index);
    onChange();
  };

  const onMidpointClick = (
    e: ReactMouseEvent<SVGCircleElement>,
    index: number,
    mid: [number, number],
  ): void => {
    e.stopPropagation();
    if (zone === undefined) return;
    draft.insertVertex(zone.id, index, mid);
    onChange();
  };

  return (
    <>
      {open && drawing.points.length > 0 && (
        <g key={`draw-${epoch}`} pointerEvents="none">
          {drawing.points.length > 1 && (
            <polyline
              points={drawing.points
                .map((p) => {
                  const [x, y] = api.toScreen(p);
                  return `${x},${y}`;
                })
                .join(" ")}
              fill="none"
              stroke="var(--rgb)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {drawing.points.map((p, i) => {
            const [x, y] = api.toScreen(p);
            return <circle key={`d${i}`} cx={x} cy={y} r={3} fill="var(--rgb)" />;
          })}
        </g>
      )}

      {zone !== undefined && (
        <g>
          {zone.kind !== "line" &&
            zone.points.map((p, i) => {
              const next = zone.points[(i + 1) % zone.points.length];
              const mid: [number, number] = [(p[0] + next[0]) / 2, (p[1] + next[1]) / 2];
              const [x, y] = api.toScreen(mid);
              return (
                <circle
                  key={`m${i}`}
                  cx={x}
                  cy={y}
                  r={MIDPOINT_R}
                  fill="var(--panel-2)"
                  stroke="var(--muted)"
                  strokeWidth={1}
                  pointerEvents="all"
                  cursor="copy"
                  onClick={(e) => onMidpointClick(e, i + 1, mid)}
                >
                  <title>Köşe ekle</title>
                </circle>
              );
            })}
          {zone.points.map((p, i) => {
            const pos = drag !== null && drag.index === i ? drag.p : p;
            const [x, y] = api.toScreen(pos);
            return (
              <circle
                key={`v${i}`}
                cx={x}
                cy={y}
                r={VERTEX_R}
                fill="#fff"
                stroke="var(--line)"
                strokeWidth={1}
                pointerEvents="all"
                cursor="grab"
                onPointerDown={(e) => onVertexDown(e, i)}
                onPointerMove={(e) => onVertexMove(e, i)}
                onPointerUp={onVertexUp}
                onPointerCancel={onVertexCancel}
                onDoubleClick={(e) => onVertexDelete(e, i)}
                onContextMenu={(e) => onVertexDelete(e, i)}
              >
                <title>Köşe {i + 1}: sürükle; çift tıkla ya da sağ tıkla sil</title>
              </circle>
            );
          })}
        </g>
      )}
    </>
  );
}
