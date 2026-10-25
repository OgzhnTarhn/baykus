// Araç kartı (Faz 4 tasarımı §5.8): seçili aracın özeti — başlık, durum rozeti, süre, hareket
// şeridi, duruş / yol satırları ve bölge ziyaretleri. Haritanın üstünde yüzer; kapatılabilir.
import type { CSSProperties } from "react";
import type { CardView } from "@/lib/card";
import type { Motion } from "@/lib/client";
import { MOTION_TEXT, text } from "@/lib/texts";

// Hareket şeridi: hareketli düz dolgu, duran kesikli tarama, bilinmeyen nötr zemin.
const SEGMENT_STYLE: Record<Motion, CSSProperties> = {
  moving: { background: "var(--text)" },
  stationary: {
    background:
      "repeating-linear-gradient(45deg, var(--text) 0 2px, transparent 2px 5px)",
  },
  unknown: { background: "var(--panel-2)" },
};

function CloseButton({ onClose }: { onClose(): void }) {
  return (
    <button
      type="button"
      onClick={onClose}
      aria-label="Kartı kapat"
      className="rounded border border-line px-1.5 text-xs text-muted hover:text-text"
    >
      ×
    </button>
  );
}

function SegmentStrip({ segments }: { segments: CardView["segments"] }) {
  return (
    <div className="flex h-2.5 gap-px overflow-hidden rounded-sm" aria-label="Hareket şeridi">
      {segments.map((s, i) => (
        <span
          key={i}
          className="h-full"
          style={{ flexGrow: s.ratio, flexBasis: 0, ...SEGMENT_STYLE[s.state] }}
          title={text(MOTION_TEXT, s.state)}
        />
      ))}
    </div>
  );
}

export function TrackCard({
  view,
  onClose,
  onShowOnMap,
}: {
  view: CardView | null;
  onClose(): void;
  onShowOnMap(): void;
}) {
  if (!view) {
    return (
      <section className="absolute right-4 top-4 z-20 w-72 rounded border border-line bg-panel p-3">
        <div className="flex items-start justify-between gap-2">
          <p className="text-xs text-muted">Seçili araç için özet henüz yok.</p>
          <CloseButton onClose={onClose} />
        </div>
      </section>
    );
  }

  return (
    <section
      className={`absolute right-4 top-4 z-20 w-72 rounded border border-line bg-panel p-3 ${
        view.ended ? "opacity-60" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <h2 className="text-sm font-semibold">{view.title}</h2>
          <span className="inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-xs text-muted">
            {view.tag ? <span className="font-semibold">{view.tag}</span> : null}
            {view.stateText}
          </span>
        </div>
        <CloseButton onClose={onClose} />
      </div>

      <p className="mt-2 text-xs">
        {view.durationText} <span className="text-muted">{view.fpsNote}</span>
      </p>

      <div className="mt-2">
        <SegmentStrip segments={view.segments} />
      </div>

      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-muted">Toplam duruş</dt>
        <dd className="num text-right">{view.stationaryText}</dd>
        <dt className="text-muted">Göreli yol</dt>
        <dd className="num text-right">{view.pathText}</dd>
      </dl>

      <div className="mt-2 flex flex-col gap-1 border-t border-line pt-2">
        <h3 className="text-xs font-semibold text-muted">Bölge ziyaretleri</h3>
        {view.zones.length === 0 ? (
          <p className="text-xs text-muted">Bölge ziyareti yok.</p>
        ) : (
          <ul className="flex flex-col gap-0.5 text-xs">
            {view.zones.map((z) => (
              <li key={z.zone_id} className="flex justify-between gap-2">
                <span>{z.name}</span>
                <span className="num text-right text-muted">{z.text}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {view.ended && view.lastSeenText ? (
        <p className="mt-2 text-xs text-muted">{view.lastSeenText}</p>
      ) : null}

      <button
        type="button"
        onClick={onShowOnMap}
        className="mt-2 w-full rounded border border-line px-2 py-1 text-xs hover:bg-panel-2"
      >
        Haritada göster
      </button>
    </section>
  );
}
