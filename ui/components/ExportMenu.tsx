"use client";
// Dışa aktarma menüsü (Faz 4 §5.13): "Dışa aktar" düğmesi küçük bir menü açar; olay günlüğü
// (JSON/CSV) ve harita (PNG: tamamı / görünen) indirmeleri üst bileşenden gelir. Menü tıklamadan
// sonra, Esc ile ve dışarı tıklanınca kapanır; iş sürerken düğmeler kilitlenir.
import { useEffect, useRef, useState } from "react";

export type ExportMenuProps = {
  onEventsJson(): Promise<void>;
  onEventsCsv(): Promise<void>;
  onMapFull(): Promise<void>;
  onMapView(): Promise<void>;
  disabled?: boolean;
  busy?: boolean;
  error?: string | null;
};

export function ExportMenu({
  onEventsJson,
  onEventsCsv,
  onMapFull,
  onMapView,
  disabled = false,
  busy = false,
  error = null,
}: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  // Menü açıkken Esc kapatır, dışarı tıklama da kapatır; dinleyiciler yalnız açıkken kurulur.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: PointerEvent): void => {
      if (root.current !== null && !root.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  // Tıklama menüyü hemen kapatır; hata üst bileşende `error` olarak gösterilir.
  const run = (fn: () => Promise<void>): void => {
    setOpen(false);
    fn().catch(() => {});
  };

  const items: { label: string; run(): Promise<void> }[] = [
    { label: "Olaylar (JSON)", run: onEventsJson },
    { label: "Olaylar (CSV)", run: onEventsCsv },
    { label: "Harita (PNG: tamamı)", run: onMapFull },
    { label: "Harita (PNG: görünen)", run: onMapView },
  ];
  const locked = disabled || busy;

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        className="rounded border border-line px-3 py-1 text-sm hover:border-text disabled:opacity-40"
        aria-expanded={open}
        aria-haspopup="true"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        Dışa aktar
      </button>

      {open && (
        <div className="absolute right-0 z-20 mt-1 flex w-56 flex-col gap-1 rounded border border-line bg-panel-2 p-2 shadow-lg">
          {items.map((item) => (
            <button
              key={item.label}
              type="button"
              className="rounded border border-line px-2 py-1 text-left text-sm hover:border-text disabled:opacity-40"
              disabled={locked}
              onClick={() => run(item.run)}
            >
              {item.label}
            </button>
          ))}
          {busy && <p className="text-xs text-muted">hazırlanıyor…</p>}
          <p className="text-xs text-muted">İnternetsiz: dosyalar bu bilgisayara indirilir.</p>
        </div>
      )}

      {error !== null && (
        <p role="alert" className="text-xs text-alarm">
          {error}
        </p>
      )}
    </div>
  );
}
