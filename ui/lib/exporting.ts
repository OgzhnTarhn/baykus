// Dışa aktarma (Faz 4 §5.13): internetsiz, tarayıcıda Blob indirme. Olay günlüğü JSON'u meta +
// tüm olayları taşır, CSV motorun metnini BOM ile korur, dosya adları yerel saatte kararlıdır.
import type { ZoneDef, ZoneEvent } from "@/lib/client";

// Dışa aktarılan dosyanın başlığı: oturum künyesi ve o anki bölge tanımları.
export type ExportMeta = {
  sequence: string;
  session: string;
  created_at: string;
  model_sha256: string | null;
  motion_sha256: string | null;
  fps_assumed: number;
  zones: ZoneDef[];
};

export type ExportKind = "olaylar" | "harita";
export type ExportExt = "json" | "csv" | "png";

const pad = (v: number, width: number = 2) => String(v).padStart(width, "0");

// Yerel saat bileşenleriyle ad: baykus_<sequence>_<kind>_YYYYMMDD-HHmmss.<ext>.
export function fileName(
  kind: ExportKind,
  sequence: string,
  created: Date,
  ext: ExportExt,
): string {
  const stamp =
    `${created.getFullYear()}${pad(created.getMonth() + 1)}${pad(created.getDate())}` +
    `-${pad(created.getHours())}${pad(created.getMinutes())}${pad(created.getSeconds())}`;
  return `baykus_${sequence}_${kind}_${stamp}.${ext}`;
}

// Olay günlüğü: meta + tüm olaylar, 2 boşlukla girintili.
export function eventsJson(meta: ExportMeta, events: ZoneEvent[]): Blob {
  const body = JSON.stringify({ meta, events }, null, 2);
  return new Blob([body], { type: "application/json" });
}

// Motorun CSV metni olduğu gibi korunur; yalnız başına UTF-8 BOM eklenir (Excel Türkçe).
export function csvBlob(csv: string): Blob {
  return new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" });
}

// Blob'u geçici bir <a download> ile indirir; nesne URL'si tıklamadan sonra serbest bırakılır.
export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Hemen bırakılırsa bazı tarayıcılar indirmeyi başlatmadan keser.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Harita tuvalini PNG'ye kodlar; kodlama başarısızsa Türkçe hata.
export function mapPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("PNG kodlanamadı"));
    }, "image/png");
  });
}
