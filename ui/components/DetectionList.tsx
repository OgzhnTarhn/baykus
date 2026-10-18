// Tespit listesi (Faz 3 tasarımı §4.4 sağ panel) ve skor açıklaması (bilgi düğmesinde).
import type { Detection } from "@/lib/client";
import { classColor } from "@/components/Overlay";

export const SCORE_NOTE =
  "kalibre skor: DroneVehicle val, iki kamera açık; M3OT'ta ve kapalı kanalda doğrulanmadı";

export function DetectionList({ detections }: { detections: Detection[] | null }) {
  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">
          Tespitler{detections ? <span className="num text-muted"> ({detections.length})</span> : null}
        </h2>
        <details className="relative text-xs">
          <summary
            className="cursor-pointer list-none rounded-full border border-line px-1.5 text-muted"
            aria-label="Skor nedir?"
          >
            i
          </summary>
          <p className="absolute right-0 z-10 mt-1 w-64 rounded border border-line bg-panel-2 p-2">
            {SCORE_NOTE}
          </p>
        </details>
      </div>
      {!detections || detections.length === 0 ? (
        <p className="text-sm text-muted">
          {detections ? "Bu karede eşiği geçen tespit yok." : "Henüz kare işlenmedi."}
        </p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted">
              <th className="font-normal">İz</th>
              <th className="font-normal">Sınıf</th>
              <th className="text-right font-normal">Skor</th>
            </tr>
          </thead>
          <tbody className="num">
            {detections.map((d, i) => (
              <tr key={d.track_id ?? `i${i}`}>
                <td>{d.track_id === null ? "–" : `#${d.track_id}`}</td>
                <td style={{ color: classColor(d.cls) }}>{d.cls}</td>
                <td className="text-right">{d.score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
