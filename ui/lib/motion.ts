// Hareket durumu gösterimi (Faz 4 tasarımı §5.8): duran araç kesikli ince kutu + "D", hareketli
// araç düz kalın kutu + "H"; bilinmeyen/yok Faz 3 görünümü (düz, orta, etiketsiz). Stil yalnız
// renge dayanmaz: çizgi deseni, kalınlık ve harf etiketi birlikte ayırt eder.
import type { Detection, Motion } from "@/lib/client";

export type BoxStyle = {
  dash: string | null; // SVG stroke-dasharray; null = düz çizgi
  width: number;
  tag: string | null; // kutunun yanındaki harf etiketi
};

const STATIONARY: BoxStyle = { dash: "6 4", width: 1, tag: "D" };
const MOVING: BoxStyle = { dash: null, width: 3, tag: "H" };
const UNKNOWN: BoxStyle = { dash: null, width: 2, tag: null };

export function boxStyle(motion: Motion | undefined): BoxStyle {
  if (motion === "stationary") return STATIONARY;
  if (motion === "moving") return MOVING;
  return UNKNOWN;
}

// "Yalnız hareketli" süzgeci: yalnız tespit listesini süzer, girdi dizisine dokunmaz.
export function visible(detections: Detection[], onlyMoving: boolean): Detection[] {
  if (!onlyMoving) return detections;
  return detections.filter((d) => d.motion === "moving");
}
