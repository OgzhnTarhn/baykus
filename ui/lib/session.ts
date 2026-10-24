// Motor oturumu (Faz 3 tasarımı §4.4): sayfa açılışında UUID. Takipçi her istekte tam bir kare
// ilerlediğini varsayar; bunun dışındaki her geçişte oturum sıfırlanır. Faz 4 §5.8: sıfırlama
// kapsamlıdır — aynı dizide ileri atlama > 1, geri adım, aynı kareyi yeniden isteme ve döngü başı
// yalnız takipçiyi sıfırlar (takip haritası, bölgeler ve olay günlüğü korunur); dizi değişimi tüm
// oturumu sıfırlar.
import type { ResetScope } from "@/lib/client";

export type Position = { seq: string; index: number };

export function newSessionId(): string {
  return crypto.randomUUID();
}

export function resetScope(prev: Position | null, next: Position): ResetScope | null {
  if (prev === null) return null; // ilk kare: oturum zaten boş
  if (prev.seq !== next.seq) return "all";
  return next.index - prev.index === 1 ? null : "tracks";
}

// Yanıt damgası (S12): istek gönderilirken oturum ve dönem kaydedilir; dizi değişince ya da oturum
// sıfırlanınca dönem artar. Yanıt geldiğinde damga geçerli değilse yanıt alarm, şerit ve gecikme
// penceresine yazılmaz (eski dizinin geç gelen yanıtı yeni dizinin izlemesini kirletmez).
export type Stamp = { session: string; epoch: number };

export function nextEpoch(epoch: number): number {
  return epoch + 1;
}

export function isCurrent(sent: Stamp, now: Stamp): boolean {
  return sent.session === now.session && sent.epoch === now.epoch;
}

// Takip kapalıyken motor takipçiyi güncellemez (tasarım §4.1 adım 4); yeniden açılınca takipçi
// aradaki kareleri görmemiş olur, oturum sıfırlanır.
export function trackResumed(prev: boolean, next: boolean): boolean {
  return !prev && next;
}
