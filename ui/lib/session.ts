// Motor oturumu (Faz 3 tasarımı §4.4): sayfa açılışında UUID. Takipçi her istekte tam bir kare
// ilerlediğini varsayar; bunun dışındaki her geçişte (dizi değişimi, ileri atlama > 1, geri adım,
// aynı kareyi yeniden isteme, döngü başı) oturum sıfırlanır.
export type Position = { seq: string; index: number };

export function newSessionId(): string {
  return crypto.randomUUID();
}

export function shouldReset(prev: Position | null, next: Position): boolean {
  if (prev === null) return false; // ilk kare: oturum zaten boş
  return prev.seq !== next.seq || next.index - prev.index !== 1;
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
