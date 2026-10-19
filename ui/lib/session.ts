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

// Takip kapalıyken motor takipçiyi güncellemez (tasarım §4.1 adım 4); yeniden açılınca takipçi
// aradaki kareleri görmemiş olur, oturum sıfırlanır.
export function trackResumed(prev: boolean, next: boolean): boolean {
  return !prev && next;
}
