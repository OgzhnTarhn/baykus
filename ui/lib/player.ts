// Oynatıcı durum makinesi (Faz 3 tasarımı §4.4 akış kontrolü; plan Görev 29):
// - aynı anda tek istek; sonraki kare yanıt gelince istenir;
// - hız bir üst sınırdır (kare aralığının kalanı beklenir), kare atlanmaz;
// - motor hatasında durur ve iletiyi gösterir (yeniden deneme döngüsü yok);
// - dizi sonunda başa döner; oturum kuralı (session.ts) gerekirse önce motor oturumunu sıfırlar.
import { shouldReset, type Position } from "@/lib/session";

export type PlayerState = "idle" | "playing" | "paused";

export type PlayerSnapshot<R> = {
  state: PlayerState;
  seq: string;
  frames: number[];
  index: number; // gösterilen karenin dizideki yeri; henüz kare yoksa -1
  frame: number | null;
  result: R | null;
  error: string | null;
  busy: boolean;
  fps: number;
};

export type PlayerDeps<R> = {
  detect(seq: string, frame: number): Promise<R>;
  reset(): Promise<void>;
  sleep(ms: number): Promise<void>;
  now(): number;
  onChange(s: PlayerSnapshot<R>): void;
  onReset?(): void;
};

export type Player<R> = {
  load(seq: string, frames: number[]): void;
  play(): void;
  pause(): void;
  step(): void;
  seek(index: number): void;
  requestReset(): void;
  setSpeed(fps: number): void;
  snapshot(): PlayerSnapshot<R>;
};

export const DEFAULT_FPS = 10;

export function createPlayer<R>(deps: PlayerDeps<R>): Player<R> {
  let state: PlayerState = "idle";
  let seq = "";
  let frames: number[] = [];
  let index = -1;
  let result: R | null = null;
  let error: string | null = null;
  let busy = false; // istek yolda (oturum sıfırlama dahil)
  let waiting = false; // oynatırken kare aralığının kalanı bekleniyor
  let pendingSeek: number | null = null;
  let fps = DEFAULT_FPS;
  let last: Position | null = null; // motora en son başarıyla verilen kare
  let resetRequested = false; // operatörün "oturumu sıfırla"sı; yoldaki istek bitince yapılır

  const snapshot = (): PlayerSnapshot<R> => ({
    state,
    seq,
    frames,
    index,
    frame: index >= 0 ? frames[index] : null,
    result,
    error,
    busy,
    fps,
  });
  const emit = () => deps.onChange(snapshot());
  const nextIndex = () => (index + 1 >= frames.length ? 0 : index + 1);

  async function doReset(): Promise<void> {
    resetRequested = false;
    await deps.reset();
    last = null; // boş oturum: sonraki kare yeniden sıfırlatmaz
    deps.onReset?.();
  }

  // Yalnız oturum sıfırlama (istek yokken düğmeye basıldı ya da yoldaki istek bitti).
  async function resetOnly(): Promise<void> {
    busy = true;
    emit();
    try {
      await doReset();
    } catch (e) {
      state = "paused";
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
      emit();
    }
  }

  async function run(target: number): Promise<void> {
    busy = true;
    emit();
    const pos = { seq, index: target };
    const started = deps.now();
    try {
      if (resetRequested || shouldReset(last, pos)) await doReset();
      const r = await deps.detect(seq, frames[target]);
      if (pos.seq !== seq) return; // bu arada başka dizi yüklendi
      last = pos;
      index = target;
      result = r;
      error = null;
    } catch (e) {
      state = "paused";
      error = e instanceof Error ? e.message : String(e);
      pendingSeek = null;
      return;
    } finally {
      busy = false;
      emit();
    }
    if (pendingSeek !== null) {
      const t = pendingSeek;
      pendingSeek = null;
      return run(t);
    }
    if (state !== "playing") {
      if (resetRequested) await resetOnly();
      return;
    }
    waiting = true;
    await deps.sleep(Math.max(0, 1000 / fps - (deps.now() - started)));
    waiting = false;
    if (state === "playing" && !busy && frames.length > 0) return run(nextIndex());
  }

  const start = (target: number) => {
    if (frames.length === 0) return;
    if (busy) {
      pendingSeek = target;
      return;
    }
    void run(target);
  };

  return {
    load(s, f) {
      seq = s;
      frames = f;
      index = -1;
      result = null;
      error = null;
      pendingSeek = null;
      state = "idle";
      emit();
    },
    play() {
      if (frames.length === 0 || state === "playing") return;
      state = "playing";
      emit();
      if (!busy && !waiting) start(nextIndex());
    },
    pause() {
      if (state === "playing") state = "paused";
      emit();
    },
    step() {
      if (busy) return;
      if (state === "playing") state = "paused";
      start(nextIndex());
    },
    seek(i) {
      if (i < 0 || i >= frames.length) return;
      start(i);
    },
    requestReset() {
      resetRequested = true;
      if (!busy && !waiting) void resetOnly();
    },
    setSpeed(v) {
      fps = Math.max(0.1, v);
      emit();
    },
    snapshot,
  };
}
