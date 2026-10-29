// Bölge eşitleme (Faz 4 §5.8): her değişiklik PUT ile motora gider; yolda tek istek uçar,
// aradaki değişiklikler tek takip isteğinde birleşir. Motor kabul ederse onaylı küme ve askı
// durumu yanıttan alınır; bekleyen yerel düzenleme varsa çalışma kümesi korunur, yoksa yanıt
// çalışma kümesini de belirler. Motor reddederse son kabul edilen kümeye dönülür, hata taslakta
// tutulur ve reddedilen kümenin üstüne binen bekleyen düzenleme düşer (takip PUT'u doğurmaz).
import type { ZoneDef } from "@/lib/client";
import type { ZoneDraft } from "@/lib/editor";

export type PutResult =
  | { ok: true; zones: ZoneDef[] }
  | { ok: false; error: string };

export type ZonePut = (defs: ZoneDef[]) => Promise<PutResult>;

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export class ZoneSync {
  private draft: ZoneDraft;
  private put: ZonePut;
  private onChange: () => void;
  private inFlight = false;
  private scheduled = false;
  private pending: ZoneDef[] | null = null;
  private waiters: (() => void)[] = [];

  constructor(draft: ZoneDraft, put: ZonePut, onChange: () => void) {
    this.draft = draft;
    this.put = put;
    this.onChange = onChange;
  }

  // Düzenleme eşzamanlı uygulanır; gönderilecek küme o anki tanımlardan alınır.
  edit(fn: (draft: ZoneDraft) => void): void {
    fn(this.draft);
    const snapshot = this.draft.defs();
    this.onChange();
    this.pending = snapshot;
    if (this.inFlight) return;
    this.schedule();
  }

  busy(): boolean {
    return this.inFlight || this.pending !== null;
  }

  // Yolda istek ve bekleyen değişiklik kalmadığında çözülür.
  idle(): Promise<void> {
    if (!this.busy()) return Promise.resolve();
    return new Promise<void>((resolve) => this.waiters.push(resolve));
  }

  private schedule(): void {
    if (this.scheduled) return;
    this.scheduled = true;
    queueMicrotask(() => {
      this.scheduled = false;
      this.flush();
    });
  }

  private flush(): void {
    if (this.inFlight || this.pending === null) return;
    const defs = this.pending;
    this.pending = null;
    this.inFlight = true;
    this.put(defs).then(
      (result) => this.settle(result),
      (e: unknown) => this.settle({ ok: false, error: message(e) }),
    );
  }

  private settle(result: PutResult): void {
    this.inFlight = false;
    if (result.ok) {
      // Motor kimliği ve durumu kendi yanıtıyla tazeler; bekleyen yerel düzenleme
      // varsa çalışma kümesi korunur, yoksa yanıt çalışma kümesini de belirler.
      this.draft.accept(result.zones, this.pending === null);
    } else {
      // Reddedilen kümenin üstüne binen düzenleme de düşer: takip PUT'u doğurmaz.
      this.draft.commit({ ok: false, error: result.error });
      this.pending = null;
    }
    this.onChange();
    if (this.pending !== null) {
      this.flush();
      return;
    }
    const waiters = this.waiters;
    this.waiters = [];
    for (const resolve of waiters) resolve();
  }
}
