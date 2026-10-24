// Alarm sözleşmesi (Faz 3 tasarımı §4.4; Python eşi baykus.health.m3ot.Alarms):
// - kayıp, gürültü ve donma: son 5 karenin en az 3'ünde bayrak; hizasız: geçerli karenin
//   pair.misaligned'ı (zaman kuralı motorda zaten pencereli; ikinci pencere yok);
// - donma alarmları (rgb_frozen / ir_frozen) Faz 4 tasarımı §5.6: rgb/ir.frozen === true pencereli;
// - histerezis yok: koşul tutunca başlar, tutmayınca temizlenir; olaylar kare numarasıyla geçmişe;
// - metin alarm kimliğinden (sabit tablo), motorun kapı reason'ından değil;
// - reset() pencereleri, açık alarmları ve geçmişi temizler. Pencere sabit, yapılandırılmaz.
export const ALARM_IDS = [
  "rgb_lost",
  "ir_lost",
  "ir_noisy",
  "misaligned",
  "rgb_frozen",
  "ir_frozen",
] as const;
export type AlarmId = (typeof ALARM_IDS)[number];

export const ALARM_TEXT: Record<AlarmId, string> = {
  rgb_lost: "RGB kamera kayıp",
  ir_lost: "IR kamera kayıp",
  ir_noisy: "IR gürültülü",
  misaligned: "IR hizasız",
  rgb_frozen: "RGB görüntü donmuş",
  ir_frozen: "IR görüntü donmuş",
};

export const ALARM_WINDOW = 5;
export const ALARM_MIN = 3;

export type AlarmInput = {
  rgb: { flags: string[]; frozen?: boolean | null };
  ir: { flags: string[]; frozen?: boolean | null };
  pair: { misaligned: boolean };
};

export type AlarmEvent = { id: AlarmId; kind: "start" | "clear"; frame: number };

// Pencere sütunları: rgb kayıp, ir kayıp, ir gürültülü, rgb donmuş, ir donmuş.
const WINDOWED: readonly AlarmId[] = ["rgb_lost", "ir_lost", "ir_noisy", "rgb_frozen", "ir_frozen"];

export class Alarms {
  private window: boolean[][] = [];
  private on: AlarmId[] = [];
  private events: AlarmEvent[] = [];

  update(h: AlarmInput, frame: number): AlarmEvent[] {
    this.window.push([
      h.rgb.flags.includes("lost"),
      h.ir.flags.includes("lost"),
      h.ir.flags.includes("noisy"),
      h.rgb.frozen === true,
      h.ir.frozen === true,
    ]);
    if (this.window.length > ALARM_WINDOW) this.window.shift();
    const next: AlarmId[] = [];
    for (const id of ALARM_IDS) {
      if (id === "misaligned") {
        if (h.pair.misaligned) next.push(id);
        continue;
      }
      const col = WINDOWED.indexOf(id);
      const count = this.window.filter((w) => w[col]).length;
      if (count >= ALARM_MIN) next.push(id);
    }

    const out: AlarmEvent[] = [];
    for (const id of ALARM_IDS) {
      const was = this.on.includes(id);
      const is = next.includes(id);
      if (is && !was) out.push({ id, kind: "start", frame });
      if (!is && was) out.push({ id, kind: "clear", frame });
    }
    this.on = next;
    this.events.push(...out);
    return out;
  }

  active(): AlarmId[] {
    return [...this.on];
  }

  items(): { id: AlarmId; text: string }[] {
    return this.on.map((id) => ({ id, text: ALARM_TEXT[id] }));
  }

  history(): AlarmEvent[] {
    return [...this.events];
  }

  reset(): void {
    this.window = [];
    this.on = [];
    this.events = [];
  }
}

// Kapı rozeti: alarmlardan ayrı, her karenin gate / applied alanından (kapı anında çalışır).
export type GateBadge = { text: string; level: "applied" | "advise" };

export function gateBadge(h: { gate: string; applied: boolean }): GateBadge | null {
  if (h.gate === "none") return null;
  if (!h.applied) return { text: "yalnız uyarı", level: "advise" };
  if (h.gate === "ir_off") return { text: "IR modele verilmiyor", level: "applied" };
  if (h.gate === "rgb_off") return { text: "RGB modele verilmiyor", level: "applied" };
  return null;
}
