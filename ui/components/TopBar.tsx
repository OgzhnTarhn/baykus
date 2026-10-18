// Üst çubuk (Faz 3 tasarımı §4.4): dizi, conf kaydırıcısı, takip aç/kapa, kapı modu, oturum sıfırla.
import type { Gate, SequenceIndex } from "@/lib/client";

const GATES: { value: Gate; label: string }[] = [
  { value: "off", label: "kapalı" },
  { value: "advise", label: "uyar" },
  { value: "auto", label: "otomatik" },
];

export type TopBarProps = {
  sequences: SequenceIndex[];
  seq: string;
  onSeq(seq: string): void;
  conf: number;
  onConf(v: number): void;
  track: boolean;
  onTrack(v: boolean): void;
  gate: Gate;
  onGate(v: Gate): void;
  onReset(): void;
};

export function TopBar(p: TopBarProps) {
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-line bg-panel px-4 py-3">
      <h1 className="text-lg font-semibold tracking-tight">Baykuş</h1>

      <label className="flex items-center gap-2 text-sm">
        <span className="text-muted">Dizi</span>
        <select
          className="rounded border border-line bg-panel-2 px-2 py-1"
          value={p.seq}
          onChange={(e) => p.onSeq(e.target.value)}
          disabled={p.sequences.length === 0}
        >
          {p.sequences.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} ({s.frames.length} kare)
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm">
        <span className="text-muted">Güven eşiği</span>
        <input
          type="range"
          min={0.05}
          max={0.95}
          step={0.05}
          value={p.conf}
          onChange={(e) => p.onConf(Number(e.target.value))}
          aria-valuetext={p.conf.toFixed(2)}
        />
        <span className="num w-10">{p.conf.toFixed(2)}</span>
      </label>

      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={p.track}
          onChange={(e) => p.onTrack(e.target.checked)}
        />
        <span>Takip</span>
      </label>

      <fieldset className="flex items-center gap-1 text-sm">
        <legend className="sr-only">Kapı modu</legend>
        <span className="mr-1 text-muted">Kapı</span>
        {GATES.map((g) => (
          <label
            key={g.value}
            className={`cursor-pointer rounded border px-2 py-1 ${
              p.gate === g.value ? "border-rgb text-rgb" : "border-line text-muted"
            }`}
          >
            <input
              type="radio"
              name="gate"
              className="sr-only"
              value={g.value}
              checked={p.gate === g.value}
              onChange={() => p.onGate(g.value)}
            />
            {g.label}
          </label>
        ))}
      </fieldset>

      <button
        type="button"
        className="ml-auto rounded border border-line px-3 py-1 text-sm hover:border-text"
        onClick={p.onReset}
      >
        Oturumu sıfırla
      </button>
    </header>
  );
}
