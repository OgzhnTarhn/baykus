// Arıza simülasyonu paneli (FR-U4): karartma, IR gürültü, IR kaydırma, bulanıklık, kamerayı kapat,
// kareyi dondur.
// Değişiklik bir sonraki kareden geçerli olur.
import { LEVELS, NO_FAULTS, faultedChannels, type FaultConfig } from "@/lib/faults";

type Option = { value: number; label: string };

function Choice({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: number;
  options: Option[];
  onChange(v: number): void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-sm">
      <span>{label}</span>
      <select
        className="rounded border border-line bg-panel-2 px-2 py-1"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

const off = (v: number): Option => ({ value: v, label: "kapalı" });

export function FaultPanel({
  faults,
  onChange,
}: {
  faults: FaultConfig;
  onChange(f: FaultConfig): void;
}) {
  const set = (patch: Partial<FaultConfig>) => onChange({ ...faults, ...patch });
  const ch = faultedChannels(faults);
  const pixel = ch.rgb || ch.ir;
  const on = pixel || faults.freezeRgb || faults.freezeIr;
  const frozen = [faults.freezeRgb && "RGB", faults.freezeIr && "IR"].filter(Boolean).join(" ve ");
  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Arıza simülasyonu</h2>
        {on && (
          <button
            type="button"
            className="rounded border border-line px-2 py-0.5 text-xs hover:border-text"
            onClick={() => onChange(NO_FAULTS)}
          >
            Arızaları kapat
          </button>
        )}
      </div>
      <Choice
        label="RGB karartma"
        value={faults.darken}
        options={[off(1), ...LEVELS.darken.map((v) => ({ value: v, label: `× ${v}` }))]}
        onChange={(v) => set({ darken: v })}
      />
      <Choice
        label="IR gürültü"
        value={faults.noise}
        options={[off(0), ...LEVELS.noise.map((v) => ({ value: v, label: `σ ${v}` }))]}
        onChange={(v) => set({ noise: v })}
      />
      <Choice
        label="IR kaydırma"
        value={faults.shift}
        options={[off(0), ...LEVELS.shift.map((v) => ({ value: v, label: `${v} px sağa` }))]}
        onChange={(v) => set({ shift: v })}
      />
      <Choice
        label="Bulanıklık (iki kamera)"
        value={faults.blur}
        options={[off(0), ...LEVELS.blur.map((v) => ({ value: v, label: `${v} × ${v}` }))]}
        onChange={(v) => set({ blur: v })}
      />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={faults.offRgb}
            onChange={(e) => set({ offRgb: e.target.checked })}
          />
          RGB kamerayı kapat
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={faults.offIr}
            onChange={(e) => set({ offIr: e.target.checked })}
          />
          IR kamerayı kapat
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={faults.freezeRgb}
            onChange={(e) => set({ freezeRgb: e.target.checked })}
          />
          RGB kareyi dondur
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={faults.freezeIr}
            onChange={(e) => set({ freezeIr: e.target.checked })}
          />
          IR kareyi dondur
        </label>
      </div>
      <p className="text-xs text-muted">
        {pixel
          ? `Bozulan kanal tarayıcıda üretilip PNG olarak gönderiliyor: ${[
              ch.rgb && "RGB",
              ch.ir && "IR",
            ]
              .filter(Boolean)
              .join(" ve ")}.`
          : "Arıza yok: iki kanalın özgün dosyaları motora gidiyor."}
        {frozen && ` Dondurulan kanal: ${frozen} (aynı karenin özgün baytları tekrar gönderiliyor).`}
      </p>
    </section>
  );
}
