// Sensör sağlık paneli (Faz 3 tasarımı §4.4): her kamera için durum rozeti, çift için hizasız,
// son 50 karenin durum şeridi ve ölçülen değerler (gürültü σ, kayma px, yanıt).
import type { CameraHealth, Ego, Health } from "@/lib/client";

export const STRIP_LEN = 50;

export type StripEntry = { frame: number; rgb: string; ir: string; misaligned: boolean };

const STATE_TEXT: Record<string, string> = {
  ok: "tamam",
  lost: "kayıp",
  noisy: "gürültülü",
  dark: "karanlık",
  blurry: "bulanık",
};

const STATE_COLOR: Record<string, string> = {
  ok: "var(--ok)",
  lost: "var(--alarm)",
  noisy: "var(--ir)",
  dark: "var(--warn)",
  blurry: "var(--warn)",
};

const fmt = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "–" : v.toFixed(digits);

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs"
      style={{ borderColor: color, color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} aria-hidden />
      {text}
    </span>
  );
}

function CameraRow({ name, color, cam }: { name: string; color: string; cam: CameraHealth }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-sm font-medium" style={{ color }}>
        {name}
      </span>
      <span className="flex flex-wrap justify-end gap-1">
        <Badge text={STATE_TEXT[cam.state] ?? cam.state} color={STATE_COLOR[cam.state] ?? "var(--muted)"} />
        {cam.flags
          .filter((f) => f !== cam.state)
          .map((f) => (
            <Badge key={f} text={STATE_TEXT[f] ?? f} color={STATE_COLOR[f] ?? "var(--muted)"} />
          ))}
        {cam.frozen === true && <Badge text="donmuş" color="var(--warn)" />}
      </span>
    </div>
  );
}

function Strip({ strip }: { strip: StripEntry[] }) {
  const rows: { label: string; color: (e: StripEntry) => string }[] = [
    { label: "RGB", color: (e) => STATE_COLOR[e.rgb] ?? "var(--muted)" },
    { label: "IR", color: (e) => STATE_COLOR[e.ir] ?? "var(--muted)" },
    { label: "Çift", color: (e) => (e.misaligned ? "var(--alarm)" : "var(--ok)") },
  ];
  const pad = STRIP_LEN - strip.length;
  return (
    <div className="flex flex-col gap-0.5" aria-label={`Son ${STRIP_LEN} karenin durumu`}>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-1.5">
          <span className="w-7 text-[10px] text-muted">{r.label}</span>
          <div className="flex flex-1 gap-px">
            {Array.from({ length: pad }, (_, i) => (
              <span key={`p${i}`} className="h-2.5 flex-1 bg-panel-2" />
            ))}
            {strip.map((e) => (
              <span
                key={e.frame}
                className="h-2.5 flex-1"
                style={{ background: r.color(e), opacity: 0.85 }}
                title={`kare ${e.frame}`}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

const EGO_SOURCE_TEXT: Record<Ego["source"], string> = { ir: "IR", rgb: "RGB", none: "yok" };

function EgoSection({ ego, lastReloc }: { ego?: Ego | null; lastReloc?: number | null }) {
  return (
    <div className="flex flex-col gap-1.5 border-t border-line pt-2">
      <h3 className="text-xs font-semibold text-muted">Kamera hareketi (ego)</h3>
      {ego ? (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
          <dt className="text-muted">Kanal</dt>
          <dd className="text-right">{EGO_SOURCE_TEXT[ego.source] ?? ego.source}</dd>
          <dt className="text-muted">Durum</dt>
          <dd className="text-right">{ego.ok ? "izleniyor" : "kayıp"}</dd>
          <dt className="text-muted">Harita nesli</dt>
          <dd className="num text-right">
            {ego.map_gen} ({ego.provisional ? "geçici" : "kalıcı"})
          </dd>
          <dt className="text-muted">İç nokta</dt>
          <dd className="num text-right">
            {ego.inliers === null || ego.inlier_ratio === null
              ? "–"
              : `${fmt(ego.inliers, 0)} (${fmt(ego.inlier_ratio, 2).replace(".", ",")})`}
          </dd>
          <dt className="text-muted">Son yeniden konumlama</dt>
          <dd className="num text-right">{lastReloc ? `kare ${lastReloc}` : "–"}</dd>
        </dl>
      ) : (
        <p className="text-xs text-muted">ego yok (takip kapalı ya da oturumsuz)</p>
      )}
    </div>
  );
}

export function HealthPanel({
  health,
  strip,
  ego,
  lastReloc,
}: {
  health: Health | null;
  strip: StripEntry[];
  ego?: Ego | null;
  lastReloc?: number | null;
}) {
  return (
    <section className="flex flex-col gap-3 rounded border border-line bg-panel p-3">
      <h2 className="text-sm font-semibold">Sensör sağlığı</h2>
      {health ? (
        <>
          <CameraRow name="RGB" color="var(--rgb)" cam={health.rgb} />
          <CameraRow name="IR" color="var(--ir)" cam={health.ir} />
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Çift</span>
            {health.pair.misaligned ? (
              <Badge text="hizasız" color="var(--alarm)" />
            ) : (
              <Badge text="hizalı" color="var(--ok)" />
            )}
          </div>
          <Strip strip={strip} />
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
            <dt className="text-muted">RGB gürültü σ</dt>
            <dd className="num text-right">{fmt(health.rgb.noise, 2)}</dd>
            <dt className="text-muted">IR gürültü σ</dt>
            <dd className="num text-right">{fmt(health.ir.noise, 2)}</dd>
            <dt className="text-muted">RGB kare farkı</dt>
            <dd className="num text-right">{fmt(health.rgb.diff_mad, 2)}</dd>
            <dt className="text-muted">IR kare farkı</dt>
            <dd className="num text-right">{fmt(health.ir.diff_mad, 2)}</dd>
            <dt className="text-muted">Kayma</dt>
            <dd className="num text-right">
              {fmt(health.pair.misalign_px)} px ({fmt(health.pair.dx_px)}, {fmt(health.pair.dy_px)})
            </dd>
            <dt className="text-muted">Eşleşme yanıtı</dt>
            <dd className="num text-right">{fmt(health.pair.response, 3)}</dd>
          </dl>
          <EgoSection ego={ego} lastReloc={lastReloc} />
        </>
      ) : (
        <p className="text-sm text-muted">Oynatma başlayınca her karenin sağlığı burada görünür.</p>
      )}
    </section>
  );
}
