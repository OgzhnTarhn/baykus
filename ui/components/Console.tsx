"use client";

// Operatör ekranı (Faz 3 tasarımı §4.4): oynatıcı durum makinesini (lib/player) tarayıcıya bağlar.
// Ayarlar (conf, takip, kapı, küçük hedef modu, seçili iz) bir sonraki kareden geçerli olur; oturum
// sayfa açılışında UUID. Faz 4 (§5.6, §5.8): kareyi dondurma, kapsamlı sıfırlama, ego son yeniden
// konumlama, yalnız hareketliler süzgeci, göreli harita ve araç kartı burada bağlanır.
import { useCallback, useEffect, useState } from "react";

import { AlarmList } from "@/components/AlarmList";
import { DetectionList } from "@/components/DetectionList";
import { FaultPanel } from "@/components/FaultPanel";
import { Footer } from "@/components/Footer";
import { LatencyChart, type LatencyView } from "@/components/LatencyChart";
import { GateBadge } from "@/components/GateBadge";
import { HealthPanel, STRIP_LEN, type StripEntry } from "@/components/HealthPanel";
import { MapPanel } from "@/components/MapPanel";
import { Player, type FrameView, type View } from "@/components/Player";
import { TopBar } from "@/components/TopBar";
import { TrackCard } from "@/components/TrackCard";
import { Alarms, type AlarmEvent, type AlarmId } from "@/lib/alarms";
import {
  applyFaults,
  buildFaultForm,
  encodePng,
  faultedChannels,
  faultSummary,
  holdFrames,
  mulberry32,
  NO_FAULTS,
  type FaultConfig,
  type Pixels,
} from "@/lib/faults";
import {
  fetchSequences,
  frameUrl,
  postDetect,
  postDetectForm,
  postReset,
  type DetectSettings,
  type Freeze,
  type Gate,
  type SequenceIndex,
  type Tiles,
} from "@/lib/client";
import { cardView } from "@/lib/card";
import { LatencyWindow } from "@/lib/latency";
import { Mosaic } from "@/lib/mosaic";
import { createPlayer, type PlayerSnapshot } from "@/lib/player";
import { isCurrent, newSessionId, nextEpoch, trackResumed, type Stamp } from "@/lib/session";
import { DEFAULT_SETTINGS } from "@/lib/settings";


// Harita karo küçük resmi kare ölçüsünün dörtte biri (Mosaic thumbScale).
const THUMB_SCALE = 4;

async function preload(src: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = src;
  try {
    await img.decode();
  } catch {
    throw new Error("kare görüntüsü yüklenemedi");
  }
  return img;
}

function toPixels(img: HTMLImageElement): Pixels {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas açılamadı");
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height);
  return { width: d.width, height: d.height, data: d.data };
}

function toPng(p: Pixels): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = p.width;
  c.height = p.height;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("canvas açılamadı");
  ctx.putImageData(new ImageData(new Uint8ClampedArray(p.data), p.width, p.height), 0, 0);
  return encodePng(c);
}

// Harita karo küçük resmi: kaynak kanalın karesi dörtte bir ölçekte tuvale çizilir.
function thumbOf(img: HTMLImageElement, size: [number, number]): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(size[0] / 4));
  c.height = Math.max(1, Math.round(size[1] / 4));
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("canvas açılamadı");
  ctx.drawImage(img, 0, 0, c.width, c.height);
  return c;
}

// Oynatıcı ve istek ayarları bileşen dışında kurulur: ayarlar render sırasında okunmaz, yalnız
// istek anında (sonraki kare) okunur.
// Oturum kimliği ilk istekte üretilir (tarayıcıda; sunucu ön-render'ında rastgele değer yok).
type Settings = Omit<DetectSettings, "session" | "tiles"> & {
  faults: FaultConfig;
  tiles: Tiles;
  focus: number | null; // seçili iz; istekle motora gider (yanıtta focus_track)
};

export type Monitor = {
  active: AlarmId[];
  history: AlarmEvent[];
  strip: StripEntry[];
  latency: LatencyView;
  lastReloc: number | null; // ego'nun son yeniden konumlandığı kare (Faz 4 §5.7)
  mapVersion: number; // harita her geçerli yanıtta yeniden çizilir (Faz 4 §5.8)
};
const EMPTY_MONITOR: Monitor = {
  active: [],
  history: [],
  strip: [],
  latency: { totals: [], stats: null, breakdown: [] },
  lastReloc: null,
  mapVersion: 0,
};

// Alarmlar ve durum şeridi motor oturumunu izler: oturum sıfırlanınca (ya da motor oturumu
// kendiliğinden yeniden açınca, session_new) pencereler ve geçmiş temizlenir. Gecikme penceresi
// oturumdan bağımsızdır (son 100 kare).
function makeConsole(
  onChange: (s: PlayerSnapshot<FrameView>) => void,
  onMonitor: (m: Monitor) => void,
) {
  let current: Settings = {
    ...DEFAULT_SETTINGS,
    faults: NO_FAULTS,
    tiles: 1,
    focus: null,
  };
  const blobUrls: string[][] = []; // bozulmuş karelerin nesne URL'leri; iki kare sonra bırakılır
  let session: string | null = null;
  const sid = () => (session ??= newSessionId());
  let epoch = 0; // dizi değişimi ve oturum sıfırlamasında artar (S12)
  const stamp = (): Stamp => ({ session: sid(), epoch });
  const alarms = new Alarms();
  const latency = new LatencyWindow();
  let strip: StripEntry[] = [];
  let held: Freeze = {}; // dondurulan kanalların tuttuğu kare (Faz 4 §5.6)
  let lastReloc: number | null = null; // ego'nun son yeniden konumlandığı kare
  let size: [number, number] = [640, 512]; // dizinin kare ölçüsü (load ile gelir)
  let mosaic = new Mosaic(size, THUMB_SCALE); // göreli harita (Faz 4 §5.8)
  let mapVersion = 0; // her geçerli yanıtta artar; harita yeniden çizim tetiğidir
  const emit = () =>
    onMonitor({
      active: alarms.active(),
      history: alarms.history(),
      strip,
      latency: { totals: latency.totals(), stats: latency.stats(), breakdown: latency.breakdown() },
      lastReloc,
      mapVersion,
    });
  const clearMonitor = () => {
    epoch = nextEpoch(epoch);
    alarms.reset();
    strip = [];
    emit();
  };
  const player = createPlayer<FrameView>({
    detect: async (seq, frame) => {
      const { faults, focus, ...rest } = current;
      const params = { ...rest, session: sid() };
      const sent = stamp();
      // Dondurulan kanal ilk kareyi tutar; görüntü ve motora giden baytlar o kareden gelir.
      held = holdFrames(held, faults, frame);
      const freeze = held.rgb !== undefined || held.ir !== undefined ? held : undefined;
      let rgbSrc = frameUrl(seq, held.rgb ?? frame, "rgb");
      let irSrc = frameUrl(seq, held.ir ?? frame, "ir");
      const ch = faultedChannels(faults);
      let response;
      let rgbImg: HTMLImageElement;
      let irImg: HTMLImageElement;
      if (!ch.rgb && !ch.ir) {
        const [res, rgb, ir] = await Promise.all([
          postDetect(seq, frame, params, { focus, freeze }),
          preload(rgbSrc),
          preload(irSrc),
        ]);
        response = res;
        rgbImg = rgb;
        irImg = ir;
      } else {
        // Bozulan kanal canvas'ta üretilir ve PNG gider; gürültü kare başına tohumlu (tekrarlanabilir).
        const rng = mulberry32(frame);
        [rgbImg, irImg] = await Promise.all([preload(rgbSrc), preload(irSrc)]);
        const parts: { rgb?: Blob; ir?: Blob } = {};
        const urls: string[] = [];
        if (ch.rgb) parts.rgb = await toPng(applyFaults(toPixels(rgbImg), "rgb", faults, rng));
        if (ch.ir) parts.ir = await toPng(applyFaults(toPixels(irImg), "ir", faults, rng));
        if (parts.rgb) urls.push((rgbSrc = URL.createObjectURL(parts.rgb)));
        if (parts.ir) urls.push((irSrc = URL.createObjectURL(parts.ir)));
        blobUrls.push(urls);
        if (blobUrls.length > 2) blobUrls.shift()!.forEach((u) => URL.revokeObjectURL(u));
        const meta = { seq, frame, ...params, focus, fault: faultSummary(faults), freeze };
        response = await postDetectForm(buildFaultForm(meta, parts));
      }
      // dizi değişti ya da oturum sıfırlandı: eski yanıt izlemeye yazılmaz (oynatıcı da göstermez)
      if (!isCurrent(sent, stamp())) return { seq, frame, rgbSrc, irSrc, response };
      if (response.session_new) {
        clearMonitor();
        mosaic.reset(epoch); // motor oturumu yeniden açıldı: nesiller 1'den başlar, eski harita atılır
      }
      if (response.ego?.relocalized) lastReloc = frame;
      // Harita: gerekiyorsa ego'nun kaynak kanal karesi karo küçük resmi olur.
      const ego = response.ego ?? null;
      const thumb =
        ego !== null && mosaic.needsThumb(ego)
          ? thumbOf(ego.source === "rgb" ? rgbImg : irImg, size)
          : null;
      mosaic.push(frame, ego, response.tracks ?? [], thumb);
      mapVersion += 1;
      const h = response.health;
      alarms.update(h, frame);
      strip = [
        ...strip,
        { frame, rgb: h.rgb.state, ir: h.ir.state, misaligned: h.pair.misaligned },
      ].slice(-STRIP_LEN);
      latency.push(response.timing_ms);
      emit();
      return { seq, frame, rgbSrc, irSrc, response };
    },
    reset: (scope) => postReset(sid(), scope),
    onReset: (scope) => {
      if (scope === "all") lastReloc = null;
      clearMonitor();
      // "tracks" sıfırlaması haritayı korur; yalnız kapsamlı sıfırlama haritayı boşaltır.
      if (scope === "all") mosaic.reset(epoch);
    },
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    now: () => performance.now(),
    onChange,
  });
  const settings = {
    set: (v: Settings) => {
      if (trackResumed(current.track, v.track)) player.requestReset("tracks");
      current = v;
    },
  };
  // Dizi değişimi yeni dönem açar: yoldaki eski dizi yanıtı izlemeye yazılmaz.
  const load = (name: string, frames: number[], frameSize: [number, number] = [640, 512]) => {
    epoch = nextEpoch(epoch);
    held = {};
    lastReloc = null;
    size = frameSize;
    mosaic = new Mosaic(size, THUMB_SCALE); // yeni dizi yeni harita
    mosaic.reset(epoch);
    player.load(name, frames);
  };
  return { player, settings, load, getMosaic: () => mosaic };
}

// Ana alan düzeni (Faz 4 §5.8): "İkisi" oynatıcı ve haritayı yan yana (xl altı alt alta) gösterir.
type Layout = "video" | "map" | "both";
const LAYOUTS: { value: Layout; label: string }[] = [
  { value: "video", label: "Görüntü" },
  { value: "map", label: "Harita" },
  { value: "both", label: "İkisi" },
];

export function Console() {
  const [sequences, setSequences] = useState<SequenceIndex[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [seq, setSeq] = useState("");
  const [conf, setConf] = useState(DEFAULT_SETTINGS.conf);
  const [track, setTrack] = useState(DEFAULT_SETTINGS.track);
  const [gate, setGate] = useState<Gate>(DEFAULT_SETTINGS.gate);
  const [tiles, setTiles] = useState<Tiles>(1);
  const [onlyMoving, setOnlyMoving] = useState(false);
  const [view, setView] = useState<View>("both");
  const [layout, setLayout] = useState<Layout>("both");
  const [selected, setSelected] = useState<number | null>(null);
  const [follow, setFollow] = useState(0); // "Haritada göster" sayacı
  const [faults, setFaults] = useState<FaultConfig>(NO_FAULTS);
  const [snapState, setSnap] = useState<PlayerSnapshot<FrameView> | null>(null);
  const [monitor, setMonitor] = useState<Monitor>(EMPTY_MONITOR);
  const [{ player, settings, load, getMosaic }] = useState(() =>
    makeConsole(setSnap, setMonitor),
  );
  const snap = snapState ?? player.snapshot();

  useEffect(() => {
    settings.set({ conf, track, gate, faults, tiles, focus: selected });
  }, [settings, conf, track, gate, faults, tiles, selected]);

  useEffect(() => {
    let alive = true;
    fetchSequences()
      .then((list) => {
        if (!alive) return;
        setSequences(list);
        if (list.length > 0) {
          setSeq(list[0].name);
          load(list[0].name, list[0].frames, list[0].size);
        }
      })
      .catch((e: Error) => alive && setListError(e.message));
    return () => {
      alive = false;
      player.pause();
    };
  }, [player, load]);

  const chooseSeq = useCallback(
    (name: string) => {
      const s = sequences.find((x) => x.name === name);
      if (!s) return;
      setSeq(name);
      load(s.name, s.frames, s.size);
    },
    [load, sequences],
  );

  // Klavye: boşluk oynat/durdur, sağ ok sonraki kare (form alanı odaktayken değil).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(t.tagName)) return;
      if (e.key === " ") {
        e.preventDefault();
        if (player.snapshot().state === "playing") player.pause();
        else player.play();
      } else if (e.key === "ArrowRight") {
        player.step();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player]);

  const current = sequences.find((s) => s.name === seq);
  const size: [number, number] = current?.size ?? [640, 512];
  const health = snap.result?.response.health ?? null;
  const tracks = snap.result?.response.tracks ?? [];
  const focusTrack = snap.result?.response.focus_track ?? null;
  // Araç kartı: seçili izin özeti (focus) yanıtta varsa; fps dizinin index.json değerinden.
  const card =
    focusTrack !== null && snap.frame !== null
      ? cardView(focusTrack, current?.fps ?? 0, snap.frame)
      : null;

  return (
    <div className="flex min-h-screen flex-col">
      <TopBar
        sequences={sequences}
        seq={seq}
        onSeq={chooseSeq}
        conf={conf}
        onConf={setConf}
        track={track}
        onTrack={setTrack}
        onlyMoving={onlyMoving}
        onOnlyMoving={setOnlyMoving}
        tiles={tiles}
        onTiles={setTiles}
        gate={gate}
        onGate={setGate}
        onReset={() => player.requestReset("all")}
      />
      <main className="flex flex-1 flex-col gap-3 p-4">
        <div role="group" aria-label="Düzen" className="flex gap-1">
          {LAYOUTS.map((l) => (
            <button
              key={l.value}
              type="button"
              aria-pressed={layout === l.value}
              className={`rounded border px-2 py-1 ${
                layout === l.value ? "border-rgb text-rgb" : "border-line text-muted"
              }`}
              onClick={() => setLayout(l.value)}
            >
              {l.label}
            </button>
          ))}
        </div>
        <div className="grid flex-1 gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
          {listError ? (
            <p role="alert" className="text-sm">
              Dizi listesi okunamadı: {listError}. Demo verisinin /data/sequences altında olduğunu
              denetleyin.
            </p>
          ) : sequences.length === 0 ? (
            <p className="text-sm text-muted">
              Gösterilecek dizi yok. Demo verisini /data/sequences altına koyun.
            </p>
          ) : (
            <div className="relative flex min-w-0 flex-col gap-4 xl:flex-row">
              {layout !== "map" && (
                <div className="flex min-w-0 flex-1 flex-col">
                  <Player
                    badge={<GateBadge health={health} />}
                    snap={snap}
                    size={size}
                    view={view}
                    onlyMoving={onlyMoving}
                    selected={selected}
                    onSelect={setSelected}
                    onView={setView}
                    onPlay={() => player.play()}
                    onPause={() => player.pause()}
                    onStep={() => player.step()}
                    onSeek={(i) => player.seek(i)}
                    onSpeed={(v) => player.setSpeed(v)}
                  />
                </div>
              )}
              {layout !== "video" && (
                <div className="flex min-w-0 flex-1 flex-col">
                  <MapPanel
                    mosaic={getMosaic()}
                    version={monitor.mapVersion}
                    tracks={tracks}
                    selected={selected}
                    onSelect={setSelected}
                    follow={follow}
                  />
                </div>
              )}
              {selected !== null && (
                <TrackCard
                  view={card}
                  onClose={() => setSelected(null)}
                  onShowOnMap={() => {
                    setLayout((l) => (l === "video" ? "both" : l));
                    setFollow((f) => f + 1);
                  }}
                />
              )}
            </div>
          )}
          <aside className="flex flex-col gap-4" aria-label="Durum">
            <AlarmList active={monitor.active} history={monitor.history} />
            <HealthPanel
              health={health}
              strip={monitor.strip}
              ego={snap.result?.response.ego ?? null}
              lastReloc={monitor.lastReloc}
            />
            <DetectionList
              detections={snap.result?.response.detections ?? null}
              onlyMoving={onlyMoving}
            />
            <LatencyChart view={monitor.latency} />
            <FaultPanel faults={faults} onChange={setFaults} />
          </aside>
        </div>
      </main>
      <Footer />
    </div>
  );
}
