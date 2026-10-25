"use client";

// Operatör ekranı (Faz 3 tasarımı §4.4): oynatıcı durum makinesini (lib/player) tarayıcıya bağlar.
// Ayarlar (conf, takip, kapı, küçük hedef modu, seçili iz) bir sonraki kareden geçerli olur; oturum
// sayfa açılışında UUID. Faz 4 (§5.6, §5.8): kareyi dondurma, kapsamlı sıfırlama, ego son yeniden
// konumlama, yalnız hareketliler süzgeci, göreli harita ve araç kartı burada bağlanır.
import { useCallback, useEffect, useState } from "react";

import { AlarmList } from "@/components/AlarmList";
import { AlarmPanel } from "@/components/AlarmPanel";
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
import { ZoneEditorLayer, ZoneToolbar } from "@/components/ZoneEditor";
import { ZoneList } from "@/components/ZoneList";
import {
  Alarms,
  ZoneAlarmBook,
  type AlarmEvent,
  type AlarmId,
  type ZoneAlarmHistory,
} from "@/lib/alarms";
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
  fetchZones,
  frameUrl,
  postAck,
  postDetect,
  postDetectForm,
  postReset,
  putZones,
  type Alarm,
  type DetectSettings,
  type Freeze,
  type Gate,
  type LineCount,
  type SequenceIndex,
  type Tiles,
  type ZoneCounts,
} from "@/lib/client";
import { cardView } from "@/lib/card";
import { ZoneDraft, type Tool } from "@/lib/editor";
import { LatencyWindow } from "@/lib/latency";
import { Mosaic } from "@/lib/mosaic";
import { createPlayer, type PlayerSnapshot } from "@/lib/player";
import { isCurrent, newSessionId, nextEpoch, trackResumed, type Stamp } from "@/lib/session";
import { DEFAULT_SETTINGS } from "@/lib/settings";
import { ZoneSync } from "@/lib/zone-sync";


// Harita karo küçük resmi kare ölçüsünün dörtte biri (Mosaic thumbScale).
const THUMB_SCALE = 4;

// Bölge listesini değiştiren olay türleri: ekran bu olaylarda listeyi yeniden yükler.
const ZONE_EVENT_KINDS = new Set(["zones_updated", "map_new_generation", "map_merged"]);

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
  zoneAlarms: Alarm[]; // açık bölge alarmları (Faz 4 §5.5)
  zoneHistory: ZoneAlarmHistory[]; // kapanan bölge alarmları (onaylı / onaysız)
  unacked: number; // onaylanmamış açık bölge alarmı sayısı
  zoneCounts: ZoneCounts; // bölge sayaçları (motorun son yanıtı)
  mapGen: number | null; // ego'nun son harita nesli; ego yoksa korunur
  zonesChanged: number; // bölge listesi değişim sayacı (ekran listeyi yeniler)
};
const EMPTY_MONITOR: Monitor = {
  active: [],
  history: [],
  strip: [],
  latency: { totals: [], stats: null, breakdown: [] },
  lastReloc: null,
  mapVersion: 0,
  zoneAlarms: [],
  zoneHistory: [],
  unacked: 0,
  zoneCounts: {},
  mapGen: null,
  zonesChanged: 0,
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
  const zoneBook = new ZoneAlarmBook();
  const latency = new LatencyWindow();
  let strip: StripEntry[] = [];
  let held: Freeze = {}; // dondurulan kanalların tuttuğu kare (Faz 4 §5.6)
  let lastReloc: number | null = null; // ego'nun son yeniden konumlandığı kare
  let size: [number, number] = [640, 512]; // dizinin kare ölçüsü (load ile gelir)
  let mosaic = new Mosaic(size, THUMB_SCALE); // göreli harita (Faz 4 §5.8)
  let mapVersion = 0; // her geçerli yanıtta artar; harita yeniden çizim tetiğidir
  let zoneCounts: ZoneCounts = {}; // motorun son bölge sayaçları
  let mapGen: number | null = null; // ego'nun son harita nesli; ego yoksa korunur
  let zonesChanged = 0; // bölge listesi değişim sayacı (ekran listeyi yeniler)
  const emit = () =>
    onMonitor({
      active: alarms.active(),
      history: alarms.history(),
      strip,
      latency: { totals: latency.totals(), stats: latency.stats(), breakdown: latency.breakdown() },
      lastReloc,
      mapVersion,
      zoneAlarms: zoneBook.open(),
      zoneHistory: zoneBook.history(),
      unacked: zoneBook.unacked(),
      zoneCounts,
      mapGen,
      zonesChanged,
    });
  const clearMonitor = () => {
    epoch = nextEpoch(epoch);
    alarms.reset();
    zoneBook.reset();
    zoneCounts = {};
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
      // Bölge alarmları ve sayaçları motorun son yanıtından; liste değişimi ekranı yeniler.
      zoneBook.update(response.alarms ?? [], response.events ?? []);
      zoneCounts = response.zone_counts ?? {};
      if (response.ego) mapGen = response.ego.map_gen;
      if ((response.events ?? []).some((e) => ZONE_EVENT_KINDS.has(e.kind))) zonesChanged += 1;
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
      if (scope === "all") {
        lastReloc = null;
        mapGen = null;
        zonesChanged += 1; // ekran bölge listesini yeniden yükler
      }
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
    mapGen = null;
    zonesChanged += 1; // yeni dizi: ekran bölge listesini yeniden yükler
    size = frameSize;
    mosaic = new Mosaic(size, THUMB_SCALE); // yeni dizi yeni harita
    mosaic.reset(epoch);
    player.load(name, frames);
  };
  return { player, settings, load, getMosaic: () => mosaic, session: () => sid() };
}

// Ana alan düzeni (Faz 4 §5.8): "İkisi" oynatıcı ve haritayı yan yana (xl altı alt alta) gösterir.
type Layout = "video" | "map" | "both";
const LAYOUTS: { value: Layout; label: string }[] = [
  { value: "video", label: "Görüntü" },
  { value: "map", label: "Harita" },
  { value: "both", label: "İkisi" },
];

// Sağ sütun sekmeleri (Faz 4 §5.8): her sekme tek bir durum paneli gösterir.
type Tab = "detections" | "zones" | "alarms" | "health" | "latency";
const TABS: { value: Tab; label: string }[] = [
  { value: "detections", label: "Tespitler" },
  { value: "zones", label: "Bölgeler" },
  { value: "alarms", label: "Alarmlar" },
  { value: "health", label: "Sağlık" },
  { value: "latency", label: "Gecikme" },
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
  const [{ player, settings, load, getMosaic, session }] = useState(() =>
    makeConsole(setSnap, setMonitor),
  );
  const snap = snapState ?? player.snapshot();
  const [tab, setTab] = useState<Tab>("detections");
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [ackError, setAckError] = useState<string | null>(null);

  // Bölge editörü (Faz 4 §5.8): taslak ve eşitleyici bileşen ömrü boyunca tek; fps dizinin
  // index.json değerinden gelir (dizi yüklenmeden önce 30).
  const fps = sequences.find((s) => s.name === seq)?.fps ?? 30;
  const [zoneVersion, setZoneVersion] = useState(0);
  const bump = useCallback(() => setZoneVersion((n) => n + 1), []);
  const [tool, setTool] = useState<Tool | null>(null);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const [draft] = useState(() => new ZoneDraft(fps));
  const [sync] = useState(() => new ZoneSync(draft, (defs) => putZones(session(), defs), bump));

  useEffect(() => {
    settings.set({ conf, track, gate, faults, tiles, focus: selected });
  }, [settings, conf, track, gate, faults, tiles, selected]);

  // Bölge listesi değişti (ya da harita nesli sıfırlandı): taslak motordan tazelenir. Harita yoksa
  // bölge de yoktur; okunamayan liste taslağı olduğu gibi bırakır.
  useEffect(() => {
    let alive = true;
    const zones = monitor.mapGen === null ? Promise.resolve([]) : fetchZones(session());
    zones
      .then((list) => {
        if (!alive) return;
        draft.load(list);
        bump();
      })
      .catch(() => {
        // liste okunamadı: taslak son hâlinde kalır
      });
    return () => {
      alive = false;
    };
  }, [monitor.zonesChanged, monitor.mapGen, draft, session, bump]);

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

  // Bölge listesi ve harita aynı taslaktan okunur; askıdaki bölge listede gri görünür.
  const zones = draft
    .defs()
    .map((z) => (draft.isSuspended(z.id) ? { ...z, status: "suspended" as const } : z));
  const alarmZones = new Set(monitor.zoneAlarms.map((a) => a.zone_id));

  // Harita tıklaması: araç yoksa seçim yapılır; araç varsa nokta eklenir, çift tıklama bitirir.
  const onMapClick = (p: [number, number], ev: { double: boolean }): boolean => {
    if (tool === null || monitor.mapGen === null) return false;
    if (draft.drawing().tool === null) draft.start(tool, monitor.mapGen);
    const z = ev.double && tool !== "line" ? draft.finish() : draft.addPoint(p);
    if (z !== null) {
      sync.edit(() => {});
      setTool(null);
      setSelectedZone(z.id);
    }
    bump();
    return true;
  };

  // Liste düzenlemeleri taslağa uygulanır ve eşitleyiciye (PUT) devredilir.
  const onRename = (id: string, name: string) => sync.edit((d) => d.rename(id, name));
  const onDwellSeconds = (id: string, s: number) =>
    sync.edit((d) => d.setDwellSeconds(id, s, fps));
  const onCount = (id: string, c: LineCount) => sync.edit((d) => d.setCount(id, c));
  const onFlip = (id: string) => sync.edit((d) => d.flipLine(id));
  const onRemove = (id: string) => sync.edit((d) => d.remove(id));

  // Alarm onayı (Faz 4 §5.8): istek sürerken düğme kilitli; hata satırı panelin üstünde görünür.
  const onAck = (id: string) => {
    setPending((p) => new Set(p).add(id));
    setAckError(null);
    postAck(session(), id)
      .catch((e: Error) => setAckError(e.message))
      .finally(() =>
        setPending((p) => {
          const next = new Set(p);
          next.delete(id);
          return next;
        }),
      );
  };

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
        unacked={monitor.unacked}
        onAlarms={() => setTab("alarms")}
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
                    zones={zones}
                    alarmZones={alarmZones}
                    counts={monitor.zoneCounts}
                    onMapClick={onMapClick}
                    overlay={(api) => (
                      <ZoneEditorLayer
                        draft={draft}
                        version={zoneVersion}
                        selectedZone={selectedZone}
                        api={api}
                        onChange={() => sync.edit(() => {})}
                      />
                    )}
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
            <div role="tablist" aria-label="Durum sekmeleri" className="flex flex-wrap gap-1">
              {TABS.map((t) => (
                <button
                  key={t.value}
                  type="button"
                  role="tab"
                  aria-selected={tab === t.value}
                  className={`rounded border px-2 py-1 text-sm ${
                    tab === t.value ? "border-rgb text-rgb" : "border-line text-muted"
                  }`}
                  onClick={() => setTab(t.value)}
                >
                  {t.label}
                  {t.value === "alarms" && monitor.unacked > 0 ? ` (${monitor.unacked})` : ""}
                </button>
              ))}
            </div>
            <div role="tabpanel" className="flex flex-col gap-4">
              {tab === "detections" && (
                <DetectionList
                  detections={snap.result?.response.detections ?? null}
                  onlyMoving={onlyMoving}
                />
              )}
              {tab === "zones" && (
                <ZoneList
                  zones={zones}
                  counts={monitor.zoneCounts}
                  fps={fps}
                  error={draft.error()}
                  selected={selectedZone}
                  onSelect={setSelectedZone}
                  onRename={onRename}
                  onDwellSeconds={onDwellSeconds}
                  onCount={onCount}
                  onFlip={onFlip}
                  onRemove={onRemove}
                  toolbar={
                    <ZoneToolbar tool={tool} onTool={setTool} disabled={monitor.mapGen === null} />
                  }
                />
              )}
              {tab === "alarms" && (
                <>
                  <AlarmList active={monitor.active} history={monitor.history} />
                  {ackError !== null && (
                    <p role="alert" className="text-sm">
                      Alarm onaylanamadı: {ackError}
                    </p>
                  )}
                  <AlarmPanel
                    open={monitor.zoneAlarms}
                    history={monitor.zoneHistory}
                    zones={zones}
                    frame={snap.frame ?? 0}
                    fps={fps}
                    pending={pending}
                    onAck={onAck}
                  />
                </>
              )}
              {tab === "health" && (
                <>
                  <HealthPanel
                    health={health}
                    strip={monitor.strip}
                    ego={snap.result?.response.ego ?? null}
                    lastReloc={monitor.lastReloc}
                  />
                  <FaultPanel faults={faults} onChange={setFaults} />
                </>
              )}
              {tab === "latency" && <LatencyChart view={monitor.latency} />}
            </div>
          </aside>
        </div>
      </main>
      <Footer />
    </div>
  );
}
