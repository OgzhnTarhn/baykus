"use client";

// Operatör ekranı (Faz 3 tasarımı §4.4): oynatıcı durum makinesini (lib/player) tarayıcıya bağlar.
// Ayarlar (conf, takip, kapı) bir sonraki kareden geçerli olur; oturum sayfa açılışında UUID.
import { useCallback, useEffect, useState } from "react";

import { AlarmList } from "@/components/AlarmList";
import { FaultPanel } from "@/components/FaultPanel";
import { GateBadge } from "@/components/GateBadge";
import { HealthPanel, STRIP_LEN, type StripEntry } from "@/components/HealthPanel";
import { Player, type FrameView, type View } from "@/components/Player";
import { TopBar } from "@/components/TopBar";
import { Alarms, type AlarmEvent, type AlarmId } from "@/lib/alarms";
import {
  applyFaults,
  buildFaultForm,
  encodePng,
  faultedChannels,
  faultSummary,
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
  type Gate,
  type SequenceIndex,
} from "@/lib/client";
import { createPlayer, type PlayerSnapshot } from "@/lib/player";
import { newSessionId } from "@/lib/session";

const DEFAULT_CONF = 0.25;

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

// Oynatıcı ve istek ayarları bileşen dışında kurulur: ayarlar render sırasında okunmaz, yalnız
// istek anında (sonraki kare) okunur.
// Oturum kimliği ilk istekte üretilir (tarayıcıda; sunucu ön-render'ında rastgele değer yok).
type Settings = Omit<DetectSettings, "session"> & { faults: FaultConfig };

export type Monitor = { active: AlarmId[]; history: AlarmEvent[]; strip: StripEntry[] };
const EMPTY_MONITOR: Monitor = { active: [], history: [], strip: [] };

// Alarmlar ve durum şeridi motor oturumunu izler: oturum sıfırlanınca (ya da motor oturumu
// kendiliğinden yeniden açınca, session_new) pencereler ve geçmiş temizlenir.
function makeConsole(
  onChange: (s: PlayerSnapshot<FrameView>) => void,
  onMonitor: (m: Monitor) => void,
) {
  let current: Settings = { conf: DEFAULT_CONF, track: true, gate: "auto", faults: NO_FAULTS };
  const blobUrls: string[][] = []; // bozulmuş karelerin nesne URL'leri; iki kare sonra bırakılır
  let session: string | null = null;
  const sid = () => (session ??= newSessionId());
  const settings = { set: (v: Settings) => (current = v) };
  const alarms = new Alarms();
  let strip: StripEntry[] = [];
  const clearMonitor = () => {
    alarms.reset();
    strip = [];
    onMonitor(EMPTY_MONITOR);
  };
  const player = createPlayer<FrameView>({
    detect: async (seq, frame) => {
      const { faults, ...rest } = current;
      const params = { ...rest, session: sid() };
      let rgbSrc = frameUrl(seq, frame, "rgb");
      let irSrc = frameUrl(seq, frame, "ir");
      const ch = faultedChannels(faults);
      let response;
      if (!ch.rgb && !ch.ir) {
        [response] = await Promise.all([
          postDetect(seq, frame, params),
          preload(rgbSrc),
          preload(irSrc),
        ]);
      } else {
        // Bozulan kanal canvas'ta üretilir ve PNG gider; gürültü kare başına tohumlu (tekrarlanabilir).
        const rng = mulberry32(frame);
        const [rgbImg, irImg] = await Promise.all([preload(rgbSrc), preload(irSrc)]);
        const parts: { rgb?: Blob; ir?: Blob } = {};
        const urls: string[] = [];
        if (ch.rgb) parts.rgb = await toPng(applyFaults(toPixels(rgbImg), "rgb", faults, rng));
        if (ch.ir) parts.ir = await toPng(applyFaults(toPixels(irImg), "ir", faults, rng));
        if (parts.rgb) urls.push((rgbSrc = URL.createObjectURL(parts.rgb)));
        if (parts.ir) urls.push((irSrc = URL.createObjectURL(parts.ir)));
        blobUrls.push(urls);
        if (blobUrls.length > 2) blobUrls.shift()!.forEach((u) => URL.revokeObjectURL(u));
        const meta = { seq, frame, ...params, fault: faultSummary(faults) };
        response = await postDetectForm(buildFaultForm(meta, parts));
      }
      if (response.session_new) clearMonitor();
      const h = response.health;
      alarms.update(h, frame);
      strip = [
        ...strip,
        { frame, rgb: h.rgb.state, ir: h.ir.state, misaligned: h.pair.misaligned },
      ].slice(-STRIP_LEN);
      onMonitor({ active: alarms.active(), history: alarms.history(), strip });
      return { seq, frame, rgbSrc, irSrc, response };
    },
    reset: () => postReset(sid()),
    onReset: clearMonitor,
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    now: () => performance.now(),
    onChange,
  });
  return { player, settings };
}

export function Console() {
  const [sequences, setSequences] = useState<SequenceIndex[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [seq, setSeq] = useState("");
  const [conf, setConf] = useState(DEFAULT_CONF);
  const [track, setTrack] = useState(true);
  const [gate, setGate] = useState<Gate>("auto");
  const [view, setView] = useState<View>("both");
  const [faults, setFaults] = useState<FaultConfig>(NO_FAULTS);
  const [snapState, setSnap] = useState<PlayerSnapshot<FrameView> | null>(null);
  const [monitor, setMonitor] = useState<Monitor>(EMPTY_MONITOR);
  const [{ player, settings }] = useState(() => makeConsole(setSnap, setMonitor));
  const snap = snapState ?? player.snapshot();

  useEffect(() => {
    settings.set({ conf, track, gate, faults });
  }, [settings, conf, track, gate, faults]);

  useEffect(() => {
    let alive = true;
    fetchSequences()
      .then((list) => {
        if (!alive) return;
        setSequences(list);
        if (list.length > 0) {
          setSeq(list[0].name);
          player.load(list[0].name, list[0].frames);
        }
      })
      .catch((e: Error) => alive && setListError(e.message));
    return () => {
      alive = false;
      player.pause();
    };
  }, [player]);

  const chooseSeq = useCallback(
    (name: string) => {
      const s = sequences.find((x) => x.name === name);
      if (!s) return;
      setSeq(name);
      player.load(s.name, s.frames);
    },
    [player, sequences],
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
        gate={gate}
        onGate={setGate}
        onReset={() => player.requestReset()}
      />
      <main className="grid flex-1 gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
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
          <Player
            badge={<GateBadge health={health} />}
            snap={snap}
            size={size}
            view={view}
            onView={setView}
            onPlay={() => player.play()}
            onPause={() => player.pause()}
            onStep={() => player.step()}
            onSeek={(i) => player.seek(i)}
            onSpeed={(v) => player.setSpeed(v)}
          />
        )}
        <aside className="flex flex-col gap-4" aria-label="Durum">
          <AlarmList active={monitor.active} history={monitor.history} />
          <HealthPanel health={health} strip={monitor.strip} />
          <FaultPanel faults={faults} onChange={setFaults} />
        </aside>
      </main>
    </div>
  );
}
