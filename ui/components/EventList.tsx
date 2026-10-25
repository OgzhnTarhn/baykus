"use client";

// Olaylar sekmesi (Faz 4 tasarımı §5.8) ve olay ayrıntısı: günlük kayıtları en yeniden eskiye
// listelenir (en çok 300 satır); satır seçimi kareyi, iz bağlantısı araç kartını açar. Ayrıntı
// paneli olayın metnini, karesini, bölgesini, izini ve oturum içi 128 × 128 RGB / IR kırpmalarını
// gösterir; "O kareye git" oynatıcıyı olay karesine taşır.
import type { ZoneEvent } from "@/lib/client";
import { DIR_TEXT, EVENT_TEXT, text } from "@/lib/texts";

// Uzun oturumlarda tablo şişmesin: yalnız en yeni 300 olay çizilir.
export const EVENT_ROWS_MAX = 300;

export function EventList({
  events,
  selectedSeq,
  onSelect,
  onTrack,
}: {
  events: ZoneEvent[];
  selectedSeq: number | null;
  onSelect(seq: number): void;
  onTrack(id: number): void;
}) {
  const shown = [...events].sort((a, b) => b.seq - a.seq).slice(0, EVENT_ROWS_MAX);
  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <h2 className="text-sm font-semibold">
        Olaylar<span className="num text-muted"> ({events.length})</span>
      </h2>
      {shown.length === 0 ? (
        <p className="text-sm text-muted">Henüz olay yok.</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted">
              <th className="font-normal">Kare</th>
              <th className="font-normal">Olay</th>
              <th className="font-normal">Bölge</th>
              <th className="font-normal">İz</th>
              <th className="font-normal">Yön</th>
            </tr>
          </thead>
          <tbody className="num">
            {shown.map((e) => {
              const selected = selectedSeq === e.seq;
              const trackId = e.track_id;
              return (
                <tr
                  key={e.seq}
                  aria-selected={selected}
                  className={`cursor-pointer ${
                    selected ? "bg-panel-2 text-text" : "hover:bg-panel-2"
                  }`}
                  onClick={() => onSelect(e.seq)}
                >
                  <td>{e.frame}</td>
                  <td>{text(EVENT_TEXT, e.kind)}</td>
                  <td>{e.zone_name ?? "–"}</td>
                  <td>
                    {trackId === null ? (
                      "–"
                    ) : (
                      <button
                        type="button"
                        className="rounded border border-line px-1.5 text-xs hover:border-text"
                        aria-label={`#${trackId} aracını aç`}
                        onClick={(ev) => {
                          // İz bağlantısı satırı seçmez: araç kartı açılır.
                          ev.stopPropagation();
                          onTrack(trackId);
                        }}
                      >
                        #{trackId}
                      </button>
                    )}
                  </td>
                  <td>{e.dir === null ? "–" : text(DIR_TEXT, e.dir)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

export function EventDetail({
  event,
  snapshot,
  onGoTo,
  onClose,
}: {
  event: ZoneEvent;
  snapshot: { rgb: string; ir: string } | null;
  onGoTo(frame: number): void;
  onClose(): void;
}) {
  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <div className="flex items-start justify-between gap-2">
        <h2 className="text-sm font-semibold">Olay ayrıntısı</h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Ayrıntıyı kapat"
          className="rounded border border-line px-1.5 text-xs text-muted hover:text-text"
        >
          ×
        </button>
      </div>

      <p className="text-sm">{text(EVENT_TEXT, event.kind)}</p>

      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-muted">Kare</dt>
        <dd className="num text-right">{event.frame}</dd>
        <dt className="text-muted">Bölge</dt>
        <dd className="text-right">{event.zone_name ?? "–"}</dd>
        <dt className="text-muted">İz</dt>
        <dd className="num text-right">
          {event.track_id === null ? "–" : `#${event.track_id}`}
        </dd>
      </dl>

      {snapshot ? (
        <div className="flex gap-2">
          <figure className="flex flex-col items-center gap-0.5">
            {/* eslint-disable-next-line @next/next/no-img-element -- kırpma data URL; optimizasyon yok */}
            <img
              src={snapshot.rgb}
              alt="Olay anı RGB kırpma"
              width={128}
              height={128}
              className="h-32 w-32 rounded border border-line object-cover"
            />
            <figcaption className="text-xs text-muted">RGB</figcaption>
          </figure>
          <figure className="flex flex-col items-center gap-0.5">
            {/* eslint-disable-next-line @next/next/no-img-element -- kırpma data URL; optimizasyon yok */}
            <img
              src={snapshot.ir}
              alt="Olay anı IR kırpma"
              width={128}
              height={128}
              className="h-32 w-32 rounded border border-line object-cover"
            />
            <figcaption className="text-xs text-muted">IR</figcaption>
          </figure>
        </div>
      ) : (
        <p className="text-xs text-muted">Anlık görüntü yok (yalnız bu oturumda alınır).</p>
      )}

      <button
        type="button"
        onClick={() => onGoTo(event.frame)}
        className="w-full rounded border border-line px-2 py-1 text-xs hover:bg-panel-2"
      >
        O kareye git
      </button>
    </section>
  );
}
