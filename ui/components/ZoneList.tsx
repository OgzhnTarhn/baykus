"use client";

// Bölge listesi (Faz 4 tasarımı §5.8 "Bölgeler" sekmesi): her bölge için tür, ad, sayaçlar ve türüne
// göre düzenlenebilir alanlar. Her değişiklik üstteki editörle motora gider; motor reddederse hata
// burada görünür ve taslak son kabul edilen kümeye döner. Askıdaki bölge gri, yalnız silinebilir.
import { useRef, useState, type KeyboardEvent, type ReactNode } from "react";

import type { LineCount, ZoneCount, ZoneCounts, ZoneDef, ZoneKind } from "@/lib/client";
import { ZONE_KIND_TEXT, text } from "@/lib/texts";

const COUNT_OPTIONS: { value: LineCount; label: string }[] = [
  { value: "both", label: "İki yön" },
  { value: "a_to_b", label: "A → B" },
  { value: "b_to_a", label: "B → A" },
];

const num = (v: number | null | undefined): string =>
  v === null || v === undefined ? "–" : String(v);

// Enter ve blur aynı işi yapar: alan bırakılmadan değişiklik motora gitmez.
function commitOnEnter(e: KeyboardEvent<HTMLInputElement>, commit: () => void): void {
  if (e.key !== "Enter") return;
  e.preventDefault();
  commit();
  e.currentTarget.blur();
}

function NameField({
  name,
  disabled,
  onCommit,
}: {
  name: string;
  disabled: boolean;
  onCommit(name: string): void;
}) {
  // Taslak yalnız yazarken tutulur; bırakınca alan yine üstten gelen adı gösterir.
  const [draft, setDraft] = useState<string | null>(null);
  const committed = useRef<string | null>(null);
  const commit = () => {
    const v = (draft ?? "").trim();
    setDraft(null);
    if (v === "" || v === name || v === committed.current) return;
    committed.current = v;
    onCommit(v);
  };
  return (
    <input
      className="w-full rounded border border-line bg-panel-2 px-2 py-1 text-sm"
      value={draft ?? name}
      aria-label="Bölge adı"
      disabled={disabled}
      onChange={(e) => {
        committed.current = null;
        setDraft(e.target.value);
      }}
      onBlur={commit}
      onKeyDown={(e) => commitOnEnter(e, commit)}
    />
  );
}

function DwellField({
  frames,
  fps,
  disabled,
  onCommit,
}: {
  frames: number;
  fps: number;
  disabled: boolean;
  onCommit(seconds: number): void;
}) {
  const seconds = fps > 0 ? frames / fps : 0;
  const [draft, setDraft] = useState<string | null>(null);
  const committed = useRef<number | null>(null);
  const commit = () => {
    const v = Number((draft ?? "").replace(",", "."));
    setDraft(null);
    if (!Number.isFinite(v) || v <= 0 || v === seconds || v === committed.current) return;
    committed.current = v;
    onCommit(v);
  };
  return (
    <label className="flex flex-col gap-0.5 text-xs">
      <span className="text-muted">Bekleme (sn)</span>
      <input
        type="number"
        className="num w-24 rounded border border-line bg-panel-2 px-2 py-1 text-sm"
        value={draft ?? seconds.toFixed(1)}
        min={0.1}
        step={0.1}
        disabled={disabled}
        onChange={(e) => {
          committed.current = null;
          setDraft(e.target.value);
        }}
        onBlur={commit}
        onKeyDown={(e) => commitOnEnter(e, commit)}
      />
      <span className="text-muted">{frames} kare (30 kare/sn varsayımı)</span>
    </label>
  );
}

function Counters({ kind, c }: { kind: ZoneKind; c: ZoneCount | undefined }) {
  return (
    <p className="num text-xs">
      {kind === "line" ? (
        <>
          A→B {num(c?.a_to_b)} · B→A {num(c?.b_to_a)}
        </>
      ) : (
        <>
          içeride {num(c?.inside)} (H hareketli, D duran)
          {kind === "dwell" ? ` · aşan ${num(c?.exceeded_total)}` : ""}
          {kind === "area" || kind === "restricted" ? ` · giren ${num(c?.entered_total)}` : ""}
        </>
      )}
    </p>
  );
}

function ZoneRow({
  zone,
  count,
  fps,
  selected,
  onSelect,
  onRename,
  onDwellSeconds,
  onCount,
  onFlip,
  onRemove,
}: {
  zone: ZoneDef;
  count: ZoneCount | undefined;
  fps: number;
  selected: boolean;
  onSelect(id: string | null): void;
  onRename(id: string, name: string): void;
  onDwellSeconds(id: string, s: number): void;
  onCount(id: string, c: LineCount): void;
  onFlip(id: string): void;
  onRemove(id: string): void;
}) {
  const suspended = zone.status === "suspended" || count?.status === "suspended";
  return (
    <li
      id={`zone-${zone.id}`}
      role="option"
      className={`flex flex-col gap-2 rounded border px-2 py-2 ${
        selected ? "border-rgb bg-panel-2" : "border-line"
      } ${suspended ? "opacity-60" : ""}`}
      aria-selected={selected}
      onClick={() => onSelect(zone.id)}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-muted">{text(ZONE_KIND_TEXT, zone.kind)}</span>
        {suspended && <span className="text-xs text-muted">harita bağlantısı yok</span>}
        <button
          type="button"
          className="rounded border border-line px-2 py-0.5 text-xs hover:border-text"
          aria-label={`${zone.name} bölgesini sil`}
          onClick={(e) => {
            // Silme satırı seçmez: seçim silinen bölgeye takılı kalmasın.
            e.stopPropagation();
            onRemove(zone.id);
          }}
        >
          Sil
        </button>
      </div>

      <NameField
        name={zone.name}
        disabled={suspended}
        onCommit={(name) => onRename(zone.id, name)}
      />

      <Counters kind={zone.kind} c={count} />

      {zone.kind === "dwell" && (
        <DwellField
          frames={zone.dwell_frames ?? 0}
          fps={fps}
          disabled={suspended}
          onCommit={(s) => onDwellSeconds(zone.id, s)}
        />
      )}
      {zone.kind === "line" && (
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-0.5 text-xs">
            <span className="text-muted">Sayılan yön</span>
            <select
              className="rounded border border-line bg-panel-2 px-2 py-1 text-sm"
              value={zone.count ?? "both"}
              disabled={suspended}
              onChange={(e) => onCount(zone.id, e.target.value as LineCount)}
            >
              {COUNT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="rounded border border-line px-2 py-1 text-xs hover:border-text"
            disabled={suspended}
            onClick={() => onFlip(zone.id)}
          >
            Yönü çevir
          </button>
        </div>
      )}
    </li>
  );
}

export function ZoneList({
  zones,
  counts,
  fps,
  error,
  selected,
  onSelect,
  onRename,
  onDwellSeconds,
  onCount,
  onFlip,
  onRemove,
  toolbar,
}: {
  zones: ZoneDef[];
  counts: ZoneCounts;
  fps: number;
  error: string | null;
  selected: string | null;
  onSelect(id: string | null): void;
  onRename(id: string, name: string): void;
  onDwellSeconds(id: string, s: number): void;
  onCount(id: string, c: LineCount): void;
  onFlip(id: string): void;
  onRemove(id: string): void;
  toolbar?: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3 rounded border border-line bg-panel p-3">
      <h2 className="text-sm font-semibold">Bölgeler</h2>
      {toolbar}
      {error && (
        <p role="alert" className="text-xs text-alarm">
          Motor bölgeleri kabul etmedi: {error}
        </p>
      )}
      {zones.length === 0 ? (
        <p className="text-sm text-muted">Bölge yok. Haritada bir araç seçip çizin.</p>
      ) : (
        <ul
          className="flex flex-col gap-2"
          role="listbox"
          aria-label="Bölgeler"
          aria-activedescendant={selected ? `zone-${selected}` : undefined}
        >
          {zones.map((z) => (
            <ZoneRow
              key={z.id}
              zone={z}
              count={counts[z.id]}
              fps={fps}
              selected={selected === z.id}
              onSelect={onSelect}
              onRename={onRename}
              onDwellSeconds={onDwellSeconds}
              onCount={onCount}
              onFlip={onFlip}
              onRemove={onRemove}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
