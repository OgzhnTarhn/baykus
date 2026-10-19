"""Faz 3 raporu: results/**/*.json → docs/system.md (tasarım §6, plan Görev 37).

Bütün sayılar JSON'lardan; metin sabit, elle düzenlenmez. Kabul 1-8 kanıt dosyalarından okunur:
dosya yoksa satır "kanıt yok" ve ❌ (eski bir "geçti" varsayılmaz). Eksik isteğe bağlı sayı `—`.
"""

from __future__ import annotations

import json
from pathlib import Path

# Kabul → kanıt dosyası (results/ altında; CI aynı biçimde üretir). Kabul 2 Kabul 1'in dosyasında,
# Kabul 6'nın elle kısmı (Wi-Fi kapalı demo) Kabul 7'nin dosyasında.
EVIDENCE = {
    "k1": "engine/parity_detect.json",
    "k3": "engine/parity_health.json",
    "k4": "engine/parity_track.json",
    "k5": "bench/e3c_onnx_fp32_engine_laptop-docker.json",
    "k6": "system/offline.json",
    "k7": "system/manual.json",
    "k8": "system/api.json",
}
OPTIONAL = {
    "bench_http": "bench/e3c_onnx_fp32_engine_laptop-docker-http.json",
    "dev_detect": "engine/m3ot_dev_detect.json",
    "test_detect": "engine/m3ot_test_detect.json",
    "dev_engine": "track/m3ot_dev_engine.json",
    "dev_oracle": "track/m3ot_dev_oracle.json",
    "test_engine": "track/m3ot_test_engine.json",
    "test_oracle": "track/m3ot_test_oracle.json",
    "gate_val": "health/gate_model_val.json",
    "dev_health": "health/m3ot_dev_health.json",
    "test_health": "health/m3ot_test_health.json",
}
PHASE1_M3OT = "m3ot/*e3c*_m3ot_test_ir.json"  # Faz 1 E7 (PyTorch, agnostic_nms)
MISSING = "kanıt yok"
STAGES = ("decode", "health", "pre", "infer", "post", "track", "total")
RULES = {"phase2": "Faz 2", "frame": "kare (`r_min`)", "frame+temporal": "kare + zaman (ürün)"}


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_results(root: Path) -> dict[str, dict | None]:
    root = Path(root)
    res = {k: _read(root / rel) for k, rel in {**EVIDENCE, **OPTIONAL}.items()}
    phase1 = sorted(root.glob(PHASE1_M3OT))
    res["phase1_m3ot"] = _read(phase1[-1]) if phase1 else None
    return res


def _num(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"%{100 * v:.1f}"


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _get(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _k1(r: dict) -> tuple[bool, str]:
    n = r.get("n_frames", len(r.get("frames", [])))
    mx = max((_get(f, "raw", "max_diff") or 0.0 for f in r.get("frames", [])), default=None)
    return r.get("passed") is True, f"{n} fikstür; ham ORT çıktısında en büyük fark {_num(mx, 6)}"


def _k2(r: dict) -> tuple[bool, str]:
    frames = r.get("frames", [])
    eq = sum(f.get("letterbox_sha_equal") is True for f in frames)
    return bool(frames) and eq == len(frames), f"{eq}/{len(frames)} fikstürde tensör sha256 aynı"


def _k3(r: dict) -> tuple[bool, str]:
    sf = r.get("single_frames", {})
    ok = r.get("passed") is True
    clip = _get(r, "clip", "passed") is True and _get(r, "temporal_unit", "passed") is True
    return ok and clip, (
        f"{sf.get('equal', '—')}/{sf.get('frames', '—')} kare durumlar aynı; zaman kuralı uçtan "
        f"uca "
        f"ve birim {'geçti' if clip else 'DÜŞTÜ'}"
    )


def _k4(r: dict) -> tuple[bool, str]:
    seqs = r.get("per_seq", {})
    parts = [f"{s} {v['compared']}/{v['frames']} kare" for s, v in sorted(seqs.items())]
    ok = r.get("verdict") in ("pass", "pass_with_exception") and bool(seqs)
    return ok, f"karar `{r.get('verdict')}`; " + ", ".join(parts)


def _k5(r: dict) -> tuple[bool, str]:
    k = r.get("kabul5") or {}
    return k.get("passed") is True, (
        f"motor p95 {_num(k.get('total_p95_ms'), 1)} ms ≤ {_num(k.get('threshold_ms'), 0)} ms "
        "(Docker, HTTP hariç)"
    )


def _k6(r: dict, manual: dict | None) -> tuple[bool, str]:
    auto = r.get("passed") is True
    wifi = _get(manual, "wifi_off") is True
    ext = r.get("browser_requests_external")
    return auto and wifi, (
        f"otomatik kanıt 1-4 {'geçti' if auto else 'DÜŞTÜ'} (tarayıcıdan dış istek {ext}); "
        f"Wi-Fi kapalı demo {'kayıtlı' if wifi else 'yok'}"
    )


def _k7(r: dict) -> tuple[bool, str]:
    items = r.get("items", [])
    ok = sum(i.get("ok") is True for i in items)
    return bool(items) and ok == len(items), f"{ok}/{len(items)} ekran maddesi"


def _k8(r: dict) -> tuple[bool, str]:
    cases = r.get("cases", [])
    ok = sum(c.get("ok") is True for c in cases)
    fz = r.get("fuzz", {})
    healthy = fz.get("healthy_after") is True and fz.get("server_errors") == 0
    return bool(cases) and ok == len(cases) and healthy, (
        f"{ok}/{len(cases)} hata durumu; {fz.get('n', '—')} rastgele istekten sonra "
        f"`/health` {'ok' if healthy else 'DÜŞTÜ'}"
    )


ROWS = [
    (1, "tespit eşliği (C++ ↔ Python)", "k1", _k1),
    (2, "letterbox baytları", "k1", _k2),
    (3, "sağlık eşliği (kare + zaman kuralı)", "k3", _k3),
    (4, "takip eşliği", "k4", _k4),
    (5, "gecikme p95 ≤ 100 ms", "k5", _k5),
    (6, "kapalı ağ (otomatik + Wi-Fi kapalı)", "k6", None),
    (7, "ekran (FR-U1-U4, alarmlar, kapı)", "k7", _k7),
    (8, "API sağlamlığı", "k8", _k8),
]


def acceptance(res: dict[str, dict | None]) -> list[dict]:
    rows = []
    for n, crit, key, fn in ROWS:
        rec = res.get(key)
        if rec is None:
            ok, detail = False, MISSING
        elif n == 6:
            ok, detail = _k6(rec, res.get("k7"))
        else:
            ok, detail = fn(rec)
        rows.append(
            {"kabul": n, "criterion": crit, "ok": ok, "detail": detail, "file": EVIDENCE[key]}
        )
    return rows


def exit_ok(rows: list[dict]) -> bool:
    """Tasarım §6.7: Kabul 1-8'in hepsi çıkışı engeller."""
    return all(r["ok"] for r in rows)


def _acceptance_table(rows: list[dict]) -> list[str]:
    out = ["| Kabul | Ölçüt | Sonuç | Ayrıntı | Kanıt |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['kabul']} | {r['criterion']} | {_mark(r['ok'])} | {r['detail']} | "
            f"`results/{r['file']}` |"
        )
    verdict = "**izin veriyor**" if exit_ok(rows) else "**izin vermiyor**"
    return [
        *out,
        "",
        f"Kabul 1-8'in hepsi çıkışı engeller (tasarım §6.7); faz çıkışına {verdict}.",
        "",
    ]


def _latency(res: dict) -> list[str]:
    mem, http = res.get("k5"), res.get("bench_http")
    out = [
        "| Aşama | Bellek içi p50 / p95 (ms) | HTTP kipi p50 / p95 (ms) |",
        "|---|---|---|",
    ]
    for s in STAGES:
        a, b = _get(mem, "stages", s), _get(http, "stages", s)
        out.append(
            f"| {s} | {_num(_get(a, 'p50'), 2)} / {_num(_get(a, 'p95'), 2)} | "
            f"{_num(_get(b, 'p50'), 2)} / {_num(_get(b, 'p95'), 2)} |"
        )
    ht = _get(http, "http_total")
    out.append(
        f"| HTTP dahil gidiş-dönüş | — | {_num(_get(ht, 'p50'), 2)} / {_num(_get(ht, 'p95'), 2)} |"
    )
    sysinfo = _get(mem, "system") or {}
    return [
        *out,
        "",
        f"Bellek içi {_num(_get(mem, 'fps'), 1)} FPS, tepe RSS "
        f"{_num(_get(mem, 'peak_rss_mb'), 0)} MB; "
        f"{sysinfo.get('cpu_model', '—')}, Docker Desktop {sysinfo.get('docker_desktop', '—')}, "
        f"güç {sysinfo.get('power', '—')}. Protokol H5 (50 ısınma + 500 ölçüm, 20 görüntü, "
        "`gate=advise`, takip açık).",
        "",
    ]


def _detect_rows(res: dict) -> list[str]:
    out = [
        "| Bölüm | Dizi | AP50 | ECE ham → kalibre | Çift kutu | Kare |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (("dev_detect", "val (prova)"), ("test_detect", "**test**")):
        r = res.get(key)
        if r is None:
            out.append(f"| {label} | — | — | — | — | — |")
            continue
        for seq, v in sorted((r.get("per_seq") or {}).items()):
            out.append(f"| {label} | {seq} | {_det_cells(v)}")
        out.append(f"| {label} | birleşik | {_det_cells(r)}")
    p1 = _get(res.get("phase1_m3ot"), "metrics", "map50")
    return [
        *out,
        "",
        f"Faz 1 E3c'nin M3OT test AP50'si {_num(p1)} (PyTorch + `agnostic_nms`, E7). Fark tek bir "
        f"nedene "
        "bağlanmaz: ONNX, uçtan uca çıktının 300 kutu sınırı, sınıflar arası çift kutu, NMS "
        "yokluğu "
        "(tasarım §4.3). Kalibrasyon DroneVehicle val'den; M3OT ECE'si alan dışı davranışı "
        "gösterir.",
        "",
    ]


def _det_cells(v: dict) -> str:
    return (
        f"{_num(v.get('ap50'))} | {_num(v.get('ece_raw'))} → {_num(v.get('ece_calibrated'))} | "
        f"{_num(v.get('duplicate_rate'))} | {v.get('n_images', '—')} |"
    )


def _tracking(res: dict) -> list[str]:
    out = ["| Bölüm | Satır | Dizi | HOTA | MOTA | IDF1 |", "|---|---|---|---|---|---|"]
    for part, label in (("dev", "val (prova)"), ("test", "**test**")):
        for row in ("oracle", "engine"):
            r = res.get(f"{part}_{row}")
            name = "oracle" if row == "oracle" else "motor"
            if r is None:
                out.append(f"| {label} | {name} | — | — | — | — |")
                continue
            for seq, m in [
                *sorted((r.get("per_seq") or {}).items()),
                ("birleşik", r.get("combined")),
            ]:
                out.append(
                    f"| {label} | {name} | {seq} | {_num(_get(m, 'HOTA'))} | "
                    f"{_num(_get(m, 'MOTA'))} | {_num(_get(m, 'IDF1'))} |"
                )
    return [
        *out,
        "",
        "Oracle: M3OT'un kendi kutuları skor 1 ile (takipçinin modelden bağımsız ölçümü); motor: "
        "`baykus_detect` tespitleri, takip katmanı ham ≥ 0,1. `trackers` 2.6.1 varsayılan "
        "parametreleri, "
        "IR etiketleri, sınıfsız. Mutlak değer değil iki satırın farkı yorumlanır (ADR 0008).",
        "",
    ]


def _exploration(res: dict) -> list[str]:
    g = res.get("gate_val")
    out = [
        "Etiket: **val'de keşif, PyTorch E3c** (sevk edilen ONNX değil); Faz 3'ün kabul ölçütü "
        "değil, "
        "ürün varsayılanını (`gate=advise`) değiştirmez.",
        "",
    ]
    if g is None:
        return [*out, "_ölçüm dosyası yok._", ""]
    out += [
        "| Koşul | Kapı oranı | E3c mAP50 | E3c + kapı | Fark (puan) [%95 GA] |",
        "|---|---|---|---|---|",
    ]
    for r in g.get("rows", []):
        ci = r.get("ci95") or [None, None]
        ci_s = "—" if None in ci else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
        diff = r.get("diff")
        out.append(
            f"| {r['condition']} | {_pct(r.get('gate_rate'))} | {_num(r.get('base_map50'))} | "
            f"{_num(r.get('gate_map50'))} | {'—' if diff is None else f'{100 * diff:+.1f}'} "
            f"{ci_s} |"
        )
    acc = g.get("acceptance") or {}
    k1, k2, k3 = acc.get("k1") or {}, acc.get("k2") or {}, acc.get("k3") or {}
    ci2 = k2.get("ci95") or [None, None]
    out += [
        "",
        f"Faz 2 Kabul tanımlarıyla: K1 temiz fark {_num(_get(k1, 'value'), 4)} ≥ "
        f"{_num(_get(k1, 'limit'), 3)} {_mark(k1.get('passed') is True)} · K2 bozuk IR kazancı GA "
        f"{'—' if None in ci2 else f'[{ci2[0]:.3f}, {ci2[1]:.3f}]'} "
        f"{_mark(k2.get('passed') is True)} · "
        f"K3 temiz yanlış alarm {_pct(k3.get('value'))} ≤ {_pct(k3.get('limit'))} "
        f"{_mark(k3.get('passed') is True)}.",
        "",
    ]
    catch = g.get("misaligned_catch") or {}
    for cond, v in sorted(catch.items()):
        out.append(
            f"- {cond} yakalama: tümü {_pct(v.get('all'))}, karanlık {_pct(v.get('dark'))}, "
            f"aydınlık {_pct(v.get('bright'))}"
        )
    return [*out, ""]


def _health(res: dict) -> list[str]:
    out = []
    for key, label in (("dev_health", "val (V, demo seçimi)"), ("test_health", "test (T)")):
        h = res.get(key)
        out += [f"**{label}**", ""]
        if h is None:
            out += ["_ölçüm dosyası yok._", ""]
            continue
        out += [
            "| Kural | temiz hizasız | misalign-20 | drop_ir IR kayıp | noise-20 IR gürültülü | "
            "temiz AP50 auto / off |",
            "|---|---|---|---|---|---|",
        ]
        for rule in h.get("rules", []):
            c = _get(h, "combined", rule) or {}

            def rate(cond: str, state: str, c=c) -> str:
                return _pct(_get(c, cond, "alarm_rates", state))

            out.append(
                f"| {RULES.get(rule, rule)} | {rate('clean', 'misaligned')} | "
                f"{rate('misalign-20', 'misaligned')} | {rate('drop_ir', 'ir_lost')} | "
                f"{rate('noise-20', 'ir_noisy')} | {_num(_get(c, 'clean', 'ap50_auto'))} / "
                f"{_num(_get(c, 'clean', 'ap50_off'))} |"
            )
        demo = [
            f"{s['seq']} {'tuttu' if _get(s, 'demo', 'passed') else 'tutmadı'}"
            for s in h.get("sequences", [])
        ]
        out += [
            "",
            f"Demo ölçütü (temiz ≤ %2, misalign-20 ≥ %90, gecikme ≤ 10): {', '.join(demo)}.",
            "",
        ]
    return out


ARCH = [
    "## Mimari",
    "",
    "- **Motor (`engine/`, C++17):** ONNX Runtime 1.30 + OpenCV 5.0; tek model E3c (erken füzyon, "
    "RGB + IR 4 kanal, ONNX FP32). Çözme → sağlık (Faz 2 dedektörü, kare + zaman kuralı) → kapı → "
    "letterbox → çıkarım → kalibre skor → ByteTrack (`trackers` 2.6.1 uyarlaması, oturum başına). "
    "HTTP sunucusu yalnız compose'un iç `back` ağında.",
    "- **Ekran (`ui/`, Next.js 16):** tek sayfa; oynatıcı, RGB / IR / yan yana görünüm, sağlık "
    "paneli, "
    "alarmlar (5 karenin 3'ü), arıza simülasyonu, kapı modu, gecikme grafiği. Tarayıcı yalnız "
    "`127.0.0.1:3000`'e konuşur; motor isteklerini Next sunucusu iletir.",
    "- **Kurulum:** `docker compose up` (iki servis); `compose.offline.yml` bütün ağları "
    "`internal` "
    "yapar. `scripts/prepare.sh --bundle` internetsiz bilgisayara taşınacak paketi hazırlar.",
    "- **Referans:** her C++ adımı Python referansıyla (`python/src/baykus/`) eşlik testli; CI her "
    "commit'te sınar.",
    "",
    "## API",
    "",
    "| Uç | Nerede | İş |",
    "|---|---|---|",
    "| `GET /health` | motor | durum (`ok`) |",
    "| `GET /v1/model` | motor | model ve dosya sha256'ları |",
    "| `POST /v1/detect` | motor | RGB + IR (multipart) → tespitler, sağlık, kapı, izler, süreler "
    "|",
    "| `POST /v1/session/reset` | motor | oturumun takipçisini ve zaman penceresini sıfırlar |",
    "| `GET /api/sequences`, `GET /api/frame/…` | ekran | demo dizileri ve kare görüntüleri |",
    "| `POST /api/detect`, `POST /api/session/reset` | ekran | motora iletir (arızalı kanal "
    "multipart) |",
    "",
    "Hatalar: eksik parça, farklı boyut, bozuk görüntü → 400; sınır üstü görüntü → 413 (Kabul 8).",
    "",
]

LIMITS = [
    "## Sınırlamalar",
    "",
    "- **Doğal kayma:** M3OT'ta RGB ve IR kameralar arasında doğal kayma var ve faz korelasyonu "
    "yanıtı "
    "düşük; ürün kuralı (`r_min` + zaman) temizde yanlış alarm vermiyor ama gerçek 20 px kaymayı "
    "da "
    "yakalamıyor. Sabit bir doğal kayma zaman kuralını geçer; eşik ayarlanmadı.",
    "- **Titreyen arıza:** kayma vektörü sabit kalmayan gerçek arıza (titreyen bağlantı) zaman "
    "kuralını "
    "geçmez; kayıp ve gürültü alarmı 5 karenin 3'ü kuralıyla gecikmeli açılır.",
    "- **Kalibrasyonun alanı:** kalibrasyon DroneVehicle val'den öğrenildi; M3OT'ta skorlar fazla "
    "iyimser (ECE yüksek) ve kalibrasyon az düzeltiyor.",
    "- **Karşılaştırılamaz hız:** H5'teki 60 ms Windows yerel Python ORT ölçümü; motor Docker'da "
    "(WSL2) ve çözme + sağlık dahil. İki sayı yan yana konmaz.",
    "- **Sınıflar:** model DroneVehicle'ın 5 sınıfıyla eğitildi (car, truck, bus, van, "
    "freight_car); "
    "motosiklet yok. Yoğun sahnede (M3OT val, kare başına ≈ 11 araç) tespit belirgin düşük.",
    "- **Az test verisi:** M3OT test iki dizi, ardışık video kareleri; güven aralığı yok.",
    "",
]


def system_markdown(res: dict[str, dict | None]) -> str:
    rows = acceptance(res)
    lines = [
        "# Faz 3 — Sistem: C++ motor, operatör ekranı, kapalı ağ",
        "",
        "> Bu dosya `baykus-system report` ile `results/**/*.json`'dan üretildi; elle düzenlenmez. "
        'Kabul kanıtı yoksa satır "kanıt yok" ile düşer; eksik sayı `—`.',
        "",
        *ARCH,
        "## Kabul 1-8",
        "",
        *_acceptance_table(rows),
        "## Gecikme (Kabul 5)",
        "",
        *_latency(res),
        "## M3OT tespit doğruluğu (motor, sınıfsız `vehicle`, IR etiketleri)",
        "",
        "Test bölümü sistem dondurulduktan sonra bir kez ölçüldü; val aynı komutların provası.",
        "",
        *_detect_rows(res),
        "## Takip (M3OT)",
        "",
        *_tracking(res),
        "## Kare kuralıyla kapı (DroneVehicle val)",
        "",
        *_exploration(res),
        "## M3OT sağlık kontrolü (yalnız raporlanır)",
        "",
        *_health(res),
        *LIMITS,
    ]
    return "\n".join(lines)
