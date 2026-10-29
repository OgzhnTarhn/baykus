"""Faz 4 raporu: results/**/*.json → docs/motion.md (tasarım faz4 §6.6, plan Görev 23).

Kabul satırları kanıt dosyalarından okunur: dosya yoksa satır "kanıt yok" ve ❌ (eski bir "geçti"
varsayılmaz). Hareket ve ürün iddiası §6.6 kuralıyla ayrı hesaplanır; eksik isteğe bağlı sayı `—`.
Açıklama metni sabit, sayılar JSON'lardan; dosya elle düzenlenmez.
"""

from __future__ import annotations

import json
from pathlib import Path

# Kabul → kanıt dosyası (results/ altında; CI aynı biçimde üretir). Kabul 6'nın elle kısmı
# `system/manual.json` Faz 4 bölümünde, otomatik kısmı `system/offline.json`'da.
EVIDENCE = {
    "k1": "engine/parity_ego.json",
    "k4": "motion/label_audit_val.json",
    "k5": "bench/e3c_onnx_fp32_engine_laptop-docker_f4.json",
    "k6": "system/manual.json",
    "k6b": "system/offline.json",
    "k9": "motion/m3ot_dev_oracle.json",
    "ci": "system/ci_f4.json",
}
OPTIONAL = {
    "dev_oracle": "motion/m3ot_dev_oracle.json",
    "dev_engine": "motion/m3ot_dev_engine.json",
    "test_oracle": "motion/m3ot_test_oracle.json",
    "test_engine": "motion/m3ot_test_engine.json",
    "dev_ego": "motion/m3ot_dev_ego.json",
    "test_ego": "motion/m3ot_test_ego.json",
    "test_audit": "motion/label_audit_test.json",
    "bench_noego": "bench/e3c_onnx_fp32_engine_laptop-docker_f4_noego.json",
    "bench_irdrop": "bench/e3c_onnx_fp32_engine_laptop-docker_f4_irdrop.json",
    "bench_seek": "bench/e3c_onnx_fp32_engine_laptop-docker_f4_seek.json",
}
MISSING = "kanıt yok"
# İddia kuralı eşikleri (tasarım §6.6; ölçümden önce sabit).
MOTION_BA_ORACLE, MOTION_BA_ENGINE, MOTION_P_MOV = 0.85, 0.70, 0.80
PRODUCT_IDS = ("Kabul 6", "Kabul 7", "Kabul 10 (a, c)", "Kabul 10 b", "Kabul 11")


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_results(root: Path) -> dict[str, dict | None]:
    root = Path(root)
    return {k: _read(root / rel) for k, rel in {**EVIDENCE, **OPTIONAL}.items()}


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


# --- kabul satırları --------------------------------------------------------------------------


def _k1(rec: dict, res: dict) -> tuple[bool, str]:
    sc = rec.get("per_scenario") or {}
    mx = max((v.get("max_corner_px") or 0.0 for v in sc.values()), default=None)
    return rec.get("kabul1") == "pass", (f"{len(sc)} senaryo; en büyük köşe farkı {_num(mx, 6)} px")


def _ci(rec: dict, res: dict) -> tuple[bool, str]:
    ok = rec.get("conclusion") == "success"
    return ok, f"CI koşusu `{rec.get('run_id', '—')}` `{rec.get('conclusion', '—')}`"


def _k4(rec: dict, res: dict) -> tuple[bool, str]:
    a = rec.get("a_agreement") or {}
    return rec.get("verdict") == "pass", (
        f"karar `{rec.get('verdict')}`; (a) uyum {a.get('k', '—')}/{a.get('n', '—')}"
    )


def _k5(rec: dict, res: dict) -> tuple[bool, str]:
    k = rec.get("kabul5") or {}
    return k.get("passed") is True, (
        f"motor p95 {_num(k.get('total_p95_ms'), 1)} ms ≤ {_num(k.get('threshold_ms'), 0)} ms "
        "(Docker, HTTP hariç)"
    )


def _k6(rec: dict, res: dict) -> tuple[bool, str]:
    f4 = rec.get("faz4") or {}
    items = f4.get("items") or []
    ok_items = sum(i.get("ok") is True for i in items)
    wifi = f4.get("wifi_off") is True
    auto = _get(res.get("k6b"), "passed") is True
    ok = bool(items) and ok_items == len(items) and wifi and auto
    return ok, (
        f"{ok_items}/{len(items)} ekran maddesi; Wi-Fi kapalı demo "
        f"{'kayıtlı' if wifi else 'yok'}; kapalı ağ otomatik kanıtı "
        f"{'geçti' if auto else 'DÜŞTÜ'}"
    )


def _k9(rec: dict, res: dict) -> tuple[bool, str]:
    g = rec.get("gate_k9") or {}
    return g.get("pass") is True, (
        f"BA {_num(g.get('BA'))} ≥ 0,85; R_mov {_num(g.get('R_mov'))}, "
        f"R_sta {_num(g.get('R_sta'))} ≥ 0,80"
    )


def _k10b(rec: dict, res: dict) -> tuple[bool, str]:
    k = rec.get("kabul10b") or {}
    # Ölçülen değer de yazılır (yalnız sınır değil): senaryo başına C++ köşe farkı.
    worst = ", ".join(
        f"{r.get('scenario')} {_num(r.get('corner_px'), 2)} px" for r in k.get("cpp") or []
    )
    return k.get("pass") is True, f"köşe farkı sınırı 2,0 px; ölçülen: {worst or '—'}"


ROWS = [
    ("Kabul 1", "uçtan uca eşlik (klip senaryoları)", ("k1",), _k1),
    ("Kabul 2", "hareket birim eşliği", ("ci",), _ci),
    ("Kabul 3", "sentetik doğruluk", ("ci",), _ci),
    ("Kabul 4", "kör insan örneklemi (val)", ("k4",), _k4),
    ("Kabul 5", "gecikme p95 ≤ 100 ms", ("k5",), _k5),
    ("Kabul 6", "ekran (elle, demo dizisi)", ("k6", "k6b"), _k6),
    ("Kabul 7", "bölgeler ve olaylar (CI)", ("ci",), _ci),
    ("Kabul 8", "gerileme ve API sağlamlığı (CI)", ("ci",), _ci),
    ("Kabul 9", "oracle işlev kapısı (val)", ("k9",), _k9),
    ("Kabul 10 (a, c)", "yeniden konumlama doğruluğu (CI)", ("ci",), _ci),
    ("Kabul 10 b", "yeniden konumlanan dönüşüm (seek, long_gap)", ("k1",), _k10b),
    ("Kabul 11", "araç özeti ve günlük (CI)", ("ci",), _ci),
]


def acceptance(res: dict[str, dict | None]) -> list[dict]:
    rows = []
    for rid, crit, keys, fn in ROWS:
        rec = res.get(keys[0])
        files = ", ".join(f"`results/{EVIDENCE[k]}`" for k in keys)
        if rec is None:
            ok, detail = False, MISSING
        else:
            ok, detail = fn(rec, res)
        known = None
        if rid == "Kabul 10 b" and not ok:
            path = _get(rec, "known_fail", "kabul10b")
            if path and _get(rec, "known_fail", "still_failing") is True:
                known = path
        rows.append(
            {"id": rid, "ok": ok, "text": f"{crit}: {detail} ({files})", "known_fail": known}
        )
    return rows


def exit_ok(rows: list[dict]) -> bool:
    """Tasarım §6.6: Kabul 1-11 çıkışı engeller; Kabul 10 b ADR ile kapanabilir."""
    return all(r["ok"] or r["known_fail"] for r in rows)


# --- iddia kuralı -----------------------------------------------------------------------------


def _motion_claim(res: dict[str, dict | None]) -> dict:
    reasons = []
    oracle = _get(res.get("test_oracle"), "labeled") or {}
    engine = _get(res.get("test_engine"), "labeled") or {}
    p_mov = _get(res.get("test_engine"), "precision", "moving", "P")
    audit = res.get("test_audit")
    ba_o, ba_e = oracle.get("BA"), engine.get("BA")
    if ba_o is None or ba_o < MOTION_BA_ORACLE:
        reasons.append(f"test oracle BA {_num(ba_o)} < {MOTION_BA_ORACLE:.2f}")
    if ba_e is None or ba_e < MOTION_BA_ENGINE:
        reasons.append(f"test motor BA {_num(ba_e)} < {MOTION_BA_ENGINE:.2f}")
    if p_mov is None or p_mov < MOTION_P_MOV:
        reasons.append(f"test motor P_mov {_num(p_mov)} < {MOTION_P_MOV:.2f}")
    if audit is None:
        reasons.append("test audit yok")
    elif _get(audit, "a_agreement", "pass") is not True:
        reasons.append("test audit (a) tutmadı")
    return {"ok": not reasons, "reasons": reasons}


def _product_claim(rows: list[dict]) -> dict:
    reasons = [
        f"{r['id']} düştü"
        for r in rows
        if r["id"] in PRODUCT_IDS and not r["ok"] and not r["known_fail"]
    ]
    ok = all(r["ok"] or r["known_fail"] for r in rows if r["id"] in PRODUCT_IDS)
    return {"ok": ok, "reasons": reasons}


def claims(res: dict[str, dict | None], rows: list[dict]) -> dict:
    """Tasarım §6.6: hareket iddiası test ölçümlerine, ürün iddiası Kabul 6, 7, 10, 11'e bağlı."""
    return {"motion": _motion_claim(res), "product": _product_claim(rows)}


# --- markdown ---------------------------------------------------------------------------------


def _acceptance_table(rows: list[dict]) -> list[str]:
    out = ["| Kabul | Durum | Kanıt |", "|---|---|---|"]
    for r in rows:
        mark = f"❌ (ADR: {r['known_fail']})" if r["known_fail"] else _mark(r["ok"])
        out.append(f"| {r['id']} | {mark} | {r['text']} |")
    verdict = "**izin veriyor**" if exit_ok(rows) else "**izin vermiyor**"
    return [
        *out,
        "",
        f"Kabul 1-11 çıkışı engeller (tasarım §6.6); Kabul 10 b ADR ile kapanabilir. "
        f"Faz çıkışına {verdict}.",
        "",
    ]


def _claim_rule(c: dict) -> list[str]:
    m, p = c["motion"], c["product"]
    out = ["## İddia kuralı", ""]
    out.append(f'Hareket iddiası ("hareketli / duran ayrımı çalışıyor") {_mark(m["ok"])}.')
    if m["ok"]:
        out.append(
            "Test oracle BA ≥ 0,85, test motor BA ≥ 0,70 ve P_mov ≥ 0,80, test audit (a) tuttu."
        )
    else:
        out.append("Tutmayan koşullar: " + "; ".join(m["reasons"]) + ".")
        out.append(
            'README "hareketli / duran ayrımı çalışıyor" cümlesini kullanmaz; sayılar olduğu '
            "gibi yazılır."
        )
    out.append("")
    out.append(f"Ürün iddiası (bölgeler, alarmlar, harita belleği, dışa aktarma) {_mark(p['ok'])}.")
    if p["ok"]:
        out.append(
            "Kabul 6, 7, 10 ve 11 tuttu; olayların gerçek dünyadaki doğruluğu iddia edilmez."
        )
    else:
        out.append("Tutmayan koşullar: " + "; ".join(p["reasons"]) + ".")
    return [*out, ""]


def _motion_table(res: dict[str, dict | None], split: str) -> list[str]:
    out = [
        "| Satır | Dizi | n | hareketli | duran | R_mov | R_sta | BA | kapsam | P_mov | P_sta |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    zero = False
    for row, name in (("oracle", "oracle"), ("engine", "motor")):
        r = res.get(f"{split}_{row}")
        if r is None:
            out.append(f"| {name} | — | — | — | — | — | — | — | — | — | — |")
            continue
        prec = r.get("precision") or {}
        seqs = [*sorted((r.get("per_seq") or {}).items()), ("birleşik", r.get("labeled"))]
        for seq, m in seqs:
            m = m or {}
            zero = zero or m.get("n_stationary") == 0
            p_mov = _get(prec, "moving", "P") if seq == "birleşik" else None
            p_sta = _get(prec, "stationary", "P") if seq == "birleşik" else None
            out.append(
                f"| {name} | {seq} | {m.get('n', '—')} | {m.get('n_moving', '—')} | "
                f"{m.get('n_stationary', '—')} | {_num(m.get('R_mov'))} | "
                f"{_num(m.get('R_sta'))} | {_num(m.get('BA'))} | {_num(m.get('coverage'))} | "
                f"{_num(p_mov)} | {_num(p_sta)} |"
            )
    if zero:
        out += ["", "Bu bölümde duran araç yok: R_sta ve BA tanımsız."]
    return [*out, ""]


def _human(res: dict[str, dict | None]) -> list[str]:
    out = []
    for key, label in (("k4", "val"), ("test_audit", "test")):
        a = res.get(key)
        if a is None:
            out += [f"**{label}**: _kanıt yok._", ""]
            continue
        ag = a.get("a_agreement") or {}
        out.append(
            f"**{label}** karar `{a.get('verdict', '—')}`; (a) uyum "
            f"{ag.get('k', '—')}/{ag.get('n', '—')} ({_num(ag.get('rate'))}) "
            f"{_mark(ag.get('pass') is True)}."
        )
    out += ["", "| Bölüm | Satır | İnsanla uyum | Oran |", "|---|---|---|---|"]
    for split, label in (("dev", "val"), ("test", "test")):
        for row, name in (("oracle", "oracle"), ("engine", "motor")):
            h = _get(res.get(f"{split}_{row}"), "vs_human")
            if h is None:
                out.append(f"| {label} | {name} | — | — |")
            else:
                out.append(
                    f"| {label} | {name} | {h.get('agree', '—')}/{h.get('n', '—')} | "
                    f"{_num(h.get('rate'))} |"
                )
    return [*out, ""]


def _drift(d: dict | None, k: str) -> str:
    return f"{_num(_get(d, k, 'median_px'), 2)} / {_num(_get(d, k, 'p95_px'), 2)}"


def _map(res: dict[str, dict | None]) -> list[str]:
    out = [
        "| Bölüm | Dizi | ego ok | kayma 15 ortanca / p95 (px) | kayma 60 ortanca / p95 (px) | "
        "yeniden konumlama |",
        "|---|---|---|---|---|---|",
    ]
    for split, label in (("dev", "val"), ("test", "test")):
        e = res.get(f"{split}_ego")
        if e is None:
            out.append(f"| {label} | — | — | — | — | — |")
            continue
        for seq, v in sorted((e.get("per_seq") or {}).items()):
            d = v.get("map_drift") or {}
            out.append(
                f"| {label} | {seq} | {_num(_get(v, 'ego', 'ok_rate'))} | {_drift(d, '15')} | "
                f"{_drift(d, '60')} | {_num(_get(v, 'reloc', 'success_rate'))} |"
            )
    return [*out, ""]


def _latency(res: dict[str, dict | None]) -> list[str]:
    b = res.get("k5")
    k = _get(b, "kabul5") or {}
    out = [
        f"Faz 4 bench: p50 {_num(_get(b, 'stages', 'total', 'p50'), 2)} ms, p95 "
        f"{_num(k.get('total_p95_ms'), 2)} ms ≤ {_num(k.get('threshold_ms'), 0)} ms "
        f"{_mark(k.get('passed') is True)} (Kabul 5).",
        "",
        "| Varyant | p50 (ms) | p95 (ms) |",
        "|---|---|---|",
        f"| f4 (tam) | {_num(_get(b, 'stages', 'total', 'p50'), 2)} | "
        f"{_num(k.get('total_p95_ms'), 2)} |",
    ]
    for key, label in (
        ("bench_noego", "ego kapalı"),
        ("bench_irdrop", "IR kaybı"),
        ("bench_seek", "atlama"),
    ):
        v = res.get(key)
        out.append(
            f"| {label} | {_num(_get(v, 'stages', 'total', 'p50'), 2)} | "
            f"{_num(_get(v, 'kabul5', 'total_p95_ms'), 2)} |"
        )
    return [*out, ""]


LIMITS = [
    "## Sınırlamalar",
    "",
    "- **Az veri:** yalnız iki val ve iki test dizisi; güven aralığı yok, genelleme güvencesi "
    "değil.",
    "- **Kare hızı:** 30 kare/sn varsayımı; hız ve bekleme süreleri bu varsayıma bağlı.",
    "- **Harita göreli:** harita koordinatı ürünün kendi kestirimi; mutlak konum iddia edilmez.",
    "- **Olay doğruluğu:** olayların gerçek dünyadaki doğruluğu ölçülmedi.",
    "- **Kabul 10 b (ADR 0011):** uzun boşluktan sonra harita birkaç piksel kayabilir.",
    "",
]


def motion_markdown(res: dict[str, dict | None]) -> str:
    rows = acceptance(res)
    lines = [
        "# Faz 4: hareket ve ürün davranışı",
        "",
        "> Bu dosya `baykus-motion report` ile `results/**/*.json`'dan üretildi; elle düzenlenmez. "
        'Kabul kanıtı yoksa satır "kanıt yok" ile düşer; eksik sayı `—`.',
        "",
        "## Kabul",
        "",
        *_acceptance_table(rows),
        *_claim_rule(claims(res, rows)),
        "## Hareket: val",
        "",
        *_motion_table(res, "dev"),
        "## Hareket: test",
        "",
        *_motion_table(res, "test"),
        "## İnsan örneklemi",
        "",
        *_human(res),
        "## Harita kayması ve yeniden konumlama",
        "",
        *_map(res),
        "## Gecikme",
        "",
        *_latency(res),
        *LIMITS,
    ]
    return "\n".join(lines)
