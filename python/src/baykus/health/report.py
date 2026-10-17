"""Faz 2 raporu: results/health/*.json → docs/health.md + güvenilirlik diyagramı.

Bütün sayılar JSON'lardan; metin sabit. Biçim mevcut raporlarla aynı (puan, ondalık nokta).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baykus.report.tables import fmt  # noqa: E402

PENDING = "_henüz ölçülmedi._"
STATE_LABELS = {
    "rgb_lost": "RGB kayıp",
    "ir_lost": "IR kayıp",
    "ir_noisy": "IR gürültülü",
    "misaligned": "IR hizasız",
    "rgb_blurry": "RGB bulanık (bilgi)",
    "ir_blurry": "IR bulanık (bilgi)",
}


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}"


def _diff(d: float | None, ci: list | None) -> str:
    if d is None:
        return "—"
    s = f"{100 * d:+.1f}"
    if ci and None not in ci:
        s += f" {_ci(ci)}"
    return s


def _ci(ci: list | None) -> str:
    if not ci or None in ci:
        return "—"
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def _mark(ok: bool | None) -> str:
    return "—" if ok is None else ("✅" if ok else "❌")


def _thresholds(th: dict) -> list[str]:
    out = [
        "## Eşikler (train)",
        "",
        f"Kaynak: {th['source']['split']} bölümünden {th['source']['n']} görüntü "
        f"(seed {th['source']['seed']}), sağlamlık paketinin bütün koşulları; Youden J. "
        "TPR / FPR: train'de, %.",
        "",
        "| Durum | Kod | Özellik | Yön | Eşik | TPR | FPR |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in th["states"].items():
        arrow = "≥" if s["direction"] == "above" else "≤"
        t = s["train"]
        out.append(
            f"| {STATE_LABELS.get(name, name)} | {name} | {s['feature']} | {arrow} | "
            f"{s['threshold']:.3f} | {_pct(t['tpr'])} | {_pct(t['fpr'])} |"
        )
    out += [
        "",
        f"`karanlık` (RGB, bilgi): parlaklık < {th['dark_brightness']:.2f} (ADR 0005 t_dark). "
        f"Train temiz karelerde kapı oranı: {_pct(th['train_clean_gate_rate'])}%.",
        "",
    ]
    return out


def _detection(gate: dict) -> list[str]:
    out = [
        "## Arıza tespiti (val)",
        "",
        "Görüntü oranı, %. Zararlı şiddetler: IR gürültüsü σ 20, hizasızlık 20 / 40 px, "
        "kamera kaybı. Karanlık ve bulanık yalnız bilgi.",
        "",
        "| Koşul | Kapı | IR gürültülü | IR hizasız | IR kayıp | RGB kayıp | RGB karanlık "
        "| RGB bulanık | IR bulanık |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, d in gate["detection"].items():
        out.append(
            f"| {cid} | {_pct(d['gate_rate'])} | {_pct(d['ir']['noisy'])} | "
            f"{_pct(d['misaligned'])} | {_pct(d['ir']['lost'])} | {_pct(d['rgb']['lost'])} | "
            f"{_pct(d['rgb']['dark'])} | {_pct(d['rgb']['blurry'])} | {_pct(d['ir']['blurry'])} |"
        )
    return [*out, ""]


def _gate_table(gate: dict) -> list[str]:
    out = [
        "| Koşul | Kapı % | E3c | E3c + kapı | Δ puan [%95 GA] |",
        "|---|---|---|---|---|",
    ]
    for r in gate["rows"]:
        out.append(
            f"| {r['condition']} | {_pct(r['gate_rate'])} | {fmt(r['base_map50'])} | "
            f"{fmt(r['gate_map50'])} | {_diff(r['diff'], r['ci95'])} |"
        )
    h = gate.get("harmful")
    if h:
        out.append(
            f"| **Z ortalaması** ({', '.join(h['conditions'])}) | | | | "
            f"**{_diff(h['diff'], h['ci95'])}** |"
        )
    return [*out, ""]


def _acceptance(gate_val: dict | None, calib: dict | None) -> list[str]:
    acc = (gate_val or {}).get("acceptance") or {}
    k1, k2, k3 = acc.get("k1"), acc.get("k2"), acc.get("k3")
    k4 = (calib or {}).get("k4")

    def pct(k: dict) -> str:
        return f"{100 * k['value']:.2f}%"

    def points(k: dict) -> str:
        return f"{100 * k['value']:.3f}"

    items = [
        ("Kabul 1", "temiz val'de fark ≥ −0,5 puan", k1, lambda k: f"{100 * k['value']:+.2f}"),
        ("Kabul 2", "Z birleşik %95 GA tamamen 0'ın üstünde", k2, lambda k: _ci(k["ci95"])),
        ("Kabul 3", "temiz val'de kapı alarmı ≤ %2 görüntü", k3, pct),
        ("Kabul 4", "kalibrasyonla \\|ΔmAP50\\| ≤ 0,1 puan (test)", k4, points),
    ]
    lines = [
        "## Kabul ölçütleri",
        "",
        "Ölçümden önce sabit (`docs/design/faz2.md` §6).",
        "",
        "| # | Ölçüt | Değer | Sonuç |",
        "|---|---|---|---|",
    ]
    for name, rule, k, show in items:
        lines.append(f"| {name} | {rule} | {show(k) if k else '—'} | {_mark(k and k['passed'])} |")
    return [*lines, ""]


def _calibration(calib: dict | None, fig_rel: str) -> list[str]:
    out = ["## Kalibrasyon (test)", ""]
    if not calib:
        return [*out, PENDING, ""]
    t = calib["test"]
    return [
        *out,
        "E3c · ONNX FP32 (sevk edilen); tablo val'de izotonik regresyonla, ECE testte ham "
        f"skoru ≥ 0,25 olan {t['n_eval']} tespitte, 15 eşit aralık.",
        "",
        f"ECE {t['ece_before']:.3f} → {t['ece_after']:.3f}.",
        "",
        "| | Önce | Sonra |",
        "|---|---|---|",
        f"| ECE | {t['ece_before']:.3f} | {t['ece_after']:.3f} |",
        f"| mAP50 | {fmt(t['map50_before'])} | {fmt(t['map50_after'])} |",
        "",
        f"![Güvenilirlik diyagramı]({fig_rel})",
        "",
    ]


def health_markdown(
    thresholds: dict,
    gate_val: dict | None,
    gate_test: dict | None,
    calib: dict | None,
    fig_rel: str,
) -> str:
    boot = gate_val or gate_test or {}
    lines = [
        "# Faz 2 — Sensör sağlığı kapısı ve güven kalibrasyonu",
        "",
        "> Bu dosya `baykus-health report` ile `results/health/*.json`'dan üretildi; elle "
        "düzenlenmez. Hepsi IR etiketleri; GA = görüntü düzeyinde eşleştirilmiş bootstrap "
        f"({boot.get('n_boot', '—')}, seed {boot.get('seed', '—')}). mAP50 ve farklar puan.",
        "",
        "## Yöntem",
        "",
        "- Her karede RGB ve IR için kural tabanlı ölçüler: piksel std (kayıp), Immerkær gürültü "
        "kestirimi, Laplace varyansı (bulanık), parlaklık (karanlık, ADR 0005), kenar haritaları "
        "arasında faz korelasyonu kayması (hizasız). Eğitim yok.",
        "- Kapı: modele zarar veren kamera ön işlemeden önce sıfırlanır (E3c değişmez). "
        "Öncelik kayıp > gürültülü > hizasız; iki kanal birden kapatılmaz; RGB'de yalnız kayıp "
        "kapıya girer.",
        "- Kapı bir koşulun hiçbir pikselini değiştirmediyse kapılı tahminler Faz 1 tahminleriyle "
        "aynı (fark tam 0).",
        "",
        *_thresholds(thresholds),
    ]
    if gate_val:
        lines += [*_detection(gate_val), "## Kapı (val)", "", *_gate_table(gate_val)]
    else:
        lines += ["## Kapı (val)", "", PENDING, ""]
    lines += _acceptance(gate_val, calib)
    lines += ["## Test kontrolü", ""]
    if gate_test:
        lines += [
            "Tek sefer, 3 koşul; seçim yapılmadı (tasarım §6.4).",
            "",
            *_gate_table(gate_test),
        ]
    else:
        lines += [PENDING, ""]
    lines += _calibration(calib, fig_rel)
    lines += [
        "## Sınırlamalar",
        "",
        "- Bozulmalar sentetik (sağlamlık paketi); gerçek kamera arızasında dedektör ayrıca "
        "doğrulanmadı (Faz 3'te M3OT dizilerinde niteliksel kontrol).",
        "- Zarar düzeyleri Faz 1'in val ölçümlerinden, eşikler train'den; val değerlendirmesi "
        "hafif iyimser olabilir. Test kontrolü bunun için.",
        "",
    ]
    return "\n".join(lines)


def calibration_plot(calib: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="ideal")
    t = calib["test"]
    for when, label, marker in (("before", "önce", "o"), ("after", "sonra", "s")):
        pts = [(b["conf"], b["acc"]) for b in t[f"bins_{when}"] if b["n"]]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker=marker, label=f"{label} (ECE {t[f'ece_{when}']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("güven (skor)")
    ax.set_ylabel("doğru tespit oranı")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
