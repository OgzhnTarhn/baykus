"""results/robust/*.json → sağlamlık tabloları ve eğrileri (FR-V4, spec §12.3).

"—" ölçülmedi, "uygulanmaz" bozulma modelin görmediği modaliteye ait.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baykus.report.tables import fmt  # noqa: E402
from baykus.robustness.corruptions import IDENTITY, SUITE, Condition, applies  # noqa: E402

CURVES = ("darken", "noise", "blur", "misalign")
TITLES = {
    "darken": "Karartma (RGB, piksel çarpanı)",
    "noise": "Gauss gürültüsü (IR, σ)",
    "blur": "Bulanıklık (RGB + IR, Gauss çekirdeği)",
    "misalign": "Hizasızlık (IR yatay kaydırma, piksel)",
}
NA = "uygulanmaz"


def load_robust(folder: Path, split: str = "val") -> list[dict]:
    """Sadece istenen bölümün sağlamlık kayıtları (val ile test karışmasın)."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.json"))]
    return [r for r in recs if "robustness" in r and r.get("split") == split]


def model_key(r: dict) -> str:
    return Path(r["config"]).stem


def _index(recs: list[dict]) -> dict[tuple[str, str], dict]:
    """(config adı, koşul) → kayıt. Aynı config'in iki eğitimi aynı hücreye düşerse hata:
    hangisinin gösterildiği belirsiz kalmasın."""
    idx: dict[tuple[str, str], dict] = {}
    for r in recs:
        key = (model_key(r), r["robustness"]["id"])
        if key in idx:
            raise ValueError(
                f"{key}: iki kayıt var ({idx[key]['run_id']}, {r['run_id']}); "
                "eski çalıştırmanın dosyalarını results/robust'tan çıkar"
            )
        idx[key] = r
    return idx


def _cell(r: dict | None) -> str:
    if r is None:
        return "—"
    m = r["metrics"]
    return fmt(m.get("map50"), (m.get("ci95") or {}).get("map50"))


def _cond_id(corruption: str, sev: float) -> str:
    if sev == IDENTITY.get(corruption):
        return "clean"
    return Condition(corruption, float(sev), SUITE[corruption][0]).id


def _applies_any(corruption: str, mode: str) -> bool:
    target, sevs = SUITE[corruption]
    return applies(Condition(corruption, float(sevs[-1]), target), mode)


def robust_markdown(recs: list[dict], fig_rel: str) -> str:
    idx = _index(recs)
    models = sorted({model_key(r) for r in recs})
    modes = {model_key(r): r["model"]["mode"] for r in recs}
    head = "| Şiddet | " + " | ".join(models) + " |\n|---|" + "---|" * len(models)
    out = [
        f"# Sağlamlık paketi ({recs[0]['split']}, IR etiketleri)",
        "",
        "> Bu dosya `baykus-robust report` ile `results/robust/*.json`'dan üretildi; elle "
        "düzenlenmez.",
        "> Hücre: mAP50 [görüntü düzeyinde bootstrap %95 GA]. `—` ölçülmedi; `uygulanmaz` "
        "bozulma modelin görmediği modaliteye ait ya da tek modaliteli modelde hizasızlık.",
        "> Karartma piksel değerini çarpanla ölçekler (gamma eğrisi değil). Hizasızlıkta IR "
        "kaydırılır, etiketler yerinde kalır.",
        "",
    ]
    for c in CURVES:
        out += [f"## {TITLES[c]}", "", head]
        for s in SUITE[c][1]:
            label = f"{s:g} (temiz)" if s == IDENTITY[c] else f"{s:g}"
            cells = [
                _cell(idx.get((mk, _cond_id(c, s)))) if _applies_any(c, modes[mk]) else NA
                for mk in models
            ]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Kamera kaybı", "", head.replace("Şiddet", "Koşul", 1)]
    for label, cid, corr in (("Temiz", "clean", None), ("RGB kaybı", "drop_rgb", "drop_rgb"),
                             ("IR kaybı", "drop_ir", "drop_ir")):  # fmt: skip
        cells = [
            _cell(idx.get((mk, cid))) if corr is None or _applies_any(corr, modes[mk]) else NA
            for mk in models
        ]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["", f"![Sağlamlık eğrileri]({fig_rel})", ""]
    return "\n".join(out)


def robust_figure(recs: list[dict]):
    """Dört bozulma paneli; her model bütün panellerde aynı renkte."""
    idx = _index(recs)
    models = sorted({model_key(r) for r in recs})
    modes = {model_key(r): r["model"]["mode"] for r in recs}
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = {mk: palette[i % len(palette)] for i, mk in enumerate(models)}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, c in zip(axes.flat, CURVES, strict=True):
        for mk in models:
            if not _applies_any(c, modes[mk]):
                continue
            xs, ys, lo, hi = [], [], [], []
            for s in SUITE[c][1]:
                r = idx.get((mk, _cond_id(c, s)))
                if r is None or r["metrics"].get("map50") is None:
                    continue
                v = 100 * r["metrics"]["map50"]
                ci = (r["metrics"].get("ci95") or {}).get("map50") or [None, None]
                xs.append(s)
                ys.append(v)
                lo.append(v if ci[0] is None else 100 * ci[0])
                hi.append(v if ci[1] is None else 100 * ci[1])
            if xs:
                ax.plot(xs, ys, marker="o", label=mk, color=color[mk])
                ax.fill_between(xs, lo, hi, alpha=0.15, color=color[mk])
        ax.set_title(TITLES[c], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("mAP50 (%)")
        if c == "darken":
            ax.invert_xaxis()  # sağa doğru şiddet artar
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def robust_plot(recs: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = robust_figure(recs)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
