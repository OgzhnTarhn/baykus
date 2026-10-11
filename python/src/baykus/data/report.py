"""E0 veri raporu: sayımlar, iki etiket seti farkı, aydınlatma ve boyut dağılımı (FR-D7)."""

from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baykus.data.layout import SPLITS  # noqa: E402


def _centers(coco: dict) -> dict[str, list[tuple[int, float, float]]]:
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    out: dict[str, list[tuple[int, float, float]]] = collections.defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[names[a["image_id"]]].append((a["category_id"], x + w / 2, y + h / 2))
    return out


def label_shift(coco_ir: dict, coco_rgb: dict, max_dist: float = 40.0) -> dict:
    """Aynı nesnenin IR ve RGB etiketlerindeki merkez kayması (piksel)."""
    ir, rgb = _centers(coco_ir), _centers(coco_rgb)
    offsets: list[float] = []
    un_ir = un_rgb = 0
    for name in sorted(set(ir) | set(rgb)):
        a, b = ir.get(name, []), rgb.get(name, [])
        pairs = sorted(
            (d, i, j)
            for i, p in enumerate(a)
            for j, q in enumerate(b)
            if p[0] == q[0] and (d := math.dist(p[1:], q[1:])) <= max_dist
        )
        used_a: set[int] = set()
        used_b: set[int] = set()
        for d, i, j in pairs:
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                offsets.append(d)
        un_ir += len(a) - len(used_a)
        un_rgb += len(b) - len(used_b)
    return {
        "matched": len(offsets),
        "unmatched_ir": un_ir,
        "unmatched_rgb": un_rgb,
        "offsets": offsets,
    }


def summarize(coco: dict) -> dict:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_img = collections.Counter(a["image_id"] for a in coco["annotations"])
    per_cls = collections.Counter(names[a["category_id"]] for a in coco["annotations"])
    size = {"small": 0, "medium": 0, "large": 0}
    for a in coco["annotations"]:
        size["small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large"] += 1
    return {
        "images": len(coco["images"]),
        "boxes": len(coco["annotations"]),
        "per_class": {n: per_cls.get(n, 0) for n in names.values()},
        "max_per_image": max(per_img.values(), default=0),
        "empty_images": len(coco["images"]) - len(per_img),
        "size": size,
    }


def _table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_report(out: Path, meta: Path, fig_dir: Path) -> str:
    fig_dir.mkdir(parents=True, exist_ok=True)
    s = {
        (sp, ls): summarize(json.loads((out / "coco" / f"{sp}_{ls}.json").read_text("utf-8")))
        for sp in SPLITS
        for ls in ("ir", "rgb")
    }
    verify = json.loads((meta / "verify_report.json").read_text("utf-8"))
    conv = json.loads((meta / "convert_stats.json").read_text("utf-8"))
    illum = collections.Counter()
    with (meta / "illumination.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            illum[(row["split"], row["illumination"])] += 1

    md = [
        "# E0 — DroneVehicle veri raporu",
        "",
        "Bu rapor `baykus-data report` ile üretildi; elle düzenlenmez.",
        "",
        "## Bütünlük",
        "",
        f"- Geçerli çiftler: {verify['ok_counts']}",
        f"- Dışlananlar (nedene göre): {verify['issues_by_reason']}",
        "",
        "## Etiket dönüşümü",
        "",
        "```json",
        json.dumps(conv, indent=1),
        "```",
        "",
        "## Görüntü ve kutu sayıları (IR vs RGB etiket seti)",
        "",
    ]
    rows = []
    for sp in SPLITS:
        a, b = s[(sp, "ir")], s[(sp, "rgb")]
        diff = (b["boxes"] - a["boxes"]) / a["boxes"] * 100 if a["boxes"] else 0
        rows.append(
            [
                sp,
                a["images"],
                a["boxes"],
                b["boxes"],
                f"{diff:+.1f}%",
                a["max_per_image"],
                a["empty_images"],
            ]
        )
    md += [
        _table(
            rows,
            [
                "split",
                "görüntü",
                "kutu (IR)",
                "kutu (RGB)",
                "RGB-IR farkı",
                "maks kutu/görüntü (IR)",
                "boş görüntü",
            ],
        ),
        "",
    ]
    md += ["## Sınıf dağılımı (train)", ""]
    md += [
        _table(
            [
                [c, s[("train", "ir")]["per_class"][c], s[("train", "rgb")]["per_class"][c]]
                for c in s[("train", "ir")]["per_class"]
            ],
            ["sınıf", "IR", "RGB"],
        ),
        "",
    ]
    md += ["## Boyut dağılımı (COCO: küçük < 32², orta < 96²)", ""]
    md += [
        _table(
            [[sp, *s[(sp, "ir")]["size"].values()] for sp in SPLITS],
            ["split", "küçük", "orta", "büyük"],
        ),
        "",
    ]
    md += ["## Aydınlatma (türetilmiş, ADR 0005)", ""]
    md += [
        _table(
            [
                [sp, illum[(sp, "day")], illum[(sp, "night")], illum[(sp, "dark_night")]]
                for sp in SPLITS
            ],
            ["split", "gündüz", "gece", "karanlık gece"],
        ),
        "",
    ]

    tr = {
        ls: json.loads((out / "coco" / f"train_{ls}.json").read_text("utf-8"))
        for ls in ("ir", "rgb")
    }
    sh = label_shift(tr["ir"], tr["rgb"])
    off = np.asarray(sh["offsets"])
    p50, p90, p99 = np.percentile(off, [50, 90, 99]) if off.size else (0.0, 0.0, 0.0)
    over10 = float((off > 10).mean() * 100) if off.size else 0.0
    md += [
        "## RGB-IR etiket kayması (train)",
        "",
        "Aynı nesnenin iki etiket setindeki merkez farkı (piksel). Görüntü başına aynı "
        "sınıftaki kutular artan uzaklığa göre eşlendi, en fazla 40 px. Eşsiz kutular ya "
        "diğer sette etiketlenmemiş ya da 40 px'ten fazla kaymış. Spec §8.2'deki '1-20 px' "
        "tahmininin ölçülmüş hali; E6 hizasızlık şiddetleri buna göre yorumlanır.",
        "",
        _table(
            [
                [
                    sh["matched"],
                    sh["unmatched_ir"],
                    sh["unmatched_rgb"],
                    f"{p50:.1f}",
                    f"{p90:.1f}",
                    f"{p99:.1f}",
                    f"{over10:.1f}%",
                ]
            ],
            ["eşleşen", "eşsiz IR", "eşsiz RGB", "p50 px", "p90 px", "p99 px", "> 10 px"],
        ),
        "",
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(off, bins=40, range=(0, 40))
    ax.set_xlabel("merkez kayması (px)")
    ax.set_ylabel("kutu çifti")
    ax.set_title("train — RGB-IR etiket kayması")
    fig.tight_layout()
    fig.savefig(fig_dir / "label_shift_train.png", dpi=120)
    plt.close(fig)
    md += ["![etiket kayması](figures/data/label_shift_train.png)", ""]

    fig, ax = plt.subplots(figsize=(6, 3))
    cls = list(s[("train", "ir")]["per_class"])
    x = range(len(cls))
    ax.bar([i - 0.2 for i in x], [s[("train", "ir")]["per_class"][c] for c in cls], 0.4, label="IR")
    ax.bar(
        [i + 0.2 for i in x], [s[("train", "rgb")]["per_class"][c] for c in cls], 0.4, label="RGB"
    )
    ax.set_xticks(list(x), cls)
    ax.set_title("train — sınıf başına kutu")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "class_counts_train.png", dpi=120)
    plt.close(fig)
    md += [
        "![sınıf dağılımı](figures/data/class_counts_train.png)",
        "",
        "## Aydınlatma etiketi doğrulaması",
        "",
        "Gece/gündüz sınırı, gündüz-gece geçiş bölgesinden seçilen 160 elle etiketli örnekle "
        "belirlendi (ADR 0005; etiketler `docs/data/illum_check_labels.csv`). Numaralı sayfalar:",
        "",
        *(f"![etiketli örnekler {i}](figures/data/illum_check_{i:02d}.jpg)\n" for i in range(4)),
    ]
    return "\n".join(md)
