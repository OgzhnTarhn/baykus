"""baykus-data: veri hattı alt komutları."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baykus.data.layout import SPLITS, split_dirs


def _cmd_verify(a: argparse.Namespace) -> int:
    from baykus.data.verify import verify_split, write_outputs

    ok, issues = {}, []
    for split in SPLITS:
        s_ok, s_issues = verify_split(split_dirs(Path(a.raw), split), split)
        ok[split], issues = s_ok, issues + s_issues
    write_outputs(Path(a.meta), ok, issues)
    print(json.dumps({k: len(v) for k, v in ok.items()}), f"excluded={len(issues)}")
    return 0


def _cmd_convert(a: argparse.Namespace) -> int:
    from baykus.data.convert import convert_split

    ok = json.loads((Path(a.meta) / "ok_stems.json").read_text(encoding="utf-8"))
    stats = {
        s: convert_split(split_dirs(Path(a.raw), s), s, ok.get(s, []), Path(a.out)) for s in SPLITS
    }
    (Path(a.meta) / "convert_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def _cmd_activate(a: argparse.Namespace) -> int:
    from baykus.data.convert import activate_label_set

    print(activate_label_set(Path(a.out), a.modality, a.label_set))
    return 0


def _cmd_illum(a: argparse.Namespace) -> int:
    import csv

    import cv2
    import numpy as np

    from baykus.data.illumination import (
        LEVELS,
        OFFICIAL_COUNTS,
        assign,
        fit_thresholds,
        mean_brightness,
    )
    from baykus.io import imread_any, imwrite_any

    out, meta, fig = Path(a.out), Path(a.meta), Path(a.fig)
    rows = []
    for split in SPLITS:
        for p in sorted((out / "rgb/images" / split).glob("*.jpg")):
            rows.append((split, p.stem, mean_brightness(imread_any(p))))
    values = np.array([r[2] for r in rows])
    th = fit_thresholds(values, OFFICIAL_COUNTS)
    labels = assign(values, th)

    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "illumination.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "brightness", "illumination"])
        for (s, st, b), lab in zip(rows, labels, strict=True):
            w.writerow([s, st, f"{b:.2f}", lab])
    (meta / "illumination_thresholds.json").write_text(
        json.dumps({"t_dark": th[0], "t_night": th[1]}), encoding="utf-8"
    )

    # her seviyeden 16 örneklik mozaik (gözle doğrulama için)
    rng = np.random.default_rng(0)
    for level in LEVELS:
        idx = [i for i, lab in enumerate(labels) if lab == level]
        if not idx:
            continue
        pick = rng.choice(idx, size=min(16, len(idx)), replace=False)
        tiles = [
            cv2.resize(
                imread_any(out / "rgb/images" / rows[i][0] / f"{rows[i][1]}.jpg"), (320, 256)
            )
            for i in pick
        ]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r : r + 4]) for r in range(0, len(tiles), 4)])
        imwrite_any(fig / f"illum_samples_{level}.jpg", grid)

    counts = {lv: labels.count(lv) for lv in LEVELS}
    print(json.dumps({"thresholds": th, "counts": counts}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-data")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="RGB/IR çift ve etiket bütünlüğü")
    v.add_argument("--raw", required=True)
    v.add_argument("--meta", required=True)
    v.set_defaults(func=_cmd_verify)
    c = sub.add_parser("convert", help="kenarlık kırp, etiketleri YOLO formatına çevir")
    c.add_argument("--raw", required=True)
    c.add_argument("--meta", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_convert)
    al = sub.add_parser(
        "activate-labels", help="bir etiket setini <modality>/labels altına kopyala"
    )
    al.add_argument("--out", required=True)
    al.add_argument("--modality", required=True, choices=["rgb", "ir", "rgbt4"])
    al.add_argument("--label-set", required=True, choices=["ir", "rgb"])
    al.set_defaults(func=_cmd_activate)
    il = sub.add_parser("illum", help="aydınlatma etiketi (parlaklık eşikleri)")
    il.add_argument("--out", required=True)
    il.add_argument("--meta", required=True)
    il.add_argument("--fig", default="../docs/figures/data")
    il.set_defaults(func=_cmd_illum)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
