"""M3OT ham verisinin gerçek yapısı (H4 KAPI M). Veri toplama scripti; test yazılmaz.

Kullanım: python inspect_m3ot.py <ham_kök> <çıktı.json> [--figs KLASÖR]
Her `gt/gt.txt` için dizi özeti, ağacın ilk 3 seviyesi ve ilk iki dizinin ortadaki karesine
süzülmüş kutuların çizimi (gözle hiza kontrolü).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from baykus.io import imread_any, imwrite_any

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def tree(root: Path, depth: int = 3, limit: int = 12) -> list[str]:
    lines = []

    def walk(d: Path, level: int) -> None:
        kids = sorted(d.iterdir())
        dirs = [k for k in kids if k.is_dir()]
        files = [k for k in kids if k.is_file()]
        exts = Counter(f.suffix.lower() for f in files)
        lines.append(f"{'  ' * level}{d.name}/  ({len(dirs)} klasör, {dict(exts)})")
        if level + 1 < depth:
            for k in dirs[:limit]:
                walk(k, level + 1)
            if len(dirs) > limit:
                lines.append(f"{'  ' * (level + 1)}… {len(dirs) - limit} klasör daha")

    walk(root, 0)
    return lines


def gt_summary(path: Path) -> dict:
    rows = [ln.split(",") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def col(i: int) -> dict:
        return dict(Counter(r[i].strip() for r in rows if len(r) > i).most_common(8))

    vis = [float(r[8]) for r in rows if len(r) > 8]
    frames = [int(float(r[0])) for r in rows]
    return {
        "rows": len(rows),
        "ncols": dict(Counter(len(r) for r in rows)),
        "frames": [min(frames), max(frames)] if frames else None,
        "tracks": len({r[1] for r in rows}),
        "col7_flag": col(6),
        "col8_class": col(7),
        "col9_vis": {"min": min(vis), "max": max(vis), "le_0.25": sum(v <= 0.25 for v in vis)}
        if vis
        else None,
        "head": [",".join(r) for r in rows[:3]],
    }


def seq_summary(seq: Path, raw: Path) -> dict:
    imgs = sorted(p for p in (seq / "img1").iterdir() if p.suffix.lower() in IMG)
    first = imread_any(imgs[0], cv2.IMREAD_UNCHANGED) if imgs else None
    ini = seq / "seqinfo.ini"
    return {
        "path": seq.relative_to(raw).as_posix(),
        "n_images": len(imgs),
        "exts": dict(Counter(p.suffix.lower() for p in imgs)),
        "first": imgs[0].name if imgs else None,
        "last": imgs[-1].name if imgs else None,
        "shape": list(first.shape) if first is not None else None,
        "dtype": str(first.dtype) if first is not None else None,
        "seqinfo": ini.read_text(encoding="utf-8") if ini.exists() else None,
        "gt": gt_summary(seq / "gt" / "gt.txt"),
    }


def draw(seq: Path, out: Path) -> None:
    imgs = sorted(p for p in (seq / "img1").iterdir() if p.suffix.lower() in IMG)
    mid = imgs[len(imgs) // 2]
    frame = int("".join(ch for ch in mid.stem if ch.isdigit()) or 0)
    img = imread_any(mid, cv2.IMREAD_COLOR)
    for ln in (seq / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
        f = ln.split(",")
        if len(f) < 9 or int(float(f[0])) != frame:
            continue
        x, y, w, h = (int(float(v)) for v in f[2:6])
        ok = int(float(f[6])) == 1 and int(float(f[7])) == 1 and float(f[8]) > 0.25
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0) if ok else (0, 0, 255), 1)
    name = seq.relative_to(seq.parents[2]).as_posix().replace("/", "__")
    imwrite_any(out / f"{name}__{mid.stem}.jpg", img)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--figs", type=Path, default=None)
    a = p.parse_args()
    seqs = sorted(g.parent.parent for g in a.raw.rglob("gt/gt.txt"))
    report = {
        "root": str(a.raw),
        "tree": tree(a.raw),
        "sequences": [seq_summary(s, a.raw) for s in seqs],
        "other_files": sorted(
            f"{f.relative_to(a.raw).as_posix()} ({f.stat().st_size} B)"
            for f in a.raw.rglob("*")
            if f.is_file()
            and f.suffix.lower() in {".json", ".txt", ".ini", ".csv", ".md"}
            and f.name != "gt.txt"
        )[:40],
    }
    if a.figs:
        a.figs.mkdir(parents=True, exist_ok=True)
        for s in seqs[:4]:
            draw(s, a.figs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(report["tree"]))
    print(f"\n{len(seqs)} dizi")
    for s in report["sequences"]:
        g = s["gt"]
        print(f"{s['path']} | {s['n_images']} görüntü {s['exts']} {s['first']}..{s['last']} "
              f"{s['shape']} | gt {g['rows']} satır, kare {g['frames']}, iz {g['tracks']}, "
              f"flag {g['col7_flag']}, sınıf {g['col8_class']}, vis {g['col9_vis']}")  # fmt: skip
    print("\ndiğer dosyalar:", *report["other_files"], sep="\n  ")


if __name__ == "__main__":
    main()
