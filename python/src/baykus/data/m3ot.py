"""M3OT → bizim işlenmiş düzen (FR-D8, kanca K5, spec §8.3).

Ham düzen (Spike C, KAPI M): her dizi img1/*.jpg ve gt/gt.txt taşıyan bir klasör. gt.txt (MOT):
frame, track_id, x, y, w, h, flag, class, visibility. Bölüm ve modalite yol parçalarından ya da
dizi adının eklerinden okunur (test/RGB/seq03, test/seq03_ir, test/seq03/ir).
Çıktı DroneVehicle düzeniyle aynı; stem = <dizi>_<kare:06d>. Dizi, kare ve iz numarası
meta/m3ot_tracks.csv'de korunur (K5: Faz 3 takip, Faz 4 hareketli/duran).
Süzgeç resmi tools/convert_M3OT_to_yolo.py ile aynı: flag == 1, sınıf == 1, görünürlük > 0.25.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass
from pathlib import Path

import cv2

from baykus.data.coco import yolo_to_coco
from baykus.data.convert import to_single_channel
from baykus.data.yamls import write_data_yamls
from baykus.io import imread_any, imwrite_any

CLASS_NAME = "vehicle"
SPLITS = ("train", "val", "test")
MODALITY_TOKENS = {"rgb": {"rgb", "visible", "vis"}, "ir": {"ir", "infrared", "thermal", "tir"}}
MIN_VISIBILITY = 0.25
IMG_EXTS = {".jpg", ".jpeg", ".png"}
_SEP = re.compile(r"[_\-. ]+")
TRACK_FIELDS = ["split", "modality", "seq", "frame", "stem", "track_id", "x", "y", "w", "h",
                "flag", "cls", "vis"]  # fmt: skip


@dataclass(frozen=True)
class Seq:
    split: str
    name: str
    modality: str
    path: Path


@dataclass(frozen=True)
class GtRow:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    flag: int
    cls: int
    vis: float


def _tokens(part: str) -> list[str]:
    return [t for t in _SEP.split(part) if t]


def _modality(part: str) -> str | None:
    toks = {t.lower() for t in _tokens(part)}
    found = [m for m, names in MODALITY_TOKENS.items() if toks & names]
    if len(found) > 1:
        raise ValueError(f"{part!r}: hem RGB hem IR adı geçiyor")
    return found[0] if found else None


def _strip(part: str) -> str:
    names = set().union(*MODALITY_TOKENS.values())
    return "_".join(t for t in _tokens(part) if t.lower() not in names)


def classify(rel: Path) -> tuple[str, str, str]:
    """Ham köke göre dizi klasörü → (bölüm, modalite, dizi adı)."""
    parts = Path(rel).parts
    split = next((p.lower() for p in parts if p.lower() in SPLITS), None)
    mods = {m for p in parts if (m := _modality(p))}
    if split is None or len(mods) != 1:
        raise ValueError(
            f"{Path(rel).as_posix()}: bölüm ya da modalite yoldan okunamadı "
            "(KAPI M çıktısını Claude'a getir)"
        )
    name = _strip(parts[-1]) or _strip(parts[-2])  # test/seq03/ir → seq03
    modality = mods.pop()
    if modality == "ir":  # KAPI M: IR dizi adı sondaki T eki taşır (1-03T → 1_03)
        name = re.sub(r"^(.*\d)[Tt]$", r"\1", name)
    return split, modality, name


def find_sequences(raw: Path) -> list[Seq]:
    raw = Path(raw)
    out = []
    for gt in sorted(raw.rglob("gt/gt.txt")):
        seq_dir = gt.parent.parent
        if (seq_dir / "img1").is_dir():
            split, modality, name = classify(seq_dir.relative_to(raw))
            out.append(Seq(split, name, modality, seq_dir))
    return out


def pair_sequences(seqs: Iterable[Seq]) -> tuple[list[tuple[Seq, Seq]], list[Seq]]:
    by: dict[tuple[str, str], dict[str, Seq]] = defaultdict(dict)
    for s in seqs:
        if s.modality in by[(s.split, s.name)]:
            raise ValueError(f"{s.split}/{s.name}: iki {s.modality} dizisi var")
        by[(s.split, s.name)][s.modality] = s
    pairs, alone = [], []
    for key in sorted(by):
        d = by[key]
        if "rgb" in d and "ir" in d:
            pairs.append((d["rgb"], d["ir"]))
        else:
            alone += [d[m] for m in sorted(d)]
    return pairs, alone


def read_gt(path: Path) -> list[GtRow]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        f = [v.strip() for v in line.split(",")]
        if len(f) < 9:
            raise ValueError(f"{path}:{n}: 9 sütun bekleniyordu, {len(f)} var")
        i = [int(float(v)) for v in (f[0], f[1], f[6], f[7])]
        x, y, w, h, vis = (float(v) for v in (f[2], f[3], f[4], f[5], f[8]))
        rows.append(GtRow(i[0], i[1], x, y, w, h, i[2], i[3], vis))
    return rows


def keep(r: GtRow) -> bool:
    return r.flag == 1 and r.cls == 1 and r.vis > MIN_VISIBILITY and r.w > 0 and r.h > 0


def yolo_lines(rows: Iterable[GtRow], width: int, height: int) -> list[str]:
    """Kutu görüntüye kırpılır, boş kalan atlanır. Sınıf 0: değerlendirme sınıfsız (single_cls)."""
    out = []
    for r in rows:
        x1, y1 = max(r.x, 0.0), max(r.y, 0.0)
        x2, y2 = min(r.x + r.w, float(width)), min(r.y + r.h, float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        out.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return out


def frame_images(img_dir: Path) -> dict[int, Path]:
    out = {}
    for p in sorted(Path(img_dir).iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        digits = re.findall(r"\d+", p.stem)
        if not digits:
            raise ValueError(f"{p}: kare numarası okunamadı")
        out[int(digits[-1])] = p
    return out


def _write_frame(job: tuple) -> tuple[int, int]:
    rgb_src, ir_src, out, split, stem = job
    bgr = imread_any(rgb_src, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"okunamadı: {rgb_src}")
    ir = imread_any(ir_src, cv2.IMREAD_UNCHANGED)
    if ir is None:
        raise OSError(f"okunamadı: {ir_src}")
    ir = to_single_channel(ir)
    if bgr.shape[:2] != ir.shape[:2]:
        raise ValueError(f"{stem}: RGB {bgr.shape[:2]} ≠ IR {ir.shape[:2]}")
    dst = out / "rgb/images" / split / f"{stem}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rgb_src.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copy2(rgb_src, dst)  # yeniden sıkıştırma yok
    else:
        imwrite_any(dst, bgr)
    imwrite_any(out / "ir/images" / split / f"{stem}.png", ir)
    return ir.shape[1], ir.shape[0]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def convert_m3ot(raw: Path, out: Path, splits: Iterable[str] = ("test",), workers: int = 1) -> dict:
    raw, out, splits = Path(raw), Path(out), tuple(splits)
    pairs, alone = pair_sequences(s for s in find_sequences(raw) if s.split in splits)
    if not pairs:
        raise ValueError(f"{raw}: {list(splits)} için RGB-IR dizi çifti bulunamadı")
    stats = {
        "pairs": 0,
        "sequences": len(pairs),
        "unpaired_sequences": [f"{s.split}/{s.name}/{s.modality}" for s in alone],
        "frames_rgb_only": 0,
        "frames_ir_only": 0,
        "boxes": {"rgb": 0, "ir": 0},
        "dropped_rows": {"rgb": 0, "ir": 0},
        # süzgeçten geçen ama karesi tek modalitede olduğu için etikete girmeyen satırlar
        "rows_unpaired_frames": {"rgb": 0, "ir": 0},
    }
    jobs, labels, index, tracks = [], {}, [], []
    for rgb, ir in pairs:
        f_rgb, f_ir = frame_images(rgb.path / "img1"), frame_images(ir.path / "img1")
        common = sorted(set(f_rgb) & set(f_ir))
        stats["frames_rgb_only"] += len(set(f_rgb) - set(f_ir))
        stats["frames_ir_only"] += len(set(f_ir) - set(f_rgb))
        by_frame: dict[str, dict[int, list[GtRow]]] = {}
        for m, seq in (("rgb", rgb), ("ir", ir)):
            by_frame[m] = defaultdict(list)
            for r in read_gt(seq.path / "gt" / "gt.txt"):
                stem = f"{rgb.name}_{r.frame:06d}"
                tracks.append([rgb.split, m, rgb.name, r.frame, stem, *astuple(r)[1:]])
                if keep(r):
                    by_frame[m][r.frame].append(r)
                else:
                    stats["dropped_rows"][m] += 1
            stats["rows_unpaired_frames"][m] += sum(
                len(rows) for fr, rows in by_frame[m].items() if fr not in set(common)
            )
        for fr in common:
            stem = f"{rgb.name}_{fr:06d}"
            jobs.append((f_rgb[fr], f_ir[fr], out, rgb.split, stem))
            labels[(rgb.split, stem)] = {m: by_frame[m].get(fr, []) for m in ("rgb", "ir")}
            index.append([rgb.split, stem, rgb.name, fr])
    if workers <= 1:
        sizes = [_write_frame(j) for j in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            sizes = list(ex.map(_write_frame, jobs, chunksize=32))
    if len(set(sizes)) != 1:
        raise ValueError(f"görüntü boyutları farklı: {sorted(set(sizes))[:5]}")
    width, height = sizes[0]
    for (split, stem), per_mod in labels.items():
        for m, rows in per_mod.items():
            lines = yolo_lines(rows, width, height)
            stats["boxes"][m] += len(lines)
            p = out / f"labels_{m}" / split / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    (out / "coco").mkdir(parents=True, exist_ok=True)
    for split in splits:
        stems = sorted(stem for s, stem in labels if s == split)
        for m in ("ir", "rgb"):
            if stems:
                coco = yolo_to_coco(
                    out / f"labels_{m}" / split, stems, width, height, single_class=CLASS_NAME
                )
                (out / "coco" / f"{split}_{m}.json").write_text(json.dumps(coco), encoding="utf-8")
    stats["pairs"] = len(jobs)
    meta = out / "meta"
    _write_csv(meta / "m3ot_index.csv", ["split", "stem", "seq", "frame"], index)
    _write_csv(meta / "m3ot_tracks.csv", TRACK_FIELDS, tracks)
    (meta / "m3ot_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_data_yamls(out)
    return stats
