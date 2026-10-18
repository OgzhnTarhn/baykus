"""M3OT geliştirme ve demo paketleri (Faz 3 tasarımı §4.4, §4.6).

`baykus-data m3ot` çıktısındaki dizileri operatör ekranının okuduğu düzene kopyalar:
`sequences/<dizi>/{rgb/<kare:06d>.jpg, ir/<kare:06d>.png, index.json}`; ölçüm için etiketler
`labels/{<split>_ir.json, tracks_ir.csv}`. Görüntüler bayt bayt kopyalanır (yeniden kodlama yok).
Zaman kuralı ve takip ardışık kare varsaydığı için kare numaraları ardışık olmayan dizi paketlenmez.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from baykus.io import imread_any

FPS = 30  # M3OT'un kare hızı doğrulanmadı; resmi araçtaki değer (tasarım §4.3)
SOURCE = "M3OT {split}, CC BY 4.0"


def _frames(index_csv: Path, split: str) -> dict[str, list[int]]:
    by_seq: dict[str, list[int]] = defaultdict(list)
    with Path(index_csv).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                by_seq[row["seq"]].append(int(row["frame"]))
    return {seq: sorted(fr) for seq, fr in by_seq.items()}


def _is_consecutive(frames: list[int]) -> bool:
    return bool(frames) and frames == list(range(frames[0], frames[0] + len(frames)))


def consecutive_sequences(index_csv: Path, split: str) -> dict[str, dict]:
    """Dizi → kare sayısı ve kare numaralarının aralıksız artıp artmadığı."""
    return {
        seq: {"frames": len(fr), "consecutive": _is_consecutive(fr)}
        for seq, fr in sorted(_frames(index_csv, split).items())
    }


def pack_sequence(root: Path, split: str, seq: str, out: Path) -> Path:
    root, out = Path(root), Path(out)
    frames = _frames(root / "meta/m3ot_index.csv", split).get(seq)
    if not frames:
        raise ValueError(f"{split}/{seq}: dizi yok")
    if not _is_consecutive(frames):
        raise ValueError(f"{split}/{seq}: kare numaraları ardışık değil, paketlenmez")
    src = [
        (
            root / "rgb/images" / split / f"{seq}_{f:06d}.jpg",
            root / "ir/images" / split / f"{seq}_{f:06d}.png",
        )
        for f in frames
    ]
    rgb0 = imread_any(src[0][0], cv2.IMREAD_COLOR)
    ir0 = imread_any(src[0][1], cv2.IMREAD_GRAYSCALE)
    if rgb0 is None or ir0 is None:
        raise OSError(f"{split}/{seq}: ilk kare okunamadı")
    if rgb0.shape[:2] != ir0.shape[:2]:
        raise ValueError(
            f"{split}/{seq}: RGB ve IR boyutu farklı: {rgb0.shape[:2]} / {ir0.shape[:2]}"
        )
    seq_dir = out / "sequences" / seq
    (seq_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (seq_dir / "ir").mkdir(parents=True, exist_ok=True)
    for f, (rgb, ir) in zip(frames, src, strict=True):
        shutil.copyfile(rgb, seq_dir / "rgb" / f"{f:06d}.jpg")
        shutil.copyfile(ir, seq_dir / "ir" / f"{f:06d}.png")
    index = {
        "name": seq,
        "split": split,
        "frames": frames,
        "fps": FPS,
        "size": [int(rgb0.shape[1]), int(rgb0.shape[0])],
        "source": SOURCE.format(split=split),
    }
    (seq_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return seq_dir


def pack_labels(root: Path, split: str, seqs: list[str], out: Path) -> Path:
    """Seçilen dizilerin IR COCO etiketleri (görüntü kimlikleri korunur) ve IR iz satırları."""
    root, labels = Path(root), Path(out) / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(f"{s}_" for s in seqs)
    coco = json.loads((root / "coco" / f"{split}_ir.json").read_text(encoding="utf-8"))
    images = [im for im in coco["images"] if im["file_name"].startswith(prefixes)]
    ids = {im["id"] for im in images}
    sub = {
        "images": images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        "categories": coco["categories"],
    }
    (labels / f"{split}_ir.json").write_text(json.dumps(sub), encoding="utf-8")
    keep = set(seqs)
    with (root / "meta/m3ot_tracks.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            r for r in reader if r["split"] == split and r["modality"] == "ir" and r["seq"] in keep
        ]
        fields = reader.fieldnames
    with (labels / "tracks_ir.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return labels


def pack(
    root: Path, split: str, seqs: list[str], out: Path, *, allow_test: bool = False
) -> list[Path]:
    if split == "test" and not allow_test:
        raise ValueError(
            "test bölümü yalnız allow_test (--allow-test) ile paketlenir; test yalnız son ölçümde"
        )
    dirs = [pack_sequence(root, split, s, out) for s in seqs]
    pack_labels(root, split, seqs, out)
    return dirs
