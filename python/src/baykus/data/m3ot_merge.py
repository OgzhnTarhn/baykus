"""M3OT train'i DroneVehicle eğitim ağacına katmak (Görev 15B; tasarım faz4-kapsama.md §3 B).

relabel — `labels_ir/<split>/` dosyalarını `meta/m3ot_tracks.csv`'den yeniden yazar; sınıf öğretmen
          etiketinden (`teacher[<seq>:<track_id>]["cls_id"]`), öğretmen yoksa ya da iz yoksa `car`.
          Süzgeç ve kutu biçimi dönüştürücüyle aynı (IR, flag 1, sınıf 1, vis > 0,25; kırpma).
merge   — M3OT RGB / IR görüntülerini ve IR etiketlerini DroneVehicle köküne önekli adla kopyalar
          (`train` gibi; test reddedilir). Var olan dosyanın üstüne yazmaz.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from baykus.classes import CLASS_NAMES
from baykus.data.teacher import track_rows
from baykus.io import imread_any


def _image_size(root: Path, split: str) -> tuple[int, int]:
    """(genişlik, yükseklik): dönüştürücü bütün karelerin aynı boyda olduğunu denetledi."""
    first = next(iter(sorted((root / "ir/images" / split).glob("*.png"))), None)
    if first is None:
        raise ValueError(f"{root}: ir/images/{split} boş")
    img = imread_any(first)
    if img is None:
        raise OSError(f"okunamadı: {first}")
    return img.shape[1], img.shape[0]


def _line(cls_id: int, box: list[float], width: int, height: int) -> str | None:
    """`m3ot.yolo_lines` ile aynı kırpma ve biçim; sınıf parametreli."""
    x1, y1 = max(box[0], 0.0), max(box[1], 0.0)
    x2, y2 = min(box[2], float(width)), min(box[3], float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def relabel(root: Path, split: str, teacher: dict[str, dict] | None) -> dict[str, int]:
    """Etiket dosyalarını yeniden yazar; sınıf adı → kutu sayısı döner."""
    root = Path(root)
    rows = track_rows(root, split)  # test bölümünü reddeder
    width, height = _image_size(root, split)
    lines: dict[str, list[str]] = {
        p.stem: [] for p in (root / "labels_ir" / split).glob("*.txt")
    }  # kutusuz kareler de boş dosya olarak kalır
    counts = {name: 0 for name in CLASS_NAMES}
    for key, per in rows.items():
        cls_id = int((teacher or {}).get(key, {}).get("cls_id", 0))
        for r in per:
            ln = _line(cls_id, r["box"], width, height)
            if ln is None:
                continue
            lines.setdefault(r["stem"], []).append(ln)
            counts[CLASS_NAMES[cls_id]] += 1
    out = root / "labels_ir" / split
    out.mkdir(parents=True, exist_ok=True)
    for stem, ls in lines.items():
        (out / f"{stem}.txt").write_text("".join(x + "\n" for x in ls), encoding="utf-8")
    return counts


def merge(m3ot_root: Path, dv_root: Path, split: str = "train", prefix: str = "m3ot_") -> int:
    """Önekli kopyalama; kopyalanan RGB-IR çifti sayısı."""
    if split == "test":
        raise ValueError("test bölümü eğitime katılmaz")
    src, dst = Path(m3ot_root), Path(dv_root)
    stems = sorted(p.stem for p in (src / "ir/images" / split).glob("*.png"))
    if not stems:  # açılmamış kaynak sessizce 0 çift vermesin (eğitim M3OT'suz sürerdi)
        raise ValueError(f"M3OT kaynağında çift yok: {src / 'ir/images' / split}")
    jobs = []
    for stem in stems:
        for rel, ext in (("rgb/images", ".jpg"), ("ir/images", ".png"), ("labels_ir", ".txt")):
            a = src / rel / split / f"{stem}{ext}"
            b = dst / rel / split / f"{prefix}{stem}{ext}"
            if b.exists():
                raise ValueError(f"hedefte zaten var: {b}")
            jobs.append((a, b))
    for a, b in jobs:  # önce bütün hedefler denetlenir: yarım birleştirme kalmaz
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, b)
    return len(stems)
