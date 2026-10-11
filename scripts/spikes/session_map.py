"""Spike (ADR 0005, seçenek C): görüntü → çekim oturumu eşlemesi ve oturum başına örnek sayfaları.

Ham XML'deki <folder> alanı (ör. "5-11") çekim oturumu olarak alınır. Çıktılar:
  <out>/session_map.csv        split, stem, folder, filename (RGB etiket XML'inden)
  <out>/sessions_NN.jpg        her satır bir oturum: dosya adına göre sıralı, eşit aralıklı örnekler
  <out>/sessions_summary.json  oturum sayısı ve oturum başına görüntü sayısı

Çalıştırma (Colab):
  python scripts/spikes/session_map.py /content/raw/VisDrone-DroneVehicle /content/sessions
"""

from __future__ import annotations

import collections
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
CROP = 100
THUMB = (160, 128)  # 640x512 kırpılmış görüntünün 1/4'ü
PER_ROW = 6
ROWS_PER_SHEET = 12
LABEL_W = 220


def read_rows(raw: Path) -> list[dict]:
    rows = []
    for split in SPLITS:
        for xml in sorted((raw / split / f"{split}label").glob("*.xml")):
            try:
                r = ET.parse(xml).getroot()
            except ET.ParseError:
                continue
            rows.append(
                {
                    "split": split,
                    "stem": xml.stem,
                    "folder": (r.findtext("folder") or "").strip(),
                    "filename": (r.findtext("filename") or "").strip(),
                }
            )
    return rows


def thumb(path: Path) -> np.ndarray:
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        return np.zeros((THUMB[1], THUMB[0], 3), np.uint8)
    return cv2.resize(a[CROP:-CROP, CROP:-CROP], THUMB, interpolation=cv2.INTER_AREA)


def session_row(raw: Path, folder: str, items: list[dict]) -> np.ndarray:
    items = sorted(items, key=lambda d: d["filename"])
    idx = np.linspace(0, len(items) - 1, min(PER_ROW, len(items))).round().astype(int)
    tiles = [
        thumb(raw / items[i]["split"] / f"{items[i]['split']}img" / f"{items[i]['stem']}.jpg")
        for i in idx
    ]
    tiles += [np.zeros((THUMB[1], THUMB[0], 3), np.uint8)] * (PER_ROW - len(tiles))
    label = np.full((THUMB[1], LABEL_W, 3), 255, np.uint8)
    cv2.putText(label, folder[:18], (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(label, f"n={len(items)}", (6, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return np.hstack([label, *tiles])


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(raw)
    with (out / "session_map.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", "folder", "filename"])
        w.writeheader()
        w.writerows(rows)
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[r["folder"]].append(r)
    folders = sorted(groups)
    summary = {
        "n_images": len(rows),
        "n_sessions": len(folders),
        "per_session": {k: len(groups[k]) for k in folders},
        "splits_per_session": {
            k: dict(collections.Counter(d["split"] for d in groups[k])) for k in folders
        },
    }
    (out / "sessions_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    for page, start in enumerate(range(0, len(folders), ROWS_PER_SHEET)):
        sheet = np.vstack(
            [session_row(raw, k, groups[k]) for k in folders[start : start + ROWS_PER_SHEET]]
        )
        ok, enc = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        enc.tofile(str(out / f"sessions_{page:02d}.jpg"))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "splits_per_session"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
