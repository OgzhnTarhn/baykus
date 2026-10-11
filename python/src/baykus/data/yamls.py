"""Ultralytics data_*.yaml dosyaları (modalite başına bir dosya)."""

from __future__ import annotations

from pathlib import Path

from baykus.classes import CLASS_NAMES

CHANNELS = {"rgb": 3, "ir": 3, "rgbt4": 4}


def write_data_yamls(out: Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES))
    paths = []
    for modality, ch in CHANNELS.items():
        p = out / f"data_{modality}.yaml"
        p.write_text(
            f"path: {(out / modality).as_posix()}\ntrain: images/train\nval: images/val\n"
            f"test: images/test\nchannels: {ch}\nnames:\n{names}",
            encoding="utf-8",
        )
        paths.append(p)
    return paths
