from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baykus.data.layout import RAW_SIZE, SplitDirs, split_dirs
from baykus.io import imwrite_any

XML_TMPL = """<annotation><size><width>840</width><height>712</height></size>{objs}</annotation>"""
OBJ_TMPL = (
    "<object><name>{name}</name><polygon><x1>{x1}</x1><y1>{y1}</y1><x2>{x2}</x2><y2>{y1}</y2>"
    "<x3>{x2}</x3><y3>{y2}</y3><x4>{x1}</x4><y4>{y2}</y4></polygon></object>"
)


def write_xml(path: Path, objs: list[tuple[str, float, float, float, float]]) -> None:
    body = "".join(OBJ_TMPL.format(name=n, x1=a, y1=b, x2=c, y2=d) for n, a, b, c, d in objs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(XML_TMPL.format(objs=body), encoding="utf-8")


def make_raw_dataset(
    root: Path,
    split: str,
    stems: list[str],
    objects: dict | None = None,
    brightness: dict | None = None,
) -> SplitDirs:
    """Her stem için 840x712 RGB + IR görüntü ve iki XML yazar.
    objects: {stem: [(name, x1, y1, x2, y2), ...]} — hem RGB hem IR etiketine yazılır.
    brightness: {stem: 0-255} — RGB görüntünün sabit parlaklığı (varsayılan 128)."""
    d = split_dirs(root, split)
    w, h = RAW_SIZE
    for s in stems:
        v = (brightness or {}).get(s, 128)
        imwrite_any(d.rgb / f"{s}.jpg", np.full((h, w, 3), v, np.uint8))
        imwrite_any(d.ir / f"{s}.jpg", np.full((h, w, 3), 90, np.uint8))
        objs = (objects or {}).get(s, [("car", 200, 150, 240, 170)])
        write_xml(d.label_rgb / f"{s}.xml", objs)
        write_xml(d.label_ir / f"{s}.xml", objs)
    return d


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"
