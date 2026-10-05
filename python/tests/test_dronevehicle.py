import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from baykus.classes import CLASS_NAMES, class_id, coco_category_id, normalize_class
from baykus.data.dronevehicle import parse_annotation
from baykus.io import imread_any, imwrite_any

FIX = Path(__file__).parent / "fixtures" / "dv_annotation.xml"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("car", "car"),
        (" Car ", "car"),
        ("feright car", "freight_car"),
        ("freight car", "freight_car"),
        ("feright_car", "freight_car"),
        ("truvk", "truck"),
        ("tank", None),
        ("", None),
    ],
)
def test_normalize_class(raw, expected):
    assert normalize_class(raw) == expected


def test_class_ids():
    assert CLASS_NAMES == ("car", "truck", "bus", "van", "freight_car")
    assert class_id("van") == 3
    assert coco_category_id(0) == 1


def test_parse_polygon_bndbox_and_aliases():
    res = parse_annotation(FIX)
    names = [o.cls_name for o in res.objects]
    assert names == ["car", "freight_car", "truck"]
    assert res.objects[0].points.shape == (4, 2)
    np.testing.assert_allclose(
        res.objects[2].points, [[400, 400], [460, 400], [460, 430], [400, 430]]
    )


def test_parse_reports_unknown_and_skipped():
    res = parse_annotation(FIX)
    assert res.unknown_classes == ["tank"]
    assert res.skipped_shapes == 1  # <point> şekli desteklenmiyor


def test_parse_broken_xml_raises(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<annotation><object>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        parse_annotation(bad)


def test_imread_imwrite_roundtrip_non_ascii(tmp_path):
    p = tmp_path / "çğü" / "a.png"
    p.parent.mkdir()
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    imwrite_any(p, img)
    np.testing.assert_array_equal(imread_any(p), img)


def test_imread_missing_returns_none(tmp_path):
    assert imread_any(tmp_path / "yok.png") is None
