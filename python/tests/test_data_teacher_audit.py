"""Araç türü kör denetimi testleri (Faz 4 tasarımı §3 B)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.data.teacher_audit import render, sample, score
from baykus.io import imread_any


def _teacher(n_car: int, n_truck: int, n_bus: int) -> dict[str, dict]:
    teacher: dict[str, dict] = {}
    i = 0
    for cls_id, count in ((0, n_car), (1, n_truck), (2, n_bus)):
        for _ in range(count):
            teacher[f"k{i:04d}"] = {"cls_id": cls_id}
            i += 1
    return teacher


def test_audit_sample_stratified_seeded() -> None:
    teacher = _teacher(100, 30, 10)
    picked = sample(teacher, n=20, non_car_max=8, seed=42)
    assert len(picked) == 20
    non_car = [k for k in picked if teacher[k]["cls_id"] != 0]
    assert len(non_car) == 8
    truck = [k for k in non_car if teacher[k]["cls_id"] == 1]
    bus = [k for k in non_car if teacher[k]["cls_id"] == 2]
    assert len(truck) == 6
    assert len(bus) == 2
    assert sample(teacher, n=20, non_car_max=8, seed=42) == picked


def test_audit_page_hides_prediction(tmp_path: Path) -> None:
    keys = [f"k{i:04d}" for i in range(6)]
    crops = {
        k: (np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), (0.0, 0.0, 8.0, 8.0))
        for k in keys
    }
    render(keys, crops, tmp_path, seed=42)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    for key in keys:
        assert key not in html
    assert "cls_id" not in html
    for label in ("Otomobil", "Kamyon", "Otobüs", "Minibüs", "Yük aracı", "Bilmiyorum"):
        assert label in html


def test_audit_crops_mark_the_asked_vehicle(tmp_path: Path) -> None:
    ir = np.zeros((32, 40), np.uint8)
    rgb = np.zeros((32, 40, 3), np.uint8)
    box = (10.0, 8.0, 20.0, 16.0)
    render(["k0000"], {"k0000": (ir, rgb, box)}, tmp_path, seed=42)
    for name in ("ir", "rgb"):
        img = imread_any(tmp_path / "crops" / f"t001_{name}.jpg", cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape == (128, 160, 3)
        # Çerçeve çizgisi: kutunun 3 px dışı (kutu üstü 8*4 = 32, çizgi 27-28).
        b, g, r = (int(v) for v in img[8 * 4 - 4, 15 * 4])
        assert b < 80 and g > 170 and r > 170
        # Kutunun içi (araç) boyanmaz.
        assert all(int(v) < 60 for v in img[12 * 4, 15 * 4])


def test_audit_score_excludes_unknown_and_decides() -> None:
    teacher = {f"k{i:04d}": {"cls_id": 0} for i in range(100)}
    ids = {f"t{i:03d}": f"k{i:04d}" for i in range(100)}

    answers = {sid: ("car" if i < 80 else "truck") for i, sid in enumerate(ids)}
    result = score(ids, teacher, answers)
    assert result["accuracy"] == pytest.approx(0.80)
    assert result["decision"] == "teacher"

    answers79 = {sid: ("car" if i < 79 else "truck") for i, sid in enumerate(ids)}
    result79 = score(ids, teacher, answers79)
    assert result79["accuracy"] == pytest.approx(0.79)
    assert result79["decision"] == "all_car"

    answers_unknown: dict[str, str] = {}
    for i, sid in enumerate(ids):
        if i < 80:
            answers_unknown[sid] = "car"
        elif i < 90:
            answers_unknown[sid] = "truck"
        else:
            answers_unknown[sid] = "unknown"
    result_unknown = score(ids, teacher, answers_unknown)
    assert result_unknown["answered"] == 90
    assert result_unknown["unknown"] == 10
    assert result_unknown["accuracy"] == pytest.approx(80 / 90)


def test_audit_score_all_unknown_is_all_car() -> None:
    teacher = {f"k{i:04d}": {"cls_id": 0} for i in range(10)}
    ids = {f"t{i:03d}": f"k{i:04d}" for i in range(10)}
    answers = {sid: "unknown" for sid in ids}
    result = score(ids, teacher, answers)
    assert result["accuracy"] is None
    assert result["decision"] == "all_car"


def test_audit_score_missing_answer_raises() -> None:
    teacher = {f"k{i:04d}": {"cls_id": 0} for i in range(3)}
    ids = {f"t{i:03d}": f"k{i:04d}" for i in range(3)}
    answers = {"t000": "car", "t001": "car"}
    with pytest.raises(ValueError):
        score(ids, teacher, answers)
