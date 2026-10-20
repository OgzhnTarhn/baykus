import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.motion.audit import (
    CENTER,
    Sample,
    centered,
    population,
    render,
    sample,
    score,
    wilson,
)
from baykus.motion.images import load_sequence
from baykus.motion.labels import LabelRow

from .motion_synth import make_video, write_sequence


def _gt(n_tracks: int, n_frames: int) -> dict[str, dict[int, dict[int, tuple]]]:
    return {"s": {t: {f: (0, 0, 10, 10) for f in range(1, n_frames + 1)} for t in range(n_tracks)}}


def _labels(gt, label_of) -> dict[tuple[str, int, int], LabelRow]:
    out = {}
    for seq, tracks in gt.items():
        for tid, per in tracks.items():
            for f in per:
                lab = label_of(tid, f)
                out[(seq, f, tid)] = LabelRow(seq, f, tid, 0.0, lab)
    return out


def test_population_ignores_labels():
    gt = _gt(3, 60)
    frames = {"s": list(range(1, 61))}
    pop = population(gt, frames, k=15)
    assert len(pop) == 3 * (60 - 30)  # t ± 15 karede iz var: 16..45
    assert min(p[1] for p in pop) == 16 and max(p[1] for p in pop) == 45
    # etiketler nüfusu değiştirmez: imzada etiket yok
    assert population(gt, frames, k=15) == pop


def test_sample_counts_and_track_cap():
    gt = _gt(60, 100)
    frames = {"s": list(range(1, 101))}
    labels = _labels(gt, lambda t, f: "band" if t % 2 == 0 else "moving")
    pop = population(gt, frames, k=15)
    a = sample(pop, labels, seed=42)
    b = sample(pop, labels, seed=42)
    assert a == b
    assert sum(s.kind == "uniform" for s in a) == 80
    assert sum(s.kind == "targeted" for s in a) == 40
    per_track: dict[tuple[str, int], int] = {}
    for s in a:
        per_track[(s.seq, s.track_id)] = per_track.get((s.seq, s.track_id), 0) + 1
    assert max(per_track.values()) <= 3
    assert len({(s.seq, s.frame, s.track_id) for s in a}) == 120  # tekrar yok
    for s in a:
        if s.kind == "targeted":  # bu kurguda geçiş yok: hedefliler bant izlerinden
            assert labels[(s.seq, s.frame, s.track_id)].label == "band"


def test_targeted_includes_transitions():
    gt = _gt(40, 120)
    frames = {"s": list(range(1, 121))}
    # her izde 60. karede duran → hareketli geçişi; bant yok
    labels = _labels(gt, lambda t, f: "stationary" if f < 60 else "moving")
    a = sample(population(gt, frames, k=15), labels, seed=42)
    targeted = [s for s in a if s.kind == "targeted"]
    assert targeted and all(abs(s.frame - 60) <= 15 for s in targeted)


def test_page_hides_labels(tmp_path: Path):
    tracks = {1: {"start": (60, 60), "speed": (0, 0)}, 2: {"start": (150, 140), "speed": (3, 0)}}
    frames, gt = make_video(n=40, tracks=tracks)
    d = write_sequence(tmp_path / "dev", "syn", frames, gt)
    seq = load_sequence(d, allow_test=False)
    samples = [
        Sample("syn", 20, 1, "uniform"),
        Sample("syn", 21, 2, "targeted"),
    ]
    out = tmp_path / "audit"
    ids = render(samples, {"syn": seq}, {"syn": gt}, out, k=15)
    html = (out / "index.html").read_text(encoding="utf-8")
    names = " ".join(p.name for p in (out / "crops").iterdir())
    for word in ("band", "v_ref", "uniform", "targeted", "unregistered", "syn"):
        assert word not in html, word
    for word in ("moving", "stationary", "band", "uniform", "targeted", "syn"):
        assert word not in names, word
    assert re.search(r"a\d{3}", html)  # örnek kimlikleri yalnız sıra numarası
    assert len(ids) == 2 and set(ids.values()) == {s for s in samples}
    assert len(list((out / "crops").iterdir())) == 2 * 11 * 2  # örnek × kare × kanal
    assert "http://" not in html and "https://" not in html  # internetsiz


def _key(n_uniform: int, n_targeted: int) -> list[Sample]:
    s = [Sample("s", 100 + i, i, "uniform") for i in range(n_uniform)]
    s += [Sample("s", 300 + i, 500 + i, "targeted") for i in range(n_targeted)]
    return s


def test_score_thresholds():
    key = _key(80, 2)
    ids = {f"a{i + 1:03d}": s for i, s in enumerate(key)}
    labels: dict[tuple[str, int, int], LabelRow] = {}
    answers: dict[str, str] = {}
    # 80 düzgün: 12 emin değilim (%15); 68 insan-kararlı, bunların 53'ü etiketleyicide etiketli
    # (15'i bant), 53'ün 51'i uyumlu
    for i, (sid, s) in enumerate(list(ids.items())[:80]):
        if i < 12:
            ans, lab = "unsure", "moving"
        elif i < 27:
            ans, lab = "moving", "band"
        elif i < 29:
            ans, lab = "moving", "stationary"  # uyumsuz
        else:
            ans, lab = ("moving", "moving") if i % 2 else ("stationary", "stationary")
        answers[sid] = ans
        labels[(s.seq, s.frame, s.track_id)] = LabelRow(s.seq, s.frame, s.track_id, 0.0, lab)
    for sid, s in list(ids.items())[80:]:
        answers[sid] = "moving"
        labels[(s.seq, s.frame, s.track_id)] = LabelRow(s.seq, s.frame, s.track_id, None, "band")
    r = score(ids, labels, answers)
    assert r["a_agreement"]["k"] == 51 and r["a_agreement"]["n"] == 53
    assert r["a_agreement"]["pass"] is True  # %96,2 ≥ %95
    assert r["b_unsure"]["k"] == 12 and r["b_unsure"]["n"] == 80
    assert r["b_unsure"]["pass"] is True  # %15 ≤ %15 (sınır dahil)
    assert r["c_covered"]["k"] == 53 and r["c_covered"]["n"] == 68
    assert r["c_covered"]["pass"] is False  # %77,9 < %80
    assert r["verdict"] == "fail"
    assert r["targeted"]["moving"]["band"] == 2


def test_wilson_interval():
    lo, hi = wilson(62, 65)
    assert lo == pytest.approx(0.8731, abs=1e-3)
    assert hi == pytest.approx(0.9842, abs=1e-3)


def test_answers_must_cover_every_sample():
    key = _key(2, 0)
    ids = {"a001": key[0], "a002": key[1]}
    labels = {
        (s.seq, s.frame, s.track_id): LabelRow(s.seq, s.frame, s.track_id, 0.0, "moving")
        for s in key
    }
    with pytest.raises(ValueError, match="cevap"):
        score(ids, labels, {"a001": "moving"})
    with pytest.raises(ValueError, match="cevap"):
        score(ids, labels, {"a001": "moving", "a002": "evet"})


def test_audit_score_cli_exit_code(tmp_path: Path):
    from baykus.motion.cli import main

    key = {"a001": {"seq": "s", "frame": 1, "track_id": 1, "kind": "uniform"}}
    (tmp_path / "key.json").write_text(json.dumps({"samples": key}), encoding="utf-8")
    (tmp_path / "labels_val_s.csv").write_text(
        "seq,frame,track_id,v_ref,label\ns,1,1,0.5,moving\n", encoding="utf-8"
    )
    (tmp_path / "ans.json").write_text(
        json.dumps([{"id": "a001", "answer": "stationary"}]), encoding="utf-8"
    )
    out = tmp_path / "audit.json"
    code = main(
        [
            "audit-score",
            "--key",
            str(tmp_path / "key.json"),
            "--labels-dir",
            str(tmp_path),
            "--answers",
            str(tmp_path / "ans.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 1  # tek örnek, uyumsuz → fail
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "fail"


def _render_one(tmp_path: Path, speed: tuple[float, float]):
    tracks = {1: {"start": (160, 128), "speed": speed}}
    frames, gt = make_video(n=40, cam=(2.0, 1.0), tracks=tracks)
    d = write_sequence(tmp_path / "dev", "syn", frames, gt)
    seq = load_sequence(d, allow_test=False)
    out = tmp_path / "audit"
    ids = render([Sample("syn", 20, 1, "uniform")], {"syn": seq}, {"syn": gt}, out, k=15)
    sid = next(iter(ids))
    from baykus.io import imread_any

    def crop(j: int, ch: str):
        return imread_any(out / "crops" / f"{sid}_{j:02d}_{ch}.jpg", cv2.IMREAD_GRAYSCALE)

    return crop, frames, gt


def _background_change(crop) -> float:
    a, b = crop(0, "ir").astype(float), crop(10, "ir").astype(float)
    ring = np.ones_like(a, bool)
    ring[70:122, 70:122] = False  # aracın çevresi hariç arka plan
    return float(np.abs(a - b)[ring].mean())


def test_crop_follows_vehicle(tmp_path: Path):
    """Pencere araca ortalanır: duran araçta arka plan kıpırdamaz, hareketlide kayar (kamera da
    kayıyor)."""
    still = _background_change(_render_one(tmp_path / "a", (0, 0))[0])
    moving = _background_change(_render_one(tmp_path / "b", (4, 0))[0])
    assert still < 6.0
    assert moving > 3 * still


def test_rgb_crop_has_no_box_over_vehicle(tmp_path: Path):
    """RGB kutusu IR etiketinden geldiği için kayar (kullanıcı geri bildirimi): RGB'de kutu
    çizilmez, yalnız aracın dışında köşe işaretleri."""
    crop, frames, gt = _render_one(tmp_path, (0, 0))
    x1, y1, x2, y2 = gt[1][20]  # j = 5 → t − 15 + 15 = 20
    want = centered(frames[20], (x1 + x2) / 2, (y1 + y2) / 2).astype(float)
    got = crop(5, "rgb").astype(float)
    c = slice(CENTER - 15, CENTER + 15)
    assert np.abs(got[c, c] - want[c, c]).mean() < 4.0  # aracın üstünde çizgi yok
    corner = (slice(CENTER - 23, CENTER - 20), slice(CENTER - 23, CENTER - 14))
    assert np.abs(got[corner] - want[corner]).max() > 60  # köşe işareti var


def test_sample_excludes_previous_round():
    gt = _gt(60, 100)
    frames = {"s": list(range(1, 101))}
    labels = _labels(gt, lambda t, f: "band" if t % 2 == 0 else "moving")
    pop = population(gt, frames, k=15)
    first = sample(pop, labels, seed=42)
    seen = {(s.seq, s.frame, s.track_id) for s in first}
    second = sample(pop, labels, seed=43, exclude=seen)
    assert not seen & {(s.seq, s.frame, s.track_id) for s in second}
    assert len(second) == 120


def test_page_states_majority_rule(tmp_path: Path):
    """Klibin bir kısmında duran, bir kısmında hareket eden araç için kural yazılı (1. tur notu)."""
    crop, frames, gt = _render_one(tmp_path, (0, 0))
    html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    assert "çoğunlukla" in html and "yarı yarıya" in html


def test_crop_outside_image_is_black():
    """Görüntü dışına taşan pencere siyah dolgulu (tekrarlı kenar çizgileri dikkat dağıtıyordu)."""
    img = np.full((100, 120), 200, np.uint8)
    c = centered(img, 5.0, 5.0)
    assert c.shape == (192, 192)
    assert c[0:80, 0:80].max() == 0  # sol üst: görüntü dışı
    assert c[CENTER + 10, CENTER + 10] == 200  # görüntü içi
