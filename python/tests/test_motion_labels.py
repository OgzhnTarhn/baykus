from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.motion.labels import gt_tracks, label_sequence, register_direct

from .motion_synth import MemFrames, make_video, write_sequence

CLIP = Path(__file__).resolve().parents[2] / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"


def _corner_error(h_est: np.ndarray, h_true: np.ndarray, w: int, hgt: int) -> float:
    pts = np.float32([[[0, 0]], [[w, 0]], [[w, hgt]], [[0, hgt]]])
    a = cv2.perspectiveTransform(pts, h_est)
    b = cv2.perspectiveTransform(pts, h_true)
    return float(np.max(np.linalg.norm(a - b, axis=2)))


def test_register_direct_recovers_known_shift():
    a = cv2.imread(str(CLIP / "ir" / "000001.png"), cv2.IMREAD_GRAYSCALE)
    h, w = a.shape
    t = np.float32([[1, 0, 12], [0, 1, -7]])
    b = cv2.warpAffine(a, t, (w, h))
    est = register_direct(a, b, [], [])
    assert est is not None
    assert _corner_error(est, np.vstack([t, [0, 0, 1]]).astype(np.float64), w, h) <= 0.5


def test_register_direct_blank_returns_none():
    z = np.zeros((256, 320), np.uint8)
    assert register_direct(z, z, [], []) is None


def _labels_by(rows, track):
    return {r.frame: r for r in rows if r.track_id == track}


def test_label_thresholds():
    tracks = {
        1: {"start": (60, 60), "speed": (0, 0)},  # durağan (kamera kayıyor)
        2: {"start": (150, 140), "speed": (3, 0)},  # 3 px/kare, kutu 14 → 0,21
        3: {"start": (240, 100), "speed": (0, 0.8)},  # 0,8 / 14 ≈ 0,057 → bant
    }
    frames, gt = make_video(n=40, tracks=tracks)
    rows = label_sequence(MemFrames(frames), gt, k=15, lo=0.03, hi=0.10)
    for tid, want in ((1, "stationary"), (2, "moving"), (3, "band")):
        mid = _labels_by(rows, tid)[20]
        assert mid.label == want, (tid, mid)
    assert _labels_by(rows, 1)[10].label == "edge"  # t − K yok
    assert _labels_by(rows, 1)[30].label == "edge"  # t + K yok
    assert _labels_by(rows, 2)[20].v_ref == pytest.approx(3 / 14, rel=0.05)


def test_v_ref_denominator_uses_t_plus_k():
    frames, gt = make_video(n=31, tracks={1: {"start": (100, 100), "speed": (2, 0)}})
    gt[1][31] = (gt[1][31][0], gt[1][31][1], gt[1][31][0] + 20, gt[1][31][1] + 20)  # t+K'de 20 px
    gt[1][1] = (gt[1][1][0], gt[1][1][1], gt[1][1][0] + 10, gt[1][1][1] + 10)  # t−K'de 10 px
    rows = label_sequence(MemFrames(frames), gt, k=15, lo=0.03, hi=0.10)
    r = _labels_by(rows, 1)[16]
    # kamera 30 karede (30, 15) px ilerler: durağan nokta görüntüde −(30, 15) kayar
    c0 = np.array([gt[1][1][0] + 5, gt[1][1][1] + 5]) - np.array([30.0, 15.0])
    c1 = np.array([gt[1][31][0] + 10, gt[1][31][1] + 10])
    want = float(np.linalg.norm(c1 - c0)) / 30 / 20
    assert r.v_ref == pytest.approx(want, rel=0.05)


def test_unregistered_when_registration_fails():
    frames, gt = make_video(n=31, tracks={1: {"start": (100, 100), "speed": (0, 0)}})
    frames[31] = np.zeros_like(frames[31])
    rows = label_sequence(MemFrames(frames), gt, k=15, lo=0.03, hi=0.10)
    assert _labels_by(rows, 1)[16].label == "unregistered"
    assert _labels_by(rows, 1)[16].v_ref is None


def test_gt_tracks_applies_phase3_filter(tmp_path: Path):
    frames, gt = make_video(n=2, tracks={1: {"start": (50, 50), "speed": (0, 0)}})
    write_sequence(tmp_path, "s", frames, gt)
    csv = tmp_path / "labels" / "tracks_ir.csv"
    with csv.open("a", encoding="utf-8", newline="") as fh:
        fh.write("val,ir,s,1,s_000001,9,1,1,5,5,1,1,0.2\n")  # görünürlük ≤ 0,25: süzülür
        fh.write("val,ir,s,1,s_000001,8,1,1,5,5,0,1,1.0\n")  # flag 0: süzülür
        fh.write("val,rgb,s,1,s_000001,7,1,1,5,5,1,1,1.0\n")  # RGB modalitesi: alınmaz
    tr = gt_tracks(csv, "s")
    assert set(tr) == {1}
    assert tr[1][1] == pytest.approx(gt[1][1])


def test_labels_refuse_test_split(tmp_path: Path):
    from baykus.motion.cli import main

    frames, gt = make_video(n=2)
    d = write_sequence(tmp_path, "t", frames, gt, split="test")
    out = tmp_path / "labels.csv"
    args = ["labels", "--seq-dir", str(d), "--tracks", str(tmp_path / "labels" / "tracks_ir.csv")]
    assert main([*args, "--out", str(out)]) == 2  # CLI kalıbı: ValueError → çıkış 2, ileti
    assert not out.exists()
