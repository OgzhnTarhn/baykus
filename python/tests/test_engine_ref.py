import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.edge.frames import Letterbox, postprocess
from baykus.engine_ref.cli import main as engine_main
from baykus.engine_ref.compose import decode_pair
from baykus.engine_ref.models import Models, load_models
from baykus.engine_ref.pipeline import Det, Pipeline, decode_rows, display, order
from baykus.health.calibrate import apply_isotonic
from baykus.health.gate import Thresholds
from baykus.health.guard import GateModel
from tests.test_health_features import scene
from tests.test_health_gate import TH

CALIB_X = np.array([0.1, 0.25, 0.3, 1.0])
CALIB_Y = np.array([0.0, 0.2, 0.3, 0.9])
GM = GateModel(0.2, 5, 3, 3.0, "g", TH.sha256)


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def _raw(rows: list[tuple]) -> np.ndarray:
    out = np.zeros((1, 300, 6), np.float32)
    for i, r in enumerate(rows):
        out[0, i] = r
    return out


class FakeBackend:
    imgsz = 640
    channels = 4

    def __init__(self, out: np.ndarray) -> None:
        self.out = out
        self.inputs: list[np.ndarray] = []

    def infer(self, x: np.ndarray) -> np.ndarray:
        self.inputs.append(x)
        return self.out


def _models(th: Thresholds = TH) -> Models:
    return Models(Path("e3c.onnx"), th, CALIB_X, CALIB_Y, GM)


def _pair() -> tuple[bytes, bytes]:
    g = scene()
    return _jpg(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)), _png(255 - g)


ROWS = [
    (10, 70, 50, 110, 0.0005, 0),  # değerlendirme katmanının altında
    (10, 70, 50, 110, 0.05, 1),
    (100, 100, 140, 160, 0.2, 2),
    (200, 100, 260, 150, 0.3, 0),
    (300, 64, 300, 90, 0.9, 0),  # genişlik 0 → atılır
    (-20, 50, 30, 120, 0.6, 3),  # kırpılır
]


def test_decode_pair_gray_ir_from_3ch_png():
    g = scene()
    bgr, ir = decode_pair(_jpg(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)), _png(cv2.merge([g, g, g])))
    assert bgr.shape == (512, 640, 3) and ir.shape == (512, 640) and ir.dtype == np.uint8
    assert np.array_equal(ir, g)


def test_decode_pair_16bit_ir_is_8bit_gray():
    g16 = (scene().astype(np.uint16) * 257).astype(np.uint16)
    _, ir = decode_pair(_jpg(cv2.cvtColor(scene(), cv2.COLOR_GRAY2BGR)), _png(g16))
    assert ir.dtype == np.uint8 and ir.shape == (512, 640)


def test_decode_pair_size_mismatch_raises():
    with pytest.raises(ValueError, match="boyut"):
        decode_pair(_jpg(np.zeros((512, 640, 3), np.uint8)), _png(np.zeros((500, 640), np.uint8)))


def test_decode_pair_garbage_raises():
    with pytest.raises(ValueError, match="RGB"):
        decode_pair(b"not an image", _png(np.zeros((8, 8), np.uint8)))


def test_decode_rows_matches_postprocess():
    rng = np.random.default_rng(0)
    out = np.zeros((1, 300, 6), np.float32)
    out[0, :, :4] = np.sort(rng.uniform(-40, 700, (300, 2, 2)), axis=1).reshape(300, 4)
    out[0, :, 4] = rng.uniform(0, 1, 300)
    out[0, :, 5] = rng.integers(0, 5, 300)
    lb = Letterbox(1.0, 0, 64, 640, 512)
    ref = postprocess(out, lb, 0.1, image_id=0)
    got = sorted(decode_rows(out, lb, 0.1), key=lambda d: (-d.score_raw, d.index))
    assert len(got) == len(ref)
    for d, r in zip(got, ref, strict=True):
        x1, y1, x2, y2 = d.box
        assert [x1, y1, x2 - x1, y2 - y1] == pytest.approx(r["bbox"], abs=1e-9)
        assert d.score_raw == r["score"] and d.cls_id + 1 == r["category_id"]


def test_decode_rows_index_is_ort_row():
    dets = decode_rows(_raw(ROWS), Letterbox(1.0, 0, 64, 640, 512), 0.001)
    assert [d.index for d in dets] == [1, 2, 3, 5]
    assert dets[-1].box[0] == 0.0  # kırpıldı


def _det(index, raw, score):
    return Det(index, 0, raw, score, (0.0, 0.0, 1.0, 1.0))


def test_order_breaks_ties_by_raw_then_index():
    dets = [_det(5, 0.5, 0.3), _det(1, 0.6, 0.3), _det(2, 0.5, 0.3), _det(0, 0.9, 0.2)]
    assert [d.index for d in order(dets)] == [1, 2, 5, 0]


def test_floor_layers():
    p = Pipeline(_models(), backend=FakeBackend(_raw(ROWS)))
    rgb, ir = _pair()
    assert len(p.run(rgb, ir, floor=0.001)["dets_all"]) == 4
    dets = p.run(rgb, ir, floor=0.1)["dets_all"]
    assert len(dets) == 3
    shown = display(dets, 0.25)  # kalibre: 0.6 → 0.9, 0.3 → 0.3, 0.2 → 0.2
    assert [d.index for d in shown] == [5, 3]


def test_calibrated_score_is_double():
    p = Pipeline(_models(), backend=FakeBackend(_raw(ROWS)))
    for d in p.run(*_pair(), floor=0.001)["dets_all"]:
        assert type(d.score) is float and type(d.score_raw) is float
        assert d.score == apply_isotonic(np.float64(d.score_raw), CALIB_X, CALIB_Y).item()
        assert d.score_raw == float(np.float32(d.score_raw))


def _noisy_th() -> Thresholds:
    return Thresholds(**{**TH.__dict__, "ir_noisy": 0.0})  # her IR "gürültülü"


def test_auto_gate_zeroes_ir_and_sets_applied():
    fake = FakeBackend(_raw(ROWS))
    r = Pipeline(_models(_noisy_th()), backend=fake).run(*_pair(), gate="auto")
    assert r["health"]["gate"] == "ir_off" and r["health"]["applied"] is True
    assert not fake.inputs[-1][0, 3, 64:576, :].any()  # IR kanalı görüntü bölgesinde sıfır
    assert fake.inputs[-1][0, :3, 64:576, :].any()


def test_advise_never_changes_pixels():
    p = Pipeline(_models(_noisy_th()), backend=FakeBackend(_raw(ROWS)))
    adv = p.run(*_pair(), gate="advise")
    off = p.run(*_pair(), gate="off")
    assert adv["tensor_sha256"] == off["tensor_sha256"]
    assert adv["health"]["gate"] == "ir_off" and adv["health"]["applied"] is False
    assert off["health"]["gate"] == "none" and off["health"]["applied"] is False


def test_tensor_sha_is_of_letterbox_tensor():
    fake = FakeBackend(_raw(ROWS))
    r = Pipeline(_models(), backend=fake).run(*_pair())
    assert r["tensor_sha256"] == hashlib.sha256(fake.inputs[-1].tobytes()).hexdigest()
    assert r["lb"] == Letterbox(1.0, 0, 64, 640, 512) and r["raw"].shape == (1, 300, 6)


def test_unknown_gate_mode_raises():
    with pytest.raises(ValueError, match="gate"):
        Pipeline(_models(), backend=FakeBackend(_raw(ROWS))).run(*_pair(), gate="maybe")


def test_health_uses_r_min_and_vector():
    h = Pipeline(_models(), backend=FakeBackend(_raw(ROWS))).run(*_pair())["health"]
    assert {"dx_px", "dy_px", "misaligned_frame"} <= set(h["pair"]) and "reason_code" in h


def _model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "e3c.onnx").write_bytes(b"onnx")
    th = {
        "states": {
            n: {"threshold": 1.0}
            for n in ("rgb_lost", "ir_lost", "ir_noisy", "misaligned", "rgb_blurry", "ir_blurry")
        },
        "dark_brightness": 48.0,
    }
    (d / "thresholds.json").write_text(json.dumps(th), encoding="utf-8")
    calib = {"table": {"x": CALIB_X.tolist(), "y": CALIB_Y.tolist()}}
    (d / "calibration.json").write_text(json.dumps(calib), encoding="utf-8")
    sha = hashlib.sha256((d / "thresholds.json").read_bytes()).hexdigest()
    gm = {"r_min": 0.2, "temporal": {"window": 5, "min_frames": 3, "tol_px": 3.0}}
    (d / "gate_model.json").write_text(json.dumps({**gm, "thresholds_sha256": sha}), "utf-8")
    return d


def _manifest(d: Path, tmp_path: Path, override: dict[str, str] | None = None) -> Path:
    files = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in d.iterdir()
    }
    for name, sha in (override or {}).items():
        files[name]["sha256"] = sha
    m = tmp_path / "models.json"
    m.write_text(json.dumps({"files": files}), encoding="utf-8")
    return m


def test_load_models_reads_tables(tmp_path: Path):
    d = _model_dir(tmp_path)
    m = load_models(d, _manifest(d, tmp_path))
    assert m.calib_x.dtype == np.float64 and m.calib_y.tolist() == CALIB_Y.tolist()
    assert m.gate.r_min == 0.2 and m.thresholds.ir_noisy == 1.0 and m.onnx == d / "e3c.onnx"


def test_load_models_sha_mismatch_raises(tmp_path: Path):
    d = _model_dir(tmp_path)
    with pytest.raises(ValueError, match="calibration.json"):
        load_models(d, _manifest(d, tmp_path, {"calibration.json": "0" * 64}))


def test_load_models_gate_from_other_thresholds_raises(tmp_path: Path):
    d = _model_dir(tmp_path)
    raw = json.loads((d / "gate_model.json").read_text(encoding="utf-8"))
    raw["thresholds_sha256"] = "baska"
    (d / "gate_model.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="gate_model.json"):
        load_models(d, None)


def _seq(tmp_path: Path, split: str) -> Path:
    s = tmp_path / "seq"
    (s / "rgb").mkdir(parents=True)
    (s / "ir").mkdir()
    rgb, ir = _pair()
    for f in (1, 2):
        (s / "rgb" / f"{f:06d}.jpg").write_bytes(rgb)
        (s / "ir" / f"{f:06d}.png").write_bytes(ir)
    index = {"name": "1-01", "split": split, "frames": [1, 2], "fps": 30, "size": [640, 512]}
    (s / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return s


def test_cli_detect_refuses_test_split(tmp_path: Path, capsys):
    s = _seq(tmp_path, "test")
    rc = engine_main(
        ["detect", "--models", str(tmp_path / "yok"), "--seq-dir", str(s), "--out", "x.json.gz"]
    )
    assert rc == 2 and "--i-know-this-is-test" in capsys.readouterr().err


def test_cli_detect_writes_frames(tmp_path: Path, monkeypatch):
    import baykus.engine_ref.cli as cli

    d = _model_dir(tmp_path)
    monkeypatch.setattr(cli, "OrtBackend", lambda path: FakeBackend(_raw(ROWS)))
    out = tmp_path / "dets.json.gz"
    rc = engine_main(
        ["detect", "--models", str(d), "--seq-dir", str(_seq(tmp_path, "val")), "--out", str(out)]
    )
    assert rc == 0
    rec = json.loads(gzip.decompress(out.read_bytes()))
    assert rec["seq"] == "1-01" and rec["split"] == "val"
    assert [f["frame"] for f in rec["frames"]] == [1, 2]
    assert len(rec["frames"][0]["dets"]) == 4  # varsayılan floor 0,001
    assert set(rec["frames"][0]["dets"][0]) == {"index", "cls_id", "score_raw", "score", "box"}
