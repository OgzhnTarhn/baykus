import json

import numpy as np
import pytest

from baykus.engine_ref.fixtures import (
    calibration_probe,
    check_margins,
    clip_violations,
    generate,
)
from baykus.health.gate import assess
from baykus.health.guard import GateModel
from tests.test_engine_ref import CALIB_X, CALIB_Y, ROWS, FakeBackend, _raw
from tests.test_health_gate import OK, TH
from tests.test_health_m3ot import _models, _seq

GM = GateModel(0.2, 5, 3, 3.0, "g", TH.sha256)


def _h(**over) -> dict:
    return assess({**OK, **over}, TH, GM.r_min)


def test_margin_ok_far_from_thresholds():
    assert check_margins(_h(), GM, TH) == []


def test_margin_rejects_near_threshold():  # misalign_px = t + 1 → ihlal
    v = check_margins(_h(misalign_px=TH.misaligned + 1, misalign_response=0.9), GM, TH)
    assert any("misalign_px" in s for s in v)


def test_margin_rejects_relative_feature():
    v = check_margins(_h(ir_noise=TH.ir_noisy * (1 + 5e-4)), GM, TH)
    assert any("ir_noise" in s for s in v)


def test_margin_response_only_near_misalignment():
    near = _h(misalign_px=TH.misaligned + 10, misalign_response=GM.r_min + 0.1)
    assert any("response" in s for s in check_margins(near, GM, TH))
    far = _h(misalign_px=TH.misaligned - 6, misalign_response=GM.r_min + 0.01)
    assert check_margins(far, GM, TH) == []


def test_response_irrelevant_below_t_minus_5():
    for r in np.linspace(0, 1, 21):
        h = _h(misalign_px=TH.misaligned - 5 - 1e-6, misalign_response=float(r))
        assert h["pair"]["misaligned_frame"] is False


def _flagged(dx, dy=0.0):
    return _h(misalign_px=30.0, misalign_response=0.9, misalign_dx=dx, misalign_dy=dy)


def test_clip_margin_on_S_only():
    # S = {(0,0), (0,0), (3,0)}: üçüncünün ortancaya uzaklığı 3 → |3 − 3| < 2: ihlal
    assert clip_violations([_flagged(0), _flagged(0), _flagged(3)], GM)
    # işaretsiz kare S'ye girmez: vektörü ne olursa olsun ihlal sayılmaz
    clean = _h(misalign_dx=3.0, misalign_dy=0.0)
    assert clip_violations([_flagged(0), _flagged(0), clean], GM) == []
    assert clip_violations([_flagged(0), _flagged(0), _flagged(5.5)], GM) == []


def test_probe_values_are_float32_neighbors():
    probes = calibration_probe(CALIB_X, CALIB_Y)
    assert len(probes) == len(CALIB_X)
    for p, x in zip(probes, CALIB_X, strict=True):
        f = np.float32(x)
        want = [np.nextafter(f, np.float32(-np.inf)), f, np.nextafter(f, np.float32(np.inf))]
        assert [q["score"] for q in p["probes"]] == [float(w) for w in want]
        for q in p["probes"]:
            idx = np.clip(np.searchsorted(CALIB_X, q["score"], side="left"), 0, len(CALIB_X) - 1)
            assert q["calibrated"] == float(CALIB_Y[idx])


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("fx")
    dev = root / "m3ot_dev"
    _seq(dev, "1_08", n=40)
    cfg = {"split": "val", "sequences": ["1_08"], "demo": None}
    out = root / "fixtures"
    index = generate(_models(), dev, cfg, out, backend=FakeBackend(_raw(ROWS)), n_frames=3)
    return out, index


def test_frame_fixture_has_modules_and_pipeline(generated):
    out, index = generated
    frames = {e["name"]: e for e in index if e["kind"] == "frame"}
    noise = json.loads((out / frames["noise_ir"]["path"]).read_text(encoding="utf-8"))
    assert set(noise) >= {"inputs", "modules", "pipeline"}
    assert noise["modules"]["health"]["gate"] == "ir_off"
    assert noise["pipeline"]["applied"] is True
    m = noise["modules"]
    assert len(m["raw"]) == 300 and len(m["raw"][0]) == 6
    assert set(m["dets"]) == {"eval", "track", "display"}
    assert set(m["letterbox"]) == {"gain", "pad_x", "pad_y", "sha256"}


def test_index_lists_every_file_with_kind(generated):
    out, index = generated
    listed = {f for e in index for f in [e["path"], *e.get("files", [])]}
    docs = {"index.json", "m3ot/NOTICE.md", "m3ot/README.md"}
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - docs
    assert on_disk == listed
    kinds = {e["kind"] for e in index}
    assert {"frame", "clip", "calibration_probe", "temporal_unit", "alarm_sequences"} <= kinds
    assert json.loads((out / "index.json").read_text(encoding="utf-8")) == index


def test_resize_fixture_uses_resize_path(generated):
    out, index = generated
    e = next(e for e in index if e["name"] == "resize_800x600")
    rec = json.loads((out / e["path"]).read_text(encoding="utf-8"))
    assert rec["modules"]["letterbox"]["gain"] != 1.0


def test_clip_fixture_frames_and_alarms(generated):
    out, index = generated
    clips = [e for e in index if e["kind"] == "clip"]
    assert {e["name"] for e in clips} == {"clip_clean", "clip_shift_all", "clip_shift_from8"}
    rec = json.loads((out / clips[0]["path"]).read_text(encoding="utf-8"))
    assert len(rec["frames"]) == 15 and set(rec["frames"][0]) == {"frame", "health", "alarms"}


def test_generate_refuses_test_split(tmp_path):
    with pytest.raises(ValueError, match="test"):
        generate(_models(), tmp_path, {"split": "test", "sequences": []}, tmp_path / "o")


def test_calib_tables_from_engine_ref_models():
    assert CALIB_X.dtype == np.float64 and CALIB_Y.dtype == np.float64


def test_misalign_margin_irrelevant_when_response_far_below_r_min():
    # yanıt r_min − 0,2'nin altında: kayma ne olursa olsun karar "hizalı", sınır payı gerekmez
    low = _h(misalign_px=TH.misaligned + 0.5, misalign_response=GM.r_min - 0.2 - 1e-6)
    assert low["pair"]["misaligned_frame"] is False
    assert check_margins(low, GM, TH) == []
    edge = _h(misalign_px=TH.misaligned + 0.5, misalign_response=GM.r_min - 0.2 + 1e-3)
    assert any("misalign_px" in s for s in check_margins(edge, GM, TH))
