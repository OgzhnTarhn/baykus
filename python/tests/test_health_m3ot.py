import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.engine_ref.models import Models
from baykus.health.cli import main as health_main
from baykus.health.gate import assess
from baykus.health.m3ot import (
    ALARM_IDS,
    CONDITIONS,
    alarm_states,
    corrupt,
    delay,
    demo_criterion,
    pick_demo,
    run_sequence,
)
from baykus.io import imwrite_any
from baykus.robustness.corruptions import shift
from tests.test_engine_ref import CALIB_X, CALIB_Y, GM, ROWS, FakeBackend, _raw
from tests.test_health_features import scene
from tests.test_health_gate import OK, TH


def test_conditions_fixed():
    assert CONDITIONS == ("clean", "misalign-20", "misalign-20@K", "noise-20", "drop_ir")
    assert ALARM_IDS == ("rgb_lost", "ir_lost", "ir_noisy", "misaligned")


def test_delay_counts_from_k():
    mis = [False] * 6 + [False, False, True, True, True, True]
    assert delay(mis, 6) == (2, False)


def test_delay_null_when_never():
    assert delay([False] * 12, 6) == (None, False)


def test_alarm_before_onset_flag():
    mis = [False] * 5 + [True] + [True] * 6
    assert delay(mis, 6) == (0, True)


def test_misalign_constant_vector():
    g = scene()
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for f in range(4):
        _, ir = corrupt(bgr, g, "misalign-20", f, 2, f"s_{f:06d}")
        assert np.array_equal(ir, shift(g, 20, 0))
        _, ir_k = corrupt(bgr, g, "misalign-20@K", f, 2, f"s_{f:06d}")
        assert np.array_equal(ir_k, shift(g, 20, 0) if f >= 2 else g)


def test_noise_is_seeded_by_stem():
    g = scene()
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    a = corrupt(bgr, g, "noise-20", 0, 0, "s_000001")[1]
    assert np.array_equal(a, corrupt(bgr, g, "noise-20", 5, 0, "s_000001")[1])
    assert not np.array_equal(a, corrupt(bgr, g, "noise-20", 0, 0, "s_000002")[1])


def test_drop_ir_and_clean():
    g = scene()
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    assert not corrupt(bgr, g, "drop_ir", 0, 0, "s")[1].any()
    b2, i2 = corrupt(bgr, g, "clean", 0, 0, "s")
    assert np.array_equal(b2, bgr) and np.array_equal(i2, g)


def _h(**over) -> dict:
    return assess({**OK, **over}, TH)


def test_alarm_states_open_clear_reset():
    lost, ok = _h(ir_std=0.0, misalign_px=None), _h()
    seq = [lost, lost, lost, ok, ok, ok, ok]
    got = alarm_states(seq)
    # son 5 karenin 3'ü: 3. karede açılır; 6. karede pencere (2..6) 2 kayıp → kapanır
    assert [("ir_lost" in s) for s in got] == [False, False, True, True, True, False, False]
    assert alarm_states(seq[3:]) == [[], [], [], []]  # yeni oturum: boş pencere


def test_alarm_states_misaligned_is_current_frame():
    mis = _h(misalign_px=30.0)
    assert alarm_states([mis, _h(), mis]) == [["misaligned"], [], ["misaligned"]]


def test_alarm_order_is_fixed():
    both = _h(ir_noise=12.0, misalign_px=30.0)
    assert alarm_states([both] * 3)[-1] == ["ir_noisy", "misaligned"]


def test_demo_criterion():
    rates = {"clean": 0.02, "misalign-20": 0.9}
    assert demo_criterion(rates, 10)["passed"] is True
    assert demo_criterion({**rates, "clean": 0.021}, 10)["clean_le_2pct"] is False
    assert demo_criterion(rates, None)["delay_le_10"] is False
    assert demo_criterion(rates, 11)["passed"] is False


def _seq(root: Path, name: str, split: str = "val", n: int = 12) -> Path:
    s = root / "sequences" / name
    (s / "rgb").mkdir(parents=True)
    (s / "ir").mkdir()
    g = scene()
    for f in range(1, n + 1):
        imwrite_any(s / "rgb" / f"{f:06d}.jpg", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        imwrite_any(s / "ir" / f"{f:06d}.png", 255 - g)
    idx = {"name": name, "split": split, "frames": list(range(1, n + 1)), "fps": 30}
    (s / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    labels = root / "labels"
    labels.mkdir(exist_ok=True)
    p = labels / f"{split}_ir.json"
    coco = (
        json.loads(p.read_text(encoding="utf-8"))
        if p.exists()
        else {"images": [], "annotations": [], "categories": [{"id": 1, "name": "vehicle"}]}
    )
    for f in range(1, n + 1):
        i = len(coco["images"]) + 1
        coco["images"].append(
            {"id": i, "file_name": f"{name}_{f:06d}", "width": 640, "height": 512}
        )
        coco["annotations"].append(
            {"id": i, "image_id": i, "category_id": 1, "bbox": [200, 36, 60, 50], "area": 3000,
             "iscrowd": 0}
        )  # fmt: skip
    p.write_text(json.dumps(coco), encoding="utf-8")
    return s


def _models() -> Models:
    return Models(Path("e3c.onnx"), TH, CALIB_X, CALIB_Y, GM)


def test_run_sequence_shape(tmp_path: Path):
    s = _seq(tmp_path, "1_08")
    r = run_sequence(s, tmp_path / "labels", _models(), backend=FakeBackend(_raw(ROWS)))
    assert r["seq"] == "1_08" and r["n_frames"] == 12 and r["k"] == 6
    assert set(r["rules"]) == {"phase2", "frame", "frame+temporal"}
    for rule in r["rules"].values():
        assert set(rule) == set(CONDITIONS)
        mk = rule["misalign-20@K"]
        assert "delay" in mk and "alarm_before_onset" in mk
        for c in rule.values():
            assert set(c["alarm_rates"]) == set(ALARM_IDS)
            assert {"ap50_auto", "ap50_off", "ap50_diff", "gate_rate"} <= set(c)
    assert set(r["demo"]) == {"clean_le_2pct", "misalign_ge_90pct", "delay_le_10", "passed"}
    json.dumps(r, allow_nan=False)


def test_drop_ir_auto_gate_changes_detection_input(tmp_path: Path):
    s = _seq(tmp_path, "1_08", n=4)
    fake = FakeBackend(_raw(ROWS))
    r = run_sequence(s, tmp_path / "labels", _models(), backend=fake)
    assert r["rules"]["frame"]["drop_ir"]["gate_rate"] == 1.0  # IR kayıp → ir_off


def test_test_split_needs_flag(tmp_path: Path, capsys):
    root = tmp_path / "m3ot_test"
    _seq(root, "1_03", split="test")
    rc = health_main(["m3ot", "--test", "--data", str(root), "--models", "x", "--out", "o.json"])
    assert rc == 2 and "--i-know-this-is-test" in capsys.readouterr().err


def _result(name: str, passed: bool) -> dict:
    return {"seq": name, "demo": {"passed": passed}}


def test_pick_demo_first_passing(tmp_path: Path):
    cfg = tmp_path / "m3ot_dev.json"
    cfg.write_text(
        json.dumps({"split": "val", "sequences": ["1_08", "2_08"], "demo": None}), "utf-8"
    )
    choice = pick_demo([_result("2_08", True), _result("1_08", False)], cfg)
    assert choice == {"demo": "2_08", "criterion_passed": True}
    assert json.loads(cfg.read_text(encoding="utf-8"))["demo"] == "2_08"


def test_pick_demo_falls_back_to_first(tmp_path: Path):
    cfg = tmp_path / "m3ot_dev.json"
    cfg.write_text(
        json.dumps({"split": "val", "sequences": ["1_08", "2_08"], "demo": None}), "utf-8"
    )
    choice = pick_demo([_result("1_08", False), _result("2_08", False)], cfg)
    assert choice == {"demo": "1_08", "criterion_passed": False}


@pytest.mark.parametrize("cond", ["misalign-20@X", "blur-3"])
def test_unknown_condition_is_error(cond):
    g = scene()
    with pytest.raises(ValueError):
        corrupt(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), g, cond, 0, 0, "s")
