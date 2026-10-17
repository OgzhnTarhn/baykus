import json
from pathlib import Path

import pytest

from baykus.health.cli import main
from baykus.health.features import FEATURES
from baykus.health.fit import STATES, feature_rows, fit_thresholds, youden
from baykus.robustness.corruptions import suite_conditions
from tests.synthetic import make_synthetic_dataset


def test_youden_above_picks_max_j():
    r = youden([10, 12, 14], [1, 2, 3, 11], "above")
    assert r["threshold"] == 6.5 and r["tpr"] == 1.0 and r["fpr"] == 0.25
    assert r["j"] == pytest.approx(0.75) and (r["n_pos"], r["n_neg"]) == (3, 4)


def test_youden_below_separates_lost_camera():
    r = youden([0.0, 0.0], [5.0, 7.0], "below")
    assert r["threshold"] == 2.5 and r["tpr"] == 1.0 and r["fpr"] == 0.0


def test_youden_tie_prefers_fewer_alarms():
    assert youden([10, 30], [0, 20], "above")["threshold"] == 25  # J 0,5 iki yerde; fpr 0 olan
    assert youden([0, 20], [10, 30], "below")["threshold"] == 5


def test_youden_drops_none_and_nan_and_rejects_empty():
    assert youden([None, 10.0], [0.0, float("nan")], "above")["n_pos"] == 1
    assert youden([None, 10.0], [0.0, float("nan")], "above")["n_neg"] == 1
    with pytest.raises(ValueError):
        youden([None], [1.0], "above")
    with pytest.raises(ValueError):
        youden([1.0], [1.0], "sideways")


def test_states_are_preregistered():
    assert STATES["ir_noisy"].positives == ("noise-20",)
    assert STATES["ir_noisy"].negatives == ("clean", "noise-5", "noise-10")
    assert STATES["misaligned"].positives == ("misalign-20", "misalign-40")
    assert STATES["misaligned"].negatives == ("clean", "misalign-5", "misalign-10")
    assert STATES["rgb_lost"].positives == ("drop_rgb",) and STATES["rgb_lost"].negatives is None
    assert STATES["ir_lost"].positives == ("drop_ir",) and STATES["ir_lost"].negatives is None
    assert STATES["rgb_blurry"].positives == ("blur-7", "blur-11")
    assert STATES["ir_blurry"].negatives == ("clean", "blur-3")
    ids = {c.id for c in suite_conditions()}
    for s in STATES.values():
        assert set(s.positives) <= ids and set(s.negatives or ()) <= ids
        assert s.feature in FEATURES


def _rows() -> list[dict]:
    base = {
        "rgb_std": 40.0,
        "rgb_noise": 3.0,
        "rgb_blur": 300.0,
        "rgb_brightness": 120.0,
        "ir_std": 30.0,
        "ir_noise": 3.0,
        "ir_blur": 200.0,
        "misalign_px": 3.0,
        "misalign_response": 0.3,
    }
    over = {
        "noise-20": {"ir_noise": 15.0},
        "drop_ir": {"ir_std": 0.0, "ir_noise": 0.0, "ir_blur": 0.0, "misalign_px": None},
        "drop_rgb": {"rgb_std": 0.0, "rgb_blur": 0.0, "misalign_px": None},
        "misalign-20": {"misalign_px": 21.0},
        "misalign-40": {"misalign_px": 41.0},
        "blur-7": {"rgb_blur": 20.0, "ir_blur": 20.0},
        "blur-11": {"rgb_blur": 10.0, "ir_blur": 10.0},
    }
    return [
        {"stem": s, "condition": c.id, **base, **over.get(c.id, {})}
        for s in ("a", "b")
        for c in suite_conditions()
    ]


def test_fit_thresholds_on_rows():
    out = fit_thresholds(_rows(), dark_brightness=48.66)
    assert set(out["states"]) == set(STATES) and out["dark_brightness"] == 48.66
    assert out["states"]["ir_noisy"]["threshold"] == 9.0
    assert out["states"]["misaligned"]["threshold"] == 12.0
    assert out["states"]["ir_lost"]["threshold"] == 15.0  # 0 ile 30 arası
    assert out["states"]["ir_noisy"]["train"]["tpr"] == 1.0
    assert out["states"]["ir_noisy"]["direction"] == "above"
    assert out["states"]["rgb_lost"]["negatives"] is None
    assert out["train_clean_gate_rate"] == 0.0


def test_feature_rows_deterministic_and_parallel_equal(tmp_path: Path):
    root = make_synthetic_dataset(tmp_path / "d", n_train=3, with_ir=True)
    stems = ["00001", "00002", "00003"]
    a = feature_rows(root, "train", stems, seed=42)
    conds = [c.id for c in suite_conditions()]
    assert len(a) == 3 * len(conds)
    assert [r["condition"] for r in a[: len(conds)]] == conds
    assert [r["stem"] for r in a[:: len(conds)]] == stems
    assert set(a[0]) == {"stem", "condition", *FEATURES}
    assert a == feature_rows(root, "train", stems, seed=42, workers=2)


def test_cli_fit_writes_json(tmp_path: Path):
    root = make_synthetic_dataset(tmp_path / "d", n_train=3, with_ir=True)
    (root / "meta/illumination_thresholds.json").write_text(
        json.dumps({"t_dark": 48.66, "t_day": 114.14}), encoding="utf-8"
    )
    out, feats = tmp_path / "res/health/thresholds.json", tmp_path / "f.csv"
    args = ["fit", "--data-root", str(root), "--meta", str(root / "meta"), "--out", str(out)]
    assert main([*args, "--features-out", str(feats), "--n", "3"]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["source"]["split"] == "train" and rec["source"]["n"] == 3
    assert rec["source"]["seed"] == 42 and len(rec["source"]["stems_sha256"]) == 64
    assert set(rec["states"]) == set(STATES) and rec["dark_brightness"] == 48.66
    lines = feats.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 3 * len(suite_conditions())


def test_cli_fit_without_illumination_thresholds_is_error(tmp_path: Path, capsys):
    root = make_synthetic_dataset(tmp_path / "d", n_train=3, with_ir=True)
    args = ["fit", "--data-root", str(root), "--meta", str(root / "meta")]
    assert main([*args, "--out", str(tmp_path / "t.json"), "--n", "3"]) == 2
    assert capsys.readouterr().err.startswith("hata: ")


def test_youden_exact_tie_is_not_broken_by_float_rounding():
    # İki aday matematikte J = 0,3: (tpr 1/2, fpr 2/10) ve (tpr 2/2, fpr 7/10).
    # Kayan noktada ikincisi 0,30000000000000004 çıkar; kural: eşitlikte düşük FPR.
    pos = [100.0, 50.0]
    neg = [100.0] * 2 + [50.0] * 5 + [0.0] * 3
    r = youden(pos, neg, "above")
    assert r["threshold"] == 75.0 and r["fpr"] == 0.2 and r["tpr"] == 0.5
