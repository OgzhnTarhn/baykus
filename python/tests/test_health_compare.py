import hashlib
import json
from pathlib import Path

import pytest

from baykus.eval.predictions import save_predictions
from baykus.health.compare import HARMFUL, acceptance, detection_rates, gate_report
from baykus.health.suite import HealthError, gate_name, summary_path
from baykus.runinfo import write_json
from tests.coco_helpers import make_gt, perfect

RUN = "R"
GT = make_gt({i: [(1, [10, 10, 20, 20]), (2, [100, 100, 30, 30])] for i in range(1, 21)})
FLAGS0 = {"lost": 0, "noisy": 0, "dark": 0, "blurry": 0}


def _summary(cond: str, *, evaluated: bool, ir_off: int = 0, n: int = 20) -> dict:
    return {
        "run_id": RUN,
        "condition": cond,
        "thresholds_sha256": "s" * 64,
        "evaluated": evaluated,
        "n_images": n,
        "gate": {"none": n - ir_off, "ir_off": ir_off, "rgb_off": 0},
        "changed": ir_off if evaluated else 0,
        "rgb": dict(FLAGS0),
        "ir": {**FLAGS0, "noisy": ir_off},
        "misaligned": 0,
    }


def _setup(tmp_path: Path, split: str, conds: list[str], gated: set[str]) -> dict:
    """Taban: zararlı koşullarda görüntülerin yarısında tahmin yok; kapılı: kusursuz."""
    preds, res = tmp_path / "robust", tmp_path / "res"
    half = [d for d in perfect(GT) if d["image_id"] <= 10]
    for c in conds:
        base = half if c in HARMFUL else perfect(GT)
        save_predictions(preds / f"{RUN}_{split}_{c}" / "predictions_coco.json.gz", base)
        if c in gated:
            p = preds / gate_name(RUN, split, c) / "predictions_coco.json.gz"
            save_predictions(p, perfect(GT))
        s = _summary(c, evaluated=c in gated, ir_off=20 if c in gated else 0)
        write_json(summary_path(res, RUN, split, c), {**s, "split": split})
    return {"preds": preds, "res": res}


def _report(tmp_path, split, conds, gated, **kw):
    d = _setup(tmp_path, split, conds, gated)
    return gate_report(
        run_id=RUN,
        split=split,
        robust_preds=d["preds"],
        gt=GT,
        results=d["res"],
        n_boot=kw.pop("n_boot", 50),
        **kw,
    )


def test_acceptance_boundaries():
    a = acceptance(-0.005, [0.001, 0.2], 0.02)
    assert a["k1"]["passed"] and a["k2"]["passed"] and a["k3"]["passed"]
    assert a["k1"]["limit"] == -0.005 and a["k3"]["limit"] == 0.02
    b = acceptance(-0.0051, [0.0, 0.2], 0.0201)
    assert not (b["k1"]["passed"] or b["k2"]["passed"] or b["k3"]["passed"])
    assert acceptance(0.0, [None, None], 0.0)["k2"]["passed"] is False


def test_unchanged_condition_reuses_baseline_and_diff_is_zero(tmp_path: Path):
    rep = _report(tmp_path, "val", ["clean", *HARMFUL], set(HARMFUL))
    row = next(r for r in rep["rows"] if r["condition"] == "clean")
    assert row["diff"] == 0.0 and row["ci95"] == [0.0, 0.0] and row["significant"] is False
    assert row["base_map50"] == row["gate_map50"] == pytest.approx(1.0)
    assert row["changed"] == 0 and row["gate_rate"] == 0.0


def test_gate_gain_on_harmful_conditions_passes_k2(tmp_path: Path):
    rep = _report(tmp_path, "val", ["clean", *HARMFUL], set(HARMFUL))
    assert [r["condition"] for r in rep["rows"]] == [
        "clean",
        "noise-20",
        "misalign-20",
        "misalign-40",
    ]
    assert rep["harmful"]["conditions"] == list(HARMFUL)
    assert rep["harmful"]["diff"] > 0 and rep["harmful"]["ci95"][0] > 0
    acc = rep["acceptance"]
    assert acc["k1"]["passed"] and acc["k2"]["passed"] and acc["k3"]["value"] == 0.0
    assert rep["n_boot"] == 50 and rep["seed"] == 42 and rep["thresholds_sha256"] == "s" * 64


def test_missing_baseline_preds_is_error_before_bootstrap(tmp_path: Path, monkeypatch):
    d = _setup(tmp_path, "val", ["clean", "noise-20"], {"noise-20"})
    (d["preds"] / f"{RUN}_val_noise-20" / "predictions_coco.json.gz").unlink()
    monkeypatch.setattr(
        "baykus.health.compare.paired_diff_ci", lambda *a, **k: pytest.fail("bootstrap")
    )
    with pytest.raises(HealthError, match="noise-20"):
        gate_report(run_id=RUN, split="val", robust_preds=d["preds"], gt=GT, results=d["res"])


def test_missing_gated_preds_is_error(tmp_path: Path):
    d = _setup(tmp_path, "val", ["clean", "noise-20"], {"noise-20"})
    (d["preds"] / gate_name(RUN, "val", "noise-20") / "predictions_coco.json.gz").unlink()
    with pytest.raises(HealthError, match="_gate"):
        gate_report(run_id=RUN, split="val", robust_preds=d["preds"], gt=GT, results=d["res"])


def test_no_summaries_is_error(tmp_path: Path):
    with pytest.raises(HealthError, match="özet"):
        gate_report(run_id=RUN, split="val", robust_preds=tmp_path, gt=GT, results=tmp_path)


def test_test_split_has_no_acceptance_and_partial_z(tmp_path: Path):
    rep = _report(
        tmp_path, "test", ["clean", "noise-20", "misalign-20"], {"noise-20", "misalign-20"}
    )
    assert rep["acceptance"] is None
    assert rep["harmful"]["conditions"] == ["noise-20", "misalign-20"]


def test_val_without_all_harmful_conditions_has_no_acceptance(tmp_path: Path):
    rep = _report(tmp_path, "val", ["clean", "noise-20"], {"noise-20"})
    assert rep["acceptance"] is None


def test_detection_rates():
    s = {
        "noise-20": {
            "n_images": 4,
            "gate": {"none": 1, "ir_off": 3, "rgb_off": 0},
            "rgb": dict(FLAGS0),
            "ir": {**FLAGS0, "noisy": 3},
            "misaligned": 1,
        }
    }
    r = detection_rates(s)["noise-20"]
    assert r["ir"]["noisy"] == 0.75 and r["gate_rate"] == 0.75 and r["misaligned"] == 0.25
    assert r["rgb"]["lost"] == 0.0


def test_cli_compare_writes_json(tmp_path: Path, capsys):
    from baykus.health.cli import main

    d = _setup(tmp_path, "val", ["clean", *HARMFUL], set(HARMFUL))
    _with_thresholds_file(d["res"])
    coco = tmp_path / "dv" / "coco"
    coco.mkdir(parents=True)
    (coco / "val_ir.json").write_text(json.dumps(GT.dataset), encoding="utf-8")
    args = ["compare", "--run-id", RUN, "--split", "val", "--robust-preds", str(d["preds"])]
    args += ["--data-root", str(tmp_path / "dv"), "--results", str(d["res"]), "--n-boot", "20"]
    assert main(args) == 0
    rec = json.loads((d["res"] / "health" / "gate_val.json").read_text(encoding="utf-8"))
    assert rec["acceptance"]["k2"]["passed"] is True
    assert "Kabul 2" in capsys.readouterr().out


def test_detection_rates_empty_condition_is_null():
    s = {"clean": {"n_images": 0, "gate": {"none": 0, "ir_off": 0, "rgb_off": 0},
                   "rgb": dict(FLAGS0), "ir": dict(FLAGS0), "misaligned": 0}}  # fmt: skip
    r = detection_rates(s)["clean"]
    assert r["gate_rate"] is None and r["ir"]["noisy"] is None and r["misaligned"] is None


def _with_thresholds_file(res: Path) -> Path:
    """results/health/thresholds.json yazar ve özetlerin sha256'sını ona eşitler."""
    th = res / "health" / "thresholds.json"
    th.write_text('{"states": {}}', encoding="utf-8")
    sha = hashlib.sha256(th.read_bytes()).hexdigest()
    for p in (res / "health").glob("*_health.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        write_json(p, {**rec, "thresholds_sha256": sha})
    return th


def test_gate_report_rejects_other_thresholds_file(tmp_path: Path):
    d = _setup(tmp_path, "val", ["clean", "noise-20"], {"noise-20"})
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(HealthError, match="eşik"):
        gate_report(run_id=RUN, split="val", robust_preds=d["preds"], gt=GT, results=d["res"],
                    thresholds=other)  # fmt: skip


def test_cli_compare_without_thresholds_file_is_error(tmp_path: Path, capsys):
    from baykus.health.cli import main

    d = _setup(tmp_path, "val", ["clean"], set())
    args = ["compare", "--run-id", RUN, "--robust-preds", str(d["preds"])]
    assert main([*args, "--data-root", str(tmp_path), "--results", str(d["res"])]) == 2
    assert "thresholds.json" in capsys.readouterr().err


def test_acceptance_with_uncomputable_false_alarm_is_not_a_pass():
    k3 = acceptance(0.0, [0.1, 0.2], None)["k3"]
    assert k3 == {"value": None, "limit": 0.02, "passed": False}


def test_gate_report_checks_results_thresholds_file_by_default(tmp_path: Path):
    d = _setup(tmp_path, "val", ["clean", "noise-20"], {"noise-20"})
    (d["res"] / "health" / "thresholds.json").write_text("{}", encoding="utf-8")  # başka eşikler
    with pytest.raises(HealthError, match="eşik"):
        gate_report(run_id=RUN, split="val", robust_preds=d["preds"], gt=GT, results=d["res"])
