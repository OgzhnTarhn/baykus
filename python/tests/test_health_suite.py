import gzip
import json
from pathlib import Path

import pytest

from baykus.health.suite import HealthError, run_gated_suite, summarize_health, summary_path
from baykus.robustness.corruptions import select_conditions, suite_conditions
from tests.synthetic import make_synthetic_dataset
from tests.test_robust_cli import FakeEval, _spec

NEVER = {  # hiçbir durum tetiklenmez
    "rgb_lost": -1.0,
    "ir_lost": -1.0,
    "ir_noisy": 1e9,
    "misaligned": 1e9,
    "rgb_blurry": -1.0,
    "ir_blurry": -1.0,
}


def thresholds_file(path: Path, **over) -> Path:
    states = {k: {"threshold": v} for k, v in {**NEVER, **over}.items()}
    path.write_text(json.dumps({"states": states, "dark_brightness": -1.0}), encoding="utf-8")
    return path


class RecordingEval(FakeEval):
    def __init__(self):
        super().__init__()
        self.kwargs: list[dict] = []

    def __call__(self, cfg, **kw):
        self.kwargs.append(kw)
        return super().__call__(cfg, **kw)


def _run(tmp_path: Path, th: Path, fake, conds=None, mode: str = "early_fusion") -> int:
    src = tmp_path / "dv"
    if not src.exists():
        make_synthetic_dataset(src, with_ir=True)
    return run_gated_suite(
        _spec(tmp_path, mode),
        conds or suite_conditions(),
        th,
        src=src,
        work=tmp_path / "work",
        meta=src / "meta",
        split="val",
        project=tmp_path / "proj",
        results=tmp_path / "res",
        seed=42,
        evaluate=fake,
    )


RUN = "2026-10-20_x-early_fusion_n_abc1234"


def test_no_gate_no_evaluation(tmp_path: Path):
    fake = RecordingEval()
    assert _run(tmp_path, thresholds_file(tmp_path / "t.json"), fake) == 0
    assert fake.calls == []
    s = json.loads(summary_path(tmp_path / "res", RUN, "val", "noise-20").read_text("utf-8"))
    assert s["gate"] == {"none": 4, "ir_off": 0, "rgb_off": 0} and s["changed"] == 0
    assert s["evaluated"] is False and s["n_images"] == 4 and s["condition"] == "noise-20"
    assert not any((tmp_path / "work").iterdir())


def test_gate_everywhere_evaluates_changed_conditions_only(tmp_path: Path):
    fake = RecordingEval()
    n = _run(tmp_path, thresholds_file(tmp_path / "t.json", ir_lost=1e9), fake)
    want = [f"{c.id}_gate" for c in suite_conditions() if c.id != "drop_ir"]
    assert [c for _, c in fake.calls] == want and n == len(want)
    assert all(k["results_subdir"] == "gate" and k["n_boot"] == 0 for k in fake.kwargs)
    assert all(k["label_sets"] == ("ir",) for k in fake.kwargs)
    cond = fake.kwargs[0]["condition"]
    assert (
        cond["base"] == "clean"
        and cond["gate"]["ir_off"] == 4
        and len(cond["thresholds_sha256"]) == 64
    )
    log = tmp_path / "proj" / f"{RUN}_val_noise-20_gate" / "health.jsonl.gz"
    recs = [json.loads(x) for x in gzip.decompress(log.read_bytes()).decode("utf-8").splitlines()]
    assert len(recs) == 4 and recs[0]["gate"] == "ir_off" and recs[0]["ir"]["state"] == "lost"
    drop = json.loads(summary_path(tmp_path / "res", RUN, "val", "drop_ir").read_text("utf-8"))
    assert drop["evaluated"] is False and drop["gate"]["ir_off"] == 4 and drop["changed"] == 0
    assert not list((tmp_path / "proj").glob("*/predictions.json"))


def test_resume_skips_and_other_thresholds_is_error(tmp_path: Path):
    th = thresholds_file(tmp_path / "t.json", ir_lost=1e9)
    conds = select_conditions(["clean", "noise-20"])
    _run(tmp_path, th, RecordingEval(), conds)
    again = RecordingEval()
    assert _run(tmp_path, th, again, conds) == 0 and again.calls == []
    thresholds_file(th, ir_lost=5.0)
    with pytest.raises(HealthError, match="eşik"):
        _run(tmp_path, th, RecordingEval(), conds)


def test_non_fusion_model_is_error(tmp_path: Path):
    with pytest.raises(HealthError, match="early_fusion"):
        _run(tmp_path, thresholds_file(tmp_path / "t.json"), RecordingEval(), mode="ir")


def test_summarize_health_counts_flags_with_zeros():
    frames = [
        {"gate": "ir_off", "changed": True, "rgb": {"flags": ["dark"]},
         "ir": {"flags": ["noisy", "blurry"]}, "pair": {"misaligned": True}},
        {"gate": "none", "changed": False, "rgb": {"flags": []},
         "ir": {"flags": []}, "pair": {"misaligned": False}},
    ]  # fmt: skip
    s = summarize_health(frames)
    assert s["n_images"] == 2 and s["changed"] == 1 and s["misaligned"] == 1
    assert s["gate"] == {"none": 1, "ir_off": 1, "rgb_off": 0}
    assert s["rgb"] == {"lost": 0, "noisy": 0, "dark": 1, "blurry": 0}
    assert s["ir"] == {"lost": 0, "noisy": 1, "dark": 0, "blurry": 1}


def test_cli_suite_passes_conditions_and_paths(tmp_path: Path, monkeypatch):
    import baykus.health.cli as hc

    got = {}
    spec = _spec(tmp_path, "early_fusion")
    monkeypatch.setattr(hc, "load_model_spec", lambda s: spec)

    def fake(model, conds, thresholds, **kw):
        got.update(conds=[c.id for c in conds], th=thresholds, **kw)
        return 0

    monkeypatch.setattr(hc, "run_gated_suite", fake)
    args = ["suite", "--model", "c=w", "--thresholds", "t.json", "--data-root", "dv",
            "--meta", "m", "--split", "test", "--conditions", "noise-20", "clean"]  # fmt: skip
    assert hc.main(args) == 0
    assert got["conds"] == ["clean", "noise-20"] and got["split"] == "test"
    assert got["th"] == Path("t.json") and got["src"] == Path("dv") and got["seed"] == 42
    assert hc.main([*args[:-2], "noise-7"]) == 2
