from pathlib import Path

import pytest

from baykus.eval.cli import train_record
from baykus.robustness.cli import (
    ModelSpec,
    RobustError,
    load_model_spec,
    main,
    plan,
    result_path,
    run_suite,
)
from baykus.robustness.corruptions import suite_conditions
from baykus.runinfo import write_json
from baykus.train.config import parse_config
from tests.fake_ultralytics import make_weights
from tests.synthetic import make_synthetic_dataset

DATA = {"rgb": "data_rgb.yaml", "ir": "data_ir.yaml", "early_fusion": "data_rgbt4.yaml"}


def _spec(tmp_path: Path, mode: str) -> ModelSpec:
    raw = {"experiment": f"x_{mode}", "mode": mode, "model": "yolo26n.pt", "data": DATA[mode]}
    run_id = f"2026-10-20_x-{mode}_n_abc1234"
    w = make_weights(tmp_path / "runs" / run_id, run_id)
    return ModelSpec(f"configs/x_{mode}.yaml", w, parse_config(raw), run_id)


class FakeEval:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, cfg, *, weights, data, split, results, condition, results_subdir,
                 label_sets, project, n_boot, **kw):  # fmt: skip
        assert data.exists(), data  # koşul klasörünün data yaml'ı
        img_dir = data.parent / cfg.modality / "images" / split
        assert any(img_dir.glob("*")), img_dir
        self.calls.append((cfg.mode, condition["id"]))
        run_id = train_record(weights)["run_id"]
        name = f"{run_id}_{split}_{condition['id']}"
        (Path(project) / name).mkdir(parents=True, exist_ok=True)
        (Path(project) / name / "predictions.json").write_text("[]", encoding="utf-8")
        out = Path(results) / results_subdir / f"{name}_{label_sets[0]}.json"
        write_json(out, {"robustness": condition, "n_boot": n_boot})
        return [out]


def _suite(tmp_path: Path, models, only, fake, seed: int = 42, n_boot: int = 10) -> int:
    src = make_synthetic_dataset(tmp_path / "dv", with_ir=True)
    return run_suite(
        models,
        suite_conditions(only),
        src=src,
        work=tmp_path / "work",
        meta=src / "meta",
        split="val",
        project=tmp_path / "proj",
        results=tmp_path / "results",
        seed=seed,
        n_boot=n_boot,
        evaluate=fake,
    )


def test_plan_counts_per_mode(tmp_path: Path):
    models = [_spec(tmp_path, m) for m in ("rgb", "ir", "early_fusion")]
    p = plan(models, suite_conditions(), tmp_path / "results", "val")
    per_mode: dict[str, int] = {}
    for _, todo in p:
        for m in todo:
            per_mode[m.cfg.mode] = per_mode.get(m.cfg.mode, 0) + 1
    assert per_mode == {"rgb": 8, "ir": 8, "early_fusion": 16}


def test_run_suite_evaluates_cleans_and_resumes(tmp_path: Path):
    models = [_spec(tmp_path, "rgb"), _spec(tmp_path, "early_fusion")]
    fake = FakeEval()
    n = _suite(tmp_path, models, ["darken", "misalign"], fake)
    # clean ×2 + darken 3×2 + misalign 4 (sadece füzyon)
    assert n == 12 and len(fake.calls) == 12
    assert ("rgb", "misalign-5") not in fake.calls and ("early_fusion", "misalign-40") in fake.calls
    assert not any((tmp_path / "work").iterdir())  # koşul klasörleri silindi
    assert not list((tmp_path / "proj").glob("*/predictions.json"))  # Ultralytics kopyası silindi
    for m in models:
        assert result_path(tmp_path / "results", m, "val", suite_conditions()[0]).exists()
    fake2 = FakeEval()
    assert _suite(tmp_path, models, ["darken", "misalign"], fake2) == 0  # oturum koptu → tekrar
    assert fake2.calls == []


def test_run_suite_replaces_stale_condition_folder(tmp_path: Path):
    stale = tmp_path / "work" / "clean" / "rgb" / "images" / "val"
    stale.mkdir(parents=True)
    (stale / "99999.png").write_bytes(b"bozuk")
    fake = FakeEval()
    _suite(tmp_path, [_spec(tmp_path, "rgb")], ["darken"], fake)
    assert ("rgb", "clean") in fake.calls
    assert not (tmp_path / "work" / "clean").exists()


def test_only_needed_modalities_written(tmp_path: Path):
    seen = []

    class Spy(FakeEval):
        def __call__(self, cfg, *, data, **kw):
            seen.append(sorted(p.name for p in data.parent.iterdir() if p.is_dir()))
            return super().__call__(cfg, data=data, **kw)

    _suite(tmp_path, [_spec(tmp_path, "ir")], ["noise"], Spy())
    assert seen and all("rgb" not in s and "rgbt4" not in s and "ir" in s for s in seen)


def test_load_model_spec_requires_train_record(tmp_path: Path):
    cfg = tmp_path / "e1.yaml"
    cfg.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\ndata: data/dv_rgbt/data_rgb.yaml\n",
        encoding="utf-8",
    )
    w = tmp_path / "run" / "weights" / "best.pt"
    with pytest.raises(RobustError, match="baykus_train.json"):
        load_model_spec(f"{cfg}={w}")
    with pytest.raises(RobustError, match="="):
        load_model_spec(str(cfg))
    make_weights(tmp_path / "run", "2026-10-11_e1-rgb_s_49da0cb")
    spec = load_model_spec(f"{cfg}={w}")
    assert spec.run_id == "2026-10-11_e1-rgb_s_49da0cb" and spec.cfg.mode == "rgb"


def test_cli_rejects_bad_model_arg(tmp_path: Path, capsys):
    rc = main(["run", "--model", "x.yaml", "--data-root", str(tmp_path), "--meta", str(tmp_path)])
    assert rc == 2 and "hata" in capsys.readouterr().err


def test_result_path_name(tmp_path: Path):
    m = _spec(tmp_path, "rgb")
    cond = suite_conditions(["darken"])[1]
    p = result_path(tmp_path / "r", m, "val", cond)
    assert p == tmp_path / "r" / "robust" / f"{m.run_id}_val_darken-0.5_ir.json"


def test_resume_refuses_results_with_other_seed_or_n_boot(tmp_path: Path):
    models = [_spec(tmp_path, "rgb")]
    _suite(tmp_path, models, ["darken"], FakeEval())
    with pytest.raises(RobustError, match="seed"):
        _suite(tmp_path, models, ["darken"], FakeEval(), seed=7)
    with pytest.raises(RobustError, match="n_boot"):
        _suite(tmp_path, models, ["darken"], FakeEval(), n_boot=100)
    assert _suite(tmp_path, models, ["darken"], FakeEval()) == 0  # aynı ayar: atlanır


def test_failed_condition_build_leaves_no_folder(tmp_path: Path, monkeypatch):
    import baykus.robustness.cli as rc

    def boom(src, out, *a, **kw):
        (Path(out) / "rgb").mkdir(parents=True)
        raise OSError("disk dolu")

    monkeypatch.setattr(rc, "make_condition", boom)
    with pytest.raises(OSError, match="disk dolu"):
        _suite(tmp_path, [_spec(tmp_path, "rgb")], ["darken"], FakeEval())
    assert not (tmp_path / "work" / "clean").exists()


def test_condition_uses_main_label_set(tmp_path: Path, monkeypatch):
    import baykus.robustness.cli as rc

    seen, real = [], rc.make_condition

    def spy(*a, **kw):
        seen.append(kw.get("label_set"))
        return real(*a, **kw)

    monkeypatch.setattr(rc, "make_condition", spy)
    _suite(tmp_path, [_spec(tmp_path, "rgb")], ["darken"], FakeEval())
    assert seen and set(seen) == {"ir"}


def test_cli_bad_config_is_error_not_traceback(tmp_path: Path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "experiment: x\nmode: rgb\nmodel: m.pt\ndata: data_rgb.yaml\nfoo: 1\n", encoding="utf-8"
    )
    w = make_weights(tmp_path / "run", "2026-10-20_x_n_abc1234")
    rc = main(["run", "--model", f"{cfg}={w}", "--data-root", str(tmp_path),
               "--meta", str(tmp_path)])  # fmt: skip
    assert rc == 2 and "foo" in capsys.readouterr().err


def test_ir_misalign_adds_ir_model_runs(tmp_path: Path):
    models = [_spec(tmp_path, m) for m in ("rgb", "ir")]
    p = plan(models, suite_conditions(), tmp_path / "results", "val", ir_misalign=True)
    ir_conds = [c.id for c, todo in p for m in todo if m.cfg.mode == "ir"]
    rgb_conds = [c.id for c, todo in p for m in todo if m.cfg.mode == "rgb"]
    assert len(ir_conds) == 12 and "misalign-40" in ir_conds
    assert not any(c.startswith("misalign") for c in rgb_conds)  # RGB'ye hâlâ uygulanmaz


def test_cli_ir_misalign_flag_reaches_suite(tmp_path: Path, monkeypatch):
    import baykus.robustness.cli as rc

    seen = {}
    monkeypatch.setattr(rc, "run_suite", lambda *a, **kw: seen.update(kw) or 0)
    w = make_weights(tmp_path / "run", "2026-10-20_e2-ir_s_abc1234")
    cfg = tmp_path / "e2.yaml"
    cfg.write_text("experiment: e2_ir\nmode: ir\nmodel: yolo26s.pt\ndata: data_ir.yaml\n",
                   encoding="utf-8")  # fmt: skip
    assert main(["run", "--model", f"{cfg}={w}", "--data-root", str(tmp_path),
                 "--meta", str(tmp_path), "--ir-misalign"]) == 0  # fmt: skip
    assert seen["ir_misalign"] is True
