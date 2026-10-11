from pathlib import Path

from baykus.train.cli import SMOKE_FRACTION, build_parser, resolve_data, ultralytics_train_args
from baykus.train.config import parse_config

CFG = parse_config(
    {
        "experiment": "e1_rgb",
        "mode": "rgb",
        "model": "yolo26s.pt",
        "data": "data/dv_rgbt/data_rgb.yaml",
    }
)


def _args(**kw):
    base = dict(
        data=Path("d/data_rgb.yaml"),
        project=Path("runs/train"),
        name="r1",
        smoke=False,
        epochs=None,
        device=None,
    )
    return ultralytics_train_args(CFG, **{**base, **kw})


def test_defaults_follow_spec_10_1():
    a = _args()
    assert a["epochs"] == 100 and a["patience"] == 20 and a["batch"] == 16
    assert a["seed"] == 42 and a["deterministic"] is True and a["close_mosaic"] == 10
    assert a["cache"] is False and a["amp"] is True and a["save_period"] == 5
    assert a["exist_ok"] is False and "fraction" not in a and "device" not in a


def test_smoke_overrides():
    a = _args(smoke=True)
    assert a["epochs"] == 1 and a["fraction"] == SMOKE_FRACTION == 0.05
    assert a["save_period"] == -1 and a["plots"] is False


def test_epochs_override_and_device():
    a = _args(epochs=1, device="cpu")
    assert a["epochs"] == 1 and a["device"] == "cpu" and "fraction" not in a


def test_resolve_data():
    assert resolve_data(CFG, None) == Path("data/dv_rgbt/data_rgb.yaml")
    assert resolve_data(CFG, Path("/content/dv_rgbt")) == Path("/content/dv_rgbt/data_rgb.yaml")


def test_parser():
    a = build_parser().parse_args(["--config", "c.yaml", "--smoke"])
    assert a.smoke and a.resume is None and a.epochs is None
