from types import SimpleNamespace

import numpy as np
import pytest

from baykus.train.config import parse_config
from baykus.train.multimodal import (
    DROP_IR,
    DROP_RGB,
    KEEP,
    callbacks_for,
    sample_drop_codes,
)

EF = {
    "experiment": "e3",
    "mode": "early_fusion",
    "model": "yolo26s.pt",
    "data": "d/data_rgbt4.yaml",
}


def test_zero_probabilities_keep_everything():
    codes = sample_drop_codes(1000, 0.0, 0.0, np.random.default_rng(0))
    assert codes.dtype == np.int8 and (codes == KEEP).all()


def test_full_probability():
    assert (sample_drop_codes(50, 1.0, 0.0, np.random.default_rng(0)) == DROP_RGB).all()
    assert (sample_drop_codes(50, 0.0, 1.0, np.random.default_rng(0)) == DROP_IR).all()


def test_frequencies_and_exclusivity():
    codes = sample_drop_codes(200_000, 0.15, 0.15, np.random.default_rng(1))
    assert set(np.unique(codes)) <= {KEEP, DROP_RGB, DROP_IR}
    assert abs((codes == DROP_RGB).mean() - 0.15) < 0.005
    assert abs((codes == DROP_IR).mean() - 0.15) < 0.005


def test_deterministic_with_seed():
    a = sample_drop_codes(100, 0.3, 0.3, np.random.default_rng(7))
    b = sample_drop_codes(100, 0.3, 0.3, np.random.default_rng(7))
    assert (a == b).all()


def test_apply_drop_codes_zeroes_right_channels():
    torch = pytest.importorskip("torch")
    from baykus.train.multimodal import apply_drop_codes

    img = torch.ones(3, 4, 2, 2)
    apply_drop_codes(img, np.array([KEEP, DROP_RGB, DROP_IR], np.int8))
    assert img[0].eq(1).all()
    assert img[1, :3].eq(0).all() and img[1, 3].eq(1).all()
    assert img[2, :3].eq(1).all() and img[2, 3].eq(0).all()


def _fake_model(torch, in_ch: int):
    conv = SimpleNamespace(weight=torch.randn(8, in_ch, 3, 3))
    return SimpleNamespace(model=[SimpleNamespace(conv=conv)])


def test_set_rgb_mean_updates_model_and_ema():
    torch = pytest.importorskip("torch")
    from baykus.train.multimodal import set_rgb_mean

    model, ema = _fake_model(torch, 4), _fake_model(torch, 4)
    trainer = SimpleNamespace(model=model, ema=SimpleNamespace(ema=ema))
    set_rgb_mean(trainer)
    for m in (model, ema):
        w = m.model[0].conv.weight
        assert torch.allclose(w[:, 3], w[:, :3].mean(1))
    assert trainer.rgb_mean_applied is True


def test_set_rgb_mean_rejects_three_channel_model():
    torch = pytest.importorskip("torch")
    from baykus.train.multimodal import set_rgb_mean

    trainer = SimpleNamespace(model=_fake_model(torch, 3), ema=None)
    with pytest.raises(ValueError, match="4 kanal"):
        set_rgb_mean(trainer)


def test_callbacks_only_for_fresh_rgb_mean_runs():
    rgb_mean = parse_config({**EF, "first_conv_init": "rgb_mean"})
    assert [e for e, _ in callbacks_for(rgb_mean, resume=False)] == ["on_pretrain_routine_end"]
    # devam ederken eğitilmiş T kanalı ezilmemeli (Review Focus 2)
    assert callbacks_for(rgb_mean, resume=True) == []
    assert callbacks_for(parse_config(dict(EF)), resume=False) == []


def test_trainer_class_subclasses_detection_trainer():
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect import DetectionTrainer

    from baykus.train.multimodal import make_trainer_class

    drop = parse_config({**EF, "modality_dropout": {"p_rgb": 0.15, "p_ir": 0.15}})
    cls = make_trainer_class(drop)
    assert issubclass(cls, DetectionTrainer) and cls is not DetectionTrainer
    assert make_trainer_class(parse_config(dict(EF))) is DetectionTrainer
