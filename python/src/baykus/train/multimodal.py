"""Erken füzyon eğitimi: modality dropout (FR-T7) ve rgb_mean başlatması (FR-T8).

Ultralytics'e yama yapılmaz: dropout DetectionTrainer.preprocess_batch alt sınıfında,
rgb_mean on_pretrain_routine_end callback'inde uygulanır (H1 Spike A). Validasyon etkilenmez.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from baykus.train.config import ExperimentConfig

KEEP, DROP_RGB, DROP_IR = 0, 1, 2


def sample_drop_codes(n: int, p_rgb: float, p_ir: float, rng: np.random.Generator) -> np.ndarray:
    """Örnek başına kod: KEEP, DROP_RGB veya DROP_IR. Aynı örnekte ikisi birden asla."""
    u = rng.random(n)
    codes = np.full(n, KEEP, dtype=np.int8)
    codes[u < p_rgb] = DROP_RGB
    codes[(u >= p_rgb) & (u < p_rgb + p_ir)] = DROP_IR
    return codes


def apply_drop_codes(img, codes: np.ndarray):
    """img: torch (B, 4, H, W). Kanal 0-2 = RGB, kanal 3 = T (ADR 0004). Yerinde sıfırlar."""
    import torch

    c = torch.as_tensor(codes, device=img.device)
    img[c == DROP_RGB, :3] = 0
    img[c == DROP_IR, 3:] = 0
    return img


def set_rgb_mean(trainer) -> None:
    """Callback: ilk katmanın T kanalını RGB ağırlıklarının ortalamasıyla başlatır (model + EMA)."""
    import torch

    ema = getattr(trainer, "ema", None)
    with torch.no_grad():
        for m in (trainer.model, ema.ema if ema is not None else None):
            if m is None:
                continue
            w = m.model[0].conv.weight
            if w.shape[1] != 4:
                raise ValueError(f"rgb_mean: ilk katman 4 kanal bekleniyordu, {w.shape[1]} bulundu")
            w[:, 3] = w[:, :3].mean(1)
    trainer.rgb_mean_applied = True


def callbacks_for(cfg: ExperimentConfig, resume: bool) -> list[tuple[str, Callable]]:
    if cfg.first_conv_init == "rgb_mean" and not resume:
        return [("on_pretrain_routine_end", set_rgb_mean)]
    return []


def make_trainer_class(cfg: ExperimentConfig) -> type:
    from ultralytics.models.yolo.detect import DetectionTrainer

    drop = cfg.modality_dropout
    if drop is None or (drop.p_rgb == 0 and drop.p_ir == 0):
        return DetectionTrainer
    p_rgb, p_ir, seed = drop.p_rgb, drop.p_ir, cfg.train.seed

    class ModalityDropoutTrainer(DetectionTrainer):
        _rng = np.random.default_rng(seed)

        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            codes = sample_drop_codes(batch["img"].shape[0], p_rgb, p_ir, self._rng)
            apply_drop_codes(batch["img"], codes)
            return batch

    return ModalityDropoutTrainer
