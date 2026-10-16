"""Uçtan uca edge testleri (slow): Ultralytics LetterBox ile birebir uyum."""

import numpy as np
import pytest

from baykus.edge.frames import preprocess

pytestmark = pytest.mark.slow


def test_letterbox_matches_ultralytics():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    for shape in [(512, 640, 3), (200, 300, 4)]:
        img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
        ref = LetterBox(new_shape=(640, 640), auto=False)(image=img)
        x, _ = preprocess(img)
        mine = x[0].transpose(1, 2, 0) * 255
        if shape[2] == 3:
            mine = mine[..., ::-1]
        assert np.array_equal(mine.round().astype(np.uint8), ref)
