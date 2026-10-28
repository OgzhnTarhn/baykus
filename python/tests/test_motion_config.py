import json
from pathlib import Path

import pytest

from baykus.motion.config import MotionCfg

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "engine" / "motion.json"


def test_loads_mask_min_score():
    cfg = MotionCfg.load(CFG_PATH)
    assert cfg.ego.mask_min_score == 0.6


def test_missing_mask_min_score_rejected(tmp_path):
    raw = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    del raw["ego"]["mask_min_score"]
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="mask_min_score"):
        MotionCfg.load(path)
