import json
from pathlib import Path

from baykus.motion.cli import main

from .motion_synth import make_video, write_sequence


def test_explore_record_fields(tmp_path: Path):
    data = tmp_path / "dev"
    tracks = {1: {"start": (60, 60), "speed": (0, 0)}, 2: {"start": (150, 140), "speed": (3, 0)}}
    for name in ("a", "b"):
        frames, gt = make_video(n=40, tracks=tracks)
        write_sequence(data, name, frames, gt)
    cfg = tmp_path / "dev.json"
    cfg.write_text(json.dumps({"split": "val", "sequences": ["a", "b"]}), encoding="utf-8")
    out = tmp_path / "explore.json"
    assert main(["explore", "--dev", str(cfg), "--data", str(data), "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert {"per_seq", "git_commit", "created_at", "method"} <= set(rec)
    s = rec["per_seq"]["a"]
    for mod in ("ir", "rgb"):
        m = s[mod]
        assert {"median", "p90", "max"} <= set(m["step_px"])
        assert {"rot_deg_median", "scale_median", "chained_scale"} <= set(m)
        assert 0.8 < m["step_px"]["median"] < 1.4  # kamera (1, 0,5) px/kare → ‖·‖ ≈ 1,12
    assert {"first_block", "last_block", "ratio"} <= set(s["box_growth"])
    hist = s["v_hist"]
    assert {"lo", "band", "hi", "total"} <= set(hist)
    assert hist["lo"] + hist["band"] + hist["hi"] == hist["total"]
    assert hist["lo"] > 0 and hist["hi"] > 0  # durağan ve hareketli iz ayrılıyor
    assert s["tracks"] == 2
