import gzip
import json
from pathlib import Path

import pytest

from baykus.track.cli import main
from baykus.track.sweep import pick


def _row(activation: float, high_conf: float, hota: float) -> dict:
    return {
        "activation": activation,
        "high_conf": high_conf,
        "HOTA": hota,
        "MOTA": 0.0,
        "IDF1": 0.0,
        "tracks_per_frame": 0.0,
    }


def _write(path: Path, ids_per_frame: list[list[tuple[int, int]]], split: str = "val") -> Path:
    seq = {
        "seq": "1_08",
        "split": split,
        "frames": [
            {"frame": f + 1, "tracks": [{"index": i, "track_id": t} for i, t in fr]}
            for f, fr in enumerate(ids_per_frame)
        ],
    }
    path.write_bytes(gzip.compress(json.dumps({"sequences": [seq]}).encode("utf-8")))
    return path


def test_pick_highest_hota():
    """HOTA farkları `tol` üstünde → en yüksek HOTA satırı seçilir."""
    rows = [_row(0.3, 0.2, 0.30), _row(0.5, 0.3, 0.40), _row(0.7, 0.4, 0.35)]
    assert pick(rows)["HOTA"] == 0.40


def test_pick_tie_prefers_higher_activation():
    """HOTA farkı ≤ `tol` → etkinleşme eşiği yüksek olan satır."""
    rows = [_row(0.3, 0.2, 0.300), _row(0.5, 0.2, 0.296)]
    assert pick(rows)["activation"] == 0.5


def test_pick_tie_then_higher_high_conf():
    """Aynı activation ve HOTA eşit aralıkta → high_conf büyük olan satır."""
    rows = [_row(0.5, 0.2, 0.300), _row(0.5, 0.4, 0.298)]
    assert pick(rows)["high_conf"] == 0.4


def test_pick_empty_raises():
    with pytest.raises(ValueError):
        pick([])


def test_sweep_cli_rejects_test_split(tmp_path: Path, capsys):
    """`split` test olan dets ile sweep reddedilir (main hatayı yakalar, çıkış kodu 2)."""
    dets = _write(tmp_path / "dets.json.gz", [[(0, 1)]], split="test")
    gt = tmp_path / "gt.csv"
    gt.write_text("1,1,0,0,10,10,1\n", encoding="utf-8")
    out = tmp_path / "sweep.json"
    code = main(["sweep", "--dets", str(dets), "--gt-csv", str(gt), "--out", str(out)])
    assert code == 2
    assert "test" in capsys.readouterr().err
    assert not out.exists()
