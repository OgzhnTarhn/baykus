"""Drive'daki eğitim klasörleri: devam/değerlendirme için bulma, checkpoint durumu."""

import json
from pathlib import Path

from baykus.train.checkpoints import checkpoint_status, find_runs, status_report

HEADER = "epoch,train/box_loss,metrics/mAP50(B)\n"


def _mk(root: Path, name: str, status: str | None = None, epochs_done: int = 0) -> Path:
    d = root / name
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "last.pt").write_bytes(b"x")
    if status is not None:
        rec = {"status": status, "epochs_requested": 50}
        (d / "baykus_train.json").write_text(json.dumps(rec), encoding="utf-8")
    if epochs_done:
        rows = "".join(f"{i},1.0,0.5\n" for i in range(1, epochs_done + 1))
        (d / "results.csv").write_text(HEADER + rows, encoding="utf-8")
    return d


def _layout(root: Path) -> None:
    _mk(root, "2026-10-11_e1-rgb_s_1a92f50_smoke", "finished", 1)
    _mk(root, "2026-10-11_e1-rgb_s_1a92f50_ep1", None, 1)
    _mk(root, "2026-10-10_e1-rgb_s_aaaaaaa", "finished", 50)
    _mk(root, "2026-10-11_e1-rgb_s_49da0cb", None, 38)  # kaydı olmayan, 38'de kopan
    _mk(root, "2026-10-11_e2-ir_s_49da0cb", "running", 3)
    (root / "mlruns").mkdir()


def test_find_runs_skips_smoke_and_epoch_timing(tmp_path: Path):
    _layout(tmp_path)
    names = [p.name for p in find_runs(tmp_path, "e1_rgb", finished=False)]
    assert names == ["2026-10-11_e1-rgb_s_49da0cb"]
    assert [p.name for p in find_runs(tmp_path, "e1_rgb", finished=True)] == [
        "2026-10-10_e1-rgb_s_aaaaaaa"
    ]
    assert [p.name for p in find_runs(tmp_path, "e2_ir", finished=False)] == [
        "2026-10-11_e2-ir_s_49da0cb"
    ]


def test_find_runs_missing_folder(tmp_path: Path):
    assert find_runs(tmp_path / "yok", "e1_rgb", finished=False) == []


def test_checkpoint_status_counts_saved_epochs(tmp_path: Path):
    _layout(tmp_path)
    s = checkpoint_status(tmp_path / "2026-10-11_e1-rgb_s_49da0cb")
    assert s["epochs_done"] == 38 and s["epochs_requested"] is None and s["finished"] is False
    assert s["last_saved"] is not None and s["last_saved"].endswith("Z")
    f = checkpoint_status(tmp_path / "2026-10-10_e1-rgb_s_aaaaaaa")
    assert f["epochs_done"] == 50 and f["epochs_requested"] == 50 and f["finished"] is True


def test_checkpoint_status_without_checkpoint(tmp_path: Path):
    (tmp_path / "r").mkdir()
    s = checkpoint_status(tmp_path / "r")
    assert s["epochs_done"] == 0 and s["last_saved"] is None


def test_status_report_lines(tmp_path: Path):
    _layout(tmp_path)
    text = status_report(tmp_path, "e1_rgb")
    assert "2026-10-11_e1-rgb_s_49da0cb: 38/? epoch kaydedildi, yarım" in text
    assert "2026-10-10_e1-rgb_s_aaaaaaa: 50/50 epoch kaydedildi, bitti" in text
    assert "smoke" not in text and "e2-ir" not in text
    assert status_report(tmp_path, "e3_early_fusion") == "e3_early_fusion için eğitim yok"
