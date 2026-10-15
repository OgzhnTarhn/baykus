"""Drive'daki eğitim klasörleri ve checkpoint durumu (02_train notebook'u kullanır).

Ultralytics her epoch sonunda weights/last.pt ve results.csv'ye bir satır yazar; devam
(baykus-train --resume) last.pt'den sürer. Smoke ve 1 epoch ölçümleri tam eğitim sayılmaz.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from baykus.runinfo import iso

# Smoke ve epoch ölçümleri; aynı gün yeniden çalıştırılınca Ultralytics ada -2 / 2 ekler
_TRIAL = re.compile(r"_(smoke|ep\d+)(-?\d+)?$")


def _record(run_dir: Path) -> dict:
    p = run_dir / "baykus_train.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def find_runs(runs_root: Path, experiment: str, finished: bool) -> list[Path]:
    """Bu deneyin tam eğitim klasörleri (eskiden yeniye); finished=False → yarım kalanlar."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    slug = experiment.replace("_", "-")
    out = []
    for d in sorted(root.glob(f"*_{slug}_*")):
        if _TRIAL.search(d.name) or not (d / "weights" / "last.pt").exists():
            continue
        if (_record(d).get("status") == "finished") == finished:
            out.append(d)
    return out


def find_trial_runs(runs_root: Path, experiment: str, kind: str) -> list[Path]:
    """Bu deneyin biten smoke ya da epoch ölçümleri (kind: "smoke", "ep1"); yeniden
    "Tümünü çalıştır" denince tekrar koşmamak için."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    slug = experiment.replace("_", "-")
    pat = re.compile(rf"_{re.escape(kind)}(-?\d+)?$")
    return [
        d
        for d in sorted(root.glob(f"*_{slug}_*"))
        if pat.search(d.name) and _record(d).get("status") == "finished"
    ]


def checkpoint_status(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    csv = run_dir / "results.csv"
    lines = csv.read_text(encoding="utf-8").splitlines() if csv.exists() else []
    last = run_dir / "weights" / "last.pt"
    saved = iso(dt.datetime.fromtimestamp(last.stat().st_mtime, dt.UTC)) if last.exists() else None
    rec = _record(run_dir)
    return {
        "run": run_dir.name,
        "epochs_done": max(len([ln for ln in lines if ln.strip()]) - 1, 0),
        "epochs_requested": rec.get("epochs_requested"),
        "finished": rec.get("status") == "finished",
        "last_saved": saved,
    }


def status_report(runs_root: Path, experiment: str) -> str:
    runs = find_runs(runs_root, experiment, finished=True)
    runs += find_runs(runs_root, experiment, finished=False)
    if not runs:
        return f"{experiment} için eğitim yok"
    lines = []
    for d in sorted(runs):
        s = checkpoint_status(d)
        total = s["epochs_requested"] if s["epochs_requested"] is not None else "?"
        state = "bitti" if s["finished"] else "yarım"
        lines.append(
            f"{s['run']}: {s['epochs_done']}/{total} epoch kaydedildi, {state} "
            f"(son kayıt {s['last_saved']} UTC)"
        )
    return "\n".join(lines)
