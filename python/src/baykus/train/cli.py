"""baykus-train: config'ten Ultralytics eğitimi + MLflow + eğitim kaydı (FR-T1, T3, T5, T6)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from baykus.runinfo import git_info, iso, make_run_id, software_versions, utc_now, write_json
from baykus.train.config import ExperimentConfig, load_config

SMOKE_FRACTION = 0.05
RECORD = "baykus_train.json"


def resolve_data(cfg: ExperimentConfig, data_root: Path | None) -> Path:
    return Path(data_root) / Path(cfg.data).name if data_root else Path(cfg.data)


def ultralytics_train_args(
    cfg: ExperimentConfig,
    *,
    data: Path,
    project: Path,
    name: str,
    smoke: bool,
    epochs: int | None,
    device: str | None,
) -> dict:
    t = cfg.train
    args = dict(
        data=str(data),
        imgsz=t.imgsz,
        epochs=t.epochs,
        patience=t.patience,
        batch=t.batch,
        seed=t.seed,
        deterministic=True,
        close_mosaic=t.close_mosaic,
        workers=t.workers,
        save_period=t.save_period,
        cache=False,
        amp=True,
        project=str(project),
        name=name,
        exist_ok=False,
        plots=True,
    )
    if epochs is not None:
        args["epochs"] = epochs
    if smoke:
        args.update(epochs=1, fraction=SMOKE_FRACTION, save_period=-1, plots=False)
    if device is not None:
        args["device"] = device
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-train")
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", type=Path, default=None, help="data_*.yaml'ların klasörü")
    p.add_argument("--project", type=Path, default=Path("runs/train"))
    p.add_argument("--results", type=Path, default=Path("../results"))
    p.add_argument("--smoke", action="store_true", help="%%5 veri, 1 epoch (FR-T6)")
    p.add_argument("--epochs", type=int, default=None, help="ör. 1: epoch süresi ölçümü")
    p.add_argument("--device", default=None)
    p.add_argument("--resume", type=Path, default=None, help="yarıda kalan çalıştırmanın klasörü")
    return p


def _start_mlflow(
    cfg: ExperimentConfig, project: Path, run_name: str, tags: dict, resume_id: str | None
) -> str | None:
    try:
        import mlflow
        from ultralytics import settings
    except ImportError:
        print("uyarı: mlflow kurulu değil, MLflow kaydı atlandı", file=sys.stderr)
        return None
    uri = (Path(project).resolve() / "mlruns").as_uri()
    # MLflow 3.x dosya deposunu varsayılan olarak reddeder; sqlite Drive (FUSE) üzerinde
    # kilitlenme sorunlu olduğu için dosya deposunda kalınır.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.update(
        MLFLOW_TRACKING_URI=uri, MLFLOW_EXPERIMENT_NAME=cfg.mlflow_experiment, MLFLOW_RUN=run_name
    )
    settings.update({"mlflow": True})
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    run = mlflow.start_run(run_id=resume_id) if resume_id else mlflow.start_run(run_name=run_name)
    mlflow.set_tags({k: str(v) for k, v in tags.items()})
    return run.info.run_id


def _end_mlflow() -> None:
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is not None:
        mlflow.end_run()


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = load_config(Path(a.config))
    from ultralytics import YOLO

    from baykus.data.convert import activate_label_set
    from baykus.train.multimodal import callbacks_for, make_trainer_class

    start = utc_now()
    if a.resume:
        run_dir = Path(a.resume)
        last, rec_path = run_dir / "weights" / "last.pt", run_dir / RECORD
        if not last.exists() or not rec_path.exists():
            print(f"hata: devam için {last} ve {rec_path} gerekli", file=sys.stderr)
            return 2
        record = json.loads(rec_path.read_text(encoding="utf-8"))
        _start_mlflow(cfg, run_dir.parent, record["run_id"], {}, record.get("mlflow_run_id"))
        t0 = time.monotonic()
        try:
            YOLO(str(last)).train(resume=True, trainer=make_trainer_class(cfg))
        finally:
            _end_mlflow()
        record["train_seconds"] = round(record.get("train_seconds", 0) + time.monotonic() - t0, 1)
        record["resumed_at"] = iso(utc_now())
        write_json(rec_path, record)
        return 0

    data = resolve_data(cfg, a.data_root)
    if not data.exists():
        print(f"hata: {data} yok (baykus-data yaml çalıştı mı?)", file=sys.stderr)
        return 2
    n_labels = activate_label_set(data.parent, cfg.modality, cfg.label_set)
    print(f"etiket seti {cfg.label_set} → {cfg.modality}/labels ({n_labels} dosya)")

    git = git_info()
    run_id = make_run_id(cfg.experiment, cfg.model, git["git_commit"], start)
    name = run_id + ("_smoke" if a.smoke else f"_ep{a.epochs}" if a.epochs else "")
    args = ultralytics_train_args(
        cfg,
        data=data,
        project=a.project,
        name=name,
        smoke=a.smoke,
        epochs=a.epochs,
        device=a.device,
    )
    tags = {"config": a.config, "config_hash": cfg.config_hash, **git, "smoke": a.smoke}
    mlflow_id = _start_mlflow(cfg, a.project, name, tags, None)

    model = YOLO(cfg.model)
    for event, fn in callbacks_for(cfg, resume=False):
        model.add_callback(event, fn)
    t0 = time.monotonic()
    try:
        model.train(trainer=make_trainer_class(cfg), **args)
    finally:
        _end_mlflow()
    run_dir = Path(model.trainer.save_dir)
    record = {
        "run_id": name,
        "experiment": cfg.experiment,
        "config": a.config,
        "config_hash": cfg.config_hash,
        **git,
        "mlflow_run_id": mlflow_id,
        "seed": cfg.train.seed,
        "mode": cfg.mode,
        "model": cfg.model,
        "label_set": cfg.label_set,
        "smoke": a.smoke,
        "epochs_requested": args["epochs"],
        "train_seconds": round(time.monotonic() - t0, 1),
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
        "software": software_versions(),
        "created_at": iso(start),
    }
    write_json(run_dir / RECORD, record)
    if not a.smoke:
        write_json(Path(a.results) / "train" / f"{name}.json", record)
    print(json.dumps(record, ensure_ascii=False))
    return 0
