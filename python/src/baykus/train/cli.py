"""baykus-train: config'ten Ultralytics eğitimi + MLflow + eğitim kaydı (FR-T1, T3, T5, T6)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from baykus.runinfo import (
    git_info,
    hardware_info,
    iso,
    make_run_id,
    software_versions,
    utc_now,
    write_json,
)
from baykus.train.config import ExperimentConfig, load_config

SMOKE_FRACTION = 0.05
RECORD = "baykus_train.json"


class ResumeError(ValueError):
    pass


def resolve_data(cfg: ExperimentConfig, data_root: Path | None) -> Path:
    return Path(data_root) / Path(cfg.data).name if data_root else Path(cfg.data)


def make_record(
    cfg: ExperimentConfig,
    *,
    config: str,
    run_id: str,
    git: dict,
    mlflow_run_id: str | None,
    smoke: bool,
    epochs: int,
    run_dir: Path,
    start,
) -> dict:
    """Eğitim kaydı eğitim BAŞINDA yazılır (status=running); oturum koparsa devam bunu kullanır."""
    run_dir = Path(run_dir)
    return {
        "run_id": run_id,
        "experiment": cfg.experiment,
        "config": config,
        "config_hash": cfg.config_hash,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "mlflow_run_id": mlflow_run_id,
        "seed": cfg.train.seed,
        "mode": cfg.mode,
        "model": cfg.model,
        "label_set": cfg.label_set,
        "smoke": smoke,
        "epochs_requested": epochs,
        "status": "running",
        "train_seconds": None,
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
        "hardware": hardware_info(),
        "software": software_versions(),
        "created_at": iso(start),
        "finished_at": None,
    }


def finish_record(record: dict, seconds: float, when) -> dict:
    record["train_seconds"] = round((record.get("train_seconds") or 0) + seconds, 1)
    record["status"] = "finished"
    record["finished_at"] = iso(when)
    return record


def prepare_resume(
    cfg: ExperimentConfig, run_dir: Path, data_root: Path | None, config: str
) -> tuple[dict, Path]:
    """Yeni oturumda devam: last.pt ve veri kontrolü, etiket seti kurulumu, kayıt yükleme."""
    from baykus.data.convert import activate_label_set

    run_dir = Path(run_dir)
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        raise ResumeError(f"devam için {last} gerekli")
    data = resolve_data(cfg, data_root)
    if not data.exists():
        raise ResumeError(f"{data} yok (veri indirme hücresi ve baykus-data yaml çalıştı mı?)")
    n_labels = activate_label_set(data.parent, cfg.modality, cfg.label_set)
    print(f"etiket seti {cfg.label_set} → {cfg.modality}/labels ({n_labels} dosya)")
    rec_path = run_dir / RECORD
    if rec_path.exists():
        return json.loads(rec_path.read_text(encoding="utf-8")), data
    # Eski sürüm kaydı sadece eğitim sonunda yazıyordu: kopan çalıştırmanın kaydı yeniden kurulur.
    # Başlangıç commit'i run_id'nin son parçasında (kısa hash).
    git = {"git_commit": run_dir.name.rsplit("_", 1)[-1], "git_dirty": None}
    record = make_record(
        cfg,
        config=config,
        run_id=run_dir.name,
        git=git,
        mlflow_run_id=None,
        smoke=False,
        epochs=cfg.train.epochs,
        run_dir=run_dir,
        start=utc_now(),
    )
    record["record_reconstructed"] = True
    write_json(rec_path, record)
    return record, data


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


def _save_record(record: dict, results: Path) -> None:
    write_json(Path(record["run_dir"]) / RECORD, record)
    if not record["smoke"]:
        write_json(Path(results) / "train" / f"{record['run_id']}.json", record)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = load_config(Path(a.config))
    from ultralytics import YOLO

    from baykus.data.convert import activate_label_set
    from baykus.train.multimodal import callbacks_for, make_trainer_class

    start = utc_now()
    if a.resume:
        try:
            record, _ = prepare_resume(cfg, Path(a.resume), a.data_root, a.config)
        except ResumeError as e:
            print(f"hata: {e}", file=sys.stderr)
            return 2
        run_dir = Path(a.resume)
        mlflow_id = _start_mlflow(
            cfg, run_dir.parent, record["run_id"], {}, record.get("mlflow_run_id")
        )
        if record.get("mlflow_run_id") is None:  # yeniden kurulan kayıt: devamın MLflow kaydı
            record["mlflow_run_id"] = mlflow_id
        t0 = time.monotonic()
        try:
            YOLO(str(run_dir / "weights" / "last.pt")).train(
                resume=True, trainer=make_trainer_class(cfg)
            )
        finally:
            _end_mlflow()
        record.setdefault("resumed_at", []).append(iso(start))
        record["hardware_resumed"] = hardware_info()
        _save_record(finish_record(record, time.monotonic() - t0, utc_now()), a.results)
        print(json.dumps(record, ensure_ascii=False))
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
    record: dict = {}

    def write_start_record(trainer) -> None:
        # save_dir Ultralytics'te belli olur (ad çakışırsa sonuna sayı ekler)
        record.update(
            make_record(
                cfg,
                config=a.config,
                run_id=name,
                git=git,
                mlflow_run_id=mlflow_id,
                smoke=a.smoke,
                epochs=args["epochs"],
                run_dir=Path(trainer.save_dir),
                start=start,
            )
        )
        write_json(Path(trainer.save_dir) / RECORD, record)

    model = YOLO(cfg.model)
    model.add_callback("on_pretrain_routine_start", write_start_record)
    for event, fn in callbacks_for(cfg, resume=False):
        model.add_callback(event, fn)
    t0 = time.monotonic()
    try:
        model.train(trainer=make_trainer_class(cfg), **args)
    finally:
        _end_mlflow()
    _save_record(finish_record(record, time.monotonic() - t0, utc_now()), a.results)
    print(json.dumps(record, ensure_ascii=False))
    return 0
