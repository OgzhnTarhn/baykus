"""baykus-data: veri hattı alt komutları."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baykus.data.layout import SPLITS, split_dirs


def _cmd_verify(a: argparse.Namespace) -> int:
    from baykus.data.verify import verify_split, write_outputs

    ok, issues = {}, []
    for split in SPLITS:
        s_ok, s_issues = verify_split(split_dirs(Path(a.raw), split), split)
        ok[split], issues = s_ok, issues + s_issues
    write_outputs(Path(a.meta), ok, issues)
    print(json.dumps({k: len(v) for k, v in ok.items()}), f"excluded={len(issues)}")
    return 0


def _cmd_convert(a: argparse.Namespace) -> int:
    from baykus.data.convert import convert_split

    ok = json.loads((Path(a.meta) / "ok_stems.json").read_text(encoding="utf-8"))
    stats = {
        s: convert_split(split_dirs(Path(a.raw), s), s, ok.get(s, []), Path(a.out)) for s in SPLITS
    }
    (Path(a.meta) / "convert_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def _cmd_activate(a: argparse.Namespace) -> int:
    from baykus.data.convert import activate_label_set

    print(activate_label_set(Path(a.out), a.modality, a.label_set))
    return 0


def _cmd_illum(a: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from baykus.data.illumination import LEVELS, assign, mean_brightness, thresholds
    from baykus.io import imread_any, imwrite_any

    out, meta, fig = Path(a.out), Path(a.meta), Path(a.fig)
    rows = []
    for split in SPLITS:
        for p in sorted((out / "rgb/images" / split).glob("*.jpg")):
            rows.append((split, p.stem, mean_brightness(imread_any(p))))
    values = np.array([r[2] for r in rows])
    th = thresholds(values)
    labels = assign(values, th)
    _write_illumination(meta, rows, labels, th)

    # her seviyeden 16 örneklik mozaik (gözle doğrulama için)
    rng = np.random.default_rng(0)
    for level in LEVELS:
        idx = [i for i, lab in enumerate(labels) if lab == level]
        if not idx:
            continue
        pick = rng.choice(idx, size=min(16, len(idx)), replace=False)
        tiles = [
            cv2.resize(
                imread_any(out / "rgb/images" / rows[i][0] / f"{rows[i][1]}.jpg"), (320, 256)
            )
            for i in pick
        ]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r : r + 4]) for r in range(0, len(tiles), 4)])
        imwrite_any(fig / f"illum_samples_{level}.jpg", grid)

    counts = {lv: labels.count(lv) for lv in LEVELS}
    print(json.dumps({"thresholds": th, "counts": counts}))
    return 0


def _write_illumination(meta: Path, rows: list, labels: list[str], th: tuple) -> None:
    import csv

    from baykus.data.illumination import METHOD

    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "illumination.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "brightness", "illumination"])
        for (s, st, b), lab in zip(rows, labels, strict=True):
            w.writerow([s, st, f"{b:.2f}", lab])
    (meta / "illumination_thresholds.json").write_text(
        json.dumps({"t_dark": th[0], "t_day": th[1], "method": METHOD}, ensure_ascii=False),
        encoding="utf-8",
    )


def _cmd_illum_relabel(a: argparse.Namespace) -> int:
    """Görüntü okumadan: illumination.csv'deki parlaklıkla etiketleri yeniden yazar."""
    import csv

    import numpy as np

    from baykus.data.illumination import LEVELS, assign, thresholds

    meta = Path(a.meta)
    with (meta / "illumination.csv").open(encoding="utf-8") as f:
        rows = [(r["split"], r["stem"], float(r["brightness"])) for r in csv.DictReader(f)]
    th = thresholds(np.array([r[2] for r in rows]))
    labels = assign(np.array([r[2] for r in rows]), th)
    _write_illumination(meta, rows, labels, th)
    print(json.dumps({"thresholds": th, "counts": {lv: labels.count(lv) for lv in LEVELS}}))
    return 0


def _write_slices(out: Path, meta: Path) -> None:
    """slices_<split>.json. Kimlikler COCO JSON'daki dosya adından okunur (dışlananlar kaydırır)."""
    from baykus.data.coco import illumination_slices

    for split in SPLITS:
        coco = json.loads((out / "coco" / f"{split}_ir.json").read_text(encoding="utf-8"))
        stem_to_id = {im["file_name"]: im["id"] for im in coco["images"]}
        slices = {"illumination": illumination_slices(meta / "illumination.csv", split, stem_to_id)}
        (meta / f"slices_{split}.json").write_text(json.dumps(slices), encoding="utf-8")


def _cmd_slices(a: argparse.Namespace) -> int:
    _write_slices(Path(a.out), Path(a.meta))
    print("slices ok")
    return 0


def _cmd_rgbt4(a: argparse.Namespace) -> int:
    import time

    from baykus.data.multichannel import build_rgbt4
    from baykus.data.yamls import write_data_yamls

    out = Path(a.out)
    t0 = time.monotonic()
    n = build_rgbt4(out, splits=a.splits, workers=a.workers)
    write_data_yamls(out)
    print(f"rgbt4 images: {n} ({time.monotonic() - t0:.0f} sn, {a.workers} işçi, {a.splits})")
    return 0


def _cmd_yaml(a: argparse.Namespace) -> int:
    from baykus.data.yamls import write_data_yamls

    for p in write_data_yamls(Path(a.out)):
        print(p)
    return 0


def _cmd_coco(a: argparse.Namespace) -> int:
    from baykus.data.coco import yolo_to_coco
    from baykus.data.layout import OUT_SIZE

    out, meta = Path(a.out), Path(a.meta)
    (out / "coco").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        stems = sorted(p.stem for p in (out / "rgb/images" / split).glob("*.jpg"))
        for label_set in ("ir", "rgb"):
            coco = yolo_to_coco(out / f"labels_{label_set}" / split, stems, *OUT_SIZE)
            (out / "coco" / f"{split}_{label_set}.json").write_text(
                json.dumps(coco), encoding="utf-8"
            )
    _write_slices(out, meta)
    print("coco ok")
    return 0


def _cmd_report(a: argparse.Namespace) -> int:
    from baykus.data.report import build_report

    text = build_report(Path(a.out), Path(a.meta), Path(a.fig))
    Path(a.md).write_text(text, encoding="utf-8")
    print(f"yazıldı: {a.md}")
    return 0


def _cmd_m3ot(a: argparse.Namespace) -> int:
    from baykus.data.m3ot import convert_m3ot

    stats = convert_m3ot(Path(a.raw), Path(a.out), a.splits, a.workers)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def _cmd_m3ot_seqs(a: argparse.Namespace) -> int:
    from baykus.data.demo import consecutive_sequences

    seqs = consecutive_sequences(Path(a.root) / "meta/m3ot_index.csv", a.split)
    print(json.dumps(seqs, ensure_ascii=False, indent=2))
    return 0


def _cmd_demo_pack(a: argparse.Namespace) -> int:
    from baykus.data.demo import pack

    dirs = pack(Path(a.root), a.split, a.seqs, Path(a.out), allow_test=a.allow_test)
    print(json.dumps([str(d) for d in dirs], ensure_ascii=False))
    return 0


def _cmd_m3ot_relabel(a: argparse.Namespace) -> int:
    from baykus.data.m3ot_merge import relabel

    if (a.teacher is not None) == a.all_car:
        raise ValueError("--teacher ya da --all-car'dan tam biri verilmeli")
    teacher = None
    if a.teacher is not None:
        teacher = json.loads(Path(a.teacher).read_text(encoding="utf-8"))["tracks"]
    counts = relabel(Path(a.root), a.split, teacher)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


def _cmd_m3ot_merge(a: argparse.Namespace) -> int:
    from baykus.data.m3ot_merge import merge

    n = merge(Path(a.m3ot), Path(a.into), a.split)
    print(f"birleştirilen çift: {n}")
    return 0


def _no_test(split: str) -> None:
    if split == "test":
        raise ValueError("M3OT test bölümü öğretmen etiketine / denetime girmez")


def _cmd_m3ot_teacher(a: argparse.Namespace) -> int:
    _no_test(a.split)  # modeller yüklenmeden
    from baykus.classes import CLASS_NAMES
    from baykus.data import teacher as t
    from baykus.edge.backends import OrtBackend
    from baykus.engine_ref.models import load_models
    from baykus.engine_ref.pipeline import Pipeline
    from baykus.runinfo import git_info, iso, utc_now, write_json

    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    tracks = t.label_tracks(
        Path(a.root), a.split, Pipeline(models, backend=OrtBackend(models.onnx))
    )
    counts = {name: 0 for name in CLASS_NAMES}
    for v in tracks.values():
        counts[CLASS_NAMES[v["cls_id"]]] += 1
    rule = {
        "window": list(t.WINDOW),
        "frames_per_track": t.FRAMES_PER_TRACK,
        "iou_min": t.IOU_MIN,
        "score_min": t.SCORE_MIN,
        "floor": t.FLOOR,
    }
    rec = {"split": a.split, "tracks": tracks, "counts": counts, "rule": rule}
    rec.update(git_commit=git_info()["git_commit"], created_at=iso(utc_now()))
    print(write_json(Path(a.out), rec))
    print(f"{len(tracks)} iz: {json.dumps(counts, ensure_ascii=False)}")
    return 0


def _cmd_m3ot_audit(a: argparse.Namespace) -> int:
    _no_test(a.split)
    from baykus.data import teacher_audit
    from baykus.data.teacher import audit_crops
    from baykus.runinfo import write_json

    tracks = json.loads(Path(a.teacher).read_text(encoding="utf-8"))["tracks"]
    keys = teacher_audit.sample(tracks)
    ids = teacher_audit.render(keys, audit_crops(Path(a.root), a.split, keys), Path(a.out))
    print(write_json(Path(a.out) / "key.json", ids))
    print(f"{len(ids)} örnek → {Path(a.out) / 'index.html'}")
    return 0


def _cmd_m3ot_audit_score(a: argparse.Namespace) -> int:
    from baykus.data.teacher_audit import score
    from baykus.runinfo import git_info, iso, utc_now, write_json

    tracks = json.loads(Path(a.teacher).read_text(encoding="utf-8"))["tracks"]
    ids = json.loads(Path(a.key).read_text(encoding="utf-8"))
    answers = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    rec = score(ids, tracks, answers)
    rec.update(git_commit=git_info()["git_commit"], created_at=iso(utc_now()))
    print(write_json(Path(a.out), rec))
    print(f"doğruluk {rec['accuracy']} ({rec['answered']} cevap) → karar {rec['decision']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-data")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="RGB/IR çift ve etiket bütünlüğü")
    v.add_argument("--raw", required=True)
    v.add_argument("--meta", required=True)
    v.set_defaults(func=_cmd_verify)
    c = sub.add_parser("convert", help="kenarlık kırp, etiketleri YOLO formatına çevir")
    c.add_argument("--raw", required=True)
    c.add_argument("--meta", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_convert)
    al = sub.add_parser(
        "activate-labels", help="bir etiket setini <modality>/labels altına kopyala"
    )
    al.add_argument("--out", required=True)
    al.add_argument("--modality", required=True, choices=["rgb", "ir", "rgbt4"])
    al.add_argument("--label-set", required=True, choices=["ir", "rgb"])
    al.set_defaults(func=_cmd_activate)
    il = sub.add_parser("illum", help="aydınlatma etiketi (parlaklık eşikleri)")
    il.add_argument("--out", required=True)
    il.add_argument("--meta", required=True)
    il.add_argument("--fig", default="../docs/figures/data")
    il.set_defaults(func=_cmd_illum)
    ir_ = sub.add_parser("illum-relabel", help="illumination.csv'yi görüntüsüz yeniden etiketle")
    ir_.add_argument("--meta", required=True)
    ir_.set_defaults(func=_cmd_illum_relabel)
    sl = sub.add_parser("slices", help="COCO JSON + illumination.csv → slices_<split>.json")
    sl.add_argument("--out", required=True)
    sl.add_argument("--meta", required=True)
    sl.set_defaults(func=_cmd_slices)
    r4 = sub.add_parser("rgbt4", help="RGB+IR → 4 kanallı TIFF ve data_*.yaml")
    r4.add_argument("--out", required=True)
    r4.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    r4.add_argument("--workers", type=int, default=1, help="paralel işçi (Colab: os.cpu_count())")
    r4.set_defaults(func=_cmd_rgbt4)
    ym = sub.add_parser("yaml", help="data_*.yaml dosyalarını yaz (rgbt4 beklemeden)")
    ym.add_argument("--out", required=True)
    ym.set_defaults(func=_cmd_yaml)
    co = sub.add_parser("coco", help="COCO JSON + kırılım listeleri")
    co.add_argument("--out", required=True)
    co.add_argument("--meta", required=True)
    co.set_defaults(func=_cmd_coco)
    rp = sub.add_parser("report", help="E0 veri raporu")
    rp.add_argument("--out", required=True)
    rp.add_argument("--meta", required=True)
    rp.add_argument("--md", default="../docs/data-report.md")
    rp.add_argument("--fig", default="../docs/figures/data")
    rp.set_defaults(func=_cmd_report)
    m = sub.add_parser("m3ot", help="M3OT → işlenmiş düzen (FR-D8, K5)")
    m.add_argument("--raw", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--splits", nargs="+", default=["test"], choices=list(SPLITS))
    m.add_argument("--workers", type=int, default=1)
    m.set_defaults(func=_cmd_m3ot)
    ms = sub.add_parser("m3ot-seqs", help="M3OT dizileri: kare sayısı ve ardışıklık (JSON)")
    ms.add_argument("--root", required=True, help="baykus-data m3ot --out klasörü")
    ms.add_argument("--split", required=True, choices=list(SPLITS))
    ms.set_defaults(func=_cmd_m3ot_seqs)
    dp = sub.add_parser("demo-pack", help="M3OT dizileri → operatör ekranı düzeni + etiketler")
    dp.add_argument("--root", required=True, help="baykus-data m3ot --out klasörü")
    dp.add_argument("--split", required=True, choices=list(SPLITS))
    dp.add_argument("--seqs", nargs="+", required=True)
    dp.add_argument("--out", required=True)
    dp.add_argument("--allow-test", action="store_true", help="test bölümü (yalnız son ölçüm için)")
    dp.set_defaults(func=_cmd_demo_pack)
    mr = sub.add_parser("m3ot-relabel", help="M3OT etiketlerini öğretmen sınıflarıyla yeniden yaz")
    mr.add_argument("--root", required=True, help="baykus-data m3ot --out klasörü")
    mr.add_argument("--split", default="train", choices=list(SPLITS))
    mr.add_argument("--teacher", help="öğretmen JSON (içinde 'tracks')")
    mr.add_argument("--all-car", action="store_true", help="bütün kutuları car say")
    mr.set_defaults(func=_cmd_m3ot_relabel)
    mm = sub.add_parser("m3ot-merge", help="M3OT train'i DroneVehicle ağacına önekli kopyala")
    mm.add_argument("--m3ot", required=True, help="baykus-data m3ot --out klasörü")
    mm.add_argument("--into", required=True, help="DroneVehicle kökü")
    mm.add_argument("--split", default="train", choices=list(SPLITS))
    mm.set_defaults(func=_cmd_m3ot_merge)
    mt = sub.add_parser("m3ot-teacher", help="M3OT izlerine öğretmen modelle sınıf (tasarım §3 B)")
    mt.add_argument("--root", required=True, help="baykus-data m3ot --out klasörü")
    mt.add_argument("--split", default="train", choices=list(SPLITS))
    mt.add_argument("--models", required=True, help="model dizini (e3c.onnx + JSON'lar)")
    mt.add_argument("--manifest", default=None, help="configs/engine/models.json (sha256)")
    mt.add_argument("--out", required=True)
    mt.set_defaults(func=_cmd_m3ot_teacher)
    ma = sub.add_parser("m3ot-audit", help="öğretmen etiketlerinin kör denetim sayfası")
    ma.add_argument("--root", required=True)
    ma.add_argument("--split", default="train", choices=list(SPLITS))
    ma.add_argument("--teacher", required=True, help="m3ot-teacher çıktısı")
    ma.add_argument("--out", required=True, help="sayfa klasörü (index.html, crops/, key.json)")
    ma.set_defaults(func=_cmd_m3ot_audit)
    ms = sub.add_parser("m3ot-audit-score", help="kör denetim cevapları → doğruluk ve karar")
    ms.add_argument("--teacher", required=True)
    ms.add_argument("--key", required=True, help="m3ot-audit'in key.json'u")
    ms.add_argument("--answers", required=True, help="sayfadan indirilen cevaplar")
    ms.add_argument("--out", required=True)
    ms.set_defaults(func=_cmd_m3ot_audit_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
