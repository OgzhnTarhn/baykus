"""baykus-engine: C++ motorunun Python referansı.

detect — bir dizi klasöründen (operatör ekranı düzeni: rgb/, ir/, index.json) tespit dosyası
         (değerlendirme katmanı, varsayılan ham skor ≥ 0,001). Test dizisi bayraksız okunmaz.
measure — tespit dosyaları + pack_labels etiketleri → sınıfsız AP50, ECE (ham / kalibre),
          çift sınıf oranı (dizi başına ve birleşik).
coverage — dizi × koşul (temiz / RGB kapalı / RGB yerine IR) AP50, bulma oranı, kare başına
           tespit (dizi başına ve birleşik). Test dizisi reddedilir.
fixtures — C++ eşlik fikstürleri (M3OT geliştirme bölümü; test reddedilir) ve fixtures/index.json.
raw-ref — kare fikstürlerinin ham ORT çıktısı bu makinede (CI'da Kabul 1 a aynı işlemcide).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

from baykus.edge.backends import OrtBackend
from baykus.engine_ref.measure import measure
from baykus.engine_ref.models import load_models
from baykus.engine_ref.pipeline import GATE_MODES, Pipeline
from baykus.runinfo import git_info, iso, utc_now, write_json

TEST_FLAG = "--i-know-this-is-test"


def read_index(seq_dir: Path, allow_test: bool) -> dict:
    index = json.loads((Path(seq_dir) / "index.json").read_text(encoding="utf-8"))
    if index.get("split") == "test" and not allow_test:
        raise ValueError(f"{seq_dir}: test dizisi; yalnız son ölçümde {TEST_FLAG} ile okunur")
    return index


def _cmd_detect(a: argparse.Namespace) -> int:
    seq = Path(a.seq_dir)
    index = read_index(seq, a.i_know_this_is_test)  # modellerden önce: test hiç yüklenmesin
    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    pipe = Pipeline(models, backend=OrtBackend(models.onnx))
    frames = []
    for f in index["frames"]:
        rgb = (seq / "rgb" / f"{f:06d}.jpg").read_bytes()
        ir = (seq / "ir" / f"{f:06d}.png").read_bytes()
        r = pipe.run(rgb, ir, gate=a.gate, floor=a.floor, tiles=a.tiles)
        frames.append({"frame": f, "dets": [asdict(d) for d in r["dets_all"]]})
    rec = {
        "seq": index["name"],
        "split": index["split"],
        "floor": a.floor,
        "gate": a.gate,
        "tiles": a.tiles,
        "frames": frames,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(json.dumps(rec, ensure_ascii=False).encode("utf-8")))
    print(f"{out} ({len(frames)} kare)")
    return 0


def _cmd_measure(a: argparse.Namespace) -> int:
    files = [json.loads(gzip.decompress(Path(f).read_bytes())) for f in a.dets]
    splits = sorted({f["split"] for f in files})
    if len(splits) != 1:
        raise ValueError(f"tespit dosyaları tek bir bölümden olmalı: {splits}")
    rec = measure(files, Path(a.labels), split=splits[0], allow_test=a.i_know_this_is_test)
    rec["dets_files"] = [Path(f).name for f in a.dets]
    rec["floor"] = sorted({f.get("floor") for f in files}, key=str)
    rec["git_commit"] = git_info()["git_commit"]
    rec["created_at"] = iso(utc_now())
    print(write_json(Path(a.out), rec))
    for name, r in [*rec["per_seq"].items(), ("birleşik", rec)]:
        e0, e1 = r["ece_raw"], r["ece_calibrated"]
        ece = "—" if e0 is None else f"{e0:.4f} → {e1:.4f}"
        print(
            f"{name}: AP50 {r['ap50']:.4f} · ECE {ece} · çift sınıf {r['duplicate_rate']:.4f} "
            f"({r['n_images']} kare, {r['n_dets']} tespit)"
        )
    return 0


def _manifest_sha256(manifest: Path, name: str) -> str | None:
    """models.json içinde `name` (e3c.onnx) için sha256 varsa döndürür."""

    def walk(node) -> str | None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == name:
                    if isinstance(v, dict) and isinstance(v.get("sha256"), str):
                        return v["sha256"]
                    if isinstance(v, str):
                        return v
                found = walk(v)
                if found:
                    return found
        elif isinstance(node, list):
            for v in node:
                found = walk(v)
                if found:
                    return found
        return None

    return walk(json.loads(Path(manifest).read_text(encoding="utf-8")))


def _cmd_coverage(a: argparse.Namespace) -> int:
    from baykus.engine_ref.coverage import CONDITIONS, run_coverage

    seqs = [Path(s) for s in a.seq_dir]
    for seq in seqs:  # modellerden önce: test hiç yüklenmesin
        read_index(seq, False)
    conds = [c.strip() for c in a.conds.split(",")] if a.conds else list(CONDITIONS)
    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    pipe = Pipeline(models, backend=OrtBackend(models.onnx))
    rec = run_coverage(seqs, Path(a.labels), pipe, conds)
    rec["models_sha256"] = _manifest_sha256(Path(a.manifest), "e3c.onnx") if a.manifest else None
    rec["git_commit"] = git_info()["git_commit"]
    rec["created_at"] = iso(utc_now())
    print(write_json(Path(a.out), rec))
    for cond, r in rec["combined"].items():
        hit = r["recall"]["0.3"]
        print(
            f"birleşik/{cond}: AP50 {r['ap50']:.4f} · bulma@0,3 "
            f"IoU0,5 {hit['iou_0.5']:.4f} / IoU0,3 {hit['iou_0.3']:.4f} · "
            f"tespit/kare@0,3 {r['dets_per_frame']['0.3']:.2f}"
        )
    return 0


def _cmd_fixtures(a: argparse.Namespace) -> int:
    from baykus.engine_ref.fixtures import generate

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    index = generate(models, Path(a.dev_root), cfg, Path(a.out))
    out = Path(a.out)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    kinds: dict[str, int] = {}
    for e in index:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"{out / 'index.json'}: {kinds} · toplam {total / 1e6:.1f} MB")
    return 0


def _cmd_tiles_fixtures(a: argparse.Namespace) -> int:
    from baykus.engine_ref.fixtures import tiles_fixtures

    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    added = tiles_fixtures(Pipeline(models, backend=OrtBackend(models.onnx)), Path(a.fixtures))
    print(f"{len(added)} parçalı fikstür kaydı → {Path(a.fixtures) / 'index.json'}")
    return 0


def _cmd_raw_ref(a: argparse.Namespace) -> int:
    from baykus.engine_ref.fixtures import raw_reference

    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    rec = raw_reference(Pipeline(models, backend=OrtBackend(models.onnx)), Path(a.fixtures))
    print(write_json(Path(a.out), rec))
    print(f"{len(rec['frames'])} kare, onnxruntime {rec['onnxruntime']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="dizi klasörü → tespit dosyası (.json.gz)")
    d.add_argument("--models", required=True, help="model dizini (e3c.onnx + JSON'lar)")
    d.add_argument("--manifest", default=None, help="configs/engine/models.json (sha256 denetimi)")
    d.add_argument("--seq-dir", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--floor", type=float, default=0.001, help="ham skor taban süzgeci")
    d.add_argument("--gate", default="advise", choices=GATE_MODES)
    d.add_argument("--tiles", type=int, default=1, choices=(1, 4), help="karo sayısı (1 veya 4)")
    d.add_argument(TEST_FLAG, action="store_true", help="test dizisi (yalnız son ölçüm)")
    d.set_defaults(func=_cmd_detect)
    m = sub.add_parser("measure", help="tespit dosyaları → AP50, ECE, çift sınıf oranı")
    m.add_argument("--dets", nargs="+", required=True, help="detect çıktıları (.json.gz)")
    m.add_argument("--labels", required=True, help="pack_labels klasörü (<split>_ir.json)")
    m.add_argument("--out", required=True)
    m.add_argument(TEST_FLAG, action="store_true", help="test bölümü (yalnız son ölçüm)")
    m.set_defaults(func=_cmd_measure)
    c = sub.add_parser("coverage", help="dizi × koşul AP50, bulma oranı, kare başına tespit")
    c.add_argument("--models", required=True, help="model dizini (e3c.onnx + JSON'lar)")
    c.add_argument("--manifest", default=None, help="configs/engine/models.json (sha256 denetimi)")
    c.add_argument("--seq-dir", nargs="+", required=True, help="dizi klasörleri")
    c.add_argument("--labels", required=True, help="pack_labels klasörü (<split>_ir.json)")
    c.add_argument("--conds", default=None, help="virgüllü koşullar (vars. tümü)")
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_coverage)
    f = sub.add_parser("fixtures", help="C++ eşlik fikstürleri")
    f.add_argument("--models", required=True)
    f.add_argument("--manifest", default=None)
    f.add_argument("--config", default="../configs/m3ot_dev.json")
    f.add_argument("--dev-root", default="../data/m3ot_dev")
    f.add_argument("--out", default="../fixtures")
    f.set_defaults(func=_cmd_fixtures)
    r = sub.add_parser("raw-ref", help="kare fikstürlerinin ham ORT çıktısı (bu makinede)")
    r.add_argument("--models", required=True)
    r.add_argument("--manifest", default=None)
    r.add_argument("--fixtures", default="../fixtures")
    r.add_argument("--out", required=True)
    r.set_defaults(func=_cmd_raw_ref)
    t = sub.add_parser(
        "tiles-fixtures", help="kare fikstürlerinin tiles=4 beklentileri (Görev 15D)"
    )
    t.add_argument("--models", required=True)
    t.add_argument("--manifest", default=None)
    t.add_argument("--fixtures", default="../fixtures")
    t.set_defaults(func=_cmd_tiles_fixtures)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows konsolu (cp1254) → yazamaz: kodlanamayan karakter '?' olur, komut çökmez
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
