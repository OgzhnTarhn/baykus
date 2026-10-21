"""`baykus-motion`: Faz 4 referans etiketleri, keşif ve ölçüm (docs/design/faz4.md §4, §5.3, §6.3).

Komutlar `python/` içinden; yollar varsayılan olarak repo düzenine göre (`../data/m3ot_dev`).
Test dizisi okuyan komutlar `--i-know-this-is-test` ister (yalnız son ölçüm, tasarım §6.3).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from baykus.engine_ref.cli import TEST_FLAG
from baykus.motion import drift, measure
from baykus.motion.audit import Sample, population, render, sample, score
from baykus.motion.config import MotionCfg
from baykus.motion.explore import METHOD, explore_sequence
from baykus.motion.images import load_sequence
from baykus.motion.labels import gt_tracks, label_sequence, read_labels, write_labels
from baykus.runinfo import git_info, iso, utc_now, write_json

K, LO, HI = 15, 0.03, 0.10  # tasarım §5.2 / §5.3 (ölçümden önce sabit)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cmd_labels(a: argparse.Namespace) -> int:
    seq = load_sequence(Path(a.seq_dir), a.i_know_this_is_test)  # test korumalı: önce
    rows = label_sequence(seq, gt_tracks(Path(a.tracks), seq.name), k=K, lo=LO, hi=HI)
    out = write_labels(Path(a.out), rows)
    counts = Counter(r.label for r in rows)
    meta = {
        "seq": seq.name,
        "split": seq.split,
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "k": K,
        "lo": LO,
        "hi": HI,
        "csv_sha256": _sha256(out),
        "tracks_sha256": _sha256(Path(a.tracks)),
        "labeler_commit": git_info()["git_commit"],
        "git_dirty": git_info()["git_dirty"],
        "created_at": iso(utc_now()),
    }
    write_json(out.with_suffix(".json"), meta)
    print(f"{seq.name}: {len(rows)} satır {dict(sorted(counts.items()))} → {out}")
    return 0


def _cmd_explore(a: argparse.Namespace) -> int:
    cfg = json.loads(Path(a.dev).read_text(encoding="utf-8"))
    data = Path(a.data)
    tracks = data / "labels" / "tracks_ir.csv"
    per_seq = {}
    for name in cfg["sequences"]:
        seq = load_sequence(data / "sequences" / name, allow_test=False)
        per_seq[name] = explore_sequence(seq, seq.rgb, gt_tracks(tracks, name))
        s = per_seq[name]
        print(
            f"{name}: IR adım ortanca {s['ir']['step_px']['median']:.2f} px, "
            f"zincir ölçek {s['ir']['chained_scale']:.2f}, v_hist {s['v_hist']}"
        )
    rec = {
        "split": cfg["split"],
        "sequences": cfg["sequences"],
        "method": METHOD,
        "per_seq": per_seq,
        "tracks_sha256": _sha256(tracks),
        **git_info(),
        "created_at": iso(utc_now()),
    }
    write_json(Path(a.out), rec)
    print(f"→ {a.out}")
    return 0


def _labels_file(labels_dir: Path, seq: str) -> Path:
    found = sorted(Path(labels_dir).glob(f"labels_*_{seq}.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{labels_dir}: {seq} için tek bir labels_<bölüm>_{seq}.csv yok")
    return found[0]


def _load_labels(labels_dir: Path, seqs: list[str]) -> tuple[dict, dict[str, str]]:
    labels, shas = {}, {}
    for seq in seqs:
        f = _labels_file(labels_dir, seq)
        shas[seq] = _sha256(f)
        labels.update({(r.seq, r.frame, r.track_id): r for r in read_labels(f)})
    return labels, shas


def _cmd_audit(a: argparse.Namespace) -> int:
    cfg = json.loads(Path(a.dev).read_text(encoding="utf-8"))
    data = Path(a.data)
    tracks = data / "labels" / "tracks_ir.csv"
    seqs = {
        n: load_sequence(data / "sequences" / n, allow_test=a.i_know_this_is_test)
        for n in cfg["sequences"]
    }
    gts = {n: gt_tracks(tracks, n) for n in seqs}
    labels, shas = _load_labels(Path(a.labels_dir), list(seqs))
    pop = population(gts, {n: s.frames for n, s in seqs.items()}, k=K)
    seen: set = set()
    for prev in a.exclude_key:  # önceki turlarda gösterilen örnekler
        for s in json.loads(Path(prev).read_text(encoding="utf-8"))["samples"].values():
            seen.add((s["seq"], s["frame"], s["track_id"]))
    picked = sample(pop, labels, seed=a.seed, k=K, exclude=seen)
    ids = render(picked, seqs, gts, Path(a.out), k=K, seed=a.seed)
    key = {
        "split": cfg["split"],
        "seed": a.seed,
        "k": K,
        "population": len(pop),
        "excluded": len(seen),
        "exclude_keys": [_sha256(Path(p)) for p in a.exclude_key],
        "samples": {i: s._asdict() for i, s in ids.items()},
        "labels_sha256": shas,
        "tracks_sha256": _sha256(tracks),
        **git_info(),
        "created_at": iso(utc_now()),
    }
    write_json(Path(a.key), key)
    kinds = Counter(s.kind for s in picked)
    print(f"nüfus {len(pop)}, örnek {dict(kinds)} → {Path(a.out) / 'index.html'} (anahtar {a.key})")
    return 0


def _cmd_audit_score(a: argparse.Namespace) -> int:
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    ids = {i: Sample(**s) for i, s in key["samples"].items()}
    labels, shas = _load_labels(Path(a.labels_dir), sorted({s.seq for s in ids.values()}))
    raw = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    answers = {r["id"]: r["answer"] for r in raw}
    rec = {
        **score(ids, labels, answers),
        "split": key.get("split"),
        "key_sha256": _sha256(Path(a.key)),
        "answers_sha256": _sha256(Path(a.answers)),
        "labels_sha256": shas,
        **git_info(),
        "created_at": iso(utc_now()),
    }
    write_json(Path(a.out), rec)
    for name in ("a_agreement", "b_unsure", "c_covered"):
        r = rec[name]
        print(f"{name}: {r['k']}/{r['n']} → {'geçti' if r['pass'] else 'geçmedi'}")
    print(f"Kabul 4: {rec['verdict']} → {a.out}")
    return 0 if rec["verdict"] == "pass" else 1


def _split(a: argparse.Namespace) -> tuple[str, Path, list[str]]:
    """`dev`: `configs/m3ot_dev.json` dizileri; `test`: `data/m3ot_test/sequences` (bayrakla)."""
    if a.split == "test":
        if not a.i_know_this_is_test:
            raise ValueError(f"M3OT test yalnız son ölçümde {TEST_FLAG} ile ölçülür")
        data = Path(a.data or "../data/m3ot_test")
        return "test", data, sorted(p.name for p in (data / "sequences").iterdir())
    cfg = json.loads(Path(a.dev).read_text(encoding="utf-8"))
    return cfg["split"], Path(a.data or "../data/m3ot_dev"), list(cfg["sequences"])


def _inputs(a: argparse.Namespace):
    split, data, names = _split(a)
    tracks = data / "labels" / "tracks_ir.csv"
    seqs = {n: load_sequence(data / "sequences" / n, a.i_know_this_is_test) for n in names}
    for n, sq in seqs.items():
        if sq.split != split:
            raise ValueError(f"{n}: index.json split {sq.split!r}, beklenen {split!r}")
    gts = {n: gt_tracks(tracks, n) for n in names}
    labels, label_shas = _load_labels(Path(a.labels_dir), names)
    cfg = MotionCfg.load(Path(a.motion))
    meta = {
        "split": split,
        "sequences": names,
        "labels_sha256": label_shas,
        "tracks_sha256": _sha256(tracks),
        "motion_sha256": cfg.sha256,
    }
    return seqs, gts, labels, cfg, meta


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _cmd_measure(a: argparse.Namespace) -> int:
    seqs, gts, labels, cfg, meta = _inputs(a)
    if a.row == "oracle":
        runs = {n: measure.oracle_run(sq, gts[n], cfg) for n, sq in seqs.items()}
        meta["runs_sha256"] = None
    else:
        if not a.run:
            raise ValueError("--row engine için her dizinin motor koşusu --run ile verilmeli")
        runs = dict(measure.load_engine_run(Path(p)) for p in a.run)
        missing = sorted(set(seqs) - set(runs))
        if missing:
            raise ValueError(f"motor koşusu eksik: {missing}")
        meta["runs_sha256"] = {Path(p).name: _sha256(Path(p)) for p in a.run}
    matched = a.row == "engine"
    rec = {"row": a.row, **meta, **measure.score(list(labels.values()), runs, gts, matched)}
    if a.row == "oracle":
        rec["gate_k9"] = measure.k9(rec)
    rec["vs_human"] = None
    if a.audit_key and a.audit_answers:
        key = json.loads(Path(a.audit_key).read_text(encoding="utf-8"))
        ids = {i: Sample(**v) for i, v in key["samples"].items()}
        raw = json.loads(Path(a.audit_answers).read_text(encoding="utf-8"))
        answers = {r["id"]: r["answer"] for r in raw}
        rec["vs_human"] = {
            **measure.vs_human(ids, answers, runs, gts, matched),
            "key_sha256": _sha256(Path(a.audit_key)),
            "answers_sha256": _sha256(Path(a.audit_answers)),
        }
    rec.update({**git_info(), "created_at": iso(utc_now())})
    write_json(Path(a.out), rec)
    lab = rec["labeled"]
    print(
        f"{a.row}: BA {_fmt(lab['BA'])} · R_mov {_fmt(lab['R_mov'])} · R_sta {_fmt(lab['R_sta'])}"
        f" · kapsam {_fmt(lab['coverage'])} · P_mov {_fmt(rec['precision']['moving']['P'])}"
    )
    if rec["vs_human"]:
        h = rec["vs_human"]
        print(f"insanla uyum: {h['agree']}/{h['n']} ({_fmt(h['rate'])})")
    if a.row == "oracle":
        print(f"Kabul 9: {'geçti' if rec['gate_k9']['pass'] else 'GEÇMEDİ'} → {a.out}")
        return 0 if rec["gate_k9"]["pass"] else 1
    print(f"→ {a.out}")
    return 0


def _cmd_ego_report(a: argparse.Namespace) -> int:
    seqs, gts, labels, cfg, meta = _inputs(a)
    per_seq, trials, ratios = {}, [], []
    for n, sq in seqs.items():
        run = measure.oracle_run(sq, gts[n], cfg)
        lab = [r for r in labels.values() if r.seq == n]
        st = drift.stationary_stability(lab, run)
        tr = drift.reloc_trials(sq, gts[n], cfg)
        trials += tr
        ratios += st["ratios"]
        per_seq[n] = {
            "ego": measure.ego_quality(run, cfg.ego.relink_max_gap),
            "map_drift": drift.map_drift(sq, run, gts[n]),
            "stability": st,
            "reloc": drift.reloc_summary(tr),
            "frozen": measure.frozen_rates(sq, cfg),
        }
        e, d = per_seq[n]["ego"], per_seq[n]["map_drift"]
        drift_txt = " / ".join(_fmt(d[k]["median_px"]) for k in ("15", "60", "150"))
        print(
            f"{n}: ok {_fmt(e['ok_rate'])}, nesiller {e['map_gens']}, kayma Δ15/60/150 ortanca "
            f"{drift_txt} px, yeniden konumlama {per_seq[n]['reloc']['success']}/{len(tr)}",
            flush=True,
        )
    rec = {
        **meta,
        "row": "oracle",
        "per_seq": per_seq,
        "stability": {
            "segments": len(ratios),
            "median": drift.pct(ratios, 50),
            "p95": drift.pct(ratios, 95),
        },
        "reloc": drift.reloc_summary(trials),
        **git_info(),
        "created_at": iso(utc_now()),
    }
    write_json(Path(a.out), rec)
    r = rec["reloc"]
    print(
        f"yeniden konumlama {r['success']}/{r['trials']}, "
        f"yanlış bağlama {r['false_links']} → {a.out}"
    )
    return 0


def _cmd_fixtures(a: argparse.Namespace) -> int:
    from baykus.engine_ref.models import load_models
    from baykus.motion import fixtures as fx
    from baykus.track.runner import load_cfg

    dev_cfg = json.loads(Path(a.dev).read_text(encoding="utf-8"))
    models = load_models(Path(a.models), Path(a.models_manifest))
    cfg = MotionCfg.load(Path(a.motion))
    meta = fx.build(
        models,
        Path(a.data),
        dev_cfg,
        cfg,
        load_cfg(Path(a.tracker)),
        Path(a.stage),
        log=lambda s: print(s, flush=True),
        start_from=a.start_from,
        candidates=[int(x) for x in a.candidates.split(",")] if a.candidates else None,
    )
    sha, size = fx.write_tar(Path(a.stage), Path(a.out_tar))
    name = Path(a.out_tar).name
    manifest = {
        "release": a.release,
        "base_url": f"https://github.com/OgzhnTarhn/baykus/releases/download/{a.release}/",
        "files": {name: {"sha256": sha, "bytes": size}},
    }
    write_json(Path(a.manifest), manifest)

    out = Path(a.out)
    dets = json.loads(gzip.decompress(Path(a.dets).read_bytes()))
    tracks = json.loads(gzip.decompress(Path(a.tracks).read_bytes()))
    maps = fx.maps_val(Path(a.data), dets, cfg)
    maps["dets_sha256"] = _sha256(Path(a.dets))
    maps_path = out / "motion" / "maps_val.json.gz"
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.write_bytes(fx.gz_json(maps))
    exp = fx.motion_expected(maps, dets, tracks, cfg)
    exp["maps_sha256"] = _sha256(maps_path)
    exp["tracks_sha256"] = _sha256(Path(a.tracks))
    exp_path = out / "expected" / "motion_val.json"
    text = json.dumps(exp, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    exp_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    readme = (Path(a.stage) / "README.md").read_text(encoding="utf-8")
    readme += f"\nPaket: `{name}` sha256 `{sha}`, {size} bayt (Release `{a.release}`).\n"
    (out / "motion" / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(f"klip {meta['source']} · denenen {len(meta['notes'])} · eşiğe yakın {meta['near']}")
    print(f"{a.out_tar}: {size / 1e6:.1f} MB, sha256 {sha}")
    for p in (maps_path, exp_path):
        print(f"{p}: {p.stat().st_size / 1e3:.0f} KB")
    if size > 40e6:
        print("UYARI: paket > 40 MB (plan: klibi 45 kareye indir)")
    return 0


def _cmd_clip_ref(a: argparse.Namespace) -> int:
    import onnxruntime

    from baykus.edge.backends import OrtBackend
    from baykus.engine_ref.models import load_models
    from baykus.engine_ref.pipeline import Pipeline
    from baykus.motion import fixtures as fx
    from baykus.track.runner import load_cfg

    models = load_models(Path(a.models), Path(a.models_manifest))
    cfg = MotionCfg.load(Path(a.motion))
    root = Path(a.tar)
    scen = json.loads((root / "scenarios.json").read_text(encoding="utf-8"))
    pipe = Pipeline(models, backend=OrtBackend(models.onnx))
    runs = fx.run_scenarios(pipe, load_cfg(Path(a.tracker)), cfg, root, scen)
    rec = {
        "scenarios": {k: v["frames"] for k, v in runs.items()},
        "gate": "advise",
        "onnxruntime": onnxruntime.__version__,
        "motion_sha256": cfg.sha256,
        "scenarios_sha256": _sha256(root / "scenarios.json"),
        **git_info(),
        "created_at": iso(utc_now()),
    }
    write_json(Path(a.out), rec)
    print(f"{len(runs)} senaryo, {sum(len(v['frames']) for v in runs.values())} kare → {a.out}")
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--dev", default="../configs/m3ot_dev.json")
    p.add_argument("--data", default=None, help="varsayılan ../data/m3ot_<dev|test>")
    p.add_argument("--labels-dir", default="../data/motion")
    p.add_argument("--motion", default="../configs/engine/motion.json")
    p.add_argument(TEST_FLAG, action="store_true", dest="i_know_this_is_test")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-motion")
    sub = p.add_subparsers(required=True)
    lab = sub.add_parser("labels", help="dizi + GT izleri → referans hareket etiketleri (CSV)")
    lab.add_argument("--seq-dir", required=True)
    lab.add_argument("--tracks", required=True, help="pack_labels tracks_ir.csv")
    lab.add_argument("--out", required=True)
    lab.add_argument(TEST_FLAG, action="store_true", dest="i_know_this_is_test")
    lab.set_defaults(func=_cmd_labels)
    ex = sub.add_parser("explore", help="tasarım §4 val keşfini yeniden üretir")
    ex.add_argument("--dev", default="../configs/m3ot_dev.json")
    ex.add_argument("--data", default="../data/m3ot_dev")
    ex.add_argument("--out", default="../results/motion/explore_val.json")
    ex.set_defaults(func=_cmd_explore)
    au = sub.add_parser("audit", help="kör insan örneklemi sayfası (tasarım §5.3)")
    au.add_argument("--dev", default="../configs/m3ot_dev.json")
    au.add_argument("--data", default="../data/m3ot_dev")
    au.add_argument("--labels-dir", default="../data/motion")
    au.add_argument("--out", default="../data/motion/audit")
    au.add_argument("--key", default="../data/motion/audit_key_val.json")
    au.add_argument("--seed", type=int, default=42)
    au.add_argument("--exclude-key", action="append", default=[], help="önceki turun anahtarı")
    au.add_argument(TEST_FLAG, action="store_true", dest="i_know_this_is_test")
    au.set_defaults(func=_cmd_audit)
    sc = sub.add_parser("audit-score", help="kör örneklem cevapları → Kabul 4")
    sc.add_argument("--key", default="../data/motion/audit_key_val.json")
    sc.add_argument("--labels-dir", default="../data/motion")
    sc.add_argument("--answers", required=True)
    sc.add_argument("--out", default="../results/motion/label_audit_val.json")
    sc.set_defaults(func=_cmd_audit_score)
    me = sub.add_parser("measure", help="hareketli / duran ölçümü (tasarım §6.3; oracle: Kabul 9)")
    _common(me)
    me.add_argument("--row", choices=("oracle", "engine"), required=True)
    me.add_argument("--run", action="append", default=[], help="motor dizi koşusu (dizi başına)")
    me.add_argument("--audit-key", default=None)
    me.add_argument("--audit-answers", default=None)
    me.add_argument("--out", required=True)
    me.set_defaults(func=_cmd_measure)
    eg = sub.add_parser("ego-report", help="ego, harita kayması, kararlılık, yeniden konumlama")
    _common(eg)
    eg.add_argument("--out", required=True)
    eg.set_defaults(func=_cmd_ego_report)
    fx = sub.add_parser("fixtures", help="hareket eşlik fikstürleri ve Release paketi (§5.9)")
    fx.add_argument("--dev", default="../configs/m3ot_dev.json")
    fx.add_argument("--data", default="../data/m3ot_dev")
    fx.add_argument("--models", default="../models")
    fx.add_argument("--models-manifest", default="../configs/engine/models.json")
    fx.add_argument("--motion", default="../configs/engine/motion.json")
    fx.add_argument("--tracker", default="../configs/engine/tracker.json")
    fx.add_argument("--dets", default="../fixtures/tracks/dev_dets.json.gz")
    fx.add_argument("--tracks", default="../fixtures/tracks/dev_tracks_py.json.gz")
    fx.add_argument("--stage", default="../build/fixtures-f4")
    fx.add_argument("--out-tar", default="../build/fixtures-f4.tar")
    fx.add_argument("--out", default="../fixtures")
    fx.add_argument("--manifest", default="../configs/engine/fixtures.json")
    fx.add_argument("--release", default="fixtures-f4")
    fx.add_argument("--start-from", type=int, default=1, help="taramaya bu başlangıçtan devam")
    fx.add_argument(
        "--candidates", default=None, help="virgüllü başlangıçlar: en az sorunlu, kapsamı tam olan"
    )
    fx.set_defaults(func=_cmd_fixtures)
    cr = sub.add_parser("clip-ref", help="klip senaryolarının Python çıktısı (bu makinede)")
    cr.add_argument("--models", default="../models")
    cr.add_argument("--models-manifest", default="../configs/engine/models.json")
    cr.add_argument("--motion", default="../configs/engine/motion.json")
    cr.add_argument("--tracker", default="../configs/engine/tracker.json")
    cr.add_argument("--tar", default="../fixtures/f4", help="açılmış fixtures-f4 paketi")
    cr.add_argument("--out", default="../evidence/clip_ref.json")
    cr.set_defaults(func=_cmd_clip_ref)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
