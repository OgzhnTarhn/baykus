"""baykus-track: ByteTrack referansı (trackers==2.6.1), eşlik ve MOT ölçümü.

pack-dets — detect çıktılarını takip katmanına (ham ≥ 0,1) süzüp tek fikstür dosyasında toplar.
run       — tespit dosyası → iz dosyası (dizi başına yeni takipçi; --dump ile iç durum dökümü).
compare   — Python ↔ C++ iz dosyaları → Kabul 4 (parity_track.json).
eval      — iz + tespit + pack_labels iz CSV'si → HOTA, MOTA, IDF1 (test bayrakla).
oracle    — pack_labels iz CSV'si → M3OT kutuları skor 1 ile tespit dosyası (oracle satırı).
sweep     — eşik ızgarası → HOTA/MOTA/IDF1 ve sabit kuralla seçim (tasarım §3 A).
fixtures  — atama çözücüsü ve Kalman birim fikstürleri.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

from baykus.runinfo import git_info, iso, utc_now, write_json
from baykus.track import sweep
from baykus.track.parity import compare, evidence, verdict
from baykus.track.runner import (
    KALMAN_INIT,
    TRACK_FLOOR,
    evaluate_mot,
    kalman_steps,
    load_cfg,
    lsap_cases,
    oracle_frames,
    run_tracks,
)

TEST_FLAG = "--i-know-this-is-test"


def _read_gz(path: Path) -> dict:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def _write_gz(path: Path, rec: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: aynı içerik → aynı bayt (fikstür sha256'sı kararlı)
    data = json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0))
    return path


def _sequences(rec: dict) -> list[dict]:
    """detect çıktısı (tek dizi) ya da fikstür (`sequences`) → dizi kayıtları."""
    return rec["sequences"] if "sequences" in rec else [rec]


def _cmd_pack_dets(a: argparse.Namespace) -> int:
    seqs = []
    for f in a.dets:
        for s in _sequences(_read_gz(Path(f))):
            if s["split"] == "test":
                raise ValueError("test dizileri eşlik fikstürüne girmez (tasarım §4.6)")
            frames = [
                {
                    "frame": fr["frame"],
                    "dets": [d for d in fr["dets"] if d["score_raw"] >= TRACK_FLOOR],
                }
                for fr in s["frames"]
            ]
            seqs.append({"seq": s["seq"], "split": s["split"], "frames": frames})
    out = _write_gz(
        Path(a.out), {"floor": TRACK_FLOOR, "source": "M3OT, CC BY 4.0", "sequences": seqs}
    )
    n = sum(len(fr["dets"]) for s in seqs for fr in s["frames"])
    print(f"{out} ({len(seqs)} dizi, {n} tespit, {out.stat().st_size / 1e6:.2f} MB)")
    return 0


def _cmd_run(a: argparse.Namespace) -> int:
    cfg = load_cfg(Path(a.config))
    out = []
    for f in a.dets:
        for s in _sequences(_read_gz(Path(f))):
            if s["split"] == "test" and not a.i_know_this_is_test:
                raise ValueError(f"{s['seq']}: test dizisi; yalnız son ölçümde {TEST_FLAG} ile")
            tracks, dumps = run_tracks(s["frames"], cfg, set(a.dump or []))
            out.append(
                {
                    "seq": s["seq"],
                    "split": s["split"],
                    "frames": tracks,
                    "dumps": {str(k): v for k, v in dumps.items()},
                }
            )
    path = _write_gz(Path(a.out), {"tracker": "trackers==2.6.1", "cfg": cfg, "sequences": out})
    print(f"{path} ({len(out)} dizi)")
    return 0


def _by_seq(path: Path) -> dict[str, dict]:
    return {s["seq"]: s for s in _read_gz(path)["sequences"]}


def _cmd_compare(a: argparse.Namespace) -> int:
    py, cpp = _by_seq(Path(a.py)), _by_seq(Path(a.cpp))
    cmp = compare(py, cpp)
    ev: dict[str, str | None] = {}
    for seq, r in cmp.items():
        if r["first_diff"] is None:
            continue
        key = str(r["first_diff"])
        dp, dc = py[seq].get("dumps", {}).get(key), cpp[seq].get("dumps", {}).get(key)
        if dp is None or dc is None:
            print(f"[{seq}] kare {key} için döküm yok: iki tarafı --dump {key} ile yeniden koş")
            ev[seq] = None
        else:
            ev[seq] = evidence(dp, dc)
    rec = {
        "verdict": verdict(cmp, ev),
        "per_seq": cmp,
        "evidence": ev,
        "git_commit": git_info()["git_commit"],
        "created_at": iso(utc_now()),
    }
    print(write_json(Path(a.out), rec))
    for seq, r in cmp.items():
        print(f"[{seq}] {r['compared']}/{r['frames']} kare · ilk fark {r['first_diff']}")
    print(f"Kabul 4: {rec['verdict']}")
    return 0 if rec["verdict"] != "fail" else 1  # istisnalı geçiş (tasarım §6.1) de geçiştir


def _cmd_eval(a: argparse.Namespace) -> int:
    tracks = _by_seq(Path(a.tracks))
    dets = {s["seq"]: s for f in a.dets for s in _sequences(_read_gz(Path(f)))}
    if any(s["split"] == "test" for s in tracks.values()) and not a.i_know_this_is_test:
        raise ValueError(f"M3OT test yalnız son ölçümde {TEST_FLAG} ile ölçülür")
    rec = evaluate_mot(
        {s: (t["frames"], dets[s]["frames"]) for s, t in tracks.items()}, Path(a.gt_csv)
    )
    rec.update(
        label_set="ir",
        sequences=sorted(tracks),
        git_commit=git_info()["git_commit"],
        created_at=iso(utc_now()),
    )
    print(write_json(Path(a.out), rec))
    for s, m in [*rec["per_seq"].items(), ("birleşik", rec["combined"])]:
        print(f"{s}: HOTA {m['HOTA']:.4f} · MOTA {m['MOTA']:.4f} · IDF1 {m['IDF1']:.4f}")
    return 0


def _cmd_oracle(a: argparse.Namespace) -> int:
    seqs = []
    for d in a.seq_dir:
        idx = json.loads((Path(d) / "index.json").read_text(encoding="utf-8"))
        if idx["split"] == "test" and not a.i_know_this_is_test:
            raise ValueError(f"{idx['name']}: test dizisi; yalnız son ölçümde {TEST_FLAG} ile")
        frames = oracle_frames(Path(a.gt_csv), idx["name"], idx["frames"])
        seqs.append({"seq": idx["name"], "split": idx["split"], "frames": frames})
    out = _write_gz(Path(a.out), {"source": "oracle", "sequences": seqs})
    n = sum(len(fr["dets"]) for s in seqs for fr in s["frames"])
    print(f"{out} ({len(seqs)} dizi, {n} kutu)")
    return 0


def _cmd_sweep(a: argparse.Namespace) -> int:
    seqs: dict[str, list[dict]] = {}
    for f in a.dets:
        for s in _sequences(_read_gz(Path(f))):
            if s["split"] == "test":
                raise ValueError(f"{s['seq']}: test dizisi; eşik ızgarasına girmez")
            seqs[s["seq"]] = s["frames"]
    base = load_cfg(Path(a.config))
    rows = sweep.run_grid(seqs, Path(a.gt_csv), base)
    chosen = sweep.pick(rows)
    rec = {
        "rows": rows,
        "pick": chosen,
        "base": base,
        "label_set": "ir",
        "sequences": sorted(seqs),
        "dets_sha256": {
            Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.dets
        },
        "git_commit": git_info()["git_commit"],
        "created_at": iso(utc_now()),
    }
    print(write_json(Path(a.out), rec))
    for r in rows:
        print(
            f"activation {r['activation']:.2f} · high {r['high_conf']:.2f}: "
            f"HOTA {r['HOTA']:.4f} · MOTA {r['MOTA']:.4f} · IDF1 {r['IDF1']:.4f} · "
            f"iz/kare {r['tracks_per_frame']:.3f}"
        )
    print(
        f"seçilen: activation {chosen['activation']:.2f} · high {chosen['high_conf']:.2f} "
        f"(HOTA {chosen['HOTA']:.4f})"
    )
    return 0


def _cmd_fixtures(a: argparse.Namespace) -> int:
    out = Path(a.out_dir)
    recs = {
        "lsap_cases.json": {
            "seed": 42,
            "source": "scipy linear_sum_assignment(maximize=True)",
            "cases": lsap_cases(),
        },
        "kalman_steps.json": {
            "seed": 42,
            "source": "trackers==2.6.1 ByteTrackTracklet",
            "init": KALMAN_INIT,
            "steps": kalman_steps(),
        },
    }
    for name, rec in recs.items():
        print(write_json(out / name, rec))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-track")
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack-dets", help="detect çıktıları → takip fikstürü (ham ≥ 0,1)")
    pk.add_argument("--dets", nargs="+", required=True)
    pk.add_argument("--out", required=True)
    pk.set_defaults(func=_cmd_pack_dets)
    r = sub.add_parser("run", help="tespit dosyası → iz dosyası")
    r.add_argument("--dets", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default="../configs/engine/tracker.json")
    r.add_argument("--dump", nargs="*", type=int, help="iç durumu dökülecek kare numaraları")
    r.add_argument(TEST_FLAG, action="store_true")
    r.set_defaults(func=_cmd_run)
    c = sub.add_parser("compare", help="Python ↔ C++ iz dosyaları (Kabul 4)")
    c.add_argument("--py", required=True)
    c.add_argument("--cpp", required=True)
    c.add_argument("--out", default="../results/engine/parity_track.json")
    c.set_defaults(func=_cmd_compare)
    e = sub.add_parser("eval", help="HOTA, MOTA, IDF1")
    e.add_argument("--tracks", required=True)
    e.add_argument("--dets", nargs="+", required=True)
    e.add_argument("--gt-csv", required=True, help="pack_labels tracks_ir.csv")
    e.add_argument("--out", required=True)
    e.add_argument(TEST_FLAG, action="store_true")
    e.set_defaults(func=_cmd_eval)
    o = sub.add_parser("oracle", help="M3OT kutuları skor 1 ile tespit dosyası")
    o.add_argument("--gt-csv", required=True, help="pack_labels tracks_ir.csv")
    o.add_argument("--seq-dir", nargs="+", required=True, help="index.json'lu dizi klasörleri")
    o.add_argument("--out", required=True)
    o.add_argument(TEST_FLAG, action="store_true")
    o.set_defaults(func=_cmd_oracle)

    s = sub.add_parser("sweep", help="eşik ızgarası → HOTA/MOTA/IDF1 ve seçim")
    s.add_argument("--dets", nargs="+", required=True)
    s.add_argument("--gt-csv", required=True, help="pack_labels tracks_ir.csv")
    s.add_argument("--config", default="../configs/engine/tracker.json")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_sweep)

    fx = sub.add_parser("fixtures", help="atama ve Kalman birim fikstürleri")
    fx.add_argument("--out-dir", default="../fixtures/expected")
    fx.set_defaults(func=_cmd_fixtures)
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
