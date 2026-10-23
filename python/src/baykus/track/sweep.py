"""Takipçi eşik ızgarası (tasarım §3 A, `docs/design/faz4-kapsama.md`).

`run_grid` her (etkinleşme, yüksek güven) çifti için dizileri `runner.run_tracks` ile koşar ve
`runner.evaluate_mot` birleşik ölçütlerini toplar. `pick` ölçümden önce sabitlenen kuralı uygular:
en yüksek HOTA'dan en çok `tol` kadar düşük satırlar arasında etkinleşme eşiği yüksek olan (daha az
sahte iz), o da eşitse yüksek eşiği yüksek olan.
"""

from __future__ import annotations

from pathlib import Path

from baykus.track import runner

GRID_ACTIVATION = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
GRID_HIGH = (0.6, 0.5, 0.4, 0.3, 0.2)


def run_grid(seqs: dict[str, list[dict]], gt_csv: Path, base: dict) -> list[dict]:
    """Izgaranın her çifti için birleşik HOTA / MOTA / IDF1 ve kare başına iz sayısı."""
    rows = []
    for activation in GRID_ACTIVATION:
        for high in GRID_HIGH:
            cfg = dict(base)
            cfg["track_activation_threshold"] = activation
            cfg["high_conf_det_threshold"] = high
            out = {seq: (runner.run_tracks(frames, cfg)[0], frames) for seq, frames in seqs.items()}
            combined = runner.evaluate_mot(out, gt_csv)["combined"]
            frames_total = sum(len(frames) for frames in seqs.values())
            n_tracks = sum(
                1
                for tracks, _ in out.values()
                for fr in tracks
                for t in fr["tracks"]
                if t["track_id"] >= 0
            )
            rows.append(
                {
                    "activation": activation,
                    "high_conf": high,
                    "HOTA": combined["HOTA"],
                    "MOTA": combined["MOTA"],
                    "IDF1": combined["IDF1"],
                    "tracks_per_frame": n_tracks / frames_total if frames_total else 0.0,
                }
            )
    return rows


def pick(rows: list[dict], tol: float = 0.005) -> dict:
    """Tasarım §3 A: HOTA farkı ≤ `tol` olanlarda activation, sonra high_conf büyük olan."""
    if not rows:
        raise ValueError("sweep: boş satır listesi")
    best = max(r["HOTA"] for r in rows)
    near = [r for r in rows if r["HOTA"] >= best - tol]
    return max(near, key=lambda r: (r["activation"], r["high_conf"]))
