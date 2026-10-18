"""Takip eşliği: Python referansı ↔ C++ (`baykus_track`), Kabul 4 (tasarım §6.1).

Karşılaştırma girdi indeksiyle; kimlikler dizi başına birebir eşlenir (kimlik sayıları farklı
olabilir, eşleme tutarlı olmalı); `-1` yalnız `-1` ile. İlk farklı karede durulur. İstisna yalnız
mekanik kanıtla: (i) IoU 0,1 eşiğine 1e-5'ten yakın, (ii) skor 0,6 ya da 0,7'ye 1e-6'dan yakın,
(iii) iki atama da geçerli ve toplam IoU'ları 1e-6 içinde eşit. Her dizide karşılaştırılan
kare ≥ %90.
"""

from __future__ import annotations

IOU_THR, IOU_TOL = 0.1, 1e-5
SCORE_THRS, SCORE_TOL = (0.6, 0.7), 1e-6
COST_TOL = 1e-6
MIN_RATIO = 0.9


def _frame_ok(py: list[dict], cpp: list[dict], fwd: dict, back: dict) -> bool:
    a = {t["index"]: t["track_id"] for t in py}
    b = {t["index"]: t["track_id"] for t in cpp}
    if a.keys() != b.keys():
        return False
    for i, ta in a.items():
        tb = b[i]
        if (ta == -1) != (tb == -1):
            return False
        if ta == -1:
            continue
        if fwd.setdefault(ta, tb) != tb or back.setdefault(tb, ta) != ta:
            return False
    return True


def compare(py: dict[str, dict], cpp: dict[str, dict]) -> dict:
    out = {}
    for seq, p in py.items():
        cf = {fr["frame"]: fr["tracks"] for fr in cpp[seq]["frames"]}
        fwd: dict[int, int] = {}
        back: dict[int, int] = {}
        n, compared, first = len(p["frames"]), 0, None
        for fr in p["frames"]:
            if not _frame_ok(fr["tracks"], cf.get(fr["frame"], []), fwd, back):
                first = fr["frame"]
                break
            compared += 1
        out[seq] = {
            "frames": n,
            "compared": compared,
            "first_diff": first,
            "ratio": compared / n if n else 1.0,
        }
    return out


def _valid(assign: list[list[int]]) -> bool:
    rows = [r for r, _ in assign]
    cols = [c for _, c in assign]
    return len(set(rows)) == len(rows) and len(set(cols)) == len(cols)


def _total(iou: list[list[float]], assign: list[list[int]]) -> float:
    return sum(iou[r][c] for r, c in assign)


def evidence(dump_py: dict, dump_cpp: dict) -> str | None:
    """İlk farklı karenin iki dökümü → kanıt türü ya da None."""
    for d in (dump_py, dump_cpp):
        for m in (d.get("iou1", []), d.get("iou2", [])):
            if any(abs(v - IOU_THR) < IOU_TOL for row in m for v in row):
                return "iou_boundary"
        for det in d.get("dets", []):
            if any(abs(det["score_raw"] - s) < SCORE_TOL for s in SCORE_THRS):
                return "score_boundary"
    for stage in ("1", "2"):
        a, b = dump_py.get(f"assign{stage}", []), dump_cpp.get(f"assign{stage}", [])
        iou = dump_py.get(f"iou{stage}", [])
        if sorted(map(tuple, a)) == sorted(map(tuple, b)) or not (_valid(a) and _valid(b)):
            continue
        if abs(_total(iou, a) - _total(iou, b)) <= COST_TOL:
            return "equal_cost_assignment"
    return None


def verdict(cmp: dict, evidences: dict[str, str | None]) -> str:
    diffs = [s for s, r in cmp.items() if r["first_diff"] is not None]
    if not diffs:
        return "pass"
    if all(evidences.get(s) for s in diffs) and all(r["ratio"] >= MIN_RATIO for r in cmp.values()):
        return "pass_with_exception"
    return "fail"
