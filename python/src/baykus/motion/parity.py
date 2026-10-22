"""Klip senaryoları: Python referansı ile C++ motorunun karşılaştırması (Faz 4 §6.1 Kabul 1, 10 b).

Kare başına: ego kararları (ok, source, keyframe, relinked, relocalized, provisional, map_gen,
merge from/to) ve sağlık bayrakları / donmuş kare birebir; `frame_to_map` ve `merge.transform`
köşelerde ≤ 0,25 px; iz kimlik kümeleri birebir; iz başına `motion` aynı, `speed_rel` ≤ 2e-3,
`map_xy` ≤ 0,25 px. Histerezis bandı (v, 0,03 ya da 0,10'a 2e-3'ten yakın): kararlar aynıysa satır
karşılaştırılır ve `in_band` sayılır; ayrışırsa satır serbesttir ve ayrışma o izin sonraki
satırlarına kararlar yeniden eşitlenene kadar taşınır. Senaryo başına karşılaştırılan / satır
≥ %99. Kabul 10 b: `seek` ve `long_gap`'te yeniden konumlanan karelerin `frame_to_map`'i aynı
karenin `clean` koşusundakinden köşelerde ≤ 2,0 px (iki taraf ayrı ayrı).
"""

from __future__ import annotations

import numpy as np

from baykus.motion import geometry

_EGO_FIELDS = (
    "ok",
    "source",
    "keyframe",
    "relinked",
    "relocalized",
    "provisional",
    "map_gen",
)


def _corner(a, b, w: float, h: float) -> float:
    aa = np.array(a, np.float64).reshape(2, 3)
    bb = np.array(b, np.float64).reshape(2, 3)
    return geometry.corner_error(aa, bb, w, h)


def _near(speed, lo: float, hi: float, band: float) -> bool:
    if speed is None:
        return False
    return abs(speed - lo) < band or abs(speed - hi) < band


def _compare_ego(name, frame, re, ce, errors, corner_tol, w, h):
    errs = 0
    max_corner = 0.0
    if (re is None) != (ce is None):
        errors.append(f"{name} kare {frame}: ego null farkı")
        return 1, 0.0
    if re is None:
        return 0, 0.0
    for key in _EGO_FIELDS:
        if re.get(key) != ce.get(key):
            errors.append(f"{name} kare {frame}: ego.{key} farklı")
            errs += 1
    rftm = re.get("frame_to_map")
    cftm = ce.get("frame_to_map")
    if (rftm is None) != (cftm is None):
        errors.append(f"{name} kare {frame}: ego.frame_to_map null farkı")
        errs += 1
    elif rftm is not None:
        ce_err = _corner(rftm, cftm, w, h)
        max_corner = max(max_corner, ce_err)
        if ce_err > corner_tol:
            errors.append(f"{name} kare {frame}: frame_to_map köşe hatası {ce_err:.3f}")
            errs += 1
    rm = re.get("merge")
    cm = ce.get("merge")
    if (rm is None) != (cm is None):
        errors.append(f"{name} kare {frame}: ego.merge null farkı")
        errs += 1
    elif rm is not None:
        if rm.get("from") != cm.get("from") or rm.get("to") != cm.get("to"):
            errors.append(f"{name} kare {frame}: ego.merge from/to farklı")
            errs += 1
        rtr = rm.get("transform")
        ctr = cm.get("transform")
        if (rtr is None) != (ctr is None):
            errors.append(f"{name} kare {frame}: ego.merge.transform null farkı")
            errs += 1
        elif rtr is not None:
            ce_err = _corner(rtr, ctr, w, h)
            max_corner = max(max_corner, ce_err)
            if ce_err > corner_tol:
                errors.append(f"{name} kare {frame}: merge.transform köşe hatası {ce_err:.3f}")
                errs += 1
    return errs, max_corner


def _compare_health(name, frame, rh, ch, errors) -> int:
    errs = 0
    rh = rh or {}
    ch = ch or {}
    for band_name in ("rgb", "ir"):
        rb = rh.get(band_name) or {}
        cb = ch.get(band_name) or {}
        if set(rb.get("flags", [])) != set(cb.get("flags", [])):
            errors.append(f"{name} kare {frame}: health.{band_name}.flags farklı")
            errs += 1
        if rb.get("frozen") != cb.get("frozen"):
            errors.append(f"{name} kare {frame}: health.{band_name}.frozen farklı")
            errs += 1
    rp = (rh.get("pair") or {}).get("misaligned")
    cp = (ch.get("pair") or {}).get("misaligned")
    if rp != cp:
        errors.append(f"{name} kare {frame}: health.pair.misaligned farklı")
        errs += 1
    return errs


def _compare_tracks(name, frame, rt, ct, spreading, errors, lo, hi, band, speed_tol, xy_tol):
    rows = compared = free = in_band = errs = 0
    max_speed = max_xy = 0.0
    rmap = {t["track_id"]: t for t in rt}
    cmap = {t["track_id"]: t for t in ct}
    if set(rmap) != set(cmap):
        errors.append(f"{name} kare {frame}: track_id kümesi farklı")
        errs += 1
    for tid in sorted(set(rmap) & set(cmap), key=str):
        rows += 1
        r = rmap[tid]
        c = cmap[tid]
        r_speed = r.get("speed_rel")
        c_speed = c.get("speed_rel")
        motion_eq = r.get("motion") == c.get("motion")

        if tid in spreading:
            if motion_eq:
                spreading.discard(tid)
                is_free = False
                do_compare = True
            else:
                is_free = True
                do_compare = False
        else:
            # Serbestlik yalnız kullanıldığı yerde: bantta kararlar aynıysa satır karşılaştırılır
            # (in_band); bantta ayrışırsa serbest, ayrışma izin sonraki satırlarına taşınır.
            near = _near(r_speed, lo, hi, band) or _near(c_speed, lo, hi, band)
            in_band += near
            is_free = near and not motion_eq
            do_compare = not is_free
            if is_free:
                spreading.add(tid)
            elif not motion_eq:
                errors.append(f"{name} kare {frame} kimlik {tid}: motion farklı")
                errs += 1

        if is_free:
            free += 1
        if do_compare:
            compared += 1
            if (r_speed is None) != (c_speed is None):
                errors.append(f"{name} kare {frame} kimlik {tid}: speed_rel null farkı")
                errs += 1
            elif r_speed is not None:
                d = abs(r_speed - c_speed)
                max_speed = max(max_speed, d)
                if d > speed_tol:
                    errors.append(f"{name} kare {frame} kimlik {tid}: speed_rel farkı {d:.4f}")
                    errs += 1
            rxy = r.get("map_xy")
            cxy = c.get("map_xy")
            if (rxy is None) != (cxy is None):
                errors.append(f"{name} kare {frame} kimlik {tid}: map_xy null farkı")
                errs += 1
            elif rxy is not None:
                d = max(abs(rxy[0] - cxy[0]), abs(rxy[1] - cxy[1]))
                max_xy = max(max_xy, d)
                if d > xy_tol:
                    errors.append(f"{name} kare {frame} kimlik {tid}: map_xy farkı {d:.3f}")
                    errs += 1
    return rows, compared, free, in_band, errs, max_speed, max_xy


def _compare_scenario(
    name, ref_frames, cpp_frames, errors, lo, hi, band, corner_tol, speed_tol, xy_tol, w, h
):
    rows = compared = free = in_band = err_count = 0
    max_corner = max_speed = max_xy = 0.0
    spreading: set = set()

    if len(ref_frames) != len(cpp_frames):
        errors.append(f"{name}: kare sayısı farklı ref={len(ref_frames)} cpp={len(cpp_frames)}")
        err_count += 1

    for rf, cf in zip(ref_frames, cpp_frames, strict=False):  # boy farkı yukarıda hata
        frame = rf.get("frame")
        if frame != cf.get("frame"):
            errors.append(f"{name}: frame farklı ref={frame} cpp={cf.get('frame')}")
            err_count += 1
        if rf.get("reset_before") != cf.get("reset_before"):
            errors.append(f"{name} kare {frame}: reset_before farklı")
            err_count += 1

        e, mc = _compare_ego(name, frame, rf.get("ego"), cf.get("ego"), errors, corner_tol, w, h)
        err_count += e
        max_corner = max(max_corner, mc)

        err_count += _compare_health(name, frame, rf.get("health"), cf.get("health"), errors)

        r_rows, r_comp, r_free, r_band, r_err, r_ms, r_mx = _compare_tracks(
            name,
            frame,
            rf.get("tracks") or [],
            cf.get("tracks") or [],
            spreading,
            errors,
            lo,
            hi,
            band,
            speed_tol,
            xy_tol,
        )
        rows += r_rows
        compared += r_comp
        free += r_free
        in_band += r_band
        err_count += r_err
        max_speed = max(max_speed, r_ms)
        max_xy = max(max_xy, r_mx)

    ratio_ok = rows == 0 or compared / rows >= 0.99
    passed = err_count == 0 and ratio_ok
    return {
        "rows": rows,
        "compared": compared,
        "free": free,
        "in_band": in_band,
        "errors": err_count,
        "max_corner_px": max_corner,
        "max_speed_diff": max_speed,
        "max_xy_diff": max_xy,
        "pass": passed,
    }


def _kabul10b(ref_scen, cpp_scen, reloc_tol, w, h):
    result = {"ref": [], "cpp": [], "pass": True}
    for side, scen in (("ref", ref_scen), ("cpp", cpp_scen)):
        clean = scen.get("clean") or []
        clean_by_frame = {f.get("frame"): f for f in clean}
        entries = []
        ok = True
        for sname in ("seek", "long_gap"):
            for f in scen.get(sname) or []:
                ego = f.get("ego")
                if not ego or not ego.get("relocalized"):
                    continue
                ftm = ego.get("frame_to_map")
                cf = clean_by_frame.get(f.get("frame"))
                if ftm is None or cf is None:
                    continue
                cftm = (cf.get("ego") or {}).get("frame_to_map")
                if cftm is None:
                    continue
                ce = _corner(ftm, cftm, w, h)
                entries.append({"scenario": sname, "frame": f.get("frame"), "corner_px": ce})
                if ce > reloc_tol:
                    ok = False
        result[side] = entries
        if not ok:
            result["pass"] = False
    return result


def compare(
    ref,
    cpp,
    lo: float = 0.03,
    hi: float = 0.10,
    band: float = 2e-3,
    corner_tol: float = 0.25,
    speed_tol: float = 2e-3,
    xy_tol: float = 0.25,
    reloc_tol: float = 2.0,
    w: float = 640,
    h: float = 512,
) -> dict:
    errors: list[str] = []
    per_scenario: dict[str, dict] = {}
    ref_scen = ref.get("scenarios", {})
    cpp_scen = cpp.get("scenarios", {})

    if set(ref_scen) != set(cpp_scen):
        errors.append(f"senaryo kümeleri farklı: ref={sorted(ref_scen)} cpp={sorted(cpp_scen)}")

    for name in ref_scen:
        if name not in cpp_scen:
            continue
        per_scenario[name] = _compare_scenario(
            name,
            ref_scen[name],
            cpp_scen[name],
            errors,
            lo,
            hi,
            band,
            corner_tol,
            speed_tol,
            xy_tol,
            w,
            h,
        )

    kabul10b = _kabul10b(ref_scen, cpp_scen, reloc_tol, w, h)

    ok = not errors and all(s["pass"] for s in per_scenario.values()) and kabul10b["pass"]
    return {
        "verdict": "pass" if ok else "fail",
        "per_scenario": per_scenario,
        "kabul10b": kabul10b,
        "errors": errors[:50],
    }
