"""Hareket raporu (baykus.motion.report) testleri — Faz 4 kabul satırları ve iddia kuralı.

Kanıt dosyaları results/ altında; burada tmp_path'te en küçük sentetik JSON'lar üretilir.
"""

import json
from pathlib import Path

from baykus.motion.report import (
    acceptance,
    claims,
    exit_ok,
    load_results,
    motion_markdown,
)

# Kabul satırı kimlikleri (rapor sırasıyla).
IDS = [
    "Kabul 1",
    "Kabul 2",
    "Kabul 3",
    "Kabul 4",
    "Kabul 5",
    "Kabul 6",
    "Kabul 7",
    "Kabul 8",
    "Kabul 9",
    "Kabul 10 (a, c)",
    "Kabul 10 b",
    "Kabul 11",
]

CI_IDS = ["Kabul 2", "Kabul 3", "Kabul 7", "Kabul 8", "Kabul 10 (a, c)", "Kabul 11"]


def _write(root: Path, rel: str, rec: dict) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")


def _labeled(
    n_stationary: int = 40,
    ba: float | None = 0.9,
    r_mov: float = 0.9,
    r_sta: float = 0.9,
    p_mov: float = 0.9,
    p_sta: float = 0.9,
) -> dict:
    """M3OT hareket kaydı: etiketli metrikler, hassasiyet, insan örneklemi, dizi kırılımı."""
    labeled = {
        "n": 100,
        "n_moving": 60,
        "n_stationary": n_stationary,
        "R_mov": r_mov,
        "R_sta": r_sta,
        "BA": ba,
        "coverage": 0.95,
    }
    return {
        "labeled": labeled,
        "precision": {"moving": {"P": p_mov}, "stationary": {"P": p_sta}},
        "vs_human": {"n": 50, "agree": 45, "rate": 0.9},
        "per_seq": {"1_03": dict(labeled)},
    }


def _ego() -> dict:
    """Ego/harita kayması kaydı: dizi başına ok oranı, kayma ve yeniden konumlanma."""
    return {
        "per_seq": {
            "1_03": {
                "ego": {"ok_rate": 0.98},
                "map_drift": {
                    "15": {"median_px": 1.2, "p95_px": 3.4},
                    "60": {"median_px": 2.0, "p95_px": 5.0},
                },
                "reloc": {"success_rate": 1.0},
            }
        }
    }


def _passing(root: Path) -> Path:
    """Bütün kanıt dosyaları geçen hâliyle yazılır."""
    _write(root, "engine/parity_ego.json", {"kabul1": "pass", "kabul10b": {"pass": True}})
    audit = {"verdict": "pass", "a_agreement": {"k": 40, "n": 50, "rate": 0.8, "pass": True}}
    _write(root, "motion/label_audit_val.json", audit)
    _write(root, "motion/label_audit_test.json", audit)
    _write(
        root,
        "bench/e3c_onnx_fp32_engine_laptop-docker_f4.json",
        {
            "kabul5": {"passed": True, "threshold_ms": 100.0, "total_p95_ms": 82.19},
            "stages": {"total": {"p50": 41.1, "p95": 82.19}},
        },
    )
    _write(
        root,
        "system/manual.json",
        {"faz4": {"wifi_off": True, "items": [{"id": "FR-U1 oynat", "file": "a.jpg", "ok": True}]}},
    )
    _write(root, "system/offline.json", {"passed": True})
    _write(
        root,
        "system/ci_f4.json",
        {
            "run_id": 123456,
            "head_sha": "abc123",
            "conclusion": "success",
            "url": "https://example.invalid/run/123456",
        },
    )
    dev_oracle = _labeled()
    dev_oracle["gate_k9"] = {"BA": 0.9, "R_mov": 0.9, "R_sta": 0.9, "pass": True}
    _write(root, "motion/m3ot_dev_oracle.json", dev_oracle)
    _write(root, "motion/m3ot_dev_engine.json", _labeled())
    _write(root, "motion/m3ot_test_oracle.json", _labeled())
    _write(root, "motion/m3ot_test_engine.json", _labeled())
    _write(root, "motion/m3ot_dev_ego.json", _ego())
    _write(root, "motion/m3ot_test_ego.json", _ego())
    return root


def _row(rows: list[dict], rid: str) -> dict:
    return next(r for r in rows if r["id"] == rid)


def test_acceptance_rows_ids(tmp_path):
    """Satır kimlikleri ve anahtarları sabit; hepsi geçince çıkış serbest."""
    rows = acceptance(load_results(_passing(tmp_path)))
    assert [r["id"] for r in rows] == IDS
    for r in rows:
        assert set(r) == {"id", "ok", "text", "known_fail"}
        assert isinstance(r["ok"], bool) and isinstance(r["text"], str)
    assert all(r["ok"] for r in rows) and exit_ok(rows)


def test_claim_rule(tmp_path):
    """Hareket iddiası: test oracle BA ≥ 0,85, motor BA ≥ 0,70, P_mov ≥ 0,80, insan örneklemi."""
    root = _passing(tmp_path)
    res = load_results(root)
    c = claims(res, acceptance(res))
    assert c["motion"]["ok"] is True

    # Motor hareket hassasiyeti eşiğin altında → iddia düşer, gerekçe P_mov'u anar.
    _write(root, "motion/m3ot_test_engine.json", _labeled(p_mov=0.79))
    res = load_results(root)
    c = claims(res, acceptance(res))
    assert c["motion"]["ok"] is False
    assert any("P_mov" in r for r in c["motion"]["reasons"])

    # Duran araç yok → BA None; eksik değer iddiayı düşürür ve gerekçe yazılır.
    _write(root, "motion/m3ot_test_engine.json", _labeled(n_stationary=0, ba=None))
    res = load_results(root)
    c = claims(res, acceptance(res))
    assert c["motion"]["ok"] is False
    assert c["motion"]["reasons"]


def test_product_claims_need_acceptance(tmp_path):
    """Ürün iddiası Kabul 6, 7, 10 (a, c), 11'e ve Kabul 10 b'ye (veya bilinen hatasına) bağlı."""
    root = _passing(tmp_path)
    # Kabul 10 b düşük ama bilinen hata olarak işaretli → engellemez.
    _write(
        root,
        "engine/parity_ego.json",
        {
            "kabul1": "pass",
            "kabul10b": {"pass": False},
            "known_fail": {
                "kabul10b": "docs/adr/0011-kabul-10b.md",
                "still_failing": True,
            },
        },
    )
    res = load_results(root)
    rows = acceptance(res)
    k10b = _row(rows, "Kabul 10 b")
    assert k10b["ok"] is False and k10b["known_fail"] == "docs/adr/0011-kabul-10b.md"
    assert exit_ok(rows) is True
    assert claims(res, rows)["product"]["ok"] is True

    # CI kırmızı → Kabul 2, 3, 7, 8, 10 (a, c), 11 düşer, ürün iddiası da düşer.
    _write(
        root,
        "system/ci_f4.json",
        {
            "run_id": 123456,
            "head_sha": "abc123",
            "conclusion": "failure",
            "url": "https://example.invalid/run/123456",
        },
    )
    res = load_results(root)
    rows = acceptance(res)
    assert claims(res, rows)["product"]["ok"] is False

    # Kabul 10 b bilinen hata olmadan düşerse çıkış da ürün iddiası da engellenir.
    _write(root, "engine/parity_ego.json", {"kabul1": "pass", "kabul10b": {"pass": False}})
    res = load_results(root)
    rows = acceptance(res)
    assert _row(rows, "Kabul 10 b")["known_fail"] is None
    assert exit_ok(rows) is False
    assert claims(res, rows)["product"]["ok"] is False


def test_missing_evidence_row_drops(tmp_path):
    """Kanıt dosyası yoksa satır "kanıt yok" ile düşer; eski bir "geçti" varsayılmaz."""
    root = _passing(tmp_path)
    (root / "system/ci_f4.json").unlink()
    res = load_results(root)
    rows = acceptance(res)
    for rid in CI_IDS:
        r = _row(rows, rid)
        assert r["ok"] is False and "kanıt yok" in r["text"]
    assert exit_ok(rows) is False
    assert claims(res, rows)["product"]["ok"] is False
    assert "kanıt yok" in motion_markdown(res)


def test_markdown_sections(tmp_path):
    """Markdown bölümleri ve test motorunun R_mov'u 3 ondalıkla; duran araç yoksa yazılır."""
    root = _passing(tmp_path)
    _write(root, "motion/m3ot_test_engine.json", _labeled(n_stationary=0, ba=None, r_mov=0.8123))
    md = motion_markdown(load_results(root))
    for heading in (
        "## Kabul",
        "## İddia kuralı",
        "## Hareket: val",
        "## Hareket: test",
        "## İnsan örneklemi",
        "## Harita kayması",
        "## Gecikme",
        "## Sınırlamalar",
    ):
        assert heading in md
    assert "0.812" in md
    assert "duran araç yok" in md
