"""Gizli bilgi taşıyan kabuk komutunu çıktısız çalıştırma (Faz 3 planı Görev 2-3).

Colab'da figshare indirmesi tarayıcıdan kopyalanan çerezli bir cURL komutuyla yapılır. Komut,
standart çıktı ve hata akışı hiçbir zaman yazdırılmaz ya da istisnaya konmaz: notebook çıktısı
kaydedilse bile çerez dosyaya düşmez. Yalnız etiket ve çıkış kodu görünür.
"""

from __future__ import annotations

import subprocess


def run_secret(cmd: str, *, label: str) -> int:
    if not cmd.strip():
        raise ValueError(f"{label}: komut boş")
    try:
        rc = subprocess.run(cmd, shell=True, capture_output=True, check=False).returncode
    except Exception:  # noqa: BLE001 — istisna metni komutu içerebilir; yalnız etiket yazılır
        rc = 1
    if rc != 0:
        print(f"{label}: başarısız (çıkış {rc})", flush=True)
    return rc
