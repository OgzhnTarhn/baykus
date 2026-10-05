"""notebooks/colab/*.ipynb dosyalarını üretir. Notebook'lar elle düzenlenmez.

Çalıştırma (repo kökünde): uvx --with nbformat python scripts/spikes/make_notebooks.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

REPO = "github.com/OgzhnTarhn/baykus.git"


def md(text: str) -> tuple[str, str]:
    return "md", dedent(text).strip()


def code(text: str) -> tuple[str, str]:
    return "code", dedent(text).strip()


CLONE = code(
    f"""
    # Repo private ise Secrets'a GITHUB_TOKEN ekle (fine-grained, sadece bu repo, 'Contents: read').
    import os
    from google.colab import userdata
    try:
        tok = userdata.get('GITHUB_TOKEN')
    except Exception:
        tok = None
    url = f'https://{{tok}}@{REPO}' if tok else 'https://{REPO}'
    if os.path.isdir('/content/baykus/.git'):
        !git -C /content/baykus pull -q {{url}} main
    else:
        !git clone -q {{url}} /content/baykus
        !git -C /content/baykus remote set-url origin https://{REPO}
    %cd /content/baykus
    !git log --oneline -3
    """
)

KAGGLE_TOKEN = code(
    """
    import os
    from google.colab import userdata
    os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
    print('token yüklendi (değer yazdırılmadı)')
    """
)

SETUP_CELLS = [
    md(
        """
        # 00 — Colab ortam spike'ı
        Çalıştırmadan önce: **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU**.
        Secrets'ta (sol menü 🔑) `KAGGLE_API_TOKEN` olmalı; repo private ise `GITHUB_TOKEN` da.
        """
    ),
    code(
        """
        !nvidia-smi
        !df -h /content
        !free -h
        !nproc
        """
    ),
    code(
        """
        from google.colab import drive
        drive.mount('/content/drive')
        !mkdir -p /content/drive/MyDrive/baykus
        !echo ok > /content/drive/MyDrive/baykus/write_test.txt
        !cat /content/drive/MyDrive/baykus/write_test.txt
        """
    ),
    KAGGLE_TOKEN,
    code(
        """
        %%time
        !pip -q install kaggle
        RAW = '/content/raw/dronevehicle'
        !kaggle datasets download brendanalvey/visdrone-dronevehicle -p {RAW} --unzip -q
        !du -sh {RAW}
        """
    ),
    CLONE,
    code(
        """
        %%time
        !pip -q install 'ultralytics>=8.4' tensorrt
        from ultralytics import YOLO
        m = YOLO('yolo26n.pt')
        print(m.export(format='engine', half=True, imgsz=640, nms=False))
        """
    ),
    md("Sonuçları `docs/spikes/03-colab-env.md` dosyasına yazmak için çıktıları Claude'a getir."),
]


DATA_CELLS = [
    md(
        """
        # 01 — Veri hazırlama (her Colab oturumunda bir kez)
        **CPU çalışma zamanı yeterli**; GPU kotası harcamaz. Secrets: `KAGGLE_API_TOKEN`
        (repo private ise `GITHUB_TOKEN` da). Çıktı: `/content/dv_rgbt` ve `/content/meta`.
        Her `%%time` hücresinin süresini ve son hücrenin disk çıktısını Claude'a getir (ADR 0004).
        """
    ),
    KAGGLE_TOKEN,
    CLONE,
    code(
        """
        %%time
        !pip -q install uv
        !uv pip install --system -q -e /content/baykus/python
        !baykus-data --help | head -3
        """
    ),
    code(
        """
        %%time
        !pip -q install kaggle
        !kaggle datasets download brendanalvey/visdrone-dronevehicle -p /content/raw --unzip -q
        !du -sh /content/raw
        """
    ),
    code(
        """
        # Ham kök: içinde train/ val/ test/ olan klasör (Kaggle zip'i bir alt klasör ekleyebilir)
        from pathlib import Path
        cands = [p for p in [Path('/content/raw'), *Path('/content/raw').glob('*')]
                 if (p / 'train').is_dir() and (p / 'val').is_dir()]
        assert cands, 'train/ ve val/ içeren klasör bulunamadı; /content/raw yapısına bak'
        RAW = cands[0].as_posix()
        print('RAW =', RAW)
        !ls {RAW}/train
        """
    ),
    code(
        """
        %%time
        !baykus-data verify --raw {RAW} --meta /content/meta
        !baykus-data convert --raw {RAW} --meta /content/meta --out /content/dv_rgbt
        !baykus-data illum --out /content/dv_rgbt --meta /content/meta --fig /content/figs
        !baykus-data rgbt4 --out /content/dv_rgbt
        !baykus-data coco --out /content/dv_rgbt --meta /content/meta
        """
    ),
    code(
        """
        !du -sh /content/dv_rgbt/* /content/raw
        !df -h /content
        """
    ),
]


def build(cells: list[tuple[str, str]], path: Path, gpu: bool) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    for i, cell in enumerate(nb.cells):
        cell.id = f"{path.stem}-{i:02d}"  # sabit kimlik: yeniden üretim gereksiz diff yaratmasın
    if gpu:  # Colab bu alanı görünce GPU çalışma zamanı ister; veri hazırlığı kota harcamasın
        nb.metadata["accelerator"] = "GPU"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


if __name__ == "__main__":
    build(SETUP_CELLS, Path("notebooks/colab/00_setup.ipynb"), gpu=True)
    build(DATA_CELLS, Path("notebooks/colab/01_data.ipynb"), gpu=False)
    print("ok")
