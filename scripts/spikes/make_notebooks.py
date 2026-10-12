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
        Secrets'ta (sol menü 🔑) repo private ise `GITHUB_TOKEN` olmalı.
        Ham veri indirme süresi `01_data`'da ölçüldü (CPU); burada GPU kotası harcanmasın diye yok.
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
        %%time
        # Spike B (H1 Görev 4): ham verinin gerçek yapısı. Bu hücrenin çıktısının TAMAMINI Claude'a getir.
        import json
        !python /content/baykus/scripts/spikes/inspect_dronevehicle.py /content/raw /content/meta/dv_inspect.json
        for s in json.load(open('/content/meta/dv_inspect.json', encoding='utf-8'))['xml_samples']:
            print('=====', s['path'])
            print(s['head'])
        """
    ),
    md(
        """
        🚧 **KAPI B — burada dur.** Aşağıdaki hücreler verinin beklenen düzende olduğunu varsayıyor
        (`<split>/<split>img`, `<split>imgr`, `<split>label`, `<split>labelr`). Önce yukarıdaki çıktıyı
        Claude'a getir; düzen farklıysa kod güncellenir. Claude "devam" deyince aşağıdan sürdür.
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


CLONE_BRANCH = code(
    f"""
    # Repo private ise Secrets'a GITHUB_TOKEN ekle (fine-grained, sadece bu repo, 'Contents: read').
    import os
    from google.colab import userdata
    try:
        tok = userdata.get('GITHUB_TOKEN')
    except Exception:
        tok = None
    url = f'https://{{tok}}@{REPO}' if tok else 'https://{REPO}'
    if not os.path.isdir('/content/baykus/.git'):
        !git clone -q {{url}} /content/baykus
    !git -C /content/baykus fetch -q {{url}} {{BRANCH}}
    !git -C /content/baykus checkout -q -B {{BRANCH}} FETCH_HEAD
    !git -C /content/baykus remote set-url origin https://{REPO}
    %cd /content/baykus
    !git log --oneline -3
    """
)

TRAIN_CELLS = [
    md(
        """
        # 02 — Eğitim ve değerlendirme (GPU)
        **Çalışma zamanı → GPU** (T4, L4 veya A100; hangisi olduğu kurulum hücresinde yazar).
        Secrets: `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`.
        Sıra: smoke → 1 epoch (süre ölç) → Claude'a getir → tam eğitim → değerlendirme.
        Veri: private Kaggle veri seti `ouzhantarhan/baykus-dv-rgbt` (ADR 0004), birkaç dakikada iner.
        Çıktılar Drive'da `MyDrive/baykus/{runs,eval,results}`. Her epoch `weights/last.pt` olarak
        Drive'a yazılır; oturum koparsa kurulum hücrelerinden sonra "3b) devam" hücresi.
        """
    ),
    code(
        """
        CONFIG = 'configs/e1_rgb.yaml'   # e2_ir.yaml, e3_early_fusion.yaml ...
        BRANCH = 'main'                  # haftalık dalı denemek için ör. 'h3'
        DRIVE = '/content/drive/MyDrive/baykus'
        from google.colab import drive
        drive.mount('/content/drive')

        import glob, json, os
        from pathlib import Path

        def runs(finished: bool) -> list[str]:
            # Bu config'in tam eğitim klasörleri (smoke ve 1 epoch hariç), eskiden yeniye
            slug = Path(CONFIG).stem.replace('_', '-')
            out = []
            for d in sorted(glob.glob(f'{DRIVE}/runs/*_{slug}_*')):
                if d.endswith(('_smoke', '_ep1')) or not os.path.exists(f'{d}/weights/last.pt'):
                    continue
                rec = f'{d}/baykus_train.json'
                status = json.load(open(rec, encoding='utf-8')).get('status') if os.path.exists(rec) else None
                if (status == 'finished') == finished:
                    out.append(d)
            return out
        """
    ),
    KAGGLE_TOKEN,
    CLONE_BRANCH,
    code(
        """
        %%time
        !pip -q install uv kaggle
        !python --version   # Spike C: Colab görüntüsü zamanla değişir
        !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        !nproc              # CPU çekirdeği: veri yükleme hızını belirler (gpu-budget.md)
        !uv pip install --system -q -e '/content/baykus/python[train]'
        import yaml
        MODE = yaml.safe_load(open(f'/content/baykus/{CONFIG}', encoding='utf-8'))['mode']
        print('MODE =', MODE)
        """
    ),
    code(
        """
        %%time
        # İşlenmiş veri: private Kaggle veri seti (ADR 0004). Kaggle yüklenen tar'ları klasöre açtı;
        # CLI klasör seçerek indiremediği için veri setinin tamamı iner (~9,8 GB).
        DS = 'ouzhantarhan/baykus-dv-rgbt'
        !kaggle datasets download {DS} -p /content/dv_rgbt --unzip -q
        !mv /content/dv_rgbt/meta /content/meta
        !baykus-data illum-relabel --meta /content/meta   # ADR 0005: Kaggle'daki etiketler eski eşikle
        !baykus-data slices --out /content/dv_rgbt --meta /content/meta
        !baykus-data yaml --out /content/dv_rgbt
        !du -sh /content/dv_rgbt/* /content/meta
        from pathlib import Path
        NEED = {'rgb': ['rgb'], 'ir': ['ir'], 'early_fusion': ['rgb', 'ir']}[MODE]
        # H1 Görev 10'dan taşındı: kutular araçların üstüne oturuyor mu? (gözle)
        # Her set kendi görüntüsünde: solda RGB etiketleri RGB'de, sağda IR etiketleri IR'de.
        import random, cv2, numpy as np
        from IPython.display import Image, display
        random.seed(1)
        root = Path('/content/dv_rgbt')

        def boxed(img_path, label_path):
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # IR gri → 3 kanal
            h_im, w_im = im.shape[:2]
            lines = label_path.read_text(encoding='utf-8').splitlines() if label_path.exists() else []
            for ln in lines:
                c, cx, cy, w, h = map(float, ln.split())
                cv2.rectangle(im, (int((cx - w / 2) * w_im), int((cy - h / 2) * h_im)),
                              (int((cx + w / 2) * w_im), int((cy + h / 2) * h_im)), (0, 255, 0), 1)
            return im, len(lines)

        for s in random.sample(sorted(p.stem for p in (root / 'rgb/images/train').glob('*.jpg')), 3):
            rgb, n_rgb = boxed(root / f'rgb/images/train/{s}.jpg', root / f'labels_rgb/train/{s}.txt')
            ir, n_ir = boxed(root / f'ir/images/train/{s}.png', root / f'labels_ir/train/{s}.txt')
            print(f'{s}: RGB {n_rgb} kutu, IR {n_ir} kutu')
            cv2.imwrite(f'/content/check_{s}.jpg', np.hstack([rgb, ir]))
            display(Image(f'/content/check_{s}.jpg'))
        for d in NEED:  # beklenen düzen: <modalite>/images/{train,val,test} (Data Explorer'da görüldü)
            for split in ('train', 'val', 'test'):
                assert Path(f'/content/dv_rgbt/{d}/images/{split}').is_dir(), f"{d}/images/{split} yok; ls çıktısını Claude'a getir"
        """
    ),
    code(
        """
        %%time
        if MODE == 'early_fusion':  # rgbt4 yüklenmedi (ADR 0004); H3'te bu hücre yeniden ele alınır
            !baykus-data rgbt4 --out /content/dv_rgbt
        """
    ),
    code(
        """
        %%time
        # 1) smoke: %5 veri, 1 epoch — hattın uçtan uca çalıştığını doğrular
        !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results --smoke
        """
    ),
    code(
        """
        %%time
        # 2) 1 tam epoch — süresi GPU bütçesine yazılır (spec §10.2)
        !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results --epochs 1
        """
    ),
    md("🚧 **KAPI: Burada dur.** Yukarıdaki iki hücrenin süresini Claude'a getir; bütçe onaylanınca devam."),
    code(
        """
        # 3) tam eğitim (config'teki epoch sayısı)
        !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results
        """
    ),
    code(
        """
        # 3b) devam: oturum koptuysa önce Drive → token → klonlama → kurulum → veri hücrelerini
        # çalıştır, sonra bunu. Bu config'in yarım kalan en son eğitimi kaldığı epoch'tan sürer.
        todo = runs(finished=False)
        print('devam edilecek:', todo[-1] if todo else 'yok')
        if todo:
            RUN_DIR = todo[-1]
            !cd /content/baykus/python && baykus-train --config ../{CONFIG} --resume "{RUN_DIR}" --data-root /content/dv_rgbt --results {DRIVE}/results
        """
    ),
    code(
        """
        # 4) değerlendirme (val): bu config'in biten en son tam eğitiminin best.pt'si
        done = runs(finished=True)
        assert done, 'biten tam eğitim yok; önce 3) veya 3b) bitmeli'
        W = f'{done[-1]}/weights/best.pt'
        print(W)
        !cd /content/baykus/python && baykus-eval run --config ../{CONFIG} --weights "{W}" --data-root /content/dv_rgbt --meta /content/meta --project {DRIVE}/eval --results {DRIVE}/results
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
    build(TRAIN_CELLS, Path("notebooks/colab/02_train.ipynb"), gpu=True)
    print("ok")
