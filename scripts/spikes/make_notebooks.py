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

DATA_DOWNLOAD = code(
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
    """
)

TRAIN_CELLS = [
    md(
        """
        # 02 — Eğitim ve değerlendirme (GPU)
        **Çalışma zamanı → GPU** (T4, L4 veya A100; hangisi olduğu kurulum hücresinde yazar).
        Secrets: `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`.
        **Kullanım:** ilk kod hücresinde `CONFIG` ve adım anahtarlarını ayarla, sonra
        **Çalışma zamanı → Tümünü çalıştır**. Kapalı adımlar kendiliğinden atlanır.
        Yeni deneyde sıra: smoke + 1 epoch (süre ölç) → Claude'a getir → `TRAIN` + `EVAL`.
        Veri: private Kaggle veri seti `ouzhantarhan/baykus-dv-rgbt` (ADR 0004), birkaç dakikada iner.
        Çıktılar Drive'da `MyDrive/baykus/{runs,eval,results}`. Her epoch `weights/last.pt` olarak
        Drive'a yazılır. Oturum koparsa ayarlara dokunmadan yeniden **Tümünü çalıştır**: eğitim
        hücresi yarım kalan eğitimi bulup kaldığı epoch'tan sürdürür.
        """
    ),
    code(
        """
        CONFIG = 'configs/e1_rgb.yaml'   # e2_ir.yaml, e3_early_fusion.yaml ...
        BRANCH = 'main'                  # haftalık dalı denemek için ör. 'h3'
        # Adım anahtarları (True = çalışır, False = atlanır):
        SMOKE = True    # 1) %5 veri, 1 epoch: hattın çalıştığını doğrular (yeni deneyde ilk iş)
        EPOCH1 = True   # 2) 1 tam epoch: süre ölçümü (GPU bütçesi)
        TRAIN = False   # 3) tam eğitim; yarım kalan varsa devam eder, biten varsa atlar
        EVAL = False    # 4) değerlendirme (val): biten tam eğitimin best.pt'si
        COMPARE = False # 5) E3 − E1, E3 − E2, E2 − E1 farkları (üç modelin de 4. adımı bitmiş olmalı)
        DRIVE = '/content/drive/MyDrive/baykus'
        from google.colab import drive
        drive.mount('/content/drive')

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
        import sys  # editable kurulumun .pth'si çalışan çekirdekte okunmaz; checkpoint hücreleri için
        sys.path.insert(0, '/content/baykus/python/src')
        import yaml
        _cfg = yaml.safe_load(open(f'/content/baykus/{CONFIG}', encoding='utf-8'))
        MODE, EXP = _cfg['mode'], _cfg['experiment']
        print('MODE =', MODE, '· EXP =', EXP)
        """
    ),
    DATA_DOWNLOAD,
    code(
        """
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
        # ADR 0004: rgbt4 Kaggle'da yok; her E3 oturumunda rgb + ir'den üretilir (sadece train + val).
        # Süreyi Claude'a getir (ADR 0004'e yazılır).
        import os
        if MODE == 'early_fusion':
            NPROC = os.cpu_count()
            !baykus-data rgbt4 --out /content/dv_rgbt --splits train val --workers {NPROC}
            !du -sh /content/dv_rgbt/rgbt4
            !df -h /content
        """
    ),
    code(
        """
        %%time
        # 1) smoke: %5 veri, 1 epoch — hattın uçtan uca çalıştığını doğrular
        if SMOKE:
            !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results --smoke
        else:
            print('SMOKE = False → atlandı')
        """
    ),
    code(
        """
        %%time
        # 2) 1 tam epoch — süresi GPU bütçesine yazılır (spec §10.2)
        if EPOCH1:
            !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results --epochs 1
        else:
            print('EPOCH1 = False → atlandı')
        """
    ),
    md(
        "🚧 **KAPI:** Yeni bir deneyde 1) ve 2) bitince süreleri Claude'a getir; bütçe "
        "onaylanınca ilk hücrede `TRAIN = True`, `EVAL = True` yapıp yeniden Tümünü çalıştır."
    ),
    code(
        """
        %%time
        # 3) tam eğitim (config'teki epoch sayısı). Checkpoint: her epoch sonunda
        # {DRIVE}/runs/<run>/weights/last.pt ve 5 epoch'ta bir epochN.pt Drive'a yazılır.
        # Yarım kalan eğitim varsa kaldığı epoch'tan sürer; biten varsa yeniden eğitmez.
        from baykus.train.checkpoints import find_runs, status_report
        if TRAIN:
            todo = find_runs(f'{DRIVE}/runs', EXP, finished=False)
            done = find_runs(f'{DRIVE}/runs', EXP, finished=True)
            if todo:
                RUN_DIR = str(todo[-1])
                print('yarım kalan eğitim, devam:', RUN_DIR)
                !cd /content/baykus/python && baykus-train --config ../{CONFIG} --resume "{RUN_DIR}" --data-root /content/dv_rgbt --results {DRIVE}/results
            elif done:
                print('bu deneyin biten tam eğitimi var, yeniden eğitilmedi:', done[-1])
            else:
                !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results
        else:
            print('TRAIN = False → atlandı')
        print(status_report(f'{DRIVE}/runs', EXP))  # 📌 Drive'da hangi epoch'a kadar kayıtlı
        """
    ),
    code(
        """
        # 4) değerlendirme (val): bu config'in biten en son tam eğitiminin best.pt'si
        from baykus.train.checkpoints import find_runs
        if EVAL:
            done = [str(p) for p in find_runs(f'{DRIVE}/runs', EXP, finished=True)]
            assert done, 'biten tam eğitim yok; önce 3) bitmeli'
            W = f'{done[-1]}/weights/best.pt'
            print(W)
            !cd /content/baykus/python && baykus-eval run --config ../{CONFIG} --weights "{W}" --data-root /content/dv_rgbt --meta /content/meta --project {DRIVE}/eval --results {DRIVE}/results
        else:
            print('EVAL = False → atlandı')
        """
    ),
    code(
        """
        # 5) eşleştirilmiş karşılaştırma (val, IR etiketleri; FR-V8): A − B, aydınlatma kırılımıyla.
        # Sonuç: {DRIVE}/results/compare/*.json
        import json
        from pathlib import Path
        from baykus.train.checkpoints import find_runs

        def preds(exp):
            d = find_runs(f'{DRIVE}/runs', exp, finished=True)[-1]
            run_id = json.loads((d / 'baykus_train.json').read_text(encoding='utf-8'))['run_id']
            p = Path(f'{DRIVE}/eval/{run_id}_val/predictions_coco.json.gz')
            assert p.exists(), f'{p} yok: önce {exp} için 4) değerlendirme'
            return p

        if COMPARE:
            for A, B in [('e3_early_fusion', 'e1_rgb'), ('e3_early_fusion', 'e2_ir'), ('e2_ir', 'e1_rgb')]:
                PA, PB = preds(A), preds(B)
                !cd /content/baykus/python && baykus-eval compare --gt /content/dv_rgbt/coco/val_ir.json --a "{PA}" --b "{PB}" --slices /content/meta/slices_val.json --name {A}-vs-{B}_val_ir --results {DRIVE}/results
        else:
            print('COMPARE = False → atlandı')
        """
    ),
]

ROBUST_CELLS = [
    md(
        """
        # 03 — Sağlamlık paketi (E6, GPU)
        **Çalışma zamanı → GPU.** Secrets: `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`.
        **Kullanım:** ilk kod hücresini kontrol et, sonra **Çalışma zamanı → Tümünü çalıştır**.
        Her bozulma koşulu için val'in bozulmuş kopyası üretilir, uygun her model değerlendirilir,
        kopya silinir. Oturum koparsa yeniden **Tümünü çalıştır**: biten (model, koşul) çiftleri atlanır.
        Çıktılar: `MyDrive/baykus/results/robust/*.json`, tahminler `MyDrive/baykus/robust/`.
        """
    ),
    code(
        """
        CONFIGS = ['configs/e1_rgb.yaml', 'configs/e2_ir.yaml', 'configs/e3_early_fusion.yaml']
        BRANCH = 'main'                  # haftalık dalı denemek için ör. 'h3'
        DRIVE = '/content/drive/MyDrive/baykus'
        from google.colab import drive
        drive.mount('/content/drive')
        """
    ),
    KAGGLE_TOKEN,
    CLONE_BRANCH,
    code(
        """
        %%time
        !pip -q install uv kaggle
        !python --version
        !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        !nproc
        !uv pip install --system -q -e '/content/baykus/python[train]'
        import sys
        sys.path.insert(0, '/content/baykus/python/src')
        """
    ),
    DATA_DOWNLOAD,
    code(
        """
        # Modeller: her config'in biten en son tam eğitiminin best.pt'si
        import os, yaml
        from baykus.train.checkpoints import find_runs
        MODEL_ARGS = []
        for c in CONFIGS:
            exp = yaml.safe_load(open(f'/content/baykus/{c}', encoding='utf-8'))['experiment']
            done = find_runs(f'{DRIVE}/runs', exp, finished=True)
            assert done, f'{exp}: biten tam eğitim yok'
            MODEL_ARGS.append(f'--model ../{c}={done[-1]}/weights/best.pt')
            print(exp, '→', done[-1])
        NPROC = os.cpu_count()
        !du -sh {DRIVE}/eval {DRIVE}/robust 2>/dev/null   # Drive'da yer: tahmin dosyaları büyük olabilir
        """
    ),
    code(
        """
        %%time
        # E6: sağlamlık paketi (val, IR etiketleri). Süreyi ve son 20 satırı Claude'a getir.
        ARGS = ' '.join(MODEL_ARGS)
        !cd /content/baykus/python && baykus-robust run {ARGS} --data-root /content/dv_rgbt --meta /content/meta --work /content/robust_work --project {DRIVE}/robust --results {DRIVE}/results --workers {NPROC}
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
    build(ROBUST_CELLS, Path("notebooks/colab/03_robust.ipynb"), gpu=True)
    print("ok")
