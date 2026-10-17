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
        CONFIG = 'configs/e1_rgb.yaml'   # e2_ir, e3_early_fusion, e3c_modality_dropout
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
        # 1) smoke: %5 veri, 1 epoch — hattın uçtan uca çalıştığını doğrular.
        # Bu deneyin biten smoke'u varsa tekrar koşmaz (yeniden ölçmek için Drive'daki klasörü sil).
        from baykus.train.checkpoints import find_trial_runs
        _prev = find_trial_runs(f'{DRIVE}/runs', EXP, 'smoke')
        if SMOKE and _prev:
            print('smoke zaten var, atlandı:', _prev[-1].name)
        elif SMOKE:
            !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results --smoke
        else:
            print('SMOKE = False → atlandı')
        """
    ),
    code(
        """
        %%time
        # 2) 1 tam epoch — süresi GPU bütçesine yazılır (spec §10.2). Biten varsa tekrar koşmaz.
        from baykus.train.checkpoints import find_trial_runs
        _prev = find_trial_runs(f'{DRIVE}/runs', EXP, 'ep1')
        if EPOCH1 and _prev:
            print('1 epoch ölçümü zaten var, atlandı:', _prev[-1].name)
        elif EPOCH1:
            !cd /content/baykus/python && baykus-train --config ../{CONFIG} --data-root /content/dv_rgbt --project {DRIVE}/runs --results {DRIVE}/results --epochs 1
        else:
            print('EPOCH1 = False → atlandı')
        """
    ),
    md(
        "🚧 **KAPI:** Yeni bir deneyde 1) ve 2) bitince süreleri Claude'a getir; bütçe "
        "onaylanınca ilk hücrede `SMOKE = False`, `EPOCH1 = False`, `TRAIN = True`, `EVAL = True` "
        "yapıp yeniden Tümünü çalıştır."
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
        # Biten eğitimi ya da değerlendirmesi olmayan deney atlanır; var olan karşılaştırma tekrar yapılmaz.
        # H4'ün karşılaştırmaları (E3c, E4) 04_fusion'da.
        import json
        from pathlib import Path
        from baykus.train.checkpoints import find_runs

        def preds(exp):
            runs = find_runs(f'{DRIVE}/runs', exp, finished=True)
            if not runs:
                return None
            run_id = json.loads((runs[-1] / 'baykus_train.json').read_text(encoding='utf-8'))['run_id']
            p = Path(f'{DRIVE}/eval/{run_id}_val/predictions_coco.json.gz')
            return p if p.exists() else None

        if COMPARE:
            for A, B in [('e3_early_fusion', 'e1_rgb'), ('e3_early_fusion', 'e2_ir'), ('e2_ir', 'e1_rgb')]:
                name = f'{A}-vs-{B}_val_ir'
                if Path(f'{DRIVE}/results/compare/{name}.json').exists():
                    print('zaten var:', name)
                    continue
                PA, PB = preds(A), preds(B)
                if PA is None or PB is None:
                    print('atlandı (biten eğitim ya da değerlendirme yok):', name)
                    continue
                !cd /content/baykus/python && baykus-eval compare --gt /content/dv_rgbt/coco/val_ir.json --a "{PA}" --b "{PB}" --slices /content/meta/slices_val.json --split val --name {name} --results {DRIVE}/results
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
        CONFIGS = ['configs/e1_rgb.yaml', 'configs/e2_ir.yaml', 'configs/e3_early_fusion.yaml',
                   'configs/e3c_modality_dropout.yaml']
        BRANCH = 'h4'
        IR_MISALIGN = True   # E2'yi kaydırılmış IR'de de ölç: geç füzyonun (04_fusion) hizasızlık girdisi
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
        # Modeller: her config'in biten en son tam eğitiminin best.pt'si (bitmemiş olan atlanır)
        import os, yaml
        from baykus.train.checkpoints import find_runs
        MODEL_ARGS = []
        for c in CONFIGS:
            exp = yaml.safe_load(open(f'/content/baykus/{c}', encoding='utf-8'))['experiment']
            done = find_runs(f'{DRIVE}/runs', exp, finished=True)
            if not done:
                print(exp, '→ biten tam eğitim yok, atlandı')
                continue
            MODEL_ARGS.append(f'--model ../{c}={done[-1]}/weights/best.pt')
            print(exp, '→', done[-1])
        assert MODEL_ARGS, 'hiç biten eğitim yok'
        NPROC = os.cpu_count()
        !du -sh {DRIVE}/eval {DRIVE}/robust 2>/dev/null   # Drive'da yer: tahmin dosyaları büyük olabilir
        """
    ),
    code(
        """
        %%time
        # E6: sağlamlık paketi (val, IR etiketleri). Biten (model, koşul) atlanır.
        # Süreyi ve son 20 satırı Claude'a getir.
        ARGS = ' '.join(MODEL_ARGS) + (' --ir-misalign' if IR_MISALIGN else '')
        !cd /content/baykus/python && baykus-robust run {ARGS} --data-root /content/dv_rgbt --meta /content/meta --work /content/robust_work --project {DRIVE}/robust --results {DRIVE}/results --workers {NPROC}
        """
    ),
]


M3OT_CELLS = [
    md(
        """
        # 05 — M3OT veri setleri arası test (E7)
        Ham veri Drive'da `MyDrive/baykus/m3ot_raw/` altında (zip ya da açılmış klasör).
        **İlk çalıştırma (inceleme, KAPI M):** Çalışma zamanı → **CPU**; ilk hücrede `INSPECT = True`,
        diğerleri `False` → Tümünü çalıştır → inceleme çıktısının tamamını Claude'a getir.
        **Ölçüm:** Çalışma zamanı → **GPU** (T4 yeterli); `INSPECT = False`, `CONVERT = EVAL = FUSION = True`.
        Secrets: `GITHUB_TOKEN`. Kaggle gerekmez (DroneVehicle inmez).
        """
    ),
    code(
        """
        BRANCH = 'h4'
        INSPECT = True   # 0) ham düzeni incele (KAPI M)
        CONVERT = False  # 1) M3OT → bizim düzen (test bölümü) + 4 kanal
        EVAL = False     # 2) her modeli M3OT test'te sınıfsız değerlendir (GPU)
        FUSION = False   # 3) geç füzyon (E4'ün val'de seçilen parametreleriyle)
        CONFIGS = ['configs/e1_rgb.yaml', 'configs/e2_ir.yaml', 'configs/e3_early_fusion.yaml',
                   'configs/e3c_modality_dropout.yaml']
        DRIVE = '/content/drive/MyDrive/baykus'
        from google.colab import drive
        drive.mount('/content/drive')
        """
    ),
    CLONE_BRANCH,
    code(
        """
        %%time
        !pip -q install uv
        !nproc
        !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'GPU yok (CPU çalışma zamanı)'
        EXTRA = '[train]' if EVAL else ''
        !uv pip install --system -q -e "/content/baykus/python{EXTRA}"
        import sys
        sys.path.insert(0, '/content/baykus/python/src')
        """
    ),
    code(
        """
        %%time
        # Ham veri: Drive'dan yerel diske (Drive'dan doğrudan okumak yavaş)
        import glob, os
        RAW = '/content/m3ot_raw'
        os.makedirs(RAW, exist_ok=True)
        zips = sorted(glob.glob(f'{DRIVE}/m3ot_raw/*.zip'))
        print('zip:', [os.path.basename(z) for z in zips])
        for z in zips:
            !unzip -q -n "{z}" -d {RAW}
        if not zips:
            !cp -r {DRIVE}/m3ot_raw/. {RAW}/
        !du -sh {RAW}
        !df -h /content
        """
    ),
    code(
        """
        # 0) KAPI M: bu hücrenin çıktısının TAMAMINI Claude'a getir; çizimler Drive'da baykus/m3ot_figs/
        if INSPECT:
            !python /content/baykus/scripts/spikes/inspect_m3ot.py {RAW} {DRIVE}/m3ot_inspect.json --figs {DRIVE}/m3ot_figs
        else:
            print('INSPECT = False → atlandı')
        """
    ),
    md(
        "🚧 **KAPI M:** İlk çalıştırmada burada dur, inceleme çıktısını Claude'a getir. Dönüştürücü "
        "bu çıktıya göre kesinleşir; Claude 'devam' deyince GPU çalışma zamanıyla ölçüme geç."
    ),
    code(
        """
        %%time
        # 1) Dönüştürme (sadece test bölümü) + 4 kanal. İstatistik satırını Claude'a getir.
        if CONVERT:
            NPROC = os.cpu_count()
            !baykus-data m3ot --raw {RAW} --out /content/m3ot --splits test --workers {NPROC}
            !baykus-data rgbt4 --out /content/m3ot --splits test --workers {NPROC}
            !cp /content/m3ot/meta/m3ot_stats.json {DRIVE}/results/m3ot_stats.json
            !du -sh /content/m3ot/*
        else:
            print('CONVERT = False → atlandı')
        """
    ),
    code(
        """
        %%time
        # 2) E7: her model M3OT test'te, sınıfsız 'vehicle' AP (spec §8.3). 'map50' satırlarını getir.
        import json, yaml
        from pathlib import Path
        from baykus.train.checkpoints import find_runs
        if EVAL:
            for c in CONFIGS:
                exp = yaml.safe_load(open(f'/content/baykus/{c}', encoding='utf-8'))['experiment']
                done = find_runs(f'{DRIVE}/runs', exp, finished=True)
                if not done:
                    print('atlandı (biten eğitim yok):', exp)
                    continue
                W = f'{done[-1]}/weights/best.pt'
                !cd /content/baykus/python && baykus-eval run --config ../{c} --weights "{W}" --data-root /content/m3ot --meta /content/m3ot/meta --split test --dataset m3ot --class-agnostic --project {DRIVE}/m3ot --results {DRIVE}/results
        else:
            print('EVAL = False → atlandı')
        """
    ),
    code(
        """
        # 3) Geç füzyon: E1 ve E2'nin M3OT tahminleri, E4'ün val'de seçilen parametreleriyle.
        # E4b uygulanmaz (M3OT'ta aydınlatma etiketi yok).
        if FUSION:
            def rid(exp):
                d = find_runs(f'{DRIVE}/runs', exp, finished=True)[-1]
                return json.loads((d / 'baykus_train.json').read_text(encoding='utf-8'))['run_id']
            P = lambda r: f'{DRIVE}/m3ot/{r}_m3ot_test/predictions_coco.json.gz'
            PR, PI = P(rid('e1_rgb')), P(rid('e2_ir'))
            !cd /content/baykus/python && baykus-fusion apply --config ../configs/fusion/e4_late_fusion.yaml --selection {DRIVE}/results/fusion/e4_late_fusion_val_selection.json --rgb-preds "{PR}" --ir-preds "{PI}" --data-root /content/m3ot --meta /content/m3ot/meta --split test --dataset m3ot --class-agnostic --project {DRIVE}/m3ot --results {DRIVE}/results
        else:
            print('FUSION = False → atlandı')
        """
    ),
]


FUSION_CELLS = [
    md(
        """
        # 04 — Geç füzyon (E4, E4b) ve karşılaştırmalar (CPU)
        **Çalışma zamanı → CPU** (GPU gerekmez). Secrets: `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`.
        Gerekenler Drive'da: E1 ve E2'nin değerlendirme tahminleri (`eval/`); sağlamlık için
        03_robust'ın çıktıları (`robust/`, E2 hizasızlığı dahil); karşılaştırmalar için E3 ve E3c.
        **Kullanım:** ilk hücrede anahtarları ayarla → **Çalışma zamanı → Tümünü çalıştır**.
        Biten adımlar atlanır; oturum koparsa yeniden Tümünü çalıştır.
        """
    ),
    code(
        """
        BRANCH = 'h4'
        SELECT = True    # 1) E4 ve E4b parametre seçimi (sadece val) + run JSON'ları
        ROBUST = True    # 2) geç füzyonun sağlamlık paketi (03_robust bitmiş olmalı)
        COMPARE = True   # 3) eşleştirilmiş karşılaştırmalar (temiz + kamera kaybı)
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
        !nproc   # işçi sayısı: birleştirme süresini belirler
        !uv pip install --system -q -e /content/baykus/python
        import sys
        sys.path.insert(0, '/content/baykus/python/src')
        """
    ),
    DATA_DOWNLOAD,
    code(
        """
        # Modeller: her deneyin biten en son tam eğitimi → run_id
        import json, os
        from pathlib import Path
        from baykus.train.checkpoints import find_runs
        NPROC = os.cpu_count()

        def train_run(exp):
            done = find_runs(f'{DRIVE}/runs', exp, finished=True)
            return json.loads((done[-1] / 'baykus_train.json').read_text(encoding='utf-8'))['run_id'] if done else None

        RUN = {e: train_run(e) for e in ['e1_rgb', 'e2_ir', 'e3_early_fusion', 'e3c_modality_dropout']}
        for e, r in RUN.items():
            print(e, '→', r)
        assert RUN['e1_rgb'] and RUN['e2_ir'], 'E1 ve E2 biten eğitim yok'
        EVAL = lambda r: f'{DRIVE}/eval/{r}_val/predictions_coco.json.gz'
        SEL = lambda exp: f'{DRIVE}/results/fusion/{exp}_val_selection.json'
        """
    ),
    code(
        """
        %%time
        # 1) E4: sabit ağırlık (12 hücreli ızgara); E4b: aydınlatmaya göre ağırlık (E4'ün IoU eşiğiyle).
        # Süreyi ve 'map50' satırlarını Claude'a getir.
        COMMON = (f'--rgb-preds {EVAL(RUN["e1_rgb"])} --ir-preds {EVAL(RUN["e2_ir"])} '
                  f'--rgb-run-id {RUN["e1_rgb"]} --ir-run-id {RUN["e2_ir"]} '
                  f'--data-root /content/dv_rgbt --meta /content/meta '
                  f'--project {DRIVE}/eval --results {DRIVE}/results --workers {NPROC}')
        if SELECT:
            !cd /content/baykus/python && baykus-fusion select --config ../configs/fusion/e4_late_fusion.yaml {COMMON}
            !cd /content/baykus/python && baykus-fusion select --config ../configs/fusion/e4b_late_fusion_illum.yaml --base {SEL('e4_late_fusion')} {COMMON}
        else:
            print('SELECT = False → atlandı')
        """
    ),
    code(
        """
        %%time
        # 2) Sağlamlık: E1 ve E2'nin her koşuldaki tahminleri birleştirilir (GPU yok). Biten koşul atlanır.
        if ROBUST:
            for EXP in ['e4_late_fusion', 'e4b_late_fusion_illum']:
                !cd /content/baykus/python && baykus-fusion robust --config ../configs/fusion/{EXP}.yaml --selection {SEL(EXP)} --robust-preds {DRIVE}/robust --data-root /content/dv_rgbt --meta /content/meta --project {DRIVE}/robust --results {DRIVE}/results --workers {NPROC}
        else:
            print('ROBUST = False → atlandı')
        """
    ),
    code(
        """
        # 3) Eşleştirilmiş karşılaştırmalar (val, IR etiketleri; FR-V8). Eksik model/koşul atlanır.
        for EXP in ['e4_late_fusion', 'e4b_late_fusion_illum']:
            p = Path(SEL(EXP))
            RUN[EXP] = json.loads(p.read_text(encoding='utf-8'))['run_id'] if p.exists() else None

        def preds(exp, cond):
            r = RUN.get(exp)
            if r is None:
                return None
            p = Path(EVAL(r) if cond == 'clean' else f'{DRIVE}/robust/{r}_val_{cond}/predictions_coco.json.gz')
            return p if p.exists() else None

        PAIRS = [('e3c_modality_dropout', 'e3_early_fusion', 'clean'),
                 ('e4_late_fusion', 'e3_early_fusion', 'clean'),
                 ('e4_late_fusion', 'e2_ir', 'clean'),
                 ('e4b_late_fusion_illum', 'e4_late_fusion', 'clean')]
        PAIRS += [(a, b, c) for c in ('drop_rgb', 'drop_ir') for a, b in [
            ('e3c_modality_dropout', 'e3_early_fusion'),
            ('e4_late_fusion', 'e3_early_fusion'),
            ('e4_late_fusion', 'e3c_modality_dropout')]]
        # Kamera kaybı puanı (seçim kuralı): iki kayıp koşulunun ortalaması, birleşik bootstrap
        CAMLOSS = [('e4_late_fusion', 'e3c_modality_dropout'),
                   ('e3c_modality_dropout', 'e3_early_fusion'),
                   ('e4_late_fusion', 'e3_early_fusion')]
        if COMPARE:
            for A, B in CAMLOSS:
                name = f'{A}-vs-{B}_val_camloss_ir'
                if Path(f'{DRIVE}/results/compare/{name}.json').exists():
                    print('zaten var:', name)
                    continue
                PA = [preds(A, c) for c in ('drop_rgb', 'drop_ir')]
                PB = [preds(B, c) for c in ('drop_rgb', 'drop_ir')]
                if None in PA + PB:
                    print('atlandı (tahmin yok):', name)
                    continue
                !cd /content/baykus/python && baykus-eval compare --gt /content/dv_rgbt/coco/val_ir.json --a "{PA[0]}" "{PA[1]}" --b "{PB[0]}" "{PB[1]}" --slices /content/meta/slices_val.json --split val --name {name} --results {DRIVE}/results
            for A, B, C in PAIRS:
                name = f'{A}-vs-{B}_val_ir' if C == 'clean' else f'{A}-vs-{B}_val_{C}_ir'
                if Path(f'{DRIVE}/results/compare/{name}.json').exists():
                    print('zaten var:', name)
                    continue
                PA, PB = preds(A, C), preds(B, C)
                if PA is None or PB is None:
                    print('atlandı (tahmin yok):', name)
                    continue
                !cd /content/baykus/python && baykus-eval compare --gt /content/dv_rgbt/coco/val_ir.json --a "{PA}" --b "{PB}" --slices /content/meta/slices_val.json --split val --name {name} --results {DRIVE}/results
        else:
            print('COMPARE = False → atlandı')
        """
    ),
]


EDGE_INSTALL = code(
    """
    %%time
    # Kurulum (≈ 1 dk). Sürüm satırlarını Claude'a getir.
    !pip -q install uv kaggle
    if PART == 'trt':
        !pip -q install tensorrt
    !uv pip install --system -q -e '/content/baykus/python[train]'
    import sys
    sys.path.insert(0, '/content/baykus/python/src')
    !python --version && nproc && (nvidia-smi -L || true)
    !python -c "import onnxruntime as o, ultralytics as u; print('onnxruntime', o.__version__, '· ultralytics', u.__version__)"
    !python -c "import tensorrt as t; print('tensorrt', t.__version__)" || true
    """
)

EDGE_MODELS = code(
    """
    %%time
    # Modeller ve kalibrasyon listesi (≈ 1-5 dk; T4'ün 2 çekirdeğinde rgbt4 üretimi daha uzun).
    import json, os
    from pathlib import Path
    import yaml
    from baykus.train.config import load_config
    NPROC = os.cpu_count()
    REPO = Path('/content/baykus')
    E = yaml.safe_load((REPO / 'configs/edge/e9_edge.yaml').read_text(encoding='utf-8'))
    RUN = {k: v['run_id'] for k, v in E['models'].items()}
    MOD = {k: load_config(REPO / v['config']).modality for k, v in E['models'].items()}
    W = {k: Path(f'{DRIVE}/runs/{r}/weights/best.pt') for k, r in RUN.items()}
    missing = [str(p) for p in W.values() if not p.exists()]
    assert not missing, f'best.pt yok: {missing}'
    MODELS_DIR = f'{DRIVE}/edge/models'
    PROJ = f'{DRIVE}/edge/eval'
    CFG = '../configs/edge/e9_edge.yaml'
    for m in sorted(set(MOD.values())):
        !cd /content/baykus/python && baykus-edge calib --config {CFG} --data-root /content/dv_rgbt --modality {m} --out /content/calib --build-rgbt4 --workers {NPROC}
    if 'rgbt4' in MOD.values():
        !baykus-data rgbt4 --out /content/dv_rgbt --splits val --workers {NPROC}
    VARIANTS = ['onnx_fp32', 'onnx_fp16', 'onnx_int8', 'onnx_int8_headfp32'] if PART == 'onnx' else ['trt_fp32', 'trt_fp16', 'trt_int8']
    EXT = '.onnx' if PART == 'onnx' else '.engine'
    for k in RUN:
        print(k, RUN[k], MOD[k])
    """
)

EDGE_EXPORT = code(
    """
    %%time
    # Export (onnx: model başına ≈ 1-3 dk, INT8 kalibrasyonu dahil; trt: motor başına ≈ 3-6 dk).
    # Biten varyant manifest'ten atlanır. Wall time'ı ve son 20 satırı Claude'a getir.
    for k in RUN:
        !cd /content/baykus/python && baykus-edge export --config {CFG} --model {k} --weights "{W[k]}" --out {MODELS_DIR} --variants {' '.join(VARIANTS)} --data-root /content/dv_rgbt --calib-dir /content/calib
    !ls -la {MODELS_DIR}
    """
)

EDGE_ACCURACY = code(
    """
    %%time
    # Doğruluk (val, IR etiketleri, bootstrap yok; FR-X4). ONNX: ORT CPU, bütün çekirdekler;
    # ilk varyantın süresine bakarak kalanı tahmin et. 'map50=' satırlarını Claude'a getir.
    THREADS = f'--threads {NPROC}' if PART == 'onnx' else ''
    for k in RUN:
        for V in VARIANTS:
            F = f'{MODELS_DIR}/{RUN[k]}_{V}{EXT}'
            !cd /content/baykus/python && baykus-edge eval --config {CFG} --model {k} --variant {V} --file "{F}" --data-root /content/dv_rgbt --meta /content/meta --project {PROJ} --results {DRIVE}/results {THREADS}
    # E4: iki modelin aynı varyantı, H4'te val'de seçilen WBF parametreleriyle
    for V in VARIANTS:
        R = f'{PROJ}/{RUN["e1_rgb"]}_val_{V}/predictions_coco.json.gz'
        I = f'{PROJ}/{RUN["e2_ir"]}_val_{V}/predictions_coco.json.gz'
        !cd /content/baykus/python && baykus-fusion apply --config ../configs/fusion/e4_late_fusion.yaml --selection ../results/fusion/e4_late_fusion_val_selection.json --rgb-preds "{R}" --ir-preds "{I}" --data-root /content/dv_rgbt --meta /content/meta --split val --dataset dronevehicle --variant {V} --n-boot 0 --project {PROJ} --results {DRIVE}/results --workers {NPROC}
    """
)

EDGE_LAPTOP = code(
    """
    %%time
    # Yalnız PART='onnx': laptop için modeller + bench girdileri tek zip'te (≈ 1 dk).
    # 'du' satırlarını Claude'a getir; boyutu görmeden indirme.
    if PART == 'onnx':
        !rm -rf /content/edge_laptop && mkdir -p /content/edge_laptop/models
        !cd /content/baykus/python && baykus-edge bench-inputs --config {CFG} --data-root /content/dv_rgbt --split val --preds-dir {PROJ} --out /content/edge_laptop/bench_inputs
        !cp {MODELS_DIR}/*.onnx /content/edge_laptop/models/
        !du -sh /content/edge_laptop/models /content/edge_laptop/bench_inputs
        !cd /content/edge_laptop && zip -q -r -0 {DRIVE}/edge/edge_laptop.zip models bench_inputs
        !du -sh {DRIVE}/edge/edge_laptop.zip
    else:
        print("PART='trt' → atlandı")
    """
)

EDGE_BENCH = code(
    """
    %%time
    # Yalnız PART='trt': T4 gecikmesi (§14: 50 ısınma + 500 ölçüm; ölçüm başına ≈ 1 dk).
    if PART == 'trt':
        !cd /content/baykus/python && baykus-edge bench-inputs --config {CFG} --data-root /content/dv_rgbt --split val --preds-dir {PROJ} --out /content/bench_inputs --variants {' '.join(VARIANTS)}
        for P in E['pipelines']:
            for V in VARIANTS:
                !cd /content/baykus/python && baykus-bench run --config {CFG} --pipeline {P} --variant {V} --models-dir {MODELS_DIR} --inputs /content/bench_inputs --hardware colab-t4 --results {DRIVE}/results
    else:
        print("PART='onnx' → atlandı (laptop bench'i Görev 14)")
    """
)

EDGE_CELLS = [
    md(
        """
        # 06 — Edge: export, nicemleme, doğruluk (E9)
        **H5'te eğitim yok:** eğitilmiş `s` modelleri (E1, E2, E3c) dışa aktarılır ve ölçülür.
        - `PART = 'onnx'` → **Çalışma zamanı → A100** (GPU kullanılmaz; 12 CPU çekirdeği ORT CPU
          doğruluk ölçümünü hızlandırır). ONNX FP32/FP16/INT8/INT8-head-FP32 + laptop zip'i.
        - `PART = 'trt'` → **Çalışma zamanı → T4** (TensorRT motoru donanıma özgü). FP32/FP16/INT8,
          doğruluk ve T4 gecikmesi.

        Secrets: `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`. **Kullanım:** ilk hücrede `PART`'ı seç →
        **Tümünü çalıştır**. Biten adımlar atlanır; oturum koparsa yeniden Tümünü çalıştır.
        İş bitince oturumu hemen sonlandır.
        """
    ),
    code(
        """
        BRANCH = 'h5'
        PART = 'onnx'   # 'onnx' (A100 çalışma zamanı) | 'trt' (T4)
        DRIVE = '/content/drive/MyDrive/baykus'
        from google.colab import drive
        drive.mount('/content/drive')
        """
    ),
    KAGGLE_TOKEN,
    CLONE_BRANCH,
    EDGE_INSTALL,
    DATA_DOWNLOAD,
    EDGE_MODELS,
    EDGE_EXPORT,
    EDGE_ACCURACY,
    EDGE_LAPTOP,
    EDGE_BENCH,
]

FINAL_CELLS = [
    md(
        """
        # 07 — Final test: test seti, bir kez
        ⚠️ **Test seti bir kez ölçülür; bu notebook'tan sonra hiçbir seçim değişmez.**
        Sevk kararı (H5 Görev 16) verilmeden çalıştırma.
        **Çalışma zamanı → A100** (bootstrap 12 çekirdekte paralel). Secrets: `KAGGLE_API_TOKEN`,
        `GITHUB_TOKEN`. Oturum koparsa yeniden **Tümünü çalıştır**: biten sonuçlar atlanır.
        """
    ),
    code(
        """
        BRANCH = 'h5'
        SHIPPED_PIPELINE = ''   # Görev 16'daki karar, ör. 'e4_late_fusion'
        SHIPPED_VARIANT = ''    # ör. 'onnx_int8'
        assert SHIPPED_PIPELINE and SHIPPED_VARIANT, 'önce sevk kararı (H5 Görev 16)'
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
        !uv pip install --system -q -e '/content/baykus/python[train]'
        import sys
        sys.path.insert(0, '/content/baykus/python/src')
        !python --version && nproc && (nvidia-smi -L || true)
        """
    ),
    DATA_DOWNLOAD,
    code(
        """
        %%time
        # Test bölümünün 4 kanallı TIFF'leri (E3, E3c; ≈ 2-3 dk, 12 çekirdek) ve modeller
        import json, os
        from pathlib import Path
        from baykus.train.checkpoints import find_runs
        NPROC = os.cpu_count()
        !baykus-data rgbt4 --out /content/dv_rgbt --splits test --workers {NPROC}

        def train_run(exp):
            done = find_runs(f'{DRIVE}/runs', exp, finished=True)
            return json.loads((done[-1] / 'baykus_train.json').read_text(encoding='utf-8'))['run_id'] if done else None

        EXPS = ['e1_rgb', 'e2_ir', 'e3_early_fusion', 'e3c_modality_dropout']
        RUN = {e: train_run(e) for e in EXPS}
        for e, r in RUN.items():
            print(e, '→', r)
        assert all(RUN.values()), 'biten eğitim eksik'
        DONE = lambda name: Path(f'{DRIVE}/results/runs/{name}_ir.json').exists()
        """
    ),
    code(
        """
        %%time
        # 1) E1, E2, E3, E3c test (iki etiket seti, 1000 bootstrap; model başına ≈ birkaç dk).
        for e in EXPS:
            if DONE(f'{RUN[e]}_test'):
                print('zaten var:', e)
                continue
            W = f'{DRIVE}/runs/{RUN[e]}/weights/best.pt'
            !cd /content/baykus/python && baykus-eval run --config ../configs/{e}.yaml --weights "{W}" --data-root /content/dv_rgbt --meta /content/meta --split test --project {DRIVE}/eval --results {DRIVE}/results --workers {NPROC}
        """
    ),
    code(
        """
        %%time
        # 2) E4 ve E4b test: val'de seçilen parametrelerle (yeniden seçim yok).
        R = f'{DRIVE}/eval/{RUN["e1_rgb"]}_test/predictions_coco.json.gz'
        I = f'{DRIVE}/eval/{RUN["e2_ir"]}_test/predictions_coco.json.gz'
        for EXP in ['e4_late_fusion', 'e4b_late_fusion_illum']:
            !cd /content/baykus/python && baykus-fusion apply --config ../configs/fusion/{EXP}.yaml --selection ../results/fusion/{EXP}_val_selection.json --rgb-preds "{R}" --ir-preds "{I}" --data-root /content/dv_rgbt --meta /content/meta --split test --dataset dronevehicle --project {DRIVE}/eval --results {DRIVE}/results --workers {NPROC}
        """
    ),
    code(
        """
        %%time
        # 3) Sevk edilen varyant test'te (ORT CPU, bütün çekirdekler; iki etiket seti, 1000 bootstrap).
        import yaml
        E = yaml.safe_load(Path('/content/baykus/configs/edge/e9_edge.yaml').read_text(encoding='utf-8'))
        P = E['pipelines'][SHIPPED_PIPELINE]
        V, MODELS_DIR, PROJ = SHIPPED_VARIANT, f'{DRIVE}/edge/models', f'{DRIVE}/edge/eval'
        EXT = '.engine' if V.startswith('trt_') else '.onnx'
        names = [P['model']] if P['kind'] == 'single' else [P['rgb'], P['ir']]
        for k in names:
            r = E['models'][k]['run_id']
            !cd /content/baykus/python && baykus-edge eval --config ../configs/edge/e9_edge.yaml --model {k} --variant {V} --file "{MODELS_DIR}/{r}_{V}{EXT}" --data-root /content/dv_rgbt --meta /content/meta --split test --label-sets ir rgb --n-boot 1000 --workers {NPROC} --threads {NPROC} --project {PROJ} --results {DRIVE}/results
        if P['kind'] == 'late_fusion':
            R = f"{PROJ}/{E['models'][P['rgb']]['run_id']}_test_{V}/predictions_coco.json.gz"
            I = f"{PROJ}/{E['models'][P['ir']]['run_id']}_test_{V}/predictions_coco.json.gz"
            !cd /content/baykus/python && baykus-fusion apply --config ../configs/fusion/e4_late_fusion.yaml --selection ../{P['selection']} --rgb-preds "{R}" --ir-preds "{I}" --data-root /content/dv_rgbt --meta /content/meta --split test --dataset dronevehicle --variant {V} --project {PROJ} --results {DRIVE}/results --workers {NPROC}
        """
    ),
]


HEALTH_CELLS = [
    md(
        """
        # 08 — Faz 2: sensör sağlığı kapısı (tek oturum)
        ⚠️ **Test seti bu notebook'ta bir kez, 3 koşulda ölçülür; sonra eşik ya da kural değişmez.**
        **Çalışma zamanı → A100** (GPU: Ultralytics val; 12 CPU çekirdeği: koşul üretimi ve
        bootstrap). Secrets: `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`. Oturum koparsa yeniden
        **Tümünü çalıştır**: eşik dosyası ve biten koşullar atlanır.
        Süreler tahmindir (Faz 1'de E6 koşul başına ≈ 1 dk). Bütün çıktıları Claude'a getir.
        """
    ),
    code(
        """
        BRANCH = 'f2'
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
        !uv pip install --system -q -e '/content/baykus/python[train]'
        import sys
        sys.path.insert(0, '/content/baykus/python/src')
        !python --version && nproc && (nvidia-smi -L || true)
        """
    ),
    DATA_DOWNLOAD,
    code(
        """
        # Model ve Faz 1 taban tahminleri (kapısız E3c; val, 16 koşul). Eksik varsa dur.
        import json, os
        from pathlib import Path
        from baykus.robustness.corruptions import suite_conditions
        from baykus.train.checkpoints import find_runs
        NPROC = os.cpu_count()
        EXPECT = '2026-10-15_e3c-modality-dropout_s_d4e1280'   # results/robust'taki E3c kayıtları
        done = find_runs(f'{DRIVE}/runs', 'e3c_modality_dropout', finished=True)
        assert done, 'E3c tam eğitimi bulunamadı'
        RUN = json.loads((done[-1] / 'baykus_train.json').read_text(encoding='utf-8'))['run_id']
        print('E3c →', RUN)
        assert RUN == EXPECT, f"run kimliği {RUN} ≠ {EXPECT}: Claude'a söyle"
        MODEL = f'../configs/e3c_modality_dropout.yaml={done[-1]}/weights/best.pt'
        missing = [c.id for c in suite_conditions()
                   if not Path(f'{DRIVE}/robust/{RUN}_val_{c.id}/predictions_coco.json.gz').exists()]
        print('eksik Faz 1 taban tahmini:', missing or 'yok')
        assert not missing, "eksik koşulları Claude'a söyle (03_robust ile yeniden üretilir)"
        TH = f'{DRIVE}/results/health/thresholds.json'
        COMMON = f'--data-root /content/dv_rgbt --meta /content/meta --project {DRIVE}/robust --results {DRIVE}/results --workers {NPROC}'
        TEST_CONDS = 'clean noise-20 misalign-20'
        """
    ),
    code(
        """
        %%time
        # 1) Eşikler: train'den 500 görüntü (seed 42) × 16 koşul, Youden J (≈ 2-4 dk).
        # Dosya varsa yeniden üretilmez: değişirse biten koşullar "başka eşik" hatası verir.
        if Path(TH).exists():
            print('eşikler zaten var:', TH)
        else:
            !cd /content/baykus/python && baykus-health fit --data-root /content/dv_rgbt --meta /content/meta --out {TH} --features-out {DRIVE}/health/train_features.csv --workers {NPROC}
        print(json.dumps({k: round(v['threshold'], 3) for k, v in json.load(open(TH))['states'].items()}))
        """
    ),
    code(
        """
        %%time
        # 2) Val, kapılı sağlamlık paketi: 16 koşul; yalnız kapının piksel değiştirdiği koşullar
        # yeniden değerlendirilir (≈ 15-25 dk). Son 20 satırı getir.
        !cd /content/baykus/python && baykus-health suite --model {MODEL} --thresholds {TH} --split val --work /content/health_work {COMMON}
        """
    ),
    code(
        """
        %%time
        # 3) Test, taban (kapısız E3c): temiz, IR gürültüsü σ 20, hizasızlık 20 px (≈ 12-18 dk).
        !cd /content/baykus/python && baykus-robust run --model {MODEL} --split test --conditions {TEST_CONDS} --n-boot 0 --work /content/robust_work {COMMON}
        """
    ),
    code(
        """
        %%time
        # 4) Test, kapılı: aynı 3 koşul (≈ 10-15 dk).
        !cd /content/baykus/python && baykus-health suite --model {MODEL} --thresholds {TH} --split test --conditions {TEST_CONDS} --work /content/health_work {COMMON}
        """
    ),
    code(
        """
        %%time
        # 5) E3c + kapı − E3c: 1000 eşleştirilmiş bootstrap (val ≈ 3-6 dk, test ≈ 5-10 dk).
        # Kabul satırlarını getir.
        for SPLIT in ['val', 'test']:
            !cd /content/baykus/python && baykus-health compare --run-id {RUN} --split {SPLIT} --robust-preds {DRIVE}/robust --data-root /content/dv_rgbt --results {DRIVE}/results --workers {NPROC}
        """
    ),
    code(
        """
        # 6) İndirilecekler: sonuç klasörleri ve laptopa inecek ONNX test tahminlerinin boyutu.
        !du -sh {DRIVE}/results/health {DRIVE}/results/gate
        !ls {DRIVE}/results/robust | grep _test_
        ONNX_TEST = f'{DRIVE}/edge/eval/{RUN}_test_onnx_fp32/predictions_coco.json.gz'
        !ls -la "{ONNX_TEST}"
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
    build(FUSION_CELLS, Path("notebooks/colab/04_fusion.ipynb"), gpu=False)
    build(M3OT_CELLS, Path("notebooks/colab/05_m3ot.ipynb"), gpu=False)  # ilk açılış CPU (inceleme)
    build(EDGE_CELLS, Path("notebooks/colab/06_edge.ipynb"), gpu=True)
    build(FINAL_CELLS, Path("notebooks/colab/07_final_test.ipynb"), gpu=True)
    build(HEALTH_CELLS, Path("notebooks/colab/08_health.ipynb"), gpu=True)
    print("ok")
