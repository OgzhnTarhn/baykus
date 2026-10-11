# Baykuş (baykus) — Claude Code kuralları

## Proje
RGB+IR füzyonlu araç tespiti (YOLO26, yatay kutu) + C++ çıkarım motoru + Next.js operatör ekranı.
Tasarım: docs/spec.md · Haftalık plan: docs/plans/ · Kararlar: docs/adr/ · Spike bulguları: docs/spikes/ · Deney günlüğü: docs/experiments.md

## Ortam
- Windows host, repo C:\baykus (ASCII yol — kullanıcı klasöründe "ğ" var, oraya dosya koyma).
- Python komutları python/ içinden: `uv sync`, `uv run pytest -q`, `uv run ruff check . && uv run ruff format --check .`
- Veri CLI: `uv run baykus-data <alt-komut> ...`
- Eğitim/GPU işleri Google Colab'da kullanıcı tarafından çalıştırılır (notebooks/colab/). Sen script yazarsın, çalıştırmazsın.
- Python: yerel 3.11, Colab T4 3.13 (Spike C). Kod ikisinde de çalışmalı.
- İşlenmiş veri private Kaggle veri setinde: ouzhantarhan/baykus-dv-rgbt (ADR 0004). Ham veri yeniden işlenmez; dizüstüne büyük veri indirilmez.

## Kurallar
- data/, runs/, mlruns/, *.pt, *.onnx, *.engine ASLA commit edilmez.
- Deterministik kod için önce test (TDD): geometri, ayrıştırma, dönüştürme, IoU, NMS, WBF, letterbox, bozulmalar.
- (S) işaretli görevlerde implementasyonu kullanıcı yazar; sen test yazar ve incelersin.
- Görüntü okuma/yazma baykus.io.imread_any / imwrite_any ile; metin dosyaları encoding="utf-8".
- Sınıf sırası: car, truck, bus, van, freight_car (baykus.classes). COCO category_id = cls_id + 1.
- pycocotools değerlendirmesinde maxDets en az 1000.
- Metrik = results/**/*.json veya docs'taki rapor; kanıtsız sayı yazma, tahmin uydurma.
- Test seti sadece final ölçümde; seçimler val ile.
- Token/şifre hiçbir dosyaya yazılmaz.
- Conventional Commits. "Bitti" demeden önce testleri çalıştır ve çıktıyı göster.
- Commit'ler atomik: her commit tek bir mantıksal değişiklik. Commit mesajlarına ve PR açıklamalarına Claude imzası (Co-Authored-By, "Generated with Claude Code") eklenmez.
- Haftanın işi kendi dalında (h2, h3, ...); hafta sonunda main'e birleştirilir. Kullanıcı "at" derse commit + push. Push başarısızsa kullanıcıya `cd C:\baykus; git push` komutunu ver.
- Her görev sonunda dur: ne yaptığını ve nedenini kullanıcıya açıkla, onay bekle.
