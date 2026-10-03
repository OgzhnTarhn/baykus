# Baykuş (baykus) — Claude Code kuralları

## Proje
RGB+IR füzyonlu araç tespiti (YOLO26, yatay kutu) + C++ çıkarım motoru + Next.js operatör ekranı.
Tasarım: docs/spec.md · Haftalık plan: docs/plans/ · Kararlar: docs/adr/ · Spike bulguları: docs/spikes/

## Ortam
- Windows host, repo C:\baykus (ASCII yol — kullanıcı klasöründe "ğ" var, oraya dosya koyma).
- Python komutları python/ içinden: `uv sync`, `uv run pytest -q`, `uv run ruff check . && uv run ruff format --check .`
- Veri CLI: `uv run baykus-data <alt-komut> ...`
- Eğitim/GPU işleri Google Colab'da kullanıcı tarafından çalıştırılır (notebooks/colab/). Sen script yazarsın, çalıştırmazsın.

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
- Her görev sonunda dur: ne yaptığını ve nedenini kullanıcıya açıkla, onay bekle.
