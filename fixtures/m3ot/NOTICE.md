# M3OT fikstürleri: atıf ve lisans

Bu klasördeki görüntüler (`frames/`, `clip/`) ve `../synthetic/`, `../tracks/` içindeki veriler
**M3OT** veri setinden türetilmiştir:

- Kaynak: M3OT, *Scientific Data* 2025 · figshare https://figshare.com/s/01fa8d1163f4e9a5a13a
- Lisans: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Repo AGPL olsa da bu
  dosyalar CC BY 4.0 altında kalır.
- Yalnız geliştirme bölümü (val); test dizileri fikstür olmaz.
- Değişiklikler: RGB PNG → JPEG; IR tek kanal PNG; sentetik bozulmalar (siyah IR, IR gürültü σ 20,
  IR 20 px kaydırma, siyah RGB, RGB karartma × 0,25, 800×600 ve 33×17 yeniden boyutlandırma);
  takip fikstüründe yalnız modelin tespitleri (ham skor ≥ 0,1).
