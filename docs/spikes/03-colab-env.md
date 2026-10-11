# Spike C — Colab ortamı

- **Tarih:** 2026-10-11
- **Notebook'lar:** `notebooks/colab/00_setup.ipynb` (T4 GPU çalışma zamanı), `notebooks/colab/01_data.ipynb` (CPU çalışma zamanı)
- **Değişiklik:** `00_setup`'taki ham veri indirme hücresi çıkarıldı; indirme `01_data`'da CPU'da ölçüldü (GPU kotası harcanmasın).

## Bulgular

| Soru | Bulgu |
|---|---|
| GPU / sürücü / CUDA | **Tesla T4**, 15.360 MiB (Ultralytics 14.913 MiB görüyor). Sürücü 580.82.07, CUDA 13.0 |
| Disk, RAM, CPU (T4 oturumu) | `/content` 113 GB, başlangıçta 71 GB boş · RAM 12 GiB (11 GiB kullanılabilir), swap yok · **2 CPU çekirdeği** |
| Disk (CPU oturumu, işleme sonu) | 108 GB'ın 71 GB'ı dolu (ham 14 GB + işlenmiş 37 GB) |
| Drive yazma | ok (`/content/drive/MyDrive/baykus/write_test.txt`) |
| Kaggle indirme (CPU oturumu) | 5 dk 14 sn, 14 GB (`brendanalvey/visdrone-dronevehicle`) |
| Yazılım (T4 oturumu) | **Python 3.13.15**, torch 2.11.0+cu130, Ultralytics 8.4.175, TensorRT 11.4.0.106 |
| ONNX export (yolo26n) | ok, 10,4 sn, 9,5 MB, opset 18 |
| **TensorRT FP16 engine** | **ok**, 189,5 sn, `yolo26n.engine` 53 MB. Giriş `(1, 3, 640, 640)`, çıkış `(1, 300, 6)` (uçtan uca, NMS'siz; Spike A ile tutarlı) |
| Kurulum + export toplamı | 5 dk 52 sn |
| Veri her oturumda işlenebilir mi? | **Hayır.** İndirme + işleme ~1 saat (ADR 0004). İşlenmiş veri private Kaggle veri setine yüklendi; eğitim oturumları onu indirir. |

## Dikkat çeken notlar

1. **Python 3.13.** T4 oturumu Python 3.13 ile geldi. Paketimiz `requires-python = ">=3.11,<3.13"` diyor; bu ortamda `uv pip install -e python` başarısız olur. `01_data`'daki CPU oturumunda kurulum çalıştı (sürüm o oturumda yazdırılmadı). **Aksiyon (H2 Görev 4):** aralık `<3.14`'e genişletilir, CI'a 3.13 eklenir. Colab görüntüleri zamanla değişir; her notebook başında `python --version` yazdırılır.
2. **`half` parametresi eskimiş.** Ultralytics 8.4.175: "'half' is deprecated ... Use 'quantize' instead". H5 export kodu `quantize=16` / `quantize=8` ile yazılır.
3. **FP16'da kafanın bir kısmı FP32 kaldı.** TensorRT export ModelOpt AutoCast kullanıyor: 400 düğümün 372'si FP16'ya çevrildi; değer aralığı ±512'yi aşan `model.23` (kafa: kutu çözümleme, TopK, GatherElements) düğümleri FP32 bırakıldı. Spec FR-X2'deki "INT8'de head FP32" varyantıyla uyumlu bir gözlem.
4. **Otomatik bağımlılık güncellemeleri.** Export sırasında Ultralytics `onnx`, `onnxruntime-gpu`, `onnxslim` ve `nvidia-modelopt` kurdu, bazılarını sürüm düşürerek değiştirdi ("Restart runtime" uyarısı). CuPy için iki paket çakıştı (`cupy-cuda12x`, `cupy-cuda13x`). Bu seferki export'u etkilemedi. H5'te export notebook'unda bu paketler baştan sabit sürümle kurulur.
5. **Yerel ile Colab sürümleri farklı.** Yerel: Python 3.11, torch 2.14.1 (CPU), Ultralytics 8.4.171. Colab: Python 3.13, torch 2.11.0, Ultralytics 8.4.175. Run kayıtları sürümleri yazmalı (H2: `hardware` alanının yanına yazılım sürümleri eklenebilir).
