# Spike A — Ultralytics YOLO26 + 4 kanal

- **Tarih:** 2026-10-05
- **Ortam:** Windows 11, CPU. Python 3.11.0, ultralytics 8.4.171, torch 2.14.1+cpu, onnx 1.23.1, onnxruntime 1.30.0, opencv 5.0.0, tifffile 2026.3.3, numpy 2.4.6
- **Script:** `scripts/spikes/ultralytics_4ch.py` (sentetik 128×128 veri, 16 train / 4 val, `yolo26n.pt`, 1 epoch)
- **Ham çıktı:** `data/spikes/u4ch/result_{hwc,chw,pages}.json`, `data/spikes/u4ch_log.txt` (git'e girmez)

## Sonuçlar

| | `hwc` (tek sayfa, contig) | `chw` (tek sayfa, separate) | **`pages` (4 gri sayfa)** |
|---|---|---|---|
| Ultralytics `imread` şekli | 128×128×4 | 128×128×4 | 128×128×4 |
| Okunan = yazılan R,G,B,T? | Hayır | Hayır | **Evet** |
| Okunan = B,G,R,T? | Hayır | Hayır | Hayır |
| Modelin gördüğü (yazılan 10,20,30,40) | **[5, 3, 2, 40]** | **[5, 3, 2, 40]** | **[10, 20, 30, 40]** |
| Eğitim / val / export / onnxruntime | ok | ok | ok |

### 1. Tek sayfalı 4 kanallı TIFF sessizce bozuluyor (kritik)
tifffile, RGB fotometrili 4 örnekli TIFF'te 4. örneği **alfa** olarak işaretliyor. OpenCV (`imdecodemulti`, `IMREAD_UNCHANGED`) okurken kanalları BGR'ye çeviriyor ve renk kanallarını alfa ile çarpıyor: 30·40/255 ≈ 5, 20·40/255 ≈ 3, 10·40/255 ≈ 2.

Gerçek veride bu, **RGB görüntünün termal parlaklığıyla karartılması** demek: soğuk bölgelerde RGB neredeyse siyaha döner. Hata vermez; eğitim çalışır ama erken füzyon bozuk veriyle eğitilir.

Ek deneme (atılmak üzere yazılmış bir probe):
- 4. örneği "belirtilmemiş" olarak işaretleyince (`extrasamples=[0]`) çarpma kalkıyor ama sıra B,G,R,T oluyor.
- 4 örnekli tek sayfa `minisblack` dosyayı Ultralytics okuyamıyor ("corrupt image").

### 2. Doğru biçim: 4 sayfalı, her sayfa tek kanal gri
`utils/patches.py::imread` birden fazla sayfayı `np.stack(frames, axis=2)` ile sırayla yığıyor; renk dönüşümü devreye girmiyor. Dosyaya R,G,B,T sayfaları yazılınca model R,G,B,T görüyor. COCO'dan kopyalanan ilk katman ağırlıkları (kanal 0-2 = R,G,B) böylece doğru kanala denk geliyor.

### 3. İlk katman aktarımı
- `w3` (COCO) şekli `[16, 3, 3, 3]`, `w4` (4 kanallı model) şekli `[16, 4, 3, 3]`.
- Kanal 0-2 COCO ağırlıklarıyla **birebir aynı** (`ch0_2_equal_coco: true`).
- Kanal 3 rastgele (`ch3_equals_rgb_mean: false`). std 0,094; COCO ağırlıklarında 0,286.
- Log: `Transferred 606/708` (eğitimsiz yükleme) ve `612/708` (eğitim). Kalan kalemler 80 sınıflı COCO başlığının 1 sınıflı başlığa uymamasından ve ilk katmanın özel yoldan kopyalanmasından geliyor.
- Kaynak koddaki davranışla tutarlı (`nn/tasks.py::BaseModel.load`).

### 4. `rgb_mean` callback'i
`on_pretrain_routine_end` callback'i model ve EMA'nın ilk katmanında kanal 3'ü RGB ortalamasıyla değiştirdi. Eğitim hatasız tamamlandı (`rgb_mean_callback_applied: true`). Ultralytics'e yama gerekmedi.

### 5. ONNX export: uçtan uca çıktı varsayılan değil (kritik)
| `export(..., nms=...)` | Çıktı | TopK düğümü | Anlamı |
|---|---|---|---|
| `None` (**varsayılan**) | `[1, 5, 336]` | 0 | Ham çıktı (4 kutu + 1 sınıf × anchor), **harici NMS gerekir** |
| `False` | `[1, 300, 6]` | 2 | YOLO26'nın NMS'siz uçtan uca çıktısı (x1, y1, x2, y2, skor, sınıf) |

Spec, YOLO26'nın uçtan uca çıktısını varsayıyordu (FR-E1). Bu çıktı ancak `nms=False` açıkça verilirse geliyor.

Exporter kaynağında ayrıca şu not var: "Static activation quantization collapses the end2end class-index output" (LiteRT INT8 için uçtan uca dal kapatılıyor). ORT INT8'de de benzer bir risk olabilir. Bu yüzden C++ motorunda NMS yolu (FR-E3) yedek değil, INT8 için gerekli olabilir; H5'te ölçülecek.

### 6. Diğer
- `val_map50 = 0.0`: 16 görüntü ve 1 epoch'la beklenen bir sonuç, bilgi taşımıyor.
- `yolo26n.pt` ilk çalıştırmada indirildi (`python/yolo26n.pt`, git'e girmez).

## Kararlar
- **TIFF_LAYOUT = `pages`** (her kanal ayrı gri sayfa, deflate), **CHANNEL_WRITE_ORDER = `RGBT`**. Görev 12 buna göre güncellendi. ADR 0004'e işlenecek.
- **E3 başlatma varyantları:** `ul_default` (Ultralytics varsayılanı, doğrulandı), `rgb_mean` (callback, doğrulandı), `random`.
- **Export:** Uçtan uca model için `nms=False`; ham çıktı + NMS yolu da desteklenir (INT8 için). Spec FR-E1, FR-X1 ve §14 güncellendi.

## Açık sorular
- 4 sayfalı TIFF'in dosya boyutu ve okuma hızı tek sayfalıya göre ne kadar farklı? Görev 12'de gerçek veriyle ölçülecek.
- ORT `quantize_static` uçtan uca grafikle (`TopK` + sınıf indeksi) doğru çalışıyor mu? H5'te iki çıktı türüyle denenecek.
