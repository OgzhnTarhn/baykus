# 0001 — Dedektör: Ultralytics YOLO26 (yedek YOLO11)

- Durum: Kabul edildi
- Tarih: 2026-10-05

## Bağlam
Proje; RGB, IR ve 4 kanallı erken füzyon modellerini 10 haftada Colab T4'te eğitmeyi, ONNX/TensorRT'ye çevirmeyi ve C++ motorunda çalıştırmayı gerektiriyor. Çok kanallı girdi, export ve takip desteği olan, iyi belgelenmiş bir çerçeveye ihtiyaç var.

## Karar
- **Ultralytics YOLO26** (ultralytics 8.4.x; H1 Spike A'da 8.4.171 ile doğrulandı). Ana model `s`, edge karşılaştırması `n`.
- **Yedek:** YOLO11. Aynı API, NMS'li çıktı.
- **Export:** Uçtan uca (NMS'siz) çıktı için `nms=False` açıkça verilir → `[1, 300, 6]`. Varsayılan (`nms=None`) ham çıktı verir (Spike A).
- **Çok kanal:** `data.yaml` içinde `channels: 4`; ilk katmanda kanal 0-2 COCO ağırlığı, kanal 3 rastgele (`ul_default`). `rgb_mean` başlatması `on_pretrain_routine_end` callback'iyle (Spike A).

## Değerlendirilen alternatifler
- **MMRotate / MMDetection:** OBB literatürünün standardı ama kurulumu ağır, çok kanal ve export zahmetli. HBB seçildi (ADR 0002).
- **RT-DETR / RF-DETR / D-FINE (Apache-2.0):** Lisans açısından üretime daha uygun. Ancak çok kanallı eğitim ve edge export yolu Ultralytics kadar hazır değil; 10 haftalık takvimde risk.
- **YOLOv11-RGBT (AGPL):** Orta füzyon için güçlü ama YOLO26'yı desteklemiyor; sadece E10 ek deneyi için.
- **YOLO27:** Ultralytics duyurdu, ancak 2026-10-05 itibarıyla ağırlıklar yayımlanmadı ve tarih yok. Proje içinde model değiştirilmez; gelecek iş.

## Sonuçlar (iyi / kötü / risk)
- İyi: Eğitim, değerlendirme, export ve takip tek API'de; çok kanal resmi olarak destekleniyor.
- Kötü: **AGPL-3.0.** Repo da AGPL. Kapalı savunma ürünlerinde Ultralytics, Enterprise lisans olmadan kullanılamaz. Üretimde bu karar lisans yüzünden değişirdi; izinli alternatifler RF-DETR, D-FINE, RT-DETR (Apache-2.0) ve MultimediaTechLab/YOLO (MIT).
- Risk: Ultralytics varsayılanları sürümden sürüme değişiyor (Spike A'daki export örneği). Sürüm `uv.lock` ile sabitlenir; spike'lar yeni sürümde yeniden çalıştırılır.
