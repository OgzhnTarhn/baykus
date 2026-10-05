# 0002 — Kutu tipi: yatay kutu (HBB); OBB ek deney

- Durum: Kabul edildi
- Tarih: 2026-10-05

## Bağlam
DroneVehicle etiketleri döndürülmüş kutu (OBB) olarak geliyor ve literatürün büyük kısmı OBB ile raporluyor. Ancak projenin hedefi savunma sanayiindeki video tespit ve takip işlerine benzer bir sistem.

## Karar
Ana hat **yatay kutu (HBB)**. OBB poligonları, 4 köşeyi kapsayan en küçük eksen hizalı dikdörtgene çevrilir ve 100 px kenarlık kırpımından sonra görüntü sınırına kırpılır (FR-D2, FR-D3). OBB, E11 ek deneyi olarak kalır (YOLO26-OBB, sadece IR).

## Değerlendirilen alternatifler
- **OBB ana hat:** Literatürle doğrudan kıyas sağlar. Ancak takipçiler (ByteTrack), değerlendirme araçları (pycocotools, TrackEval), TEKNOFEST formatı ve C++ motorundaki NMS/WBF hep HBB üzerine kurulu. OBB, her bileşende ek iş demek.

## Sonuçlar (iyi / kötü / risk)
- İyi: Standart araçlar değişmeden kullanılıyor; C++ tarafı basit.
- Kötü: Sonuçlar OBB tablolarıyla doğrudan kıyaslanamaz. README'deki her literatür satırına kutu tipi, etiket seti, kenarlık kırpma ve imgsz sütunları eklenir.
- Risk: Döndürülmüş araçlarda HBB arka planı fazla kapsar; yoğun otoparklarda kutular üst üste biner. E0 veri raporu ve kırılımlı değerlendirme bunun etkisini gösterir.
