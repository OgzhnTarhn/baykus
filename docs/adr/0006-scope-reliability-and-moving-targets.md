# 0006 — Kapsam genişletme: güvenilirlik farkındalığı ve hareketli hedef

- Durum: Kabul edildi (yol haritası düzeyinde; Faz 2 ve Faz 4'ün ayrıntıları kendi tasarımlarında)
- Tarih: 2026-10-07

## Bağlam
Spec v3.0'ın iddiası "dürüst ölçüm + uçtan uca sistem" (§3.2). Alan çok çalışılmış: onlarca DroneVehicle makalesi, hazır RGB-T çerçeveleri, Jetson'da çalışan füzyon modelleri var. Kullanıcının kaygısı, projenin mülakatta "herkes bunu yapmış" diye görülmesi. İstediği fark iki türlü: **kendi önerdiği bir yöntem** ve **operatörün gerçekten kullanacağı bir sistem katmanı (operasyonel zekâ)**. Süre 13 Aralık 2026'dan sonraya uzayabilir; hedef mezuniyet (Haziran 2027).

Mevcut plan bu iki yönün temelini zaten içeriyor: modality dropout (E3c), aydınlatmaya göre geç füzyon (E4b), sağlamlık paketi (E6), takip (ByteTrack) ve ek hedeflerde ilk sırada hareketli/duran araç ayrımı.

## Karar
Proje beş faza bölünür. Faz 1, v3.0'ın H1-H5'i, değişmeden kalır.

| Faz | Dönem | İçerik |
|---|---|---|
| 1. Çekirdek | 5 Eki - 8 Kas 2026 | v3.0 H1-H5: veri, E1-E4b, sağlamlık, M3OT, edge, MVP (`v0.1.0`) |
| 2. Güvenilirlik farkındalığı (A) | 9 Kas - 6 Ara 2026 | Kare başına RGB/IR güvenilirlik tahmini, buna göre uyarlanan füzyon, güven kalibrasyonu, sensör arıza tespiti |
| 3. Sistem | 7 Ara 2026 - 31 Oca 2027 | v3.0 H6-H9 (C++ motor, takip, UI, Docker) + sensör sağlık paneli ve güven tabanlı alarmlar |
| 4. Hareketli hedef (B) | 1 Şub - 28 Mar 2027 | Kamera hareketi telafisi, hareketli/duran ayrımı, göreli mozaik harita, bölge olayları (M3OT videoları) |
| 5. Sunum | 1-30 Nis 2027 | README, demo videosu, teknik yazı, isteğe bağlı arXiv |

Planlama üç katmanda yapılır: yol haritası (bu ADR ve spec v4.0), faz tasarımı (her faz başlamadan önce, o ana kadarki sonuçlarla) ve haftalık plan (`docs/plans/`).

Faz 1, sonraki fazların ihtiyaç duyacağı birkaç küçük **kanca** ile uygulanır (spec §6.1). Bunlar Faz 1'in kapsamını değiştirmez, sonradan yeniden iş yapmayı önler.

## Değerlendirilen alternatifler
- **Sadece A (güvenilirlik):** H1 işini en çok kullanan, GPU'su makul yön. Tek başına operasyonel tarafı zayıf kalıyordu.
- **Sadece B (hareketli hedef):** TEKNOFEST görevine birebir uyuyor. DroneVehicle tek karelerden oluştuğu için ağırlığı M3OT'a ve takibe kayıyor; "kendi yöntem" kısmı zayıf.
- **C: Hizasızlığa dayanıklı füzyon (saf araştırma):** Literatür çok yoğun (TSFADet, C²Former, CAFNet). Ultralytics dışı mimari gerektiriyor, export ve C++ zorlaşıyor; GPU maliyeti ve "sonuç çıkmama" riski en yüksek. Reddedildi.
- **Gerçek donanım / saha verisi, simülasyonla sentetik veri:** Kullanıcı tercih etmedi; maliyet ve kurulum yükü yüksek.
- **Kapsamı değiştirmemek:** Kullanıcının asıl kaygısını (ayırt edicilik) karşılamıyor.

## Sonuçlar (iyi / kötü / risk)
- İyi: Mülakat hikâyesi güçleniyor: "Sistemim ne zaman yanıldığını biliyor, bunu operatöre söylüyor; araçların hareket edip etmediğini de ayırıyor." Faz 1 ve H1 işi tamamen korunuyor. Her faz kendi başına gösterilebilir bir sürümle bitiyor.
- İyi: Faz 4, TEKNOFEST 2027 başvuru dönemiyle (Şubat) çakışıyor ve yarışmanın hareketli/duran taşıt görevine doğrudan hazırlık oluyor.
- Kötü: Süre ~10 haftadan ~7 aya çıkıyor. Proje bitişi 30 Nisan 2027.
- Risk: Faz 2'nin yöntemi sabit füzyona (E3c/E4b) göre anlamlı kazanç sağlamayabilir. Bu da bir sonuç olarak raporlanır; yöntem "yenilik" diye değil, ölçülmüş bir deneme olarak sunulur. Benzer bir fikir literatürde var (LER-YOLO, 2026, güvenilirlik haritası); Faz 2 tasarımında farkı açıkça yazılır.
- Risk: Gerçek coğrafi konum (piksel → GPS) bu veri setleriyle yapılamaz; kamera irtifa ve poz bilgisi yok. Harita göreli mozaik olarak tasarlanır.
- Risk: Uzun süre ve dönem içi ders yükü (özellikle Ocak final dönemi, varsayım). Önlem: Faz 3'e 8 hafta ayrıldı; bir faz bitmeden sonrakine geçilmez; Faz 4 tamamen kesilse bile Faz 1-3 eksiksiz bir proje.
- ADR numaraları: v3.0'da 0006 (gecikme hedefi) ve 0007 (takip değerlendirmesi) için ayrılmış numaralar 0007 ve 0008 oldu.
