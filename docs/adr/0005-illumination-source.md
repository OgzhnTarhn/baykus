# 0005 — Aydınlatma etiketinin kaynağı

- Durum: Önerildi (gerçek veride eşikler ve görsel kontrol bekleniyor)
- Tarih: 2026-10-06

## Bağlam
Ana iddialardan biri "füzyon gece daha iyi" (spec §12.2 kırılımları, FR-D4). DroneVehicle'ın resmi dağılımı üç aydınlatma seviyesi veriyor (spec §8.1, FR-D4): gündüz 14.478, gece 5.468, karanlık gece 8.493. Ancak Kaggle kopyasında bu etiketin görüntü başına hali olup olmadığı belli değil. Bunu Spike B'nin gerçek dosya listesi gösterecek (`docs/spikes/02-dronevehicle-format.md`, `other_files`).

## Karar
**Spike B resmi etiket dosyası bulursa:** O dosya kullanılır. Bu ADR "Kabul edildi, kaynak: resmi" olarak güncellenir; parlaklık yöntemi sadece çapraz kontrol olarak kalır.

**Bulmazsa (önerilen varsayılan):** Etiket RGB parlaklığından türetilir (`baykus.data.illumination`, `baykus-data illum`).
- Ölçü: RGB görüntünün HSV `V` kanalı ortalaması (kırpılmış 640×512 görüntüde).
- Eşikler: Bütün görüntülerin parlaklık dağılımı, resmi oranlara (14.478 : 5.468 : 8.493) uyacak şekilde iki eşikle bölünür. Eşikler `meta/illumination_thresholds.json`'a yazılır.
- Doğrulama: Her seviyeden 16 örneklik mozaik (`docs/figures/data/illum_samples_*.jpg`). Kullanıcı gözle kontrol eder.
- Raporlarda "türetilmiş etiket, resmi değil" diye açıkça yazılır.

**Kesinleşme ölçütü:** Mozaiklerde belirgin karışıklık yoksa (ör. gündüz kareleri "karanlık gece" mozaiğinde değilse) kabul edilir. Varsa yöntem yetersiz sayılır ve kullanıcıyla alternatif konuşulur (aşağıda).

## Değerlendirilen alternatifler
- **Sabit mutlak eşikler (ör. V < 40 karanlık):** Kalibrasyonsuz. Kamera pozlaması değişince kayar; resmi dağılımla uyuşması garanti değil.
- **IR / RGB parlaklık oranı:** Gece RGB karanlık, IR değil; oran ayırt edici olabilir. Ama termal görüntünün parlaklığı sıcaklığa ve sahneye bağlı; ek gürültü getiriyor. Parlaklık yetersiz çıkarsa ilk denenecek seçenek bu.
- **Küçük bir sınıflandırıcı eğitmek:** Görüntü başına etiket olmadan eğitilemez; elle etiket gerektirir. Kapsam dışı.
- **Kırılımı hiç yapmamak:** "Gece füzyon daha iyi" iddiası ölçülemez olur. Reddedildi.

## Sonuçlar (iyi / kötü / risk)
- İyi: Ek etiketleme yok, tekrarlanabilir. Seviye oranları resmi dağılımla birebir tutarlı (yapı gereği).
- Kötü: Oranların tutması doğruluk kanıtı değil. Doğruluğun tek kanıtı görsel kontrol.
- Risk: Alacakaranlık kareleri gece ve gündüz arasında kayabilir; "gece" kırılımı gürültülü olur. Eşikler test dahil bütün bölümlerin **görüntü istatistiğinden** hesaplanıyor. Bu, etiket sızıntısı değil, ama test setine dokunan tek ön işleme adımı; raporda belirtilir.
