# 0005 — Aydınlatma etiketinin kaynağı

- Durum: Kabul edildi (parlaklık; gündüz sınırı elle etiketli örnekle düzeltildi)
- Tarih: 2026-10-06 · Güncelleme: 2026-10-11

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

## Gerçek veride sonuç (2026-10-11)

**Kaynak:** Spike B resmi aydınlatma etiketi bulmadı (`docs/spikes/02-dronevehicle-format.md`). Varsayılan yöntem (RGB HSV `V` ortalaması, iki eşik resmi orana uydurulmuş) uygulandı: eşikler 48,66 / 82,65; sayılar 8.482 / 5.461 / 14.460.

**Görsel kontrol başarısız:** "Karanlık gece" ve "gece" mozaikleri tutarlıydı, ama "gündüz" mozaiğinin 16 karesinin ~6'sı sokak lambalarıyla aydınlatılmış gece sahnesiydi. Ortalama parlaklık doğru ölçü, ama gündüz eşiği resmi orana uyması için fazla düşük kalıyordu. Kesinleşme ölçütü karşılanmadı; iki alternatif denendi.

**Alternatif C — çekim oturumu (başarısız):** Ham XML'deki `<folder>` alanı (`scripts/spikes/session_map.py`). 93 değer çıktı; tarih klasörleri (`5-13`, `6-1 (2)`…) aynı günün hem gündüz hem gece uçuşlarını içeriyor, genel klasörler (`B` 8.685, `image` 2.530, `modalA` 2.182 görüntü) karışık. Görüntülerin yalnızca %23'ü tek tip bir klasörde. Oturum etiketi olarak kullanılamaz.

**Alternatif D — elle etiketli örnekle eşik (kabul):** `scripts/spikes/illum_features.py` her görüntü için parlaklık özellikleri hesapladı (ortalama, yüzdelikler, karanlık piksel oranları) ve gündüz-gece geçiş bölgesinden (ortalama ≥ 50, karanlık piksel oranına göre 8 eşit dilimden 20'şer) 160 örnek seçti. Örnekler Claude tarafından etiketlendi, kullanıcı tamamını gözden geçirdi; kararsız 3 örneği (42, 93, 117) kullanıcı etiketledi. Etiketler: `docs/data/illum_check_labels.csv` (85 gündüz, 75 gece), sayfalar: `docs/figures/data/illum_check_0{0..3}.jpg`.

| Kural (160 örnek) | Doğruluk | Gece → "gündüz" | Gündüz → "gece" |
|---|---|---|---|
| Eski: ortalama ≥ 82,65 | %77,5 | 36 | 0 |
| **Yeni: ortalama ≥ 114,14** | **%95,0** | 7 | 1 |
| Ortalama + karanlık piksel oranı (iki eşik) | %96,8 (157 örnekte) | | |

Birini-dışarıda-bırak doğruluğu (tek eşik): %92,5. İki özellikli kural küçük örnekte yalnızca 2 örnek fark ettiği için ezber riski taşıyor; tek eşik seçildi. Kalan hatalar çok parlak, renkli ışıklarla aydınlatılmış gece kareleri (8, 23, 42, 44, 65, 91, 102). Örnekler bilerek zor bölgeden seçildi; tüm veri setindeki doğruluk ölçülmedi, daha yüksek olması beklenir ama iddia edilmez.

## Güncel karar
- `t_dark`: resmi karanlık gece oranına uydurulur (değişmedi; mozaikle doğrulandı) → 48,66.
- `t_day`: **sabit 114,14** (`baykus.data.illumination.DAY_MIN_BRIGHTNESS`).
- Sonuç: gündüz **11.050**, gece **8.871**, karanlık gece **8.482** (resmi: 14.478 / 5.468 / 8.493). Gündüz/gece sayıları resmi oranla artık tutmuyor; raporlarda belirtilir.
- Uygulama: `baykus-data illum` (görüntüden) ve `baykus-data illum-relabel` (mevcut `illumination.csv`'deki parlaklıktan, görüntüsüz) aynı kuralı kullanır; `baykus-data slices` kırılım listelerini COCO JSON'daki dosya adlarından yeniden üretir. Kaggle veri setindeki `meta/illumination.csv` eski etiketleri taşır ama parlaklık değerleri aynıdır; eğitim notebook'u indirmeden sonra `illum-relabel` + `slices` çalıştırır (H2 Görev 9).

## Sınırlama (raporlara girer)
"Gündüz/Gece/Karanlık gece" kırılımı **türetilmiş** bir etikettir: RGB parlaklığı + 160 örnekle seçilmiş eşik. Geçiş bölgesindeki örnekte ~%5 hata; çok parlak aydınlatılmış geceler "gündüz" sayılabilir.
