# 0008 — Takip değerlendirmesi (M3OT)

- Durum: Kabul edildi (2026-10-19; protokol ölçümden önce sabitti, sonuç bir kez ölçüldü)
- Tarih: 2026-10-19

## Bağlam
Faz 3'te motor ByteTrack'i C++'ta çalıştırıyor (`trackers==2.6.1` uyarlaması, Python referansıyla eşlik
Kabul 4). Eşlik "C++ Python'la aynı mı" sorusunu cevaplar; "takip ne kadar iyi" sorusunu cevaplamaz.
Spec §16 takibin HOTA, MOTA ve IDF1 ile ölçülmesini istiyor; tasarım §4.3 ve §6.3 protokolü ölçümden önce
sabitledi. M3OT test bölümü (`1_03`, `2_03`, 2 × 480 kare) yalnız bu son ölçümde, sistem dondurulduktan
sonra (`f3-freeze`) bir kez kullanıldı.

## Karar
1. **Referans:** `trackers==2.6.1` ByteTrack; motor C++ uyarlamasını (`baykus_track`) kullanır.
   Parametreler `configs/engine/tracker.json` = paketin varsayılanları (ayar yapılmadı): kayıp iz tamponu 30,
   30 kare/sn, etkinleştirme 0,7, ardışık kare 2, IoU 0,1, yüksek güven 0,6.
2. **Girdi:** takip katmanı ham skor ≥ 0,1 (tasarım §4.1). GT: IR etiketleri, AP etiketleriyle aynı süzgeç
   (flag 1, sınıf 1, görünürlük > 0,25), sınıfsız `vehicle`.
3. **Ölçüm:** `trackers eval` (TrackEval eşdeğeri; aynı sabit paket) ile HOTA, MOTA, IDF1; dizi başına ve
   iki dizinin birleşiği; bootstrap yok (iki dizi).
4. **İki satır:** (a) **oracle**: M3OT'un kendi kutuları skor 1 ile (`baykus-track oracle`), takipçiyi
   modelden ayırır; (b) **motor**: `baykus_detect` tespitleri (ham ≥ 0,001 dosyası, takipte ≥ 0,1).
5. **Komutlar:** `scripts/m3ot_measure.sh` — önce geliştirme bölümünde (`dev`, M3OT val) prova, sonra
   aynı betik `test --i-know-this-is-test` ile bir kez.

## Sonuç (ölçülen)

| Bölüm | Satır | HOTA | MOTA | IDF1 | Kaynak |
|---|---|---|---|---|---|
| val (prova) | oracle | 0,974 | 0,993 | 0,969 | `results/track/m3ot_dev_oracle.json` |
| val (prova) | motor | 0,220 | 0,152 | 0,296 | `results/track/m3ot_dev_engine.json` |
| **test** | oracle | **0,949** | **0,981** | **0,941** | `results/track/m3ot_test_oracle.json` |
| **test** | motor | **0,489** | **0,547** | **0,677** | `results/track/m3ot_test_engine.json` |

Test dizi başına (motor): `1_03` HOTA 0,535 · MOTA 0,591 · IDF1 0,744; `2_03` HOTA 0,432 · MOTA 0,497 ·
IDF1 0,599. Aynı tespitlerin AP50'si test 0,860, val 0,396 (`results/engine/m3ot_{test,dev}_detect.json`).

**Yorum:** Doğru kutular verildiğinde takipçi neredeyse kusursuz (oracle HOTA test 0,95); motor satırındaki
düşüş tespitten geliyor. Val dizileri testten çok daha zor: kare başına IR kutusu val'de ≈ 11,2
(13.474 / 1200), testte ≈ 2,6 (2465 / 960); AP50 0,40'a karşı 0,86. Mutlak değer değil iki satırın farkı
yorumlanır (tasarım §8 risk satırı).

## Değerlendirilen alternatifler
- **Yalnız motor satırı:** takipçi hatasıyla dedektör hatası ayrılamazdı; reddedildi.
- **Parametreleri val'de ayarlamak:** Faz 3 kapsamı dışı (paket varsayılanları, eşlik onlarla kanıtlı);
  ayar yapılsaydı val'de yapılırdı, testte asla.
- **Makaledeki ByteTrack sayılarıyla yan yana koymak:** farklı dedektör ve algoritma; konmadı.

## Sonuçlar (iyi / kötü / risk)
- İyi: takipçi ile dedektörün katkısı ayrı ölçüldü; protokol ve komutlar ölçümden önce sabit ve val'de
  sınanmıştı.
- Kötü: yalnız iki test dizisi ve ardışık video kareleri; güven aralığı yok, sayılar genelleme için zayıf.
- Risk: val ile test arasındaki büyük fark, sonucun sahne yoğunluğuna çok bağlı olduğunu gösteriyor;
  yoğun trafikte motor satırı val'deki gibi düşük olabilir.
