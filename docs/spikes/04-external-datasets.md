# Spike C (2) — Harici veri setleri

- **Tarih:** 2026-10-06
- **Yöntem:** Makale, resmi GitHub depoları ve dönüştürme scriptleri okundu (web). M3OT'un kendisi henüz indirilmedi; figshare otomatik indirmeyi engelliyor (HTTP 202 WAF), indirme tarayıcıdan yapılacak.
- Kesin olmayan maddeler **(doğrulanmadı)** diye işaretli.

## M3OT (çekirdek: veri setleri arası test + takip ölçümü)

| Konu | Bulgu | Kaynak |
|---|---|---|
| İçerik | 21.580 kare, **10.790 RGB-IR çifti**, ~220 bin kutu; banliyö, şehir, gündüz, alacakaranlık, gece | Makale, README |
| Bölümler | 17.260 train · 2.400 val · 1.920 test **görüntü** (iki modalite birlikte sayılmış: toplam 21.580) | Makale |
| Platform | İki DJI Mavic 3T; 100-120 m irtifa | Makale |
| Çözünürlük | IR 640×512. RGB (4000×3000 tele) afin dönüşüm + kırpma ile IR'ye hizalanıp **640×512** yayımlanmış | Makale |
| Sınıf | **Tek sınıf `vehicle`** (COCO `id=1`) | Makale, `tools/m3OT2coco.py` |
| Etiket | Her modalite için ayrı. MOTChallenge `gt.txt`: `frame, track_id, x, y, w, h, conf/ignore, class_id, visibility`. COCO JSON'lar `Annotations/` klasöründe | Makale, `tools/` |
| Düzen | `<split>/<dizi>/img1/*.jpg` + `<split>/<dizi>/gt/gt.txt` (MOT düzeni) | `tools/convert_M3OT_to_yolo.py` |
| Kare hızı | Resmi araçta `fps=30` sabit yazılmış. Makalede kare hızı ya da örnekleme aralığı geçmiyor. **(doğrulanmadı)** | `tools/m3OT2coco.py` |
| Kutu boyutu | Hepsi < 1024 px², tepe 100-400 px² (DroneVehicle'dan küçük) | Makale |
| Lisans | **CC BY 4.0** | Makale |
| Takip taban çizgisi | ByteTrack: HOTA 36,9 · IDF1 56,9 · MOTA 37,7 | Makale |
| İndirme | https://figshare.com/s/01fa8d1163f4e9a5a13a (tarayıcıdan) | Makale |

**Spec'e düzeltme:** Spec §8.1 ve §16 "8 dizi" diyor. Kaynakta geçen ifade **"8 saatlik videodan çıkarılmış 21.580 kare"**; dizi sayısı belli değil. Dizi sayısı ve uzunlukları indirmeden sonra sayılacak.

**Proje için sonuçlar:**
- Tek sınıf olduğu için spec §8.3'teki plan geçerli: 5 sınıf → sınıfsız `vehicle` AP. Ek ADR gerekmiyor.
- Kutular çok küçük. 640 girdiyle M3OT AP'si DroneVehicle AP'sinden belirgin düşük çıkabilir. E7'de mutlak değer değil, **modlar arası sıralama** yorumlanmalı.
- M3OT RGB'si tele kameradan küçültülmüş. DroneVehicle RGB'sinden keskinlik ve renk farkı (alan kayması) beklenir; E7'nin amacı da bu.

**İndirmeden sonra kontrol edilecekler ([KULLANICI] tarayıcıdan indirip `C:\baykus\data\raw\m3ot\` altına açar):**
- [ ] Dizi sayısı, dizi başına kare sayısı; RGB ve IR dizilerinin adlandırması (aynı dizi adı + modalite eki mi?)
- [ ] `gt.txt` 7. sütun (conf / ignore) ve 9. sütun (visibility) değerleri; `m3OT2coco.py`'deki `USELESS` / `IGNORES` sınıf kodları gerçekte geçiyor mu?
- [ ] Kare hızı (dosya adlarından ya da `seqinfo.ini`'den)
- [ ] RGB ve IR kare sayıları eşit mi, aynı kare numaraları hizalı mı?
- [ ] İki modalite arasındaki etiket sayısı farkı

## HIT-UAV (ek hedef)

| Konu | Bulgu |
|---|---|
| İçerik | 43.470 kareden seçilmiş **2.898 IR görüntü**; okul, otopark, yol, oyun alanı; gündüz + gece; 60-130 m irtifa, 30-90° bakış açısı |
| Sınıflar | Person, Car, Bicycle, OtherVehicle (+ DontCare) |
| Etiket | `normal_xml` / `normal_json` (HBB, VOC + COCO), `rotate_xml` / `rotate_json` (OBB) |
| İndirme | GitHub `suojiashun/HIT-UAV-Infrared-Thermal-Dataset`, Kaggle `pandrii000/hituav-a-highaltitude-infrared-thermal-dataset`, Zenodo |
| Lisans | GitHub README: **CC BY 4.0**. Zenodo kaydı arama sonucunda CC0 görünüyor **(doğrulanmadı)**. İkisi de projeye uygun; atıfta CC BY 4.0 esas alınır. |

Eşleme önerisi (yapılırsa ADR'ye girer): `Car` → car; `OtherVehicle` → sınıfsız `vehicle` değerlendirmesine dahil, 5 sınıfa dağıtılmaz; `Person` ve `Bicycle` dışarıda. Yalnızca IR modeli test edilebilir (RGB yok).

## VT-MOT (yedek)

- 582 RGB-T dizi çifti, 401 bin kare çifti; gözetleme, drone ve el kamerası.
- İndirme: Baidu Pan (çıkarma kodu PFTrack README'sinde). Çin dışından erişim ve hesap gerektirebilir.
- Lisans: README'de ve makalede açık bir lisans bulunamadı. Bu yüzden portfolyoda yeniden dağıtım ya da fikstür olarak kullanılmamalı.
- Sonuç: Yedek olarak kalıyor. M3OT erişimi sorunsuz olursa kullanılmayacak.

## TEKNOFEST Havacılıkta Yapay Zeka (spec §25 için)

- 2026 takvimi: şartname 16.02, başvuru 28.02, ön tasarım raporu 22.04, simülasyon 16.07, final 18-20.09.2026. **2027 şartnamesi Şubat 2027'de çıkacak.**
- Görev 1: taşıt, insan, UAP, UAİ tespiti; taşıtta **hareketli / duran** ayrımı; UAP ve UAİ'de iniş uygunluğu. Görev 2: GPS'siz pozisyon kestirimi. Görev 3: referans görüntüyle nesne bulma.
- Kısıtlar: ~7,5 FPS akış, internetsiz çalışma, sonuçlar sunucuya JSON olarak gönderiliyor.
- Baykuş karşılığı: taşıt tespiti, internetsiz çalışma ve JSON API ortak. Hareketli/duran ayrımı ek hedeflerde ilk sırada.
- Ayrıntı: `docs/research/2026-10-04-derin-arastirma.md` §5.4.

## Kaynaklar
- M3OT makalesi (Scientific Data 2025): https://www.nature.com/articles/s41597-025-06204-0 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12686387/
- M3OT GitHub (`tools/convert_M3OT_to_yolo.py`, `tools/m3OT2coco.py`): https://github.com/M3OT/M3OT
- HIT-UAV: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · https://arxiv.org/abs/2204.03245
- VT-MOT / PFTrack: https://github.com/wqw123wqw/PFTrack · https://arxiv.org/abs/2408.00969
- TEKNOFEST: https://teknofest.org/tr/yarismalar/havacilikta-yapay-zeka-yarismasi/
