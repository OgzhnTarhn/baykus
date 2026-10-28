# Faz 4 ek tasarım — M3OT'ta araç kapsamı (takipçi eşikleri, M3OT ile eğitim, parçalı çıkarım)

**Durum:** onaylı (kullanıcı, 2026-10-23). Onaydan sonra F4 planına Görev 15A-15E olarak
girer (Görev 16'dan önce; motor satırı yeni kapsamla ölçülür).

## 1. Neden

Görev 15'in klip gözden geçirmesinde kullanıcı gördü: karede 30'dan fazla araç görünüyor, motor 0-5 iz
veriyor. Tanı (2026-10-23, laptop, yalnız M3OT val `1_08`, `2_08`, 1200 kare; geçici betikler, sayılar
Görev 15A'da `results/`'a resmî olarak yazılır):

| Ölçüm | Değer |
|---|---|
| Etiketli araç / kare (IR, flag 1, sınıf 1, vis > 0,25) | 11,2; ortanca kutu 13 px |
| E3c bulma oranı (IoU ≥ 0,5), ham ≥ 0,05 / 0,1 / 0,3 / 0,7 | %40 / %35 / %23 / %6 |
| E3c AP50 temiz / RGB kapalı / RGB yerine IR | 0,40 / 0,39 / 0,30 → **hizasızlık neden değil** |
| E3c 4 parçalı (2 × 2, ≈ 1,8× büyütme) AP50 / bulma ham ≥ 0,1 / ≥ 0,3 | 0,50 / %50 / %41 |
| Takip HOTA / MOTA / iz-kare, eşikler 0,7-0,6 (bugün) | 0,22 / 0,15 / 2,5 |
| aynı, etkinleşme 0,3, yüksek 0,3 | 0,29 / 0,21 / 5,0 |
| Takip, GT kutularıyla (oracle, `results/track/m3ot_dev_oracle.json`) | HOTA 0,97 |

**Sonuç:** asıl darboğaz model (DroneVehicle'da eğitildi; M3OT val'deki 13 px'lik araçların çoğunu
görmüyor), ikincisi takipçinin yüksek etkinleşme eşiği (bulunan araçların yarısı iz açmıyor). Takipçinin
kendisi sağlam (oracle 0,97).

## 2. Hedefler, hedef dışı, anti-hedefler

**Amaç (kullanıcı):** videodaki araçlar bulunsun ve takip edilsin; ürünün bütün katmanları (hareket,
bölgeler, alarmlar, özet) buna dayanıyor.

**Hedefler** (üç yolun üçü de, kullanıcı kararı 2026-10-23):
1. **A — Takipçi eşikleri** val'de ölçülen bir kuralla yeniden seçilir (önce E3c ile, yeni modelle tekrar).
2. **B — Modeli M3OT eğitim dizileriyle de eğitmek** (Colab'da kullanıcı çalıştırır).
3. **C — Parçalı çıkarım** motorda ve Python referansında seçilebilir bir mod olarak.

**Hedef dışı:** yeni mimari (orta füzyon vb.), takipçi algoritmasını değiştirmek (ByteTrack, `trackers==2.6.1`
aynen; yalnız parametreler), RGB-IR hizalama (ölçüldü: neden değil).

**Anti-hedefler (teknik olarak çalışsa da başarısızlık sayılır):**
- M3OT **test** dizileri herhangi bir seçimde kullanılırsa (yalnız Görev 23'te bir kez).
- DroneVehicle'daki başarı sessizce düşerse (aşağıdaki kural).
- Varsayılan yol (laptop CPU) gecikme hedefini (ADR 0007, p95 ≤ 100 ms) sessizce aşarsa.
- Eşlik (Python ↔ C++) bozulursa ya da fikstürler eski modelle kalırsa.

## 3. Kararlar (ölçümden önce sabit)

### A — Takipçi eşikleri (Görev 15A, E3c ile; Görev 15C, yeni modelle tekrar)
- Araç: `baykus-track sweep` (val; test bayrağı yok, test dizisi reddedilir) → `results/track/m3ot_dev_sweep_<model>.json`.
- Izgara: `track_activation_threshold` ∈ {0,7; 0,6; 0,5; 0,4; 0,3; 0,2} × `high_conf_det_threshold` ∈
  {0,6; 0,5; 0,4; 0,3; 0,2}; diğer parametreler aynen (`lost_track_buffer` 30, `minimum_consecutive_frames` 2,
  `minimum_iou_threshold` 0,1). Eşikler ham skora (Faz 3 gibi).
- **Kural:** iki dizinin birleşik HOTA'sı en yüksek çift; HOTA farkı ≤ 0,005 olanlar arasında etkinleşme
  eşiği yüksek olan (daha az sahte iz), o da eşitse yüksek eşiği yüksek olan.
- Seçilen değerler `configs/engine/tracker.json`'a; Faz 3 tasarımındaki "referansın varsayılanları"
  satırı gerekçesiyle güncellenir (eski değer silinmez).

### B — M3OT ile eğitim (Görev 15B, Colab)
- **Model adı:** E3c-M. E3c'nin ağırlıklarından ince ayar; tarif E3c'ninki (erken füzyon, kanal kaybı
  eğitimi, aynı hiperparametreler, aynı görüntü boyu 640).
- **Veri:** DroneVehicle train (aynen) + M3OT train (14 dizi, 9.590 çift; IR etiketleri, flag 1, sınıf 1,
  vis > 0,25). M3OT'ta araç türü yok (13.474 val kutusunun hepsi tek sınıf) → **öğretmen etiketi**
  (kullanıcı kararı 2026-10-23; "hepsi otomobil" ve "yeni sınıf" seçenekleri reddedildi):
  - Her M3OT train izinden (dizi + iz kimliği) eşit aralıklı en çok **10** kare. Kutunun merkezinde
    **160 × 128** pencere (görüntü sınırına kaydırılarak), RGB ve IR aynı koordinatla kırpılır, bugünkü ön
    işlemeyle 640'a (≈ 4× büyütme), E3c (ONNX FP32, Python referansı) ham ≥ 0,1.
  - O karenin oyu: kutuyla (pencere koordinatında) IoU ≥ **0,3** olanların en yüksek IoU'lusu, ham skoru
    ≥ **0,5** ise sınıfı; değilse oy yok.
  - İzin sınıfı: oyların çoğunluğu (eşitlikte skor toplamı büyük olan, o da eşitse küçük `cls_id`);
    hiç oy yoksa `car`.
  - **İnsan denetimi (kör, ≈ 15 dk):** 100 iz; tahmini `car` olmayanlardan en çok 60 (sınıf başına
    orantılı), kalanı `car`. Kullanıcı tahmini görmeden IR ve RGB kırpıntısına bakıp türü seçer ya da
    "bilmiyorum" der. Doğruluk = cevaplananlarda tahminle aynı oranı (Wilson %95 GA, sınıf başına da).
  - **Kural:** doğruluk ≥ **0,80** → öğretmen etiketleri; altında → bütün M3OT kutuları `car` (ve ADR'de
    sınırlama). Ölçümden önce sabit.
- **Doğrulama:** seçimler DroneVehicle val + M3OT val ile; M3OT test kullanılmaz.
- **Sevk kuralı:** (i) DroneVehicle val mAP50 (ONNX FP32, motor yolu) E3c'nin 0,841'inden en çok **1,0
  puan** düşük (ADR 0007'deki Δ ≤ 1,0 kuralı) **ve** sınıf başına AP50'de hiçbir sınıf 3,0 puandan fazla
  düşmez; (ii) M3OT val AP50 (sınıfsız, motor yolu) E3c'nin değerinden yüksek. İkisi tutarsa E3c-M sevk
  edilir, tutmazsa E3c kalır ve sonuç raporlanır (A ve C yine uygulanır).
- Kalibrasyon (`calibration.json`) ve sağlık / kapı dosyaları Faz 2-3'teki aynı yordamla yeniden üretilir.
  Gecikme mimari aynı olduğu için değişmez; bench yine koşulur (Görev 17).
- Sonuç ADR 0010 (model değişimi) + `docs/experiments.md` E11.

### C — Parçalı çıkarım (Görev 15D)
- Mod `tiles`: `1` (bugünkü) ya da `4` (2 × 2 parça, 360 × 288 pencere, örtüşme yatay 80 / düşey 64 px, her
  parça bugünkü ön işlemeyle 640'a; kutular kareye geri taşınır, birleşik liste sınıfsız NMS IoU 0,5,
  skor sırası). Python referansı ve C++ motoru aynı tanımla; eşlik fikstürleri her iki mod için.
- API: `POST /v1/detect?tiles=1|4` (varsayılan motor yapılandırmasından); `/v1/model` geçerli varsayılanı
  ve parça tanımını döner. Ekranda ayarlar menüsünde "küçük hedef modu" (Görev 18'e eklenir).
- **Varsayılan kuralı:** bench'te (Görev 17) cihazda `tiles=4` p95 ≤ 100 ms ise varsayılan 4, değilse 1
  (laptop CPU'da beklenen 1; 4060'ta ölçülür). Ölçümden önce sabit.
- Ölçüm: son model için val AP50 / bulma oranı / HOTA, `tiles` 1 ve 4 → `results/engine/m3ot_dev_tiles.json`.
- **Sonuç ve karar (2026-10-28):** E3c-M ile val'de `tiles=4` her ölçüde daha kötü: AP50 0,885 → 0,833, bulma
  0,857 → 0,784 (`results/engine/m3ot_dev_tiles.json`), HOTA 0,652 → 0,597, MOTA 0,760 → 0,656
  (`results/track/m3ot_dev_engine.json`, `m3ot_dev_tiles4_engine.json`). Yukarıdaki kural yalnız gecikmeye bakıyordu
  ve parçalı modun kapsamı artıracağını varsayıyordu; ölçüm bunu E3c-M için çürüttü. Kullanıcı kararı: varsayılan
  **her cihazda 1**; `tiles=4` API'de isteğe bağlı kalır, Görev 17'deki `tiles=4` gecikme satırı yalnız bilgi.

## 4. Sıra ve etkiler

1. **15A** ölçüm araçları (`sweep`, parçalı / koşullu AP) + tanının resmî kaydı + A (E3c) → takip
   fikstürleri, klip beklentileri (Kabul 1 referansı, `zones_clip.json`) yeniden.
2. **15B** öğretmen etiketi aracı + kör denetim sayfası; Colab defteri (`notebooks/colab/10_m3ot_train.ipynb`):
   M3OT indirme (figshare API, çerezsiz) + train dönüştürme + öğretmen etiketleri + denetim örneği
   **[KULLANICI - COLAB]**, laptopta kör denetim **[KULLANICI - LAPTOP, ≈ 15 dk]**, sonra aynı defterde
   ince ayar + ONNX + DroneVehicle val + M3OT val ölçümü **[KULLANICI - COLAB]**.
3. **15C** sevk kuralı; sevk edilirse model dosyaları, manifest, kalibrasyon, sağlık / kapı, **bütün
   eşlik fikstürleri** (Faz 2-3 tespit, sağlık, takip, F4 klip ve Release `fixtures-f4` yeni sürüm) yeniden;
   A yeni modelle tekrar.
4. **15D** parçalı mod (Python + C++ + API + eşlik).
5. **15E** spec v4.6 notu, ADR 0010, `docs/experiments.md`; sonra Görev 16 (motor satırı yeni kapsamla).

**Süre:** 15A 1 gün; 15B araç + defter 2 gün + Colab (indirme ≈ 7 dk, dönüştürme ≈ 10 dk, eğitim birkaç saat;
defterde hücre başına süre); 15C 2 gün; 15D 3-4 gün; 15E 0,5 gün → ≈ 1,5-2 hafta.

## 5. Riskler

- M3OT train ve val aynı platform / bölgeden: val sonucu iyimser olabilir; test Görev 23'te ayırt eder.
- Öğretmen etiketleri bugünkü modelin yanlılığını taşır (M3OT'ta zayıf olduğu için); kör denetim bunu
  ölçer, sınıf başına AP kuralı DroneVehicle'daki bedeli yakalar. 13 px'lik araçta türü insan da her
  zaman ayırt edemeyebilir ("bilmiyorum" cevapları raporlanır).
- Model değişimi bütün fikstürleri değiştirir: Görev 9-15'te sabitlenen beklentiler (klip olayları dahil)
  yeniden üretilir ve gözden geçirilir.

## 6. Builder'a bırakılanlar

- Colab'da epoch sayısı ve erken durdurma (E3c tarifinin süresine göre; defterde süre tahmini).
- `sweep` çıktı biçimi; parçalı NMS'in uygulama ayrıntısı (tanım yukarıda sabit).
- Fikstür yeniden üretiminin sırası ve Release sürüm etiketi.
