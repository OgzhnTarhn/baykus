# Faz 4 Tasarımı — Hareketli hedef ve operasyonel ürün

> **Tarih:** 2026-10-20 (taslak v1) · 2026-10-20 (v2: Codex incelemesi doğrulanıp işlendi, §12) ·
> 2026-10-20 (v3: kullanıcının ürün paketleri — harita + araç kartı, bölge kuralları, olay kaydı ve dışa
> aktarma, olaylar motorda; harita belleği) ·
> **Durum:** v3, onaylı (2026-10-20; ürün paketleri A-D, kesme yok; §10/1 = A; plan `docs/plans/F4.md`) · **Dal:** `f4`
> **Dayanak:** spec §2.3 S11, §3.2 madde 5, §6 Faz 4, §16, §17, §20.6 kesme sırası, §22 Faz 4 satırı, §23 son
> risk satırı; ADR 0006, 0007, 0008; Faz 3 sonuçları (`docs/system.md`), Faz 3 kod incelemesi
> (`docs/reviews/2026-10-19-faz3-kod-incelemesi.md`).
> **Kullanıcı yönergesi (2026-10-20):** "Faz 4 tasarımını yap; Codex'e sorun var mı diye soracağım, sonra
> birlikte revize edeceğiz." İnceleme sonrası: "Codex'in dediklerini araştır; mantıklıysa değiştir." Sorular
> atlandı; varsayılanlar **Claude'un kararı** olarak işaretli (§10). Yalın, boş iş yok (Faz 2-3 ile aynı ilke).
> Kodlama başka oturumda; bu belge ve ondan yazılacak plan (`docs/plans/F4.md`) yeterli olmalı.
> **Kullanıcı yönergesi (2026-10-20, v3):** "Bunları detaylı ve ayrıntılı yapabiliriz; planda basit bir
> çalışır gibi değil, gerçekten ürün özellikleri göstermeli." Seçilen paketler (dördü de): **(A)** harita
> etkileşimi + araç bilgi kartı, **(B)** bölge kuralları (yasak bölge, bekleme sınırı, yönlü sayım hattı),
> **(C)** olay kaydı, zaman çizelgesi ve dışa aktarma, **(D)** bölgeler ve olaylar motorda (API). "Yalın"
> ilkesi ölçüm ve kanıt tarafında sürer (boş ölçüm yok); ürün tarafında operatörün gerçekten kullanacağı
> davranışlar tam tanımlanır.

## 1. Neden ve ne

Faz 3 sonunda sistem araçları bulup numaralıyor, ama her şeyi **görüntü koordinatında** yapıyor. Drone
kendisi hareket ettiği için görüntüde her araç kayıyor: park etmiş bir araç da ekranda kayıyor. Operatörün
sorusu "hangisi gerçekten hareket ediyor, nereye gidiyor, şu kavşağa kim girdi" — Faz 3 bunu cevaplamıyor.
Faz 4:

**Motor her karede kameranın kendi hareketini (ego-hareket) arka plandan kestirir, izleri bir "harita"
koordinatına taşır ve her izi hareketli / duran diye işaretler; kesintiden (kamera arızası, videoda atlama)
sonra kendini aynı haritada yeniden konumlar. Operatörün tanımladığı bölgeler (alan, yasak bölge, bekleme
sınırı, sayım hattı) motorda değerlendirilir; olaylar, alarmlar ve araç özetleri API'den akar ve günlüğe
yazılır. Ekran haritayı kareleri birleştirerek (göreli mozaik) etkileşimli çizer; araç kartı, bölge
editörü, alarm paneli, olay zaman çizelgesi ve internetsiz dışa aktarma sunar.**

Mülakat hikâyesi (demo): "Drone uçuyor, görüntü kayıyor. Düz çerçeveli kutular hareketli, kesikli kutular
park etmiş araçlar — kamera hareketi çıkarıldığı için. Yanda kareler birleşip bir harita oluşuyor; kaydırıp
yakınlaştırıyorum. Kavşağa bir sayım hattı çekiyorum, yan sokağı yasak bölge, otoparkı 20 saniye bekleme
sınırlı bölge yapıyorum. Yasak bölgeye giren araç kırmızı alarm veriyor, onaylıyorum; bir araca tıklayınca
kartında ne zamandır durduğu, hangi bölgelerden geçtiği görünüyor. IR kamerayı ekrandan kapatıyorum; sistem
RGB'ye geçiyor, IR dönünce eski haritaya geri bağlanıyor, bölgelerim yerinde. Olay günlüğünü CSV, haritayı
PNG olarak indiriyorum — hepsi internetsiz. Olaylar motorun API'sinden geldiği için başka bir istemci de aynı
JSON'u alabilir. Hareketli/duran ayrımı M3OT'ta ölçüldü; etiketleri iz yörüngelerinden ayrı bir yöntemle
türettim ve rastgele bir örneklemi kör olarak elle doğruladım." Bu metin, §6'daki ölçümler ve iddia
koşulları (§6.6) tutmadan yazılmaz.

Yöntem yeni değil, bilerek: ego-hareket kestirimi BoT-SORT'un kamera hareketi telafisindeki (GMC) yaygın
"seyrek optik akış + benzerlik dönüşümü + RANSAC" tarifi; hareket kararı telafi edilmiş hız eşiği. Ayırt
edici olan **dürüst ölçüm**: üründen ayrı etiketleyici, etiketleyiciden bağımsız seçilmiş kör insan
örneklemi, sabit paydalar, oracle / motor satırları.

## 2. Hedefler

1. **Ego-hareket (motorda):** her karede kare → harita benzerlik dönüşümü; anahtar kareye kayıtla birikimli
   kayma sınırlı; açık **harita nesli** sözleşmesi (kanal değişimi, uzun kayıp ve atlamada geçici nesil + yeniden konumlama, kısa kayıpta
   yeniden bağlama); IR öncelikli, IR bozuksa RGB; tespit kutuları maskelenir. C++ ↔ Python eşlikli.
2. **Hareket durumu (motorda):** iz başına `moving / stationary / unknown`, telafi edilmiş ve kutu boyuyla
   normalleştirilmiş hızdan; eşikler ölçümden önce sabit. C++ ↔ Python eşlikli.
3. **Ölçüm (S11):** M3OT'ta iz yörüngelerinden türetilmiş **referans etiketler** (ürünle aynı olmayan ayrı
   kayıt yöntemi), etiketleyiciden bağımsız rastgele seçilmiş **kör insan örneklemi**; oracle ve motor
   satırları, sabit paydalarla; val'de geliştirme ve oracle işlev kapısı, test dondurmadan sonra bir kez.
4. **Harita belleği ve yeniden konumlama (motorda):** anahtar kare kütüphanesi; süreklilik koptuğunda
   (uzun kayıp, kanal dönüşü, videoda atlama) aynı kanalın eski haritasına ORB ile yeniden bağlanma; geçici
   neslin birleştirilmesi. Bölgeler arızadan ve atlamadan sonra yerinde kalır. C++ ↔ Python eşlikli.
5. **Bölgeler, kurallar ve olaylar (motorda, API):** alan, **yasak bölge**, **bekleme sınırı**, **yönlü
   sayım hattı**; olaylar (`giriş`, `çıkış`, `çıkış (kayıp)`, `durdu`, `hattı geçti`), alarmlar (başlar,
   onaylanır, temizlenir), sayaçlar; gösterim süzgecinden bağımsız **bütün takip akışından**. Oturum başına
   olay günlüğü (artımlı okuma, JSON / CSV).
6. **Araç özeti (motorda):** iz başına sınıf, durum ve ne zamandan beri, durum geçmişi, toplam duruş, göreli
   yol uzunluğu, bölge ziyaretleri.
7. **Operatör ekranı:** etkileşimli göreli mozaik harita (kaydır, yakınlaştır, kamerayı izle), araç kartı,
   bölge editörü (çiz, düzenle, türü ve parametresi), alarm paneli (onay), olay zaman çizelgesi (süzgeç,
   anlık görüntü, o kareye git), dışa aktarma (olaylar JSON / CSV, harita PNG) — hepsi internetsiz.
8. **Donmuş kare:** kare düzeyinde bayrak (ego için çekirdek) + `rgb_frozen` / `ir_frozen` alarmı.
9. **Gerileme yok:** Faz 3 Kabul 1-8 yeşil kalır; gecikme p95 ≤ 100 ms (ego + hareket + bölgeler dahil,
   gerçek ardışık video girdisiyle).

## 3. Hedef dışı (bilerek kesilenler)

| Kesilen | Neden |
|---|---|
| Piksel → GPS, metre, km/sa hız | İrtifa, GSD ve kare hızı doğrulanmadı (spec §6 kapsam dışı; M3OT `fps=30` resmi araçta varsayım). Hız yalnız "kutu boyu / kare" biriminde, ekranda sayı olarak gösterilmez. |
| Takipçiye kamera hareketi telafisi (BoT-SORT GMC tarzı Kalman düzeltmesi) | **Kapsam tercihi:** takipçiyi değiştirir; Faz 3 Kabul 4 eşliği ve ADR 0008 ölçümü `trackers==2.6.1`'e bağlı. Kazancın küçük olacağı iddia edilmez (val'deki oracle 0,974 / motor 0,220 HOTA farkı tespit kaybını gösterir, kamera hareketinden gelen ilişkilendirme hatasını ayırmaz). Faz 5 notu. |
| Öğrenilmiş hareket sınıflandırıcısı, yeni eğitim | "Train only what ships"; eşik kuralı yeterli mi önce ölçülür. |
| Ürün yolunda homografi (8 serbestlik) | Nadir görüş; val keşfinde dönme ≈ 0,02°/kare, ölçek ≈ %0,2/kare (§4). Model hatası val'de **ölçülür** (§6.3 harita kayması, referans homografiye karşı); büyük çıkarsa Faz 5'te ele alınır. (Tarayıcı tuvalinin afin çizebilmesi kolaylık, gerekçe değil.) |
| Kanallar arası (RGB ↔ IR) harita bağlama | Kesitler arası kayıt ayrı bir problem; M3OT'ta doğal RGB-IR kayması belgeli (`docs/system.md` Sınırlamalar). Kanal değişince geçici nesil; aynı kanal geri gelince o kanalın eski haritasına yeniden konumlama (§5.11). RGB dönemindeki izler IR haritasına taşınmaz. |
| Döngü kapatma, global optimizasyon (bundle adjustment) | Harita "göreli" ve niteliksel; kayma ölçülüp raporlanır (§6.3). |
| Diske kalıcı kayıt, veritabanı, oturumlar arası bölge saklama, çoklu kullanıcı, sesli alarm | Spec §17 kapsam dışı ilkesi. Bölgeler, olay günlüğü ve araç özetleri motorun oturum belleğinde (oturum yaşadıkça: atlama ve kamera arızası dahil); kalıcılık yerine dışa aktarma (§5.13). Sayfa yenilenip yeni oturum açılınca bölgeler gider (harita koordinatı oturuma bağlı). |
| Olay doğruluğunun veri setinde ölçülmesi | Bölgeler operatör tanımı; GT'de bölge yok ve harita koordinatı ürünün kendi kestirimi. Olay mantığı senaryo testleriyle (Kabul 7), haritanın doğruluğu harita kayması ve yeniden konumlama ölçümleriyle (§6.3) dolaylı kanıtlanır. |
| TEKNOFEST biçiminde JSON çıktısı | Şartname okunmadı; Faz 5'te ayrı iş. |
| Yeni veri seti (VT-MOT vb.) | M3OT yeterli (iz numaraları kanca K5 ile korunuyor). |
| Faz 3 incelemesinin K4-K6, K8-K11'i | Faz 4'ün dokunmadığı yerler; açık kalır. (S1, S3, S6, S11, S12, K7 girer, §5.10.) |

**Anti-hedefler (çalışsa bile başarısız sayılır):**
- Referans etiketler ürünün kendi ego tahmincisiyle üretilip ürünün "doğruluğu" diye raporlanıyor (döngüsel ölçüm).
- İnsan doğrulaması yalnız etiketleyicinin emin olduğu örneklerden seçiliyor (dışlanan nüfus görünmüyor).
- Test dizileri bir eşik, parametre, fikstür, demo ya da tasarım kararında kullanılıyor.
- Faz 3 eşlikleri, kapalı ağ kanıtları ya da 100 ms bütçesi bozuluyor ve CI / bench bunu göstermiyor.
- Harita gerçek konum gibi, hız gerçek hız gibi sunuluyor.
- Donmuş bir kanal ego-harekete girip bütün araçları "duran" gösteriyor.
- Operatörün güven kaydırıcısı bölge olaylarını ya da sayaçları değiştiriyor.
- Bir kesintiden (kamera arızası, atlama) sonra bölgeler **sessizce yanlış yerde** çalışıyor: yeniden
  konumlama doğrulanmadıysa bölgeler askıya alınır ve ekranda öyle görünür; asla tahmini bir konumda olay
  üretmez.
- Bir alarm operatör onaylamadan kendiliğinden kayboluyor (alarm temizlenince geçmişte kalır, görünür).

## 4. Val keşfi (2026-10-20, laptop, yalnız M3OT val `1_08`, `2_08`; test'e bakılmadı)

Geçici betikle (scratchpad; plan Görev 1'de `baykus-motion explore` olarak yeniden üretilip
`results/motion/explore_val.json`'a yazılır — o zamana kadar bu sayılar **kanıt değil keşif notu**):

- **Kamera hareket ediyor:** IR'de ORB + RANSAC ile ardışık kare benzerlik dönüşümü; öteleme ortanca
  1,50 px/kare (`1_08`, p90 2,50), 1,06 px/kare (`2_08`, p90 1,79); dönme ortanca ≈ 0,02°/kare; ölçek değişimi
  ortanca ≈ %0,2/kare, hep aynı yönde (yaklaşma). RGB'de benzer.
- **Zincir kayması gerçek:** ardışık dönüşümleri 600 kare çarpınca ölçek 4,3 (`1_08`) ve 3,0 (`2_08`) çıkıyor;
  oysa GT kutu boyu (√alan ortancası, 100 karelik bloklar) yalnız 12,4 → 15,6 ve 10,2 → 16,4 px büyüyor
  (≈ 1,3-1,6 kat). Ardışık kare zinciri sistematik ölçek kayması biriktiriyor → anahtar kareye kayıt şart (§5.1).
- **Hareketli / duran ayrılıyor:** GT kutuları arka planda maskelenip, 15 karelik pencerede telafi edilmiş
  merkez yer değiştirmesi / 15 / kutu boyu (`v`, "kutu boyu / kare"):

  | Dizi | `v ≤ 0,03` | `0,03 < v < 0,10` | `v ≥ 0,10` | Toplam (iz, kare) |
  |---|---|---|---|---|
  | `1_08` | 1309 | 412 (%8,3) | 3253 | 4974 |
  | `2_08` | 2381 | 848 (%11,7) | 4043 | 7272 |

  **Yeniden üretim (2026-10-20, `baykus-motion explore` → `results/motion/explore_val.json`; bu kayıt
  esastır):** adım ve zincir ölçeği aynı (1,50 / 1,06 px; 4,31 / 3,03); histogram `1_08` 1277 / 434 (%8,7) /
  3263, `2_08` 2375 / 846 (%11,6) / 4051. Küçük fark geçici betiğe göre maske kutularının yuvarlanmasından ve
  `≤ 0,03` sınırından (betik `< 0,03` saymıştı); eşikler ve yorum değişmez.
  Dağılım iki tepeli: duranlar `v ≈ 0,01-0,02`'de (GT kutu titremesi), hareketliler 0,15-0,5'te; `1_08`'de
  0,04-0,08 arası çukur, `2_08`'de daha düz (yavaşlayan / duran-kalkan araçlar). Dizi başına 39 iz; iz uzunluğu
  ortanca 122 ve 188 kare. **Eşikler (0,03 ve 0,10) bu keşiften, ölçümden önce sabitlenir** (§5.2); val
  sonuçları bu yüzden "val'de keşif" etiketi taşır (Faz 3 `r_min` ile aynı düzen).
- **Birim hakkında:** kutu boyuyla normalleştirme yaklaşma / irtifa değişimine karşı ölçek bağımsızlığı verir.
  Kare başına birim kare hızını **bilmeyi** gerektirmez ama ona **bağlıdır**: aynı fiziksel hareket daha düşük
  kare hızında kare başına daha büyük görünür. Eşikler M3OT'un (doğrulanmamış) kare hızına aittir; başka kare
  hızında yeniden ölçülmeden kullanılmaz.

## 5. Mimari

```
motor (C++, oturum başına durum)
decode → health(+donmuş) → gate → pre → infer → post ─┐
                                                       ├→ ego (anahtar kare, LK, RANSAC, nesil,
                                                       │       harita belleği + yeniden konumlama)
                                                       │  → track (değişmez) → motion → zones/alarms
                                                       │  → track summary → event log → JSON
                                                       └─ ego maskesi: takip katmanı kutuları (ham ≥ 0,1)
JSON: detections[] (gösterim, conf süzgeçli) · tracks[] (bütün takip çıktısı, süzgeçsiz)
      ego{frame_to_map, map_gen, merge, relocalized, ...} · events[] · alarms[] · zone_counts{}
      focus_track{...} · health.*.frozen
API:  /v1/session/zones (PUT/GET) · /v1/session/events (artımlı, JSON/CSV) · /v1/session/alarms/ack
      /v1/session/track · /v1/session/reset?scope=all|tracks

ekran (Next.js) → oynatıcı (detections[]; kutu biçimi = durum; tıkla → araç kartı)
               → harita (mozaik, izler, bölgeler; kaydır/yakınlaştır) + bölge editörü
               → alarm paneli · zaman çizelgesi + olay listesi · araç kartı · dışa aktarma
```

Kural: **ölçülen her şey motorda ve Python referansıyla eşlikli** (ego, nesil, harita belleği, hareket
durumu, donmuş kare). **Ölçülmeyen ama ürünün sözleşmesi olan mantık motorda, C++ senaryo testleriyle**
(bölgeler, kurallar, alarmlar, olay günlüğü, araç özeti; Python referansı yok — veri setinde karşılığı
yok). **Ekran yalnız gösterim ve etkileşim** (mozaik çizimi, editör, paneller, dışa aktarma; Vitest).
Takipçi değişmez; ego takipten bağımsız, hareket durumu takip kimliklerini kullanır.

### 5.1 Ego-hareket ve harita nesli (`engine/src/ego.cpp`, Python `baykus.motion.ego`)

**Bulgu (2026-10-20, kör örneklem sırasında; val `1_08`, `2_08` kareleri gözle):** M3OT val'de RGB ve
IR kareleri **aynı hizada değil**: görüş alanı, ölçek ve konum belirgin farklı (yol IR'de ortada, RGB'de
sağda; kavşaklar farklı yerde). Faz 1'in test bölümü ölçümü (etiket IoU ortanca 0,35) bunun hafif hâliydi.
**Sonuç (Claude'un kararı, plan Görev 4 ve 11'e işlenir):** ego **yalnız IR**; aşağıdaki RGB yedeği bu
veride kullanılamaz (RGB'den kestirilen kamera hareketi IR koordinatındaki izlere uygulanamaz) ve
`ChannelSelector` kaldırılır. IR `lost` / `noisy` / donmuş alarmı → ego kaybı (`source: none`); IR dönünce
§5.11 yeniden konumlama. Kabul 1 `ir_drop` senaryosu buna göre: RGB geçici nesli yok, IR dönüşünde birleşme.
Ekrandaki RGB görünümünde kutular M3OT'ta araçların üstüne oturmaz (sınırlama; hizalı kameralarda doğru).

**Kaynak kanal (v3 metni; yukarıdaki bulguyla RGB yedeği kalktı):**
- IR; IR'nin kare bayraklarında `lost` ya da `noisy` varsa ya da IR **donmuş alarmı** (§5.6, 5 karenin 3'ü)
  açıksa RGB (gri); ikisi de uygun değilse `source: none`. Karar ham girdinin sağlığına göre (kapı
  sıfırlamasından önce).
- RGB'den IR'ye dönüş: IR art arda **5** kare temiz (bayraksız, donmuş alarmı kapalı) olunca. Titremeyi süzer.
- **Donmuş kare (kare düzeyi):** seçili kanalın o karesi `frozen_frame` ise (§5.6) kare kayda girmez:
  `ego_ok=false` (aşağıdaki kayıp kuralı). Kanal değişimi ancak alarm koşulunda (yukarıda).
- **Çözünürlük:** yarı çözünürlük gri (320 × 256), sağlık modülünün küçültmesiyle aynı:
  `cv2.resize(..., (w // 2, h // 2), INTER_AREA)` (`baykus.health.features`). Yarı çözünürlükteki piksel
  `i` tam çözünürlükte `2i + 0,5` merkezlidir.
- **Maske:** takip katmanındaki kutular (ham ≥ 0,1), 4 px genişletilmiş, yarı çözünürlüğe ölçekli; 8 px kenar
  bandı (kaydırma dolgusu). Tespit ego'dan önce koştuğu için maske aynı karenin kutularıdır.

**Kayıt** (BoT-SORT GMC "sparseOptFlow" tarifinin anahtar kareli hâli; parametreler `configs/engine/motion.json`):
1. **Anahtar kare** `k`: maskeli `goodFeaturesToTrack` (en çok 300 köşe, kalite 0,01, en az aralık 8 px, blok 3).
2. Her yeni kare `t`: anahtar karenin köşeleri `calcOpticalFlowPyrLK` ile **doğrudan anahtar kareden** `t`'ye
   (pencere 21, 3 seviye); ileri-geri denetimi (geri takip hatası ≤ 1 px); `t`'nin maskesindeki noktalar atılır.
3. `estimateAffinePartial2D` (4 serbestlik), RANSAC eşiği 1,0 px (yarı çözünürlük), en çok 2000 yineleme,
   güven 0,995, iyileştirme 10. `ok` = iç nokta ≥ 30 **ve** oranı ≥ 0,4.
4. **Yeni anahtar kare** (`ok` olan geçerli kare olur, köşeler yeniden): iç nokta < 60 ya da oran < 0,5 ya da
   anahtar kareden bu yana 30 kare ya da öteleme > 40 px (yarı çözünürlük). `ok` olmayan kare anahtar kare
   **olamaz** (v1'deki "ok değilse yeni anahtar" kuralı kaldırıldı: kayıp aralığın hareketini haritadan düşürürdü).
5. **Harita dönüşümü:** `A` anahtar kare → `t` (yarı çözünürlük); tam çözünürlüğe `A_full = S·A·S⁻¹`,
   `S = [[2,0,0,5],[0,2,0,5]]`. `Mₜ = M_k · A_full⁻¹` (kare → harita). Neslin ilk karesinde `M = I`.

**Harita nesli (`map_gen`, oturum başına sayaç; ekranın ve hareket geçmişinin geçerlilik sözleşmesi):**
- Oturumun ilk karesi nesil 1. Bir nesil içinde bütün `Mₜ`'ler aynı harita koordinatındadır.
- **Kayıp:** `ok` olmayan kare (kayıt başarısız, `source: none`, donmuş kare): `ego_ok=false`, `Mₜ` yok
  (`frame_to_map: null`; ekran ve hareket modülü o kareyi kullanmaz). Anahtar kare ve `M_k` korunur.
- **Kısa kayıptan dönüş (aynı kanal, kayıp ≤ 10 kare):** geçerli kare **son geçerli anahtar kareye** doğrudan
  kaydedilir (aynı adımlar, LK 4 seviye; anahtar kareden bu yana kare sayısı kuralı bu denemede uygulanmaz).
  `ok` ise nesil sürer (bağlandı); değilse süreklilik kopması (geçici nesil, aşağıda).
- **Süreklilik kopması:** kanal değişimi (IR ↔ RGB) ya da > 10 karelik kayıp ya da bağlama başarısız ya da
  `session/reset?scope=tracks` (atlama) → **geçici nesil**: `map_gen` = yeni numara, `provisional: true`,
  geçerli `ok` kare onun ilk karesi (`M = I`), yeni anahtar kare. Aynı anda **yeniden konumlama** başlar
  (§5.11): başarılı olursa geçici nesil eski nesle **birleştirilir**, olmazsa 30 kare sonra kalıcı yeni nesil olur.
- **Etkisi:** hareket geçmişi (§5.2) geçici neslin başında silinir (birleşmede dönüşümle taşınır, §5.11);
  bölgeler yalnız kendi neslinde değerlendirilir, başka nesildeyken **askıda** (§5.5); ekran mozaiği nesil
  başına katman olarak tutar (§5.4).
- `session/reset` (`scope=all`, varsayılan) nesli 1'e döndürür, harita belleğini, bölgeleri, günlüğü ve her
  şeyi siler. Oturumsuz istek ego hesaplamaz (`ego: null`).

### 5.2 Hareket durumu (`engine/src/motion.cpp`, Python `baykus.motion.state`)

- İz başına halka arabellek (kimliği olan izler; `track_id: null` → `unknown`), yalnız `ego_ok` karelerde:
  harita merkezi `cₜ = Mₜ(kutu merkezi)` ve harita boyu `sₜ = √(w·h) · σ(Mₜ)`, `σ = √|det(Mₜ'nin 2×2 kısmı)|`.
  Pay ve payda aynı (harita) koordinatında.
- `K = 15` kare. Geçerli karede ve tam `t − K` karesinde kaydı olan, aynı nesilde ve `(t − K, t]` aralığındaki
  bütün karelerde `ego_ok` olan iz için `v = ‖cₜ − cₜ₋ₖ‖ / K / sₜ`.
- **Karar (histerezis):** `v ≤ 0,03` → `stationary`; `v ≥ 0,10` → `moving`; arada → önceki karar (yoksa
  `unknown`). `v` hesaplanamazsa önceki karar en çok `K` kare tutulur, sonra `unknown`. Yeni (geçici) nesilde
  bütün kararlar `unknown`'dan başlar; birleşmede geçmiş `cₜ` ve `sₜ` değerleri birleştirme dönüşümüyle
  (`cₜ ← T(cₜ)`, `sₜ ← σ(T)·sₜ`) hedef nesle taşınır ve kararlar sürer.
- Yeni iz ilk `K` karede `unknown`. Parametreler (15; 0,03; 0,10; tutma `K`) **ölçümden önce sabit, ayarlanmaz**.
- Çıktı (iz başına): `motion`, `speed_rel` (`v`, yoksa `null`), `map_xy` (`cₜ`, ego yoksa `null`).
- **Karar gecikmesi (kare cinsinden; raporlanır, §6.3):** pencere geçmişe baktığı için durum değişimi ≈ `K`
  kare gecikir; bölge `durdu` olayı buna ek 30 kare bekler. 30 kare/sn'de ≈ 0,5 s + 1 s; ekranın varsayılan
  10 kare/sn oynatmasında ≈ 1,5 s + 3 s. Bilinen sınırlar: çok yavaş araç (`v < 0,10` sürekli) `moving`
  olmaz; yüksek yapıların paralaksı (100-120 m'de küçük, varsayım) kaba RANSAC'la sınırlı.

### 5.3 Referans etiketler ve kör insan örneklemi (yalnız Python, `baykus-motion labels / audit`)

Etiket ürünle **aynı yöntemle** üretilirse ego hataları iki tarafta da aynı olur ve ölçüm iyimser çıkar.
Referans etiketleyici bilerek farklı:

| | Ürün (§5.1-5.2) | Referans etiket |
|---|---|---|
| Kayıt | anahtar kareye LK, benzerlik (4 sd), yarı çözünürlük | **doğrudan** `t − K` ↔ `t + K`, SIFT (2000) + oran testi 0,75 + **homografi** RANSAC 1,5 px, tam çözünürlük |
| Maske | model tespitleri | GT kutuları (6 px genişletilmiş) |
| Pencere | geçmiş `[t − K, t]` (nedensel) | ortalanmış `[t − K, t + K]` |
| Kutular | motor / oracle | GT (IR etiketleri, Faz 3 süzgeci: flag 1, sınıf 1, görünürlük > 0,25) |

Eşikler (0,03 / 0,10) iki tarafta aynı: bu bir döngü değil, görevin **tanımı** ("hareketli" = telafi edilmiş
hız ≥ 0,10 kutu boyu / kare). Döngüsellik ortak **kestirim hatasından** gelir; o yüzden kestirici ayrı ve
etiketleyicinin kendisi kör insan örneklemiyle sınanır.

- `v_ref = ‖H(c_{t−K}) − c_{t+K}‖ / (2K) / √(w_{t+K}·h_{t+K})` — pay `t + K` karesinin pikselinde, payda da
  (v1'de payda `t` karesindeydi; yaklaşmada ≈ %3 sistematik fark yaratıyordu). `≤ 0,03` → duran, `≥ 0,10` →
  hareketli; arada → **bant**; kayıt başarısız (iç nokta < 50) → **kayıtsız**; iz `t ± K`'de yok → **kenar**.
  Bant, kayıtsız ve kenar birlikte **dışlanan** nüfus; ölçümde ayrıca raporlanır (§6.3).
- Etiketler `data/motion/labels_<bölüm>.csv` (git'e girmez; komutla yeniden üretilir; sha256 sonuç JSON'una).
  Etiketleyicinin kodu ve parametreleri **val audit'inden sonra dondurulur** (`motion-labels-v1` etiketi ya da
  commit sha'sı sonuç JSON'larına); test etiketleri dondurulmuş kodla üretilir.

**Kör insan örneklemi (KAPI, kullanıcı adımı ≈ 20-30 dk; val):** `baykus-motion audit` yerel bir HTML sayfası
üretir (`data/motion/audit/index.html`, internet gerekmez, git'e girmez).
- **Nüfus:** val'deki bütün GT örnekleri (iz, kare), etiketleyicinin çıktısından **bağımsız**; yalnız aracın
  `t ± K` aralığında görünür olması şartı (insanın hareketi görebilmesi için; kenar örnekleri bu yüzden
  dışarıda, sayıları raporlanır).
- **Seçim (seed 42, iz başına en çok 3, karışık sıra):** **80 düzgün rastgele** örnek (asıl ölçüt) + **40
  hedefli** örnek: etiketleyicinin bant / kayıtsız dediği ve etiket değişiminin ±`K` çevresindeki (geçiş)
  örnekler — dışlanan nüfusta insan ne görüyor, bilgi için.
- **Gösterim:** etiket ve hız gösterilmez. Aracın çevresinden 192 × 192 px, `t − K … t + K` arası her 3. kare,
  **telafisiz** (2. turdan itibaren araca ortalanmış; aşağıdaki 1. tur notu) (herhangi bir kestiricinin kaydı kullanılırsa doğrulama döngüsel olur; insan arka plana göre
  karar verir), oynatılabilir; aracın kutusu ince çerçeveyle.
- Kullanıcı "hareketli / duran / emin değilim" işaretler; sayfa cevapları JSON olarak indirir →
  `baykus-motion audit-score` → `results/motion/label_audit_val.json`.
- **1. tur (2026-10-20) ve düzeltme:** (a) 65/69 = %94,2 (GA %86-98) → ❌; (b) 0/80 ✅; (c) 69/80 ✅
  (`results/motion/label_audit_val_r1.json`). Uyumsuz 4 örneğin hepsi "insan hareketli / etiketleyici duran",
  `v_ref` 0,021-0,029. Kullanıcı geri bildirimi: RGB'de kutu araçtan kayıyor (IR etiketi + doğal kayma) ve
  bir araç klibin sonunda hareket ediyordu. İnceleme: sabit pencerede kamera hareketi park etmiş araçları
  da kayıyor gösteriyordu. **Tek düzeltme hakkı sunuma kullanıldı** (etiketleyici değişmedi): pencere her
  karede aracın GT kutusuna ortalanır (kestirici yok, döngüsel değil), dışı siyah; RGB'de kutu yerine
  köşe işaretleri; sayfaya "çoğunlukla hareketli / çoğunlukla duran / yarı yarıya → emin değilim" kuralı.
  1. turun cevapları sonuç görüldükten sonra değiştirilmedi. **2. tur:** seed 43, 1. turun 120 örneği
  hariç; aynı ölçüt. Tutmazsa B planı (kullanıcı kararı, ADR).
- **Kabul 4 ölçütü (ölçümden önce), 80 düzgün örnekte:**
  (a) insanın ve etiketleyicinin karar verdiği örneklerde uyum **≥ %95**;
  (b) "emin değilim" **≤ %15**;
  (c) insanın karar verdiği örneklerin **≥ %80**'i etiketleyicinin etiketli nüfusunda (dışlama ölçülen
  nüfusu fazla daraltmıyor).
  80 örnekte %95'in güven aralığı geniştir; sonuç Wilson %95 GA ile yazılır. Tutmazsa etiket yöntemi val'de
  **bir kez** düzeltilir ve yeni seed'le yeniden doğrulanır; yine tutmazsa ölçüm kısmı spec §23 B planına
  iner (niteliksel demo; kullanıcı kararıyla ADR).
- **Test (dondurmadan sonra, kullanıcı adımı ≈ 20-30 dk):** aynı protokolle test dizilerinde kör örneklem
  (`label_audit_test.json`). Sistemi değiştirmez; yalnız raporlanır. Ayrıca ürünün kararları bu örneklemlerde
  doğrudan insan cevabıyla karşılaştırılır (etiketleyiciden tamamen bağımsız, küçük örneklem; §6.3).

### 5.4 Göreli mozaik harita (ekran, `ui/lib/mosaic.ts`, `ui/components/MapPanel.tsx`)

- Ego kanalının görüntüsü her **5.** `ego_ok` karede çeyrek çözünürlükte (160 × 128) saklanır ve
  `frame_to_map` ile tuvale çizilir (`setTransform`). Yeni kareler eskilerin üstüne. Bellek: nesil başına en
  çok 400 küçük resim (dolunca eşit aralıkla seyreltilir; harita kapsamı korunur).
- **Nesil katmanları:** her nesil ayrı katman. Ekranda yalnız **geçerli neslin** katmanı ve onun bölgeleri;
  eski nesiller saklanır (bölgeleri askıda). Birleşmede (`ego.merge = {from, to, transform}`) kaynak
  katmanın küçük resimleri, kuyrukları ve görüş alanı dönüşümle hedef katmana taşınır.
- **Etkileşim:** sürükleyerek kaydırma; tekerlek ve düğmelerle yakınlaştırma (0,25× – 8×, imleç çevresinde);
  "Sığdır"; "Kamerayı izle" (görüş alanını ortada tutar; kullanıcı kaydırınca kapanır). Fiziksel ölçek yok;
  yalnız göreli yakınlaştırma yüzdesi.
- Üstünde: kameranın şu anki görüş alanı (kare dörtgeni), iz kuyrukları (son 150 kare, `tracks[].map_xy`,
  durum rengi ve çizgi biçimiyle), seçili izin tam yolu (araç özetinden, §5.12), bölgeler türüne göre
  (§5.5; sayım hattında yön oku ve iki yön sayacı), aktif alarmlı bölge kalın kırmızı kenarla (yanıp sönmez).
- **Seçim:** izin kuyruğuna ya da işaretçisine tıklama → araç kartı (§5.8). İmleç altında iz no balonu.
- `ego_ok=false` karelerde küçük resim eklenmez; kuyruk o karede kesilir. Geçici nesilde başlıkta
  "konum aranıyor…" rozeti; kalıcı yeni nesil olunca "yeni harita (nesil n) — önceki bölgeler askıda".
- Panel başlığı: "göreli harita (kamera hareketinden; ilk kareye göre; kanal değişince yeniden konumlanır)".
- Harita gerçek konum değildir; ölçek, kuzey, metre gösterilmez.

### 5.5 Bölgeler, kurallar, alarmlar ve olaylar (motor, `engine/src/zones.cpp`)

Bölgeler operatör tanımıdır, ama ürünün davranış sözleşmesidir: tarayıcı kapalıyken de, başka bir istemci
bağlıyken de aynı olaylar üretilmeli. Bu yüzden **motorda** ve oturum başına. Veri setinde karşılığı
olmadığı için Python referansı yok; doğruluk C++ senaryo testleriyle (Kabul 7).

**Bölge tanımı** (`PUT /v1/session/zones` gövdesi, tamamını değiştirir):

```json
{"zones": [
  {"id": "z1", "name": "Kavşak", "kind": "area", "gen": 1, "points": [[0, 0], [10, 0], [10, 10]]},
  {"id": "z2", "name": "Yan sokak", "kind": "restricted", "gen": 1, "points": [[...]]},
  {"id": "z3", "name": "Otopark", "kind": "dwell", "gen": 1, "points": [[...]], "dwell_frames": 600},
  {"id": "h1", "name": "Ana yol", "kind": "line", "gen": 1, "points": [[0, 5], [20, 5]], "count": "both"}
]}
```

- Koordinatlar harita pikseli, `gen` neslinde. **Doğrulama** (hata → 400, Türkçe ileti, hiçbir bölge
  değişmez): `id` benzersiz, `[A-Za-z0-9_-]{1,16}`; `name` 1-40 karakter; en çok **16** bölge; çokgen türleri
  3-32 nokta, basit (komşu olmayan kenarlar kesişmez), alan > 1 px²; `line` tam 2 farklı nokta;
  `dwell_frames` 1-36000; `count` `both | a_to_b | b_to_a`; `gen` var olan bir nesil.
- `GET /v1/session/zones` saklananları ve durumlarını döner (`active | suspended`).
- Bölgeler `scope=tracks` sıfırlamasında ve nesil değişiminde **korunur**; yalnız `scope=all` siler.

**Gözlem:** iz × bölge her karede `içerde` / `dışarıda` / `yok` (iz `tracks[]`'ta ve `map_xy` geçerli neslin
koordinatında; değilse `yok`). Bölge yalnız `gen` = geçerli nesil ve nesil kalıcıyken (`provisional: false`)
**aktif**; değilse **askıda**: gözlem yok, iz durumları silinir, sayaçlar 0, açık alarmları "askıya alındı"
nedeniyle temizlenir. Birleşmeyle geçerli nesil bölgenin nesli olursa yeniden aktif.

**"Art arda" kuralı (bütün türler):** sayaçlar yalnız `içerde` / `dışarıda` gözlemlerinde ilerler; araya en
çok **5** `yok` kare girebilir; daha uzun `yok` dizisi sayacı sıfırlar (iz–bölge durumu korunur).

**Türler ve davranış:**

| Tür | Olaylar | Alarm | Sayaç |
|---|---|---|---|
| `area` (alan) | `enter` (3 `içerde` gözlem art arda; kare = ilkinin karesi), `exit` (3 `dışarıda`), `exit_lost` (içerdeyken 30 kare `tracks[]`'ta yok), `stopped` (içerdeyken 30 `stationary` gözlem art arda; kalış başına bir kez; `unknown` ve `moving` sayacı sıfırlar) | — | içerdeki iz; hareketli / duran / bilinmiyor |
| `restricted` (yasak) | `area` olayları | `intrusion`: bölgede en az bir iz içerde durumdayken açık (bölge düzeyi; `track_ids` güncel); son iz çıkınca temizlenir | `area` gibi + bugüne kadar giren iz sayısı |
| `dwell` (bekleme sınırı) | `area` olayları | `dwell_exceeded`: iz içerde ve `dwell_frames` gözlem art arda `stationary` → iz düzeyinde açılır; iz çıkınca, kaybolunca ya da 3 `moving` gözlemle temizlenir | `area` gibi + sınırı aşan iz sayısı |
| `line` (sayım hattı) | `line_cross` (`dir: a_to_b` ya da `b_to_a`) | — | yön başına geçiş sayısı (`count` ayarındaki yönler) |

**Sayım hattı geçişi:** taraf `s = sign((B − A) × (p − A))` (`> 0` sol, `< 0` sağ, `= 0` önceki taraf).
Bir iz için **kararlı taraf** = 3 gözlem art arda aynı taraf. Kararlı taraf değiştiğinde, eski taraftaki son
gözlemin konumu ile yeni taraftaki ilk gözlemin konumu arasındaki doğru parçası **AB parçasını kesiyorsa**
(uç noktalar dahil) geçiş sayılır; kesmiyorsa (hattın uzantısından dolanma) sayılmaz. `a_to_b` = soldan
sağa (A'dan B'ye bakan gözlemciye göre). Kare = yeni taraftaki ilk gözlemin karesi. Bir iz aynı hattı
tekrar geçebilir (her geçiş ayrı olay).

**Alarmlar:** `alarms[]` (her yanıtta açık alarmların tamamı): `{id, kind, zone_id, track_ids, since_n,
since_frame, acked}`. `POST /v1/session/alarms/ack?session=&alarm=<id>` → `acked: true` (alarm açık kalır,
yalnız işaretlenir; günlüğe `alarm_ack`). Temizlenen alarm günlükte `alarm_clear` olarak kalır; onaylanmadan
temizlenen ekranın geçmiş listesinde "onaylanmadan kapandı" diye işaretli. Donmuş, kayıp, gürültü ve
hizasızlık alarmları Faz 3 sözleşmesinde (ekranda) kalır; bölge alarmları motorda.

**Kimlikler:** API'deki `track_id` oturum içinde **benzersiz ve artan**: `scope=tracks` sıfırlamasında
takipçi kimlikleri 0'dan başlar; motor bir taban ekler (`api_id = tracker_id + base`; `base` = o ana kadar
verilen en büyük kimlik + 1). Faz 3 takip eşliği takipçinin kendi kimlikleriyle (değişmez).

### 5.6 Donmuş kare (motor `health`, Python `baykus.health.frozen`)

- Kanal başına, oturumlu istekte: yarı çözünürlük gri görüntünün önceki kareyle ortalama mutlak farkı
  `diff_mad` (0-255); `frozen_frame` = `diff_mad ≤ 0,5`. Sıfır görüntü (`lost`) donmuş sayılmaz (`lost`
  öncelikli).
- **İki kullanım:** (1) **kare bayrağı** ego'ya anında girer (§5.1: o kare kayda girmez) — çekirdek, kesilmez;
  (2) **alarm** `rgb_frozen` "RGB görüntü donmuş" / `ir_frozen` "IR görüntü donmuş" Faz 3 alarm sözleşmesiyle
  (son 5 karenin en az 3'ü); motor da aynı koşulu kanal seçiminde kullanır (§5.1). Kapı kararını
  **değiştirmez** (Faz 2-3 kuralları aynen).
- Ekran arıza simülasyonuna "kareyi dondur (RGB / IR)": o kanalın son gönderilen baytları tekrar gönderilir.
- Ölçüm (yalnız rapor): M3OT val temizde kanal başına `frozen_frame` ve alarm oranı; sentetik `freeze@K`
  (K = ⌊n/2⌋'den sonra aynı kare) gecikmesi. Eşik 0,5 ölçümden önce sabit; temizde alarm çıkarsa ayarlanmaz,
  sınırlama.

### 5.7 API değişiklikleri (geriye uyumlu: Faz 3 alanları aynen, yalnız ekleme)

```
POST /v1/detect  sorgu: Faz 3 + frame=<dizi kare no> (isteğe bağlı) + focus=<track_id> (isteğe bağlı)
  →  Faz 3 yanıtı +
  detections[i].motion, .speed_rel, .map_xy            (gösterim katmanı, conf süzgeçli — Faz 3 gibi)
  tracks: [{"track_id","cls_id","box","score_raw","score","motion","speed_rel","map_xy"}]
          takip katmanındaki (ham ≥ 0,1) kimliği olan bütün tespitler, conf süzgeci YOK
  ego: {"ok","source":"ir|rgb|none","map_gen","provisional","frame_to_map":[a,b,tx,c,d,ty] | null,
        "inliers","inlier_ratio","keyframe","relinked","relocalized",
        "merge": {"from","to","transform":[6]} | null} | null
  events: [...]  (bu karenin günlük kayıtları, §5.13)
  alarms: [...]  (açık bölge alarmlarının tamamı, §5.5)
  zone_counts: {"<zone_id>": {"status","inside","moving","stationary","unknown",
                              "entered_total","exceeded_total","a_to_b","b_to_a"}}
  focus_track: {...} | null   (§5.12)
  health.rgb.frozen, health.ir.frozen, health.rgb.diff_mad, health.ir.diff_mad   (oturumsuz: null)
  timing_ms.ego, timing_ms.motion, timing_ms.zones
PUT  /v1/session/zones?session=        gövde §5.5 → 200 {zones} | 400 {error}
GET  /v1/session/zones?session=        → {zones:[{...,"status"}]}
GET  /v1/session/events?session=&since=<seq>&format=json|csv   (§5.13)
POST /v1/session/alarms/ack?session=&alarm=<id>  → 200 | 404
GET  /v1/session/track?session=&id=    → araç özeti | 404
POST /v1/session/reset?session=&scope=all|tracks  (varsayılan all; Faz 3 davranışı)
GET  /v1/model → + "motion": {configs/engine/motion.json parametreleri}, "motion_sha256"
```
- `tracks[]` motorun mevcut takip girdisinden kurulur (`runtime.cpp`: takipçi `dets_all`'ın ham ≥ 0,1 kısmını
  görür; gösterim listesi ayrıca `filter_display(conf)` ile süzülür — Faz 3'te kimlikler yalnız süzülmüş
  listeye yazılıyordu).
- Ego, hareket, bölgeler, günlük yalnız `track=true` + `session` ile (oturumsuz: `ego: null`, `tracks: []`,
  `events: []`, `alarms: []`, `motion: "unknown"`). Oturum uçları bilinmeyen oturumda 404 (oturum açmaz).
- Hatalar Faz 3 gibi (400 / 404 / 413 / 500); gövde > 64 KB bölge isteği 413. Motor tek hatalı istekle çökmez
  (Kabul 8 genişler: bölge ucu bozuk JSON, sınır dışı değerler, kendini kesen çokgen).
- `configs/engine/motion.json` repoda (`ego`, `reloc`, `motion`, `frozen`, `zones`, `log`, `summary`
  parametreleri); motor açılışta okur; sha256 `/v1/model`'de. Bench için `ego.enabled` (§6.4).

### 5.8 Ekran (`ui/`)

Düzen (tek sayfa, Faz 3 düzeni korunur, yeni paneller eklenir; masaüstü ≥ 1440 px hedef, 1280 px'te
sağ sütun sekmeli):
- **Sol:** oynatıcı (Faz 3) + altında **zaman çizelgesi**.
- **Orta / sağ:** **harita paneli** (§5.4) — oynatıcıyla yan yana ya da sekme ("Görüntü / Harita / İkisi").
- **Sağ sütun sekmeleri:** "Tespitler" (Faz 3 listesi + durum sütunu), "Bölgeler" (bölge listesi + sayaçlar
  + editör araçları), "Alarmlar" (açık + geçmiş), "Olaylar" (süzgeçli liste), "Sağlık" (Faz 3 paneli + ego
  durumu), "Gecikme" (Faz 3).
- **Araç kartı:** seçili iz için yüzen kart (kapatılabilir).

Bileşenler:
- **Kutular:** `moving` düz kalın çerçeve + "H"; `stationary` kesikli ince çerçeve + "D"; `unknown` Faz 3
  görünümü. Yalnız renge dayanmaz. Seçili iz vurgulu. Üst çubuğa "yalnız hareketliler" süzgeci (yalnız
  gösterim; olaylar etkilenmez). Kutuya tıklama → araç kartı.
- **Araç kartı** (`focus` ile her karede güncel): iz no, sınıf, durum rozeti ve "N kare (≈ s) dir duruyor /
  hareket ediyor", durum zaman şeridi (`segments`), toplam duruş, göreli yol (kutu boyu), bölge ziyaretleri
  (giriş / çıkış kareleri), son görülme; "Haritada göster" (haritayı ize ortalar); iz kaybolduysa gri ve
  "son görülme". Süreler kare cinsinden; saniye `index.json` `fps`'inden ve "(30 kare/sn varsayımı)" notuyla.
- **Bölge editörü** (harita üzerinde): araç düğmeleri "Alan / Yasak bölge / Bekleme sınırı / Sayım hattı";
  tıklayarak nokta, çift tıklama ya da Enter ile bitir, Esc iptal; köşe sürükleyerek düzenle, köşe ekle /
  sil; ad değiştir; bekleme sınırı saniye girilir (kareye `fps` ile çevrilir, yuvarlanır, notlu); sayım
  hattında yön seçimi ve A / B etiketi; sil. Her değişiklik `PUT /api/session/zones` ile motora; motorun
  400 iletisi editörde gösterilir, yerel değişiklik geri alınır. Askıdaki bölgeler listede gri, "harita
  bağlantısı yok", yalnız silinebilir.
- **Alarm paneli:** açık alarmlar (tür, bölge, iz(ler), başlangıç karesi, süre), "Onayla" düğmesi; yasak
  bölge kırmızı, bekleme sınırı turuncu; onaylanmamış alarm sayısı üst çubukta rozet. Geçmiş: temizlenenler
  ("onaylandı" / "onaylanmadan kapandı").
- **Zaman çizelgesi:** dizinin kare ekseni; olay işaretleri türüne göre biçim + renk (renge dayanmaz);
  alarm aralıkları bant olarak; geçerli kare imleci; süzgeçler (tür, bölge, iz); işarete tıklama → olay
  ayrıntısı + anlık görüntü (§5.13) + "o kareye git" (atlama: `scope=tracks`, §5.11).
- **Olay listesi:** zaman çizelgesiyle aynı süzgeç; satırda kare, tür (Türkçe metin tablosu), bölge, iz, yön;
  tıklama → araç kartı.
- **Dışa aktarma:** "Olaylar (JSON)", "Olaylar (CSV)", "Harita (PNG: tamamı / görünen)" (§5.13).
- **Sağlık paneli:** donmuş rozeti; alarm listesine `rgb_frozen`, `ir_frozen`; ego durumu (kanal, nesil,
  geçici / kalıcı, iç nokta, son yeniden konumlama).
- **Arıza simülasyonu:** "kareyi dondur (RGB / IR)".
- **Oturum kuralları (Faz 3'ten değişen):** aynı dizide ileri atlama / geri adım → `reset?scope=tracks`
  (harita, bölgeler, günlük korunur); dizi değişimi ve "Oturumu sıfırla" düğmesi → `scope=all` (onay
  penceresi değil, düğmede "bölgeler ve olaylar silinir" uyarı metni). Tek istek, kare atlanmaz, internetsiz
  kuralları aynen. İstekler her karede `frame` (dizi kare numarası) taşır.
- **Next rotaları:** `/api/session/zones` (GET / PUT), `/api/session/events` (GET, `format`),
  `/api/session/alarms/ack` (POST), `/api/session/track` (GET), `/api/session/reset` (`scope`); hepsi motora
  iletir (tarayıcı motoru görmez, Faz 3 kuralı).

### 5.9 Fikstürler (yalnız M3OT val, CC BY 4.0; `fixtures/m3ot/NOTICE.md` güncellenir)

- **Uçtan uca video klibi:** `1_08`'den ardışık **60** kare, kodlanmış RGB JPG + IR PNG. Repoya değil,
  modeller gibi GitHub Release'e (`fixtures-f4`, sha256 `configs/engine/fixtures.json`'da; ≈ 30 MB, builder
  ölçer); CI ve `scripts/prepare` indirip denetler. Senaryolar (girdi baytları Python'da üretilip aynı
  pakette; iki taraf aynı dosyaları okur; her istek `frame` taşır):
  1. `clean`: temiz 60 kare (en az 3 anahtar kare değişimi; başlangıç sınır kuralıyla seçilir);
  2. `ir_drop`: 25-34. karelerde IR sıfır PNG → RGB geçici nesil; IR dönünce geçici IR nesli → **IR nesil
     1'e yeniden konumlama ve birleşme** (RGB geçici nesli kapanır);
  3. `ir_freeze`: 20-29. karelerde IR = 19. karenin baytları (kare bayrağı, alarm, kanal geçişi);
  4. `short_gap`: 40-44. karelerde iki kanal sıfır (`source: none`; ≤ 10 kare → LK bağlama);
  5. `long_gap`: 30-45. karelerde iki kanal sıfır (> 10 kare → geçici nesil → yeniden konumlama, birleşme);
  6. `seek`: 1-40, sonra `reset?scope=tracks`, sonra 10-60 (geri atlama → yeniden konumlama, nesil 1);
  7. `scene_change`: 21-60. kareler **yatay aynalanmış** (benzerlik dönüşümüyle eşleşemez) → yeniden
     konumlama 30 karede başarısız → **kalıcı yeni nesil**, bölgeler askıda;
  8. `reset_all`: 40. karede `reset?scope=all`.
- **Bölge senaryosu (yalnız C++ regresyonu):** `clean` ve `long_gap` + sabit bölge kümesi
  (`fixtures/motion/zones_clip.json`: alan, yasak, bekleme `dwell_frames` 10, sayım hattı; klip karelerine
  bakılarak val'de elle yerleştirilir) → beklenen olaylar `fixtures/expected/zones_clip.json` (ilk üretimde
  Claude ve kullanıcı olayları kare kare gözden geçirir, sonra sabitlenir).
- **Sentetik dönüşüm:** klipten 10 kare × 5 bilinen benzerlik dönüşümü (öteleme ±20 px, dönme ±1°, ölçek ±%3;
  sabit liste), Python `warpAffine` ile üretilip aynı Release paketinde (Kabul 3); ayrıca 10 kare × 3 büyük
  öteleme (60, 120, 200 px) yeniden konumlama için (Kabul 10).
- **Uzun geçmişli hareket birim eşliği:** mevcut `fixtures/tracks/m3ot_val_dets.json.gz` (repoda; CI
  `dev_dets.json.gz` adıyla) + Python'un tam çözünürlük karelerden ürettiği kare başına `frame_to_map`,
  `ego_ok`, `map_gen` (`fixtures/motion/maps_val.json.gz`, iki dizi × 600 kare, küçük, repoda). Hareket
  modülünü 600 karelik geçmişte, ego'dan bağımsız sınar.
- **Sınır kuralı:** Faz 3 §4.6 gibi üretici denetler; anahtar kare / bağlama / yeniden konumlama / nesil
  kararına giren ölçüler (iç nokta, oran, öteleme) eşiğinden en az 2 nokta / 0,02 / 1 px uzak olmalı;
  tutmayan senaryoda başlangıç karesi kaydırılır (yalnız val, yalnız eşlik için).

### 5.10 Faz 3'ten devreden açık bulgular (Faz 4'ün dokunduğu yollar)

Görev 0'da kapatılır; K4-K6, K8-K11 açık kalır:
- **S1** `runtime.cpp`: oturum çözümden önce açılıyor → ego / nesil durumu da bozuk ilk istekte kaybolur.
- **S3** çok küçük görüntüde yarı çözünürlük 0 boyut → 500 (ego aynı küçültmeyi kullanıyor).
- **S6** `baykus-track compare` (`track/cli.py` `_cmd_compare`) karar `fail` olsa da 0 döndürüyor. CI'daki
  takip eşliği **kapısı** bundan etkilenmiyor (`ctest` içindeki `Tracker.FixtureParity` her farkta düşer,
  `REQ=1`), ama kanıt adımı başarısız karşılaştırmayı yeşil gösterebilir ve Faz 4 aynı kalıpla yeni kanıt
  adımları ekliyor → karar `fail` ise ≠ 0.
- **S11** `ui/lib/engine.ts`: motor isteğinde zaman aşımı yok.
- **S12** `Console.tsx`: eski dizinin geç gelen yanıtı yeni oturuma yazılıyor → harita ve olaylara da yazılır.
- **K7** `system/report.py`: bench protokolü ve sürümler sabit metin; Faz 4 bench protokolünü değiştirdiği için
  JSON'dan okunur.

### 5.11 Harita belleği ve yeniden konumlama (motor `ego`, Python `baykus.motion.reloc`)

Amaç: kamera arızası, uzun kayıp ya da videoda atlama haritayı ve bölgeleri **kaybettirmesin**; ama yanlış
bir bağlama bölgeleri **sessizce kaydırmasın** (anti-hedef). Bu yüzden yeniden konumlama ölçütlü ve
doğrulanmamışsa bölgeler askıda kalır.

- **Kütüphane:** oturum ve kanal başına; her **anahtar karede** (§5.1 adım 4) bir kayıt:
  `{gen, M_k, frame (istekteki dizi kare numarası, yoksa null), n, ORB anahtar noktaları + tanımlayıcıları}`.
  ORB: yarı çözünürlükte 1000 nokta, ego maskesiyle aynı maske. En çok **300** kayıt (dolunca en eski çıkar).
  Geçici nesil kayıtları da girer (birleşince hedef nesle dönüşümle yazılır).
- **Ne zaman denenir:** geçici nesil açıldığında (§5.1 süreklilik kopması). Geçici neslin ilk karesinde ve
  sonra her **5.** `ok` karede, en çok **30** kare boyunca.
- **Aday kayıtlar:** aynı kanal, geçici nesil dışındaki nesiller. İstekte `frame` varsa `|Δframe| ≤ 300` olan
  en yakın **5** kayıt; yoksa en yeni 5 kayıt.
- **Eşleştirme:** geçerli kareden ORB (aynı parametreler, geçerli tespit maskesi) → her adayla Hamming,
  çapraz denetim; `estimateAffinePartial2D` RANSAC 1,5 px (yarı çözünürlük), 2000 yineleme, güven 0,995.
  Aday `ok` = iç nokta ≥ **40** ve oranı ≥ **0,4**; en çok iç noktalı `ok` aday seçilir (eşitlikte daha
  yakın `frame`, sonra daha yeni `n`).
- **Başarı (birleşme):** seçilen kaydın nesli `g`, `A` kayıt → geçerli kare. Geçerli karenin hedef nesildeki
  dönüşümü `M' = M_rec · A_full⁻¹`; geçici neslin koordinatından hedefe **birleştirme dönüşümü**
  `T = M' · M_prov⁻¹` (`M_prov` geçerli karenin geçici nesildeki dönüşümü). Yanıtta
  `ego.merge = {from: geçici, to: g, transform: T}`, `relocalized: true`; günlüğe `map_relocalized` +
  `map_merged`. Bundan sonra kareler `g` neslinde; geçici neslin hareket geçmişi `T` ile taşınır (§5.2);
  geçici nesil kapanır. `g`'nin bölgeleri yeniden aktif (§5.5).
- **Başarısızlık:** 30 karede başarı yoksa geçici nesil **kalıcı** olur (`provisional: false`, günlüğe
  `map_new_generation`); eski bölgeler askıda kalır; ekran "yeni harita — önceki bölgeler askıda" der.
  Operatör yeni haritada bölge çizebilir ya da askıdakileri silebilir. Sonraki bir kopmada eski nesillere
  yeniden bağlanma yine denenir (kütüphane korunur), yani askıdaki bölgeler sonradan geri gelebilir.
- **Geçici neslin içinde yeni kopma** (ör. RGB geçici nesli sürerken IR geri gelir): eski geçici nesil
  birleşmeden **kapanır** (kalıcı olur, bölgesi yoktur), yeni geçici nesil açılır ve onun için yeniden
  konumlama baştan başlar (adaylar bütün kalıcı nesiller).
- **Atlama:** ekran aynı dizide ileri / geri atlamada artık `scope=tracks` ister: takipçi, hareket, zaman
  kuralı penceresi ve donmuş durumu sıfırlanır; ego geçici nesle geçer ve kütüphanede `frame`'e en yakın
  kayıtlarla yeniden konumlanır. Haritanın henüz görmediği bir yere (kütüphanede yakın kayıt yok) ileri
  atlamada başarısızlık doğaldır: yeni nesil. Dizi değişiminde `scope=all`.
- **Parametreler** `configs/engine/motion.json` `reloc`: `orb_features 1000`, `candidates 5`,
  `max_dframe 300`, `window_frames 30`, `every 5`, `ransac_px 1.5`, `min_inliers 40`, `min_ratio 0.4`,
  `library_max 300`. Ölçümden önce sabit.
- **Maliyet:** ORB çıkarımı yalnız anahtar karelerde (≈ her 30 karede) ve deneme karelerinde; aşama
  `timing_ms.ego` içinde, deneme kareleri bench'te ayrıca raporlanır (§6.4).
- Python referansı (`baykus.motion.reloc`) ve eşlik (Kabul 1 senaryoları), doğruluk Kabul 10.

### 5.12 Araç özeti (motor, `engine/src/track_summary.cpp`)

İz başına (API kimliği): `first_frame`, `last_frame` (dizi kare numarası; istekte `frame` yoksa oturum
sayacı), `class` (gözlemlerdeki çoğunluk sınıfı; eşitlikte küçük `cls_id`), `motion` + `motion_since_frame`,
`segments` (durum değişimleri `[{state, from_frame, to_frame}]`, histerezisten sonraki kararlarla; en çok 200,
dolunca en eski iki bölüm birleştirilir), `stationary_frames` (toplam), `path_rel` (harita adımlarının toplamı
/ harita kutu boyu ortancası; "kutu boyu" birimi; nesiller arası birikmez), `zones`
(`[{zone_id, enter_frame, exit_frame | null}]`), `alive`. Bellek: yaşayan izler + bitmiş son 500 iz.
`GET /v1/session/track?session=&id=` ve `/v1/detect?...&focus=<id>` → yanıtta `focus_track`. Veri setinde
ölçülmez; C++ senaryo testleriyle (Kabul 11).

### 5.13 Olay günlüğü ve dışa aktarma

- **Günlük (motor):** oturum başına halka arabellek, en çok **20.000** kayıt (dolunca en eskiler çıkar;
  `dropped` sayacı). Kayıt: `{seq, n, frame, kind, zone_id, zone_name, zone_kind, track_id, dir, gen,
  alarm_id}` (`seq` oturumda artan). Türler: bölge olayları (§5.5), `alarm_start`, `alarm_ack`, `alarm_clear`,
  `map_provisional`, `map_relocalized`, `map_merged`, `map_new_generation`, `zones_updated`,
  `session_reset_tracks`. Her `/v1/detect` yanıtında o karenin `events[]`'i.
- `GET /v1/session/events?session=&since=<seq>&format=json|csv`: `since`'ten sonraki kayıtlar (artımlı
  okuma; başka bir istemci yalnız bunu çağırarak olayları alır). CSV: UTF-8, virgül, sabit başlık
  `seq,n,frame,kind,zone_id,zone_name,zone_kind,track_id,dir,gen,alarm_id`, alıntılama RFC 4180.
- **Ekrandan dışa aktarma (internetsiz, tarayıcıda Blob indirme):**
  - "Olayları indir (JSON)": `{meta: {sequence, session, created_at, model_sha256 (`/v1/model`),
    motion_sha256, fps_assumed, zones}, events: [...]}` — günlüğün tamamı motordan.
  - "Olayları indir (CSV)": motorun CSV'si aynen.
  - "Haritayı indir (PNG)": geçerli neslin tam kapsamı (en çok 4096 px kenar) ya da yalnız görünen alan; izler,
    bölgeler, sayaçlar, lejant ve "göreli harita" başlığıyla.
- **Anlık görüntü (ekran):** her bölge olayında ve alarm başlangıcında ekran, o karenin RGB ve IR
  görüntüsünden izin kutusu çevresinde 128 × 128 kırpar (bellekte PNG, en çok 500, en eski çıkar). Zaman
  çizelgesinde olaya tıklayınca gösterilir. Dışa aktarmaya girmez (yalnız JSON / CSV / PNG).

## 6. Değerlendirme ve kabul ölçütleri (ölçümden önce sabit)

Her sayı bir JSON'a bağlı; `docs/motion.md` bu dosyalardan üretilir (`baykus-system report` genişletmesi ya da
ayrı `baykus-motion report`; builder), elle düzenlenmez.

| Sonuç dosyası | İçerik |
|---|---|
| `results/motion/explore_val.json` | §4 keşfi (yeniden üretilmiş) |
| `results/motion/label_audit_{val,test}.json` | kör insan örneklemi (Kabul 4; test yalnız rapor) |
| `results/engine/parity_ego.json`, `parity_motion.json` | Kabul 1-3, 10 |
| `results/engine/zones_tests.json` (ctest özeti) | Kabul 7, 11 |
| `results/motion/m3ot_{dev,test}_{oracle,engine}.json` | hareketli / duran ölçümü |
| `results/motion/m3ot_{dev,test}_ego.json` | ego kalitesi, harita kayması, kararlılık, yeniden konumlama, donmuş kare |
| `results/bench/e3c_onnx_fp32_engine_laptop-docker_f4.json` | Kabul 5 (Faz 3 dosyası değişmez) |

### 6.1 Eşlik ve doğruluk (CI, her commit)
- **Kabul 1 — uçtan uca eşlik (asıl kapı):** klip senaryolarının (§5.9) hepsi iki **tam** pipeline'dan
  (kodlanmış baytlardan, oturumlu, `gate=advise`, takip açık, her istek `frame` ile) geçer. Her karede
  **aynı**: `ego.ok`, `source`, `keyframe`, `relinked`, `relocalized`, `provisional`, `map_gen`,
  `merge.from/to`, `health.*.frozen`, sağlık alarm durumları, `tracks[]` kimlikleri (Faz 3 takip eşliği zaten
  birebir), `motion`. `merge.transform` köşelerde ≤ 0,25 px. Toleranslı: `frame_to_map` ile tam çözünürlük kare köşelerinin dönüşüm farkı
  ≤ **0,25 px**; `map_xy` ≤ 0,25 px; `speed_rel` ≤ **2e-3** (köşe toleransının hıza yansıması: 0,25 / 15 / 10).
  - **Histerezisle serbest satırlar:** bir iz × karede iki taraftan birinin `v`'si 0,03 ya da 0,10'a `2e-3`'ten
    yakınsa o satır serbesttir. Serbest satırda kararlar ayrışırsa, o izin sonraki satırları da iki tarafın
    kararı yeniden aynı olana kadar serbesttir (histerezis ayrışmayı taşır; her iki taraf `v` bandı terk edince
    yeniden eşitlenir). Kanıtsız ayrışma (serbest olmayan satırda) ❌. Serbest satırlar sayılır; her senaryoda
    karşılaştırılan iz × kare oranı ≥ **%99**.
  - Anahtar kare / bağlama / yeniden konumlama / nesil kararı farklı çıkarsa ❌ (sınır kuralı bunu fikstürde dışarıda tutar);
    istisna yalnız karara giren ölçünün eşiğe `1e-6` (oran) / 0 (tam sayı) içinde olduğu mekanik kanıtla.
- **Kabul 2 — hareket birim eşliği:** `maps_val` + takip fikstüründe (ego Python'dan sabit) her iz × karede
  `motion` aynı, `speed_rel` ≤ 1e-6, `map_xy` ≤ 1e-4 px; serbest satır kuralı aynı (`1e-6` payıyla). Uzun
  geçmişte (600 kare) tutma, histerezis ve nesil başı davranışını sınar.
- **Kabul 3 — doğruluk (sentetik):** bilinen dönüşümlerin **%95**'inde geri kestirilen dönüşümün kare
  köşelerindeki hatası ≤ **0,5 px** (tam çözünürlük); iki tarafta aynı dosyalardan. Donmuş kare: sentetik
  tekrar karede `frozen_frame=true`, ardışık iki farklı val karesinde `false` (birim test, iki tarafta).

### 6.2 Etiketler
- **Kabul 4 — kör insan örneklemi (val):** §5.3 ölçütü (a)-(c). Hareket ölçümünün (§6.3) ön koşulu.

### 6.3 Hareketli / duran ölçümü (val geliştirme, test dondurmadan sonra bir kez)

İki satır (ADR 0008 düzeni):
- **(a) oracle:** GT kutuları + GT iz numaraları takip çıktısı yerine; ego maskesi GT kutuları; ürünün ego,
  nesil ve hareket modülü (Python referansı = motor, Kabul 1-2). Ego + karar kuralını tespitten ayırır.
- **(b) motor:** C++ motorun dizi koşucusu (builder: `baykus_detect`'e oturum kipi ya da yeni
  `baykus_sequence`), `gate=advise`, takip açık, `tracks[]`. Her karede motor izleri (kimliği olan) GT
  kutularıyla IoU ≥ 0,5 Macar eşleşmesi.

**Paydalar (sabit):**
- **Etiketli nüfus** = GT örnekleri (iz, kare) arasında referans etiketi hareketli ya da duran olanlar
  (dışlanan nüfus: bant, kayıtsız, kenar; sayıları ve oranları ayrı raporlanır).
- Her etiketli GT örneği için sonuç: `moving` / `stationary` / `unknown` / **`kaçırıldı`** (motor: eşleşen
  kimlikli iz yok; oracle'da olmaz).
- **Geri çağırma** `R_mov` = (GT hareketli ∧ tahmin `moving`) / GT hareketli; `R_sta` aynı biçimde. `unknown`
  ve `kaçırıldı` paydada, payda değil.
- **Dengeli doğruluk (birincil)** `BA = (R_mov + R_sta) / 2`, bütün etiketli nüfus üzerinden.
- **Kapsam** = karar verilen (`moving` ya da `stationary`) / etiketli; `BA_karar` yalnız karar verilenlerde
  (ikincil).
- **Kesinlik** (motor): `P_mov` = GT hareketli bir örnekle eşleşen `moving` çıktıları / **bütün** `moving`
  çıktıları (eşleşmeyen ya da dışlanan GT'yle eşleşen `moving` çıktıları ayrı sayılır; eşleşmeyen = yanlış).
  `P_sta` aynı biçimde. Yanlış `moving` çıktılarının kare başına sayısı ayrıca.
- Dışlanan nüfusta ürün çıktılarının dağılımı (bilgi).
- Aynı ölçüler **kararlı bölümde** (GT etiketinin değiştiği karelerin ±`K` çevresi hariç).
- **Karar gecikmesi:** GT etiketinin değiştiği her geçişte, ürünün yeni duruma geçtiği ilk kareye kadar kare
  sayısı (ortanca, p90; geçmeyen geçiş sayısı).
- Belirsizlik: iz düzeyinde bootstrap %95 GA (1000, seed 42). Yalnız dizi içi örnekleme belirsizliğini
  gösterir; iki dizinin ortak kamera / sahne etkisine karşı **genelleme güvencesi değildir** ve öyle yazılır.
- **İnsan örneklemiyle doğrudan karşılaştırma:** audit örneklerinde (val ve test) ürünün (oracle ve motor)
  kararı insanın cevabıyla; etiketleyiciden tamamen bağımsız, küçük örneklem, Wilson GA.

**Ego kalitesi:**
- `ok` kare oranı, anahtar kare ve nesil sayısı, bağlama başarı oranı, iç nokta ortancası.
- **Yeniden konumlama (val, yalnız rapor):** her val dizisinde 100 karede bir benzetilmiş atlama (geri 50
  kare, ileri 50 kare, `scope=tracks`); başarı oranı, deneme sayısı, yeniden konumlanan `frame_to_map`'in
  kesintisiz koşudaki aynı kareye göre köşe farkı (ortanca, p95) ve yanlış bağlama sayısı (fark > 10 px).
- **Harita kayması (etiketlerden bağımsız):** aynı nesildeki `t` ve `t + Δ` kareleri için (Δ ∈ {15, 60, 150},
  `t` her 10 karede), ürünün `P = M_{t+Δ}⁻¹ · M_t` dönüşümü ile referansın **doğrudan** SIFT + homografi
  kaydının (GT kutuları maskeli, arka plan) dört köşe + merkezdeki farkı, px; ortanca ve p95. Hem birikimli
  kaymayı hem benzerlik modelinin hatasını ölçer (§3 homografi satırı).
- **Duran araç kararlılığı:** GT'de kesintisiz en az 30 kare duran **bölümler** (iz başına ayrı ayrı); her
  bölümde `map_xy`'nin standart sapması / bölümdeki harita kutu boyu ortancası (ikisi de harita biriminde);
  bölümlerin ortancası. Aynı aracın farklı yerlerdeki duruşları ayrı bölümdür.

**Donmuş kare:** temizde kanal başına `frozen_frame` ve alarm oranı; `freeze@K` gecikmesi.

**Oracle işlev kapısı (Kabul 9, val, çıkışı engeller):** oracle satırında `BA ≥ 0,85` ve `R_mov ≥ 0,80`,
`R_sta ≥ 0,80` (birleşik val). Eşikler val'e bakılarak seçildiği ve GT kutuları kullanıldığı için bu bir
**kalite iddiası değil işlev kapısıdır**: ego, nesil ya da karar kuralı bozuksa (ör. her şeye `unknown`)
burada düşer. Tutmazsa düzeltilir ya da faz yalnız kullanıcının ADR'siyle kapanır.

**Demo ölçütü (ölçümden önce; val, motor satırı):** demo dizisi Faz 3 kuralıyla `1_08`; motor satırında
`BA ≥ 0,70` ve `P_mov ≥ 0,80`. Tutmazsa `2_08`; hiçbiri tutmazsa demo yine yapılır ama §6.6 iddia kuralı
uygulanır. Test'te yalnız raporlanır.

**Test bölümü:** M3OT test dizileri (`1_03`, `2_03`) Faz 3'te tespit, takip ve sağlık için bir kez ölçüldü
(`f3-freeze`); hareket görevi için hiç kullanılmadı, test etiketleri hiç üretilmedi. Faz 4'ün hiçbir kararı
test sonuçlarına dayanmaz (v1'deki tek test atfı val sayısıyla değiştirildi, §3). Test sonucu "M3OT test
bölümünde, hareket görevi için ilk ölçüm; aynı bölüm Faz 3'te başka görevlerde kullanıldı" diye yazılır.
Ürün kodu, eşikler ve etiketleyici **dondurulduktan sonra** (`f4-freeze`) bir kez; ADR 0009 kayda alır.
Ek dokunulmamış bölüm yok (kullanıcı kararı §10/1 = A).

### 6.4 Gecikme
- **Kabul 5:** motor konteynerinde, laptop prizde, `gate=advise`, takip + ego + hareket + donmuş + 4 bölge
  (her türden biri) açık, tek oturum. Girdi **val `1_08`'in ardışık 550 karesi** (bellekteki kodlanmış baytlar; ≈ 150 MB): ilk 50 ısınma,
  sonraki **500 ölçüm**, döngü yok, sıfırlama yok (H5 sayıları, gerçek video). Motor p95 (HTTP hariç, bütün
  aşamaların **kare başına toplamının** p95'i) ≤ **100 ms**.
- Ayrıca raporlanır (eşik yok): aşama başına p50 / p95; anahtar kare ve bağlama karelerinin ayrı p95'i;
  kanal geçişi (IR kaybı) ve atlama (her 100 karede `scope=tracks`) senaryolarının p95'i ve yeniden konumlama
  deneme karelerinin ayrı p95'i; bölgeler açıkken (16 bölge) `zones` aşaması; **aynı girdide
  `ego.enabled=false`** ile Faz 3 yolu (karşılaştırma
  aynı girdiyle; Faz 3'ün 20 görüntülük sonucu ayrı protokol olarak kalır).
- *Tahmin (ölçülmedi; aşama p95'leri toplanmaz):* ego yarı çözünürlükte birkaç ms ekler; Faz 3 p95'i 82,2 ms
  (20 görüntülük girdi). Yeni girdide pay bilinmiyor; ilk ölçüm C++ ego görevinin sonunda yapılır (erken uyarı).
- Bütçe aşılırsa: yalnız ölçülmüş bir değişiklikle (ör. köşe sayısı, LK seviyesi) ve kullanıcının kararıyla;
  "her N karede bir ego" seçeneği bu tasarımda tanımlı değildir (atlanan karelerin `ego_ok`, hız ve olay
  sayaçlarına etkisi yeni tasarım ister).

### 6.5 Ürün davranışı: ekran, bölgeler, harita belleği, özet, günlük
- **Kabul 6 — ekran (elle, demo dizisinde; ekran görüntüsü / GIF ile `results/system/manual.json` Faz 4
  bölümü):** her madde ayrı onaylanır:
  1. kutu durum biçimleri (H / D, kesikli), "yalnız hareketliler" süzgeci, liste durum sütunu;
  2. harita: mozaik oluşuyor, görüş alanı, kuyruklar, kaydır / yakınlaştır / sığdır / kamerayı izle;
  3. araç seçimi (kutudan ve haritadan), araç kartı alanları, "haritada göster";
  4. bölge editörü: dört türü çiz, köşe düzenle, ad değiştir, bekleme saniyesi, hat yönü, sil; motorun 400
     iletisi (kendini kesen çokgen) editörde;
  5. alan olayları ve sayaçlar; yasak bölgeye girişte kırmızı alarm → onayla → çıkınca geçmişe;
     bekleme sınırı aşımında turuncu alarm; sayım hattında iki yön sayacı;
  6. güven kaydırıcısı 1,0'da olaylar ve sayaçlar değişmiyor;
  7. IR'yi kapat → "konum aranıyor" → IR geri gelince aynı harita ve bölgeler aktif (birleşme);
  8. geri atla → harita ve bölgeler korunuyor, yeniden konumlama; dizi değiştir → her şey sıfır;
  9. zaman çizelgesi: işaretler, süzgeçler, anlık görüntü, "o kareye git";
  10. dışa aktarma: JSON, CSV (bir hesap tablosunda açılıyor), harita PNG (tamamı / görünen);
  11. donmuş rozeti + alarm + "kareyi dondur"; ego durumu sağlık panelinde;
  12. Wi-Fi kapalıyken 1-11 (internetsiz).
- **Kabul 7 — bölgeler ve olaylar (CI, C++ GoogleTest `zones_test.cpp`):** her tür için senaryolar: 2 gözlem
  → olay yok, 3 → olay (kare = ilk gözlem); ≤ 5 `yok` art arda sayılır, 6 `yok` sayacı sıfırlar; kenar üstü
  nokta içerde; içbükey çokgen; `exit_lost` 30 kare; `stopped` kalış başına bir kez, `unknown` / `moving`
  sıfırlar; yasak bölge alarmı çoklu izle açık kalır, son iz çıkınca temizlenir; bekleme alarmı tam
  `dwell_frames`'te açılır, 3 `moving`'le temizlenir; sayım hattı: iki yön, uzantıdan dolanma sayılmaz, uç
  noktadan geçiş sayılır, tek kare titreme sayılmaz, tekrar geçiş ayrı olay, `count` süzgeci; askıya alma ve
  birleşmeyle yeniden aktifleşme; `scope=tracks`'te bölgeler ve günlük korunur, kimlik tabanı artar;
  `scope=all` siler; doğrulama (16 bölge, 32 nokta, kendini kesen çokgen, eşsiz `id`, `gen`, ad uzunluğu)
  hiçbir bölgeyi değiştirmeden 400; alarm onayı. Ayrıca **klip regresyonu** (§5.9 bölge senaryosu): beklenen
  olay listesi birebir.
- **Kabul 10 — yeniden konumlama doğruluğu:** (a) sentetik büyük ötelemelerde (§5.9; 60 / 120 / 200 px)
  yeniden konumlanan dönüşümün kare köşelerindeki hatası ≤ **1,0 px** (%95'i), iki tarafta; (b) `seek` ve
  `long_gap` senaryolarında yeniden konumlanan karelerin `frame_to_map`'i aynı karelerin `clean` koşusundaki
  değerinden köşelerde ≤ **2,0 px**; (c) `scene_change`'de birleşme **olmaz** (yanlış bağlama yok) ve
  bölgeler askıya alınır. CI'da (Release fikstürüyle).
- **Kabul 11 — araç özeti ve günlük (CI, C++):** özet senaryoları (çoğunluk sınıfı, durum bölümleri
  histerezisten sonra, 200 bölüm sınırında birleştirme, `path_rel`, bölge ziyaretleri, bitmiş 500 iz
  sınırı); günlük (artan `seq`, `since` ile artımlı okuma, 20.000 sınırında `dropped`, CSV başlığı ve RFC 4180
  alıntılama — virgül / tırnak / Türkçe karakter içeren bölge adı).
- **Kabul 8 — gerileme ve API sağlamlığı:** Faz 3 Kabul 1-4, 6, 8 CI'da yeşil (takip fikstürü ve
  `Tracker.FixtureParity` değişmeden); `compose.offline.yml` kanıtları yeni imajlarla; Faz 3 API sağlamlık
  testi yeni uçlarla genişler (bozuk JSON, > 64 KB gövde, bilinmeyen oturum 404, sınır dışı değerler; ardından
  1000 rastgele istek ve `/health` ok); S6 düzeltmesiyle kanıt adımları başarısızlıkta düşer.

### 6.6 Faz çıkışı ve iddia kuralı

| Ölçüt | Etkisi |
|---|---|
| Kabul 1, 2, 3 (eşlik, sentetik doğruluk) | **çıkışı engeller** |
| Kabul 4 (kör insan örneklemi, val) | **çıkışı engeller**; iki kez tutmazsa yalnız kullanıcının ADR'siyle B planına iner |
| Kabul 5 (gecikme) | **çıkışı engeller**; aşılırsa ADR'yle kapanır, README'de olumsuz sonuç |
| Kabul 6, 7, 8, 10, 11 (ürün davranışı) | **çıkışı engeller** |
| Kabul 9 (oracle işlev kapısı, val) | **çıkışı engeller**; tutmazsa ADR |
| Motor satırı, test sonuçları, ego kalitesi, harita kayması, kararlılık, yeniden konumlama val raporu, donmuş kare, demo ölçütü, test audit'i | **yalnız raporlanır** |

**İddia kuralı (ölçümden önce):** README, `docs/motion.md` ve demo metni "hareketli / duran ayrımı çalışıyor"
cümlesini yalnız şu koşulda kullanır: test oracle `BA ≥ 0,85`, test motor `BA ≥ 0,70` ve `P_mov ≥ 0,80`,
test audit'inde (a) tutmuş. Tutmayan her koşul olduğu gibi yazılır ("ego ve kural çalışıyor ama tespit
yetersiz" gibi); olumsuz sonuç da sonuçtur (Faz 2 düzeni). Ürün özellikleri (bölgeler, alarmlar, harita
belleği, dışa aktarma) "çalışıyor" diye yazılabilmesi için Kabul 6, 7, 10, 11 yeterli; olayların gerçek
dünyadaki doğruluğu iddia edilmez (§3).

Çıkış: S11 işaretli; `docs/motion.md`; README'ye Faz 4 bölümü, ekran görüntüleri ve kısa GIF; ADR 0009;
spec v4.6; `main`'e birleştirme ve `v0.3.0` (kullanıcı "at" derse push).

## 7. İş yükü ve takvim

Bütün kodu Claude yazar (Faz 3 kararı 4 sürüyor); kullanıcı adımları: kör örneklem (val ≈ 20-30 dk, test
≈ 20-30 dk), klip bölge olaylarının gözden geçirilmesi (≈ 15 dk), GitHub Release `fixtures-f4` yüklemesi,
Docker Desktop açık, ekran kontrolü (Kabul 6, ≈ 1 saat), onaylar. **GPU yok; Colab yok** (§10 karar 1 = A).
Val ve test dizileri laptopta (`data/m3ot_dev` 346 MB, `data/m3ot_test` 309 MB).

| # | İş | Tahmin |
|---|---|---|
| 0 | Faz 3 devri: S1, S3, S6, S11, S12, K7 (§5.10) | 1 gün |
| 1 | Referans etiketleyici, keşif, kör örneklem → **KAPI: kullanıcı val örneklemi** (Kabul 4) → etiketleyici dondurulur | 2-3 gün |
| 2 | Python ego + nesil + **harita belleği / yeniden konumlama** + hareket + donmuş + oturumlu referans (TDD) · `motion.json` · oracle satırı val + Kabul 9 · harita kayması, yeniden konumlama val raporu · fikstür üretici (8 senaryo, Release) | 6-7 gün |
| 3 | Docker `deps`: OpenCV `video`, `calib3d`, `features2d`, `flann` · C++ ego + nesil + yeniden konumlama · Kabul 3, 10 · erken bench | 4-5 gün |
| 4 | C++ hareket + donmuş, Runtime, `tracks[]`, kimlik tabanı, `scope`, dizi ve klip koşucuları · Kabul 1, 2 · motor satırı val · demo ölçütü | 4-5 gün |
| 5 | C++ bölgeler, kurallar, alarmlar, günlük, araç özeti, yeni API uçları · Kabul 7, 11, 8 (API) · klip regresyonu (kullanıcıyla gözden geçirme) | 5-6 gün |
| 6 | Bench (550 ardışık kare, `ego.enabled=false`, IR kaybı ve atlama satırları) + Kabul 5 | 1 gün |
| 7 | Ekran I: tipler, kutu durumları, liste, sağlık / ego, donmuş alarmı ve arıza, Next rotaları, oturum kuralları (`scope`) | 3-4 gün |
| 8 | Ekran II: harita (mozaik, nesil katmanları, birleşme, kaydır / yakınlaştır / izle), araç kartı | 4-5 gün |
| 9 | Ekran III: bölge editörü, alarm paneli, zaman çizelgesi + olay listesi + anlık görüntü, dışa aktarma | 4-5 gün |
| 10 | CI (Release fikstürü, yeni kanıt adımları, UI testleri), kapalı ağ kanıtları · Kabul 8 · Kabul 6 (elle) | 2-3 gün |
| 11 | **Dondur → test bir kez:** test etiketleri, oracle + motor satırı, ego kalitesi, kullanıcı test örneklemi · iddia kuralı · ADR 0009 · `docs/motion.md`, README, spec v4.6, `v0.3.0` | 2-3 gün |

**Toplam ≈ 8-10 hafta** (v2'de 5-6 hafta; ürün paketleri ≈ +3-4 hafta). Spec penceresi 8 hafta (1 Şub –
28 Mar 2027); takvimin ≈ 3 ay önündeyiz, Faz 4 erken başlayıp penceresi içinde biter. En riskli adımlar: 1
(etiketler), 2-3 (yeniden konumlama ve nesil birleşmesi: durum makinesi + eşlik), 5 (bölge durum makinesi
ayrıntıları), 8 (harita etkileşimi ve birleşmede katman dönüşümü). **Kesme yok (kullanıcı kararı,
2026-10-20):** "Zaman yeter, bunların hepsinin olmasını istiyorum." Bu tasarımdaki bütün özellikler faz
çıkışının parçası; süre aşılırsa iş uzar, özellik düşmez. (v3 taslağındaki kesme sırası — anlık görüntüler →
harita PNG → zaman çizelgesi → sayım hattı → bekleme sınırı → kamerayı izle — bu kararla kaldırıldı.)

## 8. Riskler

| Risk | Önlem |
|---|---|
| Etiket ile ürün aynı hatayı paylaşır (döngüsel) | Farklı özellik, model, pencere ve maske (§5.3 tablo); telafisiz, etiketleyiciden bağımsız seçilmiş kör insan örneklemi; oracle satırı; harita kayması referans kayda karşı. |
| Etiketleyicinin dışladığı nüfus (bant, kayıtsız) yüksek doğruluğu "kolay örnekler"e kaydırır | Dışlanan nüfus ve ürünün oradaki çıktısı raporlanır; Kabul 4 (c) dışlamanın insan-kararlı örnekleri %20'den fazla dışarıda bırakmamasını ister; hedefli 40 örnek. |
| GT kutu titremesi (duranlarda `v ≈ 0,01-0,02`) duranları banda iter | Keşifte duranların büyük kısmı ≤ 0,03; dışlanan oran raporlanır; eşik ayarlanmaz. |
| Duran-kalkan araçlar (kavşak) bant içinde uzun kalır | Ürün histerezisle önceki kararı tutar; "kararlı bölüm" ve karar gecikmesi ayrıca raporlanır. |
| Ego başarısız (dokusuz alan, gece RGB, su) | IR öncelikli; kayıp / bağlama / nesil sözleşmesi; `ok` oranı ve nesil sayısı raporlanır. |
| Kanal değişimi sık olur, harita sürekli yeniden başlar | IR'ye dönüş 5 temiz kareyle; nesil sayısı raporlanır; temiz M3OT'ta IR kaybı beklenmez (arıza simülasyonu dışında). |
| Anahtar kareli zincirde de kayma birikir; harita ve bölgeler kayar | Harita kayması Δ = 150'ye kadar ölçülür ve raporlanır; harita "göreli"; bölge olaylarının gerçek doğruluğu iddia edilmez (§3). |
| Yeniden konumlama yanlış yere bağlar → bölgeler sessizce kayar (anti-hedef) | Ölçütlü kabul (iç nokta ≥ 40, oran ≥ 0,4, aynı kanal, aday sınırı); `scene_change` senaryosunda birleşme olmaması Kabul 10 (c); val'de yanlış bağlama sayısı raporlanır; başarısızlıkta bölgeler askıda. |
| Geçici nesil / birleşme durum makinesi karmaşık; C++ ↔ Python ayrışır | 8 klip senaryosu uçtan uca eşlikte (Kabul 1); birim testler her geçiş için; tampon. |
| Bölge kurallarında uç durumlar (titreme, hattın ucundan geçiş, uzun kesinti) | Kabul 7 senaryo listesi; klip regresyonu kullanıcıyla gözden geçirilip sabitlenir. |
| Ürün kapsamı büyür (ek bölge türü, kalıcılık, çoklu kullanıcı) | §3 listesi; bu tasarım dışındaki yeni istek Faz 5'e (kesme yok, ama ekleme de yok). |
| Ekran karmaşıklaşır, operatör kaybolur | Sekmeli sağ sütun, tek seçili iz, sabit Türkçe metin tablosu; Kabul 6 maddeleri tek tek elle denetlenir. |
| Yaklaşma / irtifa değişimi hız eşiğini kaydırır | Kutu boyuyla normalleştirme; pay ve payda aynı koordinatta (ürün harita, referans `t + K`). |
| Kare hızı farklı veride eşikler geçersiz | Birim kare başına; başka kare hızında yeniden ölçülmeden kullanılmaz (§4). |
| LK / köşe seçimi CI işlemcisinde farklı (SIMD) | Aynı OpenCV 5.0.0; tolerans + sınır kuralı + histerezis serbest satır kuralı + mekanik kanıtlı istisna (Kabul 1). |
| Gecikme 100 ms'yi aşar | C++ ego görevinde erken bench; aşama kırılımı (`ego`, `motion`, `zones`, yeniden konumlama kareleri ayrı); ölçülmüş değişiklik ya da ADR. |
| Motor satırı val'de düşük (yoğun sahne, AP50 0,40) | Oracle satırı yöntemi tespitten ayırır; motor satırı yalnız rapor; iddia kuralı. |
| Test yalnız iki dizi | Bootstrap yalnız dizi içi; genelleme iddiası yok; ek bölüm yok (§10/1 = A, kullanıcı kararı). |
| Kullanıcı örnekleminde yorulma / yanlılık | 120 örnek, karışık sıra, etiket gizli, "emin değilim" seçeneği; test örneklemi ayrı günde yapılabilir. |

## 9. Builder'a bırakılanlar (varsayılanlarıyla; plana yazılır)

- OpenCV ek modüllerinin CMake bayrakları (varsayılan: yalnız gerekenler; wheel bilgisiyle `opencv_build.json`).
- Dizi koşucusunun biçimi (`baykus_detect` oturum kipi ya da yeni uygulama).
- Release fikstür paketinin biçimi (`.tar` + sha256) ve `scripts/prepare` / CI indirme adımı (model gibi).
- Renkler, ikonlar, panel boyutları (düzen §5.8'de); harita küçük resim seyreltme ayrıntısı; editörde köşe
  tutamaç boyutu.
- Basit çokgen denetiminin algoritması (varsayılan: komşu olmayan kenar çiftlerinde parça kesişimi, O(n²), n ≤ 32).
- Kör örneklem sayfasının görünümü (oynatma hızı, büyütme); cevap JSON biçimi.
- `docs/motion.md` üreticisi (`baykus-system report` genişletmesi ya da ayrı komut).
- Klip uzunluğu (60; senaryolar sığmazsa 75) ve senaryo kare aralıklarının sınır kuralına göre kaydırılması.

## 10. Kararlar (inceleme için)

**Kullanıcı kararı (2026-10-20): A** — test yalnız `1_03`, `2_03`; ek dokunulmamış bölüm yok, Colab yok.

1. **Test için ek dokunulmamış bölüm.** Codex: mevcut test bölümü Faz 3'te kullanıldı, "temiz test" denmemeli.
   Doğrulama: Faz 4'ün hiçbir kararı test sonucuna dayanmıyor (v1'deki tek atıf düzeltildi), hareket görevi
   testte hiç ölçülmedi; aynı test bölümünün başka bir görevde yeniden kullanılması standart ve dürüstçe
   yazılabilir (§6.3). Ama yalnız 2 dizi var ve genelleme zayıf.
   - **A (varsayılan, yalın):** test = `1_03`, `2_03`, §6.3'teki ifadeyle.
   - **B:** A + M3OT **train** dizileri (14 dizi; model DroneVehicle'da eğitildi, M3OT train hiçbir ölçümde
     kullanılmadı) yalnız **oracle satırı** için ek dokunulmamış bölüm. Bir Colab CPU oturumu (figshare
     indirmesi + dönüşüm + yalnız Python ego / etiket; laptopa iniş yok), ≈ 1-2 saat kullanıcı süresi.
     Ego + kuralın 14 dizide genellemesini ölçer; motor satırı yine 2 test dizisinde.

**Kullanıcı kararı (2026-10-20, v3):** ürün paketleri A (harita + araç kartı), B (bölge kuralları), C (olay
kaydı ve dışa aktarma), D (olaylar motorda) — dördü de girer. Bunun sonucu Claude'un kararları:

- **Harita belleği ve yeniden konumlama** (§5.11) eklendi: bölgeler motorda ve oturum boyunca kalıcı
  olacaksa kamera arızası ve atlama haritayı kaybettirmemeli; v2'de her kopmada bölgeler siliniyordu.
- **Atlama** artık `scope=tracks` (harita, bölgeler, günlük korunur); Faz 3 kuralı (atlama = tam sıfırlama)
  bu yüzden değişti.
- API `track_id`'si oturumda benzersiz ve artan (taban), günlükte kimlik çakışmasın.
- Bölge mantığı, araç özeti ve günlük için Python referansı yok: veri setinde karşılıkları yok, ölçülmüyor;
  C++ senaryo testleri + klip regresyonu (Kabul 7, 11).
- Bölge türleri 4 ile sınırlı (alan, yasak, bekleme, sayım hattı); en çok 16 bölge.
- Kalıcılık yerine dışa aktarma; oturumlar arası saklama yok.

**Claude'un kararları (inceleme sonrası hâli):**

2. Eşikler (0,03 / 0,10, `K` = 15) val keşfinden; val sonuçları "keşif". (Codex: makul.)
3. Kör insan örneklemini kullanıcı yapar: val (kapı) + test (rapor), her biri ≈ 20-30 dk; örneklem
   etiketleyiciden bağımsız seçilir (v1'de etiketleyicinin çıktısından seçiliyordu — Codex haklı, düzeltildi).
4. ~~Bölge olayları ekranda~~ → **kullanıcı kararıyla (v3, paket D) motorda** (§5.5); `tracks[]` + `map_gen` +
   `yok` gözlemi sözleşmesi aynen motora taşındı.
5. Ürün yolunda benzerlik dönüşümü; model hatası harita kaymasıyla val'de ölçülür.
6. Takipçiye kamera telafisi yok — kapsam tercihi, kazanç iddiası yok.
7. Donmuş kare: kare bayrağı çekirdek (ego), alarm ve arıza simülasyonu kesilebilir.
8. Faz 3 bulgularından S1, S3, S6, S11, S12, K7 Görev 0'da.
9. Bench: val'den 550 ardışık kare, döngüsüz; aynı girdide `ego.enabled=false` satırı.
10. Oracle işlev kapısı (Kabul 9) çıkışı engeller; motor satırı yalnız rapor + iddia kuralı (§6.6).
11. Hız hiçbir yerde fiziksel birimle gösterilmez; harita "göreli" etiketli.

## 11. Spec'e yansıyacaklar (onaydan sonra, v4.6)

§2.3 S11 metni (ölçütler §6; ürün özellikleri: bölge kuralları, alarmlar, harita belleği, dışa aktarma),
§6 Faz 4 maddesi (donmuş kare ve ürün paketleri eklenir, takipçi telafisi ve kanallar arası bağlama kapsam
dışı), §7.4 FR-E5 API alanları ve yeni uçlar (§5.7), FR-U eklemeleri (§5.8), §15.1 modül listesi (`ego`,
`motion`, `frozen`, `zones`, `track_summary`, `event_log`), §14 bench protokolü (Faz 4 satırı), §16 not
(takipçi değişmedi; API kimlik tabanı), §17 düzen (§5.8; atlama `scope=tracks`), §18 OpenCV modülleri ve
Release fikstürü, §22 Faz 4 satırı (≈ 8-10 hafta), §23 son risk satırının güncellenmesi; ADR 0009 numarası; `docs/health.md` donmuş kare
sınırlamasının kapanması.

## 12. İnceleme kaydı (Codex, 2026-10-20; her madde kod ve belgeyle doğrulandı)

| # | Bulgu (özet) | Doğrulama | Karar |
|---|---|---|---|
| 1 | Test sonucu tasarım gerekçesinde; test "temiz" değil | v1 §3'te takipçi kararı **test** HOTA'sıyla gerekçelendirilmişti (doğru). Faz 4'ün başka kararı teste dayanmıyor; hareket görevi testte hiç ölçülmedi. | **Kısmen kabul:** atıf val sayısıyla değişti; test ifadesi §6.3'te dürüstçe; ek bölüm kullanıcı kararı (§10/1). "Mevcut test artık kullanılamaz" kısmı kabul edilmedi: başka görevde bir kez ölçülmüş bölümün yeni görevde kullanılması, kararlar ona dayanmadıkça sızıntı değil. |
| 2 | Etiketler bağımsız değil; audit etiketleyiciden örnekleniyor | Audit örneklemi gerçekten etiketleyicinin hareketli / duran katmanlarından seçiliyordu (doğru). "Aynı eşikler" itirazı: eşik görevin tanımı, döngü kestirim hatasından gelir. | **Kabul** (eşik kısmı hariç): örneklem nüfustan bağımsız, hedefli dışlanan örnekler, Kabul 4 (c), test örneklemi, ürün ↔ insan doğrudan karşılaştırma. |
| 3 | Kanal değişimi / kayıp sonrası harita bağlantısı tanımsız | Doğru: v1 kanal değişiminde yeni anahtar kare açıyor ama eskiye bağlamıyordu; `ok` olmayan kare anahtar oluyordu (kayıp hareket haritadan düşerdi). | **Kabul:** harita nesli sözleşmesi, kısa kayıpta yeniden bağlama, `ok` olmayan kare anahtar olamaz (§5.1). |
| 4 | Bölge olayları conf süzgeçli listeye bağlı | Doğru: `runtime.cpp` `r.dets = filter_display(fr.dets_all, req.conf)` ve kimlikler yalnız bu listeye yazılıyor. | **Kabul:** süzgeçsiz `tracks[]` (§5.7); olaylar ondan. |
| 5 | Metrikler başarısız ürünü ayırmıyor; paydalar belirsiz | Doğru: eşleşmeyen tahminler cezasızdı, `unknown` paydada belirsizdi, kalite kapısı yoktu. | **Kısmen kabul:** sabit paydalar, geri çağırma, kesinlik, oracle işlev kapısı (Kabul 9), iddia kuralı. Motor satırı için çıkış kapısı **kabul edilmedi**: tespit kalitesi Faz 4'te değişmiyor; proje düzeni olumsuz araştırma sonucunu raporlamak (Faz 2). |
| 6 | Eşlik testleri gerçek C++ ego → hareket zincirini sınamıyor | Doğru: v1'de ego hazır küçültülmüş girdiden, hareket Python ego çıktısından besleniyordu. | **Kabul:** uçtan uca klip, 6 senaryo (Kabul 1); birim testler tanı için kalır. |
| 7 | CI kapısı gerçek kapı değil; S6 ertelenemez | Kısmen yanlış: CI'daki kapı `ctest` içindeki `Tracker.FixtureParity` (`REQ=1`), her farkta düşer; `baykus-track compare` yalnız kanıt dosyasını yazıyor. Ama `compare`'in 0 döndürmesi gerçek ve Faz 4 aynı kalıpla kanıt adımları ekliyor. K7 de bench protokolü değiştiği için yerinde. | **Kabul (gerekçe düzeltilerek):** S6 ve K7 Görev 0'da (§5.10). |
| 8 | 20 karelik döngü güvenilir gecikme kabulü değil | Doğru: döngü başındaki sıçrama gerçek video davranışı değil; 600 karelik val dizisi 50 + 500 için yeter. | **Kabul:** 550 ardışık kare, döngüsüz; anahtar kare ve kanal geçişi ayrı; aynı girdide Faz 3 yolu. |
| 9 | Gecikme tahmini karışık; karar gecikmesi ayrı; "her 2 karede" B planı tanımsız | Doğru: aşama p95'leri toplanmaz; varsayılan oynatma 10 kare/sn (`ui/lib/player.ts` `DEFAULT_FPS = 10`). | **Kabul:** tahmin metni düzeltildi, karar gecikmesi kare cinsinden raporlanıyor, B planı kaldırıldı. |
| 10 | Referans hız formülünde koordinatlar karışık; "kare hızından bağımsız" yanlış | Doğru: pay `t + K`, payda `t` karesindeydi (≈ %3 sistematik); birim cümlesi yanlıştı. | **Kabul:** payda `t + K`; birim cümlesi düzeltildi (§4). |
| 11 | Histerezisli serbest satır ve olay sayaçlarındaki boşluklar tanımsız | Doğru: serbest satır ayrışmayı sonraki karelere taşıyabilir; "art arda" kesintide tanımsızdı. | **Kabul:** serbest satır yayılım kuralı (Kabul 1-2), `yok` gözlemi ve ≤ 5 kare kuralı (§5.5). |
| 12 | Harita kararlılığı ölçüsü bağımsız / yeterli değil; bootstrap genelleme vermez | Doğru: duran kareler birleştiriliyordu, birim belirsizdi, ölçü etiketlere bağlıydı. | **Kabul:** etiketten bağımsız harita kayması (referans kayda karşı), kesintisiz duruş bölümleri, aynı birim; bootstrap sınırlaması yazıldı. |
| §10/6 | Donmuş kare "çekirdek ama ilk kesilir" çelişkisi | Doğru. | **Kabul:** kare bayrağı çekirdek, alarm kesilebilir. |
| §10/4 | Tuval yeteneği modeli belirlememeli | Doğru (gerekçe sırası). | **Kabul:** gerekçe nadir görüş + ölçülen model hatası. |

v3 (ürün paketleri) Codex incelemesinden sonra eklendi; v3'ün kendisi henüz bağımsız incelemeden geçmedi.
