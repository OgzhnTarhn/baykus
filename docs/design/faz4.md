# Faz 4 Tasarımı — Hareketli hedef (yalın)

> **Tarih:** 2026-10-20 (taslak v1) · 2026-10-20 (v2: Codex incelemesi doğrulanıp işlendi, §12) ·
> **Durum:** v2, onaylı (2026-10-20; kullanıcı kararı §10/1 = A; plan `docs/plans/F4.md`) · **Dal:** `f4`
> **Dayanak:** spec §2.3 S11, §3.2 madde 5, §6 Faz 4, §16, §17, §20.6 kesme sırası, §22 Faz 4 satırı, §23 son
> risk satırı; ADR 0006, 0007, 0008; Faz 3 sonuçları (`docs/system.md`), Faz 3 kod incelemesi
> (`docs/reviews/2026-10-19-faz3-kod-incelemesi.md`).
> **Kullanıcı yönergesi (2026-10-20):** "Faz 4 tasarımını yap; Codex'e sorun var mı diye soracağım, sonra
> birlikte revize edeceğiz." İnceleme sonrası: "Codex'in dediklerini araştır; mantıklıysa değiştir." Sorular
> atlandı; varsayılanlar **Claude'un kararı** olarak işaretli (§10). Yalın, boş iş yok (Faz 2-3 ile aynı ilke).
> Kodlama başka oturumda; bu belge ve ondan yazılacak plan (`docs/plans/F4.md`) yeterli olmalı.

## 1. Neden ve ne

Faz 3 sonunda sistem araçları bulup numaralıyor, ama her şeyi **görüntü koordinatında** yapıyor. Drone
kendisi hareket ettiği için görüntüde her araç kayıyor: park etmiş bir araç da ekranda kayıyor. Operatörün
sorusu "hangisi gerçekten hareket ediyor, nereye gidiyor, şu kavşağa kim girdi" — Faz 3 bunu cevaplamıyor.
Faz 4:

**Motor her karede kameranın kendi hareketini (ego-hareket) arka plandan kestirir, izleri bir "harita"
koordinatına taşır ve her izi hareketli / duran diye işaretler; ekran bu haritayı kareleri birleştirerek
(göreli mozaik) çizer, operatörün çizdiği bölgeler için giriş / çıkış / durdu olayları üretir.**

Mülakat hikâyesi (demo): "Drone uçuyor, görüntü kayıyor. Düz çerçeveli kutular hareketli, kesikli kutular
park etmiş araçlar — kamera hareketi çıkarıldığı için. Yanda kareler birleşip bir harita oluşuyor, izler
üzerinde. Kavşağa bir bölge çiziyorum; giren her araç olay listesine düşüyor, orada duran araç 'durdu' diye
işaretleniyor. Hareketli/duran ayrımı M3OT'ta ölçüldü; etiketleri iz yörüngelerinden ayrı bir yöntemle
türettim ve rastgele bir örneklemi kör olarak elle doğruladım." Bu metin, §6'daki ölçümler ve iddia
koşulları (§6.6) tutmadan yazılmaz.

Yöntem yeni değil, bilerek: ego-hareket kestirimi BoT-SORT'un kamera hareketi telafisindeki (GMC) yaygın
"seyrek optik akış + benzerlik dönüşümü + RANSAC" tarifi; hareket kararı telafi edilmiş hız eşiği. Ayırt
edici olan **dürüst ölçüm**: üründen ayrı etiketleyici, etiketleyiciden bağımsız seçilmiş kör insan
örneklemi, sabit paydalar, oracle / motor satırları.

## 2. Hedefler

1. **Ego-hareket (motorda):** her karede kare → harita benzerlik dönüşümü; anahtar kareye kayıtla birikimli
   kayma sınırlı; açık **harita nesli** sözleşmesi (kanal değişimi ve uzun kayıpta yeni nesil, kısa kayıpta
   yeniden bağlama); IR öncelikli, IR bozuksa RGB; tespit kutuları maskelenir. C++ ↔ Python eşlikli.
2. **Hareket durumu (motorda):** iz başına `moving / stationary / unknown`, telafi edilmiş ve kutu boyuyla
   normalleştirilmiş hızdan; eşikler ölçümden önce sabit. C++ ↔ Python eşlikli.
3. **Ölçüm (S11):** M3OT'ta iz yörüngelerinden türetilmiş **referans etiketler** (ürünle aynı olmayan ayrı
   kayıt yöntemi), etiketleyiciden bağımsız rastgele seçilmiş **kör insan örneklemi**; oracle ve motor
   satırları, sabit paydalarla; val'de geliştirme ve oracle işlev kapısı, test dondurmadan sonra bir kez.
4. **Göreli mozaik harita (ekranda):** kareler ego dönüşümüyle tuvale yerleştirilir; izler, kameranın şu
   anki görüş alanı ve bölgeler harita üzerinde.
5. **Bölge olayları (ekranda):** operatörün çizdiği çokgenler; `giriş`, `çıkış`, `durdu`; bölge sayaçları;
   gösterim süzgecinden bağımsız **bütün takip akışından** beslenir.
6. **Donmuş kare:** kare düzeyinde bayrak (ego için çekirdek) + `rgb_frozen` / `ir_frozen` alarmı (ekran).
7. **Gerileme yok:** Faz 3 Kabul 1-8 yeşil kalır; gecikme p95 ≤ 100 ms (ego + hareket dahil, gerçek ardışık
   video girdisiyle).

## 3. Hedef dışı (bilerek kesilenler)

| Kesilen | Neden |
|---|---|
| Piksel → GPS, metre, km/sa hız | İrtifa, GSD ve kare hızı doğrulanmadı (spec §6 kapsam dışı; M3OT `fps=30` resmi araçta varsayım). Hız yalnız "kutu boyu / kare" biriminde, ekranda sayı olarak gösterilmez. |
| Takipçiye kamera hareketi telafisi (BoT-SORT GMC tarzı Kalman düzeltmesi) | **Kapsam tercihi:** takipçiyi değiştirir; Faz 3 Kabul 4 eşliği ve ADR 0008 ölçümü `trackers==2.6.1`'e bağlı. Kazancın küçük olacağı iddia edilmez (val'deki oracle 0,974 / motor 0,220 HOTA farkı tespit kaybını gösterir, kamera hareketinden gelen ilişkilendirme hatasını ayırmaz). Faz 5 notu. |
| Öğrenilmiş hareket sınıflandırıcısı, yeni eğitim | "Train only what ships"; eşik kuralı yeterli mi önce ölçülür. |
| Ürün yolunda homografi (8 serbestlik) | Nadir görüş; val keşfinde dönme ≈ 0,02°/kare, ölçek ≈ %0,2/kare (§4). Model hatası val'de **ölçülür** (§6.3 harita kayması, referans homografiye karşı); büyük çıkarsa Faz 5'te ele alınır. (Tarayıcı tuvalinin afin çizebilmesi kolaylık, gerekçe değil.) |
| Kanallar arası (RGB ↔ IR) harita bağlama | Kesitler arası kayıt ayrı bir problem; M3OT'ta doğal RGB-IR kayması belgeli (`docs/system.md` Sınırlamalar). Kanal değişince yeni harita nesli (§5.1). |
| Döngü kapatma, global optimizasyon (bundle adjustment) | Harita "göreli" ve niteliksel; kayma ölçülüp raporlanır (§6.3). |
| Kalıcı bölgeler, veritabanı, bölge dışa aktarma, sesli alarm | Spec §17 kapsam dışı ilkesi. Bölgeler oturum ve harita nesli belleğinde. |
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
motor (C++, oturum başına durum)                                       ekran (Next.js)
decode → health(+donmuş) → gate → pre → infer → post ─┐
                                                       ├→ ego (anahtar kare, LK, RANSAC, harita nesli)
                                                       │      → track (değişmez) → motion → JSON
                                                       └─ ego maskesi: takip katmanı kutuları (ham ≥ 0,1)
JSON: detections[] (gösterim, conf süzgeçli) · tracks[] (bütün takip çıktısı, süzgeçsiz)
      ego{frame_to_map, ok, source, map_gen, ...} · health.*.frozen
                                                     → oynatıcı (detections[]; kutu biçimi = durum)
                                                     → harita paneli + bölge olayları (tracks[] + ego)
```

Kural: **ölçülen her şey motorda ve Python referansıyla eşlikli** (ego, harita nesli, hareket durumu, donmuş
kare); **operatör ayarına bağlı mantık ekranda** (mozaik çizimi, bölge olayları; Vitest senaryo testli).
Takipçi değişmez; ego takipten bağımsız hesaplanır, hareket durumu takip kimliklerini kullanır.

### 5.1 Ego-hareket ve harita nesli (`engine/src/ego.cpp`, Python `baykus.motion.ego`)

**Kaynak kanal:**
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
  `ok` ise nesil sürer (bağlandı); değilse yeni nesil.
- **Yeni nesil:** kanal değişimi (IR ↔ RGB) ya da > 10 karelik kayıp ya da bağlama başarısız → `map_gen + 1`,
  geçerli `ok` kare yeni neslin ilk karesi (`M = I`), yeni anahtar kare. Bir sonraki `ok` kareye kadar ego yok.
- **Etkisi:** hareket geçmişi (§5.2) yeni neslin başında silinir; ekran yeni nesilde mozaiği, bölgeleri ve bölge
  durumlarını temizler ve olay listesine "harita yeniden başladı (nesil n)" yazar (§5.4-5.5).
- `session/reset` nesli 1'e döndürür ve her şeyi siler. Oturumsuz istek ego hesaplamaz (`ego: null`).

### 5.2 Hareket durumu (`engine/src/motion.cpp`, Python `baykus.motion.state`)

- İz başına halka arabellek (kimliği olan izler; `track_id: null` → `unknown`), yalnız `ego_ok` karelerde:
  harita merkezi `cₜ = Mₜ(kutu merkezi)` ve harita boyu `sₜ = √(w·h) · σ(Mₜ)`, `σ = √|det(Mₜ'nin 2×2 kısmı)|`.
  Pay ve payda aynı (harita) koordinatında.
- `K = 15` kare. Geçerli karede ve tam `t − K` karesinde kaydı olan, aynı nesilde ve `(t − K, t]` aralığındaki
  bütün karelerde `ego_ok` olan iz için `v = ‖cₜ − cₜ₋ₖ‖ / K / sₜ`.
- **Karar (histerezis):** `v ≤ 0,03` → `stationary`; `v ≥ 0,10` → `moving`; arada → önceki karar (yoksa
  `unknown`). `v` hesaplanamazsa önceki karar en çok `K` kare tutulur, sonra `unknown`. Yeni nesilde bütün
  kararlar `unknown`'dan başlar.
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
  **telafisiz** (herhangi bir kestiricinin kaydı kullanılırsa doğrulama döngüsel olur; insan arka plana göre
  karar verir), oynatılabilir; aracın kutusu ince çerçeveyle.
- Kullanıcı "hareketli / duran / emin değilim" işaretler; sayfa cevapları JSON olarak indirir →
  `baykus-motion audit-score` → `results/motion/label_audit_val.json`.
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

### 5.4 Göreli mozaik harita (ekran, `ui/components/MapPanel.tsx`, `ui/lib/mosaic.ts`)

- Ego kanalının görüntüsü her **5.** `ego_ok` karede çeyrek çözünürlükte (160 × 128) saklanır ve
  `frame_to_map` ile tuvale çizilir (`setTransform`). Yeni kareler eskilerin üstüne. Bellek: 600 karelik dizide
  ≈ 120 küçük resim.
- Harita sınırları büyüdükçe görünüm "sığdır" ölçeğiyle yeniden çizilir.
- Üstünde: kameranın şu anki görüş alanı (kare dikdörtgeni), iz kuyrukları (son 150 kare, `tracks[].map_xy`),
  durum biçimi, bölgeler.
- `ego_ok=false` karelerde küçük resim eklenmez; kuyruk o karede kesilir.
- **Yeni harita nesli** (`map_gen` arttı) ve `session/reset` mozaiği, bölgeleri ve olay durumlarını temizler;
  olay listesine "harita yeniden başladı" satırı (sıfırlamada liste de temizlenir, Faz 3 kuralı). Bilinen sınır,
  ekranda yazılı.
- Panel başlığı: "göreli harita (kamera hareketinden; ilk kareye göre; kanal değişince yeniden başlar)".

### 5.5 Bölge olayları (ekran, `ui/lib/zones.ts`, saf fonksiyonlar)

- **Girdi: yanıtın `tracks[]` alanı** (bütün takip çıktısı, `conf` süzgecinden bağımsız; §5.7) ve `ego`.
  `detections[]` kullanılmaz; güven kaydırıcısı olayları ve sayaçları değiştirmez.
- Operatör harita üzerinde çokgen çizer (tıkla, çift tıkla kapat); en çok 8 bölge; ad `B1…B8`; silinebilir.
  Bölge harita koordinatında ve **çizildiği harita nesline** ait.
- İz × bölge gözlemi her karede üç değerli: `içerde` / `dışarıda` (iz `tracks[]`'ta ve `map_xy` var; ışın atma,
  kenar dahil) ya da `yok` (iz bu karede çıktıda değil ya da `ego_ok=false`).
- **Durum makinesi** (kare numarasıyla; sabit, yapılandırılmaz):
  - Sayaçlar yalnız `içerde` / `dışarıda` gözlemlerinde ilerler. **Art arda** = araya `yok` girebilir ama en
    çok **5** `yok` kare; 5'ten uzun `yok` dizisi sayaçları sıfırlar (bölge içi/dışı durum korunur).
  - `giriş`: durum dışarıdayken 3 `içerde` gözlem art arda → olay, kare = ilk `içerde` gözlemin karesi.
  - `çıkış`: durum içerdeyken 3 `dışarıda` gözlem art arda → olay, kare = ilk `dışarıda` gözlemin karesi.
  - `çıkış (kayıp)`: durum içerdeyken iz art arda **30** kare `tracks[]`'ta yok (takipçinin
    `lost_track_buffer`'ı; `ego_ok=false` kareler de sayılır) → olay; iz–bölge durumu silinir.
  - `durdu`: durum içerdeyken `motion = stationary` olan 30 gözlem art arda (aynı "art arda" kuralı) → olay;
    aynı kalışta bir kez. `unknown` ve `moving` sayacı sıfırlar.
  - Yeni harita nesli ve `session/reset`: bütün bölgeler ve durumlar silinir.
- Bölge sayaçları: şu an içerde (durum) iz sayısı, kaçı hareketli / duran / bilinmiyor.
- Olay listesi paneli: kare, iz no, bölge, olay; tıklayınca kareye gitmez (atlama oturumu sıfırlar).
- **Neden ekranda (Claude'un kararı, §10):** bölge operatör ayarıdır ve tek bir tarayıcı oturumuna aittir;
  motorda olsa ayar için yeni bir uç (`/v1/session/zones`), oturum durumu ve Python referansı + eşlik
  gerekirdi. Veri sözleşmesi (`tracks[]`, `map_gen`, `yok` gözlemi) yukarıda tanımlı olduğu için mantık
  saf fonksiyon olarak Vitest'te tam sınanır (Kabul 7).

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

### 5.7 API değişiklikleri (geriye uyumlu: yalnız alan ekleniyor)

```
POST /v1/detect  (sorgu aynı)  →  Faz 3 yanıtı +
  detections[i].motion, .speed_rel, .map_xy           (gösterim katmanı, conf süzgeçli — Faz 3 gibi)
  tracks: [{"track_id","cls_id","box","score_raw","score","motion","speed_rel","map_xy"}]
          takip katmanındaki (ham ≥ 0,1) kimliği olan bütün tespitler, conf süzgeci YOK
  ego: {"ok","source":"ir|rgb|none","map_gen","frame_to_map":[a,b,tx,c,d,ty] | null,
        "inliers","inlier_ratio","keyframe","relinked"} | null
  health.rgb.frozen, health.ir.frozen, health.rgb.diff_mad, health.ir.diff_mad   (oturumsuz: null)
  timing_ms.ego, timing_ms.motion
GET /v1/model → + "motion": {configs/engine/motion.json parametreleri}
```
- `tracks[]` motorun mevcut takip girdisinden kurulur (`runtime.cpp`: takipçi `dets_all`'ın ham ≥ 0,1
  kısmını görür; gösterim listesi ayrıca `filter_display(conf)` ile süzülür — Faz 3'te kimlikler yalnız süzülmüş
  listeye yazılıyordu).
- Ego ve hareket yalnız `track=true` + `session` ile (oturumsuz `ego: null`, `tracks: []`, `motion: "unknown"`).
- `configs/engine/motion.json` repoda (ego, nesil, hareket, donmuş parametreleri); motor açılışta okur;
  sha256 `/v1/model`'de. Bench için `ego.enabled` bayrağı (Faz 3 yolunun aynı girdide ölçümü, §6.4).

### 5.8 Ekran değişiklikleri (`ui/`)

- **Kutular:** `moving` düz kalın çerçeve + "H"; `stationary` kesikli ince çerçeve + "D"; `unknown` Faz 3
  görünümü. Yalnız renge dayanmaz. Üst çubuğa "yalnız hareketliler" süzgeci (yalnız gösterim).
- **Tespit listesi:** durum sütunu.
- **Harita paneli** (§5.4) ve **olay paneli** (§5.5): sağ sütunda sekme ya da oynatıcının yanında ikinci
  görünüm (builder; yan yana görünümle çakışmasın).
- **Sağlık paneli:** donmuş rozeti; alarm listesine iki yeni kimlik; ego durumu (kanal, nesil, iç nokta).
- **Arıza simülasyonu:** "kareyi dondur".
- Faz 3'ün bütün kuralları geçerli (tek istek, kare atlanmaz, oturum sıfırlama kuralları, internetsiz).

### 5.9 Fikstürler (yalnız M3OT val, CC BY 4.0; `fixtures/m3ot/NOTICE.md` güncellenir)

- **Uçtan uca video klibi:** `1_08`'den ardışık **60** kare, kodlanmış RGB JPG + IR PNG (≈ 20 MB; builder
  ölçer). Repoya değil, modeller gibi GitHub Release'e (`fixtures-f4`, sha256 `configs/engine/fixtures.json`'da);
  CI ve `scripts/prepare` indirip denetler. Senaryolar (girdi baytları Python'da üretilip aynı pakette; iki
  taraf aynı dosyaları okur):
  1. temiz 60 kare;
  2. IR kaybı: 25-34. karelerde IR sıfır PNG (RGB'ye geçiş = nesil 2, IR'ye dönüş = nesil 3);
  3. IR donması: 20-29. karelerde IR = 19. karenin baytları (kare bayrağı, alarm, kanal geçişi);
  4. ego kaybı ve bağlama: 40-44. karelerde iki kanal sıfır (`source: none`; 5 kare ≤ 10 → yeniden bağlama
     denemesi);
  5. uzun kayıp: 30-45. karelerde iki kanal sıfır (> 10 kare → yeni nesil);
  6. 40. karede `session/reset`.
  En az 3 anahtar kare değişimi temiz senaryoda bulunmalı (başlangıç karesi kaydırılarak sağlanır).
- **Sentetik dönüşüm:** klipten 10 kare × 5 bilinen benzerlik dönüşümü (öteleme ±20 px, dönme ±1°, ölçek ±%3;
  sabit liste), Python `warpAffine` ile üretilip aynı Release paketinde.
- **Uzun geçmişli hareket birim eşliği:** mevcut `fixtures/tracks/m3ot_val_dets.json.gz` (repoda; CI
  `dev_dets.json.gz` adıyla) + Python'un tam çözünürlük karelerden ürettiği kare başına `frame_to_map`,
  `ego_ok`, `map_gen` (`fixtures/motion/maps_val.json.gz`, iki dizi × 600 kare, küçük, repoda). Hareket
  modülünü 600 karelik geçmişte, ego'dan bağımsız sınar.
- **Sınır kuralı:** Faz 3 §4.6 gibi üretici denetler; anahtar kare / bağlama / nesil kararına giren ölçüler
  (iç nokta, oran, öteleme) eşiğinden en az 2 nokta / 0,02 / 1 px uzak olmalı; tutmayan senaryoda başlangıç
  karesi kaydırılır (yalnız val, yalnız eşlik için).

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

## 6. Değerlendirme ve kabul ölçütleri (ölçümden önce sabit)

Her sayı bir JSON'a bağlı; `docs/motion.md` bu dosyalardan üretilir (`baykus-system report` genişletmesi ya da
ayrı `baykus-motion report`; builder), elle düzenlenmez.

| Sonuç dosyası | İçerik |
|---|---|
| `results/motion/explore_val.json` | §4 keşfi (yeniden üretilmiş) |
| `results/motion/label_audit_{val,test}.json` | kör insan örneklemi (Kabul 4; test yalnız rapor) |
| `results/engine/parity_ego.json`, `parity_motion.json` | Kabul 1-3 |
| `results/motion/m3ot_{dev,test}_{oracle,engine}.json` | hareketli / duran ölçümü |
| `results/motion/m3ot_{dev,test}_ego.json` | ego kalitesi, harita kayması, kararlılık, donmuş kare |
| `results/bench/e3c_onnx_fp32_engine_laptop-docker_f4.json` | Kabul 5 (Faz 3 dosyası değişmez) |

### 6.1 Eşlik ve doğruluk (CI, her commit)
- **Kabul 1 — uçtan uca eşlik (asıl kapı):** klip senaryolarının (§5.9) hepsi iki **tam** pipeline'dan
  (kodlanmış baytlardan, oturumlu, `gate=advise`, takip açık) geçer. Her karede **aynı**: `ego.ok`, `source`,
  `keyframe`, `relinked`, `map_gen`, `health.*.frozen`, alarm durumları, `tracks[]` kimlikleri (Faz 3 takip
  eşliği zaten birebir), `motion`. Toleranslı: `frame_to_map` ile tam çözünürlük kare köşelerinin dönüşüm farkı
  ≤ **0,25 px**; `map_xy` ≤ 0,25 px; `speed_rel` ≤ **2e-3** (köşe toleransının hıza yansıması: 0,25 / 15 / 10).
  - **Histerezisle serbest satırlar:** bir iz × karede iki taraftan birinin `v`'si 0,03 ya da 0,10'a `2e-3`'ten
    yakınsa o satır serbesttir. Serbest satırda kararlar ayrışırsa, o izin sonraki satırları da iki tarafın
    kararı yeniden aynı olana kadar serbesttir (histerezis ayrışmayı taşır; her iki taraf `v` bandı terk edince
    yeniden eşitlenir). Kanıtsız ayrışma (serbest olmayan satırda) ❌. Serbest satırlar sayılır; her senaryoda
    karşılaştırılan iz × kare oranı ≥ **%99**.
  - Anahtar kare / bağlama / nesil kararı farklı çıkarsa ❌ (sınır kuralı bunu fikstürde dışarıda tutar);
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
- **Kabul 5:** motor konteynerinde, laptop prizde, `gate=advise`, takip + ego + hareket + donmuş açık, tek
  oturum. Girdi **val `1_08`'in ardışık 550 karesi** (bellekteki kodlanmış baytlar; ≈ 150 MB): ilk 50 ısınma,
  sonraki **500 ölçüm**, döngü yok, sıfırlama yok (H5 sayıları, gerçek video). Motor p95 (HTTP hariç, bütün
  aşamaların **kare başına toplamının** p95'i) ≤ **100 ms**.
- Ayrıca raporlanır (eşik yok): aşama başına p50 / p95; anahtar kare ve bağlama karelerinin ayrı p95'i;
  kanal geçişi senaryosu (IR kaybı) p95'i; **aynı girdide `ego.enabled=false`** ile Faz 3 yolu (karşılaştırma
  aynı girdiyle; Faz 3'ün 20 görüntülük sonucu ayrı protokol olarak kalır).
- *Tahmin (ölçülmedi; aşama p95'leri toplanmaz):* ego yarı çözünürlükte birkaç ms ekler; Faz 3 p95'i 82,2 ms
  (20 görüntülük girdi). Yeni girdide pay bilinmiyor; ilk ölçüm Görev 3 sonunda yapılır (erken uyarı).
- Bütçe aşılırsa: yalnız ölçülmüş bir değişiklikle (ör. köşe sayısı, LK seviyesi) ve kullanıcının kararıyla;
  "her N karede bir ego" seçeneği bu tasarımda tanımlı değildir (atlanan karelerin `ego_ok`, hız ve olay
  sayaçlarına etkisi yeni tasarım ister).

### 6.5 Ekran ve olaylar
- **Kabul 6 — ekran:** demo dizisinde, ekran görüntüsü / GIF ile deney günlüğüne: kutu durum biçimleri ve
  süzgeç, liste sütunu, harita (mozaik, görüş alanı, kuyruklar), bölge çizme / silme, olay listesi ve sayaçlar,
  güven kaydırıcısının olayları değiştirmediği, sıfırlamada ve IR kaybında (yeni nesil) temizlenme, donmuş
  rozeti + alarm + "kareyi dondur".
- **Kabul 7 — bölge olayları (CI, Vitest):** senaryo testleri: giriş / çıkış titreşimi (2 gözlem → olay yok,
  3 → olay), araya ≤ 5 `yok` giren ve 5'ten uzun `yok`la bölünen diziler, kenar üstü nokta, içbükey çokgen,
  `çıkış (kayıp)` (30 kare), `durdu` bir kez ve `unknown`'la sıfırlanması, `ego_ok=false` kareler, kimliksiz
  tespit, `conf` değişiminin olaylara etkisizliği (aynı `tracks[]`), yeni nesil ve sıfırlama, 8 bölge sınırı.
- **Kabul 8 — gerileme:** Faz 3 Kabul 1-4, 6, 8 CI'da yeşil (takip fikstürü ve `Tracker.FixtureParity`
  değişmeden); `compose.offline.yml` kanıtları yeni imajlarla yeniden; S6 düzeltmesiyle kanıt adımları da
  başarısızlıkta düşer.

### 6.6 Faz çıkışı ve iddia kuralı

| Ölçüt | Etkisi |
|---|---|
| Kabul 1, 2, 3 (eşlik, sentetik doğruluk) | **çıkışı engeller** |
| Kabul 4 (kör insan örneklemi, val) | **çıkışı engeller**; iki kez tutmazsa yalnız kullanıcının ADR'siyle B planına iner |
| Kabul 5 (gecikme) | **çıkışı engeller**; aşılırsa ADR'yle kapanır, README'de olumsuz sonuç |
| Kabul 6, 7, 8 | **çıkışı engeller** |
| Kabul 9 (oracle işlev kapısı, val) | **çıkışı engeller**; tutmazsa ADR |
| Motor satırı, test sonuçları, ego kalitesi, harita kayması, kararlılık, donmuş kare, demo ölçütü, test audit'i | **yalnız raporlanır** |

**İddia kuralı (ölçümden önce):** README, `docs/motion.md` ve demo metni "hareketli / duran ayrımı çalışıyor"
cümlesini yalnız şu koşulda kullanır: test oracle `BA ≥ 0,85`, test motor `BA ≥ 0,70` ve `P_mov ≥ 0,80`,
test audit'inde (a) tutmuş. Tutmayan her koşul olduğu gibi yazılır ("ego ve kural çalışıyor ama tespit
yetersiz" gibi); olumsuz sonuç da sonuçtur (Faz 2 düzeni).

Çıkış: S11 işaretli; `docs/motion.md`; README'ye Faz 4 bölümü ve ekran görüntüsü; ADR 0009; spec v4.6;
`main`'e birleştirme ve `v0.3.0` (kullanıcı "at" derse push).

## 7. İş yükü ve takvim

Bütün kodu Claude yazar (Faz 3 kararı 4 sürüyor); kullanıcı adımları: kör örneklem (val ≈ 20-30 dk, test
≈ 20-30 dk), GitHub Release `fixtures-f4` yüklemesi, Docker Desktop açık, ekran kontrolü, onaylar.
**GPU yok; Colab yok** (§10 karar 1 = A). Val ve test dizileri laptopta
(`data/m3ot_dev` 346 MB, `data/m3ot_test` 309 MB).

| # | İş | Tahmin |
|---|---|---|
| 0 | Faz 3 devri: S1, S3, S6, S11, S12, K7 (§5.10) | 1 gün |
| 1 | `baykus.motion`: keşif CLI (`explore`), referans etiketleyici (`labels`), kör örneklem sayfası (`audit`, `audit-score`) → **KAPI: kullanıcı val örneklemi** (Kabul 4) → etiketleyici dondurulur | 2-3 gün |
| 2 | Python ego + nesil + hareket + donmuş (TDD: sentetik dönüşüm, kayıp / bağlama / nesil, histerezis, tutma) · `configs/engine/motion.json` · oracle satırı val + Kabul 9 · harita kayması · fikstür üretici (klip senaryoları, Release paketi) | 4-5 gün |
| 3 | Docker `deps`: OpenCV'ye `video`, `calib3d` (+ `features2d`, `flann`) modülleri (bir kez yeniden derleme, 30-60 dk; CI önbelleği bir kez geçersiz) · C++ ego + nesil · Kabul 3 · **erken bench** (ego aşaması, 550 kare) | 3-4 gün |
| 4 | C++ hareket + donmuş, pipeline, `tracks[]` ve API alanları, dizi koşucusu · Kabul 1, 2 · motor satırı val · demo ölçütü | 4-5 gün |
| 5 | Bench protokolü (550 ardışık kare, `ego.enabled=false` satırı) + Kabul 5 | 1 gün |
| 6 | Ekran: kutu durumları, liste, harita paneli, bölge çizimi + olaylar (TDD, Kabul 7), nesil temizliği, donmuş arıza + alarm · Kabul 6 | 5-7 gün |
| 7 | CI (Release fikstürü, yeni kanıt adımları, `ui` testleri), kapalı ağ kanıtlarının yeniden üretimi · Kabul 8 | 1-2 gün |
| 8 | **Dondur → test bir kez:** test etiketleri (dondurulmuş etiketleyici), oracle + motor satırı, ego kalitesi, **kullanıcı test örneklemi** · iddia kuralı · ADR 0009 · `docs/motion.md`, README, spec v4.6, `v0.3.0` | 2-3 gün |

**Toplam ≈ 5-6 hafta** (spec'te 8 hafta, 1 Şub – 28 Mar 2027; takvimin ≈ 3 ay önündeyiz). En riskli adımlar:
1 (etiketler doğrulanmazsa ölçüm kısmı düşer), 3-4 (OpenCV modülleri, LK eşliği, nesil durum makinesi).
Kesme sırası spec §20.6: önce mozaik çizimi (ego, nesil ve hareket kalır), sonra bölge olayları; donmuş
**alarmı ve arıza simülasyonu** mozaikten de önce kesilir — donmuş **kare bayrağı** ego'nun parçası, kesilmez.

## 8. Riskler

| Risk | Önlem |
|---|---|
| Etiket ile ürün aynı hatayı paylaşır (döngüsel) | Farklı özellik, model, pencere ve maske (§5.3 tablo); telafisiz, etiketleyiciden bağımsız seçilmiş kör insan örneklemi; oracle satırı; harita kayması referans kayda karşı. |
| Etiketleyicinin dışladığı nüfus (bant, kayıtsız) yüksek doğruluğu "kolay örnekler"e kaydırır | Dışlanan nüfus ve ürünün oradaki çıktısı raporlanır; Kabul 4 (c) dışlamanın insan-kararlı örnekleri %20'den fazla dışarıda bırakmamasını ister; hedefli 40 örnek. |
| GT kutu titremesi (duranlarda `v ≈ 0,01-0,02`) duranları banda iter | Keşifte duranların büyük kısmı ≤ 0,03; dışlanan oran raporlanır; eşik ayarlanmaz. |
| Duran-kalkan araçlar (kavşak) bant içinde uzun kalır | Ürün histerezisle önceki kararı tutar; "kararlı bölüm" ve karar gecikmesi ayrıca raporlanır. |
| Ego başarısız (dokusuz alan, gece RGB, su) | IR öncelikli; kayıp / bağlama / nesil sözleşmesi; `ok` oranı ve nesil sayısı raporlanır. |
| Kanal değişimi sık olur, harita sürekli yeniden başlar | IR'ye dönüş 5 temiz kareyle; nesil sayısı raporlanır; temiz M3OT'ta IR kaybı beklenmez (arıza simülasyonu dışında). |
| Anahtar kareli zincirde de kayma birikir; harita bozulur | Harita kayması Δ = 150'ye kadar ölçülür; harita "göreli" ve niteliksel; kesme sırasında ilk giden. |
| Yaklaşma / irtifa değişimi hız eşiğini kaydırır | Kutu boyuyla normalleştirme; pay ve payda aynı koordinatta (ürün harita, referans `t + K`). |
| Kare hızı farklı veride eşikler geçersiz | Birim kare başına; başka kare hızında yeniden ölçülmeden kullanılmaz (§4). |
| LK / köşe seçimi CI işlemcisinde farklı (SIMD) | Aynı OpenCV 5.0.0; tolerans + sınır kuralı + histerezis serbest satır kuralı + mekanik kanıtlı istisna (Kabul 1). |
| Gecikme 100 ms'yi aşar | Görev 3'te erken bench; aşama kırılımı; ölçülmüş değişiklik ya da ADR. |
| Motor satırı val'de düşük (yoğun sahne, AP50 0,40) | Oracle satırı yöntemi tespitten ayırır; motor satırı yalnız rapor; iddia kuralı. |
| Test yalnız iki dizi | Bootstrap yalnız dizi içi; genelleme iddiası yok; ek bölüm yok (§10/1 = A, kullanıcı kararı). |
| Kullanıcı örnekleminde yorulma / yanlılık | 120 örnek, karışık sıra, etiket gizli, "emin değilim" seçeneği; test örneklemi ayrı günde yapılabilir. |

## 9. Builder'a bırakılanlar (varsayılanlarıyla; plana yazılır)

- OpenCV ek modüllerinin CMake bayrakları (varsayılan: yalnız gerekenler; wheel bilgisiyle `opencv_build.json`).
- Dizi koşucusunun biçimi (`baykus_detect` oturum kipi ya da yeni uygulama).
- Release fikstür paketinin biçimi (`.tar` + sha256) ve `scripts/prepare` / CI indirme adımı (model gibi).
- Harita panelinin yeri (sekme ya da ikinci görünüm), renkler, kuyruk uzunluğu ayrıntısı, yakınlaştırma.
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

**Claude'un kararları (inceleme sonrası hâli):**

2. Eşikler (0,03 / 0,10, `K` = 15) val keşfinden; val sonuçları "keşif". (Codex: makul.)
3. Kör insan örneklemini kullanıcı yapar: val (kapı) + test (rapor), her biri ≈ 20-30 dk; örneklem
   etiketleyiciden bağımsız seçilir (v1'de etiketleyicinin çıktısından seçiliyordu — Codex haklı, düzeltildi).
4. Bölge olayları ekranda, `tracks[]` + `map_gen` + `yok` gözlemi sözleşmesiyle (v1'deki "metrik yok"
   gerekçesi kaldırıldı).
5. Ürün yolunda benzerlik dönüşümü; model hatası harita kaymasıyla val'de ölçülür.
6. Takipçiye kamera telafisi yok — kapsam tercihi, kazanç iddiası yok.
7. Donmuş kare: kare bayrağı çekirdek (ego), alarm ve arıza simülasyonu kesilebilir.
8. Faz 3 bulgularından S1, S3, S6, S11, S12, K7 Görev 0'da.
9. Bench: val'den 550 ardışık kare, döngüsüz; aynı girdide `ego.enabled=false` satırı.
10. Oracle işlev kapısı (Kabul 9) çıkışı engeller; motor satırı yalnız rapor + iddia kuralı (§6.6).
11. Hız hiçbir yerde fiziksel birimle gösterilmez; harita "göreli" etiketli.

## 11. Spec'e yansıyacaklar (onaydan sonra, v4.6)

§2.3 S11 metni (ölçütler §6), §6 Faz 4 maddesi (donmuş kare eklenir, takipçi telafisi ve kanallar arası
bağlama kapsam dışı), §7.4 FR-E5 API alanları (§5.7, `tracks[]`), FR-U eklemeleri (harita, bölgeler, olaylar,
durum biçimleri, donmuş arıza), §15.1 modül listesi (`ego`, `motion`), §14 bench protokolü (Faz 4 satırı),
§16 not (takipçi değişmedi), §17 düzen, §18 OpenCV modülleri ve Release fikstürü, §22 Faz 4 satırı
(≈ 5-6 hafta), §23 son risk satırının güncellenmesi; ADR 0009 numarası; `docs/health.md` donmuş kare
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
