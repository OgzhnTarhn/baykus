# Baykuş — Nerede kaldık? (geri dönüş özeti)

> **Tarih:** 2026-10-18 · **Dal:** `f3` · Bu belge, projeye ara verip dönen biri için tek oturumda
> okunacak bir özet. Ayrıntı her bölümün sonundaki bağlantılarda. Buradaki her sayı başka bir belgeden
> (README, `docs/results*.md`, `docs/health.md`, `docs/experiments.md`) alındı; yeni ölçüm yok.

---

## 1. Tek paragrafta proje

Drone'dan çekilmiş **normal (RGB)** ve **termal (IR)** görüntüleri birleştirip yerdeki araçları
(car, truck, bus, van, freight_car) gece-gündüz bulan bir sistem. Amaç yalnız iyi bir model değil:
**bir kamera bozulunca da çalışan**, **laptop işlemcisinde gerçek zamana yakın** koşan, sonunda
**C++ motoru + web ekranı + internetsiz kurulum** olan ve her sayısı ölçümle kanıtlı bir proje.
Hedef: mezuniyette (Haziran 2027) savunma sanayiinde ML / görüntü işleme mühendisliği başvurusu.

Proje 5 faza bölünmüş durumda (ADR 0006):

| Faz | İçerik | Durum |
|---|---|---|
| 1. Çekirdek | Veri, modeller (E1-E4b), sağlamlık, M3OT testi, edge ölçümü | ✅ Bitti (`v0.1.0`) |
| 2. Güvenilirlik | Kamera sağlığını kare kare ölçen kapı + güven kalibrasyonu | ✅ Bitti, **olumsuz sonuç** (`v0.1.1`) |
| 3. Sistem | C++ motoru, takip, operatör ekranı, Docker (internetsiz) | 📝 Tasarım yazıldı, **senin onayını bekliyor** |
| 4. Hareketli hedef | Kamera hareketi telafisi, hareketli/duran araç ayrımı | Bekliyor |
| 5. Sunum | README, demo videosu, teknik yazı | Bekliyor |

Takvimin yaklaşık **7 hafta önündeyiz** (spec'te Faz 3 7 Aralık'ta başlıyordu).

---

## 2. Şu anki durum (git ve sıradaki adım)

- **Dal:** `f3` (yerelde; henüz push edilmedi). Üzerinde tek commit var: Faz 3 tasarımı
  (`dc71be5`, [docs/design/faz3.md](design/faz3.md)).
- **`main`:** Faz 2 sonuna kadar her şey içinde ve GitHub'a push edilmiş (`d0f58be`).
- **Etiketler:** `v0.1.0` GitHub'da; **`v0.1.1` yalnız yerelde** (push bekliyor).
- **Testler:** 486 test yeşil (`cd python; uv run pytest -q`, 2026-10-18).

**Sıradaki adım senin:** Faz 3 tasarımını oku ve onayla (ya da değiştirmek istediklerini söyle).
Onaydan sonra sırayla: `docs/plans/F3.md` planı yazılır → spec v4.5 güncellenir → kodlama başlar.

Bekleyen push (sen "at" dersen yapılır): `git push origin f3 v0.1.1`

---

## 3. Bilmen gereken kavramlar (kısa sözlük)

| Terim | Ne demek | Bizde nerede |
|---|---|---|
| **mAP50** | Tespit doğruluğu, 0-100. Bir kutu, gerçek kutuyla en az %50 örtüşürse (IoU ≥ 0,5) doğru sayılır; sınıf başına ortalama. | Ana metrik, IR etiketleriyle |
| **IoU** | İki kutunun kesişim / birleşim oranı (0-1). | Eşleştirme, eşlik testleri |
| **train / val / test** | Eğitim / seçim / final ölçüm bölümleri. Bütün seçimler val'de; test **bir kez**, en sonda. | 17.954 / 1.469 / 8.980 görüntü çifti |
| **GA (güven aralığı)** | Görüntüleri yeniden örnekleyerek (bootstrap, 1000 kez) bulunan %95 aralık: "bu sayı ne kadar oynayabilir". | Her tabloda `[alt; üst]` |
| **Erken füzyon (E3)** | RGB'nin 3 kanalı + IR'nin 1 kanalı = **4 kanallı tek görüntü**, tek model. | Sevk edilen model bu tip |
| **Geç füzyon (E4)** | İki ayrı model (RGB, IR) çalışır, kutuları **WBF** ile birleştirilir. | Daha sağlam ama 2 kat yavaş |
| **WBF** | Weighted Boxes Fusion: iki modelin örtüşen kutularını skorla ağırlıklı ortalayarak tek kutu yapar. | `baykus-fusion` |
| **Modality dropout (E3c)** | Eğitimde rastgele bir kamerayı (%15 olasılıkla) siyah yapmak; model tek kameraya yaslanmayı bırakır. | Sevk edilen model **E3c** |
| **ONNX** | Modelin PyTorch'tan bağımsız, her yerde çalışan dosya biçimi. ONNX Runtime (ORT) ile koşar. | Sevk biçimi: ONNX FP32 |
| **FP32 / FP16 / INT8** | Sayı hassasiyeti. Küçüldükçe dosya küçülür, *genelde* hız artar, doğruluk düşebilir. | Laptopta INT8 **yavaşlattı** |
| **TensorRT** | NVIDIA'nın GPU için hızlandırıcısı. | Colab T4'te ölçüldü, sevk edilmedi |
| **p50 / p95** | Gecikmenin ortancası / karelerin %95'inin altında kaldığı süre. p95 takılmaları gösterir. | Hedef: p95 ≤ 100 ms |
| **Letterbox** | Görüntüyü oranını bozmadan 640×640'a sığdırıp kenarları gri (114) doldurmak. | Ön işleme; C++'ta birebir aynı olmalı |
| **Sağlamlık paketi** | Görüntüleri bilerek bozup (karartma, gürültü, bulanıklık, kayma, kamera kaybı) doğruluğu ölçmek. | `baykus-robust`, 16 koşul |
| **Kapı (gate)** | Faz 2: her karede kameranın sağlığını ölçüp bozuksa o kanalı modele **sıfır** olarak vermek. | Olumsuz sonuç verdi |
| **Kalibrasyon / ECE** | Modelin "%80 eminim" dediği tespitlerin gerçekten %80'i doğru mu? ECE bu sapmanın ölçüsü (0 = mükemmel). İzotonik regresyonla düzeltilir. | 0,080 → 0,007 |
| **Youden J** | Bir eşik seçme kuralı: doğru pozitif oranı − yanlış pozitif oranı en büyük olan eşik. | Faz 2 eşikleri |
| **ByteTrack** | Kareler arası araç takibi: Kalman filtresi (hareket tahmini) + Macar algoritması (eşleştirme). | Faz 3'te C++'ta yazılacak |
| **Eşlik (parity)** | C++ çıktısının Python referansıyla tolerans içinde aynı olması; CI'da otomatik test. | Faz 3 başarı kriteri S7 |

---

## 4. Faz 1 — ne yaptık, ne bulduk

### 4.1 Veri
- **DroneVehicle** (Kaggle): RGB-IR görüntü çiftleri, 5 araç sınıfı. Etiketler döndürülmüş kutu (OBB);
  biz **yatay kutuya (HBB)** çevirdik (ADR 0002), 100 piksellik kenarlığı kırptık (640×512).
- İki etiket seti var (RGB ve IR ayrı etiketlenmiş). **Ana tablo IR etiketleriyle** (ADR 0003), çünkü
  gece RGB'de görünmeyen araçlar IR setinde var.
- 4 kanallı görüntüler TIFF'te **her kanal ayrı sayfa** olarak saklanıyor; tek sayfalı biçimde OpenCV
  4. kanalı "alfa" sanıp sessizce bozuyordu (ADR 0004, Spike A). İşlenmiş veri private Kaggle veri
  setinde (`ouzhantarhan/baykus-dv-rgbt`).
- Aydınlatma etiketi (gündüz / gece / karanlık gece) resmi dosyada yoktu; RGB parlaklığından türettik
  (ADR 0005).
- İkinci veri seti **M3OT** (CC-BY-4.0): hiç eğitimde görülmeyen, video dizileri. Hem "başka veride ne
  oluyor" testi, hem Faz 3'ün demo / takip verisi.

### 4.2 Modeller (hepsi YOLO26s, Ultralytics, Colab'da eğitildi, tek seed)

Test seti, mAP50, IR etiketleri:

| Model | Ne | mAP50 | Karanlık gece |
|---|---|---|---|
| E1 | Sadece RGB | 73,3 | 66,9 |
| E2 | Sadece IR | 81,0 | 82,5 |
| E3 | Erken füzyon (4 kanal) | 82,3 | 82,9 |
| **E3c** | **E3 + modality dropout** | **82,5** | 83,1 |
| E4 | Geç füzyon (E1 + E2, WBF) | 82,0 | 82,0 |
| E4b | E4 + aydınlatmaya göre ağırlık | 82,2 | 82,0 |

**Ana çıkarımlar:**
1. Karanlıkta RGB çöküyor; füzyon modelleri 82-83'te kalıyor.
2. Temiz veride füzyon modelleri birbirinden **ayırt edilemiyor** (güven aralıkları örtüşüyor).
   Fark **bozulmalarda** ortaya çıkıyor.

### 4.3 Sağlamlık (val, mAP50)

| Koşul | E3 | E3c | E4 |
|---|---|---|---|
| Temiz | 85,7 | 85,1 | 84,8 |
| IR kamera kapalı | **12,8** | 72,0 | 76,1 |
| RGB kamera kapalı | 71,8 | 82,9 | 83,9 |
| Bulanıklık (11) | 64,1 | 65,2 | **80,9** |
| IR 20 px kayık | 25,8 | 26,6 | **48,5** |

- Düz erken füzyon (E3) IR kapanınca çöküyor; **modality dropout bunu düzeltiyor** (12,8 → 72,0).
- Geç füzyon (E4) bulanıklık ve iki kamera arasındaki **kaymaya** karşı belirgin daha dayanıklı.
  E3c'nin zayıf noktası bu (ve Faz 2'nin çıkış noktası).

### 4.4 Edge ve sevk kararı
Kural **ölçümden önce** yazıldı (ADR 0007): laptop CPU'da (Ryzen 7 8845HS) **p95 ≤ 100 ms** ve en
fazla 1 puan doğruluk kaybı; aday sırası önce E4, sonra E3c.

- E4 (iki model) laptopta p95 **131,6 ms** → kural dışı, elendi.
- **Sevk edilen: E3c · ONNX FP32** — p50 46 ms, **p95 60 ms (21 FPS)**, test mAP50 **81,4**, 38 MB.
- INT8 bu işlemcide **2 kat yavaşladı** (yalnız konvolüsyonlar nicemlendiği için sürekli dönüşüm —
  tahmin, ölçülmedi) ve doğruluk 3-8 puan düştü.
- GPU'da (Colab T4, TensorRT) ikisi de gerçek zamanlı: E3c FP16 ≈ 100 FPS, E4 INT8 ≈ 37 FPS.
- ONNX, PyTorch'tan ≈ 1 puan düşük (82,5 → 81,4). Kayıp küçük nesnelerde; olası nedenler (300 kutu
  sınırı, kare 640×640 girdi) tam ölçülmedi — açık iş.

Ayrıntı: [README.md](../README.md), [results-test.md](results-test.md), [robustness.md](robustness.md),
[results.md](results.md) "Edge (E9)".

---

## 5. Faz 2 — kamera bozulunca fark etmek (olumsuz sonuç)

**Soru:** E3c, IR kamera **bozuk veri** verirken çöküyor (IR 40 px kayınca val 9,0), ama IR'yi hiç
vermemek 72,0 alıyor. Yeniden eğitmeden, her karede kameraların sağlığını ölçüp bozuk olanı modele
vermeyen bir kapı bu farkı kapatır mı?

**Yöntem (eğitim yok, kural tabanlı):** her karede gürültü, bulanıklık, parlaklık, kayıp kamera ve
RGB–IR kayması (kenar haritalarında faz korelasyonu) ölçülüyor; eşikler yalnız train'den (Youden J).
Bozuk bulunan kanal ön işlemeden önce sıfırlanıyor. Kabul ölçütleri ölçümden önce sabitlendi.

| mAP50 | E3c | E3c + kapı | Fark [%95 GA] |
|---|---|---|---|
| Bozuk IR, val (ortalama) | 29,7 | 70,4 | **+40,7** [+38,3; +43,1] |
| Bozuk IR, test | 41,8 | 67,5 | **+25,7** |
| Temiz val | 85,1 | 82,0 | **−3,1** [−4,4; −1,8] |
| Temiz test | 82,5 | 79,2 | **−3,3** |

**Sonuç:** kapı bozuk IR'de çok işe yarıyor ama temiz karelerin **%4,1'inde yanlış alarm** veriyor ve
temiz veride sistemi 3 puan kötüleştiriyor → önceden yazılan kurala göre **başarısız**
(Kabul 1 ❌, 2 ✅, 3 ❌).

**Neden bulundu:** yanlış alarmların hepsi "IR hizasız", %80'i **karanlık gece**. RGB karanlıkken iki
kameranın kenarları eşleşmiyor, faz korelasyonunun güven yanıtı çok düşük (0,05; normalde 0,56) ve
kayma kestirimi anlamsız bir sayı buluyor. Önerilen düzeltme: *yanıt zayıfsa hizasızlık kararı verme*.
Val'e bakılarak bulunduğu için Faz 2'de ölçülmedi, **Faz 3'e devredildi** (senin kararın).

**Kalibrasyon:** ECE 0,080 → 0,007 (çok iyi), ama kalibre skorlar 80 farklı değere indiği için eşit
skorlar sıralamayı bozup mAP'yi 0,11 puan düşürüyor (sınır 0,1) → Kabul 4 ❌. Eşitlikte ham skorla
sıralayınca kayıp sıfır; bu da Faz 3'e devredildi.

Neden önemli: olumsuz sonucu dürüstçe, nedeniyle birlikte raporlamak mülakatta güçlü bir hikâye.
Ayrıntı: [health.md](health.md), [design/faz2.md](design/faz2.md).

---

## 6. Faz 3 — sıradaki iş (tasarım, onay bekliyor)

Tek cümle: **Bir C++ motoru görüntü çiftlerini alır, araçları bulur, takip eder, kameraların
sağlığını ölçer; bir web ekranı videoyu oynatıp kutuları, takip numaralarını ve kamera uyarılarını
gösterir; hepsi internete çıkmadan `docker compose up` ile açılır.**

Demo hikâyesi: M3OT videosu oynuyor, araçlar kutulu ve numaralı. Ekrandan IR'yi 20 px kaydırıyorsun;
birkaç kare sonra "IR hizasız" uyarısı çıkıyor.

| Parça | Ne | Not |
|---|---|---|
| C++ motoru (`engine/`) | letterbox → ONNX Runtime → çıktı çözümleme → sağlık → kalibrasyon → takip → HTTP JSON | Yalnız Docker'da (Linux) derlenir; Windows'a C++ araç zinciri kurulmaz |
| Eşlik testleri | C++ = Python (kutular IoU ≥ 0,99, skor farkı ≤ 1e-3, letterbox piksel farkı 0) | CI'da her commit'te |
| Sağlık kapısı | Faz 2 kuralı + **yanıt koruması**; varsayılan **yalnız uyarır** (`advise`), piksel kesmez | Düzeltme val'de "keşif" olarak ölçülür, yeni Colab yok |
| Takip | Kendi C++ ByteTrack'imiz, `roboflow/trackers` referansıyla birebir | M3OT test'te HOTA / MOTA / IDF1 |
| Operatör ekranı (`ui/`) | Next.js tek sayfa: oynatıcı, kutular, sağlık paneli, alarm, arıza simülasyonu, gecikme grafiği | Font/CDN yok, tamamen yerel |
| Kapalı ağ | Motor iç ağda (`internal: true`), dışarı çıkamaz; otomatik kanıt + Wi-Fi kapalı elle demo | Başarı kriteri S8 |

**Bilerek kesilenler:** motorda geç füzyon/WBF (E4 sevk edilmedi), TensorRT/GPU yolu, dikdörtgen
export, yeni eğitim, kullanıcı girişi/veritabanı.

**Kabul ölçütleri (ölçümden önce sabit):** tespit, letterbox, sağlık ve takip eşliği (Kabul 1-4);
motorun uçtan uca p95 ≤ 100 ms (Kabul 5); internetsiz çalışma kanıtı (Kabul 6).

**Süre:** ≈ 5-6 hafta, GPU gerekmez; yalnız bir **Colab CPU** oturumu (M3OT demo paketi hazırlamak).
Sonunda `docs/system.md`, README güncellemesi ve `v0.2.0`.

Ayrıntı: [design/faz3.md](design/faz3.md).

---

## 7. Kod haritası

```
C:\baykus
├─ python/src/baykus/     Python paketi (her alt klasör bir CLI)
│  ├─ data/               baykus-data   : kırpma, OBB→HBB, 4 kanallı TIFF, COCO, aydınlatma, M3OT
│  ├─ train/              baykus-train  : Ultralytics eğitimi, modality dropout
│  ├─ eval/               baykus-eval   : pycocotools mAP, bootstrap GA, kırılımlar, karşılaştırma
│  ├─ robustness/         baykus-robust : bozulma paketi (16 koşul)
│  ├─ fusion/             baykus-fusion : WBF geç füzyon
│  ├─ edge/               baykus-edge   : ONNX / INT8 / TensorRT export ve doğruluk
│  ├─ bench/              baykus-bench  : p50 / p95 gecikme ölçümü
│  ├─ health/             baykus-health : Faz 2 sağlık özellikleri, kapı, kalibrasyon, rapor
│  ├─ report/             baykus-report : docs/results*.md tablolarını JSON'lardan üretir
│  ├─ io.py, classes.py   ortak: görüntü okuma/yazma, sınıf sırası
├─ configs/               deney config'leri (e1…e3c, fusion, edge)
├─ notebooks/colab/       00_setup → 08_health (eğitim/GPU işleri; sen çalıştırırsın)
├─ results/               her sayının JSON kanıtı (config + commit + seed ile)
└─ docs/                  spec, ADR'ler, planlar, raporlar, deney günlüğü
```

Faz 3'te eklenecek: `engine/` (C++), `ui/` (Next.js), `docker/`, `docker-compose.yml`,
`.github/workflows/ci.yml`.

---

## 8. Çalışma düzeni (hatırlatma)

- **İş bölümü:** Claude kod, test, plan ve notebook hazırlar; **eğitim ve GPU işlerini sen Colab'da
  çalıştırırsın**, çıktıyı yapıştırırsın, hemen deney günlüğüne yazılır.
- **Dürüst ölçüm kuralları:** kabul ölçütü ölçümden önce yazılır; seçim val'de, test bir kez; kanıtsız
  sayı yok; olumsuz sonuç da raporlanır.
- **Yalınlık:** yalnız sevk edilecek şey eğitilir; notebook'larda yalnız karar adımları.
- **Git:** her faz kendi dalında (`f3`); commit'ler atomik, Conventional Commits; sen "at" deyince
  commit + push.
- **Komutlar:** `cd python; uv sync; uv run pytest -q; uv run ruff check .`
- **Colab bütçesi:** 2026-10-17 akşamı 244,05 birim kalmıştı (T4 ≈ 1,07 / saat, A100 ≈ 5,3 / saat);
  Faz 2 oturumunun harcaması ölçülmedi. Faz 3 GPU istemiyor. Ayrıntı: [gpu-budget.md](gpu-budget.md).

---

## 9. Kararlar (ADR'ler, birer satır)

| ADR | Karar |
|---|---|
| 0001 | Dedektör Ultralytics YOLO26 (yedek YOLO11); lisans AGPL-3.0, repo da AGPL |
| 0002 | Yatay kutu (HBB); OBB yalnız ek deney |
| 0003 | Ana etiket seti IR; RGB setiyle de ölçülüp ayrı raporlanır |
| 0004 | 4 kanal: her kanal ayrı TIFF sayfası, R,G,B,T sırası; T kanalı RGB ağırlık ortalamasıyla başlar |
| 0005 | Aydınlatma etiketi RGB parlaklığından türetildi (resmi değil) |
| 0006 | Proje 5 faza bölündü (güvenilirlik + sistem + hareketli hedef + sunum) |
| 0007 | Hedef donanım laptop CPU, p95 ≤ 100 ms, ≤ 1 puan kayıp → E3c · ONNX FP32 sevk edildi |

Faz 3'te ADR 0008 (takip değerlendirmesi) eklenecek.

---

## 10. Açık uçlar ve ertelenenler

- **Kapı düzeltmesi** (yanıt koruması) ve **kalibrasyonda eşitlik sıralaması** → Faz 3 tasarımında.
- **ONNX ≈ 1 puan farkı:** nedeni tam ölçülmedi; dikdörtgen 640×512 export Faz 5 tamponuna.
- **Donmuş kare tespiti** → Faz 4 (video altyapısıyla).
- **RTX 4060:** laptopta var ama hedef donanım CPU; isteğe bağlı satır, Faz 3'te yok.
- **Sınırlamalar:** tek seed (seed varyansı GA'larda yok); val sayıları seçimler yüzünden iyimser;
  M3OT kareleri ardışık video olduğu için oradaki GA'lar iyimser.
- README'nin üstündeki durum satırı hâlâ "`v0.1.0` — Faz 1" diyor; Faz 2 bölümü eklenmiş ama başlık
  güncellenmemiş (küçük düzeltme).

---

## 11. Daha derine inmek istersen — okuma sırası

1. [README.md](../README.md) — sonuç tabloları, 10 dakika.
2. [design/faz3.md](design/faz3.md) — onaylaman gereken belge, 20 dakika.
3. [health.md](health.md) — Faz 2'nin olumsuz sonucu ve nedeni.
4. [experiments.md](experiments.md) — her deneyin hipotezi, kurulumu, sonucu, kararı (tarih sırasıyla).
5. [adr/](adr/) — neden böyle seçtik.
6. [spec.md](spec.md) — bütün proje tanımı (uzun; §1-2 ve §22 yeterli).
