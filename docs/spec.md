# Baykuş — RGB + Termal Füzyonlu Hava Görüntüsü Tespiti, Edge için Optimize, Uçtan Uca Sistem

> **Proje adı:** Baykuş · **Repo / paket adı:** `baykus` · **Alt başlık (EN):** *RGB-Thermal aerial vehicle detection on the edge*
> **Sahibi:** Oğuzhan Tarhan · **Tarih:** 2026-10-05 · **Sürüm:** v3.0 (araştırma sonrası yol haritası; eğitim: Colab, kullanıcı çalıştırır · laptop: AMD Ryzen 7 8845HS) · **Süre:** 10 hafta (5 Ekim-13 Aralık 2026) · **Durum:** Onaylandı (2026-10-02)

---

## İçindekiler

0. [v1'den bu yana ne değişti](#0-v1den-bu-yana-ne-değişti)
1. [Tek cümlede proje](#1-tek-cümlede-proje)
2. [Amaç, hedef ve başarı kriterleri](#2-amaç-hedef-ve-başarı-kriterleri)
3. [Neden bu proje — araştırma özeti](#3-neden-bu-proje--araştırma-özeti)
4. [YOLO 101 — bu proje için bilmen gerekenler](#4-yolo-101--bu-proje-için-bilmen-gerekenler)
5. [Teknoloji yığını — sektörde ve GitHub'da ne kullanılıyor, biz ne seçtik](#5-teknoloji-yığını--sektörde-ve-githubda-ne-kullanılıyor-biz-ne-seçtik)
6. [Kapsam ve kapsam dışı](#6-kapsam-ve-kapsam-dışı)
7. [Gereksinimler](#7-gereksinimler)
8. [Veri](#8-veri)
9. [Mimari](#9-mimari)
10. [Eğitim tarifi](#10-eğitim-tarifi)
11. [Füzyon yöntemleri](#11-füzyon-yöntemleri)
12. [Değerlendirme protokolü](#12-değerlendirme-protokolü)
13. [Deney matrisi](#13-deney-matrisi)
14. [Edge optimizasyonu ve ölçüm protokolü](#14-edge-optimizasyonu-ve-ölçüm-protokolü)
15. [C++ motor](#15-c-motor)
16. [Takip (tracking)](#16-takip-tracking)
17. [Operatör ekranı (UI)](#17-operatör-ekranı-ui)
18. [Paketleme, kapalı ağ ve CI](#18-paketleme-kapalı-ağ-ve-ci)
19. [Repo yapısı](#19-repo-yapısı)
20. [Proje yönetimi](#20-proje-yönetimi)
21. [Claude Code ile çalışma düzeni](#21-claude-code-ile-çalışma-düzeni)
22. [10 haftalık takvim ve issue listesi](#22-10-haftalık-takvim-ve-issue-listesi)
23. [Riskler ve B planları](#23-riskler-ve-b-planları)
24. [1. hafta doğrulama listesi ve açık kararlar](#24-1-hafta-doğrulama-listesi-ve-açık-kararlar)
25. [Stratejik not: TEKNOFEST 2027](#25-stratejik-not-teknofest-2027)
26. [Proje bitince: CV, README, tanıtım](#26-proje-bitince-cv-readme-tanıtım)
27. [Kaynaklar](#27-kaynaklar)

---

## 0. v1'den bu yana ne değişti

### v3.0 (2026-10-05)
Kaynak: `docs/research/2026-10-04-derin-arastirma.md` (§1-§10). Kullanıcı kararları: süre 10 hafta; eğitim Colab'da ve kullanıcı tarafından; Claude kod, plan ve notebook hazırlar.

| Konu | v2.3 | **v3.0** | Neden |
|---|---|---|---|
| Süre | 8 hafta | **10 hafta** (5 Eki-13 Ara), MVP H5'te | GPU bütçesi: tam eğitimler ≈ 70-90 T4 saati; haftada en fazla 2 eğitim |
| Kamera kaybı | Sadece ölçülüyordu (E6) | **E3c modality dropout** ile eğitiliyor (FR-T7) | Literatürde standart çözüm; en güçlü mülakat hikâyesi |
| Geç füzyon | Sabit WBF ağırlığı | + **E4b** aydınlatmaya göre ağırlık | GPU gerektirmeyen, literatürle uyumlu iyileştirme |
| Belirsizlik | Yok | **Bootstrap %95 güven aralığı + eşleştirilmiş fark** (FR-V8) | Tek seed ile 1-2 puanlık farklar anlamsız olabilir |
| Veri setleri arası test | HIT-UAV, sadece IR modeli | **M3OT**, tüm modlar (S5, E7) | M3OT RGB-IR, araç, CC-BY-4.0 |
| Takip değerlendirmesi | VT-MOT (belirsiz) | **M3OT** (MOT formatı) | Drone, araç, lisansı net |
| C++ takip | ByteTrack-cpp | **Kendi ByteTrack'imiz** + roboflow/trackers eşliği | ByteTrack-cpp'nin bakımı 2024'te durdu |
| Edge | FP32/FP16/INT8 | + **INT8 head FP32**, + **laptop NPU denemesi** (FR-X6) | INT8 kaybı head'e çok duyarlı; laptopta NPU var |
| Ek hedefler | E10, E11, HIT-UAV, Jetson | + **hareketli/duran araç** (ilk sırada) | TEKNOFEST görevi |

### v2.3 (2026-10-04)
Kaynak: `docs/research/2026-10-04-derin-arastirma.md`. Sadece H1'i etkileyen düzeltmeler işlendi; sonraki haftaların önerileri (modality dropout, bootstrap güven aralığı, INT8'de head FP32, NPU spike'ı, takipçi kararı) ilgili haftanın planında ele alınacak.

| Konu | v2.2 | **v2.3** |
|---|---|---|
| İlk katman başlatması | `rgb_mean \| random` | `ul_default` (Ultralytics varsayılanı: RGB kopya + T rastgele) eklendi. E3 = `rgb_mean`, E3b = `ul_default` (§4.4, §10.3, §13) |
| 4 kanal sırası | Belirsiz | Ultralytics BGR→RGB çevirmesini sadece 3 kanalda yapıyor; dosyaya yazma sırası Spike A'da ölçülür (§4.4, ADR 0004) |
| Hizasızlık | "1-20 px" tahmini | E0 raporunda iki etiket setinin merkez kayması ölçülür (FR-D7, §8.2) |
| Model | YOLO26 | Değişmedi. YOLO27 duyuruldu ama ağırlık yok; ADR 0001'e not |

### v2'ye kadar
Ayrıntılı araştırma üç önemli kararı değiştirdi:

| Konu | v1 | **v2** | Neden |
|---|---|---|---|
| Kutu tipi | Döndürülmüş kutu (OBB) | **Yatay kutu (HBB)**. OBB ek deneye indi. | Savunma sektöründe video tespiti, TEKNOFEST Havacılıkta Yapay Zeka yarışması, en popüler RGB-T GitHub çerçevesi (YOLOv11-RGBT, ~720★), pycocotools/TrackEval gibi standart araçlar ve ByteTrack **yatay kutu kullanıyor**. OBB uzaktan algılamaya özgü kalıyor. |
| Model | "YOLO11 mi YOLO26 mı, sonra karar" | **YOLO26** (Ultralytics'in Ocak 2026'da çıkan en yeni resmi modeli). YOLO11 yedek. | NMS'siz uçtan uca çıktı ile edge'e deploy daha basit, CPU'da daha hızlı. TEKNOFEST 2026 takımları da RGB ve termal için YOLO26 kullanıyor. |
| Erken füzyon | "6 kanal mümkün mü bilinmiyor" | **4 kanal (R, G, B, T)**, Ultralytics'in yerleşik çok kanallı desteğiyle | Ultralytics 8.3.112'den beri `channels` alanı resmi olarak destekleniyor. Termal tek kanal olduğu için 6 yerine 4 kanal yeterli, disk ve bellek de 1/3 azalıyor. |

Ayrıca sektör standardı araçlar plana eklendi:
- **MLflow:** Ultralytics'e yerleşik deney takibi
- **ensemble-boxes (WBF):** Geç füzyon için
- **pycocotools:** Kırılımlı metrikler için
- **ByteTrack:** C++ takipçi kendimiz yazılıyor; Python referansı `roboflow/trackers` (Apache-2.0). (v3.0: ByteTrack-cpp'nin bakımı 2024'te durdu)
- **TrackEval:** Takip metrikleri için
- **uv / ruff / pre-commit:** Python ortamı ve kod kalitesi için

YOLO'yu bilmediğin için **§4'te sıfırdan bir YOLO özeti** var.

---

## 1. Tek cümlede proje

İnsansız hava aracından çekilmiş **normal (RGB) ve termal (IR)** görüntüleri birleştirerek gece-gündüz **araçları tespit edip takip eden**, uç cihazlar için **küçültülüp hızlandırılmış**, **C++ ile çalışan**, **internetsiz kurulabilen** ve sonuçlarını **dürüst, kırılımlı metriklerle** raporlayan uçtan uca bir sistem.

**Ne değil:** Drone takip / anti-drone sistemi değil. Drone burada kameranın taşıyıcısı; tespit edilen şey yerdeki araçlar.

**Kullanım çerçevesi:** Arama-kurtarma, gözetleme, sınır ve tesis güvenliği gibi gözlem amaçlı kullanım. Hedefleme ya da silah entegrasyonu kapsam dışı.

---

## 2. Amaç, hedef ve başarı kriterleri

### 2.1 Amaç
Mezuniyette (Haziran 2027) **Türk savunma sanayii** şirketlerinde (ASELSAN, HAVELSAN, Baykar, STM, TUSAŞ, Roketsan vb.) **ML / Yapay Zeka / Görüntü İşleme Mühendisi** pozisyonları için, ilanlarda en sık aranan becerileri tek bir projede kanıtlamak.

### 2.2 CV'de kapatılacak boşluklar

| Boşluk (şu anki CV) | Bu projede nasıl kapanıyor |
|---|---|
| PyTorch yok (sadece TF/Keras) | Ultralytics (PyTorch) ile eğitim, çok kanallı model, PyTorch tabanlı bozulma testleri |
| Bilgisayarlı görü yok | Nesne tespiti, çok modlu füzyon, takip |
| Edge / model optimizasyonu yok | ONNX, FP16/INT8, TensorRT, ölçülmüş hız-doğruluk tablosu |
| C/C++ var ama somut ML kanıtı yok | C++ çıkarım motoru: ön işleme, çıktı çözümleme, WBF, takip, HTTP servisi |
| Gerçek zamanlı / gömülü sistem dili yok | Gecikme bütçesi, p50/p95 ölçümü, kapalı ağda kurulum |

### 2.3 Başarı kriterleri (ölçülebilir)

- [ ] **S1 — Tekrarlanabilirlik:** README'deki her sayı bir config, bir git commit'i ve bir MLflow çalıştırmasına bağlı. `results/` altındaki JSON'dan yeniden üretilebiliyor.
- [ ] **S2 — Füzyonun katkısı:** Sadece RGB, sadece IR ve en az iki füzyon yöntemi **Gündüz / Gece / Karanlık gece** kırılımlarıyla karşılaştırılmış.
- [ ] **S3 — Etiket seti etkisi:** Aynı tahminler hem IR hem RGB etiketleriyle ölçülmüş, fark raporlanmış.
- [ ] **S4 — Sağlamlık:** En az 4 bozulma türünde doğruluk eğrileri çıkarılmış.
- [ ] **S5 — Veri setleri arası test:** DroneVehicle'da eğitilen RGB, IR ve füzyon modelleri, görmediği M3OT verisinde (RGB-IR, araç) ölçülmüş.
- [ ] **S6 — Edge tablosu:** En az 3 hassasiyet (FP32/FP16/INT8) × en az 2 donanım (laptop CPU, Colab T4) için doğruluk, boyut ve p50/p95 gecikme. INT8'de head FP32 varyantı dahil. Laptop NPU satırı varsa eklenir (FR-X6).
- [ ] **S7 — C++/Python eşliği:** C++ motor çıktısı Python referansıyla tolerans içinde aynı. Bu CI'da test ediliyor.
- [ ] **S8 — Kapalı ağ:** `docker compose up` dış ağa çıkmadan çalışıyor.
- [ ] **S9 — Sunum:** README, demo videosu ve teknik yazı yayında.

---

## 3. Neden bu proje — araştırma özeti

### 3.1 Savunma sanayii ML'de ne arıyor? (Ekim 2026 itibarıyla ilanlar ve programlar)

| Alan | Kimler | Yoğunluk |
|---|---|---|
| **Görüntü işleme: termal + RGB tespit ve takip, edge (Jetson, TensorRT, ONNX, quantization), C++** | Baykar, STM, HAVELSAN, Roketsan, Lasersan, Artron; SSB SAYZEK programı | **En yüksek** |
| Kapalı ağda Türkçe LLM | HAVELSAN (MAIN GPT) | Yükselen |
| Arıza yönetimi / sistem sağlığı | ASELSAN | Orta |
| RL / otonomi / sürü | Baykar, ASELSAN SST | Orta |
| RF / elektronik harp | ASELSAN | Genelde elektrik-elektronik mezunlarına yönelik |

İlanlarda ortak noktalar: **C++ ve Python birlikte**, PyTorch, Linux, gerçek zamanlı sistem, edge'e deploy.

SSB'nin SAYZEK programında yapılan iş, bu projeyle neredeyse aynı: *"havadan termal ve RGB görüntülerde insan/araç tespiti; PyTorch, YOLO, ONNX, TensorRT, Jetson ile eğitim, sıkıştırma ve uç cihaz ölçümü."*

**TEKNOFEST Havacılıkta Yapay Zeka Yarışması** da aynı alanda: hava aracı kamerasıyla taşıt ve insan tespiti, **RGB ve termal**, mAP@0.5 ile değerlendirme, **internetsiz ortamda kendi bilgisayarında çalıştırma zorunlu**. Savunma şirketlerindeki mülakatçılar bu yarışmayı yakından tanır (bkz. §25).

### 3.2 Dürüst durum tespiti
- Bu alan **çok çalışılmış.** Onlarca DroneVehicle makalesi var, Jetson'da çalışan füzyon modelleri var, GitHub'da hazır RGB-T çerçeveleri var.
- RGB-IR hizasızlığı ve iki görüntünün ayrı etiketlenmesi de literatürde biliniyor.

**Projenin iddiası "ilk ben yaptım" değil.** İddia şu üçü:
1. **Derinlik ve dürüstlük:** Tutarlı bir protokol; kırılımlı ve tekrarlanabilir sonuçlar; etiket setinin skoru nasıl etkilediği açıkça gösterilmiş.
2. **Operasyonel sağlamlık:** "Bir kamera bozulunca ne oluyor?" sorusu sistemin içinde ölçülüyor.
3. **Uçtan uca, ölçülmüş, açık kaynak sistem:** Model, C++ motoru, takip, arayüz ve kapalı ağ kurulumu bir arada.

---

## 4. YOLO 101 — bu proje için bilmen gerekenler

> Bu bölüm 1. hafta okuması. Burada geçen her kavramı mülakatta anlatabilecek seviyede bilmelisin.

### 4.1 Nesne tespiti nedir?
Bir görüntü verildiğinde model, içindeki her nesne için şunları döndürür:
- **Kutu:** `(x1, y1, x2, y2)` ya da `(cx, cy, w, h)`
- **Sınıf:** car, truck, …
- **Güven skoru:** 0-1 arası

Sınıflandırmadan farkı: Sınıflandırma "resimde araba var" der, tespit "şu 12 noktada araba var" der.

### 4.2 YOLO nedir?
"You Only Look Once." Görüntüye tek seferde bakıp bütün kutuları birden tahmin eden, gerçek zamanlı çalışan bir model ailesi. Sektörde en yaygın gerçek zamanlı tespit ailesi bu. Bugün fiili standart **Ultralytics** kütüphanesi: YOLOv8 → YOLO11 → **YOLO26** (Ocak 2026).

Bir YOLO modelinin üç parçası var:

```
Görüntü ─► Backbone (omurga) ─► Neck (boyun) ─► Head (kafa) ─► kutular + sınıflar + skorlar
           özellik çıkarır       farklı ölçekleri    her ölçekte
           (kenar, doku, şekil)  birleştirir          tahmin üretir
```

### 4.3 Model boyutları
Her modelin `n, s, m, l, x` boyutları var. Örnekler: `yolo26n.pt` (nano: en küçük, en hızlı, en az doğru), `yolo26s.pt` (small), `yolo26m.pt` (medium). **Edge için genelde n ve s kullanılır.** Bu projede s ana model, n edge karşılaştırması için, m (bütçe varsa) üst sınır için kullanılacak.

### 4.4 Önceden eğitilmiş ağırlıklar ve transfer öğrenme
- `yolo26s.pt` dosyası COCO veri setinde (80 sınıf, günlük fotoğraflar) önceden eğitilmiş.
- Biz bu ağırlıklardan başlayıp kendi verimizle ince ayar (fine-tuning) yapacağız. Sıfırdan eğitmekten çok daha hızlı ve daha iyi sonuç verir.
- **4 kanallı modelde dikkat edilecek nokta:** COCO ağırlıkları 3 kanal için. Ultralytics (8.4.x) 4 kanallı modelde ilk katmanın kanal 0-2'sine COCO'nun RGB ağırlıklarını kopyalıyor, kanal 3'ü (termal) **rastgele** bırakıyor (`ul_default`). Literatürdeki yaygın alternatif termal kanalı RGB ağırlıklarının ortalamasıyla başlatmak (`rgb_mean`). Bunu deney olarak ölçeceğiz (E3, E3b).
- **Kanal sırası:** Kopyalanan ağırlıklar kanal 0-2'nin R, G, B olduğunu varsayar. Ultralytics BGR→RGB çevirmesini sadece 3 kanallı görüntüde yapar; 4 kanallı TIFF dosyadan çözüldüğü sırayla modele gider. Bu yüzden dosyaya hangi sırayla yazılacağı H1 Spike A'da ölçülür (ADR 0004).

### 4.5 Veri formatı (Ultralytics / YOLO formatı)

```
dataset/
├── images/
│   ├── train/  00001.jpg ...
│   └── val/    ...
└── labels/
    ├── train/  00001.txt ...   ← her görüntü için aynı isimde bir .txt
    └── val/    ...
```

Her `.txt` satırı bir nesne. Değerler **görüntü boyutuna oranlanmış (0-1)** haldedir:
```
<sınıf_id> <cx> <cy> <w> <h>
0 0.512 0.334 0.061 0.028
```

Veri seti tanımı (`data.yaml`):
```yaml
path: /content/dv_rgbt
train: images/train
val: images/val
test: images/test
channels: 4            # R, G, B, Termal  (normal RGB için 3; yazılmazsa 3 varsayılır)
names:
  0: car
  1: truck
  2: bus
  3: van
  4: freight_car
```

### 4.6 Eğitim, doğrulama, tahmin, export, takip — 5 temel komut

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")                                   # önceden eğitilmiş model
model.train(data="data.yaml", imgsz=640, epochs=100, batch=16, seed=42)  # eğit
metrics = model.val(data="data.yaml", split="test")          # ölç (mAP vb.)
results = model.predict("ornek.tif", conf=0.25)              # tahmin et
model.export(format="onnx")                                  # ONNX'e çevir
model.track("video.mp4", tracker="bytetrack.yaml")           # video üzerinde takip
```

Eğitim sonunda `runs/detect/train/` klasörü oluşur. İçinde `weights/best.pt` (en iyi model), `results.csv` (her epoch'un metrikleri) ve grafikler bulunur.

### 4.7 Temel kavramlar sözlüğü

| Kavram | Anlamı |
|---|---|
| **IoU** | İki kutunun kesişim alanı / birleşim alanı. Tahmin ile gerçek kutu ne kadar örtüşüyor (0-1). |
| **NMS** | Aynı nesne için üretilen üst üste kutulardan en iyisini tutup diğerlerini atma. **YOLO26 buna ihtiyaç duymuyor** (NMS'siz, uçtan uca). |
| **Precision** | "Bulduklarımın ne kadarı doğru?" |
| **Recall** | "Gerçekte olanların ne kadarını buldum?" |
| **AP / mAP50** | IoU ≥ 0.5 eşiğinde precision-recall eğrisinin altındaki alan, sınıflar üzerinden ortalama. **Ana metriğimiz.** |
| **mAP50-95** | IoU eşiği 0.5'ten 0.95'e kadar ortalaması. Kutu hassasiyetini daha sıkı ölçer. |
| **conf eşiği** | Bu skorun altındaki tahminler atılır. |
| **imgsz** | Modelin girdi boyutu (biz 640). |
| **Epoch** | Eğitim verisinin tamamının bir kez geçilmesi. |
| **Augmentation** | Eğitimde görüntüleri çevirme, kırpma, mozaik gibi çoğaltma. Ultralytics otomatik yapar. Çok kanallı girdide renk (HSV) çoğaltması otomatik kapanır. |
| **Letterbox** | Görüntüyü en-boy oranını bozmadan 640'a sığdırıp kenarları gri dolgu ile doldurma. **C++ tarafında birebir aynısını yazacağız.** |
| **FP32 / FP16 / INT8** | Ağırlıkların sayı hassasiyeti. Düştükçe model küçülür ve hızlanır, doğruluk biraz düşebilir. |
| **ONNX** | Modellerin çerçeveden bağımsız taşınma formatı. PyTorch'ta eğit, C++'ta çalıştır. |
| **TensorRT** | NVIDIA donanımına özel hızlandırıcı. (Intel'in karşılığı OpenVINO; bu projede kullanılmıyor.) |

### 4.8 Öğrenme kaynakları (1. hafta, ~6-8 saat)
1. Ultralytics dokümanları: *Detect*, *Train*, *Val*, *Export*, *Track* sayfaları ve *YOLO26* model sayfası
2. *YOLO26 Training Recipe* rehberi (Ultralytics)
3. Ultralytics'in *Google Colab* entegrasyon rehberi
4. YOLO26 makalesi: yalnızca giriş ve mimari bölümleri
5. Pratik: Colab'da `yolo26n.pt` ile hazır küçük bir veri setinde 5 epoch eğit, val et, ONNX'e çevir. Bütün döngüyü bir kez elle gör.

---

## 5. Teknoloji yığını — sektörde ve GitHub'da ne kullanılıyor, biz ne seçtik

| Katman | Sektörde / GitHub'da yaygın olan | **Seçimimiz** | Not |
|---|---|---|---|
| Tespit modeli | Ultralytics YOLO (v8/11/26), RT-DETR, RF-DETR | **Ultralytics YOLO26** (yedek: YOLO11) | AGPL-3.0, repo da AGPL olur (§7.6) |
| RGB-T çerçevesi | **YOLOv11-RGBT** (~720★; erken/orta/geç/skor füzyonu) | Erken füzyon: **Ultralytics'in yerleşik çok kanal desteği**. Orta füzyon: **YOLOv11-RGBT** ile ek deney | YOLOv11-RGBT eski bir Ultralytics tabanına dayanıyor ve YOLO26'yı desteklemiyor. Ana hatta değil, referans olarak kullanılıyor. |
| Geç füzyon (kutu birleştirme) | **ensemble-boxes** (WBF; Kaggle'da standart) | **ensemble-boxes** (Python), C++'ta kendi WBF uygulamamız | Python sonucu C++ için referans olur |
| Metrikler | Ultralytics `val`, **pycocotools** (COCO standardı) | İkisi birlikte. Kırılımlar pycocotools ile | Çapraz kontrol (FR-V6) |
| Görselleştirme / yardımcı | **supervision** (Roboflow, MIT) | Görselleştirme ve README görselleri | |
| Deney takibi | MLflow, W&B, ClearML | **MLflow** (Ultralytics'e yerleşik, internetsiz çalışır) + `results/*.json` | Savunmada kapalı ağ nedeniyle MLflow tercih edilir |
| Python ortamı | **uv**, poetry, pip | **uv** + `pyproject.toml` | |
| Kod kalitesi | **ruff**, **pre-commit**, pytest | Üçü de | |
| Export / hızlandırma | ONNX, TensorRT, OpenVINO | **ONNX + TensorRT** (Ultralytics `export`). OpenVINO Intel'e özel olduğu ve laptop AMD olduğu için çıkarıldı. | INT8 kalibrasyonu `data` ve `fraction` parametreleriyle |
| C++ çıkarım | ONNX Runtime + OpenCV; referans repolar: **YOLOs-CPP** (YOLO26 destekli, Python eşlik testli), TensorRT-YOLO | **Kendi motorumuz:** ONNX Runtime C++ + OpenCV. Mimari ve eşlik testi yaklaşımı için YOLOs-CPP referans | Motoru kendimiz yazıyoruz, mülakat değeri burada |
| C++ takip | ByteTrack resmi C++ sürümü, ByteTrack-cpp (MIT, Eigen; son commit 2024-04) | **Kendi ByteTrack'imiz** (C++17, Eigen; Kalman + Macar + iki aşamalı eşleşme) | Bakımı durmuş bağımlılık yerine mülakat malzemesi |
| Python takip | Ultralytics `track`, **roboflow/trackers** (Apache-2.0; SORT, ByteTrack, OC-SORT, BoT-SORT + `trackers eval`), BoxMOT (AGPL) | **roboflow/trackers ByteTrack** (C++ eşlik referansı ve HOTA/MOTA/IDF1) | Lisansı temiz, aktif |
| Takip metrikleri | **TrackEval** (HOTA, MOTA, IDF1) | TrackEval | |
| C++ derleme | CMake, vcpkg / Conan, FetchContent | **CMake + FetchContent** (GoogleTest, nlohmann/json, cpp-httplib); ONNX Runtime hazır paket; OpenCV ve Eigen sistem paketi | |
| C++ kalite | clang-format, clang-tidy, GoogleTest | Üçü de | |
| Web arayüzü | React / Next.js, Tailwind, shadcn/ui | **Next.js (App Router) + TypeScript + Tailwind + shadcn/ui** | Senin günlük stack'in |
| Paketleme | Docker (multi-stage), Docker Compose | Aynısı, `internal` ağ ile | |
| CI | GitHub Actions | GitHub Actions | |
| Proje yönetimi | GitHub Projects / Jira | **GitHub Projects + Issues + Milestones** | |

---

## 6. Kapsam ve kapsam dışı

### Kapsamda
- DroneVehicle üzerinde **5 sınıf araç tespiti** (car, truck, bus, van, freight car), **yatay kutu (HBB)**
- Modeller: sadece RGB, sadece IR, erken füzyon (4 kanal), geç füzyon (WBF)
- Kırılımlı değerlendirme, etiket seti karşılaştırması, sağlamlık testi, veri setleri arası test (M3OT; tüm modlar)
- ONNX / TensorRT export; FP32, FP16, INT8; laptop CPU ve Colab T4 ölçümleri
- C++ çıkarım motoru, HTTP API, ByteTrack takibi
- Next.js operatör ekranı
- Docker Compose ile internetsiz kurulum, GitHub Actions CI

### Ek hedefler (zaman kalırsa, bu sırayla)
1. Hareketli / duran araç ayrımı (kamera hareketi telafisi + iz hızı; TEKNOFEST görevi)
2. Orta seviye füzyon (YOLOv11-RGBT ile)
3. OBB deneyi: literatürle doğrudan karşılaştırma için YOLO26-OBB, sadece IR
4. İnsan tespiti ve IR-only ikinci veri setleri arası test (HIT-UAV termal)
5. Gerçek bir Jetson cihazında ölçüm

### Kapsam dışı
- SOTA skor iddiası, makale yayını
- Gerçek drone, kamera veya donanım entegrasyonu
- Kubernetes, bulut deploy
- Hedefleme ya da silah sistemi entegrasyonu
- Laptopta model eğitimi

---

## 7. Gereksinimler

### 7.1 Donanım ve hesaplar

| Gereksinim | Detay | Zorunlu mu |
|---|---|---|
| Laptop | Windows 11 + WSL2. **En az 16 GB RAM**, **≥ 60 GB boş disk** (test verisi, Docker imajları, derleme). CPU: AMD Ryzen 7 8845HS (CPU çıkarımı ONNX Runtime ile). | ✅ |
| Yerel NVIDIA GPU | Gerekmez | ❌ |
| **Google Colab (ücretsiz)** | **Eğitim ortamı.** Tek T4 GPU (16 GB VRAM), ~12-13 GB RAM, geçici disk ~35-78 GB (oturuma göre değişir, oturum kapanınca silinir). Oturum en fazla 12 saat, ~90 dk hareketsizlikte kopar. Haftalık GPU süresi sabit değil (kabaca 15-30 saat) ve GPU her zaman garanti değil. | ✅ |
| **Google Drive** | Checkpoint, MLflow kayıtları ve işlenmiş verinin kalıcı saklanması (ücretsiz 15 GB) | ✅ |
| Kaggle hesabı | **Veri indirmek için.** DroneVehicle kopyası Kaggle'da (14,5 GB). API token'ı ile Colab'a indirilir. | ✅ |
| Kaggle GPU (telefon doğrulamalı) | Yedek GPU: haftada garanti ~30 saat, arka planda çalışma, veri seti indirmeden bağlanır. Colab kotası yetmezse açılır. | ❌ (yedek) |
| Colab Pro | Ücretsiz kota ve Kaggle yedeği yetmezse isteğe bağlı | ❌ |
| GitHub hesabı | Repo, Projects, Actions, Releases | ✅ |
| Jetson Orin Nano vb. | Gerçek edge ölçümü | ❌ (ek hedef) |

### 7.2 Yazılım ortamı

**Laptop (WSL2 Ubuntu 24.04):**

| Katman | Araç | Sürüm |
|---|---|---|
| Python | Python | 3.11 veya 3.12 (Colab ile uyumlu olan) |
| | uv | güncel |
| | ultralytics | **≥ 8.4.x** (YOLO26 ve çok kanal desteği). Kesin sürüm `uv.lock` ile sabitlenir. |
| | torch, torchvision | Ultralytics ile uyumlu, Colab'daki sürüm |
| | onnx, onnxruntime, onnxslim | güncel |
| | pycocotools, ensemble-boxes, supervision | güncel |
| | mlflow | güncel |
| | tifffile, opencv-python, numpy, pandas, matplotlib | güncel |
| | pytest, ruff, pre-commit | güncel |
| | TrackEval | GitHub'dan |
| C++ | GCC ≥ 12, **C++17** | |
| | CMake ≥ 3.22, Ninja | |
| | ONNX Runtime C++ | Python tarafıyla aynı ana sürüm (hazır paket) |
| | OpenCV 4.x, Eigen 3 | apt |
| | GoogleTest, nlohmann/json, cpp-httplib | CMake FetchContent |
| | clang-format, clang-tidy | |
| Web | Node.js 20 LTS veya üstü, pnpm veya npm | |
| | Next.js, TypeScript, Tailwind, shadcn/ui | güncel |
| Altyapı | Docker Desktop (WSL2 backend), Compose v2, Git | |

**Colab:** T4 GPU çalışma zamanı, Google Drive bağlı, Kaggle token'ı **Colab Secrets**'ta (`KAGGLE_API_TOKEN`), TensorRT pip paketi.

### 7.3 Öğrenilmesi gerekenler (zamanlamalı)

| Hafta | Konu | Kaynak |
|---|---|---|
| H1 | §4 YOLO 101, IoU / mAP, YOLO veri formatı | Ultralytics dokümanları |
| H1 | Lisanslar: AGPL-3.0, MIT, CC-BY-4.0 | choosealicense.com |
| H2 | Transfer öğrenme, overfitting, erken durdurma | Ultralytics *Train* sayfası |
| H2 | Bootstrap güven aralığı, eşleştirilmiş karşılaştırma | docs/research §2.6 |
| H3 | Çok modlu füzyon: erken / orta / geç, WBF, modality dropout | YOLOv11-RGBT makalesi (giriş), ensemble-boxes README, Scarf-DETR (giriş) |
| H5 | Quantization: PTQ, kalibrasyon, FP16, QDQ, VNNI | Ultralytics *Export* ve *TensorRT* sayfaları, ORT quantization dokümanı |
| H6 | ONNX Runtime C++ API, letterbox | ORT C++ örnekleri, YOLOs-CPP kaynak kodu |
| H8 | Kalman filtresi, Macar algoritması, ByteTrack, MOTA / IDF1 / HOTA | ByteTrack makalesi, roboflow/trackers kaynak kodu, TrackEval README |

### 7.4 Fonksiyonel gereksinimler (FR)

**Veri (FR-D)**
- **FR-D1:** Bütünlük kontrol scripti. Dosya sayıları, okunamayan görüntüler, eşi olmayan RGB/IR dosyaları, boş etiketler raporlanır. Bozuklar `data/excluded.txt` listesine yazılır.
- **FR-D2:** 100 piksellik beyaz kenarlık kırpılır: 840×712 → 640×512. Etiket koordinatları da kaydırılır.
- **FR-D3:** OBB etiketleri yatay kutuya çevrilir. Dönüşüm: döndürülmüş kutunun 4 köşesini kapsayan en küçük eksen hizalı dikdörtgen, görüntü sınırına kırpılır. **IR ve RGB etiket setleri ayrı ayrı** üretilir.
- **FR-D4:** Aydınlatma etiketi (Gündüz / Gece / Karanlık gece). Resmi kaynaktan alınır. Yoksa türetme yöntemi (ör. RGB ortalama parlaklık eşikleri) belgelenir ve resmi sayılarla (14.478 / 5.468 / 8.493) karşılaştırılarak doğrulanır.
- **FR-D5:** 4 kanallı görüntü üretimi: RGB (3) + IR (1) → tek bir 4 kanallı TIFF. Biçim, Ultralytics'in beklediği kanal düzenine göre 1. haftada kesinleşir. Sıkıştırma: deflate/LZW.
- **FR-D6:** COCO JSON üretimi (her etiket seti için) ve kırılım listeleri (aydınlatma, boyut).
- **FR-D7:** Veri raporu `docs/data-report.md`: sınıf, boyut ve aydınlatma dağılımı; iki etiket seti arasındaki sayı farkı; **aynı nesnenin iki etiket setindeki merkez kayması dağılımı** (p50/p90/p99, histogram); örnek hizasızlık görselleri.
- **FR-D8:** M3OT dönüştürücü (COCO JSON + MOT `gt.txt` → bizim düzen; RGB, IR ve 4 kanal) ve sınıf eşlemesi (§8.3). HIT-UAV dönüştürücü ek hedef.

**Eğitim (FR-T)**
- **FR-T1:** Tek CLI: `uv run baykus-train --config configs/e3_early_fusion.yaml`
- **FR-T2:** Config şeması (§10.3). Bilinmeyen anahtar gelirse hata verir.
- **FR-T3:** MLflow'a otomatik kayıt (Ultralytics entegrasyonu) ve sonunda `results/runs/<run_id>.json`.
- **FR-T4:** Modlar: `rgb`, `ir`, `early_fusion`. Geç füzyon eğitim değil, çıkarım zamanında yapılır (FR-V7).
- **FR-T5:** Seed, `deterministic=True`, git commit ve config hash kaydedilir.
- **FR-T6:** `--smoke` bayrağı: %5 veri, 1 epoch. Pipeline'ı GPU kotasını harcamadan doğrular.
- **FR-T7:** Modality dropout (E3c): config'teki `modality_dropout` ile, Ultralytics'in belgelenmiş özel trainer API'si üzerinden (`DetectionTrainer` alt sınıfı, `preprocess_batch()`). Batch GPU'dayken örnek başına RGB (kanal 0-2) veya T (kanal 3) sıfırlanır; validasyon etkilenmez. Ultralytics'e yama yapılmaz. Sıfırlama fonksiyonu birim testli.
- **FR-T8:** `first_conv_init=rgb_mean`, `on_pretrain_routine_end` callback'iyle model ve EMA'ya birlikte uygulanır (H1 Spike A ile doğrulanır).

**Değerlendirme (FR-V)**
- **FR-V1:** mAP50, mAP50-95, sınıf bazında AP, precision, recall.
- **FR-V2:** Kırılımlar: aydınlatma (3), COCO boyut sınıfları (küçük < 32², orta < 96², büyük), sınıf.
- **FR-V3:** Aynı tahminler iki etiket setine karşı ölçülür.
- **FR-V4:** Sağlamlık paketi (§12.3) tek komutla çalışır, eğri grafikleri üretir.
- **FR-V5:** M3OT veri setleri arası test tek komutla, her mod (rgb, ir, early_fusion, late_fusion) için çalışır.
- **FR-V6:** Ultralytics `val` ile pycocotools sonuçları karşılaştırılır. Fark %1'den büyükse nedeni `docs/` altında açıklanır (iki kütüphane AP'yi farklı enterpolasyonla hesaplar).
- **FR-V7:** Geç füzyon: RGB ve IR modellerinin tahminleri ensemble-boxes WBF ile birleştirilir. Ağırlıklar ve IoU eşiği **sadece val** setinde seçilir. E4b: WBF ağırlıkları aydınlatma kırılımı başına (gündüz / gece / karanlık gece) ayrı seçilir; çıkarımda görüntünün aydınlatma etiketi kullanılır (FR-D4 ile aynı türetme kuralı, sadece RGB'den).
- **FR-V8 — belirsizlik:** Ana tablodaki her mAP50 için görüntü düzeyinde bootstrap %95 güven aralığı (1000 yeniden örnekleme). Model karşılaştırmalarında (ör. E3 − E1, karanlık gece) **eşleştirilmiş bootstrap**: iki model aynı yeniden örneklemle ölçülür, farkın aralığı raporlanır. Uygulama: pycocotools `evaluateImg` sonuçları bir kez hesaplanır, her yeniden örneklemede sadece birikim (accumulate) tekrarlanır. Oyuncak örnekte elle hesaplanan değerle test edilir.

**Export ve edge (FR-X)**
- **FR-X1:** ONNX export (sabit boyut, batch 1, 4 kanal girdi korunur).
- **FR-X2:** Varyantlar: ONNX FP32, ONNX FP16, ONNX INT8 (tam), **ONNX INT8 (head FP32)**; TensorRT FP16 ve INT8 (Colab T4).
- **FR-X6 — NPU denemesi (zaman kutusu 3 saat):** Aynı ONNX modeli laptop NPU'sunda (Ryzen 7 8845HS, XDNA; Ryzen AI Software, AMD Quark ile INT8, ONNX Runtime VitisAI EP, Windows). Kaç düğümün NPU'da çalıştığı ve gecikme raporlanır. Çalışmazsa bulgu olarak yazılır; S6'yı engellemez.
- **FR-X3:** INT8 kalibrasyonu **sadece eğitim setinden** yapılır (`fraction` ile ~500 görüntü). Test setine dokunulmaz.
- **FR-X4:** Her varyantın doğruluğu yeniden ölçülür.
- **FR-X5:** Benchmark: §14 protokolü; p50/p95, boyut, tepe bellek, donanım bilgisi.

**C++ motor (FR-E)**
- **FR-E1:** Letterbox ön işleme (Python ile piksel düzeyinde aynı), ONNX Runtime çıkarımı, YOLO26 uçtan uca çıktı çözümleme, koordinatların orijinal görüntüye geri ölçeklenmesi.
- **FR-E2:** Modlar: rgb, ir, early_fusion (4 kanal) ve late_fusion (iki model + C++ WBF).
- **FR-E3:** YOLO11 yedeği için NMS'li çıktı yolu da desteklenir.
- **FR-E4:** Kendi C++ ByteTrack'imizle takip, oturum (session) bazında. Fikstür dizisinde Python referansıyla (roboflow/trackers ByteTrack, aynı parametreler) eşlik testi: aynı tespit dizisi verildiğinde iz atamaları ID permütasyonu dışında aynı; fark varsa nedeni belgelenir.
- **FR-E5:** HTTP API (§15.3).
- **FR-E6 — eşlik testi:** Fikstür görüntülerde C++ çıktısı ile Python referans JSON'u karşılaştırılır. Kutu sayısı aynı, eşleşen kutularda IoU ≥ 0.99, skor farkı ≤ 1e-3. CI'da çalışır.
- **FR-E7:** Her yanıtta aşama süreleri: pre / infer / post / track / total.

**UI (FR-U)**
- **FR-U1:** Görüntü dizisi veya video oynatma: oynat, durdur, kare kare, hız ayarı.
- **FR-U2:** RGB / IR / yan yana görünüm. Kutu, sınıf, skor ve takip numarası.
- **FR-U3:** Mod seçimi ve canlı gecikme grafiği (son N kare, p50/p95).
- **FR-U4:** "Kamera arızası" simülasyonu: RGB'yi karart, IR'a gürültü ekle, bir modaliteyi kapat. Sonucu anlık gör.

### 7.5 Fonksiyonel olmayan gereksinimler (NFR)
- **NFR-1 Kapalı ağ:** Çalışma zamanında hiçbir dış ağ çağrısı yok (font, CDN, telemetri dahil). Compose ağı `internal: true`. Ultralytics ayarlarında `sync=False`.
- **NFR-2 Tekrarlanabilirlik:** config + commit + seed + donanım + `uv.lock`.
- **NFR-3 Test:** Deterministik kod pytest ve GoogleTest ile test edilir. CI yeşil olmadan merge yok.
- **NFR-4 Kod kalitesi:** ruff, clang-format, eslint. pre-commit kancalarıyla zorunlu.
- **NFR-5 Gecikme hedefi:** H5'teki ilk ölçümden sonra ADR-0006 ile sabitlenir. Yön: s modeli laptop CPU'da FP32 ile gerçek zamanlıya yakın, INT8 ile daha hızlı.
- **NFR-6 Bellek:** Motor, laptopta tek model için makul bir RAM sınırı içinde kalır. Ölçülür ve raporlanır.

### 7.6 Lisans ve veri gereksinimleri
- Ultralytics **AGPL-3.0** olduğu için repo da **AGPL-3.0** olur. README'de "üretim kullanımı Ultralytics Enterprise lisansı gerektirir" notu yer alır.
- Kullanılan kütüphanelerin lisansları: ONNX Runtime (MIT), OpenCV (Apache-2.0), Eigen (MPL-2.0), roboflow/trackers (Apache-2.0; sadece Python referansı), ensemble-boxes (MIT), cpp-httplib (MIT), nlohmann/json (MIT), YOLOs-CPP (AGPL; sadece referans).
- **DroneVehicle'ın lisansı belirtilmemiş:** yeniden dağıtılmaz, atıf yapılır.
- **M3OT ve HIT-UAV CC-BY-4.0:** küçük fikstürler atıfla repoya girebilir.
- Model ağırlıkları GitHub Releases'ta AGPL ile yayımlanır.

---

## 8. Veri

### 8.1 Veri setleri

| Veri seti | İçerik | Erişim | Lisans | Rol |
|---|---|---|---|---|
| **DroneVehicle** | 28.439 RGB-IR çifti, 5 araç sınıfı, OBB (biz HBB'ye çeviriyoruz), modalite başına ayrı etiket, ~953 bin kutu. Gündüz 14.478 / Gece 5.468 / Karanlık gece 8.493 | **Kaggle kopyası** `brendanalvey/visdrone-dronevehicle` (Kaggle API ile Colab'a indirilir). Resmi: GitHub, BaiduYun | Belirtilmemiş, atıf | Ana eğitim ve test |
| **M3OT** (Scientific Data 2025) | 8 video dizisi, 10.790 RGB-IR çifti, 640×512, hizalı (afin + kırpma), 100-120 m irtifa, **sadece araç**, ~220 bin kutu, **modalite başına ayrı etiket**, MOT `gt.txt` + COCO JSON | figshare | **CC-BY-4.0** | **Veri setleri arası test (tüm modlar) ve takip değerlendirmesi (HOTA/MOTA/IDF1)**, CI fikstürleri |
| **HIT-UAV** | 2.898 termal görüntü, 60-130 m irtifa, Person/Car/Bicycle/OtherVehicle, hem HBB hem OBB, YOLO formatı hazır | GitHub | **CC-BY-4.0** | Ek hedef: IR-only ikinci veri setleri arası test, insan tespiti |
| VT-MOT | 582 RGB-T video çifti (gözetleme / drone / el kamerası); drone dizi sayısı ve sınıflar belirsiz | Makale ve GitHub (PFTrack) | Belirsiz | M3OT'a yedek |

### 8.2 Bilinen sorunlar
- Kaggle kopyasında birkaç eğitim görüntüsü bozuk veya eksik → FR-D1.
- RGB-IR hizasızlığı yaygın (1-20 piksel, tahmin) → etiket seti karşılaştırması (E5) ve hizasızlık bozulması (E6). Gerçek dağılım E0 veri raporunda iki etiket setinin merkez kaymasından ölçülür (FR-D7).
- 100 piksellik beyaz kenarlık → FR-D2.
- IR görüntüleri 3 kanallı kopya olarak saklanmış olabilir → 1. haftada kontrol; tek kanal alınır.
- **Disk:** 4 kanallı TIFF'ler sıkıştırılmamış halde ~28k × 640 × 512 × 4 bayt ≈ **37 GB**. Colab'ın geçici diskine sığabilir ama kalıcı değil; Drive'ın ücretsiz 15 GB'ına sığmaz → §23'teki B planları.

### 8.3 Sınıf eşlemesi (veri setleri arası test)

**M3OT (çekirdek):** M3OT'ta tek bir araç sınıfı var. Modelin 5 sınıfı sınıfsız **`vehicle`**'a indirgenir ve sınıfsız AP raporlanır. Her mod için ayrı satır: rgb (RGB etiketiyle), ir (IR etiketiyle), early_fusion ve late_fusion (IR etiketiyle, ana tabloyla tutarlı). M3OT'un gerçek sınıf listesi ve etiket formatı H1 Spike C'de doğrulanır; birden fazla sınıf çıkarsa eşleme ADR'ye yazılır.

**HIT-UAV (ek hedef):** DroneVehicle'da insan yok, bu yüzden test sadece araçlarla yapılır:
- HIT-UAV `Car` → `car`
- HIT-UAV `OtherVehicle` + DroneVehicle'ın `truck/bus/van/freight_car` sınıfları → sınıfsız **`vehicle`**
- `Person`, `Bicycle` ve `DontCare` değerlendirme dışı

Sonuç iki satır olarak raporlanır: `car` AP ve sınıfsız `vehicle` AP. Test sadece **IR modeliyle** yapılır (HIT-UAV sadece termal).

### 8.4 Bölümler
- DroneVehicle'ın resmi train / val / test bölümleri aynen kullanılır.
- Hiperparametre, WBF ağırlıkları ve eşikler **sadece val** üzerinde seçilir. **Test her deneyde bir kez** ölçülür.

### 8.5 İşlenmiş veri düzeni
```
dv_rgbt/
├── rgb/       images/{train,val,test}/*.jpg   labels_rgb/{...}/*.txt
├── ir/        images/{train,val,test}/*.png   labels_ir/{...}/*.txt
├── rgbt4/     images/{train,val,test}/*.tif   (labels → ir etiketlerine sembolik bağ)
├── coco/      {split}_{ir|rgb}.json
├── meta/      illumination.csv, excluded.txt, split_stats.json
└── data_*.yaml
```

---

## 9. Mimari

```
┌──────────────────────── KAGGLE (GPU, eğitim) ─────────────────────────┐
│ notebook: git clone → uv pip install -e python/ → baykus-* komutları    │
│                                                                       │
│ baykus.data ──► baykus.train ──► baykus.eval ──► baykus.export ──► baykus.bench │
│ (FR-D)        (Ultralytics,  (pycocotools,  (ONNX/         (p50/p95) │
│               MLflow)        WBF, robust.)  TensorRT)                 │
│        çıktılar: best.pt · mlruns/ · results/runs/*.json · *.onnx     │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │ GitHub Release (model.onnx)
┌──────────────── LAPTOP / HEDEF MAKİNE (CPU, internetsiz) ─────────────┐
│                                  ▼                                    │
│  ┌──────────────────────────────────────┐  HTTP   ┌────────────────┐  │
│  │ engine (C++17)                       │◄──────► │ ui (Next.js)   │  │
│  │ letterbox → ONNX Runtime → çözümleme │  JSON   │ oynatıcı,      │  │
│  │ → (WBF) → ByteTrack (kendi) → JSON   │         │ kutular, mod,  │  │
│  └──────────────────────────────────────┘         │ gecikme, arıza │  │
│              docker compose (internal network)    └────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

| Bileşen | Tek sorumluluğu | Girdi → Çıktı |
|---|---|---|
| `baykus.data` | Ham veriyi temiz, ortak formata çevirmek | ham veri → YOLO/COCO etiketleri, 4 kanallı TIFF, meta |
| `baykus.train` | Config'e göre eğitmek | config → best.pt, MLflow çalıştırması, run JSON |
| `baykus.eval` | Tahminleri metriklere çevirmek | model + split + etiket seti → metrik JSON |
| `baykus.fusion` | Geç füzyon | iki tahmin listesi → birleşik liste (WBF) |
| `baykus.robustness` | Kontrollü bozulma | görüntü çifti + parametre → bozulmuş çift |
| `baykus.export` | Dağıtılabilir formatlar | best.pt → onnx/engine + doğruluk JSON |
| `baykus.bench` | Gecikme ölçümü | model dosyası → benchmark JSON |
| `baykus.report` | Tablo ve grafik üretimi | results/*.json → markdown tablolar, PNG |
| `engine` | Hızlı, deterministik çıkarım ve takip | görüntü(ler) → tespit + takip JSON |
| `ui` | Operatöre gösterim | engine API → ekran |

---

## 10. Eğitim tarifi

### 10.1 Varsayılan hiperparametreler (başlangıç noktası)

| Parametre | Değer | Not |
|---|---|---|
| model | `yolo26s.pt` | n ve m E8'de |
| imgsz | 640 | Görüntüler 640×512, letterbox ile 640×640 |
| epochs | 100 | H2'deki zaman ölçümüne göre güncellenir |
| patience | 20 | Erken durdurma |
| batch | 16 (T4) veya `-1` (otomatik) | Colab tek T4; bellek hatası olursa 8 |
| optimizer | `auto` | Ultralytics varsayılanı |
| amp | True | |
| seed / deterministic | 42 / True | |
| close_mosaic | 10 | Son 10 epoch mozaiksiz |
| cache | False | Colab RAM'i ~12 GB, RAM cache sığmaz |
| project | `/content/drive/MyDrive/baykus/runs` | Checkpoint'ler Drive'a yazılır; oturum koparsa `resume=True` ile `last.pt`'den devam |
| save_period | 5 | Her 5 epoch'ta bir ara checkpoint |
| workers | 4 | |

### 10.2 GPU bütçesi yöntemi
1. Her yeni deneyde önce `--smoke` (FR-T6) çalıştırılır.
2. Ardından **1 tam epoch** ölçülür → `epoch_süresi`.
3. `tahmini_süre = epoch_süresi × epochs`. Haftalık kalan kota ile karşılaştırılır, `docs/gpu-budget.md`'ye yazılır.
4. Bütçe yetmiyorsa sırasıyla: epoch azalt → `n` modeli kullan → eğitim alt kümesi kullan.

### 10.3 Config şeması (örnek `configs/e3_early_fusion.yaml`)
```yaml
experiment: e3_early_fusion
mode: early_fusion            # rgb | ir | early_fusion
model: yolo26s.pt
data: data/dv_rgbt/data_rgbt4.yaml
label_set: ir                 # eğitim etiketleri: ir | rgb
first_conv_init: rgb_mean     # early_fusion için: ul_default | rgb_mean | random
modality_dropout:             # sadece early_fusion; E3c. Yoksa kapalı.
  p_rgb: 0.0                  # örnek başına RGB kanallarını sıfırlama olasılığı
  p_ir: 0.0                   # örnek başına T kanalını sıfırlama olasılığı (aynı örnekte ikisi birden sıfırlanmaz)
train:
  imgsz: 640
  epochs: 100
  patience: 20
  batch: 16
  seed: 42
  close_mosaic: 10
eval:
  splits: [val, test]
  label_sets: [ir, rgb]
  slices: [illumination, size, class]
tracking:
  mlflow_experiment: baykus
```

---

## 11. Füzyon yöntemleri

| Yöntem | Nasıl | Araç | Avantaj | Dezavantaj |
|---|---|---|---|---|
| **Sadece RGB** (E1) | 3 kanal | Ultralytics | Basit | Gece çöker |
| **Sadece IR** (E2) | Termal 1 kanal (3'e kopyalanabilir) | Ultralytics | Gece iyi | Renk ve doku bilgisi yok |
| **Erken füzyon** (E3) | R, G, B, T → 4 kanal, tek model | Ultralytics `channels: 4` | Tek model, en hızlı füzyon | Hizasızlığa duyarlı. İlk katmanın T kanalı yeniden başlatılıyor. Bir kamera giderse çökmesi beklenir. |
| **Erken füzyon + modality dropout** (E3c) | E3 + eğitimde olasılıkla RGB veya T'yi sıfırlama | Özel trainer (`preprocess_batch`) | Tek model kalır, kamera kaybına dayanıklı olması beklenir | Tam modalitede küçük bir doğruluk bedeli olabilir (ölçülecek) |
| **Geç füzyon** (E4, E4b) | İki ayrı model, kutular WBF ile birleşir; E4b'de ağırlıklar aydınlatmaya göre | ensemble-boxes (Python), kendi WBF (C++) | Bir kamera giderse diğeri çalışır, sağlam | 2 kat hesaplama |
| **Orta füzyon** (E10, ek) | İki omurga, özellik seviyesinde birleşim | YOLOv11-RGBT | Genelde en doğru | Karmaşık, export zor |

---

## 12. Değerlendirme protokolü

### 12.1 Ana metrikler
- **mAP50 (ana)** ve mAP50-95; sınıf AP; precision ve recall (conf=0.25'te)
- Ana tablo **IR etiket setiyle**. RGB etiket seti sonucu ayrı sütunda (E5).
- **Belirsizlik (FR-V8):** Her ana sayının yanında bootstrap %95 güven aralığı. İki model arasındaki fark, eşleştirilmiş bootstrap aralığı sıfırı içermiyorsa "anlamlı" diye yazılır; içeriyorsa "fark ölçüm belirsizliği içinde" denir. Tek seed kullanıldığı için seed varyansı bu aralığa dahil değildir; README'de bu sınırlama açıkça yazılır.

### 12.2 Kırılımlar
- Aydınlatma: Gündüz / Gece / Karanlık gece
- Boyut: COCO standardı (küçük < 32², orta < 96², büyük)
- Sınıf: 5 sınıf

### 12.3 Sağlamlık paketi

| Bozulma | Parametreler | Modalite |
|---|---|---|
| Karartma (gamma) | ×1.0, ×0.5, ×0.25, ×0.1 | RGB |
| Gauss gürültüsü (8-bit) | σ = 0, 5, 10, 20 | IR |
| Bulanıklık (Gauss) | çekirdek 0, 3, 7, 11 | Her ikisi |
| Hizasızlık (IR kaydırma) | 0, 5, 10, 20, 40 piksel | IR |
| Kamera kaybı | RGB = 0 veya IR = 0 | Biri |

Çıktı: her model için "bozulma şiddeti → mAP50" eğrisi ve özet tablo. Mülakattaki ana soru buradan çıkar: *"Termal kamera kapandığında hangi model ne kadar dayanıyor?"*

### 12.4 Sonuç JSON şeması (`results/runs/<run_id>.json`)
```json
{
  "run_id": "2026-10-20_e3-early-fusion_s_a1b2c3",
  "git_commit": "a1b2c3d",
  "config": "configs/e3_early_fusion.yaml",
  "config_hash": "sha256:...",
  "mlflow_run_id": "...",
  "seed": 42,
  "dataset": "dronevehicle",
  "split": "test",
  "label_set": "ir",
  "model": {"family": "yolo26", "size": "s", "mode": "early_fusion", "params_m": null, "gflops": null},
  "format": "pytorch",
  "precision": "fp32",
  "hardware": {"device": "Colab T4", "driver": null},
  "metrics": {
    "map50": null, "map50_95": null, "precision": null, "recall": null,
    "per_class": {"car": null, "truck": null, "bus": null, "van": null, "freight_car": null},
    "per_illumination": {"day": null, "night": null, "dark_night": null},
    "per_size": {"small": null, "medium": null, "large": null},
    "ci95": {"map50": [null, null], "per_illumination": {"day": [null, null], "night": [null, null], "dark_night": [null, null]}}
  },
  "created_at": "2026-10-20T14:00:00Z"
}
```
`null` değerleri gerçek ölçümle doldurulur. **Tahmini bir değer asla yazılmaz.**

---

## 13. Deney matrisi

| ID | Deney | Soru | Hafta |
|---|---|---|---|
| E0 | Veri raporu | Veri ne kadar temiz? Etiket setleri ne kadar farklı? | H1 |
| E1 | Sadece RGB (s) | Normal kamera gece ne kadar çöküyor? | H2 |
| E2 | Sadece IR (s) | Termal tek başına ne kadar iyi? | H3 |
| E3 | Erken füzyon 4 kanal (s), `first_conv_init=rgb_mean` | Basit birleştirme yardım ediyor mu? | H3 |
| E3b | E3 + `first_conv_init=ul_default` (Ultralytics varsayılanı: RGB kopya + T rastgele); bütçe varsa `random` | İlk katman başlatması ne kadar fark ediyor? | H4 |
| E3c | E3 + modality dropout (`p_rgb`, `p_ir` val'de seçilmez; sabit 0.15 / 0.15) | Kamera kaybına karşı eğitmek tam modalitede ne kadara mal oluyor, kayıpta ne kadar kurtarıyor? | H4 |
| E4 | Geç füzyon (E1 + E2, WBF) | İki model mi, tek birleşik model mi? | H4 |
| E4b | E4 + aydınlatmaya göre WBF ağırlıkları (val'de seçilir) | Gece RGB'ye daha az güvenmek geç füzyonu iyileştiriyor mu? GPU gerekmez. | H4 |
| E5 | E1-E4 için iki etiket seti | Skor ne kadar değişiyor? | H3-H4 |
| E6 | Sağlamlık paketi (E1-E4b) | Hangi model hangi arızada dayanıyor? | H3-H4 |
| E7 | M3OT veri setleri arası test (E1-E4, sınıfsız `vehicle` AP) | Hangi mod başka bir drone/sensör verisinde genelleşiyor? | H4 |
| E8 | Boyut: n / s (/ m) için en iyi mod | Doğruluk ve hız ödünleşimi | H5 |
| E9 | Format × hassasiyet × donanım (+ INT8 head FP32, + NPU denemesi) | Edge tablosu | H5 |
| E10 *(ek)* | Orta füzyon (YOLOv11-RGBT) | Karmaşık füzyon farkı kapatır mı? | H4 / tampon |
| E11 *(ek)* | YOLO26-OBB, sadece IR | OBB literatürüyle karşılaştırma | tampon |

---

## 14. Edge optimizasyonu ve ölçüm protokolü

**Export komutları (Ultralytics):**
```python
model.export(format="onnx", imgsz=640, half=False)                       # ONNX FP32
model.export(format="onnx", imgsz=640, half=True)                        # ONNX FP16
model.export(format="engine", imgsz=640, half=True)                      # TensorRT FP16 (T4)
model.export(format="engine", imgsz=640, int8=True, data="data.yaml", fraction=...)  # TensorRT INT8
```
ONNX INT8 için ONNX Runtime'ın `quantize_static` fonksiyonu kullanılır (QDQ, kanal başına ağırlık). Kalibrasyon eğitim setinden yapılır.

**INT8 notları (araştırmadan):**
- YOLO26'da INT8 kaybı detection head'in de INT8'e çevrilip çevrilmemesine çok bağlı. Bir deploy çalışmasında TensorRT'de %5,65, ORT'de %0,72 mAP kaybı raporlanmış. Bu yüzden iki ORT varyantı ölçülür: **tam INT8** ve **head FP32** (`nodes_to_exclude` ile son tespit başlığı düğümleri). Hangi düğümlerin hariç tutulduğu bench JSON'una yazılır.
- INT8'in CPU'da hızlanması komut setine bağlı. VNNI'siz bir CPU'da INT8, FP32'den yavaş ölçülmüş. Ryzen 7 8845HS (Zen 4) AVX-512 VNNI destekliyor; CPU bayrakları bench JSON'una yazılır ve sonuç bu bağlamla raporlanır.
- Laptop NPU denemesi FR-X6'da.

**Ölçüm kuralları:**
- batch 1, sabit 640×640 girdi
- 50 ısınma + **500 ölçüm**; p50, p95 ve ortalama raporlanır
- Ayrı ayrı ölçülen süreler: **ön işleme + çıkarım + son işleme**, ayrıca toplam
- Kaydedilen bilgiler: CPU modeli, thread sayısı, ORT/TRT sürümleri, güç modu (laptop prize takılı)
- Her varyant için doğruluk yeniden ölçülür. Tablo sütunları: format, hassasiyet, boyut (MB), mAP50, ΔmAP50, p50, p95, FPS
- Hızlı genel tablo için Ultralytics `benchmark` fonksiyonu. Rapordaki sayılar ise kendi `baykus.bench` aracımızdan gelir.

---

## 15. C++ motor

### 15.1 Modül yapısı
```
engine/
├── CMakeLists.txt                // FetchContent: GoogleTest, nlohmann_json, httplib
├── cmake/                        // FindONNXRuntime.cmake
├── include/baykus/
│   ├── image_io.hpp              // RGB/IR okuma, 4 kanal birleştirme
│   ├── preprocess.hpp            // letterbox, normalize, HWC→CHW (Python ile birebir)
│   ├── detector.hpp              // ORT oturumu; YOLO26 uçtan uca + YOLO11 (NMS) çözümleme
│   ├── nms.hpp                   // YOLO11 yedeği için klasik NMS
│   ├── wbf.hpp                   // geç füzyon için Weighted Box Fusion
│   ├── kalman.hpp                // sabit hızlı Kalman filtresi (Eigen)
│   ├── assignment.hpp            // Macar algoritması (doğrusal atama)
│   ├── tracker.hpp               // ByteTrack: iki aşamalı eşleşme, iz yaşam döngüsü, oturum yönetimi
│   ├── pipeline.hpp              // mod → bileşen zinciri, zamanlama
│   └── server.hpp                // HTTP API
├── src/…  apps/baykus_server.cpp  apps/baykus_bench.cpp
└── tests/  test_preprocess.cpp test_nms.cpp test_wbf.cpp test_tracker.cpp test_parity.cpp
```

### 15.2 Kendimiz yazacaklarımız ve hazır kullanacaklarımız
- **Biz yazıyoruz (mülakat malzemesi):** letterbox ön işleme, çıktı çözümleme, NMS, WBF, pipeline, zamanlama, HTTP katmanı, eşlik testi.
- **Hazır kullanıyoruz (sektör pratiği):** ONNX Runtime (çıkarım), OpenCV (görüntü), Eigen (lineer cebir), cpp-httplib (HTTP), nlohmann/json (JSON).

### 15.3 HTTP API sözleşmesi
```
GET  /health          → {"status":"ok","model_loaded":true}
GET  /v1/model        → {"name","family","mode","precision","input_size":[640,640],"channels":4,"classes":[...]}
POST /v1/detect       multipart: rgb=<jpg/png>, ir=<jpg/png>   (mod hangisini istiyorsa)
     sorgu: ?conf=0.25&track=true&session=<id>
     → {
         "detections":[{"cls":"car","cls_id":0,"score":0.91,
                        "box":{"x1":301.2,"y1":190.4,"x2":339.8,"y2":214.1},
                        "track_id":7}],
         "timing_ms":{"pre":1.2,"infer":18.4,"post":0.3,"track":0.2,"total":20.1}
       }
POST /v1/session/reset?session=<id>   → takipçiyi sıfırlar
```

**Hata yanıtları:**
- **400:** eksik modalite, boyut uyuşmazlığı, okunamayan görüntü
- **413:** dosya çok büyük
- **500:** iç hata (loglanır)

Motor tek bir hatalı istek yüzünden çökmez.

### 15.4 Eşlik testi (FR-E6) nasıl çalışır
1. Python, fikstür görüntülerde ONNX modeliyle `fixtures/expected/*.json` üretir. Bu dosya repoda tutulur.
2. CI'da C++ motoru aynı görüntü ve aynı ONNX modeliyle çalışır.
3. Kutular IoU ile eşleştirilip toleranslar kontrol edilir.
4. CI'da kullanılan model fikstür amaçlı küçük bir modeldir (`n`, Release'ten indirilir).

---

## 16. Takip (tracking)

- **Algoritma: ByteTrack.** Düşük skorlu tespitleri de eşleştirmede kullanan, Kalman + Macar algoritması tabanlı, sektörde en yaygın hafif takipçi.
- **Python referansı:** Ultralytics `model.track(..., tracker="bytetrack.yaml")`
- **C++:** Kendi ByteTrack uygulamamız (Kalman + Macar + iki aşamalı eşleşme). Motorda oturum başına bir takipçi tutulur. Kalman ve atama birim testli; tam takipçi Python referansıyla eşlik testli (FR-E4).
- **Değerlendirme:** HOTA, MOTA ve IDF1 (TrackEval veya `roboflow/trackers eval`). Veri: **M3OT** (8 drone dizisi, araç, RGB-IR, MOT formatı, CC-BY-4.0).
  - **B planı 1:** M3OT'un hazır kutularıyla ("oracle tespit") sadece takipçi ölçülür. Bu ölçüm modelden bağımsız olduğu için her durumda raporlanır; model tespitiyle olan ölçüm onun yanında durur.
  - **B planı 2:** Takip niceliksel değil, sadece niteliksel olarak (demo videosu) gösterilir.
- **Not:** M3OT de araç ve 100-120 m irtifa; DroneVehicle ile görüş açısı uyumlu. Kare hızı ve dizi uzunlukları H1 Spike C'de kontrol edilir. M3OT uygunsuz çıkarsa yedek VT-MOT.

---

## 17. Operatör ekranı (UI)

- **Yığın:** Next.js (App Router) + TypeScript + Tailwind + shadcn/ui. Çizim için `<canvas>`, gecikme grafiği için hafif bir grafik bileşeni.
- **Sayfa düzeni (tek sayfa):**
  - Üst çubuk: model/mod seçici, conf kaydırıcısı, takip aç/kapa
  - Sol (büyük): oynatıcı; RGB / IR / yan yana sekmeleri; kutular ve takip numaraları
  - Sağ: tespit listesi (sınıf, skor, ID), canlı gecikme grafiği (p50/p95)
  - Alt: **arıza simülasyonu paneli** (RGB karart, IR gürültü, modalite kapat)
- **Veri kaynağı:** Yerel klasördeki görüntü dizileri veya video. Dış ağ çağrısı yok. Fontlar yerelde, CDN yok.
- **Kapsam dışı:** Kullanıcı girişi, veritabanı, çoklu kullanıcı.

---

## 18. Paketleme, kapalı ağ ve CI

**Docker:**
- `engine.Dockerfile`: multi-stage (build: GCC + CMake → runtime: sadece kütüphaneler + binary + model)
- `ui.Dockerfile`: Next.js `output: standalone`
- `docker-compose.yml`: iki servis, `networks: { baykus: { internal: true } }`. UI dış dünyaya yalnızca localhost portundan açılır.

**NFR-1 testi:** Dış ağ kapalıyken (`internal` ağ ve test sırasında internet bağlantısı kesik) sistem açılıyor ve `/health` ile `/v1/detect` çalışıyor.

**CI (`.github/workflows/ci.yml`):**

| İş | Adımlar |
|---|---|
| python | uv sync → ruff → pytest (veri gerektirmeyen testler) |
| engine | apt (OpenCV, Eigen) → ORT indir → CMake build → ctest (eşlik testi dahil) |
| ui | install → eslint → build |
| docker (main'de) | imajlar derlenir |

**Release (`release.yml`):** Tag atıldığında ONNX modeli ve sürüm notlarıyla bir GitHub Release oluşturulur.

---

## 19. Repo yapısı

```
baykus/
├── CLAUDE.md
├── README.md
├── LICENSE                         # AGPL-3.0
├── .pre-commit-config.yaml
├── docs/
│   ├── spec.md                     # bu doküman
│   ├── yolo-101.md                 # §4'ün genişletilmiş hali, kendi notların
│   ├── plans/H1.md … H10.md        # haftalık uygulama planları
│   ├── adr/0001-…md                # karar kayıtları
│   ├── data-report.md
│   ├── experiments.md              # deney günlüğü
│   └── gpu-budget.md
├── python/
│   ├── pyproject.toml  uv.lock
│   ├── src/baykus/{data,train,eval,fusion,robustness,export,bench,report}/
│   └── tests/
├── configs/                        # e1_rgb.yaml … e10_mid_fusion.yaml
├── notebooks/colab/                # 00_setup, 01_data, 02_train, 03_eval, 04_export
├── engine/                         # §15
├── ui/                             # §17
├── fixtures/                       # M3OT (CC-BY) + sentetik görüntüler, expected/*.json
├── results/{runs,bench,figures}/
├── docker/{engine,ui}.Dockerfile
├── docker-compose.yml
└── .github/{workflows,ISSUE_TEMPLATE}/ pull_request_template.md
```

`.gitignore` içinde olmalı: `data/`, `datasets/`, `runs/`, `mlruns/`, `*.pt`, `*.onnx` (fikstür modeli hariç), `*.engine`, `build/`, `node_modules/`, `.venv/`

---

## 20. Proje yönetimi

### 20.1 Araçlar
- **GitHub Projects (Board):** `Backlog → Bu hafta → Devam ediyor → İncelemede → Bitti`
- **Milestone'lar:** `H1`…`H10`
- **Etiketler:** `alan/python`, `alan/engine`, `alan/ui`, `alan/altyapı`, `alan/doküman`, `tür/özellik`, `tür/deney`, `tür/hata`, `tür/spike`, `öncelik/yüksek`

### 20.2 Issue şablonları
**Özellik**
```
## Ne
## Neden (ilgili FR/NFR)
## Kabul kriterleri
- [ ]
## Test planı
```
**Deney**
```
## Hipotez
## Kurulum (config, model, veri, seed)
## Tahmini GPU süresi (smoke + 1 epoch ölçümüne göre)
## Sonuç (run JSON + MLflow bağlantısı)
## Yorum / sonraki adım
```
**Spike (araştırma / doğrulama)**
```
## Soru
## Zaman kutusu (ör. 2 saat)
## Bulgu
## Karar (gerekirse ADR)
```

### 20.3 Git akışı
- `main` korumalı. CI yeşil olmadan merge yok.
- Dal adları: `feat/<no>-kisa-ad`, `exp/<no>-e3-early-fusion`, `fix/…`, `spike/…`
- **Conventional Commits:** `feat(engine): add letterbox preprocess`
- Küçük PR'lar (< 400 satır). Her PR bir issue kapatır.
- Merge öncesi Claude Code ile `/code-review` yapılır.
- Sürümler: `v0.1.0` (H5 MVP), `v0.2.0` (H8), `v1.0.0` (H10)

### 20.4 Tamamlanma tanımı (DoD)
- [ ] Kabul kriterleri karşılandı
- [ ] Testler yazıldı ve geçiyor (deterministik kodda)
- [ ] CI yeşil
- [ ] Gerekiyorsa docs / README güncellendi
- [ ] Deneyse: run JSON + MLflow + `experiments.md` girdisi var
- [ ] **Kodu satır satır açıklayabiliyorum**

### 20.5 Haftalık ritim
- **Pazartesi (15 dk):** Haftanın planını (`docs/plans/H<n>.md`) aç, issue'ları "Bu hafta" sütununa al
- **Her çalışma oturumu:** bir issue
- **Pazar (30 dk):** Değerlendirme — ne bitti, hangi sayılar çıktı, GPU bütçesi, riskler, gelecek haftanın planı

### 20.6 Zaman varsayımı
Haftada **~15 saat**, 10 hafta (~150 saat). 2. hafta sonunda gerçek tempoya ve ölçülen epoch süresine göre takvim güncellenir. Kesme sırası: ek hedefler → E3b → takip model-tespitli ölçümü (oracle kalır) → UI ekstraları → NPU denemesi.

### 20.7 Karar kayıtları (ADR)
| No | Konu | Ne zaman |
|---|---|---|
| 0001 | Ultralytics YOLO26 (AGPL); YOLO11 yedek; MMRotate ve RT-DETR neden değil | H1 |
| 0002 | HBB vs OBB | H1 |
| 0003 | Ana etiket seti IR | H1 |
| 0004 | 4 kanallı TIFF biçimi ve depolama stratejisi | H1 |
| 0005 | Aydınlatma etiketinin kaynağı | H1 |
| 0006 | Gecikme hedefi | H5 |
| 0007 | Takip değerlendirme stratejisi (M3OT model tespiti / oracle / niteliksel) | H8 |

---

## 21. Claude Code ile çalışma düzeni

### 21.1 İlkeler
1. **Claude Code yazar, sen anlarsın.** Her görev sonunda "bunu bana açıkla" de. Anlamadığın kod merge edilmez.
2. **Çekirdek 3 parçayı önce sen yaz, Claude incelesin:** (a) IoU + NMS, (b) letterbox (Python ve C++), (c) WBF. Bunlar mülakatın en çok soru gelen yerleri.
3. **Bir oturum = bir issue = bir dal.** Konu dağılırsa yeni oturum aç.
4. **Plan önce:** Her hafta başında `docs/plans/H<n>.md` Claude Code ile yazılır, sen onaylarsın, sonra görev görev uygulanır.
5. **Deterministik kodda TDD.**
6. **GPU işini Claude Code yapamaz.** Script'leri o yazar, sen Colab'da çalıştırırsın, çıktıyı geri getirirsin.
7. **Kanıtsız sayı yok.** Her metrik bir JSON'dan gelir.

### 21.2 `CLAUDE.md` taslağı
```markdown
# baykus — Claude Code kuralları

## Proje
RGB+IR füzyonlu araç tespiti (YOLO26, HBB) + C++ çıkarım motoru + Next.js operatör ekranı.
Tasarım: docs/spec.md · Haftalık plan: docs/plans/ · Kararlar: docs/adr/

## Komutlar
- Python ortam: `cd python && uv sync`
- Python test/lint: `uv run pytest -q` · `uv run ruff check . && uv run ruff format --check .`
- C++: `cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure`
- UI: `cd ui && npm run lint && npm run build`
- Sistem: `docker compose up --build`

## Kurallar
- Veri ve ağırlıkları ASLA commit etme (data/, runs/, mlruns/, *.pt, *.onnx, *.engine).
- Deterministik kod için önce test yaz (IoU, NMS, WBF, letterbox, dönüşümler, bozulmalar).
- Letterbox / normalize Python ve C++'ta birebir aynı; değişirse fixtures/expected yeniden üretilir.
- Runtime kodu dış ağa çıkmaz; Ultralytics sync=False.
- Metrik = results/**/*.json; kanıtsız sayı yazma, tahmin uydurma.
- Test seti sadece final ölçümde; seçimler val ile.
- Ultralytics'in iç API'lerine yama yapma; gerekiyorsa önce ADR öner.
- Conventional Commits. Bitti demeden önce testleri çalıştır ve çıktıyı göster.

## Colab
Notebook = ince kabuk: Drive bağla → clone → `uv pip install -e python/` → `baykus-* --config ...`. İş mantığı notebook'a yazılmaz.
Eğitim çıktıları (runs/, mlruns/) Drive'a yazılır; uzun eğitimler `resume=True` ile kaldığı yerden devam edebilmeli.
Kaggle token'ı sadece Colab Secrets'tan okunur; hiçbir dosyaya yazılmaz.
```

### 21.3 Bir görevin yaşam döngüsü
```
1. Issue → "Devam ediyor", dal aç
2. Claude Code: "docs/plans/H1.md Görev 4'ü uygula (issue #7). Önce testleri yaz, çalıştır, kırmızı olduğunu göster."
3. Test kırmızı → kod → test yeşil
4. Sen: "Bu kodu satır satır açıkla. OBB→HBB dönüşümünde neden görüntü sınırına kırpıyoruz?"
5. /code-review → düzeltmeler
6. Commit → PR → CI yeşil → merge
```

### 21.4 Colab döngüsü
```
Laptop (Claude Code): config + kod + test → push
   ↓
Colab: 02_train notebook → Drive bağla → git pull → baykus-train --config configs/e3.yaml (önce --smoke)
   (oturum koparsa: aynı komut + resume → Drive'daki last.pt'den devam)
   ↓
Çıktı: best.pt, mlruns/, results/runs/<id>.json  → Google Drive
   ↓
Laptop: JSON + mlruns'u indir → results/'a ekle → `mlflow ui` ile incele → experiments.md → PR
```

### 21.5 Claude Code'a verilmeyecek işler
- Kaggle/Google/GitHub giriş bilgileri, API anahtarları ve token'lar
- Ölçülmemiş sonuçların yazılması
- Çalıştırmadan "test geçti" denmesi

---

## 22. 10 haftalık takvim ve issue listesi

> Her haftanın bir **çıkış kriteri** var. Kriter karşılanmadan sonraki haftaya geçilmez.
> **(S)** = önce sen yaz, Claude incelesin. **[GPU]** = kullanıcı Colab'da çalıştırır; Claude script/notebook/config hazırlar.
> **İş bölümü:** GPU işleri arka planda koşarken CPU tarafı (kod, test, C++) paralel ilerler. C++ motoru eğitilmiş modele ihtiyaç duymadan COCO'lu `yolo26n` ile geliştirilir; eğitilmiş model hazır olunca sadece model dosyası değişir.

| Hafta | Tarih (2026) | Tema | GPU (Colab) |
|---|---|---|---|
| H1 | 5-11 Eki | Temel, doğrulama, veri, ölçüm | — (sadece Spike C denemesi) |
| H2 | 12-18 Eki | Eğitim ve değerlendirme altyapısı, E1 | E1 (+ E2 smoke ve 1 epoch ölçümü) |
| H3 | 19-25 Eki | Tek modalite + erken füzyon | E2, E3 |
| H4 | 26 Eki-1 Kas | Füzyon tamam, sağlamlık, M3OT | E3b, E3c |
| H5 | 2-8 Kas | Edge optimizasyonu · 🚩 MVP | E8-n, TensorRT ölçümleri |
| H6 | 9-15 Kas | C++ motor I: ön işleme, çıkarım, eşlik | — |
| H7 | 16-22 Kas | C++ motor II: WBF, HTTP, C++ bench | — |
| H8 | 23-29 Kas | Takip (kendi ByteTrack) + M3OT takip ölçümü | — |
| H9 | 30 Kas-6 Ara | Operatör ekranı + Docker / kapalı ağ | — |
| H10 | 7-13 Ara | Sürüm, sunum ve tampon | — |

### H1 — Temel, doğrulama, ölçüm
| # | Issue | Kabul kriteri |
|---|---|---|
| 1 | Repo iskeleti: uv, pyproject, ruff, pre-commit, AGPL LICENSE, .gitignore | `uv sync` ve `pre-commit run -a` temiz |
| 2 | CLAUDE.md, docs yapısı, ADR şablonu | Dosyalar repoda |
| 3 | GitHub Projects, milestone'lar, etiketler, issue/PR şablonları | Pano hazır, H1 issue'ları içinde |
| 4 | CI iskeleti (python job) | Boş test bile olsa CI yeşil |
| 5 | Colab `00_setup` notebook'u: Drive bağla + Secrets'tan Kaggle token + veri indir + clone + install | Notebook baştan sona çalışıyor |
| 6 | **Spike:** §24'teki doğrulama listesi | Her madde için bulgu ve karar yazıldı |
| 7 | Kenarlık kırpma + OBB→HBB + YOLO formatı, iki etiket seti (TDD) | Testler yeşil; 5 örnek görselde kutular doğru oturuyor |
| 8 | Bütünlük kontrolü (FR-D1) | `excluded.txt` ve rapor üretildi |
| 9 | 4 kanallı TIFF üretimi (FR-D5) + `data_rgbt4.yaml` | Ultralytics bir 4 kanallı örneği hatasız yüklüyor |
| 10 | Aydınlatma etiketi (FR-D4) + COCO JSON + kırılım listeleri | Sayılar resmi dağılımla tutarlı |
| 11 | IoU **(S)** + pycocotools sarmalayıcı + kırılımlı değerlendirme (TDD) | Oyuncak örneklerde elle hesapla aynı |
| 12 | E0 veri raporu + ADR 0001-0005 | `data-report.md` ve ADR'ler yazıldı |

**Çıkış:** Veri hazır, ölçüm modülü testli, kritik belirsizlikler çözüldü.

### H2 — Eğitim ve değerlendirme altyapısı
| # | Issue | Kabul kriteri |
|---|---|---|
| 13 | Eğitim CLI'ı + config şeması (FR-T1-T2, T5-T6) + MLflow + run JSON | `--smoke` Colab'da uçtan uca çalışıyor; yarıda kesilen eğitim `resume` ile devam ediyor |
| 14 | Özel trainer: modality dropout (FR-T7) + `rgb_mean` callback (FR-T8), TDD | Sıfırlama fonksiyonu ve callback birim testli; smoke koşusunda etkinleşiyor |
| 15 | Değerlendirme CLI'ı (iki etiket seti, kırılımlar) | Bir model için tam JSON üretiyor |
| 16 | Bootstrap güven aralığı + eşleştirilmiş fark (FR-V8), TDD | Oyuncak örnekte elle hesapla aynı; 1000 örneklem val'de makul sürede |
| 17 | Çapraz kontrol: Ultralytics val vs pycocotools (FR-V6) | Fark ≤ %1 ya da açıklaması yazılmış |
| 18 | `baykus.report`: JSON → markdown tablo + grafik (güven aralıklı) | README'ye yapıştırılabilir tablo |
| 19 | [GPU] E1 RGB: smoke → 1 epoch süresi → tam eğitim; E2 smoke + 1 epoch süresi | Run JSON + MLflow; `gpu-budget.md` gerçek sürelerle güncel |

**Çıkış:** Altyapı testli; E1 tabloda (güven aralıklı); GPU bütçesi ölçülmüş süreyle yeniden hesaplanmış.

### H3 — Tek modalite + erken füzyon
| # | Issue | Kabul kriteri |
|---|---|---|
| 20 | [GPU] E2 IR tam eğitim | Run JSON + MLflow |
| 21 | [GPU] E3 erken füzyon (`rgb_mean`) | Run JSON + MLflow |
| 22 | E5 etiket seti tablosu (E1-E3) | Tablo + yorum |
| 23 | Bozulma modülü (TDD) + sağlamlık CLI'ı; E6'nın E1-E3 kısmı | Eğriler; hizasızlık şiddetleri E0'daki kayma dağılımına göre yorumlanmış |

**Çıkış:** E1, E2, E3 aydınlatma kırılımı ve güven aralığıyla tabloda.

### H4 — Füzyon tamam, sağlamlık, veri setleri arası test
| # | Issue | Kabul kriteri |
|---|---|---|
| 24 | [GPU] E3b (`ul_default`) + E3c (modality dropout) | İki run JSON |
| 25 | WBF **(S)** sarmalayıcısı + E4 geç füzyon + E4b aydınlatmaya göre ağırlık (val'de seçilir) | Run JSON'lar; seçilen parametreler notta |
| 26 | E6 tamamı (E1-E4b; kamera kaybı dahil) | E3 vs E3c vs E4 kamera kaybı karşılaştırması güven aralığıyla |
| 27 | M3OT dönüştürücü (FR-D8) + E7 | Her mod için sınıfsız `vehicle` AP |
| 28 | *(ek)* E10 orta füzyon, YOLOv11-RGBT | Run JSON veya "yapılmadı" notu |

**Çıkış:** Ana sonuç tablosu, sağlamlık grafikleri ve M3OT tablosu hazır.

### H5 — Edge optimizasyonu · 🚩 MVP
| # | Issue | Kabul kriteri |
|---|---|---|
| 29 | Export script: ONNX FP32/FP16/INT8 (tam + head FP32), TensorRT FP16/INT8 | Tüm dosyalar üretiliyor |
| 30 | Her varyantın doğruluğu (FR-X4) | Varyant başına metrik JSON |
| 31 | `baykus.bench` (§14 protokolü; CPU bayrakları dahil) | p50/p95 JSON, donanım bilgisiyle |
| 32 | [GPU] E8: en iyi mod için `n` modeli (+ bütçe varsa `m`) | Tablo |
| 33 | NPU denemesi (FR-X6, zaman kutusu 3 saat) | Bulgu notu (çalıştı / çalışmadı ve neden) |
| 34 | README v1 + ADR 0006 + `v0.1.0` tag | README'de ana tablo, aydınlatma, sağlamlık, edge tabloları + 1 grafik |

**Çıkış (MVP):** Repo herkese gösterilebilir. Proje burada kalsa bile güçlü.

### H6 — C++ motor I
| # | Issue | Kabul kriteri |
|---|---|---|
| 35 | CMake iskeleti + bağımlılıklar + CI engine job'u | CI'da boş test yeşil |
| 36 | Letterbox **(S)** + testler + Python ile piksel eşliği | Max piksel farkı 0 |
| 37 | ORT oturumu + YOLO26 çıktı çözümleme (+ YOLO11 NMS **(S)**) | Tek görüntüde doğru kutular |
| 38 | Eşlik testi (FR-E6) CI'da | CI yeşil |

**Çıkış:** C++ motor tek görüntüde Python ile aynı kutuları veriyor; CI'da kanıtlı.

### H7 — C++ motor II
| # | Issue | Kabul kriteri |
|---|---|---|
| 39 | C++ WBF + testler (Python ensemble-boxes ile eşlik) | Testler yeşil |
| 40 | HTTP API + hata yönetimi + entegrasyon testleri | §15.3 sözleşmesi karşılanıyor |
| 41 | `baykus_bench` C++ benchmark'ı → edge tablosuna ek satır | Python ve C++ gecikme karşılaştırması |

**Çıkış:** `POST /v1/detect` çalışıyor, eşlik testi yeşil.

### H8 — Takip
| # | Issue | Kabul kriteri |
|---|---|---|
| 42 | C++ Kalman + Macar algoritması (TDD) | Birim testler yeşil |
| 43 | C++ ByteTrack (iki aşamalı eşleşme, iz yaşam döngüsü) + oturum yönetimi + Python eşlik testi | Fikstür dizisinde roboflow/trackers ile eşlik (FR-E4) |
| 44 | M3OT takip ölçümü (oracle + model tespiti) + ADR 0007 | HOTA/MOTA/IDF1 tablosu |

**Çıkış:** Takip ölçülmüş ve C++'ta eşlik testli. `v0.2.0`.

### H9 — Operatör ekranı ve kapalı ağ
| # | Issue | Kabul kriteri |
|---|---|---|
| 45 | UI iskeleti (Next.js, Tailwind, shadcn) + oynatıcı + canvas | Yerel diziyi oynatıyor |
| 46 | UI ↔ motor bağlantısı; mod seçimi, conf, gecikme grafiği, arıza simülasyonu | FR-U1-U4 |
| 47 | Dockerfile'lar + compose (internal) + NFR-1 testi | İnternetsiz demo çalışıyor |

**Çıkış:** Temiz bir makinede `docker compose up` ile internetsiz demo.

### H10 — Sürüm, sunum ve tampon
| # | Issue | Kabul kriteri |
|---|---|---|
| 48 | CI tamamlama + release workflow + `v1.0.0` | Tag ile Release oluşuyor; model ve notlar içinde |
| 49 | README son hali: GIF, mimari, sonuçlar, "dürüst ölçüm notları", sınırlamalar, atıflar | Gözden geçirildi |
| 50 | 2-3 dakikalık demo videosu | Yayında |
| 51 | Teknik yazı (TR + EN): "Gece görüşünde füzyon ve bozulan kamera" | Yayında |
| 52 | CV ve LinkedIn güncellemesi (§26) | Güncellendi |

**Çıkış:** S1-S9'un hepsi işaretli.

### Ek hedefler (zaman kalırsa, bu sırayla)
1. **Hareketli / duran araç:** kamera hareketi telafisi (ORB + RANSAC homografi) + iz hızı. TEKNOFEST görevine doğrudan hizmet eder.
2. E10 orta füzyon (YOLOv11-RGBT)
3. E11 YOLO26-OBB, sadece IR
4. HIT-UAV: IR-only ikinci veri setleri arası test + insan tespiti
5. Gerçek bir Jetson cihazında ölçüm

---

## 23. Riskler ve B planları

| Risk | Olasılık | Etki | Önlem / B planı |
|---|---|---|---|
| Kaggle kopyasında ciddi eksik | Orta | Yüksek | FR-D1 ile tespit. Resmi BaiduYun'u dene. Olmazsa eksikleri çıkar ve belgele. |
| **4 kanallı TIFF'ler disk sınırını aşar (~37 GB)** | Orta | Yüksek | (1) Sıkıştırılmış TIFF; (2) Colab'ın geçici diskinde her oturum başında yeniden üretmek (süresi H1'de ölçülür); (3) Drive'da sadece sıkıştırılmış ham veriyi (zip) tutup oturum başında açmak; (4) eğitim alt kümesi |
| Ultralytics çok kanal + YOLO26 + detect kombinasyonunda hata | Düşük-Orta | Orta | YOLO11 ile dene. Olmazsa YOLOv11-RGBT ile erken füzyon. |
| Colab GPU kotası yetmiyor veya GPU verilmiyor | **Yüksek** | Orta | §10.2 bütçe yöntemi (epoch azalt → `n` → alt küme); haftada en fazla 2 tam eğitim (§22); eğitimi gece/sabah erken saatlerde başlatmak; **Kaggle GPU yedeği** (telefon doğrulaması, haftada ~30 saat garanti); gerekirse 1 ay Colab Pro |
| Colab oturumu eğitim ortasında kopuyor | **Yüksek** | Orta | Checkpoint'ler Drive'a, `save_period=5`, `resume=True` |
| Kaggle API ile indirme de izin vermiyor | Düşük | Yüksek | Resmi BaiduYun bağlantısı; M3OT (figshare, hesap gerekmez) ile RGB-IR geliştirmeye başla |
| TensorRT kurulumu sorunlu | Orta | Düşük | ONNX Runtime CUDA/TensorRT EP. TensorRT ek hedefe iner. |
| M3OT uygunsuz (format, kare hızı, erişim) | Düşük-Orta | Orta | Yedek VT-MOT; oracle tespitle takip ölçümü veya niteliksel demo (ADR 0007) |
| Windows'ta C++ zorluğu | Yüksek | Orta | Baştan WSL2. Docker imajı tek doğru kaynak. |
| Zaman aşımı | Yüksek | Yüksek | H5 MVP kapısı ve §20.6 kesme sırası |
| AGPL | Düşük | Düşük | Açık portfolyo için uygun. README'de açıkça belirt. |
| Kodu anlamadan ilerlemek | Orta | **Yüksek (mülakat)** | §21.1 ilkeleri ve DoD |

---

## 24. 1. hafta doğrulama listesi ve açık kararlar

**Doğrulanacaklar (Issue #6 — spike):**
- [ ] DroneVehicle Kaggle kopyası: train/val/test dosya sayıları, bozuk dosya sayısı
- [ ] Etiket dosya formatı (XML?); iki etiket setinin gerçek sayı farkı
- [ ] IR görüntüleri 1 kanal mı, 3 kanal kopya mı?
- [ ] Aydınlatma etiketi dosyalarda var mı?
- [ ] Ultralytics güncel sürümünde `channels: 4` ile **YOLO26 detect** 1 epoch eğitim + val + ONNX export sorunsuz mu? İlk katman ağırlıkları nasıl yükleniyor?
- [ ] 4 kanallı TIFF'in Ultralytics'in beklediği biçimi (kanal sırası, çok sayfalı mı)
- [ ] Kaggle API token'ı ile DroneVehicle'ın Colab'a indirilebildiği (telefon doğrulaması olmadan)
- [ ] Colab disk alanı, RAM, veri indirme + işleme süresi
- [ ] Drive'a checkpoint yazma ve `resume` ile devam etme
- [ ] Colab'da TensorRT kurulumu (küçük deneme)
- [ ] M3OT: indirme, sınıf listesi, etiket formatı (MOT + COCO), kare hızı, RGB-IR hizası
- [ ] (ek hedef) HIT-UAV etiket formatı

**Açık kararlar (H1 sonunda ADR olarak kapanacak):**
- [ ] YOLO26 (varsayılan) mı, YOLO11 mi? Spike sonucuna göre.
- [ ] Depolama stratejisi (§23)
- [ ] Aydınlatma etiketinin kaynağı
- [ ] Proje adı

---

## 25. Stratejik not: TEKNOFEST 2027

**Havacılıkta Yapay Zeka Yarışması** bu projeyle neredeyse aynı alanda:
- Hava aracı kamerasıyla **taşıt ve insan tespiti**, RGB ve termal, mAP@0.5 ile değerlendirme
- İnternetsiz ortamda kendi bilgisayarında çalıştırma zorunluluğu
- Mezunlar da katılabiliyor. 2-5 kişilik takımlar.
- 2026 takvimi örnek olarak: başvuru Şubat, rapor Nisan, simülasyon Temmuz, final Eylül
- Ödüller 200-250 bin ₺ aralığında

**Öneri:** Bu proje bittiğinde (Aralık 2026) elinde hazır bir RGB-T tespit, takip ve internetsiz çalışma altyapısı olacak. **Şubat 2027'de GDSC çevrenden bir takımla TEKNOFEST 2027'ye başvurmak** doğal bir sonraki adım. Savunma şirketleri TEKNOFEST derecelerini çok yakından takip eder, Baykar başta olmak üzere. Bu projenin kapsamı değil, ama 1. haftada yarışmanın şartnamesini okuyup sınıf ve format seçimlerini uyumlu tutmak az maliyetli bir yatırım.

---

## 26. Proje bitince: CV, README, tanıtım

### CV maddesi taslağı
> Köşeli parantezler ölçülen gerçek değerlerle doldurulacak. Ölçülmemiş sayı yazılmaz.

**Baykuş — RGB-Thermal Aerial Detection:** YOLO26 ile drone görüntülerinde RGB ve termali birleştiren araç tespiti ve takibi. Füzyon, karanlık gece mAP50'sini tek RGB'ye göre [X] puan artırdı. Dört kamera arızası senaryosunda sağlamlık ölçüldü. Model INT8'e quantize edildi ([A] MB → [B] MB, ΔmAP50 [C]). Kendi yazılan C++ ONNX Runtime motoru, ByteTrack takibiyle birlikte laptop CPU'da [N] ms p50'de çalışıyor. Python ile eşliği CI'da test ediliyor, Docker ile internetsiz kuruluyor. PyTorch, Ultralytics, ONNX, TensorRT, C++17, Next.js.

### README iskeleti
1. Tek cümle + demo GIF
2. Sonuçlar: ana tablo, aydınlatma kırılımı, sağlamlık grafiği, edge tablosu
3. Dürüst ölçüm notları: etiket seti farkı, hizasızlık, sınırlamalar
4. Mimari
5. 5 dakikada çalıştır (`docker compose up`)
6. Tekrarlama (eğitim ve ölçüm komutları)
7. Lisans ve atıflar

### Mülakatta anlatılacak 3 hikâye
1. "Etiket setini değiştirince skor şu kadar değişti; bu yüzden ikisini de raporladım."
2. "Termal kamera kapanınca erken füzyon şuna düştü, geç füzyon ayakta kaldı; bu yüzden sahada şunu seçerdim."
3. "INT8'de şu kadar hız kazandım, şu kadar doğruluk kaybettim; C++ ile Python çıktısının aynı olduğunu CI'da şöyle garanti ettim."

---

## 27. Kaynaklar

**YOLO / Ultralytics**
- YOLO26 model sayfası — https://docs.ultralytics.com/models/yolo26
- YOLO11 vs YOLO26 — https://docs.ultralytics.com/compare/yolo11-vs-yolo26
- YOLO26 makalesi — https://arxiv.org/html/2606.03748v1
- YOLO26 eğitim tarifi — https://docs.ultralytics.com/guides/yolo26-training-recipe
- Detect görevi — https://docs.ultralytics.com/tasks/detect
- Export — https://docs.ultralytics.com/modes/export
- TensorRT entegrasyonu — https://docs.ultralytics.com/integrations/tensorrt
- Track modu — https://docs.ultralytics.com/modes/track
- MLflow entegrasyonu — https://docs.ultralytics.com/integrations/mlflow
- Colab entegrasyonu — https://docs.ultralytics.com/integrations/google-colab
- Çok kanal desteği PR'ı (8.3.112) — https://github.com/ultralytics/ultralytics/pull/20223
- RGB-IR çift girdi PR'ı — https://github.com/ultralytics/ultralytics/pull/18985

**RGB-T çerçeveleri ve C++ / takip**
- YOLOv11-RGBT — https://github.com/wandahangFY/YOLOv11-RGBT · https://arxiv.org/abs/2506.14696
- YOLO-multi-channel (YOLO26 çok kanal fork'u) — https://github.com/Xiaoguaiyaya/YOLO-multi-channel
- YOLOs-CPP — https://github.com/Geekgineer/YOLOs-CPP
- YOLO_OBB_CPP — https://github.com/mohamedsamirx/YOLO_OBB_CPP
- YOLOv8-TensorRT — https://github.com/triple-Mu/YOLOv8-TensorRT
- ByteTrack — https://github.com/FoundationVision/ByteTrack
- ByteTrack-cpp (MIT, bakımı durmuş; sadece okuma referansı) — https://github.com/Vertical-Beach/ByteTrack-cpp
- roboflow/trackers (Apache-2.0) — https://github.com/roboflow/trackers
- Lisans karşılaştırmaları — https://blog.roboflow.com/roboflow-vs-ultralytics/ · https://www.lightly.ai/blog/best-ultralytics-alternatives-in-2026

**Veri setleri**
- DroneVehicle — https://github.com/VisDrone/DroneVehicle · https://arxiv.org/abs/2003.02437
- DroneVehicle Kaggle kopyası — https://www.kaggle.com/datasets/brendanalvey/visdrone-dronevehicle/data
- HIT-UAV — https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · https://arxiv.org/abs/2204.03245
- M3OT — https://pmc.ncbi.nlm.nih.gov/articles/PMC12686387/ · https://figshare.com/s/01fa8d1163f4e9a5a13a
- VT-MOT — https://arxiv.org/abs/2408.00969
- Awesome-RGBT-Fusion — https://github.com/yuanmaoxun/Awesome-RGBT-Fusion

**İlgili çalışmalar**
- TDSF-Net (2026) — https://link.springer.com/article/10.1007/s11554-026-01972-3
- Zayıf hizalı görünür-IR tespit (2026) — https://arxiv.org/pdf/2609.18124
- DOD-SA — https://arxiv.org/pdf/2508.10445
- Spektral güvenilirlik ile sağlam tespit (2026) — https://arxiv.org/pdf/2606.01173
- DroneVehicle aydınlatma bölümleri — https://www.researchgate.net/figure/Visualization-of-images-in-the-DroneVehicle-dataset-a-Day-scene-b-Night-scene-c_fig7_367346713

**Altyapı**
- Colab ücretsiz katman sınırları — https://joshthompson.co.uk/ai/google-colab-2026-guide-free-compute-automations-pro-tips/ · https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025
- Kaggle verisini Colab'a indirme — https://www.freecodecamp.org/news/how-to-download-kaggle-dataset-to-google-colab/
- Colab'da TensorRT kurulumu — https://github.com/pratikratadiya/tensorrt-installation-colab

**Sektör / TEKNOFEST**
- TEKNOFEST Havacılıkta Yapay Zeka — https://teknofest.org/tr/yarismalar/havacilikta-yapay-zeka-yarismasi/
- Yarışma GitHub reposu — https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi
- Baykar Yapay Zeka Mühendisi — https://kariyer.baykartech.com/tr/acik-pozisyonlar/detay/yapay-zeka-muhendisi
- Lasersan Senior Yapay Zeka Mühendisi — https://www.kariyer.net/is-ilani/lasersan-advanced-technology-systems-a-s-senior-yapay-zeka-muhendisi-4502677
- STM Bilgisayarlı Görü Mühendisi — https://www.kariyer.net/is-ilani/stm-savunma-teknolojileri-muhendislik-ve-ticaret-a-uzman-kidemli-bilgisayarli-goru-muhendisi-3131069
- HAVELSAN HTR Gömülü Linux — https://www.kariyer.net/is-ilani/havelsan-teknoloji-radar-htr-ar-ge-gomulu-linux-yazilim-muhendisi-4047658
- ASELSAN 2026 alımları — https://www.isinolsa.com/aselsan-personel-alimi-ilani/
- SAYZEK — https://sayzek.org.tr/en
- SSB EVREN — https://www.aa.com.tr/tr/bilim-teknoloji/savunma-sanayisinin-milli-yapay-zeka-platformu-evren-kullanima-acildi/4070613
