# 0004 — 4 kanallı girdi: biçim, ilk katman ve depolama

- Durum: Kabul edildi (biçim, ilk katman, depolama, 4 kanalın beslenmesi).
- Tarih: 2026-10-06 · Güncelleme: 2026-10-11 (Colab ölçümleri) · 2026-10-14 (H3: 4 kanalın beslenmesi)

## Bağlam
Erken füzyon (E3) RGB ve IR'yi tek bir 4 kanallı görüntü olarak modele veriyor. Ultralytics çok kanallı girdiyi TIFF'ten okuyor. Spike A (`docs/spikes/01-ultralytics-4ch.md`) üç şeyi gösterdi:

- Tek sayfalı 4 örnekli TIFF'te OpenCV 4. kanalı **alfa** sayıyor, kanalları BGR'ye çeviriyor ve RGB'yi T ile çarpıyor (yazılan 10,20,30,40 → okunan 5,3,2,40). Hata vermiyor, sessizce bozuyor.
- Her kanalı ayrı gri sayfa olarak yazınca Ultralytics sayfaları sırayla yığıyor; model yazılan sırayı birebir görüyor.
- COCO ağırlıklarının ilk katmanı kanal 0-2'ye R,G,B sırasıyla kopyalanıyor, kanal 3 rastgele başlıyor.

DroneVehicle'ın 4 kanallı hali, sıkıştırmasız tahminle ~37 GB (spec §23). Colab'ın `/content` diski oturumla birlikte siliniyor, Drive kotası sınırlı.

## Karar
**Biçim (kesin):**
- `TIFF_LAYOUT = "pages"`: her kanal ayrı gri sayfa (`photometric="minisblack"`), `deflate` sıkıştırma.
- `CHANNEL_WRITE_ORDER = RGBT`. Modelin R, G, B, T görmesi şart, çünkü COCO ilk katman ağırlıkları kanal 0-2'de R, G, B bekliyor. Ultralytics'in BGR→RGB çevirmesi sadece 3 kanallı girdide çalışıyor; 4 kanalda sıra dosyadaki sıradır.
- Uygulama: `baykus.data.multichannel.write_rgbt4`. Gidiş-dönüş testi tek sayfalı biçimde başarısız oluyor (bozulmayı yakalıyor).

**İlk katman (kesin, E3 varyantları):**
- `ul_default`: Ultralytics varsayılanı (RGB = COCO, T rastgele). E3b.
- `rgb_mean`: `on_pretrain_routine_end` callback'i T kanalını RGB ağırlıklarının ortalamasıyla doldurur (model + EMA). E3.
- `random`: tüm ilk katman rastgele (alt sınır).

**Depolama (kesin, 2026-10-11 ölçümüyle):**

Ölçüm (`01_data.ipynb`, Colab CPU çalışma zamanı, 2 çekirdek):

| Adım | Süre / boyut |
|---|---|
| Ham veri indirme (Kaggle) | 5 dk 14 sn · 14 GB |
| `verify` + `convert` + `illum` + `rgbt4` + `coco` | **54 dk 28 sn** (adım başına ayrı ölçülmedi) |
| İşlenmiş `rgb` (JPEG q95) | 3,6 GB |
| İşlenmiş `ir` (PNG) | 5,7 GB |
| `rgbt4` (4 sayfalı TIFF, deflate) | **27 GB** (görüntü başına ~950 KB; 3 kanallı JPEG'in ~7 katı) |
| Etiketler + COCO | 0,35 GB |
| `/content` diski (işleme sonu) | 108 GB'ın 71 GB'ı dolu |

Kesinleşme ölçütü (≲ 30 dk) karşılanmadı: her oturumda yeniden üretim kabul edilmedi.

Karar:
- İşlenmiş veri **bir kez** Colab'da üretilir ve kullanıcının **private Kaggle veri setine** yüklenir: `ouzhantarhan/baykus-dv-rgbt` (sürüm 1, 9,84 GB). Her klasör ayrı tar olarak yüklendi (`--dir-mode tar`, 4 dk 54 sn, ~90 MB/sn); Kaggle bunları klasöre açtı: `rgb`, `ir`, `labels_ir`, `labels_rgb`, `coco`, `meta`, `figs`. Düzen: `rgb/images/{train,val,test}/*.jpg`.
- Eğitim oturumları ham veri yerine bu veri setini indirir. Kaggle CLI klasör seçerek indiremediği için veri setinin tamamı iner (~9,8 GB; ham verinin 14 GB'ı 5 dk sürmüştü). `data_*.yaml` oturumda `baykus-data yaml` ile yazılır (yollar `/content` altında).
- **`rgbt4` yüklenmedi.** Bilgi olarak `rgb` + `ir`'den türetilmiş bir kopya; gerekirse aynı kodla yeniden üretilir. E1, E2 ve geç füzyon (E4) buna ihtiyaç duymaz. İlk ihtiyaç H3'te E3.
- Drive'a sadece `runs/` (checkpoint), `mlruns/` ve `results/` yazılır.

**H3'e açık soru: 4 kanal nasıl beslenecek?**
1. E3 oturumunda `baykus-data rgbt4` ile TIFF'leri yeniden üretmek (kesin çalışır; süresi ayrıca ölçülmeli, toplam 54 dakikanın büyük kısmı olabilir).
2. Eğitim sırasında `rgb` + `ir`'yi okuyup bellekte 4 kanala çevirmek. Bu ADR'nin ilk hâlinde reddedilmişti (aşağıda). 27 GB ve uzun üretim süresi bu kararı yeniden değerlendirmek için yeterli sebep. Yama değil, Ultralytics'in belgelenmiş genişletme noktası üzerinden (özel trainer'ın `build_dataset`'i ve `YOLODataset.load_image` alt sınıfı) yapılabilir mi, H3 planında kısa bir spike ile ölçülür. Kanal sırası ve piksel eşliği, TIFF yoluyla birebir aynı olmalı (test).
**Karar (2026-10-14, H3 planı): seçenek 1.** E3 oturumu `rgbt4`'ü `rgb` + `ir`'den yeniden üretir:
`baykus-data rgbt4 --splits train val --workers <çekirdek sayısı>`. Test bölümü final ölçümde
ayrıca üretilir.
- Seçenek 2 reddedildi: eğitim (`build_dataset`), doğrulama (`DetectionValidator`) ve tahmin
  (`predict`) yollarının üçünün de alt sınıflanması gerekir; sağlamlık paketi (bozulmuş val
  kümeleri) ve C++ eşlik testleri zaten diskteki 4 kanallı dosyayı kullanıyor.
- Süre H3 Görev 10'da ölçülür ve buraya yazılır. A100'de 20 dakikayı geçerse seçenek 2 H4'te
  yeniden değerlendirilir.

## Değerlendirilen alternatifler
- **Tek sayfalı RGB+alfa TIFF:** Sessizce bozuk veri (yukarıda). Reddedildi.
- **Tek sayfa, `extrasamples=[0]`:** Çarpma kalkıyor ama sıra B,G,R,T oluyor. İlk katman ağırlıklarıyla uyuşmuyor. Reddedildi.
- **4 örnekli `minisblack` tek sayfa:** Ultralytics okuyamıyor ("corrupt image").
- **Kanalları eğitim sırasında birleştirmek (özel dataloader):** Diskte 4 kanal tutmayı gerektirmez ama Ultralytics'in veri hattını değiştirmek demek; bakım yükü ve hata riski yüksek. Spec §21.2'deki "Ultralytics'in iç API'lerine yama yapma" kuralıyla çelişiyor. *(2026-10-06'da reddedildi; 2026-10-11'de 27 GB ölçümüyle H3'te yeniden değerlendirilmek üzere açıldı, yukarıda.)*
- **Her oturumda yeniden üretim:** 54 dk ölçüldü; GPU oturumunda bu süre kotadan yer. Reddedildi.
- **İşlenmiş veriyi Drive'da tutmak:** Ücretsiz Drive 15 GB ve Gmail ile paylaşılıyor; 9,5 GB sığsa bile `runs/` ve `mlruns/` için yer kalmayabilir. Binlerce küçük dosyayı Drive'dan okumak yavaş; tek arşiv olarak kopyalanırsa çalışır. Kaggle veri setine göre avantajı yok; B planı.
- **Veriyi yerelde işlemek:** 14 GB indirme ve işlenmiş verinin evden Colab'a yüklenmesi gerekir. Reddedildi (kullanıcı kararı).

## Sonuçlar (iyi / kötü / risk)
- İyi: Modelin gördüğü kanal sırası testle güvence altında. Ultralytics'e yama yok.
- Kötü: 4 sayfalı TIFF, tek sayfalıdan biraz daha büyük olabilir ve okuması biraz yavaş olabilir (Spike A'nın açık sorusu; Görev 12 gerçek veride ölçecek).
- İyi: Eğitim oturumunda veri hazırlığı ~1 saatten indirme süresine (birkaç dakika) iner; `verify`/`convert` bir daha çalışmaz, her oturum aynı veriyi görür.
- Kötü: İşlenmiş veri, kodda hata bulunursa yeniden üretilip veri setinin yeni sürümü olarak yüklenmeli (`kaggle datasets version`). Hangi sürümün kullanıldığı run kaydına yazılmalı (H2 Görev 9).
- Risk: Kaggle tar'ları klasöre açtığı için indirilen düzen ilk `02_train` çalıştırmasında `assert` ile doğrulanır (H2 Görev 9).
