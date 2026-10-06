# 0004 — 4 kanallı girdi: biçim, ilk katman ve depolama

- Durum: Önerildi (biçim ve ilk katman kararı kesin; depolama kararı Colab ölçümüne bağlı)
- Tarih: 2026-10-06

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

**Depolama (önerilen varsayılan, ölçüm bekliyor):**
- Colab'da her oturumda: Kaggle'dan indir → `baykus-data` hattıyla işle → `/content/dv_rgbt` altında tut (`notebooks/colab/01_data.ipynb`).
- Drive'a sadece `runs/` (checkpoint) ve `mlruns/` yazılır.
- **Kesinleşme ölçütü:** `01_data` toplam süresi (indirme + işleme) ≲ 30 dk ve `/content`'te yer kalıyorsa bu karar kabul edilir. Değilse spec §23'teki sıradaki seçenek: işlenmiş veriyi tek arşiv olarak Drive'da tutup oturum başında açmak; o da yetmezse eğitim alt kümesi.

## Değerlendirilen alternatifler
- **Tek sayfalı RGB+alfa TIFF:** Sessizce bozuk veri (yukarıda). Reddedildi.
- **Tek sayfa, `extrasamples=[0]`:** Çarpma kalkıyor ama sıra B,G,R,T oluyor. İlk katman ağırlıklarıyla uyuşmuyor. Reddedildi.
- **4 örnekli `minisblack` tek sayfa:** Ultralytics okuyamıyor ("corrupt image").
- **Kanalları eğitim sırasında birleştirmek (özel dataloader):** Diskte 4 kanal tutmayı gerektirmez ama Ultralytics'in veri hattını değiştirmek demek; bakım yükü ve hata riski yüksek. Spec §21.2'deki "Ultralytics'in iç API'lerine yama yapma" kuralıyla çelişiyor.
- **İşlenmiş veriyi kalıcı olarak Drive'da tutmak:** Drive'dan binlerce küçük dosya okumak Colab'da çok yavaş; ancak tek arşiv olarak kopyalanıp açılırsa anlamlı. B planı olarak duruyor.

## Sonuçlar (iyi / kötü / risk)
- İyi: Modelin gördüğü kanal sırası testle güvence altında. Ultralytics'e yama yok.
- Kötü: 4 sayfalı TIFF, tek sayfalıdan biraz daha büyük olabilir ve okuması biraz yavaş olabilir (Spike A'nın açık sorusu; Görev 12 gerçek veride ölçecek).
- Risk: Her oturumda yeniden üretim, GPU oturumunun ilk dakikalarını yiyebilir. Bu yüzden `01_data` CPU çalışma zamanında koşar; GPU oturumunda ise tekrar süresi `docs/gpu-budget.md`'ye yazılır.
