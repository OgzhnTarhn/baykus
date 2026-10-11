# Spike B — DroneVehicle Kaggle kopyası

- **Tarih:** 2026-10-10
- **Ortam:** Google Colab (CPU çalışma zamanı), `notebooks/colab/01_data.ipynb` Spike B hücresi. Veri yerel makineye indirilmedi (H1 planı, Görev 4 değişiklik notu).
- **Kaynak:** Kaggle `brendanalvey/visdrone-dronevehicle`
- **Script:** `scripts/spikes/inspect_dronevehicle.py` (görüntü istatistiği 400 rastgele görüntülük örneklemden; XML sayımları tüm dosyalardan)
- **Ham çıktı:** Colab'da `/content/meta/dv_inspect.json` (geçici, git'e girmez). Özet aşağıda.

## Bulgular

| Soru | Bulgu |
|---|---|
| İndirme süresi / toplam boyut | Bu çalıştırmada ölçülmedi. Kaggle sayfasındaki boyut ~13 GB. Gerçek süre ve disk, Görev 16'da `01_data.ipynb`'nin `%%time` çıktılarından ADR 0004'e yazılır. |
| Klasör düzeni | `<kök>/VisDrone-DroneVehicle/<split>/<split>img` (RGB), `<split>imgr` (IR), `<split>label` (RGB etiket), `<split>labelr` (IR etiket). Beklenen düzenle aynı; tek fark zip'in eklediği **`VisDrone-DroneVehicle/` üst klasörü**. `--raw` bu klasörü göstermeli. `01_data.ipynb` bunu otomatik buluyor (`train/` ve `val/` içeren klasör). |
| Dosya sayıları | train 17.990, val 1.469, test 8.980; her split'te 4 klasör eşit sayıda. Toplam **28.439 çift**, resmi sayıyla aynı. Ek dosya (txt/csv/json) yok. |
| Etiket formatı | VOC benzeri XML. `<size>` 840×712×3. Her `<object>`'te `<name>`, `<pose>`, `<truncated>`, `<difficult>` ve bir şekil etiketi. Örnek aşağıda. |
| Şekil etiketleri | `polygon` 887.905, `bndbox` 65.184, **`point` 75**; ayrıca 169.536 `<id>` etiketi (şekil değil, ek alan). polygon + bndbox + point = 953.164 = toplam nesne sayısı; her nesnede tam bir şekil var. |
| Ham sınıf adları | `car` 817.926 · `truck` 48.085 · `bus` 31.924 · `van` 24.643 · `feright_car` 20.303 · `feright car` 10.278 · `feright` 2 · `*` 2 · `truvk` 1 |
| Dosya başına en fazla nesne | **206** (pycocotools varsayılanı `maxDets=100`'ü aşıyor; `MAX_DETS=1000` gerekli) |
| Görüntü boyutları | Örneklemdeki tüm görüntüler **840×712**. RGB her zaman 3 kanal. |
| IR kanal durumu | **Karışık:** örneklemde IR'nin çoğu 3 kanal kopya (`gray3`, 189/209), bir kısmı tek kanal (20/209). Gri olmayan 3 kanallı IR görülmedi. |
| Okunamayan dosya | Örneklemde 0. Bozuk XML 0 (56.878 dosyanın hepsi ayrıştı). |
| Aydınlatma metadata'sı | **Yok.** Ek dosya yok; XML'de de aydınlatma alanı yok. ADR 0005'in parlaklık yöntemi kullanılır. |

Not: Sınıf sayıları **iki etiket setinin toplamı** (56.878 XML = 2 × 28.439). Toplam 953.164 nesne, spec §8.1'deki "~953 bin kutu" ile tutarlı. `car` nesnelerin ~%86'sı; sınıf dengesizliği belirgin.

Not: XML'deki `<folder>` (ör. `5-11`) ve `<filename>` (ör. `DJI_0458`) özgün çekim oturumunu ve kare adını taşıyor gibi görünüyor. Bölümler resmi olduğu için şimdilik kullanılmıyor; E0 raporunda oturum başına dağılım ilginç olabilir.

### Örnek XML (test/testlabel/00001.xml)
```xml
<annotation>
  <folder>5-11</folder>
  <filename>DJI_0458</filename>
  <size><width>840</width><height>712</height><depth>3</depth></size>
  <object>
    <name>car</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <polygon>
      <x1>574</x1><y1>395</y1><x2>604</x2><y2>393</y2>
      <x3>606</x3><y3>469</y3><x4>578</x4><y4>466</y4>
    </polygon>
  </object>
</annotation>
```

## Görev 8-11 için sapmalar (KAPI B)

**Sonuç: Kapı B geçildi; kodda değişiklik gerekmiyor.**

| Kontrol | Kod | Durum |
|---|---|---|
| Klasör adları | `baykus.data.layout.split_dirs` | Aynı. Sadece `--raw` için üst klasör eklenmeli (notebook otomatik buluyor). |
| Sınıf varyantları | `baykus.classes.CLASS_ALIASES` | `feright car`, `feright_car`, `feright`, `truvk` zaten tabloda. `*` (2 nesne) bilinmeyen sınıf olarak raporlanır, atılır. |
| Şekiller | `baykus.data.dronevehicle._points` | `polygon` ve `bndbox` işleniyor. `point` (75 nesne) kutu üretemez; `skipped_shapes` olarak sayılır ve raporlanır. `<id>` yok sayılır. |
| IR kanal | `baykus.data.convert` | Tek kanal ve `gray3` ikisi de işleniyor (ilk kanal alınır); diğer 3 kanallı durumda gri dönüşüm. |
| Görüntü boyutu | `layout.RAW_SIZE = (840, 712)` | Aynı. |
| Kalabalık görüntü | `coco_eval.MAX_DETS = 1000` | 206 nesneyi kapsıyor. |
| Aydınlatma | ADR 0005 | Resmi etiket yok → parlaklıktan türetilir. |

Kaybolan nesneler: `*` 2 + `point` 75 = **77 / 953.164** (%0,008). Önemsiz; E0 raporunda sayısı yazılır.
