# M3OT (veri setleri arası test, E7)

> Kaynak: M3OT, *Scientific Data* 2025 · figshare https://figshare.com/s/01fa8d1163f4e9a5a13a · lisans **CC BY 4.0**.
> Ayrıntılı bulgular: `docs/spikes/04-external-datasets.md` (Spike C + H4 KAPI M).

## Ham düzen (indirilen `M3OT.zip`, 10 GB, sıkıştırmasız)

```
M3OT/<platform 1|2>/<rgb|ir>/<train|val|test>/<dizi>/
    img1/NNNNNN.PNG      640×512 (IR de 3 kanal PNG)
    gt/gt.txt            MOT: frame, track_id, x, y, w, h, flag, class, visibility
    gt/labels.txt        sınıf adı
    seqinfo.ini
M3OT/Annotations/<platform>/<rgb|ir>/<split>_cocoformat.json
```

- RGB ve IR aynı dizi adını taşır; IR dizilerinin adında `T` eki var (`1-03` ↔ `1-03T`).
- Test bölümü: 2 dizi (`1-03`, `2-03`) × 480 kare = **960 RGB-IR çifti**; RGB 2473, IR 2465 kutu; dizi başına 19 iz.
- Drive'da yalnız test bölümü tutulur: `MyDrive/baykus/m3ot_raw/m3ot_test.zip` (1,1 GB).
- İndirme: figshare otomatik istekleri engelliyor (HTTP 403). Tarayıcıda "Copy as cURL" ile alınan çerezlerle Colab'a indirilir; çerezler kişiseldir, hiçbir dosyaya yazılmaz.

## Dönüştürme (`baykus-data m3ot`, FR-D8)

```
baykus-data m3ot --raw /content/m3ot_raw --out /content/m3ot --splits test --workers N
baykus-data rgbt4 --out /content/m3ot --splits test --workers N
```

| Çıktı | İçerik |
|---|---|
| `rgb/images/test/<dizi>_<kare:06d>.jpg` | RGB (PNG → JPG, DroneVehicle düzeniyle aynı) |
| `ir/images/test/<stem>.png` | IR tek kanal |
| `labels_{rgb,ir}/test/<stem>.txt` | YOLO, sınıf 0 (sınıfsız) |
| `coco/test_{rgb,ir}.json` | tek kategori `{"id": 1, "name": "vehicle"}` |
| `meta/m3ot_index.csv` | `split, stem, seq, frame` |
| `meta/m3ot_tracks.csv` | bütün `gt.txt` satırları + `split, modality, seq, frame, stem` (kanca K5: Faz 3 takip, Faz 4 hareketli/duran) |
| `meta/m3ot_stats.json` | çift, dizi, kutu sayıları; tek modalitede kalan kare ve diziler; süzülen satırlar |

- Süzgeç resmi `tools/convert_M3OT_to_yolo.py` ile aynı: `flag == 1`, `sınıf == 1`, `görünürlük > 0,25`. Test bölümünde bütün satırlar bu koşulu sağlıyor (KAPI M).
- Koordinatlar 0 tabanlı kabul edilir (resmi araç gibi), görüntüye kırpılır; çizimle doğrulandı (KAPI M, 241. kare).
- Kare eşleşmesi dosya adındaki numarayla; yalnız iki modalitede de olan kareler alınır.

## Değerlendirme (E7, spec §8.3)

- DroneVehicle'da eğitilen her model M3OT **test** bölümünde, sınıfsız `vehicle` AP: `baykus-eval run ... --dataset m3ot --class-agnostic` (Ultralytics `single_cls=True`, `agnostic_nms=True`; pycocotools'ta bütün tahminler kategori 1).
- Tabloda rgb modu RGB etiketiyle, diğer modlar IR etiketiyle; geç füzyon E4'ün val'de seçilen parametreleriyle (`baykus-fusion apply`), E4b uygulanmaz (aydınlatma etiketi yok).
- Kareler aynı videodan ve ardışık: görüntü düzeyinde bootstrap güven aralığı iyimser (dar) çıkabilir. M3OT kutuları DroneVehicle'dan küçük (< 1024 px²). Mutlak değer değil, **modlar arası sıra** yorumlanır.

## Sonuçlar (E7, 2026-10-16, Colab A100, `05_m3ot`)

Kaynak: notebook çıktısındaki `baykus-eval` / `baykus-fusion apply` satırları (pycocotools, `maxDets=1000`).
JSON dosyaları `results/m3ot/` altına indirilince bu tablo onlarla doğrulanacak.

| Deney | Model | mAP50 IR etiketi | mAP50 RGB etiketi | Tablodaki değer |
|---|---|---|---|---|
| E1 | `2026-10-11_e1-rgb_s_49da0cb` | 0,1009 | **0,8522** | 0,852 (RGB etiketi) |
| E2 | `2026-10-14_e2-ir_s_02c4098` | **0,9806** | 0,0756 | 0,981 |
| E3 | `2026-10-14_e3-early-fusion_s_67ace37` | **0,9706** | 0,1015 | 0,971 |
| E3c | `2026-10-15_e3c-modality-dropout_s_d4e1280` | **0,8889** | 0,1677 | 0,889 |
| E4 | `2026-10-16_e4-late-fusion_s_b346199` (E1 + E2, val'de seçilen ağırlıklar) | **0,9713** | 0,4262 | 0,971 |

- Dönüştürme: 960 çift, 2 dizi, eşleşmeyen dizi ya da kare yok; kutu RGB 2473, IR 2465; süzülen satır yok.
  `rgbt4` 960 görüntü 9 sn (12 işçi). Diskte: `rgb` 128 MB, `ir` 180 MB, `rgbt4` 975 MB, etiketler 3,8 MB × 2.
- Ultralytics taraması: 960 görüntü, **10 arka plan** (IR etiketinde kutusuz kare).
- `dropped=0` (E1-E3c); E4 satırı `dropped` yazmıyor.
- Sıra (IR etiketi): E2 > E4 ≈ E3 > E3c. E3c, DroneVehicle val'deki gibi temiz koşulda geride; burada fark daha büyük (−8,2 puan E3'e göre).
- **RGB-IR etiket hizası (Colab, `meta/m3ot_tracks.csv`, test):** RGB 2473, IR 2465 kutu; aynı kare + iz
  numarasıyla 2455 çift eşleşti (iz numaraları iki modalitede tutarlı). Kutu genişliği ortanca ≈ 12,5 px.
  RGB-IR kutu IoU: ortalama 0,38, ortanca 0,35, çeyrekler 0,24 / 0,50 → çiftlerin ≈ dörtte biri IoU ≥ 0,5.
  Kayma (IR − RGB, piksel): dx ortanca −0,95 (std 3,1); dy ortanca +5,5 (std 10,8, çeyrekler −0,2 / 14,7,
  uçlar −103 / +54). İki dizide de aynı (1-03: IoU 0,35, dy 5,3 · 2-03: IoU 0,36, dy 5,8).
  - Sonuç: iki modalitenin etiketleri 12 px'lik kutular ölçeğinde hizalı değil; asıl kayma düşeyde ve sabit değil.
    Çapraz etiket skorlarının (≈ 0,1) sebebi bu; o sütun model kalitesi hakkında bilgi taşımaz.
  - Tablo kuralı (rgb modu RGB etiketiyle, öbürleri IR etiketiyle) bu yüzden doğru. İki girdiyi birlikte kullanan
    modeller (E3, E3c, E4) M3OT'ta hizasız girdiyle çalışıyor; E3c'nin düşüşü ve E4'ün E2'nin altında kalması
    bununla uyumlu, ama ayrıca ölçülmedi. DroneVehicle'daki E6 hizasızlık sonucuyla da uyumlu (dropout
    hizasızlığa karşı bir şey öğretmiyor).
