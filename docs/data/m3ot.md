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
