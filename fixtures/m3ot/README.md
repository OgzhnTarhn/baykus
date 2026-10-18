# Eşlik fikstürleri (üretici: `baykus-engine fixtures`)

Bölüm: `val`, diziler ['1_08', '2_08']. Seçim yalnız eşlik testi için (tasarım §4.6 sınır kuralı); model ya da eşik seçimi değil.

## Tek kareler
- `1_08_000061`
- `1_08_000181`
- `1_08_000301`
- `1_08_000421`
- `1_08_000542`
- `2_08_000061`
- `2_08_000187`
- `2_08_000301`
- `2_08_000422`
- `2_08_000541`

## Sentetik durumlar

Taban kare: `1_08_000181`. black_ir, noise_ir, shift_ir, black_rgb, darken_rgb, resize_800x600, tiny_33x17.

## Klip

`2_08` kare 270-284; değişkenler `clean`, `shift_all`, `shift_from8` (IR `shift(ir, 20, 0)`).

## Atlananlar (sınır kuralı)
- 1_08 kare 541 atlandı: response 0.0872 r_min'e (0.2450) 0.2'den yakın
- 2_08 kare 181 atlandı: misalign_px 18.667 eşiğe (16.158) 5 px'ten yakın
- 2_08 kare 182 atlandı: misalign_px 18.992 eşiğe (16.158) 5 px'ten yakın
- 2_08 kare 183 atlandı: misalign_px 18.824 eşiğe (16.158) 5 px'ten yakın
- 2_08 kare 184 atlandı: misalign_px 17.472 eşiğe (16.158) 5 px'ten yakın
- 2_08 kare 185 atlandı: misalign_px 19.274 eşiğe (16.158) 5 px'ten yakın
- 2_08 kare 186 atlandı: misalign_px 20.052 eşiğe (16.158) 5 px'ten yakın
- 2_08 kare 421 atlandı: misalign_px 20.963 eşiğe (16.158) 5 px'ten yakın
- sentetik taban 1_08_000061 atlandı: ('shift_ir', ["misalign_px 17.270 eşiğe (16.158) 5 px'ten yakın", "response 0.0960 r_min'e (0.2450) 0.2'den yakın"])
- klip: 855 başlangıç karesi sınır kuralını geçmedi
-   örnek klip 1_08 1: clean 1: response 0.0671 r_min'e (0.2450) 0.2'den yakın
-   örnek klip 1_08 2: clean 1: misalign_px 12.557 eşiğe (16.158) 5 px'ten yakın
-   örnek klip 1_08 3: clean 4: response 0.0584 r_min'e (0.2450) 0.2'den yakın
