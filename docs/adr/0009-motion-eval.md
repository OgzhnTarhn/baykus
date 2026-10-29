# 0009 — Hareket değerlendirmesi (M3OT)

- Durum: Kabul edildi (2026-10-29; protokol ölçümden önce sabitti, test bir kez ölçüldü)
- Tarih: 2026-10-29

## Bağlam
Faz 4'te motor her izi `moving` / `stationary` / `unknown` diye etiketliyor (kamera hareketi çıkarılmış
haritada 15 karelik yol, tasarım faz4 §5.2). Eşlik (Kabul 1-2) "C++ Python'la aynı mı" sorusunu cevaplar;
"karar doğru mu" sorusunu cevaplamaz. M3OT'ta hareket etiketi yok; tasarım §5.3 referansı üründen bilerek
farklı bir etiketleyiciyle (SIFT + homografi, ortalanmış pencere, GT kutuları) üretti ve kör insan
örneklemiyle doğruladı. Protokol, eşikler ve iddia kuralı (§6.3, §6.6) ölçümden önce sabitti. M3OT test
bölümü (`1_03`, `2_03`, 2 × 480 kare) yalnız bu son ölçümde, `f4-freeze` (`041a274`) sonrasında bir kez
kullanıldı.

## Karar
1. **Referans etiket:** `baykus-motion labels` (etiketleyici `motion-labels-v1` kuralları, `041a274`),
   `K = 15`, `v ≤ 0,03` duran, `v ≥ 0,10` hareketli; arası bant, kayıt yoksa kayıtsız, ilk / son `K` kare
   kenar (dışlanır).
2. **Doğrulama:** kör insan örneklemi, 80 düzgün örnek (+ val'de 40 hedefli); ölçüt (a) uyum ≥ %95,
   (b) "emin değilim" ≤ %15, (c) insan-kararlı örneklerin ≥ %80'i etiketli nüfusta (Kabul 4 val, test
   yalnız raporlanır).
3. **İki satır:** (a) **oracle**: M3OT kutuları + ürünün ego'su ve kuralı (modelden bağımsız); (b) **motor**:
   C++ `baykus_sequence` (E3c-M, takip, ego, hareket), GT ile IoU ≥ 0,5 eşleşme.
4. **Ölçüler:** `R_mov`, `R_sta`, `BA = (R_mov + R_sta) / 2` (birincil), kapsam, `P_mov`, `P_sta`,
   kararlı bölüm, karar gecikmesi, insanla doğrudan uyum; ego kalitesi ve harita kayması ayrı
   (`baykus-motion ego-report`).
5. **İddia kuralı (§6.6):** "hareketli / duran ayrımı çalışıyor" yalnız test oracle `BA ≥ 0,85`, motor
   `BA ≥ 0,70` ve `P_mov ≥ 0,80`, test örnekleminde (a) tutarsa. Komut `baykus-motion report` →
   `docs/motion.md`.

## Sonuç (ölçülen)

| Bölüm | Satır | n (hareketli / duran) | R_mov | R_sta | BA | P_mov | Kaynak |
|---|---|---|---|---|---|---|---|
| val | oracle | 9813 (6450 / 3363) | 0,985 | 0,913 | 0,949 | 0,791 | `results/motion/m3ot_dev_oracle.json` |
| val | motor | 9813 (6450 / 3363) | 0,835 | 0,811 | 0,823 | 0,717 | `results/motion/m3ot_dev_engine.json` |
| **test** | oracle | 1325 (1325 / 0) | **0,998** | — | — | 0,699 | `results/motion/m3ot_test_oracle.json` |
| **test** | motor | 1325 (1325 / 0) | **0,918** | — | — | 0,679 | `results/motion/m3ot_test_engine.json` |

Kör örneklem: val (a) 72/73, test (a) 80/80 — ikisi de geçti (`results/motion/label_audit_{val,test}.json`).
Ürünün insanla uyumu: test oracle 80/80, motor 72/80 (8'i `unknown`).

**Yorum:** Test dizilerinde referansa göre duran araç yok; `R_sta` ve `BA` tanımsız, bu yüzden hareket
iddiası kuralı tutmuyor ve README o cümleyi kullanmaz. Ölçülebilen kısım güçlü: hareketli araçların
%92'si motorda hareketli bulunuyor (oracle %99,8). `P_mov` düşük görünüyor çünkü tanım gereği kenar
karelerindeki (dışlanan) `moving` çıktıları da paydada; oracle'da bile 0,70. Val'de (duran araç var) motor
BA 0,823; düşüş tespitten ve takipten (oracle 0,949).

## Değerlendirilen alternatifler
- **Etiketi ürünle aynı yöntemle üretmek:** ego hataları iki tarafta aynı olur, ölçüm iyimser çıkar;
  reddedildi (§5.3).
- **Test sonucuna bakıp eşikleri ayarlamak:** test bir kez kullanılır; reddedildi.
- **Duran araç içeren başka bir test dizisi seçmek:** test bölümü M3OT'un kendi ayrımı; sonuç görüldükten
  sonra dizi değiştirmek ölçütü değiştirmek olur; reddedildi.

## Sonuçlar (iyi / kötü / risk)
- İyi: protokol, eşikler ve iddia kuralı ölçümden önce sabitti; etiket insanla iki bölümde doğrulandı;
  oracle ve motor satırı modelin ve kuralın katkısını ayırıyor.
- Kötü: yalnız iki val ve iki test dizisi; test'te duran araç yok, ayrımın testte doğruluğu ölçülemedi.
  Test örneklemi kör değil sayılmalı: kullanıcı örneklemden önce testte duran araç olmadığını öğrendi
  (`docs/experiments.md` 2026-10-29). 30 kare/sn varsayımı; harita göreli; olayların gerçek dünyadaki
  doğruluğu ölçülmedi.
- Risk: duran araç yoğun sahnelerde motorun `R_sta` değeri yalnız val'den biliniyor (0,811).
