# 0007 — Gecikme hedefi ve sevk kuralı

- Durum: Önerildi (hedef ölçümden önce sabitlendi, 2026-10-17)
- Tarih: 2026-10-17

## Bağlam
NFR-5 gecikme hedefini H5'teki ilk ölçümden sonra bu ADR'ye bırakmıştı. Hedef ölçüm görülerek konursa ölçüme göre kayar; H4'teki E3c / E4 seçim kuralı gibi önce sabitlenir (spec v4.3, H5 planı Spec yorumları 1).

H4'ün seçimi **E4** (E1 + E2 geç füzyon, WBF): kamera kaybında E3c'den anlamlı olarak iyi (K_E4 − K_E3c +0,026 [+0,018; +0,034]) ama cihazda iki model çalıştırıyor. **E3c** tek modelli yedek. **E8-n** (`n` boyutu) yalnız `s` cihazda yeterince hızlı değilse eğitilir (spec §0 v4.2).

## Karar
1. **Hedef donanım:** laptop CPU (Ryzen 7 8845HS, prize takılı, Windows güç modu "En iyi performans"), ONNX Runtime CPU EP.
2. **Ölçüt:** `baykus-bench` toplam süresinin (ön işleme + çıkarım + son işleme, geç füzyonda + WBF) **p95'i**; batch 1, 640×640, 50 ısınma + 500 ölçüm (spec §14).
3. **Hedef: p95 ≤ 100 ms** (≥ 10 kare/sn): operatör ekranında akıcı izleme ve takip için alt sınır; NFR-5'in yönüyle uyumlu.
4. **Doğruluk koşulu:** varyantın val mAP50'si (IR etiketleri) aynı pipeline'ın ONNX FP32'sinden **en fazla 1,0 puan** düşük (nokta tahmin; tam −1,0 geçer).
5. **Aday sırası:** E4 → E3c. Her aday için koşulu sağlayan laptop varyantlarından p95'i en düşük olan alınır; hedefi tutan **ilk** aday ve o varyant sevk edilir.
6. **Hiçbiri tutmazsa:** E8-n eğitilir; hangi modun `n` sürümü olacağı ölçümlerle birlikte kararlaştırılır.
7. Kural `baykus-report`'un JSON'lardan hesapladığı tabloyla uygulanır (`configs/edge/e9_edge.yaml` `rule` bölümü); sonuç `docs/experiments.md`'ye "Sevk kararı" girdisi olarak yazılır ve bu ADR "Kabul edildi"ye geçer.

Kural sonradan değişirse bu metin silinmez; değişiklik gerekçesiyle aşağıya eklenir.

## Değerlendirilen alternatifler
- **Hedefi ilk ölçümden sonra koymak (spec'in ilk hâli):** hedefin ölçülen sayıya göre seçilmesi riski; reddedildi.
- **FPS ortalaması:** kuyruktaki takılmaları gizler; operatör ekranı için p95 daha dürüst.
- **Doğruluk koşulunda güven aralığı:** varyant başına bootstrap Colab CPU'sunda pahalı; H4 kuralının eleme adımı gibi nokta tahmin.

## Sonuçlar (iyi / kötü / risk)
- İyi: sevk kararı ölçümden bağımsız bir kurala dayanıyor; E8-n eğitimi ancak gerçekten gerekirse yapılır.
- Kötü: 100 ms tek bir laptop için konmuş bir sayı; başka donanımda yeniden değerlendirilmeli.
- Risk: laptopun ısınması ya da arka plan yükü p95'i bozabilir; bench kaydı güç durumunu ve sistem bilgisini yazar.

### Ölçümler (H5 Görev 16'da doldurulur)
