# 0003 — Ana etiket seti: IR

- Durum: Kabul edildi
- Tarih: 2026-10-05

## Bağlam
DroneVehicle her görüntü çifti için iki ayrı etiket seti veriyor: RGB ve IR. RGB-IR hizasızlığı ve gece RGB'nin görünmez olması yüzünden iki set farklı; IR setinde daha fazla nesne var (ör. car: RGB 389.779, IR 428.086).

## Karar
- Eğitim ve ana tablo **IR etiket setiyle**.
- Aynı tahminler RGB etiket setiyle de ölçülür ve ayrı sütunda raporlanır (E5, FR-V3).
- İki set arasındaki merkez kayması E0 veri raporunda ölçülür (FR-D7).

## Değerlendirilen alternatifler
- **RGB etiket seti:** Gece görünmeyen araçları içermiyor; gece kırılımı ölçümünü yanıltır.
- **İki setin birleşimi:** Hizasızlık yüzünden aynı nesne iki kez sayılabilir; birleştirme kuralı ayrı bir araştırma konusu.
- **"Daha çok etiketi olan modalite":** Literatürde yaygın (ör. UAVD-Mamba); pratikte çoğunlukla IR'a denk geliyor ama görüntüden görüntüye değişiyor, tutarsız.

## Sonuçlar (iyi / kötü / risk)
- İyi: Gece ve karanlık gece ölçümü anlamlı; literatürdeki yaygın tercihle uyumlu.
- Kötü: RGB modelleri, RGB'de görünmeyen nesneler yüzünden IR setinde dezavantajlı. Bu yüzden E5 zorunlu.
- Risk: Etiket seti seçimi skorları birkaç puan oynatabilir. Mülakatta anlatılacak ilk hikâye tam olarak bu (spec §26).
