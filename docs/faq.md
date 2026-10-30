# Baykuş: teknik sorular ve cevaplar

Projeyi inceleyen birinin en çok soracağı sorular, kısa cevapları ve kaynak belgeleri. Sayıların hepsi
`results/**/*.json` dosyalarından; ayrıntılı tablolar [details.md](details.md).

## Model ve veri

**Neden iki kamera? Tek IR yetmiyor mu?**
IR tek başına güçlü (test mAP50 81,0), RGB karanlıkta çöküyor (karanlık gecede 66,9). Füzyon temiz veride
yalnız biraz daha iyi (E3c 82,5), asıl faydası bir kamera bozulduğunda ortaya çıkıyor: sistem kalan kamerayla
çalışmaya devam ediyor. [details.md](details.md#ana-tablo-test-seti-bir-kez-ölçüldü)

**Erken füzyon ile geç füzyon arasındaki fark ne? Neden erken füzyon seçildi?**
Erken füzyon (E3) RGB ve IR'yi 4 kanallı tek görüntü olarak tek modele verir. Geç füzyon (E4) iki ayrı model
çalıştırıp kutuları WBF ile birleştirir. E4 bulanıklık ve kamera kaymasına karşı daha sağlam, ama laptop
işlemcisinde iki kat yavaş (p95 132 ms). Seçim kuralı ölçümden önce yazıldı: p95 ≤ 100 ms ve en çok 1 puan
doğruluk kaybı. Bu kurala göre E3c sevk edildi. [ADR 0007](adr/0007-latency-target.md)

**Modality dropout nedir, ne kazandırdı?**
Eğitimde her örnekte %15 olasılıkla RGB, %15 olasılıkla IR kanalları sıfırlanıyor (ikisi birden değil). Model
böylece tek kameraya güvenmeyi öğreniyor. Val'de IR kamera kapandığında normal erken füzyon 12,8'e düşerken
dropout'lu E3c 72,0'da kalıyor; temiz veride kayıp 0,6 puan. [robustness.md](robustness.md)

**4 kanalı Ultralytics'e nasıl verdiniz? İlk katman ağırlıkları ne oldu?**
Görüntüler RGBT sırasıyla 4 sayfalı TIFF olarak saklanıyor; `data.yaml`'da `channels: 4`. İlk konvolüsyonun
R, G, B kanalları COCO ağırlıklarını koruyor, T kanalı bu üçünün ortalamasıyla başlatılıyor (`rgb_mean`).
Ultralytics'e yama yapılmadı; dropout belgelenmiş trainer alt sınıfıyla eklendi.
[ADR 0004](adr/0004-four-channel-storage.md), [ADR 0001](adr/0001-ultralytics-yolo26.md)

**DroneVehicle'da döndürülmüş kutular var; neden yatay kutu (HBB)?**
Takipçi (ByteTrack), değerlendirme araçları (pycocotools, TrackEval) ve C++ motorundaki NMS / WBF yatay
kutu üzerine kurulu; OBB her bileşende ek iş demekti. OBB'ler en küçük kapsayan dikdörtgene çevrildi. Bedeli:
sonuçlar literatürdeki OBB tablolarıyla doğrudan kıyaslanamıyor ve yoğun otoparklarda kutular üst üste
binebiliyor. [ADR 0002](adr/0002-hbb-vs-obb.md)

**RGB ve IR etiketleri farklıysa hangisini kullandınız?**
IR etiketlerini. İki set RGB-IR hizasızlığı ve gece RGB'de görünmeyen araçlar yüzünden farklı; IR setinde
daha çok nesne var (ör. car: RGB 389.779, IR 428.086). Aynı tahminler RGB etiketleriyle de ölçülüp ayrı
raporlandı. [ADR 0003](adr/0003-ir-primary-label-set.md)

**M3OT'ta sonuçlar neden önce kötüydü, nasıl düzeldi?**
M3OT'ta araçlar çok küçük (≈ 13 px) ve kalabalık; E3c gerçek araçların yalnız %23'ünü buluyordu. Model
DroneVehicle + M3OT train ile ince ayarlandı (E3c-M). M3OT'ta araç türü etiketi olmadığı için türler E3c'nin
iz boyunca oylarından alındı ve kör insan denetimiyle sınandı (doğruluk 0,92). Sonuç: M3OT test AP50 86,0 →
98,8, DroneVehicle val'de kayıp yok. [ADR 0010](adr/0010-e3c-m3ot.md)

## Ölçüm

**Sayılarınıza neden güveneyim?**
Üç kural: (1) bütün seçimler val'de yapıldı, test seti en sonda bir kez ölçüldü; (2) kabul ölçütleri ve
eşikler ölçümden önce yazıldı ve kayıtlı; (3) her sayı commit'i, config'i ve seed'iyle bir JSON dosyasında.
Güven aralıkları görüntü düzeyinde bootstrap ile. Kısıtı da açık: her model tek seed'le eğitildi.

**Başarısız olan bir şey oldu mu?**
Evet, iki tane yazıldı. (1) Faz 2'nin sensör sağlığı kapısı bozuk IR'de +25,7 puan kazandırdı ama temiz
veride 3,3 puan kaybettirdi, önceden yazılan kurala göre başarısız sayıldı; neden de bulundu: yanlış alarmların
%80'i karanlık gece karesi. [health.md](health.md) (2) Faz 4'te "hareketli / duran ayrımı çalışıyor" iddiası
kurulamadı, çünkü test dizilerinde duran araç yoktu. [ADR 0009](adr/0009-motion-eval.md)

**Hareket kararının doğruluğunu etiket olmadan nasıl ölçtünüz?**
M3OT'ta hareket etiketi yok. Referans etiket üründen bilerek farklı bir yöntemle üretildi (SIFT + homografi,
ortalanmış pencere, gerçek kutular), yoksa ürünün hataları referansta da olurdu. Referans kör insan
örneklemiyle doğrulandı (test 80/80). İki satır raporlandı: gerçek kutularla "oracle" (yalnız hareket
kuralını ölçer) ve tam motor (tespit ve takip hataları dahil). [ADR 0009](adr/0009-motion-eval.md)

## Motor ve sistem

**Neden C++ motor? Python yetmez miydi?**
Hedef, sahada internetsiz ve tek bilgisayarda çalışan bir ürün. C++ motor ONNX Runtime ve OpenCV ile tek bir
Docker imajında çalışıyor. Python referans olarak kaldı: her C++ adımı (letterbox, tespit, sağlık, takip,
hareket) Python'la aynı sonucu vermek zorunda ve bu CI'da her commit'te sınanıyor. [system.md](system.md)

**C++ ve Python'un aynı sonucu verdiğini nasıl kanıtlıyorsunuz?**
Python referansı sabit girdilerden "fikstür" dosyaları üretiyor; C++ testleri aynı girdiyle çalışıp bu
dosyalarla karşılaştırıyor. 17 tespit fikstürü birebir; takip 2 × 600 karede fark 0. Kayan nokta sonuçları
işlemciye göre son hanede değişebildiği için hareket fikstürleri küçük bir payla karşılaştırılıyor.

**Kamera hareketini nasıl çıkarıyorsunuz?**
IR görüntüsünde anahtar karede köşe noktaları seçiliyor, Lucas-Kanade optik akışıyla yeni kareye taşınıyor,
RANSAC'lı 4 serbestlikli benzerlik dönüşümü kestiriliyor. Ardışık kareler yerine anahtar kareye kayıt
yapılıyor, çünkü ardışık zincir ölçek hatası biriktiriyordu. Görüntü uzun süre kesilirse ORB ile eski
haritaya yeniden bağlanılıyor. [faz4 tasarımı](design/faz4.md) §5.1

**Bir aracın hareketli olduğuna nasıl karar veriyorsunuz?**
Araç merkezi haritaya taşınıyor; 15 karelik yer değiştirme kutu boyuna ve 15'e bölünüyor (v). v ≤ 0,03
duran, v ≥ 0,10 hareketli, arada önceki karar korunuyor (histerezis). Kutu boyuna bölmek, drone yükselip
alçaldığında eşiğin ölçekten bağımsız kalmasını sağlıyor. Piksel → metre çevrimi yok, çünkü irtifa ve
çözünürlük bilinmiyor.

**Takip nasıl yapılıyor?**
`trackers` paketindeki ByteTrack'in C++ uyarlaması, parametreler paketin varsayılanları (ayar yapılmadı).
Gerçek kutularla takip neredeyse kusursuz (HOTA 0,949); motordaki kayıp tespitten geliyor (E3c-M ile HOTA
0,771). [ADR 0008](adr/0008-tracking-eval.md)

**Gecikme bütçesi neye harcanıyor?**
Faz 3'te çözme + sağlık + tespit + takip p95 82,2 ms idi; Faz 4'te kamera hareketi, hareket kararı ve 4
bölgeyle 97,9 ms (eşik 100). Pay dar: koşudan koşuya 7-13 ms oynuyor. INT8 bu işlemcide hızlandırmadı,
yavaşlattı; GPU'da (T4) E3c ≈ 100 FPS. [details.md](details.md#edge-format--hassasiyet--donanım-val)

## Sınırlar

**Bu sistem sahada kullanılabilir mi?**
Henüz değil. Ölçümler açık veri setlerinde (DroneVehicle, M3OT); gerçek bir drone'dan canlı görüntüyle
denenmedi. Hareket kararı yalnız iki val ve iki test dizisinde ölçüldü. Uzun görüntü kesintisinden sonra
harita birkaç piksel kayabiliyor ([ADR 0011](adr/0011-kabul-10b.md)). Lisans da AGPL-3.0 (Ultralytics).

**Bir daha yapsanız neyi farklı yapardınız?**
Faz 2'nin kapısında karanlık gece karelerini ilk tasarımda ayrı ele alırdım; hareket değerlendirmesi için
duran araç içeren test dizilerini önceden garanti ederdim; birden çok seed'le eğitip seed varyansını
güven aralığına katardım.
