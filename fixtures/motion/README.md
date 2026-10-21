# fixtures-f4 (üretici: `baykus-motion fixtures`)

Klip: M3OT val 1_08 kare 243-302, CC BY 4.0 (60 kare, özgün baytlar). Seçim yalnız eşlik testi için (tasarım §5.9 sınır kuralı).

Senaryolar (`scenarios.json`): `clean`, `ir_drop`, `ir_freeze`, `short_gap`, `long_gap`, `seek`, `scene_change`, `reset_all`.
Dönüşümler (`warps.json`): 80 IR görüntü.

Eşiğe yakın kalan kayıt: 19 (`margins.json`). 1-541 arasında sınır kuralını tam geçen başlangıç yok (ego oranı anahtar kare eşiği 0,5'e sık sık ±0,02 yakın); kullanıcı kararıyla (2026-10-21) kapsamı tam, en az sorunlu aday seçildi. C++ ego (plan Görev 11) bu kayıtlarda ölçülür; karar farkı çıkarsa fikstür yeniden seçilir.

Görüntüler M3OT veri setinden (Scientific Data 2025, figshare https://figshare.com/s/01fa8d1163f4e9a5a13a), lisans CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Değişiklikler: yatay aynalama, sıfır görüntüler, benzerlik dönüşümleri ve ötelemeler.

## Denenen başlangıçlar
- başlangıç 243: 19 sorun
- başlangıç 250: 35 sorun
- başlangıç 252: 23 sorun
- başlangıç 254: 44 sorun
- başlangıç 258: 43 sorun
- başlangıç 262: 28 sorun, kapsam eksik (long_gap: nesil 1'e tek birleşme yok (0 yeniden konumlama))
- başlangıç 266: 40 sorun
- başlangıç 270: 25 sorun
- başlangıç 274: 35 sorun
- başlangıç 278: 25 sorun
- başlangıç 282: 35 sorun
- başlangıç 286: 34 sorun
- başlangıç 290: 51 sorun, kapsam eksik (long_gap: nesil 1'e tek birleşme yok (0 yeniden konumlama))
- başlangıç 294: 47 sorun
- başlangıç 298: 28 sorun, kapsam eksik (long_gap: nesil 1'e tek birleşme yok (0 yeniden konumlama))
- başlangıç 302: 39 sorun
- başlangıç 306: 35 sorun, kapsam eksik (long_gap: nesil 1'e tek birleşme yok (0 yeniden konumlama))
- başlangıç 310: 29 sorun
- başlangıç 314: 31 sorun
- başlangıç 318: 27 sorun
- başlangıç 322: 39 sorun
- başlangıç 326: 41 sorun
- başlangıç 330: 37 sorun, kapsam eksik (ir_drop: nesil 1'e tek birleşme yok (0 yeniden konumlama))
- başlangıç 334: 32 sorun, kapsam eksik (ir_drop: nesil 1'e tek birleşme yok (0 yeniden konumlama))
- başlangıç 338: 17 sorun, kapsam eksik (long_gap: nesil 1'e tek birleşme yok (0 yeniden konumlama))

Paket: `fixtures-f4.tar` sha256 `6434da5b5a6b50b4d3cc011b81b50e226425a01c4777031d0ec3c2ea8c771ed8`, 41000960 bayt (Release `fixtures-f4`).
