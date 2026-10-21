# fixtures-f4 (üretici: `baykus-motion fixtures`)

Klip: M3OT val 1_08 kare 243-302, CC BY 4.0 (60 kare, özgün baytlar). Seçim yalnız eşlik testi için (tasarım §5.9 sınır kuralı).

Senaryolar (`scenarios.json`): `clean`, `ir_drop`, `ir_freeze`, `short_gap`, `long_gap`, `seek`, `scene_change`, `reset_all`.
Dönüşümler (`warps.json`): 80 IR görüntü.

Eşiğe yakın kalan kayıt: 19 (`margins.json`). 1-541 arasında sınır kuralını tam geçen başlangıç yok (ego oranı anahtar kare eşiği 0,5'e sık sık ±0,02 yakın); kullanıcı kararıyla (2026-10-21) kapsamı tam, en az sorunlu aday seçildi. C++ ego (plan Görev 11) bu kayıtlarda ölçülür; karar farkı çıkarsa fikstür yeniden seçilir.

Görüntüler M3OT veri setinden (Scientific Data 2025, figshare https://figshare.com/s/01fa8d1163f4e9a5a13a), lisans CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Değişiklikler: yatay aynalama, sıfır görüntüler, benzerlik dönüşümleri ve ötelemeler.

## Denenen başlangıçlar
- başlangıç 243: 19 sorun

Paket: `fixtures-f4.tar` sha256 `8175bf8df10af0eede10a1c0461662ac161350d68b2fbd4e181fdcd41a7de5cf`, 41000960 bayt (Release `fixtures-f4`).
