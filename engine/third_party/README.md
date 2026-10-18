# Üçüncü taraf kodu

Bu klasör C++ motoruna kopyalanan dış kodu ve lisans notlarını tutar (Faz 3 tasarımı §4.1, §4.3).

| Bileşen | Kaynak | Lisans | Görev |
|---|---|---|---|
| `httplib.h` | cpp-httplib (tek başlık) | MIT | 25 |
| `lsap/` | SciPy `rectangular_lsap` C kodunun C++ uyarlaması | BSD-3-Clause | 22 |

Derleme bağımlılıkları (kopyalanmaz; Docker `build` aşamasında): OpenCV 5.0.0 (Apache-2.0, kaynaktan),
ONNX Runtime 1.30.0 (MIT, hazır paket), Eigen3 (MPL-2.0), nlohmann/json (MIT), GoogleTest (BSD-3-Clause),
OpenSSL libcrypto (Apache-2.0).
