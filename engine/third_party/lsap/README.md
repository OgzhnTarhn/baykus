# SciPy `rectangular_lsap`

`rectangular_lsap.cpp` ve `rectangular_lsap.h` SciPy'nin
[`scipy/optimize/rectangular_lsap/`](https://github.com/scipy/scipy/tree/v1.17.1/scipy/optimize/rectangular_lsap)
klasöründen **değiştirilmeden** kopyalandı (etiket `v1.17.1`; `v1.18.1`'de dosya aynı). Python
`scipy.optimize.linear_sum_assignment` aynı C++ kodunu çağırır; eşit maliyetlerde de aynı atamayı
vermesi için kendi Macar ya da LAPJV uygulamamız yerine bu kod kullanılır (Faz 3 tasarımı §4.3).

Lisans: BSD-3-Clause, `LICENSE.txt` (SciPy deposunun lisans dosyası). Motorun sarmalayıcısı
`engine/src/lsap.cpp`.
