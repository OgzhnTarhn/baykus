"""Büyük JSON kayıtlarını değerle karşılaştırma: ilk farkın yolunu kısaca verir."""

import math


def first_diff(a, b, path="$"):
    """Ondalıklar `rel/abs 1e-9` payla (CI işlemcisine göre son hane değişebiliyor), gerisi tam."""
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, int | float) and isinstance(b, int | float)
        return (
            None
            if ok and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            else f"{path}: {a!r} != {b!r}"
        )
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: anahtarlar {sorted(a)} != {sorted(b)}"
        return next((d for k in a if (d := first_diff(a[k], b[k], f"{path}.{k}"))), None)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: uzunluk {len(a)} != {len(b)}"
        return next(
            (
                d
                for i, (x, y) in enumerate(zip(a, b, strict=True))
                if (d := first_diff(x, y, f"{path}[{i}]"))
            ),
            None,
        )
    return None if a == b else f"{path}: {a!r} != {b!r}"
