#!/usr/bin/env bash
# Hazırlık (Faz 3 tasarımı §4.5; plan Görev 34). İnternet açıkken bir kez çalıştırılır:
#   1. imajlar: baykus-engine (runtime) ve baykus-ui yerelde derlenir;
#   2. modeller: models/ manifestle (sha256) doğrulanır, eksik ya da farklıysa Release'ten indirilir;
#   3. demo verisi: demo-data/sequences/<dizi>/{index.json, rgb/, ir/} denetlenir (test bölümü yok).
# Kullanım: bash scripts/prepare.sh                 → sonra: docker compose up
#           bash scripts/prepare.sh --fixtures      → yalnız Faz 4 eşlik paketi: build/fixtures-f4.tar
#             (configs/engine/fixtures.json sha256; eksik ya da farklıysa Release'ten) → fixtures/f4/
#           bash scripts/prepare.sh --bundle out/   → internetsiz bilgisayara taşınacak klasör:
#             baykus-images.tar.gz (docker save), docker-compose.yml, compose.offline.yml,
#             configs/engine/, models/, demo-data/, README-offline.md
set -euo pipefail
cd "$(dirname "$0")/.."

bundle=""
if [ "${1:-}" = "--fixtures" ]; then
  manifest=configs/engine/fixtures.json
  tar_name=fixtures-f4.tar
  sum="$(tr -d ' \n\r\t' < "$manifest" | grep -o '"sha256":"[0-9a-f]\{64\}"' | cut -d'"' -f4)"
  if [ -f "build/$tar_name" ] &&
    printf '%s  %s\n' "$sum" "build/$tar_name" | sha256sum -c --quiet 2>/dev/null; then
    echo "fikstür paketi zaten doğru (sha256): build/$tar_name"
  else
    bash scripts/fetch_models.sh build "$manifest"
  fi
  rm -rf fixtures/f4
  mkdir -p fixtures/f4
  tar -xf "build/$tar_name" -C fixtures/f4
  echo "fikstürler açıldı: fixtures/f4 ($(find fixtures/f4 -type f | wc -l) dosya)"
  exit 0
elif [ "${1:-}" = "--bundle" ]; then
  bundle="${2:?kullanım: prepare.sh [--fixtures | --bundle <klasör>]}"
elif [ -n "${1:-}" ]; then
  echo "kullanım: prepare.sh [--fixtures | --bundle <klasör>]" >&2
  exit 2
fi

if [ -n "$bundle" ] && [ -d "$bundle" ] && [ -n "$(ls -A "$bundle")" ] &&
  [ ! -f "$bundle/README-offline.md" ]; then
  # paket adımı klasördeki models/ ve demo-data/'yı yeniler; depo kökü ya da başka bir klasör
  # verilirse gerçek veri silinmesin
  echo "HATA: $bundle boş değil ve bir Baykuş paketi değil; boş ya da yeni bir klasör verin" >&2
  exit 2
fi

echo "== 1/3 imajlar"
docker build -f docker/engine.Dockerfile --target runtime -t baykus-engine .
docker build -f docker/ui.Dockerfile -t baykus-ui .

echo "== 2/3 modeller (models/)"
manifest=configs/engine/models.json
sums="$(tr -d ' \n\r\t' < "$manifest" | grep -o '"[^"]*":{"sha256":"[0-9a-f]\{64\}"' \
  | sed 's/^"\([^"]*\)":{"sha256":"\([0-9a-f]*\)"$/\2  \1/')"
if [ -d models ] && (cd models && printf '%s\n' "$sums" | sha256sum -c --quiet) 2>/dev/null; then
  echo "modeller zaten doğru (sha256)"
else
  bash scripts/fetch_models.sh models "$manifest"
fi

echo "== 3/3 demo verisi (demo-data/sequences)"
found=0
for idx in demo-data/sequences/*/index.json; do
  [ -f "$idx" ] || continue
  dir="$(dirname "$idx")"
  name="$(basename "$dir")"
  flat="$(tr -d ' \n\r\t' < "$idx")"
  split="$(printf '%s' "$flat" | sed -n 's/.*"split":"\([^"]*\)".*/\1/p')"
  if [ "$split" = "test" ]; then
    echo "HATA: $name test bölümünden; demo verisine test dizisi girmez" >&2
    exit 1
  fi
  frames="$(printf '%s' "$flat" | sed -n 's/.*"frames":\[\([0-9,]*\)\].*/\1/p' | tr ',' '\n' | grep -c . || true)"
  rgb="$(find "$dir/rgb" -name '*.jpg' | wc -l)"
  ir="$(find "$dir/ir" -name '*.png' | wc -l)"
  if [ "$frames" -eq 0 ] || [ "$rgb" -ne "$frames" ] || [ "$ir" -ne "$frames" ]; then
    echo "HATA: $name: index $frames kare, rgb $rgb, ir $ir dosya" >&2
    exit 1
  fi
  echo "$name: $split, $frames kare"
  found=$((found + 1))
done
if [ "$found" -eq 0 ]; then
  echo "HATA: demo-data/sequences altında dizi yok (demo paketi: baykus-data demo)" >&2
  exit 1
fi

if [ -z "$bundle" ]; then
  echo "hazır: docker compose up → http://127.0.0.1:3000"
  exit 0
fi

echo "== paket: $bundle"
mkdir -p "$bundle/configs/engine"
docker save baykus-engine baykus-ui | gzip > "$bundle/baykus-images.tar.gz"
cp docker-compose.yml compose.offline.yml "$bundle/"
cp configs/engine/models.json configs/engine/tracker.json "$bundle/configs/engine/"
rm -rf "$bundle/models" "$bundle/demo-data"
cp -r models "$bundle/models"
mkdir -p "$bundle/demo-data"
cp -r demo-data/sequences "$bundle/demo-data/sequences"
cat > "$bundle/README-offline.md" <<'EOF'
# Baykuş — internetsiz kurulum

Gerekli: Docker Desktop (ya da Docker Engine + Compose v2). İnternet gerekmez.
Bu klasörde bir terminal açıp sırayla:

1. `docker load -i baykus-images.tar.gz`
2. `docker compose up`
3. Tarayıcıda `http://127.0.0.1:3000`

Kapatmak için terminalde Ctrl+C, sonra `docker compose down`.
Görüntüler: M3OT veri seti (Scientific Data, 2025), CC BY 4.0
(creativecommons.org/licenses/by/4.0).
EOF
du -sh "$bundle" "$bundle/baykus-images.tar.gz" "$bundle/models" "$bundle/demo-data"
