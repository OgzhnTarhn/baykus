#!/usr/bin/env bash
# Model dosyalarını Release'ten indirir ve sha256 ile doğrular (Faz 3 plan Görev 8).
# Kullanım: scripts/fetch_models.sh <out_dir> [manifest]   (varsayılan configs/engine/models.json)
# BASE_URL ortam değişkeni manifestteki adresi geçersiz kılar (yerel deneme). Uyuşmazlıkta çıkış 1.
set -euo pipefail

out="${1:?kullanım: fetch_models.sh <out_dir> [manifest]}"
manifest="${2:-$(dirname "$0")/../configs/engine/models.json}"
flat="$(tr -d ' \n\r\t' < "$manifest")"
base="${BASE_URL:-$(printf '%s' "$flat" | sed -n 's/.*"base_url":"\([^"]*\)".*/\1/p')}"
sums="$(printf '%s' "$flat" | grep -o '"[^"]*":{"sha256":"[0-9a-f]\{64\}"' \
  | sed 's/^"\([^"]*\)":{"sha256":"\([0-9a-f]*\)"$/\2  \1/')"
if [ -z "$base" ] || [ -z "$sums" ]; then
  echo "manifest okunamadı: $manifest" >&2
  exit 1
fi

mkdir -p "$out"
while read -r _ name; do
  curl -fsSL --retry 3 -o "$out/$name" "${base}${name}"
done <<< "$sums"
(cd "$out" && printf '%s\n' "$sums" | sha256sum -c --quiet) || {
  echo "sha256 uyuşmuyor: $out" >&2
  exit 1
}
echo "modeller doğrulandı: $out"
