#!/usr/bin/env bash
# Kapalı ağ kanıtları 1-4 (Faz 3 tasarımı §4.5, Kabul 6 otomatik kısmı); CI `compose` işi ve yerel
# aynı adımlar. Önkoşul: baykus-engine ve baykus-ui imajları, models/ (fetch_models.sh), ui/'da
# npm ci ve Playwright tarayıcısı (yerelde PW_CHANNEL=chrome ile sistem Chrome'u).
# Kullanım: bash scripts/offline_proof.sh <kanıt klasörü> [offline.json çıktısı]
#   BAYKUS_DATA (varsayılan ./fixtures/m3ot/clip): /data olarak bağlanan demo verisi
set -euo pipefail
cd "$(dirname "$0")/.."
ev="${1:?kullanım: offline_proof.sh <kanıt klasörü> [offline.json]}"
out="${2:-$ev/offline.json}"
mkdir -p "$ev"
# önceki koşunun kanıtları silinir: düşen adım dosya yazmazsa rapor eski "geçti"yi okumasın
for f in offline_check egress_ui egress_engine browser scan; do rm -f "$ev/$f.json"; done
export BAYKUS_DATA="${BAYKUS_DATA:-./fixtures/m3ot/clip}" MSYS_NO_PATHCONV=1
offline=(-f docker-compose.yml -f compose.offline.yml)

echo "== kanıt 1-2: kapalı katman (bütün ağlar internal, port yok)"
docker compose "${offline[@]}" up -d
docker compose exec -T ui node /app/scripts/offline_check.mjs > "$ev/offline_check.json" || true
docker compose exec -T ui node /app/scripts/egress_check.mjs > "$ev/egress_ui.json" || true
docker compose exec -T engine baykus_server --egress-check > "$ev/egress_engine.json" || true
cat "$ev/offline_check.json" "$ev/egress_ui.json" "$ev/egress_engine.json"
docker compose "${offline[@]}" down

echo "== kanıt 4: imajdaki derlenmiş Next çıktısında adres taraması"
cid="$(docker create baykus-ui)"
rm -rf "$ev/next"
docker cp "$cid:/app/.next" "$ev/next"
docker rm "$cid" > /dev/null
node scripts/scan_next.mjs "$ev/next" scripts/scan_allow.txt --out "$ev/scan.json" > /dev/null || true
rm -rf "$ev/next"

echo "== kanıt 3: normal compose, başsız tarayıcı"
docker compose up -d
(cd ui && EVIDENCE_DIR="../$ev" npx playwright test) || true
docker compose down

node scripts/offline_report.mjs "$ev" "$out"
