#!/usr/bin/env bash
# M3OT ölçümü (Faz 3 tasarımı §4.3, §6.3; plan Görev 36): motor tespitleri (C++ baykus_detect, ham ≥ 0,001)
# → AP50 / ECE / çift kutu; takip (C++ baykus_track) motor tespitleri ve oracle (M3OT kutuları skor 1)
# → HOTA / MOTA / IDF1. Önce geliştirme bölümünde (dev) prova edilir; test yalnız son ölçümde, bir kez.
# Önkoşul: baykus-engine imajı, models/, data/m3ot_<dev|test>/{sequences,labels}, python/'da uv sync.
# TILES=4 bash scripts/m3ot_measure.sh dev  → parçalı çıkarım (Görev 15D): motor tespitleri 4 karo
# üzerinden çalışır; çıktıları ayrı dosyalara yazar ( *_tiles4 ), tiles=1 koşumu bozulmaz. Varsayılan 1.
# Kullanım: bash scripts/m3ot_measure.sh dev
#           bash scripts/m3ot_measure.sh test --i-know-this-is-test
#           TILES=4 bash scripts/m3ot_measure.sh dev
#           bash scripts/m3ot_measure.sh dev motion
# OUT_SUFFIX=_e3cm: sonuç dosyalarının adına ek (Faz 4 Görev 23: E3c-M test ölçümü Faz 3'ün E3c test
# dosyalarının üzerine yazmasın). Varsayılan boş.
set -euo pipefail
cd "$(dirname "$0")/.."
part="${1:?kullanım: m3ot_measure.sh <dev|test> [--i-know-this-is-test]}"
flag=()
case "$part" in
  dev) split=val ;;
  test)
    [[ " ${*:2} " == *" --i-know-this-is-test "* ]] || {
      echo "HATA: test yalnız son ölçümde, --i-know-this-is-test ile" >&2
      exit 2
    }
    split=test
    flag=(--i-know-this-is-test)
    ;;
  *)
    echo "kullanım: m3ot_measure.sh <dev|test> [--i-know-this-is-test]" >&2
    exit 2
    ;;
esac
data="data/m3ot_$part"
# Faz 4 hareket satırı (plan F4 Görev 16): dev motion | test motion --i-know-this-is-test (sıra serbest) → C++
# baykus_sequence (ego + hareket + takip) → baykus-motion measure --row engine; dev'de kör örneklem r2.
if [[ " ${*:2} " == *" motion "* ]]; then
  work="runs/m3ot_${part}_motion"
  export MSYS_NO_PATHCONV=1
  mkdir -p "$work" results/motion
  rm -f "$work"/*.json.gz
  runs=()
  for d in "$data"/sequences/*/; do
    s="$(basename "$d")"
    docker run --rm -v "$PWD/models:/models:ro" -v "$PWD/configs:/configs:ro" \
      -v "$PWD/$data:/data:ro" -v "$PWD/$work:/work" --entrypoint baykus_sequence baykus-engine \
      --models /models --manifest /configs/engine/models.json --motion /configs/engine/motion.json \
      --tracker /configs/engine/tracker.json --seq-dir "/data/sequences/$s" \
      --out "/work/$s.json.gz" "${flag[@]}"
    runs+=(--run "../$work/$s.json.gz")
  done
  audit=()
  key="data/motion/audit_key_val_r2.json" answers="data/motion/audit_answers_val_r2.json"
  if [ "$part" = dev ] && [ -f "$key" ] && [ -f "$answers" ]; then
    audit=(--audit-key "../$key" --audit-answers "../$answers")
  fi
  (cd python && uv run baykus-motion measure --split "$part" --row engine "${runs[@]}" "${audit[@]}" \
    --out "../results/motion/m3ot_${part}_engine.json" "${flag[@]}")
  exit 0
fi
# TILES: 1 (varsayılan, tüm kutu) veya 4 (parçalı çıkarım, Görev 15D)
tiles="${TILES:-1}"
tiles_suf=""
tiles_flag=()
case "$tiles" in
  1) ;;
  4)
    tiles_suf="_tiles4"
    tiles_flag=(--tiles 4)
    ;;
  *)
    echo "HATA: TILES yalnız 1 veya 4 olabilir (TILES=$tiles)" >&2
    exit 2
    ;;
esac
work="runs/m3ot_$part$tiles_suf"
export MSYS_NO_PATHCONV=1
mkdir -p "$work/dets" results/engine results/track
rm -f "$work"/dets/*.json.gz "$work"/*.json.gz

seqs=()
for d in "$data"/sequences/*/; do seqs+=("$(basename "$d")"); done
echo "== dizi: ${seqs[*]} ($split)"

engine=(docker run --rm -v "$PWD/models:/models:ro" -v "$PWD/configs:/configs:ro"
  -v "$PWD/$data:/data:ro" -v "$PWD/$work:/work" --entrypoint)
echo "== 1/4 motor tespitleri (baykus_detect, ham ≥ 0,001)"
for s in "${seqs[@]}"; do
  "${engine[@]}" baykus_detect baykus-engine --models /models \
    --manifest /configs/engine/models.json --seq-dir "/data/sequences/$s" \
    --out "/work/dets/$s.json.gz" --floor 0.001 "${tiles_flag[@]}" "${flag[@]}"
done
dets=()
for s in "${seqs[@]}"; do dets+=("../$work/dets/$s.json.gz"); done

echo "== 2/4 AP50, ECE, çift kutu"
(cd python && uv run baykus-engine measure --dets "${dets[@]}" --labels "../$data/labels" \
  --out "../results/engine/m3ot_${part}${tiles_suf}${OUT_SUFFIX:-}_detect.json" "${flag[@]}")

if [ "$tiles" = "4" ]; then
  # Oracle tespiti/takibi dedektörden bağımsızdır (M3OT kutuları skor 1); 4 karo koşumunda tekrarlanmaz.
  echo "== 3/4 oracle tespitleri atlandı (TILES=4: oracle dedektörden bağımsız)"
else
  echo "== 3/4 oracle tespitleri (M3OT kutuları skor 1)"
  dirs=()
  for s in "${seqs[@]}"; do dirs+=("../$data/sequences/$s"); done
  (cd python && uv run baykus-track oracle --gt-csv "../$data/labels/tracks_ir.csv" \
    --seq-dir "${dirs[@]}" --out "../$work/oracle_dets.json.gz" "${flag[@]}")
fi

echo "== 4/4 takip (baykus_track) ve HOTA, MOTA, IDF1"
for s in "${seqs[@]}"; do
  "${engine[@]}" baykus_track baykus-engine --dets "/work/dets/$s.json.gz" \
    --config /configs/engine/tracker.json --out "/work/tracks_$s.json.gz" "${flag[@]}"
done
if [ "$tiles" != "4" ]; then
  "${engine[@]}" baykus_track baykus-engine --dets /work/oracle_dets.json.gz \
    --config /configs/engine/tracker.json --out /work/oracle_tracks.json.gz "${flag[@]}"
fi
(cd python && uv run python - "../$work" "${seqs[@]}" <<'EOF'
# dizi başına iz dosyalarını tek dosyada topla (eval tek --tracks alır)
import gzip, json, sys
from pathlib import Path
work, seqs = Path(sys.argv[1]), sys.argv[2:]
recs = [json.loads(gzip.decompress((work / f"tracks_{s}.json.gz").read_bytes())) for s in seqs]
out = {**{k: v for k, v in recs[0].items() if k != "sequences"},
       "sequences": [x for r in recs for x in r["sequences"]]}
data = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
(work / "engine_tracks.json.gz").write_bytes(gzip.compress(data, mtime=0))
EOF
)
(cd python && uv run baykus-track eval --tracks "../$work/engine_tracks.json.gz" --dets "${dets[@]}" \
  --gt-csv "../$data/labels/tracks_ir.csv" --out "../results/track/m3ot_${part}${tiles_suf}${OUT_SUFFIX:-}_engine.json" "${flag[@]}")
if [ "$tiles" = "4" ]; then
  echo "== oracle takibi/ölçümü atlandı (TILES=4: oracle dedektörden bağımsız)"
else
  (cd python && uv run baykus-track eval --tracks "../$work/oracle_tracks.json.gz" \
    --dets "../$work/oracle_dets.json.gz" --gt-csv "../$data/labels/tracks_ir.csv" \
    --out "../results/track/m3ot_${part}${OUT_SUFFIX:-}_oracle.json" "${flag[@]}")
fi
