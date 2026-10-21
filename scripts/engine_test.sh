#!/usr/bin/env bash
# Motor testlerini derleme imajında koşar; modeller ve fikstürler salt okunur bağlanır (Faz 3 Görev 16).
# Kullanım: REQ=1 bash scripts/engine_test.sh   (REQ=1: eksik fikstür/model test hatasıdır; CI böyle)
# RAW_REF=<dosya>: Kabul 1 a ham referansı aynı makinede üretilmiş dosyadan (baykus-engine raw-ref)
# EGO_TRACE=<dosya>: ego hareketi izi (baykus-engine ego-trace) aynı makinede üretilmiş dosyadan
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
image="${IMAGE:-baykus-engine-build}"
# Git Bash'te Windows yolu: docker'a C:/... biçiminde verilir
to_docker() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
root="$(to_docker "$root")"
raw_ref=()
if [ -n "${RAW_REF:-}" ]; then
  ref="$(to_docker "$(cd "$(dirname "$RAW_REF")" && pwd)/$(basename "$RAW_REF")")"
  raw_ref=(-e BAYKUS_RAW_REF=/raw_ref.json -v "$ref:/raw_ref.json:ro")
fi
ego_trace=()
if [ -n "${EGO_TRACE:-}" ]; then
  trace="$(to_docker "$(cd "$(dirname "$EGO_TRACE")" && pwd)/$(basename "$EGO_TRACE")")"
  ego_trace=(-e BAYKUS_EGO_TRACE=/ego_trace.json -v "$trace:/ego_trace.json:ro")
fi
# EVIDENCE=<klasör>: testlerin yazdığı kanıtlar (ör. api.json) oraya
evidence=()
if [ -n "${EVIDENCE:-}" ]; then
  mkdir -p "$EVIDENCE"
  evidence=(-e BAYKUS_EVIDENCE_DIR=/evidence -v "$(to_docker "$(cd "$EVIDENCE" && pwd)"):/evidence")
fi
MSYS_NO_PATHCONV=1 docker run --rm \
  -e BAYKUS_REQUIRE_FIXTURES="${REQ:-0}" -e BAYKUS_MODELS=/models -e BAYKUS_FIXTURES=/fixtures \
  -v "$root/models:/models:ro" -v "$root/fixtures:/fixtures:ro" -v "$root/configs:/configs:ro" \
  "${raw_ref[@]}" "${ego_trace[@]}" "${evidence[@]}" "$image" ctest --test-dir /build --output-on-failure "$@"
