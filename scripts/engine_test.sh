#!/usr/bin/env bash
# Motor testlerini derleme imajında koşar; modeller ve fikstürler salt okunur bağlanır (Faz 3 Görev 16).
# Kullanım: REQ=1 bash scripts/engine_test.sh   (REQ=1: eksik fikstür/model test hatasıdır; CI böyle)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
image="${IMAGE:-baykus-engine-build}"
# Git Bash'te Windows yolu: docker'a C:/... biçiminde verilir
if command -v cygpath >/dev/null 2>&1; then root="$(cygpath -m "$root")"; fi
MSYS_NO_PATHCONV=1 docker run --rm \
  -e BAYKUS_REQUIRE_FIXTURES="${REQ:-0}" -e BAYKUS_MODELS=/models -e BAYKUS_FIXTURES=/fixtures \
  -v "$root/models:/models:ro" -v "$root/fixtures:/fixtures:ro" -v "$root/configs:/configs:ro" \
  "$image" ctest --test-dir /build --output-on-failure "$@"
