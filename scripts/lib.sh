#!/usr/bin/env bash
# Shared helpers for the OpenStats release scripts. Source this file; do not run it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${OPENSTATS_DIST:-$ROOT/dist}"
TAURI_DIR="$ROOT/openstats-desktop/tauri/src-tauri"
WEB_SRC="$ROOT/openstats-desktop/web"

log() { printf '==> %s\n' "$*" >&2; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required. $2"; }

# Run a command, or only print it when OPENSTATS_DRY_RUN is set.
run() {
  if [ -n "${OPENSTATS_DRY_RUN:-}" ]; then
    printf '[dry-run] %s\n' "$*" >&2
  else
    "$@"
  fi
}

# The release version: the workspace package version.
version() {
  sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml" | head -n 1
}

# SHA-256 of a file, as lowercase hex.
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    die "sha256sum or shasum is required"
  fi
}

# Write DIST/SHA256SUMS for every regular file directly in DIST.
write_checksums() {
  local f
  : >"$DIST/SHA256SUMS"
  for f in "$DIST"/*; do
    [ -f "$f" ] || continue
    [ "$(basename "$f")" = SHA256SUMS ] && continue
    printf '%s  %s\n' "$(sha256_file "$f")" "$(basename "$f")" >>"$DIST/SHA256SUMS"
  done
}
