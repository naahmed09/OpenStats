#!/usr/bin/env bash
# One command from a clean checkout to release artefacts in dist/.
#
#   scripts/release.sh [--skip-checks] [--web-only]
#
# 1. verifies the versions agree, then runs rustfmt, clippy (-D warnings), the full test suite
#    and the JavaScript tests (skipped with --skip-checks);
# 2. builds the static web bundle (always);
# 3. builds the native installer for the machine it runs on: .dmg on macOS, .msi/.exe on Windows
#    (through PowerShell). Linux builds the web bundle only. Skipped with --web-only.
# Finally writes dist/SHA256SUMS. Set OPENSTATS_DRY_RUN=1 to print the commands instead.
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

checks=1
native=1
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-checks) checks=0 ;;
    --web-only) native=0 ;;
    -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

"$ROOT/scripts/check-versions.sh"

if [ "$checks" -eq 1 ]; then
  need cargo "Install Rust from https://rustup.rs."
  cd "$ROOT"
  log "rustfmt"
  run cargo fmt --all -- --check
  log "clippy"
  run cargo clippy --workspace --all-targets --locked -- -D warnings
  log "tests"
  run cargo test --workspace --locked
  if command -v node >/dev/null 2>&1; then
    log "javascript tests"
    run node --test "$WEB_SRC"/test/*.test.mjs
  else
    warn "node not found: skipping the JavaScript tests"
  fi
fi

mkdir -p "$DIST"
"$ROOT/scripts/build-web.sh"

if [ "$native" -eq 1 ]; then
  case "$(uname -s)" in
    Darwin) "$ROOT/scripts/package-macos.sh" ;;
    MINGW* | MSYS* | CYGWIN*) run powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT/scripts/package-windows.ps1" ;;
    *) warn "no native installer is built on $(uname -s); the web bundle is complete" ;;
  esac
fi

[ -n "${OPENSTATS_DRY_RUN:-}" ] || write_checksums
log "artefacts in $DIST"
