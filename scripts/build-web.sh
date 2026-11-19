#!/usr/bin/env bash
# Builds the static web bundle: the WebAssembly module, the page and the JavaScript glue.
#
#   scripts/build-web.sh [--wasm FILE] [--out DIR] [--no-compress]
#
# Without --wasm the module is compiled from source for wasm32-unknown-unknown with the
# size-optimised `web` profile (needs the target: `rustup target add wasm32-unknown-unknown`).
# The result in DIR (default dist/web) is self-contained: serve it from any static host, or let
# Tauri embed it. Also writes dist/openstats-web-VERSION.tar.gz (and .zip when `zip` exists).
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

wasm=""
out="$DIST/web"
compress=1
while [ $# -gt 0 ]; do
  case "$1" in
    --wasm) wasm="${2:?--wasm needs a file}"; shift 2 ;;
    --out) out="${2:?--out needs a directory}"; shift 2 ;;
    --no-compress) compress=0; shift ;;
    -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done

ver="$(version)"

if [ -z "$wasm" ]; then
  need cargo "Install Rust from https://rustup.rs."
  if command -v rustup >/dev/null 2>&1; then
    run rustup target add wasm32-unknown-unknown
  fi
  log "compiling openstats-host for wasm32 (profile web)"
  (cd "$ROOT" && run cargo build -p openstats-host --target wasm32-unknown-unknown --profile web --locked)
  wasm="$ROOT/target/wasm32-unknown-unknown/web/openstats_host.wasm"
fi
[ -f "$wasm" ] || die "no WebAssembly module at $wasm"
[ "$(head -c 4 "$wasm" | od -An -tx1 | tr -d ' \n')" = "0061736d" ] || die "$wasm is not a WebAssembly module"

rm -rf "$out"
mkdir -p "$out"
cp "$WEB_SRC/index.html" "$WEB_SRC/graph.html" "$WEB_SRC/openstats.js" "$WEB_SRC/tauri-shell.js" "$out/"
cp "$wasm" "$out/openstats_host.wasm"
cp "$ROOT/LICENSE-APACHE" "$ROOT/LICENSE-MIT" "$out/"

if command -v wasm-opt >/dev/null 2>&1; then
  log "optimising with wasm-opt"
  wasm-opt -Oz --enable-bulk-memory --enable-sign-ext --enable-mutable-globals "$out/openstats_host.wasm" -o "$out/openstats_host.wasm.opt" &&
    mv "$out/openstats_host.wasm.opt" "$out/openstats_host.wasm"
else
  warn "wasm-opt not found; shipping the module as built"
fi

# Hosting hints (Netlify, Cloudflare Pages): correct MIME type, immutable caching of the module.
cat >"$out/_headers" <<'HEADERS'
/*.wasm
  Content-Type: application/wasm
  Cache-Control: public, max-age=31536000, immutable
/*.js
  Content-Type: text/javascript; charset=utf-8
  Cache-Control: public, max-age=3600
/*
  X-Content-Type-Options: nosniff
  Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:
HEADERS

printf '{"name":"openstats","version":"%s","wasm":{"file":"openstats_host.wasm","bytes":%s,"sha256":"%s"}}\n' \
  "$ver" "$(wc -c <"$out/openstats_host.wasm" | tr -d ' ')" "$(sha256_file "$out/openstats_host.wasm")" >"$out/version.json"

# Every module the pages import and every module they fetch must be in the bundle.
for page in index.html graph.html openstats.js tauri-shell.js; do
  refs="$(grep -ohE "(from|import\()? *['\"]\./[A-Za-z0-9_.-]+\.(js|wasm|html)['\"]" "$out/$page" | grep -oE "\./[A-Za-z0-9_.-]+" | sort -u || true)"
  for r in $refs; do
    [ -f "$out/${r#./}" ] || die "$page refers to $r, which is not in the bundle"
  done
done

if [ "$compress" -eq 1 ] && command -v gzip >/dev/null 2>&1; then
  for f in index.html graph.html openstats.js tauri-shell.js openstats_host.wasm; do
    gzip -9 -k -n -f "$out/$f"
  done
fi

mkdir -p "$DIST"
archive="$DIST/openstats-web-$ver"
tar -C "$out" -czf "$archive.tar.gz" .
if command -v zip >/dev/null 2>&1; then
  rm -f "$archive.zip"
  (cd "$out" && zip -qr "$archive.zip" .)
fi
log "web bundle: $out ($(wc -c <"$out/openstats_host.wasm" | tr -d ' ') byte module) and $archive.tar.gz"
