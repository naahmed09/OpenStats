#!/usr/bin/env bash
# Tests the release scripts without a wasm toolchain: builds the web bundle around a minimal valid
# WebAssembly module and checks the result. Run directly or through `cargo test -p openstats-host`.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# Every script parses.
for f in "$root"/scripts/*.sh "$here"/*.sh; do bash -n "$f" || fail "syntax error in $f"; done

# Versions agree.
"$root/scripts/check-versions.sh" 2>/dev/null || fail "versions disagree"

# A mismatch is detected (work on a copy of the tree's small files).
mkdir -p "$tmp/mini/scripts" "$tmp/mini/openstats-desktop/tauri/src-tauri"
cp "$root"/scripts/lib.sh "$root"/scripts/check-versions.sh "$tmp/mini/scripts/"
cp "$root/Cargo.toml" "$tmp/mini/"
cp "$root"/openstats-desktop/tauri/src-tauri/Cargo.toml "$root"/openstats-desktop/tauri/src-tauri/tauri.conf.json "$tmp/mini/openstats-desktop/tauri/src-tauri/"
sed -i.bak 's/"version": "[^"]*"/"version": "9.9.9"/' "$tmp/mini/openstats-desktop/tauri/src-tauri/tauri.conf.json"
if "$tmp/mini/scripts/check-versions.sh" 2>/dev/null; then fail "a version mismatch was not detected"; fi

# The smallest valid module: magic + version.
printf '\0asm\1\0\0\0' >"$tmp/m.wasm"
out="$tmp/dist/web"
OPENSTATS_DIST="$tmp/dist" "$root/scripts/build-web.sh" --wasm "$tmp/m.wasm" --out "$out" 2>/dev/null

for f in index.html graph.html openstats.js tauri-shell.js openstats_host.wasm _headers version.json LICENSE-APACHE LICENSE-MIT; do
  [ -f "$out/$f" ] || fail "bundle lacks $f"
done
cmp -s "$tmp/m.wasm" "$out/openstats_host.wasm" || fail "the module was altered"
ver="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -n 1)"
grep -q "\"version\":\"$ver\"" "$out/version.json" || fail "version.json has the wrong version"
want="$(sha256sum "$tmp/m.wasm" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$tmp/m.wasm" | cut -d' ' -f1)"
grep -q "$want" "$out/version.json" || fail "version.json has the wrong checksum"
grep -q 'application/wasm' "$out/_headers" || fail "_headers lacks the wasm MIME type"
# Test files and fixtures never ship.
[ ! -e "$out/test" ] || fail "test files leaked into the bundle"
if command -v gzip >/dev/null 2>&1; then
  gzip -dc "$out/openstats.js.gz" | cmp -s - "$out/openstats.js" || fail "openstats.js.gz does not match"
fi
tar -tzf "$tmp/dist/openstats-web-$ver.tar.gz" >"$tmp/listing"
grep -q 'openstats_host.wasm' "$tmp/listing" || fail "archive lacks the module"

# A file that is not WebAssembly is refused, as is a missing module.
printf 'not wasm' >"$tmp/bad.wasm"
if OPENSTATS_DIST="$tmp/dist2" "$root/scripts/build-web.sh" --wasm "$tmp/bad.wasm" --out "$tmp/o2" 2>/dev/null; then fail "a non-wasm file was accepted"; fi
if OPENSTATS_DIST="$tmp/dist2" "$root/scripts/build-web.sh" --wasm "$tmp/none.wasm" --out "$tmp/o3" 2>/dev/null; then fail "a missing module was accepted"; fi

# The macOS script refuses to run elsewhere.
if [ "$(uname -s)" != Darwin ]; then
  if "$root/scripts/package-macos.sh" 2>"$tmp/err"; then fail "package-macos.sh ran on $(uname -s)"; fi
  grep -q 'only be built on macOS' "$tmp/err" || fail "package-macos.sh gave the wrong reason"
fi

# release.sh --skip-checks --web-only in dry-run mode builds the bundle and checksums.
OPENSTATS_DIST="$tmp/dist3" OPENSTATS_DRY_RUN=1 "$root/scripts/release.sh" --skip-checks --web-only >/dev/null 2>"$tmp/rel" && fail "dry-run release should stop at the missing wasm build"
grep -q 'dry-run' "$tmp/rel" || fail "dry-run mode printed no commands"

# write_checksums lists every file in DIST and the list verifies.
(
  export OPENSTATS_DIST="$tmp/dist"
  # shellcheck source=../lib.sh
  source "$root/scripts/lib.sh"
  write_checksums
  grep -q "openstats-web-$ver.tar.gz" "$DIST/SHA256SUMS" || exit 1
  ! grep -q SHA256SUMS "$DIST/SHA256SUMS" || exit 1
  cd "$DIST"
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -c --quiet SHA256SUMS; else shasum -a 256 -c --quiet SHA256SUMS; fi
) || fail "SHA256SUMS does not verify"
echo "scripts ok"
