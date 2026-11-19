#!/usr/bin/env bash
# Builds the universal (arm64 + x86_64) OpenStats.app and a .dmg installer around it.
#
#   scripts/package-macos.sh
#
# Needs macOS with Xcode command line tools, Rust (rustup), and the Tauri CLI
# (`cargo install tauri-cli --version '^2'`). Output: dist/OpenStats_VERSION_universal.dmg.
#
# Signing and notarisation are optional and read from the environment (Tauri's own variables):
#   APPLE_SIGNING_IDENTITY       "Developer ID Application: Name (TEAMID)"
#   APPLE_ID, APPLE_PASSWORD, APPLE_TEAM_ID    notarisation with an app-specific password
# Without them the app is built unsigned and Gatekeeper will ask the user to approve it.
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "$(uname -s)" = "Darwin" ] || die "the macOS installer can only be built on macOS (this is $(uname -s))"
need cargo "Install Rust from https://rustup.rs."
need rustup "Install Rust from https://rustup.rs."
cargo tauri --version >/dev/null 2>&1 || die "the Tauri CLI is required: cargo install tauri-cli --version '^2' --locked"

"$ROOT/scripts/check-versions.sh"
ver="$(version)"

run rustup target add aarch64-apple-darwin x86_64-apple-darwin
"$ROOT/scripts/build-web.sh"

# CI passes unset secrets as empty strings, and Tauri treats an empty identity as a request to
# sign with nothing ("no identity found"). Drop the empty ones.
for v in APPLE_SIGNING_IDENTITY APPLE_ID APPLE_PASSWORD APPLE_TEAM_ID APPLE_CERTIFICATE APPLE_CERTIFICATE_PASSWORD; do
  [ -n "${!v:-}" ] || unset "$v"
done
if [ -z "${APPLE_SIGNING_IDENTITY:-}" ]; then
  warn "APPLE_SIGNING_IDENTITY is not set: the app gets an ad-hoc signature (no developer identity)"
  # "-" is an ad-hoc signature: it keeps the universal binary launchable on Apple Silicon.
  export APPLE_SIGNING_IDENTITY="-"
fi

log "building the universal app bundle"
(cd "$TAURI_DIR" && run cargo tauri build --target universal-apple-darwin --bundles app)

bundle="$TAURI_DIR/target/universal-apple-darwin/release/bundle"
mkdir -p "$DIST"
dmg_out="$DIST/OpenStats_${ver}_universal.dmg"
if [ -n "${OPENSTATS_DRY_RUN:-}" ]; then
  log "dry run: would wrap $bundle/macos/OpenStats.app in $dmg_out"
  exit 0
fi
# The disk image is made here with hdiutil rather than by Tauri's dmg bundler, which styles the
# window through Finder (AppleScript) and fails on machines without a GUI session.
app="$bundle/macos/OpenStats.app"
[ -d "$app" ] || die "Tauri produced no app bundle at $app"
stage="$(mktemp -d)"
cp -R "$app" "$stage/"
ln -s /Applications "$stage/Applications"
rm -f "$dmg_out"
hdiutil create -volname "OpenStats" -srcfolder "$stage" -ov -format UDZO "$dmg_out" >/dev/null
rm -rf "$stage"

# Verify what was built: the disk image is intact and the executable holds both architectures.
hdiutil verify "$dmg_out" >/dev/null
exe="$(find "$bundle/macos/OpenStats.app/Contents/MacOS" -type f -perm -u+x -print -quit 2>/dev/null || true)"
if [ -n "$exe" ]; then
  archs="$(lipo -archs "$exe")"
  case "$archs" in *x86_64*arm64* | *arm64*x86_64*) ;; *) die "the executable is not universal (found: $archs)" ;; esac
fi

write_checksums
log "wrote $DIST/OpenStats_${ver}_universal.dmg"
