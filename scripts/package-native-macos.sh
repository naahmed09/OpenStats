#!/usr/bin/env bash
# Builds the NATIVE (no web view) universal OpenStats.app and a .dmg around it.
#
#   scripts/package-native-macos.sh
#
# Needs macOS with Xcode command line tools and Rust (rustup). Output:
# dist/OpenStats-native_VERSION_universal.dmg. Optional signing: APPLE_SIGNING_IDENTITY.
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "$(uname -s)" = "Darwin" ] || die "this packages a macOS app and must run on macOS (this is $(uname -s))"
need cargo "Install Rust from https://rustup.rs."
need rustup "Install Rust from https://rustup.rs."
need lipo "Install the Xcode command line tools: xcode-select --install"
need hdiutil "hdiutil ships with macOS."

ver="$(version)"
native="$ROOT/openstats-desktop/native"
icons="$ROOT/openstats-desktop/tauri/src-tauri/icons"
app="$DIST/OpenStats.app"

run rustup target add aarch64-apple-darwin x86_64-apple-darwin
for t in aarch64-apple-darwin x86_64-apple-darwin; do
  log "building $t"
  (cd "$native" && run cargo build --release --target "$t")
done

if [ -n "${OPENSTATS_DRY_RUN:-}" ]; then
  log "dry run: would assemble $app and $DIST/OpenStats-native_${ver}_universal.dmg"
  exit 0
fi

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
lipo -create -output "$app/Contents/MacOS/OpenStats" \
  "$native/target/aarch64-apple-darwin/release/OpenStats" \
  "$native/target/x86_64-apple-darwin/release/OpenStats"
cp "$icons/icon.icns" "$app/Contents/Resources/icon.icns"
cat >"$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>OpenStats</string>
  <key>CFBundleDisplayName</key><string>OpenStats</string>
  <key>CFBundleIdentifier</key><string>org.openstats.desktop</string>
  <key>CFBundleExecutable</key><string>OpenStats</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${ver}</string>
  <key>CFBundleVersion</key><string>${ver}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key><array><dict>
    <key>CFBundleTypeName</key><string>OpenStats project</string>
    <key>CFBundleTypeExtensions</key><array><string>otb</string></array>
    <key>CFBundleTypeRole</key><string>Editor</string>
  </dict></array>
</dict></plist>
PLIST

if [ -n "${APPLE_SIGNING_IDENTITY:-}" ]; then
  codesign --force --options runtime --sign "$APPLE_SIGNING_IDENTITY" "$app"
else
  warn "APPLE_SIGNING_IDENTITY is not set: signing ad hoc (Gatekeeper will ask the user to approve)"
  codesign --force --sign - "$app"
fi

stage="$(mktemp -d)"
cp -R "$app" "$stage/"
ln -s /Applications "$stage/Applications"
dmg="$DIST/OpenStats-native_${ver}_universal.dmg"
rm -f "$dmg"
hdiutil create -volname "OpenStats" -srcfolder "$stage" -ov -format UDZO "$dmg" >/dev/null
rm -rf "$stage"
hdiutil verify "$dmg" >/dev/null
archs="$(lipo -archs "$app/Contents/MacOS/OpenStats")"
case "$archs" in *x86_64*arm64* | *arm64*x86_64*) ;; *) die "the executable is not universal (found: $archs)" ;; esac
write_checksums
log "wrote $dmg"
