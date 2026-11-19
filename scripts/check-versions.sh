#!/usr/bin/env bash
# Fails unless the workspace, the Tauri crate and tauri.conf.json agree on one version.
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

want="$(version)"
[ -n "$want" ] || die "cannot read the workspace version from Cargo.toml"

conf="$(sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' "$TAURI_DIR/tauri.conf.json" | head -n 1)"
crate="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$TAURI_DIR/Cargo.toml" | head -n 1)"

status=0
[ "$conf" = "$want" ] || { warn "tauri.conf.json is $conf, workspace is $want"; status=1; }
[ "$crate" = "$want" ] || { warn "src-tauri/Cargo.toml is $crate, workspace is $want"; status=1; }
[ "$status" -eq 0 ] || die "version mismatch"
log "all versions are $want"
