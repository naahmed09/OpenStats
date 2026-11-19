# OpenStats native desktop app

A real window (winit + softbuffer), the OS menu bar (muda), native file dialogs (rfd), the system
clipboard (arboard) and system fonts (fontdue). **No browser, web view or JavaScript.** The
browser build in `../web` is separate and is the only part of OpenStats that runs in a browser.

All behaviour lives in `openstats-host::native` (std only, tested in `crates/openstats-host/tests/native.rs`):
frames, input, menus, and the actions (save, open, clipboard, graph windows) the shell must perform.
This crate only adapts an OS window to it.

```sh
cd openstats-desktop/native
cargo run --release          # launch
```

Packaging a macOS app bundle and `.dmg`: `scripts/package-native-macos.sh` (from the repository root).

## Status

Written against winit 0.30, softbuffer 0.4, fontdue 0.9, muda 0.15, rfd 0.15, arboard 3 and **not
compiled** where it was written (no registry access); expect small API fixes on first build.

Known limits:
* Printing writes PNG page files; there is no system print dialog.
* The Linux build has no native menu bar (GTK needed); the keyboard shortcuts work.
* Text is drawn with Arial / Segoe UI / Liberation Sans if present, else the built-in bitmap face; there is no
  kerning or complex-script shaping.
* The Dock/taskbar icon comes from the app bundle (`package-native-macos.sh`); a bare `cargo run` shows a generic icon.
* Screen readers cannot see inside a drawn UI.
