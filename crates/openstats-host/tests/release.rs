//! The release configuration: icons, versions, bundle settings and the packaging scripts.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn src_tauri() -> PathBuf {
    root().join("openstats-desktop/tauri/src-tauri")
}

/// The value of the first `"key": "value"` (JSON) or `key = "value"` (TOML) after `from`.
fn string_after(text: &str, from: &str, key: &str) -> Option<String> {
    let start = text.find(from)? + from.len();
    let rest = &text[start..];
    for line in rest.lines() {
        let l = line.trim().trim_end_matches(',');
        for sep in ["\": \"", "\":\"", " = \""] {
            let pat = format!("{key}{}", sep.trim_start_matches('"'));
            let pat2 = format!("\"{key}{sep}");
            if let Some(v) = l.strip_prefix(&pat2).or_else(|| l.strip_prefix(pat.as_str())) {
                return Some(v.trim_end_matches('"').to_string());
            }
        }
    }
    None
}

fn png_size(b: &[u8]) -> (u32, u32) {
    assert_eq!(&b[..8], b"\x89PNG\r\n\x1a\n", "PNG signature");
    assert_eq!(&b[12..16], b"IHDR");
    (u32::from_be_bytes(b[16..20].try_into().unwrap()), u32::from_be_bytes(b[20..24].try_into().unwrap()))
}

#[test]
fn png_icons_have_the_sizes_their_names_promise() {
    let dir = src_tauri().join("icons");
    for (name, size) in [("32x32.png", 32), ("128x128.png", 128), ("128x128@2x.png", 256), ("icon.png", 512)] {
        let b = std::fs::read(dir.join(name)).unwrap();
        assert_eq!(png_size(&b), (size, size), "{name}");
        assert!(b.len() > 200, "{name} is not blank");
    }
}

#[test]
fn ico_directory_points_at_png_images_of_the_declared_sizes() {
    let b = std::fs::read(src_tauri().join("icons/icon.ico")).unwrap();
    assert_eq!(&b[..4], &[0, 0, 1, 0]);
    let n = usize::from(u16::from_le_bytes([b[4], b[5]]));
    assert_eq!(n, 7);
    let mut sizes = vec![];
    for i in 0..n {
        let e = &b[6 + 16 * i..6 + 16 * (i + 1)];
        let declared = if e[0] == 0 { 256 } else { u32::from(e[0]) };
        let len = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
        let off = u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize;
        assert_eq!(u16::from_le_bytes([e[6], e[7]]), 32, "32 bits per pixel");
        assert!(off + len <= b.len());
        assert_eq!(png_size(&b[off..off + len]), (declared, declared));
        sizes.push(declared);
    }
    assert_eq!(sizes, [16, 24, 32, 48, 64, 128, 256]);
}

#[test]
fn icns_chunks_tile_the_file_exactly() {
    let b = std::fs::read(src_tauri().join("icons/icon.icns")).unwrap();
    assert_eq!(&b[..4], b"icns");
    assert_eq!(u32::from_be_bytes(b[4..8].try_into().unwrap()) as usize, b.len());
    let mut pos = 8;
    let mut kinds = vec![];
    while pos < b.len() {
        let kind = String::from_utf8(b[pos..pos + 4].to_vec()).unwrap();
        let len = u32::from_be_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let side = match kind.as_str() {
            "ic07" => 128,
            "ic08" => 256,
            "ic09" => 512,
            "ic10" => 1024,
            other => panic!("unexpected chunk {other}"),
        };
        assert_eq!(png_size(&b[pos + 8..pos + len]), (side, side), "{kind}");
        kinds.push(kind);
        pos += len;
    }
    assert_eq!(pos, b.len());
    assert_eq!(kinds, ["ic07", "ic08", "ic09", "ic10"]);
}

#[test]
fn every_version_agrees_and_the_bundle_config_is_complete() {
    let ws = read(&root().join("Cargo.toml"));
    let version = string_after(&ws, "[workspace.package]", "version").expect("workspace version");
    assert_eq!(version, "1.0.0");
    let conf = read(&src_tauri().join("tauri.conf.json"));
    assert_eq!(string_after(&conf, "{", "version").as_deref(), Some(version.as_str()));
    let crate_toml = read(&src_tauri().join("Cargo.toml"));
    assert_eq!(string_after(&crate_toml, "[package]", "version").as_deref(), Some(version.as_str()));
    // Path dependencies are versioned alongside.
    for krate in ["core", "math", "stats", "charts", "ui"] {
        let line = ws.lines().find(|l| l.starts_with(&format!("openstats-{krate} = "))).unwrap();
        assert!(line.contains(&format!("version = \"{version}\"")), "{line}");
    }
    // Every bundle icon exists, the license matches the workspace, and the three installers are built.
    for icon in ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.icns", "icons/icon.ico"] {
        assert!(conf.contains(&format!("\"{icon}\"")), "{icon} listed in tauri.conf.json");
        assert!(src_tauri().join(icon).is_file(), "{icon} exists");
    }
    assert!(conf.contains("\"license\": \"Apache-2.0 OR MIT\""));
    for target in ["dmg", "msi", "nsis"] {
        assert!(conf.contains(&format!("\"{target}\"")), "{target} target");
    }
    assert!(conf.contains("\"frontendDist\": \"../../../dist/web\""));
    assert!(conf.contains("\"otb\""), "the .otb file association");
    assert!(conf.contains("wasm-unsafe-eval"), "the CSP lets the module compile");
}

#[test]
fn capabilities_cover_exactly_the_plugins_the_shell_uses() {
    let main = read(&src_tauri().join("src/main.rs"));
    let caps = read(&src_tauri().join("capabilities/default.json"));
    for (plugin, perm) in [
        ("tauri_plugin_dialog", "dialog:allow-open"),
        ("tauri_plugin_dialog", "dialog:allow-save"),
        ("tauri_plugin_fs", "fs:allow-read-file"),
        ("tauri_plugin_fs", "fs:allow-write-file"),
        ("tauri_plugin_clipboard_manager", "clipboard-manager:allow-write-text"),
    ] {
        assert!(main.contains(&format!("{plugin}::init()")), "{plugin} initialised");
        assert!(caps.contains(perm), "{perm} granted");
    }
    let cargo = read(&src_tauri().join("Cargo.toml"));
    for dep in ["tauri-plugin-dialog", "tauri-plugin-fs", "tauri-plugin-clipboard-manager"] {
        assert!(cargo.contains(dep), "{dep} in Cargo.toml");
    }
    // The commands the JS calls exist in Rust and are registered.
    let js = read(&root().join("openstats-desktop/web/tauri-shell.js"));
    for cmd in ["set_menu", "open_graph_window"] {
        assert!(js.contains(&format!("'{cmd}'")), "{cmd} called from JS");
        assert!(main.contains(&format!("fn {cmd}(")), "{cmd} defined in Rust");
        assert!(main.contains(cmd), "{cmd} registered");
    }
    // Menu events are emitted under the name the JS listens for.
    assert!(main.contains("\"menu-command\"") && js.contains("'menu-command'"));
    assert!(main.contains("\"graph-ready\"") && js.contains("'graph-ready'"));
    assert!(main.contains("\"graph-open\"") && js.contains("'graph-open'"));
}

#[test]
fn the_web_pages_only_reference_files_the_bundle_ships() {
    let web = root().join("openstats-desktop/web");
    let shipped = ["index.html", "graph.html", "openstats.js", "tauri-shell.js", "openstats_host.wasm"];
    for page in ["index.html", "graph.html", "openstats.js", "tauri-shell.js"] {
        let text = read(&web.join(page));
        for part in text.split("'./").skip(1).chain(text.split("\"./").skip(1)) {
            let name: String =
                part.chars().take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')).collect();
            if name.ends_with(".js") || name.ends_with(".wasm") || name.ends_with(".html") {
                assert!(
                    shipped.contains(&name.as_str()),
                    "{page} refers to ./{name}, which build-web.sh does not ship"
                );
            }
        }
    }
    let script = read(&root().join("scripts/build-web.sh"));
    for f in shipped.iter().filter(|f| !f.ends_with(".wasm")) {
        assert!(script.contains(f), "build-web.sh copies {f}");
    }
}

#[test]
fn the_packaging_scripts_pass_their_own_tests() {
    // The shell scripts are exercised on Linux and macOS. On Windows `bash` is usually the WSL
    // launcher (which fails when no distribution is installed), so the check is skipped there.
    let bash = std::process::Command::new("bash").arg("--version").output();
    if cfg!(windows) || !bash.map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("a working bash is not available; skipping the script tests");
        return;
    }
    let out = std::process::Command::new("bash")
        .arg(root().join("scripts/test/test-scripts.sh"))
        .output()
        .expect("run test-scripts.sh");
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("scripts ok"));
}
