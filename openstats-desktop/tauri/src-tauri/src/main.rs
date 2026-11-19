//! Tauri v2 shell for OpenStats.
//!
//! The application itself (`openstats_host.wasm` + `web/openstats.js`) runs inside the
//! webview exactly as in a browser. This shell adds what a browser cannot do:
//!
//! * a **native menu bar**, built from the menu tree the workspace publishes as JSON
//!   (the webview sends it with `set_menu`; choosing an item emits `menu-command`
//!   back to the window that owns the menu);
//! * **native open/save dialogs, file access and clipboard** (via the dialog, fs and
//!   clipboard-manager plugins, driven from `web/tauri-shell.js`);
//! * **graph tear-off windows**: `open_graph_window` creates a webview that loads
//!   `graph.html` and receives the chart spec by event.
//!
//! Menu items deliberately carry no native accelerators: the workspace handles its own
//! shortcuts, and a native accelerator would fire the command twice.
//!
//! Status: written against the Tauri v2 API; not compiled in the environment that
//! produced it (no registry access).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use tauri::menu::{Menu, MenuBuilder, MenuItemBuilder, PredefinedMenuItem, Submenu, SubmenuBuilder};
use tauri::{AppHandle, Emitter, Listener, Manager, Runtime, WebviewUrl, WebviewWindowBuilder};

/// One node of the menu tree produced by `Host::menu_json` (parsed by the webview and
/// forwarded as a structured value, so serde deserialises it directly).
#[derive(Deserialize)]
struct Node {
    label: Option<String>,
    id: Option<String>,
    shortcut: Option<String>,
    enabled: Option<bool>,
    separator: Option<bool>,
    entries: Option<Vec<Node>>,
}

fn submenu<R: Runtime>(app: &AppHandle<R>, label: &str, nodes: &[Node]) -> tauri::Result<Submenu<R>> {
    let mut sub = SubmenuBuilder::new(app, label);
    for n in nodes {
        if n.separator == Some(true) {
            sub = sub.separator();
        } else if let Some(children) = &n.entries {
            let inner = submenu(app, n.label.as_deref().unwrap_or(""), children)?;
            sub = sub.item(&inner);
        } else if let Some(id) = &n.id {
            let mut label = n.label.clone().unwrap_or_default();
            if let Some(s) = n.shortcut.as_deref().filter(|s| !s.is_empty()) {
                label = format!("{label}\t{s}");
            }
            let item = MenuItemBuilder::with_id(id.clone(), label)
                .enabled(n.enabled.unwrap_or(true))
                .build(app)?;
            sub = sub.item(&item);
        }
    }
    sub.build()
}

fn build_menu<R: Runtime>(app: &AppHandle<R>, top: &[Node]) -> tauri::Result<Menu<R>> {
    let mut bar = MenuBuilder::new(app);
    for m in top {
        let sub = submenu(app, m.label.as_deref().unwrap_or(""), m.entries.as_deref().unwrap_or(&[]))?;
        bar = bar.item(&sub);
    }
    // Standard window menu so minimise / close behave natively on every platform.
    let window = SubmenuBuilder::new(app, "Window")
        .item(&PredefinedMenuItem::minimize(app, None)?)
        .item(&PredefinedMenuItem::close_window(app, None)?)
        .build()?;
    bar.item(&window).build()
}

/// Replace the native menu bar. Called by the webview whenever the menu tree changes
/// (for example when Undo/Redo become available).
#[tauri::command]
fn set_menu(app: AppHandle, tree: Vec<Node>) -> Result<(), String> {
    let menu = build_menu(&app, &tree).map_err(|e| e.to_string())?;
    app.set_menu(menu).map(|_| ()).map_err(|e| e.to_string())
}

/// Payload delivered to a graph window once it announces itself.
#[derive(Clone, Serialize)]
struct GraphPayload {
    name: String,
    json: String,
}

/// Open a graph in its own window. The chart spec is delivered once the new page
/// announces itself with a `graph-ready` event, so it is never sent before the page
/// is listening.
#[tauri::command]
fn open_graph_window(app: AppHandle, name: String, json: String) -> Result<String, String> {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let label = format!("graph-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    let w = WebviewWindowBuilder::new(&app, &label, WebviewUrl::App("graph.html".into()))
        .title(&name)
        .inner_size(720.0, 480.0)
        .build()
        .map_err(|e| e.to_string())?;
    let handle = app.clone();
    let target = label.clone();
    let payload = GraphPayload { name, json };
    w.once("graph-ready", move |_| {
        let _ = handle.emit_to(target.as_str(), "graph-open", payload.clone());
    });
    Ok(label)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .invoke_handler(tauri::generate_handler![set_menu, open_graph_window])
        .on_menu_event(|app, event| {
            // Menu commands always act on the main workspace window.
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.emit("menu-command", event.id().0.clone());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running OpenStats");
}
