//! OpenStats native desktop shell: a window, a menu bar, dialogs and the clipboard around
//! `openstats_host::native`. There is no browser, web view or JavaScript anywhere in this app.
//!
//! Status: written against winit 0.30, softbuffer 0.4, fontdue 0.9, muda 0.15, rfd 0.15 and
//! arboard 3; not compiled where it was written (no registry access).

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod fonts;
mod keymap;

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use muda::{Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use openstats_charts::render::raster::TextPainter;
use openstats_host::native::{to_xrgb, Action, MenuNode, NativeApp, NativeGraph};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowId};

type Surface = softbuffer::Surface<Arc<Window>, Arc<Window>>;

struct MainWin {
    window: Arc<Window>,
    surface: Surface,
    app: NativeApp,
    mods: ModifiersState,
    cursor: (f64, f64),
    last_click: Option<(Instant, (f64, f64), u32)>,
    /// Last input-method candidate-window position sent to the OS. Telling the OS again on every
    /// frame made it answer with input-method events, which asked for another frame: a busy loop.
    ime_area: Option<(f64, f64, f64)>,
}

struct GraphWin {
    window: Arc<Window>,
    surface: Surface,
    graph: NativeGraph,
}

enum Win {
    Main(MainWin),
    Graph(GraphWin),
}

struct Shell {
    wins: HashMap<WindowId, Win>,
    main_id: Option<WindowId>,
    saved_theme: String,
    font: Option<fonts::SystemFont>,
    clipboard: Option<arboard::Clipboard>,
    menu: Option<Menu>,
}

/// Where the chosen theme is remembered between launches.
fn theme_file() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(std::path::PathBuf::from(home).join(".openstats").join("theme"))
}

fn new_surface(window: &Arc<Window>) -> Result<Surface, String> {
    let ctx = softbuffer::Context::new(window.clone()).map_err(|e| e.to_string())?;
    softbuffer::Surface::new(&ctx, window.clone()).map_err(|e| e.to_string())
}

/// Paint `img` into the window's surface.
fn present(window: &Window, surface: &mut Surface, img: &openstats_charts::render::png::Image) {
    let size = window.inner_size();
    let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else { return };
    if surface.resize(w, h).is_err() {
        return;
    }
    if let Ok(mut buf) = surface.buffer_mut() {
        to_xrgb(img, &mut buf, size.width as usize, size.height as usize);
        let _ = buf.present();
    }
}

fn menu_label(label: &str, shortcut: &str) -> String {
    if shortcut.is_empty() {
        return label.to_string();
    }
    if cfg!(target_os = "macos") {
        let s = shortcut.replace("Ctrl+", "⌘").replace("Shift+", "⇧").replace("Alt+", "⌥");
        format!("{label}    {s}")
    } else {
        format!("{label}\t{shortcut}")
    }
}

fn build_submenu(label: &str, entries: &[MenuNode]) -> Submenu {
    let sub = Submenu::new(label, true);
    for e in entries {
        match e {
            MenuNode::Separator => {
                let _ = sub.append(&PredefinedMenuItem::separator());
            }
            MenuNode::Item { label, id, shortcut, enabled } => {
                // No native accelerators: the workspace handles its own shortcuts, and a native
                // key equivalent would swallow them (breaking text editing in cells).
                let _ = sub.append(&MenuItem::with_id(id.as_str(), menu_label(label, shortcut), *enabled, None));
            }
            MenuNode::Submenu { label, entries } => {
                let _ = sub.append(&build_submenu(label, entries));
            }
        }
    }
    sub
}

impl Shell {
    fn main_mut(&mut self) -> Option<&mut MainWin> {
        let id = self.main_id?;
        match self.wins.get_mut(&id) {
            Some(Win::Main(m)) => Some(m),
            _ => None,
        }
    }

    fn open_main(&mut self, el: &ActiveEventLoop) -> Result<(), String> {
        let attrs = Window::default_attributes()
            .with_title("OpenStats")
            .with_inner_size(LogicalSize::new(1280.0, 800.0))
            .with_min_inner_size(LogicalSize::new(800.0, 500.0));
        let window = Arc::new(el.create_window(attrs).map_err(|e| e.to_string())?);
        window.set_ime_allowed(true);
        let surface = new_surface(&window)?;
        let size = window.inner_size().to_logical::<f64>(window.scale_factor());
        let mut app = NativeApp::new(size.width, size.height, window.scale_factor());
        if let Some(saved) = theme_file().and_then(|p| std::fs::read_to_string(p).ok()) {
            app.set_theme_by_name(&saved);
        }
        let id = window.id();
        self.wins.insert(
            id,
            Win::Main(MainWin {
                window: window.clone(),
                surface,
                app,
                mods: ModifiersState::empty(),
                cursor: (0.0, 0.0),
                last_click: None,
                ime_area: None,
            }),
        );
        self.main_id = Some(id);
        self.sync_chrome();
        window.request_redraw();
        Ok(())
    }

    fn open_graph(&mut self, el: &ActiveEventLoop, name: &str, json: &str) {
        let graph = match NativeGraph::from_json(json) {
            Ok(g) => g,
            Err(e) => return message(&format!("Cannot open the graph: {e}")),
        };
        let attrs = Window::default_attributes().with_title(name).with_inner_size(LogicalSize::new(720.0, 480.0));
        let Ok(window) = el.create_window(attrs).map(Arc::new) else { return };
        let Ok(surface) = new_surface(&window) else { return };
        window.request_redraw();
        self.wins.insert(window.id(), Win::Graph(GraphWin { window, surface, graph }));
    }

    /// Push a changed title and menu bar to the operating system.
    fn sync_chrome(&mut self) {
        let (theme, menu) = {
            let Some(m) = self.main_mut() else { return };
            if let Some(t) = m.app.title_if_changed() {
                m.window.set_title(&t);
            }
            (m.app.theme_name(), m.app.menu_if_changed())
        };
        if theme != self.saved_theme {
            self.saved_theme = theme.to_string();
            if let Some(path) = theme_file() {
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                let _ = std::fs::write(path, theme);
            }
        }
        if let Some(tree) = menu {
            let bar = Menu::new();
            if cfg!(target_os = "macos") {
                // macOS shows the first menu as the application menu.
                let app = Submenu::new("OpenStats", true);
                let _ = app.append(&PredefinedMenuItem::about(None, None));
                let _ = app.append(&PredefinedMenuItem::separator());
                let _ = app.append(&PredefinedMenuItem::hide(None));
                let _ = app.append(&PredefinedMenuItem::hide_others(None));
                let _ = app.append(&PredefinedMenuItem::show_all(None));
                let _ = app.append(&PredefinedMenuItem::separator());
                let _ = app.append(&PredefinedMenuItem::quit(None));
                let _ = bar.append(&app);
            }
            for node in &tree {
                if let MenuNode::Submenu { label, entries } = node {
                    let _ = bar.append(&build_submenu(label, entries));
                }
            }
            if cfg!(target_os = "macos") {
                let win = Submenu::new("Window", true);
                let _ = win.append(&PredefinedMenuItem::minimize(None));
                let _ = win.append(&PredefinedMenuItem::maximize(None));
                let _ = bar.append(&win);
            }
            self.attach_menu(&bar);
            self.menu = Some(bar);
        }
    }

    #[cfg(target_os = "macos")]
    fn attach_menu(&self, bar: &Menu) {
        bar.init_for_nsapp();
    }

    #[cfg(target_os = "windows")]
    fn attach_menu(&self, bar: &Menu) {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let Some(id) = self.main_id else { return };
        if let Some(Win::Main(m)) = self.wins.get(&id) {
            if let Ok(h) = m.window.window_handle() {
                if let RawWindowHandle::Win32(w) = h.as_raw() {
                    // SAFETY: the handle belongs to the live main window, which outlives the menu.
                    let _ = unsafe { bar.init_for_hwnd(w.hwnd.get()) };
                }
            }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    fn attach_menu(&self, _bar: &Menu) {
        // GTK menu bars need a gtk container; on Linux the keyboard shortcuts and the in-window
        // menu bar drawn by the workspace remain available.
    }

    /// Carry out what the workspace asked for. Follow-up actions (opening a file produces more)
    /// are handled in the same loop.
    fn perform(&mut self, el: &ActiveEventLoop, first: Vec<Action>) {
        let mut queue = first;
        while let Some(a) = queue.pop() {
            match a {
                Action::SaveFile { name, bytes } => save_file(&name, &bytes),
                Action::Clipboard(t) => {
                    if let Some(c) = self.clipboard.as_mut() {
                        let _ = c.set_text(t);
                    }
                }
                Action::PickProject => {
                    // File > Open accepts data files too (routed by extension), so a CSV or workbook
                    // chosen here is imported instead of being greyed out.
                    let picked = rfd::FileDialog::new()
                        .add_filter("OpenStats projects and data", &["otb", "csv", "tsv", "txt", "xlsx", "json", "gz", "mac"])
                        .add_filter("OpenStats project", &["otb"])
                        .pick_file();
                    if let Some(path) = picked {
                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        let is_project = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("otb"));
                        match std::fs::read(&path) {
                            Ok(bytes) => {
                                if let Some(m) = self.main_mut() {
                                    queue.extend(if is_project {
                                        m.app.open_project(&bytes)
                                    } else {
                                        m.app.import_data(&name, &bytes)
                                    });
                                }
                            }
                            Err(e) => message(&format!("Cannot read {}: {e}", path.display())),
                        }
                    }
                }
                Action::PickData => {
                    let picked = rfd::FileDialog::new()
                        .add_filter("Data files and macros", &["csv", "tsv", "txt", "xlsx", "json", "gz", "mac"])
                        .pick_file();
                    if let Some(path) = picked {
                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        match std::fs::read(&path) {
                            Ok(bytes) => {
                                if let Some(m) = self.main_mut() {
                                    queue.extend(m.app.import_data(&name, &bytes));
                                }
                            }
                            Err(e) => message(&format!("Cannot read {}: {e}", path.display())),
                        }
                    }
                }
                Action::TearOff { name, json } => self.open_graph(el, &name, &json),
                Action::SavePages { base, pages } => {
                    // Printing writes one PNG per page; the system print dialog is not used.
                    let Some(first) = rfd::FileDialog::new()
                        .set_file_name(format!("{base}.png"))
                        .add_filter("PNG image", &["png"])
                        .save_file()
                    else {
                        continue;
                    };
                    let stem = first.with_extension("");
                    for (i, p) in pages.iter().enumerate() {
                        let path = if i == 0 {
                            first.clone()
                        } else {
                            let mut s = stem.clone().into_os_string();
                            s.push(format!("-{}.png", i + 1));
                            s.into()
                        };
                        if let Err(e) = std::fs::write(&path, p) {
                            message(&format!("Cannot write {}: {e}", path.display()));
                        }
                    }
                }
                // Errors are shown in the window title by `sync_chrome`.
                Action::Error(_) => {}
            }
        }
        self.sync_chrome();
    }

    /// A menu item was chosen.
    fn menu_command(&mut self, el: &ActiveEventLoop, id: &str) {
        let acts = if id == "edit.paste" {
            // Paste needs the system clipboard, which the workspace cannot read itself.
            let text = self.clipboard.as_mut().and_then(|c| c.get_text().ok());
            match (text, self.main_mut()) {
                (Some(t), Some(m)) => m.app.paste(&t),
                _ => Vec::new(),
            }
        } else {
            self.main_mut().map(|m| m.app.command(id)).unwrap_or_default()
        };
        self.perform(el, acts);
        self.redraw_main();
    }

    /// Ask for a repaint of the main window (identical frames are skipped, so this is cheap).
    fn redraw_main(&mut self) {
        if let Some(m) = self.main_mut() {
            m.window.request_redraw();
        }
    }

    fn main_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let mut acts = Vec::new();
        let mut paste_from_clipboard = false;
        let is_redraw = matches!(event, WindowEvent::RedrawRequested);
        {
            let Some(Win::Main(m)) = self.wins.get_mut(&id) else { return };
            let sf = m.window.scale_factor();
            match event {
                WindowEvent::CloseRequested => {
                    el.exit();
                    return;
                }
                WindowEvent::Resized(size) => {
                    let l = size.to_logical::<f64>(sf);
                    m.app.resize(l.width, l.height, sf);
                    m.app.invalidate();
                    m.window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    let l = m.window.inner_size().to_logical::<f64>(scale_factor);
                    m.app.resize(l.width, l.height, scale_factor);
                    m.app.invalidate();
                    m.window.request_redraw();
                }
                WindowEvent::ModifiersChanged(mods) => m.mods = mods.state(),
                WindowEvent::CursorMoved { position, .. } => {
                    let p: LogicalPosition<f64> = position.to_logical(sf);
                    m.cursor = (p.x, p.y);
                    acts = m.app.pointer(1, p.x, p.y, 0, keymap::mod_bits(m.mods), 0);
                    m.window.request_redraw();
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let b = match button {
                        MouseButton::Left => 0,
                        MouseButton::Right => 2,
                        _ => return,
                    };
                    let (x, y) = m.cursor;
                    let bits = keymap::mod_bits(m.mods);
                    if state == ElementState::Pressed {
                        let now = Instant::now();
                        let clicks = match m.last_click {
                            Some((t, (lx, ly), n))
                                if now.duration_since(t) < Duration::from_millis(500)
                                    && (lx - x).abs() < 4.0
                                    && (ly - y).abs() < 4.0 =>
                            {
                                (n + 1).min(3)
                            }
                            _ => 1,
                        };
                        m.last_click = Some((now, (x, y), clicks));
                        acts = m.app.pointer(0, x, y, b, bits, clicks);
                    } else {
                        acts = m.app.pointer(2, x, y, b, bits, 1);
                    }
                    m.window.request_redraw();
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    // The workspace scrolls like a browser: positive = content moves up.
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (-f64::from(x) * 40.0, -f64::from(y) * 40.0),
                        MouseScrollDelta::PixelDelta(PhysicalPosition { x, y }) => (-x / sf, -y / sf),
                    };
                    m.app.wheel(m.cursor.0, m.cursor.1, dx, dy);
                    m.window.request_redraw();
                }
                WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                    let bits = keymap::mod_bits(m.mods);
                    match keymap::key_code(&event, m.mods) {
                        // Cmd/Ctrl+V: the clipboard text arrives as a paste, not as a key.
                        Some(c) if bits & 2 != 0 && bits & 4 == 0 && (c == 'v' as u32 || c == 'V' as u32) => {
                            paste_from_clipboard = true;
                        }
                        Some(code) => acts = m.app.key(code, bits),
                        None => {}
                    }
                    m.window.request_redraw();
                }
                WindowEvent::Ime(ime) => {
                    match ime {
                        Ime::Preedit(text, cursor) => {
                            if text.is_empty() {
                                m.app.ime_cancel();
                            } else {
                                let caret = cursor.map_or(text.chars().count(), |(start, _)| {
                                    text.get(..start).map_or(text.chars().count(), |s| s.chars().count())
                                });
                                m.app.ime_update(&text, caret);
                            }
                        }
                        Ime::Commit(text) => m.app.ime_commit(&text),
                        Ime::Disabled => m.app.ime_cancel(),
                        Ime::Enabled => {}
                    }
                    m.window.request_redraw(); // skipped by `frame_if_changed` when nothing changed
                }
                WindowEvent::RedrawRequested => {
                    // Only rasterise and present when the picture really changed.
                    let img = match self.font.as_mut() {
                        Some(f) => m.app.frame_if_changed(Some(f as &mut dyn TextPainter)),
                        None => m.app.frame_if_changed(None),
                    };
                    if let Some(img) = img {
                        present(&m.window, &mut m.surface, &img);
                    }
                    // Keep the candidate window next to the cell being edited, but only tell the
                    // OS when the position changed.
                    let anchor = m.app.ime_anchor();
                    if anchor != m.ime_area {
                        m.ime_area = anchor;
                        if let Some((x, y, h)) = anchor {
                            m.window.set_ime_cursor_area(LogicalPosition::new(x, y), LogicalSize::new(2.0, h));
                        }
                    }
                }
                _ => {}
            }
        }
        if paste_from_clipboard {
            let text = self.clipboard.as_mut().and_then(|c| c.get_text().ok());
            if let (Some(t), Some(m)) = (text, self.main_mut()) {
                acts = m.app.paste(&t);
            }
        }
        self.perform(el, acts);
        // Never ask for a repaint while answering one: that is an endless redraw loop.
        if !is_redraw {
            self.redraw_main();
        }
    }

    fn graph_event(&mut self, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.wins.remove(&id);
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(Win::Graph(g)) = self.wins.get(&id) {
                    g.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(Win::Graph(g)) = self.wins.get_mut(&id) {
                    let sf = g.window.scale_factor();
                    let size = g.window.inner_size().to_logical::<f64>(sf);
                    let img = match self.font.as_mut() {
                        Some(f) => g.graph.frame(size.width, size.height, sf, Some(f as &mut dyn TextPainter)),
                        None => g.graph.frame(size.width, size.height, sf, None),
                    };
                    present(&g.window, &mut g.surface, &img);
                }
            }
            _ => {}
        }
    }
}

fn save_file(name: &str, bytes: &[u8]) {
    let ext = std::path::Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("").to_string();
    let mut dlg = rfd::FileDialog::new().set_file_name(name);
    if !ext.is_empty() {
        dlg = dlg.add_filter(ext.to_uppercase(), &[ext.as_str()]);
    }
    if let Some(path) = dlg.save_file() {
        if let Err(e) = std::fs::write(&path, bytes) {
            message(&format!("Cannot write {}: {e}", path.display()));
        }
    }
}

fn message(text: &str) {
    rfd::MessageDialog::new().set_title("OpenStats").set_description(text).set_level(rfd::MessageLevel::Error).show();
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        set_dock_icon();
        if self.main_id.is_none() {
            if let Err(e) = self.open_main(el) {
                message(&format!("Cannot open the window: {e}"));
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if Some(id) == self.main_id {
            self.main_event(el, id, event);
        } else {
            self.graph_event(id, event);
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Native menu choices arrive on a channel of their own.
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            let id = ev.id.0.clone();
            self.menu_command(el, &id);
        }
        el.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(50)));
    }
}

fn main() {
    let event_loop = match EventLoop::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("OpenStats cannot start: {e}");
            std::process::exit(1);
        }
    };
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut shell = Shell {
        wins: HashMap::new(),
        main_id: None,
        saved_theme: String::new(),
        font: fonts::SystemFont::load(),
        clipboard: arboard::Clipboard::new().ok(),
        menu: None,
    };
    if let Err(e) = event_loop.run_app(&mut shell) {
        eprintln!("OpenStats stopped: {e}");
        std::process::exit(1);
    }
}

/// Show the OpenStats icon in the Dock even when run with `cargo run` (outside an app bundle, which
/// would otherwise show a generic "exec" icon). A bundle built by `package-native-macos.sh`
/// carries the same icon itself.
#[cfg(target_os = "macos")]
fn set_dock_icon() {
    use objc2::{rc::Retained, ClassType};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::{MainThreadMarker, NSData};
    static ICON: &[u8] = include_bytes!("../../tauri/src-tauri/icons/icon.png");
    let Some(mtm) = MainThreadMarker::new() else { return };
    let data = NSData::with_bytes(ICON);
    let image: Option<Retained<NSImage>> = NSImage::initWithData(NSImage::alloc(), &data);
    if let Some(image) = image {
        let app = NSApplication::sharedApplication(mtm);
        unsafe { app.setApplicationIconImage(Some(&image)) };
    }
}

#[cfg(not(target_os = "macos"))]
fn set_dock_icon() {}
