//! The flat C ABI.
//!
//! Every function takes and returns integers or floats. Byte buffers never cross as pointers *into*
//! Rust: the shell asks [`ot_input_reserve`] for room, writes into linear memory at the returned
//! offset, then calls the function that consumes it with the length. Results are copied into an
//! output buffer whose offset is [`ot_out_ptr`] and whose length the call returned. Offsets and
//! buffers are valid until the next call into the module.
//!
//! One [`Host`] (the application window) and a table of [`GraphHost`]s (torn-off graph windows)
//! live in thread-local state, which is the whole state of a single-threaded WebAssembly instance.
//!
//! Outcomes: after any event call [`ot_outcome_next`] until it returns `0`. Its result is the kind
//! (`1` file, `2` clipboard, `3` print, `4` needs a project file, `5` needs a data file, `6` error,
//! `7` graph tear-off); slots `0..=3` hold, as bytes, `0` the name / text / message, `1` the MIME
//! type, `2` the file contents, `3` the graph JSON. For prints, [`ot_print_pages`] and
//! [`ot_print_frame`] give each page as a canvas stream.
#![allow(unsafe_code)] // only for the `#[no_mangle]` attributes below; there are no `unsafe` blocks

use super::{GraphHost, Host};
use openstats_ui::workspace::Outcome;
use std::cell::RefCell;

#[derive(Default)]
struct State {
    host: Option<Host>,
    graphs: Vec<Option<GraphHost>>,
    out: Vec<u8>,
    slots: [Vec<u8>; 4],
    pages: usize,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(|s| f(&mut s.borrow_mut()))
}

fn with_host<R: Default>(f: impl FnOnce(&mut Host, &mut Vec<u8>) -> R) -> R {
    with(|s| match s.host.as_mut() {
        Some(h) => f(h, &mut s.out),
        None => R::default(),
    })
}

fn input_text(h: &Host, len: usize) -> String {
    String::from_utf8_lossy(&h.input()[..len.min(h.input().len())]).into_owned()
}

/// ABI version (bump on any incompatible change).
#[no_mangle]
pub extern "C" fn ot_abi_version() -> u32 {
    1
}

/// Create (or replace) the application window. Returns `1`.
#[no_mangle]
pub extern "C" fn ot_init(width: f64, height: f64) -> u32 {
    with(|s| s.host = Some(Host::new(width, height)));
    1
}

/// Resize the window.
#[no_mangle]
pub extern "C" fn ot_resize(width: f64, height: f64) {
    with_host(|h, _| h.resize(width, height));
}

/// Pointer event: kind `0` down, `1` move, `2` up; button `0` left, `2` right; modifier bits
/// `1` shift, `2` ctrl, `4` alt; `clicks` is the click count.
#[no_mangle]
pub extern "C" fn ot_pointer(kind: u32, x: f64, y: f64, button: u32, mods: u32, clicks: u32) {
    with_host(|h, _| h.pointer(kind, x, y, button, mods, clicks));
}

/// Wheel / trackpad scroll in pixels.
#[no_mangle]
pub extern "C" fn ot_wheel(x: f64, y: f64, dx: f64, dy: f64) {
    with_host(|h, _| h.wheel(x, y, dx, dy));
}

/// Key press (see [`crate::keys`]).
#[no_mangle]
pub extern "C" fn ot_key(code: u32, mods: u32) {
    with_host(|h, _| h.key(code, mods));
}

/// Reserve `len` bytes of input and return their offset in linear memory.
#[no_mangle]
pub extern "C" fn ot_input_reserve(len: u32) -> usize {
    with_host(|h, _| h.reserve_input(len as usize).as_mut_ptr() as usize)
}

/// Offset of the output buffer.
#[no_mangle]
pub extern "C" fn ot_out_ptr() -> usize {
    with(|s| s.out.as_ptr() as usize)
}

/// Paste the UTF-8 text in the input buffer.
#[no_mangle]
pub extern "C" fn ot_paste(len: u32) {
    with_host(|h, _| {
        let t = input_text(h, len as usize);
        h.paste(&t);
    });
}

/// Input-method composition update: the input buffer holds the text; `caret` is a character index.
#[no_mangle]
pub extern "C" fn ot_ime_update(len: u32, caret: u32) {
    with_host(|h, _| {
        let t = input_text(h, len as usize);
        h.ime_update(&t, caret as usize);
    });
}

/// Input-method commit of the text in the input buffer.
#[no_mangle]
pub extern "C" fn ot_ime_commit(len: u32) {
    with_host(|h, _| {
        let t = input_text(h, len as usize);
        h.ime_commit(&t);
    });
}

/// Input-method composition cancelled.
#[no_mangle]
pub extern "C" fn ot_ime_cancel() {
    with_host(|h, _| h.ime_cancel());
}

/// Candidate-window anchor: `1` and the values readable through [`ot_ime_x`], [`ot_ime_y`],
/// [`ot_ime_h`] when a cell is being edited, else `0`.
#[no_mangle]
pub extern "C" fn ot_ime_anchor() -> u32 {
    with_host(|h, _| u32::from(h.ime_anchor().is_some()))
}

/// Candidate-window x (NaN when not editing).
#[no_mangle]
pub extern "C" fn ot_ime_x() -> f64 {
    with(|s| s.host.as_ref().and_then(Host::ime_anchor).map_or(f64::NAN, |a| a.0))
}

/// Candidate-window top y (NaN when not editing).
#[no_mangle]
pub extern "C" fn ot_ime_y() -> f64 {
    with(|s| s.host.as_ref().and_then(Host::ime_anchor).map_or(f64::NAN, |a| a.1))
}

/// Candidate-window line height (NaN when not editing).
#[no_mangle]
pub extern "C" fn ot_ime_h() -> f64 {
    with(|s| s.host.as_ref().and_then(Host::ime_anchor).map_or(f64::NAN, |a| a.2))
}

/// Run the command whose text id is in the input buffer. Returns `1` if the id is known.
#[no_mangle]
pub extern "C" fn ot_command(len: u32) -> u32 {
    with_host(|h, _| {
        let id = input_text(h, len as usize);
        u32::from(h.command(&id))
    })
}

/// Open the project bytes in the input buffer.
#[no_mangle]
pub extern "C" fn ot_open_project(len: u32) {
    with_host(|h, _| {
        let b = h.input()[..(len as usize).min(h.input().len())].to_vec();
        h.open_project(&b);
    });
}

/// Import a data file: the input buffer holds `name_len` bytes of file name followed by the file.
#[no_mangle]
pub extern "C" fn ot_import(name_len: u32, total_len: u32) {
    with_host(|h, _| {
        let b = h.input()[..(total_len as usize).min(h.input().len())].to_vec();
        let n = (name_len as usize).min(b.len());
        h.import_data(&String::from_utf8_lossy(&b[..n]), &b[n..]);
    });
}

/// Save the project; the `.otb` bytes are in the output buffer (returns the length).
#[no_mangle]
pub extern "C" fn ot_save_project() -> u32 {
    with_host(|h, out| {
        *out = h.save_project();
        out.len() as u32
    })
}

/// Draw the window; the binary canvas stream is in the output buffer (returns the length).
#[no_mangle]
pub extern "C" fn ot_render() -> u32 {
    with_host(|h, out| {
        *out = h.render().to_vec();
        out.len() as u32
    })
}

/// The menu bar as JSON, in the output buffer (returns the length).
#[no_mangle]
pub extern "C" fn ot_menu_json() -> u32 {
    with_host(|h, out| {
        *out = h.menu_json().into_bytes();
        out.len() as u32
    })
}

/// The window title, in the output buffer (returns the length).
#[no_mangle]
pub extern "C" fn ot_title() -> u32 {
    with_host(|h, out| {
        *out = h.title().into_bytes();
        out.len() as u32
    })
}

/// The active theme's name (`light`, `dark`, `high-contrast`), in the output buffer (returns the
/// length). The page saves it and restores it at start-up with the `view.theme_*` commands.
#[no_mangle]
pub extern "C" fn ot_theme() -> u32 {
    with_host(|h, out| {
        *out = h.theme_name().as_bytes().to_vec();
        out.len() as u32
    })
}

/// Take the next queued outcome (see the module documentation); `0` when there is none.
#[no_mangle]
pub extern "C" fn ot_outcome_next() -> u32 {
    with(|s| {
        let Some(h) = s.host.as_mut() else { return 0 };
        s.slots = Default::default();
        s.pages = 0;
        if !h.next_outcome() {
            return 0;
        }
        match h.outcome().cloned() {
            Some(Outcome::File(f)) => {
                s.slots[0] = f.name.into_bytes();
                s.slots[1] = f.mime.as_bytes().to_vec();
                s.slots[2] = f.bytes;
                1
            }
            Some(Outcome::Clipboard(t)) => {
                s.slots[0] = t.into_bytes();
                2
            }
            Some(Outcome::Print(p)) => {
                s.pages = p.len();
                3
            }
            Some(Outcome::NeedOpenFile) => 4,
            Some(Outcome::NeedImportFile) => 5,
            Some(Outcome::Error(m)) => {
                s.slots[0] = m.into_bytes();
                6
            }
            Some(Outcome::TearOffGraph { name, json }) => {
                s.slots[0] = name.into_bytes();
                s.slots[3] = json.into_bytes();
                7
            }
            Some(Outcome::None) | None => 0,
        }
    })
}

/// Offset of outcome slot `slot` (`0..=3`).
#[no_mangle]
pub extern "C" fn ot_slot_ptr(slot: u32) -> usize {
    with(|s| s.slots.get(slot as usize).map_or(0, |v| v.as_ptr() as usize))
}

/// Length of outcome slot `slot`.
#[no_mangle]
pub extern "C" fn ot_slot_len(slot: u32) -> u32 {
    with(|s| s.slots.get(slot as usize).map_or(0, |v| v.len() as u32))
}

/// Number of pages of the current print outcome.
#[no_mangle]
pub extern "C" fn ot_print_pages() -> u32 {
    with(|s| s.pages as u32)
}

/// Draw print page `i`; the canvas stream is in the output buffer (returns the length).
#[no_mangle]
pub extern "C" fn ot_print_frame(i: u32) -> u32 {
    with_host(|h, out| {
        *out = h.page_frame(i as usize).to_vec();
        out.len() as u32
    })
}

/// Open a graph window from the `SessionReport` JSON in the input buffer. Returns a handle
/// (`> 0`), or `0` on error (the message is in the output buffer, length from [`ot_out_len`]).
#[no_mangle]
pub extern "C" fn ot_graph_open(len: u32) -> u32 {
    with(|s| {
        let Some(h) = s.host.as_ref() else { return 0 };
        let json = String::from_utf8_lossy(&h.input()[..(len as usize).min(h.input().len())]).into_owned();
        match GraphHost::from_json(&json) {
            Ok(g) => {
                s.graphs.push(Some(g));
                s.graphs.len() as u32
            }
            Err(e) => {
                s.out = e.into_bytes();
                0
            }
        }
    })
}

/// Length of the output buffer.
#[no_mangle]
pub extern "C" fn ot_out_len() -> u32 {
    with(|s| s.out.len() as u32)
}

/// A graph window's title, in the output buffer (returns the length).
#[no_mangle]
pub extern "C" fn ot_graph_title(handle: u32) -> u32 {
    with(|s| {
        let t = s.graphs.get((handle as usize).wrapping_sub(1)).and_then(Option::as_ref).map(|g| g.name.clone());
        s.out = t.unwrap_or_default().into_bytes();
        s.out.len() as u32
    })
}

/// Draw a graph window at `width × height`; the canvas stream is in the output buffer.
#[no_mangle]
pub extern "C" fn ot_graph_render(handle: u32, width: f64, height: f64) -> u32 {
    with(|s| {
        let bytes = s
            .graphs
            .get_mut((handle as usize).wrapping_sub(1))
            .and_then(Option::as_mut)
            .map(|g| g.render(width, height).to_vec());
        s.out = bytes.unwrap_or_default();
        s.out.len() as u32
    })
}

/// Close a graph window.
#[no_mangle]
pub extern "C" fn ot_graph_close(handle: u32) {
    with(|s| {
        if let Some(slot) = s.graphs.get_mut((handle as usize).wrapping_sub(1)) {
            *slot = None;
        }
    });
}

/// Safe stand-ins for reading and writing linear memory, for native embedders and tests: where a
/// WebAssembly shell writes at [`ot_input_reserve`]'s offset and reads at [`ot_out_ptr`]'s, native
/// code calls these instead.
pub mod memory {
    use super::{with, with_host};

    /// Put `bytes` in the input buffer (what a shell does through [`super::ot_input_reserve`]).
    pub fn write_input(bytes: &[u8]) {
        with_host(|h, _| h.reserve_input(bytes.len()).copy_from_slice(bytes));
    }

    /// Copy of the output buffer.
    pub fn read_out() -> Vec<u8> {
        with(|s| s.out.clone())
    }

    /// Copy of outcome slot `slot`.
    pub fn read_slot(slot: usize) -> Vec<u8> {
        with(|s| s.slots.get(slot).cloned().unwrap_or_default())
    }
}
