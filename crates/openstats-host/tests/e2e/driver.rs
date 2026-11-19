//! A user at the keyboard and mouse, speaking only the C ABI.
//!
//! Everything here goes through the `ot_*` functions exactly as the JavaScript shell does: bytes
//! in through the input buffer, a canvas command stream and outcome slots out. UI elements are
//! found the way a person finds them, by the text that was drawn, and clicked at that position.

use openstats_charts::render::canvas::{from_binary, Command};
use openstats_host::abi::{self, memory};
use openstats_host::keys::SPECIAL;

/// An outcome the shell must act on (the decoded `ot_outcome_next` slots).
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    File { name: String, mime: String, bytes: Vec<u8> },
    Clipboard(String),
    Print(usize),
    NeedOpen,
    NeedImport,
    Error(String),
    TearOff { name: String, json: String },
}

/// Modifier bits of the wire format.
pub const SHIFT: u32 = 1;
pub const CTRL: u32 = 2;
pub const ALT: u32 = 4;

/// A drawn piece of text.
#[derive(Debug, Clone)]
pub struct Label {
    pub x: f64,
    pub y: f64,
    pub size: f64,
    pub anchor: u8,
    pub text: String,
}

pub struct Driver;

impl Driver {
    pub fn start(width: f64, height: f64) -> Driver {
        assert_eq!(abi::ot_abi_version(), 1);
        assert_eq!(abi::ot_init(width, height), 1);
        Driver
    }

    // ------------------------------------------------------------------ seeing

    pub fn frame(&self) -> Vec<Command> {
        assert!(abi::ot_render() > 8);
        from_binary(&memory::read_out()).expect("a valid OTC1 frame")
    }

    pub fn labels(&self) -> Vec<Label> {
        self.frame()
            .into_iter()
            .filter_map(|c| match c {
                Command::Text { x, y, text, size, anchor, .. } => {
                    Some(Label { x: f64::from(x), y: f64::from(y), size: f64::from(size), anchor, text })
                }
                _ => None,
            })
            .collect()
    }

    pub fn texts(&self) -> Vec<String> {
        self.labels().into_iter().map(|l| l.text).collect()
    }

    pub fn shows(&self, needle: &str) -> bool {
        self.texts().iter().any(|t| t.contains(needle))
    }

    /// The window background colour (the first command is `Clear`).
    pub fn background(&self) -> (u8, u8, u8) {
        match self.frame().first() {
            Some(Command::Clear { color, .. }) => (color.r, color.g, color.b),
            other => panic!("a frame starts with Clear, not {other:?}"),
        }
    }

    pub fn title(&self) -> String {
        abi::ot_title();
        String::from_utf8(memory::read_out()).unwrap()
    }

    // ------------------------------------------------------------------ mouse

    pub fn pointer(&self, kind: u32, x: f64, y: f64, button: u32, mods: u32, clicks: u32) {
        abi::ot_pointer(kind, x, y, button, mods, clicks);
    }

    pub fn click_at(&self, x: f64, y: f64) {
        self.pointer(0, x, y, 0, 0, 1);
        self.pointer(2, x, y, 0, 0, 1);
    }

    pub fn right_click_at(&self, x: f64, y: f64) {
        self.pointer(0, x, y, 2, 0, 1);
        self.pointer(2, x, y, 2, 0, 1);
    }

    pub fn double_click_at(&self, x: f64, y: f64) {
        self.pointer(0, x, y, 0, 0, 1);
        self.pointer(2, x, y, 0, 0, 1);
        self.pointer(0, x, y, 0, 0, 2);
        self.pointer(2, x, y, 0, 0, 2);
    }

    /// Where a label can be clicked: the last (top-most) drawn text equal to `text`.
    pub fn find(&self, text: &str) -> Label {
        self.labels()
            .into_iter()
            .rev()
            .find(|l| l.text == text)
            .unwrap_or_else(|| panic!("no \"{text}\" on screen; saw {:?}", self.texts()))
    }

    fn centre(l: &Label) -> (f64, f64) {
        let w = openstats_charts::core::font::text_width(&l.text, l.size, false);
        let x = match l.anchor {
            1 => l.x,
            2 => l.x - w / 2.0,
            _ => l.x + w / 2.0,
        };
        (x, l.y - l.size * 0.35)
    }

    pub fn click_text(&self, text: &str) {
        let (x, y) = Self::centre(&self.find(text));
        self.click_at(x, y);
    }

    pub fn right_click_text(&self, text: &str) {
        let (x, y) = Self::centre(&self.find(text));
        self.right_click_at(x, y);
    }

    pub fn double_click_text(&self, text: &str) {
        let (x, y) = Self::centre(&self.find(text));
        self.double_click_at(x, y);
    }

    /// Click through a menu path such as `["Stat", "Regression", "Regression..."]`.
    pub fn menu(&self, path: &[&str]) {
        for p in path {
            self.click_text(p);
        }
    }

    pub fn wheel(&self, x: f64, y: f64, dy: f64) {
        abi::ot_wheel(x, y, 0.0, dy);
    }

    // ------------------------------------------------------------------ keyboard

    pub fn key(&self, c: char, mods: u32) {
        abi::ot_key(c as u32, mods);
    }

    pub fn special(&self, n: u32, mods: u32) {
        abi::ot_key(SPECIAL + n, mods);
    }

    pub fn up(&self) {
        self.special(0, 0);
    }
    pub fn down(&self) {
        self.special(1, 0);
    }
    pub fn enter(&self) {
        self.special(8, 0);
    }
    pub fn tab(&self) {
        self.special(9, 0);
    }
    pub fn escape(&self) {
        self.special(10, 0);
    }
    pub fn f6(&self) {
        self.special(14, 0);
    }

    pub fn type_text(&self, s: &str) {
        for c in s.chars() {
            self.key(c, 0);
        }
    }

    pub fn ctrl(&self, c: char) {
        self.key(c, CTRL);
    }

    /// In a dialog's variable list: highlight column `c` (0-based) and press Space.
    pub fn pick_column(&self, c: usize) {
        for _ in 0..64 {
            self.up();
        }
        for _ in 0..c {
            self.down();
        }
        self.key(' ', 0);
    }

    // ------------------------------------------------------------------ commands and files

    pub fn command(&self, id: &str) -> bool {
        memory::write_input(id.as_bytes());
        abi::ot_command(id.len() as u32) == 1
    }

    pub fn import(&self, name: &str, data: &[u8]) {
        let mut buf = name.as_bytes().to_vec();
        buf.extend_from_slice(data);
        memory::write_input(&buf);
        abi::ot_import(name.len() as u32, buf.len() as u32);
    }

    pub fn open_project(&self, bytes: &[u8]) {
        memory::write_input(bytes);
        abi::ot_open_project(bytes.len() as u32);
    }

    pub fn save_project(&self) -> Vec<u8> {
        let n = abi::ot_save_project() as usize;
        let b = memory::read_out();
        assert_eq!(b.len(), n);
        b
    }

    pub fn paste(&self, text: &str) {
        memory::write_input(text.as_bytes());
        abi::ot_paste(text.len() as u32);
    }

    /// Every pending outcome, decoded.
    pub fn outcomes(&self) -> Vec<Out> {
        let mut v = Vec::new();
        loop {
            let k = abi::ot_outcome_next();
            let s = |i: usize| String::from_utf8_lossy(&memory::read_slot(i)).into_owned();
            match k {
                0 => return v,
                1 => v.push(Out::File { name: s(0), mime: s(1), bytes: memory::read_slot(2) }),
                2 => v.push(Out::Clipboard(s(0))),
                3 => v.push(Out::Print(abi::ot_print_pages() as usize)),
                4 => v.push(Out::NeedOpen),
                5 => v.push(Out::NeedImport),
                6 => v.push(Out::Error(s(0))),
                7 => v.push(Out::TearOff { name: s(0), json: s(3) }),
                other => panic!("unknown outcome {other}"),
            }
        }
    }

    /// Run a command that produces exactly one file and return it as `(name, bytes)`.
    pub fn file_from(&self, id: &str) -> (String, Vec<u8>) {
        assert!(self.command(id), "{id}");
        match self.outcomes().as_slice() {
            [Out::File { name, bytes, .. }] => (name.clone(), bytes.clone()),
            other => panic!("{id} should give one file, got {other:?}"),
        }
    }

    /// The Session window as text (File > Export Session).
    pub fn session_text(&self) -> String {
        String::from_utf8(self.file_from("file.export_session").1).unwrap()
    }

    /// The active worksheet as CSV (File > Export Worksheet).
    pub fn worksheet_csv(&self) -> String {
        String::from_utf8(self.file_from("file.export_worksheet").1).unwrap().replace("\r\n", "\n")
    }

    /// Open a torn-off graph window from `SessionReport` JSON; returns its handle and title.
    pub fn graph_open(&self, json: &str) -> (u32, String) {
        memory::write_input(json.as_bytes());
        let h = abi::ot_graph_open(json.len() as u32);
        assert!(h > 0, "{}", String::from_utf8_lossy(&memory::read_out()));
        abi::ot_graph_title(h);
        (h, String::from_utf8(memory::read_out()).unwrap())
    }

    /// Texts drawn by a graph window at a given size.
    pub fn graph_texts(&self, handle: u32, w: f64, h: f64) -> Vec<String> {
        assert!(abi::ot_graph_render(handle, w, h) > 8);
        from_binary(&memory::read_out())
            .unwrap()
            .into_iter()
            .filter_map(|c| if let Command::Text { text, .. } = c { Some(text) } else { None })
            .collect()
    }

    pub fn graph_close(&self, handle: u32) {
        abi::ot_graph_close(handle);
    }
}
