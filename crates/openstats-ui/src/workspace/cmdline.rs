//! The command line and the History list.
//!
//! The command line takes the same language as a macro (`openstats-script`): one or more
//! commands, `;` subcommands, `LET` and the rest. Every command typed is echoed into the Session
//! window as `MTB > …`, runs as one undo step, and is kept in the history together with the
//! dialogs and macros that were run. Up and Down recall earlier commands.

use crate::grid::{EditTarget, Editor};

/// What a history row records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    /// Text typed on the command line (can be recalled and run again).
    Command,
    /// A dialog that was run from a menu.
    Dialog,
    /// A macro that was run from the Macro Editor.
    Macro,
}

impl HistoryKind {
    fn code(self) -> &'static str {
        match self {
            HistoryKind::Command => "C",
            HistoryKind::Dialog => "D",
            HistoryKind::Macro => "M",
        }
    }

    fn from_code(c: &str) -> Option<HistoryKind> {
        match c {
            "C" => Some(HistoryKind::Command),
            "D" => Some(HistoryKind::Dialog),
            "M" => Some(HistoryKind::Macro),
            _ => None,
        }
    }
}

/// One row of the History list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryEntry {
    /// What it was.
    pub kind: HistoryKind,
    /// The command text, the dialog title or the macro name.
    pub text: String,
    /// Whether it completed without an error.
    pub ok: bool,
}

/// Most history rows kept (the oldest are dropped).
pub const HISTORY_LIMIT: usize = 500;

/// State of the command line and the History list.
#[derive(Debug, Clone)]
pub struct CommandLine {
    /// Whether the command line is shown.
    pub visible: bool,
    /// Whether the History list is shown.
    pub history_open: bool,
    /// The text being typed.
    pub editor: Editor,
    entries: Vec<HistoryEntry>,
    /// Position in the recalled commands (an index into `entries`), when Up/Down is in use.
    recall: Option<usize>,
    /// What was typed before recalling started.
    draft: String,
}

impl Default for CommandLine {
    fn default() -> Self {
        CommandLine {
            visible: false,
            history_open: false,
            editor: Editor::new(EditTarget::ColumnName(0), String::new(), true),
            entries: Vec::new(),
            recall: None,
            draft: String::new(),
        }
    }
}

impl CommandLine {
    /// The history, oldest first.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Text typed so far.
    pub fn text(&self) -> &str {
        &self.editor.buffer
    }

    /// Replace the text (the caret goes to the end).
    pub fn set_text(&mut self, text: &str) {
        self.editor = Editor::new(EditTarget::ColumnName(0), text.to_string(), true);
    }

    /// Append a row to the history.
    pub fn record(&mut self, kind: HistoryKind, text: &str, ok: bool) {
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.entries.push(HistoryEntry { kind, text, ok });
        if self.entries.len() > HISTORY_LIMIT {
            let extra = self.entries.len() - HISTORY_LIMIT;
            self.entries.drain(..extra);
        }
        self.recall = None;
    }

    /// Forget every row.
    pub fn clear_history(&mut self) {
        self.entries.clear();
        self.recall = None;
    }

    /// Step back (`older`) or forward through the commands that were typed. Going past the newest
    /// restores what was being typed.
    pub fn recall_step(&mut self, older: bool) {
        let cmds: Vec<usize> =
            self.entries.iter().enumerate().filter(|(_, e)| e.kind == HistoryKind::Command).map(|(i, _)| i).collect();
        if cmds.is_empty() {
            return;
        }
        let pos = self.recall.and_then(|r| cmds.iter().position(|&i| i == r));
        let next = match (pos, older) {
            (None, true) => {
                self.draft = self.editor.buffer.clone();
                Some(cmds.len() - 1)
            }
            (None, false) => return,
            (Some(p), true) => Some(p.saturating_sub(1)),
            (Some(p), false) => (p + 1 < cmds.len()).then_some(p + 1),
        };
        match next {
            Some(p) => {
                self.recall = Some(cmds[p]);
                let t = self.entries[cmds[p]].text.clone();
                self.set_text(&t);
            }
            None => {
                self.recall = None;
                let d = std::mem::take(&mut self.draft);
                self.set_text(&d);
            }
        }
    }

    /// The rows as text for saving in a project: `kind<TAB>ok<TAB>text` per line (line breaks in
    /// the text become the two characters `\n`).
    pub fn save_text(&self) -> String {
        self.entries
            .iter()
            .map(|e| {
                format!(
                    "{}\t{}\t{}",
                    e.kind.code(),
                    if e.ok { 1 } else { 0 },
                    e.text.replace('\\', "\\\\").replace('\n', "\\n").replace('\t', " ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Restore rows written by [`CommandLine::save_text`]; unreadable lines are skipped.
    pub fn load_text(&mut self, text: &str) {
        self.entries.clear();
        for line in text.lines() {
            let mut p = line.splitn(3, '\t');
            let (Some(k), Some(ok), Some(t)) = (p.next(), p.next(), p.next()) else { continue };
            let Some(kind) = HistoryKind::from_code(k) else { continue };
            let mut out = String::new();
            let mut it = t.chars();
            while let Some(c) = it.next() {
                if c == '\\' {
                    match it.next() {
                        Some('n') => out.push('\n'),
                        Some(o) => out.push(o),
                        None => {}
                    }
                } else {
                    out.push(c);
                }
            }
            self.entries.push(HistoryEntry { kind, text: out, ok: ok == "1" });
        }
        let extra = self.entries.len().saturating_sub(HISTORY_LIMIT);
        self.entries.drain(..extra);
        self.recall = None;
    }
}
