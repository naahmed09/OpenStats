//! Custom keyboard shortcuts (Phase 6a): chords such as `Ctrl+Shift+K` or `F2` bound to a built-in command or to
//! any dialog of the registry. They are kept with the project.

use crate::dialogs::menu::Command;
use crate::dialogs::Registry;
use crate::grid::{Key, Mods};
use std::collections::BTreeMap;

/// Built-in commands that a shortcut can run, by the name typed in the Customize Shortcuts dialog.
pub const COMMANDS: [(&str, Command); 18] = [
    ("new project", Command::NewProject),
    ("open project", Command::OpenProject),
    ("save project", Command::SaveProject),
    ("new worksheet", Command::NewWorksheet),
    ("import data", Command::ImportData),
    ("print session", Command::PrintSession),
    ("undo", Command::Undo),
    ("redo", Command::Redo),
    ("clear session", Command::ClearSession),
    ("command line", Command::CommandLine),
    ("history", Command::HistoryPane),
    ("formula bar", Command::FormulaBar),
    ("next pane", Command::NextPane),
    ("next theme", Command::ToggleTheme),
    ("keyboard shortcuts", Command::ShowShortcuts),
    ("insert rows", Command::InsertRows),
    ("delete rows", Command::DeleteRows),
    ("run macro", Command::MacroRun),
];

/// Chords that keep their text-editing meaning.
const RESERVED: [&str; 7] = ["ctrl+c", "ctrl+x", "ctrl+v", "ctrl+a", "ctrl+z", "ctrl+y", "ctrl+shift+z"];

/// Function keys the hosts can send.
const FKEYS: [(&str, Key); 7] = [
    ("f1", Key::F1),
    ("f2", Key::F2),
    ("f5", Key::F5),
    ("f6", Key::F6),
    ("f8", Key::F8),
    ("f9", Key::F9),
    ("f10", Key::F10),
];

/// The normal form of a chord typed by the user (`ctrl+alt+shift+key`, lower case), or why it is refused.
pub fn normalize(text: &str) -> Result<String, String> {
    let (mut ctrl, mut alt, mut shift) = (false, false, false);
    let mut key: Option<String> = None;
    for part in text.split('+').map(|p| p.trim().to_ascii_lowercase()).filter(|p| !p.is_empty()) {
        match part.as_str() {
            "ctrl" | "cmd" | "control" => ctrl = true,
            "alt" | "option" => alt = true,
            "shift" => shift = true,
            k => {
                if key.replace(k.to_string()).is_some() {
                    return Err("a shortcut has one key besides Ctrl, Alt and Shift".into());
                }
            }
        }
    }
    let key = key.ok_or("type a shortcut such as Ctrl+Shift+K or F2")?;
    let is_f = FKEYS.iter().any(|(n, _)| *n == key);
    let is_char = key.chars().count() == 1 && key.chars().all(|c| c.is_ascii_alphanumeric());
    if !is_f && !is_char {
        return Err(format!(
            "\"{key}\" is not a key a shortcut can use (letters, digits, F1, F2, F5, F6, F8, F9, F10)"
        ));
    }
    if is_char && !ctrl && !alt {
        return Err("a letter or digit needs Ctrl or Alt".into());
    }
    let mut out = String::new();
    for (on, name) in [(ctrl, "ctrl+"), (alt, "alt+"), (shift, "shift+")] {
        if on {
            out.push_str(name);
        }
    }
    out.push_str(&key);
    if RESERVED.contains(&out.as_str()) {
        return Err(format!("{out} keeps its text-editing meaning and cannot be reassigned"));
    }
    Ok(out)
}

/// The chord of a key press in the normal form, when a shortcut could be bound to it.
pub fn chord_of(key: Key, mods: Mods) -> Option<String> {
    let name = match key {
        Key::Char(c) if c.is_ascii_alphanumeric() => c.to_ascii_lowercase().to_string(),
        k => FKEYS.iter().find(|(_, f)| *f == k)?.0.to_string(),
    };
    let mut out = String::new();
    for (on, n) in [(mods.ctrl, "ctrl+"), (mods.alt, "alt+"), (mods.shift, "shift+")] {
        if on {
            out.push_str(n);
        }
    }
    out.push_str(&name);
    Some(out)
}

/// The shortcuts the user set: chord → command (a built-in name or a dialog id).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyMap {
    map: BTreeMap<String, String>,
}

impl KeyMap {
    /// Whether no shortcut is set.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// All shortcuts, by chord.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(a, b)| (a.as_str(), b.as_str()))
    }

    /// Bind `chord` to `command` (a built-in name or a dialog id); an empty command removes the binding.
    pub fn set(&mut self, reg: &Registry, chord: &str, command: &str) -> Result<String, String> {
        let chord = normalize(chord)?;
        let command = command.trim();
        if command.is_empty() {
            return match self.map.remove(&chord) {
                Some(_) => Ok(format!("Removed the shortcut {chord}")),
                None => Err(format!("{chord} has no custom shortcut")),
            };
        }
        let name = resolve_name(reg, command).ok_or_else(|| {
            format!(
                "\"{command}\" is neither a command name (such as Undo) nor a dialog id (such as stat.basic.ttest1)"
            )
        })?;
        self.map.insert(chord.clone(), name.clone());
        Ok(format!("{chord} now runs {name}"))
    }

    /// Remove every custom shortcut.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// The command bound to this key press.
    pub fn lookup(&self, reg: &Registry, key: Key, mods: Mods) -> Option<Command> {
        let name = self.map.get(&chord_of(key, mods)?)?;
        command_named(reg, name)
    }

    /// Text form for the project file: one `chord<TAB>command` per line.
    pub fn encode(&self) -> String {
        self.map.iter().map(|(a, b)| format!("{a}\t{b}\n")).collect()
    }

    /// Inverse of [`KeyMap::encode`]; lines that no longer make sense are skipped.
    pub fn decode(text: &str, reg: &Registry) -> KeyMap {
        let mut k = KeyMap::default();
        for line in text.lines() {
            if let Some((a, b)) = line.split_once('\t') {
                let _ = k.set(reg, a, b);
            }
        }
        k
    }
}

fn resolve_name(reg: &Registry, command: &str) -> Option<String> {
    let lower = command.to_ascii_lowercase();
    if COMMANDS.iter().any(|(n, _)| *n == lower) {
        return Some(lower);
    }
    reg.get(command).map(|a| a.spec.id.clone())
}

/// The command a name stands for.
pub fn command_named(reg: &Registry, name: &str) -> Option<Command> {
    if let Some((_, c)) = COMMANDS.iter().find(|(n, _)| *n == name) {
        return Some(c.clone());
    }
    reg.get(name).map(|a| Command::OpenDialog(a.spec.id.clone()))
}
