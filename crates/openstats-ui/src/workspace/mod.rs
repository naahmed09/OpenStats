//! The split workspace: Project Navigator, Session window, worksheet with formula bar, menu bar
//! and modal dialogs, composed into one scene.

pub mod cmdline;
pub mod export;
pub mod helpdoc;
pub mod keymap;
pub mod linked;
pub mod macro_editor;
pub mod navigator;
pub mod project;
pub mod session;
mod shell;

pub use cmdline::{CommandLine, HistoryEntry, HistoryKind};
pub use macro_editor::{EditorEvent, MacroEditor, RunState};
pub use navigator::{Button, Folder, NavAction, NavEvent, NavModel, Navigator, Target};
pub use project::{graph_stem, GraphItem, ItemId, MacroItem, SessionItem};
pub use session::{SessionHit, SessionView};
pub use shell::{ExportFile, Focus, Outcome, Top, Workspace};
