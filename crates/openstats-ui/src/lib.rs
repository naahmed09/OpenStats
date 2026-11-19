//! OpenStats UI logic (Phases 3d and 3e), std only.
//!
//! This crate holds the *behaviour* of the user interface as plain, testable Rust, with no
//! DOM, window system or JavaScript dependency, so the same code drives the WebAssembly
//! build and the Tauri desktop shell:
//!
//! * [`grid`] — the virtualized worksheet grid. A [`grid::GridController`] owns a
//!   `Workbook` with its undo history, the scroll position, selection, in-cell editor and
//!   formula bar, and the clipboard. Hosts forward pointer / keyboard / wheel / paste events
//!   and draw the [`openstats_charts::Scene`] returned by [`grid::GridController::render`]
//!   (through the SVG, canvas-command or PNG renderers of `openstats-charts`). Only the
//!   visible window is ever laid out, so a worksheet of millions of rows costs the same
//!   per frame as a small one.
//!
//! * [`dialogs`] — a declarative dialog system. A `DialogSpec` lists fields (variable
//!   listboxes, numbers, choices, checkboxes) on the Main / Options / Graphs / Storage pages;
//!   the `Registry` binds each spec to a routine in `openstats-stats`, validates the input
//!   against the worksheet and returns a `SessionReport` section plus columns to store. The
//!   `MenuBar` is built from the registry, and `DialogView` is the interactive dialog.
//! * [`workspace`] — the window. [`workspace::Workspace`] composes menu bar, Project
//!   Navigator, Session window, formula bar, grid and the modal dialog, routes abstract input
//!   events by focus (dialog, then menu, then shortcuts, then the focused pane) and draws one
//!   `Scene`. Projects save to `.otb` with Session windows and graphs kept in workbook metadata.
//!
//! The grid does not depend on the dialogs or the workspace.

pub mod dialogs;
pub mod grid;
pub mod style;
pub mod workspace;
