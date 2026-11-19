//! A virtualized worksheet grid.
//!
//! [`GridController`] owns a [`openstats_core::Workbook`] with its undo history and all view
//! state (scroll, selection, editor). Input events go in through [`GridController::key_down`],
//! [`GridController::pointer_down`] and friends; [`GridController::render`] returns a
//! [`openstats_charts::Scene`] holding only the visible cells, which the existing canvas, SVG and
//! PNG renderers draw. The crate has no platform bindings: a Wasm host forwards DOM events and
//! paints the scene.

pub mod clipboard;
mod controller;
pub mod format;
mod input;
pub mod metrics;
mod render;
pub mod selection;

pub use controller::{EditTarget, Editor, GridController, Layout};
pub use input::{Hit, Key, Mods};
pub use metrics::{ColumnWidths, Metrics, ScrollProxy, View, MAX_COLS, MAX_ROWS};
pub use selection::{CellRange, CellRef, Selection};
