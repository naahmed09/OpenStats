//! # openstats-core
//!
//! The columnar data engine behind OpenStats. Everything here is written against
//! the Rust standard library only.
//!
//! * [`value`] / [`missing`] — cell values and Minitab-compatible missing-value
//!   semantics (`*` for numeric/date-time, empty string for text).
//! * [`datetime`] — proleptic-Gregorian calendar arithmetic on `i64`
//!   microseconds since 1970-01-01T00:00:00Z, parsing and formatting.
//! * [`mod@column`], [`worksheet`], [`workbook`] — typed columns, worksheets with
//!   columns/constants/matrices, and multi-worksheet projects.
//! * [`edit`] / [`history`] — reversible edits and the undo/redo stack.
//! * [`ops`] — data-manipulation operations (stack, unstack, sort, rank, …).
//! * [`formula`] — the calculator: parser, vectorised evaluator, ~150
//!   functions and dependency-ordered column formulas.
//! * [`io`] — CSV, JSON, native `.otb`, DEFLATE/zlib, ZIP, XML, XLSX import,
//!   OLE2 compound files and Minitab container inspection.

pub mod column;
pub mod datetime;
pub mod edit;
pub mod error;
pub mod formula;
pub mod history;
pub mod io;
pub mod locale;
pub mod missing;
pub mod ops;
pub mod refs;
pub mod value;
pub mod workbook;
pub mod worksheet;

pub use column::{Column, ColumnData, ColumnKind, ColumnProps, NumericFormat};
pub use datetime::DateTime;
pub use edit::Edit;
pub use error::{CoreError, Result};
pub use history::History;
pub use value::Value;
pub use workbook::Workbook;
pub use worksheet::{StoredMatrix, Worksheet};
