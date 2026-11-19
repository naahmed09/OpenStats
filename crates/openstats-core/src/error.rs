//! Error type shared by every module of `openstats-core`.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, CoreError>;

/// All recoverable failures raised by the data engine.
#[derive(Debug, Clone, PartialEq)]
pub enum CoreError {
    /// A column, worksheet, constant or matrix reference did not resolve.
    NotFound(String),
    /// A row/column/worksheet index was outside the valid range.
    IndexOutOfRange {
        what: &'static str,
        index: usize,
        len: usize,
    },
    /// A value of one type was written to a column of another type.
    TypeMismatch {
        expected: &'static str,
        found: &'static str,
    },
    /// Columns that must have equal length did not.
    LengthMismatch {
        expected: usize,
        found: usize,
    },
    /// Text could not be parsed (date-time, number, reference, JSON, CSV …).
    Parse {
        context: &'static str,
        message: String,
        position: Option<usize>,
    },
    /// Invalid argument supplied to an operation.
    InvalidArgument(String),
    /// A name already exists where names must be unique.
    Duplicate(String),
    /// Binary/JSON project file is corrupt or of an unsupported version.
    Format(String),
    /// Underlying I/O error (message only, so the error stays `Clone`).
    Io(String),
    /// Undo/redo stack has nothing to apply.
    NothingToUndo,
    NothingToRedo,
}

impl CoreError {
    pub(crate) fn parse(context: &'static str, message: impl Into<String>, position: Option<usize>) -> Self {
        CoreError::Parse { context, message: message.into(), position }
    }
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        CoreError::InvalidArgument(message.into())
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::NotFound(s) => write!(f, "not found: {s}"),
            CoreError::IndexOutOfRange { what, index, len } => {
                write!(f, "{what} index {index} out of range (length {len})")
            }
            CoreError::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {expected}, found {found}")
            }
            CoreError::LengthMismatch { expected, found } => {
                write!(f, "length mismatch: expected {expected}, found {found}")
            }
            CoreError::Parse { context, message, position } => match position {
                Some(p) => write!(f, "{context} parse error at {p}: {message}"),
                None => write!(f, "{context} parse error: {message}"),
            },
            CoreError::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            CoreError::Duplicate(s) => write!(f, "duplicate name: {s}"),
            CoreError::Format(s) => write!(f, "file format error: {s}"),
            CoreError::Io(s) => write!(f, "I/O error: {s}"),
            CoreError::NothingToUndo => write!(f, "nothing to undo"),
            CoreError::NothingToRedo => write!(f, "nothing to redo"),
        }
    }
}

impl std::error::Error for CoreError {}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::Io(e.to_string())
    }
}
