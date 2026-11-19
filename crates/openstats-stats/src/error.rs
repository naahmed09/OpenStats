//! Error type for statistical procedures.

use openstats_math::MathError;
use std::fmt;

/// Failure of a statistical procedure.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// An argument is outside its valid range or inputs disagree in shape.
    InvalidArgument(String),
    /// Too few usable observations for the requested quantity.
    InsufficientData { what: &'static str, needed: usize, got: usize },
    /// The data contain ±∞ (missing values must be `NaN`).
    NonFinite,
    /// The computation exceeds a documented size limit.
    TooLarge(String),
    /// A numerical routine from `openstats-math` failed.
    Math(MathError),
}

/// `Result` specialised to [`StatsError`].
pub type Result<T> = std::result::Result<T, StatsError>;

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            StatsError::InsufficientData { what, needed, got } => {
                write!(f, "{what} needs at least {needed} observations, got {got}")
            }
            StatsError::NonFinite => f.write_str("data contain infinite values"),
            StatsError::TooLarge(m) => write!(f, "problem too large: {m}"),
            StatsError::Math(e) => write!(f, "numerical error: {e:?}"),
        }
    }
}

impl std::error::Error for StatsError {}

impl From<MathError> for StatsError {
    fn from(e: MathError) -> Self {
        StatsError::Math(e)
    }
}

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(StatsError::InvalidArgument(msg.into()))
}
