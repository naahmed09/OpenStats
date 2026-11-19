//! Error type for numerical routines.

use std::fmt;

pub type Result<T> = std::result::Result<T, MathError>;

#[derive(Debug, Clone, PartialEq)]
pub enum MathError {
    /// Operand shapes are incompatible.
    DimensionMismatch {
        op: &'static str,
        left: (usize, usize),
        right: (usize, usize),
    },
    /// The matrix must be square.
    NotSquare {
        rows: usize,
        cols: usize,
    },
    /// The matrix is singular (or numerically singular) at the given pivot.
    Singular {
        pivot: usize,
    },
    /// Cholesky failed: the matrix is not (numerically) positive definite.
    NotPositiveDefinite {
        pivot: usize,
    },
    /// An iterative algorithm failed to converge.
    NoConvergence {
        algorithm: &'static str,
        iterations: usize,
    },
    /// The input contains NaN or infinite values.
    NonFinite {
        op: &'static str,
    },
    InvalidArgument(String),
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathError::DimensionMismatch { op, left, right } => {
                write!(f, "{op}: incompatible dimensions {}×{} and {}×{}", left.0, left.1, right.0, right.1)
            }
            MathError::NotSquare { rows, cols } => write!(f, "matrix must be square, got {rows}×{cols}"),
            MathError::Singular { pivot } => write!(f, "matrix is singular (pivot {pivot})"),
            MathError::NotPositiveDefinite { pivot } => {
                write!(f, "matrix is not positive definite (pivot {pivot})")
            }
            MathError::NoConvergence { algorithm, iterations } => {
                write!(f, "{algorithm} did not converge after {iterations} iterations")
            }
            MathError::NonFinite { op } => write!(f, "{op}: input contains NaN or infinite values"),
            MathError::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
        }
    }
}

impl std::error::Error for MathError {}
