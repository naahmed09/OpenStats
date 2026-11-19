//! Numerically careful statistical building blocks.
//!
//! These are the first pieces of the statistics layer, written to be
//! certifiable against the NIST Statistical Reference Datasets:
//!
//! | item | contents |
//! |---|---|
//! | [`summary`] | compensated sums, two-pass mean, variance, standard deviation, lag-k autocorrelation |
//! | [`ols`] | linear least squares by Householder QR with iterative refinement: coefficients, standard errors, covariance, R², residual SD |
//! | [`anova_oneway`] | one-way analysis of variance table |

mod anova;
mod ols;
pub mod summary;

pub use anova::{anova_oneway, OneWayAnova};
pub use ols::{ols, OlsFit, R2Kind};
