//! # openstats-math
//!
//! OpenStats's numerical kernel, written against the Rust standard library only.
//!
//! Phase 1c provides [`linalg`]: dense matrices with cache-blocked GEMM, LU,
//! Householder QR with column pivoting, Cholesky / LDLᵀ / modified Cholesky,
//! symmetric eigensolvers (tridiagonal QL and cyclic Jacobi), the general
//! real eigensolver (Hessenberg + Francis double-shift QR), Golub–Kahan–Reinsch
//! SVD, pseudo-inverse and least squares, plus sparse CSR/CSC matrices with a
//! minimum-degree sparse Cholesky and preconditioned conjugate gradients.
//!
//! Phase 1d adds [`special`] functions (Γ family, erf family, incomplete
//! gamma/beta and inverses, Bessel, studentized range, Dunnett), the
//! continuous distributions in [`dist`], quadrature ([`integrate`]), root
//! finding ([`roots`]) and — with the default `formula` feature — calculator
//! functions in `calc` (NSCORES, PDF, CDF, SFCDF, INVCDF).
//!
//! Phase 1e adds Owen's T, E₁/Ei, ζ, Lambert W, the discrete and non-central
//! distributions. Phase 1f adds [`optim`] (Brent, Nelder–Mead, BFGS, L-BFGS,
//! bounded L-BFGS, Newton and dogleg trust region, Levenberg–Marquardt,
//! Gauss–Newton with parameter covariance) and [`random`] (MT19937, MT19937-64,
//! PCG64, xoshiro256**, exact variate samplers for every distribution,
//! multivariate normal, Wishart, Dirichlet, multinomial) together with the
//! calculator functions `RANDOM`, `RANDOMSEED` and `BASE`.

#[cfg(feature = "formula")]
pub mod calc;
pub mod complex;
pub mod dist;
pub mod error;
pub mod integrate;
pub mod linalg;
pub mod optim;
pub mod random;
pub mod roots;
pub mod special;
pub mod stats;

pub use complex::Complex;
pub use error::{MathError, Result};
pub use linalg::Matrix;
