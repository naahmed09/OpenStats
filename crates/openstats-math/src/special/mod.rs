//! Special functions.
//!
//! | module | functions |
//! |---|---|
//! | [`gamma`] | Γ, lnΓ, ψ (digamma), ψ′ (trigamma), B, ln B, factorials, sin(πx) |
//! | [`erf`] | erf, erfc, erfcx, inverses; standard normal pdf/cdf/sf/quantile |
//! | [`incgamma`] | regularised incomplete gamma P(a,x), Q(a,x) and inverses |
//! | [`incbeta`] | regularised incomplete beta I_x(a,b) and its inverse |
//! | [`bessel`] | J_n, Y_n, I_n, K_n (integer order) |
//! | [`tukey`] | studentized range distribution (cdf and quantile) |
//! | [`dunnett`] | Dunnett's many-to-one distribution (one- and two-sided) |
//! | [`owen`] | Owen's T function |
//! | [`expint`] | exponential integrals E₁, Ei |
//! | [`zeta`] | Riemann zeta function ζ(s) |
//! | [`lambertw`] | Lambert W₀ and W₋₁ |
//!
//! Accuracy target: relative error ≲ 1e-14 in the bulk and ≲ 1e-12 in the
//! far tails wherever IEEE doubles can represent the result; functions near
//! their real zeros (lnΓ near 1 and 2 is the exception: it is accurate there)
//! are accurate in the absolute sense. Every module is checked against SciPy
//! reference values in `tests/scipy_reference.rs`.

pub mod bessel;
pub mod dunnett;
pub mod erf;
pub mod expint;
pub mod gamma;
pub mod incbeta;
pub mod incgamma;
pub mod lambertw;
pub mod owen;
pub mod tukey;
pub mod zeta;

/// Smallest positive normal double divided by machine epsilon: guards
/// modified-Lentz continued fractions against division by zero.
pub(crate) const FPMIN: f64 = f64::MIN_POSITIVE / f64::EPSILON;
