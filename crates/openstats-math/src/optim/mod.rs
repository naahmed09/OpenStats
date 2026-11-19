//! Optimization.
//!
//! | routine | problem |
//! |---|---|
//! | [`brent_minimize`], [`bracket_minimum`] | one-dimensional minimisation (Brent's method, golden section + parabolic interpolation) |
//! | [`nelder_mead`] | derivative-free simplex search with adaptive coefficients and restarts |
//! | [`bfgs`], [`lbfgs`] | quasi-Newton with strong-Wolfe line search (dense / limited memory) |
//! | [`lbfgs_bounded`] | box-constrained projected limited-memory quasi-Newton |
//! | [`newton`], [`newton_dogleg`] | damped Newton (line search) and trust-region dogleg, both on a modified-Cholesky shifted Hessian |
//! | [`levenberg_marquardt`], [`gauss_newton`] | nonlinear least squares with parameter covariance |
//!
//! Gradients and Jacobians may be supplied analytically; otherwise they are
//! formed by central (gradient, Hessian) or forward (Jacobian) differences.
//! Routines report non-convergence through [`Minimum::converged`] and
//! [`Termination`] rather than discarding the best point found.

mod brent;
mod least_squares;
mod line_search;
mod nelder_mead;
mod newton;
mod quasi_newton;

pub use brent::{bracket_minimum, brent_minimize, Bracket, Min1d};
pub use least_squares::{gauss_newton, levenberg_marquardt, LeastSquaresFit, LsOptions};
pub use line_search::{backtracking, strong_wolfe, LineSearch};
pub use nelder_mead::{nelder_mead, NelderMeadOptions};
pub use newton::{newton, newton_dogleg};
pub use quasi_newton::{bfgs, lbfgs, lbfgs_bounded};

use crate::error::{MathError, Result};

/// Objective function `f: ℝⁿ → ℝ`.
pub type Objective<'a> = &'a mut dyn FnMut(&[f64]) -> f64;
/// Analytic gradient.
pub type GradientFn<'a> = &'a mut dyn FnMut(&[f64]) -> Vec<f64>;
/// Analytic Hessian (row-major `n × n` matrix).
pub type HessianFn<'a> = &'a mut dyn FnMut(&[f64]) -> crate::Matrix;

/// Why an iteration stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The (projected) gradient is below tolerance.
    Gradient,
    /// The objective stopped decreasing (relative change below `ftol`).
    Function,
    /// The step (or simplex) became smaller than `xtol`.
    Step,
    /// The iteration limit was reached.
    MaxIterations,
    /// No acceptable step length could be found along a descent direction.
    LineSearchFailed,
}

/// Common stopping rules.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub max_iter: usize,
    /// Stop when ‖∇f‖∞ ≤ `gtol`·(1 + |f|).
    pub gtol: f64,
    /// Stop when the reduction in f is ≤ `ftol`·max(|f|, 1).
    pub ftol: f64,
    /// Stop when ‖step‖∞ ≤ `xtol`·(1 + ‖x‖∞).
    pub xtol: f64,
    /// History length for [`lbfgs`] / [`lbfgs_bounded`].
    pub memory: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { max_iter: 2000, gtol: 1e-8, ftol: 1e-15, xtol: 1e-14, memory: 10 }
    }
}

/// Result of a minimisation.
#[derive(Debug, Clone)]
pub struct Minimum {
    pub x: Vec<f64>,
    pub f: f64,
    /// Gradient at `x` (empty for derivative-free methods).
    pub grad: Vec<f64>,
    pub iterations: usize,
    pub f_evals: usize,
    pub g_evals: usize,
    pub converged: bool,
    pub termination: Termination,
}

pub(crate) fn invalid(msg: impl Into<String>) -> MathError {
    MathError::InvalidArgument(msg.into())
}

pub(crate) fn check_start(x0: &[f64], who: &str) -> Result<()> {
    if x0.is_empty() {
        return Err(invalid(format!("{who}: empty starting point")));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: "optimizer starting point" });
    }
    Ok(())
}

pub(crate) fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Central-difference gradient. Step `h_i = ε^{1/3}·max(|x_i|, 1)`; the
/// perturbation is made exactly representable so the actual step is used in
/// the quotient.
pub fn gradient_fd(f: &mut dyn FnMut(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut xx = x.to_vec();
    let mut g = vec![0.0; x.len()];
    for i in 0..x.len() {
        let h0 = f64::EPSILON.cbrt() * x[i].abs().max(1.0);
        let (xp, xm) = (x[i] + h0, x[i] - h0);
        let h = (xp - xm) / 2.0;
        xx[i] = xp;
        let fp = f(&xx);
        xx[i] = xm;
        let fm = f(&xx);
        xx[i] = x[i];
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

/// Central-difference Hessian from function values (symmetric).
pub fn hessian_fd(f: &mut dyn FnMut(&[f64]) -> f64, x: &[f64]) -> crate::Matrix {
    let n = x.len();
    let f0 = f(x);
    let h: Vec<f64> = x.iter().map(|v| f64::EPSILON.powf(0.25) * v.abs().max(1.0)).collect();
    let mut hess = crate::Matrix::zeros(n, n);
    let mut xx = x.to_vec();
    for i in 0..n {
        xx[i] = x[i] + h[i];
        let fp = f(&xx);
        xx[i] = x[i] - h[i];
        let fm = f(&xx);
        xx[i] = x[i];
        hess[(i, i)] = (fp - 2.0 * f0 + fm) / (h[i] * h[i]);
        for j in 0..i {
            xx[i] = x[i] + h[i];
            xx[j] = x[j] + h[j];
            let fpp = f(&xx);
            xx[j] = x[j] - h[j];
            let fpm = f(&xx);
            xx[i] = x[i] - h[i];
            let fmm = f(&xx);
            xx[j] = x[j] + h[j];
            let fmp = f(&xx);
            xx[i] = x[i];
            xx[j] = x[j];
            let v = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j]);
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
    hess
}

/// Hessian from gradient differences, symmetrised.
pub fn hessian_from_gradient(g: &mut dyn FnMut(&[f64]) -> Vec<f64>, x: &[f64]) -> crate::Matrix {
    let n = x.len();
    let mut hess = crate::Matrix::zeros(n, n);
    let mut xx = x.to_vec();
    for j in 0..n {
        let h0 = f64::EPSILON.cbrt() * x[j].abs().max(1.0);
        let (xp, xm) = (x[j] + h0, x[j] - h0);
        let h = (xp - xm) / 2.0;
        xx[j] = xp;
        let gp = g(&xx);
        xx[j] = xm;
        let gm = g(&xx);
        xx[j] = x[j];
        for i in 0..n {
            hess[(i, j)] = (gp[i] - gm[i]) / (2.0 * h);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
    hess
}

/// Test problems used by the unit tests and examples.
pub mod testfn {
    /// Generalised Rosenbrock Σ 100(x_{i+1} − x_i²)² + (1 − x_i)²; minimum 0 at (1, …, 1).
    pub fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
    }
    pub fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
        }
        g
    }
    pub fn rosenbrock_hess(x: &[f64]) -> crate::Matrix {
        let n = x.len();
        let mut h = crate::Matrix::zeros(n, n);
        for i in 0..n - 1 {
            h[(i, i)] += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
            h[(i, i + 1)] += -400.0 * x[i];
            h[(i + 1, i)] += -400.0 * x[i];
            h[(i + 1, i + 1)] += 200.0;
        }
        h
    }
    /// Himmelblau (x² + y − 11)² + (x + y² − 7)²: four minima of value 0.
    pub fn himmelblau(x: &[f64]) -> f64 {
        (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2)
    }
    pub fn himmelblau_grad(x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0);
        vec![4.0 * x[0] * a + 2.0 * b, 2.0 * a + 4.0 * x[1] * b]
    }
    /// Beale; minimum 0 at (3, 0.5).
    pub fn beale(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (1.5 - a + a * b).powi(2) + (2.25 - a + a * b * b).powi(2) + (2.625 - a + a * b * b * b).powi(2)
    }
    /// Powell's singular function in four variables; minimum 0 at the origin
    /// (with a singular Hessian there).
    pub fn powell_singular(x: &[f64]) -> f64 {
        (x[0] + 10.0 * x[1]).powi(2)
            + 5.0 * (x[2] - x[3]).powi(2)
            + (x[1] - 2.0 * x[2]).powi(4)
            + 10.0 * (x[0] - x[3]).powi(4)
    }
    /// Wood's function; minimum 0 at (1, 1, 1, 1).
    pub fn wood(x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0] * x[0]).powi(2)
            + (1.0 - x[0]).powi(2)
            + 90.0 * (x[3] - x[2] * x[2]).powi(2)
            + (1.0 - x[2]).powi(2)
            + 10.1 * ((x[1] - 1.0).powi(2) + (x[3] - 1.0).powi(2))
            + 19.8 * (x[1] - 1.0) * (x[3] - 1.0)
    }
    /// Goldstein–Price; global minimum 3 at (0, −1).
    pub fn goldstein_price(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (1.0 + (a + b + 1.0).powi(2) * (19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b + 6.0 * a * b + 3.0 * b * b))
            * (30.0
                + (2.0 * a - 3.0 * b).powi(2)
                    * (18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b + 27.0 * b * b))
    }
    /// Booth; minimum 0 at (1, 3).
    pub fn booth(x: &[f64]) -> f64 {
        (x[0] + 2.0 * x[1] - 7.0).powi(2) + (2.0 * x[0] + x[1] - 5.0).powi(2)
    }
}
