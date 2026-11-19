//! Nonlinear least squares: Levenberg–Marquardt and Gauss–Newton.

use super::{check_start, dot, invalid, Termination};
use crate::error::{MathError, Result};
use crate::linalg::{Matrix, Qr, Svd};

/// Residual function r(x) ∈ ℝᵐ.
pub type ResidualFn<'a> = &'a mut dyn FnMut(&[f64]) -> Vec<f64>;
/// Analytic Jacobian J(x) (m × n, J_ij = ∂r_i/∂x_j).
pub type JacobianFn<'a> = &'a mut dyn FnMut(&[f64]) -> Matrix;

/// Settings for [`levenberg_marquardt`] and [`gauss_newton`].
#[derive(Debug, Clone, Copy)]
pub struct LsOptions {
    pub max_iter: usize,
    /// Stop when the cosine of the angle between r and every column of J is
    /// ≤ `gtol` (the MINPACK criterion; scale free), or r = 0.
    pub gtol: f64,
    /// Stop when an accepted step reduces ‖r‖² by a relative amount ≤ `ftol`
    /// and the model predicts no more.
    pub ftol: f64,
    /// Stop when the scaled step ‖D·p‖ ≤ `xtol`·(‖D·x‖ + xtol).
    pub xtol: f64,
    /// Difference the Jacobian centrally (2n residual evaluations, more
    /// accurate) instead of forwards (n evaluations).
    pub central_jacobian: bool,
}

impl Default for LsOptions {
    fn default() -> Self {
        LsOptions { max_iter: 500, gtol: 1e-12, ftol: 1e-15, xtol: 1e-14, central_jacobian: false }
    }
}

/// Result of a nonlinear least-squares fit.
#[derive(Debug, Clone)]
pub struct LeastSquaresFit {
    pub x: Vec<f64>,
    pub residuals: Vec<f64>,
    /// Σ rᵢ².
    pub sse: f64,
    /// Jacobian at `x`.
    pub jacobian: Matrix,
    pub iterations: usize,
    pub f_evals: usize,
    pub j_evals: usize,
    pub converged: bool,
    pub termination: Termination,
}

impl LeastSquaresFit {
    /// Residual degrees of freedom m − n (may be ≤ 0).
    pub fn dof(&self) -> isize {
        self.residuals.len() as isize - self.x.len() as isize
    }

    /// Residual mean square SSE/(m − n); `None` unless m > n.
    pub fn mse(&self) -> Option<f64> {
        (self.dof() > 0).then(|| self.sse / self.dof() as f64)
    }

    /// Estimated parameter covariance σ²·(JᵀJ)⁻¹ with σ² = SSE/(m − n),
    /// computed from the SVD of J (so it is accurate for ill-conditioned
    /// problems). Singular values below max(m, n)·ε·σ_max are treated as
    /// zero, giving the pseudo-inverse covariance of a rank-deficient fit.
    /// `None` unless m > n.
    pub fn covariance(&self) -> Result<Option<Matrix>> {
        let Some(s2) = self.mse() else { return Ok(None) };
        let svd = Svd::new(&self.jacobian)?;
        let n = self.x.len();
        let smax = svd.s.first().copied().unwrap_or(0.0);
        let tol = self.jacobian.rows().max(n) as f64 * f64::EPSILON * smax;
        let mut cov = Matrix::zeros(n, n);
        for (k, &sk) in svd.s.iter().enumerate() {
            if sk > tol {
                for i in 0..n {
                    for j in 0..n {
                        cov[(i, j)] += svd.v[(i, k)] * svd.v[(j, k)] / (sk * sk);
                    }
                }
            }
        }
        for v in cov.as_mut_slice() {
            *v *= s2;
        }
        Ok(Some(cov))
    }

    /// Standard errors √diag(covariance).
    pub fn std_errors(&self) -> Result<Option<Vec<f64>>> {
        Ok(self.covariance()?.map(|c| (0..c.rows()).map(|i| c[(i, i)].sqrt()).collect()))
    }

    /// Numerical rank of the Jacobian at the solution.
    pub fn rank(&self) -> Result<usize> {
        Ok(Svd::new(&self.jacobian)?.rank(None))
    }
}

struct Eval<'a, 'b> {
    r: ResidualFn<'a>,
    j: Option<JacobianFn<'b>>,
    central: bool,
    m: usize,
    n: usize,
    f_evals: usize,
    j_evals: usize,
}

impl Eval<'_, '_> {
    fn resid(&mut self, x: &[f64]) -> Result<Vec<f64>> {
        self.f_evals += 1;
        let r = (self.r)(x);
        if r.len() != self.m {
            return Err(invalid(format!("residual function returned {} values, expected {}", r.len(), self.m)));
        }
        Ok(r)
    }

    fn jacobian(&mut self, x: &[f64], r0: &[f64]) -> Result<Matrix> {
        self.j_evals += 1;
        let (m, n) = (self.m, self.n);
        if let Some(j) = self.j.as_mut() {
            let jm = j(x);
            if jm.shape() != (m, n) {
                return Err(invalid(format!("Jacobian is {}×{}, expected {m}×{n}", jm.rows(), jm.cols())));
            }
            if jm.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(MathError::NonFinite { op: "Jacobian" });
            }
            return Ok(jm);
        }
        let mut jm = Matrix::zeros(m, n);
        let mut xx = x.to_vec();
        for k in 0..n {
            if self.central {
                let h0 = f64::EPSILON.cbrt() * x[k].abs().max(1.0);
                let (xp, xm) = (x[k] + h0, x[k] - h0);
                let h = (xp - xm) / 2.0;
                xx[k] = xp;
                let rp = self.resid(&xx)?;
                xx[k] = xm;
                let rm = self.resid(&xx)?;
                for i in 0..m {
                    jm[(i, k)] = (rp[i] - rm[i]) / (2.0 * h);
                }
            } else {
                let h0 = f64::EPSILON.sqrt() * x[k].abs().max(1.0);
                let xp = x[k] + h0;
                let h = xp - x[k];
                xx[k] = xp;
                let rp = self.resid(&xx)?;
                for i in 0..m {
                    jm[(i, k)] = (rp[i] - r0[i]) / h;
                }
            }
            xx[k] = x[k];
        }
        if jm.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(MathError::NonFinite { op: "finite-difference Jacobian" });
        }
        Ok(jm)
    }
}

fn col_norms(j: &Matrix) -> Vec<f64> {
    (0..j.cols()).map(|c| (0..j.rows()).map(|i| j[(i, c)] * j[(i, c)]).sum::<f64>().sqrt()).collect()
}

/// ∞-norm of the gradient cosine: max_j |J_jᵀr| / (‖J_j‖·‖r‖).
fn gradient_cosine(j: &Matrix, r: &[f64], norms: &[f64]) -> f64 {
    let rn = dot(r, r).sqrt();
    if rn == 0.0 {
        return 0.0;
    }
    (0..j.cols())
        .map(|c| {
            if norms[c] == 0.0 {
                0.0
            } else {
                ((0..j.rows()).map(|i| j[(i, c)] * r[i]).sum::<f64>() / (norms[c] * rn)).abs()
            }
        })
        .fold(0.0, f64::max)
}

fn finish(
    ev: &Eval,
    x: Vec<f64>,
    r: Vec<f64>,
    j: Matrix,
    iters: usize,
    term: Termination,
    opts: &LsOptions,
) -> LeastSquaresFit {
    let sse = dot(&r, &r);
    let norms = col_norms(&j);
    // A zero Jacobian column means the residuals do not respond to that
    // parameter at all (e.g. the model has underflowed): the gradient test is
    // vacuously met there, but the point is degenerate, not a solution.
    let degenerate = sse != 0.0 && norms.contains(&0.0);
    let converged = !degenerate
        && matches!(term, Termination::Gradient | Termination::Function | Termination::Step)
        && (sse == 0.0
            || gradient_cosine(&j, &r, &norms) <= 1e4 * opts.gtol.max(1e-14)
            || term == Termination::Gradient);
    LeastSquaresFit {
        x,
        residuals: r,
        sse,
        jacobian: j,
        iterations: iters,
        f_evals: ev.f_evals,
        j_evals: ev.j_evals,
        converged,
        termination: term,
    }
}

/// Levenberg–Marquardt with Marquardt (Moré) column scaling: each step solves
/// min ‖[J; √λ·D]·p + [r; 0]‖ by QR, where D holds the largest column norms of
/// J seen so far. λ is adapted from the gain ratio (Nielsen's rule).
/// `jac = None` differences the Jacobian numerically.
pub fn levenberg_marquardt(
    resid: ResidualFn,
    jac: Option<JacobianFn>,
    x0: &[f64],
    opts: &LsOptions,
) -> Result<LeastSquaresFit> {
    check_start(x0, "levenberg_marquardt")?;
    let n = x0.len();
    let mut ev = Eval { r: resid, j: jac, central: opts.central_jacobian, m: 0, n, f_evals: 0, j_evals: 0 };
    let mut x = x0.to_vec();
    let mut r = {
        ev.f_evals += 1;
        (ev.r)(&x)
    };
    ev.m = r.len();
    let m = r.len();
    if m == 0 {
        return Err(invalid("levenberg_marquardt: no residuals"));
    }
    if r.iter().any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: "residuals at the starting point" });
    }
    let mut j = ev.jacobian(&x, &r)?;
    let mut d = col_norms(&j).into_iter().map(|v| if v > 0.0 { v } else { 1.0 }).collect::<Vec<_>>();
    let mut sse = dot(&r, &r);
    let (mut lambda, mut nu) = (1e-3f64, 2.0f64);
    let mut iters = 0;
    let mut term = Termination::MaxIterations;
    'outer: while iters < opts.max_iter {
        let norms = col_norms(&j);
        if sse == 0.0 || gradient_cosine(&j, &r, &norms) <= opts.gtol {
            term = Termination::Gradient;
            break;
        }
        for (dk, nk) in d.iter_mut().zip(&norms) {
            *dk = dk.max(*nk);
        }
        // Inner loop: raise λ until a step reduces the sum of squares.
        loop {
            let mut a = Matrix::zeros(m + n, n);
            let mut b = vec![0.0; m + n];
            for i in 0..m {
                a.row_mut(i).copy_from_slice(j.row(i));
                b[i] = -r[i];
            }
            let sl = lambda.sqrt();
            for k in 0..n {
                a[(m + k, k)] = sl * d[k];
            }
            let p = Qr::new(&a)?.solve_least_squares(&b)?;
            let jp = j.matvec(&p)?;
            let pred = 0.5 * (sse - r.iter().zip(&jp).map(|(ri, ji)| (ri + ji) * (ri + ji)).sum::<f64>());
            let xt: Vec<f64> = x.iter().zip(&p).map(|(a, c)| a + c).collect();
            let step_scaled = (0..n).map(|k| (d[k] * p[k]).powi(2)).sum::<f64>().sqrt();
            let x_scaled = (0..n).map(|k| (d[k] * x[k]).powi(2)).sum::<f64>().sqrt();
            let small_step = step_scaled <= opts.xtol * (x_scaled + opts.xtol);
            let rt = {
                ev.f_evals += 1;
                (ev.r)(&xt)
            };
            if rt.len() != m {
                return Err(invalid(format!("residual function returned {} values, expected {m}", rt.len())));
            }
            let sse_t = dot(&rt, &rt);
            let rho = if sse_t.is_finite() && pred > 0.0 { 0.5 * (sse - sse_t) / pred } else { -1.0 };
            if rho > 1e-4 {
                iters += 1;
                let rel_drop = (sse - sse_t) / sse;
                x = xt;
                r = rt;
                sse = sse_t;
                lambda *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                if small_step {
                    term = Termination::Step;
                    j = ev.jacobian(&x, &r)?;
                    break 'outer;
                }
                if rel_drop <= opts.ftol && pred <= opts.ftol * sse.max(f64::MIN_POSITIVE) {
                    term = Termination::Function;
                    j = ev.jacobian(&x, &r)?;
                    break 'outer;
                }
                j = ev.jacobian(&x, &r)?;
                break;
            }
            lambda *= nu;
            nu *= 2.0;
            if small_step || lambda > 1e100 {
                term = if small_step { Termination::Step } else { Termination::LineSearchFailed };
                break 'outer;
            }
        }
    }
    Ok(finish(&ev, x, r, j, iters, term, opts))
}

/// Gauss–Newton with Armijo backtracking on ‖r‖². The step solves the linear
/// least-squares problem min ‖J·p + r‖ by column-pivoted QR (basic solution
/// when J is rank deficient). Converges quadratically on small-residual
/// problems; may fail where Levenberg–Marquardt succeeds.
pub fn gauss_newton(
    resid: ResidualFn,
    jac: Option<JacobianFn>,
    x0: &[f64],
    opts: &LsOptions,
) -> Result<LeastSquaresFit> {
    check_start(x0, "gauss_newton")?;
    let n = x0.len();
    let mut ev = Eval { r: resid, j: jac, central: opts.central_jacobian, m: 0, n, f_evals: 0, j_evals: 0 };
    let mut x = x0.to_vec();
    let mut r = {
        ev.f_evals += 1;
        (ev.r)(&x)
    };
    ev.m = r.len();
    let m = r.len();
    if m == 0 {
        return Err(invalid("gauss_newton: no residuals"));
    }
    if r.iter().any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: "residuals at the starting point" });
    }
    let mut j = ev.jacobian(&x, &r)?;
    let mut sse = dot(&r, &r);
    let mut iters = 0;
    let mut term = Termination::MaxIterations;
    while iters < opts.max_iter {
        let norms = col_norms(&j);
        if sse == 0.0 || gradient_cosine(&j, &r, &norms) <= opts.gtol {
            term = Termination::Gradient;
            break;
        }
        let neg_r: Vec<f64> = r.iter().map(|v| -v).collect();
        let p = Qr::new_pivoted(&j)?.solve_least_squares(&neg_r)?;
        let jp = j.matvec(&p)?;
        let slope = 2.0 * dot(&r, &jp);
        if slope.is_nan() || slope >= 0.0 {
            // No descent along the Gauss-Newton direction (rank-deficient or
            // badly scaled J). That is stationarity only if the gradient is
            // genuinely negligible; otherwise it is a failure, not convergence.
            term = if gradient_cosine(&j, &r, &norms) <= 1e4 * opts.gtol.max(1e-14) {
                Termination::Gradient
            } else {
                Termination::LineSearchFailed
            };
            break;
        }
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let xt: Vec<f64> = x.iter().zip(&p).map(|(a, c)| a + alpha * c).collect();
            ev.f_evals += 1;
            let rt = (ev.r)(&xt);
            if rt.len() != m {
                return Err(invalid(format!("residual function returned {} values, expected {m}", rt.len())));
            }
            let st = dot(&rt, &rt);
            if st.is_finite() && st <= sse + 1e-4 * alpha * slope {
                accepted = Some((xt, rt, st));
                break;
            }
            alpha *= 0.5;
        }
        let Some((xt, rt, st)) = accepted else {
            term = Termination::LineSearchFailed;
            break;
        };
        iters += 1;
        let step = (0..n).map(|k| (alpha * p[k]).abs()).fold(0.0, f64::max);
        let xn = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let rel_drop = (sse - st) / sse;
        x = xt;
        r = rt;
        sse = st;
        j = ev.jacobian(&x, &r)?;
        if step <= opts.xtol * (1.0 + xn) {
            term = Termination::Step;
            break;
        }
        if rel_drop <= opts.ftol {
            term = Termination::Function;
            break;
        }
    }
    Ok(finish(&ev, x, r, j, iters, term, opts))
}
