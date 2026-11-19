//! Newton's method with line search and trust-region dogleg.

use super::line_search::backtracking;
use super::{
    check_start, dot, gradient_fd, hessian_fd, hessian_from_gradient, inf_norm, invalid, GradientFn, HessianFn,
    Minimum, Objective, Options, Termination,
};
use crate::error::{MathError, Result};
use crate::linalg::cholesky::modified_cholesky;
use crate::Matrix;

/// Evaluation of f, ∇f and ∇²f with analytic or finite-difference fallbacks.
struct Model<'a, 'b, 'c> {
    f: Objective<'a>,
    grad: Option<GradientFn<'b>>,
    hess: Option<HessianFn<'c>>,
    f_evals: usize,
    g_evals: usize,
}

impl Model<'_, '_, '_> {
    fn value(&mut self, x: &[f64]) -> f64 {
        self.f_evals += 1;
        (self.f)(x)
    }
    fn gradient(&mut self, x: &[f64]) -> Vec<f64> {
        self.g_evals += 1;
        match self.grad.as_mut() {
            Some(g) => g(x),
            None => {
                self.f_evals += 2 * x.len();
                gradient_fd(self.f, x)
            }
        }
    }
    fn hessian(&mut self, x: &[f64]) -> Matrix {
        if let Some(h) = self.hess.as_mut() {
            return h(x);
        }
        if let Some(g) = self.grad.as_mut() {
            return hessian_from_gradient(g, x);
        }
        self.f_evals += 2 * x.len() * x.len() + 1;
        hessian_fd(self.f, x)
    }
}

fn setup(
    f: Objective,
    grad: Option<GradientFn>,
    hess: Option<HessianFn>,
    x0: &[f64],
    who: &str,
) -> Result<(Vec<f64>, f64)> {
    check_start(x0, who)?;
    let mut m = Model { f, grad, hess, f_evals: 0, g_evals: 0 };
    let fx = m.value(x0);
    if !fx.is_finite() {
        return Err(MathError::NonFinite { op: "objective at the starting point" });
    }
    Ok((x0.to_vec(), fx))
}

fn check_hessian(h: &Matrix, n: usize) -> Result<()> {
    if h.shape() != (n, n) {
        return Err(invalid(format!("Hessian is {}×{}, expected {n}×{n}", h.rows(), h.cols())));
    }
    if h.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: "Hessian" });
    }
    Ok(())
}

/// Damped Newton method. The Newton system uses `H + τI` with τ ≥ 0 from the
/// modified Cholesky factorisation, so every step is a descent direction even
/// where the Hessian is indefinite or singular; the step is accepted by
/// Armijo backtracking. Hessian source, in order of preference: `hess`,
/// differences of `grad`, differences of `f`.
pub fn newton(
    f: Objective,
    grad: Option<GradientFn>,
    hess: Option<HessianFn>,
    x0: &[f64],
    opts: &Options,
) -> Result<Minimum> {
    let (mut x, mut fx) = setup(&mut *f, None, None, x0, "newton")?;
    let n = x.len();
    let mut m = Model { f, grad, hess, f_evals: 1, g_evals: 0 };
    let mut g = m.gradient(&x);
    let mut iters = 0;
    let mut termination = Termination::MaxIterations;
    while iters < opts.max_iter {
        if g.len() != n || g.iter().any(|v| !v.is_finite()) {
            return Err(MathError::NonFinite { op: "gradient" });
        }
        if inf_norm(&g) <= opts.gtol * (1.0 + fx.abs()) {
            termination = Termination::Gradient;
            break;
        }
        let h = m.hessian(&x);
        check_hessian(&h, n)?;
        let (chol, _tau) = modified_cholesky(&h)?;
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let mut d = chol.solve(&neg_g)?;
        let mut slope = dot(&g, &d);
        if slope.is_nan() || slope >= 0.0 {
            d = neg_g;
            slope = -dot(&g, &g);
        }
        let xr = x.clone();
        let search = |m: &mut Model, d: &[f64], slope: f64| {
            backtracking(
                |a| {
                    let xt: Vec<f64> = xr.iter().zip(d).map(|(x, d)| x + a * d).collect();
                    m.value(&xt)
                },
                fx,
                slope,
                1.0,
                1e-4,
                0.5,
                60,
            )
        };
        let mut ls = search(&mut m, &d, slope);
        if ls.is_none() {
            d = g.iter().map(|v| -v).collect();
            slope = -dot(&g, &g);
            let scale = 1.0 / dot(&g, &g).sqrt().max(1.0);
            let d2: Vec<f64> = d.iter().map(|v| v * scale).collect();
            slope *= scale;
            d = d2;
            ls = search(&mut m, &d, slope);
        }
        let Some(ls) = ls else {
            termination = Termination::LineSearchFailed;
            break;
        };
        iters += 1;
        let step: Vec<f64> = d.iter().map(|v| ls.alpha * v).collect();
        let f_old = fx;
        for (xi, si) in x.iter_mut().zip(&step) {
            *xi += si;
        }
        fx = ls.value;
        g = m.gradient(&x);
        if inf_norm(&step) <= opts.xtol * (1.0 + inf_norm(&x)) {
            termination = Termination::Step;
            break;
        }
        if f_old - fx <= opts.ftol * f_old.abs().max(1.0) && inf_norm(&g) <= 1e3 * opts.gtol * (1.0 + fx.abs()) {
            termination = Termination::Function;
            break;
        }
    }
    let converged = matches!(termination, Termination::Gradient | Termination::Function | Termination::Step)
        && inf_norm(&g) <= 1e3 * opts.gtol.max(1e-12) * (1.0 + fx.abs());
    Ok(Minimum { x, f: fx, grad: g, iterations: iters, f_evals: m.f_evals, g_evals: m.g_evals, converged, termination })
}

/// Trust-region Newton with the dogleg step (Nocedal & Wright, Algorithm
/// 4.1). The model Hessian is `B = H + τI` from the modified Cholesky
/// factorisation, so the dogleg path is well defined for indefinite `H`.
pub fn newton_dogleg(
    f: Objective,
    grad: Option<GradientFn>,
    hess: Option<HessianFn>,
    x0: &[f64],
    opts: &Options,
) -> Result<Minimum> {
    let (mut x, mut fx) = setup(&mut *f, None, None, x0, "newton_dogleg")?;
    let n = x.len();
    let mut m = Model { f, grad, hess, f_evals: 1, g_evals: 0 };
    let mut g = m.gradient(&x);
    let mut radius = 1.0f64;
    let max_radius = 1e8;
    let mut iters = 0;
    let mut termination = Termination::MaxIterations;
    while iters < opts.max_iter {
        if g.len() != n || g.iter().any(|v| !v.is_finite()) {
            return Err(MathError::NonFinite { op: "gradient" });
        }
        if inf_norm(&g) <= opts.gtol * (1.0 + fx.abs()) {
            termination = Termination::Gradient;
            break;
        }
        let h = m.hessian(&x);
        check_hessian(&h, n)?;
        let (chol, tau) = modified_cholesky(&h)?;
        let mut b = h.clone();
        for i in 0..n {
            b[(i, i)] += tau;
        }
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let p_b = chol.solve(&neg_g)?;
        let gbg = dot(&g, &b.matvec(&g)?);
        let gg = dot(&g, &g);
        let p_u: Vec<f64> = g.iter().map(|v| -gg / gbg * v).collect();
        let norm = |v: &[f64]| dot(v, v).sqrt();
        // Inner loop: shrink the region until the step is accepted.
        let mut accepted = false;
        while !accepted {
            let p: Vec<f64> = if norm(&p_b) <= radius {
                p_b.clone()
            } else if norm(&p_u) >= radius {
                g.iter().map(|v| -radius / gg.sqrt() * v).collect()
            } else {
                // ‖p_u + t(p_b − p_u)‖ = Δ, t ∈ (0, 1).
                let diff: Vec<f64> = p_b.iter().zip(&p_u).map(|(a, c)| a - c).collect();
                let (qa, qb, qc) = (dot(&diff, &diff), 2.0 * dot(&p_u, &diff), dot(&p_u, &p_u) - radius * radius);
                let disc = (qb * qb - 4.0 * qa * qc).max(0.0).sqrt();
                let t = (-qb + disc) / (2.0 * qa);
                p_u.iter().zip(&diff).map(|(u, d)| u + t * d).collect()
            };
            let pn = norm(&p);
            if pn <= opts.xtol * (1.0 + inf_norm(&x)) {
                termination = Termination::Step;
                break;
            }
            let bp = b.matvec(&p)?;
            let pred = -(dot(&g, &p) + 0.5 * dot(&p, &bp));
            let xt: Vec<f64> = x.iter().zip(&p).map(|(a, c)| a + c).collect();
            let ft = m.value(&xt);
            let actual = fx - ft;
            let rho = if pred > 0.0 && ft.is_finite() { actual / pred } else { -1.0 };
            if rho < 0.25 {
                radius = 0.25 * pn;
            } else if rho > 0.75 && (pn - radius).abs() <= 1e-10 * radius {
                radius = (2.0 * radius).min(max_radius);
            }
            if rho > 1e-4 {
                accepted = true;
                iters += 1;
                let f_old = fx;
                x = xt;
                fx = ft;
                g = m.gradient(&x);
                if f_old - fx <= opts.ftol * f_old.abs().max(1.0) && inf_norm(&g) <= 1e3 * opts.gtol * (1.0 + fx.abs())
                {
                    termination = Termination::Function;
                }
            }
            if radius < 1e-300 {
                termination = Termination::Step;
                break;
            }
        }
        if matches!(termination, Termination::Step | Termination::Function) {
            break;
        }
    }
    let converged = matches!(termination, Termination::Gradient | Termination::Function | Termination::Step)
        && inf_norm(&g) <= 1e3 * opts.gtol.max(1e-12) * (1.0 + fx.abs());
    Ok(Minimum { x, f: fx, grad: g, iterations: iters, f_evals: m.f_evals, g_evals: m.g_evals, converged, termination })
}
