//! BFGS, L-BFGS and a bound-constrained projected L-BFGS.

use super::line_search::strong_wolfe;
use super::{check_start, dot, gradient_fd, inf_norm, invalid, GradientFn, Minimum, Objective, Options, Termination};
use crate::error::{MathError, Result};
use std::collections::VecDeque;

/// Objective with an optional analytic gradient and evaluation counters.
struct Problem<'a, 'b> {
    f: Objective<'a>,
    grad: Option<GradientFn<'b>>,
    f_evals: usize,
    g_evals: usize,
}

impl Problem<'_, '_> {
    fn value(&mut self, x: &[f64]) -> f64 {
        self.f_evals += 1;
        (self.f)(x)
    }
    fn gradient(&mut self, x: &[f64]) -> Vec<f64> {
        self.g_evals += 1;
        match self.grad.as_mut() {
            Some(g) => g(x),
            None => {
                let g = gradient_fd(self.f, x);
                self.f_evals += 2 * x.len();
                g
            }
        }
    }
}

fn axpy(x: &[f64], a: f64, d: &[f64]) -> Vec<f64> {
    x.iter().zip(d).map(|(xi, di)| xi + a * di).collect()
}

fn check_gradient(g: &[f64], n: usize) -> Result<()> {
    if g.len() != n {
        return Err(invalid(format!("gradient has {} entries, expected {n}", g.len())));
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: "gradient at the starting point" });
    }
    Ok(())
}

/// Shared quasi-Newton driver; `Hessian` is the strategy for applying the
/// inverse-Hessian approximation.
trait InverseHessian {
    fn apply(&self, g: &[f64]) -> Vec<f64>;
    fn update(&mut self, s: &[f64], y: &[f64], first: bool);
    fn reset(&mut self);
}

/// Dense BFGS inverse Hessian.
struct Dense {
    n: usize,
    h: Vec<f64>,
}

impl Dense {
    fn new(n: usize) -> Self {
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        Dense { n, h }
    }
}

impl InverseHessian for Dense {
    fn apply(&self, g: &[f64]) -> Vec<f64> {
        (0..self.n).map(|i| dot(&self.h[i * self.n..(i + 1) * self.n], g)).collect()
    }
    fn update(&mut self, s: &[f64], y: &[f64], first: bool) {
        let n = self.n;
        let sy = dot(s, y);
        if first {
            // Scale the initial matrix to the curvature along the first step.
            let yy = dot(y, y);
            let scale = sy / yy;
            for i in 0..n {
                self.h[i * n + i] = scale;
            }
        }
        let rho = 1.0 / sy;
        let hy = self.apply(y);
        let yhy = dot(y, &hy);
        for i in 0..n {
            for j in 0..n {
                self.h[i * n + j] += -rho * (s[i] * hy[j] + hy[i] * s[j]) + (rho * rho * yhy + rho) * s[i] * s[j];
            }
        }
    }
    fn reset(&mut self) {
        let n = self.n;
        self.h.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            self.h[i * n + i] = 1.0;
        }
    }
}

/// Limited-memory inverse Hessian (two-loop recursion).
struct Limited {
    m: usize,
    pairs: VecDeque<(Vec<f64>, Vec<f64>, f64)>,
    gamma: f64,
}

impl InverseHessian for Limited {
    fn apply(&self, g: &[f64]) -> Vec<f64> {
        let mut q = g.to_vec();
        let mut alpha = vec![0.0; self.pairs.len()];
        for (k, (s, y, rho)) in self.pairs.iter().enumerate().rev() {
            alpha[k] = rho * dot(s, &q);
            for (qi, yi) in q.iter_mut().zip(y) {
                *qi -= alpha[k] * yi;
            }
        }
        for qi in &mut q {
            *qi *= self.gamma;
        }
        for (k, (s, y, rho)) in self.pairs.iter().enumerate() {
            let beta = rho * dot(y, &q);
            for (qi, si) in q.iter_mut().zip(s) {
                *qi += (alpha[k] - beta) * si;
            }
        }
        q
    }
    fn update(&mut self, s: &[f64], y: &[f64], _first: bool) {
        let sy = dot(s, y);
        self.gamma = sy / dot(y, y);
        if self.pairs.len() == self.m {
            self.pairs.pop_front();
        }
        self.pairs.push_back((s.to_vec(), y.to_vec(), 1.0 / sy));
    }
    fn reset(&mut self) {
        self.pairs.clear();
        self.gamma = 1.0;
    }
}

fn quasi_newton(problem: &mut Problem, x0: &[f64], opts: &Options, h: &mut dyn InverseHessian) -> Result<Minimum> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = problem.value(&x);
    if !fx.is_finite() {
        return Err(MathError::NonFinite { op: "objective at the starting point" });
    }
    let mut g = problem.gradient(&x);
    check_gradient(&g, n)?;
    let mut first = true;
    let mut iters = 0;
    let mut termination = Termination::MaxIterations;
    while iters < opts.max_iter {
        if inf_norm(&g) <= opts.gtol * (1.0 + fx.abs()) {
            termination = Termination::Gradient;
            break;
        }
        let mut d: Vec<f64> = h.apply(&g).into_iter().map(|v| -v).collect();
        let mut slope = dot(&g, &d);
        if slope.is_nan() || slope >= 0.0 || d.iter().any(|v| !v.is_finite()) {
            h.reset();
            first = true;
            d = g.iter().map(|v| -v).collect();
            slope = -dot(&g, &g);
        }
        let g2 = dot(&g, &g).sqrt();
        let a0 = if first { (1.0 / g2).min(1.0) } else { 1.0 };
        let mut cache: Vec<(f64, Vec<f64>)> = Vec::new();
        let attempt = |p: &mut Problem, d: &[f64], slope: f64, cache: &mut Vec<(f64, Vec<f64>)>| {
            let (xr, fr) = (&x, fx);
            strong_wolfe(
                |a| {
                    let xt = axpy(xr, a, d);
                    let v = p.value(&xt);
                    if !v.is_finite() {
                        return (v, 0.0);
                    }
                    let gt = p.gradient(&xt);
                    let s = dot(&gt, d);
                    cache.push((a, gt));
                    (v, s)
                },
                fr,
                slope,
                a0,
                1e-4,
                0.9,
                60,
            )
        };
        let mut found = attempt(problem, &d, slope, &mut cache);
        if found.is_none() && !first {
            // Poor curvature model: retry once along steepest descent.
            h.reset();
            first = true;
            d = g.iter().map(|v| -v).collect();
            slope = -dot(&g, &g);
            cache.clear();
            found = attempt(problem, &d, slope, &mut cache);
        }
        let Some(ls) = found else {
            termination = Termination::LineSearchFailed;
            break;
        };
        iters += 1;
        let g_new = cache
            .iter()
            .rev()
            .find(|(a, _)| *a == ls.alpha)
            .map(|(_, gg)| gg.clone())
            .expect("accepted step was evaluated");
        let x_new = axpy(&x, ls.alpha, &d);
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let f_old = fx;
        x = x_new;
        fx = ls.value;
        g = g_new;
        let sy = dot(&s, &y);
        if sy > 1e-10 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            h.update(&s, &y, first);
            first = false;
        }
        if inf_norm(&s) <= opts.xtol * (1.0 + inf_norm(&x)) {
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
    Ok(Minimum {
        x,
        f: fx,
        grad: g,
        iterations: iters,
        f_evals: problem.f_evals,
        g_evals: problem.g_evals,
        converged,
        termination,
    })
}

/// BFGS with a dense inverse-Hessian approximation and strong-Wolfe line
/// search. `grad = None` uses central differences.
pub fn bfgs(f: Objective, grad: Option<GradientFn>, x0: &[f64], opts: &Options) -> Result<Minimum> {
    check_start(x0, "bfgs")?;
    let mut p = Problem { f, grad, f_evals: 0, g_evals: 0 };
    let mut h = Dense::new(x0.len());
    quasi_newton(&mut p, x0, opts, &mut h)
}

/// Limited-memory BFGS (history `opts.memory`) with strong-Wolfe line search.
pub fn lbfgs(f: Objective, grad: Option<GradientFn>, x0: &[f64], opts: &Options) -> Result<Minimum> {
    check_start(x0, "lbfgs")?;
    if opts.memory == 0 {
        return Err(invalid("lbfgs: memory must be at least 1"));
    }
    let mut p = Problem { f, grad, f_evals: 0, g_evals: 0 };
    let mut h = Limited { m: opts.memory, pairs: VecDeque::new(), gamma: 1.0 };
    quasi_newton(&mut p, x0, opts, &mut h)
}

/// Armijo backtracking along the projected path x(α) = P(x + αd).
fn projected_armijo(
    p: &mut Problem,
    clamp: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    fx: f64,
    g: &[f64],
    d: &[f64],
    a0: f64,
) -> Option<super::line_search::LineSearch> {
    let mut alpha = a0;
    for k in 1..=60 {
        let xt = clamp(&axpy(x, alpha, d));
        let v = p.value(&xt);
        let dec: f64 = g.iter().zip(&xt).zip(x).map(|((gi, xi), x0)| gi * (xi - x0)).sum();
        if v.is_finite() && dec < 0.0 && v <= fx + 1e-4 * dec {
            return Some(super::line_search::LineSearch { alpha, value: v, slope: f64::NAN, evals: k });
        }
        alpha *= 0.5;
    }
    None
}

/// Box-constrained minimisation `lower ≤ x ≤ upper` (use ±∞ for no bound) by
/// a projected limited-memory quasi-Newton method: variables held at a bound
/// by the gradient are fixed, the remaining ones move along the L-BFGS
/// direction, and each trial point is projected back onto the box with an
/// Armijo backtracking search on the projected path. Convergence is on the
/// projected gradient ‖P(x − ∇f) − x‖∞. (This is a projected-gradient /
/// active-set method in the Bertsekas family, not the Cauchy-point
/// subspace algorithm of Byrd et al.'s L-BFGS-B.)
pub fn lbfgs_bounded(
    f: Objective,
    grad: Option<GradientFn>,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    opts: &Options,
) -> Result<Minimum> {
    check_start(x0, "lbfgs_bounded")?;
    let n = x0.len();
    if lower.len() != n
        || upper.len() != n
        || (0..n).any(|i| lower[i].is_nan() || upper[i].is_nan() || lower[i] > upper[i])
    {
        return Err(invalid("lbfgs_bounded: need lower ≤ upper with one bound per variable"));
    }
    if opts.memory == 0 {
        return Err(invalid("lbfgs_bounded: memory must be at least 1"));
    }
    let clamp = |x: &[f64]| -> Vec<f64> { x.iter().enumerate().map(|(i, &v)| v.clamp(lower[i], upper[i])).collect() };
    let mut p = Problem { f, grad, f_evals: 0, g_evals: 0 };
    let mut x = clamp(x0);
    let mut fx = p.value(&x);
    if !fx.is_finite() {
        return Err(MathError::NonFinite { op: "objective at the (projected) starting point" });
    }
    let mut g = p.gradient(&x);
    check_gradient(&g, n)?;
    let mut h = Limited { m: opts.memory, pairs: VecDeque::new(), gamma: 1.0 };
    let proj_grad = |x: &[f64], g: &[f64]| -> f64 {
        (0..n).fold(0.0f64, |m, i| m.max(((x[i] - g[i]).clamp(lower[i], upper[i]) - x[i]).abs()))
    };
    let mut iters = 0;
    let mut termination = Termination::MaxIterations;
    while iters < opts.max_iter {
        if proj_grad(&x, &g) <= opts.gtol * (1.0 + fx.abs()) {
            termination = Termination::Gradient;
            break;
        }
        let free: Vec<bool> =
            (0..n).map(|i| !((x[i] <= lower[i] && g[i] > 0.0) || (x[i] >= upper[i] && g[i] < 0.0))).collect();
        let gf: Vec<f64> = (0..n).map(|i| if free[i] { g[i] } else { 0.0 }).collect();
        let mut d: Vec<f64> =
            h.apply(&gf).into_iter().enumerate().map(|(i, v)| if free[i] { -v } else { 0.0 }).collect();
        if dot(&g, &d).is_nan() || dot(&g, &d) >= 0.0 || d.iter().any(|v| !v.is_finite()) {
            h.reset();
            d = gf.iter().map(|v| -v).collect();
        }
        let first = h.pairs.is_empty();
        let a0 = if first { (1.0 / inf_norm(&d).max(1e-300)).min(1.0) } else { 1.0 };
        let mut found = projected_armijo(&mut p, &clamp, &x, fx, &g, &d, a0);
        if found.is_none() && !h.pairs.is_empty() {
            // Poor curvature model: retry along the projected steepest descent.
            h.reset();
            d = gf.iter().map(|v| -v).collect();
            let a1 = (1.0 / inf_norm(&d).max(1e-300)).min(1.0);
            found = projected_armijo(&mut p, &clamp, &x, fx, &g, &d, a1);
        }
        let Some(ls) = found else {
            termination = Termination::LineSearchFailed;
            break;
        };
        iters += 1;
        let x_new = clamp(&axpy(&x, ls.alpha, &d));
        let g_new = p.gradient(&x_new);
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let f_old = fx;
        x = x_new;
        fx = ls.value;
        g = g_new;
        let sy = dot(&s, &y);
        if sy > 1e-10 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            h.update(&s, &y, false);
        }
        if inf_norm(&s) <= opts.xtol * (1.0 + inf_norm(&x)) {
            termination = Termination::Step;
            break;
        }
        if f_old - fx <= opts.ftol * f_old.abs().max(1.0) && proj_grad(&x, &g) <= 1e3 * opts.gtol * (1.0 + fx.abs()) {
            termination = Termination::Function;
            break;
        }
    }
    let converged = matches!(termination, Termination::Gradient | Termination::Function | Termination::Step)
        && proj_grad(&x, &g) <= 1e3 * opts.gtol.max(1e-12) * (1.0 + fx.abs());
    Ok(Minimum { x, f: fx, grad: g, iterations: iters, f_evals: p.f_evals, g_evals: p.g_evals, converged, termination })
}
