//! Nelder–Mead simplex search.

use super::{check_start, Minimum, Objective, Termination};
use crate::error::Result;

/// Nelder–Mead settings.
#[derive(Debug, Clone, Copy)]
pub struct NelderMeadOptions {
    /// Maximum number of simplex iterations per run.
    pub max_iter: usize,
    /// Stop a run when the simplex spans ≤ `xtol` (∞-norm of vertex offsets,
    /// scaled by 1 + ‖best‖∞) …
    pub xtol: f64,
    /// … and the vertex function values differ by ≤ `ftol` (scaled by max(1, |f|)).
    pub ftol: f64,
    /// Use the dimension-adaptive coefficients of Gao & Han (2012), which
    /// keep the method well behaved for n ≳ 5; otherwise the classical
    /// (1, 2, ½, ½).
    pub adaptive: bool,
    /// After a run converges, rebuild the simplex around the best point and
    /// run again, up to this many extra times, while f still improves.
    /// Guards against the method's known premature stagnation.
    pub restarts: usize,
    /// Relative size of the initial simplex edges (5%; 0.00025 absolute for
    /// zero coordinates).
    pub initial_step: f64,
}

impl Default for NelderMeadOptions {
    fn default() -> Self {
        NelderMeadOptions { max_iter: 5000, xtol: 1e-10, ftol: 1e-14, adaptive: true, restarts: 3, initial_step: 0.05 }
    }
}

fn build_simplex(f: &mut dyn FnMut(&[f64]) -> f64, x: &[f64], step: f64, evals: &mut usize) -> Vec<(f64, Vec<f64>)> {
    let n = x.len();
    let mut s = Vec::with_capacity(n + 1);
    *evals += 1;
    s.push((f(x), x.to_vec()));
    for i in 0..n {
        let mut v = x.to_vec();
        v[i] = if v[i] != 0.0 { v[i] * (1.0 + step) } else { 0.000_25 };
        *evals += 1;
        s.push((f(&v), v));
    }
    s
}

/// Minimise `f` from `x0` without derivatives.
pub fn nelder_mead(f: Objective, x0: &[f64], opts: &NelderMeadOptions) -> Result<Minimum> {
    check_start(x0, "nelder_mead")?;
    let n = x0.len();
    let nf = n as f64;
    let (alpha, beta, gamma, delta) = if opts.adaptive {
        (1.0, 1.0 + 2.0 / nf, 0.75 - 1.0 / (2.0 * nf), 1.0 - 1.0 / nf)
    } else {
        (1.0, 2.0, 0.5, 0.5)
    };
    // Rank with NaN treated as +∞ so that undefined points are never best.
    let key = |v: f64| if v.is_nan() { f64::INFINITY } else { v };
    let mut evals = 0usize;
    let mut iters = 0usize;
    let mut best_x = x0.to_vec();
    let mut best_f = f64::INFINITY;
    let mut termination = Termination::MaxIterations;
    let mut start = x0.to_vec();
    let mut step = opts.initial_step;
    for run in 0..=opts.restarts {
        let mut s = build_simplex(f, &start, step, &mut evals);
        let mut converged_run = false;
        for _ in 0..opts.max_iter {
            s.sort_by(|a, b| key(a.0).total_cmp(&key(b.0)));
            iters += 1;
            let fspread = (s[n].0 - s[0].0).abs();
            let xspread = s[1..]
                .iter()
                .map(|(_, v)| v.iter().zip(&s[0].1).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())))
                .fold(0.0, f64::max);
            let xscale = 1.0 + s[0].1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if xspread <= opts.xtol * xscale && fspread <= opts.ftol * s[0].0.abs().max(1.0) {
                converged_run = true;
                break;
            }
            // Centroid of the best n vertices.
            let mut c = vec![0.0; n];
            for (_, v) in &s[..n] {
                for (ci, vi) in c.iter_mut().zip(v) {
                    *ci += vi / nf;
                }
            }
            let worst = s[n].clone();
            let point = |t: f64| -> Vec<f64> { c.iter().zip(&worst.1).map(|(ci, wi)| ci + t * (ci - wi)).collect() };
            let xr = point(alpha);
            let fr = f(&xr);
            evals += 1;
            if key(fr) < key(s[0].0) {
                let xe = point(alpha * beta);
                let fe = f(&xe);
                evals += 1;
                s[n] = if key(fe) < key(fr) { (fe, xe) } else { (fr, xr) };
            } else if key(fr) < key(s[n - 1].0) {
                s[n] = (fr, xr);
            } else {
                let outside = key(fr) < key(worst.0);
                let xc = if outside { point(alpha * gamma) } else { point(-gamma) };
                let fc = f(&xc);
                evals += 1;
                if key(fc) <= key(if outside { fr } else { worst.0 }) {
                    s[n] = (fc, xc);
                } else {
                    // Shrink toward the best vertex.
                    let b = s[0].1.clone();
                    for item in s.iter_mut().skip(1) {
                        for (vi, bi) in item.1.iter_mut().zip(&b) {
                            *vi = bi + delta * (*vi - bi);
                        }
                        item.0 = f(&item.1);
                        evals += 1;
                    }
                }
            }
        }
        s.sort_by(|a, b| key(a.0).total_cmp(&key(b.0)));
        let improved = key(s[0].0) < best_f - opts.ftol * best_f.abs().max(1.0) || !best_f.is_finite();
        if key(s[0].0) <= best_f || !best_f.is_finite() {
            best_f = s[0].0;
            best_x = s[0].1.clone();
        }
        termination = if converged_run { Termination::Step } else { Termination::MaxIterations };
        if !converged_run {
            break;
        }
        if run > 0 && !improved {
            break;
        }
        start = best_x.clone();
        step = (step * 0.5).max(1e-3);
    }
    let converged = termination == Termination::Step;
    Ok(Minimum {
        x: best_x,
        f: best_f,
        grad: Vec::new(),
        iterations: iters,
        f_evals: evals,
        g_evals: 0,
        converged,
        termination,
    })
}
