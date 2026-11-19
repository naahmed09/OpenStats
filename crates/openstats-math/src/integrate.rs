//! Numerical quadrature: Gauss–Legendre rules (nodes computed by Newton's
//! method on the Legendre recurrence) and adaptive bisection on top of them.

use crate::error::{MathError, Result};

/// An n-point Gauss–Legendre rule on [−1, 1].
#[derive(Debug, Clone)]
pub struct GaussLegendre {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl GaussLegendre {
    /// Build the n-point rule (exact for polynomials of degree 2n − 1).
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "need at least one node");
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let m = n.div_ceil(2);
        for i in 0..m {
            // Tricomi's initial guess, then Newton.
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..100 {
                let (p, d) = legendre(n, x);
                dp = d;
                let dx = p / d;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            let (_, d) = legendre(n, x);
            if d != 0.0 {
                dp = d;
            }
            let w = 2.0 / ((1.0 - x * x) * dp * dp);
            nodes[i] = -x;
            nodes[n - 1 - i] = x;
            weights[i] = w;
            weights[n - 1 - i] = w;
        }
        if n % 2 == 1 {
            nodes[n / 2] = 0.0;
        }
        GaussLegendre { nodes, weights }
    }

    /// ∫ₐᵇ f.
    pub fn integrate(&self, a: f64, b: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
        let (h, c) = (0.5 * (b - a), 0.5 * (b + a));
        let mut s = 0.0;
        for (x, w) in self.nodes.iter().zip(&self.weights) {
            s += w * f(c + h * x);
        }
        s * h
    }
}

/// P_n(x) and P_n′(x) by the three-term recurrence.
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    let d = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, d)
}

/// Adaptive Gauss–Legendre: accept a panel when the 20-point estimate and
/// the sum of its two halves agree to `max(abs_tol, rel_tol·|I|)`.
pub fn adaptive(f: &mut dyn FnMut(f64) -> f64, a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> Result<f64> {
    let rule = GaussLegendre::new(20);
    let whole = rule.integrate(a, b, &mut *f);
    let mut stack = vec![(a, b, whole, 0usize)];
    let mut total = 0.0;
    let mut evals = 0usize;
    while let Some((lo, hi, est, depth)) = stack.pop() {
        let mid = 0.5 * (lo + hi);
        let left = rule.integrate(lo, mid, &mut *f);
        let right = rule.integrate(mid, hi, &mut *f);
        evals += 40;
        let both = left + right;
        let tol = abs_tol.max(rel_tol * both.abs());
        if (both - est).abs() <= tol || depth >= 40 || (hi - lo).abs() < 1e-14 * (1.0 + lo.abs()) {
            total += both;
        } else {
            stack.push((lo, mid, left, depth + 1));
            stack.push((mid, hi, right, depth + 1));
        }
        if evals > 2_000_000 {
            return Err(MathError::NoConvergence { algorithm: "adaptive quadrature", iterations: evals });
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_are_exact_for_polynomials() {
        for n in [1, 2, 5, 20, 64] {
            let r = GaussLegendre::new(n);
            assert!((r.weights.iter().sum::<f64>() - 2.0).abs() < 1e-14);
            let deg = 2 * n - 1;
            let v = r.integrate(0.0, 1.0, |x| x.powi(deg as i32));
            assert!((v - 1.0 / (deg as f64 + 1.0)).abs() < 1e-14, "n={n}");
        }
    }

    #[test]
    fn adaptive_handles_peaks() {
        let v = adaptive(&mut |x: f64| (-x * x).exp(), -10.0, 10.0, 1e-15, 1e-14).unwrap();
        assert!((v - std::f64::consts::PI.sqrt()).abs() < 1e-14);
        let v = adaptive(&mut |x: f64| x.sqrt(), 0.0, 1.0, 1e-15, 1e-14).unwrap();
        assert!((v - 2.0 / 3.0).abs() < 1e-13);
        let v = adaptive(&mut |x: f64| 1.0 / (1e-4 + x * x), -1.0, 1.0, 1e-13, 1e-13).unwrap();
        assert!((v - 2.0 * 100.0 * (100.0f64).atan()).abs() < 1e-9);
    }
}
