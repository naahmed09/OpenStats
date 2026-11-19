//! Scalar root finding: Brent's method on a bracket, and bracket expansion.

use crate::error::{MathError, Result};

/// Brent's method (inverse quadratic interpolation with bisection safeguard).
/// Requires f(a)·f(b) ≤ 0. Converges to |b − a| ≤ 2·ε·|x| + `xtol`.
pub fn brent(mut f: impl FnMut(f64) -> f64, a: f64, b: f64, xtol: f64, max_iter: usize) -> Result<f64> {
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.is_nan() || fb.is_nan() || fa.signum() == fb.signum() {
        return Err(MathError::InvalidArgument("brent: interval does not bracket a root".into()));
    }
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    for _ in 0..max_iter {
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtol;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                let qa = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * m * qa * (qa - r) - (b - a) * (r - 1.0));
                q = (qa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b);
    }
    Err(MathError::NoConvergence { algorithm: "Brent", iterations: max_iter })
}

/// Expand [lo, hi] geometrically (keeping `lo ≥ min`) until f changes sign.
pub fn expand_bracket(mut f: impl FnMut(f64) -> f64, lo: f64, hi: f64, min: f64, max: f64) -> Result<(f64, f64)> {
    let (mut lo, mut hi) = (lo, hi);
    let (mut flo, mut fhi) = (f(lo), f(hi));
    for _ in 0..200 {
        if flo.signum() != fhi.signum() || flo == 0.0 || fhi == 0.0 {
            return Ok((lo, hi));
        }
        let w = hi - lo;
        if flo.abs() < fhi.abs() {
            lo = (lo - 2.0 * w).max(min);
            flo = f(lo);
        } else {
            hi = (hi + 2.0 * w).min(max);
            fhi = f(hi);
        }
    }
    Err(MathError::NoConvergence { algorithm: "bracket expansion", iterations: 200 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_roots() {
        let r = brent(|x| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, 0.0, 100).unwrap();
        assert!((r - 2.094_551_481_542_326_5).abs() < 1e-15);
        let r = brent(|x: f64| x.cos() - x, 0.0, 1.0, 0.0, 100).unwrap();
        assert!((r - 0.739_085_133_215_160_6).abs() < 1e-15);
        assert!(brent(|x| x * x + 1.0, -1.0, 1.0, 0.0, 100).is_err());
        let (lo, hi) = expand_bracket(|x| x - 100.0, 0.0, 1.0, 0.0, f64::INFINITY).unwrap();
        assert!(lo <= 100.0 && hi >= 100.0);
    }
}
