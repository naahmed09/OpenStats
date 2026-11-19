//! One-dimensional minimisation.

use super::invalid;
use crate::error::Result;

/// Result of [`brent_minimize`].
#[derive(Debug, Clone, Copy)]
pub struct Min1d {
    pub x: f64,
    pub f: f64,
    pub iterations: usize,
    pub f_evals: usize,
    pub converged: bool,
}

/// A bracketing triple a < b < c (or c < b < a) with f(b) ≤ f(a), f(b) ≤ f(c).
#[derive(Debug, Clone, Copy)]
pub struct Bracket {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub fa: f64,
    pub fb: f64,
    pub fc: f64,
}

const GOLD: f64 = 1.618_033_988_749_895;

/// Find a bracketing triple by downhill golden-section expansion from two
/// starting abscissae (Numerical-Recipes-style `mnbrak` with parabolic
/// extrapolation, written from the algorithm's description).
pub fn bracket_minimum(mut f: impl FnMut(f64) -> f64, x0: f64, x1: f64) -> Result<Bracket> {
    if !x0.is_finite() || !x1.is_finite() || x0 == x1 {
        return Err(invalid("bracket_minimum: need two distinct finite starting points"));
    }
    let (mut a, mut b) = (x0, x1);
    let (mut fa, mut fb) = (f(a), f(b));
    if fa.is_nan() || fb.is_nan() {
        return Err(invalid("bracket_minimum: objective is NaN at a starting point"));
    }
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = b + GOLD * (b - a);
    let mut fc = f(c);
    for _ in 0..200 {
        if fb <= fc {
            return Ok(Bracket { a, b, c, fa, fb, fc });
        }
        // Parabolic extrapolation through a, b, c.
        let r = (b - a) * (fb - fc);
        let q = (b - c) * (fb - fa);
        let denom = 2.0 * (q - r).abs().max(1e-300) * if q - r >= 0.0 { 1.0 } else { -1.0 };
        let mut u = b - ((b - c) * q - (b - a) * r) / denom;
        let ulim = b + 100.0 * (c - b);
        let fu;
        if (b - u) * (u - c) > 0.0 {
            // u between b and c
            let v = f(u);
            if v < fc {
                return Ok(Bracket { a: b, b: u, c, fa: fb, fb: v, fc });
            } else if v > fb {
                return Ok(Bracket { a, b, c: u, fa, fb, fc: v });
            }
            u = c + GOLD * (c - b);
            fu = f(u);
        } else if (c - u) * (u - ulim) > 0.0 {
            let v = f(u);
            if v < fc {
                let nu = u + GOLD * (u - c);
                let (nb, nfb) = (c, fc);
                let (nc, nfc) = (u, v);
                // shift (b, c, u) → (c, u, nu)
                b = nb;
                fb = nfb;
                c = nc;
                fc = nfc;
                u = nu;
                fu = f(u);
            } else {
                fu = v;
            }
        } else if (u - ulim) * (ulim - c) >= 0.0 {
            u = ulim;
            fu = f(u);
        } else {
            u = c + GOLD * (c - b);
            fu = f(u);
        }
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = u;
        fc = fu;
        if !fc.is_finite() && fc.is_nan() {
            return Err(invalid("bracket_minimum: objective became NaN"));
        }
    }
    Err(crate::MathError::NoConvergence { algorithm: "bracket_minimum", iterations: 200 })
}

/// Brent's minimisation of `f` on `[a, b]` (golden-section search accelerated
/// by parabolic interpolation). `xtol` is the absolute tolerance on x, added
/// to the relative `√ε·|x|`; note a minimum can be located no better than
/// about `√ε·|x|` because f is flat there.
pub fn brent_minimize(mut f: impl FnMut(f64) -> f64, a: f64, b: f64, xtol: f64, max_iter: usize) -> Result<Min1d> {
    if a.is_nan() || b.is_nan() || a >= b || !a.is_finite() || !b.is_finite() {
        return Err(invalid("brent_minimize: need finite a < b"));
    }
    if xtol.is_nan() || xtol < 0.0 {
        return Err(invalid("brent_minimize: xtol must be non-negative"));
    }
    const CGOLD: f64 = 0.381_966_011_250_105_1;
    let sqrt_eps = f64::EPSILON.sqrt();
    let (mut lo, mut hi) = (a, b);
    let mut x = lo + CGOLD * (hi - lo);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    if fx.is_nan() {
        return Err(invalid("brent_minimize: objective is NaN"));
    }
    let (mut fw, mut fv) = (fx, fx);
    let (mut d, mut e) = (0.0f64, 0.0f64);
    let mut evals = 1;
    for iter in 0..max_iter {
        let xm = 0.5 * (lo + hi);
        let tol1 = sqrt_eps * x.abs() + xtol / 3.0;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (hi - lo) {
            return Ok(Min1d { x, f: fx, iterations: iter, f_evals: evals, converged: true });
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // Trial parabolic fit through x, w, v.
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (lo - x) && p < q * (hi - x) {
                d = p / q;
                let u = x + d;
                if u - lo < tol2 || hi - u < tol2 {
                    d = if xm >= x { tol1 } else { -tol1 };
                }
                golden = false;
            }
        }
        if golden {
            e = if x >= xm { lo - x } else { hi - x };
            d = CGOLD * e;
        }
        let u = if d.abs() >= tol1 { x + d } else { x + if d >= 0.0 { tol1 } else { -tol1 } };
        let fu = f(u);
        evals += 1;
        if fu.is_nan() {
            return Err(invalid("brent_minimize: objective is NaN"));
        }
        if fu <= fx {
            if u >= x {
                lo = x;
            } else {
                hi = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Ok(Min1d { x, f: fx, iterations: max_iter, f_evals: evals, converged: false })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_minima() {
        let r = brent_minimize(|x| (x - 2.0) * (x - 2.0) + 1.0, -10.0, 10.0, 1e-12, 200).unwrap();
        assert!(r.converged && (r.x - 2.0).abs() < 1e-7 && (r.f - 1.0).abs() < 1e-14);
        // cos has its minimum −1 at π in [2, 5]; accuracy is ~√ε because f is flat.
        let r = brent_minimize(f64::cos, 2.0, 5.0, 0.0, 200).unwrap();
        assert!((r.x - std::f64::consts::PI).abs() < 1e-7 && (r.f + 1.0).abs() < 1e-15);
        // x·ln x has its minimum −1/e at 1/e.
        let r = brent_minimize(|x| x * x.ln(), 1e-3, 3.0, 1e-12, 200).unwrap();
        assert!((r.x - (-1.0f64).exp()).abs() < 1e-7);
        // Asymmetric, non-quadratic: minimum of e^x − 3x at ln 3.
        let r = brent_minimize(|x| x.exp() - 3.0 * x, -5.0, 5.0, 1e-12, 200).unwrap();
        assert!((r.x - 3.0f64.ln()).abs() < 1e-7);
    }

    #[test]
    fn boundary_and_flat_cases() {
        // Monotone function: converges to the left end point.
        let r = brent_minimize(|x| x, 1.0, 4.0, 1e-10, 300).unwrap();
        assert!(r.x < 1.0 + 1e-6);
        // Quartic: very flat minimum, so only ~ε^{1/4} accuracy in x.
        let r = brent_minimize(|x| (x - 1.0).powi(4), -3.0, 5.0, 1e-12, 300).unwrap();
        assert!((r.x - 1.0).abs() < 2e-3 && r.f < 1e-10);
        assert!(brent_minimize(|x| x, 2.0, 1.0, 1e-8, 10).is_err());
        assert!(brent_minimize(|_| f64::NAN, 0.0, 1.0, 1e-8, 10).is_err());
    }

    #[test]
    fn bracketing_then_minimising() {
        let f = |x: f64| (x - 7.0).powi(2) * (1.0 + 0.1 * x.sin()) + 2.0;
        let b = bracket_minimum(f, 0.0, 1.0).unwrap();
        assert!(b.fb <= b.fa && b.fb <= b.fc);
        let (lo, hi) = if b.a < b.c { (b.a, b.c) } else { (b.c, b.a) };
        assert!(lo < b.b && b.b < hi);
        let r = brent_minimize(f, lo, hi, 1e-12, 200).unwrap();
        assert!((r.x - 7.0).abs() < 1e-6 && (r.f - 2.0).abs() < 1e-11);
        // Downhill in the negative direction.
        let b = bracket_minimum(|x| (x + 100.0).powi(2), 0.0, 1.0).unwrap();
        assert!(b.fb <= b.fa && b.fb <= b.fc);
        // Unbounded below: no bracket.
        assert!(bracket_minimum(|x| -x, 0.0, 1.0).is_err());
    }
}
