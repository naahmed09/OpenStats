//! Regularised incomplete beta function Iₓ(a, b) and its inverse.
//!
//! * The prefix xᵃyᵇ/B(a,b) (y = 1 − x supplied separately so the caller can
//!   pass it without cancellation) is computed for a, b ≥ 10 as
//!   √(ab/(2π(a+b)))·exp(a·log1pmx((x−p)/p) + b·log1pmx((y−q)/q) + Δ) with
//!   p = a/(a+b), q = b/(a+b) and Δ the Stirling corrections — no large
//!   logarithms cancel.
//! * The continued fraction (Didonato & Morris form, evaluated with modified
//!   Lentz) is applied on the side x < (a+1)/(a+b+2) where it converges
//!   fastest; the other tail follows by symmetry, and both Iₓ and 1 − Iₓ are
//!   returned with full relative accuracy.
//! * The inverse uses the Numerical Recipes initial guess and Halley steps
//!   with a bisection safeguard, solving in the tail that keeps precision.

use super::gamma::{lgamma_correction, ln_beta, log1pmx};
use super::FPMIN;
use std::f64::consts::PI;

const EPS: f64 = 1e-16;
const MAX_ITER: usize = 200_000;

/// xᵃ yᵇ / B(a, b) with y = 1 − x.
pub fn beta_prefix(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 || y <= 0.0 {
        return 0.0;
    }
    if a >= 10.0 && b >= 10.0 {
        let s = a + b;
        let (p, q) = (a / s, b / s);
        let (dx, dy) = if x < p {
            let d = x - p;
            (d, -d)
        } else {
            let d = y - q;
            (-d, d)
        };
        let corr = lgamma_correction(s) - lgamma_correction(a) - lgamma_correction(b);
        // a·(ln(x/p) − (x/p − 1)) + b·(ln(y/q) − (y/q − 1)): series near the
        // mode, direct logarithms far from it.
        let term = |c: f64, v: f64, m: f64, d: f64| -> f64 {
            let t = d / m;
            if t.abs() < 0.5 {
                c * log1pmx(t)
            } else {
                c * ((v / m).ln() - t)
            }
        };
        let e = term(a, x, p, dx) + term(b, y, q, dy) + corr;
        (a * b / (2.0 * PI * s)).sqrt() * e.exp()
    } else {
        // ln x and ln y from whichever of (x, y) is the exact input: when
        // y = 1 − x is tiny, ln x = ln(1 − y) must come from ln_1p(−y), since
        // x itself is rounded and a·ln x would otherwise lose a·ε.
        let ln_x = if y < 0.5 { (-y).ln_1p() } else { x.ln() };
        let ln_y = if x < 0.5 { (-x).ln_1p() } else { y.ln() };
        (a * ln_x + b * ln_y - ln_beta(a, b)).exp()
    }
}

/// Continued fraction for Iₓ(a,b)·a / prefix.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// (Iₓ(a, b), 1 − Iₓ(a, b)) given x and y = 1 − x.
pub fn beta_inc_pair(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    if a.is_nan() || b.is_nan() || x.is_nan() || y.is_nan() || a <= 0.0 || b <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    if y <= 0.0 {
        return (1.0, 0.0);
    }
    if x < (a + 1.0) / (a + b + 2.0) {
        let i = (beta_prefix(a, b, x, y) * betacf(a, b, x) / a).min(1.0);
        (i, 1.0 - i)
    } else {
        let j = (beta_prefix(a, b, x, y) * betacf(b, a, y) / b).min(1.0);
        (1.0 - j, j)
    }
}

/// Iₓ(a, b).
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    beta_inc_pair(a, b, x, 1.0 - x).0
}

/// 1 − Iₓ(a, b).
pub fn beta_inc_c(a: f64, b: f64, x: f64) -> f64 {
    beta_inc_pair(a, b, x, 1.0 - x).1
}

/// Solve Iₓ(a, b) = target (target ≤ ½ recommended) starting from x0.
fn solve(a: f64, b: f64, target: f64, x0: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut x = x0.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON / 2.0);
    for _ in 0..300 {
        let y = 1.0 - x;
        let (i, _) = beta_inc_pair(a, b, x, y);
        let f = i - target;
        if f == 0.0 {
            return (x, y);
        }
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let deriv = beta_prefix(a, b, x, y) / (x * y);
        let mut xn = if deriv > 0.0 && deriv.is_finite() {
            let u = f / deriv;
            let k = (a - 1.0) / x - (b - 1.0) / y;
            let corr = 1.0 - 0.5 * (u * k).clamp(-1.0, 1.0);
            x - u / corr
        } else {
            f64::NAN
        };
        if xn.is_finite() && (xn - x).abs() <= 2e-16 * x.abs() {
            return (xn, 1.0 - xn);
        }
        if !(xn > lo && xn < hi) || !xn.is_finite() {
            xn = if lo == 0.0 && hi < 1e-3 {
                (hi * hi.max(1e-300)).sqrt().max(0.5 * hi * 1e-3)
            } else {
                0.5 * (lo + hi)
            };
            if xn <= lo || xn >= hi {
                xn = 0.5 * (lo + hi);
            }
        }
        if (xn - x).abs() <= 2e-16 * xn.abs() || hi - lo <= 2e-16 * hi {
            return (xn, 1.0 - xn);
        }
        x = xn;
    }
    (x, 1.0 - x)
}

/// Numerical Recipes initial approximation for Iₓ(a,b) = p.
fn initial_guess(a: f64, b: f64, p: f64) -> f64 {
    if a >= 1.0 && b >= 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut x = (2.307_53 + t * 0.270_61) / (1.0 + t * (0.992_29 + t * 0.044_81)) - t;
        if p < 0.5 {
            x = -x;
        }
        let al = (x * x - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = x * (al + h).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * (2.0 * w).exp())
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        if p < t / w {
            (a * w * p).powf(1.0 / a)
        } else {
            1.0 - (b * w * (1.0 - p)).powf(1.0 / b)
        }
    }
}

/// Inverse: returns (x, 1 − x) with Iₓ(a, b) = p and 1 − Iₓ(a, b) = q,
/// where q = 1 − p is supplied separately (so tiny upper tails keep precision).
pub fn beta_inc_inv_pair(a: f64, b: f64, p: f64, q: f64) -> (f64, f64) {
    if a.is_nan() || b.is_nan() || p.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return (f64::NAN, f64::NAN);
    }
    if p == 0.0 && q > 0.0 {
        return (0.0, 1.0);
    }
    if q == 0.0 {
        return (1.0, 0.0);
    }
    if p <= q {
        let x0 = initial_guess(a, b, p);
        let (x, y) = solve(a, b, p, x0);
        if x > 0.5 {
            // Refine the small complement directly: I_y(b, a) = q.
            let (y2, x2) = solve(b, a, q, y);
            return (x2, y2);
        }
        (x, y)
    } else {
        let (y, x) = beta_inc_inv_pair(b, a, q, p);
        (x, y)
    }
}

/// x with Iₓ(a, b) = p.
pub fn beta_inc_inv(a: f64, b: f64, p: f64) -> f64 {
    beta_inc_inv_pair(a, b, p, 1.0 - p).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        if a == b {
            0.0
        } else {
            (a - b).abs() / b.abs()
        }
    }

    #[test]
    fn closed_forms() {
        // I_x(1, b) = 1 − (1−x)^b ; I_x(a, 1) = x^a ; I_x(1/2,1/2) = (2/π) asin √x.
        for &x in &[1e-10, 0.01, 0.3, 0.5, 0.9, 0.999_999] {
            assert!(rel(beta_inc(1.0, 3.5, x), -(3.5 * (-x).ln_1p()).exp_m1()) < 1e-14, "I(1,b) x={x}");
            assert!(rel(beta_inc(2.5, 1.0, x), x.powf(2.5)) < 1e-14, "I(a,1) x={x}");
            // Compare whichever tail is better conditioned for the reference formula.
            let (i, j) = beta_inc_pair(0.5, 0.5, x, 1.0 - x);
            if x < 0.5 {
                assert!(rel(i, 2.0 / PI * x.sqrt().asin()) < 1e-14, "arcsine x={x}");
            } else {
                assert!(rel(j, 2.0 / PI * (1.0 - x).sqrt().asin()) < 1e-14, "arcsine tail x={x}");
            }
        }
        // Binomial identity: I_p(k, n−k+1) = P(X ≥ k), X ~ Bin(n, p).
        let (n, k, p) = (20u32, 7u32, 0.3f64);
        let tail: f64 = (k..=n)
            .map(|j| {
                let c = super::super::gamma::factorial(n)
                    / (super::super::gamma::factorial(j) * super::super::gamma::factorial(n - j));
                c * p.powi(j as i32) * (1.0 - p).powi((n - j) as i32)
            })
            .sum();
        assert!(rel(beta_inc(k as f64, (n - k + 1) as f64, p), tail) < 1e-13);
    }

    #[test]
    fn symmetry_and_large_parameters() {
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (50.0, 70.0, 0.45), (1e4, 2e4, 0.334), (0.1, 30.0, 0.01)] {
            let (i, j) = beta_inc_pair(a, b, x, 1.0 - x);
            let (i2, j2) = beta_inc_pair(b, a, 1.0 - x, x);
            assert!(rel(i, j2) < 1e-13 && rel(j, i2) < 1e-13, "a={a} b={b}");
            assert!((i + j - 1.0).abs() < 1e-15);
        }
        // a = b ⇒ I_{1/2} = 1/2. Continued-fraction rounding grows with the
        // parameters (documented limit: ≈1e-13 at a = b = 1e6).
        for (a, tol) in [(1e2, 2e-15), (1e4, 2e-14), (1e6, 1e-12)] {
            assert!((beta_inc(a, a, 0.5) - 0.5).abs() < tol, "a={a}");
        }
    }

    #[test]
    fn inverse_roundtrips() {
        for &(a, b) in &[(0.5, 0.5), (1.0, 1.0), (2.0, 5.0), (0.05, 3.0), (30.0, 0.7), (200.0, 300.0), (1e4, 1e4)] {
            for k in 1..50 {
                let p = 10f64.powf(-(k as f64) / 3.0);
                let (x, y) = beta_inc_inv_pair(a, b, p, 1.0 - p);
                let (i, _) = beta_inc_pair(a, b, x, y);
                // Solutions in the subnormal range cannot be represented precisely.
                if x > 1e-290 {
                    assert!(rel(i, p) < 1e-11, "a={a} b={b} p={p:e} x={x:e}");
                }
                let (x2, y2) = beta_inc_inv_pair(a, b, 1.0 - p, p);
                let (_, j) = beta_inc_pair(a, b, x2, y2);
                assert!(y2 < 1e-290 || rel(j, p) < 1e-11, "upper a={a} b={b} q={p:e} y={y2:e}");
            }
        }
    }
}
