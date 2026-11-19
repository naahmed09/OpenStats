//! Regularised incomplete gamma functions
//! P(a, x) = γ(a, x)/Γ(a) and Q(a, x) = Γ(a, x)/Γ(a), and their inverses.
//!
//! * The common factor xᵃe⁻ˣ/Γ(a) is computed for a ≥ 10 as
//!   √(a/2π)·exp(a·log1pmx((x−a)/a) − corr(a)), which avoids the
//!   catastrophic cancellation of a·ln x − x − lnΓ(a) for large a.
//! * P uses the power series when x < a + 1, Q Legendre's continued fraction
//!   otherwise; the smaller of P and Q is always computed directly, so both
//!   tails keep full relative accuracy.
//! * Inverses: Numerical Recipes initial guess, then Halley iterations with a
//!   bisection safeguard, solving on whichever tail is smaller.

use super::gamma::{lgamma_correction, ln_gamma, log1pmx};
use super::FPMIN;
use std::f64::consts::PI;

const EPS: f64 = 1e-16;
const MAX_ITER: usize = 1_000_000;

/// xᵃ e⁻ˣ / Γ(a).
pub fn gamma_prefix(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if a < 10.0 {
        (a * x.ln() - x - ln_gamma(a)).exp()
    } else {
        // a·(ln(x/a) − (x/a − 1)): series in t near 1, direct logarithm when
        // x/a is far from 1 (forming 1 + t there would lose precision).
        let t = (x - a) / a;
        let r = if t.abs() < 0.5 { a * log1pmx(t) } else { a * ((x / a).ln() - t) };
        (a / (2.0 * PI)).sqrt() * (r - lgamma_correction(a)).exp()
    }
}

/// Both P(a, x) and Q(a, x), each with full relative accuracy.
pub fn gamma_pq(a: f64, x: f64) -> (f64, f64) {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if x.is_infinite() {
        return (1.0, 0.0);
    }
    let pre = gamma_prefix(a, x);
    if x < a + 1.0 {
        // Series: P = pre · Σ xⁿ / (a(a+1)…(a+n)).
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        let p = (sum * pre).min(1.0);
        (p, 1.0 - p)
    } else {
        // Legendre continued fraction for Q (modified Lentz).
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAX_ITER {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
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
        let q = (pre * h).min(1.0);
        (1.0 - q, q)
    }
}

/// Lower regularised incomplete gamma P(a, x).
pub fn gamma_p(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).0
}

/// Upper regularised incomplete gamma Q(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).1
}

/// Density of the standard gamma distribution, xᵃ⁻¹e⁻ˣ/Γ(a) (= d P/dx).
fn dpdx(a: f64, x: f64) -> f64 {
    gamma_prefix(a, x) / x
}

/// Solve P(a, x) = p (or Q(a, x) = q when `upper`) for x.
fn invert(a: f64, target: f64, upper: bool) -> f64 {
    let p = if upper { 1.0 - target } else { target };
    // Initial guess (Numerical Recipes, invgammp).
    let mut x = if a > 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let pp = if upper && target < 0.5 { target } else { pp };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.307_53 + t * 0.270_61) / (1.0 + t * (0.992_29 + t * 0.044_81)) - t;
        if p < 0.5 {
            z = -z;
        }
        (a * (1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt())).powi(3)).max(1e-3)
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t {
            (p / t).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t) / (1.0 - t)).ln()
        }
    };
    if !x.is_finite() || x <= 0.0 {
        x = a.max(1e-300);
    }
    // Halley with bracketing. f(x) increasing in x for both formulations.
    let f = |x: f64| -> f64 {
        let (pp, qq) = gamma_pq(a, x);
        if upper {
            target - qq
        } else {
            pp - target
        }
    };
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    for _ in 0..200 {
        let fx = f(x);
        if fx == 0.0 {
            return x;
        }
        if fx < 0.0 {
            lo = lo.max(x);
        } else {
            hi = hi.min(x);
        }
        let d = dpdx(a, x);
        let mut xn = if d > 0.0 && d.is_finite() {
            let u = fx / d;
            let corr = 1.0 - 0.5 * (u * ((a - 1.0) / x - 1.0)).min(1.0);
            x - u / corr
        } else {
            f64::NAN
        };
        if xn.is_finite() && (xn - x).abs() <= 2e-16 * x.abs() {
            return xn;
        }
        if !(xn > lo && xn < hi) || !xn.is_finite() {
            xn = if hi.is_finite() { 0.5 * (lo + hi) } else { 2.0 * x.max(lo) + 1.0 };
            if lo == 0.0 && hi.is_finite() && xn == 0.5 * hi {
                // Geometric bisection near zero (x may be extremely small).
                xn = (hi * hi.max(f64::MIN_POSITIVE) * 1e-3).sqrt().min(0.5 * hi);
            }
        }
        if (xn - x).abs() <= 2e-16 * xn.abs() {
            return xn;
        }
        x = xn;
    }
    x
}

/// x such that P(a, x) = p.
pub fn gamma_p_inv(a: f64, p: f64) -> f64 {
    if a.is_nan() || p.is_nan() || a <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        invert(a, 1.0 - p, true)
    } else {
        invert(a, p, false)
    }
}

/// x such that Q(a, x) = q (accurate for tiny q).
pub fn gamma_q_inv(a: f64, q: f64) -> f64 {
    if a.is_nan() || q.is_nan() || a <= 0.0 || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return 0.0;
    }
    if q > 0.5 {
        invert(a, 1.0 - q, false)
    } else {
        invert(a, q, true)
    }
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
        // P(1, x) = 1 − e⁻ˣ; Q(1/2, x²) = erfc(x).
        for &x in &[1e-8, 0.3, 1.0, 5.0, 40.0] {
            let (p, q) = gamma_pq(1.0, x);
            assert!(rel(p, -(-x).exp_m1()) < 1e-14, "P(1,{x})");
            assert!(rel(q, (-x).exp()) < 1e-14, "Q(1,{x})");
            assert!(rel(gamma_q(0.5, x * x), super::super::erf::erfc(x)) < 1e-13, "Q(1/2,{x}²)");
        }
        // Poisson identity: Q(n, λ) = Σ_{k<n} e^{−λ} λᵏ/k!
        let (n, lam) = (7.0, 3.5f64);
        let s: f64 = (0..7).map(|k| (-lam).exp() * lam.powi(k) / super::super::gamma::factorial(k as u32)).sum();
        assert!(rel(gamma_q(n, lam), s) < 1e-14);
    }

    #[test]
    fn large_a_prefix_is_accurate() {
        // Q(a, a) → 1/2 + 1/(3√(2πa)) + … ; compare with the series value at a = 1e6.
        let a = 1e6;
        let q = gamma_q(a, a);
        // Q(a, a) = 1/2 − 1/(3√(2πa)) + O(a^{-3/2}) (the median of Gamma(a) is ≈ a − 1/3).
        let approx = 0.5 - 1.0 / (3.0 * (2.0 * PI * a).sqrt());
        assert!((q - approx).abs() < 1e-8, "{q} vs {approx}");
        // Exact consistency: P + Q = 1 and both computed directly on their side.
        let (p, q2) = gamma_pq(a, a * 1.01);
        assert!((p + q2 - 1.0).abs() < 1e-15 && q2 < 1e-10);
    }

    #[test]
    fn inverses_roundtrip() {
        for &a in &[0.01, 0.5, 1.0, 3.7, 25.0, 1000.0, 1e5] {
            for k in 1..60 {
                let p = 10f64.powf(-(k as f64) / 4.0);
                let x = gamma_p_inv(a, p);
                // Skip targets whose solution underflows the double range.
                if gamma_p(a, f64::MIN_POSITIVE) < p {
                    assert!(rel(gamma_p(a, x), p) < 1e-12, "P⁻¹ a={a} p={p:e} x={x:e}");
                }
                let y = gamma_q_inv(a, p);
                assert!(rel(gamma_q(a, y), p) < 1e-12, "Q⁻¹ a={a} q={p:e} y={y:e}");
            }
        }
        assert_eq!(gamma_p_inv(2.0, 0.0), 0.0);
        assert!(gamma_q_inv(2.0, 0.0).is_infinite());
        assert!(gamma_p_inv(-1.0, 0.5).is_nan());
    }
}
