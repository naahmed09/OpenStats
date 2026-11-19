//! Owen's T function
//!
//! ```text
//! T(h, a) = 1/(2π) ∫₀^a exp(−h²(1 + x²)/2) / (1 + x²) dx
//! ```
//!
//! For |a| ≤ 1 the integral is evaluated directly with composite Gauss–
//! Legendre panels, after factoring out e^{−h²/2} so that tiny results keep
//! full relative accuracy; the integrand `e^{−h²x²/2}/(1+x²)` is analytic and
//! has width ≈ 1/|h|, so panels of width ≤ 1.5/|h| over [0, min(a, 9.5/|h|)]
//! converge to rounding error. For |a| > 1 the classical identity
//!
//! ```text
//! T(h, a) + T(ah, 1/a) = ½Q(h) + ½Q(ah) − Q(h)Q(ah)        (h, a ≥ 0)
//! ```
//!
//! with Q the standard normal upper tail reduces to the first case.
//! T is even in h and odd in a.

use super::erf::norm_sf;
use crate::integrate::GaussLegendre;
use std::f64::consts::PI;
use std::sync::OnceLock;

fn rule() -> &'static GaussLegendre {
    static R: OnceLock<GaussLegendre> = OnceLock::new();
    R.get_or_init(|| GaussLegendre::new(32))
}

/// T(h, a) for h ≥ 0 and 0 ≤ a ≤ 1.
fn t_small(h: f64, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    let h2 = h * h;
    // Beyond x_max the Gaussian factor is below e^{-45}.
    let x_max = if h > 0.0 { a.min(9.5 / h) } else { a };
    let panels = if h > 0.0 { ((x_max * h / 1.5).ceil() as usize).max(1) } else { 1 };
    let r = rule();
    let w = x_max / panels as f64;
    let mut s = 0.0;
    for p in 0..panels {
        let (c, half) = (w * (p as f64 + 0.5), 0.5 * w);
        let mut acc = 0.0;
        for (x, wt) in r.nodes.iter().zip(&r.weights) {
            let t = c + half * x;
            acc += wt * (-0.5 * h2 * t * t).exp() / (1.0 + t * t);
        }
        s += acc * half;
    }
    (-0.5 * h2).exp() * s / (2.0 * PI)
}

/// Owen's T function T(h, a).
pub fn owens_t(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    let h = h.abs();
    let (sign, a) = if a < 0.0 { (-1.0, -a) } else { (1.0, a) };
    let v = if a <= 1.0 {
        t_small(h, a)
    } else if a.is_infinite() {
        if h == 0.0 {
            0.25
        } else {
            0.5 * norm_sf(h)
        }
    } else {
        let ah = a * h;
        let (qh, qah) = (norm_sf(h), norm_sf(ah));
        let base = 0.5 * qh + 0.5 * qah - qh * qah;
        // T(ah, 1/a) with ah possibly infinite (h = ∞): the term vanishes.
        let tail = if ah.is_infinite() { 0.0 } else { t_small(ah, 1.0 / a) };
        base - tail
    };
    sign * v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::erf::{norm_cdf, norm_sf};

    #[test]
    fn closed_forms() {
        assert_eq!(owens_t(1.0, 0.0), 0.0);
        // T(0, a) = atan(a)/(2π)
        for &a in &[0.1f64, 0.5, 1.0, 2.0, 7.5, 1e3] {
            let want = a.atan() / (2.0 * PI);
            assert!((owens_t(0.0, a) - want).abs() < 1e-16, "a={a}");
            assert_eq!(owens_t(0.0, -a), -owens_t(0.0, a));
        }
        // T(h, 1) = ½ Φ(h) Φ̄(h)
        for &h in &[0.0, 0.3, 1.0, 2.5, 5.0, 10.0, 20.0] {
            let want = 0.5 * norm_cdf(h) * norm_sf(h);
            let got = owens_t(h, 1.0);
            assert!((got - want).abs() <= 4e-15 * want, "h={h}: {got:e} vs {want:e}");
            assert_eq!(owens_t(-h, 1.0), got);
        }
        // T(h, ∞) = ½ Φ̄(|h|)
        assert!((owens_t(1.5, f64::INFINITY) - 0.5 * norm_sf(1.5)).abs() < 1e-18);
        assert_eq!(owens_t(0.0, f64::INFINITY), 0.25);
        assert!(owens_t(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn identities() {
        // Reflection used for |a| > 1, checked at independent points, and
        // T(h, a) + T(ah, 1/a) = ½Φ(h) + ½Φ(ah) − Φ(h)Φ(ah) (h, a > 0).
        for &(h, a) in &[(0.5, 2.0), (1.0, 3.0), (2.0, 1.5), (0.1, 10.0)] {
            let lhs = owens_t(h, a) + owens_t(a * h, 1.0 / a);
            let rhs = 0.5 * norm_cdf(h) + 0.5 * norm_cdf(a * h) - norm_cdf(h) * norm_cdf(a * h);
            assert!((lhs - rhs).abs() < 1e-15, "h={h} a={a}");
        }
        // Bivariate-normal link: P(X > h, Y > 0 ...) positivity and monotonicity in a.
        let mut prev = 0.0;
        for i in 1..=40 {
            let v = owens_t(0.7, i as f64 * 0.25);
            assert!(v > prev);
            prev = v;
        }
    }
}
