//! Exponential integrals E₁ and Ei.
//!
//! ```text
//! E₁(x) = ∫ₓ^∞ e^{−t}/t dt            (x > 0)
//! Ei(x) = −PV ∫_{−x}^∞ e^{−t}/t dt    (x ≠ 0),   Ei(−x) = −E₁(x)
//! ```
//!
//! E₁ uses its Maclaurin series for x ≤ 1 and the Legendre continued
//! fraction (modified Lentz) for larger x. Ei uses the all-positive series
//! for 0 < x ≤ 40 and the asymptotic expansion beyond (smallest term < e^{−x}
//! relative).

use super::FPMIN;

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// E₁(x). NaN for x < 0 (use [`ei`] for the real part), +∞ at 0.
pub fn e1(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x > 745.2 {
        return 0.0;
    }
    if x <= 1.0 {
        // −γ − ln x − Σ_{k≥1} (−x)^k / (k k!)
        let mut sum = 0.0;
        let mut term = 1.0; // (−x)^k / k!
        for k in 1..60 {
            term *= -x / k as f64;
            let t = term / k as f64;
            sum += t;
            if t.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        -EULER_GAMMA - x.ln() - sum
    } else {
        // E₁(x) = e^{−x} · 1/(x + 1 − 1/(x + 3 − 4/(x + 5 − …)))
        let mut b = x + 1.0;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -((i * i) as f64);
            b += 2.0;
            d = 1.0 / (an * d + b);
            c = b + an / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// Ei(x) (principal value for x > 0). −∞ at 0; Ei(−x) = −E₁(x).
pub fn ei(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -e1(-x);
    }
    if x > 710.0 {
        return f64::INFINITY;
    }
    if x <= 40.0 {
        // γ + ln x + Σ x^k / (k k!), all terms positive.
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..400 {
            term *= x / k as f64;
            let t = term / k as f64;
            sum += t;
            if t < 1e-17 * sum {
                break;
            }
        }
        EULER_GAMMA + x.ln() + sum
    } else {
        // e^x/x · Σ k!/x^k, summed until the terms start to grow.
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..200 {
            let next = term * k as f64 / x;
            if next >= term || next < 1e-17 {
                break;
            }
            term = next;
            sum += term;
        }
        x.exp() / x * sum
    }
}

/// The unique positive zero of Ei: ln μ with μ = 1.4513692348833810… the
/// Ramanujan–Soldner constant (li(μ) = Ei(ln μ) = 0).
pub const EI_ZERO: f64 = 0.372_507_410_781_366_6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // E1(1) = 0.21938393439552029, Ei(1) = 1.8951178163559368
        assert!((e1(1.0) - 0.219_383_934_395_520_3).abs() < 2e-16);
        assert!((ei(1.0) - 1.895_117_816_355_936_8).abs() < 5e-16);
        assert!(ei(EI_ZERO).abs() < 2e-16);
        assert!((EI_ZERO.exp() - 1.451_369_234_883_381).abs() < 1e-15);
        assert_eq!(e1(0.0), f64::INFINITY);
        assert_eq!(ei(0.0), f64::NEG_INFINITY);
        assert!(e1(-1.0).is_nan());
        assert_eq!(e1(1000.0), 0.0);
        assert!((ei(-2.0) + e1(2.0)).abs() == 0.0);
    }

    #[test]
    fn identities() {
        // d/dx E1 = −e^{−x}/x by central difference, continuity across the
        // series/CF switch at 1 and across Ei's switch at 40.
        for &x in &[0.1f64, 0.5, 0.999, 1.001, 3.0, 10.0, 30.0] {
            let h = 1e-4 * x.min(1.0);
            let d = (e1(x + h) - e1(x - h)) / (2.0 * h);
            assert!((d + (-x).exp() / x).abs() < 1e-8 * (-x).exp() / x, "x={x}");
        }
        assert!((e1(1.0 - 1e-15) - e1(1.0 + 1e-15)).abs() < 1e-14);
        let (a, b) = (ei(40.0), ei(40.0 + 8e-15));
        assert!((a - b).abs() < 2e-14 * a, "series/asymptotic switch");
        // E1(x) ≈ e^{−x}/x (1 − 1/x + 2/x² − …) for large x
        let x = 200.0f64;
        let asym = (-x).exp() / x * (1.0 - 1.0 / x + 2.0 / (x * x) - 6.0 / x.powi(3) + 24.0 / x.powi(4));
        assert!((e1(x) - asym).abs() < 1e-8 * asym);
    }
}
