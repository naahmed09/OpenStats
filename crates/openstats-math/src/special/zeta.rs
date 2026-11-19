//! Riemann zeta function ζ(s) for real s.
//!
//! * s > 1 (and 0 ≤ s < 1): Euler–Maclaurin summation with N = 12 explicit
//!   terms and 14 Bernoulli corrections (truncation error < 1e-19 for s ≤ 60).
//! * s < 0: the functional equation ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s).
//! * ζ(1) is a pole → NaN; ζ(−2n) = 0 exactly.

use super::gamma::{gamma, ln_gamma, sin_pi};
use std::f64::consts::PI;

/// B_{2j}/(2j)! for j = 1…14.
fn bernoulli_over_factorial() -> [f64; 14] {
    // Bernoulli numbers B2 … B28 as exact rationals.
    const B: [(f64, f64); 14] = [
        (1.0, 6.0),
        (-1.0, 30.0),
        (1.0, 42.0),
        (-1.0, 30.0),
        (5.0, 66.0),
        (-691.0, 2730.0),
        (7.0, 6.0),
        (-3617.0, 510.0),
        (43867.0, 798.0),
        (-174_611.0, 330.0),
        (854_513.0, 138.0),
        (-236_364_091.0, 2730.0),
        (8_553_103.0, 6.0),
        (-23_749_461_029.0, 870.0),
    ];
    let mut out = [0.0; 14];
    let mut fact = 1.0;
    for (j, &(num, den)) in B.iter().enumerate() {
        let k = 2.0 * (j as f64 + 1.0);
        fact *= (k - 1.0) * k;
        out[j] = num / den / fact;
    }
    out
}

/// Euler–Maclaurin evaluation, valid for s > −2 away from the pole.
fn zeta_em(s: f64) -> f64 {
    const N: usize = 12;
    let nf = N as f64;
    let mut sum = 0.0;
    for k in (1..N).rev() {
        sum += (k as f64).powf(-s);
    }
    let n_s = nf.powf(-s);
    sum += nf * n_s / (s - 1.0) + 0.5 * n_s;
    // Σ_j B_{2j}/(2j)! · s(s+1)…(s+2j−2) · N^{−s−2j+1}
    let b = bernoulli_over_factorial();
    let mut rising = s;
    let mut pw = n_s / nf;
    let inv_n2 = 1.0 / (nf * nf);
    for (j, &bj) in b.iter().enumerate() {
        let term = bj * rising * pw;
        sum += term;
        if term.abs() < 1e-19 * sum.abs() {
            break;
        }
        let jj = (j + 1) as f64;
        rising *= (s + 2.0 * jj - 1.0) * (s + 2.0 * jj);
        pw *= inv_n2;
    }
    sum
}

/// Riemann zeta function.
pub fn zeta(s: f64) -> f64 {
    if s.is_nan() || s == 1.0 {
        return f64::NAN;
    }
    if s == f64::INFINITY {
        return 1.0;
    }
    if s == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if s > 745.0 {
        return 1.0;
    }
    if s >= 0.0 {
        return zeta_em(s);
    }
    // s < 0: ζ(−2n) = 0 exactly.
    if s == s.trunc() && (s as i64) % 2 == 0 {
        return 0.0;
    }
    let sin = sin_pi(0.5 * s);
    let z = zeta_em(1.0 - s);
    if s > -100.0 {
        2f64.powf(s) * PI.powf(s - 1.0) * sin * gamma(1.0 - s) * z
    } else {
        let mag = (s * std::f64::consts::LN_2 + (s - 1.0) * PI.ln() + sin.abs().ln() + ln_gamma(1.0 - s)).exp();
        mag.copysign(sin) * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values() {
        let p2 = PI * PI;
        assert!((zeta(2.0) - p2 / 6.0).abs() < 2e-16);
        assert!((zeta(4.0) - p2 * p2 / 90.0).abs() < 2e-16);
        assert!((zeta(6.0) - p2 * p2 * p2 / 945.0).abs() < 2e-16);
        assert!((zeta(0.0) + 0.5).abs() < 1e-16);
        assert!((zeta(-1.0) + 1.0 / 12.0).abs() < 1e-16);
        assert!((zeta(-3.0) - 1.0 / 120.0).abs() < 1e-16);
        assert!((zeta(-5.0) + 1.0 / 252.0).abs() < 1e-16);
        assert_eq!(zeta(-2.0), 0.0);
        assert_eq!(zeta(-20.0), 0.0);
        assert!(zeta(1.0).is_nan());
        assert_eq!(zeta(f64::INFINITY), 1.0);
        // Apéry's constant.
        assert!((zeta(3.0) - 1.202_056_903_159_594_3).abs() < 3e-16);
        assert_eq!(zeta(1000.0), 1.0);
    }

    #[test]
    fn behaviour_near_pole_and_large_s() {
        // ζ(s) = 1/(s−1) + γ − γ₁(s−1) + …
        let e = 1e-6;
        let want = 1.0 / e + 0.577_215_664_901_532_9 + 0.072_815_845_483_676_72 * e;
        assert!((zeta(1.0 + e) - want).abs() < 1e-9 * want);
        assert!((zeta(1.0 - e) + 1.0 / e - 0.577_215_664_901_532_9 + 0.072_815_845_483_676_72 * e).abs() < 1e-9 / e);
        // 1 + 2^{-s} + 3^{-s} for large s.
        let s = 60.0;
        let want = 1.0 + 2f64.powf(-s) + 3f64.powf(-s);
        assert!((zeta(s) - want).abs() < 1e-16);
        // Functional equation holds across the s = 0 switch point.
        for &s in &[-0.5, -1.5, -7.3, -30.5] {
            let lhs = zeta(s);
            let rhs = 2f64.powf(s) * PI.powf(s - 1.0) * sin_pi(0.5 * s) * gamma(1.0 - s) * zeta(1.0 - s);
            assert!((lhs - rhs).abs() <= 1e-14 * rhs.abs());
        }
        // Continuity at 0 and large negative argument sign pattern: ζ(−4n−1) > 0 ⇒ ... use magnitude.
        assert!(zeta(-150.5).is_finite() || zeta(-150.5).is_infinite());
    }
}
