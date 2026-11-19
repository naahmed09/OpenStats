//! Gamma, log-gamma, digamma, trigamma and beta functions.
//!
//! * lnΓ uses a Taylor series in ζ(k) on [0.5, 2.5] (so it keeps full
//!   *relative* accuracy around its zeros at 1 and 2), downward recurrence
//!   for 2.5 < x < 10, and Stirling's series with eight Bernoulli terms for
//!   x ≥ 10. Negative arguments use the reflection formula with an exact
//!   sin(πx).
//! * Γ uses Stirling's series directly (split power to avoid premature
//!   overflow) for x ≥ 10, upward recurrence below, exact factorials for
//!   integers, and reflection for x < 0.
//! * ln B follows the Stirling-correction formulation used by R's `lbeta`, so
//!   large arguments do not lose digits to cancellation.

use std::f64::consts::PI;
use std::sync::OnceLock;

/// ln √(2π)
pub const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
/// Euler–Mascheroni constant γ.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// sin(πx), exact at integers and half-integers.
pub fn sin_pi(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let y = x.abs();
    if y >= 4_503_599_627_370_496.0 {
        return 0.0; // every such double is an integer
    }
    let n = (2.0 * y).round();
    let r = y - n / 2.0; // |r| ≤ 1/4, exact
    let v = match (n as i64).rem_euclid(4) {
        0 => (PI * r).sin(),
        1 => (PI * r).cos(),
        2 => -(PI * r).sin(),
        _ => -(PI * r).cos(),
    };
    let v = if v == 0.0 { 0.0 } else { v };
    if x < 0.0 {
        -v
    } else {
        v
    }
}

/// cos(πx), exact at integers and half-integers.
pub fn cos_pi(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let y = x.abs();
    if y >= 4_503_599_627_370_496.0 {
        return if (y % 2.0) == 0.0 { 1.0 } else { -1.0 };
    }
    let n = (2.0 * y).round();
    let r = y - n / 2.0;
    let v = match (n as i64).rem_euclid(4) {
        0 => (PI * r).cos(),
        1 => -(PI * r).sin(),
        2 => -(PI * r).cos(),
        _ => (PI * r).sin(),
    };
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// Stirling remainder lnΓ(x) − [(x − ½)ln x − x + ln√(2π)] for x ≥ 10
/// (truncation error < 2e-18).
pub fn lgamma_correction(x: f64) -> f64 {
    const C: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360_360.0,
        1.0 / 156.0,
        -3617.0 / 122_400.0,
    ];
    let z = 1.0 / (x * x);
    let mut s = C[7];
    for &c in C[..7].iter().rev() {
        s = s * z + c;
    }
    s / x
}

/// ζ(k) for k = 2..=63 (index k), computed once by direct summation with an
/// Euler–Maclaurin tail.
fn zeta_table() -> &'static [f64; 64] {
    static T: OnceLock<[f64; 64]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0.0; 64];
        const N: f64 = 64.0;
        for (k, slot) in t.iter_mut().enumerate().skip(2) {
            let kf = k as f64;
            // Sum small terms first for accuracy.
            let mut s = 0.0;
            for n in (1..64).rev() {
                s += (n as f64).powf(-kf);
            }
            let nk = N.powf(-kf);
            // Euler–Maclaurin: N^{1-k}/(k-1) + N^{-k}/2 + Σ B_{2j}/(2j)! (k)_{2j-1} N^{-k-2j+1}
            let mut tail = N * nk / (kf - 1.0) + nk / 2.0;
            let b = [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0];
            let mut rising = kf; // (k)_1
            let mut fact = 2.0; // (2j)!
            let mut pw = nk / N; // N^{-k-1}
            for (j, &bj) in b.iter().enumerate() {
                tail += bj / fact * rising * pw;
                let jj = (j + 1) as f64;
                rising *= (kf + 2.0 * jj - 1.0) * (kf + 2.0 * jj);
                fact *= (2.0 * jj + 1.0) * (2.0 * jj + 2.0);
                pw /= N * N;
            }
            *slot = s + tail;
        }
        t
    })
}

/// lnΓ(1 + z) for |z| ≤ 0.5 by the ζ series (full relative accuracy).
fn lgamma1p_small(z: f64) -> f64 {
    let zt = zeta_table();
    let mut sum = 0.0;
    let mut pw = z * z; // z^k, k = 2
    for (k, &zk) in zt.iter().enumerate().skip(2) {
        let term = zk * pw / k as f64;
        let signed = if k % 2 == 0 { term } else { -term };
        sum += signed;
        if term.abs() < 1e-17 * sum.abs().max(f64::MIN_POSITIVE) {
            break;
        }
        pw *= z;
    }
    -EULER_GAMMA * z + sum
}

/// Natural logarithm of |Γ(x)|. Returns +∞ at the poles (0, −1, −2, …).
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx)
        return (PI / sin_pi(x).abs()).ln() - ln_gamma(1.0 - x);
    }
    if x < 0.5 {
        // lnΓ(x) = lnΓ(1+x) − ln x
        return lgamma1p_small(x) - x.ln();
    }
    if x <= 1.5 {
        return lgamma1p_small(x - 1.0);
    }
    if x <= 2.5 {
        let z = x - 2.0;
        return z.ln_1p() + lgamma1p_small(z);
    }
    if x < 10.0 {
        // Downward recurrence into [1.5, 2.5].
        let mut y = x;
        let mut prod = 1.0;
        while y > 2.5 {
            y -= 1.0;
            prod *= y;
        }
        let z = y - 2.0;
        return prod.ln() + z.ln_1p() + lgamma1p_small(z);
    }
    (x - 0.5) * x.ln() - x + LN_SQRT_2PI + lgamma_correction(x)
}

/// n! as f64 (exact up to 22!, correctly rounded products up to 170!).
pub fn factorial(n: u32) -> f64 {
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let mut v = vec![1.0f64; 171];
        for i in 1..171 {
            v[i] = v[i - 1] * i as f64;
        }
        v
    });
    if n as usize >= t.len() {
        f64::INFINITY
    } else {
        t[n as usize]
    }
}

/// ln(n!).
pub fn ln_factorial(n: u64) -> f64 {
    if n <= 170 {
        factorial(n as u32).ln()
    } else {
        ln_gamma(n as f64 + 1.0)
    }
}

/// Γ(x). NaN at the poles; overflows to +∞ for x > 171.62.
pub fn gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        if x < -184.0 {
            // |Γ(x)| underflows; Γ is negative on (k−1, k) for odd floor(x).
            return if (x.floor() as i64) % 2 != 0 { -0.0 } else { 0.0 };
        }
        return PI / (sin_pi(x) * gamma(1.0 - x));
    }
    if x == x.floor() && x <= 171.0 {
        return factorial(x as u32 - 1);
    }
    if x >= 10.0 {
        if x > 171.7 {
            return f64::INFINITY;
        }
        // √(2π) x^{x−½} e^{−x} e^{corr}, with the power split in halves.
        let p = x.powf(0.5 * x - 0.25);
        return (2.0 * PI).sqrt() * p * (p * (-x).exp()) * lgamma_correction(x).exp();
    }
    if x < 1e-8 {
        // Γ(x) = 1/x − γ + O(x)
        return 1.0 / x - EULER_GAMMA;
    }
    // Upward recurrence to ≥ 10.
    let mut y = x;
    let mut prod = 1.0;
    while y < 10.0 {
        prod *= y;
        y += 1.0;
    }
    gamma(y) / prod
}

/// Digamma ψ(x) = d/dx lnΓ(x). NaN at the poles.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // ψ(1−x) − ψ(x) = π cot(πx)
        return digamma(1.0 - x) - PI * cos_pi(x) / sin_pi(x);
    }
    let mut acc = 0.0;
    let mut y = x;
    if y < 1e-6 {
        // ψ(x) = −1/x − γ + (π²/6)x + O(x²)
        return -1.0 / y - EULER_GAMMA + PI * PI / 6.0 * y;
    }
    while y < 10.0 {
        acc -= 1.0 / y;
        y += 1.0;
    }
    let z = 1.0 / (y * y);
    let series = z
        * (1.0 / 12.0
            - z * (1.0 / 120.0
                - z * (1.0 / 252.0 - z * (1.0 / 240.0 - z * (1.0 / 132.0 - z * (691.0 / 32_760.0 - z / 12.0))))));
    acc + y.ln() - 0.5 / y - series
}

/// Trigamma ψ′(x). NaN at the poles.
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // ψ′(1−x) + ψ′(x) = π² / sin²(πx)
        let s = sin_pi(x);
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }
    if x < 1e-4 {
        // ψ′(x) = 1/x² + π²/6 − 2ζ(3)x + …
        return 1.0 / (x * x) + PI * PI / 6.0 - 2.404_113_806_319_188_5 * x;
    }
    let mut acc = 0.0;
    let mut y = x;
    while y < 10.0 {
        acc += 1.0 / (y * y);
        y += 1.0;
    }
    let z = 1.0 / (y * y);
    let series = 1.0 / y
        + 0.5 * z
        + z / y
            * (1.0 / 6.0
                - z * (1.0 / 30.0
                    - z * (1.0 / 42.0 - z * (1.0 / 30.0 - z * (5.0 / 66.0 - z * (691.0 / 2730.0 - z * 7.0 / 6.0))))));
    acc + series
}

/// ln B(a, b) for a, b > 0, without cancellation for large arguments.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    let (p, q) = if a < b { (a, b) } else { (b, a) };
    if p >= 10.0 {
        let corr = lgamma_correction(p) + lgamma_correction(q) - lgamma_correction(p + q);
        -0.5 * q.ln() + LN_SQRT_2PI + corr + (p - 0.5) * (p / (p + q)).ln() + q * (-p / (p + q)).ln_1p()
    } else if q >= 10.0 {
        let corr = lgamma_correction(q) - lgamma_correction(p + q);
        ln_gamma(p) + corr + p - p * (p + q).ln() + (q - 0.5) * (-p / (p + q)).ln_1p()
    } else {
        ln_gamma(p) + ln_gamma(q) - ln_gamma(p + q)
    }
}

/// B(a, b) for a, b > 0.
pub fn beta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if a + b < 171.0 {
        gamma(a) / gamma(a + b) * gamma(b)
    } else {
        ln_beta(a, b).exp()
    }
}

/// ln(1+t) − t, accurate for small |t| (used by incomplete-gamma prefixes).
pub fn log1pmx(t: f64) -> f64 {
    if t.abs() < 0.5 {
        // Σ_{k≥2} (−1)^{k+1} t^k / k
        let mut term = t * t;
        let mut s = 0.0;
        let mut k = 2.0;
        loop {
            let v = term / k;
            s += if (k as i64) % 2 == 0 { -v } else { v };
            if v.abs() <= 1e-17 * s.abs() {
                break;
            }
            term *= t;
            k += 1.0;
            if k > 200.0 {
                break;
            }
        }
        s
    } else {
        t.ln_1p() - t
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
    fn sin_cos_pi_exact() {
        assert_eq!(sin_pi(1.0), 0.0);
        assert_eq!(sin_pi(0.5), 1.0);
        assert_eq!(sin_pi(-1.5), 1.0);
        assert_eq!(cos_pi(1.0), -1.0);
        assert_eq!(cos_pi(0.5), 0.0);
        assert!(rel(sin_pi(1e-10), PI * 1e-10) < 1e-15);
    }

    #[test]
    fn identities() {
        assert_eq!(gamma(5.0), 24.0);
        assert!(rel(gamma(0.5), PI.sqrt()) < 1e-15);
        assert!(rel(gamma(-0.5), -2.0 * PI.sqrt()) < 1e-15);
        assert!(gamma(-3.0).is_nan());
        assert!(gamma(172.0).is_infinite());
        assert!(rel(gamma(170.5), 5.562_092_414_559_999e305) < 2e-14);
        assert_eq!(ln_gamma(1.0), 0.0);
        assert_eq!(ln_gamma(2.0), 0.0);
        // Near the zeros of lnΓ: relative accuracy (reference values from SciPy).
        // High-precision reference (50-digit series at the exact double 1+1e-8); SciPy is off by 2e-9 here.
        assert!(rel(ln_gamma(1.0 + 1e-8), -5.772_156_531_688_512e-9) < 1e-15);
        assert!(rel(ln_gamma(2.0 - 1e-6), -4.227_840_125_965_860e-7) < 1e-14); // 50-digit reference
        assert!(rel(ln_gamma(1e-300), 690.775_527_898_213_7) < 1e-15);
        assert!(rel(ln_gamma(1e6), 12_815_504.569_147_611) < 1e-15);
        assert!(rel(ln_gamma(-2.5), -0.056_243_716_497_674_01) < 1e-12);
        assert!(rel(digamma(1.0), -EULER_GAMMA) < 1e-15);
        assert!(rel(digamma(0.5), -EULER_GAMMA - 2.0 * 2f64.ln()) < 1e-15);
        assert!(rel(trigamma(1.0), PI * PI / 6.0) < 1e-15);
        assert!(rel(trigamma(0.5), PI * PI / 2.0) < 1e-15);
        assert!(rel(beta(2.0, 3.0), 1.0 / 12.0) < 1e-15);
        // Large-argument ln B against lnΓ differences evaluated in a stable regime.
        assert!(rel(ln_beta(1e6, 1e6), -1_386_300.003_362_920_1) < 1e-12);
        assert!(rel(ln_beta(0.5, 1e8), -8.637_975_427_801_484) < 1e-12);
        assert!(rel(log1pmx(1e-5), -4.999_966_666_916_666e-11) < 1e-14);
    }
}
