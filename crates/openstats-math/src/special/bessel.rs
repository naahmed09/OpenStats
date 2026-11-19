//! Bessel functions of integer order: Jₙ, Yₙ (first and second kind) and
//! Iₙ, Kₙ (modified).
//!
//! Methods (all from first principles, no tabulated rational fits):
//!
//! | function | x ≤ 2 | 2 < x < 25 | x ≥ 25 |
//! |---|---|---|---|
//! | Jₙ | power series | Miller backward recurrence normalised by J₀ + 2ΣJ₂ₖ = 1 | Hankel asymptotic (J₀, J₁) + forward recurrence for n < x, Miller otherwise |
//! | Y₀, Y₁ | power series (A&S 9.1.11) | Neumann series in the Miller Jₖ | Hankel asymptotic |
//! | Yₙ | forward recurrence from Y₀, Y₁ (stable) | | |
//! | Iₙ | power series (positive terms) up to x = 30 or while n² ≥ x | | asymptotic e^x/√(2πx) series |
//! | K₀, K₁ | trapezoidal rule on ∫₀^∞ e^{−x cosh t} cosh(νt) dt (exponentially convergent) | | |
//! | Kₙ | forward recurrence from K₀, K₁ (stable) | | |
//!
//! Jₙ and Yₙ are accurate in the absolute sense near their zeros, as usual.

use super::gamma::{digamma, EULER_GAMMA};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

const HANKEL_X: f64 = 25.0;

/// Power series Jₙ(x) = Σ (−1)ᵏ (x/2)^{2k+n} / (k!(n+k)!), for small x.
fn j_series(n: u32, x: f64) -> f64 {
    let h = 0.5 * x;
    let mut term = 1.0;
    for k in 1..=n {
        term *= h / k as f64;
    }
    let mut sum = term;
    let q = -h * h;
    let mut k = 1.0;
    loop {
        term *= q / (k * (k + n as f64));
        sum += term;
        if term.abs() <= 1e-17 * sum.abs() || k > 300.0 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// Miller's algorithm: normalised J₀(x), …, J_M(x) for x > 0.
fn miller(x: f64, nmax: usize) -> Vec<f64> {
    let big = nmax.max(x.ceil() as usize);
    let mut m = big + 30 + (10.0 * (big as f64).sqrt()) as usize;
    if m % 2 == 1 {
        m += 1;
    }
    let mut j = vec![0.0; m + 2];
    j[m] = 1e-300;
    for k in (1..=m).rev() {
        j[k - 1] = 2.0 * k as f64 / x * j[k] - j[k + 1];
        if j[k - 1].abs() > 1e250 {
            for v in j.iter_mut().skip(k - 1) {
                *v *= 1e-250;
            }
        }
    }
    let mut norm = j[0];
    for k in (2..=m).step_by(2) {
        norm += 2.0 * j[k];
    }
    j.truncate(m + 1);
    j.iter_mut().for_each(|v| *v /= norm);
    j
}

/// Hankel asymptotic expansion: (P, Q) for order ν at large x.
fn hankel_pq(nu: f64, x: f64) -> (f64, f64) {
    let mu = 4.0 * nu * nu;
    let mut p = 1.0;
    let mut q = 0.0;
    let mut term = 1.0;
    let mut last = f64::INFINITY;
    for k in 1..60 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0).powi(2)) / (kf * 8.0 * x);
        if term.abs() > last {
            break; // asymptotic series started diverging
        }
        last = term.abs();
        match k % 4 {
            1 => q += term,
            2 => p -= term,
            3 => q -= term,
            _ => p += term,
        }
        if term.abs() < 1e-17 {
            break;
        }
    }
    (p, q)
}

/// (J_ν, Y_ν) for ν ∈ {0, 1} by the Hankel expansion.
fn hankel_jy(nu: u32, x: f64) -> (f64, f64) {
    let (p, q) = hankel_pq(nu as f64, x);
    let (s, c) = x.sin_cos();
    // χ = x − (2ν+1)π/4 ; cos χ and sin χ via exact angle-sum with known φ.
    let (cphi, sphi) = if nu == 0 { (FRAC_1_SQRT_2, FRAC_1_SQRT_2) } else { (-FRAC_1_SQRT_2, FRAC_1_SQRT_2) };
    let cos_chi = c * cphi + s * sphi;
    let sin_chi = s * cphi - c * sphi;
    let f = (2.0 / (PI * x)).sqrt();
    (f * (p * cos_chi - q * sin_chi), f * (p * sin_chi + q * cos_chi))
}

/// J₀(x).
pub fn j0(x: f64) -> f64 {
    jn(0, x)
}

/// J₁(x).
pub fn j1(x: f64) -> f64 {
    jn(1, x)
}

/// Jₙ(x) for integer n (negative n via J₋ₙ = (−1)ⁿJₙ).
pub fn jn(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let sign_n = if n < 0 && n % 2 != 0 { -1.0 } else { 1.0 };
    let m = n.unsigned_abs();
    let sign_x = if x < 0.0 && m % 2 == 1 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let v = if ax == 0.0 {
        if m == 0 {
            1.0
        } else {
            0.0
        }
    } else if ax.is_infinite() {
        0.0
    } else if ax <= 2.0 {
        j_series(m, ax)
    } else if ax < HANKEL_X {
        miller(ax, m as usize)[m as usize]
    } else {
        let (j0v, _) = hankel_jy(0, ax);
        if m == 0 {
            j0v
        } else {
            let (j1v, _) = hankel_jy(1, ax);
            if (m as f64) < ax {
                let (mut a, mut b) = (j0v, j1v);
                for k in 1..m {
                    let c = 2.0 * k as f64 / ax * b - a;
                    a = b;
                    b = c;
                }
                b
            } else {
                miller(ax, m as usize)[m as usize]
            }
        }
    };
    sign_n * sign_x * v
}

/// (Y₀, Y₁) for x > 0.
fn y01(x: f64) -> (f64, f64) {
    if x <= 2.0 {
        let h = 0.5 * x;
        let lh = h.ln();
        let q = -h * h;
        // Y₀ = (2/π)(ln(x/2) + γ)J₀ − (2/π) Σ_{k≥1} H_k (−x²/4)^k/(k!)²
        let mut term = 1.0;
        let mut hk = 0.0;
        let mut s0 = 0.0;
        // Y₁ = −2/(πx) + (2/π)ln(x/2)J₁ − (1/π)(x/2) Σ (ψ(k+1)+ψ(k+2)) (−x²/4)^k/(k!(k+1)!)
        let mut t1 = 1.0;
        let mut s1 = 0.0;
        for k in 0..200 {
            let kf = k as f64;
            if k > 0 {
                term *= q / (kf * kf);
                hk += 1.0 / kf;
                s0 += hk * term;
                t1 *= q / (kf * (kf + 1.0));
            }
            let c = digamma(kf + 1.0) + digamma(kf + 2.0);
            s1 += c * t1;
            if k > 2 && term.abs() < 1e-18 && t1.abs() < 1e-18 {
                break;
            }
        }
        let y0 = 2.0 / PI * ((lh + EULER_GAMMA) * j_series(0, x) - s0);
        let y1 = -2.0 / (PI * x) + 2.0 / PI * lh * j_series(1, x) - h / PI * s1;
        (y0, y1)
    } else if x < HANKEL_X {
        let j = miller(x, 2);
        let l = (0.5 * x).ln() + EULER_GAMMA;
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut k = 1;
        while 2 * k + 1 < j.len() {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            s0 += sign * j[2 * k] / k as f64;
            s1 += sign * (j[2 * k - 1] - j[2 * k + 1]) / k as f64;
            k += 1;
        }
        let y0 = 2.0 / PI * l * j[0] - 4.0 / PI * s0;
        let y1 = -2.0 / PI * (j[0] / x - l * j[1]) + 2.0 / PI * s1;
        (y0, y1)
    } else {
        (hankel_jy(0, x).1, hankel_jy(1, x).1)
    }
}

/// Y₀(x) (NaN for x < 0, −∞ at 0).
pub fn y0(x: f64) -> f64 {
    yn(0, x)
}

/// Y₁(x).
pub fn y1(x: f64) -> f64 {
    yn(1, x)
}

/// Yₙ(x) for integer n (Y₋ₙ = (−1)ⁿYₙ).
pub fn yn(n: i32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let sign = if n < 0 && n % 2 != 0 { -1.0 } else { 1.0 };
    let m = n.unsigned_abs();
    let (a0, a1) = y01(x);
    if m == 0 {
        return a0;
    }
    let (mut a, mut b) = (a0, a1);
    for k in 1..m {
        let c = 2.0 * k as f64 / x * b - a;
        a = b;
        b = c;
        if b.is_infinite() {
            break;
        }
    }
    sign * b
}

/// Iₙ(x) for integer n.
pub fn in_(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let m = n.unsigned_abs(); // I₋ₙ = Iₙ
    let sign_x = if x < 0.0 && m % 2 == 1 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax == 0.0 {
        return if m == 0 { 1.0 } else { 0.0 };
    }
    let mf = m as f64;
    let v = if ax <= 30.0 || mf * mf >= ax {
        let h = 0.5 * ax;
        // Leading term computed in logs to avoid overflow for large n.
        let lead = (mf * h.ln() - super::gamma::ln_gamma(mf + 1.0)).exp();
        let mut term = 1.0;
        let mut sum = 1.0;
        let q = h * h;
        let mut k = 1.0;
        loop {
            term *= q / (k * (k + mf));
            sum += term;
            if term <= 1e-17 * sum || k > 2000.0 {
                break;
            }
            k += 1.0;
        }
        lead * sum
    } else {
        // I_n(x) ~ e^x/√(2πx) Σ (−1)^k a_k(n)/x^k
        let mu = 4.0 * mf * mf;
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut last = f64::INFINITY;
        for k in 1..60 {
            let kf = k as f64;
            term *= -(mu - (2.0 * kf - 1.0).powi(2)) / (kf * 8.0 * ax);
            if term.abs() > last {
                break;
            }
            last = term.abs();
            sum += term;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        let e = (0.5 * ax).exp(); // split to delay overflow
        e * (e * sum / (2.0 * PI * ax).sqrt())
    };
    sign_x * v
}

/// I₀(x).
pub fn i0(x: f64) -> f64 {
    in_(0, x)
}

/// I₁(x).
pub fn i1(x: f64) -> f64 {
    in_(1, x)
}

/// e^x·K_ν(x) by the trapezoidal rule on ∫₀^∞ e^{−x(cosh t − 1)} cosh(νt) dt.
fn k_scaled(nu: f64, x: f64) -> f64 {
    // The integrand is ≈ e^{−x t²/2} near 0: keep ≥ 2 points per width 1/√x
    // (trapezoid error ~ exp(−2π²/(h√x)²)), and h ≤ 0.1 for the
    // analytic-strip bound at small x.
    let h = 0.1f64.min(0.5 / x.sqrt());
    let f = |t: f64| {
        let s = (0.5 * t).sinh();
        (-x * 2.0 * s * s).exp() * (nu * t).cosh()
    };
    let mut sum = 0.5 * f(0.0);
    let mut k = 1.0;
    loop {
        let v = f(k * h);
        sum += v;
        if v < 1e-18 * sum && k * h * x.sqrt().max(1.0) > 1.0 {
            break;
        }
        k += 1.0;
        if k > 20_000.0 {
            break;
        }
    }
    sum * h
}

/// Kₙ(x) for x > 0 (NaN for x < 0, +∞ at 0).
pub fn kn(n: i32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x > 745.0 {
        return 0.0;
    }
    let m = n.unsigned_abs(); // K₋ₙ = Kₙ
    let e = (-x).exp();
    let (k0, k1) = (k_scaled(0.0, x) * e, k_scaled(1.0, x) * e);
    if m == 0 {
        return k0;
    }
    let (mut a, mut b) = (k0, k1);
    for k in 1..m {
        let c = 2.0 * k as f64 / x * b + a;
        a = b;
        b = c;
        if b.is_infinite() {
            break;
        }
    }
    b
}

/// K₀(x).
pub fn k0(x: f64) -> f64 {
    kn(0, x)
}

/// K₁(x).
pub fn k1(x: f64) -> f64 {
    kn(1, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wronskians_and_identities() {
        // J₁Y₀ − J₀Y₁ = 2/(πx) and I₀K₁ + I₁K₀ = 1/x across all regimes.
        for &x in &[0.01, 0.5, 1.9, 2.1, 5.0, 12.3, 24.9, 25.1, 60.0, 400.0] {
            let w = jn(1, x) * yn(0, x) - jn(0, x) * yn(1, x);
            assert!((w * PI * x / 2.0 - 1.0).abs() < 1e-13, "JY Wronskian at {x}: {w}");
            if x < 600.0 {
                let wi = in_(0, x) * kn(1, x) + in_(1, x) * kn(0, x);
                assert!((wi * x - 1.0).abs() < 1e-13, "IK Wronskian at {x}");
            }
        }
        // Σ Jₖ² normalisation: J₀² + 2ΣJₖ² = 1.
        for &x in &[1.0, 7.5, 30.0] {
            let s: f64 = jn(0, x).powi(2) + 2.0 * (1..120).map(|k| jn(k, x).powi(2)).sum::<f64>();
            assert!((s - 1.0).abs() < 1e-13, "sum rule at {x}");
        }
        // Symmetries.
        assert_eq!(jn(-3, 2.5), -jn(3, 2.5));
        assert_eq!(jn(3, -2.5), -jn(3, 2.5));
        assert_eq!(in_(-2, 1.5), in_(2, 1.5));
        assert_eq!(jn(0, 0.0), 1.0);
        assert!(yn(0, 0.0).is_infinite() && yn(0, -1.0).is_nan());
        assert!(kn(0, 0.0).is_infinite());
    }

    #[test]
    fn known_zeros_and_values() {
        // First zero of J₀: 2.404825557695773; of Y₀: 0.8935769662791675.
        assert!(jn(0, 2.404_825_557_695_773).abs() < 1e-15);
        assert!(yn(0, 0.893_576_966_279_167_5).abs() < 1e-15);
        // I₀(1) = 1.2660658777520082, K₀(1) = 0.42102443824070834 (SciPy).
        assert!((i0(1.0) - 1.266_065_877_752_008_2).abs() < 1e-15);
        assert!((k0(1.0) / 0.421_024_438_240_708_34 - 1.0).abs() < 1e-14);
    }
}
