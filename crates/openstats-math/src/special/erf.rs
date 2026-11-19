//! Error functions and the standard normal distribution.
//!
//! * erf for |x| < 1.5 uses the all-positive series
//!   erf(x) = (2/√π) e^{−x²} Σ 2ⁿ x^{2n+1} / (2n+1)!!.
//! * erfc / erfcx for x ≥ 1.5 use Legendre's continued fraction for
//!   Γ(½, x²) evaluated with the modified Lentz method.
//! * e^{−x²} is evaluated with a hi/lo split of x so that it stays accurate
//!   to a few ulps even when x² is large (deep tails).
//! * The normal survival function for |x| > 2.5 uses the Laplace continued
//!   fraction for Mills' ratio, avoiding the rounding error of x/√2.
//! * Inverses start from the Abramowitz–Stegun 26.2.23 approximation and are
//!   polished with Halley iterations on the accurate forward functions.

use super::FPMIN;
use std::f64::consts::{FRAC_2_SQRT_PI, PI, SQRT_2};

/// 1/√(2π)
pub const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

fn split(x: f64) -> (f64, f64) {
    let hi = f64::from_bits(x.to_bits() & 0xFFFF_FFFF_F800_0000);
    (hi, x - hi)
}

/// e^{−x²} accurate to a few ulps for all x.
pub fn exp_neg_sq(x: f64) -> f64 {
    let x = x.abs();
    let (hi, lo) = split(x);
    (-hi * hi).exp() * (-(2.0 * hi * lo + lo * lo)).exp()
}

/// e^{−x²/2} accurate to a few ulps for all x.
pub fn exp_neg_half_sq(x: f64) -> f64 {
    let x = x.abs();
    let (hi, lo) = split(x);
    (-0.5 * hi * hi).exp() * (-(hi * lo + 0.5 * lo * lo)).exp()
}

/// Series erf for |x| < 1.5 (positive terms only).
fn erf_series(x: f64) -> f64 {
    let x2 = 2.0 * x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 1.0;
    while term.abs() > 1e-17 * sum.abs() {
        term *= x2 / (2.0 * n + 1.0);
        sum += term;
        n += 1.0;
        if n > 500.0 {
            break;
        }
    }
    FRAC_2_SQRT_PI * exp_neg_sq(x) * sum
}

/// Continued fraction K with erfc(x) = e^{−x²}·x·K/√π (x ≥ 1.5), from
/// Γ(a, z) = e^{−z} z^a · K(a, z) at a = ½, z = x².
fn erfc_cf(x: f64) -> f64 {
    let z = x * x;
    let a = 0.5;
    let mut b = z + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..10_000 {
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
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// erf(x).
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() < 1.5 {
        erf_series(x)
    } else if x > 0.0 {
        1.0 - erfc(x)
    } else {
        erfc(-x) - 1.0
    }
}

/// erfc(x) = 1 − erf(x), with full relative accuracy for large x.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 1.5 {
        return 1.0 - erf_series(x);
    }
    if x > 27.3 {
        return 0.0;
    }
    exp_neg_sq(x) * x * erfc_cf(x) / PI.sqrt()
}

/// Scaled complementary error function erfcx(x) = e^{x²} erfc(x).
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 1.5 {
        if x < -26.7 {
            return f64::INFINITY;
        }
        // e^{x²}·erfc(x) with e^{x²} = 1/e^{−x²}.
        return erfc(x) / exp_neg_sq(x);
    }
    if x > 1e8 {
        // erfcx(x) = 1/(x√π) · (1 − 1/(2x²) + …)
        return 1.0 / (x * PI.sqrt()) * (1.0 - 0.5 / (x * x));
    }
    x * erfc_cf(x) / PI.sqrt()
}

/// Inverse error function: erf(erf_inv(y)) = y for −1 < y < 1.
pub fn erf_inv(y: f64) -> f64 {
    if y.is_nan() || !(-1.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }
    if y == -1.0 {
        return f64::NEG_INFINITY;
    }
    if y == 0.0 {
        return y;
    }
    if y.abs() < 0.5 {
        // Series start, then Halley on erf.
        let mut x = y * PI.sqrt() / 2.0 * (1.0 + PI * y * y / 12.0);
        for _ in 0..8 {
            let k = FRAC_2_SQRT_PI * exp_neg_sq(x);
            let u = (erf(x) - y) / k;
            let dx = u / (1.0 + x * u);
            x -= dx;
            if dx.abs() <= 1e-17 * x.abs() {
                break;
            }
        }
        return x;
    }
    let s = y.signum();
    s * erfc_inv(1.0 - y.abs())
}

/// Inverse complementary error function: erfc(erfc_inv(z)) = z for 0 < z < 2.
pub fn erfc_inv(z: f64) -> f64 {
    if z.is_nan() || !(0.0..=2.0).contains(&z) {
        return f64::NAN;
    }
    if z == 0.0 {
        return f64::INFINITY;
    }
    if z == 2.0 {
        return f64::NEG_INFINITY;
    }
    if z > 1.0 {
        return -erfc_inv(2.0 - z);
    }
    if z > 0.5 {
        return erf_inv(1.0 - z);
    }
    let mut x = norm_isf(z / 2.0) / SQRT_2;
    for _ in 0..4 {
        let k = FRAC_2_SQRT_PI * exp_neg_sq(x);
        if k == 0.0 {
            break;
        }
        let u = (erfc(x) - z) / k;
        let dx = u / (1.0 - x * u);
        x += dx;
        if dx.abs() <= 1e-17 * x.abs() {
            break;
        }
    }
    x
}

/// Standard normal density φ(x).
pub fn norm_pdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    FRAC_1_SQRT_2PI * exp_neg_half_sq(x)
}

/// ln φ(x).
pub fn norm_ln_pdf(x: f64) -> f64 {
    -0.5 * x * x - super::gamma::LN_SQRT_2PI
}

/// Laplace's continued fraction for Mills' ratio R(x) = Q(x)/φ(x), x > 0:
/// R = 1/(x + 1/(x + 2/(x + 3/(x + …)))).
fn mills_cf(x: f64) -> f64 {
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for n in 1..5000 {
        let an = n as f64;
        d = x + an * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = x + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = c * d;
        f *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    1.0 / f
}

/// Standard normal survival function Q(x) = 1 − Φ(x).
pub fn norm_sf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 2.5 {
        if x > 38.6 {
            return 0.0;
        }
        norm_pdf(x) * mills_cf(x)
    } else if x < -2.5 {
        1.0 - norm_sf(-x)
    } else {
        0.5 * erfc(x / SQRT_2)
    }
}

/// Standard normal distribution function Φ(x).
pub fn norm_cdf(x: f64) -> f64 {
    norm_sf(-x)
}

/// Upper-tail quantile: Q(x) = q (x = Φ⁻¹(1 − q) without forming 1 − q).
pub fn norm_isf(q: f64) -> f64 {
    if q.is_nan() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return f64::NEG_INFINITY;
    }
    if q == 0.5 {
        return 0.0;
    }
    if q > 0.5 {
        return -norm_isf(1.0 - q);
    }
    // A&S 26.2.23 (|error| < 4.5e-4), then Halley on Q.
    let t = (-2.0 * q.ln()).sqrt();
    let mut x =
        t - (2.515_517 + t * (0.802_853 + t * 0.010_328)) / (1.0 + t * (1.432_788 + t * (0.189_269 + t * 0.001_308)));
    for _ in 0..6 {
        let pdf = norm_pdf(x);
        if pdf == 0.0 {
            break;
        }
        let u = (norm_sf(x) - q) / pdf;
        let dx = u / (1.0 - 0.5 * x * u);
        x += dx;
        if dx.abs() <= 2e-16 * x.abs() {
            break;
        }
    }
    x
}

/// Standard normal quantile Φ⁻¹(p).
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p < 0.5 {
        -norm_isf(p)
    } else if p > 0.5 {
        // Solve Φ(x) = p directly so that p close to 0.5 keeps precision.
        if p > 0.75 {
            return norm_isf(1.0 - p);
        }
        let mut x = norm_isf(1.0 - p);
        for _ in 0..4 {
            let pdf = norm_pdf(x);
            let u = (norm_cdf(x) - p) / pdf;
            let dx = u / (1.0 + 0.5 * x * u);
            x -= dx;
            if dx.abs() <= 2e-16 * x.abs() {
                break;
            }
        }
        x
    } else {
        0.0
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
    fn known_values() {
        // References: SciPy.
        assert!(rel(erf(0.5), 0.520_499_877_813_046_5) < 1e-15);
        assert!(rel(erf(1e-10), 1.128_379_167_095_512_6e-10) < 1e-15);
        assert!(rel(erfc(2.0), 0.004_677_734_981_047_266) < 1e-15);
        assert!(rel(erfc(10.0), 2.088_487_583_762_544_7e-45) < 1e-14);
        assert!(rel(erfc(26.0), 5.663_192_408_856_145e-296) < 1e-13);
        assert!(rel(erfcx(30.0), 0.018_795_888_861_416_75) < 1e-15);
        assert!(rel(norm_sf(5.0), 2.866_515_718_791_933e-07) < 1e-14);
        assert!(rel(norm_sf(37.5), 4.605_353_009_581_954e-308) < 1e-12);
        assert!(rel(norm_cdf(-1.0), 0.158_655_253_931_457_05) < 1e-15);
        assert!(rel(norm_ppf(0.975), 1.959_963_984_540_054) < 1e-15);
        assert!(rel(norm_ppf(1e-300), -37.047_096_299_361_2) < 1e-14);
        assert!(rel(norm_isf(1e-10), 6.361_340_902_404_056) < 1e-15);
    }

    #[test]
    fn inverse_roundtrips() {
        for i in 1..200 {
            let y = -0.999 + i as f64 * 0.00999;
            let x = erf_inv(y);
            assert!((erf(x) - y).abs() <= 5e-16, "erf_inv({y})");
        }
        for k in 1..300 {
            let z = 10f64.powf(-(k as f64));
            let x = erfc_inv(z);
            assert!(rel(erfc(x), z) < 1e-13, "erfc_inv({z:e})");
        }
        for k in 1..300 {
            // Round trips are limited by the conditioning of Φ: a 1-ulp change
            // in x changes Φ(x) by about x²·ε relative.
            let p = 10f64.powf(-(k as f64));
            let x = norm_ppf(p);
            assert!(rel(norm_cdf(x), p) < 4e-16 * (1.0 + x * x) + 1e-15, "p={p:e}");
        }
        assert!(rel(norm_ppf(0.5 + 1e-12), 2.506_628_274_6e-12) < 1e-4);
        assert!(norm_ppf(0.0).is_infinite() && norm_ppf(1.0).is_infinite());
        assert!(norm_ppf(1.5).is_nan());
    }

    #[test]
    fn symmetry_and_limits() {
        for &x in &[0.1, 1.0, 2.4, 2.6, 7.0] {
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 1e-16);
            assert!((erf(x) + erf(-x)).abs() < 1e-16);
            assert!(rel(erfcx(x), erfc(x) / exp_neg_sq(x)) < 1e-14);
        }
        assert_eq!(erfc(30.0), 0.0);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert!(erfcx(-30.0).is_infinite());
    }
}
