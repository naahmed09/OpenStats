//! Arithmetic and trigonometric functions.

use super::{def, map1, map1_opt, map2, Category::*, FunctionRegistry};
use crate::formula::array::Arr;
use crate::formula::eval::pow;

/// Log-gamma via the Lanczos approximation (g = 7, n = 9), relative error
/// below 2e-15 for x > 0; reflection formula for x < 0.5. (`openstats-math`
/// provides the high-precision special-function suite; this keeps the core
/// calculator self-contained.)
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Γ(x)Γ(1−x) = π / sin(πx)
        let s = (std::f64::consts::PI * x).sin().abs();
        return std::f64::consts::PI.ln() - s.ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Gamma function (exact for small positive integers).
pub fn gamma(x: f64) -> f64 {
    if x.is_nan() || (x <= 0.0 && x.fract() == 0.0) {
        return f64::NAN;
    }
    if x.fract() == 0.0 && x <= 171.0 {
        return factorial(x - 1.0);
    }
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }
    ln_gamma(x).exp()
}

/// n! for integer n ≥ 0 (exact up to 22!, exact products up to 170!).
pub fn factorial(n: f64) -> f64 {
    if n.is_nan() || n < 0.0 || n.fract() != 0.0 {
        return f64::NAN;
    }
    if n > 170.0 {
        return f64::INFINITY;
    }
    let mut r = 1.0;
    let mut k = 2.0;
    while k <= n {
        r *= k;
        k += 1.0;
    }
    r
}

/// Binomial coefficient C(n, k).
pub fn comb(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() || n.fract() != 0.0 || k.fract() != 0.0 || n < 0.0 || k < 0.0 {
        return f64::NAN;
    }
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    if n <= 1000.0 {
        let mut r = 1.0;
        let mut i = 1.0;
        while i <= k {
            r = r * (n - k + i) / i;
            i += 1.0;
        }
        r.round()
    } else {
        (ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)).exp()
    }
}

/// Permutations P(n, k) = n!/(n−k)!.
pub fn perm(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() || n.fract() != 0.0 || k.fract() != 0.0 || n < 0.0 || k < 0.0 {
        return f64::NAN;
    }
    if k > n {
        return 0.0;
    }
    let mut r = 1.0;
    let mut i = 0.0;
    while i < k {
        r *= n - i;
        i += 1.0;
    }
    r
}

/// Decimal rounding modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    /// Half away from zero (ROUND).
    HalfAway,
    /// Away from zero (ROUNDUP).
    Up,
    /// Toward zero (ROUNDDOWN / TRUNC).
    Down,
}

/// Round `x` to `d` decimal places (negative `d` rounds to tens, hundreds, …)
/// on its shortest round-trip decimal representation, so results match the
/// digits the user sees: ROUND(2.675, 2) = 2.68, TRUNC(0.29, 2) = 0.29.
pub fn decimal_round(x: f64, d: f64, mode: RoundMode) -> f64 {
    if !x.is_finite() || d.is_nan() || d.fract() != 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return x;
    }
    let d = d.clamp(-400.0, 400.0) as i64;
    let s = format!("{:e}", x.abs());
    let (mant, exp) = s.split_once('e').expect("exponent form");
    let exp: i64 = exp.parse().expect("integer exponent");
    let digits: Vec<u8> = mant.bytes().filter(|c| c.is_ascii_digit()).collect();
    let point = exp + 1; // digits before the decimal point
    let keep = point + d;
    if keep >= digits.len() as i64 {
        return x;
    }
    let (kept, rest): (&[u8], &[u8]) = if keep <= 0 { (&[], &digits[..]) } else { digits.split_at(keep as usize) };
    let bump = match mode {
        RoundMode::HalfAway => keep >= 0 && rest.first().is_some_and(|&c| c >= b'5'),
        RoundMode::Up => rest.iter().any(|&c| c != b'0'),
        RoundMode::Down => false,
    };
    let mut k: Vec<u8> = kept.to_vec();
    if bump {
        let mut i = k.len();
        loop {
            if i == 0 {
                k.insert(0, b'1');
                break;
            }
            i -= 1;
            if k[i] == b'9' {
                k[i] = b'0';
            } else {
                k[i] += 1;
                break;
            }
        }
    }
    if k.is_empty() {
        return 0.0;
    }
    // When keep < 0 and rounding up away from zero, the unit is 10^(-d).
    let ks = String::from_utf8(k).expect("ascii digits");
    let v: f64 = format!("{ks}e{}", -d).parse().expect("valid float");
    v.copysign(x)
}

/// Round half away from zero to `d` decimals.
pub fn round_to(x: f64, d: f64) -> f64 {
    decimal_round(x, d, RoundMode::HalfAway)
}

/// Truncate toward zero to `d` decimals.
pub fn trunc_to(x: f64, d: f64) -> f64 {
    decimal_round(x, d, RoundMode::Down)
}

fn gcd(a: f64, b: f64) -> f64 {
    if a.fract() != 0.0 || b.fract() != 0.0 || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let (mut a, mut b) = (a.abs(), b.abs());
    while b > 0.0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn round_multiple(x: f64, m: f64, f: fn(f64) -> f64) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    let q = x / m;
    let snapped = if (q - q.round()).abs() < 1e-9 { q.round() } else { q };
    f(snapped) * m
}

pub(super) fn register(r: &mut FunctionRegistry) {
    use std::f64::consts::PI;
    macro_rules! unary {
        ($name:literal, $cat:expr, $help:literal, $f:expr) => {
            r.register(def($name, 1, Some(1), $cat, $help, |a, _| map1(&a[0], $f)));
        };
    }
    unary!("ABS", Arithmetic, "ABS(x): absolute value", f64::abs);
    unary!("SQRT", Arithmetic, "SQRT(x): square root", f64::sqrt);
    unary!("EXP", Arithmetic, "EXP(x): e raised to x", f64::exp);
    unary!("LN", Arithmetic, "LN(x): natural logarithm", |x: f64| if x > 0.0 { x.ln() } else { f64::NAN });
    unary!("LOGTEN", Arithmetic, "LOGTEN(x): base-10 logarithm", |x: f64| if x > 0.0 { x.log10() } else { f64::NAN });
    unary!("ANTILOG", Arithmetic, "ANTILOG(x): 10 raised to x", |x: f64| 10f64.powf(x));
    unary!("SIGNS", Arithmetic, "SIGNS(x): -1, 0 or 1", |x: f64| if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x * 0.0
    });
    unary!("INT", Arithmetic, "INT(x): largest integer ≤ x", f64::floor);
    unary!("EVEN", Arithmetic, "EVEN(x): round away from zero to an even integer", |x: f64| {
        let r = (x.abs() / 2.0).ceil() * 2.0;
        r.copysign(x)
    });
    unary!("ODD", Arithmetic, "ODD(x): round away from zero to an odd integer", |x: f64| {
        let a = x.abs().ceil();
        let r = if a % 2.0 == 0.0 { a + 1.0 } else { a };
        r.copysign(x)
    });
    unary!("GAMMA", Arithmetic, "GAMMA(x): gamma function", gamma);
    unary!("LNGAMMA", Arithmetic, "LNGAMMA(x): natural log of |gamma(x)|", ln_gamma);
    unary!("FACTORIAL", Arithmetic, "FACTORIAL(n): n!", factorial);
    unary!("SQUARE", Arithmetic, "SQUARE(x): x squared", |x: f64| x * x);
    unary!("SIN", Trigonometry, "SIN(x): sine (radians)", f64::sin);
    unary!("COS", Trigonometry, "COS(x): cosine (radians)", f64::cos);
    unary!("TAN", Trigonometry, "TAN(x): tangent (radians)", f64::tan);
    unary!("ASIN", Trigonometry, "ASIN(x): arcsine", f64::asin);
    unary!("ACOS", Trigonometry, "ACOS(x): arccosine", f64::acos);
    unary!("ATAN", Trigonometry, "ATAN(x): arctangent", f64::atan);
    unary!("SINH", Trigonometry, "SINH(x): hyperbolic sine", f64::sinh);
    unary!("COSH", Trigonometry, "COSH(x): hyperbolic cosine", f64::cosh);
    unary!("TANH", Trigonometry, "TANH(x): hyperbolic tangent", f64::tanh);
    unary!("ASINH", Trigonometry, "ASINH(x): inverse hyperbolic sine", f64::asinh);
    unary!("ACOSH", Trigonometry, "ACOSH(x): inverse hyperbolic cosine", f64::acosh);
    unary!("ATANH", Trigonometry, "ATANH(x): inverse hyperbolic tangent", f64::atanh);
    unary!("DEGREES", Trigonometry, "DEGREES(x): radians to degrees", f64::to_degrees);
    unary!("RADIANS", Trigonometry, "RADIANS(x): degrees to radians", f64::to_radians);

    r.register(def("ROUND", 1, Some(2), Arithmetic, "ROUND(x, [decimals]): round half away from zero", |a, _| {
        map1_opt(a, 0.0, round_to)
    }));
    r.register(def("ROUNDUP", 1, Some(2), Arithmetic, "ROUNDUP(x, [decimals]): round away from zero", |a, _| {
        map1_opt(a, 0.0, |x, d| decimal_round(x, d, RoundMode::Up))
    }));
    r.register(def("ROUNDDOWN", 1, Some(2), Arithmetic, "ROUNDDOWN(x, [decimals]): round toward zero", |a, _| {
        map1_opt(a, 0.0, trunc_to)
    }));
    r.register(def("TRUNC", 1, Some(2), Arithmetic, "TRUNC(x, [decimals]): truncate toward zero", |a, _| {
        map1_opt(a, 0.0, trunc_to)
    }));
    r.register(def("CEILING", 1, Some(2), Arithmetic, "CEILING(x, [multiple]): round up to a multiple", |a, _| {
        map1_opt(a, 1.0, |x, m| round_multiple(x, m, f64::ceil))
    }));
    r.register(def("FLOOR", 1, Some(2), Arithmetic, "FLOOR(x, [multiple]): round down to a multiple", |a, _| {
        map1_opt(a, 1.0, |x, m| round_multiple(x, m, f64::floor))
    }));
    r.register(def("MOD", 2, Some(2), Arithmetic, "MOD(x, y): remainder with the sign of y", |a, _| {
        map2(&a[0], &a[1], |x, y| if y == 0.0 { f64::NAN } else { x - y * (x / y).floor() })
    }));
    r.register(def("QUOTIENT", 2, Some(2), Arithmetic, "QUOTIENT(x, y): integer part of x/y", |a, _| {
        map2(&a[0], &a[1], |x, y| (x / y).trunc())
    }));
    r.register(def("POWER", 2, Some(2), Arithmetic, "POWER(x, y): x raised to y", |a, _| map2(&a[0], &a[1], pow)));
    r.register(def("LOG", 1, Some(2), Arithmetic, "LOG(x, [base]): logarithm (default base 10)", |a, _| {
        map1_opt(a, 10.0, |x, b| if x > 0.0 && b > 0.0 && b != 1.0 { x.ln() / b.ln() } else { f64::NAN })
    }));
    r.register(def("ATAN2", 2, Some(2), Trigonometry, "ATAN2(y, x): angle of the point (x, y)", |a, _| {
        map2(&a[0], &a[1], f64::atan2)
    }));
    r.register(def("COMBINATIONS", 2, Some(2), Arithmetic, "COMBINATIONS(n, k): ways to choose k of n", |a, _| {
        map2(&a[0], &a[1], comb)
    }));
    r.register(def(
        "PERMUTATIONS",
        2,
        Some(2),
        Arithmetic,
        "PERMUTATIONS(n, k): ordered arrangements of k of n",
        |a, _| map2(&a[0], &a[1], perm),
    ));
    r.register(def("GCD", 2, Some(2), Arithmetic, "GCD(a, b): greatest common divisor", |a, _| {
        map2(&a[0], &a[1], gcd)
    }));
    r.register(def("LCM", 2, Some(2), Arithmetic, "LCM(a, b): least common multiple", |a, _| {
        map2(&a[0], &a[1], |x, y| {
            let g = gcd(x, y);
            if g == 0.0 {
                0.0
            } else {
                (x * y).abs() / g
            }
        })
    }));
    r.register(def("PI", 0, Some(0), Arithmetic, "PI(): the constant π", |_, _| Ok(Arr::num(PI))));
    r.register(def("E", 0, Some(0), Arithmetic, "E(): the constant e", |_, _| Ok(Arr::num(std::f64::consts::E))));

    r.alias("LOG10", "LOGTEN");
    r.alias("SIGN", "SIGNS");
    r.alias("COMB", "COMBINATIONS");
    r.alias("PERM", "PERMUTATIONS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_values() {
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
        assert!((gamma(5.5) - 52.342_777_784_553_52).abs() / 52.34 < 1e-14);
        assert!((gamma(-1.5) - 2.363_271_801_207_355).abs() < 1e-13);
        assert_eq!(gamma(6.0), 120.0);
        assert!(gamma(0.0).is_nan());
        assert!((ln_gamma(100.0) - 359.134_205_369_575_4).abs() < 1e-10);
        assert_eq!(factorial(10.0), 3_628_800.0);
        assert_eq!(comb(52.0, 5.0), 2_598_960.0);
        assert_eq!(comb(5.0, 7.0), 0.0);
        assert_eq!(perm(10.0, 3.0), 720.0);
    }

    #[test]
    fn rounding() {
        assert_eq!(round_to(2.675, 2.0), 2.68);
        assert_eq!(round_to(-2.5, 0.0), -3.0);
        assert_eq!(round_to(1234.5, -2.0), 1200.0);
        assert_eq!(round_to(0.125, 2.0), 0.13);
        assert_eq!(round_to(1.005, 2.0), 1.01);
        assert_eq!(round_to(2.4999, 0.0), 2.0);
        assert_eq!(trunc_to(0.29, 2.0), 0.29);
        assert_eq!(trunc_to(-3.789, 1.0), -3.7);
        assert_eq!(decimal_round(0.0001, 2.0, RoundMode::Up), 0.01);
        assert_eq!(decimal_round(-9.99, 1.0, RoundMode::HalfAway), -10.0);
        assert_eq!(decimal_round(4.0, -1.0, RoundMode::HalfAway), 0.0);
        assert_eq!(decimal_round(5.0, -1.0, RoundMode::HalfAway), 10.0);
        assert_eq!(decimal_round(1.2, 0.0, RoundMode::Up), 2.0);
        assert_eq!(decimal_round(123.0, -5.0, RoundMode::Up), 100000.0);
        assert_eq!(decimal_round(1e-20, 2.0, RoundMode::HalfAway), 0.0);
        assert_eq!(round_multiple(7.0, 5.0, f64::ceil), 10.0);
        assert_eq!(round_multiple(0.3, 0.1, f64::floor), 0.30000000000000004);
    }
}
