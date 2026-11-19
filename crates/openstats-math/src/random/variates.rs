//! Exact random variate algorithms.
//!
//! All samplers draw from `&mut dyn Rng` and are exact (up to double
//! precision arithmetic): there are no approximate normal or Poisson
//! shortcuts. Algorithms:
//!
//! * normal: Marsaglia's polar method;
//! * exponential: inversion `−ln U`;
//! * gamma: Marsaglia–Tsang squeeze for shape ≥ 1, with the Stuart boost
//!   `G(a) = G(a+1)·U^{1/a}` below 1, also available on the log scale so
//!   tiny shapes never underflow inside other samplers;
//! * beta: ratio of log-scale gammas (exact for every a, b > 0);
//! * Poisson: sequential inversion below mean 10, Hörmann's transformed
//!   rejection (PTRS) above;
//! * binomial: sequential inversion when n·min(p, 1−p) < 30, otherwise
//!   Knuth's beta splitting recursion (exact, O(log n) beta draws);
//! * hypergeometric: inversion walking outward from the mode with pmf
//!   ratio recurrences;
//! * negative binomial: gamma–Poisson mixture; geometric: inversion;
//! * non-central χ², t, F, beta: from their defining constructions.

use super::Rng;
use crate::special::gamma::ln_gamma;

/// Standard normal variate (Marsaglia polar method).
pub fn std_normal(rng: &mut dyn Rng) -> f64 {
    loop {
        let u = 2.0 * rng.next_f64() - 1.0;
        let v = 2.0 * rng.next_f64() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

/// Standard exponential variate.
pub fn std_exponential(rng: &mut dyn Rng) -> f64 {
    -rng.next_f64_open().ln()
}

/// ln of a Gamma(shape, 1) variate. Finite for every shape > 0.
pub fn ln_gamma_variate(rng: &mut dyn Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        // G(a) = G(a+1) · U^{1/a}
        let g = gamma_ge1(rng, shape + 1.0);
        return g.ln() + rng.next_f64_open().ln() / shape;
    }
    gamma_ge1(rng, shape).ln()
}

fn gamma_ge1(rng: &mut dyn Rng, shape: f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = std_normal(rng);
        let t = 1.0 + c * x;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u = rng.next_f64_open();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Gamma(shape, 1) variate.
pub fn std_gamma(rng: &mut dyn Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        return ln_gamma_variate(rng, shape).exp();
    }
    gamma_ge1(rng, shape)
}

/// Chi-square variate with `df` degrees of freedom.
pub fn chi_square(rng: &mut dyn Rng, df: f64) -> f64 {
    2.0 * std_gamma(rng, 0.5 * df)
}

/// Beta(a, b) variate.
pub fn beta(rng: &mut dyn Rng, a: f64, b: f64) -> f64 {
    let lx = ln_gamma_variate(rng, a);
    let ly = ln_gamma_variate(rng, b);
    1.0 / (1.0 + (ly - lx).exp())
}

/// Student t variate; `df = ∞` is the normal distribution.
pub fn student_t(rng: &mut dyn Rng, df: f64) -> f64 {
    let z = std_normal(rng);
    if df.is_infinite() {
        return z;
    }
    z / (chi_square(rng, df) / df).sqrt()
}

/// Fisher F variate, computed on the log scale so that small degrees of
/// freedom cannot produce 0/0.
pub fn fisher_f(rng: &mut dyn Rng, d1: f64, d2: f64) -> f64 {
    let l1 = ln_gamma_variate(rng, 0.5 * d1);
    let l2 = ln_gamma_variate(rng, 0.5 * d2);
    (l1 - l2).exp() * (d2 / d1)
}

/// Poisson(mean) variate.
pub fn poisson(rng: &mut dyn Rng, mean: f64) -> u64 {
    if mean <= 0.0 {
        return 0;
    }
    if mean < 10.0 {
        let u = rng.next_f64();
        let mut k = 0u64;
        let mut p = (-mean).exp();
        let mut s = p;
        while u > s {
            k += 1;
            p *= mean / k as f64;
            let ns = s + p;
            if ns == s {
                break; // remaining mass is below double precision
            }
            s = ns;
        }
        return k;
    }
    // Hörmann (1993), "The transformed rejection method for generating
    // Poisson random variables" (PTRS).
    let slam = mean.sqrt();
    let loglam = mean.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.next_f64_open() - 0.5;
        let v = rng.next_f64_open();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + mean + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k as u64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln() <= -mean + k * loglam - ln_gamma(k + 1.0) {
            return k as u64;
        }
    }
}

/// Binomial(n, p) variate.
pub fn binomial(rng: &mut dyn Rng, n: u64, p: f64) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    if p > 0.5 {
        return n - binomial(rng, n, 1.0 - p);
    }
    let nf = n as f64;
    if nf * p < 30.0 {
        // BINV: sequential inversion from 0.
        let q = 1.0 - p;
        let s = p / q;
        let a = (nf + 1.0) * s;
        let mut r = (nf * (-p).ln_1p()).exp();
        let mut u = rng.next_f64();
        let mut x = 0u64;
        while u > r {
            u -= r;
            x += 1;
            if x > n {
                return n;
            }
            r *= a / x as f64 - s;
            if r <= 0.0 {
                break;
            }
        }
        return x;
    }
    // Knuth: with a = 1 + ⌊n/2⌋, b = n + 1 − a and X ~ Beta(a, b) the
    // a-th order statistic of n uniforms: if X ≥ p then Bin(n, p) =
    // Bin(a − 1, p/X), else a + Bin(b − 1, (p − X)/(1 − X)).
    let a = 1 + n / 2;
    let b = n + 1 - a;
    let x = beta(rng, a as f64, b as f64);
    if x >= p {
        binomial(rng, a - 1, p / x)
    } else {
        a + binomial(rng, b - 1, (p - x) / (1.0 - x))
    }
}

/// Geometric variate counting trials (support 1, 2, …).
pub fn geometric(rng: &mut dyn Rng, p: f64) -> u64 {
    if p >= 1.0 {
        return 1;
    }
    let k = (rng.next_f64_open().ln() / (-p).ln_1p()).floor() + 1.0;
    if k >= 1.8e19 {
        u64::MAX
    } else {
        k as u64
    }
}

/// Negative binomial variate counting trials (support r, r+1, …): a Poisson
/// whose mean is Gamma(r, (1−p)/p), plus r.
pub fn negative_binomial(rng: &mut dyn Rng, r: u64, p: f64) -> u64 {
    if p >= 1.0 {
        return r;
    }
    let lambda = std_gamma(rng, r as f64) * ((1.0 - p) / p);
    r.saturating_add(poisson(rng, lambda))
}

/// Hypergeometric variate: successes among `draws` items taken without
/// replacement from `population` items of which `successes` are marked.
/// Inversion over the pmf, walking outward from the mode.
pub fn hypergeometric(rng: &mut dyn Rng, population: u64, successes: u64, draws: u64) -> u64 {
    let (nn, kk, n) = (population, successes, draws);
    let lo = (n + kk).saturating_sub(nn);
    let hi = n.min(kk);
    if lo == hi {
        return lo;
    }
    let mode = (((n as f64 + 1.0) * (kk as f64 + 1.0) / (nn as f64 + 2.0)).floor() as u64).clamp(lo, hi);
    let d = crate::dist::Hypergeometric { population, successes, draws };
    let p_mode = crate::dist::DiscreteDistribution::pmf(&d, mode as f64);
    // ratio(k) = P(k+1)/P(k)
    let ratio =
        |k: u64| -> f64 { ((kk - k) as f64 * (n - k) as f64) / ((k + 1) as f64 * ((nn - kk + k + 1) - n) as f64) };
    let u = rng.next_f64();
    let mut cum = p_mode;
    if u <= cum {
        return mode;
    }
    let (mut up, mut p_up) = (mode, p_mode);
    let (mut dn, mut p_dn) = (mode, p_mode);
    loop {
        let mut progressed = false;
        if up < hi {
            p_up *= ratio(up);
            up += 1;
            cum += p_up;
            progressed = true;
            if u <= cum {
                return up;
            }
        }
        if dn > lo {
            p_dn /= ratio(dn - 1);
            dn -= 1;
            cum += p_dn;
            progressed = true;
            if u <= cum {
                return dn;
            }
        }
        if !progressed {
            // Rounding left u above the accumulated mass: return the side
            // that carried more of it.
            return if p_up >= p_dn { up } else { dn };
        }
    }
}

/// Uniform integer in `a..=b`.
pub fn uniform_int(rng: &mut dyn Rng, a: i64, b: i64) -> i64 {
    let width = (i128::from(b) - i128::from(a) + 1) as u128;
    if width > u128::from(u64::MAX) {
        return (i128::from(a) + i128::from(rng.next_u64())) as i64;
    }
    (i128::from(a) + i128::from(rng.below(width as u64))) as i64
}

/// Non-central χ²(df, λ): Poisson(λ/2) mixture of central χ² variates.
pub fn nc_chi_square(rng: &mut dyn Rng, df: f64, lambda: f64) -> f64 {
    let j = if lambda > 0.0 { poisson(rng, 0.5 * lambda) as f64 } else { 0.0 };
    let total = df + 2.0 * j;
    if total <= 0.0 {
        0.0
    } else {
        chi_square(rng, total)
    }
}

/// Non-central t(df, δ) = (Z + δ)/√(χ²_df/df).
pub fn nc_t(rng: &mut dyn Rng, df: f64, delta: f64) -> f64 {
    let z = std_normal(rng) + delta;
    if df.is_infinite() {
        return z;
    }
    z / (chi_square(rng, df) / df).sqrt()
}

/// Non-central F: (χ'²_{d1}(λ)/d1) / (χ²_{d2}/d2).
pub fn nc_f(rng: &mut dyn Rng, d1: f64, d2: f64, lambda: f64) -> f64 {
    (nc_chi_square(rng, d1, lambda) / d1) / (chi_square(rng, d2) / d2)
}

/// Non-central beta: X/(X+Y) with X ~ χ'²_{2a}(λ), Y ~ χ²_{2b}.
pub fn nc_beta(rng: &mut dyn Rng, a: f64, b: f64, lambda: f64) -> f64 {
    let x = nc_chi_square(rng, 2.0 * a, lambda);
    let y = chi_square(rng, 2.0 * b);
    x / (x + y)
}
