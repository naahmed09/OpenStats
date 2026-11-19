//! Discrete distributions: Binomial, Poisson, Hypergeometric, Negative
//! binomial, Geometric, Discrete uniform (Minitab "Integer") and the general
//! user-specified Discrete distribution.
//!
//! Probability masses use the saddle-point (Loader) formulation
//! `pmf = exp(stirlerr(n) − stirlerr(x) − stirlerr(n−x) − bd0(x, np) − bd0(n−x, nq)) / √(2π x(n−x)/n)`,
//! which keeps full relative accuracy for huge n and far-tail x. The Binomial,
//! Poisson and Negative binomial distribution functions are exact
//! incomplete-beta / incomplete-gamma identities; the hypergeometric one sums
//! the pmf recurrence from the requested point towards the nearer tail, so
//! both tails keep relative accuracy.
//!
//! **Parameterisation (as in Minitab).** The negative binomial and geometric
//! variables count *trials*: `NegativeBinomial(r, p)` has support r, r+1, …
//! and `Geometric(p)` support 1, 2, …. Quantiles follow the usual convention:
//! `quantile(p)` is the smallest k with `cdf(k) ≥ p` and `isf(q)` the smallest
//! k with `sf(k) ≤ q`, each with a 64-ulp fuzz so that round trips of
//! probabilities that were themselves rounded return the same k.

use super::require;
use crate::error::Result;
use crate::special::erf::norm_ppf;
use crate::special::gamma::{lgamma_correction, ln_gamma, LN_SQRT_2PI};
use crate::special::incbeta::beta_inc_pair;
use crate::special::incgamma::gamma_pq;
use std::f64::consts::PI;

/// Interface shared by all discrete distributions. Arguments are `f64` so the
/// calculator can pass worksheet values directly: the mass at a non-integer is
/// zero and distribution functions use ⌊x⌋.
pub trait DiscreteDistribution {
    fn name(&self) -> &'static str;
    /// P(X = x); 0 if x is not in the support.
    fn pmf(&self, x: f64) -> f64;
    /// ln P(X = x); −∞ outside the support.
    fn ln_pmf(&self, x: f64) -> f64 {
        self.pmf(x).ln()
    }
    /// P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;
    /// P(X > x), computed without cancellation.
    fn sf(&self, x: f64) -> f64;
    /// Smallest k with P(X ≤ k) ≥ p.
    fn quantile(&self, p: f64) -> f64;
    /// Smallest k with P(X > k) ≤ q.
    fn isf(&self, q: f64) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// (smallest, largest) value; the upper bound is +∞ if unbounded.
    fn support(&self) -> (f64, f64);
    /// One random variate; the default is inversion of the quantile
    /// function at a uniform on (0, 1).
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.quantile(rng.next_f64_open())
    }
    /// `n` independent variates.
    fn sample_n(&self, rng: &mut dyn crate::random::Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

// ---------------------------------------------------------------------------
// Loader's saddle-point ingredients
// ---------------------------------------------------------------------------

/// Stirling error ln n! − [(n + ½) ln n − n + ½ ln 2π], with stirlerr(0) = 0.
pub(crate) fn stirlerr(n: f64) -> f64 {
    if n >= 10.0 {
        lgamma_correction(n)
    } else if n == 0.0 {
        0.0
    } else {
        ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - LN_SQRT_2PI
    }
}

/// Deviance term x ln(x/np) + np − x given the accurately computed difference
/// `d = x − np` (series when x ≈ np, direct logarithm otherwise).
pub(crate) fn bd0_d(x: f64, np: f64, d: f64) -> f64 {
    if !x.is_finite() || !np.is_finite() || np == 0.0 {
        return f64::NAN;
    }
    if d.abs() < 0.1 * (x + np) {
        let mut v = d / (x + np);
        let mut s = d * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s {
                return s1;
            }
            s = s1;
        }
        s
    } else {
        x * (x / np).ln() + np - x
    }
}

/// Deviance term x ln(x/np) + np − x.
pub(crate) fn bd0(x: f64, np: f64) -> f64 {
    bd0_d(x, np, x - np)
}

/// ln of the binomial mass C(n, x) pˣ qⁿ⁻ˣ, with p and q = 1 − p both supplied
/// accurately. Neither np nor ln q is formed from a rounded complement:
/// x − np uses a fused multiply–add and ln q = ln(1 − p) uses `ln_1p`, so the
/// result is accurate for p as small as the double format allows.
pub(crate) fn ln_dbinom_raw(x: f64, n: f64, p: f64, q: f64) -> f64 {
    if p == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if q == 0.0 {
        return if x == n { 0.0 } else { f64::NEG_INFINITY };
    }
    if x < 0.0 || x > n {
        return f64::NEG_INFINITY;
    }
    let (ln_p, ln_q) = if p < 0.5 { (p.ln(), (-p).ln_1p()) } else { ((-q).ln_1p(), q.ln()) };
    if x == 0.0 {
        return if n == 0.0 { 0.0 } else { n * ln_q };
    }
    if x == n {
        return n * ln_p;
    }
    let d = (-n).mul_add(p, x); // x − np with a single rounding
    let dev = if d.abs() < 0.1 * (x + n * p) {
        bd0_d(x, n * p, d) + bd0_d(n - x, n * q, -d)
    } else {
        x * (x.ln() - n.ln() - ln_p) + (n - x) * ((-x / n).ln_1p() - ln_q)
    };
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x);
    let lf = (2.0 * PI).ln() + x.ln() + (-x / n).ln_1p();
    lc - dev - 0.5 * lf
}

/// ln of the Poisson mass λˣ e^{−λ} / x!.
pub(crate) fn ln_dpois_raw(x: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if x < 0.0 {
        return f64::NEG_INFINITY;
    }
    if x <= lambda * f64::MIN_POSITIVE {
        return -lambda;
    }
    if lambda < x * f64::MIN_POSITIVE {
        return -lambda + x * lambda.ln() - ln_gamma(x + 1.0);
    }
    -stirlerr(x) - bd0(x, lambda) - 0.5 * (2.0 * PI * x).ln()
}

// ---------------------------------------------------------------------------
// Generic quantile search on the integers
// ---------------------------------------------------------------------------

fn is_int(x: f64) -> bool {
    x.is_finite() && x == x.floor()
}

/// Smallest integer k in [lo, hi] with `pred(k)` true, for a monotone
/// predicate (false … false true … true) that holds at `hi`; `hi` may be +∞.
fn smallest_k(lo: f64, hi: f64, guess: f64, pred: impl Fn(f64) -> bool) -> f64 {
    let g = if guess.is_finite() { guess.round().clamp(lo, hi) } else { lo };
    let (mut f, mut t);
    if pred(g) {
        t = g;
        let mut step = 1.0;
        loop {
            if t <= lo {
                return lo;
            }
            let c = (t - step).max(lo);
            if pred(c) {
                t = c;
                step *= 2.0;
            } else {
                f = c;
                break;
            }
        }
    } else {
        f = g;
        let mut step = 1.0;
        let mut iters = 0;
        loop {
            iters += 1;
            let c = (f + step).min(hi);
            if pred(c) {
                t = c;
                break;
            }
            if c >= hi || iters > 2000 {
                return hi;
            }
            f = c;
            step *= 2.0;
        }
    }
    while t - f > 1.0 {
        let mid = ((f + t) / 2.0).floor();
        if pred(mid) {
            t = mid;
        } else {
            f = mid;
        }
    }
    t
}

const FUZZ: f64 = 64.0 * f64::EPSILON;

/// Smallest k with cdf(k) ≥ p. In the upper half the comparison is made on
/// the survival side with the exactly representable q = 1 − p, so that
/// probabilities within a few ulp of 1 keep their full resolution.
fn quantile_search(p: f64, lo: f64, hi: f64, guess: f64, cdf: impl Fn(f64) -> f64, sf: impl Fn(f64) -> f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return lo;
    }
    if p == 1.0 {
        return hi;
    }
    if p > 0.5 {
        let target = (1.0 - p) * (1.0 + FUZZ);
        smallest_k(lo, hi, guess, |k| sf(k) <= target)
    } else {
        let target = p * (1.0 - FUZZ);
        smallest_k(lo, hi, guess, |k| cdf(k) >= target)
    }
}

/// Smallest k with sf(k) ≤ q (see [`quantile_search`] for the tail choice).
fn isf_search(q: f64, lo: f64, hi: f64, guess: f64, cdf: impl Fn(f64) -> f64, sf: impl Fn(f64) -> f64) -> f64 {
    if q.is_nan() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 1.0 {
        return lo;
    }
    if q == 0.0 {
        return hi;
    }
    if q > 0.5 {
        let target = (1.0 - q) * (1.0 - FUZZ);
        smallest_k(lo, hi, guess, |k| cdf(k) >= target)
    } else {
        let target = q * (1.0 + FUZZ);
        smallest_k(lo, hi, guess, |k| sf(k) <= target)
    }
}

/// Cornish–Fisher-free normal estimate of the quantile used as search start.
fn normal_guess(mean: f64, sd: f64, p: f64) -> f64 {
    if p > 0.0 && p < 1.0 && sd.is_finite() {
        mean + sd * norm_ppf(p)
    } else {
        mean
    }
}

// ---------------------------------------------------------------------------
// Binomial
// ---------------------------------------------------------------------------

/// Binomial(n, p): number of events in n independent trials.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Result<Self> {
        require((0.0..=1.0).contains(&p), || format!("Binomial: need 0 ≤ p ≤ 1 (got {p})"))?;
        Ok(Binomial { n, p })
    }
    fn nf(&self) -> f64 {
        self.n as f64
    }
}

impl DiscreteDistribution for Binomial {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::binomial(rng, self.n, self.p) as f64
    }
    fn name(&self) -> &'static str {
        "Binomial"
    }
    fn ln_pmf(&self, x: f64) -> f64 {
        if !is_int(x) || x < 0.0 || x > self.nf() {
            return f64::NEG_INFINITY;
        }
        ln_dbinom_raw(x, self.nf(), self.p, 1.0 - self.p)
    }
    fn pmf(&self, x: f64) -> f64 {
        self.ln_pmf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 {
            return 0.0;
        }
        if x >= self.nf() || self.p == 0.0 {
            return 1.0;
        }
        let k = x.floor();
        if self.p == 1.0 {
            return 0.0;
        }
        beta_inc_pair(self.nf() - k, k + 1.0, 1.0 - self.p, self.p).0
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 {
            return 1.0;
        }
        if x >= self.nf() || self.p == 0.0 {
            return 0.0;
        }
        let k = x.floor();
        if self.p == 1.0 {
            return 1.0;
        }
        beta_inc_pair(k + 1.0, self.nf() - k, self.p, 1.0 - self.p).0
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_search(
            p,
            0.0,
            self.nf(),
            normal_guess(self.mean(), self.std_dev(), p),
            |k| self.cdf(k),
            |k| self.sf(k),
        )
    }
    fn isf(&self, q: f64) -> f64 {
        isf_search(
            q,
            0.0,
            self.nf(),
            normal_guess(self.mean(), self.std_dev(), 1.0 - q),
            |k| self.cdf(k),
            |k| self.sf(k),
        )
    }
    fn mean(&self) -> f64 {
        self.nf() * self.p
    }
    fn variance(&self) -> f64 {
        self.nf() * self.p * (1.0 - self.p)
    }
    fn support(&self) -> (f64, f64) {
        (0.0, self.nf())
    }
}

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------

/// Poisson(λ) with mean λ ≥ 0 (λ = 0 is the point mass at 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    pub mean: f64,
}

impl Poisson {
    pub fn new(mean: f64) -> Result<Self> {
        require(mean >= 0.0 && mean.is_finite(), || format!("Poisson: need mean ≥ 0 (got {mean})"))?;
        Ok(Poisson { mean })
    }
}

impl DiscreteDistribution for Poisson {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::poisson(rng, self.mean) as f64
    }
    fn name(&self) -> &'static str {
        "Poisson"
    }
    fn ln_pmf(&self, x: f64) -> f64 {
        if !is_int(x) || x < 0.0 {
            return f64::NEG_INFINITY;
        }
        ln_dpois_raw(x, self.mean)
    }
    fn pmf(&self, x: f64) -> f64 {
        self.ln_pmf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 {
            return 0.0;
        }
        if self.mean == 0.0 || x == f64::INFINITY {
            return 1.0;
        }
        gamma_pq(x.floor() + 1.0, self.mean).1
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 {
            return 1.0;
        }
        if self.mean == 0.0 || x == f64::INFINITY {
            return 0.0;
        }
        gamma_pq(x.floor() + 1.0, self.mean).0
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_search(
            p,
            0.0,
            f64::INFINITY,
            normal_guess(self.mean, self.mean.sqrt(), p),
            |k| self.cdf(k),
            |k| self.sf(k),
        )
    }
    fn isf(&self, q: f64) -> f64 {
        isf_search(
            q,
            0.0,
            f64::INFINITY,
            normal_guess(self.mean, self.mean.sqrt(), 1.0 - q),
            |k| self.cdf(k),
            |k| self.sf(k),
        )
    }
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.mean
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

// ---------------------------------------------------------------------------
// Hypergeometric
// ---------------------------------------------------------------------------

/// Hypergeometric: `draws` items sampled without replacement from a population
/// of `population` items that contains `successes` events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hypergeometric {
    pub population: u64,
    pub successes: u64,
    pub draws: u64,
}

impl Hypergeometric {
    pub fn new(population: u64, successes: u64, draws: u64) -> Result<Self> {
        require(population >= 1, || "Hypergeometric: population size must be at least 1".to_string())?;
        require(successes <= population, || {
            format!("Hypergeometric: events {successes} exceed population {population}")
        })?;
        require(draws <= population, || format!("Hypergeometric: sample {draws} exceeds population {population}"))?;
        Ok(Hypergeometric { population, successes, draws })
    }
    fn big_n(&self) -> f64 {
        self.population as f64
    }
    fn big_k(&self) -> f64 {
        self.successes as f64
    }
    fn n(&self) -> f64 {
        self.draws as f64
    }
    fn lo(&self) -> f64 {
        (self.n() + self.big_k() - self.big_n()).max(0.0)
    }
    fn hi(&self) -> f64 {
        self.n().min(self.big_k())
    }
}

impl DiscreteDistribution for Hypergeometric {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::hypergeometric(rng, self.population, self.successes, self.draws) as f64
    }
    fn name(&self) -> &'static str {
        "Hypergeometric"
    }
    fn ln_pmf(&self, x: f64) -> f64 {
        if !is_int(x) || x < self.lo() || x > self.hi() {
            return f64::NEG_INFINITY;
        }
        let (nn, kk, n) = (self.big_n(), self.big_k(), self.n());
        let p = n / nn;
        let q = (nn - n) / nn;
        ln_dbinom_raw(x, kk, p, q) + ln_dbinom_raw(n - x, nn - kk, p, q) - ln_dbinom_raw(n, nn, p, q)
    }
    fn pmf(&self, x: f64) -> f64 {
        self.ln_pmf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.lo() {
            return 0.0;
        }
        if x >= self.hi() {
            return 1.0;
        }
        let k = x.floor();
        if k <= self.mean() {
            self.lower_sum(k)
        } else {
            1.0 - self.upper_sum(k + 1.0)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.lo() {
            return 1.0;
        }
        if x >= self.hi() {
            return 0.0;
        }
        let k = x.floor();
        if k <= self.mean() {
            1.0 - self.lower_sum(k)
        } else {
            self.upper_sum(k + 1.0)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        let g = normal_guess(self.mean(), self.std_dev(), p);
        quantile_search(p, self.lo(), self.hi(), g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn isf(&self, q: f64) -> f64 {
        let g = normal_guess(self.mean(), self.std_dev(), 1.0 - q);
        isf_search(q, self.lo(), self.hi(), g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn mean(&self) -> f64 {
        self.n() * self.big_k() / self.big_n()
    }
    fn variance(&self) -> f64 {
        let (nn, kk, n) = (self.big_n(), self.big_k(), self.n());
        if nn <= 1.0 {
            return 0.0;
        }
        n * (kk / nn) * (1.0 - kk / nn) * (nn - n) / (nn - 1.0)
    }
    fn support(&self) -> (f64, f64) {
        (self.lo(), self.hi())
    }
}

impl Hypergeometric {
    /// P(X ≤ k): pmf(k) · (1 + r_k + r_k r_{k−1} + …), r_j = pmf(j−1)/pmf(j).
    fn lower_sum(&self, k: f64) -> f64 {
        let (nn, kk, n, lo) = (self.big_n(), self.big_k(), self.n(), self.lo());
        let mut term = self.pmf(k);
        let mut sum = term;
        let mut j = k;
        while j > lo {
            term *= j * (nn - kk - n + j) / ((kk - j + 1.0) * (n - j + 1.0));
            sum += term;
            j -= 1.0;
            if term < 1e-17 * sum {
                break;
            }
        }
        sum
    }
    /// P(X ≥ k): pmf(k) · (1 + u_k + u_k u_{k+1} + …), u_j = pmf(j+1)/pmf(j).
    fn upper_sum(&self, k: f64) -> f64 {
        let (nn, kk, n, hi) = (self.big_n(), self.big_k(), self.n(), self.hi());
        let mut term = self.pmf(k);
        let mut sum = term;
        let mut j = k;
        while j < hi {
            term *= (kk - j) * (n - j) / ((j + 1.0) * (nn - kk - n + j + 1.0));
            sum += term;
            j += 1.0;
            if term < 1e-17 * sum {
                break;
            }
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Negative binomial and geometric (trial-counting forms)
// ---------------------------------------------------------------------------

/// Negative binomial: number of trials needed to observe `events` events when
/// each trial succeeds with probability p. Support r, r+1, ….
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomial {
    pub events: u64,
    pub p: f64,
}

impl NegativeBinomial {
    pub fn new(events: u64, p: f64) -> Result<Self> {
        require(events >= 1, || "NegativeBinomial: number of events must be at least 1".to_string())?;
        require(p > 0.0 && p <= 1.0, || format!("NegativeBinomial: need 0 < p ≤ 1 (got {p})"))?;
        Ok(NegativeBinomial { events, p })
    }
    fn r(&self) -> f64 {
        self.events as f64
    }
}

impl DiscreteDistribution for NegativeBinomial {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::negative_binomial(rng, self.events, self.p) as f64
    }
    fn name(&self) -> &'static str {
        "NegativeBinomial"
    }
    fn ln_pmf(&self, x: f64) -> f64 {
        if !is_int(x) || x < self.r() {
            return f64::NEG_INFINITY;
        }
        // C(x−1, r−1) pʳ qˣ⁻ʳ = (r/x) · C(x, r) pʳ qˣ⁻ʳ
        (self.r() / x).ln() + ln_dbinom_raw(self.r(), x, self.p, 1.0 - self.p)
    }
    fn pmf(&self, x: f64) -> f64 {
        self.ln_pmf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.r() {
            return 0.0;
        }
        if self.p == 1.0 || x == f64::INFINITY {
            return 1.0;
        }
        let failures = x.floor() - self.r();
        beta_inc_pair(self.r(), failures + 1.0, self.p, 1.0 - self.p).0
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.r() {
            return 1.0;
        }
        if self.p == 1.0 || x == f64::INFINITY {
            return 0.0;
        }
        let failures = x.floor() - self.r();
        beta_inc_pair(self.r(), failures + 1.0, self.p, 1.0 - self.p).1
    }
    fn quantile(&self, p: f64) -> f64 {
        let g = normal_guess(self.mean(), self.std_dev(), p);
        quantile_search(p, self.r(), f64::INFINITY, g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn isf(&self, q: f64) -> f64 {
        let g = normal_guess(self.mean(), self.std_dev(), 1.0 - q);
        isf_search(q, self.r(), f64::INFINITY, g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn mean(&self) -> f64 {
        self.r() / self.p
    }
    fn variance(&self) -> f64 {
        self.r() * (1.0 - self.p) / (self.p * self.p)
    }
    fn support(&self) -> (f64, f64) {
        (self.r(), f64::INFINITY)
    }
}

/// Geometric: number of trials until the first event. Support 1, 2, ….
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometric {
    pub p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Result<Self> {
        require(p > 0.0 && p <= 1.0, || format!("Geometric: need 0 < p ≤ 1 (got {p})"))?;
        Ok(Geometric { p })
    }
}

impl DiscreteDistribution for Geometric {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::geometric(rng, self.p) as f64
    }
    fn name(&self) -> &'static str {
        "Geometric"
    }
    fn ln_pmf(&self, x: f64) -> f64 {
        if !is_int(x) || x < 1.0 {
            return f64::NEG_INFINITY;
        }
        if self.p == 1.0 {
            return if x == 1.0 { 0.0 } else { f64::NEG_INFINITY };
        }
        self.p.ln() + (x - 1.0) * (-self.p).ln_1p()
    }
    fn pmf(&self, x: f64) -> f64 {
        self.ln_pmf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 1.0 {
            return 0.0;
        }
        if self.p == 1.0 || x == f64::INFINITY {
            return 1.0;
        }
        -(x.floor() * (-self.p).ln_1p()).exp_m1()
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 1.0 {
            return 1.0;
        }
        if self.p == 1.0 || x == f64::INFINITY {
            return 0.0;
        }
        let k = x.floor();
        let q = 1.0 - self.p;
        // When 1 − p is exact (p = ½, ¼, …) a power is exact; otherwise the
        // exponent form keeps the error at the conditioning limit k·ε.
        if 1.0 - q == self.p {
            q.powf(k)
        } else {
            (k * (-self.p).ln_1p()).exp()
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        let g = if self.p < 1.0 && p < 1.0 { ((-p).ln_1p() / (-self.p).ln_1p()).ceil() } else { 1.0 };
        quantile_search(p, 1.0, f64::INFINITY, g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn isf(&self, q: f64) -> f64 {
        let g = if self.p < 1.0 && q > 0.0 { (q.ln() / (-self.p).ln_1p()).ceil() } else { 1.0 };
        isf_search(q, 1.0, f64::INFINITY, g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
    fn support(&self) -> (f64, f64) {
        (1.0, f64::INFINITY)
    }
}

// ---------------------------------------------------------------------------
// Discrete uniform (Minitab "Integer") and general Discrete
// ---------------------------------------------------------------------------

/// Uniform distribution on the integers a, a+1, …, b (Minitab's *Integer*).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscreteUniform {
    pub a: i64,
    pub b: i64,
}

/// Minitab's name for [`DiscreteUniform`].
pub type Integer = DiscreteUniform;

impl DiscreteUniform {
    pub fn new(a: i64, b: i64) -> Result<Self> {
        require(a <= b, || format!("DiscreteUniform: need minimum ≤ maximum (got {a}, {b})"))?;
        Ok(DiscreteUniform { a, b })
    }
    fn count(&self) -> f64 {
        (self.b as f64) - (self.a as f64) + 1.0
    }
}

impl DiscreteDistribution for DiscreteUniform {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::uniform_int(rng, self.a, self.b) as f64
    }
    fn name(&self) -> &'static str {
        "DiscreteUniform"
    }
    fn pmf(&self, x: f64) -> f64 {
        if is_int(x) && x >= self.a as f64 && x <= self.b as f64 {
            1.0 / self.count()
        } else {
            0.0
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.a as f64 {
            0.0
        } else if x >= self.b as f64 {
            1.0
        } else {
            (x.floor() - self.a as f64 + 1.0) / self.count()
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < self.a as f64 {
            1.0
        } else if x >= self.b as f64 {
            0.0
        } else {
            (self.b as f64 - x.floor()) / self.count()
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        let g = self.a as f64 + p * self.count();
        quantile_search(p, self.a as f64, self.b as f64, g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn isf(&self, q: f64) -> f64 {
        let g = self.b as f64 - q * self.count();
        isf_search(q, self.a as f64, self.b as f64, g, |k| self.cdf(k), |k| self.sf(k))
    }
    fn mean(&self) -> f64 {
        0.5 * (self.a as f64 + self.b as f64)
    }
    fn variance(&self) -> f64 {
        let n = self.count();
        (n * n - 1.0) / 12.0
    }
    fn support(&self) -> (f64, f64) {
        (self.a as f64, self.b as f64)
    }
}

/// Distribution on finitely many user-specified values (Minitab *Discrete*).
/// Duplicate values are merged; probabilities must be non-negative and sum to
/// 1 within 1e-9 (they are then renormalised exactly).
#[derive(Debug, Clone, PartialEq)]
pub struct Discrete {
    values: Vec<f64>,
    probs: Vec<f64>,
    /// prefix[i] = P(X ≤ values[i]); suffix[i] = P(X ≥ values[i]).
    prefix: Vec<f64>,
    suffix: Vec<f64>,
}

impl Discrete {
    /// Inversion by binary search on the cumulative probabilities.
    pub(crate) fn draw(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        let u = rng.next_f64();
        let i = self.prefix.partition_point(|&c| c <= u);
        self.values[i.min(self.values.len() - 1)]
    }

    pub fn new(values: &[f64], probs: &[f64]) -> Result<Self> {
        require(!values.is_empty() && values.len() == probs.len(), || {
            "Discrete: need the same non-zero number of values and probabilities".to_string()
        })?;
        require(values.iter().all(|v| v.is_finite()), || "Discrete: values must be finite".to_string())?;
        require(probs.iter().all(|p| *p >= 0.0 && p.is_finite()), || {
            "Discrete: probabilities must be finite and non-negative".to_string()
        })?;
        let total: f64 = probs.iter().sum();
        require((total - 1.0).abs() <= 1e-9, || format!("Discrete: probabilities sum to {total}, not 1"))?;
        let mut pairs: Vec<(f64, f64)> = values.iter().copied().zip(probs.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut vs: Vec<f64> = Vec::new();
        let mut ps: Vec<f64> = Vec::new();
        for (v, p) in pairs {
            let v = if v == 0.0 { 0.0 } else { v };
            if vs.last() == Some(&v) {
                *ps.last_mut().expect("non-empty") += p;
            } else {
                vs.push(v);
                ps.push(p);
            }
        }
        // Normalise exactly (the input may sum to 1 only within 1e-9).
        let sum: f64 = ps.iter().sum();
        for p in &mut ps {
            *p /= sum;
        }
        let mut prefix = Vec::with_capacity(vs.len());
        let mut acc = 0.0;
        for p in &ps {
            acc += p;
            prefix.push(acc);
        }
        let mut suffix = vec![0.0; vs.len()];
        let mut acc = 0.0;
        for i in (0..vs.len()).rev() {
            acc += ps[i];
            suffix[i] = acc;
        }
        // Totals are exactly 1 by definition (guards against rounding in the running sums).
        if let Some(last) = prefix.last_mut() {
            *last = 1.0;
        }
        suffix[0] = 1.0;
        Ok(Discrete { values: vs, probs: ps, prefix, suffix })
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probs
    }
    /// Index of the last value ≤ x.
    fn upper_index(&self, x: f64) -> Option<usize> {
        let n = self.values.partition_point(|&v| v <= x);
        n.checked_sub(1)
    }
}

impl DiscreteDistribution for Discrete {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.draw(rng)
    }
    fn name(&self) -> &'static str {
        "Discrete"
    }
    fn pmf(&self, x: f64) -> f64 {
        match self.values.binary_search_by(|v| v.total_cmp(&x)) {
            Ok(i) => self.probs[i],
            Err(_) => 0.0,
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        self.upper_index(x).map_or(0.0, |i| self.prefix[i].min(1.0))
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        match self.upper_index(x) {
            None => 1.0,
            Some(i) if i + 1 < self.values.len() => self.suffix[i + 1],
            Some(_) => 0.0,
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        let target = p * (1.0 - FUZZ);
        let i = self.prefix.partition_point(|&c| c < target);
        self.values[i.min(self.values.len() - 1)]
    }
    fn isf(&self, q: f64) -> f64 {
        if q.is_nan() || !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        // Smallest value v with P(X > v) ≤ q ⇔ suffix[i+1] ≤ q (i the index of v).
        let target = q * (1.0 + FUZZ);
        for i in 0..self.values.len() {
            let tail = if i + 1 < self.values.len() { self.suffix[i + 1] } else { 0.0 };
            if tail <= target {
                return self.values[i];
            }
        }
        self.values[self.values.len() - 1]
    }
    fn mean(&self) -> f64 {
        self.values.iter().zip(&self.probs).map(|(v, p)| v * p).sum()
    }
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.values.iter().zip(&self.probs).map(|(v, p)| p * (v - m) * (v - m)).sum()
    }
    fn support(&self) -> (f64, f64) {
        (self.values[0], self.values[self.values.len() - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact binomial coefficient in integer arithmetic.
    fn choose(n: u128, k: u128) -> u128 {
        let k = k.min(n - k);
        let mut r: u128 = 1;
        for i in 0..k {
            r = r * (n - i) / (i + 1);
        }
        r
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        a == b || (a - b).abs() <= rel * a.abs().max(b.abs())
    }

    /// Consistency checks every discrete distribution must satisfy over its
    /// first `limit` support points.
    fn check<D: DiscreteDistribution>(d: &D, limit: usize) {
        let (lo, hi) = d.support();
        let mut cum = 0.0;
        let mut k = lo;
        let mut steps = 0;
        let (mut m1, mut m2) = (0.0, 0.0);
        while k <= hi && steps < limit {
            let pm = d.pmf(k);
            assert!((0.0..=1.0).contains(&pm), "{}: pmf({k}) = {pm}", d.name());
            assert!(close(d.ln_pmf(k), pm.ln(), 1e-12) || pm < 1e-300, "{}: ln_pmf({k})", d.name());
            cum += pm;
            m1 += k * pm;
            m2 += k * k * pm;
            let (c, s) = (d.cdf(k), d.sf(k));
            assert!((c - cum).abs() < 1e-13, "{}: cdf({k}) = {c}, running sum {cum}", d.name());
            assert!((c + s - 1.0).abs() < 1e-13, "{}: cdf + sf at {k}", d.name());
            // Quantile round trips are exact for points of non-negligible mass.
            // (Only for c ≤ ½: above that, p = cdf(k) rounds away the survival
            // probability that decides the quantile, so no round trip exists.)
            if pm > 1e-11 * c.max(1e-300) && c <= 0.5 && pm > 1e-13 {
                assert_eq!(d.quantile(c), k, "{}: quantile(cdf({k}))", d.name());
            }
            if pm > 1e-11 * s.max(1e-300) && s <= 0.5 && s > 1e-300 && pm > 1e-13 {
                assert_eq!(d.isf(s), k, "{}: isf(sf({k}))", d.name());
            }
            // Non-integers: no mass; cdf is a step function.
            if k + 0.5 <= hi {
                assert_eq!(d.pmf(k + 0.5), 0.0);
                assert_eq!(d.cdf(k + 0.5), c);
                assert_eq!(d.sf(k + 0.5), s);
            }
            k += 1.0;
            steps += 1;
        }
        // Below / above the support.
        assert_eq!(d.cdf(lo - 1.0), 0.0);
        assert_eq!(d.sf(lo - 1.0), 1.0);
        assert_eq!(d.pmf(lo - 1.0), 0.0);
        assert_eq!(d.ln_pmf(lo - 1.0), f64::NEG_INFINITY);
        assert!(d.quantile(f64::NAN).is_nan() && d.quantile(1.5).is_nan() && d.isf(-0.1).is_nan());
        assert_eq!(d.quantile(0.0), lo);
        assert_eq!(d.isf(1.0), lo);
        assert_eq!(d.quantile(1.0), hi);
        assert_eq!(d.isf(0.0), hi);
        if hi.is_finite() && steps as f64 > hi - lo {
            assert!((cum - 1.0).abs() < 1e-13, "{}: pmf sums to {cum}", d.name());
            assert!(
                close(m1, d.mean(), 1e-11) || (m1 - d.mean()).abs() < 1e-11,
                "{}: mean {m1} vs {}",
                d.name(),
                d.mean()
            );
            assert!(
                close(m2 - m1 * m1, d.variance(), 1e-9) || (m2 - m1 * m1 - d.variance()).abs() < 1e-9,
                "{}: variance",
                d.name()
            );
            assert_eq!(d.cdf(hi), 1.0);
            assert_eq!(d.sf(hi), 0.0);
        }
    }

    #[test]
    fn saddle_point_ingredients() {
        // stirlerr(n) = ln n! − (n + ½) ln n + n − ½ ln 2π; known: stirlerr(1) = 1 − ½ln(2π) = 0.0810614667953272…
        assert!((stirlerr(1.0) - 0.081_061_466_795_327_26).abs() < 1e-15);
        assert_eq!(stirlerr(0.0), 0.0);
        // The direct formula (used below 10) agrees with the asymptotic series at the switch point.
        let direct = ln_gamma(11.0) - 10.5 * 10f64.ln() + 10.0 - LN_SQRT_2PI;
        assert!((direct - stirlerr(10.0)).abs() < 5e-15);
        // bd0 series branch agrees with the direct formula where both are accurate.
        let (x, np) = (10.0, 11.0);
        assert!((bd0(x, np) - (x * (x / np).ln() + np - x)).abs() < 1e-14);
        assert!(bd0(1e6, 1e6 + 1.0) > 0.0);
        // Binomial mass of Binomial(1, p).
        assert!((ln_dbinom_raw(1.0, 1.0, 0.3, 0.7) - 0.3f64.ln()).abs() < 1e-15);
        assert!((ln_dpois_raw(0.0, 2.5) + 2.5).abs() < 1e-16);
    }

    #[test]
    fn binomial_exact() {
        let d = Binomial::new(10, 0.3).unwrap();
        let pm = |k: u128| choose(10, k) as f64 * 0.3f64.powi(k as i32) * 0.7f64.powi(10 - k as i32);
        for k in 0..=10 {
            assert!(close(d.pmf(k as f64), pm(k), 4e-15), "k = {k}");
        }
        assert!(close(d.cdf(3.0), (0..=3).map(pm).sum::<f64>(), 4e-15));
        assert!(close(d.sf(3.0), (4..=10).map(pm).sum::<f64>(), 4e-15));
        assert_eq!(d.quantile(0.5), 3.0);
        assert_eq!(d.quantile(0.65), 4.0);
        assert_eq!(d.mean(), 3.0);
        assert!((d.variance() - 2.1).abs() < 1e-15);
        check(&d, 100);
        check(&Binomial::new(200, 0.01).unwrap(), 300);
        check(&Binomial::new(30, 0.97).unwrap(), 100);
        check(&Binomial::new(1, 0.5).unwrap(), 10);
        check(&Binomial::new(0, 0.5).unwrap(), 10);
        // Degenerate p.
        let z = Binomial::new(5, 0.0).unwrap();
        assert_eq!((z.pmf(0.0), z.pmf(1.0), z.cdf(0.0), z.sf(0.0), z.quantile(0.3)), (1.0, 0.0, 1.0, 0.0, 0.0));
        let o = Binomial::new(5, 1.0).unwrap();
        assert_eq!((o.pmf(5.0), o.pmf(4.0), o.cdf(4.0), o.sf(4.0), o.quantile(0.3)), (1.0, 0.0, 0.0, 1.0, 5.0));
        assert!(Binomial::new(5, 1.1).is_err() && Binomial::new(5, f64::NAN).is_err());
    }

    #[test]
    fn binomial_large_n() {
        // Central term of Binomial(2m, ½): C(2m, m)/4^m = 1/√(πm) · (1 − 1/(8m) + 1/(128m²) + …)
        let m = 500_000.0f64;
        let d = Binomial::new(1_000_000, 0.5).unwrap();
        let want = 1.0 / (PI * m).sqrt() * (1.0 - 1.0 / (8.0 * m) + 1.0 / (128.0 * m * m));
        assert!(close(d.pmf(m), want, 1e-14));
        // Far tail with full relative accuracy: P(X = 0) = (1 − p)ⁿ.
        let d = Binomial::new(1_000_000_000, 1e-9).unwrap();
        assert!(close(d.pmf(0.0), (1e9 * (-1e-9f64).ln_1p()).exp(), 1e-14));
        // Poisson limit.
        let pois = Poisson::new(1.0).unwrap();
        for k in 0..8 {
            assert!(close(d.pmf(k as f64), pois.pmf(k as f64), 1e-6));
        }
        assert!(close(d.cdf(5.0), pois.cdf(5.0), 1e-6));
        // Quantiles of a large binomial are consistent with the cdf.
        let d = Binomial::new(10_000_000_000, 0.37).unwrap();
        let q = d.quantile(0.975);
        assert!(d.cdf(q) >= 0.975 * (1.0 - 1e-13) && d.cdf(q - 1.0) < 0.975 * (1.0 + 1e-13));
        let q = d.isf(1e-30);
        assert!(d.sf(q) <= 1e-30 * (1.0 + 1e-12) && d.sf(q - 1.0) > 1e-30);
    }

    #[test]
    fn poisson_exact() {
        let d = Poisson::new(2.0).unwrap();
        let e2 = (-2.0f64).exp();
        assert!(close(d.pmf(0.0), e2, 2e-16));
        assert!(close(d.pmf(3.0), e2 * 8.0 / 6.0, 4e-16));
        assert!(close(d.cdf(2.0), 5.0 * e2, 4e-16));
        assert!(close(d.sf(2.0), 1.0 - 5.0 * e2, 4e-16));
        assert_eq!(d.quantile(0.5), 2.0);
        check(&d, 80);
        check(&Poisson::new(0.01).unwrap(), 20);
        check(&Poisson::new(50.0).unwrap(), 200);
        check(&Poisson::new(1000.0).unwrap(), 1200);
        let z = Poisson::new(0.0).unwrap();
        assert_eq!((z.pmf(0.0), z.pmf(1.0), z.cdf(0.0), z.sf(0.0)), (1.0, 0.0, 1.0, 0.0));
        assert_eq!((z.quantile(0.5), z.variance()), (0.0, 0.0));
        assert!(Poisson::new(-1.0).is_err() && Poisson::new(f64::INFINITY).is_err());
        // Large mean: Stirling-accurate mass, tail relative accuracy.
        let big = Poisson::new(1e6).unwrap();
        assert!(close(big.pmf(1e6), 1.0 / (2.0 * PI * 1e6).sqrt() * (1.0 - 1.0 / 12e6), 1e-12));
        let t = Poisson::new(10.0).unwrap();
        // P(X ≥ 100) ≈ pmf(100) (1 + λ/101 + …) — compare the sf against its own series.
        let mut series = 0.0;
        for k in 100..400 {
            series += t.pmf(k as f64);
        }
        assert!(close(t.sf(99.0), series, 1e-13));
    }

    #[test]
    fn hypergeometric_exact() {
        let d = Hypergeometric::new(20, 7, 5).unwrap();
        let pm = |k: u128| choose(7, k) as f64 * choose(13, 5 - k) as f64 / choose(20, 5) as f64;
        for k in 0..=5 {
            assert!(close(d.pmf(k as f64), pm(k), 1e-14), "k = {k}");
        }
        assert!(close(d.cdf(2.0), (0..=2).map(pm).sum::<f64>(), 1e-14));
        assert!(close(d.sf(2.0), (3..=5).map(pm).sum::<f64>(), 1e-14));
        assert!((d.mean() - 1.75).abs() < 1e-15);
        // n (K/N)(1 − K/N)(N − n)/(N − 1)
        assert!((d.variance() - 5.0 * 0.35 * 0.65 * 15.0 / 19.0).abs() < 1e-14);
        check(&d, 10);
        check(&Hypergeometric::new(100, 30, 60).unwrap(), 70); // support starts at 0 + (60+30−100 < 0) → 0; hi 30
        check(&Hypergeometric::new(50, 40, 30).unwrap(), 40); // lower bound 20
        check(&Hypergeometric::new(10, 10, 4).unwrap(), 10); // degenerate at 4
        check(&Hypergeometric::new(10, 0, 4).unwrap(), 10); // degenerate at 0
        check(&Hypergeometric::new(1, 1, 1).unwrap(), 5);
        assert_eq!(Hypergeometric::new(50, 40, 30).unwrap().support(), (20.0, 30.0));
        assert!(Hypergeometric::new(10, 11, 3).is_err() && Hypergeometric::new(10, 5, 11).is_err());
        assert!(Hypergeometric::new(0, 0, 0).is_err());
        // One draw is a Bernoulli(K/N).
        let b = Hypergeometric::new(37, 11, 1).unwrap();
        assert!(close(b.pmf(1.0), 11.0 / 37.0, 1e-15) && close(b.pmf(0.0), 26.0 / 37.0, 1e-15));
        // Large population approaches the binomial; far-tail relative accuracy vs exact ratio.
        let h = Hypergeometric::new(1_000_000_000, 300_000_000, 40).unwrap();
        let bin = Binomial::new(40, 0.3).unwrap();
        for k in [5.0, 12.0, 20.0, 30.0] {
            assert!(close(h.pmf(k), bin.pmf(k), 1e-4), "k = {k}");
        }
        // Tail: P(X ≥ 30) from both directions agrees.
        let d = Hypergeometric::new(200, 100, 100).unwrap();
        let tail: f64 = (60..=100).map(|k| d.pmf(k as f64)).sum();
        assert!(close(d.sf(59.0), tail, 1e-13));
        let low: f64 = (0..=20).map(|k| d.pmf(k as f64)).sum();
        assert!(close(d.cdf(20.0), low, 1e-13));
        assert!(d.cdf(20.0) < 1e-12 && d.cdf(20.0) > 0.0, "relative accuracy needs a sub-1e-12 tail");
    }

    #[test]
    fn negative_binomial_and_geometric() {
        let d = NegativeBinomial::new(3, 0.5).unwrap();
        assert!(close(d.pmf(5.0), 6.0 / 32.0, 1e-15));
        assert_eq!(d.pmf(2.0), 0.0);
        assert_eq!(d.support(), (3.0, f64::INFINITY));
        assert_eq!((d.mean(), d.variance()), (6.0, 6.0));
        // cdf(5) = 1/8 + 3/16 + 6/32 = 0.5
        assert!(close(d.cdf(5.0), 0.5, 1e-15) && close(d.sf(5.0), 0.5, 1e-15));
        check(&d, 200);
        check(&NegativeBinomial::new(10, 0.02).unwrap(), 2000);
        check(&NegativeBinomial::new(1, 0.9).unwrap(), 50);
        let one = NegativeBinomial::new(4, 1.0).unwrap();
        assert_eq!((one.pmf(4.0), one.pmf(5.0), one.cdf(4.0), one.quantile(0.7)), (1.0, 0.0, 1.0, 4.0));
        assert!(NegativeBinomial::new(0, 0.5).is_err() && NegativeBinomial::new(2, 0.0).is_err());

        let g = Geometric::new(0.25).unwrap();
        assert!(close(g.pmf(3.0), 0.140_625, 1e-15) && close(g.cdf(3.0), 0.578_125, 1e-15));
        assert!(close(g.sf(3.0), 0.421_875, 1e-15));
        assert_eq!((g.quantile(0.5), g.isf(0.5)), (3.0, 3.0));
        assert_eq!((g.mean(), g.variance()), (4.0, 12.0));
        check(&g, 200);
        check(&Geometric::new(1e-3).unwrap(), 3000);
        let one = Geometric::new(1.0).unwrap();
        assert_eq!(
            (one.pmf(1.0), one.pmf(2.0), one.cdf(1.0), one.sf(1.0), one.quantile(0.2)),
            (1.0, 0.0, 1.0, 0.0, 1.0)
        );
        assert!(Geometric::new(0.0).is_err() && Geometric::new(1.5).is_err());
        // NB(1, p) is Geometric(p) exactly.
        for &p in &[0.01, 0.3, 0.77] {
            let (nb, g) = (NegativeBinomial::new(1, p).unwrap(), Geometric::new(p).unwrap());
            for k in 1..40 {
                let k = k as f64;
                assert!(
                    close(nb.pmf(k), g.pmf(k), 1e-13)
                        && close(nb.cdf(k), g.cdf(k), 1e-13)
                        && close(nb.sf(k), g.sf(k), 1e-13)
                );
            }
            for &q in &[1e-10, 0.1, 0.5, 0.9, 0.999] {
                assert_eq!(nb.quantile(q), g.quantile(q));
                assert_eq!(nb.isf(q), g.isf(q));
            }
        }
        // Extreme tail: P(X > k) = q^k exactly for the geometric.
        let g = Geometric::new(0.5).unwrap();
        // (conditioning: the exponent 1000·ln ½ carries a relative error ~ 1000 ε)
        assert!(close(g.sf(1000.0), 0.5f64.powi(1000), 1e-12));
        assert_eq!(g.isf(0.5f64.powi(1000)), 1000.0);
        // Memorylessness of NB sums: Σ_{r} … simple check — NB(r, p) mean from the mass function.
        let nb = NegativeBinomial::new(5, 0.3).unwrap();
        let (mut s0, mut s1) = (0.0, 0.0);
        for k in 5..4000 {
            let pm = nb.pmf(k as f64);
            s0 += pm;
            s1 += k as f64 * pm;
        }
        assert!((s0 - 1.0).abs() < 1e-13 && close(s1, nb.mean(), 1e-12));
    }

    #[test]
    fn uniform_and_general_discrete() {
        let d = DiscreteUniform::new(3, 8).unwrap();
        assert!(close(d.pmf(5.0), 1.0 / 6.0, 1e-16));
        assert_eq!((d.cdf(5.0), d.sf(5.0)), (0.5, 0.5));
        assert_eq!((d.mean(), d.support()), (5.5, (3.0, 8.0)));
        assert!((d.variance() - 35.0 / 12.0).abs() < 1e-15);
        // p·n rounding must not push the quantile up: 0.3 · 10 = 3.0000000000000004.
        let u = Integer::new(1, 10).unwrap();
        assert_eq!(u.quantile(0.3), 3.0);
        assert_eq!(u.quantile(0.30000000000000004), 3.0);
        assert_eq!(u.isf(0.7), 3.0);
        check(&d, 20);
        check(&Integer::new(-4, -4).unwrap(), 5);
        check(&Integer::new(-7, 12).unwrap(), 30);
        assert!(DiscreteUniform::new(5, 4).is_err());

        let v = Discrete::new(&[10.0, 2.0, 5.0, 2.0], &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(v.values(), &[2.0, 5.0, 10.0]);
        assert!(close(v.pmf(2.0), 0.6, 1e-15) && v.pmf(3.0) == 0.0);
        assert!(close(v.cdf(7.0), 0.9, 1e-15) && close(v.sf(7.0), 0.1, 1e-15));
        assert_eq!((v.cdf(1.0), v.sf(1.0), v.cdf(10.0), v.sf(10.0)), (0.0, 1.0, 1.0, 0.0));
        assert_eq!((v.quantile(0.6), v.quantile(0.61), v.quantile(0.0), v.quantile(1.0)), (2.0, 5.0, 2.0, 10.0));
        assert_eq!((v.isf(0.4), v.isf(0.39), v.isf(0.1), v.isf(0.0), v.isf(1.0)), (2.0, 5.0, 5.0, 10.0, 2.0));
        assert!((v.mean() - (1.2 + 1.5 + 1.0)).abs() < 1e-15);
        assert_eq!(v.support(), (2.0, 10.0));
        // A fair die as Discrete equals the Integer(1,6) distribution.
        let die = Discrete::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1.0 / 6.0; 6]).unwrap();
        let u = Integer::new(1, 6).unwrap();
        for k in 0..8 {
            let k = k as f64;
            assert!(close(die.cdf(k), u.cdf(k), 1e-15) && close(die.sf(k), u.sf(k), 1e-15));
        }
        assert!(close(die.variance(), u.variance(), 1e-14));
        for &p in &[0.1, 0.5, 0.5000000000000001, 0.9] {
            assert_eq!(die.quantile(p), u.quantile(p));
        }
        assert!(Discrete::new(&[1.0], &[0.5]).is_err());
        assert!(Discrete::new(&[1.0, 2.0], &[1.5, -0.5]).is_err());
        assert!(Discrete::new(&[], &[]).is_err());
        assert!(Discrete::new(&[1.0], &[0.5, 0.5]).is_err());
        assert!(Discrete::new(&[f64::NAN], &[1.0]).is_err());
    }
}
