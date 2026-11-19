//! Power and sample-size engine.
//!
//! Every design implements [`PowerModel`]: a power function `power(n, effect)`
//! plus the metadata needed to solve for *any one* of sample size, power or
//! effect size given the other two ([`PowerModel::solve`]).
//!
//! | Design | Struct | Power computed from |
//! |---|---|---|
//! | 1-sample Z | [`OneSampleZ`] | normal |
//! | 1-sample t, paired t, 2-sample t | [`OneSampleT`], [`PairedT`], [`TwoSampleT`] | non-central t |
//! | 1 proportion | [`OneProportion`] | normal approximation or exact binomial |
//! | 2 proportions | [`TwoProportions`] | normal approximation or Fisher's exact test |
//! | 1- / 2-sample Poisson | [`OnePoisson`], [`TwoPoisson`] | normal approximation or exact (conditional) test |
//! | 1 variance, 2 variances | [`OneVariance`], [`TwoVariances`] | central χ² / F |
//! | one-way ANOVA | [`OneWayAnova`] | non-central F |
//! | 2-level factorial, Plackett–Burman | [`Factorial`] | non-central t of the effect contrast |
//! | equivalence (TOST) | [`Tost`] | exact: the joint non-central t probability, integrated over the χ² variable |
//!
//! **Sample size** is always the number of observations *per group* (or
//! replicates of the base design for factorials; observation units of the
//! given length for Poisson). It is the smallest integer whose power reaches
//! the target. Continuous power functions are increasing in *n*, so a
//! doubling-then-bisection search is exact; for the exact discrete tests
//! (binomial, Fisher, Poisson) power is saw-toothed in *n*, so every *n* is
//! tried in turn and the first that reaches the target is returned.
//!
//! **Effect** is a signed-free magnitude in the units documented on each
//! struct (a difference, a proportion, a rate, a variance ratio, …). For
//! designs where it matters which side of the null the alternative lies on,
//! the struct has a [`Direction`] that is only consulted for two-sided tests.

use crate::error::{invalid, Result, StatsError};
use crate::hypothesis::Alternative;
use crate::poisson::poisson_p;
use crate::proportion::binomial_p;
use openstats_math::dist::{
    Binomial, ChiSquared, ContinuousDistribution, DiscreteDistribution, FisherF, NonCentralF, NonCentralT, Normal,
    Poisson, StudentT,
};
use openstats_math::integrate::adaptive;
use openstats_math::roots::brent;
use openstats_math::special::gamma::ln_factorial;

/// Which quantity a [`PowerSolution`] was solved for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolvedFor {
    /// Sample size from power and effect.
    SampleSize,
    /// Power from sample size and effect.
    Power,
    /// Effect from sample size and power.
    Effect,
}

/// Side of the null on which the alternative lies (used when solving for the
/// effect of a two-sided test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Alternative above the null value (ratio > 1, p₁ > p₀, …).
    Increase,
    /// Alternative below the null value.
    Decrease,
}

/// Method used to compute power for proportions and Poisson rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscreteMethod {
    /// Power of the normal-approximation (z) test.
    NormalApproximation,
    /// Power of the exact test (binomial / Fisher / conditional Poisson),
    /// using its true rejection region at level α.
    Exact,
}

/// Result of [`PowerModel::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSolution {
    /// The quantity that was solved for.
    pub solved_for: SolvedFor,
    /// Sample size (per group / replicates / observation units).
    pub sample_size: usize,
    /// Power attained at `sample_size` and `effect`.
    pub power: f64,
    /// Power that was asked for (`None` when power itself was solved for).
    pub target_power: Option<f64>,
    /// Effect size.
    pub effect: f64,
}

/// A design for which power can be computed and solved.
pub trait PowerModel {
    /// Name of the procedure, e.g. "2-Sample t Test".
    fn name(&self) -> String;
    /// Fixed inputs as (label, value) pairs for reports.
    fn parameters(&self) -> Vec<(String, String)>;
    /// Label of the effect size, e.g. "Difference".
    fn effect_label(&self) -> &'static str;
    /// Label of the sample size, e.g. "Sample Size (per group)".
    fn sample_size_label(&self) -> &'static str {
        "Sample Size"
    }
    /// Significance level.
    fn alpha(&self) -> f64;
    /// Power at sample size `n` and effect `effect`.
    fn power(&self, n: usize, effect: f64) -> Result<f64>;
    /// Smallest sample size for which power is defined.
    fn min_n(&self) -> Result<usize> {
        Ok(2)
    }
    /// Largest sample size the solver will try.
    fn max_n(&self) -> usize {
        5_000_000
    }
    /// `true` when power is saw-toothed in *n* (exact discrete tests): the
    /// sample-size search then tries every *n*.
    fn discrete(&self) -> bool {
        false
    }
    /// Search interval for the effect: (value at which the alternative is
    /// "nearest" the null, far end). The far end may be ±∞.
    fn effect_range(&self) -> (f64, f64);
    /// Typical size of the effect (used to expand an infinite range).
    fn effect_scale(&self) -> f64 {
        1.0
    }

    /// Smallest sample size whose power is at least `target`.
    fn sample_size(&self, effect: f64, target: f64) -> Result<usize> {
        check_power(target)?;
        let min = self.min_n()?;
        let max = self.max_n().max(min);
        if self.discrete() {
            for n in min..=max {
                if self.power(n, effect)? >= target {
                    return Ok(n);
                }
            }
            return Err(StatsError::TooLarge(format!(
                "power {target} is not reached with up to {max} observations (power at {max} is {:.4})",
                self.power(max, effect)?
            )));
        }
        if self.power(min, effect)? >= target {
            return Ok(min);
        }
        let mut lo = min;
        let mut hi = min.max(1);
        loop {
            hi = hi.saturating_mul(2).min(max);
            if self.power(hi, effect)? >= target {
                break;
            }
            if hi == max {
                return Err(StatsError::TooLarge(format!(
                    "power {target} is not reached with up to {max} observations; the effect is too small"
                )));
            }
            lo = hi;
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.power(mid, effect)? >= target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(hi)
    }

    /// The effect at which power equals `target` for sample size `n`.
    fn effect(&self, n: usize, target: f64) -> Result<f64> {
        check_power(target)?;
        let (lo, far) = self.effect_range();
        let f = |e: f64| -> Result<f64> { Ok(self.power(n, e)? - target) };
        let f_lo = f(lo)?;
        let mut hi = far;
        if far.is_infinite() {
            let step = self.effect_scale();
            let mut k = 1.0;
            loop {
                hi = lo + far.signum() * step * k;
                if f(hi)? >= 0.0 {
                    break;
                }
                k *= 2.0;
                if k > 1e15 {
                    return invalid("the requested power cannot be reached for any effect size");
                }
            }
        }
        let f_hi = f(hi)?;
        if f_lo * f_hi > 0.0 {
            return invalid(format!(
                "power {target} is not attainable for this design: power is {:.4} at {lo} and {:.4} at {hi}",
                f_lo + target,
                f_hi + target
            ));
        }
        let mut err = None;
        let root = brent(
            |e| match f(e) {
                Ok(v) => v,
                Err(x) => {
                    err = Some(x);
                    f64::NAN
                }
            },
            lo,
            hi,
            1e-13 * self.effect_scale().max(1e-300),
            300,
        );
        if let Some(e) = err {
            return Err(e);
        }
        Ok(root?)
    }

    /// Solve for whichever of `n`, `power`, `effect` is `None`; exactly one
    /// must be `None`.
    fn solve(&self, n: Option<usize>, power: Option<f64>, effect: Option<f64>) -> Result<PowerSolution>
    where
        Self: Sized,
    {
        match (n, power, effect) {
            (None, Some(p), Some(e)) => {
                let n = self.sample_size(e, p)?;
                Ok(PowerSolution {
                    solved_for: SolvedFor::SampleSize,
                    sample_size: n,
                    power: self.power(n, e)?,
                    target_power: Some(p),
                    effect: e,
                })
            }
            (Some(n), None, Some(e)) => Ok(PowerSolution {
                solved_for: SolvedFor::Power,
                sample_size: n,
                power: self.power(n, e)?,
                target_power: None,
                effect: e,
            }),
            (Some(n), Some(p), None) => {
                let e = self.effect(n, p)?;
                Ok(PowerSolution {
                    solved_for: SolvedFor::Effect,
                    sample_size: n,
                    power: self.power(n, e)?,
                    target_power: Some(p),
                    effect: e,
                })
            }
            _ => invalid("exactly one of sample size, power and effect must be left unspecified"),
        }
    }

    /// Power at each effect in `effects` for a fixed sample size.
    fn power_curve(&self, n: usize, effects: &[f64]) -> Result<Vec<f64>> {
        effects.iter().map(|&e| self.power(n, e)).collect()
    }
}

fn check_power(p: f64) -> Result<()> {
    if p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        invalid(format!("power {p} must be in (0, 1)"))
    }
}

fn check_alpha(a: f64) -> Result<()> {
    if a > 0.0 && a < 1.0 {
        Ok(())
    } else {
        invalid(format!("significance level {a} must be in (0, 1)"))
    }
}

fn check_pos(name: &str, x: f64) -> Result<()> {
    if x > 0.0 && x.is_finite() {
        Ok(())
    } else {
        invalid(format!("{name} must be positive and finite (got {x})"))
    }
}

fn check_n(n: usize, min: usize) -> Result<()> {
    if n >= min {
        Ok(())
    } else {
        Err(StatsError::InsufficientData { what: "power calculation", needed: min, got: n })
    }
}

fn alt_name(alt: Alternative) -> &'static str {
    match alt {
        Alternative::TwoSided => "not equal",
        Alternative::Less => "less than",
        Alternative::Greater => "greater than",
    }
}

fn direction_of(alt: Alternative, two_sided: Direction) -> Direction {
    match alt {
        Alternative::Less => Direction::Decrease,
        Alternative::Greater => Direction::Increase,
        Alternative::TwoSided => two_sided,
    }
}

fn fmt(x: f64) -> String {
    format!("{x}")
}

/// Power of a z-type test whose statistic has null sd `s0` and alternative
/// sd `s1`, when the true shift is `shift` (positive = towards the
/// alternative for one-sided tests, signed for two-sided).
fn z_type_power(alt: Alternative, alpha: f64, shift: f64, s0: f64, s1: f64) -> f64 {
    let norm = Normal::standard();
    match alt {
        Alternative::TwoSided => {
            let z = norm.isf(alpha / 2.0);
            norm.sf((z * s0 - shift) / s1) + norm.sf((z * s0 + shift) / s1)
        }
        _ => norm.sf((norm.isf(alpha) * s0 - shift) / s1),
    }
}

/// Power of a t test with `df` degrees of freedom and non-centrality `delta`.
fn t_power(alt: Alternative, alpha: f64, df: f64, delta: f64) -> Result<f64> {
    let t = StudentT::new(df)?;
    let nc = NonCentralT::new(df, delta)?;
    Ok(match alt {
        Alternative::TwoSided => {
            let c = t.isf(alpha / 2.0);
            (nc.sf(c) + nc.cdf(-c)).min(1.0)
        }
        _ => nc.sf(t.isf(alpha)),
    })
}

// ---------------------------------------------------------------------------
// Means: Z, t, paired t, 2-sample t
// ---------------------------------------------------------------------------

/// 1-sample Z test with known σ. Effect = |μ − μ₀|.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneSampleZ {
    /// Known standard deviation.
    pub sigma: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis.
    pub alternative: Alternative,
}

impl PowerModel for OneSampleZ {
    fn name(&self) -> String {
        "1-Sample Z Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Standard deviation".into(), fmt(self.sigma)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Difference"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn min_n(&self) -> Result<usize> {
        Ok(1)
    }
    fn power(&self, n: usize, effect: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("standard deviation", self.sigma)?;
        check_n(n, 1)?;
        let shift = effect.abs() * (n as f64).sqrt() / self.sigma;
        Ok(z_type_power(self.alternative, self.alpha, shift, 1.0, 1.0))
    }
    fn effect_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn effect_scale(&self) -> f64 {
        self.sigma
    }
}

/// 1-sample t test. Effect = |μ − μ₀|; `sigma` is the (anticipated)
/// standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneSampleT {
    /// Standard deviation.
    pub sigma: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis.
    pub alternative: Alternative,
}

impl PowerModel for OneSampleT {
    fn name(&self) -> String {
        "1-Sample t Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Standard deviation".into(), fmt(self.sigma)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Difference"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, effect: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("standard deviation", self.sigma)?;
        check_n(n, 2)?;
        let delta = effect.abs() * (n as f64).sqrt() / self.sigma;
        t_power(self.alternative, self.alpha, n as f64 - 1.0, delta)
    }
    fn effect_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn effect_scale(&self) -> f64 {
        self.sigma
    }
}

/// Paired t test. Effect = |mean difference|; `sigma_diff` is the standard
/// deviation of the *paired differences*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedT {
    /// Standard deviation of the differences.
    pub sigma_diff: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis.
    pub alternative: Alternative,
}

impl PowerModel for PairedT {
    fn name(&self) -> String {
        "Paired t Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Standard deviation of differences".into(), fmt(self.sigma_diff)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Difference"
    }
    fn sample_size_label(&self) -> &'static str {
        "Sample Size (pairs)"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, effect: f64) -> Result<f64> {
        OneSampleT { sigma: self.sigma_diff, alpha: self.alpha, alternative: self.alternative }.power(n, effect)
    }
    fn effect_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn effect_scale(&self) -> f64 {
        self.sigma_diff
    }
}

/// 2-sample t test (equal variances, equal sample sizes `n` per group).
/// Effect = |μ₁ − μ₂|.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoSampleT {
    /// Common standard deviation.
    pub sigma: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis.
    pub alternative: Alternative,
}

impl PowerModel for TwoSampleT {
    fn name(&self) -> String {
        "2-Sample t Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Standard deviation".into(), fmt(self.sigma)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Difference"
    }
    fn sample_size_label(&self) -> &'static str {
        "Sample Size (per group)"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, effect: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("standard deviation", self.sigma)?;
        check_n(n, 2)?;
        let nf = n as f64;
        let delta = effect.abs() / (self.sigma * (2.0 / nf).sqrt());
        t_power(self.alternative, self.alpha, 2.0 * nf - 2.0, delta)
    }
    fn effect_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn effect_scale(&self) -> f64 {
        self.sigma
    }
}

// ---------------------------------------------------------------------------
// Exact-test rejection regions
// ---------------------------------------------------------------------------

/// Largest `x` in `[lo, hi]` with `pred(x)` true, assuming `pred` is true on
/// a (possibly empty) prefix; `None` if `pred(lo)` is false.
fn last_true(lo: u64, hi: u64, mut pred: impl FnMut(u64) -> Result<bool>) -> Result<Option<u64>> {
    if !pred(lo)? {
        return Ok(None);
    }
    let (mut a, mut b) = (lo, hi);
    while a < b {
        let mid = a + (b - a).div_ceil(2);
        if pred(mid)? {
            a = mid;
        } else {
            b = mid - 1;
        }
    }
    Ok(Some(a))
}

/// Smallest `x` in `[lo, hi]` with `pred(x)` true, assuming `pred` is true on
/// a (possibly empty) suffix.
fn first_true(lo: u64, hi: u64, mut pred: impl FnMut(u64) -> Result<bool>) -> Result<Option<u64>> {
    if !pred(hi)? {
        return Ok(None);
    }
    let (mut a, mut b) = (lo, hi);
    while a < b {
        let mid = a + (b - a) / 2;
        if pred(mid)? {
            b = mid;
        } else {
            a = mid + 1;
        }
    }
    Ok(Some(a))
}

/// Rejection region `{x ≤ lower} ∪ {x ≥ upper}` of the exact binomial test of
/// `p0` with `n` trials at level α (the p-values are those used by
/// [`crate::proportion::one_proportion`]).
fn binomial_rejection(n: u64, p0: f64, alt: Alternative, alpha: f64) -> Result<(Option<u64>, Option<u64>)> {
    let pv = |x: u64| binomial_p(n, p0, x, alt);
    let mode = (((n + 1) as f64 * p0).floor() as u64).min(n);
    Ok(match alt {
        Alternative::Less => (last_true(0, n, |x| Ok(pv(x)? <= alpha))?, None),
        Alternative::Greater => (None, first_true(0, n, |x| Ok(pv(x)? <= alpha))?),
        Alternative::TwoSided => {
            (last_true(0, mode, |x| Ok(pv(x)? <= alpha))?, first_true(mode, n, |x| Ok(pv(x)? <= alpha))?)
        }
    })
}

/// Same for the exact Poisson test of mean `mu0` (unbounded support).
fn poisson_rejection(mu0: f64, alt: Alternative, alpha: f64) -> Result<(Option<u64>, Option<u64>)> {
    let pv = |x: u64| poisson_p(mu0, x, alt);
    let mode = mu0.floor() as u64;
    let upper_search = |lo: u64| -> Result<Option<u64>> {
        let mut hi = lo.max(1) * 2;
        while pv(hi)? > alpha {
            hi = hi.saturating_mul(2);
            if hi > u64::MAX / 8 {
                return Ok(None);
            }
        }
        first_true(lo, hi, |x| Ok(pv(x)? <= alpha))
    };
    Ok(match alt {
        Alternative::Less => {
            let mut hi = mode + 1;
            while pv(hi)? <= alpha {
                hi = hi.saturating_mul(2);
                if hi > u64::MAX / 8 {
                    break;
                }
            }
            (last_true(0, hi, |x| Ok(pv(x)? <= alpha))?, None)
        }
        Alternative::Greater => (None, upper_search(0)?),
        Alternative::TwoSided => (last_true(0, mode, |x| Ok(pv(x)? <= alpha))?, upper_search(mode)?),
    })
}

/// Probability that `X ≤ lower` or `X ≥ upper` for a binomial.
fn binomial_region_prob(n: u64, p: f64, region: (Option<u64>, Option<u64>)) -> Result<f64> {
    let b = Binomial::new(n, p)?;
    let mut s = 0.0;
    if let Some(l) = region.0 {
        s += b.cdf(l as f64);
    }
    if let Some(u) = region.1 {
        s += if u == 0 { 1.0 } else { b.sf(u as f64 - 1.0) };
    }
    Ok(s.min(1.0))
}

fn poisson_region_prob(mu: f64, region: (Option<u64>, Option<u64>)) -> Result<f64> {
    let d = Poisson::new(mu)?;
    let mut s = 0.0;
    if let Some(l) = region.0 {
        s += d.cdf(l as f64);
    }
    if let Some(u) = region.1 {
        s += if u == 0 { 1.0 } else { d.sf(u as f64 - 1.0) };
    }
    Ok(s.min(1.0))
}

fn ratio_range(alt: Alternative, dir: Direction, null: f64) -> (f64, f64) {
    match direction_of(alt, dir) {
        Direction::Increase => (null, f64::INFINITY),
        Direction::Decrease => (null, null * 1e-9),
    }
}

// ---------------------------------------------------------------------------
// Proportions
// ---------------------------------------------------------------------------

/// 1-proportion test of H₀: p = `p0`. Effect = the alternative value p₁.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneProportion {
    /// Hypothesized proportion p₀.
    pub p0: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis.
    pub alternative: Alternative,
    /// Normal approximation or exact binomial test.
    pub method: DiscreteMethod,
    /// Side of p₀ searched when solving a two-sided test for p₁.
    pub direction: Direction,
}

impl OneProportion {
    /// Design with `direction = Increase`.
    pub fn new(p0: f64, alpha: f64, alternative: Alternative, method: DiscreteMethod) -> Self {
        OneProportion { p0, alpha, alternative, method, direction: Direction::Increase }
    }

    /// Actual significance level of the exact test at sample size `n`
    /// (≤ α because the binomial is discrete).
    pub fn actual_alpha(&self, n: usize) -> Result<f64> {
        let region = binomial_rejection(n as u64, self.p0, self.alternative, self.alpha)?;
        binomial_region_prob(n as u64, self.p0, region)
    }
}

impl PowerModel for OneProportion {
    fn name(&self) -> String {
        "1 Proportion Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Comparison (null) proportion".into(), fmt(self.p0)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
            ("Method".into(), format!("{:?}", self.method)),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Alternative proportion"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn min_n(&self) -> Result<usize> {
        Ok(1)
    }
    fn max_n(&self) -> usize {
        if self.method == DiscreteMethod::Exact {
            20_000
        } else {
            5_000_000
        }
    }
    fn discrete(&self) -> bool {
        self.method == DiscreteMethod::Exact
    }
    fn power(&self, n: usize, p1: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        if !(self.p0 > 0.0 && self.p0 < 1.0) || !(p1 > 0.0 && p1 < 1.0) {
            return invalid("proportions must be strictly between 0 and 1");
        }
        check_n(n, 1)?;
        let nf = n as f64;
        match self.method {
            DiscreteMethod::NormalApproximation => {
                let s0 = (self.p0 * (1.0 - self.p0) / nf).sqrt();
                let s1 = (p1 * (1.0 - p1) / nf).sqrt();
                let shift = match self.alternative {
                    Alternative::Less => self.p0 - p1,
                    _ => p1 - self.p0,
                };
                Ok(z_type_power(self.alternative, self.alpha, shift, s0, s1))
            }
            DiscreteMethod::Exact => {
                let region = binomial_rejection(n as u64, self.p0, self.alternative, self.alpha)?;
                binomial_region_prob(n as u64, p1, region)
            }
        }
    }
    fn effect_range(&self) -> (f64, f64) {
        match direction_of(self.alternative, self.direction) {
            Direction::Increase => (self.p0, 1.0 - 1e-9),
            Direction::Decrease => (self.p0, 1e-9),
        }
    }
    fn effect_scale(&self) -> f64 {
        0.1
    }
}

/// 2-proportions test. `baseline` is p₂; the effect is the comparison
/// proportion p₁ (alternative p₁ ≠ / < / > p₂). Sample size is per group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoProportions {
    /// Baseline proportion p₂.
    pub baseline: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis about p₁ − p₂.
    pub alternative: Alternative,
    /// Normal approximation (unpooled alternative variance, pooled null
    /// variance) or Fisher's exact test.
    pub method: DiscreteMethod,
    /// Side of the baseline searched when solving a two-sided test for p₁.
    pub direction: Direction,
}

impl TwoProportions {
    /// Design with `direction = Increase`.
    pub fn new(baseline: f64, alpha: f64, alternative: Alternative, method: DiscreteMethod) -> Self {
        TwoProportions { baseline, alpha, alternative, method, direction: Direction::Increase }
    }

    /// Power of Fisher's exact test: for each total `s = x₁ + x₂` the
    /// rejection set is built from the hypergeometric null (two-sided by
    /// the small-p-values rule), then summed against the true binomial
    /// probabilities.
    fn fisher_power(&self, n: usize, p1: f64) -> Result<f64> {
        let p2 = self.baseline;
        let nn = n as u64;
        let ln_choose = |a: u64, b: u64| ln_factorial(a) - ln_factorial(b) - ln_factorial(a - b);
        let (lp1, lq1, lp2, lq2) = (p1.ln(), (1.0 - p1).ln(), p2.ln(), (1.0 - p2).ln());
        let mut power = 0.0;
        for s in 0..=2 * nn {
            let x_lo = s.saturating_sub(nn);
            let x_hi = s.min(nn);
            // Hypergeometric null pmf of x₁ given the total s.
            let ln_den = ln_choose(2 * nn, s);
            let h: Vec<f64> = (x_lo..=x_hi).map(|x| (ln_choose(nn, x) + ln_choose(nn, s - x) - ln_den).exp()).collect();
            // True joint probability of (x₁, s − x₁).
            let joint: Vec<f64> = (x_lo..=x_hi)
                .map(|x| {
                    (ln_choose(nn, x)
                        + x as f64 * lp1
                        + (nn - x) as f64 * lq1
                        + ln_choose(nn, s - x)
                        + (s - x) as f64 * lp2
                        + (nn - (s - x)) as f64 * lq2)
                        .exp()
                })
                .collect();
            let total_joint: f64 = joint.iter().sum();
            if total_joint < 1e-18 {
                continue;
            }
            let m = h.len();
            let reject: Vec<bool> = match self.alternative {
                Alternative::Greater => {
                    // p(x) = P(X₁ ≥ x)
                    let mut tail = 0.0;
                    let mut r = vec![false; m];
                    for i in (0..m).rev() {
                        tail += h[i];
                        r[i] = tail <= self.alpha;
                    }
                    r
                }
                Alternative::Less => {
                    let mut tail = 0.0;
                    let mut r = vec![false; m];
                    for i in 0..m {
                        tail += h[i];
                        r[i] = tail <= self.alpha;
                    }
                    r
                }
                Alternative::TwoSided => {
                    // p(x) = Σ_{y: h(y) ≤ h(x)(1+1e-7)} h(y), via sorted prefix sums.
                    let mut sorted: Vec<f64> = h.clone();
                    sorted.sort_by(f64::total_cmp);
                    let mut prefix = vec![0.0; m + 1];
                    for i in 0..m {
                        prefix[i + 1] = prefix[i] + sorted[i];
                    }
                    h.iter()
                        .map(|&hx| {
                            let t = hx * (1.0 + 1e-7);
                            prefix[sorted.partition_point(|&v| v <= t)] <= self.alpha
                        })
                        .collect()
                }
            };
            power += joint.iter().zip(&reject).filter(|(_, &r)| r).map(|(j, _)| j).sum::<f64>();
        }
        Ok(power.min(1.0))
    }
}

impl PowerModel for TwoProportions {
    fn name(&self) -> String {
        "2 Proportions Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Baseline proportion (p2)".into(), fmt(self.baseline)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
            (
                "Method".into(),
                match self.method {
                    DiscreteMethod::NormalApproximation => "Normal approximation".into(),
                    DiscreteMethod::Exact => "Fisher's exact test".into(),
                },
            ),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Comparison proportion (p1)"
    }
    fn sample_size_label(&self) -> &'static str {
        "Sample Size (per group)"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn min_n(&self) -> Result<usize> {
        Ok(1)
    }
    fn max_n(&self) -> usize {
        if self.method == DiscreteMethod::Exact {
            1_000
        } else {
            5_000_000
        }
    }
    fn discrete(&self) -> bool {
        self.method == DiscreteMethod::Exact
    }
    fn power(&self, n: usize, p1: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        let p2 = self.baseline;
        if !(p2 > 0.0 && p2 < 1.0) || !(p1 > 0.0 && p1 < 1.0) {
            return invalid("proportions must be strictly between 0 and 1");
        }
        check_n(n, 1)?;
        match self.method {
            DiscreteMethod::NormalApproximation => {
                let nf = n as f64;
                let pbar = 0.5 * (p1 + p2);
                let s0 = (2.0 * pbar * (1.0 - pbar) / nf).sqrt();
                let s1 = ((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / nf).sqrt();
                let shift = match self.alternative {
                    Alternative::Less => p2 - p1,
                    _ => p1 - p2,
                };
                Ok(z_type_power(self.alternative, self.alpha, shift, s0, s1))
            }
            DiscreteMethod::Exact => {
                if n > self.max_n() {
                    return Err(StatsError::TooLarge("Fisher's exact power needs n ≤ 1000 per group".into()));
                }
                self.fisher_power(n, p1)
            }
        }
    }
    fn effect_range(&self) -> (f64, f64) {
        match direction_of(self.alternative, self.direction) {
            Direction::Increase => (self.baseline, 1.0 - 1e-9),
            Direction::Decrease => (self.baseline, 1e-9),
        }
    }
    fn effect_scale(&self) -> f64 {
        0.1
    }
}

// ---------------------------------------------------------------------------
// Poisson rates
// ---------------------------------------------------------------------------

/// 1-sample Poisson rate test of H₀: λ = `rate0`. The sample size is the
/// number of observation units, each of `length`; effect = the alternative
/// rate λ₁.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePoisson {
    /// Hypothesized rate λ₀ (per unit length).
    pub rate0: f64,
    /// Length of each observation unit.
    pub length: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis.
    pub alternative: Alternative,
    /// Normal approximation or exact Poisson test.
    pub method: DiscreteMethod,
    /// Side of λ₀ searched when solving a two-sided test for λ₁.
    pub direction: Direction,
}

impl OnePoisson {
    /// Design with `length = 1` and `direction = Increase`.
    pub fn new(rate0: f64, alpha: f64, alternative: Alternative, method: DiscreteMethod) -> Self {
        OnePoisson { rate0, length: 1.0, alpha, alternative, method, direction: Direction::Increase }
    }
}

impl PowerModel for OnePoisson {
    fn name(&self) -> String {
        "1-Sample Poisson Rate Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Hypothesized rate".into(), fmt(self.rate0)),
            ("Length of each observation".into(), fmt(self.length)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
            ("Method".into(), format!("{:?}", self.method)),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Comparison rate"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn min_n(&self) -> Result<usize> {
        Ok(1)
    }
    fn max_n(&self) -> usize {
        if self.method == DiscreteMethod::Exact {
            20_000
        } else {
            5_000_000
        }
    }
    fn discrete(&self) -> bool {
        self.method == DiscreteMethod::Exact
    }
    fn power(&self, n: usize, rate1: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("hypothesized rate", self.rate0)?;
        check_pos("comparison rate", rate1)?;
        check_pos("length", self.length)?;
        check_n(n, 1)?;
        let t = n as f64 * self.length;
        match self.method {
            DiscreteMethod::NormalApproximation => {
                let s0 = (self.rate0 / t).sqrt();
                let s1 = (rate1 / t).sqrt();
                let shift = match self.alternative {
                    Alternative::Less => self.rate0 - rate1,
                    _ => rate1 - self.rate0,
                };
                Ok(z_type_power(self.alternative, self.alpha, shift, s0, s1))
            }
            DiscreteMethod::Exact => {
                let region = poisson_rejection(self.rate0 * t, self.alternative, self.alpha)?;
                poisson_region_prob(rate1 * t, region)
            }
        }
    }
    fn effect_range(&self) -> (f64, f64) {
        ratio_range(self.alternative, self.direction, self.rate0)
    }
    fn effect_scale(&self) -> f64 {
        self.rate0
    }
}

/// 2-sample Poisson rate test of H₀: λ₁ = λ₂ with equal observation lengths.
/// `baseline` is λ₂, the effect is the comparison rate λ₁; the sample size is
/// the number of observation units per group, each of `length`. The exact
/// method is the conditional binomial test given the total count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPoisson {
    /// Baseline rate λ₂ (per unit length).
    pub baseline: f64,
    /// Length of each observation unit.
    pub length: f64,
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis about λ₁ − λ₂.
    pub alternative: Alternative,
    /// Normal approximation or exact conditional test.
    pub method: DiscreteMethod,
    /// Side of the baseline searched when solving a two-sided test for λ₁.
    pub direction: Direction,
}

impl TwoPoisson {
    /// Design with `length = 1` and `direction = Increase`.
    pub fn new(baseline: f64, alpha: f64, alternative: Alternative, method: DiscreteMethod) -> Self {
        TwoPoisson { baseline, length: 1.0, alpha, alternative, method, direction: Direction::Increase }
    }
}

impl PowerModel for TwoPoisson {
    fn name(&self) -> String {
        "2-Sample Poisson Rate Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Baseline rate".into(), fmt(self.baseline)),
            ("Length of each observation".into(), fmt(self.length)),
            ("Alpha".into(), fmt(self.alpha)),
            ("Alternative".into(), alt_name(self.alternative).into()),
            ("Method".into(), format!("{:?}", self.method)),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Comparison rate"
    }
    fn sample_size_label(&self) -> &'static str {
        "Sample Size (per group)"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn min_n(&self) -> Result<usize> {
        Ok(1)
    }
    fn max_n(&self) -> usize {
        if self.method == DiscreteMethod::Exact {
            5_000
        } else {
            5_000_000
        }
    }
    fn discrete(&self) -> bool {
        self.method == DiscreteMethod::Exact
    }
    fn power(&self, n: usize, rate1: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("baseline rate", self.baseline)?;
        check_pos("comparison rate", rate1)?;
        check_pos("length", self.length)?;
        check_n(n, 1)?;
        let t = n as f64 * self.length;
        let l2 = self.baseline;
        match self.method {
            DiscreteMethod::NormalApproximation => {
                let s0 = ((l2 + rate1) / t).sqrt();
                let s1 = ((rate1 + l2) / t).sqrt();
                // Var(λ̂₁ − λ̂₂) = (λ₁ + λ₂)/t under any hypothesis for the
                // difference test with unpooled variance; the null variance
                // uses the pooled rate 2·λ̄/t with λ̄ = (λ₁+λ₂)/2, the same.
                let shift = match self.alternative {
                    Alternative::Less => l2 - rate1,
                    _ => rate1 - l2,
                };
                Ok(z_type_power(self.alternative, self.alpha, shift, s0, s1))
            }
            DiscreteMethod::Exact => {
                let mu = (rate1 + l2) * t;
                let total = Poisson::new(mu)?;
                let s_lo = total.quantile(1e-15) as u64;
                let s_hi = total.isf(1e-15) as u64 + 1;
                let pi1 = rate1 / (rate1 + l2);
                let mut power = 0.0;
                for s in s_lo..=s_hi {
                    if s == 0 {
                        continue;
                    }
                    let region = binomial_rejection(s, 0.5, self.alternative, self.alpha)?;
                    power += total.pmf(s as f64) * binomial_region_prob(s, pi1, region)?;
                }
                Ok(power.min(1.0))
            }
        }
    }
    fn effect_range(&self) -> (f64, f64) {
        ratio_range(self.alternative, self.direction, self.baseline)
    }
    fn effect_scale(&self) -> f64 {
        self.baseline
    }
}

// ---------------------------------------------------------------------------
// Variances
// ---------------------------------------------------------------------------

/// 1-variance χ² test of H₀: σ² = σ₀². Effect = ratio σ²/σ₀² of the true to
/// the hypothesized *variance* (square an SD ratio).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneVariance {
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis about σ².
    pub alternative: Alternative,
    /// Side of 1 searched when solving a two-sided test for the ratio.
    pub direction: Direction,
}

impl PowerModel for OneVariance {
    fn name(&self) -> String {
        "1 Variance Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![("Alpha".into(), fmt(self.alpha)), ("Alternative".into(), alt_name(self.alternative).into())]
    }
    fn effect_label(&self) -> &'static str {
        "Variance ratio"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, ratio: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("variance ratio", ratio)?;
        check_n(n, 2)?;
        let d = ChiSquared::new(n as f64 - 1.0)?;
        Ok(match self.alternative {
            Alternative::TwoSided => {
                (d.sf(d.isf(self.alpha / 2.0) / ratio) + d.cdf(d.quantile(self.alpha / 2.0) / ratio)).min(1.0)
            }
            Alternative::Greater => d.sf(d.isf(self.alpha) / ratio),
            Alternative::Less => d.cdf(d.quantile(self.alpha) / ratio),
        })
    }
    fn effect_range(&self) -> (f64, f64) {
        ratio_range(self.alternative, self.direction, 1.0)
    }
}

/// 2-variances F test of H₀: σ₁² = σ₂² with `n` per group. Effect = ratio
/// σ₁²/σ₂² of the true variances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoVariances {
    /// Significance level.
    pub alpha: f64,
    /// Alternative hypothesis about σ₁²/σ₂².
    pub alternative: Alternative,
    /// Side of 1 searched when solving a two-sided test for the ratio.
    pub direction: Direction,
}

impl PowerModel for TwoVariances {
    fn name(&self) -> String {
        "2 Variances Test".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![("Alpha".into(), fmt(self.alpha)), ("Alternative".into(), alt_name(self.alternative).into())]
    }
    fn effect_label(&self) -> &'static str {
        "Variance ratio"
    }
    fn sample_size_label(&self) -> &'static str {
        "Sample Size (per group)"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, ratio: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("variance ratio", ratio)?;
        check_n(n, 2)?;
        let d = FisherF::new(n as f64 - 1.0, n as f64 - 1.0)?;
        Ok(match self.alternative {
            Alternative::TwoSided => {
                (d.sf(d.isf(self.alpha / 2.0) / ratio) + d.cdf(d.quantile(self.alpha / 2.0) / ratio)).min(1.0)
            }
            Alternative::Greater => d.sf(d.isf(self.alpha) / ratio),
            Alternative::Less => d.cdf(d.quantile(self.alpha) / ratio),
        })
    }
    fn effect_range(&self) -> (f64, f64) {
        ratio_range(self.alternative, self.direction, 1.0)
    }
}

// ---------------------------------------------------------------------------
// One-way ANOVA
// ---------------------------------------------------------------------------

/// One-way ANOVA with `groups` levels and `n` observations per group. The
/// effect is the *maximum difference between means*, with the remaining
/// means at the centre (the least favourable configuration, as in Minitab):
/// λ = n·δ²/(2σ²). [`OneWayAnova::power_for_means`] takes arbitrary means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneWayAnova {
    /// Number of groups (≥ 2).
    pub groups: usize,
    /// Common standard deviation.
    pub sigma: f64,
    /// Significance level.
    pub alpha: f64,
}

impl OneWayAnova {
    fn power_lambda(&self, n: usize, lambda: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("standard deviation", self.sigma)?;
        if self.groups < 2 {
            return invalid("ANOVA needs at least 2 groups");
        }
        check_n(n, 2)?;
        let df1 = self.groups as f64 - 1.0;
        let df2 = self.groups as f64 * (n as f64 - 1.0);
        let crit = FisherF::new(df1, df2)?.isf(self.alpha);
        Ok(NonCentralF::new(df1, df2, lambda)?.sf(crit))
    }

    /// Power for the given group means (`means.len() == groups`):
    /// λ = n·Σ(μᵢ − μ̄)²/σ².
    pub fn power_for_means(&self, n: usize, means: &[f64]) -> Result<f64> {
        if means.len() != self.groups {
            return invalid(format!("expected {} means, got {}", self.groups, means.len()));
        }
        if means.iter().any(|m| !m.is_finite()) {
            return Err(StatsError::NonFinite);
        }
        let grand = means.iter().sum::<f64>() / means.len() as f64;
        let ss: f64 = means.iter().map(|m| (m - grand) * (m - grand)).sum();
        self.power_lambda(n, n as f64 * ss / (self.sigma * self.sigma))
    }
}

impl PowerModel for OneWayAnova {
    fn name(&self) -> String {
        "One-Way ANOVA".into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Number of groups".into(), self.groups.to_string()),
            ("Standard deviation".into(), fmt(self.sigma)),
            ("Alpha".into(), fmt(self.alpha)),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Maximum difference"
    }
    fn sample_size_label(&self) -> &'static str {
        "Sample Size (per group)"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, effect: f64) -> Result<f64> {
        self.power_lambda(n, n as f64 * effect * effect / (2.0 * self.sigma * self.sigma))
    }
    fn effect_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn effect_scale(&self) -> f64 {
        self.sigma
    }
}

// ---------------------------------------------------------------------------
// 2-level factorial and Plackett–Burman designs
// ---------------------------------------------------------------------------

/// Family of a two-level screening / factorial design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorialKind {
    /// Full 2ᵏ factorial.
    Full,
    /// 2^(k−p) regular fraction.
    Fractional {
        /// Number of generators p.
        generators: usize,
    },
    /// Plackett–Burman design.
    PlackettBurman,
}

/// A two-level design: its family, number of factors and base-design runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorialDesign {
    /// Design family.
    pub kind: FactorialKind,
    /// Number of factors.
    pub factors: usize,
    /// Runs (corner points) of one replicate of the base design.
    pub base_runs: usize,
}

impl FactorialDesign {
    /// Full 2ᵏ factorial, 2 ≤ k ≤ 15.
    pub fn full(factors: usize) -> Result<Self> {
        if !(2..=15).contains(&factors) {
            return invalid("a full factorial needs 2 to 15 factors");
        }
        Ok(FactorialDesign { kind: FactorialKind::Full, factors, base_runs: 1 << factors })
    }

    /// Regular fraction 2^(k−p): needs k − p ≥ 2 and room for k main
    /// effects (k ≤ 2^(k−p) − 1).
    pub fn fractional(factors: usize, generators: usize) -> Result<Self> {
        if generators == 0 || factors > 20 || factors < generators + 2 {
            return invalid("a fractional factorial needs 1 ≤ p and k − p ≥ 2 (k ≤ 20)");
        }
        let base_runs = 1usize << (factors - generators);
        if factors > base_runs - 1 {
            return invalid(format!(
                "2^({factors}-{generators}) has only {} runs: too few for {factors} main effects",
                base_runs
            ));
        }
        Ok(FactorialDesign { kind: FactorialKind::Fractional { generators }, factors, base_runs })
    }

    /// Plackett–Burman design with `runs` (a multiple of 4, 8 … 100) and
    /// 2 ≤ `factors` ≤ runs − 1.
    pub fn plackett_burman(runs: usize, factors: usize) -> Result<Self> {
        if runs % 4 != 0 || !(8..=100).contains(&runs) {
            return invalid("Plackett–Burman designs have a multiple of 4 runs between 8 and 100");
        }
        if factors < 2 || factors > runs - 1 {
            return invalid(format!("a {runs}-run Plackett–Burman design handles 2 to {} factors", runs - 1));
        }
        Ok(FactorialDesign { kind: FactorialKind::PlackettBurman, factors, base_runs: runs })
    }

    fn describe(&self) -> String {
        match self.kind {
            FactorialKind::Full => format!("2^{} full factorial", self.factors),
            FactorialKind::Fractional { generators } => {
                format!("2^({}-{}) fractional factorial", self.factors, generators)
            }
            FactorialKind::PlackettBurman => format!("{}-run Plackett-Burman", self.base_runs),
        }
    }
}

/// Power for detecting a factor effect (the difference between the mean
/// response at the high and low levels) in a replicated two-level design,
/// tested with a t test whose error degrees of freedom come from the
/// replicates and centre points. The sample size solved for is the number
/// of **replicates** of the base design.
///
/// The model has `1 + (base_runs − 1 − omitted_terms) + (blocks − 1)` terms
/// (+1 curvature term when there are centre points); error df = total runs
/// − terms. The standard error of an effect is `2σ/√(base_runs·replicates)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Factorial {
    /// The design.
    pub design: FactorialDesign,
    /// Standard deviation of the response (σ of the error).
    pub sigma: f64,
    /// Significance level (two-sided t test of the effect).
    pub alpha: f64,
    /// Number of centre points (added once to the design).
    pub center_points: usize,
    /// Number of blocks (≥ 1).
    pub blocks: usize,
    /// Number of effect terms left out of the model (their df go to error).
    pub omitted_terms: usize,
}

impl Factorial {
    /// Design with no centre points, one block and no omitted terms.
    pub fn new(design: FactorialDesign, sigma: f64, alpha: f64) -> Self {
        Factorial { design, sigma, alpha, center_points: 0, blocks: 1, omitted_terms: 0 }
    }

    fn model_terms(&self) -> Result<usize> {
        let b = self.design.base_runs;
        if self.blocks == 0 {
            return invalid("blocks must be at least 1");
        }
        if self.omitted_terms + 1 >= b {
            return invalid("too many omitted terms: no effect terms remain in the model");
        }
        Ok(1 + (b - 1 - self.omitted_terms) + (self.blocks - 1) + usize::from(self.center_points > 0))
    }

    /// Error degrees of freedom with `replicates` replicates.
    pub fn error_df(&self, replicates: usize) -> Result<i64> {
        let total = (self.design.base_runs * replicates + self.center_points) as i64;
        Ok(total - self.model_terms()? as i64)
    }

    /// Total number of runs with `replicates` replicates.
    pub fn total_runs(&self, replicates: usize) -> usize {
        self.design.base_runs * replicates + self.center_points
    }
}

impl PowerModel for Factorial {
    fn name(&self) -> String {
        format!("2-Level Factorial Design ({})", self.design.describe())
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Factors".into(), self.design.factors.to_string()),
            ("Base design runs".into(), self.design.base_runs.to_string()),
            ("Center points".into(), self.center_points.to_string()),
            ("Blocks".into(), self.blocks.to_string()),
            ("Terms omitted from model".into(), self.omitted_terms.to_string()),
            ("Standard deviation".into(), fmt(self.sigma)),
            ("Alpha".into(), fmt(self.alpha)),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Effect"
    }
    fn sample_size_label(&self) -> &'static str {
        "Replicates"
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn min_n(&self) -> Result<usize> {
        let mut r = 1;
        while self.error_df(r)? < 1 {
            r += 1;
            if r > 1_000_000 {
                return invalid("no number of replicates gives error degrees of freedom");
            }
        }
        Ok(r)
    }
    fn max_n(&self) -> usize {
        100_000
    }
    fn power(&self, replicates: usize, effect: f64) -> Result<f64> {
        check_alpha(self.alpha)?;
        check_pos("standard deviation", self.sigma)?;
        if replicates == 0 {
            return Err(StatsError::InsufficientData { what: "factorial power", needed: 1, got: 0 });
        }
        let df = self.error_df(replicates)?;
        if df < 1 {
            return invalid(format!("{replicates} replicate(s) leave {df} error degrees of freedom; need at least 1"));
        }
        let n_corner = (self.design.base_runs * replicates) as f64;
        let delta = effect.abs() / (2.0 * self.sigma / n_corner.sqrt());
        t_power(Alternative::TwoSided, self.alpha, df as f64, delta)
    }
    fn effect_range(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn effect_scale(&self) -> f64 {
        self.sigma
    }
}

// ---------------------------------------------------------------------------
// Equivalence (TOST)
// ---------------------------------------------------------------------------

/// Which equivalence design a [`Tost`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TostDesign {
    /// 1-sample: mean vs a target; effect = μ − target.
    OneSample,
    /// Paired: effect = mean difference; `sigma` = SD of the differences.
    Paired,
    /// 2-sample (equal variances, `n` per group): effect = μ₁ − μ₂.
    TwoSample,
}

/// Power of the two one-sided t tests (TOST) procedure: equivalence is
/// concluded when both `H₀: δ ≤ lower` and `H₀: δ ≥ upper` are rejected at
/// level α, δ being the true difference (the effect). Power is the exact
/// probability that both tests reject:
///
/// `P = E_W[ max(0, Φ(δ_U − t_c W) − Φ(δ_L + t_c W)) ]`,
/// `W = s/σ ~ √(χ²_ν/ν)`, `δ_U = (upper − δ)/(σ·a)`, `δ_L = (lower − δ)/(σ·a)`,
///
/// with `a = 1/√n` (one-sample, paired) or `√(2/n)` (two-sample), evaluated
/// by adaptive quadrature over the χ² variable.
///
/// When solving for the effect, the search runs from δ = 0 toward `upper`
/// ([`Direction::Increase`]) or toward `lower` ([`Direction::Decrease`]):
/// power falls as δ approaches a limit, so the solution is the largest
/// |δ| that still gives the requested power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tost {
    /// Design.
    pub design: TostDesign,
    /// Lower equivalence limit (on the difference; must be below `upper`).
    pub lower: f64,
    /// Upper equivalence limit.
    pub upper: f64,
    /// Standard deviation (of the data, or of the paired differences).
    pub sigma: f64,
    /// Level of each one-sided test.
    pub alpha: f64,
    /// Which limit the effect search heads toward.
    pub direction: Direction,
}

impl Tost {
    /// Convenience constructor with `direction = Increase`.
    pub fn new(design: TostDesign, lower: f64, upper: f64, sigma: f64, alpha: f64) -> Self {
        Tost { design, lower, upper, sigma, alpha, direction: Direction::Increase }
    }

    fn df_and_a(&self, n: usize) -> (f64, f64) {
        let nf = n as f64;
        match self.design {
            TostDesign::OneSample | TostDesign::Paired => (nf - 1.0, (1.0 / nf).sqrt()),
            TostDesign::TwoSample => (2.0 * nf - 2.0, (2.0 / nf).sqrt()),
        }
    }
}

impl PowerModel for Tost {
    fn name(&self) -> String {
        match self.design {
            TostDesign::OneSample => "Equivalence Test (TOST), 1-Sample",
            TostDesign::Paired => "Equivalence Test (TOST), Paired",
            TostDesign::TwoSample => "Equivalence Test (TOST), 2-Sample",
        }
        .into()
    }
    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("Lower equivalence limit".into(), fmt(self.lower)),
            ("Upper equivalence limit".into(), fmt(self.upper)),
            ("Standard deviation".into(), fmt(self.sigma)),
            ("Alpha (each one-sided test)".into(), fmt(self.alpha)),
        ]
    }
    fn effect_label(&self) -> &'static str {
        "Difference"
    }
    fn sample_size_label(&self) -> &'static str {
        match self.design {
            TostDesign::OneSample => "Sample Size",
            TostDesign::Paired => "Sample Size (pairs)",
            TostDesign::TwoSample => "Sample Size (per group)",
        }
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn power(&self, n: usize, diff: f64) -> Result<f64> {
        if !(0.0 < self.alpha && self.alpha < 0.5) {
            return invalid("TOST significance level must be in (0, 0.5)");
        }
        check_pos("standard deviation", self.sigma)?;
        if !self.lower.is_finite() || !self.upper.is_finite() || self.lower >= self.upper {
            return invalid("equivalence limits must be finite with lower < upper");
        }
        check_n(n, 2)?;
        let (nu, a) = self.df_and_a(n);
        let se = self.sigma * a;
        let (d_lo, d_hi) = ((self.lower - diff) / se, (self.upper - diff) / se);
        if d_hi <= d_lo {
            return Ok(0.0);
        }
        let tc = StudentT::new(nu)?.isf(self.alpha);
        let w_max = (d_hi - d_lo) / (2.0 * tc);
        let chi = ChiSquared::new(nu)?;
        let norm = Normal::standard();
        let u_lo = chi.quantile(1e-18).max(0.0).sqrt();
        let u_hi = (nu.sqrt() * w_max).min(chi.isf(1e-18).sqrt());
        if u_hi <= u_lo {
            return Ok(0.0);
        }
        let sqrt_nu = nu.sqrt();
        let mut f = |u: f64| {
            let w = u / sqrt_nu;
            let (lo, hi) = (d_lo + tc * w, d_hi - tc * w);
            if hi <= lo {
                return 0.0;
            }
            let prob = if lo + hi > 0.0 { norm.sf(lo) - norm.sf(hi) } else { norm.cdf(hi) - norm.cdf(lo) };
            prob * chi.pdf(u * u) * 2.0 * u
        };
        // Split so that a narrow χ² peak cannot fall between sample points.
        let panels = 8;
        let mut total = 0.0;
        for i in 0..panels {
            let a_i = u_lo + (u_hi - u_lo) * i as f64 / panels as f64;
            let b_i = u_lo + (u_hi - u_lo) * (i + 1) as f64 / panels as f64;
            total += adaptive(&mut f, a_i, b_i, 1e-15, 1e-11)?;
        }
        Ok(total.clamp(0.0, 1.0))
    }
    fn effect_range(&self) -> (f64, f64) {
        match self.direction {
            Direction::Increase => (0.0, self.upper),
            Direction::Decrease => (0.0, self.lower),
        }
    }
    fn effect_scale(&self) -> f64 {
        self.sigma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z() -> OneSampleZ {
        OneSampleZ { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided }
    }

    #[test]
    fn z_test_textbook_values() {
        // n = (z_{.975}+z_{.8})²/0.25 = 31.4 → 32 (exact two-sided power .8074).
        let s = z().solve(None, Some(0.8), Some(0.5)).unwrap();
        assert_eq!(s.sample_size, 32);
        assert!((s.power - 0.8074).abs() < 5e-4, "{}", s.power);
        // Power 1-sided: Φ(δ√n − z_α).
        let one = OneSampleZ { alternative: Alternative::Greater, ..z() };
        let p = one.power(25, 0.5).unwrap();
        assert!((p - Normal::standard().cdf(2.5 - 1.6448536269514722)).abs() < 1e-14);
        // Effect: round trip.
        let e = z().solve(Some(32), Some(0.9), None).unwrap();
        assert!((z().power(32, e.effect).unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn t_test_values_match_gpower() {
        let t1 = OneSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
        assert_eq!(t1.sample_size(0.5, 0.8).unwrap(), 34);
        let t2 = TwoSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
        assert_eq!(t2.sample_size(0.5, 0.8).unwrap(), 64);
        assert_eq!(t2.sample_size(0.8, 0.8).unwrap(), 26);
        let t1g = OneSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::Greater };
        assert_eq!(t1g.sample_size(0.5, 0.8).unwrap(), 27);
        let pr = PairedT { sigma_diff: 2.0, alpha: 0.05, alternative: Alternative::TwoSided };
        assert_eq!(pr.sample_size(1.0, 0.8).unwrap(), 34);
        // Power at the answer vs one below brackets the target.
        assert!(t2.power(64, 0.5).unwrap() >= 0.8 && t2.power(63, 0.5).unwrap() < 0.8);
    }

    #[test]
    fn solve_requires_exactly_one_unknown() {
        assert!(z().solve(Some(10), Some(0.8), Some(0.5)).is_err());
        assert!(z().solve(None, None, Some(0.5)).is_err());
        assert!(z().solve(Some(10), None, None).is_err());
        assert!(z().solve(None, Some(1.0), Some(0.5)).is_err());
        assert!(z().solve(None, Some(0.8), Some(0.0)).is_err()); // zero effect never reaches power
    }

    #[test]
    fn effect_solver_handles_unattainable_power() {
        // Power at the null is α, so a target below α cannot be attained.
        let t = OneSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
        assert!(t.effect(10, 0.01).is_err());
        let e = t.effect(10, 0.8).unwrap();
        assert!((t.power(10, e).unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn two_sample_t_power_is_nct_identity() {
        // The two-sided power must equal P(T' > c) + P(T' < −c) computed directly.
        let t2 = TwoSampleT { sigma: 2.0, alpha: 0.01, alternative: Alternative::TwoSided };
        let (n, d) = (15usize, 3.0);
        let nc = NonCentralT::new(28.0, d / (2.0 * (2.0f64 / 15.0).sqrt())).unwrap();
        let c = StudentT::new(28.0).unwrap().quantile(0.995);
        assert!((t2.power(n, d).unwrap() - (1.0 - nc.cdf(c) + nc.cdf(-c))).abs() < 1e-12);
        // At zero effect power equals α.
        assert!((t2.power(n, 0.0).unwrap() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn normal_proportion_power_formulas() {
        let m = OneProportion::new(0.5, 0.05, Alternative::TwoSided, DiscreteMethod::NormalApproximation);
        let p = m.power(100, 0.6).unwrap();
        let norm = Normal::standard();
        let (s0, s1) = (0.05f64, (0.24f64 / 100.0).sqrt());
        let z = 1.959963984540054;
        let expect = norm.sf((z * s0 - 0.1) / s1) + norm.sf((z * s0 + 0.1) / s1);
        assert!((p - expect).abs() < 1e-13);
        let sol = m.solve(None, Some(0.8), Some(0.6)).unwrap();
        assert!(sol.power >= 0.8 && m.power(sol.sample_size - 1, 0.6).unwrap() < 0.8);
    }

    #[test]
    fn exact_one_proportion_power_is_sawtoothed_and_scanned() {
        let m = OneProportion::new(0.5, 0.05, Alternative::Greater, DiscreteMethod::Exact);
        let n = m.sample_size(0.7, 0.8).unwrap();
        assert!(m.power(n, 0.7).unwrap() >= 0.8);
        for k in 1..n {
            assert!(m.power(k, 0.7).unwrap() < 0.8, "n={k} already reaches the target");
        }
        // Actual size never exceeds α.
        for k in [5usize, 10, 20, 50, 100] {
            assert!(m.actual_alpha(k).unwrap() <= 0.05 + 1e-12);
        }
        // Brute-force power from the full p-value table.
        let k = 17u64;
        let mut brute = 0.0;
        let b1 = Binomial::new(k, 0.7).unwrap();
        for x in 0..=k {
            if binomial_p(k, 0.5, x, Alternative::Greater).unwrap() <= 0.05 {
                brute += b1.pmf(x as f64);
            }
        }
        assert!((m.power(k as usize, 0.7).unwrap() - brute).abs() < 1e-13);
    }

    #[test]
    fn exact_two_sided_region_matches_brute_force() {
        let m = OneProportion::new(0.3, 0.05, Alternative::TwoSided, DiscreteMethod::Exact);
        for k in [8u64, 25, 60] {
            let b1 = Binomial::new(k, 0.45).unwrap();
            let brute: f64 = (0..=k)
                .filter(|&x| binomial_p(k, 0.3, x, Alternative::TwoSided).unwrap() <= 0.05)
                .map(|x| b1.pmf(x as f64))
                .sum();
            assert!((m.power(k as usize, 0.45).unwrap() - brute).abs() < 1e-13, "k={k}");
        }
    }

    #[test]
    fn two_proportions_normal_and_fisher() {
        let nm = TwoProportions::new(0.4, 0.05, Alternative::TwoSided, DiscreteMethod::NormalApproximation);
        let n = nm.sample_size(0.6, 0.8).unwrap();
        assert!(nm.power(n, 0.6).unwrap() >= 0.8 && nm.power(n - 1, 0.6).unwrap() < 0.8);
        // Fisher power at small n against full table enumeration with the
        // library's own Fisher p-value routine.
        let fm = TwoProportions::new(0.2, 0.05, Alternative::TwoSided, DiscreteMethod::Exact);
        let pw = fm.power(12, 0.7).unwrap();
        let brute = fisher_brute(12, 0.7, 0.2, 0.05);
        assert!((pw - brute).abs() < 1e-12, "{pw} vs {brute}");
        // Fisher is conservative relative to the normal approximation.
        let nmf = TwoProportions::new(0.2, 0.05, Alternative::TwoSided, DiscreteMethod::NormalApproximation);
        assert!(pw <= nmf.power(12, 0.7).unwrap() + 0.05);
    }

    /// Enumerate every 2×2 table and use the crate's two-proportion Fisher
    /// p-value to build the rejection region.
    fn fisher_brute(n: u64, p1: f64, p2: f64, alpha: f64) -> f64 {
        let b1 = Binomial::new(n, p1).unwrap();
        let b2 = Binomial::new(n, p2).unwrap();
        let mut s = 0.0;
        for x1 in 0..=n {
            for x2 in 0..=n {
                let r =
                    crate::proportion::two_proportions(x1, n, x2, n, 0.0, Alternative::TwoSided, 0.95, false).unwrap();
                if r.p_value_fisher.unwrap() <= alpha {
                    s += b1.pmf(x1 as f64) * b2.pmf(x2 as f64);
                }
            }
        }
        s
    }

    #[test]
    fn fisher_one_sided_power_matches_brute_force() {
        let fm = TwoProportions::new(0.3, 0.05, Alternative::Greater, DiscreteMethod::Exact);
        let n = 9u64;
        let b1 = Binomial::new(n, 0.8).unwrap();
        let b2 = Binomial::new(n, 0.3).unwrap();
        let mut brute = 0.0;
        for x1 in 0..=n {
            for x2 in 0..=n {
                let r =
                    crate::proportion::two_proportions(x1, n, x2, n, 0.0, Alternative::Greater, 0.95, false).unwrap();
                if r.p_value_fisher.unwrap() <= 0.05 {
                    brute += b1.pmf(x1 as f64) * b2.pmf(x2 as f64);
                }
            }
        }
        assert!((fm.power(9, 0.8).unwrap() - brute).abs() < 1e-12);
    }

    #[test]
    fn poisson_power_exact_and_normal() {
        let ex = OnePoisson::new(2.0, 0.05, Alternative::Greater, DiscreteMethod::Exact);
        // n units of length 1: total ~ Poisson(2n); brute-force region.
        let n = 6usize;
        let mu0 = 2.0 * n as f64;
        let d1 = Poisson::new(3.0 * n as f64).unwrap();
        let brute: f64 = (0..400u64)
            .filter(|&x| poisson_p(mu0, x, Alternative::Greater).unwrap() <= 0.05)
            .map(|x| d1.pmf(x as f64))
            .sum();
        assert!((ex.power(n, 3.0).unwrap() - brute).abs() < 1e-12);
        let nm = OnePoisson::new(2.0, 0.05, Alternative::Greater, DiscreteMethod::NormalApproximation);
        let n_n = nm.sample_size(3.0, 0.9).unwrap();
        assert!(nm.power(n_n, 3.0).unwrap() >= 0.9);
        let sol = ex.solve(None, Some(0.9), Some(3.0)).unwrap();
        assert!(sol.power >= 0.9);
        // Decreasing alternative.
        let dec = OnePoisson::new(5.0, 0.05, Alternative::Less, DiscreteMethod::Exact);
        assert!(dec.power(4, 2.0).unwrap() > 0.5);
    }

    #[test]
    fn two_poisson_exact_matches_brute_force() {
        let m = TwoPoisson::new(3.0, 0.05, Alternative::TwoSided, DiscreteMethod::Exact);
        let n = 3usize;
        let (t, l1, l2) = (n as f64, 6.0, 3.0);
        let (d1, d2) = (Poisson::new(l1 * t).unwrap(), Poisson::new(l2 * t).unwrap());
        let mut brute = 0.0;
        for x1 in 0..120u64 {
            for x2 in 0..120u64 {
                if x1 + x2 == 0 {
                    continue;
                }
                let p = crate::poisson::two_sample_poisson(x1, t, x2, t, 0.0, Alternative::TwoSided, 0.95)
                    .unwrap()
                    .p_value_exact
                    .unwrap();
                if p <= 0.05 {
                    brute += d1.pmf(x1 as f64) * d2.pmf(x2 as f64);
                }
            }
        }
        assert!((m.power(n, l1).unwrap() - brute).abs() < 1e-9, "{} vs {brute}", m.power(n, l1).unwrap());
        let nm = TwoPoisson::new(3.0, 0.05, Alternative::TwoSided, DiscreteMethod::NormalApproximation);
        assert!(nm.sample_size(6.0, 0.8).unwrap() > 0);
    }

    #[test]
    fn variance_power_identities() {
        let v1 = OneVariance { alpha: 0.05, alternative: Alternative::Greater, direction: Direction::Increase };
        // ratio 1 → power α.
        assert!((v1.power(20, 1.0).unwrap() - 0.05).abs() < 1e-12);
        let s = v1.solve(None, Some(0.9), Some(2.0)).unwrap();
        assert!(s.power >= 0.9 && v1.power(s.sample_size - 1, 2.0).unwrap() < 0.9);
        // Explicit χ² formula.
        let d = ChiSquared::new(19.0).unwrap();
        assert!((v1.power(20, 2.0).unwrap() - d.sf(d.isf(0.05) / 2.0)).abs() < 1e-14);
        let two = TwoVariances { alpha: 0.05, alternative: Alternative::TwoSided, direction: Direction::Increase };
        assert!((two.power(15, 1.0).unwrap() - 0.05).abs() < 1e-12);
        // Symmetry: ratio ρ and 1/ρ give the same two-sided power with equal n.
        assert!((two.power(15, 3.0).unwrap() - two.power(15, 1.0 / 3.0).unwrap()).abs() < 1e-12);
        // Effect solving on the decreasing side.
        let dec = TwoVariances { direction: Direction::Decrease, ..two };
        let e = dec.effect(15, 0.8).unwrap();
        assert!(e < 1.0 && (dec.power(15, e).unwrap() - 0.8).abs() < 1e-9);
        let inc = two.effect(15, 0.8).unwrap();
        assert!((inc * e - 1.0).abs() < 1e-7, "{inc} {e}");
    }

    #[test]
    fn anova_power_values() {
        // Cohen f = 0.25, k = 4: total N = 180 → n = 45 (G*Power); means ±0.25.
        let a = OneWayAnova { groups: 4, sigma: 1.0, alpha: 0.05 };
        let means = [-0.25, -0.25, 0.25, 0.25];
        assert!(a.power_for_means(45, &means).unwrap() > 0.80);
        assert!(a.power_for_means(44, &means).unwrap() < 0.80);
        // Max-difference parametrisation: δ = 0.5 with the rest at the centre
        // has Σ(μ−μ̄)² = δ²/2 = 0.125.
        let lam = 45.0 * 0.125;
        let direct = NonCentralF::new(3.0, 176.0, lam).unwrap().sf(FisherF::new(3.0, 176.0).unwrap().isf(0.05));
        assert!((a.power(45, 0.5).unwrap() - direct).abs() < 1e-13);
        // Two groups reduce to the two-sample t test.
        let a2 = OneWayAnova { groups: 2, sigma: 1.0, alpha: 0.05 };
        let t2 = TwoSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
        assert!((a2.power(20, 0.8).unwrap() - t2.power(20, 0.8).unwrap()).abs() < 1e-9);
        assert_eq!(a2.sample_size(0.5, 0.8).unwrap(), 64);
        assert!(a.power_for_means(10, &[0.0, 1.0]).is_err());
        let e = a.solve(Some(20), Some(0.8), None).unwrap();
        assert!((a.power(20, e.effect).unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn factorial_degrees_of_freedom_and_power() {
        // 2³ full factorial, 2 replicates: 16 runs, 8 terms → 8 error df.
        let f = Factorial::new(FactorialDesign::full(3).unwrap(), 1.0, 0.05);
        assert_eq!(f.error_df(2).unwrap(), 8);
        assert_eq!(f.error_df(1).unwrap(), 0);
        assert_eq!(f.min_n().unwrap(), 2);
        // Direct non-central t evaluation: SE = 2σ/√16 = 0.5, effect 1 → δ = 2.
        let t = StudentT::new(8.0).unwrap().quantile(0.975);
        let nc = NonCentralT::new(8.0, 2.0).unwrap();
        assert!((f.power(2, 1.0).unwrap() - (nc.sf(t) + nc.cdf(-t))).abs() < 1e-13);
        // One replicate works once terms are omitted or centre points added.
        let g = Factorial { omitted_terms: 3, ..f };
        assert_eq!(g.error_df(1).unwrap(), 3);
        assert_eq!(g.min_n().unwrap(), 1);
        let c = Factorial { center_points: 4, ..f };
        assert_eq!(c.error_df(1).unwrap(), 8 + 4 - 9);
        // Solving for replicates.
        let s = f.solve(None, Some(0.9), Some(1.5)).unwrap();
        assert!(s.power >= 0.9 && f.power(s.sample_size - 1, 1.5).map_or(true, |p| p < 0.9));
        assert!(f.power(1, 1.0).is_err());
    }

    #[test]
    fn plackett_burman_and_fractional_designs() {
        let pb = FactorialDesign::plackett_burman(12, 11).unwrap();
        assert_eq!(pb.base_runs, 12);
        let m = Factorial::new(pb, 1.0, 0.05);
        // 12 runs, all 11 main effects + constant → 0 df with 1 replicate.
        assert_eq!(m.error_df(1).unwrap(), 0);
        assert_eq!(m.error_df(2).unwrap(), 12);
        let fr = FactorialDesign::fractional(5, 1).unwrap();
        assert_eq!(fr.base_runs, 16);
        assert!(FactorialDesign::fractional(7, 3).is_ok()); // 2^(7-3): 16 runs, 7 factors
        assert!(FactorialDesign::fractional(4, 2).is_err()); // 2^(4-2) = 4 runs cannot hold 4 main effects
        assert!(FactorialDesign::fractional(5, 4).is_err()); // k - p < 2
        assert!(FactorialDesign::fractional(3, 1).is_ok());
        assert!(FactorialDesign::plackett_burman(10, 3).is_err());
        assert!(FactorialDesign::plackett_burman(12, 12).is_err());
        assert!(FactorialDesign::full(1).is_err());
        let sol =
            Factorial { omitted_terms: 0, ..Factorial::new(fr, 2.0, 0.05) }.solve(Some(2), Some(0.8), None).unwrap();
        assert!(sol.effect > 0.0);
    }

    #[test]
    fn tost_power_is_exact_and_consistent() {
        let t = Tost::new(TostDesign::OneSample, -1.0, 1.0, 1.0, 0.05);
        // Large n, diff 0: power → 1.
        assert!(t.power(200, 0.0).unwrap() > 0.999);
        // Power at the limit is below α (≤ α in the limit).
        assert!(t.power(50, 1.0).unwrap() < 0.05);
        // Symmetry in the difference for symmetric limits.
        assert!((t.power(20, 0.3).unwrap() - t.power(20, -0.3).unwrap()).abs() < 1e-12);
        // Monotone in n.
        assert!(t.power(15, 0.2).unwrap() < t.power(25, 0.2).unwrap());
        // Large-df limit: normal approximation Φ(δ_U−z) − Φ(δ_L+z).
        let big = t.power(100_000, 0.0).unwrap();
        assert!(big > 0.9999);
        let sol = t.solve(None, Some(0.9), Some(0.0)).unwrap();
        assert!(sol.power >= 0.9 && t.power(sol.sample_size - 1, 0.0).unwrap() < 0.9);
        let e = t.solve(Some(30), Some(0.8), None).unwrap();
        assert!(e.effect > 0.0 && e.effect < 1.0 && (t.power(30, e.effect).unwrap() - 0.8).abs() < 1e-8);
        let dec = Tost { direction: Direction::Decrease, ..t };
        let ed = dec.effect(30, 0.8).unwrap();
        assert!((ed + e.effect).abs() < 1e-8);
        // Paired == one-sample, two-sample uses df 2n−2 and √(2/n).
        let p = Tost::new(TostDesign::Paired, -1.0, 1.0, 1.0, 0.05);
        assert!((p.power(20, 0.3).unwrap() - t.power(20, 0.3).unwrap()).abs() < 1e-15);
        let two = Tost::new(TostDesign::TwoSample, -1.0, 1.0, 1.0, 0.05);
        assert!(two.power(20, 0.0).unwrap() < t.power(20, 0.0).unwrap());
        // Invalid inputs.
        assert!(Tost::new(TostDesign::OneSample, 1.0, -1.0, 1.0, 0.05).power(10, 0.0).is_err());
        assert!(Tost::new(TostDesign::OneSample, -1.0, 1.0, 1.0, 0.6).power(10, 0.0).is_err());
    }

    #[test]
    fn tost_power_matches_simulation() {
        use openstats_math::random::Pcg64;
        let t = Tost::new(TostDesign::TwoSample, -0.8, 1.0, 1.5, 0.05);
        let (n, diff) = (12usize, 0.1);
        let norm = Normal::standard();
        let mut rng = Pcg64::with_stream(5, 9);
        let reps = 60_000;
        let mut ok = 0u32;
        for _ in 0..reps {
            let a = norm.sample_n(&mut rng, n);
            let b = norm.sample_n(&mut rng, n);
            let ma = a.iter().sum::<f64>() / n as f64 * 1.5 + diff;
            let mb = b.iter().sum::<f64>() / n as f64 * 1.5;
            let va = a.iter().map(|v| (v - a.iter().sum::<f64>() / n as f64).powi(2)).sum::<f64>() * 2.25;
            let vb = b.iter().map(|v| (v - b.iter().sum::<f64>() / n as f64).powi(2)).sum::<f64>() * 2.25;
            let sp = ((va + vb) / (2.0 * n as f64 - 2.0)).sqrt();
            let se = sp * (2.0 / n as f64).sqrt();
            let d = ma - mb;
            let tc = StudentT::new(2.0 * n as f64 - 2.0).unwrap().isf(0.05);
            if (d - (-0.8)) / se > tc && (1.0 - d) / se > tc {
                ok += 1;
            }
        }
        let sim = f64::from(ok) / f64::from(reps);
        let exact = t.power(n, diff).unwrap();
        assert!((sim - exact).abs() < 0.006, "sim {sim} exact {exact}");
    }
}
