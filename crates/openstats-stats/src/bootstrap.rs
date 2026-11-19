//! Nonparametric bootstrap with percentile, basic, normal and bias-corrected accelerated (BCa)
//! confidence intervals.

use crate::error::{invalid, Result, StatsError};
use crate::quantile::{quantile_sorted, QuantileMethod};
use openstats_math::dist::{ContinuousDistribution, Normal};
use openstats_math::random::{Pcg64, Rng};

/// Interval method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalMethod {
    /// Percentiles of the replicates.
    Percentile,
    /// Basic (reverse percentile): `2θ̂ − q`.
    Basic,
    /// `θ̂ ± z·SE` with the bootstrap standard error.
    Normal,
    /// Bias-corrected and accelerated.
    Bca,
}

/// A confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Method.
    pub method: IntervalMethod,
    /// Lower limit.
    pub lower: f64,
    /// Upper limit.
    pub upper: f64,
}

/// Bootstrap options.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapOptions {
    /// Number of resamples (≥ 20).
    pub resamples: usize,
    /// Confidence level in (0, 1).
    pub level: f64,
    /// Generator seed.
    pub seed: u64,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        BootstrapOptions { resamples: 2000, level: 0.95, seed: 1 }
    }
}

/// Bootstrap result.
#[derive(Debug, Clone, PartialEq)]
pub struct Bootstrap {
    /// Statistic on the original data.
    pub estimate: f64,
    /// Replicates (non-finite ones are dropped and counted in `dropped`).
    pub replicates: Vec<f64>,
    /// Number of non-finite replicates that were discarded.
    pub dropped: usize,
    /// Mean of the replicates minus the estimate.
    pub bias: f64,
    /// Standard deviation of the replicates.
    pub std_error: f64,
    /// Percentile, basic, normal and BCa intervals (BCa is absent when it cannot be computed).
    pub intervals: Vec<Interval>,
    /// Bias-correction constant z₀ and acceleration of the BCa interval (when computed).
    pub bca_constants: Option<(f64, f64)>,
}

fn sorted_quantile(sorted: &[f64], p: f64) -> f64 {
    quantile_sorted(sorted, p, QuantileMethod::Type6).unwrap_or(f64::NAN)
}

/// Intervals from replicates, the original estimate and the jackknife values (used only by BCa).
pub fn intervals_from_replicates(estimate: f64, replicates: &[f64], jackknife: &[f64], level: f64) -> Vec<Interval> {
    let mut s: Vec<f64> = replicates.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let a = (1.0 - level) / 2.0;
    let nd = Normal::standard();
    let (lo, hi) = (sorted_quantile(&s, a), sorted_quantile(&s, 1.0 - a));
    let mut out = vec![Interval { method: IntervalMethod::Percentile, lower: lo, upper: hi }];
    out.push(Interval { method: IntervalMethod::Basic, lower: 2.0 * estimate - hi, upper: 2.0 * estimate - lo });
    let n = s.len() as f64;
    let mean = s.iter().sum::<f64>() / n;
    let se = (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let z = nd.quantile(1.0 - a);
    out.push(Interval { method: IntervalMethod::Normal, lower: estimate - z * se, upper: estimate + z * se });
    if let Some((z0, acc)) = bca_constants(estimate, &s, jackknife) {
        let adj = |zq: f64| nd.cdf(z0 + (z0 + zq) / (1.0 - acc * (z0 + zq)));
        let (pl, pu) = (adj(nd.quantile(a)), adj(nd.quantile(1.0 - a)));
        out.push(Interval {
            method: IntervalMethod::Bca,
            lower: sorted_quantile(&s, pl),
            upper: sorted_quantile(&s, pu),
        });
    }
    out
}

/// Bias correction z₀ (from the share of replicates below the estimate) and acceleration (from
/// the jackknife), or `None` when either is undefined.
pub fn bca_constants(estimate: f64, sorted: &[f64], jackknife: &[f64]) -> Option<(f64, f64)> {
    let n = sorted.len() as f64;
    let below = sorted.iter().filter(|&&v| v < estimate).count() as f64
        + 0.5 * sorted.iter().filter(|&&v| v == estimate).count() as f64;
    let p = below / n;
    if !(p > 0.0 && p < 1.0) || jackknife.len() < 2 {
        return None;
    }
    let z0 = Normal::standard().quantile(p);
    let jm = jackknife.iter().sum::<f64>() / jackknife.len() as f64;
    let (mut s2, mut s3) = (0.0, 0.0);
    for &j in jackknife {
        let d = jm - j;
        s2 += d * d;
        s3 += d * d * d;
    }
    if s2 == 0.0 {
        return None;
    }
    Some((z0, s3 / (6.0 * s2.powf(1.5))))
}

/// Bootstrap a statistic that is a function of resampled row indices, one group of indices per
/// stratum (a single stratum for a one-sample problem; resampling is done within each stratum).
///
/// `strata` gives the number of rows in each stratum; the statistic receives, for each stratum,
/// the indices (`0..size`) of the resampled rows.
pub fn bootstrap<F>(strata: &[usize], stat: F, opts: &BootstrapOptions) -> Result<Bootstrap>
where
    F: Fn(&[Vec<usize>]) -> f64,
{
    if strata.is_empty() || strata.iter().any(|&n| n < 2) {
        return invalid("every sample needs at least 2 observations to bootstrap");
    }
    if opts.resamples < 20 {
        return invalid("use at least 20 resamples");
    }
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    if opts.resamples > 1_000_000 {
        return Err(StatsError::TooLarge("more than 1,000,000 resamples".into()));
    }
    let ident: Vec<Vec<usize>> = strata.iter().map(|&n| (0..n).collect()).collect();
    let estimate = stat(&ident);
    if !estimate.is_finite() {
        return invalid("the statistic is not defined for the original data");
    }
    let mut rng = Pcg64::new(u128::from(opts.seed), 0x5851_F42D_4C95_7F2D);
    let mut reps = Vec::with_capacity(opts.resamples);
    let mut dropped = 0;
    for _ in 0..opts.resamples {
        let sample: Vec<Vec<usize>> =
            strata.iter().map(|&n| (0..n).map(|_| rng.below(n as u64) as usize).collect()).collect();
        let v = stat(&sample);
        if v.is_finite() {
            reps.push(v);
        } else {
            dropped += 1;
        }
    }
    if reps.len() < 20 {
        return invalid("too many resamples gave an undefined statistic");
    }
    // Jackknife: leave out each row in turn.
    let mut jack = Vec::new();
    for (si, &n) in strata.iter().enumerate() {
        for leave in 0..n {
            let mut s = ident.clone();
            s[si] = (0..n).filter(|&i| i != leave).collect();
            let v = stat(&s);
            if v.is_finite() {
                jack.push(v);
            }
        }
    }
    let nrep = reps.len() as f64;
    let mean = reps.iter().sum::<f64>() / nrep;
    let se = (reps.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nrep - 1.0)).sqrt();
    let intervals = intervals_from_replicates(estimate, &reps, &jack, opts.level);
    let mut sorted = reps.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let constants = bca_constants(estimate, &sorted, &jack);
    Ok(Bootstrap {
        estimate,
        bias: mean - estimate,
        std_error: se,
        replicates: reps,
        dropped,
        intervals,
        bca_constants: constants,
    })
}
