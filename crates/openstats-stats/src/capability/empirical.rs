//! Non-normal capability from empirical percentiles.
//!
//! `Pp = (USL − LSL)/(P99.865 − P0.135)`, `Ppl = (median − LSL)/(median − P0.135)`,
//! `Ppu = (USL − median)/(P99.865 − median)`, `Ppk = min(Ppl, Ppu)`; the
//! percentiles are sample quantiles (Minitab's type 6 by default). No
//! distribution is assumed, so the expected performance is the observed one.

use crate::error::{invalid, Result, StatsError};
use crate::quantile::{quantile_sorted, QuantileMethod};
use crate::spc::capability::{observed, ObservedPerformance, Specs};
use openstats_math::special::erf::norm_cdf;

/// Options of an empirical-percentile capability analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalOptions {
    /// Sample-quantile definition (type 6).
    pub method: QuantileMethod,
    /// When the sample is too small for the 0.135 % / 99.865 % points to lie
    /// between order statistics, `false` (default) is an error and `true` uses
    /// the sample minimum / maximum instead and records a note.
    pub allow_extremes: bool,
}

impl Default for EmpiricalOptions {
    fn default() -> Self {
        EmpiricalOptions { method: QuantileMethod::Type6, allow_extremes: false }
    }
}

/// Result of an empirical-percentile capability analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalCapability {
    /// Observations used.
    pub n: usize,
    /// Specifications.
    pub specs: Specs,
    /// Sample median.
    pub median: f64,
    /// 0.135 percentile.
    pub x_low: f64,
    /// 99.865 percentile.
    pub x_high: f64,
    /// Pp.
    pub pp: Option<f64>,
    /// PPL.
    pub ppl: Option<f64>,
    /// PPU.
    pub ppu: Option<f64>,
    /// Ppk.
    pub ppk: f64,
    /// Observed performance (also the performance estimate).
    pub observed: ObservedPerformance,
    /// Notes (for example that a percentile was replaced by an extreme).
    pub notes: Vec<String>,
}

/// Empirical-percentile capability of `data`.
pub fn capability_empirical(data: &[f64], specs: &Specs, opts: &EmpiricalOptions) -> Result<EmpiricalCapability> {
    specs.validate()?;
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let mut x: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "empirical capability", needed: 3, got: n });
    }
    x.sort_by(|a, b| a.total_cmp(b));
    let p = norm_cdf(-3.0);
    let mut notes = Vec::new();
    // The type 6 position p(n + 1) must be at least 1 (and at most n) for the
    // percentile to be interpolated between order statistics; other definitions
    // use their own positions, so the check uses the order-statistic count only.
    let interior = (n as f64 + 1.0) * p >= 1.0;
    if !interior {
        let needed = (1.0 / p).ceil() as usize - 1;
        if !opts.allow_extremes {
            return invalid(format!(
                "n = {n} is too small for the 0.135 and 99.865 percentiles (at least {needed} observations are needed); \
                 set allow_extremes to use the sample minimum and maximum"
            ));
        }
        notes.push(format!(
            "n = {n} is below {needed}: the sample minimum and maximum stand in for the 0.135 and 99.865 percentiles"
        ));
    }
    let (lo, hi) = if interior {
        (quantile_sorted(&x, p, opts.method)?, quantile_sorted(&x, 1.0 - p, opts.method)?)
    } else {
        (x[0], x[n - 1])
    };
    let med = quantile_sorted(&x, 0.5, opts.method)?;
    if !(hi > lo && med > lo && hi > med) {
        return invalid("the sample percentiles coincide; capability is undefined");
    }
    let ppl = specs.lsl.map(|l| (med - l) / (med - lo));
    let ppu = specs.usl.map(|u| (u - med) / (hi - med));
    let pp = match (specs.lsl, specs.usl) {
        (Some(l), Some(u)) => Some((u - l) / (hi - lo)),
        _ => None,
    };
    let ppk = match (ppl, ppu) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => f64::NAN,
    };
    Ok(EmpiricalCapability {
        n,
        specs: *specs,
        median: med,
        x_low: lo,
        x_high: hi,
        pp,
        ppl,
        ppu,
        ppk,
        observed: observed(&x, specs),
        notes,
    })
}
