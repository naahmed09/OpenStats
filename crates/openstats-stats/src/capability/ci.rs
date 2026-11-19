//! Confidence intervals for the capability and performance indices.
//!
//! Notation: `n` observations, `ν` the degrees of freedom of the σ estimate
//! (`n − 1` for the overall σ), `α = 1 − level`.
//!
//! * **Cp, Pp** (two-sided): `Ĉ·√(χ²_{α/2,ν}/ν)` to `Ĉ·√(χ²_{1−α/2,ν}/ν)` — exact
//!   when σ̂² ~ σ²χ²_ν/ν.
//! * **Cpl, Cpu, Ppl, Ppu** (one-sided indices): with `T = 3√n·Ĉ` the pivot is
//!   non-central t, `T ~ t_ν(δ)` with `δ = 3√n·C`. The limits are the
//!   `δ_L`, `δ_U` solving `P(T ≤ t_obs; δ_L) = 1 − α/2` and
//!   `P(T ≤ t_obs; δ_U) = α/2`, divided by `3√n`. Exact for the overall σ; for a
//!   within σ they are exact when σ̂_within² is a scaled χ²_ν (pooled standard
//!   deviation) and a close approximation otherwise.
//! * **Cpk, Ppk**: Bissell (1990), `Ĉpk ± z·√(1/(9n) + Ĉpk²/(2ν))`. There is no
//!   exact interval for the two-sided Cpk; this is an approximation.
//! * **Cpm**: Boyles' (1991) χ² approximation with the Patnaik-matched degrees
//!   of freedom `ν = n(1 + a²)²/(1 + 2a²)`, `a = (x̄ − T)/s`.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::spc::capability::NormalCapability;
use openstats_math::dist::{ChiSquared, ContinuousDistribution, NonCentralT};
use openstats_math::roots::brent;
use openstats_math::special::erf::norm_ppf;

/// Confidence intervals for the capability indices of a [`NormalCapability`].
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityIntervals {
    /// Confidence level.
    pub level: f64,
    /// Degrees of freedom of the within-σ estimate used (`None`: within intervals not computed).
    pub within_df: Option<f64>,
    /// Cp.
    pub cp: Option<Interval>,
    /// Cpl.
    pub cpl: Option<Interval>,
    /// Cpu.
    pub cpu: Option<Interval>,
    /// Cpk.
    pub cpk: Option<Interval>,
    /// Pp.
    pub pp: Option<Interval>,
    /// Ppl.
    pub ppl: Option<Interval>,
    /// Ppu.
    pub ppu: Option<Interval>,
    /// Ppk.
    pub ppk: Interval,
    /// Cpm.
    pub cpm: Option<Interval>,
}

/// Degrees of freedom of the pooled within-subgroup standard deviation, `N − k`.
pub fn pooled_within_df(groups: &[Vec<f64>]) -> Result<f64> {
    let mut n = 0usize;
    for g in groups {
        let m = g.iter().filter(|v| !v.is_nan()).count();
        if m < 2 {
            return invalid("every subgroup needs at least 2 observations");
        }
        n += m;
    }
    Ok((n - groups.len()) as f64)
}

fn check_level(level: f64) -> Result<()> {
    if !(level > 0.0 && level < 1.0) {
        return invalid(format!("confidence level {level} must be in (0, 1)"));
    }
    Ok(())
}

/// χ² interval for a two-sided index.
fn chi_interval(c: f64, df: f64, level: f64) -> Result<Interval> {
    let chi = ChiSquared::new(df)?;
    let a = 1.0 - level;
    Ok(Interval {
        lower: c * (chi.quantile(a / 2.0) / df).sqrt(),
        upper: c * (chi.quantile(1.0 - a / 2.0) / df).sqrt(),
        level,
    })
}

/// Bissell's normal approximation for Cpk / Ppk.
fn bissell(c: f64, n: usize, df: f64, level: f64) -> Interval {
    let z = norm_ppf(1.0 - (1.0 - level) / 2.0);
    let half = z * (1.0 / (9.0 * n as f64) + c * c / (2.0 * df)).sqrt();
    Interval { lower: c - half, upper: c + half, level }
}

/// Solve `P(T ≤ t; δ) = p` for the non-centrality δ (the cdf decreases in δ).
fn nct_delta(t: f64, df: f64, p: f64) -> Result<f64> {
    let f = |d: f64| -> f64 {
        match NonCentralT::new(df, d) {
            Ok(dist) => dist.cdf(t) - p,
            Err(_) => f64::NAN,
        }
    };
    let mut width = 8.0;
    loop {
        let (a, b) = (t - width, t + width);
        let (fa, fb) = (f(a), f(b));
        if fa.is_nan() || fb.is_nan() {
            return Err(StatsError::InvalidArgument("non-central t evaluation failed".into()));
        }
        if fa > 0.0 && fb < 0.0 {
            return Ok(brent(f, a, b, 1e-12, 500)?);
        }
        width *= 2.0;
        if width > 4096.0 {
            return Err(StatsError::InvalidArgument(
                "could not bracket the non-centrality parameter for the confidence limit".into(),
            ));
        }
    }
}

/// Exact confidence limits of a one-sided index `c_hat` by inverting the non-central t.
fn exact_one_sided(c_hat: f64, n: usize, df: f64, level: f64) -> Result<Interval> {
    let k = 3.0 * (n as f64).sqrt();
    let t = k * c_hat;
    let a = 1.0 - level;
    let lo = nct_delta(t, df, 1.0 - a / 2.0)?;
    let hi = nct_delta(t, df, a / 2.0)?;
    Ok(Interval { lower: lo / k, upper: hi / k, level })
}

/// Confidence intervals for all indices of `c`.
///
/// `within_df` is the degrees of freedom of the within-σ estimate (see
/// [`pooled_within_df`]); with `None` only the overall (Pp, Ppl, Ppu, Ppk) and
/// Cpm intervals are computed.
pub fn capability_intervals(c: &NormalCapability, within_df: Option<f64>, level: f64) -> Result<CapabilityIntervals> {
    check_level(level)?;
    if let Some(d) = within_df {
        if !(d > 0.0 && d.is_finite()) {
            return invalid(format!("within degrees of freedom {d} must be positive"));
        }
    }
    let n = c.n;
    let df = (n - 1) as f64;
    let two = |x: Option<f64>, d: f64| x.map(|v| chi_interval(v, d, level)).transpose();
    let one = |x: Option<f64>, d: f64| x.map(|v| exact_one_sided(v, n, d, level)).transpose();
    let pp = two(c.pp, df)?;
    let ppl = one(c.ppl, df)?;
    let ppu = one(c.ppu, df)?;
    let ppk = bissell(c.ppk, n, df, level);
    let (cp, cpl, cpu, cpk) = match within_df {
        Some(d) => (two(c.cp, d)?, one(c.cpl, d)?, one(c.cpu, d)?, Some(bissell(c.cpk, n, d, level))),
        None => (None, None, None, None),
    };
    let cpm = match (c.cpm, c.specs.target) {
        (Some(m), Some(t)) => Some(cpm_interval(m, n, c.mean, c.sd_overall, t, level)?),
        _ => None,
    };
    Ok(CapabilityIntervals { level, within_df, cp, cpl, cpu, cpk, pp, ppl, ppu, ppk, cpm })
}

fn cpm_interval(cpm: f64, n: usize, mean: f64, sd: f64, target: f64, level: f64) -> Result<Interval> {
    let a = (mean - target) / sd;
    let nu = n as f64 * (1.0 + a * a).powi(2) / (1.0 + 2.0 * a * a);
    chi_interval(cpm, nu, level)
}
