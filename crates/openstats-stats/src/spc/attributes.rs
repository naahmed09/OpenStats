//! Attribute charts beyond the basic Shewhart set: exact binomial/Poisson probability limits for
//! p, np, c and u charts, Laney P′ and U′ (overdispersion-adjusted) and the rare-event G
//! (opportunities between events) and T (time between events) charts.
//!
//! Variable subgroup or inspection-unit sizes give per-point (stair-step) limits everywhere.

use super::capability::{fit_distribution, Family};
use super::charts::{
    attribute_chart, c_chart, check_attribute, mean_of, np_chart, p_chart, u_chart, AttributeChart, AttributeKind,
    ChartOptions, ChartPanel,
};
use super::constants::constants;
use super::rules::{SpecialCauseTests, TestViolation};
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{Binomial, DiscreteDistribution, Poisson};
use openstats_math::special::erf::norm_cdf;

// ------------------------------------------------------- exact probability limits

/// How the control limits of an attribute chart are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitMethod {
    /// `centre ± k·σ` from the normal approximation (the classical limits).
    Normal,
    /// Exact binomial (p, np) or Poisson (c, u) probability limits with tail probability `Φ(−k)` on each side.
    Exact,
}

/// Make `chart` use exact limits. `limits(i)` returns the (lower, upper) limit of point `i` in count units;
/// a count below the lower or above the upper limit fails test 1. Tests 2–4 keep using the normal σ for zones.
fn with_exact_limits(
    mut chart: AttributeChart,
    k: f64,
    limits: impl Fn(usize) -> Result<(f64, f64)>,
) -> Result<AttributeChart> {
    let n = chart.panel.values.len();
    let mut bad = Vec::new();
    for i in 0..n {
        let (lo, hi) = limits(i)?;
        let scale = match chart.kind {
            AttributeKind::Np | AttributeKind::C => 1.0,
            _ => chart.sizes[i],
        };
        chart.panel.lcl[i] = lo / scale;
        chart.panel.ucl[i] = hi / scale;
        let c = chart.counts[i];
        if c < lo || c > hi {
            bad.push(i);
        }
    }
    chart.panel.tests.beyond_sigma = Some(k);
    chart.panel.violations.retain(|v| v.test != 1);
    if !bad.is_empty() {
        chart.panel.violations.insert(0, TestViolation { test: 1, points: bad });
    }
    Ok(chart)
}

fn no_test1(opts: &ChartOptions) -> ChartOptions {
    let mut o = opts.clone();
    o.tests.beyond_sigma = None;
    o
}

fn binomial_limits(n: f64, p: f64, tail: f64) -> Result<(f64, f64)> {
    if p <= 0.0 || p >= 1.0 {
        return Ok((n * p, n * p));
    }
    let d = Binomial::new(n as u64, p)?;
    Ok((d.quantile(tail), d.isf(tail)))
}

fn poisson_limits(mean: f64, tail: f64) -> Result<(f64, f64)> {
    if mean <= 0.0 {
        return Ok((0.0, 0.0));
    }
    let d = Poisson::new(mean)?;
    Ok((d.quantile(tail), d.isf(tail)))
}

/// p chart; with [`LimitMethod::Exact`] the limits are binomial quantiles for each sample size.
pub fn p_chart_with(
    defectives: &[f64],
    sizes: &[f64],
    opts: &ChartOptions,
    method: LimitMethod,
) -> Result<AttributeChart> {
    match method {
        LimitMethod::Normal => p_chart(defectives, sizes, opts),
        LimitMethod::Exact => {
            let c = p_chart(defectives, sizes, &no_test1(opts))?;
            let (tail, p) = (norm_cdf(-opts.k_sigma), c.center);
            let sz = c.sizes.clone();
            with_exact_limits(c, opts.k_sigma, |i| binomial_limits(sz[i], p, tail))
        }
    }
}

/// np chart; exact limits are binomial quantiles.
pub fn np_chart_with(
    defectives: &[f64],
    size: f64,
    opts: &ChartOptions,
    method: LimitMethod,
) -> Result<AttributeChart> {
    match method {
        LimitMethod::Normal => np_chart(defectives, size, opts),
        LimitMethod::Exact => {
            let c = np_chart(defectives, size, &no_test1(opts))?;
            let (tail, p) = (norm_cdf(-opts.k_sigma), c.center / size);
            with_exact_limits(c, opts.k_sigma, |_| binomial_limits(size, p, tail))
        }
    }
}

/// c chart; exact limits are Poisson quantiles.
pub fn c_chart_with(counts: &[f64], opts: &ChartOptions, method: LimitMethod) -> Result<AttributeChart> {
    match method {
        LimitMethod::Normal => c_chart(counts, opts),
        LimitMethod::Exact => {
            let c = c_chart(counts, &no_test1(opts))?;
            let (tail, m) = (norm_cdf(-opts.k_sigma), c.center);
            with_exact_limits(c, opts.k_sigma, |_| poisson_limits(m, tail))
        }
    }
}

/// u chart; exact limits are Poisson quantiles for the mean `ū·nᵢ` of each sample.
pub fn u_chart_with(counts: &[f64], sizes: &[f64], opts: &ChartOptions, method: LimitMethod) -> Result<AttributeChart> {
    match method {
        LimitMethod::Normal => u_chart(counts, sizes, opts),
        LimitMethod::Exact => {
            let c = u_chart(counts, sizes, &no_test1(opts))?;
            let (tail, u) = (norm_cdf(-opts.k_sigma), c.center);
            let sz = c.sizes.clone();
            with_exact_limits(c, opts.k_sigma, |i| poisson_limits(u * sz[i], tail))
        }
    }
}

// --------------------------------------------------------------- Laney charts

/// A Laney P′ or U′ chart.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneyChart {
    /// The chart (limits `centre ± k·σ_z·σᵢ`).
    pub chart: AttributeChart,
    /// Standardised values `zᵢ = (yᵢ − centre)/σᵢ` with the binomial/Poisson σᵢ.
    pub z: Vec<f64>,
    /// σ_z: average moving range of the z-scores divided by d2(2).
    pub sigma_z: f64,
}

fn laney(
    kind: AttributeKind,
    name: &str,
    counts: &[f64],
    sizes: &[f64],
    values: Vec<f64>,
    center: f64,
    sigma_i: Vec<f64>,
    hi: Vec<f64>,
    opts: &ChartOptions,
) -> Result<LaneyChart> {
    if values.len() < 3 {
        return Err(StatsError::InsufficientData { what: "Laney chart", needed: 3, got: values.len() });
    }
    if sigma_i.iter().any(|s| s.is_nan() || *s <= 0.0) {
        return invalid(
            "the process centre gives a zero standard error (all-zero or all-one data); the chart is undefined",
        );
    }
    let z: Vec<f64> = values.iter().zip(&sigma_i).map(|(v, s)| (v - center) / s).collect();
    let mr: Vec<f64> = z.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let sigma_z = mean_of(&mr) / constants(2)?.d2;
    if sigma_z <= 0.0 {
        return invalid("the z-scores do not vary; σ_z is zero");
    }
    let sig: Vec<f64> = sigma_i.iter().map(|s| s * sigma_z).collect();
    let chart = attribute_chart(kind, name, counts, sizes, values, center, sig, 0.0, hi, opts)?;
    Ok(LaneyChart { chart, z, sigma_z })
}

/// Laney P′ chart: p chart whose limits are widened (or narrowed) by σ_z, the
/// between-sample variation of the standardised proportions.
pub fn laney_p(defectives: &[f64], sizes: &[f64], opts: &ChartOptions) -> Result<LaneyChart> {
    check_attribute(defectives, sizes, opts, true)?;
    if defectives.iter().zip(sizes).any(|(d, n)| d > n) {
        return invalid("a count exceeds its sample size");
    }
    let pbar = opts.known_center.unwrap_or_else(|| defectives.iter().sum::<f64>() / sizes.iter().sum::<f64>());
    if !(0.0..=1.0).contains(&pbar) {
        return invalid("the process proportion must lie in [0, 1]");
    }
    let values: Vec<f64> = defectives.iter().zip(sizes).map(|(d, n)| d / n).collect();
    let sigma_i: Vec<f64> = sizes.iter().map(|n| (pbar * (1.0 - pbar) / n).sqrt()).collect();
    let n = values.len();
    laney(AttributeKind::LaneyP, "Proportion", defectives, sizes, values, pbar, sigma_i, vec![1.0; n], opts)
}

/// Laney U′ chart for defects per unit.
pub fn laney_u(counts: &[f64], sizes: &[f64], opts: &ChartOptions) -> Result<LaneyChart> {
    check_attribute(counts, sizes, opts, false)?;
    let ubar = opts.known_center.unwrap_or_else(|| counts.iter().sum::<f64>() / sizes.iter().sum::<f64>());
    if ubar.is_nan() || ubar < 0.0 {
        return invalid("the process mean must be non-negative");
    }
    let values: Vec<f64> = counts.iter().zip(sizes).map(|(c, n)| c / n).collect();
    let sigma_i: Vec<f64> = sizes.iter().map(|n| (ubar / n).sqrt()).collect();
    let n = values.len();
    laney(
        AttributeKind::LaneyU,
        "Sample Count Per Unit",
        counts,
        sizes,
        values,
        ubar,
        sigma_i,
        vec![f64::INFINITY; n],
        opts,
    )
}

// ------------------------------------------------------------ rare-event charts

/// G or T chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RareEventKind {
    /// Counts of opportunities (units, days, …) between events — geometric model.
    G,
    /// Time between events — Weibull model.
    T,
}

/// A rare-event chart: the plotted values with probability limits of the fitted model.
#[derive(Debug, Clone, PartialEq)]
pub struct RareEventChart {
    /// Chart kind.
    pub kind: RareEventKind,
    /// Values, median centre line and probability limits (test 1 only).
    pub panel: ChartPanel,
    /// Fitted model description.
    pub model: String,
    /// Geometric success probability (G) or Weibull shape (T).
    pub param1: f64,
    /// `NaN` (G) or Weibull scale (T).
    pub param2: f64,
    /// Tail probability beyond each limit, `Φ(−k)`.
    pub tail_probability: f64,
}

fn probability_panel(name: &str, values: Vec<f64>, lo: f64, mid: f64, hi: f64, k: f64) -> ChartPanel {
    let n = values.len();
    let violators: Vec<usize> = (0..n).filter(|&i| values[i] > hi || values[i] < lo).collect();
    ChartPanel {
        name: name.to_string(),
        center: vec![mid; n],
        ucl: vec![hi; n],
        lcl: vec![lo; n],
        sigma: vec![(hi - lo) / (2.0 * k); n],
        values,
        tests: SpecialCauseTests { beyond_sigma: Some(k), ..SpecialCauseTests::none() },
        violations: if violators.is_empty() { vec![] } else { vec![TestViolation { test: 1, points: violators }] },
    }
}

/// G chart: `between[i]` is the number of opportunities (non-events) between event i−1 and event i.
///
/// A geometric distribution on {0, 1, 2, …} is fitted (`p̂ = 1/(1 + mean)`), the centre line is its
/// median and the limits are its `Φ(−k)` and `1 − Φ(−k)` quantiles (0.00135 / 0.99865 for k = 3),
/// computed from the continuous form `ln(1 − q)/ln(1 − p) − 1` and truncated at 0.
pub fn g_chart(between: &[f64], opts: &ChartOptions) -> Result<RareEventChart> {
    opts.check()?;
    if between.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    if between.iter().any(|v| !v.is_nan() && (*v < 0.0 || v.fract() != 0.0)) {
        return invalid("a G chart needs non-negative whole counts");
    }
    let ok: Vec<f64> = between.iter().copied().filter(|v| !v.is_nan()).collect();
    if ok.len() < 2 {
        return Err(StatsError::InsufficientData { what: "G chart", needed: 2, got: ok.len() });
    }
    let mean = opts.known_center.unwrap_or_else(|| mean_of(&ok));
    if !(mean > 0.0 && mean.is_finite()) {
        return invalid("the mean count between events must be positive");
    }
    let p = 1.0 / (1.0 + mean);
    let tail = norm_cdf(-opts.k_sigma);
    let lq = (1.0 - p).ln();
    let quant = |q: f64| ((1.0 - q).ln() / lq - 1.0).max(0.0);
    let (lo, mid, hi) = (quant(tail), quant(0.5), quant(1.0 - tail));
    Ok(RareEventChart {
        kind: RareEventKind::G,
        panel: probability_panel("Count Between Events", between.to_vec(), lo, mid, hi, opts.k_sigma),
        model: "geometric".into(),
        param1: p,
        param2: f64::NAN,
        tail_probability: tail,
    })
}

/// Model for a T chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeModel {
    /// Weibull with threshold 0.
    Weibull,
    /// Exponential (Weibull with shape fixed at 1): scale = mean.
    Exponential,
}

/// T chart with the Weibull model (see [`t_chart_with`]).
pub fn t_chart(times: &[f64], opts: &ChartOptions) -> Result<RareEventChart> {
    t_chart_with(times, opts, TimeModel::Weibull)
}

/// T chart: `times[i]` is the time between event i−1 and event i (positive).
///
/// A Weibull distribution (threshold 0) is fitted by maximum likelihood; the centre line is its
/// median and the limits are its `Φ(−k)` and `1 − Φ(−k)` quantiles.
pub fn t_chart_with(times: &[f64], opts: &ChartOptions, model: TimeModel) -> Result<RareEventChart> {
    opts.check()?;
    let tail = norm_cdf(-opts.k_sigma);
    let (shape, scale, name) = match model {
        TimeModel::Weibull => {
            let fit = fit_distribution(times, Family::Weibull, None)?;
            (fit.param1, fit.param2, "Weibull")
        }
        TimeModel::Exponential => {
            if times.iter().any(|v| v.is_infinite()) {
                return Err(StatsError::NonFinite);
            }
            let ok: Vec<f64> = times.iter().copied().filter(|v| !v.is_nan()).collect();
            if ok.len() < 2 || ok.iter().any(|&v| v <= 0.0) {
                return invalid("a T chart needs at least two positive times");
            }
            (1.0, mean_of(&ok), "exponential")
        }
    };
    let q = |p: f64| scale * (-(1.0 - p).ln()).powf(1.0 / shape);
    let (lo, mid, hi) = (q(tail), q(0.5), q(1.0 - tail));
    Ok(RareEventChart {
        kind: RareEventKind::T,
        panel: probability_panel("Time Between Events", times.to_vec(), lo, mid, hi, opts.k_sigma),
        model: name.into(),
        param1: shape,
        param2: scale,
        tail_probability: tail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laney_equals_p_chart_when_z_scores_have_unit_spread() {
        // For homogeneous data σ_z is close to 1 and the limits close to the p-chart's.
        let d = [4.0, 6.0, 5.0, 3.0, 7.0, 5.0, 4.0, 6.0, 5.0, 5.0, 4.0, 6.0];
        let n = vec![100.0; 12];
        let l = laney_p(&d, &n, &ChartOptions::default()).unwrap();
        assert!(l.sigma_z > 0.0 && l.sigma_z < 2.0);
        let p = super::super::charts::p_chart(&d, &n, &ChartOptions::default()).unwrap();
        let ratio = (l.chart.panel.ucl[0] - l.chart.center) / (p.panel.ucl[0] - p.center);
        assert!((ratio - l.sigma_z).abs() < 1e-12);
    }

    #[test]
    fn laney_widens_overdispersed_limits() {
        let d = [5.0, 35.0, 6.0, 34.0, 5.0, 35.0, 6.0, 34.0, 5.0, 35.0];
        let n = vec![100.0; 10];
        let l = laney_p(&d, &n, &ChartOptions::default()).unwrap();
        assert!(l.sigma_z > 1.5);
        assert!(l.chart.panel.flagged().is_empty());
        let p = super::super::charts::p_chart(&d, &n, &ChartOptions::default()).unwrap();
        assert!(!p.panel.flagged().is_empty());
    }

    #[test]
    fn g_chart_quantiles_match_geometric_cdf() {
        let c = g_chart(&[12.0, 30.0, 8.0, 45.0, 20.0, 15.0, 60.0, 9.0], &ChartOptions::default()).unwrap();
        let p = c.param1;
        let cdf = |x: f64| 1.0 - (1.0 - p).powf(x + 1.0);
        assert!((cdf(c.panel.ucl[0]) - (1.0 - c.tail_probability)).abs() < 1e-12);
        assert!((cdf(c.panel.center[0]) - 0.5).abs() < 1e-12);
        assert!(c.panel.lcl[0] >= 0.0);
    }

    #[test]
    fn g_chart_rejects_bad_input() {
        assert!(g_chart(&[1.5, 2.0], &ChartOptions::default()).is_err());
        assert!(g_chart(&[0.0, 0.0, 0.0], &ChartOptions::default()).is_err());
        assert!(g_chart(&[3.0], &ChartOptions::default()).is_err());
    }

    #[test]
    fn t_chart_limits_are_weibull_quantiles() {
        let t = [5.0, 7.0, 4.0, 6.0, 8.0, 5.5, 6.5, 4.5, 7.5, 5.0, 6.0, 5.2];
        let c = t_chart(&t, &ChartOptions::default()).unwrap();
        let q = |p: f64| c.param2 * (-(1.0 - p).ln()).powf(1.0 / c.param1);
        assert!((c.panel.ucl[0] - q(1.0 - c.tail_probability)).abs() < 1e-9 * c.panel.ucl[0]);
        assert!((c.panel.lcl[0] - q(c.tail_probability)).abs() < 1e-9);
        assert!((c.panel.center[0] - q(0.5)).abs() < 1e-9);
        assert!(c.panel.flagged().is_empty());
        assert!(c.panel.lcl[0] > 0.0 && c.panel.center[0] > c.panel.lcl[0]);
    }
}
