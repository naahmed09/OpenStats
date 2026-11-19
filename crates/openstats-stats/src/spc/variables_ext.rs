//! Further variables charts: I-MR-R/S (between/within), zone chart, Z-MR
//! (short-run) and Box–Cox transformed charts.

use super::charts::{
    estimate_sigma, i_mr, individuals_sigma_estimate, mean_of, subgroup_chart, subgroup_stats, xbar_r, xbar_s,
    ChartOptions, ChartPanel, IndividualsSigma, SubgroupSigma, SubgroupStat, VariableChart, VariableChartKind,
};
use super::constants::constants;
use super::rules::SpecialCauseTests;
use super::transform::{BoxCox, LambdaChoice};
use crate::error::{invalid, Result, StatsError};

// ------------------------------------------------------- I-MR-R/S (between/within)

/// Which within-subgroup chart accompanies the I-MR chart of subgroup means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WithinChart {
    /// Subgroup ranges (σ_within = R̄/d2).
    Range,
    /// Subgroup standard deviations (σ_within = s̄/c4).
    StDev,
}

/// I-MR-R or I-MR-S chart: separates between-subgroup from within-subgroup variation.
#[derive(Debug, Clone, PartialEq)]
pub struct BetweenWithinChart {
    /// Which within chart was used.
    pub within_chart: WithinChart,
    /// Individuals chart of the subgroup means; limits `mean ± k√(σ_b² + σ_w²/nᵢ)`.
    pub means: ChartPanel,
    /// Moving range of consecutive subgroup means.
    pub moving_range: ChartPanel,
    /// R or S chart of the within-subgroup variation.
    pub within: ChartPanel,
    /// Grand mean.
    pub mean: f64,
    /// Within-subgroup σ.
    pub sigma_within: f64,
    /// Between-subgroup σ (0 when the moving ranges are explained by the within variation).
    pub sigma_between: f64,
    /// Total σ, `√(σ_b² + σ_w²)` (for the average subgroup size when sizes differ: `σ_w²·mean(1/nᵢ)` is used inside the limits).
    pub sigma_total: f64,
    /// Subgroup sizes.
    pub sizes: Vec<usize>,
}

/// I-MR-R (`WithinChart::Range`) or I-MR-S chart from subgroup data.
///
/// σ_within is estimated by R̄/d2 or s̄/c4. The moving ranges of the subgroup
/// means estimate `σ_m = MR̄/d2(2)`, the standard deviation of a subgroup
/// mean; the between component is `σ_b² = max(0, σ_m² − σ_w²·mean(1/nᵢ))`.
/// With unequal subgroup sizes the means do not share one variance; the
/// estimate above is then an average-variance moment estimate (a documented
/// convention, not an exact unbiased one).
pub fn i_mr_r_s(groups: &[Vec<f64>], within: WithinChart, opts: &ChartOptions) -> Result<BetweenWithinChart> {
    opts.check()?;
    let stats = subgroup_stats(groups)?;
    if stats.len() < 3 {
        return Err(StatsError::InsufficientData { what: "I-MR-R/S chart", needed: 3, got: stats.len() });
    }
    let method = match within {
        WithinChart::Range => SubgroupSigma::RBar,
        WithinChart::StDev => SubgroupSigma::SBar,
    };
    let sigma_w = match opts.known_sigma {
        Some(s) => s,
        None => estimate_sigma(&stats, opts.subgroup_sigma.unwrap_or(method))?.0,
    };
    if sigma_w.is_nan() || sigma_w <= 0.0 {
        return invalid("the within-subgroup σ is zero; limits are undefined");
    }
    let means: Vec<f64> = stats.iter().map(|s| s.mean).collect();
    let m = means.len();
    let total_n: f64 = stats.iter().map(|s| s.n as f64).sum();
    let grand = opts.known_center.unwrap_or_else(|| stats.iter().map(|s| s.n as f64 * s.mean).sum::<f64>() / total_n);
    let mut mr = vec![f64::NAN; m];
    for i in 1..m {
        mr[i] = (means[i] - means[i - 1]).abs();
    }
    let mr_bar = mean_of(&mr[1..]);
    let d2_2 = constants(2)?.d2;
    let sigma_m = mr_bar / d2_2;
    let mean_inv_n = stats.iter().map(|s| 1.0 / s.n as f64).sum::<f64>() / m as f64;
    let var_b = (sigma_m * sigma_m - sigma_w * sigma_w * mean_inv_n).max(0.0);
    let k = opts.k_sigma;
    let se: Vec<f64> = stats.iter().map(|s| (var_b + sigma_w * sigma_w / s.n as f64).sqrt()).collect();
    let location = ChartPanel::build(
        "Subgroup Mean",
        means,
        vec![grand; m],
        se.iter().map(|e| grand + k * e).collect(),
        se.iter().map(|e| grand - k * e).collect(),
        se,
        &opts.tests,
    )?;
    // The moving range of the means is charted with σ_m² = σ_b² + σ_w²·mean(1/n).
    let sm = (var_b + sigma_w * sigma_w * mean_inv_n).sqrt();
    let kc = constants(2)?;
    let moving_range = ChartPanel::build(
        "Moving Range of Means",
        mr,
        vec![kc.d2 * sm; m],
        vec![(kc.d2 + k * kc.d3) * sm; m],
        vec![((kc.d2 - k * kc.d3) * sm).max(0.0); m],
        vec![kc.d3 * sm; m],
        &opts.dispersion_tests,
    )?;
    let within_opts = ChartOptions { known_sigma: Some(sigma_w), known_center: None, ..opts.clone() };
    let kind = match within {
        WithinChart::Range => VariableChartKind::XbarR,
        WithinChart::StDev => VariableChartKind::XbarS,
    };
    let w = subgroup_chart(&stats, &within_opts, kind)?;
    Ok(BetweenWithinChart {
        within_chart: within,
        means: location,
        moving_range,
        within: w.dispersion,
        mean: grand,
        sigma_within: sigma_w,
        sigma_between: var_b.sqrt(),
        sigma_total: (var_b + sigma_w * sigma_w).sqrt(),
        sizes: stats.iter().map(|s| s.n).collect(),
    })
}

// ------------------------------------------------------------------ zone chart

/// Weights and signal rule of a zone chart.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneOptions {
    /// Scores for points within 1σ, between 1σ and 2σ, between 2σ and 3σ and beyond 3σ of the centre line.
    pub weights: [f64; 4],
    /// The chart signals when the cumulative score reaches this value.
    pub signal_at: f64,
}

impl Default for ZoneOptions {
    /// Weights 0, 2, 4, 8 and a signal at a cumulative score of 8.
    fn default() -> Self {
        ZoneOptions { weights: [0.0, 2.0, 4.0, 8.0], signal_at: 8.0 }
    }
}

/// A zone chart (cumulative zone scores).
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneChart {
    /// The underlying location chart (individuals or subgroup means), including its centre line and σ per point.
    pub panel: ChartPanel,
    /// Zone (0–3) of each point (`usize::MAX` for a missing point).
    pub zones: Vec<usize>,
    /// Cumulative score after each point (before the reset caused by a signal).
    pub scores: Vec<f64>,
    /// Weights and signal level used.
    pub options: ZoneOptions,
    /// Points at which the cumulative score reached the signal level.
    pub signals: Vec<usize>,
}

/// Zone scoring on a panel: points beyond `zone` multiples of σ get the zone's weight; the running
/// score accumulates on one side of the centre line, restarts when a point falls on the other side
/// (or exactly on the centre line) and restarts after a signal. A missing point restarts the score.
pub fn zone_scores(panel: ChartPanel, opts: &ZoneOptions) -> Result<ZoneChart> {
    if opts.weights.iter().any(|w| !(w.is_finite() && *w >= 0.0))
        || !(opts.signal_at > 0.0 && opts.signal_at.is_finite())
    {
        return invalid("zone weights must be non-negative and the signal level positive");
    }
    let z = panel.z_scores();
    let n = z.len();
    let mut zones = vec![usize::MAX; n];
    let mut scores = vec![f64::NAN; n];
    let mut signals = Vec::new();
    let (mut side, mut acc) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        if z[i].is_nan() {
            side = 0.0;
            acc = 0.0;
            continue;
        }
        let a = z[i].abs();
        let zone = if a < 1.0 {
            0
        } else if a < 2.0 {
            1
        } else if a < 3.0 {
            2
        } else {
            3
        };
        zones[i] = zone;
        let s = if z[i] > 0.0 {
            1.0
        } else if z[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        if s != side {
            side = s;
            acc = 0.0;
        }
        acc += opts.weights[zone];
        scores[i] = acc;
        if acc >= opts.signal_at {
            signals.push(i);
            acc = 0.0;
            side = 0.0;
        }
    }
    Ok(ZoneChart { panel, zones, scores, options: opts.clone(), signals })
}

/// Zone chart of individual observations (I-MR limits and σ).
pub fn zone_chart_individuals(data: &[f64], opts: &ChartOptions, zone: &ZoneOptions) -> Result<ZoneChart> {
    let o = ChartOptions { tests: SpecialCauseTests::none(), ..opts.clone() };
    zone_scores(i_mr(data, &o)?.location, zone)
}

/// Zone chart of subgroup means (X̄-R σ estimate unless `opts.subgroup_sigma` says otherwise).
pub fn zone_chart_subgroups(groups: &[Vec<f64>], opts: &ChartOptions, zone: &ZoneOptions) -> Result<ZoneChart> {
    let o = ChartOptions { tests: SpecialCauseTests::none(), ..opts.clone() };
    zone_scores(xbar_r(groups, &o)?.location, zone)
}

// ------------------------------------------------------------- Z-MR short run

/// How σ of each part is obtained in a Z-MR chart.
#[derive(Debug, Clone, PartialEq)]
pub enum ShortRunSigma {
    /// One σ for all parts: the average moving range of each part's own observations, pooled, divided by d2.
    Pooled,
    /// A separate σ per part from the part's own moving ranges (each part needs ≥ 3 observations).
    ByPart,
    /// Known σ per part, in order of first appearance of the parts.
    Known(Vec<f64>),
}

/// Z-MR (short-run) chart: observations are standardised by part and charted as I-MR with centre 0 and σ 1.
#[derive(Debug, Clone, PartialEq)]
pub struct ZmrChart {
    /// Chart of the standardised values (centre 0, σ 1; the dispersion panel is the moving range of z).
    pub chart: VariableChart,
    /// Part labels in order of first appearance.
    pub parts: Vec<String>,
    /// Target (mean) used for each part.
    pub targets: Vec<f64>,
    /// σ used for each part.
    pub sigmas: Vec<f64>,
    /// The standardised values `(x − target)/σ`.
    pub z: Vec<f64>,
}

/// Z-MR chart. `targets` (if given) holds a known target per part in order of first appearance;
/// otherwise the part mean is the target.
pub fn z_mr<S: AsRef<str>>(
    data: &[f64],
    parts: &[S],
    targets: Option<&[f64]>,
    sigma: &ShortRunSigma,
    opts: &ChartOptions,
) -> Result<ZmrChart> {
    if data.len() != parts.len() {
        return invalid(format!("{} observations but {} part labels", data.len(), parts.len()));
    }
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let mut names: Vec<String> = Vec::new();
    let idx: Vec<usize> = parts
        .iter()
        .map(|p| {
            let p = p.as_ref();
            names.iter().position(|n| n == p).unwrap_or_else(|| {
                names.push(p.to_string());
                names.len() - 1
            })
        })
        .collect();
    let np = names.len();
    let by_part: Vec<Vec<f64>> =
        (0..np).map(|p| data.iter().zip(&idx).filter(|(_, &i)| i == p).map(|(&v, _)| v).collect()).collect();
    let tg: Vec<f64> = match targets {
        Some(t) => {
            if t.len() != np {
                return invalid(format!("{} targets for {np} parts", t.len()));
            }
            t.to_vec()
        }
        None => by_part
            .iter()
            .map(|v| {
                let ok: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
                if ok.is_empty() {
                    f64::NAN
                } else {
                    mean_of(&ok)
                }
            })
            .collect(),
    };
    if tg.iter().any(|t| !t.is_finite()) {
        return invalid("every part needs at least one observation (or a finite target)");
    }
    let sg: Vec<f64> = match sigma {
        ShortRunSigma::Known(s) => {
            if s.len() != np {
                return invalid(format!("{} sigmas for {np} parts", s.len()));
            }
            s.clone()
        }
        ShortRunSigma::ByPart => by_part
            .iter()
            .map(|v| individuals_sigma_estimate(v, 2, IndividualsSigma::AverageMovingRange).map(|r| r.0))
            .collect::<Result<_>>()?,
        ShortRunSigma::Pooled => {
            let (mut sum, mut cnt) = (0.0, 0usize);
            for v in &by_part {
                for w in v.windows(2) {
                    if !w[0].is_nan() && !w[1].is_nan() {
                        sum += (w[1] - w[0]).abs();
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                return Err(StatsError::InsufficientData { what: "pooled short-run sigma", needed: 1, got: 0 });
            }
            vec![sum / cnt as f64 / constants(2)?.d2; np]
        }
    };
    if sg.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return invalid("every part σ must be positive (a part with no variation cannot be standardised)");
    }
    let z: Vec<f64> = data.iter().zip(&idx).map(|(&x, &i)| (x - tg[i]) / sg[i]).collect();
    let o = ChartOptions { known_center: Some(0.0), known_sigma: Some(1.0), ..opts.clone() };
    let mut chart = i_mr(&z, &o)?;
    chart.location.name = "Standardized Value".into();
    chart.sigma_method = "standardised by part (centre 0, σ 1)".into();
    Ok(ZmrChart { chart, parts: names, targets: tg, sigmas: sg, z })
}

// ------------------------------------------------------------ Box–Cox charts

/// A control chart drawn on Box–Cox transformed data, with limits also given in the original units.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxCoxChart {
    /// The fitted transformation.
    pub transform: BoxCox,
    /// The chart of the transformed data (all panels in transformed units).
    pub chart: VariableChart,
    /// Centre line of the location panel, back-transformed.
    pub center: Vec<f64>,
    /// Upper limit of the location panel, back-transformed (`NaN` where outside the transformation's range).
    pub ucl: Vec<f64>,
    /// Lower limit of the location panel, back-transformed (`NaN` where outside the transformation's range).
    pub lcl: Vec<f64>,
}

fn back(t: &BoxCox, v: &[f64]) -> Vec<f64> {
    v.iter().map(|&z| t.inverse(z)).collect()
}

fn finish_boxcox(t: BoxCox, chart: VariableChart) -> BoxCoxChart {
    let center = back(&t, &chart.location.center);
    let ucl = back(&t, &chart.location.ucl);
    let lcl = back(&t, &chart.location.lcl);
    BoxCoxChart { transform: t, chart, center, ucl, lcl }
}

/// I-MR chart of Box–Cox transformed individuals (positive data).
pub fn boxcox_i_mr(data: &[f64], choice: LambdaChoice, opts: &ChartOptions) -> Result<BoxCoxChart> {
    let t = BoxCox::fit(data, choice)?;
    let z: Vec<f64> = data.iter().map(|&x| t.forward(x)).collect();
    let o = transformed_opts(opts, &t);
    Ok(finish_boxcox(t, i_mr(&z, &o)?))
}

/// X̄-R (`use_range`) or X̄-S chart of Box–Cox transformed subgroup data (λ fitted to all observations).
pub fn boxcox_xbar(
    groups: &[Vec<f64>],
    use_range: bool,
    choice: LambdaChoice,
    opts: &ChartOptions,
) -> Result<BoxCoxChart> {
    let all: Vec<f64> = groups.iter().flatten().copied().collect();
    let t = BoxCox::fit(&all, choice)?;
    let zg: Vec<Vec<f64>> = groups.iter().map(|g| g.iter().map(|&x| t.forward(x)).collect()).collect();
    let o = transformed_opts(opts, &t);
    let chart = if use_range { xbar_r(&zg, &o)? } else { xbar_s(&zg, &o)? };
    Ok(finish_boxcox(t, chart))
}

/// Known centre / σ are given in original units; the centre is transformed, σ cannot be (rejected).
fn transformed_opts(opts: &ChartOptions, t: &BoxCox) -> ChartOptions {
    ChartOptions { known_center: opts.known_center.map(|c| t.forward(c)), ..opts.clone() }
}

/// Subgroup summaries re-exported for callers that build I-MR-R/S charts from statistics.
pub fn subgroup_summaries(groups: &[Vec<f64>]) -> Result<Vec<SubgroupStat>> {
    subgroup_stats(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64) / (1u64 << 31) as f64
            })
            .collect()
    }

    #[test]
    fn between_within_reduces_to_xbar_when_no_between_variation() {
        // identical-mean subgroups: all variation is within.
        let g: Vec<Vec<f64>> = (0..12)
            .map(|i| {
                let u = lcg(5, 7 + i as u64);
                u.iter().map(|v| 10.0 + v).collect()
            })
            .collect();
        let c = i_mr_r_s(&g, WithinChart::Range, &ChartOptions::default()).unwrap();
        assert!(c.sigma_within > 0.0);
        assert!(c.sigma_between >= 0.0);
        assert_eq!(c.means.values.len(), 12);
        assert!(c.moving_range.values[0].is_nan());
        let x = xbar_r(&g, &ChartOptions::default()).unwrap();
        assert!((c.within.ucl[0] - x.dispersion.ucl[0]).abs() < 1e-12);
    }

    #[test]
    fn between_variation_is_detected() {
        let g: Vec<Vec<f64>> = (0..15)
            .map(|i| {
                let shift = if i % 2 == 0 { 0.0 } else { 3.0 };
                lcg(4, 99 + i as u64).iter().map(|v| 10.0 + shift + 0.2 * v).collect()
            })
            .collect();
        let c = i_mr_r_s(&g, WithinChart::StDev, &ChartOptions::default()).unwrap();
        assert!(c.sigma_between > 1.0, "between σ {}", c.sigma_between);
        assert!(c.means.ucl[0] > 10.0 + 3.0);
    }

    #[test]
    fn zone_scores_accumulate_and_reset() {
        let p = ChartPanel {
            name: "x".into(),
            values: vec![1.5, 1.5, 0.5, -1.5, 2.5, 2.5, 0.0, 3.5],
            center: vec![0.0; 8],
            ucl: vec![3.0; 8],
            lcl: vec![-3.0; 8],
            sigma: vec![1.0; 8],
            tests: SpecialCauseTests::none(),
            violations: vec![],
        };
        let z = zone_scores(p, &ZoneOptions::default()).unwrap();
        assert_eq!(z.scores[..6], [2.0, 4.0, 4.0, 2.0, 4.0, 8.0]);
        assert_eq!(z.signals, vec![5, 7]);
        assert_eq!(z.scores[6], 0.0);
        assert_eq!(z.zones[7], 3);
    }
}
