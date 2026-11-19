//! Shewhart control charts: I-MR, X̄-R, X̄-S (variables) and p, np, c, u
//! (attributes), each with the special-cause tests of [`super::rules`].
//!
//! Conventions (Minitab's, as documented):
//!
//! * Limits are `centre ± k·σ` with `k = 3` by default. Dispersion charts use
//!   the exact sampling moments of the statistic: centre `d2·σ̂`
//!   (`c4·σ̂` for s) and limits `(d2 ± k·d3)·σ̂` (`(c4 ± k√(1−c4²))·σ̂`),
//!   lower limits truncated at 0. For a constant subgroup size this is the
//!   usual `D3·R̄`, `D4·R̄`, `B3·s̄`, `B4·s̄` form.
//! * Attribute limits are truncated to `[0, 1]` (p), `[0, n]` (np) and `≥ 0` (c, u).
//! * Missing values (`NaN`) are allowed in individual data (the affected
//!   moving ranges are missing too) and are dropped inside subgroups.

use super::constants::{c4_real, constants};
use super::rules::{evaluate, flagged_points, SpecialCauseTests, TestViolation};
use crate::error::{invalid, Result, StatsError};
use crate::report::{ChartKind, ChartSpec, Series};

/// How σ is estimated from individual observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndividualsSigma {
    /// Average moving range / d2 (default).
    AverageMovingRange,
    /// Median moving range / d4.
    MedianMovingRange,
    /// Square root of half the mean squared successive difference (span 2 only).
    Mssd,
}

/// How σ is estimated from subgroups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupSigma {
    /// Weighted average of Rᵢ/d2(nᵢ).
    RBar,
    /// Weighted average of sᵢ/c4(nᵢ).
    SBar,
    /// Pooled standard deviation, unbiased with `c4(Σ(nᵢ−1)+1)`.
    Pooled,
}

/// Options shared by the variable and attribute charts.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartOptions {
    /// Limit width in σ (3).
    pub k_sigma: f64,
    /// Tests on the location chart (I, X̄, p, np, c, u). Default: test 1.
    pub tests: SpecialCauseTests,
    /// Tests on the dispersion chart (MR, R, s). Default: test 1.
    pub dispersion_tests: SpecialCauseTests,
    /// Known process mean (or p, c, u) instead of the estimate.
    pub known_center: Option<f64>,
    /// Known process σ instead of the estimate (variable charts).
    pub known_sigma: Option<f64>,
    /// σ estimate for individuals (I-MR).
    pub individuals_sigma: IndividualsSigma,
    /// Moving-range span (2).
    pub span: usize,
    /// σ estimate for subgroups; `None` → R̄ for X̄-R and s̄ for X̄-S.
    pub subgroup_sigma: Option<SubgroupSigma>,
}

impl Default for ChartOptions {
    fn default() -> Self {
        ChartOptions {
            k_sigma: 3.0,
            tests: SpecialCauseTests::test1_only(),
            dispersion_tests: SpecialCauseTests::test1_only(),
            known_center: None,
            known_sigma: None,
            individuals_sigma: IndividualsSigma::AverageMovingRange,
            span: 2,
            subgroup_sigma: None,
        }
    }
}

impl ChartOptions {
    pub(crate) fn check(&self) -> Result<()> {
        if !(self.k_sigma > 0.0 && self.k_sigma.is_finite()) {
            return invalid(format!("k_sigma must be positive (got {})", self.k_sigma));
        }
        if let Some(s) = self.known_sigma {
            if !(s > 0.0 && s.is_finite()) {
                return invalid("known sigma must be positive");
            }
        }
        if let Some(c) = self.known_center {
            if !c.is_finite() {
                return invalid("known center must be finite");
            }
        }
        Ok(())
    }
}

/// One plotted statistic with its limits and test results.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartPanel {
    /// Name of the plotted statistic.
    pub name: String,
    /// Plotted values (`NaN` = missing).
    pub values: Vec<f64>,
    /// Centre line per point.
    pub center: Vec<f64>,
    /// Upper control limit per point.
    pub ucl: Vec<f64>,
    /// Lower control limit per point.
    pub lcl: Vec<f64>,
    /// One-sigma distance per point (zone width).
    pub sigma: Vec<f64>,
    /// Tests that were run.
    pub tests: SpecialCauseTests,
    /// Test violations.
    pub violations: Vec<TestViolation>,
}

impl ChartPanel {
    pub(crate) fn build(
        name: &str,
        values: Vec<f64>,
        center: Vec<f64>,
        ucl: Vec<f64>,
        lcl: Vec<f64>,
        sigma: Vec<f64>,
        tests: &SpecialCauseTests,
    ) -> Result<ChartPanel> {
        let z: Vec<f64> = values
            .iter()
            .zip(&center)
            .zip(&sigma)
            .map(|((v, c), s)| {
                if v.is_nan() {
                    f64::NAN
                } else if *s > 0.0 {
                    (v - c) / s
                } else if v == c {
                    0.0
                } else {
                    (v - c).signum() * f64::INFINITY
                }
            })
            .collect();
        let violations = evaluate(&z, tests)?;
        Ok(ChartPanel { name: name.to_string(), values, center, ucl, lcl, sigma, tests: tests.clone(), violations })
    }

    /// z-scores `(value − centre)/σᵢ` used by the tests.
    pub fn z_scores(&self) -> Vec<f64> {
        self.values
            .iter()
            .zip(&self.center)
            .zip(&self.sigma)
            .map(|((v, c), s)| if *s > 0.0 { (v - c) / s } else { f64::NAN })
            .collect()
    }

    /// Points flagged by any test (0-based).
    pub fn flagged(&self) -> Vec<usize> {
        flagged_points(&self.violations)
    }

    /// Points beyond the control limits (test 1 with the chart's own limits).
    pub fn beyond_limits(&self) -> Vec<usize> {
        (0..self.values.len()).filter(|&i| self.values[i] > self.ucl[i] || self.values[i] < self.lcl[i]).collect()
    }

    /// Chart data for a front end: the statistic, its limits and the flagged
    /// points (series `Flagged`, to be drawn as markers).
    pub fn chart_spec(&self, title: &str, x_label: &str) -> ChartSpec {
        let x: Vec<f64> = (1..=self.values.len()).map(|i| i as f64).collect();
        let flagged = self.flagged();
        ChartSpec {
            kind: ChartKind::Line,
            title: title.to_string(),
            x_label: x_label.to_string(),
            y_label: self.name.clone(),
            categories: Vec::new(),
            series: vec![
                Series { name: self.name.clone(), x: x.clone(), y: self.values.clone() },
                Series { name: "UCL".into(), x: x.clone(), y: self.ucl.clone() },
                Series { name: "Center".into(), x: x.clone(), y: self.center.clone() },
                Series { name: "LCL".into(), x: x.clone(), y: self.lcl.clone() },
                Series {
                    name: "Flagged".into(),
                    x: flagged.iter().map(|&i| (i + 1) as f64).collect(),
                    y: flagged.iter().map(|&i| self.values[i]).collect(),
                },
            ],
            reference_lines: Vec::new(),
        }
    }
}

/// Which variable chart a [`VariableChart`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableChartKind {
    /// Individuals and moving range.
    IMr,
    /// Subgroup means and ranges.
    XbarR,
    /// Subgroup means and standard deviations.
    XbarS,
}

/// A location/dispersion chart pair.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableChart {
    /// Chart kind.
    pub kind: VariableChartKind,
    /// Individuals or subgroup means.
    pub location: ChartPanel,
    /// Moving ranges, ranges or standard deviations.
    pub dispersion: ChartPanel,
    /// Process mean used for the centre line.
    pub mean: f64,
    /// Process σ used for the limits.
    pub sigma: f64,
    /// Description of the σ estimate.
    pub sigma_method: String,
    /// Subgroup sizes (one entry per point; all 1 for I-MR).
    pub sizes: Vec<usize>,
}

impl VariableChart {
    /// Points flagged on either panel, as `(location, dispersion)`.
    pub fn flagged(&self) -> (Vec<usize>, Vec<usize>) {
        (self.location.flagged(), self.dispersion.flagged())
    }
}

// ----------------------------------------------------------------- helpers

pub(crate) fn mean_of(v: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for &x in v {
        let y = x - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    s / v.len() as f64
}

pub(crate) fn median_of(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

pub(crate) fn check_finite(data: &[f64]) -> Result<()> {
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    Ok(())
}

// ------------------------------------------------------------------- I-MR

/// σ estimate from individual observations in time order (moving ranges of span `w`).
pub(crate) fn individuals_sigma_estimate(data: &[f64], w: usize, method: IndividualsSigma) -> Result<(f64, String)> {
    let n = data.len();
    let kc = constants(w)?;
    let mut mr = Vec::new();
    for i in (w - 1)..n {
        let win = &data[i + 1 - w..=i];
        if win.iter().all(|v| !v.is_nan()) {
            let (lo, hi) = win.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
            mr.push(hi - lo);
        }
    }
    if mr.is_empty() {
        return Err(StatsError::InsufficientData {
            what: "moving-range sigma estimate",
            needed: w,
            got: data.iter().filter(|v| !v.is_nan()).count(),
        });
    }
    Ok(match method {
        IndividualsSigma::AverageMovingRange => (mean_of(&mr) / kc.d2, format!("average moving range (span {w})")),
        IndividualsSigma::MedianMovingRange => (median_of(&mr) / kc.d4, format!("median moving range (span {w})")),
        IndividualsSigma::Mssd => {
            if w != 2 {
                return invalid("the MSSD estimate needs span 2");
            }
            let d: Vec<f64> = data
                .windows(2)
                .filter(|p| !p[0].is_nan() && !p[1].is_nan())
                .map(|p| (p[1] - p[0]) * (p[1] - p[0]))
                .collect();
            ((mean_of(&d) / 2.0).sqrt(), "square root of MSSD/2".to_string())
        }
    })
}

/// Individuals and moving-range chart.
pub fn i_mr(data: &[f64], opts: &ChartOptions) -> Result<VariableChart> {
    opts.check()?;
    check_finite(data)?;
    let w = opts.span;
    if !(2..=25).contains(&w) {
        return invalid(format!("moving-range span {w} must be in 2..=25"));
    }
    let n = data.len();
    let valid: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let mut mr = vec![f64::NAN; n];
    for i in (w - 1)..n {
        let win = &data[i + 1 - w..=i];
        if win.iter().all(|v| !v.is_nan()) {
            let (lo, hi) = win.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
            mr[i] = hi - lo;
        }
    }
    let mr_valid: Vec<f64> = mr.iter().copied().filter(|v| !v.is_nan()).collect();
    if mr_valid.len() < 2 && opts.known_sigma.is_none() {
        return Err(StatsError::InsufficientData { what: "I-MR chart", needed: w + 1, got: valid.len() });
    }
    if valid.is_empty() {
        return Err(StatsError::InsufficientData { what: "I-MR chart", needed: 1, got: 0 });
    }
    let kc = constants(w)?;
    let (sigma, method) = match opts.known_sigma {
        Some(s) => (s, "known".to_string()),
        None => individuals_sigma_estimate(data, w, opts.individuals_sigma)?,
    };
    if sigma.is_nan() || sigma <= 0.0 {
        return invalid("the estimated process σ is zero (no variation); limits are undefined");
    }
    let mean = opts.known_center.unwrap_or_else(|| mean_of(&valid));
    let k = opts.k_sigma;
    let loc = ChartPanel::build(
        "Individual Value",
        data.to_vec(),
        vec![mean; n],
        vec![mean + k * sigma; n],
        vec![mean - k * sigma; n],
        vec![sigma; n],
        &opts.tests,
    )?;
    let disp = ChartPanel::build(
        "Moving Range",
        mr,
        vec![kc.d2 * sigma; n],
        vec![(kc.d2 + k * kc.d3) * sigma; n],
        vec![((kc.d2 - k * kc.d3) * sigma).max(0.0); n],
        vec![kc.d3 * sigma; n],
        &opts.dispersion_tests,
    )?;
    Ok(VariableChart {
        kind: VariableChartKind::IMr,
        location: loc,
        dispersion: disp,
        mean,
        sigma,
        sigma_method: method,
        sizes: vec![1; n],
    })
}

// -------------------------------------------------------------- subgroups

/// Summary of one subgroup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubgroupStat {
    /// Number of observations.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Sample standard deviation (n − 1).
    pub sd: f64,
    /// Range (`NaN` when only summaries are known).
    pub range: f64,
}

/// Split a column into consecutive subgroups of `size` (a shorter final subgroup is kept).
pub fn subgroups_by_size(data: &[f64], size: usize) -> Result<Vec<Vec<f64>>> {
    if size == 0 {
        return invalid("subgroup size must be at least 1");
    }
    Ok(data.chunks(size).map(<[f64]>::to_vec).collect())
}

/// Subgroup summaries (missing values dropped; every subgroup needs ≥ 2 observations).
pub fn subgroup_stats(groups: &[Vec<f64>]) -> Result<Vec<SubgroupStat>> {
    groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            check_finite(g)?;
            let v: Vec<f64> = g.iter().copied().filter(|x| !x.is_nan()).collect();
            if v.len() < 2 {
                return invalid(format!(
                    "subgroup {} has {} usable observation(s); at least 2 are needed",
                    i + 1,
                    v.len()
                ));
            }
            let m = mean_of(&v);
            let ss: f64 = v.iter().map(|x| (x - m) * (x - m)).sum();
            let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
            Ok(SubgroupStat { n: v.len(), mean: m, sd: (ss / (v.len() - 1) as f64).sqrt(), range: hi - lo })
        })
        .collect()
}

pub(crate) fn estimate_sigma(stats: &[SubgroupStat], method: SubgroupSigma) -> Result<(f64, String)> {
    match method {
        SubgroupSigma::RBar => {
            let (mut num, mut den) = (0.0, 0.0);
            for s in stats {
                if s.range.is_nan() {
                    return invalid("the R̄ estimate needs subgroup ranges");
                }
                let c = constants(s.n)?;
                let w = (c.d2 / c.d3) * (c.d2 / c.d3);
                num += w * s.range / c.d2;
                den += w;
            }
            Ok((num / den, "R̄ / d2".into()))
        }
        SubgroupSigma::SBar => {
            let (mut num, mut den) = (0.0, 0.0);
            for s in stats {
                let c4 = c4_real(s.n as f64)?;
                let w = c4 * c4 / (1.0 - c4 * c4);
                num += w * s.sd / c4;
                den += w;
            }
            Ok((num / den, "s̄ / c4".into()))
        }
        SubgroupSigma::Pooled => {
            let df: f64 = stats.iter().map(|s| (s.n - 1) as f64).sum();
            let ss: f64 = stats.iter().map(|s| (s.n - 1) as f64 * s.sd * s.sd).sum();
            Ok(((ss / df).sqrt() / c4_real(df + 1.0)?, "pooled standard deviation / c4".into()))
        }
    }
}

pub(crate) fn subgroup_chart(
    stats: &[SubgroupStat],
    opts: &ChartOptions,
    kind: VariableChartKind,
) -> Result<VariableChart> {
    opts.check()?;
    if stats.is_empty() {
        return Err(StatsError::InsufficientData { what: "subgroup chart", needed: 1, got: 0 });
    }
    if stats.iter().any(|s| s.n < 2 || !s.mean.is_finite() || s.sd.is_nan() || s.sd < 0.0) {
        return invalid("every subgroup needs n ≥ 2, a finite mean and a standard deviation");
    }
    let default = if kind == VariableChartKind::XbarR { SubgroupSigma::RBar } else { SubgroupSigma::SBar };
    let (sigma, method) = match opts.known_sigma {
        Some(s) => (s, "known".to_string()),
        None => estimate_sigma(stats, opts.subgroup_sigma.unwrap_or(default))?,
    };
    if sigma.is_nan() || sigma <= 0.0 {
        return invalid("the estimated process σ is zero (no variation); limits are undefined");
    }
    let total: f64 = stats.iter().map(|s| s.n as f64).sum();
    let mean = opts.known_center.unwrap_or_else(|| stats.iter().map(|s| s.n as f64 * s.mean).sum::<f64>() / total);
    let k = opts.k_sigma;
    let m = stats.len();
    let se: Vec<f64> = stats.iter().map(|s| sigma / (s.n as f64).sqrt()).collect();
    let loc = ChartPanel::build(
        "Sample Mean",
        stats.iter().map(|s| s.mean).collect(),
        vec![mean; m],
        se.iter().map(|e| mean + k * e).collect(),
        se.iter().map(|e| mean - k * e).collect(),
        se,
        &opts.tests,
    )?;
    let (mut c, mut u, mut l, mut sg, mut vals) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in stats {
        if kind == VariableChartKind::XbarR {
            let kc = constants(s.n)?;
            c.push(kc.d2 * sigma);
            u.push((kc.d2 + k * kc.d3) * sigma);
            l.push(((kc.d2 - k * kc.d3) * sigma).max(0.0));
            sg.push(kc.d3 * sigma);
            vals.push(s.range);
        } else {
            let c4 = c4_real(s.n as f64)?;
            let sd = (1.0 - c4 * c4).sqrt();
            c.push(c4 * sigma);
            u.push((c4 + k * sd) * sigma);
            l.push(((c4 - k * sd) * sigma).max(0.0));
            sg.push(sd * sigma);
            vals.push(s.sd);
        }
    }
    let name = if kind == VariableChartKind::XbarR { "Sample Range" } else { "Sample StDev" };
    let disp = ChartPanel::build(name, vals, c, u, l, sg, &opts.dispersion_tests)?;
    Ok(VariableChart {
        kind,
        location: loc,
        dispersion: disp,
        mean,
        sigma,
        sigma_method: method,
        sizes: stats.iter().map(|s| s.n).collect(),
    })
}

/// X̄-R chart from subgroup data (sizes 2 … 25).
pub fn xbar_r(groups: &[Vec<f64>], opts: &ChartOptions) -> Result<VariableChart> {
    subgroup_chart(&subgroup_stats(groups)?, opts, VariableChartKind::XbarR)
}

/// X̄-S chart from subgroup data.
pub fn xbar_s(groups: &[Vec<f64>], opts: &ChartOptions) -> Result<VariableChart> {
    subgroup_chart(&subgroup_stats(groups)?, opts, VariableChartKind::XbarS)
}

/// X̄-R chart from subgroup summaries (needs `range`).
pub fn xbar_r_from_stats(stats: &[SubgroupStat], opts: &ChartOptions) -> Result<VariableChart> {
    if stats.iter().any(|s| s.range.is_nan()) {
        return invalid("an R chart needs the subgroup ranges");
    }
    subgroup_chart(stats, opts, VariableChartKind::XbarR)
}

/// X̄-S chart from subgroup summaries (`range` is ignored).
pub fn xbar_s_from_stats(stats: &[SubgroupStat], opts: &ChartOptions) -> Result<VariableChart> {
    subgroup_chart(stats, opts, VariableChartKind::XbarS)
}

// -------------------------------------------------------------- attributes

/// Attribute chart kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeKind {
    /// Proportion of nonconforming units.
    P,
    /// Number of nonconforming units (constant sample size).
    Np,
    /// Number of defects per constant inspection unit.
    C,
    /// Defects per unit (varying inspection units).
    U,
    /// Laney P′ (proportions with the between-sample variation of z-scores).
    LaneyP,
    /// Laney U′ (defects per unit with the between-sample variation of z-scores).
    LaneyU,
}

/// An attribute control chart.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeChart {
    /// Chart kind.
    pub kind: AttributeKind,
    /// The plotted statistic.
    pub panel: ChartPanel,
    /// Overall centre value (p̄, n·p̄, c̄ or ū).
    pub center: f64,
    /// Counts of nonconforming units or defects.
    pub counts: Vec<f64>,
    /// Sample sizes (inspection units).
    pub sizes: Vec<f64>,
}

pub(crate) fn check_attribute(counts: &[f64], sizes: &[f64], opts: &ChartOptions, whole_sizes: bool) -> Result<()> {
    opts.check()?;
    if opts.tests.uses_zone_tests() {
        return invalid("attribute charts support tests 1–4 only");
    }
    if counts.is_empty() {
        return Err(StatsError::InsufficientData { what: "attribute chart", needed: 1, got: 0 });
    }
    if counts.len() != sizes.len() {
        return invalid("counts and sample sizes must have the same length");
    }
    if counts.iter().any(|c| !(c.is_finite() && *c >= 0.0)) {
        return invalid("counts must be finite and non-negative");
    }
    if sizes.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return invalid("sample sizes must be positive");
    }
    if whole_sizes && sizes.iter().any(|s| s.fract() != 0.0) {
        return invalid("sample sizes must be whole numbers");
    }
    Ok(())
}

pub(crate) fn attribute_chart(
    kind: AttributeKind,
    name: &str,
    counts: &[f64],
    sizes: &[f64],
    values: Vec<f64>,
    center: f64,
    sig: Vec<f64>,
    lo: f64,
    hi: Vec<f64>,
    opts: &ChartOptions,
) -> Result<AttributeChart> {
    let k = opts.k_sigma;
    let n = values.len();
    let centers = match kind {
        AttributeKind::Np => vec![center; n],
        _ => vec![center; n],
    };
    let ucl: Vec<f64> = (0..n).map(|i| (center + k * sig[i]).min(hi[i])).collect();
    let lcl: Vec<f64> = (0..n).map(|i| (center - k * sig[i]).max(lo)).collect();
    let panel = ChartPanel::build(name, values, centers, ucl, lcl, sig, &opts.tests)?;
    Ok(AttributeChart { kind, panel, center, counts: counts.to_vec(), sizes: sizes.to_vec() })
}

/// p chart: `defectives[i]` nonconforming units in a sample of `sizes[i]`.
pub fn p_chart(defectives: &[f64], sizes: &[f64], opts: &ChartOptions) -> Result<AttributeChart> {
    check_attribute(defectives, sizes, opts, true)?;
    if defectives.iter().zip(sizes).any(|(d, n)| d > n) {
        return invalid("a count exceeds its sample size");
    }
    let pbar = opts.known_center.unwrap_or_else(|| defectives.iter().sum::<f64>() / sizes.iter().sum::<f64>());
    if !(0.0..=1.0).contains(&pbar) {
        return invalid("the process proportion must lie in [0, 1]");
    }
    let values = defectives.iter().zip(sizes).map(|(d, n)| d / n).collect();
    let sig = sizes.iter().map(|n| (pbar * (1.0 - pbar) / n).sqrt()).collect();
    attribute_chart(
        AttributeKind::P,
        "Proportion",
        defectives,
        sizes,
        values,
        pbar,
        sig,
        0.0,
        vec![1.0; sizes.len()],
        opts,
    )
}

/// np chart (constant sample size).
pub fn np_chart(defectives: &[f64], size: f64, opts: &ChartOptions) -> Result<AttributeChart> {
    let sizes = vec![size; defectives.len()];
    check_attribute(defectives, &sizes, opts, true)?;
    if defectives.iter().any(|&d| d > size) {
        return invalid("a count exceeds the sample size");
    }
    let pbar = opts
        .known_center
        .map(|c| c / size)
        .unwrap_or_else(|| defectives.iter().sum::<f64>() / (size * defectives.len() as f64));
    if !(0.0..=1.0).contains(&pbar) {
        return invalid("the process proportion must lie in [0, 1]");
    }
    let sig = vec![(size * pbar * (1.0 - pbar)).sqrt(); defectives.len()];
    attribute_chart(
        AttributeKind::Np,
        "Number Nonconforming",
        defectives,
        &sizes,
        defectives.to_vec(),
        size * pbar,
        sig,
        0.0,
        vec![size; defectives.len()],
        opts,
    )
}

/// c chart: defects per (constant) inspection unit.
pub fn c_chart(counts: &[f64], opts: &ChartOptions) -> Result<AttributeChart> {
    let sizes = vec![1.0; counts.len()];
    check_attribute(counts, &sizes, opts, false)?;
    let cbar = opts.known_center.unwrap_or_else(|| counts.iter().sum::<f64>() / counts.len() as f64);
    if cbar.is_nan() || cbar < 0.0 {
        return invalid("the process mean must be non-negative");
    }
    let sig = vec![cbar.sqrt(); counts.len()];
    attribute_chart(
        AttributeKind::C,
        "Sample Count",
        counts,
        &sizes,
        counts.to_vec(),
        cbar,
        sig,
        0.0,
        vec![f64::INFINITY; counts.len()],
        opts,
    )
}

/// u chart: `counts[i]` defects in `sizes[i]` inspection units.
pub fn u_chart(counts: &[f64], sizes: &[f64], opts: &ChartOptions) -> Result<AttributeChart> {
    check_attribute(counts, sizes, opts, false)?;
    let ubar = opts.known_center.unwrap_or_else(|| counts.iter().sum::<f64>() / sizes.iter().sum::<f64>());
    if ubar.is_nan() || ubar < 0.0 {
        return invalid("the process mean must be non-negative");
    }
    let values = counts.iter().zip(sizes).map(|(c, n)| c / n).collect();
    let sig = sizes.iter().map(|n| (ubar / n).sqrt()).collect();
    attribute_chart(
        AttributeKind::U,
        "Sample Count Per Unit",
        counts,
        sizes,
        values,
        ubar,
        sig,
        0.0,
        vec![f64::INFINITY; counts.len()],
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imr_hand_computed() {
        // x = 1, 3, 2, 5, 4: MR = 2, 1, 3, 1 → MRbar = 1.75; σ = 1.75/d2(2).
        let data = [1.0, 3.0, 2.0, 5.0, 4.0];
        let c = i_mr(&data, &ChartOptions::default()).unwrap();
        let d2 = constants(2).unwrap().d2;
        assert!((c.sigma - 1.75 / d2).abs() < 1e-12);
        assert!((c.mean - 3.0).abs() < 1e-14);
        assert!((c.location.ucl[0] - (3.0 + 3.0 * 1.75 / d2)).abs() < 1e-12);
        assert!(c.dispersion.values[0].is_nan());
        assert_eq!(&c.dispersion.values[1..], &[2.0, 1.0, 3.0, 1.0]);
        // MR chart limits: D4 · MRbar, LCL 0 for span 2.
        let d4 = constants(2).unwrap().big_d4;
        assert!((c.dispersion.ucl[1] - d4 * 1.75).abs() < 1e-12);
        assert_eq!(c.dispersion.lcl[1], 0.0);
        assert!((c.dispersion.center[1] - 1.75).abs() < 1e-12);
    }

    #[test]
    fn imr_sigma_methods_and_missing() {
        let data = [10.0, 12.0, 11.0, f64::NAN, 13.0, 12.0, 14.0];
        let avg = i_mr(&data, &ChartOptions::default()).unwrap();
        // Moving ranges touching the missing value are missing.
        assert!(avg.dispersion.values[3].is_nan() && avg.dispersion.values[4].is_nan());
        let med =
            i_mr(&data, &ChartOptions { individuals_sigma: IndividualsSigma::MedianMovingRange, ..Default::default() })
                .unwrap();
        // MR = 2, 1, 1, 2 → median 1.5 / d4(2).
        assert!((med.sigma - 1.5 / constants(2).unwrap().d4).abs() < 1e-12);
        let mssd = i_mr(
            &[1.0, 3.0, 2.0, 5.0, 4.0],
            &ChartOptions { individuals_sigma: IndividualsSigma::Mssd, ..Default::default() },
        )
        .unwrap();
        assert!((mssd.sigma - ((4.0 + 1.0 + 9.0 + 1.0) / 4.0 / 2.0f64).sqrt()).abs() < 1e-12);
        let known =
            i_mr(&data, &ChartOptions { known_center: Some(12.0), known_sigma: Some(1.0), ..Default::default() })
                .unwrap();
        assert_eq!((known.mean, known.sigma), (12.0, 1.0));
        assert!(i_mr(&[1.0, 1.0, 1.0, 1.0], &ChartOptions::default()).is_err());
        assert!(i_mr(&[1.0, 2.0], &ChartOptions::default()).is_err());
    }

    #[test]
    fn imr_span_three() {
        let data = [1.0, 4.0, 2.0, 8.0, 5.0, 6.0];
        let c = i_mr(&data, &ChartOptions { span: 3, ..Default::default() }).unwrap();
        assert!(c.dispersion.values[0].is_nan() && c.dispersion.values[1].is_nan());
        assert_eq!(&c.dispersion.values[2..], &[3.0, 6.0, 6.0, 3.0]);
        assert!((c.sigma - 4.5 / constants(3).unwrap().d2).abs() < 1e-12);
    }

    #[test]
    fn xbar_r_equal_sizes_matches_textbook_factors() {
        let groups = vec![vec![5.0, 7.0, 6.0, 8.0, 4.0], vec![6.0, 6.0, 7.0, 9.0, 5.0], vec![4.0, 5.0, 6.0, 7.0, 8.0]];
        let c = xbar_r(&groups, &ChartOptions::default()).unwrap();
        let k = constants(5).unwrap();
        let rbar = (4.0 + 4.0 + 4.0) / 3.0;
        let xbb = (6.0 + 6.6 + 6.0) / 3.0;
        assert!((c.location.ucl[0] - (xbb + k.a2 * rbar)).abs() < 1e-12);
        assert!((c.location.lcl[2] - (xbb - k.a2 * rbar)).abs() < 1e-12);
        assert!((c.dispersion.ucl[0] - k.big_d4 * rbar).abs() < 1e-12);
        assert!((c.dispersion.center[0] - rbar).abs() < 1e-12);
        assert_eq!(c.dispersion.lcl[0], 0.0);
    }

    #[test]
    fn xbar_s_equal_sizes_matches_textbook_factors() {
        let groups = vec![vec![5.0, 7.0, 6.0, 8.0, 4.0], vec![6.0, 6.0, 7.0, 9.0, 5.0], vec![4.0, 5.0, 6.0, 7.0, 8.0]];
        let c = xbar_s(&groups, &ChartOptions::default()).unwrap();
        let k = constants(5).unwrap();
        let st = subgroup_stats(&groups).unwrap();
        let sbar = st.iter().map(|s| s.sd).sum::<f64>() / 3.0;
        assert!((c.location.ucl[1] - (c.mean + k.a3 * sbar)).abs() < 1e-12);
        assert!((c.dispersion.ucl[0] - k.b4 * sbar).abs() < 1e-12);
        assert!((c.dispersion.lcl[0] - k.b3 * sbar).abs() < 1e-12);
    }

    #[test]
    fn variable_subgroup_sizes_and_pooled_sigma() {
        let groups = vec![vec![5.0, 7.0, 6.0], vec![6.0, 6.0, 7.0, 9.0, 5.0], vec![4.0, 5.0, 6.0, 7.0]];
        let c = xbar_s(&groups, &ChartOptions { subgroup_sigma: Some(SubgroupSigma::Pooled), ..Default::default() })
            .unwrap();
        assert_eq!(c.sizes, vec![3, 5, 4]);
        // Limits widen for the smaller subgroup.
        assert!(c.location.ucl[0] - c.mean > c.location.ucl[1] - c.mean);
        let st = subgroup_stats(&groups).unwrap();
        let df = 2.0 + 4.0 + 3.0;
        let sp = ((2.0 * st[0].sd.powi(2) + 4.0 * st[1].sd.powi(2) + 3.0 * st[2].sd.powi(2)) / df).sqrt();
        assert!((c.sigma - sp / c4_real(df + 1.0).unwrap()).abs() < 1e-12);
        // Weighted mean for the centre line.
        let want = (3.0 * st[0].mean + 5.0 * st[1].mean + 4.0 * st[2].mean) / 12.0;
        assert!((c.mean - want).abs() < 1e-12);
        assert!(xbar_r(&[vec![1.0], vec![2.0, 3.0]], &ChartOptions::default()).is_err());
    }

    #[test]
    fn attribute_charts() {
        let d = [4.0, 6.0, 5.0, 14.0, 3.0];
        let p = p_chart(&d, &[100.0; 5], &ChartOptions::default()).unwrap();
        let pbar = 32.0 / 500.0;
        assert!((p.center - pbar).abs() < 1e-15);
        let s = (pbar * (1.0 - pbar) / 100.0).sqrt();
        assert!((p.panel.ucl[0] - (pbar + 3.0 * s)).abs() < 1e-14);
        assert_eq!(p.panel.beyond_limits(), vec![3]);
        let np = np_chart(&d, 100.0, &ChartOptions::default()).unwrap();
        assert!((np.center - 6.4).abs() < 1e-12);
        assert!((np.panel.ucl[0] - (6.4 + 3.0 * (100.0 * pbar * (1.0 - pbar)).sqrt())).abs() < 1e-12);
        let c = c_chart(&d, &ChartOptions::default()).unwrap();
        assert!((c.panel.ucl[0] - (6.4 + 3.0 * 6.4f64.sqrt())).abs() < 1e-12);
        let u = u_chart(&d, &[2.0, 2.0, 2.0, 4.0, 1.0], &ChartOptions::default()).unwrap();
        assert!((u.center - 32.0 / 11.0).abs() < 1e-12);
        assert!(u.panel.ucl[4] > u.panel.ucl[3]); // smaller sample, wider limits
                                                  // Limits are truncated.
        let p0 = p_chart(&[0.0, 1.0, 0.0], &[10.0; 3], &ChartOptions::default()).unwrap();
        assert!(p0.panel.lcl.iter().all(|&l| l == 0.0));
        assert!(p_chart(&[11.0], &[10.0], &ChartOptions::default()).is_err());
        assert!(p_chart(&[1.0], &[10.0, 10.0], &ChartOptions::default()).is_err());
        assert!(c_chart(&d, &ChartOptions { tests: SpecialCauseTests::all(), ..Default::default() }).is_err());
    }

    #[test]
    fn tests_run_on_chart_points() {
        // A chart with a clear shift: nine points above the centre line trigger test 2.
        let mut data: Vec<f64> = (0..20).map(|i| 10.0 + if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        data.extend([11.2; 9]);
        let opts = ChartOptions { tests: SpecialCauseTests::first_four(), ..Default::default() };
        let c = i_mr(&data, &opts).unwrap();
        let t2 = c.location.violations.iter().find(|v| v.test == 2).expect("test 2 fires");
        assert_eq!(*t2.points.last().unwrap(), data.len() - 1);
        let spec = c.location.chart_spec("I Chart", "Observation");
        assert_eq!(spec.series.len(), 5);
        assert_eq!(spec.series[0].y.len(), data.len());
    }
}
