//! Staged control limits: a separate chart is built for every stage (a
//! contiguous block of observations with the same stage label) and the panels
//! are joined, so the centre line and the limits step at every stage change.
//!
//! Special-cause tests restart in each stage: a pattern never spans a stage
//! boundary, and the first moving range of a stage is missing.

use super::attributes::{g_chart, laney_p, laney_u, t_chart_with, LaneyChart, RareEventChart, TimeModel};
use super::charts::{
    c_chart, i_mr, np_chart, p_chart, u_chart, xbar_r, xbar_s, AttributeChart, AttributeKind, ChartOptions, ChartPanel,
    VariableChart, VariableChartKind,
};
use super::multivariate::{hotelling_individuals, HotellingChart, HotellingOptions};
use super::rules::TestViolation;
use super::time_weighted::{
    cusum_tabular_sided, CusumChart, CusumOptions, CusumSides, SmoothingChart, TimeWeightedInput,
};
use crate::error::{invalid, Result};

/// A contiguous stage.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    /// Stage label (as given).
    pub label: String,
    /// First point of the stage (0-based).
    pub start: usize,
    /// Number of points in the stage.
    pub len: usize,
}

/// Split consecutive points into stages: a new stage starts whenever the label changes.
pub fn split_stages<S: AsRef<str>>(labels: &[S]) -> Vec<Stage> {
    let mut out: Vec<Stage> = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        match out.last_mut() {
            Some(s) if s.label == l.as_ref() => s.len += 1,
            _ => out.push(Stage { label: l.as_ref().to_string(), start: i, len: 1 }),
        }
    }
    out
}

fn check_labels(n: usize, labels: usize) -> Result<()> {
    if n != labels {
        return invalid(format!("{n} points but {labels} stage labels"));
    }
    if n == 0 {
        return invalid("no data");
    }
    Ok(())
}

/// Join per-stage panels into one panel; violations keep their per-stage
/// evaluation and are shifted to the global point numbers.
pub(crate) fn merge_panels(parts: &[(usize, &ChartPanel)]) -> ChartPanel {
    let first = parts[0].1;
    let mut m = ChartPanel {
        name: first.name.clone(),
        values: Vec::new(),
        center: Vec::new(),
        ucl: Vec::new(),
        lcl: Vec::new(),
        sigma: Vec::new(),
        tests: first.tests.clone(),
        violations: Vec::new(),
    };
    for &(offset, p) in parts {
        m.values.extend_from_slice(&p.values);
        m.center.extend_from_slice(&p.center);
        m.ucl.extend_from_slice(&p.ucl);
        m.lcl.extend_from_slice(&p.lcl);
        m.sigma.extend_from_slice(&p.sigma);
        for v in &p.violations {
            let pts = v.points.iter().map(|i| i + offset);
            match m.violations.iter_mut().find(|x| x.test == v.test) {
                Some(x) => x.points.extend(pts),
                None => m.violations.push(TestViolation { test: v.test, points: pts.collect() }),
            }
        }
    }
    m.violations.sort_by_key(|v| v.test);
    m
}

/// A staged variables chart.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedVariableChart {
    /// Chart kind.
    pub kind: VariableChartKind,
    /// The stages in order.
    pub stages: Vec<Stage>,
    /// The chart of each stage (limits, σ estimate and tests of that stage alone).
    pub charts: Vec<VariableChart>,
    /// Location panel over all points.
    pub location: ChartPanel,
    /// Dispersion panel over all points.
    pub dispersion: ChartPanel,
}

/// A staged attributes chart.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedAttributeChart {
    /// Chart kind.
    pub kind: AttributeKind,
    /// The stages in order.
    pub stages: Vec<Stage>,
    /// The chart of each stage.
    pub charts: Vec<AttributeChart>,
    /// The joined panel.
    pub panel: ChartPanel,
}

fn join_variable(kind: VariableChartKind, stages: Vec<Stage>, charts: Vec<VariableChart>) -> StagedVariableChart {
    let loc: Vec<(usize, &ChartPanel)> = stages.iter().zip(&charts).map(|(s, c)| (s.start, &c.location)).collect();
    let dis: Vec<(usize, &ChartPanel)> = stages.iter().zip(&charts).map(|(s, c)| (s.start, &c.dispersion)).collect();
    let (location, dispersion) = (merge_panels(&loc), merge_panels(&dis));
    StagedVariableChart { kind, stages, charts, location, dispersion }
}

fn join_attribute(kind: AttributeKind, stages: Vec<Stage>, charts: Vec<AttributeChart>) -> StagedAttributeChart {
    let parts: Vec<(usize, &ChartPanel)> = stages.iter().zip(&charts).map(|(s, c)| (s.start, &c.panel)).collect();
    let panel = merge_panels(&parts);
    StagedAttributeChart { kind, stages, charts, panel }
}

/// I-MR chart with a separate centre line and limits per stage.
pub fn staged_i_mr<S: AsRef<str>>(data: &[f64], stages: &[S], opts: &ChartOptions) -> Result<StagedVariableChart> {
    check_labels(data.len(), stages.len())?;
    let st = split_stages(stages);
    let charts = st.iter().map(|s| i_mr(&data[s.start..s.start + s.len], opts)).collect::<Result<Vec<_>>>()?;
    Ok(join_variable(VariableChartKind::IMr, st, charts))
}

/// X̄-R (`use_range`) or X̄-S chart with separate limits per stage; `stages` has one label per subgroup.
pub fn staged_xbar<S: AsRef<str>>(
    groups: &[Vec<f64>],
    stages: &[S],
    use_range: bool,
    opts: &ChartOptions,
) -> Result<StagedVariableChart> {
    check_labels(groups.len(), stages.len())?;
    let st = split_stages(stages);
    let mut charts = Vec::new();
    for s in &st {
        let g = &groups[s.start..s.start + s.len];
        charts.push(if use_range { xbar_r(g, opts)? } else { xbar_s(g, opts)? });
    }
    let kind = if use_range { VariableChartKind::XbarR } else { VariableChartKind::XbarS };
    Ok(join_variable(kind, st, charts))
}

/// Staged p chart.
pub fn staged_p<S: AsRef<str>>(
    defectives: &[f64],
    sizes: &[f64],
    stages: &[S],
    opts: &ChartOptions,
) -> Result<StagedAttributeChart> {
    check_labels(defectives.len(), stages.len())?;
    if sizes.len() != defectives.len() {
        return invalid("counts and sample sizes must have the same length");
    }
    let st = split_stages(stages);
    let charts = st
        .iter()
        .map(|s| p_chart(&defectives[s.start..s.start + s.len], &sizes[s.start..s.start + s.len], opts))
        .collect::<Result<Vec<_>>>()?;
    Ok(join_attribute(AttributeKind::P, st, charts))
}

/// Staged np chart (constant sample size).
pub fn staged_np<S: AsRef<str>>(
    defectives: &[f64],
    size: f64,
    stages: &[S],
    opts: &ChartOptions,
) -> Result<StagedAttributeChart> {
    check_labels(defectives.len(), stages.len())?;
    let st = split_stages(stages);
    let charts =
        st.iter().map(|s| np_chart(&defectives[s.start..s.start + s.len], size, opts)).collect::<Result<Vec<_>>>()?;
    Ok(join_attribute(AttributeKind::Np, st, charts))
}

/// Staged c chart.
pub fn staged_c<S: AsRef<str>>(counts: &[f64], stages: &[S], opts: &ChartOptions) -> Result<StagedAttributeChart> {
    check_labels(counts.len(), stages.len())?;
    let st = split_stages(stages);
    let charts = st.iter().map(|s| c_chart(&counts[s.start..s.start + s.len], opts)).collect::<Result<Vec<_>>>()?;
    Ok(join_attribute(AttributeKind::C, st, charts))
}

/// Staged u chart.
pub fn staged_u<S: AsRef<str>>(
    counts: &[f64],
    sizes: &[f64],
    stages: &[S],
    opts: &ChartOptions,
) -> Result<StagedAttributeChart> {
    check_labels(counts.len(), stages.len())?;
    if sizes.len() != counts.len() {
        return invalid("counts and sample sizes must have the same length");
    }
    let st = split_stages(stages);
    let charts = st
        .iter()
        .map(|s| u_chart(&counts[s.start..s.start + s.len], &sizes[s.start..s.start + s.len], opts))
        .collect::<Result<Vec<_>>>()?;
    Ok(join_attribute(AttributeKind::U, st, charts))
}

// ------------------------------------------------ staged time-weighted, Laney, rare-event, T²

fn per_stage<T, S: AsRef<str>>(
    n: usize,
    labels: &[S],
    f: impl Fn(usize, usize) -> Result<T>,
) -> Result<(Vec<Stage>, Vec<T>)> {
    check_labels(n, labels.len())?;
    let st = split_stages(labels);
    let charts = st.iter().map(|s| f(s.start, s.start + s.len)).collect::<Result<Vec<_>>>()?;
    Ok((st, charts))
}

/// A staged moving-average or EWMA chart: each stage is smoothed from its own target
/// (the EWMA restarts at the stage target) with its own σ.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedSmoothingChart {
    /// The stages.
    pub stages: Vec<Stage>,
    /// One chart per stage.
    pub charts: Vec<SmoothingChart>,
    /// Joined panel.
    pub panel: ChartPanel,
}

/// Staged smoothing chart from individual observations. `build` makes the chart of one stage, e.g.
/// `|inp| ewma(inp, 0.2, 3.0, true)`. `target` and `sigma` are fixed values for every stage, or
/// estimated per stage when `None`.
pub fn staged_smoothing_individuals<S: AsRef<str>>(
    data: &[f64],
    stages: &[S],
    target: Option<f64>,
    sigma: Option<f64>,
    build: impl Fn(&TimeWeightedInput) -> Result<SmoothingChart>,
) -> Result<StagedSmoothingChart> {
    let (st, charts) =
        per_stage(data.len(), stages, |a, b| build(&TimeWeightedInput::individuals(&data[a..b], target, sigma)?))?;
    let parts: Vec<(usize, &ChartPanel)> = st.iter().zip(&charts).map(|(s, c)| (s.start, &c.panel)).collect();
    let panel = merge_panels(&parts);
    Ok(StagedSmoothingChart { stages: st, charts, panel })
}

/// Staged smoothing chart from subgroup data (one stage label per subgroup).
pub fn staged_smoothing_subgroups<S: AsRef<str>>(
    groups: &[Vec<f64>],
    stages: &[S],
    target: Option<f64>,
    sigma: Option<f64>,
    build: impl Fn(&TimeWeightedInput) -> Result<SmoothingChart>,
) -> Result<StagedSmoothingChart> {
    let (st, charts) = per_stage(groups.len(), stages, |a, b| {
        build(&TimeWeightedInput::subgroups(&groups[a..b], target, sigma, None)?)
    })?;
    let parts: Vec<(usize, &ChartPanel)> = st.iter().zip(&charts).map(|(s, c)| (s.start, &c.panel)).collect();
    let panel = merge_panels(&parts);
    Ok(StagedSmoothingChart { stages: st, charts, panel })
}

/// A staged tabular CUSUM: the sums restart at the head start in every stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedCusumChart {
    /// The stages.
    pub stages: Vec<Stage>,
    /// One CUSUM per stage.
    pub charts: Vec<CusumChart>,
    /// Upper sums over all points.
    pub upper: Vec<f64>,
    /// Lower sums over all points.
    pub lower: Vec<f64>,
    /// Upper signals (global point numbers).
    pub signals_upper: Vec<usize>,
    /// Lower signals (global point numbers).
    pub signals_lower: Vec<usize>,
}

/// Staged CUSUM of individual observations (σ and target per stage unless given).
pub fn staged_cusum_individuals<S: AsRef<str>>(
    data: &[f64],
    stages: &[S],
    target: Option<f64>,
    sigma: Option<f64>,
    opts: &CusumOptions,
    sides: CusumSides,
) -> Result<StagedCusumChart> {
    let (st, charts) = per_stage(data.len(), stages, |a, b| {
        cusum_tabular_sided(&TimeWeightedInput::individuals(&data[a..b], target, sigma)?, opts, sides)
    })?;
    let mut out = StagedCusumChart {
        stages: st,
        charts,
        upper: vec![],
        lower: vec![],
        signals_upper: vec![],
        signals_lower: vec![],
    };
    for (s, c) in out.stages.iter().zip(&out.charts) {
        out.upper.extend_from_slice(&c.upper);
        out.lower.extend_from_slice(&c.lower);
        out.signals_upper.extend(c.signals_upper.iter().map(|i| i + s.start));
        out.signals_lower.extend(c.signals_lower.iter().map(|i| i + s.start));
    }
    Ok(out)
}

/// A staged Laney chart.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedLaneyChart {
    /// Chart kind (Laney P′ or U′).
    pub kind: AttributeKind,
    /// The stages.
    pub stages: Vec<Stage>,
    /// One chart per stage (each with its own σ_z).
    pub charts: Vec<LaneyChart>,
    /// Joined panel.
    pub panel: ChartPanel,
}

/// Staged Laney P′ chart.
pub fn staged_laney_p<S: AsRef<str>>(
    defectives: &[f64],
    sizes: &[f64],
    stages: &[S],
    opts: &ChartOptions,
) -> Result<StagedLaneyChart> {
    if sizes.len() != defectives.len() {
        return invalid("counts and sample sizes must have the same length");
    }
    let (st, charts) = per_stage(defectives.len(), stages, |a, b| laney_p(&defectives[a..b], &sizes[a..b], opts))?;
    let parts: Vec<(usize, &ChartPanel)> = st.iter().zip(&charts).map(|(s, c)| (s.start, &c.chart.panel)).collect();
    let panel = merge_panels(&parts);
    Ok(StagedLaneyChart { kind: AttributeKind::LaneyP, stages: st, charts, panel })
}

/// Staged Laney U′ chart.
pub fn staged_laney_u<S: AsRef<str>>(
    counts: &[f64],
    sizes: &[f64],
    stages: &[S],
    opts: &ChartOptions,
) -> Result<StagedLaneyChart> {
    if sizes.len() != counts.len() {
        return invalid("counts and sample sizes must have the same length");
    }
    let (st, charts) = per_stage(counts.len(), stages, |a, b| laney_u(&counts[a..b], &sizes[a..b], opts))?;
    let parts: Vec<(usize, &ChartPanel)> = st.iter().zip(&charts).map(|(s, c)| (s.start, &c.chart.panel)).collect();
    let panel = merge_panels(&parts);
    Ok(StagedLaneyChart { kind: AttributeKind::LaneyU, stages: st, charts, panel })
}

/// A staged G or T chart.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedRareEventChart {
    /// The stages.
    pub stages: Vec<Stage>,
    /// One chart per stage (own fitted model and limits).
    pub charts: Vec<RareEventChart>,
    /// Joined panel.
    pub panel: ChartPanel,
}

fn join_rare(st: Vec<Stage>, charts: Vec<RareEventChart>) -> StagedRareEventChart {
    let parts: Vec<(usize, &ChartPanel)> = st.iter().zip(&charts).map(|(s, c)| (s.start, &c.panel)).collect();
    let panel = merge_panels(&parts);
    StagedRareEventChart { stages: st, charts, panel }
}

/// Staged G chart.
pub fn staged_g<S: AsRef<str>>(between: &[f64], stages: &[S], opts: &ChartOptions) -> Result<StagedRareEventChart> {
    let (st, charts) = per_stage(between.len(), stages, |a, b| g_chart(&between[a..b], opts))?;
    Ok(join_rare(st, charts))
}

/// Staged T chart.
pub fn staged_t<S: AsRef<str>>(
    times: &[f64],
    stages: &[S],
    opts: &ChartOptions,
    model: TimeModel,
) -> Result<StagedRareEventChart> {
    let (st, charts) = per_stage(times.len(), stages, |a, b| t_chart_with(&times[a..b], opts, model))?;
    Ok(join_rare(st, charts))
}

/// A staged Hotelling T² chart of individual observations: mean, covariance and limit per stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedHotellingChart {
    /// The stages.
    pub stages: Vec<Stage>,
    /// One chart per stage.
    pub charts: Vec<HotellingChart>,
    /// T² of all points.
    pub t2: Vec<f64>,
    /// Upper limit of each point (steps at stage changes).
    pub ucl: Vec<f64>,
    /// Points above their stage's limit (global numbers).
    pub flagged: Vec<usize>,
}

/// Staged T² chart for individuals (each stage needs enough points for its own Phase I estimate).
pub fn staged_hotelling_individuals<S: AsRef<str>>(
    data: &[Vec<f64>],
    stages: &[S],
    opts: &HotellingOptions,
) -> Result<StagedHotellingChart> {
    let (st, charts) = per_stage(data.len(), stages, |a, b| hotelling_individuals(&data[a..b], opts))?;
    let mut out = StagedHotellingChart { stages: st, charts, t2: vec![], ucl: vec![], flagged: vec![] };
    for (s, c) in out.stages.iter().zip(&out.charts) {
        out.t2.extend_from_slice(&c.t2);
        out.ucl.extend(std::iter::repeat_n(c.ucl, c.t2.len()));
        out.flagged.extend(c.flagged().into_iter().map(|i| i + s.start));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_split_on_label_change() {
        let s = split_stages(&["a", "a", "b", "b", "b", "a"]);
        assert_eq!(s.len(), 3);
        assert_eq!((s[1].start, s[1].len, s[1].label.as_str()), (2, 3, "b"));
        assert_eq!((s[2].start, s[2].len), (5, 1));
    }

    #[test]
    fn staged_limits_step_and_match_separate_charts() {
        let a = [10.0, 10.4, 9.8, 10.1, 9.7, 10.3, 10.0, 9.9];
        let b = [20.0, 20.9, 19.2, 20.5, 19.6, 20.1, 20.8, 19.9];
        let data: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
        let labels: Vec<&str> = (0..16).map(|i| if i < 8 { "one" } else { "two" }).collect();
        let st = staged_i_mr(&data, &labels, &ChartOptions::default()).unwrap();
        let ca = i_mr(&a, &ChartOptions::default()).unwrap();
        let cb = i_mr(&b, &ChartOptions::default()).unwrap();
        assert_eq!(st.location.ucl[0], ca.location.ucl[0]);
        assert_eq!(st.location.ucl[8], cb.location.ucl[0]);
        assert_eq!(st.location.center.len(), 16);
        assert!(st.dispersion.values[8].is_nan(), "first moving range of a stage is missing");
        // one overall chart would flag stage two, the staged chart does not
        assert!(st.location.flagged().is_empty());
    }

    #[test]
    fn violations_are_shifted_to_global_points() {
        let mut a = vec![10.0, 10.2, 9.9, 10.1, 9.8, 10.0, 10.2, 9.9, 10.1, 9.8];
        a.push(30.0);
        let b = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.2];
        let data: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
        let labels: Vec<usize> = (0..data.len()).map(|i| usize::from(i >= a.len())).collect();
        let labels: Vec<String> = labels.iter().map(ToString::to_string).collect();
        let opts = ChartOptions { known_sigma: Some(0.2), known_center: None, ..ChartOptions::default() };
        let st = staged_i_mr(&data, &labels, &opts).unwrap();
        assert!(st.location.flagged().contains(&10));
    }

    #[test]
    fn label_length_must_match() {
        assert!(staged_i_mr(&[1.0, 2.0, 3.0], &["a", "b"], &ChartOptions::default()).is_err());
    }
}
