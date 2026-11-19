//! Capability Sixpack: control charts, last subgroups, capability histogram,
//! normal probability plot and capability plot, with the capability statistics.

use super::capability::{capability_individuals, capability_subgroups, CapabilityOptions, NormalCapability, Specs};
use super::charts::{i_mr, subgroup_stats, xbar_r, xbar_s, ChartOptions, VariableChart};
use crate::error::{invalid, Result};
use crate::graphical::{
    anderson_darling, histogram, normal_probability_plot, AndersonDarling, Binning, Histogram, PlotPosition,
    ProbabilityPoint,
};
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};

/// Options of a sixpack.
#[derive(Debug, Clone, PartialEq)]
pub struct SixpackOptions {
    /// Capability options.
    pub capability: CapabilityOptions,
    /// Control-chart options.
    pub chart: ChartOptions,
    /// Subgroup data: `Some(true)` forces an X̄-S chart, `Some(false)` X̄-R;
    /// `None` uses X̄-R for subgroups of at most 8 and X̄-S otherwise.
    pub use_s_chart: Option<bool>,
    /// Number of most recent subgroups (observations) in the run plot (25).
    pub recent: usize,
}

impl Default for SixpackOptions {
    fn default() -> Self {
        SixpackOptions {
            capability: CapabilityOptions::default(),
            chart: ChartOptions::default(),
            use_s_chart: None,
            recent: 25,
        }
    }
}

/// The six panels of a Capability Sixpack.
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilitySixpack {
    /// The control-chart pair (I-MR, X̄-R or X̄-S).
    pub charts: VariableChart,
    /// Capability statistics.
    pub capability: NormalCapability,
    /// Index (0-based) of the first subgroup/observation in `recent`.
    pub recent_start: usize,
    /// Values of the most recent subgroups (a single value each for individuals).
    pub recent: Vec<Vec<f64>>,
    /// Histogram of all observations.
    pub histogram: Histogram,
    /// Normal probability plot points.
    pub normal_plot: Vec<ProbabilityPoint>,
    /// Anderson–Darling normality test.
    pub ad: AndersonDarling,
}

fn spec_lines(specs: &Specs) -> Vec<ReferenceLine> {
    let mut v = Vec::new();
    if let Some(l) = specs.lsl {
        v.push(ReferenceLine { axis: Axis::X, value: l, label: "LSL".into() });
    }
    if let Some(t) = specs.target {
        v.push(ReferenceLine { axis: Axis::X, value: t, label: "Target".into() });
    }
    if let Some(u) = specs.usl {
        v.push(ReferenceLine { axis: Axis::X, value: u, label: "USL".into() });
    }
    v
}

fn assemble(
    charts: VariableChart,
    capability: NormalCapability,
    recent_start: usize,
    recent: Vec<Vec<f64>>,
    all: &[f64],
) -> Result<CapabilitySixpack> {
    Ok(CapabilitySixpack {
        charts,
        capability,
        recent_start,
        recent,
        histogram: histogram(all, &Binning::Sturges)?,
        normal_plot: normal_probability_plot(all, PlotPosition::Benard)?,
        ad: anderson_darling(all)?,
    })
}

/// Sixpack for individual observations in time order (I-MR charts).
pub fn capability_sixpack_individuals(data: &[f64], specs: &Specs, opts: &SixpackOptions) -> Result<CapabilitySixpack> {
    if opts.recent == 0 {
        return invalid("recent must be at least 1");
    }
    let mut chart_opts = opts.chart.clone();
    chart_opts.span = opts.capability.span;
    chart_opts.individuals_sigma = opts.capability.individuals_sigma;
    let charts = i_mr(data, &chart_opts)?;
    let capability = capability_individuals(data, specs, &opts.capability)?;
    let valid: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let start = data.len().saturating_sub(opts.recent);
    let recent = data[start..].iter().map(|&v| vec![v]).collect();
    assemble(charts, capability, start, recent, &valid)
}

/// Sixpack for subgroup data (X̄-R or X̄-S charts).
pub fn capability_sixpack_subgroups(
    groups: &[Vec<f64>],
    specs: &Specs,
    opts: &SixpackOptions,
) -> Result<CapabilitySixpack> {
    if opts.recent == 0 {
        return invalid("recent must be at least 1");
    }
    let stats = subgroup_stats(groups)?;
    let max_n = stats.iter().map(|s| s.n).max().unwrap_or(0);
    let s_chart = opts.use_s_chart.unwrap_or(max_n > 8);
    let charts = if s_chart { xbar_s(groups, &opts.chart)? } else { xbar_r(groups, &opts.chart)? };
    let capability = capability_subgroups(groups, specs, &opts.capability)?;
    let all: Vec<f64> = groups.iter().flatten().copied().filter(|v| !v.is_nan()).collect();
    let start = groups.len().saturating_sub(opts.recent);
    assemble(charts, capability, start, groups[start..].to_vec(), &all)
}

impl CapabilitySixpack {
    /// The six panels as chart specifications, in Minitab's order.
    pub fn charts(&self) -> Vec<ChartSpec> {
        let c = &self.capability;
        let kind_name = match self.charts.kind {
            super::charts::VariableChartKind::IMr => ("I Chart", "MR Chart", "Observation"),
            super::charts::VariableChartKind::XbarR => ("Xbar Chart", "R Chart", "Sample"),
            super::charts::VariableChartKind::XbarS => ("Xbar Chart", "S Chart", "Sample"),
        };
        let first = self.charts.location.chart_spec(kind_name.0, kind_name.2);
        let second = self.charts.dispersion.chart_spec(kind_name.1, kind_name.2);
        let recent = {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for (i, g) in self.recent.iter().enumerate() {
                for &v in g {
                    x.push((self.recent_start + i + 1) as f64);
                    y.push(v);
                }
            }
            ChartSpec {
                kind: ChartKind::Scatter,
                title: format!(
                    "Last {} {}",
                    self.recent.len(),
                    if self.charts.kind == super::charts::VariableChartKind::IMr {
                        "Observations"
                    } else {
                        "Subgroups"
                    }
                ),
                x_label: kind_name.2.into(),
                y_label: "Values".into(),
                categories: Vec::new(),
                series: vec![Series { name: "Values".into(), x, y }],
                reference_lines: vec![ReferenceLine { axis: Axis::Y, value: c.mean, label: "Mean".into() }],
            }
        };
        let hist = ChartSpec {
            kind: ChartKind::Histogram,
            title: "Capability Histogram".into(),
            x_label: "Value".into(),
            y_label: "Frequency".into(),
            categories: Vec::new(),
            series: vec![Series {
                name: "Frequency".into(),
                x: self.histogram.edges.clone(),
                y: self.histogram.counts.iter().map(|&k| k as f64).collect(),
            }],
            reference_lines: spec_lines(&c.specs),
        };
        let prob = ChartSpec {
            kind: ChartKind::ProbabilityPlot,
            title: format!(
                "Normal Prob Plot (AD: {:.3}, P: {})",
                self.ad.a2,
                crate::report::format_p_value(self.ad.p_value)
            ),
            x_label: "Value".into(),
            y_label: "Normal score".into(),
            categories: Vec::new(),
            series: vec![Series {
                name: "Data".into(),
                x: self.normal_plot.iter().map(|p| p.x).collect(),
                y: self.normal_plot.iter().map(|p| p.z).collect(),
            }],
            reference_lines: Vec::new(),
        };
        let span = |s: f64| (c.mean - 3.0 * s, c.mean + 3.0 * s);
        let (wl, wu) = span(c.sd_within);
        let (ol, ou) = span(c.sd_overall);
        let capplot = ChartSpec {
            kind: ChartKind::IntervalPlot,
            title: "Capability Plot".into(),
            x_label: "Value".into(),
            y_label: "Spread".into(),
            categories: vec!["Within".into(), "Overall".into()],
            series: vec![
                Series { name: "estimate".into(), x: vec![c.mean, c.mean], y: vec![1.0, 2.0] },
                Series { name: "lower".into(), x: vec![wl, ol], y: vec![1.0, 2.0] },
                Series { name: "upper".into(), x: vec![wu, ou], y: vec![1.0, 2.0] },
            ],
            reference_lines: spec_lines(&c.specs),
        };
        vec![first, second, recent, hist, prob, capplot]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<f64> {
        (0..60).map(|i| 50.0 + (((i * 17) % 23) as f64 - 11.0) * 0.1).collect()
    }

    #[test]
    fn individuals_sixpack_has_six_consistent_panels() {
        let d = data();
        let s = capability_sixpack_individuals(
            &d,
            &Specs::two_sided(48.0, 52.0).with_target(50.0),
            &SixpackOptions::default(),
        )
        .unwrap();
        let ch = s.charts();
        assert_eq!(ch.len(), 6);
        assert_eq!(ch[0].title, "I Chart");
        assert_eq!(ch[1].title, "MR Chart");
        assert_eq!(s.recent.len(), 25);
        assert_eq!(s.recent_start, 35);
        assert_eq!(ch[2].series[0].x.len(), 25);
        assert_eq!(ch[3].reference_lines.len(), 3);
        assert_eq!(ch[4].series[0].x.len(), 60);
        assert_eq!(ch[5].series[1].x.len(), 2);
        // The capability plot spans ±3σ.
        let c = &s.capability;
        assert!((ch[5].series[2].x[0] - (c.mean + 3.0 * c.sd_within)).abs() < 1e-12);
        assert!((ch[5].series[1].x[1] - (c.mean - 3.0 * c.sd_overall)).abs() < 1e-12);
    }

    #[test]
    fn subgroup_sixpack_chooses_chart_by_size() {
        let d = data();
        let g5: Vec<Vec<f64>> = d.chunks(5).map(<[f64]>::to_vec).collect();
        let s = capability_sixpack_subgroups(&g5, &Specs::two_sided(48.0, 52.0), &SixpackOptions::default()).unwrap();
        assert_eq!(s.charts.dispersion.name, "Sample Range");
        assert_eq!(s.charts()[1].title, "R Chart");
        assert_eq!(s.recent.len(), 12);
        let g10: Vec<Vec<f64>> = d.chunks(10).map(<[f64]>::to_vec).collect();
        let s = capability_sixpack_subgroups(&g10, &Specs::two_sided(48.0, 52.0), &SixpackOptions::default()).unwrap();
        assert_eq!(s.charts.dispersion.name, "Sample StDev");
        assert!(capability_sixpack_subgroups(
            &g5,
            &Specs::two_sided(48.0, 52.0),
            &SixpackOptions { recent: 0, ..Default::default() }
        )
        .is_err());
    }
}
