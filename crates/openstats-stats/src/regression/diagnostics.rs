//! Residual diagnostics of a regression fit.

use crate::error::Result;
use crate::graphical::{histogram, normal_probability_plot, Binning, Histogram, PlotPosition, ProbabilityPoint};
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};

/// Per-observation diagnostics (index = position among the rows used).
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// Fitted values (including any offset).
    pub fitted: Vec<f64>,
    /// Ordinary residuals `y − ŷ`.
    pub residuals: Vec<f64>,
    /// Leverage: diagonal of the hat matrix.
    pub leverage: Vec<f64>,
    /// Standardized (internally studentized) residuals `e/(s√(1−h))`.
    pub standardized: Vec<f64>,
    /// Studentized deleted (externally studentized) residuals
    /// `e/(s₍ᵢ₎√(1−h))`, s₍ᵢ₎ from the fit without observation i.
    pub deleted: Vec<f64>,
    /// Cook's distance `e²h / (p·s²·(1−h)²)`.
    pub cooks_d: Vec<f64>,
    /// DFITS (= DFFITS): `deleted · √(h/(1−h))`.
    pub dffits: Vec<f64>,
    /// PRESS residuals `e/(1−h)`.
    pub press_residuals: Vec<f64>,
    /// Durbin–Watson statistic of the residuals in row order.
    pub durbin_watson: f64,
}

/// Which residual to plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualKind {
    /// Ordinary residuals.
    Regular,
    /// Standardized residuals.
    Standardized,
    /// Studentized deleted residuals.
    Deleted,
}

/// Why an observation is listed as unusual (Minitab's flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnusualFlag {
    /// `R`: |standardized residual| > 2.
    LargeResidual,
    /// `X`: leverage > 3p/n (or > 0.99).
    HighLeverage,
    /// Cook's distance above the median of F(p, n−p).
    Influential,
}

/// An unusual observation.
#[derive(Debug, Clone, PartialEq)]
pub struct UnusualObservation {
    /// Index among the rows used.
    pub index: usize,
    /// Flags raised.
    pub flags: Vec<UnusualFlag>,
}

/// Data for the residual "four-in-one" plot.
#[derive(Debug, Clone, PartialEq)]
pub struct FourInOne {
    /// Residual type shown.
    pub kind: ResidualKind,
    /// Normal probability plot points of the residuals.
    pub normal_plot: Vec<ProbabilityPoint>,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Residuals of the chosen kind, in row order.
    pub residuals: Vec<f64>,
    /// Histogram of the residuals (Sturges bins).
    pub histogram: Histogram,
}

/// Durbin–Watson statistic.
pub(crate) fn durbin_watson(e: &[f64]) -> f64 {
    let den: f64 = e.iter().map(|v| v * v).sum();
    if e.len() < 2 || den == 0.0 {
        return f64::NAN;
    }
    e.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])).sum::<f64>() / den
}

impl Diagnostics {
    /// The residuals of the requested kind.
    pub fn residuals_of(&self, kind: ResidualKind) -> &[f64] {
        match kind {
            ResidualKind::Regular => &self.residuals,
            ResidualKind::Standardized => &self.standardized,
            ResidualKind::Deleted => &self.deleted,
        }
    }

    /// Data of the four-in-one plot (probability plot, versus fits,
    /// histogram, versus order).
    pub fn four_in_one(&self, kind: ResidualKind) -> Result<FourInOne> {
        let r = self.residuals_of(kind).to_vec();
        Ok(FourInOne {
            kind,
            normal_plot: normal_probability_plot(&r, PlotPosition::Benard)?,
            fitted: self.fitted.clone(),
            histogram: histogram(&r, &Binning::Sturges)?,
            residuals: r,
        })
    }
}

impl FourInOne {
    /// The four charts as specifications.
    pub fn charts(&self) -> Vec<ChartSpec> {
        let name = match self.kind {
            ResidualKind::Regular => "Residual",
            ResidualKind::Standardized => "Standardized Residual",
            ResidualKind::Deleted => "Deleted Residual",
        };
        let zero = ReferenceLine { axis: Axis::Y, value: 0.0, label: "0".into() };
        let order: Vec<f64> = (1..=self.residuals.len()).map(|i| i as f64).collect();
        vec![
            ChartSpec {
                kind: ChartKind::ProbabilityPlot,
                title: format!("Normal Probability Plot of {name}s"),
                x_label: name.into(),
                y_label: "Normal score".into(),
                categories: Vec::new(),
                series: vec![Series {
                    name: name.into(),
                    x: self.normal_plot.iter().map(|p| p.x).collect(),
                    y: self.normal_plot.iter().map(|p| p.z).collect(),
                }],
                reference_lines: Vec::new(),
            },
            ChartSpec {
                kind: ChartKind::Scatter,
                title: format!("{name}s versus Fits"),
                x_label: "Fitted Value".into(),
                y_label: name.into(),
                categories: Vec::new(),
                series: vec![Series { name: name.into(), x: self.fitted.clone(), y: self.residuals.clone() }],
                reference_lines: vec![zero.clone()],
            },
            ChartSpec {
                kind: ChartKind::Histogram,
                title: format!("Histogram of {name}s"),
                x_label: name.into(),
                y_label: "Frequency".into(),
                categories: Vec::new(),
                series: vec![Series {
                    name: "Frequency".into(),
                    x: self.histogram.edges.clone(),
                    y: self.histogram.counts.iter().map(|&c| c as f64).collect(),
                }],
                reference_lines: Vec::new(),
            },
            ChartSpec {
                kind: ChartKind::Line,
                title: format!("{name}s versus Order"),
                x_label: "Observation Order".into(),
                y_label: name.into(),
                categories: Vec::new(),
                series: vec![Series { name: name.into(), x: order, y: self.residuals.clone() }],
                reference_lines: vec![zero],
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durbin_watson_known_values() {
        // Alternating residuals → DW near 4; a smooth trend → near 0.
        let alt = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert!((durbin_watson(&alt) - 10.0 / 6.0 * 2.0).abs() < 1e-12);
        let trend = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let want = (1.0 + 1.0 + 4.0 + 1.0 + 1.0) / 28.0;
        assert!((durbin_watson(&trend) - want).abs() < 1e-12);
    }

    #[test]
    fn four_in_one_has_four_charts() {
        let d = Diagnostics {
            fitted: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            residuals: vec![0.3, -0.2, 0.1, -0.4, 0.5, -0.3],
            leverage: vec![0.2; 6],
            standardized: vec![0.3, -0.2, 0.1, -0.4, 0.5, -0.3],
            deleted: vec![0.3, -0.2, 0.1, -0.4, 0.5, -0.3],
            cooks_d: vec![0.0; 6],
            dffits: vec![0.0; 6],
            press_residuals: vec![0.0; 6],
            durbin_watson: 2.0,
        };
        let f = d.four_in_one(ResidualKind::Regular).unwrap();
        assert_eq!(f.charts().len(), 4);
        assert_eq!(f.normal_plot.len(), 6);
    }
}
