//! Shared types of the Gage R&R analyses.

use crate::error::{invalid, Result};
use crate::report::ChartSpec;

/// A tolerance for the %Tolerance column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// Total tolerance `USL − LSL`.
    Width(f64),
    /// Specification limits.
    Limits {
        /// Lower specification limit.
        lsl: f64,
        /// Upper specification limit.
        usl: f64,
    },
}

impl Tolerance {
    /// The tolerance width; an error unless it is finite and positive.
    pub fn width(&self) -> Result<f64> {
        let w = match *self {
            Tolerance::Width(w) => w,
            Tolerance::Limits { lsl, usl } => usl - lsl,
        };
        if w.is_finite() && w > 0.0 {
            Ok(w)
        } else {
            invalid(format!("the tolerance {w} must be positive and finite"))
        }
    }
}

/// How the Part × Operator interaction is treated by the ANOVA method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interaction {
    /// Remove the interaction from the model when its p-value is at least
    /// [`GageOptions::alpha_interaction`] (Minitab's default behaviour).
    Auto,
    /// Always keep the interaction in the model.
    Keep,
    /// Always remove the interaction (pool it with repeatability).
    Drop,
}

/// Options of a Gage R&R analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GageOptions {
    /// Number of standard deviations in the study variation (6 by default; 5.15 is also used).
    pub study_sigma: f64,
    /// Tolerance for the %Tolerance column.
    pub tolerance: Option<Tolerance>,
    /// Interaction p-value at or above which it is dropped (0.25).
    pub alpha_interaction: f64,
    /// Interaction handling (ANOVA method).
    pub interaction: Interaction,
    /// Treat the operators as a fixed factor (the operator component is `(o−1)/o` times the
    /// mean-square estimate of the fixed-effect quadratic form; crossed ANOVA method only).
    pub operator_fixed: bool,
}

impl Default for GageOptions {
    fn default() -> Self {
        GageOptions {
            study_sigma: 6.0,
            tolerance: None,
            alpha_interaction: 0.25,
            interaction: Interaction::Auto,
            operator_fixed: false,
        }
    }
}

impl GageOptions {
    pub(crate) fn validate(&self) -> Result<Option<f64>> {
        if !(self.study_sigma.is_finite() && self.study_sigma > 0.0) {
            return invalid(format!("the study-variation multiplier {} must be positive", self.study_sigma));
        }
        if !(self.alpha_interaction > 0.0 && self.alpha_interaction < 1.0) {
            return invalid(format!("the interaction level {} must be in (0, 1)", self.alpha_interaction));
        }
        self.tolerance.map(|t| t.width()).transpose()
    }
}

/// The estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GageMethod {
    /// Crossed design, analysis of variance.
    CrossedAnova,
    /// Crossed design, X̄ and R (range) method.
    CrossedXbarR,
    /// Nested design (parts nested within operators), analysis of variance.
    NestedAnova,
}

/// A row of the variance-components table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Repeatability plus reproducibility.
    TotalGageRr,
    /// Equipment (within-cell) variation.
    Repeatability,
    /// Operator-related variation (operator plus operator × part).
    Reproducibility,
    /// Operator main effect.
    Operator,
    /// Operator × part interaction.
    OperatorByPart,
    /// Variation between parts.
    PartToPart,
    /// Total variation.
    TotalVariation,
}

impl Source {
    /// Minitab's row label.
    pub fn label(self) -> &'static str {
        match self {
            Source::TotalGageRr => "Total Gage R&R",
            Source::Repeatability => "Repeatability",
            Source::Reproducibility => "Reproducibility",
            Source::Operator => "Operator",
            Source::OperatorByPart => "Operator*Part",
            Source::PartToPart => "Part-To-Part",
            Source::TotalVariation => "Total Variation",
        }
    }
}

/// One row of the analysis-of-variance table of a Gage R&R study.
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaRow {
    /// Source label.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sum of squares.
    pub ss: f64,
    /// Mean square (`None` for the total).
    pub ms: Option<f64>,
    /// F statistic.
    pub f: Option<f64>,
    /// p-value.
    pub p: Option<f64>,
}

/// One row of the variance-components / study-variation tables.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceRow {
    /// Source.
    pub source: Source,
    /// Variance component.
    pub var: f64,
    /// Percent of the total variance.
    pub pct_contribution: f64,
    /// Standard deviation.
    pub sd: f64,
    /// Study variation `k·sd`.
    pub study_var: f64,
    /// Percent of the total study variation.
    pub pct_study_var: f64,
    /// Study variation as a percent of the tolerance.
    pub pct_tolerance: Option<f64>,
}

/// Range-based quantities of the X̄-R method.
#[derive(Debug, Clone, PartialEq)]
pub struct XbarRDetail {
    /// Average range of the part × operator cells.
    pub r_bar: f64,
    /// Range of the operator means.
    pub operator_range: f64,
    /// Range of the part means.
    pub part_range: f64,
    /// Repeatability `R̄/d2`.
    pub equipment_variation: f64,
    /// Reproducibility `√(max(0, (range/d2*)² − EV²/(parts·replicates)))`.
    pub appraiser_variation: f64,
    /// Part variation `range/d2*`.
    pub part_variation: f64,
}

/// Result of a Gage R&R analysis.
#[derive(Debug, Clone)]
pub struct GageResult {
    /// Method used.
    pub method: GageMethod,
    /// Number of parts (for a nested design: parts per operator).
    pub parts: usize,
    /// Number of operators.
    pub operators: usize,
    /// Replicates per cell.
    pub replicates: usize,
    /// ANOVA table of the model that was used for the components.
    pub anova: Vec<AnovaRow>,
    /// ANOVA table including the interaction, when the interaction was removed.
    pub anova_with_interaction: Option<Vec<AnovaRow>>,
    /// p-value of the Operator × Part interaction (crossed ANOVA).
    pub interaction_p: Option<f64>,
    /// Whether the interaction was removed from the model.
    pub interaction_dropped: bool,
    /// Variance components and study variation.
    pub components: Vec<VarianceRow>,
    /// Number of distinct categories (`None` when the gage variation is 0).
    pub ndc: Option<usize>,
    /// Study-variation multiplier.
    pub study_sigma: f64,
    /// Tolerance width, when supplied.
    pub tolerance: Option<f64>,
    /// Sources whose negative variance estimate was set to 0.
    pub negative_set_to_zero: Vec<Source>,
    /// Whether the operators were treated as a fixed factor.
    pub operator_fixed: bool,
    /// X̄-R detail (X̄-R method only).
    pub xbar_r: Option<XbarRDetail>,
    /// Six-panel chart data (five panels for a nested design).
    pub charts: Vec<ChartSpec>,
}

impl GageResult {
    /// The row of a source, if the table has one.
    pub fn component(&self, s: Source) -> Option<&VarianceRow> {
        self.components.iter().find(|r| r.source == s)
    }
}

/// Raw variance estimates entering the components table.
pub(crate) struct Estimates {
    pub repeat: f64,
    pub operator: f64,
    pub op_part: Option<f64>,
    pub part: f64,
}

/// Assemble the components table and the number of distinct categories.
pub(crate) fn components(
    est: &Estimates,
    detail: bool,
    opts: &GageOptions,
    tol: Option<f64>,
) -> (Vec<VarianceRow>, Option<usize>) {
    let reprod = est.operator + est.op_part.unwrap_or(0.0);
    let grr = est.repeat + reprod;
    let total = grr + est.part;
    let row = |source: Source, var: f64| {
        let sd = var.sqrt();
        let study = opts.study_sigma * sd;
        VarianceRow {
            source,
            var,
            pct_contribution: 100.0 * var / total,
            sd,
            study_var: study,
            pct_study_var: 100.0 * sd / total.sqrt(),
            pct_tolerance: tol.map(|t| 100.0 * study / t),
        }
    };
    let mut rows = vec![
        row(Source::TotalGageRr, grr),
        row(Source::Repeatability, est.repeat),
        row(Source::Reproducibility, reprod),
    ];
    if detail {
        rows.push(row(Source::Operator, est.operator));
        if let Some(v) = est.op_part {
            rows.push(row(Source::OperatorByPart, v));
        }
    }
    rows.push(row(Source::PartToPart, est.part));
    rows.push(row(Source::TotalVariation, total));
    let ndc = (grr > 0.0).then(|| ((1.41 * (est.part / grr).sqrt()).floor() as usize).max(1));
    (rows, ndc)
}

/// Distinct non-empty labels in order of first appearance, sorted like factor levels.
pub(crate) fn levels_of(v: &[String]) -> Vec<String> {
    let mut l: Vec<String> = Vec::new();
    for s in v {
        if !l.contains(s) {
            l.push(s.clone());
        }
    }
    crate::model::level_sort(&mut l);
    l
}
