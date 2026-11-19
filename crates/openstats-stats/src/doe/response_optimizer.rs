//! Worksheet-level response optimisation: fit one polynomial model per response from a design
//! stored as data, then hand the models and goals to [`super::optimize_responses`].
//!
//! A design is read from the columns of a worksheet (one column of natural factor values per
//! factor). Each factor is coded by its observed range (`−1` at the smallest value, `+1` at the
//! largest), which is an affine change of scale: models, predictions and optima in natural units
//! are identical to those fitted with the design's own cube levels.

use super::design::{Design, DesignKind, FactorSpec, PointType};
use super::model::PolynomialModel;
use super::optimize::Goal;
use super::rsm::{analyze_surface, SurfaceModel, SurfaceOptions};
use crate::error::{invalid, Result};
use crate::report::{Cell, Column, ColumnKind, Section, Table};

/// Order of the polynomial fitted to every response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOrder {
    /// Main effects only.
    Linear,
    /// Main effects and two-factor interactions.
    Interactions,
    /// Full quadratic (needs a response-surface design: squares are aliased in two-level designs).
    Quadratic,
}

/// A response model with its fit statistics.
#[derive(Debug, Clone)]
pub struct FittedResponse {
    /// Response name.
    pub name: String,
    /// Polynomial in coded units.
    pub model: PolynomialModel,
    /// R².
    pub r2: f64,
    /// Adjusted R².
    pub r2_adj: f64,
    /// Root mean square error.
    pub s: f64,
    /// Terms dropped as aliased.
    pub dropped: Vec<String>,
}

/// Build a [`Design`] from natural-unit factor columns (all the same length, no missing values).
pub fn design_from_columns(names: &[String], columns: &[Vec<f64>]) -> Result<Design> {
    if names.is_empty() || names.len() != columns.len() {
        return invalid("a design needs at least one factor column");
    }
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) {
        return invalid("the factor columns have different lengths");
    }
    if columns.iter().flatten().any(|v| !v.is_finite()) {
        return invalid("the factor columns contain missing or infinite values");
    }
    let mut factors = Vec::new();
    for (name, c) in names.iter().zip(columns) {
        let (lo, hi) = c.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        if hi <= lo {
            return invalid(format!("factor '{name}' has only one value"));
        }
        factors.push(FactorSpec { name: name.clone(), low: lo, high: hi });
    }
    let coded: Vec<Vec<f64>> =
        (0..n).map(|i| (0..factors.len()).map(|j| factors[j].to_coded(columns[j][i])).collect()).collect();
    let point_type = coded
        .iter()
        .map(|r| {
            if r.iter().all(|c| (c.abs() - 1.0).abs() < 1e-9) {
                PointType::Factorial
            } else if r.iter().all(|c| c.abs() < 1e-9) {
                PointType::Center
            } else {
                PointType::Axial
            }
        })
        .collect();
    Ok(Design {
        kind: DesignKind::CentralComposite,
        factors,
        coded,
        point_type,
        block: vec![1; n],
        blocks: 1,
        std_order: (1..=n).collect(),
        defining: None,
        block_words: Vec::new(),
    })
}

/// Fit a model of `order` to each response.
pub fn fit_responses(
    design: &Design,
    responses: &[(String, Vec<f64>)],
    order: ModelOrder,
) -> Result<Vec<FittedResponse>> {
    if responses.is_empty() {
        return invalid("select at least one response");
    }
    let opts = SurfaceOptions {
        model: match order {
            ModelOrder::Linear => SurfaceModel::Linear,
            ModelOrder::Interactions => SurfaceModel::LinearInteractions,
            ModelOrder::Quadratic => SurfaceModel::FullQuadratic,
        },
        level: 0.95,
    };
    let mut out = Vec::new();
    for (name, y) in responses {
        if y.iter().any(|v| !v.is_finite()) {
            return invalid(format!("response '{name}' has missing values"));
        }
        let a = analyze_surface(design, y, &opts).map_err(|e| match e {
            crate::error::StatsError::InvalidArgument(m) => {
                crate::error::StatsError::InvalidArgument(format!("response '{name}': {m}"))
            }
            other => other,
        })?;
        out.push(FittedResponse {
            name: name.clone(),
            r2: a.fit.r2,
            r2_adj: a.fit.r2_adj,
            s: a.fit.s,
            dropped: a.dropped,
            model: a.model,
        });
    }
    Ok(out)
}

/// A table of the fitted models' R², adjusted R² and S.
pub fn model_summary_section(fits: &[FittedResponse]) -> Section {
    let mut s = Section::new("Response Models");
    let cols = [
        ("Response", ColumnKind::Text),
        ("Terms", ColumnKind::Integer),
        ("S", ColumnKind::Significant(6)),
        ("R-Sq", ColumnKind::Fixed(2)),
        ("R-Sq(adj)", ColumnKind::Fixed(2)),
    ];
    let mut t = Table::new("Model summary", cols.iter().map(|(h, k)| Column::new(*h, *k)).collect());
    for f in fits {
        t.row(vec![
            Cell::text(&f.name),
            Cell::Int(f.model.terms.len() as i64),
            Cell::Num(f.s),
            Cell::Num(100.0 * f.r2),
            Cell::Num(100.0 * f.r2_adj),
        ]);
    }
    s.table(t);
    for f in fits {
        if !f.dropped.is_empty() {
            s.warn(format!(
                "{}: terms aliased with earlier terms were not estimated: {}",
                f.name,
                f.dropped.join(", ")
            ));
        }
        if f.model.df_error == 0 {
            s.warn(format!("{}: no degrees of freedom for error; standard errors of fits are unavailable.", f.name));
        }
    }
    s
}

/// Parse a goal such as `max 70 90`, `min 5 20 2` or `target 40 50 60 1 1`.
///
/// * `max LOWER TARGET [SHAPE]` — larger is better.
/// * `min TARGET UPPER [SHAPE]` — smaller is better.
/// * `target LOWER TARGET UPPER [SHAPE_LOW [SHAPE_HIGH]]` — hit the target.
///
/// Shapes (Minitab's "weights") default to 1. Words may be abbreviated to their first letters
/// (`max`, `maximize`, `maximum`, `m`…) — `m` alone is ambiguous and rejected.
pub fn parse_goal(s: &str) -> Result<Goal> {
    let mut it = s.split_whitespace();
    let kind = match it.next() {
        Some(k) => k.to_ascii_lowercase(),
        None => return invalid("an empty goal; write e.g. 'max 70 90', 'min 5 20' or 'target 40 50 60'"),
    };
    let nums: Vec<f64> = it
        .map(|t| {
            t.parse::<f64>().map_err(|_| {
                crate::error::StatsError::InvalidArgument(format!("'{t}' is not a number in the goal '{s}'"))
            })
        })
        .collect::<Result<_>>()?;
    let want = |lo: usize, hi: usize, usage: &str| -> Result<()> {
        if nums.len() < lo || nums.len() > hi {
            invalid(format!("the goal '{s}' needs {usage}"))
        } else {
            Ok(())
        }
    };
    let is = |full: &str| kind.len() >= 2 && full.starts_with(&kind);
    if is("maximize") || is("maximum") {
        want(2, 3, "LOWER TARGET [SHAPE]")?;
        Ok(Goal::Maximize { lower: nums[0], target: nums[1], shape: nums.get(2).copied().unwrap_or(1.0) })
    } else if is("minimize") || is("minimum") {
        want(2, 3, "TARGET UPPER [SHAPE]")?;
        Ok(Goal::Minimize { target: nums[0], upper: nums[1], shape: nums.get(2).copied().unwrap_or(1.0) })
    } else if is("target") {
        want(3, 5, "LOWER TARGET UPPER [SHAPE_LOW [SHAPE_HIGH]]")?;
        let lo = nums.get(3).copied().unwrap_or(1.0);
        Ok(Goal::Target {
            lower: nums[0],
            target: nums[1],
            upper: nums[2],
            shape_low: lo,
            shape_high: nums.get(4).copied().unwrap_or(lo),
        })
    } else {
        invalid(format!("unknown goal type '{kind}'; use max, min or target"))
    }
}
