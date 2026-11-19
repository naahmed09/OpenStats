//! Partial dependence and individual conditional expectation (ICE) for any
//! [`PredictiveModel`] (Friedman 2001, Goldstein et al. 2015).
//!
//! The partial dependence of a prediction on predictor *j* at value *v* is the average of the
//! model's prediction over the training observations with predictor *j* forced to *v*:
//! `PD(v) = (1/n) Σᵢ f(v, x_{i,−j})`. Each term of that sum, followed over the grid, is one ICE
//! curve. For a classifier the target is the probability of one chosen class.

use super::data::{Dataset, Feature};
use super::store::PredictiveModel;
use crate::error::{invalid, Result};
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};

/// Controls of [`partial_dependence`].
#[derive(Debug, Clone, PartialEq)]
pub struct PdOptions {
    /// Grid points for a numeric predictor (≥ 2).
    pub grid_points: usize,
    /// Quantiles of the predictor that bound the grid (default 5 % and 95 %; use (0, 1) for the
    /// full range).
    pub range: (f64, f64),
    /// Number of ICE curves kept (evenly spaced observations; 0 = none).
    pub ice_rows: usize,
    /// Class whose probability is the target (classification): index into the levels, `None` for
    /// the last level.
    pub class: Option<usize>,
    /// Observations used (evenly spaced subset; 0 = all).
    pub max_rows: usize,
}

impl Default for PdOptions {
    fn default() -> Self {
        PdOptions { grid_points: 20, range: (0.05, 0.95), ice_rows: 30, class: None, max_rows: 0 }
    }
}

/// One-predictor partial dependence.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialDependence {
    /// Predictor name.
    pub feature: String,
    /// Whether the predictor is categorical.
    pub categorical: bool,
    /// Grid: the values of a numeric predictor, or the level indices of a categorical one.
    pub grid: Vec<f64>,
    /// Grid labels (formatted numbers, or the level names).
    pub labels: Vec<String>,
    /// What is averaged: `Predicted value` or `P(class)`.
    pub target: String,
    /// Partial dependence at every grid point (mean over observations with a prediction).
    pub average: Vec<f64>,
    /// ICE curves: `(observation index, prediction at every grid point)`.
    pub ice: Vec<(usize, Vec<f64>)>,
    /// Observations averaged over.
    pub n: usize,
}

/// Two-predictor partial dependence.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialDependence2D {
    /// Predictor names `(x, y)`.
    pub features: (String, String),
    /// x grid values.
    pub x: Vec<f64>,
    /// y grid values.
    pub y: Vec<f64>,
    /// `z[j][i]` is the partial dependence at `(x[i], y[j])`.
    pub z: Vec<Vec<f64>>,
    /// What is averaged.
    pub target: String,
    /// Observations averaged over.
    pub n: usize,
}

fn quantile7(sorted: &[f64], q: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

fn numeric_grid(v: &[f64], o: &PdOptions) -> Result<Vec<f64>> {
    if o.grid_points < 2 {
        return invalid("a numeric grid needs at least 2 points");
    }
    if !(o.range.0 >= 0.0 && o.range.1 <= 1.0 && o.range.0 < o.range.1) {
        return invalid("the grid quantiles must satisfy 0 <= low < high <= 1");
    }
    let mut s: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if s.len() < 2 {
        return invalid("the predictor has fewer than two values");
    }
    s.sort_by(f64::total_cmp);
    let (lo, hi) = (quantile7(&s, o.range.0), quantile7(&s, o.range.1));
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return invalid("the predictor is (nearly) constant over the requested range");
    }
    let k = o.grid_points;
    Ok((0..k).map(|i| lo + (hi - lo) * i as f64 / (k - 1) as f64).collect())
}

fn feature_grid(ds: &Dataset, f: usize, o: &PdOptions) -> Result<(Vec<f64>, Vec<String>, bool)> {
    match ds.features.get(f) {
        None => invalid("no such predictor"),
        Some(Feature::Numeric(v)) => {
            let g = numeric_grid(v, o)?;
            let labels = g.iter().map(|x| format!("{x}")).collect();
            Ok((g, labels, false))
        }
        Some(Feature::Categorical { levels, .. }) => {
            Ok(((0..levels.len()).map(|l| l as f64).collect(), levels.clone(), true))
        }
    }
}

fn subset(ds: &Dataset, max_rows: usize) -> (Dataset, Vec<usize>) {
    let n = ds.n_rows();
    let rows: Vec<usize> = if max_rows == 0 || max_rows >= n {
        (0..n).collect()
    } else {
        (0..max_rows).map(|k| k * n / max_rows).collect()
    };
    (ds.select_rows(&rows), rows)
}

fn target_name(model: &PredictiveModel, class: usize) -> String {
    if model.is_classification() {
        format!("P({})", model.levels()[class])
    } else {
        "Predicted value".to_string()
    }
}

fn class_index(model: &PredictiveModel, o: &PdOptions) -> Result<usize> {
    let k = model.levels().len();
    match o.class {
        _ if k == 0 => Ok(0),
        None => Ok(k - 1),
        Some(c) if c < k => Ok(c),
        Some(_) => invalid("the class index is out of range"),
    }
}

fn mean_finite(v: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0, 0usize);
    for x in v {
        if x.is_finite() {
            s += x;
            c += 1;
        }
    }
    if c == 0 {
        f64::NAN
    } else {
        s / c as f64
    }
}

/// Partial dependence (and ICE curves) of `model` on predictor `feature` over the observations `ds`.
pub fn partial_dependence(
    model: &PredictiveModel,
    ds: &Dataset,
    feature: usize,
    o: &PdOptions,
) -> Result<PartialDependence> {
    let (d, rows) = subset(ds, o.max_rows);
    if d.n_rows() == 0 {
        return invalid("there are no observations");
    }
    let class = class_index(model, o)?;
    let (grid, labels, categorical) = feature_grid(&d, feature, o)?;
    let mut by_grid: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    for &g in &grid {
        by_grid.push(model.score(&d.with_constant(feature, g)?, class)?);
    }
    let average = by_grid.iter().map(|p| mean_finite(p.iter().copied())).collect();
    let k = o.ice_rows.min(d.n_rows());
    let ice = (0..k)
        .map(|c| {
            let i = c * d.n_rows() / k.max(1);
            (rows[i], by_grid.iter().map(|p| p[i]).collect())
        })
        .collect();
    Ok(PartialDependence {
        feature: ds.names[feature].clone(),
        categorical,
        grid,
        labels,
        target: target_name(model, class),
        average,
        ice,
        n: d.n_rows(),
    })
}

/// Partial dependence on two predictors jointly (a surface). Categorical predictors use their
/// levels as grid values.
pub fn partial_dependence_2d(
    model: &PredictiveModel,
    ds: &Dataset,
    fx: usize,
    fy: usize,
    o: &PdOptions,
) -> Result<PartialDependence2D> {
    if fx == fy {
        return invalid("choose two different predictors");
    }
    let (d, _) = subset(ds, o.max_rows);
    if d.n_rows() == 0 {
        return invalid("there are no observations");
    }
    let class = class_index(model, o)?;
    let (gx, _, _) = feature_grid(&d, fx, o)?;
    let (gy, _, _) = feature_grid(&d, fy, o)?;
    let mut z = Vec::with_capacity(gy.len());
    for &b in &gy {
        let row_ds = d.with_constant(fy, b)?;
        let mut row = Vec::with_capacity(gx.len());
        for &a in &gx {
            row.push(mean_finite(model.score(&row_ds.with_constant(fx, a)?, class)?.into_iter()));
        }
        z.push(row);
    }
    Ok(PartialDependence2D {
        features: (ds.names[fx].clone(), ds.names[fy].clone()),
        x: gx,
        y: gy,
        z,
        target: target_name(model, class),
        n: d.n_rows(),
    })
}

impl Reportable for PartialDependence {
    fn report_section(&self) -> Section {
        let mut s = Section::new(format!("Partial Dependence: {}", self.feature));
        s.text(format!(
            "Average of the model's {} over {} observations with {} set to each grid value.",
            self.target, self.n, self.feature
        ));
        let mut t = Table::new(
            format!("Partial Dependence of {} on {}", self.target, self.feature),
            vec![
                Column::new(
                    self.feature.clone(),
                    if self.categorical { ColumnKind::Text } else { ColumnKind::Significant(6) },
                ),
                Column::new(self.target.clone(), ColumnKind::Significant(6)),
            ],
        );
        for (i, &a) in self.average.iter().enumerate() {
            let first = if self.categorical { Cell::text(self.labels[i].clone()) } else { Cell::Num(self.grid[i]) };
            t.row(vec![first, Cell::Num(a)]);
        }
        s.table(t);
        let mut series = vec![Series {
            name: "Partial dependence".into(),
            x: if self.categorical { (1..=self.grid.len()).map(|i| i as f64).collect() } else { self.grid.clone() },
            y: self.average.clone(),
        }];
        if !self.categorical {
            for (r, curve) in &self.ice {
                series.push(Series { name: format!("ICE {}", r + 1), x: self.grid.clone(), y: curve.clone() });
            }
        }
        // ICE curves first so the average is drawn on top.
        series.rotate_left(1);
        s.chart(if self.categorical {
            ChartSpec {
                kind: ChartKind::Bar,
                title: format!("Partial Dependence on {}", self.feature),
                x_label: self.feature.clone(),
                y_label: self.target.clone(),
                categories: self.labels.clone(),
                series: vec![Series { name: "Partial dependence".into(), x: Vec::new(), y: self.average.clone() }],
                reference_lines: Vec::new(),
            }
        } else {
            ChartSpec {
                kind: ChartKind::Line,
                title: format!("Partial Dependence on {}", self.feature),
                x_label: self.feature.clone(),
                y_label: self.target.clone(),
                categories: Vec::new(),
                series,
                reference_lines: Vec::new(),
            }
        });
        s
    }
}

impl Reportable for PartialDependence2D {
    fn report_section(&self) -> Section {
        let (fx, fy) = (&self.features.0, &self.features.1);
        let mut s = Section::new(format!("Partial Dependence: {fy} and {fx}"));
        s.text(format!("Average of the model's {} over {} observations.", self.target, self.n));
        let mut cols = vec![Column::new(format!("{fy} \\ {fx}"), ColumnKind::Significant(6))];
        cols.extend(self.x.iter().map(|x| Column::new(format!("{x:.4}"), ColumnKind::Significant(6))));
        let mut t = Table::new("Partial Dependence Surface", cols);
        for (j, row) in self.z.iter().enumerate() {
            let mut cells = vec![Cell::Num(self.y[j])];
            cells.extend(row.iter().map(|&v| Cell::Num(v)));
            t.row(cells);
        }
        s.table(t);
        let (zmin, zmax) = self
            .z
            .iter()
            .flatten()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        if self.x.len() >= 2 && self.y.len() >= 2 && zmax > zmin {
            let mut series = Vec::new();
            for l in 1..=8 {
                let level = zmin + (zmax - zmin) * f64::from(l) / 9.0;
                let (mut x, mut y) = (Vec::new(), Vec::new());
                for (a, b) in crate::doe::plots::contour_segments(&self.x, &self.y, &self.z, level) {
                    x.extend([a[0], b[0], f64::NAN]);
                    y.extend([a[1], b[1], f64::NAN]);
                }
                if !x.is_empty() {
                    series.push(Series { name: format!("{level:.6}"), x, y });
                }
            }
            s.chart(ChartSpec {
                kind: ChartKind::Contour,
                title: format!("Partial Dependence Contour: {fy} vs {fx}"),
                x_label: fx.clone(),
                y_label: fy.clone(),
                categories: Vec::new(),
                series,
                reference_lines: Vec::new(),
            });
            s.chart(ChartSpec {
                kind: ChartKind::Surface,
                title: format!("Partial Dependence Surface: {fy} vs {fx}"),
                x_label: fx.clone(),
                y_label: self.target.clone(),
                categories: vec![fy.clone()],
                series: self
                    .y
                    .iter()
                    .zip(&self.z)
                    .map(|(&y, row)| Series { name: format!("{y}"), x: self.x.clone(), y: row.clone() })
                    .collect(),
                reference_lines: Vec::new(),
            });
        }
        s
    }
}
