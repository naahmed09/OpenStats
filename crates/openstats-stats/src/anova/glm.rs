//! General linear model analysis of variance for fixed factors and
//! covariates: Type I (sequential) and Type III (adjusted) sums of squares,
//! least-squares means and main-effects / interaction plot data.
//!
//! Factors are coded with sum-to-zero (deviation) contrasts by default, which
//! makes the adjusted sums of squares of the regression layer the Type III
//! sums of squares even in the presence of interactions and unequal cell
//! sizes.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::model::{AtomKind, Contrast, Data, DesignOptions, Formula};
use crate::regression::{regress, RegressionFit, RegressionOptions};
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};
use openstats_math::dist::{ContinuousDistribution, FisherF, StudentT};
use std::collections::BTreeMap;

const MAX_GRID: usize = 100_000;

/// Options of a general linear model fit.
#[derive(Debug, Clone)]
pub struct GlmOptions {
    /// Design coding (default: sum-to-zero for every factor).
    pub design: DesignOptions,
    /// Regression options (confidence level, alias tolerance).
    pub regression: RegressionOptions,
}

impl Default for GlmOptions {
    fn default() -> Self {
        GlmOptions {
            design: DesignOptions { default_contrast: Contrast::SumToZero, ..DesignOptions::default() },
            regression: RegressionOptions::default(),
        }
    }
}

/// One row of the sequential (Type I) analysis of variance.
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialRow {
    /// Term label.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sequential sum of squares.
    pub seq_ss: f64,
    /// Sequential mean square.
    pub seq_ms: f64,
    /// F statistic against the error mean square.
    pub f: Option<f64>,
    /// p-value.
    pub p_value: Option<f64>,
}

/// A least-squares mean.
#[derive(Debug, Clone, PartialEq)]
pub struct LsMean {
    /// Level of each factor of the term.
    pub levels: Vec<String>,
    /// Least-squares mean (`NaN` for a combination without observations).
    pub mean: f64,
    /// Standard error.
    pub se: f64,
    /// Confidence interval.
    pub ci: Interval,
}

/// Least-squares means of one factor term.
#[derive(Debug, Clone, PartialEq)]
pub struct LsMeansTable {
    /// Term label (`A`, `A*B`).
    pub term: String,
    /// Factors of the term.
    pub factors: Vec<String>,
    /// One row per level combination, the last factor varying fastest.
    pub rows: Vec<LsMean>,
}

/// Result of a general linear model analysis of variance.
#[derive(Debug, Clone)]
pub struct GlmAnova {
    /// The underlying regression fit (ANOVA with Seq SS and Adj SS, coefficients, diagnostics).
    pub fit: RegressionFit,
    /// The model formula.
    pub formula: String,
    /// Response column name.
    pub response: String,
    /// Sequential (Type I) tests.
    pub sequential: Vec<SequentialRow>,
    /// Least-squares means of every factor term (empty when the model has
    /// nested or numeric-as-factor terms, or the factor grid is too large).
    pub ls_means: Vec<LsMeansTable>,
    y: Vec<f64>,
    factor_cols: BTreeMap<String, Vec<String>>,
    level_order: BTreeMap<String, Vec<String>>,
}

fn grid_index(mut i: usize, sizes: &[usize]) -> Vec<usize> {
    let mut out = vec![0; sizes.len()];
    for k in (0..sizes.len()).rev() {
        out[k] = i % sizes[k];
        i /= sizes[k];
    }
    out
}

/// Fit the model `formula` (response on the left) and compute the analysis of variance.
pub fn glm(formula: &str, data: &Data, opts: &GlmOptions) -> Result<GlmAnova> {
    let parsed = Formula::parse(formula)?;
    let response = match &parsed.response {
        Some(crate::model::Expr::Var(n)) => n.clone(),
        _ => return invalid("the response must be a single worksheet column"),
    };
    let y_all = data
        .numeric(&response)
        .ok_or_else(|| StatsError::InvalidArgument(format!("response '{response}' is not a numeric column")))?;
    let fit = regress(formula, data, &opts.design, &opts.regression)?;
    let rows = fit.rows_used().to_vec();
    let y: Vec<f64> = rows.iter().map(|&r| y_all[r]).collect();

    // Atoms of the model.
    let mut factor_names: Vec<String> = Vec::new();
    let mut numeric_vars: Vec<String> = Vec::new();
    let mut simple = true;
    for t in &parsed.terms {
        if !t.nested_in.is_empty() {
            simple = false;
        }
        for a in t.factors.iter().map(|(a, _)| a).chain(t.nested_in.iter()) {
            match &a.kind {
                AtomKind::Var(n) if data.factor(n).is_some() => {
                    if !factor_names.contains(n) {
                        factor_names.push(n.clone());
                    }
                }
                AtomKind::Var(n) => {
                    if !numeric_vars.contains(n) {
                        numeric_vars.push(n.clone());
                    }
                }
                AtomKind::Expr(e) => {
                    for v in e.variables() {
                        if !numeric_vars.contains(&v) {
                            numeric_vars.push(v);
                        }
                    }
                }
                AtomKind::AsFactor(_) => simple = false,
            }
        }
    }
    let mut factor_cols = BTreeMap::new();
    let mut level_order = BTreeMap::new();
    for n in &factor_names {
        let col = data.factor(n).unwrap_or(&[]);
        let used: Vec<String> = rows.iter().map(|&r| col[r].clone()).collect();
        let mut lv: Vec<String> = Vec::new();
        for v in &used {
            if !lv.contains(v) {
                lv.push(v.clone());
            }
        }
        crate::model::level_sort(&mut lv);
        factor_cols.insert(n.clone(), used);
        level_order.insert(n.clone(), lv);
    }

    // Sequential tests.
    let mut sequential = Vec::new();
    for r in &fit.anova {
        if let Some(seq) = r.seq_ss {
            let ms = seq / r.df as f64;
            let (f, p) = if fit.mse > 0.0 {
                let f = ms / fit.mse;
                (Some(f), Some(FisherF::new(r.df as f64, fit.df_error as f64)?.sf(f)))
            } else {
                (None, None)
            };
            sequential.push(SequentialRow {
                source: r.source.clone(),
                df: r.df,
                seq_ss: seq,
                seq_ms: ms,
                f,
                p_value: p,
            });
        }
    }

    let mut out = GlmAnova {
        fit,
        formula: formula.to_string(),
        response,
        sequential,
        ls_means: Vec::new(),
        y,
        factor_cols,
        level_order,
    };
    if simple && !factor_names.is_empty() {
        out.ls_means = out.compute_ls_means(&parsed, data, &factor_names, &numeric_vars, &rows)?;
    }
    Ok(out)
}

impl GlmAnova {
    fn compute_ls_means(
        &self,
        parsed: &Formula,
        data: &Data,
        factor_names: &[String],
        numeric_vars: &[String],
        rows: &[usize],
    ) -> Result<Vec<LsMeansTable>> {
        let sizes: Vec<usize> = factor_names.iter().map(|n| self.level_order[n].len()).collect();
        let grid_n = sizes.iter().try_fold(1usize, |a, &s| a.checked_mul(s)).unwrap_or(usize::MAX);
        if grid_n > MAX_GRID || grid_n == 0 {
            return Ok(Vec::new());
        }
        let spec = match &self.fit.spec {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        // Full factor grid with covariates at their means.
        let combos: Vec<Vec<usize>> = (0..grid_n).map(|i| grid_index(i, &sizes)).collect();
        let mut grid = Data::new();
        for (k, n) in factor_names.iter().enumerate() {
            let col: Vec<&str> = combos.iter().map(|c| self.level_order[n][c[k]].as_str()).collect();
            grid = grid.with_factor(n, &col)?;
        }
        for n in numeric_vars {
            if let Some(col) = data.numeric(n) {
                let used: Vec<f64> = rows.iter().map(|&r| col[r]).collect();
                let m = used.iter().sum::<f64>() / used.len().max(1) as f64;
                grid = grid.with_numeric(n, vec![m; grid_n])?;
            }
        }
        let dm = spec.encode(&grid, false)?;
        let all: Vec<usize> = (0..dm.x.rows()).collect();
        let xk = dm.x.select(&all, &self.fit.kept);
        let b = self.fit.estimates();
        let tq = StudentT::new(self.fit.df_error as f64)?.isf((1.0 - self.fit.level) / 2.0);

        let mut tables = Vec::new();
        for t in &parsed.terms {
            let fs: Vec<String> = t
                .factors
                .iter()
                .filter_map(|(a, _)| match &a.kind {
                    AtomKind::Var(n) if factor_names.contains(n) => Some(n.clone()),
                    _ => None,
                })
                .collect();
            if fs.is_empty() || fs.len() != t.factors.len() {
                continue;
            }
            let idx: Vec<usize> = fs.iter().map(|f| factor_names.iter().position(|n| n == f).unwrap_or(0)).collect();
            let tsizes: Vec<usize> = idx.iter().map(|&i| sizes[i]).collect();
            let n_cells: usize = tsizes.iter().product();
            let mut table = LsMeansTable { term: t.label(), factors: fs.clone(), rows: Vec::with_capacity(n_cells) };
            for cell in 0..n_cells {
                let target = grid_index(cell, &tsizes);
                let mut c = vec![0.0; xk.cols()];
                let mut count = 0usize;
                for (g, combo) in combos.iter().enumerate() {
                    if idx.iter().zip(&target).all(|(&i, &t)| combo[i] == t) {
                        for (cv, xv) in c.iter_mut().zip(xk.row(g)) {
                            *cv += xv;
                        }
                        count += 1;
                    }
                }
                let levels: Vec<String> =
                    fs.iter().zip(&target).map(|(f, &l)| self.level_order[f][l].clone()).collect();
                let present =
                    (0..self.y.len()).any(|r| fs.iter().zip(&levels).all(|(f, l)| &self.factor_cols[f][r] == l));
                if !present || count == 0 {
                    table.rows.push(LsMean {
                        levels,
                        mean: f64::NAN,
                        se: f64::NAN,
                        ci: Interval { lower: f64::NAN, upper: f64::NAN, level: self.fit.level },
                    });
                    continue;
                }
                for cv in &mut c {
                    *cv /= count as f64;
                }
                let mean: f64 = c.iter().zip(&b).map(|(a, e)| a * e).sum();
                let v = self.fit.xtx_inv.matvec(&c)?;
                let q: f64 = c.iter().zip(&v).map(|(a, e)| a * e).sum();
                let se = self.fit.s * q.max(0.0).sqrt();
                table.rows.push(LsMean {
                    levels,
                    mean,
                    se,
                    ci: Interval { lower: mean - tq * se, upper: mean + tq * se, level: self.fit.level },
                });
            }
            tables.push(table);
        }
        Ok(tables)
    }

    /// Data means of the response by level of `factor` (a main-effects plot).
    pub fn main_effects_chart(&self, factor: &str) -> Result<ChartSpec> {
        let col = self
            .factor_cols
            .get(factor)
            .ok_or_else(|| StatsError::InvalidArgument(format!("'{factor}' is not a factor of the model")))?;
        let levels = &self.level_order[factor];
        let means: Vec<f64> = levels.iter().map(|l| self.cell_mean(&[(col, l)])).collect();
        let grand = self.y.iter().sum::<f64>() / self.y.len().max(1) as f64;
        Ok(ChartSpec {
            kind: ChartKind::Line,
            title: format!("Main Effects Plot for {}", self.response),
            x_label: factor.to_string(),
            y_label: format!("Mean of {}", self.response),
            categories: levels.clone(),
            series: vec![Series { name: "Mean".into(), x: (1..=levels.len()).map(|i| i as f64).collect(), y: means }],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: grand, label: "Grand mean".into() }],
        })
    }

    /// Data means of the response for every combination of the levels of
    /// `a` (x axis) and `b` (one line per level).
    pub fn interaction_chart(&self, a: &str, b: &str) -> Result<ChartSpec> {
        let ca = self
            .factor_cols
            .get(a)
            .ok_or_else(|| StatsError::InvalidArgument(format!("'{a}' is not a factor of the model")))?;
        let cb = self
            .factor_cols
            .get(b)
            .ok_or_else(|| StatsError::InvalidArgument(format!("'{b}' is not a factor of the model")))?;
        if a == b {
            return invalid("an interaction plot needs two different factors");
        }
        let la = &self.level_order[a];
        let series = self.level_order[b]
            .iter()
            .map(|lb| Series {
                name: format!("{b} = {lb}"),
                x: (1..=la.len()).map(|i| i as f64).collect(),
                y: la.iter().map(|l| self.cell_mean(&[(ca, l), (cb, lb)])).collect(),
            })
            .collect();
        Ok(ChartSpec {
            kind: ChartKind::Line,
            title: format!("Interaction Plot for {}", self.response),
            x_label: a.to_string(),
            y_label: format!("Mean of {}", self.response),
            categories: la.clone(),
            series,
            reference_lines: Vec::new(),
        })
    }

    fn cell_mean(&self, conds: &[(&Vec<String>, &String)]) -> f64 {
        let (mut s, mut n) = (0.0, 0usize);
        for (r, y) in self.y.iter().enumerate() {
            if conds.iter().all(|(c, l)| &c[r] == *l) {
                s += y;
                n += 1;
            }
        }
        if n == 0 {
            f64::NAN
        } else {
            s / n as f64
        }
    }

    /// Names of the factors in the model (in order of appearance).
    pub fn factors(&self) -> Vec<String> {
        let mut f: Vec<String> = self.factor_cols.keys().cloned().collect();
        f.sort();
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_two_way_and_plots() {
        let a = ["x", "x", "x", "x", "y", "y", "y", "y"];
        let b = ["p", "p", "q", "q", "p", "p", "q", "q"];
        let y = vec![1.0, 2.0, 4.0, 5.0, 3.0, 4.0, 9.0, 10.0];
        let d = Data::new().with_factor("A", &a).unwrap().with_factor("B", &b).unwrap().with_numeric("y", y).unwrap();
        let g = glm("y ~ A*B", &d, &GlmOptions::default()).unwrap();
        // Balanced: Type I equals Type III.
        for (s, r) in g.sequential.iter().zip(g.fit.anova.iter().skip(1)) {
            assert!((s.seq_ss - r.adj_ss).abs() < 1e-10, "{} {}", s.seq_ss, r.adj_ss);
        }
        assert_eq!(g.ls_means.len(), 3);
        assert!((g.ls_means[0].rows[0].mean - 3.0).abs() < 1e-10); // A = x: (1.5 + 4.5)/2
        assert_eq!(g.main_effects_chart("A").unwrap().series[0].y.len(), 2);
        assert_eq!(g.interaction_chart("A", "B").unwrap().series.len(), 2);
        assert!(g.main_effects_chart("zz").is_err() && g.interaction_chart("A", "A").is_err());
        assert!(glm("log(y) ~ A", &d, &GlmOptions::default()).is_err());
    }
}
