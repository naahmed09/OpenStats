//! Polynomial response models in coded factor units, and the column machinery
//! that builds them from a design.

use super::design::Design;
use crate::error::{invalid, Result, StatsError};
use crate::regression::{fit_xy, RegressionFit, RegressionOptions};
use openstats_math::linalg::{Matrix, Qr};

/// A monomial in the coded factors: `Π xᵢ^pᵢ`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyTerm {
    /// `(factor index, power)`, factors ascending, powers ≥ 1; empty = the constant.
    pub powers: Vec<(usize, u32)>,
}

impl PolyTerm {
    /// Value at the coded point `x`.
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.powers.iter().map(|&(i, p)| x[i].powi(p as i32)).product()
    }

    /// Interaction order (sum of the powers).
    pub fn order(&self) -> usize {
        self.powers.iter().map(|&(_, p)| p as usize).sum()
    }

    /// Minitab-style label: `A`, `A*B`, `A*A`.
    pub fn label(&self, names: &[String]) -> String {
        if self.powers.is_empty() {
            return "Constant".into();
        }
        let mut parts = Vec::new();
        for &(i, p) in &self.powers {
            for _ in 0..p {
                parts.push(names[i].clone());
            }
        }
        parts.join("*")
    }
}

/// A fitted polynomial in coded units with the covariance of its coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialModel {
    /// Factor names.
    pub factor_names: Vec<String>,
    /// Terms, the constant first.
    pub terms: Vec<PolyTerm>,
    /// Coefficients, one per term.
    pub coef: Vec<f64>,
    /// Covariance of the coefficients (`None` without error degrees of freedom).
    pub cov: Option<Matrix>,
    /// Mean square error (`None` without error degrees of freedom).
    pub mse: Option<f64>,
    /// Error degrees of freedom.
    pub df_error: usize,
}

impl PolynomialModel {
    /// Term vector at `x`.
    pub fn basis(&self, x: &[f64]) -> Vec<f64> {
        self.terms.iter().map(|t| t.eval(x)).collect()
    }

    /// Predicted response at the coded point `x`.
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.basis(x).iter().zip(&self.coef).map(|(b, c)| b * c).sum()
    }

    /// Standard error of the fitted mean at `x`.
    pub fn se_fit(&self, x: &[f64]) -> Option<f64> {
        let cov = self.cov.as_ref()?;
        let b = self.basis(x);
        let cb = cov.matvec(&b).ok()?;
        Some(b.iter().zip(&cb).map(|(a, c)| a * c).sum::<f64>().max(0.0).sqrt())
    }

    /// Number of factors.
    pub fn n_factors(&self) -> usize {
        self.factor_names.len()
    }
}

/// A named column of a model.
#[derive(Debug, Clone)]
pub(crate) struct Col {
    pub name: String,
    pub values: Vec<f64>,
    /// `Some` for polynomial terms (they enter predictions); `None` for blocks and curvature.
    pub term: Option<PolyTerm>,
    /// Group label used for ANOVA grouping ("Blocks", "Linear", …).
    pub group: &'static str,
}

pub(crate) fn term_values(design: &Design, t: &PolyTerm) -> Vec<f64> {
    design.coded.iter().map(|r| t.eval(r)).collect()
}

/// Effect-coded block columns (blocks 1..B−1; the last block is −1).
pub(crate) fn block_columns(design: &Design) -> Vec<Col> {
    (1..design.blocks)
        .map(|b| Col {
            name: format!("Block {b}"),
            values: design
                .block
                .iter()
                .map(|&x| {
                    if x == b {
                        1.0
                    } else if x == design.blocks {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            term: None,
            group: "Blocks",
        })
        .collect()
}

/// Result of fitting a set of columns.
pub(crate) struct ColFit {
    pub names: Vec<String>,
    pub groups: Vec<&'static str>,
    pub coef: Vec<f64>,
    pub fit: Option<RegressionFit>,
    pub sse: f64,
    pub df_error: usize,
    pub cols: Vec<Col>,
    /// Names of columns dropped as linear combinations of earlier ones.
    pub dropped: Vec<String>,
}

/// Greedy selection of linearly independent columns (Gram–Schmidt, relative tolerance 1e-10).
pub(crate) fn independent(cols: &[Col]) -> Vec<usize> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut kept = Vec::new();
    for (i, c) in cols.iter().enumerate() {
        let mut v = c.values.clone();
        let norm0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for b in &basis {
            let d: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            v.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm0 > 0.0 && norm > 1e-10 * norm0 {
            v.iter_mut().for_each(|x| *x /= norm);
            basis.push(v);
            kept.push(i);
        }
    }
    kept
}

/// Fit `y` on a constant plus `cols`; with no error degrees of freedom a plain
/// least-squares solution is returned without a `RegressionFit`.
pub(crate) fn fit_cols(cols: Vec<Col>, y: &[f64], level: f64) -> Result<ColFit> {
    let n = y.len();
    if y.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite);
    }
    if cols.iter().any(|c| c.values.len() != n) {
        return invalid("the response and the design have different numbers of runs");
    }
    let kept = independent(&cols);
    let p = kept.len() + 1;
    let use_cols: Vec<Col> = kept.iter().map(|&i| cols[i].clone()).collect();
    let dropped: Vec<String> = (0..cols.len()).filter(|i| !kept.contains(i)).map(|i| cols[i].name.clone()).collect();
    if n > p {
        let refs: Vec<(&str, &[f64])> = use_cols.iter().map(|c| (c.name.as_str(), c.values.as_slice())).collect();
        let fit = fit_xy(&refs, y, true, &RegressionOptions { level, alias_tolerance: 1e-10 })?;
        let kept_names: Vec<&str> = fit.coefficients.iter().map(|c| c.name.as_str()).collect();
        let final_cols: Vec<Col> = use_cols.iter().filter(|c| kept_names.contains(&c.name.as_str())).cloned().collect();
        return Ok(ColFit {
            names: fit.coefficients.iter().map(|c| c.name.clone()).collect(),
            groups: std::iter::once("Constant").chain(final_cols.iter().map(|c| c.group)).collect(),
            coef: fit.estimates(),
            sse: fit.sse,
            df_error: fit.df_error,
            cols: final_cols,
            dropped,
            fit: Some(fit),
        });
    }
    if n < p {
        return Err(StatsError::InsufficientData { what: "model with more terms than runs", needed: p, got: n });
    }
    // Saturated: solve the square system.
    let mut columns = vec![vec![1.0; n]];
    columns.extend(use_cols.iter().map(|c| c.values.clone()));
    let m = Matrix::from_columns(&columns)?;
    let coef = Qr::new(&m)?.solve_least_squares(y)?;
    Ok(ColFit {
        names: std::iter::once("Constant".to_string()).chain(use_cols.iter().map(|c| c.name.clone())).collect(),
        groups: std::iter::once("Constant").chain(use_cols.iter().map(|c| c.group)).collect(),
        coef,
        fit: None,
        sse: 0.0,
        df_error: 0,
        cols: use_cols,
        dropped,
    })
}

impl ColFit {
    /// The polynomial model made of the polynomial-term columns.
    pub(crate) fn model(&self, factor_names: &[String]) -> PolynomialModel {
        let mut terms = vec![PolyTerm { powers: vec![] }];
        let mut idx = vec![0usize];
        for (j, c) in self.cols.iter().enumerate() {
            if let Some(t) = &c.term {
                terms.push(t.clone());
                idx.push(j + 1);
            }
        }
        let coef: Vec<f64> = idx.iter().map(|&i| self.coef[i]).collect();
        let cov = self.fit.as_ref().and_then(|f| {
            let rows: Vec<Vec<f64>> = idx.iter().map(|&i| idx.iter().map(|&j| f.cov[(i, j)]).collect()).collect();
            Matrix::from_rows(&rows).ok()
        });
        PolynomialModel {
            factor_names: factor_names.to_vec(),
            terms,
            coef,
            mse: self.fit.as_ref().map(|f| f.mse),
            df_error: self.df_error,
            cov,
        }
    }
}
