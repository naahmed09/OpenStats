//! Ordinary least squares by Householder QR with extended-precision
//! iterative refinement.

use super::diagnostics::{durbin_watson, Diagnostics, ResidualKind, UnusualFlag, UnusualObservation};
use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::model::{build_design, Data, DesignMatrix, DesignOptions, ModelSpec, TermColumns};
use openstats_math::dist::{ContinuousDistribution, FisherF, StudentT};
use openstats_math::linalg::{Cholesky, Matrix, Qr};
use std::collections::HashMap;

mod weighted;
pub use weighted::{fit_weighted, regress_weighted};

/// Options of a regression fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionOptions {
    /// Confidence level of the coefficient and prediction intervals.
    pub level: f64,
    /// A column whose part orthogonal to the preceding columns is below
    /// `alias_tolerance` × its norm is aliased (dropped, as Minitab does).
    pub alias_tolerance: f64,
}

impl Default for RegressionOptions {
    fn default() -> Self {
        RegressionOptions { level: 0.95, alias_tolerance: 1e-10 }
    }
}

/// One row of the coefficient table.
#[derive(Debug, Clone, PartialEq)]
pub struct Coefficient {
    /// Column name.
    pub name: String,
    /// Least-squares estimate.
    pub estimate: f64,
    /// Standard error.
    pub std_err: f64,
    /// t statistic.
    pub t: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Confidence interval.
    pub ci: Interval,
    /// Variance inflation factor (`None` for the constant or with fewer than two predictors).
    pub vif: Option<f64>,
}

/// One row of the analysis-of-variance table.
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaRow {
    /// `Regression`, a term label, `Error`, `Lack-of-Fit`, `Pure Error` or `Total`.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sequential sum of squares (terms only).
    pub seq_ss: Option<f64>,
    /// Adjusted sum of squares.
    pub adj_ss: f64,
    /// Adjusted mean square.
    pub adj_ms: Option<f64>,
    /// F statistic.
    pub f: Option<f64>,
    /// p-value of `f`.
    pub p_value: Option<f64>,
}

/// Lack-of-fit test from replicated design rows.
#[derive(Debug, Clone, PartialEq)]
pub struct LackOfFit {
    /// Lack-of-fit df (distinct design rows − parameters).
    pub df_lof: usize,
    /// Lack-of-fit sum of squares.
    pub ss_lof: f64,
    /// Pure-error df (observations − distinct design rows).
    pub df_pe: usize,
    /// Pure-error sum of squares.
    pub ss_pe: f64,
    /// F statistic.
    pub f: f64,
    /// p-value.
    pub p_value: f64,
}

/// Result of a regression.
#[derive(Debug, Clone)]
pub struct RegressionFit {
    /// Number of observations used.
    pub n: usize,
    /// Whether the model has a constant.
    pub intercept: bool,
    /// Coefficient table (aliased columns excluded).
    pub coefficients: Vec<Coefficient>,
    /// Names of columns dropped because they are linear combinations of earlier ones.
    pub aliased: Vec<String>,
    /// Analysis of variance.
    pub anova: Vec<AnovaRow>,
    /// Lack-of-fit test, when the design has replicates and the test has df.
    pub lack_of_fit: Option<LackOfFit>,
    /// Error sum of squares.
    pub sse: f64,
    /// Regression sum of squares (about the mean with an intercept).
    pub ssr: f64,
    /// Total sum of squares (about the mean with an intercept, else uncentered).
    pub sst: f64,
    /// Model degrees of freedom (coefficients less the constant).
    pub df_model: usize,
    /// Error degrees of freedom.
    pub df_error: usize,
    /// Mean square error.
    pub mse: f64,
    /// Residual standard error `√MSE`.
    pub s: f64,
    /// R².
    pub r2: f64,
    /// Adjusted R².
    pub r2_adj: f64,
    /// Predicted R² = 1 − PRESS/SST.
    pub r2_pred: f64,
    /// Prediction error sum of squares.
    pub press: f64,
    /// Overall F statistic (`None` when the model has no predictors).
    pub f: Option<f64>,
    /// p-value of the overall F.
    pub f_p: Option<f64>,
    /// Per-observation diagnostics.
    pub diagnostics: Diagnostics,
    /// Confidence level used.
    pub level: f64,
    /// Estimated covariance matrix of the coefficients (kept columns).
    pub cov: Matrix,
    /// Indices of the kept columns of the original design matrix.
    pub kept: Vec<usize>,
    pub(crate) xtx_inv: Matrix,
    pub(crate) spec: Option<ModelSpec>,
    pub(crate) rows_used: Vec<usize>,
}

/// A prediction at new data.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Worksheet row of the new data.
    pub row: usize,
    /// Predicted mean response (plus the row's offset).
    pub fit: f64,
    /// Standard error of the fit.
    pub se_fit: f64,
    /// Confidence interval for the mean response.
    pub ci: Interval,
    /// Prediction interval for a new observation.
    pub pi: Interval,
}

// ---------------------------------------------------------------------------
// Extended-precision helpers
// ---------------------------------------------------------------------------

/// `y − X·b` with compensated (error-free transformation) accumulation.
fn accurate_residual(x: &Matrix, y: &[f64], b: &[f64]) -> Vec<f64> {
    (0..x.rows())
        .map(|i| {
            let (mut hi, mut lo) = (y[i], 0.0);
            for (j, bj) in b.iter().enumerate() {
                let a = -x[(i, j)];
                let p = a * bj;
                let pe = a.mul_add(*bj, -p);
                let s = hi + p;
                let bb = s - hi;
                let se = (hi - (s - bb)) + (p - bb);
                hi = s;
                lo += se + pe;
            }
            hi + lo
        })
        .collect()
}

/// Columns kept by a left-to-right Gram–Schmidt (with re-orthogonalisation).
pub(crate) fn independent_columns(x: &Matrix, tol: f64) -> (Vec<usize>, Vec<usize>) {
    let (n, p) = x.shape();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let (mut kept, mut dropped) = (Vec::new(), Vec::new());
    for j in 0..p {
        let mut v = x.col(j);
        let n0 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if n0 == 0.0 {
            dropped.push(j);
            continue;
        }
        for _ in 0..2 {
            for q in &basis {
                let c: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(q).for_each(|(a, b)| *a -= c * b);
            }
        }
        let nr = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if nr <= tol * n0 || basis.len() >= n {
            dropped.push(j);
        } else {
            v.iter_mut().for_each(|a| *a /= nr);
            basis.push(v);
            kept.push(j);
        }
    }
    (kept, dropped)
}

/// `(XᵀX)⁻¹ = R⁻¹R⁻ᵀ`.
fn xtx_inverse(qr: &Qr, p: usize) -> Matrix {
    let r = qr.r();
    let mut rinv = Matrix::zeros(p, p);
    for j in 0..p {
        rinv[(j, j)] = 1.0 / r[(j, j)];
        for i in (0..j).rev() {
            let s: f64 = (i + 1..=j).map(|k| r[(i, k)] * rinv[(k, j)]).sum();
            rinv[(i, j)] = -s / r[(i, i)];
        }
    }
    let mut m = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..=i {
            let v: f64 = (i.max(j)..p).map(|k| rinv[(i, k)] * rinv[(j, k)]).sum();
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

/// SSE of regressing `y` on the columns of `x` listed in `cols`.
pub(crate) fn sse_of(x: &Matrix, y: &[f64], cols: &[usize]) -> Result<f64> {
    if cols.is_empty() {
        return Ok(y.iter().map(|v| v * v).sum());
    }
    let rows: Vec<usize> = (0..x.rows()).collect();
    Ok(Qr::new(&x.select(&rows, cols))?.residual_ss(y)?)
}

fn vifs(xk: &Matrix, has_const: bool) -> Vec<Option<f64>> {
    let p = xk.cols();
    let first = usize::from(has_const);
    let m = p - first;
    let mut out = vec![None; p];
    if m < 2 {
        return out;
    }
    let n = xk.rows();
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(m);
    for j in first..p {
        let mut c = xk.col(j);
        if has_const {
            let mean = c.iter().sum::<f64>() / n as f64;
            c.iter_mut().for_each(|v| *v -= mean);
        }
        let nrm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm == 0.0 {
            return out;
        }
        c.iter_mut().for_each(|v| *v /= nrm);
        cols.push(c);
    }
    let Ok(z) = Matrix::from_columns(&cols) else { return out };
    let Ok(ch) = Cholesky::new(&z.gram()) else { return out };
    let Ok(inv) = ch.inverse() else { return out };
    for k in 0..m {
        out[first + k] = Some(inv[(k, k)]);
    }
    out
}

// ---------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------

/// Fit a prepared design matrix (`dm.y` must be present).
pub fn fit_design(dm: &DesignMatrix, opts: &RegressionOptions) -> Result<RegressionFit> {
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
    }
    let y_raw = dm.y.as_ref().ok_or_else(|| StatsError::InvalidArgument("the design matrix has no response".into()))?;
    let n = dm.x.rows();
    if y_raw.len() != n || dm.offset.len() != n {
        return invalid("response, offset and design matrix disagree in length");
    }
    if dm.x.as_slice().iter().chain(y_raw).any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite);
    }
    let y: Vec<f64> = y_raw.iter().zip(&dm.offset).map(|(a, o)| a - o).collect();
    let (kept, dropped) = independent_columns(&dm.x, opts.alias_tolerance);
    let p = kept.len();
    if p == 0 {
        return invalid("the design matrix has no usable columns");
    }
    if n <= p {
        return Err(StatsError::InsufficientData {
            what: "regression (observations must exceed coefficients)",
            needed: p + 1,
            got: n,
        });
    }
    let all_rows: Vec<usize> = (0..n).collect();
    let xk = dm.x.select(&all_rows, &kept);
    let has_const = dm.intercept && kept.first() == Some(&0);

    let qr = Qr::new(&xk)?;
    let mut b = qr.solve_least_squares(&y)?;
    for _ in 0..4 {
        let d = qr.solve_least_squares(&accurate_residual(&xk, &y, &b))?;
        let step = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        b.iter_mut().zip(&d).for_each(|(a, c)| *a += c);
        if step <= 1e-17 * scale {
            break;
        }
    }
    let e = accurate_residual(&xk, &y, &b);
    let fitted: Vec<f64> = y.iter().zip(&e).zip(&dm.offset).map(|((yi, ei), o)| yi - ei + o).collect();
    let sse: f64 = e.iter().map(|v| v * v).sum();
    let df_error = n - p;
    let mse = sse / df_error as f64;
    let s = mse.sqrt();
    let xtx_inv = xtx_inverse(&qr, p);
    let cov = xtx_inv.scale(mse);

    let df_total = if has_const { n - 1 } else { n };
    let sst = if has_const {
        let m = y.iter().sum::<f64>() / n as f64;
        y.iter().map(|v| (v - m) * (v - m)).sum::<f64>()
    } else {
        y.iter().map(|v| v * v).sum()
    };
    let df_model = p - usize::from(has_const);
    let ssr = (sst - sse).max(0.0);
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
    let r2_adj = if sst > 0.0 && df_total > 0 { 1.0 - mse / (sst / df_total as f64) } else { f64::NAN };

    // Leverage from the thin Q.
    let q = qr.q();
    let h: Vec<f64> = (0..n).map(|i| (0..p).map(|j| q[(i, j)] * q[(i, j)]).sum()).collect();
    let pf = p as f64;
    let mut standardized = Vec::with_capacity(n);
    let mut deleted = Vec::with_capacity(n);
    let mut cooks = Vec::with_capacity(n);
    let mut dffits = Vec::with_capacity(n);
    let mut press_res = Vec::with_capacity(n);
    for i in 0..n {
        let om = 1.0 - h[i];
        if om < 1e-12 {
            standardized.push(f64::NAN);
            deleted.push(f64::NAN);
            cooks.push(f64::NAN);
            dffits.push(f64::NAN);
            press_res.push(f64::NAN);
            continue;
        }
        standardized.push(e[i] / (s * om.sqrt()));
        let s2_i = if df_error > 1 { ((sse - e[i] * e[i] / om) / (df_error - 1) as f64).max(0.0) } else { f64::NAN };
        let t_i = e[i] / (s2_i.sqrt() * om.sqrt());
        deleted.push(t_i);
        cooks.push(e[i] * e[i] * h[i] / (pf * mse * om * om));
        dffits.push(t_i * (h[i] / om).sqrt());
        press_res.push(e[i] / om);
    }
    let press: f64 = press_res.iter().map(|v| v * v).sum();
    let r2_pred = if sst > 0.0 { 1.0 - press / sst } else { f64::NAN };
    let dw = durbin_watson(&e);

    // Coefficient table.
    let tdist = StudentT::new(df_error as f64)?;
    let tq = tdist.isf((1.0 - opts.level) / 2.0);
    let vif = vifs(&xk, has_const);
    let coefficients: Vec<Coefficient> = (0..p)
        .map(|j| {
            let se = cov[(j, j)].sqrt();
            let t = b[j] / se;
            Coefficient {
                name: dm.names[kept[j]].clone(),
                estimate: b[j],
                std_err: se,
                t,
                p_value: (2.0 * tdist.sf(t.abs())).min(1.0),
                ci: Interval { lower: b[j] - tq * se, upper: b[j] + tq * se, level: opts.level },
                vif: vif[j],
            }
        })
        .collect();
    let aliased: Vec<String> = dropped.iter().map(|&j| dm.names[j].clone()).collect();

    let (f, f_p) = if df_model > 0 && mse > 0.0 {
        let f = (ssr / df_model as f64) / mse;
        (Some(f), Some(FisherF::new(df_model as f64, df_error as f64)?.sf(f)))
    } else {
        (None, None)
    };

    // ANOVA.
    let mut qty = y.clone();
    qr.apply_qt(&mut qty)?;
    let mut anova = Vec::new();
    if df_model > 0 {
        anova.push(AnovaRow {
            source: "Regression".into(),
            df: df_model,
            seq_ss: None,
            adj_ss: ssr,
            adj_ms: Some(ssr / df_model as f64),
            f,
            p_value: f_p,
        });
        for t in &dm.terms {
            let pos: Vec<usize> = (0..p).filter(|&k| kept[k] >= t.start && kept[k] < t.start + t.len).collect();
            if pos.is_empty() {
                continue;
            }
            let seq: f64 = pos.iter().map(|&k| qty[k] * qty[k]).sum();
            let rest: Vec<usize> = (0..p).filter(|k| !pos.contains(k)).collect();
            let adj = (sse_of(&xk, &y, &rest)? - sse).max(0.0);
            let ms = adj / pos.len() as f64;
            let (ft, pt) = if mse > 0.0 {
                let ft = ms / mse;
                (Some(ft), Some(FisherF::new(pos.len() as f64, df_error as f64)?.sf(ft)))
            } else {
                (None, None)
            };
            anova.push(AnovaRow {
                source: t.label.clone(),
                df: pos.len(),
                seq_ss: Some(seq),
                adj_ss: adj,
                adj_ms: Some(ms),
                f: ft,
                p_value: pt,
            });
        }
    }
    anova.push(AnovaRow {
        source: "Error".into(),
        df: df_error,
        seq_ss: None,
        adj_ss: sse,
        adj_ms: Some(mse),
        f: None,
        p_value: None,
    });
    // Lack of fit from replicated design rows.
    let mut groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups.entry(xk.row(i).iter().map(|v| v.to_bits()).collect()).or_default().push(i);
    }
    let m_groups = groups.len();
    let mut lack_of_fit = None;
    if m_groups < n && m_groups > p {
        let ss_pe: f64 = groups
            .values()
            .map(|idx| {
                let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64;
                idx.iter().map(|&i| (y[i] - mean) * (y[i] - mean)).sum::<f64>()
            })
            .sum();
        let (df_pe, df_lof) = (n - m_groups, m_groups - p);
        let ss_lof = (sse - ss_pe).max(0.0);
        if ss_pe > 0.0 {
            let f = (ss_lof / df_lof as f64) / (ss_pe / df_pe as f64);
            let pv = FisherF::new(df_lof as f64, df_pe as f64)?.sf(f);
            anova.push(AnovaRow {
                source: "Lack-of-Fit".into(),
                df: df_lof,
                seq_ss: None,
                adj_ss: ss_lof,
                adj_ms: Some(ss_lof / df_lof as f64),
                f: Some(f),
                p_value: Some(pv),
            });
            anova.push(AnovaRow {
                source: "Pure Error".into(),
                df: df_pe,
                seq_ss: None,
                adj_ss: ss_pe,
                adj_ms: Some(ss_pe / df_pe as f64),
                f: None,
                p_value: None,
            });
            lack_of_fit = Some(LackOfFit { df_lof, ss_lof, df_pe, ss_pe, f, p_value: pv });
        }
    }
    anova.push(AnovaRow {
        source: "Total".into(),
        df: df_total,
        seq_ss: None,
        adj_ss: sst,
        adj_ms: None,
        f: None,
        p_value: None,
    });

    Ok(RegressionFit {
        n,
        intercept: has_const,
        coefficients,
        aliased,
        anova,
        lack_of_fit,
        sse,
        ssr,
        sst,
        df_model,
        df_error,
        mse,
        s,
        r2,
        r2_adj,
        r2_pred,
        press,
        f,
        f_p,
        diagnostics: Diagnostics {
            fitted,
            residuals: e,
            leverage: h,
            standardized,
            deleted,
            cooks_d: cooks,
            dffits,
            press_residuals: press_res,
            durbin_watson: dw,
        },
        level: opts.level,
        cov,
        kept,
        xtx_inv,
        spec: None,
        rows_used: dm.rows_used.clone(),
    })
}

/// Parse `formula`, build the design from `data` and fit it.
pub fn regress(formula: &str, data: &Data, design: &DesignOptions, opts: &RegressionOptions) -> Result<RegressionFit> {
    let (spec, dm) = build_design(formula, data, design)?;
    let mut fit = fit_design(&dm, opts)?;
    fit.spec = Some(spec);
    Ok(fit)
}

/// Fit `y` on the columns of `x` (name per column). With `intercept` a
/// constant column is prepended; every column is its own term.
pub fn fit_xy(
    columns: &[(&str, &[f64])],
    y: &[f64],
    intercept: bool,
    opts: &RegressionOptions,
) -> Result<RegressionFit> {
    if columns.is_empty() && !intercept {
        return invalid("no predictors and no intercept");
    }
    let n = y.len();
    if columns.iter().any(|(_, c)| c.len() != n) {
        return invalid("predictor columns and the response differ in length");
    }
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut names = Vec::new();
    if intercept {
        cols.push(vec![1.0; n]);
        names.push("Constant".to_string());
    }
    let mut terms = Vec::new();
    for (name, c) in columns {
        terms.push(TermColumns { label: (*name).to_string(), order: 1, start: cols.len(), len: 1 });
        cols.push(c.to_vec());
        names.push((*name).to_string());
    }
    let dm = DesignMatrix {
        x: Matrix::from_columns(&cols)?,
        names,
        terms,
        intercept,
        y: Some(y.to_vec()),
        offset: vec![0.0; n],
        rows_used: (0..n).collect(),
        n_dropped: 0,
    };
    fit_design(&dm, opts)
}

impl RegressionFit {
    /// Number of coefficients (kept columns).
    pub fn n_coef(&self) -> usize {
        self.coefficients.len()
    }

    /// Coefficient estimates.
    pub fn estimates(&self) -> Vec<f64> {
        self.coefficients.iter().map(|c| c.estimate).collect()
    }

    /// Indices (among the rows used) of the worksheet rows.
    pub fn rows_used(&self) -> &[usize] {
        &self.rows_used
    }

    /// AIC `n·ln(SSE/n) + 2k`, k = number of coefficients.
    pub fn aic(&self) -> f64 {
        self.n as f64 * (self.sse / self.n as f64).ln() + 2.0 * self.n_coef() as f64
    }

    /// AICc `n·ln(SSE/n) + n(n+k)/(n−k−2)` (NaN unless n > k + 2).
    pub fn aicc(&self) -> f64 {
        let (n, k) = (self.n as f64, self.n_coef() as f64);
        if n - k - 2.0 <= 0.0 {
            return f64::NAN;
        }
        n * (self.sse / n).ln() + n * (n + k) / (n - k - 2.0)
    }

    /// BIC `n·ln(SSE/n) + k·ln n`.
    pub fn bic(&self) -> f64 {
        let n = self.n as f64;
        n * (self.sse / n).ln() + self.n_coef() as f64 * n.ln()
    }

    /// Observations Minitab would list as unusual: large standardized
    /// residual (|·| > 2), high leverage (> 3p/n, or > 0.99) and influential
    /// (Cook's distance above the median of F(p, n−p)).
    pub fn unusual_observations(&self) -> Result<Vec<UnusualObservation>> {
        let (n, p) = (self.n as f64, self.n_coef() as f64);
        let cook_limit = FisherF::new(p, n - p)?.median();
        let d = &self.diagnostics;
        let mut out = Vec::new();
        for i in 0..self.n {
            let mut flags = Vec::new();
            if d.standardized[i].abs() > 2.0 {
                flags.push(UnusualFlag::LargeResidual);
            }
            if d.leverage[i] > (3.0 * p / n).min(0.99) {
                flags.push(UnusualFlag::HighLeverage);
            }
            if d.cooks_d[i] > cook_limit {
                flags.push(UnusualFlag::Influential);
            }
            if !flags.is_empty() {
                out.push(UnusualObservation { index: i, flags });
            }
        }
        Ok(out)
    }

    /// Residuals of the given kind.
    pub fn residuals(&self, kind: ResidualKind) -> &[f64] {
        self.diagnostics.residuals_of(kind)
    }

    /// Predict at new data (a model fitted with [`regress`] is required).
    /// Rows with missing predictors are skipped.
    pub fn predict(&self, data: &Data, level: f64) -> Result<Vec<Prediction>> {
        if !(level > 0.0 && level < 1.0) {
            return invalid(format!("confidence level {level} must be in (0, 1)"));
        }
        let spec = self.spec.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("prediction needs a model fitted from a formula (use `regress`)".into())
        })?;
        let dm = spec.encode(data, false)?;
        let rows: Vec<usize> = (0..dm.x.rows()).collect();
        let xn = dm.x.select(&rows, &self.kept);
        let b = self.estimates();
        let tq = StudentT::new(self.df_error as f64)?.isf((1.0 - level) / 2.0);
        let mut out = Vec::with_capacity(xn.rows());
        for i in 0..xn.rows() {
            let x0 = xn.row(i);
            let fit: f64 = x0.iter().zip(&b).map(|(a, c)| a * c).sum::<f64>() + dm.offset[i];
            let v = self.xtx_inv.matvec(x0)?;
            let q: f64 = x0.iter().zip(&v).map(|(a, c)| a * c).sum();
            let se_fit = self.s * q.max(0.0).sqrt();
            let se_pred = self.s * (1.0 + q.max(0.0)).sqrt();
            out.push(Prediction {
                row: dm.rows_used[i],
                fit,
                se_fit,
                ci: Interval { lower: fit - tq * se_fit, upper: fit + tq * se_fit, level },
                pi: Interval { lower: fit - tq * se_pred, upper: fit + tq * se_pred, level },
            });
        }
        Ok(out)
    }
}
