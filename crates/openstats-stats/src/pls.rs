//! Partial least squares regression (NIPALS, PLS1 and PLS2) with
//! cross-validated model selection (PRESS, predicted R²), VIP scores and
//! coefficients in the original units.

use crate::error::{invalid, Result, StatsError};
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};
use openstats_math::linalg::{Matrix, Svd};

/// Cross-validation scheme for choosing the number of components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossValidation {
    /// No cross-validation (PRESS columns are omitted).
    None,
    /// Leave one observation out.
    LeaveOneOut,
    /// `k` groups of rows assigned cyclically (row `i` goes to group `i mod k`).
    KFold(usize),
}

/// Options of a PLS fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlsOptions {
    /// Maximum number of components to extract.
    pub components: usize,
    /// Standardize predictors and responses to unit variance (they are always centred).
    pub standardize: bool,
    /// Cross-validation scheme.
    pub cv: CrossValidation,
}

impl Default for PlsOptions {
    fn default() -> Self {
        PlsOptions { components: 2, standardize: false, cv: CrossValidation::LeaveOneOut }
    }
}

/// Model-selection statistics for a number of components.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentStats {
    /// Number of components.
    pub components: usize,
    /// Cumulative fraction of the predictor variance explained.
    pub x_variance: f64,
    /// Error sum of squares per response.
    pub sse: Vec<f64>,
    /// R² per response.
    pub r2: Vec<f64>,
    /// Cross-validated PRESS per response.
    pub press: Option<Vec<f64>>,
    /// Predicted R² per response.
    pub r2_pred: Option<Vec<f64>>,
}

/// A fitted PLS model.
#[derive(Debug, Clone)]
pub struct PlsFit {
    /// Observations used.
    pub n: usize,
    /// Number of components extracted (≤ requested).
    pub n_components: usize,
    /// Predictor names.
    pub x_names: Vec<String>,
    /// Response names.
    pub y_names: Vec<String>,
    /// Constants of the fitted equations, one per response (original units).
    pub intercepts: Vec<f64>,
    /// Coefficients in the original units (predictors × responses).
    pub coefficients: Matrix,
    /// X scores (n × A).
    pub x_scores: Matrix,
    /// X loadings (p × A).
    pub x_loadings: Matrix,
    /// X weights (p × A).
    pub x_weights: Matrix,
    /// Y loadings (q × A).
    pub y_loadings: Matrix,
    /// Y scores (n × A).
    pub y_scores: Matrix,
    /// Fitted values (n × q).
    pub fitted: Matrix,
    /// Residuals (n × q).
    pub residuals: Matrix,
    /// Variable importance in projection per predictor.
    pub vip: Vec<f64>,
    /// Statistics for 1 … A components.
    pub selection: Vec<ComponentStats>,
    /// Number of components with the smallest summed PRESS, when cross-validated.
    pub best_by_press: Option<usize>,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    x_mean: Vec<f64>,
    x_scale: Vec<f64>,
    y_mean: Vec<f64>,
    y_scale: Vec<f64>,
}

struct Core {
    w: Matrix,
    p: Matrix,
    q: Matrix,
    t: Matrix,
    u: Matrix,
    a: usize,
}

fn col_means(m: &Matrix) -> Vec<f64> {
    (0..m.cols()).map(|j| (0..m.rows()).map(|i| m[(i, j)]).sum::<f64>() / m.rows() as f64).collect()
}

fn col_sds(m: &Matrix, means: &[f64]) -> Vec<f64> {
    (0..m.cols())
        .map(|j| ((0..m.rows()).map(|i| (m[(i, j)] - means[j]).powi(2)).sum::<f64>() / (m.rows() as f64 - 1.0)).sqrt())
        .collect()
}

fn dotc(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// NIPALS on centred (and scaled) matrices.
fn nipals(x0: &Matrix, y0: &Matrix, ncomp: usize) -> Result<Core> {
    let (n, p) = x0.shape();
    let q = y0.cols();
    let mut x = x0.clone();
    let mut y = y0.clone();
    let mut w_cols: Vec<Vec<f64>> = Vec::new();
    let mut p_cols: Vec<Vec<f64>> = Vec::new();
    let mut q_cols: Vec<Vec<f64>> = Vec::new();
    let mut t_cols: Vec<Vec<f64>> = Vec::new();
    let mut u_cols: Vec<Vec<f64>> = Vec::new();
    for _ in 0..ncomp {
        // Start with the response column of largest variance.
        let start =
            (0..q).max_by(|&a, &b| dotc(&y.col(a), &y.col(a)).total_cmp(&dotc(&y.col(b), &y.col(b)))).unwrap_or(0);
        let mut u = y.col(start);
        let mut w = vec![0.0; p];
        let mut t = vec![0.0; n];
        let mut c = vec![0.0; q];
        for _ in 0..1000 {
            let uu = dotc(&u, &u);
            if uu <= 0.0 {
                break;
            }
            w = x.t_matvec(&u)?.iter().map(|v| v / uu).collect();
            let wn = dotc(&w, &w).sqrt();
            if wn <= 1e-14 {
                break;
            }
            w.iter_mut().for_each(|v| *v /= wn);
            t = x.matvec(&w)?;
            let tt = dotc(&t, &t);
            if tt <= 1e-300 {
                break;
            }
            c = y.t_matvec(&t)?.iter().map(|v| v / tt).collect();
            let cc = dotc(&c, &c);
            if cc <= 0.0 {
                break;
            }
            let u_new: Vec<f64> = y.matvec(&c)?.iter().map(|v| v / cc).collect();
            let diff: f64 = u_new.iter().zip(&u).map(|(a, b)| (a - b).powi(2)).sum();
            let done = q == 1 || diff <= 1e-24 * dotc(&u_new, &u_new).max(1e-300);
            u = u_new;
            if done {
                break;
            }
        }
        let tt = dotc(&t, &t);
        if tt <= 1e-300 || dotc(&w, &w) <= 0.0 {
            break; // nothing left to extract
        }
        let pl: Vec<f64> = x.t_matvec(&t)?.iter().map(|v| v / tt).collect();
        // Deflate X by t pᵀ and Y by t cᵀ (regression of Y on t).
        for i in 0..n {
            for j in 0..p {
                x[(i, j)] -= t[i] * pl[j];
            }
            for j in 0..q {
                y[(i, j)] -= t[i] * c[j];
            }
        }
        w_cols.push(w);
        p_cols.push(pl);
        q_cols.push(c);
        t_cols.push(t);
        u_cols.push(u);
    }
    let a = w_cols.len();
    if a == 0 {
        return invalid("no PLS component could be extracted (the predictors or responses are constant)");
    }
    Ok(Core {
        w: Matrix::from_columns(&w_cols)?,
        p: Matrix::from_columns(&p_cols)?,
        q: Matrix::from_columns(&q_cols)?,
        t: Matrix::from_columns(&t_cols)?,
        u: Matrix::from_columns(&u_cols)?,
        a,
    })
}

/// Coefficients (scaled space, p × q) of the first `a` components.
fn coef_scaled(core: &Core, a: usize) -> Result<Matrix> {
    let p = core.w.rows();
    let q = core.q.rows();
    let wa = core.w.select(&(0..p).collect::<Vec<_>>(), &(0..a).collect::<Vec<_>>());
    let pa = core.p.select(&(0..p).collect::<Vec<_>>(), &(0..a).collect::<Vec<_>>());
    let qa = core.q.select(&(0..q).collect::<Vec<_>>(), &(0..a).collect::<Vec<_>>());
    let ptw = pa.t_matmul(&wa)?; // a × a
    let inv = Svd::new(&ptw)?.pinv(None);
    Ok(wa.matmul(&inv)?.matmul(&qa.t())?)
}

fn scale_matrix(m: &Matrix, mean: &[f64], scale: &[f64]) -> Matrix {
    Matrix::from_fn(m.rows(), m.cols(), |i, j| (m[(i, j)] - mean[j]) / scale[j])
}

/// Fit a PLS regression of the response columns `y` on the predictor
/// columns `x` (rows with any missing value are dropped).
pub fn pls_regression(x_cols: &[(&str, &[f64])], y_cols: &[(&str, &[f64])], opts: &PlsOptions) -> Result<PlsFit> {
    if x_cols.is_empty() || y_cols.is_empty() {
        return invalid("PLS needs at least one predictor and one response");
    }
    let len = x_cols[0].1.len();
    if x_cols.iter().chain(y_cols).any(|(_, c)| c.len() != len) {
        return invalid("all columns must have the same length");
    }
    if opts.components == 0 {
        return invalid("the number of components must be at least 1");
    }
    let rows_used: Vec<usize> =
        (0..len).filter(|&i| x_cols.iter().chain(y_cols).all(|(_, c)| c[i].is_finite())).collect();
    let n = rows_used.len();
    let (p, q) = (x_cols.len(), y_cols.len());
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "partial least squares", needed: 3, got: n });
    }
    let xm = Matrix::from_fn(n, p, |i, j| x_cols[j].1[rows_used[i]]);
    let ym = Matrix::from_fn(n, q, |i, j| y_cols[j].1[rows_used[i]]);
    let x_mean = col_means(&xm);
    let y_mean = col_means(&ym);
    let (x_scale, y_scale) = if opts.standardize {
        let mut xs = col_sds(&xm, &x_mean);
        let mut ys = col_sds(&ym, &y_mean);
        if xs.iter().chain(&ys).any(|&s| s <= 0.0 || !s.is_finite()) {
            // Constant columns cannot be standardized: leave them unscaled.
            xs.iter_mut().for_each(|s| {
                if *s <= 0.0 || !s.is_finite() {
                    *s = 1.0
                }
            });
            ys.iter_mut().for_each(|s| {
                if *s <= 0.0 || !s.is_finite() {
                    *s = 1.0
                }
            });
        }
        (xs, ys)
    } else {
        (vec![1.0; p], vec![1.0; q])
    };
    let xs = scale_matrix(&xm, &x_mean, &x_scale);
    let ys = scale_matrix(&ym, &y_mean, &y_scale);
    let max_a = opts.components.min(p).min(n - 1);
    let core = nipals(&xs, &ys, max_a)?;
    let a = core.a;

    // Original-unit coefficients and constants for the full model.
    let b_scaled = coef_scaled(&core, a)?;
    let coefficients = Matrix::from_fn(p, q, |j, k| b_scaled[(j, k)] * y_scale[k] / x_scale[j]);
    let intercepts: Vec<f64> =
        (0..q).map(|k| y_mean[k] - (0..p).map(|j| x_mean[j] * coefficients[(j, k)]).sum::<f64>()).collect();
    let fitted =
        Matrix::from_fn(n, q, |i, k| intercepts[k] + (0..p).map(|j| xm[(i, j)] * coefficients[(j, k)]).sum::<f64>());
    let residuals = Matrix::from_fn(n, q, |i, k| ym[(i, k)] - fitted[(i, k)]);

    // Predictor variance explained per component (of the scaled X).
    let total_x: f64 = xs.as_slice().iter().map(|v| v * v).sum();
    let tss_y: Vec<f64> = (0..q).map(|k| (0..n).map(|i| (ym[(i, k)] - y_mean[k]).powi(2)).sum()).collect();

    // Cross-validated predictions for every number of components.
    let folds: Option<Vec<Vec<usize>>> = match opts.cv {
        CrossValidation::None => None,
        CrossValidation::LeaveOneOut => Some((0..n).map(|i| vec![i]).collect()),
        CrossValidation::KFold(k) => {
            if k < 2 || k > n {
                return invalid(format!("k-fold cross-validation needs 2 ≤ k ≤ {n}"));
            }
            Some((0..k).map(|g| (0..n).filter(|i| i % k == g).collect()).collect())
        }
    };
    let mut press: Option<Vec<Vec<f64>>> = folds.as_ref().map(|_| vec![vec![0.0; q]; a]);
    if let (Some(folds), Some(press)) = (&folds, press.as_mut()) {
        for hold in folds {
            let train: Vec<usize> = (0..n).filter(|i| !hold.contains(i)).collect();
            if train.len() < 3 {
                return invalid("too few observations remain for cross-validation");
            }
            let xt = Matrix::from_fn(train.len(), p, |i, j| xm[(train[i], j)]);
            let yt = Matrix::from_fn(train.len(), q, |i, k| ym[(train[i], k)]);
            let (txm, tym) = (col_means(&xt), col_means(&yt));
            let (txs, tys) = if opts.standardize {
                let mut a1 = col_sds(&xt, &txm);
                let mut b1 = col_sds(&yt, &tym);
                a1.iter_mut().for_each(|s| {
                    if *s <= 0.0 || !s.is_finite() {
                        *s = 1.0
                    }
                });
                b1.iter_mut().for_each(|s| {
                    if *s <= 0.0 || !s.is_finite() {
                        *s = 1.0
                    }
                });
                (a1, b1)
            } else {
                (vec![1.0; p], vec![1.0; q])
            };
            let sub = nipals(&scale_matrix(&xt, &txm, &txs), &scale_matrix(&yt, &tym, &tys), a.min(train.len() - 1))?;
            for comp in 1..=a {
                let use_c = comp.min(sub.a);
                let b = coef_scaled(&sub, use_c)?;
                for &i in hold {
                    for k in 0..q {
                        let pred =
                            tym[k] + tys[k] * (0..p).map(|j| (xm[(i, j)] - txm[j]) / txs[j] * b[(j, k)]).sum::<f64>();
                        press[comp - 1][k] += (ym[(i, k)] - pred).powi(2);
                    }
                }
            }
        }
    }
    let mut selection = Vec::with_capacity(a);
    for comp in 1..=a {
        let b = coef_scaled(&core, comp)?;
        let sse: Vec<f64> = (0..q)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let pred = y_mean[k] + y_scale[k] * (0..p).map(|j| xs[(i, j)] * b[(j, k)]).sum::<f64>();
                        (ym[(i, k)] - pred).powi(2)
                    })
                    .sum()
            })
            .collect();
        let r2: Vec<f64> = (0..q).map(|k| 1.0 - sse[k] / tss_y[k]).collect();
        let var: f64 = (0..comp)
            .map(|c| {
                let t = core.t.col(c);
                let pl = core.p.col(c);
                dotc(&t, &t) * dotc(&pl, &pl)
            })
            .sum();
        let (pr, r2p) = match &press {
            Some(pv) => (Some(pv[comp - 1].clone()), Some((0..q).map(|k| 1.0 - pv[comp - 1][k] / tss_y[k]).collect())),
            None => (None, None),
        };
        selection.push(ComponentStats {
            components: comp,
            x_variance: var / total_x,
            sse,
            r2,
            press: pr,
            r2_pred: r2p,
        });
    }
    let best_by_press = press.as_ref().map(|pv| {
        let mut best = 1;
        let mut best_v = f64::INFINITY;
        for (c, row) in pv.iter().enumerate() {
            let v: f64 = row.iter().zip(&tss_y).map(|(a, b)| a / b).sum();
            if v < best_v {
                best_v = v;
                best = c + 1;
            }
        }
        best
    });

    // VIP.
    let ss: Vec<f64> = (0..a)
        .map(|c| {
            let t = core.t.col(c);
            dotc(&t, &t) * (0..q).map(|k| core.q[(k, c)].powi(2)).sum::<f64>()
        })
        .collect();
    let sst: f64 = ss.iter().sum();
    let vip: Vec<f64> =
        (0..p).map(|j| (p as f64 * (0..a).map(|c| ss[c] * core.w[(j, c)].powi(2)).sum::<f64>() / sst).sqrt()).collect();

    Ok(PlsFit {
        n,
        n_components: a,
        x_names: x_cols.iter().map(|(n, _)| n.to_string()).collect(),
        y_names: y_cols.iter().map(|(n, _)| n.to_string()).collect(),
        intercepts,
        coefficients,
        x_scores: core.t,
        x_loadings: core.p,
        x_weights: core.w,
        y_loadings: core.q,
        y_scores: core.u,
        fitted,
        residuals,
        vip,
        selection,
        best_by_press,
        rows_used,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    })
}

impl PlsFit {
    /// Predict responses (n × q) for new predictor rows (columns in the order
    /// of `x_names`).
    pub fn predict(&self, x_new: &Matrix) -> Result<Matrix> {
        if x_new.cols() != self.x_names.len() {
            return invalid(format!("{} predictor columns expected, got {}", self.x_names.len(), x_new.cols()));
        }
        let q = self.y_names.len();
        let p = self.x_names.len();
        Ok(Matrix::from_fn(x_new.rows(), q, |i, k| {
            self.intercepts[k] + (0..p).map(|j| x_new[(i, j)] * self.coefficients[(j, k)]).sum::<f64>()
        }))
    }

    /// Mean and scale used for each predictor.
    pub fn x_centering(&self) -> (&[f64], &[f64]) {
        (&self.x_mean, &self.x_scale)
    }

    /// Mean and scale used for each response.
    pub fn y_centering(&self) -> (&[f64], &[f64]) {
        (&self.y_mean, &self.y_scale)
    }
}

impl Reportable for PlsFit {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Partial Least Squares Regression");
        sec.text(format!(
            "Components extracted: {}{}",
            self.n_components,
            self.best_by_press.map_or(String::new(), |b| format!("   Components with minimum PRESS: {b}"))
        ));
        for (k, name) in self.y_names.iter().enumerate() {
            let cv = self.selection.iter().any(|s| s.press.is_some());
            let mut cols = vec![
                Column::new("Components", ColumnKind::Integer),
                Column::new("X Variance", ColumnKind::Significant(6)),
                Column::new("Error", ColumnKind::Significant(6)),
                Column::new("R-Sq", ColumnKind::Significant(6)),
            ];
            if cv {
                cols.push(Column::new("PRESS", ColumnKind::Significant(6)));
                cols.push(Column::new("R-Sq (pred)", ColumnKind::Significant(6)));
            }
            let mut t = Table::new(format!("Model Selection and Validation for {name}"), cols);
            for s in &self.selection {
                let mut row = vec![
                    Cell::Int(i64::try_from(s.components).unwrap_or(i64::MAX)),
                    Cell::Num(s.x_variance),
                    Cell::Num(s.sse[k]),
                    Cell::Num(s.r2[k]),
                ];
                if cv {
                    row.push(Cell::opt(s.press.as_ref().map(|v| v[k])));
                    row.push(Cell::opt(s.r2_pred.as_ref().map(|v| v[k])));
                }
                t.row(row);
            }
            sec.table(t);
        }
        let mut cols = vec![Column::new("Predictor", ColumnKind::Text)];
        cols.extend(self.y_names.iter().map(|n| Column::new(n, ColumnKind::Significant(6))));
        cols.push(Column::new("VIP", ColumnKind::Significant(6)));
        let mut c = Table::new("Coefficients (original units)", cols);
        let mut crow = vec![Cell::text("Constant")];
        crow.extend(self.intercepts.iter().map(|&v| Cell::Num(v)));
        crow.push(Cell::Missing);
        c.row(crow);
        for (j, name) in self.x_names.iter().enumerate() {
            let mut row = vec![Cell::text(name)];
            row.extend((0..self.y_names.len()).map(|k| Cell::Num(self.coefficients[(j, k)])));
            row.push(Cell::Num(self.vip[j]));
            c.row(row);
        }
        sec.table(c);
        if self.selection.len() > 1 {
            let x: Vec<f64> = self.selection.iter().map(|s| s.components as f64).collect();
            let mut series = Vec::new();
            for (k, name) in self.y_names.iter().enumerate() {
                if self.selection.iter().all(|s| s.press.is_some()) {
                    series.push(Series {
                        name: format!("R-Sq (pred) {name}"),
                        x: x.clone(),
                        y: self.selection.iter().map(|s| s.r2_pred.as_ref().map_or(f64::NAN, |v| v[k])).collect(),
                    });
                }
                series.push(Series {
                    name: format!("R-Sq {name}"),
                    x: x.clone(),
                    y: self.selection.iter().map(|s| s.r2[k]).collect(),
                });
            }
            sec.chart(ChartSpec {
                kind: ChartKind::Line,
                title: "Model Selection Plot".into(),
                x_label: "Number of components".into(),
                y_label: "R-Sq".into(),
                categories: vec![],
                series,
                reference_lines: vec![],
            });
        }
        if self.x_scores.cols() >= 2 {
            sec.chart(ChartSpec {
                kind: ChartKind::Scatter,
                title: "Score Plot".into(),
                x_label: "t1".into(),
                y_label: "t2".into(),
                categories: vec![],
                series: vec![Series { name: "Scores".into(), x: self.x_scores.col(0), y: self.x_scores.col(1) }],
                reference_lines: vec![],
            });
        }
        sec
    }
}
