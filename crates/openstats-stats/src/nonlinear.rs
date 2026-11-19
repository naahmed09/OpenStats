//! Nonlinear regression: least-squares fit of a user-written equation such
//! as `b0 * exp(b1 * x)` by Levenberg–Marquardt, with asymptotic standard
//! errors and intervals, parameter correlations, and confidence and
//! prediction limits for new observations (delta method).

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::model::{Data, Expr};
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::linalg::Matrix;
use openstats_math::optim::{levenberg_marquardt, LsOptions};

/// A parameter with its starting value and optional bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// Name used in the equation.
    pub name: String,
    /// Starting value.
    pub start: f64,
    /// Lower bound (`None` = unbounded).
    pub lower: Option<f64>,
    /// Upper bound (`None` = unbounded).
    pub upper: Option<f64>,
}

impl Parameter {
    /// Unbounded parameter.
    pub fn new(name: &str, start: f64) -> Parameter {
        Parameter { name: name.into(), start, lower: None, upper: None }
    }
}

/// Options of a nonlinear fit.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearOptions {
    /// Confidence level of the intervals.
    pub level: f64,
    /// Maximum Levenberg–Marquardt iterations.
    pub max_iter: usize,
}

impl Default for NonlinearOptions {
    fn default() -> Self {
        NonlinearOptions { level: 0.95, max_iter: 500 }
    }
}

/// One row of the parameter table.
#[derive(Debug, Clone, PartialEq)]
pub struct NlCoefficient {
    /// Parameter name.
    pub name: String,
    /// Estimate.
    pub estimate: f64,
    /// Asymptotic standard error.
    pub std_err: f64,
    /// Approximate confidence interval.
    pub ci: Interval,
}

/// Prediction of a fitted nonlinear model.
#[derive(Debug, Clone, PartialEq)]
pub struct NlPrediction {
    /// Worksheet row of the new data.
    pub row: usize,
    /// Fitted value.
    pub fit: f64,
    /// Standard error of the fit.
    pub se_fit: f64,
    /// Confidence interval of the mean response.
    pub ci: Interval,
    /// Prediction interval of a new observation.
    pub pi: Interval,
}

/// A fitted nonlinear regression.
#[derive(Debug, Clone)]
pub struct NonlinearFit {
    /// The equation as given.
    pub equation: String,
    /// Response column.
    pub response: String,
    /// Observations used.
    pub n: usize,
    /// Parameter table.
    pub coefficients: Vec<NlCoefficient>,
    /// Error sum of squares.
    pub sse: f64,
    /// Error degrees of freedom.
    pub df_error: usize,
    /// Mean square error.
    pub mse: f64,
    /// Residual standard error.
    pub s: f64,
    /// Iterations used.
    pub iterations: usize,
    /// Whether the optimizer met a convergence criterion.
    pub converged: bool,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Residuals.
    pub residuals: Vec<f64>,
    /// Correlation matrix of the estimates.
    pub correlation: Matrix,
    /// Covariance matrix of the estimates.
    pub cov: Matrix,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    /// Confidence level.
    pub level: f64,
    /// Predictor columns the equation reads.
    pub predictors: Vec<String>,
    /// Parameters that ended on a bound.
    pub at_bound: Vec<String>,
    expr: Expr,
    params: Vec<String>,
}

struct Plan<'a> {
    expr: &'a Expr,
    params: &'a [String],
    cols: Vec<(&'a str, &'a [f64])>,
}

impl Plan<'_> {
    fn eval(&self, theta: &[f64], row: usize) -> f64 {
        self.expr.evaluate(&|name| {
            if let Some(k) = self.params.iter().position(|p| p == name) {
                return theta[k];
            }
            self.cols.iter().find(|(n, _)| *n == name).map_or(f64::NAN, |(_, c)| c[row])
        })
    }
}

fn split_equation(eq: &str) -> &str {
    match eq.find(['=', '~']) {
        Some(i) => &eq[i + 1..],
        None => eq,
    }
}

/// Map an unbounded working variable to the bounded parameter.
fn to_param(u: f64, lo: Option<f64>, hi: Option<f64>) -> f64 {
    match (lo, hi) {
        (None, None) => u,
        (Some(l), None) => l + u * u,
        (None, Some(h)) => h - u * u,
        (Some(l), Some(h)) => l + (h - l) * (u.sin() + 1.0) / 2.0,
    }
}

fn to_working(p: f64, lo: Option<f64>, hi: Option<f64>) -> f64 {
    match (lo, hi) {
        (None, None) => p,
        (Some(l), None) => (p - l).max(0.0).sqrt(),
        (None, Some(h)) => (h - p).max(0.0).sqrt(),
        (Some(l), Some(h)) => ((2.0 * (p - l) / (h - l)) - 1.0).clamp(-1.0, 1.0).asin(),
    }
}

/// Fit `response = equation(predictors; parameters)` by least squares.
/// `equation` may be written `y = expr` or just `expr`. Rows with a missing
/// response or predictor are dropped.
pub fn nonlinear_regression(
    response: &str,
    equation: &str,
    params: &[Parameter],
    data: &Data,
    opts: &NonlinearOptions,
) -> Result<NonlinearFit> {
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
    }
    if params.is_empty() {
        return invalid("the equation needs at least one parameter");
    }
    for (i, p) in params.iter().enumerate() {
        if params[..i].iter().any(|q| q.name == p.name) {
            return invalid(format!("duplicate parameter '{}'", p.name));
        }
        if !p.start.is_finite() {
            return invalid(format!("the starting value of '{}' is not a number", p.name));
        }
        if let (Some(l), Some(h)) = (p.lower, p.upper) {
            if l >= h {
                return invalid(format!("the bounds of '{}' are not ordered", p.name));
            }
        }
        if p.lower.is_some_and(|l| p.start < l) || p.upper.is_some_and(|h| p.start > h) {
            return invalid(format!("the starting value of '{}' is outside its bounds", p.name));
        }
    }
    let expr = Expr::parse(split_equation(equation))?;
    let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
    let predictors: Vec<String> = expr.variables().into_iter().filter(|v| !names.contains(v)).collect();
    let y_all =
        data.numeric(response).ok_or_else(|| StatsError::InvalidArgument(format!("no numeric column '{response}'")))?;
    let mut pred_cols: Vec<(&str, &[f64])> = Vec::new();
    for p in &predictors {
        let c = data
            .numeric(p)
            .ok_or_else(|| StatsError::InvalidArgument(format!("'{p}' is neither a parameter nor a numeric column")))?;
        pred_cols.push((p.as_str(), c));
    }
    let rows_used: Vec<usize> =
        (0..y_all.len()).filter(|&i| y_all[i].is_finite() && pred_cols.iter().all(|(_, c)| c[i].is_finite())).collect();
    let n = rows_used.len();
    let k = params.len();
    if n <= k {
        return Err(StatsError::InsufficientData { what: "nonlinear regression", needed: k + 1, got: n });
    }
    let y: Vec<f64> = rows_used.iter().map(|&i| y_all[i]).collect();
    let used_cols: Vec<Vec<f64>> = pred_cols.iter().map(|(_, c)| rows_used.iter().map(|&i| c[i]).collect()).collect();
    let plan = Plan {
        expr: &expr,
        params: &names,
        cols: pred_cols.iter().zip(&used_cols).map(|((nm, _), c)| (*nm, c.as_slice())).collect(),
    };
    let (los, his): (Vec<Option<f64>>, Vec<Option<f64>>) = params.iter().map(|p| (p.lower, p.upper)).unzip();
    let theta_of = |u: &[f64]| -> Vec<f64> { (0..k).map(|j| to_param(u[j], los[j], his[j])).collect() };
    let mut resid = |u: &[f64]| -> Vec<f64> {
        let th = theta_of(u);
        (0..n)
            .map(|i| {
                let f = plan.eval(&th, i);
                if f.is_finite() {
                    y[i] - f
                } else {
                    1e150
                }
            })
            .collect()
    };
    let u0: Vec<f64> = (0..k).map(|j| to_working(params[j].start, los[j], his[j])).collect();
    // A start exactly on a bound has a zero derivative in the working variable: nudge it.
    let u0: Vec<f64> =
        u0.iter().map(|&u| if u == 0.0 && (los.iter().chain(&his).any(|b| b.is_some())) { 1e-4 } else { u }).collect();
    let ls = LsOptions { max_iter: opts.max_iter, central_jacobian: true, ..LsOptions::default() };
    let fit = levenberg_marquardt(&mut resid, None, &u0, &ls)?;
    let theta = theta_of(&fit.x);
    if theta.iter().any(|v| !v.is_finite()) || !fit.sse.is_finite() || fit.sse >= 1e140 {
        return invalid("the equation could not be evaluated at the starting values; try different starts");
    }
    let df_error = n - k;
    let mse = fit.sse / df_error as f64;

    // Covariance in the natural parameters: re-difference the residuals there.
    let mut res_nat = |th: &[f64]| -> Vec<f64> { (0..n).map(|i| y[i] - plan.eval(th, i)).collect() };
    let jac = numeric_jacobian(&mut res_nat, &theta);
    let jtj = jac.gram();
    let inv = openstats_math::linalg::Svd::new(&jtj)?.pinv(Some(1e-12 * jtj.norm_max().max(1e-300)));
    let cov = inv.scale(mse);
    let tq = StudentT::new(df_error as f64)?.isf((1.0 - opts.level) / 2.0);
    let coefficients: Vec<NlCoefficient> = (0..k)
        .map(|j| {
            let se = cov[(j, j)].max(0.0).sqrt();
            NlCoefficient {
                name: params[j].name.clone(),
                estimate: theta[j],
                std_err: se,
                ci: Interval { lower: theta[j] - tq * se, upper: theta[j] + tq * se, level: opts.level },
            }
        })
        .collect();
    let mut corr = Matrix::identity(k);
    for a in 0..k {
        for b in 0..k {
            corr[(a, b)] = cov[(a, b)] / (cov[(a, a)] * cov[(b, b)]).sqrt();
        }
    }
    let fitted: Vec<f64> = (0..n).map(|i| plan.eval(&theta, i)).collect();
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
    let at_bound: Vec<String> = (0..k)
        .filter(|&j| {
            let tol = 1e-6 * (1.0 + theta[j].abs());
            los[j].is_some_and(|l| (theta[j] - l).abs() < tol) || his[j].is_some_and(|h| (theta[j] - h).abs() < tol)
        })
        .map(|j| params[j].name.clone())
        .collect();
    Ok(NonlinearFit {
        equation: equation.to_string(),
        response: response.to_string(),
        n,
        coefficients,
        sse: fit.sse,
        df_error,
        mse,
        s: mse.sqrt(),
        iterations: fit.iterations,
        converged: fit.converged,
        fitted,
        residuals,
        correlation: corr,
        cov,
        rows_used,
        level: opts.level,
        predictors,
        at_bound,
        expr,
        params: names,
    })
}

fn numeric_jacobian(f: &mut dyn FnMut(&[f64]) -> Vec<f64>, x: &[f64]) -> Matrix {
    let k = x.len();
    let base = f(x);
    let m = base.len();
    let mut j = Matrix::zeros(m, k);
    for c in 0..k {
        let h = 1e-6 * x[c].abs().max(1e-3);
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[c] += h;
        xm[c] -= h;
        let (rp, rm) = (f(&xp), f(&xm));
        for i in 0..m {
            // Residual is y − f, so ∂r/∂θ = −∂f/∂θ; the Gram matrix is sign-free.
            j[(i, c)] = (rp[i] - rm[i]) / (2.0 * h);
        }
    }
    j
}

impl NonlinearFit {
    /// Parameter estimates in order.
    pub fn estimates(&self) -> Vec<f64> {
        self.coefficients.iter().map(|c| c.estimate).collect()
    }

    /// Predict at new data (every predictor column of the equation must be
    /// present); rows with missing predictors are skipped.
    pub fn predict(&self, data: &Data, level: f64) -> Result<Vec<NlPrediction>> {
        if !(level > 0.0 && level < 1.0) {
            return invalid(format!("confidence level {level} must be in (0, 1)"));
        }
        let mut cols: Vec<(&str, &[f64])> = Vec::new();
        for p in &self.predictors {
            let c = data
                .numeric(p)
                .ok_or_else(|| StatsError::InvalidArgument(format!("the new data have no numeric column '{p}'")))?;
            cols.push((p.as_str(), c));
        }
        let rows: Vec<usize> = (0..data.n_rows()).filter(|&i| cols.iter().all(|(_, c)| c[i].is_finite())).collect();
        let plan = Plan { expr: &self.expr, params: &self.params, cols };
        let theta = self.estimates();
        let tq = StudentT::new(self.df_error as f64)?.isf((1.0 - level) / 2.0);
        let k = theta.len();
        let mut out = Vec::with_capacity(rows.len());
        for &i in &rows {
            let fit = plan.eval(&theta, i);
            let mut g = vec![0.0; k];
            for c in 0..k {
                let h = 1e-6 * theta[c].abs().max(1e-3);
                let (mut a, mut b) = (theta.clone(), theta.clone());
                a[c] += h;
                b[c] -= h;
                g[c] = (plan.eval(&a, i) - plan.eval(&b, i)) / (2.0 * h);
            }
            let v = self.cov.matvec(&g)?;
            let var: f64 = g.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>().max(0.0);
            let se_fit = var.sqrt();
            let se_pred = (var + self.mse).sqrt();
            out.push(NlPrediction {
                row: i,
                fit,
                se_fit,
                ci: Interval { lower: fit - tq * se_fit, upper: fit + tq * se_fit, level },
                pi: Interval { lower: fit - tq * se_pred, upper: fit + tq * se_pred, level },
            });
        }
        Ok(out)
    }
}

impl Reportable for NonlinearFit {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(format!(
            "Nonlinear Regression: {} = {}",
            self.response,
            split_equation(&self.equation).trim()
        ));
        let mut t = Table::new(
            "Parameter Estimates",
            vec![
                Column::new("Parameter", ColumnKind::Text),
                Column::new("Estimate", ColumnKind::Significant(6)),
                Column::new("SE Estimate", ColumnKind::Significant(6)),
                Column::new(format!("{}% CI", self.level * 100.0), ColumnKind::Text),
            ],
        );
        for c in &self.coefficients {
            t.row(vec![
                Cell::text(&c.name),
                Cell::Num(c.estimate),
                Cell::Num(c.std_err),
                Cell::text(format!("({:.6}, {:.6})", c.ci.lower, c.ci.upper)),
            ]);
        }
        sec.table(t);
        let mut s = Table::new(
            "Summary",
            vec![
                Column::new("Iterations", ColumnKind::Integer),
                Column::new("Final SSE", ColumnKind::Significant(6)),
                Column::new("DFE", ColumnKind::Integer),
                Column::new("MSE", ColumnKind::Significant(6)),
                Column::new("S", ColumnKind::Significant(6)),
            ],
        );
        s.row(vec![
            Cell::Int(i64::try_from(self.iterations).unwrap_or(i64::MAX)),
            Cell::Num(self.sse),
            Cell::Int(i64::try_from(self.df_error).unwrap_or(i64::MAX)),
            Cell::Num(self.mse),
            Cell::Num(self.s),
        ]);
        sec.table(s);
        if self.coefficients.len() > 1 {
            let mut c = Table::new(
                "Correlation of Estimates",
                std::iter::once(Column::new("", ColumnKind::Text))
                    .chain(self.coefficients.iter().map(|p| Column::new(&p.name, ColumnKind::Fixed(4))))
                    .collect(),
            );
            for (a, pa) in self.coefficients.iter().enumerate() {
                let mut row = vec![Cell::text(&pa.name)];
                row.extend((0..self.coefficients.len()).map(|b| Cell::Num(self.correlation[(a, b)])));
                c.row(row);
            }
            sec.table(c);
        }
        if !self.converged {
            sec.warn("The iteration stopped before meeting the convergence criteria; try other starting values.");
        }
        if !self.at_bound.is_empty() {
            sec.warn(format!("Parameters on a bound: {}", self.at_bound.join(", ")));
        }
        sec.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: "Residuals versus Fits".into(),
            x_label: "Fitted value".into(),
            y_label: "Residual".into(),
            categories: vec![],
            series: vec![Series { name: "Residual".into(), x: self.fitted.clone(), y: self.residuals.clone() }],
            reference_lines: vec![crate::report::ReferenceLine {
                axis: crate::report::Axis::Y,
                value: 0.0,
                label: "0".into(),
            }],
        });
        sec
    }
}
