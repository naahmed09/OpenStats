//! Cox proportional hazards regression (Breslow and Efron tie handling, optional strata).

use crate::error::{invalid, Result, StatsError};
use crate::model::{build_design, Data, DesignOptions};
use openstats_math::dist::{ChiSquared, ContinuousDistribution, Normal};
use openstats_math::linalg::{Cholesky, Matrix};

/// Handling of tied event times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoxTies {
    /// Breslow's approximation.
    Breslow,
    /// Efron's approximation (the usual default).
    Efron,
}

/// Options for [`cox_ph`] and [`cox_formula`].
#[derive(Debug, Clone)]
pub struct CoxOptions {
    /// Tie handling.
    pub ties: CoxTies,
    /// Confidence level of the hazard-ratio intervals.
    pub level: f64,
    /// Maximum Newton iterations.
    pub max_iter: usize,
}

impl Default for CoxOptions {
    fn default() -> Self {
        CoxOptions { ties: CoxTies::Efron, level: 0.95, max_iter: 50 }
    }
}

/// One coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxCoefficient {
    /// Name.
    pub name: String,
    /// Estimate (log hazard ratio).
    pub coef: f64,
    /// Standard error.
    pub se: f64,
    /// Wald z.
    pub z: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Hazard ratio `exp(coef)`.
    pub hazard_ratio: f64,
    /// Lower limit of the hazard ratio.
    pub hr_lower: f64,
    /// Upper limit of the hazard ratio.
    pub hr_upper: f64,
}

/// Cox model fit.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxFit {
    /// Tie method used.
    pub ties: CoxTies,
    /// Observations used.
    pub n: usize,
    /// Events among them.
    pub events: usize,
    /// Coefficients.
    pub coefficients: Vec<CoxCoefficient>,
    /// Covariance matrix of the estimates (row-major `p × p`).
    pub covariance: Vec<f64>,
    /// Log partial likelihood at the null (β = 0).
    pub loglik_null: f64,
    /// Log partial likelihood at the estimate.
    pub loglik: f64,
    /// Likelihood-ratio statistic, Wald statistic, score statistic and the common df.
    pub lr: f64,
    /// Wald chi-square.
    pub wald: f64,
    /// Score chi-square.
    pub score: f64,
    /// Degrees of freedom of the three tests.
    pub df: usize,
    /// p-values of the LR, Wald and score tests.
    pub p_lr: f64,
    /// Wald p-value.
    pub p_wald: f64,
    /// Score p-value.
    pub p_score: f64,
    /// Harrell's concordance index.
    pub concordance: f64,
    /// Linear predictors `xβ` for the rows used.
    pub linear_predictor: Vec<f64>,
    /// Indices of the input rows used.
    pub rows_used: Vec<usize>,
    /// Newton iterations taken.
    pub iterations: usize,
    /// Whether the iteration converged.
    pub converged: bool,
    /// Number of strata.
    pub n_strata: usize,
}

struct Prepared<'a> {
    time: &'a [f64],
    event: &'a [bool],
    x: &'a [Vec<f64>], // row-major, n × p
    strata: Vec<usize>,
}

/// Log partial likelihood, score and information at `beta`.
fn lik(d: &Prepared, beta: &[f64], ties: CoxTies, need_derivs: bool) -> (f64, Vec<f64>, Vec<f64>) {
    let p = beta.len();
    let n = d.time.len();
    let eta: Vec<f64> = (0..n).map(|i| (0..p).map(|j| d.x[i][j] * beta[j]).sum()).collect();
    // Centre the linear predictor for numerical safety (does not change the likelihood).
    let shift = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shift = if shift.is_finite() { shift } else { 0.0 };
    let r: Vec<f64> = eta.iter().map(|e| (e - shift).exp()).collect();
    let mut ll = 0.0;
    let mut u = vec![0.0; p];
    let mut info = vec![0.0; p * p];
    let mut strata: Vec<usize> = d.strata.clone();
    strata.sort_unstable();
    strata.dedup();
    for &s in &strata {
        let mut idx: Vec<usize> = (0..n).filter(|&i| d.strata[i] == s).collect();
        idx.sort_by(|&a, &b| d.time[b].total_cmp(&d.time[a])); // descending
        let mut s0 = 0.0;
        let mut s1 = vec![0.0; p];
        let mut s2 = vec![0.0; p * p];
        let mut k = 0;
        while k < idx.len() {
            let t = d.time[idx[k]];
            let mut e = k;
            while e < idx.len() && d.time[idx[e]] == t {
                e += 1;
            }
            // Everyone at this time joins the risk set.
            let mut dcount = 0usize;
            let mut t0 = 0.0;
            let mut t1 = vec![0.0; p];
            let mut t2 = vec![0.0; p * p];
            for &i in &idx[k..e] {
                s0 += r[i];
                for a in 0..p {
                    s1[a] += r[i] * d.x[i][a];
                    if need_derivs {
                        for b in 0..p {
                            s2[a * p + b] += r[i] * d.x[i][a] * d.x[i][b];
                        }
                    }
                }
                if d.event[i] {
                    dcount += 1;
                    t0 += r[i];
                    ll += eta[i] - shift;
                    for a in 0..p {
                        t1[a] += r[i] * d.x[i][a];
                        u[a] += d.x[i][a];
                        if need_derivs {
                            for b in 0..p {
                                t2[a * p + b] += r[i] * d.x[i][a] * d.x[i][b];
                            }
                        }
                    }
                }
            }
            for l in 0..dcount {
                let c = match ties {
                    CoxTies::Breslow => 0.0,
                    CoxTies::Efron => l as f64 / dcount as f64,
                };
                let a0 = s0 - c * t0;
                ll -= a0.ln();
                for a in 0..p {
                    let a1a = s1[a] - c * t1[a];
                    u[a] -= a1a / a0;
                    if need_derivs {
                        for b in 0..p {
                            let a1b = s1[b] - c * t1[b];
                            let a2 = s2[a * p + b] - c * t2[a * p + b];
                            info[a * p + b] += a2 / a0 - a1a * a1b / (a0 * a0);
                        }
                    }
                }
            }
            k = e;
        }
    }
    (ll, u, info)
}

fn solve(info: &[f64], rhs: &[f64], p: usize) -> Result<Vec<f64>> {
    let m = Matrix::from_fn(p, p, |i, j| info[i * p + j]);
    Ok(Cholesky::new(&m)?.solve(rhs)?)
}

/// Fit a Cox model to the numeric covariate columns `x` (`p` columns of length `n`).
pub fn cox_ph(
    time: &[f64],
    event: &[bool],
    x: &[Vec<f64>],
    names: &[String],
    strata: Option<&[usize]>,
    opts: &CoxOptions,
) -> Result<CoxFit> {
    let n = time.len();
    let p = x.len();
    if event.len() != n || x.iter().any(|c| c.len() != n) || strata.is_some_and(|s| s.len() != n) || names.len() != p {
        return invalid("time, event, covariate and strata columns must have the same length");
    }
    if !(0.0..1.0).contains(&opts.level) || opts.level == 0.0 {
        return invalid("the confidence level must be between 0 and 1");
    }
    let rows_used: Vec<usize> = (0..n).filter(|&i| !time[i].is_nan() && x.iter().all(|c| !c[i].is_nan())).collect();
    if time.iter().any(|t| t.is_infinite()) {
        return invalid("times must be finite");
    }
    let tm: Vec<f64> = rows_used.iter().map(|&i| time[i]).collect();
    let ev: Vec<bool> = rows_used.iter().map(|&i| event[i]).collect();
    let rows: Vec<Vec<f64>> = rows_used.iter().map(|&i| x.iter().map(|c| c[i]).collect()).collect();
    let st: Vec<usize> = rows_used.iter().map(|&i| strata.map_or(0, |s| s[i])).collect();
    let events = ev.iter().filter(|&&e| e).count();
    if p == 0 {
        return invalid("select at least one predictor");
    }
    if events < 2 {
        return Err(StatsError::InsufficientData { what: "Cox regression (events)", needed: 2, got: events });
    }
    if events <= p {
        return invalid("there must be more events than coefficients");
    }
    let d = Prepared { time: &tm, event: &ev, x: &rows, strata: st.clone() };
    let mut beta = vec![0.0; p];
    let (ll0, u0, i0) = lik(&d, &beta, opts.ties, true);
    let score = {
        let s = solve(&i0, &u0, p).map_err(|_| StatsError::InvalidArgument("the covariates are collinear".into()))?;
        u0.iter().zip(&s).map(|(a, b)| a * b).sum::<f64>()
    };
    let mut ll = ll0;
    let mut converged = false;
    let mut iterations = 0;
    for it in 0..opts.max_iter {
        iterations = it + 1;
        let (l, u, info) = lik(&d, &beta, opts.ties, true);
        ll = l;
        let step = solve(&info, &u, p).map_err(|_| {
            StatsError::InvalidArgument("the covariates are collinear or the likelihood is degenerate".into())
        })?;
        let mut frac = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let cand: Vec<f64> = beta.iter().zip(&step).map(|(b, s)| b + frac * s).collect();
            let (lc, _, _) = lik(&d, &cand, opts.ties, false);
            if lc.is_finite() && lc >= l - 1e-12 {
                let gain = lc - l;
                beta = cand;
                ll = lc;
                accepted = true;
                if gain.abs() < 1e-11 * (1.0 + l.abs()) && step.iter().all(|s| s.abs() < 1e-7) {
                    converged = true;
                }
                break;
            }
            frac /= 2.0;
        }
        if !accepted || converged {
            converged = converged || step.iter().all(|s| s.abs() < 1e-8);
            break;
        }
    }
    let (llf, _, info) = lik(&d, &beta, opts.ties, true);
    ll = ll.max(llf);
    let m = Matrix::from_fn(p, p, |i, j| info[i * p + j]);
    let cov = Cholesky::new(&m).and_then(|c| c.inverse()).map_err(|_| {
        StatsError::InvalidArgument("the information matrix is singular (separation or collinearity)".into())
    })?;
    let nd = Normal::new(0.0, 1.0)?;
    let zc = nd.quantile(1.0 - (1.0 - opts.level) / 2.0);
    let mut coefs = Vec::new();
    for j in 0..p {
        let se = cov[(j, j)].sqrt();
        let z = beta[j] / se;
        coefs.push(CoxCoefficient {
            name: names[j].clone(),
            coef: beta[j],
            se,
            z,
            p_value: (2.0 * nd.sf(z.abs())).min(1.0),
            hazard_ratio: beta[j].exp(),
            hr_lower: (beta[j] - zc * se).exp(),
            hr_upper: (beta[j] + zc * se).exp(),
        });
    }
    // Wald: β' I β
    let mut wald = 0.0;
    for a in 0..p {
        for b in 0..p {
            wald += beta[a] * info[a * p + b] * beta[b];
        }
    }
    let lr = 2.0 * (ll - ll0);
    let chi = ChiSquared::new(p as f64)?;
    let lp: Vec<f64> = rows.iter().map(|r| r.iter().zip(&beta).map(|(a, b)| a * b).sum()).collect();
    let concordance = concordance_index(&tm, &ev, &lp, &st);
    let mut covv = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            covv[a * p + b] = cov[(a, b)];
        }
    }
    let n_strata = {
        let mut s = st.clone();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    Ok(CoxFit {
        ties: opts.ties,
        n: rows_used.len(),
        events,
        coefficients: coefs,
        covariance: covv,
        loglik_null: ll0,
        loglik: ll,
        lr,
        wald,
        score,
        df: p,
        p_lr: chi.sf(lr.max(0.0)),
        p_wald: chi.sf(wald),
        p_score: chi.sf(score),
        concordance,
        linear_predictor: lp,
        rows_used,
        iterations,
        converged,
        n_strata,
    })
}

/// Harrell's C within strata. Pairs with equal predictors count one half.
fn concordance_index(t: &[f64], ev: &[bool], lp: &[f64], st: &[usize]) -> f64 {
    let n = t.len();
    let (mut conc, mut total) = (0.0, 0.0);
    for i in 0..n {
        if !ev[i] {
            continue;
        }
        for j in 0..n {
            if i == j || st[i] != st[j] {
                continue;
            }
            let usable = t[i] < t[j] || (t[i] == t[j] && !ev[j]);
            if !usable {
                continue;
            }
            total += 1.0;
            if lp[i] > lp[j] {
                conc += 1.0;
            } else if lp[i] == lp[j] {
                conc += 0.5;
            }
        }
    }
    if total == 0.0 {
        f64::NAN
    } else {
        conc / total
    }
}

/// Cox regression from a model formula right-hand side such as `age + C(sex) + age:bmi`.
///
/// `time` and `event` have one entry per row of `data` (a missing time drops the row).
pub fn cox_formula(
    rhs: &str,
    data: &Data,
    time: &[f64],
    event: &[bool],
    strata: Option<&[usize]>,
    design: &DesignOptions,
    opts: &CoxOptions,
) -> Result<CoxFit> {
    let mut dd = data.clone();
    dd = dd.with_numeric("surv_time__", time.to_vec())?;
    let (_, dm) = build_design(&format!("surv_time__ ~ {rhs}"), &dd, design)?;
    let first = usize::from(dm.intercept);
    let p = dm.x.cols() - first;
    if p == 0 {
        return invalid("the model has no predictors");
    }
    let n = time.len();
    // Expand the design to the input rows (rows dropped for missing values become NaN).
    let mut cols = vec![vec![f64::NAN; n]; p];
    for (k, &row) in dm.rows_used.iter().enumerate() {
        for (j, col) in cols.iter_mut().enumerate() {
            col[row] = dm.x[(k, first + j)];
        }
    }
    let names: Vec<String> = dm.names[first..].to_vec();
    let mut fit = cox_ph(time, event, &cols, &names, strata, opts)?;
    fit.rows_used.sort_unstable();
    Ok(fit)
}
