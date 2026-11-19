//! Ordinal (cumulative-link, proportional odds) and nominal (baseline
//! category) logistic regression by Newton–Raphson with analytic derivatives.
//!
//! The ordinal model is `P(Y ≤ j) = F(θⱼ − x′β)` for `j = 1 … K−1`, so a
//! positive coefficient makes higher categories more likely and
//! `exp(β)` is the odds ratio of a higher versus a lower category. The
//! nominal model is `ln(P(Y = j)/P(Y = 0)) = x′βⱼ` against the reference
//! category 0.

use super::{std_normal, Link};
use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::model::{build_design, Data, DesignOptions, ModelSpec};
use crate::regression::independent_columns;
use openstats_math::dist::{ChiSquared, ContinuousDistribution};
use openstats_math::linalg::{Cholesky, Matrix};
use std::collections::BTreeMap;

/// Options of an ordinal or nominal logistic fit.
#[derive(Debug, Clone)]
pub struct CategoricalOptions {
    /// Category labels in order (their number is the number of categories);
    /// the response column holds the 0-based index of each observation's category.
    pub labels: Vec<String>,
    /// Link of the cumulative probabilities (ordinal only).
    pub link: Link,
    /// Reference category index (nominal only).
    pub reference: usize,
    /// Confidence level of the intervals.
    pub level: f64,
    /// Maximum Newton iterations.
    pub max_iter: usize,
}

impl CategoricalOptions {
    /// Defaults with logit link, reference category 0 and 95 % intervals.
    pub fn new(labels: Vec<String>) -> CategoricalOptions {
        CategoricalOptions { labels, link: Link::Logit, reference: 0, level: 0.95, max_iter: 100 }
    }
}

/// One estimated parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitCoef {
    /// Name (`Const(1)`, `x`, …).
    pub name: String,
    /// Estimate.
    pub estimate: f64,
    /// Standard error.
    pub std_err: f64,
    /// z statistic.
    pub z: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Confidence interval.
    pub ci: Interval,
    /// exp(estimate) for predictors.
    pub odds_ratio: Option<f64>,
    /// Confidence interval of the odds ratio.
    pub or_ci: Option<Interval>,
}

/// Likelihood-ratio and goodness-of-fit tests.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalTests {
    /// Log-likelihood of the model.
    pub loglik: f64,
    /// Log-likelihood of the model without predictors.
    pub null_loglik: f64,
    /// Likelihood-ratio statistic G = 2(ll − ll₀).
    pub g: f64,
    /// Degrees of freedom of G.
    pub g_df: usize,
    /// p-value of G.
    pub g_p: f64,
    /// Pearson χ² over covariate patterns: (statistic, df, p).
    pub pearson: Option<(f64, usize, f64)>,
    /// Deviance χ² over covariate patterns: (statistic, df, p).
    pub deviance: Option<(f64, usize, f64)>,
}

/// Ordinal logistic regression result.
#[derive(Debug, Clone)]
pub struct OrdinalFit {
    /// Link used.
    pub link: Link,
    /// Category labels in order.
    pub categories: Vec<String>,
    /// Observations per category.
    pub counts: Vec<usize>,
    /// Observations used.
    pub n: usize,
    /// Threshold parameters θ₁ … θ_{K−1}.
    pub thresholds: Vec<LogitCoef>,
    /// Predictor coefficients.
    pub coefficients: Vec<LogitCoef>,
    /// Tests.
    pub tests: CategoricalTests,
    /// Predicted category probabilities (n × K).
    pub probabilities: Matrix,
    /// Most probable category per observation.
    pub predicted: Vec<usize>,
    /// Columns dropped as aliased.
    pub aliased: Vec<String>,
    /// Newton iterations.
    pub iterations: usize,
    /// Whether the iteration converged.
    pub converged: bool,
    /// Confidence level.
    pub level: f64,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    /// Covariance of (θ, β).
    pub cov: Matrix,
    spec: Option<ModelSpec>,
    kept: Vec<usize>,
}

/// Nominal logistic regression result.
#[derive(Debug, Clone)]
pub struct NominalFit {
    /// Category labels.
    pub categories: Vec<String>,
    /// Reference category index.
    pub reference: usize,
    /// Observations per category.
    pub counts: Vec<usize>,
    /// Observations used.
    pub n: usize,
    /// Coefficients of each non-reference category's logit (constant first),
    /// in category order skipping the reference.
    pub logits: Vec<(String, Vec<LogitCoef>)>,
    /// Tests.
    pub tests: CategoricalTests,
    /// Predicted category probabilities (n × K, in category order).
    pub probabilities: Matrix,
    /// Most probable category per observation.
    pub predicted: Vec<usize>,
    /// Columns dropped as aliased.
    pub aliased: Vec<String>,
    /// Newton iterations.
    pub iterations: usize,
    /// Whether the iteration converged.
    pub converged: bool,
    /// Confidence level.
    pub level: f64,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    /// Covariance of the stacked coefficients.
    pub cov: Matrix,
    spec: Option<ModelSpec>,
    kept: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

type NewtonEval<'a> = &'a mut dyn FnMut(&[f64]) -> Option<(f64, Vec<f64>, Matrix)>;
type PatternTest = Option<(f64, usize, f64)>;

/// Damped Newton ascent. `eval(v)` returns the log-likelihood, gradient and
/// Hessian, or `None` when `v` is outside the domain.
fn newton_max(v0: Vec<f64>, max_iter: usize, eval: NewtonEval) -> Result<(Vec<f64>, f64, Matrix, usize, bool)> {
    let mut v = v0;
    let Some((mut ll, mut g, mut h)) = eval(&v) else {
        return invalid("invalid starting values for the logistic fit");
    };
    let k = v.len();
    let mut converged = false;
    let mut iters = 0;
    for it in 1..=max_iter {
        iters = it;
        let neg = h.scale(-1.0);
        let d = match Cholesky::new(&neg).and_then(|c| c.solve(&g)) {
            Ok(d) => d,
            Err(_) => {
                let mut r = neg.clone();
                let lam = 1e-6 * neg.diagonal().iter().fold(1.0f64, |a, b| a.max(b.abs()));
                for i in 0..k {
                    r[(i, i)] += lam;
                }
                Cholesky::new(&r)?.solve(&g)?
            }
        };
        let decrement: f64 = d.iter().zip(&g).map(|(a, b)| a * b).sum();
        let mut step = 1.0;
        let mut moved = false;
        for _ in 0..40 {
            let t: Vec<f64> = v.iter().zip(&d).map(|(a, b)| a + step * b).collect();
            if let Some((l2, g2, h2)) = eval(&t) {
                if l2 >= ll - 1e-12 * ll.abs() {
                    let small = (l2 - ll).abs() <= 1e-13 * (l2.abs() + 1.0);
                    v = t;
                    ll = l2;
                    g = g2;
                    h = h2;
                    moved = true;
                    if small && decrement < 1e-9 {
                        converged = true;
                    }
                    break;
                }
            }
            step *= 0.5;
        }
        if !moved || decrement < 1e-14 {
            converged = converged || decrement < 1e-8;
            break;
        }
        if converged {
            break;
        }
    }
    Ok((v, ll, h, iters, converged))
}

fn coef_row(name: String, est: f64, var: f64, level: f64, ratio: bool) -> LogitCoef {
    let se = var.max(0.0).sqrt();
    let z = est / se;
    let norm = std_normal();
    let crit = norm.isf((1.0 - level) / 2.0);
    let (lo, hi) = (est - crit * se, est + crit * se);
    LogitCoef {
        name,
        estimate: est,
        std_err: se,
        z,
        p_value: (2.0 * norm.sf(z.abs())).min(1.0),
        ci: Interval { lower: lo, upper: hi, level },
        odds_ratio: ratio.then(|| est.exp()),
        or_ci: ratio.then(|| Interval { lower: lo.exp(), upper: hi.exp(), level }),
    }
}

struct Prepared {
    spec: ModelSpec,
    x: Matrix,
    names: Vec<String>,
    kept: Vec<usize>,
    aliased: Vec<String>,
    cat: Vec<usize>,
    rows_used: Vec<usize>,
    counts: Vec<usize>,
}

fn prepare(
    formula: &str,
    data: &Data,
    design: &DesignOptions,
    opts: &CategoricalOptions,
    drop_constant: bool,
) -> Result<Prepared> {
    let k = opts.labels.len();
    if k < 2 {
        return invalid("the response needs at least two categories");
    }
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
    }
    let (spec, dm) = build_design(formula, data, design)?;
    let y = dm.y.as_ref().ok_or_else(|| StatsError::InvalidArgument("the formula needs a response".into()))?;
    let mut cat = Vec::with_capacity(y.len());
    for &v in y {
        if v < 0.0 || v.fract() != 0.0 || v as usize >= k {
            return invalid("the response must hold category indices 0 … K−1");
        }
        cat.push(v as usize);
    }
    let mut counts = vec![0usize; k];
    cat.iter().for_each(|&c| counts[c] += 1);
    if counts.contains(&0) {
        return invalid("every category must occur at least once among the complete cases");
    }
    let mut x = dm.x.clone();
    let mut names = dm.names.clone();
    let mut offset_cols = 0;
    if drop_constant && dm.intercept {
        let rows: Vec<usize> = (0..x.rows()).collect();
        let cols: Vec<usize> = (1..x.cols()).collect();
        x = x.select(&rows, &cols);
        names.remove(0);
        offset_cols = 1;
    }
    let (kept_rel, dropped) = if x.cols() == 0 { (vec![], vec![]) } else { independent_columns(&x, 1e-10) };
    let rows: Vec<usize> = (0..x.rows()).collect();
    let xk = x.select(&rows, &kept_rel);
    Ok(Prepared {
        spec,
        x: xk,
        aliased: dropped.iter().map(|&j| names[j].clone()).collect(),
        names: kept_rel.iter().map(|&j| names[j].clone()).collect(),
        kept: kept_rel.iter().map(|&j| j + offset_cols).collect(),
        cat,
        rows_used: dm.rows_used,
        counts,
    })
}

/// Pearson and deviance statistics over covariate patterns.
fn pattern_tests(
    x: &Matrix,
    cat: &[usize],
    probs: &Matrix,
    k: usize,
    n_params: usize,
) -> Result<(PatternTest, PatternTest)> {
    let mut groups: BTreeMap<Vec<u64>, (usize, Vec<f64>)> = BTreeMap::new();
    for i in 0..x.rows() {
        let key: Vec<u64> = x.row(i).iter().map(|v| v.to_bits()).collect();
        let e = groups.entry(key).or_insert_with(|| (i, vec![0.0; k]));
        e.1[cat[i]] += 1.0;
    }
    let g = groups.len();
    let cells = g * (k - 1);
    if cells <= n_params {
        return Ok((None, None));
    }
    let df = cells - n_params;
    let (mut pearson, mut dev) = (0.0, 0.0);
    for (first, counts) in groups.values() {
        let ng: f64 = counts.iter().sum();
        for c in 0..k {
            let e = ng * probs[(*first, c)];
            if e > 0.0 {
                pearson += (counts[c] - e).powi(2) / e;
            }
            if counts[c] > 0.0 {
                dev += 2.0 * counts[c] * (counts[c] / e).ln();
            }
        }
    }
    let chi = ChiSquared::new(df as f64)?;
    Ok((Some((pearson, df, chi.sf(pearson))), Some((dev, df, chi.sf(dev)))))
}

fn null_loglik(counts: &[usize]) -> f64 {
    let n: f64 = counts.iter().sum::<usize>() as f64;
    counts.iter().map(|&c| c as f64 * (c as f64 / n).ln()).sum()
}

// ---------------------------------------------------------------------------
// Ordinal
// ---------------------------------------------------------------------------

fn d2_link(link: Link, eta: f64) -> f64 {
    let f = link.d_mu(eta);
    match link {
        Link::Logit => f * (1.0 - 2.0 * link.inverse(eta)),
        Link::Probit => -eta * f,
        Link::Cloglog => f * (1.0 - eta.exp()),
        _ => 0.0,
    }
}

fn ordinal_probs(link: Link, theta: &[f64], beta: &[f64], x: &Matrix) -> Matrix {
    let n = x.rows();
    let k = theta.len() + 1;
    let mut p = Matrix::zeros(n, k);
    for i in 0..n {
        let xb: f64 = x.row(i).iter().zip(beta).map(|(a, b)| a * b).sum();
        let mut prev = 0.0;
        for j in 0..k {
            let cum = if j + 1 < k { link.inverse(theta[j] - xb) } else { 1.0 };
            p[(i, j)] = (cum - prev).max(0.0);
            prev = cum;
        }
    }
    p
}

/// Fit an ordinal (proportional-odds) logistic regression. The response of
/// `formula` holds the 0-based category index; `opts.labels` names the
/// ordered categories; the model constant is absorbed in the thresholds.
pub fn ordinal_logistic(
    formula: &str,
    data: &Data,
    design: &DesignOptions,
    opts: &CategoricalOptions,
) -> Result<OrdinalFit> {
    if !matches!(opts.link, Link::Logit | Link::Probit | Link::Cloglog) {
        return invalid("the ordinal link must be logit, probit or cloglog");
    }
    let prep = prepare(formula, data, design, opts, true)?;
    let k = opts.labels.len();
    let m = prep.x.cols();
    let n = prep.x.rows();
    let link = opts.link;
    // Start: thresholds from the marginal cumulative proportions.
    let mut theta0 = Vec::with_capacity(k - 1);
    let mut cum = 0.0;
    for c in 0..k - 1 {
        cum += prep.counts[c] as f64 / n as f64;
        theta0.push(link.link(cum));
    }
    let mut v0 = theta0.clone();
    v0.extend(std::iter::repeat_n(0.0, m));
    let np = k - 1 + m;
    let x = &prep.x;
    let cat = &prep.cat;
    let mut eval = |v: &[f64]| -> Option<(f64, Vec<f64>, Matrix)> {
        let theta = &v[..k - 1];
        if theta.windows(2).any(|w| w[1] <= w[0]) {
            return None;
        }
        let beta = &v[k - 1..];
        let mut ll = 0.0;
        let mut g = vec![0.0; np];
        let mut h = Matrix::zeros(np, np);
        for i in 0..n {
            let c = cat[i];
            let xi = x.row(i);
            let xb: f64 = xi.iter().zip(beta).map(|(a, b)| a * b).sum();
            let (e_hi, e_lo) = (
                if c + 1 < k { Some(theta[c] - xb) } else { None },
                if c > 0 { Some(theta[c - 1] - xb) } else { None },
            );
            let f_hi = e_hi.map_or((1.0, 0.0, 0.0), |e| (link.inverse(e), link.d_mu(e), d2_link(link, e)));
            let f_lo = e_lo.map_or((0.0, 0.0, 0.0), |e| (link.inverse(e), link.d_mu(e), d2_link(link, e)));
            let p = f_hi.0 - f_lo.0;
            if p.is_nan() || p <= 1e-300 || !p.is_finite() {
                return None;
            }
            ll += p.ln();
            // First derivatives of p wrt (θ_c, θ_{c−1}, β) and second derivatives.
            let (a_hi, a_lo) = (f_hi.1, f_lo.1);
            let (b_hi, b_lo) = (f_hi.2, f_lo.2);
            let mut dp = vec![0.0; np];
            if e_hi.is_some() {
                dp[c] = a_hi;
            }
            if e_lo.is_some() {
                dp[c - 1] = -a_lo;
            }
            for j in 0..m {
                dp[k - 1 + j] = -xi[j] * (a_hi - a_lo);
            }
            let mut d2 = Matrix::zeros(np, np);
            if e_hi.is_some() {
                d2[(c, c)] = b_hi;
                for j in 0..m {
                    d2[(c, k - 1 + j)] = -xi[j] * b_hi;
                    d2[(k - 1 + j, c)] = -xi[j] * b_hi;
                }
            }
            if e_lo.is_some() {
                d2[(c - 1, c - 1)] = -b_lo;
                for j in 0..m {
                    d2[(c - 1, k - 1 + j)] = xi[j] * b_lo;
                    d2[(k - 1 + j, c - 1)] = xi[j] * b_lo;
                }
            }
            for a in 0..m {
                for b in 0..m {
                    d2[(k - 1 + a, k - 1 + b)] = xi[a] * xi[b] * (b_hi - b_lo);
                }
            }
            for a in 0..np {
                g[a] += dp[a] / p;
                if dp[a] != 0.0 {
                    for b in 0..np {
                        h[(a, b)] -= dp[a] * dp[b] / (p * p);
                    }
                }
                for b in 0..np {
                    h[(a, b)] += d2[(a, b)] / p;
                }
            }
        }
        Some((ll, g, h))
    };
    let (v, ll, h, iterations, converged) = newton_max(v0, opts.max_iter, &mut eval)?;
    let cov = Cholesky::new(&h.scale(-1.0))?.inverse()?;
    let theta = &v[..k - 1];
    let beta = &v[k - 1..];
    let thresholds: Vec<LogitCoef> =
        (0..k - 1).map(|j| coef_row(format!("Const({})", j + 1), theta[j], cov[(j, j)], opts.level, false)).collect();
    let coefficients: Vec<LogitCoef> = (0..m)
        .map(|j| coef_row(prep.names[j].clone(), beta[j], cov[(k - 1 + j, k - 1 + j)], opts.level, true))
        .collect();
    let probabilities = ordinal_probs(link, theta, beta, x);
    let predicted: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = 0;
            for c in 1..k {
                if probabilities[(i, c)] > probabilities[(i, best)] {
                    best = c;
                }
            }
            best
        })
        .collect();
    let ll0 = null_loglik(&prep.counts);
    let g = 2.0 * (ll - ll0);
    let g_df = m;
    let (pearson, deviance) = pattern_tests(x, cat, &probabilities, k, np)?;
    let tests = CategoricalTests {
        loglik: ll,
        null_loglik: ll0,
        g,
        g_df,
        g_p: if g_df > 0 { ChiSquared::new(g_df as f64)?.sf(g.max(0.0)) } else { f64::NAN },
        pearson,
        deviance,
    };
    Ok(OrdinalFit {
        link,
        categories: opts.labels.clone(),
        counts: prep.counts,
        n,
        thresholds,
        coefficients,
        tests,
        probabilities,
        predicted,
        aliased: prep.aliased,
        iterations,
        converged,
        level: opts.level,
        rows_used: prep.rows_used,
        cov,
        spec: Some(prep.spec),
        kept: prep.kept,
    })
}

impl OrdinalFit {
    /// Category probabilities at new data (rows with missing predictors skipped);
    /// returns the worksheet rows and the probability matrix.
    pub fn predict(&self, data: &Data) -> Result<(Vec<usize>, Matrix)> {
        let spec = self.spec.as_ref().ok_or_else(|| StatsError::InvalidArgument("no model specification".into()))?;
        let dm = spec.encode(data, false)?;
        let rows: Vec<usize> = (0..dm.x.rows()).collect();
        let xn = dm.x.select(&rows, &self.kept);
        let theta: Vec<f64> = self.thresholds.iter().map(|t| t.estimate).collect();
        let beta: Vec<f64> = self.coefficients.iter().map(|c| c.estimate).collect();
        Ok((dm.rows_used, ordinal_probs(self.link, &theta, &beta, &xn)))
    }
}

// ---------------------------------------------------------------------------
// Nominal
// ---------------------------------------------------------------------------

fn nominal_probs(x: &Matrix, beta: &[Vec<f64>], reference: usize, k: usize) -> Matrix {
    let n = x.rows();
    let mut p = Matrix::zeros(n, k);
    for i in 0..n {
        let xi = x.row(i);
        let mut scores = vec![0.0; k];
        let mut b = 0;
        for (c, s) in scores.iter_mut().enumerate() {
            if c == reference {
                continue;
            }
            *s = xi.iter().zip(&beta[b]).map(|(a, w)| a * w).sum();
            b += 1;
        }
        let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = scores.iter().map(|s| (s - mx).exp()).sum();
        for c in 0..k {
            p[(i, c)] = (scores[c] - mx).exp() / sum;
        }
    }
    p
}

/// Fit a nominal logistic regression (baseline-category logits). The
/// response of `formula` holds the 0-based category index.
pub fn nominal_logistic(
    formula: &str,
    data: &Data,
    design: &DesignOptions,
    opts: &CategoricalOptions,
) -> Result<NominalFit> {
    let prep = prepare(formula, data, design, opts, false)?;
    let k = opts.labels.len();
    if opts.reference >= k {
        return invalid("the reference category is out of range");
    }
    let p = prep.x.cols();
    let n = prep.x.rows();
    let x = &prep.x;
    let cat = &prep.cat;
    let reference = opts.reference;
    let nb = k - 1;
    let np = nb * p;
    let others: Vec<usize> = (0..k).filter(|&c| c != reference).collect();
    let mut eval = |v: &[f64]| -> Option<(f64, Vec<f64>, Matrix)> {
        let beta: Vec<Vec<f64>> = (0..nb).map(|b| v[b * p..(b + 1) * p].to_vec()).collect();
        let probs = nominal_probs(x, &beta, reference, k);
        let mut ll = 0.0;
        let mut g = vec![0.0; np];
        let mut h = Matrix::zeros(np, np);
        for i in 0..n {
            let pc = probs[(i, cat[i])];
            if pc.is_nan() || pc <= 1e-300 {
                return None;
            }
            ll += pc.ln();
            let xi = x.row(i);
            for (b, &c) in others.iter().enumerate() {
                let pj = probs[(i, c)];
                let resid = f64::from(u8::from(cat[i] == c)) - pj;
                for a in 0..p {
                    g[b * p + a] += resid * xi[a];
                }
                for (b2, &c2) in others.iter().enumerate() {
                    let w = -pj * (f64::from(u8::from(c == c2)) - probs[(i, c2)]);
                    for a in 0..p {
                        for a2 in 0..p {
                            h[(b * p + a, b2 * p + a2)] += w * xi[a] * xi[a2];
                        }
                    }
                }
            }
        }
        Some((ll, g, h))
    };
    let (v, ll, h, iterations, converged) = newton_max(vec![0.0; np], opts.max_iter, &mut eval)?;
    let cov = Cholesky::new(&h.scale(-1.0))?.inverse()?;
    let beta: Vec<Vec<f64>> = (0..nb).map(|b| v[b * p..(b + 1) * p].to_vec()).collect();
    let has_const = prep.kept.first() == Some(&0) && prep.names.first().is_some_and(|n| n == "Constant");
    let mut logits = Vec::new();
    for (b, &c) in others.iter().enumerate() {
        let rows: Vec<LogitCoef> = (0..p)
            .map(|a| {
                let is_const = has_const && a == 0;
                coef_row(prep.names[a].clone(), beta[b][a], cov[(b * p + a, b * p + a)], opts.level, !is_const)
            })
            .collect();
        logits.push((format!("{}/{}", opts.labels[c], opts.labels[reference]), rows));
    }
    let probabilities = nominal_probs(x, &beta, reference, k);
    let predicted: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = 0;
            for c in 1..k {
                if probabilities[(i, c)] > probabilities[(i, best)] {
                    best = c;
                }
            }
            best
        })
        .collect();
    let ll0 = null_loglik(&prep.counts);
    let g = 2.0 * (ll - ll0);
    let g_df = nb * (p - usize::from(has_const));
    let (pearson, deviance) = pattern_tests(x, cat, &probabilities, k, np)?;
    let tests = CategoricalTests {
        loglik: ll,
        null_loglik: ll0,
        g,
        g_df,
        g_p: if g_df > 0 { ChiSquared::new(g_df as f64)?.sf(g.max(0.0)) } else { f64::NAN },
        pearson,
        deviance,
    };
    Ok(NominalFit {
        categories: opts.labels.clone(),
        reference,
        counts: prep.counts,
        n,
        logits,
        tests,
        probabilities,
        predicted,
        aliased: prep.aliased,
        iterations,
        converged,
        level: opts.level,
        rows_used: prep.rows_used,
        cov,
        spec: Some(prep.spec),
        kept: prep.kept,
    })
}

impl NominalFit {
    /// Category probabilities at new data (rows with missing predictors skipped).
    pub fn predict(&self, data: &Data) -> Result<(Vec<usize>, Matrix)> {
        let spec = self.spec.as_ref().ok_or_else(|| StatsError::InvalidArgument("no model specification".into()))?;
        let dm = spec.encode(data, false)?;
        let rows: Vec<usize> = (0..dm.x.rows()).collect();
        let xn = dm.x.select(&rows, &self.kept);
        let beta: Vec<Vec<f64>> = self.logits.iter().map(|(_, c)| c.iter().map(|r| r.estimate).collect()).collect();
        Ok((dm.rows_used, nominal_probs(&xn, &beta, self.reference, self.categories.len())))
    }
}
