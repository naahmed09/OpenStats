//! Generalized linear models: binary logistic, Poisson, negative-binomial,
//! gamma and Gaussian regression by iteratively reweighted least squares,
//! with the deviance table, goodness-of-fit tests, classification tables,
//! ROC curves and case diagnostics of Minitab's *Binary Logistic Regression*
//! and *Poisson Regression*; ordinal and nominal logistic regression live in
//! [`ordinal`]; term-level stepwise selection in [`stepwise`].

mod diag;
pub mod ordinal;
mod report_impl;
pub mod stepwise;

pub use diag::{
    association, classification_table, hosmer_lemeshow, roc_curve, Association, Classification, HlGroup,
    HosmerLemeshow, Roc,
};

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::model::{build_design, Data, DesignMatrix, DesignOptions, ModelSpec};
use crate::regression::independent_columns;
use openstats_math::dist::{ChiSquared, ContinuousDistribution, FisherF, Normal, StudentT};
use openstats_math::linalg::{Cholesky, Matrix, Qr};
use openstats_math::special::gamma::{digamma, ln_gamma, trigamma};

/// Error distribution of the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Binary (0/1) or events-in-trials response.
    Binomial,
    /// Counts with variance = mean.
    Poisson,
    /// Counts with variance = μ + μ²/θ (θ estimated unless given).
    NegativeBinomial,
    /// Positive continuous response with variance ∝ μ².
    Gamma,
    /// Normal response (variance constant).
    Gaussian,
}

/// Link function g(μ) = η.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// ln(μ/(1−μ)).
    Logit,
    /// Φ⁻¹(μ).
    Probit,
    /// ln(−ln(1−μ)) (complementary log-log).
    Cloglog,
    /// ln μ.
    Log,
    /// μ.
    Identity,
    /// 1/μ.
    Inverse,
}

impl Family {
    /// Canonical (default) link of the family.
    pub fn default_link(self) -> Link {
        match self {
            Family::Binomial => Link::Logit,
            Family::Poisson | Family::NegativeBinomial => Link::Log,
            Family::Gamma => Link::Inverse,
            Family::Gaussian => Link::Identity,
        }
    }

    fn check_link(self, link: Link) -> Result<()> {
        let ok = match self {
            Family::Binomial => matches!(link, Link::Logit | Link::Probit | Link::Cloglog),
            Family::Poisson | Family::NegativeBinomial => matches!(link, Link::Log | Link::Identity),
            Family::Gamma | Family::Gaussian => matches!(link, Link::Log | Link::Identity | Link::Inverse),
        };
        if ok {
            Ok(())
        } else {
            invalid(format!("the {link:?} link is not available for the {self:?} family"))
        }
    }

    pub(crate) fn estimates_dispersion(self) -> bool {
        matches!(self, Family::Gamma | Family::Gaussian)
    }
}

const EPS: f64 = 1e-12;

fn std_normal() -> Normal {
    Normal { mu: 0.0, sigma: 1.0 }
}

impl Link {
    /// η = g(μ).
    pub fn link(self, mu: f64) -> f64 {
        match self {
            Link::Logit => (mu / (1.0 - mu)).ln(),
            Link::Probit => std_normal().quantile(mu),
            Link::Cloglog => (-(1.0 - mu).ln()).ln(),
            Link::Log => mu.ln(),
            Link::Identity => mu,
            Link::Inverse => 1.0 / mu,
        }
    }

    /// μ = g⁻¹(η).
    pub fn inverse(self, eta: f64) -> f64 {
        match self {
            Link::Logit => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            Link::Probit => std_normal().cdf(eta),
            Link::Cloglog => -(-eta.exp()).exp_m1(),
            Link::Log => eta.exp(),
            Link::Identity => eta,
            Link::Inverse => 1.0 / eta,
        }
    }

    /// dμ/dη.
    pub fn d_mu(self, eta: f64) -> f64 {
        match self {
            Link::Logit => {
                let m = self.inverse(eta);
                m * (1.0 - m)
            }
            Link::Probit => std_normal().pdf(eta),
            Link::Cloglog => {
                let e = eta.exp();
                e * (-e).exp()
            }
            Link::Log => eta.exp(),
            Link::Identity => 1.0,
            Link::Inverse => -1.0 / (eta * eta),
        }
    }
}

/// Options of a GLM fit.
#[derive(Debug, Clone)]
pub struct GlmOptions {
    /// Response distribution.
    pub family: Family,
    /// Link function (must suit the family; see [`Family::default_link`]).
    pub link: Link,
    /// Confidence level of the coefficient intervals.
    pub level: f64,
    /// Number of trials per observation for a binomial response given as
    /// event counts. For [`glm`] the vector is indexed by worksheet row; for
    /// [`fit_glm`] by design-matrix row. `None` means a 0/1 response.
    pub trials: Option<Vec<f64>>,
    /// Negative-binomial θ; `None` estimates it by maximum likelihood.
    pub theta: Option<f64>,
    /// Maximum IRLS iterations.
    pub max_iter: usize,
    /// Relative change in deviance that stops the iteration.
    pub tol: f64,
}

impl GlmOptions {
    /// Options with the family's canonical link and defaults elsewhere.
    pub fn new(family: Family) -> GlmOptions {
        GlmOptions {
            family,
            link: family.default_link(),
            level: 0.95,
            trials: None,
            theta: None,
            max_iter: 100,
            tol: 1e-12,
        }
    }

    /// Binary logistic regression.
    pub fn logistic() -> GlmOptions {
        GlmOptions::new(Family::Binomial)
    }

    /// Poisson regression with the log link.
    pub fn poisson() -> GlmOptions {
        GlmOptions::new(Family::Poisson)
    }
}

/// One row of the coefficient table.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmCoefficient {
    /// Term name (`Constant`, `x`, `g_b`, …).
    pub name: String,
    /// Estimate.
    pub estimate: f64,
    /// Standard error.
    pub std_err: f64,
    /// z (fixed dispersion) or t (estimated dispersion) statistic.
    pub stat: f64,
    /// Two-sided p-value of `stat`.
    pub p_value: f64,
    /// Confidence interval of the coefficient.
    pub ci: Interval,
    /// exp(estimate): the odds ratio (logit), rate ratio (log link), …;
    /// `None` for the constant.
    pub ratio: Option<f64>,
    /// Confidence interval of `ratio`.
    pub ratio_ci: Option<Interval>,
}

/// One row of the analysis-of-deviance table.
#[derive(Debug, Clone, PartialEq)]
pub struct DevianceRow {
    /// Source (`Regression`, a term, `Error`, `Total`).
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Adjusted deviance.
    pub deviance: f64,
    /// Deviance per degree of freedom.
    pub mean: Option<f64>,
    /// χ² (or F with estimated dispersion) statistic.
    pub stat: Option<f64>,
    /// p-value of `stat`.
    pub p_value: Option<f64>,
}

/// Overall goodness-of-fit tests.
#[derive(Debug, Clone, PartialEq)]
pub struct GoodnessOfFit {
    /// Deviance χ² test: (deviance, df, p).
    pub deviance: (f64, usize, f64),
    /// Pearson χ² test: (statistic, df, p).
    pub pearson: (f64, usize, f64),
}

/// A fitted generalized linear model.
#[derive(Debug, Clone)]
pub struct GlmFit {
    /// Family fitted.
    pub family: Family,
    /// Link used.
    pub link: Link,
    /// Observations used.
    pub n: usize,
    /// Whether the model has a constant.
    pub intercept: bool,
    /// Coefficient table (aliased columns excluded).
    pub coefficients: Vec<GlmCoefficient>,
    /// Columns dropped as linear combinations of earlier ones.
    pub aliased: Vec<String>,
    /// Indices of the kept design-matrix columns.
    pub kept: Vec<usize>,
    /// Residual deviance.
    pub deviance: f64,
    /// Deviance of the constant-only model.
    pub null_deviance: f64,
    /// Model degrees of freedom (coefficients less the constant).
    pub df_model: usize,
    /// Residual degrees of freedom.
    pub df_error: usize,
    /// Pearson χ².
    pub pearson_chi2: f64,
    /// Dispersion φ (1 for binomial, Poisson and negative binomial).
    pub dispersion: f64,
    /// Log-likelihood of the model (`None` when undefined).
    pub loglik: Option<f64>,
    /// Log-likelihood of the constant-only model.
    pub null_loglik: Option<f64>,
    /// Akaike information criterion.
    pub aic: Option<f64>,
    /// Corrected AIC.
    pub aicc: Option<f64>,
    /// Bayesian information criterion.
    pub bic: Option<f64>,
    /// Analysis of deviance.
    pub deviance_table: Vec<DevianceRow>,
    /// Negative-binomial θ and its standard error.
    pub theta: Option<(f64, f64)>,
    /// Covariance matrix of the kept coefficients.
    pub cov: Matrix,
    /// Confidence level used.
    pub level: f64,
    /// Fitted means (event probabilities for binomial).
    pub fitted: Vec<f64>,
    /// Linear predictor (including any offset).
    pub eta: Vec<f64>,
    /// Response as used (proportion for binomial).
    pub y: Vec<f64>,
    /// Trials per row (1 unless events/trials).
    pub trials: Vec<f64>,
    /// Deviance residuals.
    pub deviance_residuals: Vec<f64>,
    /// Pearson residuals.
    pub pearson_residuals: Vec<f64>,
    /// Standardized deviance residuals (÷ √(φ(1−h))).
    pub std_deviance_residuals: Vec<f64>,
    /// Standardized Pearson residuals.
    pub std_pearson_residuals: Vec<f64>,
    /// Leverages (diagonal of the weighted hat matrix).
    pub leverage: Vec<f64>,
    /// Cook's distance (Pearson-residual based).
    pub cooks_d: Vec<f64>,
    /// One-step change in the Pearson χ² when the case is deleted.
    pub delta_chi2: Vec<f64>,
    /// One-step change in the deviance when the case is deleted.
    pub delta_deviance: Vec<f64>,
    /// One-step change in the coefficients (Pregibon's Δβ).
    pub delta_beta: Vec<f64>,
    /// IRLS iterations used.
    pub iterations: usize,
    /// Whether the iteration converged (false suggests separation).
    pub converged: bool,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    pub(crate) spec: Option<ModelSpec>,
}

/// Prediction at a new observation.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmPrediction {
    /// Worksheet row of the new data.
    pub row: usize,
    /// Linear predictor.
    pub eta: f64,
    /// Standard error of the linear predictor.
    pub se_eta: f64,
    /// Predicted mean (event probability).
    pub fit: f64,
    /// Confidence interval of the mean.
    pub ci: Interval,
}

// ---------------------------------------------------------------------------
// IRLS
// ---------------------------------------------------------------------------

fn ylogy(a: f64, b: f64) -> f64 {
    if a > 0.0 {
        a * (a / b).ln()
    } else {
        0.0
    }
}

fn variance(f: Family, mu: f64, theta: f64) -> f64 {
    match f {
        Family::Binomial => mu * (1.0 - mu),
        Family::Poisson => mu,
        Family::NegativeBinomial => mu + mu * mu / theta,
        Family::Gamma => mu * mu,
        Family::Gaussian => 1.0,
    }
}

fn unit_deviance(f: Family, y: f64, mu: f64, theta: f64) -> f64 {
    match f {
        Family::Binomial => 2.0 * (ylogy(y, mu) + ylogy(1.0 - y, 1.0 - mu)),
        Family::Poisson => 2.0 * (ylogy(y, mu) - (y - mu)),
        Family::NegativeBinomial => 2.0 * (ylogy(y, mu) - (y + theta) * ((y + theta) / (mu + theta)).ln()),
        Family::Gamma => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
        Family::Gaussian => (y - mu) * (y - mu),
    }
}

fn mu_valid(f: Family, mu: f64) -> bool {
    mu.is_finite()
        && match f {
            Family::Binomial => mu > 0.0 && mu < 1.0,
            Family::Poisson | Family::NegativeBinomial | Family::Gamma => mu > 0.0,
            Family::Gaussian => true,
        }
}

struct Irls {
    beta: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    weights: Vec<f64>,
    deviance: f64,
    iterations: usize,
    converged: bool,
}

struct Problem<'a> {
    x: &'a Matrix,
    y: &'a [f64],
    prior: &'a [f64],
    offset: &'a [f64],
    family: Family,
    link: Link,
    theta: f64,
    max_iter: usize,
    tol: f64,
}

fn deviance_of(p: &Problem, mu: &[f64]) -> f64 {
    (0..mu.len()).map(|i| p.prior[i] * unit_deviance(p.family, p.y[i], mu[i], p.theta)).sum()
}

fn start_mu(p: &Problem) -> Result<Vec<f64>> {
    let n = p.y.len();
    let mean = p.y.iter().sum::<f64>() / n as f64;
    let mu: Vec<f64> = (0..n)
        .map(|i| match p.family {
            Family::Binomial => (p.prior[i] * p.y[i] + 0.5) / (p.prior[i] + 1.0),
            Family::Poisson | Family::NegativeBinomial => p.y[i] + 0.1,
            Family::Gamma => p.y[i],
            Family::Gaussian => {
                if p.link == Link::Log || p.link == Link::Inverse {
                    if p.y[i] > 0.0 {
                        p.y[i]
                    } else {
                        mean.abs().max(1e-3)
                    }
                } else {
                    p.y[i]
                }
            }
        })
        .collect();
    if mu.iter().any(|&m| !mu_valid(p.family, m) || !p.link.link(m).is_finite()) {
        return invalid("the response has values outside the range of the chosen family and link");
    }
    Ok(mu)
}

fn irls(p: &Problem, start: Option<&[f64]>) -> Result<Irls> {
    let n = p.y.len();
    let k = p.x.cols();
    let mut beta: Vec<f64>;
    let mut eta: Vec<f64>;
    let mut mu: Vec<f64>;
    let mut have_beta = false;
    match start {
        Some(b) => {
            beta = b.to_vec();
            eta = (0..n).map(|i| p.x.row(i).iter().zip(&beta).map(|(a, c)| a * c).sum::<f64>() + p.offset[i]).collect();
            mu = eta.iter().map(|&e| p.link.inverse(e)).collect();
            have_beta = mu.iter().all(|&m| mu_valid(p.family, m));
            if !have_beta {
                beta = vec![0.0; k];
                mu = start_mu(p)?;
                eta = mu.iter().map(|&m| p.link.link(m)).collect();
            }
        }
        None => {
            beta = vec![0.0; k];
            mu = start_mu(p)?;
            eta = mu.iter().map(|&m| p.link.link(m)).collect();
        }
    }
    let mut dev = deviance_of(p, &mu);
    let mut weights = vec![0.0; n];
    let mut converged = false;
    let mut iterations = 0;
    for it in 1..=p.max_iter {
        iterations = it;
        let mut sx = Matrix::zeros(n, k);
        let mut sz = vec![0.0; n];
        for i in 0..n {
            let dmu = p.link.d_mu(eta[i]);
            let var = variance(p.family, mu[i], p.theta).max(EPS);
            let w = p.prior[i] * dmu * dmu / var;
            weights[i] = w;
            let sw = w.sqrt();
            let z = eta[i] - p.offset[i] + (p.y[i] - mu[i]) / dmu;
            for j in 0..k {
                sx[(i, j)] = sw * p.x[(i, j)];
            }
            sz[i] = sw * z;
        }
        if sx.as_slice().iter().chain(&sz).any(|v| !v.is_finite()) {
            return invalid("the iteration produced non-finite weights (check the response and link)");
        }
        let qr = Qr::new(&sx)?;
        let b_new = qr.solve_least_squares(&sz)?;
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let b_try: Vec<f64> = if have_beta {
                beta.iter().zip(&b_new).map(|(o, nw)| o + step * (nw - o)).collect()
            } else {
                b_new.clone()
            };
            let e_try: Vec<f64> =
                (0..n).map(|i| p.x.row(i).iter().zip(&b_try).map(|(a, c)| a * c).sum::<f64>() + p.offset[i]).collect();
            let m_try: Vec<f64> = e_try.iter().map(|&e| p.link.inverse(e)).collect();
            if m_try.iter().all(|&m| mu_valid(p.family, m)) {
                let d_try = deviance_of(p, &m_try);
                if d_try.is_finite() && (!have_beta || d_try <= dev + 1e-9 * (dev.abs() + 1.0)) {
                    accepted = Some((b_try, e_try, m_try, d_try));
                    break;
                }
            }
            if !have_beta {
                break;
            }
            step *= 0.5;
        }
        let Some((b_try, e_try, m_try, d_try)) = accepted else {
            if have_beta {
                // No further decrease possible: treat as converged at the current point.
                converged = true;
                break;
            }
            return invalid("could not find valid starting values for the fit");
        };
        let rel = (dev - d_try).abs() / (d_try.abs() + 0.1);
        beta = b_try;
        eta = e_try;
        mu = m_try;
        dev = d_try;
        have_beta = true;
        if rel < p.tol {
            converged = true;
            break;
        }
    }
    // Final working weights at the converged point.
    for i in 0..n {
        let dmu = p.link.d_mu(eta[i]);
        let var = variance(p.family, mu[i], p.theta).max(EPS);
        weights[i] = p.prior[i] * dmu * dmu / var;
    }
    Ok(Irls { beta, eta, mu, weights, deviance: dev, iterations, converged })
}

fn xwx_inverse(x: &Matrix, w: &[f64]) -> Result<Matrix> {
    let n = x.rows();
    let k = x.cols();
    let mut sx = Matrix::zeros(n, k);
    for i in 0..n {
        let sw = w[i].max(0.0).sqrt();
        for j in 0..k {
            sx[(i, j)] = sw * x[(i, j)];
        }
    }
    let ch = Cholesky::new(&sx.gram())?;
    Ok(ch.inverse()?)
}

fn loglik(family: Family, y: &[f64], mu: &[f64], prior: &[f64], theta: f64, dev: f64) -> Option<f64> {
    let n = y.len();
    match family {
        Family::Binomial => Some(
            (0..n)
                .map(|i| {
                    let m = prior[i];
                    let k = y[i] * m;
                    ln_gamma(m + 1.0) - ln_gamma(k + 1.0) - ln_gamma(m - k + 1.0)
                        + if k > 0.0 { k * mu[i].ln() } else { 0.0 }
                        + if m - k > 0.0 { (m - k) * (1.0 - mu[i]).ln() } else { 0.0 }
                })
                .sum(),
        ),
        Family::Poisson => Some((0..n).map(|i| y[i] * mu[i].ln() - mu[i] - ln_gamma(y[i] + 1.0)).sum()),
        Family::NegativeBinomial => Some(
            (0..n)
                .map(|i| {
                    ln_gamma(y[i] + theta) - ln_gamma(theta) - ln_gamma(y[i] + 1.0)
                        + theta * (theta / (theta + mu[i])).ln()
                        + if y[i] > 0.0 { y[i] * (mu[i] / (theta + mu[i])).ln() } else { 0.0 }
                })
                .sum(),
        ),
        Family::Gaussian => {
            let s2 = dev / n as f64;
            if s2 <= 0.0 {
                return None;
            }
            Some(-0.5 * n as f64 * ((2.0 * std::f64::consts::PI * s2).ln() + 1.0))
        }
        Family::Gamma => {
            let phi = dev / n as f64;
            if phi <= 0.0 {
                return None;
            }
            let a = 1.0 / phi;
            Some(
                (0..n)
                    .map(|i| prior[i] * ((a - 1.0) * y[i].ln() - y[i] * a / mu[i] - a * (mu[i] / a).ln() - ln_gamma(a)))
                    .sum(),
            )
        }
    }
}

/// Maximum-likelihood θ of the negative binomial given the means.
fn theta_ml(y: &[f64], mu: &[f64], start: f64) -> f64 {
    let mut th = start.max(1e-3);
    for _ in 0..100 {
        let (mut s, mut h) = (0.0, 0.0);
        for (&yi, &mi) in y.iter().zip(mu) {
            s += digamma(yi + th) - digamma(th) + th.ln() + 1.0 - (mi + th).ln() - (yi + th) / (mi + th);
            h += trigamma(yi + th) - trigamma(th) + 1.0 / th - 2.0 / (mi + th) + (yi + th) / ((mi + th) * (mi + th));
        }
        if h >= 0.0 {
            // Non-concave region: move in the gradient direction.
            th *= if s > 0.0 { 2.0 } else { 0.5 };
            continue;
        }
        let mut next = th - s / h;
        if next <= 0.0 {
            next = th / 2.0;
        }
        if (next - th).abs() < 1e-10 * th {
            return next;
        }
        th = next.min(1e8);
    }
    th
}

fn theta_se(y: &[f64], mu: &[f64], th: f64) -> f64 {
    let h: f64 = y
        .iter()
        .zip(mu)
        .map(|(&yi, &mi)| {
            trigamma(yi + th) - trigamma(th) + 1.0 / th - 2.0 / (mi + th) + (yi + th) / ((mi + th) * (mi + th))
        })
        .sum();
    if h < 0.0 {
        (-1.0 / h).sqrt()
    } else {
        f64::NAN
    }
}

struct Core {
    irls: Irls,
    theta: Option<f64>,
}

fn fit_core(p: &mut Problem, theta_fixed: Option<f64>, start: Option<&[f64]>) -> Result<Core> {
    if p.family != Family::NegativeBinomial {
        return Ok(Core { irls: irls(p, start)?, theta: None });
    }
    if let Some(t) = theta_fixed {
        p.theta = t;
        return Ok(Core { irls: irls(p, start)?, theta: Some(t) });
    }
    // Alternate IRLS (θ fixed) and the ML update of θ, starting from the Poisson fit.
    p.theta = 1e6;
    let mut cur = irls(p, start)?;
    let mut th = theta_ml(p.y, &cur.mu, 1.0);
    let mut ll_old = f64::NEG_INFINITY;
    for _ in 0..50 {
        p.theta = th;
        cur = irls(p, Some(&cur.beta))?;
        th = theta_ml(p.y, &cur.mu, th);
        p.theta = th;
        let ll = loglik(Family::NegativeBinomial, p.y, &cur.mu, p.prior, th, cur.deviance).unwrap_or(f64::NAN);
        if (ll - ll_old).abs() < 1e-10 * (ll.abs() + 0.1) {
            break;
        }
        ll_old = ll;
    }
    cur = irls(p, Some(&cur.beta))?;
    Ok(Core { irls: cur, theta: Some(th) })
}

fn check_response(opts: &GlmOptions, y: &[f64], trials: &[f64]) -> Result<()> {
    match opts.family {
        Family::Binomial => {
            if opts.trials.is_none() {
                if y.iter().any(|&v| v != 0.0 && v != 1.0) {
                    return invalid("a binary response must be coded 0/1 (or give events and trials)");
                }
            } else {
                if trials.iter().any(|&t| t < 1.0 || t.fract() != 0.0) {
                    return invalid("trials must be positive whole numbers");
                }
                if y.iter().zip(trials).any(|(&k, &t)| k < 0.0 || k > t || k.fract() != 0.0) {
                    return invalid("events must be whole numbers between 0 and the number of trials");
                }
            }
        }
        Family::Poisson | Family::NegativeBinomial => {
            if y.iter().any(|&v| v < 0.0 || v.fract() != 0.0) {
                return invalid("a count response must be non-negative whole numbers");
            }
        }
        Family::Gamma => {
            if y.iter().any(|&v| v <= 0.0) {
                return invalid("a gamma response must be positive");
            }
        }
        Family::Gaussian => {}
    }
    Ok(())
}

/// Fit a prepared design matrix (`dm.y` must be present). For a binomial
/// response with `opts.trials`, `dm.y` holds the event counts and `trials`
/// is indexed like the design-matrix rows.
pub fn fit_glm(dm: &DesignMatrix, opts: &GlmOptions) -> Result<GlmFit> {
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
    }
    opts.family.check_link(opts.link)?;
    let y_raw = dm.y.as_ref().ok_or_else(|| StatsError::InvalidArgument("the design matrix has no response".into()))?;
    let n = dm.x.rows();
    if y_raw.len() != n || dm.offset.len() != n {
        return invalid("response, offset and design matrix disagree in length");
    }
    if dm.x.as_slice().iter().chain(y_raw).any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite);
    }
    let trials: Vec<f64> = match &opts.trials {
        Some(t) => {
            if t.len() != n {
                return invalid("one trials value per observation is required");
            }
            t.clone()
        }
        None => vec![1.0; n],
    };
    check_response(opts, y_raw, &trials)?;
    let (y, prior): (Vec<f64>, Vec<f64>) = if opts.family == Family::Binomial {
        (y_raw.iter().zip(&trials).map(|(k, m)| k / m).collect(), trials.clone())
    } else {
        (y_raw.clone(), vec![1.0; n])
    };
    let (kept, dropped) = independent_columns(&dm.x, 1e-10);
    let p = kept.len();
    if p == 0 {
        return invalid("the design matrix has no usable columns");
    }
    if n <= p {
        return Err(StatsError::InsufficientData { what: "a generalized linear model", needed: p + 1, got: n });
    }
    let rows: Vec<usize> = (0..n).collect();
    let xk = dm.x.select(&rows, &kept);
    let has_const = dm.intercept && kept.first() == Some(&0);

    let mut prob = Problem {
        x: &xk,
        y: &y,
        prior: &prior,
        offset: &dm.offset,
        family: opts.family,
        link: opts.link,
        theta: opts.theta.unwrap_or(1.0),
        max_iter: opts.max_iter,
        tol: opts.tol,
    };
    if let Some(t) = opts.theta {
        if t <= 0.0 || !t.is_finite() {
            return invalid("theta must be positive");
        }
    }
    let core = fit_core(&mut prob, opts.theta, None)?;
    let fitted_irls = core.irls;
    let theta = core.theta.unwrap_or(1.0);
    prob.theta = theta;
    let mu = fitted_irls.mu.clone();
    let dev = fitted_irls.deviance;
    let df_error = n - p;
    let df_model = p - usize::from(has_const);

    // Null model.
    let (null_dev, null_mu) = {
        if has_const {
            let one = Matrix::filled(n, 1, 1.0);
            let mut np = Problem { x: &one, ..clone_problem(&prob) };
            let c = fit_core(&mut np, if opts.family == Family::NegativeBinomial { Some(theta) } else { None }, None)?;
            (c.irls.deviance, c.irls.mu)
        } else {
            let m: Vec<f64> = dm.offset.iter().map(|&o| opts.link.inverse(o)).collect();
            if m.iter().any(|&v| !mu_valid(opts.family, v)) {
                (f64::NAN, vec![f64::NAN; n])
            } else {
                (deviance_of(&prob, &m), m)
            }
        }
    };

    let pearson_terms: Vec<f64> = (0..n)
        .map(|i| {
            let v = variance(opts.family, mu[i], theta);
            prior[i] * (y[i] - mu[i]).powi(2) / v
        })
        .collect();
    let pearson_chi2: f64 = pearson_terms.iter().sum();
    let dispersion = if opts.family.estimates_dispersion() { pearson_chi2 / df_error as f64 } else { 1.0 };

    let inv = xwx_inverse(&xk, &fitted_irls.weights)?;
    let cov = inv.scale(dispersion);
    let b = &fitted_irls.beta;
    let est_disp = opts.family.estimates_dispersion();
    let tdist = StudentT::new(df_error as f64)?;
    let norm = std_normal();
    let crit = if est_disp { tdist.isf((1.0 - opts.level) / 2.0) } else { norm.isf((1.0 - opts.level) / 2.0) };
    let coefficients: Vec<GlmCoefficient> = (0..p)
        .map(|j| {
            let se = cov[(j, j)].max(0.0).sqrt();
            let stat = b[j] / se;
            let pv = if est_disp { 2.0 * tdist.sf(stat.abs()) } else { 2.0 * norm.sf(stat.abs()) };
            let name = dm.names[kept[j]].clone();
            let is_const = has_const && j == 0;
            let (lo, hi) = (b[j] - crit * se, b[j] + crit * se);
            GlmCoefficient {
                name,
                estimate: b[j],
                std_err: se,
                stat,
                p_value: pv.min(1.0),
                ci: Interval { lower: lo, upper: hi, level: opts.level },
                ratio: (!is_const).then(|| b[j].exp()),
                ratio_ci: (!is_const).then(|| Interval { lower: lo.exp(), upper: hi.exp(), level: opts.level }),
            }
        })
        .collect();

    // Case statistics.
    let pf = p as f64;
    let mut h = vec![0.0; n];
    for i in 0..n {
        let x0 = xk.row(i);
        let v = inv.matvec(x0)?;
        h[i] = fitted_irls.weights[i] * x0.iter().zip(&v).map(|(a, c)| a * c).sum::<f64>();
    }
    let mut dres = Vec::with_capacity(n);
    let mut pres = Vec::with_capacity(n);
    let mut sdres = Vec::with_capacity(n);
    let mut spres = Vec::with_capacity(n);
    let mut cooks = Vec::with_capacity(n);
    let mut dchi = Vec::with_capacity(n);
    let mut ddev = Vec::with_capacity(n);
    let mut dbeta = Vec::with_capacity(n);
    for i in 0..n {
        let v = variance(opts.family, mu[i], theta);
        let r_p = (y[i] - mu[i]) * prior[i].sqrt() / v.sqrt();
        let d_i = prior[i] * unit_deviance(opts.family, y[i], mu[i], theta);
        let r_d = (y[i] - mu[i]).signum() * d_i.max(0.0).sqrt();
        let om = (1.0 - h[i]).max(1e-12);
        pres.push(r_p);
        dres.push(r_d);
        sdres.push(r_d / (dispersion * om).sqrt());
        spres.push(r_p / (dispersion * om).sqrt());
        cooks.push(r_p * r_p * h[i] / (dispersion * pf * om * om));
        dchi.push(r_p * r_p / om);
        ddev.push(d_i + r_p * r_p * h[i] / om);
        dbeta.push(r_p * r_p * h[i] / (om * om));
    }

    // Likelihood and information criteria.
    let ll = loglik(opts.family, &y, &mu, &prior, theta, dev);
    let null_ll = if null_dev.is_finite() { loglik(opts.family, &y, &null_mu, &prior, theta, null_dev) } else { None };
    let extra = usize::from(opts.family == Family::NegativeBinomial && opts.theta.is_none())
        + usize::from(opts.family.estimates_dispersion());
    let kpar = (p + extra) as f64;
    let nf = n as f64;
    let aic = ll.map(|l| -2.0 * l + 2.0 * kpar);
    let aicc = aic.and_then(|a| (nf - kpar - 1.0 > 0.0).then(|| a + 2.0 * kpar * (kpar + 1.0) / (nf - kpar - 1.0)));
    let bic = ll.map(|l| -2.0 * l + kpar * nf.ln());

    // Analysis of deviance (Type III likelihood-ratio tests by term).
    let mut terms: Vec<(String, Vec<usize>)> = Vec::new();
    for t in &dm.terms {
        let pos: Vec<usize> = (0..p).filter(|&k| kept[k] >= t.start && kept[k] < t.start + t.len).collect();
        if !pos.is_empty() {
            terms.push((t.label.clone(), pos));
        }
    }
    let theta_for_sub = if opts.family == Family::NegativeBinomial { Some(theta) } else { None };
    let chisq_p = |stat: f64, df: usize| -> Result<f64> { Ok(ChiSquared::new(df as f64)?.sf(stat)) };
    let test = |delta: f64, df: usize| -> Result<(f64, f64)> {
        if est_disp {
            let f = (delta / df as f64) / dispersion;
            Ok((f, FisherF::new(df as f64, df_error as f64)?.sf(f)))
        } else {
            Ok((delta, chisq_p(delta, df)?))
        }
    };
    let mut deviance_table = Vec::new();
    if df_model > 0 && null_dev.is_finite() {
        let delta = (null_dev - dev).max(0.0);
        let (st, pv) = test(delta, df_model)?;
        deviance_table.push(DevianceRow {
            source: "Regression".into(),
            df: df_model,
            deviance: delta,
            mean: Some(delta / df_model as f64),
            stat: Some(st),
            p_value: Some(pv),
        });
        for (label, pos) in &terms {
            let rest: Vec<usize> = (0..p).filter(|k| !pos.contains(k)).collect();
            let xr = xk.select(&rows, &rest);
            let mut sp = Problem { x: &xr, ..clone_problem(&prob) };
            let sub = fit_core(&mut sp, theta_for_sub, None)?;
            let d = (sub.irls.deviance - dev).max(0.0);
            let (st, pv) = test(d, pos.len())?;
            deviance_table.push(DevianceRow {
                source: label.clone(),
                df: pos.len(),
                deviance: d,
                mean: Some(d / pos.len() as f64),
                stat: Some(st),
                p_value: Some(pv),
            });
        }
    }
    deviance_table.push(DevianceRow {
        source: "Error".into(),
        df: df_error,
        deviance: dev,
        mean: Some(dev / df_error as f64),
        stat: None,
        p_value: None,
    });
    if null_dev.is_finite() {
        deviance_table.push(DevianceRow {
            source: "Total".into(),
            df: n - usize::from(has_const),
            deviance: null_dev,
            mean: None,
            stat: None,
            p_value: None,
        });
    }

    let theta_out = core.theta.map(|t| (t, if opts.theta.is_some() { f64::NAN } else { theta_se(&y, &mu, t) }));
    Ok(GlmFit {
        family: opts.family,
        link: opts.link,
        n,
        intercept: has_const,
        coefficients,
        aliased: dropped.iter().map(|&j| dm.names[j].clone()).collect(),
        kept,
        deviance: dev,
        null_deviance: null_dev,
        df_model,
        df_error,
        pearson_chi2,
        dispersion,
        loglik: ll,
        null_loglik: null_ll,
        aic,
        aicc,
        bic,
        deviance_table,
        theta: theta_out,
        cov,
        level: opts.level,
        fitted: mu,
        eta: fitted_irls.eta.clone(),
        y,
        trials,
        deviance_residuals: dres,
        pearson_residuals: pres,
        std_deviance_residuals: sdres,
        std_pearson_residuals: spres,
        leverage: h,
        cooks_d: cooks,
        delta_chi2: dchi,
        delta_deviance: ddev,
        delta_beta: dbeta,
        iterations: fitted_irls.iterations,
        converged: fitted_irls.converged,
        rows_used: dm.rows_used.clone(),
        spec: None,
    })
}

fn clone_problem<'a>(p: &Problem<'a>) -> Problem<'a> {
    Problem {
        x: p.x,
        y: p.y,
        prior: p.prior,
        offset: p.offset,
        family: p.family,
        link: p.link,
        theta: p.theta,
        max_iter: p.max_iter,
        tol: p.tol,
    }
}

/// Parse `formula`, build the design from `data` and fit the GLM. For a
/// binomial response with events/trials, `opts.trials` is indexed by worksheet row.
pub fn glm(formula: &str, data: &Data, design: &DesignOptions, opts: &GlmOptions) -> Result<GlmFit> {
    let (spec, dm) = build_design(formula, data, design)?;
    let mut o = opts.clone();
    if let Some(t) = &opts.trials {
        if t.len() != data.n_rows() {
            return invalid("one trials value per worksheet row is required");
        }
        o.trials = Some(dm.rows_used.iter().map(|&i| t[i]).collect());
    }
    let mut fit = fit_glm(&dm, &o)?;
    fit.spec = Some(spec);
    Ok(fit)
}

impl GlmFit {
    /// Number of coefficients in the model.
    pub fn n_coef(&self) -> usize {
        self.coefficients.len()
    }

    /// Coefficient estimates in table order.
    pub fn estimates(&self) -> Vec<f64> {
        self.coefficients.iter().map(|c| c.estimate).collect()
    }

    /// Worksheet rows used.
    pub fn rows_used(&self) -> &[usize] {
        &self.rows_used
    }

    /// Deviance R² = 1 − deviance / null deviance.
    pub fn deviance_r2(&self) -> f64 {
        1.0 - self.deviance / self.null_deviance
    }

    /// Adjusted deviance R².
    pub fn deviance_r2_adj(&self) -> f64 {
        1.0 - (self.deviance / self.df_error as f64)
            / (self.null_deviance / (self.n - usize::from(self.intercept)) as f64)
    }

    /// Deviance and Pearson goodness-of-fit tests.
    pub fn goodness_of_fit(&self) -> Result<GoodnessOfFit> {
        let chi = ChiSquared::new(self.df_error as f64)?;
        Ok(GoodnessOfFit {
            deviance: (self.deviance, self.df_error, chi.sf(self.deviance)),
            pearson: (self.pearson_chi2, self.df_error, chi.sf(self.pearson_chi2)),
        })
    }

    /// Events per row (binomial) as counts.
    pub fn events(&self) -> Vec<f64> {
        self.y.iter().zip(&self.trials).map(|(p, m)| p * m).collect()
    }

    /// Predict at new data (a model fitted with [`glm`] is required); rows
    /// with missing predictors are skipped. The interval is the inverse-link
    /// image of the interval for the linear predictor.
    pub fn predict(&self, data: &Data, level: f64) -> Result<Vec<GlmPrediction>> {
        if !(level > 0.0 && level < 1.0) {
            return invalid(format!("confidence level {level} must be in (0, 1)"));
        }
        let spec = self.spec.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("prediction needs a model fitted from a formula (use `glm`)".into())
        })?;
        let dm = spec.encode(data, false)?;
        let rows: Vec<usize> = (0..dm.x.rows()).collect();
        let xn = dm.x.select(&rows, &self.kept);
        let b = self.estimates();
        let crit = if self.family.estimates_dispersion() {
            StudentT::new(self.df_error as f64)?.isf((1.0 - level) / 2.0)
        } else {
            std_normal().isf((1.0 - level) / 2.0)
        };
        let mut out = Vec::with_capacity(xn.rows());
        for i in 0..xn.rows() {
            let x0 = xn.row(i);
            let eta: f64 = x0.iter().zip(&b).map(|(a, c)| a * c).sum::<f64>() + dm.offset[i];
            let v = self.cov.matvec(x0)?;
            let se: f64 = x0.iter().zip(&v).map(|(a, c)| a * c).sum::<f64>().max(0.0).sqrt();
            let (lo, hi) = (self.link.inverse(eta - crit * se), self.link.inverse(eta + crit * se));
            out.push(GlmPrediction {
                row: dm.rows_used[i],
                eta,
                se_eta: se,
                fit: self.link.inverse(eta),
                ci: Interval { lower: lo.min(hi), upper: lo.max(hi), level },
            });
        }
        Ok(out)
    }
}
