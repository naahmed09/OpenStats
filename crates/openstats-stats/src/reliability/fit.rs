//! Maximum-likelihood fits of location–scale life distributions to censored data.

use super::data::{LifeData, Obs, ObsKind};
use super::dist::{Family, FAMILIES};
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ContinuousDistribution, Normal};
use openstats_math::linalg::Cholesky;
use openstats_math::optim::{bfgs, hessian_fd, nelder_mead, NelderMeadOptions, Options};
use openstats_math::special::gamma::gamma;
use openstats_math::Matrix;

/// Log-likelihood contribution of one observation (weight not applied).
pub fn obs_loglik(f: Family, o: &Obs, mu: f64, sigma: f64) -> f64 {
    match o.kind() {
        ObsKind::Exact => {
            let z = (f.g(o.lower) - mu) / sigma;
            f.ln_pdf0(z) - sigma.ln() - if f.is_log() { o.lower.ln() } else { 0.0 }
        }
        ObsKind::Right => f.ln_sf0((f.g(o.lower) - mu) / sigma),
        ObsKind::Left => f.ln_cdf0((f.g(o.upper) - mu) / sigma),
        ObsKind::Interval => f.ln_interval((f.g(o.lower) - mu) / sigma, (f.g(o.upper) - mu) / sigma),
    }
}

/// Check that the data suit the family and contain a failure.
pub(crate) fn validate(f: Family, obs: &[Obs]) -> Result<()> {
    if f.is_log() {
        if let Some(o) = obs.iter().find(|o| o.kind() == ObsKind::Exact && o.lower <= 0.0) {
            return invalid(format!("row {}: this distribution needs positive failure times", o.row + 1));
        }
    }
    if obs.iter().all(|o| o.kind() == ObsKind::Right) {
        return invalid("every observation is censored: at least one failure is needed to estimate the distribution");
    }
    Ok(())
}

/// Raw result of a location–scale regression fit.
#[derive(Debug, Clone)]
pub(crate) struct Core {
    pub beta: Vec<f64>,
    pub ln_sigma: f64,
    pub loglik: f64,
    /// Covariance of `(beta…, ln σ)` (without `ln σ` when the scale is fixed).
    pub cov: Option<Matrix>,
    pub converged: bool,
}

fn midpoint(f: Family, o: &Obs) -> f64 {
    match o.kind() {
        ObsKind::Exact | ObsKind::Right => f.g(o.lower),
        ObsKind::Left => f.g(o.upper),
        ObsKind::Interval => 0.5 * (f.g(o.lower) + f.g(o.upper)),
    }
}

/// Fit `μ_i = x_i·β` with a common scale by maximum likelihood.
///
/// The non-constant columns are centred and scaled for the optimisation and the Hessian (which
/// keeps strongly collinear designs, such as an Arrhenius stress, well conditioned); the results
/// are mapped back to the original columns.
pub(crate) fn fit_core(f: Family, obs: &[Obs], x: &[Vec<f64>]) -> Result<Core> {
    validate(f, obs)?;
    let k = x[0].len();
    let wsum: f64 = obs.iter().map(|o| o.weight).sum();
    let mut shift = vec![0.0; k];
    let mut scale = vec![1.0; k];
    for j in 1..k {
        let m = obs.iter().zip(x).map(|(o, r)| o.weight * r[j]).sum::<f64>() / wsum;
        let v = obs.iter().zip(x).map(|(o, r)| o.weight * (r[j] - m).powi(2)).sum::<f64>() / wsum;
        if v.sqrt().is_finite() && v.sqrt() > 0.0 {
            shift[j] = m;
            scale[j] = v.sqrt();
        }
    }
    let xs: Vec<Vec<f64>> =
        x.iter().map(|r| r.iter().enumerate().map(|(j, v)| (v - shift[j]) / scale[j]).collect()).collect();
    let c = fit_core_scaled(f, obs, &xs)?;
    let mut beta = vec![0.0; k];
    beta[0] = c.beta[0];
    for j in 1..k {
        beta[j] = c.beta[j] / scale[j];
        beta[0] -= beta[j] * shift[j];
    }
    let np = c.cov.as_ref().map_or(0, |m| m.rows());
    let cov = c.cov.as_ref().map(|cs| {
        let mut a = Matrix::zeros(np, np);
        for i in 0..np {
            a[(i, i)] = if i >= 1 && i < k { 1.0 / scale[i] } else { 1.0 };
        }
        for j in 1..k {
            a[(0, j)] = -shift[j] / scale[j];
        }
        let mut out = Matrix::zeros(np, np);
        for i in 0..np {
            for jj in 0..np {
                let mut v = 0.0;
                for p in 0..np {
                    for q in 0..np {
                        v += a[(i, p)] * cs[(p, q)] * a[(jj, q)];
                    }
                }
                out[(i, jj)] = v;
            }
        }
        out
    });
    Ok(Core { beta, ln_sigma: c.ln_sigma, loglik: c.loglik, cov, converged: c.converged })
}

fn fit_core_scaled(f: Family, obs: &[Obs], x: &[Vec<f64>]) -> Result<Core> {
    let k = x[0].len();
    let fixed = f.fixed_scale();
    let np = k + usize::from(!fixed);
    let negll = |th: &[f64]| -> f64 {
        let sigma = if fixed { 1.0 } else { th[k].exp() };
        if !sigma.is_finite() || sigma <= 0.0 {
            return 1e100;
        }
        let mut s = 0.0;
        for (o, xi) in obs.iter().zip(x) {
            let mu: f64 = xi.iter().zip(th).map(|(a, b)| a * b).sum();
            s -= o.weight * obs_loglik(f, o, mu, sigma);
        }
        if s.is_finite() {
            s
        } else {
            1e100
        }
    };

    // Least-squares start on the interval mid-points.
    let y: Vec<f64> = obs.iter().map(|o| midpoint(f, o)).collect();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for ((o, xi), yi) in obs.iter().zip(x).zip(&y) {
        for a in 0..k {
            xty[a] += o.weight * xi[a] * yi;
            for b in 0..k {
                xtx[(a, b)] += o.weight * xi[a] * xi[b];
            }
        }
    }
    for a in 0..k {
        xtx[(a, a)] *= 1.0 + 1e-10;
    }
    let beta0 = Cholesky::new(&xtx).and_then(|c| c.solve(&xty)).unwrap_or_else(|_| {
        let mut b = vec![0.0; k];
        b[0] = y.iter().sum::<f64>() / y.len() as f64;
        b
    });
    let (mut rss, mut wsum) = (0.0, 0.0);
    for ((o, xi), yi) in obs.iter().zip(x).zip(&y) {
        let r = yi - xi.iter().zip(&beta0).map(|(a, b)| a * b).sum::<f64>();
        rss += o.weight * r * r;
        wsum += o.weight;
    }
    let s0 = (rss / wsum).sqrt();
    let s0 = if s0.is_finite() && s0 > 1e-8 { s0 } else { 1.0 };

    let mut starts: Vec<Vec<f64>> = Vec::new();
    for (shift, lns) in [(0.0, (0.8 * s0).ln()), (1.0, (0.8 * s0).ln()), (0.0, s0.ln() - 1.0), (2.0, s0.ln() + 0.5)] {
        let mut th = beta0.clone();
        th[0] += shift * s0;
        if !fixed {
            th.push(lns);
        }
        starts.push(th);
    }
    let mut best: Option<(Vec<f64>, f64, bool)> = None;
    let opts = Options { max_iter: 500, ..Options::default() };
    for st in &starts {
        let mut f_ = negll;
        let mut cand = bfgs(&mut f_, None, st, &opts).ok().map(|m| (m.x, m.f, m.converged));
        if cand.as_ref().is_none_or(|c| !c.1.is_finite() || c.1 >= 1e99) {
            cand = None;
        }
        let (x0, f0, conv) = cand.unwrap_or_else(|| (st.clone(), negll(st), false));
        let mut f2 = negll;
        let nm = NelderMeadOptions { max_iter: 4000 * np, ..NelderMeadOptions::default() };
        let (xb, fb, cb) = match nelder_mead(&mut f2, &x0, &nm) {
            Ok(m) if m.f <= f0 => (m.x, m.f, conv || m.converged),
            _ => (x0, f0, conv),
        };
        if best.as_ref().is_none_or(|b| fb < b.1 - 1e-12) {
            best = Some((xb, fb, cb));
        }
    }
    let (th, fv, converged) =
        best.ok_or_else(|| StatsError::InvalidArgument("the likelihood could not be maximised".into()))?;
    if fv >= 1e99 {
        return invalid("the likelihood could not be evaluated at any starting point");
    }
    let mut f3 = negll;
    let hess = hessian_fd(&mut f3, &th);
    let cov = Cholesky::new(&hess)
        .and_then(|c| c.inverse())
        .ok()
        .filter(|m| (0..np).all(|i| m[(i, i)].is_finite() && m[(i, i)] > 0.0));
    Ok(Core { beta: th[..k].to_vec(), ln_sigma: if fixed { 0.0 } else { th[k] }, loglik: -fv, cov, converged })
}

/// Estimate with standard error and confidence limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Point estimate.
    pub est: f64,
    /// Standard error (`NaN` when not available).
    pub se: f64,
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
}

pub(crate) fn zcrit(level: f64) -> f64 {
    Normal::standard().isf((1.0 - level) / 2.0)
}

/// A fitted life distribution.
#[derive(Debug, Clone)]
pub struct LifeFit {
    /// Family.
    pub family: Family,
    /// Location μ (for the Weibull and lognormal families, the location of ln t).
    pub mu: f64,
    /// Scale σ.
    pub sigma: f64,
    /// Log-likelihood at the maximum.
    pub loglik: f64,
    /// Units in the analysis.
    pub n: f64,
    /// Units with an observed failure (exact, left or interval).
    pub failures: f64,
    /// Right-censored units.
    pub censored: f64,
    /// Covariance matrix of `(μ, ln σ)` (`ln σ` absent for the exponential); `None` when the
    /// likelihood surface is flat.
    pub cov: Option<Matrix>,
    /// `true` when the optimiser met its convergence test.
    pub converged: bool,
}

impl LifeFit {
    /// Parameters estimated.
    pub fn n_params(&self) -> usize {
        if self.family.fixed_scale() {
            1
        } else {
            2
        }
    }

    /// Akaike information criterion.
    pub fn aic(&self) -> f64 {
        -2.0 * self.loglik + 2.0 * self.n_params() as f64
    }

    /// Bayesian information criterion.
    pub fn bic(&self) -> f64 {
        -2.0 * self.loglik + self.n_params() as f64 * self.n.ln()
    }

    fn c(&self, i: usize, j: usize) -> f64 {
        match &self.cov {
            Some(m) if i < m.rows() && j < m.rows() => m[(i, j)],
            Some(_) => 0.0,
            None => f64::NAN,
        }
    }

    /// Standardised time.
    pub fn z(&self, t: f64) -> f64 {
        (self.family.g(t) - self.mu) / self.sigma
    }

    /// Reliability (survival) function.
    pub fn survival(&self, t: f64) -> f64 {
        if t <= 0.0 && self.family.is_log() {
            return 1.0;
        }
        self.family.sf0(self.z(t))
    }

    /// Failure probability.
    pub fn cdf(&self, t: f64) -> f64 {
        1.0 - self.survival(t)
    }

    /// Density.
    pub fn pdf(&self, t: f64) -> f64 {
        if t <= 0.0 && self.family.is_log() {
            return 0.0;
        }
        let j = if self.family.is_log() { t } else { 1.0 };
        self.family.pdf0(self.z(t)) / (self.sigma * j)
    }

    /// Hazard rate.
    pub fn hazard(&self, t: f64) -> f64 {
        if t <= 0.0 && self.family.is_log() {
            return 0.0;
        }
        let j = if self.family.is_log() { t } else { 1.0 };
        (self.family.ln_pdf0(self.z(t)) - self.family.ln_sf0(self.z(t))).exp() / (self.sigma * j)
    }

    /// Time by which a fraction `p` has failed.
    pub fn percentile(&self, p: f64) -> f64 {
        self.family.g_inv(self.mu + self.sigma * self.family.ppf0(p))
    }

    /// Variance of `μ + σ z`.
    fn var_x(&self, zp: f64) -> f64 {
        self.c(0, 0) + zp * zp * self.sigma * self.sigma * self.c(1, 1).max(0.0) + 2.0 * zp * self.sigma * self.c(0, 1)
    }

    /// Percentile with a normal-approximation interval (on `ln t` or `t`).
    pub fn percentile_ci(&self, p: f64, level: f64) -> Estimate {
        let zp = self.family.ppf0(p);
        let x = self.mu + self.sigma * zp;
        let se_x = self.var_x(zp).sqrt();
        let k = zcrit(level);
        let est = self.family.g_inv(x);
        let se = if self.family.is_log() { est * se_x } else { se_x };
        Estimate { est, se, lower: self.family.g_inv(x - k * se_x), upper: self.family.g_inv(x + k * se_x) }
    }

    /// Reliability at time `t` with an interval from the standardised time.
    pub fn reliability_ci(&self, t: f64, level: f64) -> Estimate {
        let z = self.z(t);
        let var_z = self.c(0, 0) / (self.sigma * self.sigma)
            + z * z * self.c(1, 1).max(0.0)
            + 2.0 * z * self.c(0, 1) / self.sigma;
        let se_z = var_z.sqrt();
        let k = zcrit(level);
        Estimate {
            est: self.family.sf0(z),
            se: self.family.pdf0(z) * se_z,
            lower: self.family.sf0(z + k * se_z),
            upper: self.family.sf0(z - k * se_z),
        }
    }

    fn mean_at(family: Family, mu: f64, sigma: f64) -> f64 {
        match family {
            Family::Weibull => mu.exp() * gamma(1.0 + sigma),
            Family::Lognormal => (mu + 0.5 * sigma * sigma).exp(),
            Family::Exponential => mu.exp(),
            Family::Normal | Family::Logistic => mu,
            Family::Loglogistic => {
                if sigma < 1.0 {
                    let a = std::f64::consts::PI * sigma;
                    mu.exp() * a / a.sin()
                } else {
                    f64::NAN
                }
            }
        }
    }

    /// Mean life (MTTF); `NaN` when it does not exist (log-logistic with σ ≥ 1).
    pub fn mean(&self) -> f64 {
        Self::mean_at(self.family, self.mu, self.sigma)
    }

    /// Standard deviation of life; `NaN` when it does not exist.
    pub fn std_dev(&self) -> f64 {
        let (mu, s) = (self.mu, self.sigma);
        match self.family {
            Family::Weibull => mu.exp() * (gamma(1.0 + 2.0 * s) - gamma(1.0 + s).powi(2)).sqrt(),
            Family::Lognormal => ((s * s).exp_m1() * (2.0 * mu + s * s).exp()).sqrt(),
            Family::Exponential => mu.exp(),
            Family::Normal => s,
            Family::Logistic => s * std::f64::consts::PI / 3f64.sqrt(),
            Family::Loglogistic => {
                if s < 0.5 {
                    let a = std::f64::consts::PI * s;
                    let b = 2.0 * a;
                    mu.exp() * (b / b.sin() - (a / a.sin()).powi(2)).sqrt()
                } else {
                    f64::NAN
                }
            }
        }
    }

    /// Mean life with an interval (delta method on `ln` mean for positive families).
    pub fn mean_ci(&self, level: f64) -> Estimate {
        let est = self.mean();
        let fam = self.family;
        let positive = fam.is_log();
        let h = |mu: f64, ls: f64| {
            let m = Self::mean_at(fam, mu, ls.exp());
            if positive {
                m.ln()
            } else {
                m
            }
        };
        let ls = self.sigma.ln();
        let (dm, dl) = (1e-6 * self.mu.abs().max(1.0), 1e-6);
        let g0 = (h(self.mu + dm, ls) - h(self.mu - dm, ls)) / (2.0 * dm);
        let g1 = if fam.fixed_scale() { 0.0 } else { (h(self.mu, ls + dl) - h(self.mu, ls - dl)) / (2.0 * dl) };
        let var = g0 * g0 * self.c(0, 0) + g1 * g1 * self.c(1, 1).max(0.0) + 2.0 * g0 * g1 * self.c(0, 1);
        let se_h = var.sqrt();
        let k = zcrit(level);
        if positive {
            Estimate { est, se: est * se_h, lower: est * (-k * se_h).exp(), upper: est * (k * se_h).exp() }
        } else {
            Estimate { est, se: se_h, lower: est - k * se_h, upper: est + k * se_h }
        }
    }

    /// Parameter table in Minitab's parameterisation: (name, estimate with interval).
    pub fn parameters(&self, level: f64) -> Vec<(String, Estimate)> {
        let k = zcrit(level);
        let loc = |name: &str, exp: bool| {
            let se = self.c(0, 0).sqrt();
            let est = if exp { self.mu.exp() } else { self.mu };
            let (lo, hi) = (self.mu - k * se, self.mu + k * se);
            if exp {
                (name.to_string(), Estimate { est, se: est * se, lower: lo.exp(), upper: hi.exp() })
            } else {
                (name.to_string(), Estimate { est, se, lower: lo, upper: hi })
            }
        };
        let scale = |name: &str, invert: bool| {
            let se_l = self.c(1, 1).sqrt();
            let (lo, hi) = (self.sigma.ln() - k * se_l, self.sigma.ln() + k * se_l);
            if invert {
                let est = 1.0 / self.sigma;
                (name.to_string(), Estimate { est, se: est * se_l, lower: (-hi).exp(), upper: (-lo).exp() })
            } else {
                (
                    name.to_string(),
                    Estimate { est: self.sigma, se: self.sigma * se_l, lower: lo.exp(), upper: hi.exp() },
                )
            }
        };
        match self.family {
            Family::Weibull => vec![scale("Shape", true), loc("Scale", true)],
            Family::Lognormal => vec![loc("Location", false), scale("Scale", false)],
            Family::Exponential => vec![loc("Mean", true)],
            Family::Normal => vec![loc("Mean", false), scale("StDev", false)],
            Family::Logistic | Family::Loglogistic => vec![loc("Location", false), scale("Scale", false)],
        }
    }
}

/// Fit one family to censored data.
pub fn fit_distribution(data: &LifeData, family: Family) -> Result<LifeFit> {
    let x: Vec<Vec<f64>> = vec![vec![1.0]; data.obs.len()];
    let core = fit_core(family, &data.obs, &x)?;
    let total = data.total();
    let failures = data.failures();
    Ok(LifeFit {
        family,
        mu: core.beta[0],
        sigma: core.ln_sigma.exp(),
        loglik: core.loglik,
        n: total,
        failures,
        censored: total - failures,
        cov: core.cov,
        converged: core.converged,
    })
}

/// Fit every family that suits the data and order by log-likelihood-based AIC.
pub fn rank_distributions(data: &LifeData) -> Vec<LifeFit> {
    let mut v: Vec<LifeFit> = FAMILIES.iter().filter_map(|&f| fit_distribution(data, f).ok()).collect();
    v.sort_by(|a, b| a.aic().total_cmp(&b.aic()));
    v
}
