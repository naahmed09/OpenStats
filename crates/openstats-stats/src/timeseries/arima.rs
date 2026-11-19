//! Seasonal ARIMA(p,d,q)×(P,D,Q)ₛ models.
//!
//! Notation follows Box–Jenkins (and Minitab): `φ(B)Φ(Bˢ)(1−B)ᵈ(1−Bˢ)ᴰ(yₜ) = c + θ(B)Θ(Bˢ)eₜ`
//! with `φ(B) = 1 − φ₁B − …` and `θ(B) = 1 − θ₁B − …` (note the minus sign on MA terms).
//!
//! Estimation works on the differenced series `wₜ` with mean `μ` (the constant term, when
//! included). Parameters are searched in an unconstrained space that maps each AR and MA block
//! through partial autocorrelations `tanh(u)` (Jones 1980), so every iterate is stationary and
//! invertible. Methods:
//!
//! * **CSS** — conditional sum of squares, pre-sample innovations set to zero.
//! * **ML** — exact Gaussian likelihood by the Kalman filter, started from the stationary state
//!   covariance (found by doubling), with σ² concentrated out.
//! * **CSS-ML** (default) — CSS first, then ML from its solution.
//!
//! Standard errors come from the numerical Hessian of the objective in the natural parameters.

use super::acf::{acf, ljung_box};
use super::{accuracy, check_series, make_forecast, Accuracy, Forecast};
use crate::error::{invalid, Result};
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::optim::{bfgs, nelder_mead, NelderMeadOptions, Options};

/// Model orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArimaSpec {
    /// Non-seasonal AR order.
    pub p: usize,
    /// Non-seasonal differences.
    pub d: usize,
    /// Non-seasonal MA order.
    pub q: usize,
    /// Seasonal AR order.
    pub sp: usize,
    /// Seasonal differences.
    pub sd: usize,
    /// Seasonal MA order.
    pub sq: usize,
    /// Seasonal period (needed when any seasonal order is positive).
    pub period: usize,
    /// Estimate a constant (the mean of the differenced series).
    pub constant: bool,
}

impl ArimaSpec {
    /// A non-seasonal model with a constant.
    pub fn new(p: usize, d: usize, q: usize) -> ArimaSpec {
        ArimaSpec { p, d, q, sp: 0, sd: 0, sq: 0, period: 1, constant: true }
    }

    /// Add seasonal orders.
    pub fn seasonal(mut self, sp: usize, sd: usize, sq: usize, period: usize) -> ArimaSpec {
        self.sp = sp;
        self.sd = sd;
        self.sq = sq;
        self.period = period;
        self
    }

    /// Include or drop the constant.
    pub fn with_constant(mut self, constant: bool) -> ArimaSpec {
        self.constant = constant;
        self
    }

    /// `ARIMA (1,1,0)(0,1,1)12`.
    pub fn name(&self) -> String {
        let base = format!("ARIMA({},{},{})", self.p, self.d, self.q);
        if self.sp + self.sd + self.sq > 0 {
            format!("{base}({},{},{}){}", self.sp, self.sd, self.sq, self.period)
        } else {
            base
        }
    }

    fn n_arma(&self) -> usize {
        self.p + self.q + self.sp + self.sq
    }

    fn lost(&self) -> usize {
        self.d + self.sd * self.period
    }
}

/// Estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArimaMethod {
    /// Conditional sum of squares.
    Css,
    /// Exact Gaussian maximum likelihood.
    Ml,
    /// CSS, then ML.
    CssMl,
}

/// Options of an ARIMA fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ArimaOptions {
    /// Method (default CSS-ML).
    pub method: ArimaMethod,
    /// Iteration limit of each optimisation.
    pub max_iter: usize,
    /// Number of residual lags for the Ljung–Box test (default: 12, or 24 / 36 for long seasons).
    pub lb_lags: Option<usize>,
}

impl Default for ArimaOptions {
    fn default() -> Self {
        ArimaOptions { method: ArimaMethod::CssMl, max_iter: 500, lb_lags: None }
    }
}

/// One row of the coefficient table.
#[derive(Debug, Clone, PartialEq)]
pub struct CoefRow {
    /// `AR 1`, `SAR 12`, `MA 1`, `SMA 12`, `Constant`.
    pub label: String,
    /// Estimate.
    pub coef: f64,
    /// Standard error.
    pub se: Option<f64>,
    /// t statistic.
    pub t: Option<f64>,
    /// Two-sided p-value.
    pub p: Option<f64>,
}

/// A fitted ARIMA model.
#[derive(Debug, Clone)]
pub struct ArimaFit {
    /// The orders.
    pub spec: ArimaSpec,
    /// Method that produced the estimates.
    pub method: ArimaMethod,
    /// The series.
    pub y: Vec<f64>,
    /// Non-seasonal AR coefficients.
    pub phi: Vec<f64>,
    /// Non-seasonal MA coefficients.
    pub theta: Vec<f64>,
    /// Seasonal AR coefficients.
    pub sphi: Vec<f64>,
    /// Seasonal MA coefficients.
    pub stheta: Vec<f64>,
    /// Mean `μ` of the differenced series (0 without a constant).
    pub mean: f64,
    /// Constant term `c = μ(1 − Σφ…)` of the model equation (0 without a constant).
    pub constant: f64,
    /// Coefficient table.
    pub coefs: Vec<CoefRow>,
    /// Innovation variance.
    pub sigma2: f64,
    /// Number of differenced observations used.
    pub n_used: usize,
    /// Log-likelihood (Gaussian; for CSS fits the conditional one).
    pub loglik: f64,
    /// AIC.
    pub aic: f64,
    /// Corrected AIC.
    pub aicc: f64,
    /// BIC.
    pub bic: f64,
    /// Sum of squared residuals over the differenced observations.
    pub ssr: f64,
    /// One-step prediction errors, aligned with `y` (`NaN` where undefined).
    pub residuals: Vec<f64>,
    /// `y − residuals`.
    pub fitted: Vec<f64>,
    /// Ljung–Box Q at each of the lags in `lb_lags`.
    pub ljung_box: Vec<LjungBoxRow>,
    /// Residual ACF.
    pub residual_acf: Vec<f64>,
    /// Accuracy of the one-step fits.
    pub accuracy: Accuracy,
    /// Whether the optimiser reported convergence.
    pub converged: bool,
    ar_full: Vec<f64>,
    ma_full: Vec<f64>,
}

/// Ljung–Box test at one lag.
#[derive(Debug, Clone, PartialEq)]
pub struct LjungBoxRow {
    /// Lag.
    pub lag: usize,
    /// Statistic.
    pub q: f64,
    /// Degrees of freedom.
    pub df: usize,
    /// p-value (`None` when `df ≤ 0`).
    pub p: Option<f64>,
}

// ------------------------------------------------------------------------------ polynomials

/// Coefficients `a` of `∏(1 − Σ cᵢBⁱ)` for the non-seasonal `c` and seasonal `sc` factors
/// (the product is `1 − Σ aᵢBⁱ`).
fn expand(c: &[f64], sc: &[f64], period: usize) -> Vec<f64> {
    let mut poly = vec![0.0; c.len() + sc.len() * period + 1];
    poly[0] = 1.0;
    let mut a = vec![0.0; c.len() + 1];
    a[0] = 1.0;
    for (i, v) in c.iter().enumerate() {
        a[i + 1] = -v;
    }
    let mut b = vec![0.0; sc.len() * period + 1];
    b[0] = 1.0;
    for (j, v) in sc.iter().enumerate() {
        b[(j + 1) * period] = -v;
    }
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            poly[i + j] += x * y;
        }
    }
    poly[1..].iter().map(|v| -v).collect()
}

/// Map partial autocorrelations to AR coefficients.
fn pacf_to_ar(r: &[f64]) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::new();
    for (m, &a) in r.iter().enumerate() {
        let prev = phi.clone();
        phi = vec![0.0; m + 1];
        for j in 0..m {
            phi[j] = prev[j] - a * prev[m - 1 - j];
        }
        phi[m] = a;
    }
    phi
}

struct Layout {
    p: usize,
    q: usize,
    sp: usize,
    sq: usize,
    mean: bool,
    period: usize,
}

struct Params {
    phi: Vec<f64>,
    theta: Vec<f64>,
    sphi: Vec<f64>,
    stheta: Vec<f64>,
    mu: f64,
}

impl Layout {
    fn dim(&self) -> usize {
        self.p + self.q + self.sp + self.sq + usize::from(self.mean)
    }

    /// Natural parameters from the unconstrained vector.
    fn natural(&self, u: &[f64], mu_scale: f64) -> Params {
        let mut i = 0;
        let mut block = |k: usize| {
            let r: Vec<f64> = u[i..i + k].iter().map(|v| v.tanh() * 0.999_999).collect();
            i += k;
            pacf_to_ar(&r)
        };
        let phi = block(self.p);
        let theta = block(self.q);
        let sphi = block(self.sp);
        let stheta = block(self.sq);
        let mu = if self.mean { u[i] * mu_scale } else { 0.0 };
        Params { phi, theta, sphi, stheta, mu }
    }

    fn flat(&self, p: &Params) -> Vec<f64> {
        let mut v = Vec::new();
        v.extend(&p.phi);
        v.extend(&p.theta);
        v.extend(&p.sphi);
        v.extend(&p.stheta);
        if self.mean {
            v.push(p.mu);
        }
        v
    }

    fn params_from_flat(&self, v: &[f64]) -> Params {
        let (p, q, sp, sq) = (self.p, self.q, self.sp, self.sq);
        Params {
            phi: v[..p].to_vec(),
            theta: v[p..p + q].to_vec(),
            sphi: v[p + q..p + q + sp].to_vec(),
            stheta: v[p + q + sp..p + q + sp + sq].to_vec(),
            mu: if self.mean { v[p + q + sp + sq] } else { 0.0 },
        }
    }

    fn polys(&self, p: &Params) -> (Vec<f64>, Vec<f64>) {
        (expand(&p.phi, &p.sphi, self.period), expand(&p.theta, &p.stheta, self.period))
    }
}

// ------------------------------------------------------------------------------ likelihoods

/// Conditional residuals; `e[t] = 0` for `t < start`.
fn css_residuals(w: &[f64], mu: f64, a: &[f64], m: &[f64]) -> (Vec<f64>, usize) {
    let n = w.len();
    let start = a.len().min(n);
    let mut e = vec![0.0; n];
    for t in start..n {
        let mut pred = 0.0;
        for (i, ai) in a.iter().enumerate() {
            pred += ai * (w[t - 1 - i] - mu);
        }
        for (j, mj) in m.iter().enumerate() {
            if t > j {
                pred -= mj * e[t - 1 - j];
            }
        }
        e[t] = (w[t] - mu) - pred;
    }
    (e, start)
}

fn css_objective(w: &[f64], mu: f64, a: &[f64], m: &[f64]) -> f64 {
    let (e, start) = css_residuals(w, mu, a, m);
    let n = (w.len() - start).max(1) as f64;
    let ss: f64 = e[start..].iter().map(|v| v * v).sum();
    if !ss.is_finite() {
        return f64::MAX;
    }
    0.5 * n * (ss.max(1e-300) / n).ln()
}

fn matmul(a: &[f64], b: &[f64], r: usize) -> Vec<f64> {
    let mut c = vec![0.0; r * r];
    for i in 0..r {
        for k in 0..r {
            let aik = a[i * r + k];
            if aik != 0.0 {
                for j in 0..r {
                    c[i * r + j] += aik * b[k * r + j];
                }
            }
        }
    }
    c
}

/// Stationary state covariance `P = T P T′ + R R′` by doubling; `None` if not stationary.
fn stationary_cov(phi: &[f64], rvec: &[f64]) -> Option<Vec<f64>> {
    let r = phi.len();
    let mut t = vec![0.0; r * r];
    for i in 0..r {
        t[i * r] = phi[i];
        if i + 1 < r {
            t[i * r + i + 1] = 1.0;
        }
    }
    let mut p = vec![0.0; r * r];
    for i in 0..r {
        for j in 0..r {
            p[i * r + j] = rvec[i] * rvec[j];
        }
    }
    let mut a = t;
    for _ in 0..64 {
        // P ← P + A P A′
        let ap = matmul(&a, &p, r);
        let mut at = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                at[i * r + j] = a[j * r + i];
            }
        }
        let apa = matmul(&ap, &at, r);
        for (x, y) in p.iter_mut().zip(&apa) {
            *x += y;
        }
        a = matmul(&a, &a, r);
        let amax = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if !amax.is_finite() || p.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if amax < 1e-15 {
            return Some(p);
        }
    }
    let amax = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    (amax < 1e-6).then_some(p)
}

struct Kalman {
    v: Vec<f64>,
    ssq: f64,
    sum_ln_f: f64,
}

/// Exact Gaussian innovations filter for `w − μ` (σ² concentrated out).
fn kalman(wc: &[f64], a: &[f64], m: &[f64]) -> Option<Kalman> {
    let n = wc.len();
    let r = a.len().max(m.len() + 1).max(1);
    let mut phi = vec![0.0; r];
    phi[..a.len()].copy_from_slice(a);
    let mut rv = vec![0.0; r];
    rv[0] = 1.0;
    for (j, mj) in m.iter().enumerate() {
        rv[j + 1] = -mj;
    }
    let mut p = stationary_cov(&phi, &rv)?;
    let mut st = vec![0.0; r];
    let mut v = Vec::with_capacity(n);
    let (mut ssq, mut sum_ln_f) = (0.0, 0.0);
    let mut tmp = vec![0.0; r * r];
    for t in 0..n {
        let f = p[0];
        if !(f.is_finite() && f > 1e-12) {
            return None;
        }
        let vt = wc[t] - st[0];
        v.push(vt);
        ssq += vt * vt / f;
        sum_ln_f += f.ln();
        // Update.
        let p0: Vec<f64> = (0..r).map(|i| p[i * r]).collect();
        let p0row: Vec<f64> = p[..r].to_vec();
        for i in 0..r {
            st[i] += p0[i] / f * vt;
            for j in 0..r {
                p[i * r + j] -= p0[i] * p0row[j] / f;
            }
        }
        // Predict: a ← T a, P ← T P T′ + R R′.
        let mut sn = vec![0.0; r];
        for i in 0..r {
            sn[i] = phi[i] * st[0] + if i + 1 < r { st[i + 1] } else { 0.0 };
        }
        st = sn;
        for i in 0..r {
            for j in 0..r {
                tmp[i * r + j] = phi[i] * p[j] + if i + 1 < r { p[(i + 1) * r + j] } else { 0.0 };
            }
        }
        for i in 0..r {
            for j in 0..r {
                p[i * r + j] = tmp[i * r] * phi[j] + if j + 1 < r { tmp[i * r + j + 1] } else { 0.0 } + rv[i] * rv[j];
            }
        }
    }
    ssq.is_finite().then_some(Kalman { v, ssq, sum_ln_f })
}

fn ml_objective(w: &[f64], mu: f64, a: &[f64], m: &[f64]) -> f64 {
    let wc: Vec<f64> = w.iter().map(|x| x - mu).collect();
    match kalman(&wc, a, m) {
        Some(k) => {
            let n = w.len() as f64;
            0.5 * (n * (k.ssq.max(1e-300) / n).ln() + k.sum_ln_f)
        }
        None => f64::MAX,
    }
}

// ------------------------------------------------------------------------------ optimisation

fn minimize(f: &mut dyn FnMut(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> (Vec<f64>, bool) {
    if x0.is_empty() {
        return (Vec::new(), true);
    }
    let o = Options { max_iter, gtol: 1e-7, ..Options::default() };
    let b = bfgs(f, None, x0, &o);
    let (mut best_x, mut best_f, mut conv) = match b {
        Ok(m) if m.f.is_finite() => (m.x, m.f, m.converged),
        _ => (x0.to_vec(), f(x0), false),
    };
    if !conv {
        let nm = NelderMeadOptions { max_iter: 400 * x0.len(), ..NelderMeadOptions::default() };
        if let Ok(m) = nelder_mead(f, &best_x, &nm) {
            if m.f <= best_f {
                best_x = m.x;
                best_f = m.f;
                conv = m.converged;
            }
        }
    }
    let _ = best_f;
    (best_x, conv)
}

/// Invert a symmetric positive-definite matrix by Gauss–Jordan; `None` if singular or indefinite.
fn invert(h: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = h.len();
    let mut a: Vec<Vec<f64>> = h
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for c in 0..n {
        let piv = (c..n).max_by(|&x, &y| a[x][c].abs().partial_cmp(&a[y][c].abs()).unwrap())?;
        if a[piv][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, piv);
        let d = a[c][c];
        for v in &mut a[c] {
            *v /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                if f != 0.0 {
                    let rowc = a[c].clone();
                    for (x, y) in a[r].iter_mut().zip(&rowc) {
                        *x -= f * y;
                    }
                }
            }
        }
    }
    let inv: Vec<Vec<f64>> = a.into_iter().map(|r| r[n..].to_vec()).collect();
    (0..n).all(|i| inv[i][i] > 0.0).then_some(inv)
}

fn hessian(f: &mut dyn FnMut(&[f64]) -> f64, x: &[f64], steps: &[f64]) -> Option<Vec<Vec<f64>>> {
    let k = x.len();
    let f0 = f(x);
    if f0 >= f64::MAX / 2.0 {
        return None;
    }
    let mut h = vec![vec![0.0; k]; k];
    let eval = |f: &mut dyn FnMut(&[f64]) -> f64, d: &[(usize, f64)]| {
        let mut y = x.to_vec();
        for &(i, s) in d {
            y[i] += s;
        }
        f(&y)
    };
    for i in 0..k {
        let (hi, fp, fm) = (steps[i], eval(f, &[(i, steps[i])]), eval(f, &[(i, -steps[i])]));
        if fp >= f64::MAX / 2.0 || fm >= f64::MAX / 2.0 {
            return None;
        }
        h[i][i] = (fp - 2.0 * f0 + fm) / (hi * hi);
        for j in i + 1..k {
            let (a, b) = (steps[i], steps[j]);
            let v = (eval(f, &[(i, a), (j, b)]) - eval(f, &[(i, a), (j, -b)]) - eval(f, &[(i, -a), (j, b)])
                + eval(f, &[(i, -a), (j, -b)]))
                / (4.0 * a * b);
            h[i][j] = v;
            h[j][i] = v;
        }
    }
    h.iter().flatten().all(|v| v.is_finite()).then_some(h)
}

// ------------------------------------------------------------------------------ fitting

/// Fit an ARIMA model.
pub fn fit_arima(y: &[f64], spec: &ArimaSpec, opts: &ArimaOptions) -> Result<ArimaFit> {
    let seasonal = spec.sp + spec.sd + spec.sq > 0;
    if seasonal && spec.period < 2 {
        return invalid("a seasonal ARIMA model needs a period of at least 2");
    }
    if spec.p > 10 || spec.q > 10 || spec.sp > 4 || spec.sq > 4 || spec.d > 3 || spec.sd > 2 {
        return invalid("orders are limited to p, q ≤ 10, P, Q ≤ 4, d ≤ 3, D ≤ 2");
    }
    let period = if seasonal { spec.period } else { 1 };
    let spec = ArimaSpec { period, ..*spec };
    let n_par = spec.n_arma() + usize::from(spec.constant);
    let min = spec.lost() + (spec.p + spec.q + spec.sp * period + spec.sq * period).max(2 * n_par).max(8);
    check_series(y, min, "ARIMA estimation")?;
    let n = y.len();
    let mut w = y.to_vec();
    for _ in 0..spec.d {
        w = (1..w.len()).map(|t| w[t] - w[t - 1]).collect();
    }
    for _ in 0..spec.sd {
        w = (period..w.len()).map(|t| w[t] - w[t - period]).collect();
    }
    let nw = w.len();
    let wmean = w.iter().sum::<f64>() / nw as f64;
    let wsd = (w.iter().map(|v| (v - wmean).powi(2)).sum::<f64>() / nw as f64).sqrt();
    if wsd <= 1e-12 * (1.0 + wmean.abs()) {
        return invalid("the differenced series is constant");
    }
    let lay = Layout { p: spec.p, q: spec.q, sp: spec.sp, sq: spec.sq, mean: spec.constant, period };
    let mu_scale = wsd;
    let mut u0 = vec![0.0; lay.dim()];
    if spec.constant {
        let k = u0.len() - 1;
        u0[k] = wmean / mu_scale;
    }
    let max_iter = opts.max_iter.max(20);
    let mut converged = true;
    let mut u = u0;
    let method_used = opts.method;
    if matches!(opts.method, ArimaMethod::Css | ArimaMethod::CssMl) {
        let mut f = |x: &[f64]| {
            let p = lay.natural(x, mu_scale);
            let (a, m) = lay.polys(&p);
            css_objective(&w, p.mu, &a, &m)
        };
        let (x, c) = minimize(&mut f, &u, max_iter);
        u = x;
        converged &= c;
    }
    if matches!(opts.method, ArimaMethod::Ml | ArimaMethod::CssMl) {
        let mut f = |x: &[f64]| {
            let p = lay.natural(x, mu_scale);
            let (a, m) = lay.polys(&p);
            ml_objective(&w, p.mu, &a, &m)
        };
        let (x, c) = minimize(&mut f, &u, max_iter);
        u = x;
        converged &= c;
    }
    let pars = lay.natural(&u, mu_scale);
    let (a_full, m_full) = lay.polys(&pars);
    // Residuals and σ².
    let ml = matches!(opts.method, ArimaMethod::Ml | ArimaMethod::CssMl);
    let (res_w, sigma2, ssr, loglik, n_eff) = if ml {
        let wc: Vec<f64> = w.iter().map(|x| x - pars.mu).collect();
        let k = match kalman(&wc, &a_full, &m_full) {
            Some(k) => k,
            None => return invalid("the likelihood could not be evaluated at the estimates"),
        };
        let nf = nw as f64;
        let s2 = k.ssq / nf;
        let ll = -0.5 * (nf * (2.0 * std::f64::consts::PI * s2).ln() + k.sum_ln_f + nf);
        let ssr: f64 = k.v.iter().map(|v| v * v).sum();
        (k.v, s2, ssr, ll, nw)
    } else {
        let (e, start) = css_residuals(&w, pars.mu, &a_full, &m_full);
        let ne = nw - start;
        let ss: f64 = e[start..].iter().map(|v| v * v).sum();
        let s2 = ss / ne as f64;
        let ll = -0.5 * ne as f64 * ((2.0 * std::f64::consts::PI * s2).ln() + 1.0);
        let res: Vec<f64> = e.iter().enumerate().map(|(t, &v)| if t < start { f64::NAN } else { v }).collect();
        (res, s2, ss, ll, ne)
    };
    let lost = spec.lost();
    let mut residuals = vec![f64::NAN; n];
    for (t, &v) in res_w.iter().enumerate() {
        residuals[t + lost] = v;
    }
    let fitted: Vec<f64> = y.iter().zip(&residuals).map(|(a, e)| a - e).collect();
    // Standard errors.
    let flat = lay.flat(&pars);
    let steps: Vec<f64> = flat
        .iter()
        .enumerate()
        .map(|(i, v)| if lay.mean && i == flat.len() - 1 { 1e-4 * wsd.max(1e-6) } else { 1e-4 * v.abs().max(0.1) })
        .collect();
    let mut nat = |x: &[f64]| {
        let p = lay.params_from_flat(x);
        let (a, m) = lay.polys(&p);
        if ml {
            ml_objective(&w, p.mu, &a, &m)
        } else {
            css_objective(&w, p.mu, &a, &m)
        }
    };
    let cov = hessian(&mut nat, &flat, &steps).and_then(|h| invert(&h));
    let k_par = flat.len();
    let df = (n_eff as f64 - k_par as f64).max(1.0);
    let dist = StudentT::new(df).ok();
    // Table order: AR, SAR, MA, SMA, Constant (Minitab order); flat order is AR, MA, SAR, SMA.
    let order: Vec<(String, usize)> = {
        let mut o = Vec::new();
        for i in 0..spec.p {
            o.push((format!("AR {}", i + 1), i));
        }
        for i in 0..spec.sp {
            o.push((format!("SAR {}", (i + 1) * period), spec.p + spec.q + i));
        }
        for i in 0..spec.q {
            o.push((format!("MA {}", i + 1), spec.p + i));
        }
        for i in 0..spec.sq {
            o.push((format!("SMA {}", (i + 1) * period), spec.p + spec.q + spec.sp + i));
        }
        if spec.constant {
            o.push(("Constant".to_string(), k_par - 1));
        }
        o
    };
    let ar_sum: f64 = 1.0 - a_full.iter().sum::<f64>();
    let constant = if spec.constant { pars.mu * ar_sum } else { 0.0 };
    let coefs: Vec<CoefRow> = order
        .into_iter()
        .map(|(label, idx)| {
            let coef = flat[idx];
            let se = cov.as_ref().map(|c| {
                if method_used == ArimaMethod::Css {
                    (2.0 * sigma2 * c[idx][idx]).sqrt()
                } else {
                    c[idx][idx].sqrt()
                }
            });
            let (coef_out, se) =
                if label == "Constant" { (constant, se.map(|s| s * ar_sum.abs())) } else { (coef, se) };
            let t = se.filter(|s| *s > 0.0).map(|s| coef_out / s);
            let p = t.and_then(|t| dist.as_ref().map(|d| 2.0 * d.sf(t.abs())));
            CoefRow { label, coef: coef_out, se, t, p }
        })
        .collect();
    let kk = (k_par + 1) as f64;
    let nf = n_eff as f64;
    let aic = -2.0 * loglik + 2.0 * kk;
    let aicc = if nf - kk - 1.0 > 0.0 { aic + 2.0 * kk * (kk + 1.0) / (nf - kk - 1.0) } else { f64::INFINITY };
    let bic = -2.0 * loglik + kk * nf.ln();
    // Residual diagnostics.
    let valid: Vec<f64> = residuals.iter().copied().filter(|v| !v.is_nan()).collect();
    let lags = opts.lb_lags.unwrap_or(if period >= 12 { 24 } else { 12 }).min(valid.len().saturating_sub(2)).max(1);
    let (residual_acf, ljung) = match acf(&valid, Some(lags), 0.95) {
        Ok(a) => {
            let mut rows = Vec::new();
            let mut marks: Vec<usize> = vec![12, 24, 36, 48];
            marks.retain(|&m| m <= lags);
            if !marks.contains(&lags) {
                marks.push(lags);
            }
            for &h in &marks {
                let (q, p) = ljung_box(&a.acf[..h], valid.len(), spec.n_arma());
                rows.push(LjungBoxRow { lag: h, q, df: h.saturating_sub(spec.n_arma()), p });
            }
            (a.acf, rows)
        }
        Err(_) => (Vec::new(), Vec::new()),
    };
    let acc = accuracy(y, &fitted);
    Ok(ArimaFit {
        spec,
        method: method_used,
        y: y.to_vec(),
        phi: pars.phi,
        theta: pars.theta,
        sphi: pars.sphi,
        stheta: pars.stheta,
        mean: pars.mu,
        constant,
        coefs,
        sigma2,
        n_used: n_eff,
        loglik,
        aic,
        aicc,
        bic,
        ssr,
        residuals,
        fitted,
        ljung_box: ljung,
        residual_acf,
        accuracy: acc,
        converged,
        ar_full: a_full,
        ma_full: m_full,
    })
}

impl ArimaFit {
    /// Forecasts for horizons 1 … `h` with `level` prediction intervals.
    pub fn forecast(&self, h: usize, level: f64) -> Result<Forecast> {
        if h == 0 {
            return invalid("the forecast horizon must be at least 1");
        }
        let n = self.y.len();
        let spec = &self.spec;
        // Φ*(B) = AR(B)·(1−B)ᵈ(1−Bˢ)ᴰ as 1 − Σ cᵢBⁱ.
        let mut poly = vec![1.0];
        poly.extend(self.ar_full.iter().map(|v| -v));
        let diff = |lag: usize, poly: &mut Vec<f64>| {
            let mut out = vec![0.0; poly.len() + lag];
            for (i, v) in poly.iter().enumerate() {
                out[i] += v;
                out[i + lag] -= v;
            }
            *poly = out;
        };
        for _ in 0..spec.d {
            diff(1, &mut poly);
        }
        for _ in 0..spec.sd {
            diff(spec.period, &mut poly);
        }
        let c: Vec<f64> = poly[1..].iter().map(|v| -v).collect();
        // Observed values extended with forecasts; residuals extended with zeros.
        let mut yy = self.y.clone();
        let mut ee: Vec<f64> = self.residuals.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
        for _ in 0..h {
            let t = yy.len();
            let mut f = self.constant;
            for (i, ci) in c.iter().enumerate() {
                if t > i {
                    f += ci * yy[t - 1 - i];
                }
            }
            for (j, mj) in self.ma_full.iter().enumerate() {
                if t > j {
                    f -= mj * ee[t - 1 - j];
                }
            }
            yy.push(f);
            ee.push(0.0);
        }
        let point = yy[n..].to_vec();
        // ψ weights of MA(B)/Φ*(B).
        let mut psi = vec![0.0; h];
        psi[0] = 1.0;
        for j in 1..h {
            let mut v = if j <= self.ma_full.len() { -self.ma_full[j - 1] } else { 0.0 };
            for (i, ci) in c.iter().enumerate() {
                if j > i {
                    v += ci * psi[j - 1 - i];
                }
            }
            psi[j] = v;
        }
        let mut cum = 0.0;
        let se: Vec<f64> = (0..h)
            .map(|j| {
                cum += psi[j] * psi[j];
                (self.sigma2 * cum).sqrt()
            })
            .collect();
        make_forecast(n - 1, point, se, level)
    }

    /// Coefficients of the full AR operator `φ(B)Φ(Bˢ)` as `aᵢ` in `1 − Σ aᵢBⁱ`.
    pub fn ar_operator(&self) -> &[f64] {
        &self.ar_full
    }

    /// Coefficients of the full MA operator `θ(B)Θ(Bˢ)` as `mⱼ` in `1 − Σ mⱼBʲ`.
    pub fn ma_operator(&self) -> &[f64] {
        &self.ma_full
    }
}

/// A candidate model with its fit statistics.
#[derive(Debug, Clone)]
pub struct RankedArima {
    /// The fitted model.
    pub fit: ArimaFit,
}

/// Fit every spec that can be fitted and order them by AICc (specs with different differencing
/// are not comparable by likelihood, so only compare candidates with the same `d` and `D`).
pub fn rank_arima(y: &[f64], specs: &[ArimaSpec], opts: &ArimaOptions) -> Result<Vec<RankedArima>> {
    let mut out: Vec<RankedArima> =
        specs.iter().filter_map(|s| fit_arima(y, s, opts).ok()).map(|fit| RankedArima { fit }).collect();
    if out.is_empty() {
        return invalid("none of the candidate models could be fitted");
    }
    out.sort_by(|a, b| a.fit.aicc.partial_cmp(&b.fit.aicc).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}
