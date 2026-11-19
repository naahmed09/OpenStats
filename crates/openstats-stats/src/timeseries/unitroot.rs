//! Unit-root and stationarity tests (augmented Dickey–Fuller, KPSS), the cross-correlation
//! function and an automatic ARIMA order search.

use super::arima::{fit_arima, ArimaFit, ArimaOptions, ArimaSpec};
use super::difference;
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ContinuousDistribution, Normal};
use openstats_math::linalg::{Cholesky, Matrix};

/// Deterministic terms in the Dickey–Fuller regression and in the KPSS null.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deterministic {
    /// No constant.
    None,
    /// Constant (level stationarity for KPSS).
    Constant,
    /// Constant and linear trend (trend stationarity for KPSS).
    Trend,
}

/// Options of [`adf_test`].
#[derive(Debug, Clone, Copy)]
pub struct AdfOptions {
    /// Deterministic terms.
    pub deterministic: Deterministic,
    /// Number of lagged differences; `None` chooses it by AIC up to Schwert's maximum.
    pub lags: Option<usize>,
}

impl Default for AdfOptions {
    fn default() -> Self {
        AdfOptions { deterministic: Deterministic::Constant, lags: None }
    }
}

/// Augmented Dickey–Fuller test result.
#[derive(Debug, Clone, PartialEq)]
pub struct AdfTest {
    /// Series length.
    pub n: usize,
    /// Lagged differences used.
    pub lags: usize,
    /// Observations in the final regression.
    pub nobs: usize,
    /// Deterministic terms used.
    pub deterministic: Deterministic,
    /// Coefficient of the lagged level.
    pub gamma: f64,
    /// Its standard error.
    pub se: f64,
    /// Test statistic `γ̂ / se`.
    pub statistic: f64,
    /// Finite-sample critical values at 1 %, 5 % and 10 % (MacKinnon response surface).
    pub critical: [f64; 3],
    /// AIC of the chosen regression (`None` when the lag was given).
    pub aic: Option<f64>,
}

impl AdfTest {
    /// Whether the unit root is rejected at the 1 %, 5 % or 10 % level (index 0, 1, 2).
    pub fn rejects(&self, level_index: usize) -> bool {
        self.statistic < self.critical[level_index]
    }
}

/// MacKinnon (2010) response-surface coefficients `[b0, b1, b2, b3]` for 1 %, 5 % and 10 %.
fn mackinnon(det: Deterministic) -> [[f64; 4]; 3] {
    match det {
        Deterministic::None => {
            [[-2.56574, -2.2358, -3.627, 0.0], [-1.94100, -0.2686, -3.365, 31.223], [-1.61682, 0.2656, -2.714, 25.364]]
        }
        Deterministic::Constant => [
            [-3.43035, -6.5393, -16.786, -79.433],
            [-2.86154, -2.8903, -4.234, -40.040],
            [-2.56677, -1.5384, -2.809, 0.0],
        ],
        Deterministic::Trend => [
            [-3.95877, -9.0531, -28.428, -134.155],
            [-3.41049, -4.3904, -9.036, -45.374],
            [-3.12705, -2.5856, -3.925, -22.380],
        ],
    }
}

fn clean_series(y: &[f64], min: usize) -> Result<Vec<f64>> {
    if y.iter().any(|v| !v.is_finite()) {
        return invalid("the series must not contain missing or infinite values");
    }
    if y.len() < min {
        return Err(StatsError::InsufficientData { what: "this test", needed: min, got: y.len() });
    }
    Ok(y.to_vec())
}

struct Ols {
    beta: Vec<f64>,
    se: Vec<f64>,
    ssr: f64,
    nobs: usize,
    k: usize,
}

fn ols(rows: &[Vec<f64>], resp: &[f64]) -> Result<Ols> {
    let (n, k) = (rows.len(), rows[0].len());
    if n <= k {
        return invalid("too few observations for the requested number of lags");
    }
    let xtx = Matrix::from_fn(k, k, |a, b| rows.iter().map(|r| r[a] * r[b]).sum());
    let xty: Vec<f64> = (0..k).map(|a| rows.iter().zip(resp).map(|(r, y)| r[a] * y).sum()).collect();
    let ch = Cholesky::new(&xtx)
        .map_err(|_| StatsError::InvalidArgument("the regression is singular (constant series?)".into()))?;
    let beta = ch.solve(&xty)?;
    let ssr: f64 = rows
        .iter()
        .zip(resp)
        .map(|(r, y)| {
            let e = y - r.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>();
            e * e
        })
        .sum();
    let inv = ch.inverse()?;
    let s2 = ssr / (n - k) as f64;
    let se = (0..k).map(|j| (s2 * inv[(j, j)]).sqrt()).collect();
    Ok(Ols { beta, se, ssr, nobs: n, k })
}

/// Regression of `Δyₜ` on `yₜ₋₁`, deterministic terms and `lags` lagged differences, using observations
/// from index `start` (so that every candidate lag can share a sample).
fn adf_regression(y: &[f64], det: Deterministic, lags: usize, start: usize) -> Result<Ols> {
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let mut rows = Vec::new();
    let mut resp = Vec::new();
    for t in start..dy.len() {
        let mut r = vec![y[t]];
        match det {
            Deterministic::None => {}
            Deterministic::Constant => r.push(1.0),
            Deterministic::Trend => {
                r.push(1.0);
                r.push((t + 1) as f64);
            }
        }
        for l in 1..=lags {
            r.push(dy[t - l]);
        }
        rows.push(r);
        resp.push(dy[t]);
    }
    ols(&rows, &resp)
}

/// Augmented Dickey–Fuller test of a unit root (`H₀`) against stationarity.
pub fn adf_test(y: &[f64], opts: &AdfOptions) -> Result<AdfTest> {
    let y = clean_series(y, 12)?;
    let n = y.len();
    let (lags, aic) = match opts.lags {
        Some(l) => (l, None),
        None => {
            let maxlag = ((12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize).min((n - 1) / 2 - 2);
            let mut best = (0usize, f64::INFINITY);
            for l in 0..=maxlag {
                let fit = adf_regression(&y, opts.deterministic, l, maxlag)?;
                let nn = fit.nobs as f64;
                let aic = nn * (fit.ssr / nn).ln() + 2.0 * fit.k as f64;
                if aic < best.1 {
                    best = (l, aic);
                }
            }
            (best.0, Some(best.1))
        }
    };
    if lags + 2 >= n {
        return invalid("too many lags for the series length");
    }
    let fit = adf_regression(&y, opts.deterministic, lags, lags)?;
    let nn = fit.nobs as f64;
    let coef = mackinnon(opts.deterministic);
    let critical = [0, 1, 2].map(|i| {
        let c = coef[i];
        c[0] + c[1] / nn + c[2] / (nn * nn) + c[3] / (nn * nn * nn)
    });
    Ok(AdfTest {
        n,
        lags,
        nobs: fit.nobs,
        deterministic: opts.deterministic,
        gamma: fit.beta[0],
        se: fit.se[0],
        statistic: fit.beta[0] / fit.se[0],
        critical,
        aic,
    })
}

/// KPSS test result.
#[derive(Debug, Clone, PartialEq)]
pub struct KpssTest {
    /// Series length.
    pub n: usize,
    /// Bartlett-window truncation lag.
    pub lags: usize,
    /// `true` for the trend-stationary null.
    pub trend: bool,
    /// Test statistic.
    pub statistic: f64,
    /// Critical values at 10 %, 5 %, 2.5 % and 1 %.
    pub critical: [f64; 4],
}

impl KpssTest {
    /// Whether stationarity is rejected at the 10 %, 5 %, 2.5 % or 1 % level (index 0–3).
    pub fn rejects(&self, level_index: usize) -> bool {
        self.statistic > self.critical[level_index]
    }
}

/// KPSS test of (level or trend) stationarity. `lags` defaults to `⌈12 (n/100)^¼⌉` ("long") when `None`.
pub fn kpss_test(y: &[f64], trend: bool, lags: Option<usize>) -> Result<KpssTest> {
    let y = clean_series(y, 8)?;
    let n = y.len();
    let nf = n as f64;
    let l = lags.unwrap_or_else(|| (12.0 * (nf / 100.0).powf(0.25)).ceil() as usize);
    if l >= n {
        return invalid("the lag truncation must be smaller than the series length");
    }
    // residuals from the level (mean) or trend regression
    let resid: Vec<f64> = if trend {
        let rows: Vec<Vec<f64>> = (0..n).map(|t| vec![1.0, (t + 1) as f64]).collect();
        let f = ols(&rows, &y)?;
        rows.iter().zip(&y).map(|(r, v)| v - r[0] * f.beta[0] - r[1] * f.beta[1]).collect()
    } else {
        let m = y.iter().sum::<f64>() / nf;
        y.iter().map(|v| v - m).collect()
    };
    let mut s = 0.0;
    let mut ss = 0.0;
    for e in &resid {
        s += e;
        ss += s * s;
    }
    let mut lrv = resid.iter().map(|e| e * e).sum::<f64>() / nf;
    for k in 1..=l {
        let w = 1.0 - k as f64 / (l as f64 + 1.0);
        let g: f64 = (k..n).map(|t| resid[t] * resid[t - k]).sum::<f64>() / nf;
        lrv += 2.0 * w * g;
    }
    let critical = if trend { [0.119, 0.146, 0.176, 0.216] } else { [0.347, 0.463, 0.574, 0.739] };
    Ok(KpssTest { n, lags: l, trend, statistic: ss / (nf * nf * lrv), critical })
}

/// Cross-correlation function of `x` and `y` at lags `−max_lag ..= max_lag`.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossCorrelation {
    /// Lags (negative: `y` leads `x`).
    pub lags: Vec<i64>,
    /// `corr(xₜ, yₜ₊ₖ)` at each lag `k`.
    pub ccf: Vec<f64>,
    /// Approximate two-sided limit `z / √n` for white noise.
    pub limit: f64,
    /// Series length.
    pub n: usize,
}

/// Cross-correlation of two series of equal length.
pub fn cross_correlation(x: &[f64], y: &[f64], max_lag: usize, level: f64) -> Result<CrossCorrelation> {
    let x = clean_series(x, 3)?;
    let y = clean_series(y, 3)?;
    if x.len() != y.len() {
        return invalid("the two series must have the same length");
    }
    let n = x.len();
    if max_lag >= n {
        return invalid("the maximum lag must be smaller than the series length");
    }
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    let nf = n as f64;
    let (mx, my) = (x.iter().sum::<f64>() / nf, y.iter().sum::<f64>() / nf);
    let sx = (x.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / nf).sqrt();
    let sy = (y.iter().map(|v| (v - my).powi(2)).sum::<f64>() / nf).sqrt();
    if sx == 0.0 || sy == 0.0 {
        return invalid("a series is constant");
    }
    let mut lags = Vec::new();
    let mut ccf = Vec::new();
    for k in -(max_lag as i64)..=(max_lag as i64) {
        let mut s = 0.0;
        for t in 0..n as i64 {
            let u = t + k;
            if (0..n as i64).contains(&u) {
                s += (x[t as usize] - mx) * (y[u as usize] - my);
            }
        }
        lags.push(k);
        ccf.push(s / (nf * sx * sy));
    }
    let z = Normal::new(0.0, 1.0)?.quantile(1.0 - (1.0 - level) / 2.0);
    Ok(CrossCorrelation { lags, ccf, limit: z / nf.sqrt(), n })
}

/// Candidate and chosen models of [`auto_arima`].
#[derive(Debug, Clone)]
pub struct AutoArima {
    /// Differencing order that was used (chosen by KPSS when not given).
    pub d: usize,
    /// All models that could be fitted, best AICc first.
    pub ranked: Vec<ArimaFit>,
}

/// Search ARIMA(p, d, q) with `p ≤ max_p`, `q ≤ max_q` and rank by AICc.
///
/// With `d = None` the number of differences (0–2) is the smallest for which the level KPSS test
/// does not reject at 5 %.
pub fn auto_arima(
    y: &[f64],
    max_p: usize,
    max_q: usize,
    d: Option<usize>,
    constant: bool,
    opts: &ArimaOptions,
) -> Result<AutoArima> {
    if max_p > 5 || max_q > 5 {
        return invalid("the maximum orders are limited to 5");
    }
    let d = match d {
        Some(d) if d <= 2 => d,
        Some(_) => return invalid("the differencing order must be 0, 1 or 2"),
        None => {
            let mut chosen = 2;
            for k in 0..=2 {
                let z = if k == 0 { y.to_vec() } else { difference(y, 1, k)? };
                if kpss_test(&z, false, None).is_ok_and(|t| !t.rejects(1)) {
                    chosen = k;
                    break;
                }
            }
            chosen
        }
    };
    let mut ranked: Vec<ArimaFit> = Vec::new();
    for p in 0..=max_p {
        for q in 0..=max_q {
            let spec = ArimaSpec { p, d, q, sp: 0, sd: 0, sq: 0, period: 1, constant: constant && d < 2 };
            if let Ok(f) = fit_arima(y, &spec, opts) {
                if f.aicc.is_finite() {
                    ranked.push(f);
                }
            }
        }
    }
    if ranked.is_empty() {
        return invalid("none of the candidate models could be fitted");
    }
    ranked.sort_by(|a, b| a.aicc.total_cmp(&b.aicc));
    Ok(AutoArima { d, ranked })
}
