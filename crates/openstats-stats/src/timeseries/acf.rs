//! Autocorrelation and partial autocorrelation, with confidence bounds and the Ljung–Box test.

use super::{check_series, z_for};
use crate::error::{invalid, Result};
use openstats_math::dist::{ChiSquared, ContinuousDistribution};

/// Sample autocorrelations.
#[derive(Debug, Clone, PartialEq)]
pub struct AcfResult {
    /// Series length.
    pub n: usize,
    /// Autocorrelation at lags 1 … K.
    pub acf: Vec<f64>,
    /// Bartlett standard errors (lag k uses the autocorrelations of lags 1 … k − 1).
    pub se: Vec<f64>,
    /// t statistics `acf / se`.
    pub t: Vec<f64>,
    /// Ljung–Box Q at each lag.
    pub lbq: Vec<f64>,
    /// Lower confidence limit (`−z·se`).
    pub lower: Vec<f64>,
    /// Upper confidence limit (`+z·se`).
    pub upper: Vec<f64>,
    /// Confidence level of the limits.
    pub level: f64,
}

/// Sample partial autocorrelations.
#[derive(Debug, Clone, PartialEq)]
pub struct PacfResult {
    /// Series length.
    pub n: usize,
    /// Partial autocorrelations at lags 1 … K.
    pub pacf: Vec<f64>,
    /// Standard error `1/√n` (the same at every lag).
    pub se: f64,
    /// Lower limit.
    pub lower: f64,
    /// Upper limit.
    pub upper: f64,
    /// Confidence level.
    pub level: f64,
}

/// Biased sample autocovariances `c₀ … c_K` (divisor `n`, mean removed).
pub(crate) fn autocovariances(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let d: Vec<f64> = x.iter().map(|v| v - mean).collect();
    (0..=k).map(|lag| (lag..n).map(|t| d[t] * d[t - lag]).sum::<f64>() / n as f64).collect()
}

fn default_lags(n: usize) -> usize {
    let a = (n as f64 / 4.0).floor() as usize;
    a.clamp(1, 60).min(n - 1)
}

/// Autocorrelations at lags 1 … `lags` (default `⌊n/4⌋`, at most 60).
pub fn acf(x: &[f64], lags: Option<usize>, level: f64) -> Result<AcfResult> {
    check_series(x, 3, "the autocorrelation function")?;
    let n = x.len();
    let k = lags.unwrap_or_else(|| default_lags(n));
    if k == 0 || k >= n {
        return invalid(format!("lags must be between 1 and {}", n - 1));
    }
    let c = autocovariances(x, k);
    if c[0] <= 0.0 {
        return invalid("the series is constant: autocorrelations are undefined");
    }
    let r: Vec<f64> = (1..=k).map(|l| c[l] / c[0]).collect();
    let z = z_for(level)?;
    let mut se = Vec::with_capacity(k);
    let mut cum = 0.0;
    for l in 0..k {
        se.push(((1.0 + 2.0 * cum) / n as f64).sqrt());
        cum += r[l] * r[l];
    }
    let t = r.iter().zip(&se).map(|(a, s)| a / s).collect();
    let mut lbq = Vec::with_capacity(k);
    let mut q = 0.0;
    for l in 0..k {
        q += r[l] * r[l] / (n - l - 1) as f64;
        lbq.push(n as f64 * (n as f64 + 2.0) * q);
    }
    Ok(AcfResult {
        n,
        lower: se.iter().map(|s| -z * s).collect(),
        upper: se.iter().map(|s| z * s).collect(),
        acf: r,
        se,
        t,
        lbq,
        level,
    })
}

/// Durbin–Levinson: partial autocorrelations and AR coefficients from autocorrelations `r₁ … r_K`.
pub(crate) fn durbin_levinson(r: &[f64]) -> Vec<f64> {
    let k = r.len();
    let mut pacf = Vec::with_capacity(k);
    let mut phi_prev: Vec<f64> = Vec::new();
    for m in 1..=k {
        let (num, den) = if m == 1 {
            (r[0], 1.0)
        } else {
            let num: f64 = r[m - 1] - (1..m).map(|j| phi_prev[j - 1] * r[m - 1 - j]).sum::<f64>();
            let den: f64 = 1.0 - (1..m).map(|j| phi_prev[j - 1] * r[j - 1]).sum::<f64>();
            (num, den)
        };
        let a = if den.abs() < 1e-300 { 0.0 } else { num / den };
        let mut phi = vec![0.0; m];
        for j in 1..m {
            phi[j - 1] = phi_prev[j - 1] - a * phi_prev[m - 1 - j];
        }
        phi[m - 1] = a;
        pacf.push(a);
        phi_prev = phi;
    }
    pacf
}

/// Partial autocorrelations at lags 1 … `lags` (default `⌊n/4⌋`, at most 60).
pub fn pacf(x: &[f64], lags: Option<usize>, level: f64) -> Result<PacfResult> {
    check_series(x, 3, "the partial autocorrelation function")?;
    let n = x.len();
    let k = lags.unwrap_or_else(|| default_lags(n));
    if k == 0 || k >= n {
        return invalid(format!("lags must be between 1 and {}", n - 1));
    }
    let c = autocovariances(x, k);
    if c[0] <= 0.0 {
        return invalid("the series is constant: partial autocorrelations are undefined");
    }
    let r: Vec<f64> = (1..=k).map(|l| c[l] / c[0]).collect();
    let z = z_for(level)?;
    let se = 1.0 / (n as f64).sqrt();
    Ok(PacfResult { n, pacf: durbin_levinson(&r), se, lower: -z * se, upper: z * se, level })
}

/// Ljung–Box statistic and p-value from autocorrelations `r₁ … r_h` of `n` residuals;
/// `df = h − fitted_params` (the p-value is `None` when that is not positive).
pub fn ljung_box(r: &[f64], n: usize, fitted_params: usize) -> (f64, Option<f64>) {
    let h = r.len();
    let nf = n as f64;
    let q = nf * (nf + 2.0) * r.iter().enumerate().map(|(l, v)| v * v / (nf - (l + 1) as f64)).sum::<f64>();
    if h <= fitted_params {
        return (q, None);
    }
    let p = ChiSquared::new((h - fitted_params) as f64).ok().map(|d| d.sf(q));
    (q, p)
}

/// Difference a series: `times` ordinary (`lag = 1`) or seasonal (`lag = s`) differences.
pub fn difference(x: &[f64], lag: usize, times: usize) -> Result<Vec<f64>> {
    if lag == 0 {
        return invalid("the difference lag must be at least 1");
    }
    let mut v = x.to_vec();
    for _ in 0..times {
        if v.len() <= lag {
            return invalid("the series is too short for this much differencing");
        }
        v = (lag..v.len()).map(|t| v[t] - v[t - lag]).collect();
    }
    Ok(v)
}
