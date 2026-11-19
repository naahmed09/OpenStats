//! Time series decomposition into trend, seasonal and irregular components.
//!
//! The trend is either a regression fitted to the data (`Linear`, `Quadratic`, `Exponential`
//! growth, or a constant) or the centred moving average of one full period. The data divided
//! (multiplicative) or differenced (additive) by the trend are averaged by season to give the
//! seasonal indices, which are scaled to average 1 (multiplicative) or sum to 0 (additive).
//! `fitted = trend × seasonal` (or `+`) and the irregular component is what remains.
//!
//! Forecasts extend the trend (regression: its formula; moving average: a least-squares line
//! through the trend values) and repeat the seasonal indices. Prediction intervals use the
//! residual standard deviation and, for regression trends, the leverage of the forecast point.

use super::{accuracy, check_series, make_forecast, Accuracy, Forecast};
use crate::error::{invalid, Result};
use openstats_math::linalg::{Matrix, Qr};

/// Additive or multiplicative combination of the components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionModel {
    /// `y = trend + seasonal + irregular`.
    Additive,
    /// `y = trend × seasonal × irregular` (positive data).
    Multiplicative,
}

/// How the trend component is obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendKind {
    /// No trend (the overall mean).
    Constant,
    /// `a + bt`.
    Linear,
    /// `a + bt + ct²`.
    Quadratic,
    /// `a·e^{bt}` (fitted to the logarithm; positive data).
    Exponential,
    /// Centred moving average of one period (`2 × m` for even `m`); the ends are missing.
    MovingAverage,
}

/// Options of a decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct DecomposeOptions {
    /// Combination of components.
    pub model: DecompositionModel,
    /// Trend.
    pub trend: TrendKind,
    /// Seasonal period (1 = no seasonality).
    pub period: usize,
    /// Position of the first observation in its season (1-based), so indices are labelled correctly.
    pub first_season: usize,
}

impl DecomposeOptions {
    /// Defaults: multiplicative, linear trend.
    pub fn new(period: usize) -> DecomposeOptions {
        DecomposeOptions {
            model: DecompositionModel::Multiplicative,
            trend: TrendKind::Linear,
            period,
            first_season: 1,
        }
    }
}

/// A decomposition.
#[derive(Debug, Clone)]
pub struct Decomposition {
    /// The options used.
    pub opts: DecomposeOptions,
    /// The series.
    pub y: Vec<f64>,
    /// Trend component (`NaN` at the ends for a moving average).
    pub trend: Vec<f64>,
    /// Data with the trend removed.
    pub detrended: Vec<f64>,
    /// Seasonal indices for seasons 1 … m.
    pub indices: Vec<f64>,
    /// Seasonal component at every observation.
    pub seasonal: Vec<f64>,
    /// Data with the seasonal component removed.
    pub deseasonalized: Vec<f64>,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Irregular component (`y − fitted`, or `y / fitted` multiplicative).
    pub irregular: Vec<f64>,
    /// Residuals `y − fitted`.
    pub residuals: Vec<f64>,
    /// Trend coefficients (`a`, `b`, `c`) for regression trends.
    pub coef: Vec<f64>,
    /// Residual standard deviation (relative for multiplicative models).
    pub sigma: f64,
    /// Accuracy measures.
    pub accuracy: Accuracy,
    /// `(XᵀX)⁻¹` of the trend regression (for forecast leverage), row-major.
    xtx_inv: Vec<Vec<f64>>,
    /// Line through the moving-average trend: intercept and slope against `t` (0-based).
    ma_line: Option<(f64, f64)>,
}

fn basis(kind: TrendKind, t: f64) -> Vec<f64> {
    match kind {
        TrendKind::Constant => vec![1.0],
        TrendKind::Linear | TrendKind::Exponential => vec![1.0, t],
        TrendKind::Quadratic => vec![1.0, t, t * t],
        TrendKind::MovingAverage => vec![],
    }
}

fn regress(x: &[Vec<f64>], y: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = x.len();
    let k = x[0].len();
    let m = Matrix::from_fn(n, k, |i, j| x[i][j]);
    let qr = Qr::new(&m)?;
    let beta = qr.solve_least_squares(y)?;
    // (XᵀX)⁻¹ by normal equations on the small k × k system via the same solver.
    let mut inv = vec![vec![0.0; k]; k];
    let g = Matrix::from_fn(k, k, |i, j| (0..n).map(|r| x[r][i] * x[r][j]).sum());
    let lu = openstats_math::linalg::Lu::new(&g)?;
    for j in 0..k {
        let mut e = vec![0.0; k];
        e[j] = 1.0;
        let col = lu.solve(&e)?;
        for i in 0..k {
            inv[i][j] = col[i];
        }
    }
    Ok((beta, inv))
}

/// Decompose a series.
pub fn decompose(y: &[f64], opts: &DecomposeOptions) -> Result<Decomposition> {
    let m = opts.period.max(1);
    let n = y.len();
    let need = if m > 1 { 2 * m } else { 4 };
    check_series(y, need.max(4), "decomposition")?;
    let mult = opts.model == DecompositionModel::Multiplicative;
    if (mult || opts.trend == TrendKind::Exponential) && y.iter().any(|&v| v <= 0.0) {
        return invalid("this decomposition needs every value to be positive");
    }
    if opts.first_season == 0 || opts.first_season > m {
        return invalid("the first observation's season must be between 1 and the period");
    }
    let mut coef = Vec::new();
    let mut xtx_inv = Vec::new();
    let mut ma_line = None;
    let trend: Vec<f64> = match opts.trend {
        TrendKind::MovingAverage => {
            if m < 2 {
                return invalid("a moving-average trend needs a seasonal period of at least 2");
            }
            let mut t = vec![f64::NAN; n];
            let half = m / 2;
            if m % 2 == 1 {
                for i in half..n - half {
                    t[i] = y[i - half..=i + half].iter().sum::<f64>() / m as f64;
                }
            } else {
                for i in half..n - half {
                    let inner: f64 = y[i - half + 1..i + half].iter().sum();
                    t[i] = (inner + 0.5 * (y[i - half] + y[i + half])) / m as f64;
                }
            }
            let pts: Vec<(f64, f64)> =
                t.iter().enumerate().filter(|(_, v)| !v.is_nan()).map(|(i, v)| (i as f64, *v)).collect();
            let k = pts.len() as f64;
            let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / k, pts.iter().map(|p| p.1).sum::<f64>() / k);
            let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
            let b = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / sxx;
            ma_line = Some((my - b * mx, b));
            t
        }
        kind => {
            let x: Vec<Vec<f64>> = (0..n).map(|i| basis(kind, (i + 1) as f64)).collect();
            let resp: Vec<f64> =
                if kind == TrendKind::Exponential { y.iter().map(|v| v.ln()).collect() } else { y.to_vec() };
            let (beta, inv) = regress(&x, &resp)?;
            let tr = x
                .iter()
                .map(|r| {
                    let lin: f64 = r.iter().zip(&beta).map(|(a, b)| a * b).sum();
                    if kind == TrendKind::Exponential {
                        lin.exp()
                    } else {
                        lin
                    }
                })
                .collect();
            coef = beta;
            xtx_inv = inv;
            tr
        }
    };
    if (mult || opts.trend == TrendKind::Exponential) && trend.iter().any(|&v| !v.is_nan() && v <= 0.0) {
        return invalid("the fitted trend is not positive; a multiplicative model does not suit these data");
    }
    let detrended: Vec<f64> = y
        .iter()
        .zip(&trend)
        .map(|(a, t)| {
            if t.is_nan() {
                f64::NAN
            } else if mult {
                a / t
            } else {
                a - t
            }
        })
        .collect();
    // Seasonal indices.
    let season_of = |i: usize| (i + opts.first_season - 1) % m;
    let mut indices = vec![0.0; m];
    if m > 1 {
        let mut sums = vec![0.0; m];
        let mut cnt = vec![0usize; m];
        for (i, d) in detrended.iter().enumerate() {
            if !d.is_nan() {
                sums[season_of(i)] += d;
                cnt[season_of(i)] += 1;
            }
        }
        if cnt.contains(&0) {
            return invalid("some season has no usable observation");
        }
        for j in 0..m {
            indices[j] = sums[j] / cnt[j] as f64;
        }
        let mean = indices.iter().sum::<f64>() / m as f64;
        for v in &mut indices {
            if mult {
                *v /= mean;
            } else {
                *v -= mean;
            }
        }
    } else if mult {
        indices[0] = 1.0;
    }
    let seasonal: Vec<f64> = (0..n).map(|i| indices[season_of(i)]).collect();
    let deseasonalized: Vec<f64> = y.iter().zip(&seasonal).map(|(a, s)| if mult { a / s } else { a - s }).collect();
    let fitted: Vec<f64> = trend
        .iter()
        .zip(&seasonal)
        .map(|(t, s)| {
            if t.is_nan() {
                f64::NAN
            } else if mult {
                t * s
            } else {
                t + s
            }
        })
        .collect();
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, f)| a - f).collect();
    let irregular: Vec<f64> = y.iter().zip(&fitted).map(|(a, f)| if mult { a / f } else { a - f }).collect();
    let valid: Vec<f64> = if mult {
        irregular.iter().copied().filter(|v| !v.is_nan()).map(|v| v - 1.0).collect()
    } else {
        residuals.iter().copied().filter(|v| !v.is_nan()).collect()
    };
    let df = (valid.len() as f64 - coef.len() as f64 - (m as f64 - 1.0)).max(1.0);
    let sigma = (valid.iter().map(|v| v * v).sum::<f64>() / df).sqrt();
    let acc = accuracy(y, &fitted);
    Ok(Decomposition {
        opts: opts.clone(),
        y: y.to_vec(),
        trend,
        detrended,
        indices,
        seasonal,
        deseasonalized,
        fitted,
        irregular,
        residuals,
        coef,
        sigma,
        accuracy: acc,
        xtx_inv,
        ma_line,
    })
}

impl Decomposition {
    /// Forecasts for the next `h` periods with `level` prediction intervals.
    pub fn forecast(&self, h: usize, level: f64) -> Result<Forecast> {
        if h == 0 {
            return invalid("the forecast horizon must be at least 1");
        }
        let n = self.y.len();
        let m = self.opts.period.max(1);
        let mult = self.opts.model == DecompositionModel::Multiplicative;
        let mut point = Vec::with_capacity(h);
        let mut se = Vec::with_capacity(h);
        for j in 1..=h {
            let i = n + j - 1;
            let (tr, lev) = match self.opts.trend {
                TrendKind::MovingAverage => {
                    let (a, b) = self.ma_line.unwrap_or((0.0, 0.0));
                    (a + b * i as f64, 0.0)
                }
                kind => {
                    let x = basis(kind, (i + 1) as f64);
                    let lin: f64 = x.iter().zip(&self.coef).map(|(a, b)| a * b).sum();
                    let lv: f64 =
                        (0..x.len()).map(|r| (0..x.len()).map(|c| x[r] * self.xtx_inv[r][c] * x[c]).sum::<f64>()).sum();
                    (if kind == TrendKind::Exponential { lin.exp() } else { lin }, lv)
                }
            };
            let s = self.indices[(i + self.opts.first_season - 1) % m];
            let f = if mult { tr * s } else { tr + s };
            point.push(f);
            let scale = (1.0 + lev).sqrt();
            se.push(if mult { self.sigma * f.abs() * scale } else { self.sigma * scale });
        }
        make_forecast(n - 1, point, se, level)
    }
}
