//! Time series: autocorrelation, exponential smoothing, seasonal ARIMA and decomposition, each
//! with forecasts and prediction intervals.
//!
//! * [`mod@acf`] — ACF with Bartlett standard errors, PACF (Durbin–Levinson), Ljung–Box Q, differencing.
//! * [`smoothing`] — single, double (Holt) and seasonal (Holt–Winters, additive and
//!   multiplicative) exponential smoothing with least-squares parameter search.
//! * [`arima`] — ARIMA(p,d,q)×(P,D,Q)ₛ by conditional least squares, exact Gaussian maximum
//!   likelihood (Kalman filter) or both in sequence.
//! * [`mod@decompose`] — trend, seasonal and irregular components (regression trend or centred
//!   moving average; additive or multiplicative) and decomposition forecasts.
//!
//! Missing values are not allowed in a series (`NaN` is rejected); the caller decides how to
//! treat them. Series are in time order, one value per period.

pub mod acf;
pub mod arima;
pub mod decompose;
pub mod report_impl;
pub mod smoothing;
pub mod unitroot;

pub use acf::{acf, difference, ljung_box, pacf, AcfResult, PacfResult};
pub use arima::{fit_arima, rank_arima, ArimaFit, ArimaMethod, ArimaOptions, ArimaSpec, CoefRow, RankedArima};
pub use decompose::{decompose, DecomposeOptions, Decomposition, DecompositionModel, TrendKind};
pub use smoothing::{fit_smoothing, select_smoothing, Seasonality, SmoothingFit, SmoothingOptions, SmoothingSpec};
pub use unitroot::{
    adf_test, auto_arima, cross_correlation, kpss_test, AdfOptions, AdfTest, AutoArima, CrossCorrelation,
    Deterministic, KpssTest,
};

use crate::error::{invalid, Result, StatsError};

/// Forecasts with prediction intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    /// Index of the last observation (forecast `h` is for period `origin + h`, 1-based).
    pub origin: usize,
    /// Point forecasts for horizons 1, 2, ….
    pub point: Vec<f64>,
    /// Lower prediction limits.
    pub lower: Vec<f64>,
    /// Upper prediction limits.
    pub upper: Vec<f64>,
    /// Standard deviation of each forecast error.
    pub se: Vec<f64>,
    /// Confidence level of the interval.
    pub level: f64,
}

/// Measures of accuracy of one-step fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    /// Number of fitted values used.
    pub n: usize,
    /// Mean absolute percentage error, percent (observations equal to 0 are skipped); `NaN` when none.
    pub mape: f64,
    /// Mean absolute deviation.
    pub mad: f64,
    /// Mean squared deviation.
    pub msd: f64,
}

/// Accuracy measures of `fitted` against `y` over the positions where `fitted` is a number.
pub fn accuracy(y: &[f64], fitted: &[f64]) -> Accuracy {
    let (mut n, mut sa, mut ss) = (0usize, 0.0, 0.0);
    let (mut np, mut sp) = (0usize, 0.0);
    for (&a, &f) in y.iter().zip(fitted) {
        if f.is_nan() {
            continue;
        }
        let e = a - f;
        n += 1;
        sa += e.abs();
        ss += e * e;
        if a != 0.0 {
            np += 1;
            sp += (e / a).abs();
        }
    }
    let nf = n.max(1) as f64;
    Accuracy { n, mape: if np == 0 { f64::NAN } else { 100.0 * sp / np as f64 }, mad: sa / nf, msd: ss / nf }
}

/// Reject empty, non-finite or missing series.
pub(crate) fn check_series(y: &[f64], min: usize, what: &'static str) -> Result<()> {
    if y.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    if y.iter().any(|v| v.is_nan()) {
        return invalid(format!("{what}: the series has missing values; time series methods need a complete series"));
    }
    if y.len() < min {
        return Err(StatsError::InsufficientData { what, needed: min, got: y.len() });
    }
    Ok(())
}

/// Two-sided standard normal multiplier for a confidence level.
pub(crate) fn z_for(level: f64) -> Result<f64> {
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    Ok(openstats_math::special::erf::norm_ppf(0.5 + level / 2.0))
}

/// Forecast skeleton from point forecasts and standard errors.
pub(crate) fn make_forecast(origin: usize, point: Vec<f64>, se: Vec<f64>, level: f64) -> Result<Forecast> {
    let z = z_for(level)?;
    let lower = point.iter().zip(&se).map(|(p, s)| p - z * s).collect();
    let upper = point.iter().zip(&se).map(|(p, s)| p + z * s).collect();
    Ok(Forecast { origin, point, lower, upper, se, level })
}
