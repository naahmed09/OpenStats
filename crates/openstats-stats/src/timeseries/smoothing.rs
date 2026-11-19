//! Exponential smoothing: single, double (Holt) and seasonal (Holt–Winters) methods.
//!
//! Recursions (m = period, `ŷₜ = (ℓₜ₋₁ + bₜ₋₁)·sₜ₋ₘ` multiplicative, `+` additive):
//!
//! ```text
//! ℓₜ = α(yₜ − sₜ₋ₘ) + (1 − α)(ℓₜ₋₁ + bₜ₋₁)        (multiplicative: α yₜ/sₜ₋ₘ)
//! bₜ = β(ℓₜ − ℓₜ₋₁) + (1 − β) bₜ₋₁
//! sₜ = γ(yₜ − ℓₜ) + (1 − γ) sₜ₋ₘ                    (multiplicative: γ yₜ/ℓₜ)
//! ```
//!
//! Unspecified smoothing weights are chosen to minimise the sum of squared one-step errors
//! (multi-start Nelder–Mead on a logistic transform of the weights, which keeps them in (0, 1)).
//! Prediction intervals use the exact variance weights of the additive-error model; for
//! multiplicative seasonality they are the same weights applied on the relative scale (an
//! approximation, documented in the phase log).

use super::{accuracy, check_series, make_forecast, Accuracy, Forecast};
use crate::error::{invalid, Result};
use openstats_math::optim::{nelder_mead, NelderMeadOptions};

/// Seasonal component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seasonality {
    /// None.
    None,
    /// Additive seasonal terms.
    Additive,
    /// Multiplicative seasonal factors (the data must be positive).
    Multiplicative,
}

/// Which components the model has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmoothingSpec {
    /// Linear trend (Holt).
    pub trend: bool,
    /// Seasonal component.
    pub seasonal: Seasonality,
    /// Seasonal period (ignored without seasonality).
    pub period: usize,
}

impl SmoothingSpec {
    /// Single exponential smoothing.
    pub fn single() -> SmoothingSpec {
        SmoothingSpec { trend: false, seasonal: Seasonality::None, period: 1 }
    }

    /// Double (Holt linear) smoothing.
    pub fn double() -> SmoothingSpec {
        SmoothingSpec { trend: true, seasonal: Seasonality::None, period: 1 }
    }

    /// Winters' method with a trend.
    pub fn winters(period: usize, seasonal: Seasonality) -> SmoothingSpec {
        SmoothingSpec { trend: true, seasonal, period }
    }

    /// Display name.
    pub fn name(&self) -> String {
        match (self.trend, self.seasonal) {
            (false, Seasonality::None) => "Single exponential smoothing".into(),
            (true, Seasonality::None) => "Double exponential smoothing (Holt)".into(),
            (t, s) => format!(
                "{}{} Winters' method (period {})",
                if t { "" } else { "No-trend " },
                if s == Seasonality::Additive { "Additive" } else { "Multiplicative" },
                self.period
            ),
        }
    }

    fn n_params(&self) -> usize {
        1 + usize::from(self.trend) + usize::from(self.seasonal != Seasonality::None)
    }
}

/// Options of a smoothing fit.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SmoothingOptions {
    /// Level weight α; `None` = optimise.
    pub alpha: Option<f64>,
    /// Trend weight β; `None` = optimise.
    pub beta: Option<f64>,
    /// Seasonal weight γ; `None` = optimise.
    pub gamma: Option<f64>,
}

/// A fitted smoothing model.
#[derive(Debug, Clone)]
pub struct SmoothingFit {
    /// Model components.
    pub spec: SmoothingSpec,
    /// Level weight.
    pub alpha: f64,
    /// Trend weight (when the model has a trend).
    pub beta: Option<f64>,
    /// Seasonal weight (when the model is seasonal).
    pub gamma: Option<f64>,
    /// The series.
    pub y: Vec<f64>,
    /// One-step fitted values.
    pub fitted: Vec<f64>,
    /// `y − fitted`.
    pub residuals: Vec<f64>,
    /// Smoothed level after each observation.
    pub level: Vec<f64>,
    /// Smoothed trend after each observation (empty without a trend).
    pub trend: Vec<f64>,
    /// Seasonal terms in force at each observation (`s` used to fit it; empty without seasonality).
    pub seasonal: Vec<f64>,
    /// Initial level, trend and seasonal terms.
    pub initial: Initial,
    /// Last `period` seasonal terms, oldest first (`sₙ₋ₘ₊₁ … sₙ`).
    pub final_seasonal: Vec<f64>,
    /// Sum of squared one-step errors.
    pub sse: f64,
    /// Accuracy measures.
    pub accuracy: Accuracy,
    /// Residual standard deviation (additive scale) or relative one (multiplicative seasonal).
    pub sigma: f64,
    /// Approximate AICc: `n ln(SSE/n) + 2k + 2k(k+1)/(n−k−1)` with `k` = weights + initial states.
    pub aicc: f64,
}

/// Initial states.
#[derive(Debug, Clone, PartialEq)]
pub struct Initial {
    /// Level before the first observation.
    pub level: f64,
    /// Trend before the first observation.
    pub trend: f64,
    /// Seasonal terms for positions 1 … m.
    pub seasonal: Vec<f64>,
}

fn ols_line(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let xm = (n - 1.0) / 2.0;
    let ym = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, v) in y.iter().enumerate() {
        sxy += (i as f64 - xm) * (v - ym);
        sxx += (i as f64 - xm).powi(2);
    }
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (ym - b * xm, b)
}

fn initial_states(y: &[f64], spec: &SmoothingSpec) -> Initial {
    let n = y.len();
    let m = spec.period;
    let seasonal = spec.seasonal != Seasonality::None;
    if !seasonal {
        if spec.trend {
            let k = n.clamp(2, 10);
            let (a, b) = ols_line(&y[..k]);
            return Initial { level: a - b, trend: b, seasonal: Vec::new() };
        }
        let k = n.min(6);
        return Initial { level: y[..k].iter().sum::<f64>() / k as f64, trend: 0.0, seasonal: Vec::new() };
    }
    let seasons = (n / m).clamp(1, 3);
    // Trend: slope between the season means (needs two seasons), else 0.
    let mean_of = |i: usize| y[i * m..(i + 1) * m].iter().sum::<f64>() / m as f64;
    let first = mean_of(0);
    let slope = if seasons >= 2 { (mean_of(seasons - 1) - first) / ((seasons - 1) * m) as f64 } else { 0.0 };
    let slope = if spec.trend { slope } else { 0.0 };
    // Level before the first observation (time −1): the first season's mean sits at time (m−1)/2.
    let level = first - slope * ((m as f64 - 1.0) / 2.0 + 1.0);
    let mult = spec.seasonal == Seasonality::Multiplicative;
    let mut s = vec![0.0; m];
    for j in 0..m {
        let mut acc = 0.0;
        for i in 0..seasons {
            let t = i * m + j;
            let base = level + slope * (t as f64 + 1.0);
            acc += if mult { y[t] / base } else { y[t] - base };
        }
        s[j] = acc / seasons as f64;
    }
    let adj = s.iter().sum::<f64>() / m as f64;
    for v in &mut s {
        if mult {
            *v /= adj;
        } else {
            *v -= adj;
        }
    }
    Initial { level, trend: slope, seasonal: s }
}

struct Run {
    fitted: Vec<f64>,
    level: Vec<f64>,
    trend: Vec<f64>,
    seasonal: Vec<f64>,
    final_seasonal: Vec<f64>,
    sse: f64,
}

fn run(y: &[f64], spec: &SmoothingSpec, init: &Initial, a: f64, b: f64, g: f64) -> Run {
    let n = y.len();
    let m = spec.period;
    let seasonal = spec.seasonal != Seasonality::None;
    let mult = spec.seasonal == Seasonality::Multiplicative;
    let mut l = init.level;
    let mut tr = init.trend;
    let mut s: Vec<f64> = if seasonal { init.seasonal.clone() } else { Vec::new() };
    let mut out = Run {
        fitted: Vec::with_capacity(n),
        level: Vec::with_capacity(n),
        trend: Vec::new(),
        seasonal: Vec::new(),
        final_seasonal: Vec::new(),
        sse: 0.0,
    };
    for t in 0..n {
        let st = if seasonal {
            s[t % m]
        } else if mult {
            1.0
        } else {
            0.0
        };
        let base = l + if spec.trend { tr } else { 0.0 };
        let f = if mult { base * st } else { base + st };
        out.fitted.push(f);
        out.sse += (y[t] - f) * (y[t] - f);
        let deseason = if mult { y[t] / st } else { y[t] - st };
        let l_new = a * deseason + (1.0 - a) * base;
        if spec.trend {
            tr = b * (l_new - l) + (1.0 - b) * tr;
            out.trend.push(tr);
        }
        if seasonal {
            out.seasonal.push(st);
            let s_new = if mult { g * y[t] / l_new + (1.0 - g) * st } else { g * (y[t] - l_new) + (1.0 - g) * st };
            s[t % m] = s_new;
        }
        l = l_new;
        out.level.push(l);
    }
    if seasonal {
        // Oldest first: positions n..n+m−1 modulo m hold sₙ₋ₘ₊₁ … sₙ.
        out.final_seasonal = (0..m).map(|j| s[(n + j) % m]).collect();
    }
    out
}

fn logistic(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn check_weight(name: &str, v: Option<f64>) -> Result<()> {
    if let Some(x) = v {
        if !(x > 0.0 && x < 1.0) {
            return invalid(format!("{name} must be strictly between 0 and 1"));
        }
    }
    Ok(())
}

/// Fit a smoothing model.
pub fn fit_smoothing(y: &[f64], spec: &SmoothingSpec, opts: &SmoothingOptions) -> Result<SmoothingFit> {
    let seasonal = spec.seasonal != Seasonality::None;
    if seasonal && spec.period < 2 {
        return invalid("the seasonal period must be at least 2");
    }
    let min = if seasonal { 2 * spec.period + 1 } else { 4 };
    check_series(y, min, "exponential smoothing")?;
    if spec.seasonal == Seasonality::Multiplicative && y.iter().any(|&v| v <= 0.0) {
        return invalid("multiplicative seasonality requires every value to be positive");
    }
    check_weight("alpha", opts.alpha)?;
    check_weight("beta", opts.beta)?;
    check_weight("gamma", opts.gamma)?;
    let init = initial_states(y, spec);
    // Free weights, in the order α, β, γ.
    let fixed =
        [opts.alpha, if spec.trend { opts.beta } else { Some(0.0) }, if seasonal { opts.gamma } else { Some(0.0) }];
    let free: Vec<usize> = (0..3).filter(|&i| fixed[i].is_none()).collect();
    let build = |u: &[f64]| -> [f64; 3] {
        let mut w = [fixed[0].unwrap_or(0.0), fixed[1].unwrap_or(0.0), fixed[2].unwrap_or(0.0)];
        for (k, &i) in free.iter().enumerate() {
            w[i] = logistic(u[k]).clamp(1e-6, 1.0 - 1e-6);
        }
        w
    };
    let w = if free.is_empty() {
        build(&[])
    } else {
        let starts: [f64; 3] = [0.1, 0.3, 0.7];
        let mut best: Option<(f64, Vec<f64>)> = None;
        for &s0 in &starts {
            let x0: Vec<f64> = free.iter().map(|&i| logit(if i == 0 { s0 } else { s0 / 3.0 + 0.02 })).collect();
            let mut f = |u: &[f64]| {
                let w = build(u);
                let r = run(y, spec, &init, w[0], w[1], w[2]);
                if r.sse.is_finite() {
                    r.sse
                } else {
                    f64::MAX
                }
            };
            let m = nelder_mead(&mut f, &x0, &NelderMeadOptions::default())?;
            if best.as_ref().is_none_or(|b| m.f < b.0) {
                best = Some((m.f, m.x));
            }
        }
        build(&best.map(|b| b.1).unwrap_or_default())
    };
    let r = run(y, spec, &init, w[0], w[1], w[2]);
    let n = y.len();
    let residuals: Vec<f64> = y.iter().zip(&r.fitted).map(|(a, b)| a - b).collect();
    let mult = spec.seasonal == Seasonality::Multiplicative;
    let sigma = if mult {
        let rel: f64 = y.iter().zip(&r.fitted).map(|(a, f)| ((a - f) / f).powi(2)).sum();
        (rel / n as f64).sqrt()
    } else {
        (r.sse / n as f64).sqrt()
    };
    let k = (spec.n_params() + 1 + spec.period * usize::from(seasonal) + usize::from(spec.trend)) as f64;
    let nf = n as f64;
    let aicc = if nf - k - 1.0 > 0.0 {
        nf * (r.sse.max(1e-300) / nf).ln() + 2.0 * k + 2.0 * k * (k + 1.0) / (nf - k - 1.0)
    } else {
        f64::INFINITY
    };
    Ok(SmoothingFit {
        spec: *spec,
        alpha: w[0],
        beta: spec.trend.then_some(w[1]),
        gamma: seasonal.then_some(w[2]),
        y: y.to_vec(),
        accuracy: accuracy(y, &r.fitted),
        fitted: r.fitted,
        residuals,
        level: r.level,
        trend: r.trend,
        seasonal: r.seasonal,
        initial: init,
        final_seasonal: r.final_seasonal,
        sse: r.sse,
        sigma,
        aicc,
    })
}

impl SmoothingFit {
    /// Forecasts for horizons 1 … `h` with `level` prediction intervals.
    pub fn forecast(&self, h: usize, level: f64) -> Result<Forecast> {
        if h == 0 {
            return invalid("the forecast horizon must be at least 1");
        }
        let n = self.y.len();
        let l = *self.level.last().unwrap_or(&self.initial.level);
        let b = self.trend.last().copied().unwrap_or(0.0);
        let m = self.spec.period;
        let mult = self.spec.seasonal == Seasonality::Multiplicative;
        let seasonal = self.spec.seasonal != Seasonality::None;
        let mut point = Vec::with_capacity(h);
        for j in 1..=h {
            let base = l + j as f64 * b;
            let s = if seasonal {
                self.final_seasonal[(j - 1) % m]
            } else if mult {
                1.0
            } else {
                0.0
            };
            point.push(if mult { base * s } else { base + s });
        }
        // Variance weights c_k of the additive-error form.
        let (a, bt, g) = (self.alpha, self.beta.unwrap_or(0.0), self.gamma.unwrap_or(0.0));
        let mut cum = 0.0;
        let mut se = Vec::with_capacity(h);
        for j in 1..=h {
            if j >= 2 {
                let k = (j - 1) as f64;
                let mut c = a * (1.0 + if self.spec.trend { k * bt } else { 0.0 });
                if seasonal && (j - 1) % m == 0 {
                    c += g * (1.0 - a);
                }
                cum += c * c;
            }
            let scale = (1.0 + cum).sqrt();
            se.push(if mult { self.sigma * point[j - 1].abs() * scale } else { self.sigma * scale });
        }
        make_forecast(n - 1, point, se, level)
    }
}

/// Fit every sensible smoothing model for the series (single, double and, when `period` ≥ 2
/// and the series covers two seasons, additive and multiplicative Winters models with and
/// without trend) and return them ordered by approximate AICc.
pub fn select_smoothing(y: &[f64], period: Option<usize>) -> Result<Vec<SmoothingFit>> {
    check_series(y, 4, "model selection")?;
    let mut specs = vec![SmoothingSpec::single(), SmoothingSpec::double()];
    if let Some(m) = period.filter(|&m| m >= 2 && y.len() > 2 * m) {
        for seasonal in [Seasonality::Additive, Seasonality::Multiplicative] {
            if seasonal == Seasonality::Multiplicative && y.iter().any(|&v| v <= 0.0) {
                continue;
            }
            for trend in [true, false] {
                specs.push(SmoothingSpec { trend, seasonal, period: m });
            }
        }
    }
    let mut fits: Vec<SmoothingFit> =
        specs.iter().filter_map(|s| fit_smoothing(y, s, &SmoothingOptions::default()).ok()).collect();
    if fits.is_empty() {
        return invalid("no smoothing model could be fitted");
    }
    fits.sort_by(|a, b| a.aicc.partial_cmp(&b.aicc).unwrap_or(std::cmp::Ordering::Equal));
    Ok(fits)
}
