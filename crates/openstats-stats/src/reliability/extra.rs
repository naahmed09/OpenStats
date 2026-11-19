//! Further life-data models: three-parameter (threshold) distributions, the gamma
//! distribution, the Turnbull non-parametric estimate for interval data and the Eyring
//! accelerated-life relationship.

use super::alt::{fit_alt, AltFit, Relationship, StressVariable};
use super::data::{LifeData, Obs, ObsKind};
use super::dist::Family;
use super::fit::{fit_distribution, zcrit, Estimate, LifeFit};
use crate::error::{invalid, Result};
use openstats_math::dist::{ChiSquared, ContinuousDistribution, Gamma};
use openstats_math::linalg::Cholesky;
use openstats_math::optim::{hessian_fd, nelder_mead, NelderMeadOptions};
use openstats_math::Matrix;

// ---------------------------------------------------------------------------------------------
// Three-parameter families
// ---------------------------------------------------------------------------------------------

/// A threshold (three-parameter) fit: `T = γ + X` with `X` following the two-parameter family.
#[derive(Debug, Clone)]
pub struct ThresholdFit {
    /// Threshold γ.
    pub threshold: f64,
    /// Profile-likelihood confidence limits for γ (`−∞` when the lower bound is not reached).
    pub threshold_lower: f64,
    /// Upper confidence limit (the smallest observed time bounds it).
    pub threshold_upper: f64,
    /// The two-parameter fit of `T − γ`.
    pub fit: LifeFit,
    /// Log-likelihood.
    pub loglik: f64,
    /// `true` when the profile has an interior maximum.
    pub interior: bool,
}

impl ThresholdFit {
    /// Reliability at `t`.
    pub fn survival(&self, t: f64) -> f64 {
        if t <= self.threshold {
            1.0
        } else {
            self.fit.survival(t - self.threshold)
        }
    }
    /// Percentile `p` (fraction failing).
    pub fn percentile(&self, p: f64) -> f64 {
        self.threshold + self.fit.percentile(p)
    }
    /// Mean life.
    pub fn mean(&self) -> f64 {
        self.threshold + self.fit.mean()
    }
}

fn shifted(data: &LifeData, g: f64) -> LifeData {
    LifeData {
        obs: data
            .obs
            .iter()
            .map(|o| Obs {
                lower: if o.lower <= 0.0 { 0.0 } else { o.lower - g },
                upper: if o.upper.is_finite() { o.upper - g } else { o.upper },
                weight: o.weight,
                row: o.row,
            })
            .collect(),
        n_skipped: data.n_skipped,
    }
}

/// Fit a Weibull, lognormal or log-logistic distribution with a threshold parameter by
/// profiling the likelihood over γ.
pub fn fit_threshold(data: &LifeData, family: Family, level: f64) -> Result<ThresholdFit> {
    if !family.is_log() || family.fixed_scale() {
        return invalid("a threshold applies to the Weibull, lognormal and log-logistic families");
    }
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;
    for o in &data.obs {
        let first = if o.lower > 0.0 { o.lower } else { o.upper };
        min_t = min_t.min(first);
        let last = if o.upper.is_finite() { o.upper } else { o.lower };
        max_t = max_t.max(last);
    }
    if data.failures() < 3.0 || !(max_t > min_t) {
        return invalid("a threshold fit needs at least three failures at different times");
    }
    let span = max_t - min_t;
    let gmax = min_t - 1e-6 * span;
    let ll = |d: f64| -> f64 {
        // d = gmax − γ > 0
        fit_distribution(&shifted(data, gmax - d), family).map_or(f64::NEG_INFINITY, |f| f.loglik)
    };
    let grid: Vec<f64> =
        (0..48).map(|k| span * (1e-4_f64.ln() + (3.0_f64.ln() - 1e-4_f64.ln()) * k as f64 / 47.0).exp()).collect();
    let vals: Vec<f64> = grid.iter().map(|&d| ll(d)).collect();
    let (kb, _) = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .fold((usize::MAX, f64::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
    if kb == usize::MAX {
        return invalid("the likelihood could not be evaluated for any threshold");
    }
    let interior = kb + 1 < grid.len();
    let (lo, hi) = (grid[kb.saturating_sub(1)].ln(), grid[(kb + 1).min(grid.len() - 1)].ln());
    let best = openstats_math::optim::brent_minimize(|x| -ll(x.exp()), lo, hi, 1e-9, 200)?;
    let d_hat = best.x.exp();
    let ll_hat = -best.f;
    let g_hat = gmax - d_hat;
    let fit = fit_distribution(&shifted(data, g_hat), family)?;
    // Profile-likelihood limits: ll(d) = ll_hat − χ²₁/2.
    let target = ll_hat - ChiSquared::new(1.0)?.quantile(level) / 2.0;
    let cross = |from: f64, to: f64| -> Option<f64> {
        let (mut a, mut b) = (from.ln(), to.ln());
        let (fa, fb) = (ll(a.exp()) - target, ll(b.exp()) - target);
        if !(fa >= 0.0) || fb.is_nan() || fb >= 0.0 {
            return None;
        }
        for _ in 0..80 {
            let m = 0.5 * (a + b);
            if ll(m.exp()) - target >= 0.0 {
                a = m;
            } else {
                b = m;
            }
        }
        Some((0.5 * (a + b)).exp())
    };
    // larger d ⇒ smaller γ (lower limit); smaller d ⇒ larger γ (upper limit)
    let d_big = cross(d_hat, grid[grid.len() - 1]);
    let d_small = cross(d_hat, grid[0]);
    Ok(ThresholdFit {
        threshold: g_hat,
        threshold_lower: d_big.map_or(f64::NEG_INFINITY, |d| gmax - d),
        threshold_upper: d_small.map_or(min_t, |d| gmax - d),
        fit,
        loglik: ll_hat,
        interior,
    })
}

// ---------------------------------------------------------------------------------------------
// Gamma
// ---------------------------------------------------------------------------------------------

/// A maximum-likelihood gamma fit.
#[derive(Debug, Clone)]
pub struct GammaFit {
    /// Shape α.
    pub shape: f64,
    /// Scale θ.
    pub scale: f64,
    /// Log-likelihood.
    pub loglik: f64,
    /// Units.
    pub n: f64,
    /// Failures.
    pub failures: f64,
    /// Covariance of `(ln α, ln θ)`.
    pub cov: Option<Matrix>,
    /// Optimiser convergence.
    pub converged: bool,
}

fn gamma_negll(data: &LifeData, ln_shape: f64, ln_scale: f64) -> f64 {
    let Ok(g) = Gamma::new(ln_shape.exp(), ln_scale.exp(), 0.0) else {
        return f64::INFINITY;
    };
    let mut s = 0.0;
    for o in &data.obs {
        let l = match o.kind() {
            ObsKind::Exact => g.ln_pdf(o.lower),
            ObsKind::Right => g.sf(o.lower).ln(),
            ObsKind::Left => g.cdf(o.upper).ln(),
            ObsKind::Interval => (g.cdf(o.upper) - g.cdf(o.lower)).ln(),
        };
        if !l.is_finite() {
            return f64::INFINITY;
        }
        s -= o.weight * l;
    }
    s
}

/// Fit the two-parameter gamma distribution to censored data.
pub fn fit_gamma(data: &LifeData) -> Result<GammaFit> {
    if data.obs.iter().any(|o| o.kind() == ObsKind::Exact && o.lower <= 0.0) {
        return invalid("the gamma distribution needs positive failure times");
    }
    if data.failures() < 2.0 {
        return invalid("a gamma fit needs at least two failures");
    }
    // moment start from the interval mid-points
    let (mut sw, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for o in &data.obs {
        let m = if o.upper.is_finite() { 0.5 * (o.lower.max(0.0) + o.upper) } else { o.lower * 1.2 };
        sw += o.weight;
        s1 += o.weight * m;
        s2 += o.weight * m * m;
    }
    let mean = s1 / sw;
    let var = (s2 / sw - mean * mean).max(mean * mean * 1e-4);
    let (a0, th0) = (mean * mean / var, var / mean);
    let mut f = |x: &[f64]| gamma_negll(data, x[0], x[1]);
    let nm = NelderMeadOptions { max_iter: 8000, ..NelderMeadOptions::default() };
    let mut best: Option<(Vec<f64>, f64, bool)> = None;
    for (da, dt) in [(0.0, 0.0), (0.7, -0.7), (-0.5, 0.5)] {
        let x0 = [a0.ln() + da, th0.ln() + dt];
        if !f(&x0).is_finite() {
            continue;
        }
        if let Ok(m) = nelder_mead(&mut f, &x0, &nm) {
            if best.as_ref().is_none_or(|b| m.f < b.1) {
                best = Some((m.x, m.f, m.converged));
            }
        }
    }
    let (x, fv, converged) = best.ok_or_else(|| {
        crate::error::StatsError::InvalidArgument("the gamma likelihood could not be maximised".into())
    })?;
    let mut f2 = |p: &[f64]| gamma_negll(data, p[0], p[1]);
    let h = hessian_fd(&mut f2, &x);
    let cov = Cholesky::new(&h).and_then(|c| c.inverse()).ok().filter(|m| m[(0, 0)] > 0.0 && m[(1, 1)] > 0.0);
    Ok(GammaFit {
        shape: x[0].exp(),
        scale: x[1].exp(),
        loglik: -fv,
        n: data.total(),
        failures: data.failures(),
        cov,
        converged,
    })
}

impl GammaFit {
    fn dist(&self) -> Gamma {
        Gamma::new(self.shape, self.scale, 0.0).expect("fitted parameters are valid")
    }
    /// Reliability at `t`.
    pub fn survival(&self, t: f64) -> f64 {
        self.dist().sf(t)
    }
    /// Percentile `p` (fraction failing).
    pub fn percentile(&self, p: f64) -> f64 {
        self.dist().quantile(p)
    }
    /// Mean life `αθ`.
    pub fn mean(&self) -> f64 {
        self.shape * self.scale
    }
    /// Akaike information criterion.
    pub fn aic(&self) -> f64 {
        -2.0 * self.loglik + 4.0
    }
    /// Shape and scale with normal-approximation limits on the log scale.
    pub fn parameters(&self, level: f64) -> Vec<(String, Estimate)> {
        let z = zcrit(level);
        let mk = |name: &str, est: f64, i: usize| {
            let se_ln = self.cov.as_ref().map_or(f64::NAN, |c| c[(i, i)].sqrt());
            (
                name.to_string(),
                Estimate { est, se: est * se_ln, lower: est * (-z * se_ln).exp(), upper: est * (z * se_ln).exp() },
            )
        };
        vec![mk("Shape", self.shape, 0), mk("Scale", self.scale, 1)]
    }
}

// ---------------------------------------------------------------------------------------------
// Turnbull
// ---------------------------------------------------------------------------------------------

/// The Turnbull non-parametric maximum-likelihood estimate.
#[derive(Debug, Clone)]
pub struct Turnbull {
    /// Innermost intervals `[q, p]` that can hold probability mass.
    pub intervals: Vec<(f64, f64)>,
    /// Probability mass of each interval.
    pub mass: Vec<f64>,
    /// `S(p)` after each interval: `1 − Σ mass up to and including it`.
    pub survival: Vec<f64>,
    /// EM iterations used.
    pub iterations: usize,
    /// Log-likelihood.
    pub loglik: f64,
}

impl Turnbull {
    /// Survival estimate at `t`: mass is treated as located at the right end of its interval.
    pub fn survival_at(&self, t: f64) -> f64 {
        1.0 - self.intervals.iter().zip(&self.mass).filter(|((_, p), _)| *p <= t).map(|(_, m)| m).sum::<f64>()
    }
}

/// Turnbull's self-consistency (EM) estimate of the survival function for any mix of exact,
/// right-, left- and interval-censored data.
pub fn turnbull(data: &LifeData, tol: f64, max_iter: usize) -> Result<Turnbull> {
    if data.obs.is_empty() {
        return invalid("no observations");
    }
    // sorted endpoints, left ends before right ends at ties
    let mut pts: Vec<(f64, u8)> = Vec::new();
    for o in &data.obs {
        // an exact time is a closed point; other records are open on the left, so their left end
        // sorts after any right end at the same value
        pts.push((o.lower.max(0.0), if o.kind() == ObsKind::Exact { 0 } else { 2 }));
        pts.push((o.upper, 1));
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut intervals = Vec::new();
    for w in pts.windows(2) {
        if w[0].1 != 1 && w[1].1 == 1 {
            intervals.push((w[0].0, w[1].0));
        }
    }
    intervals.dedup();
    if intervals.is_empty() {
        return invalid("the data define no Turnbull intervals");
    }
    let m = intervals.len();
    // membership by the right end `p` of each interval: (lower, upper] for censored records,
    // exactly `t` for an exact failure
    let inside: Vec<Vec<usize>> = data
        .obs
        .iter()
        .map(|o| {
            (0..m)
                .filter(|&j| {
                    let p = intervals[j].1;
                    if o.kind() == ObsKind::Exact {
                        p == o.lower
                    } else {
                        p > o.lower && p <= o.upper
                    }
                })
                .collect()
        })
        .collect();
    if inside.iter().any(|v| v.is_empty()) {
        return invalid("an observation contains no Turnbull interval");
    }
    let wsum: f64 = data.obs.iter().map(|o| o.weight).sum();
    let mut mass = vec![1.0 / m as f64; m];
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        let mut next = vec![0.0; m];
        for (o, js) in data.obs.iter().zip(&inside) {
            let denom: f64 = js.iter().map(|&j| mass[j]).sum();
            for &j in js {
                next[j] += o.weight * mass[j] / denom;
            }
        }
        let mut delta = 0.0_f64;
        for j in 0..m {
            next[j] /= wsum;
            delta = delta.max((next[j] - mass[j]).abs());
        }
        mass = next;
        if delta < tol {
            break;
        }
    }
    let loglik =
        data.obs.iter().zip(&inside).map(|(o, js)| o.weight * js.iter().map(|&j| mass[j]).sum::<f64>().ln()).sum();
    let mut cum = 0.0;
    let survival = mass
        .iter()
        .map(|p| {
            cum += p;
            (1.0 - cum).max(0.0)
        })
        .collect();
    Ok(Turnbull { intervals, mass, survival, iterations: iters, loglik })
}

// ---------------------------------------------------------------------------------------------
// Eyring
// ---------------------------------------------------------------------------------------------

/// An Eyring accelerated-life fit: `ln t = a − ln T + b/T` with `T` in kelvin.
#[derive(Debug, Clone)]
pub struct EyringFit {
    /// Regression of `ln(t·T)` on `1/T` (coefficients `a`, `b`).
    pub alt: AltFit,
    /// Log-likelihood on the original time scale.
    pub loglik: f64,
}

const KELVIN: f64 = 273.16;

/// Fit the Eyring relationship. `temps` are in °C, one per row of the time column.
pub fn fit_eyring(data: &LifeData, temps: &[f64], family: Family) -> Result<EyringFit> {
    if !family.is_log() {
        return invalid(
            "the Eyring relationship applies to log-time families (Weibull, lognormal, exponential, log-logistic)",
        );
    }
    let tk = |row: usize| -> Result<f64> {
        let t = *temps
            .get(row)
            .ok_or_else(|| crate::error::StatsError::InvalidArgument("temperature column shorter than time".into()))?;
        if t.is_nan() || t + KELVIN <= 0.0 {
            return invalid(format!("row {}: invalid temperature", row + 1));
        }
        Ok(t + KELVIN)
    };
    let mut scaled = Vec::with_capacity(data.obs.len());
    let mut jac = 0.0;
    for o in &data.obs {
        let k = tk(o.row)?;
        scaled.push(Obs {
            lower: o.lower * k,
            upper: if o.upper.is_finite() { o.upper * k } else { o.upper },
            weight: o.weight,
            row: o.row,
        });
        if o.kind() == ObsKind::Exact {
            jac += o.weight * k.ln();
        }
    }
    let kelvin: Vec<f64> = temps.iter().map(|t| t + KELVIN).collect();
    let alt = fit_alt(
        &LifeData { obs: scaled, n_skipped: data.n_skipped },
        &[StressVariable { name: "Temperature (K)".into(), values: kelvin, relationship: Relationship::Reciprocal }],
        family,
    )?;
    let loglik = alt.loglik + jac;
    Ok(EyringFit { alt, loglik })
}

impl EyringFit {
    /// Life distribution at use temperature `temp_c` (°C) on the original time scale.
    pub fn at_temperature(&self, temp_c: f64) -> Result<LifeFit> {
        let k = temp_c + KELVIN;
        if k <= 0.0 {
            return invalid("temperature below absolute zero");
        }
        let mut f = self.alt.at_stress(&[k])?;
        f.mu -= k.ln();
        Ok(f)
    }
    /// Acceleration factor of `test_c` relative to `use_c` for the median.
    pub fn acceleration_factor(&self, use_c: f64, test_c: f64) -> Result<f64> {
        Ok(self.at_temperature(use_c)?.percentile(0.5) / self.at_temperature(test_c)?.percentile(0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reliability::km::{kaplan_meier, CiMethod};

    fn ld(t: &[f64], cens: &[f64]) -> LifeData {
        LifeData::from_times(t, Some(cens), 1.0, None).unwrap()
    }

    #[test]
    fn turnbull_equals_km_for_right_censored() {
        let t = [3.0, 5.0, 5.0, 7.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0];
        let c = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let d = ld(&t, &c);
        let tb = turnbull(&d, 1e-13, 200_000).unwrap();
        let km = kaplan_meier(&d, 0.95, CiMethod::Plain).unwrap();
        for s in &km.steps {
            assert!(
                (tb.survival_at(s.time) - s.survival).abs() < 1e-5,
                "t={} tb={} km={}",
                s.time,
                tb.survival_at(s.time),
                s.survival
            );
        }
    }

    #[test]
    fn turnbull_interval_masses_sum_to_one() {
        let d =
            LifeData::from_intervals(&[f64::NAN, 2.0, 3.0, 5.0, 6.0], &[4.0, 6.0, 8.0, 9.0, f64::NAN], None).unwrap();
        let tb = turnbull(&d, 1e-12, 100_000).unwrap();
        assert!((tb.mass.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(tb.survival.windows(2).all(|w| w[1] <= w[0] + 1e-12));
    }

    #[test]
    fn gamma_recovers_exponential() {
        // exponential-like data: gamma shape near 1
        let t: Vec<f64> = (1..=40).map(|i| -(1.0 - (i as f64 - 0.5) / 40.0).ln() * 10.0).collect();
        let d = LifeData::from_times(&t, None, 1.0, None).unwrap();
        let g = fit_gamma(&d).unwrap();
        assert!((g.shape - 1.0).abs() < 0.25, "shape {}", g.shape);
        assert!((g.mean() - 10.0).abs() < 1.5);
        assert!(g.cov.is_some());
    }

    #[test]
    fn threshold_found_for_shifted_weibull() {
        // quantiles of a Weibull(shape 2, scale 50) shifted by 100
        let t: Vec<f64> = (1..=60).map(|i| 100.0 + 50.0 * (-(1.0 - (i as f64 - 0.5) / 60.0).ln()).powf(0.5)).collect();
        let d = LifeData::from_times(&t, None, 1.0, None).unwrap();
        let f = fit_threshold(&d, Family::Weibull, 0.95).unwrap();
        assert!(f.threshold > 60.0 && f.threshold < t[0], "gamma {}", f.threshold);
        let two = fit_distribution(&d, Family::Weibull).unwrap();
        assert!(f.loglik >= two.loglik - 1e-6);
        assert!(f.threshold_lower <= f.threshold && f.threshold <= f.threshold_upper);
    }

    #[test]
    fn eyring_recovers_trend() {
        // times shrink with temperature
        let temps = [40.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0, 60.0, 80.0, 80.0, 80.0, 80.0];
        let t = [900.0, 1100.0, 1400.0, 1800.0, 300.0, 380.0, 450.0, 600.0, 110.0, 130.0, 170.0, 220.0];
        let d = LifeData::from_times(&t, None, 1.0, None).unwrap();
        let e = fit_eyring(&d, &temps, Family::Lognormal).unwrap();
        let use_ = e.at_temperature(30.0).unwrap();
        let hot = e.at_temperature(80.0).unwrap();
        assert!(use_.percentile(0.5) > 1500.0);
        assert!((hot.percentile(0.5) - 150.0).abs() < 40.0);
        assert!(e.acceleration_factor(30.0, 80.0).unwrap() > 5.0);
    }
}
