//! Normalising transformations for capability analysis: Box–Cox and Johnson.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::anderson_darling;
use crate::quantile::{quantile_sorted, QuantileMethod};
use openstats_math::dist::{ContinuousDistribution, Johnson, JohnsonFamily};
use openstats_math::optim::brent_minimize;
use openstats_math::special::erf::norm_cdf;

/// How the Box–Cox λ is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaChoice {
    /// The maximum-likelihood λ in [−5, 5].
    Optimal,
    /// The optimal λ rounded to the nearest 0.5.
    Rounded,
    /// A user-supplied λ.
    Fixed(f64),
}

/// A fitted Box–Cox transformation `(x^λ − 1)/λ` (`ln x` at λ = 0).
///
/// The scaled form is increasing in x for every λ, so specification limits
/// keep their order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxCox {
    /// λ used by the transformation.
    pub lambda: f64,
    /// The maximum-likelihood λ (before rounding).
    pub optimal_lambda: f64,
    /// Profile log-likelihood at `lambda`.
    pub log_likelihood: f64,
}

impl BoxCox {
    /// Transform one value (`NaN` unless x > 0).
    pub fn forward(&self, x: f64) -> f64 {
        if x.is_nan() || x <= 0.0 {
            return f64::NAN;
        }
        if self.lambda.abs() < 1e-12 {
            x.ln()
        } else {
            (self.lambda * x.ln()).exp_m1() / self.lambda
        }
    }

    /// Back-transform (`NaN` when `1 + λz ≤ 0`).
    pub fn inverse(&self, z: f64) -> f64 {
        if self.lambda.abs() < 1e-12 {
            z.exp()
        } else {
            (1.0 + self.lambda * z).powf(1.0 / self.lambda)
        }
    }
}

fn profile_loglik(logs: &[f64], lambda: f64) -> f64 {
    let n = logs.len() as f64;
    let z: Vec<f64> = if lambda.abs() < 1e-12 {
        logs.to_vec()
    } else {
        logs.iter().map(|l| (lambda * l).exp_m1() / lambda).collect()
    };
    let m = z.iter().sum::<f64>() / n;
    let var = z.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
    -0.5 * n * var.ln() + (lambda - 1.0) * logs.iter().sum::<f64>()
}

impl BoxCox {
    /// Fit λ to positive data (missing values ignored).
    pub fn fit(data: &[f64], choice: LambdaChoice) -> Result<BoxCox> {
        if data.iter().any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        let v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.len() < 3 {
            return Err(StatsError::InsufficientData { what: "Box-Cox transformation", needed: 3, got: v.len() });
        }
        if v.iter().any(|&x| x <= 0.0) {
            return invalid("the Box-Cox transformation needs strictly positive data");
        }
        let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
        if lo == hi {
            return invalid("the Box-Cox transformation needs data that are not all equal");
        }
        let logs: Vec<f64> = v.iter().map(|x| x.ln()).collect();
        let mut best = (f64::NEG_INFINITY, 0.0);
        for i in 0..=200 {
            let l = -5.0 + 0.05 * i as f64;
            let ll = profile_loglik(&logs, l);
            if ll > best.0 {
                best = (ll, l);
            }
        }
        let (a, b) = ((best.1 - 0.05).max(-5.0), (best.1 + 0.05).min(5.0));
        let opt = brent_minimize(|l| -profile_loglik(&logs, l), a, b, 1e-12, 200)?;
        let optimal = if opt.f.is_finite() && -opt.f >= best.0 { opt.x } else { best.1 };
        let lambda = match choice {
            LambdaChoice::Optimal => optimal,
            LambdaChoice::Rounded => (optimal * 2.0).round() / 2.0,
            LambdaChoice::Fixed(l) => {
                if !l.is_finite() {
                    return invalid("λ must be finite");
                }
                l
            }
        };
        Ok(BoxCox { lambda, optimal_lambda: optimal, log_likelihood: profile_loglik(&logs, lambda) })
    }
}

/// Profile log-likelihood of the Box–Cox transformation at each λ in `lambdas` (positive data;
/// missing values ignored). The curve is what the Box–Cox plot draws.
pub fn box_cox_profile(data: &[f64], lambdas: &[f64]) -> Result<Vec<f64>> {
    let v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.len() < 3 {
        return Err(StatsError::InsufficientData { what: "Box-Cox transformation", needed: 3, got: v.len() });
    }
    if v.iter().any(|&x| x.is_infinite() || x <= 0.0) {
        return invalid("the Box-Cox transformation needs strictly positive, finite data");
    }
    let logs: Vec<f64> = v.iter().map(|x| x.ln()).collect();
    Ok(lambdas.iter().map(|&l| profile_loglik(&logs, l)).collect())
}

/// Likelihood-ratio confidence interval of the Box–Cox λ at `level`, searched on [−5, 5]
/// (a limit that does not cross the cut-off is reported as ±5).
pub fn box_cox_interval(data: &[f64], level: f64) -> Result<(f64, f64)> {
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be strictly between 0 and 1");
    }
    let fit = BoxCox::fit(data, LambdaChoice::Optimal)?;
    let v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let logs: Vec<f64> = v.iter().map(|x| x.ln()).collect();
    let cut = fit.log_likelihood - 0.5 * openstats_math::dist::ChiSquared::new(1.0)?.quantile(level);
    let f = |l: f64| profile_loglik(&logs, l) - cut;
    let bisect = |mut inside: f64, mut outside: f64| {
        for _ in 0..80 {
            let mid = 0.5 * (inside + outside);
            if f(mid) >= 0.0 {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        0.5 * (inside + outside)
    };
    let lo = if f(-5.0) >= 0.0 { -5.0 } else { bisect(fit.lambda, -5.0) };
    let hi = if f(5.0) >= 0.0 { 5.0 } else { bisect(fit.lambda, 5.0) };
    Ok((lo, hi))
}

/// A fitted Johnson transformation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JohnsonFit {
    /// The fitted Johnson distribution (`Z = γ + δ·g((X−ξ)/λ)` is standard normal).
    pub dist: Johnson,
    /// The z used by the quantile fit (points at ±z and ±3z).
    pub z: f64,
    /// Anderson–Darling statistic of the transformed data.
    pub a2: f64,
    /// Anderson–Darling p-value of the transformed data.
    pub p_value: f64,
}

impl JohnsonFit {
    /// Transform one value; ±∞ outside the support of SB/SL.
    pub fn forward(&self, x: f64) -> f64 {
        let d = &self.dist;
        let u = (x - d.xi) / d.lambda;
        match d.family {
            JohnsonFamily::SU => d.gamma + d.delta * u.asinh(),
            JohnsonFamily::SB => {
                if u <= 0.0 {
                    f64::NEG_INFINITY
                } else if u >= 1.0 {
                    f64::INFINITY
                } else {
                    d.gamma + d.delta * (u / (1.0 - u)).ln()
                }
            }
            JohnsonFamily::SL => {
                if u <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    d.gamma + d.delta * u.ln()
                }
            }
        }
    }
}

/// Parameters of the Johnson curve through the quantiles at −3z, −z, z, 3z
/// (Slifker–Shapiro): closed forms derived from the defining equations.
fn solve_johnson(z: f64, x: [f64; 4]) -> Option<Johnson> {
    let (m, n, p) = (x[3] - x[2], x[1] - x[0], x[2] - x[1]);
    if !(m > 0.0 && n > 0.0 && p > 0.0) {
        return None;
    }
    let qr = m * n / (p * p);
    let (mp, np) = (m / p, n / p);
    let fit = if (qr - 1.0).abs() < 1e-9 {
        // SL: m/p = e^{2z/δ} requires positive skew.
        if mp <= 1.0 {
            return None;
        }
        let r = mp.sqrt();
        let a = r.ln();
        let delta = z / a;
        let k = p * r / (mp - 1.0);
        Johnson::new(JohnsonFamily::SL, -delta * k.ln(), delta, x[2] - k * r, 1.0).ok()?
    } else if qr > 1.0 {
        let ach = (0.5 * (mp + np)).acosh();
        let delta = 2.0 * z / ach;
        let a = z / delta;
        let c = ((np - mp) / (2.0 * (qr - 1.0).sqrt())).asinh();
        let lambda = p / (2.0 * c.cosh() * a.sinh());
        let xi = 0.5 * (x[2] + x[1]) + lambda * c.sinh() * a.cosh();
        Johnson::new(JohnsonFamily::SU, delta * c, delta, xi, lambda).ok()?
    } else {
        let prod = (1.0 + p / m) * (1.0 + p / n);
        let half = 0.5 * prod.sqrt();
        if half < 1.0 {
            return None;
        }
        let a = half.acosh();
        let delta = z / a;
        let rho = ((1.0 + p / m) / (1.0 + p / n)).sqrt();
        let s = a.exp();
        let kappa = (rho * s - 1.0) / (s - rho);
        if !(kappa > 0.0 && kappa.is_finite()) {
            return None;
        }
        let l1 = kappa.ln() - 3.0 * a;
        let u = |i: usize| 1.0 / (1.0 + (-(l1 + 2.0 * a * i as f64)).exp());
        let lambda = p / (u(2) - u(1));
        let xi = x[1] - lambda * u(1);
        Johnson::new(JohnsonFamily::SB, -delta * kappa.ln(), delta, xi, lambda).ok()?
    };
    // The curve must reproduce the four quantiles.
    let scale = x[3] - x[0];
    let probs = [norm_cdf(-3.0 * z), norm_cdf(-z), norm_cdf(z), norm_cdf(3.0 * z)];
    use openstats_math::dist::ContinuousDistribution;
    for (xi, pi) in x.iter().zip(probs) {
        if (fit.quantile(pi) - xi).abs() > 1e-7 * scale {
            return None;
        }
    }
    Some(fit)
}

/// Fit a Johnson transformation (Chou, Polansky & Mason, 1998): for z = 0.25,
/// 0.26, …, 1.25 the SB / SL / SU curve through the sample quantiles (Minitab
/// type 6) at the normal scores −3z, −z, z, 3z is found in closed form; the
/// curve whose transformed data have the largest Anderson–Darling p-value is
/// kept. z values whose 3z quantile cannot be estimated (`Φ(3z) > n/(n+1)`)
/// are skipped.
pub fn fit_johnson(data: &[f64]) -> Result<JohnsonFit> {
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let mut v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n < 10 {
        return Err(StatsError::InsufficientData { what: "Johnson transformation", needed: 10, got: n });
    }
    v.sort_by(|a, b| a.total_cmp(b));
    if v[0] == v[n - 1] {
        return invalid("the Johnson transformation needs data that are not all equal");
    }
    let mut best: Option<JohnsonFit> = None;
    for step in 0..=100 {
        let z = 0.25 + 0.01 * step as f64;
        let ps = [norm_cdf(-3.0 * z), norm_cdf(-z), norm_cdf(z), norm_cdf(3.0 * z)];
        if ps[3] > n as f64 / (n as f64 + 1.0) || ps[0] < 1.0 / (n as f64 + 1.0) {
            break;
        }
        let mut x = [0.0; 4];
        for (xi, p) in x.iter_mut().zip(ps) {
            *xi = quantile_sorted(&v, p, QuantileMethod::Type6)?;
        }
        let Some(dist) = solve_johnson(z, x) else { continue };
        let cand = JohnsonFit { dist, z, a2: 0.0, p_value: 0.0 };
        let t: Vec<f64> = v.iter().map(|&xv| cand.forward(xv)).collect();
        if t.iter().any(|q| !q.is_finite()) {
            continue;
        }
        let Ok(ad) = anderson_darling(&t) else { continue };
        if best.as_ref().is_none_or(|b| ad.p_value > b.p_value) {
            best = Some(JohnsonFit { dist, z, a2: ad.a2, p_value: ad.p_value });
        }
    }
    best.ok_or_else(|| {
        StatsError::InvalidArgument("no Johnson curve fits the data (too few observations, or no admissible z)".into())
    })
}

/// A transformation to normality applied before a capability analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transformation {
    /// Box–Cox.
    BoxCox(BoxCox),
    /// Johnson.
    Johnson(JohnsonFit),
}

impl Transformation {
    /// Transform a value (`NaN` for invalid input, ±∞ outside a bounded support).
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Transformation::BoxCox(b) => b.forward(x),
            Transformation::Johnson(j) => j.forward(x),
        }
    }

    /// Short description.
    pub fn describe(&self) -> String {
        match self {
            Transformation::BoxCox(b) => format!("Box-Cox transformation with lambda = {:.6}", b.lambda),
            Transformation::Johnson(j) => {
                let d = &j.dist;
                let f = match d.family {
                    JohnsonFamily::SU => "SU",
                    JohnsonFamily::SB => "SB",
                    JohnsonFamily::SL => "SL",
                };
                format!(
                    "Johnson {f} transformation: gamma = {:.6}, eta = {:.6}, epsilon = {:.6}, lambda = {:.6}",
                    d.gamma, d.delta, d.xi, d.lambda
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_math::dist::ContinuousDistribution;

    fn lognormal_sample(n: usize) -> Vec<f64> {
        // Deterministic quantiles of a lognormal(1, 0.5) (exact normal scores).
        let nd = openstats_math::dist::Normal::standard();
        (1..=n).map(|i| (1.0 + 0.5 * nd.quantile((i as f64 - 0.5) / n as f64)).exp()).collect()
    }

    #[test]
    fn boxcox_recovers_log_for_lognormal_data() {
        let x = lognormal_sample(200);
        let b = BoxCox::fit(&x, LambdaChoice::Optimal).unwrap();
        assert!(b.optimal_lambda.abs() < 0.05, "λ = {}", b.optimal_lambda);
        let r = BoxCox::fit(&x, LambdaChoice::Rounded).unwrap();
        assert_eq!(r.lambda, 0.0);
        assert!((r.forward(std::f64::consts::E) - 1.0).abs() < 1e-12);
        let f = BoxCox::fit(&x, LambdaChoice::Fixed(0.5)).unwrap();
        assert!((f.forward(4.0) - 2.0).abs() < 1e-12); // (4^0.5 − 1)/0.5
        assert!((f.inverse(f.forward(7.3)) - 7.3).abs() < 1e-10);
        assert!(BoxCox::fit(&[1.0, 2.0, -1.0, 4.0], LambdaChoice::Optimal).is_err());
        assert!(BoxCox::fit(&[2.0, 2.0, 2.0], LambdaChoice::Optimal).is_err());
    }

    #[test]
    fn boxcox_of_squared_normal_data_is_half() {
        // x = y² with y symmetric positive → λ = 0.5 normalises.
        let nd = openstats_math::dist::Normal::standard();
        let x: Vec<f64> = (1..=300).map(|i| (20.0 + 2.0 * nd.quantile((i as f64 - 0.5) / 300.0)).powi(2)).collect();
        let b = BoxCox::fit(&x, LambdaChoice::Optimal).unwrap();
        assert!((b.optimal_lambda - 0.5).abs() < 0.03, "λ = {}", b.optimal_lambda);
    }

    #[test]
    fn johnson_closed_forms_recover_known_curves() {
        for (fam, g, d, xi, l) in [
            (JohnsonFamily::SU, 0.7, 1.3, 2.0, 3.0),
            (JohnsonFamily::SU, -0.4, 0.9, -1.0, 2.5),
            (JohnsonFamily::SB, 0.5, 1.1, 1.0, 4.0),
            (JohnsonFamily::SB, -0.6, 0.8, 0.0, 10.0),
        ] {
            let j = Johnson::new(fam, g, d, xi, l).unwrap();
            for z in [0.3, 0.75, 1.2] {
                let x = [
                    j.quantile(norm_cdf(-3.0 * z)),
                    j.quantile(norm_cdf(-z)),
                    j.quantile(norm_cdf(z)),
                    j.quantile(norm_cdf(3.0 * z)),
                ];
                let s = solve_johnson(z, x).unwrap_or_else(|| panic!("no solution for {fam:?} z={z}"));
                assert_eq!(s.family, fam);
                assert!((s.gamma - g).abs() < 1e-6 && (s.delta - d).abs() < 1e-6, "{fam:?} {s:?}");
                assert!(
                    (s.xi - xi).abs() < 1e-5 * (1.0 + xi.abs()) && (s.lambda - l).abs() < 1e-5 * l,
                    "{fam:?} {s:?}"
                );
            }
        }
        // SL: three-parameter lognormal x = ξ + exp((z − γ)/δ).
        let j = Johnson::new(JohnsonFamily::SL, 0.3, 0.8, 5.0, 1.0).unwrap();
        let z = 0.6;
        let x = [
            j.quantile(norm_cdf(-3.0 * z)),
            j.quantile(norm_cdf(-z)),
            j.quantile(norm_cdf(z)),
            j.quantile(norm_cdf(3.0 * z)),
        ];
        // The sample curve of a true SL has QR = 1 up to rounding.
        let s = solve_johnson(z, x).unwrap();
        assert_eq!(s.family, JohnsonFamily::SL);
        assert!((s.delta - 0.8).abs() < 1e-8 && (s.xi - 5.0).abs() < 1e-8 && (s.gamma - 0.3).abs() < 1e-8);
    }

    #[test]
    fn johnson_fit_normalises_skewed_data() {
        let x = lognormal_sample(300);
        let f = fit_johnson(&x).unwrap();
        assert!(f.p_value > 0.10, "p = {}", f.p_value);
        let t: Vec<f64> = x.iter().map(|&v| f.forward(v)).collect();
        let m = t.iter().sum::<f64>() / t.len() as f64;
        let sd = (t.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (t.len() - 1) as f64).sqrt();
        assert!(m.abs() < 0.2 && (sd - 1.0).abs() < 0.2, "mean {m}, sd {sd}");
        assert!(fit_johnson(&[1.0, 2.0, 3.0]).is_err());
        assert!(fit_johnson(&[2.0; 20]).is_err());
    }

    #[test]
    fn transformation_describe() {
        let b = BoxCox::fit(&lognormal_sample(50), LambdaChoice::Fixed(0.5)).unwrap();
        assert!(Transformation::BoxCox(b).describe().contains("0.5"));
        let j = fit_johnson(&lognormal_sample(100)).unwrap();
        assert!(Transformation::Johnson(j).describe().starts_with("Johnson S"));
    }
}
