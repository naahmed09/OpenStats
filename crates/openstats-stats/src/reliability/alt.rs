//! Accelerated life testing: a location–scale regression of life on transformed stress.

use super::data::LifeData;
use super::dist::Family;
use super::fit::{fit_core, zcrit, Estimate, LifeFit};
use crate::error::{invalid, Result};
use openstats_math::special::erf::norm_sf;
use openstats_math::Matrix;

/// Life–stress relationship: the transformation applied to the stress before it enters μ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relationship {
    /// Arrhenius for temperature in °C: `x = 11604.5/(T + 273.16)`.
    Arrhenius,
    /// `x = stress`.
    Linear,
    /// `x = ln(stress)` (inverse power law).
    Log,
    /// `x = 1/stress`.
    Reciprocal,
}

impl Relationship {
    /// Transform a stress value; an error names values the relationship cannot take.
    pub fn transform(self, s: f64) -> Result<f64> {
        match self {
            Relationship::Linear => Ok(s),
            Relationship::Arrhenius => {
                if s + 273.16 <= 0.0 {
                    invalid("Arrhenius stress is a temperature in °C above −273.16")
                } else {
                    Ok(11604.5 / (s + 273.16))
                }
            }
            Relationship::Log => {
                if s <= 0.0 {
                    invalid("a log relationship needs positive stress values")
                } else {
                    Ok(s.ln())
                }
            }
            Relationship::Reciprocal => {
                if s == 0.0 {
                    invalid("a reciprocal relationship needs non-zero stress values")
                } else {
                    Ok(1.0 / s)
                }
            }
        }
    }

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Relationship::Arrhenius => "Arrhenius",
            Relationship::Linear => "Linear",
            Relationship::Log => "Log",
            Relationship::Reciprocal => "Reciprocal",
        }
    }
}

/// A stress column (one value per worksheet row) and its relationship.
#[derive(Debug, Clone)]
pub struct StressVariable {
    /// Name shown in the report.
    pub name: String,
    /// Stress per input row.
    pub values: Vec<f64>,
    /// Transformation.
    pub relationship: Relationship,
}

/// A fitted accelerated-life model.
#[derive(Debug, Clone)]
pub struct AltFit {
    /// Distribution family.
    pub family: Family,
    /// Stress variables.
    pub stresses: Vec<(String, Relationship)>,
    /// Coefficients `(β₀, β₁, …)`.
    pub beta: Vec<f64>,
    /// Scale σ.
    pub sigma: f64,
    /// Log-likelihood.
    pub loglik: f64,
    /// Units.
    pub n: f64,
    /// Failures.
    pub failures: f64,
    /// Covariance of `(β…, ln σ)`.
    pub cov: Option<Matrix>,
    /// Optimiser convergence.
    pub converged: bool,
}

/// Fit `μ = β₀ + Σ βⱼ xⱼ` with a common scale.
pub fn fit_alt(data: &LifeData, stresses: &[StressVariable], family: Family) -> Result<AltFit> {
    if stresses.is_empty() {
        return invalid("select at least one stress variable");
    }
    let mut x = Vec::with_capacity(data.obs.len());
    for o in &data.obs {
        let mut row = vec![1.0];
        for s in stresses {
            let v = *s.values.get(o.row).ok_or_else(|| {
                crate::error::StatsError::InvalidArgument("stress column shorter than the time column".into())
            })?;
            if v.is_nan() {
                return invalid(format!("row {}: the stress value is missing", o.row + 1));
            }
            row.push(
                s.relationship
                    .transform(v)
                    .map_err(|e| crate::error::StatsError::InvalidArgument(format!("row {}: {e}", o.row + 1)))?,
            );
        }
        x.push(row);
    }
    for j in 1..=stresses.len() {
        let first = x[0][j];
        if x.iter().all(|r| r[j] == first) {
            return invalid(format!(
                "'{}' has a single value: accelerated life testing needs at least two stress levels",
                stresses[j - 1].name
            ));
        }
    }
    let core = fit_core(family, &data.obs, &x)?;
    Ok(AltFit {
        family,
        stresses: stresses.iter().map(|s| (s.name.clone(), s.relationship)).collect(),
        beta: core.beta,
        sigma: core.ln_sigma.exp(),
        loglik: core.loglik,
        n: data.total(),
        failures: data.failures(),
        cov: core.cov,
        converged: core.converged,
    })
}

impl AltFit {
    /// Regression table: name, estimate (with interval), z and p for each coefficient, then the scale.
    pub fn coefficients(&self, level: f64) -> Vec<(String, Estimate, f64, f64)> {
        let k = zcrit(level);
        let mut v = Vec::new();
        for (i, b) in self.beta.iter().enumerate() {
            let name = if i == 0 { "Intercept".to_string() } else { self.stresses[i - 1].0.clone() };
            let se = self.cov.as_ref().map_or(f64::NAN, |c| c[(i, i)].sqrt());
            let z = b / se;
            v.push((name, Estimate { est: *b, se, lower: b - k * se, upper: b + k * se }, z, 2.0 * norm_sf(z.abs())));
        }
        v
    }

    /// Scale parameter (`Shape` for the Weibull) with interval, in the family's parameterisation.
    pub fn scale_estimate(&self, level: f64) -> (String, Estimate) {
        let k = zcrit(level);
        let kk = self.beta.len();
        let se_l =
            if self.family.fixed_scale() { 0.0 } else { self.cov.as_ref().map_or(f64::NAN, |c| c[(kk, kk)].sqrt()) };
        if self.family == Family::Weibull {
            let est = 1.0 / self.sigma;
            let (lo, hi) = (self.sigma.ln() - k * se_l, self.sigma.ln() + k * se_l);
            ("Shape".into(), Estimate { est, se: est * se_l, lower: (-hi).exp(), upper: (-lo).exp() })
        } else {
            let name = if self.family == Family::Exponential { "Scale (fixed)" } else { "Scale" };
            let (lo, hi) = (self.sigma.ln() - k * se_l, self.sigma.ln() + k * se_l);
            (name.into(), Estimate { est: self.sigma, se: self.sigma * se_l, lower: lo.exp(), upper: hi.exp() })
        }
    }

    /// Transformed design row for stress values `at` (one per stress variable).
    pub fn design_row(&self, at: &[f64]) -> Result<Vec<f64>> {
        if at.len() != self.stresses.len() {
            return invalid("give one stress value per stress variable");
        }
        let mut row = vec![1.0];
        for (v, (_, rel)) in at.iter().zip(&self.stresses) {
            row.push(rel.transform(*v)?);
        }
        Ok(row)
    }

    /// The life distribution at the given stress values.
    pub fn at_stress(&self, at: &[f64]) -> Result<LifeFit> {
        let x = self.design_row(at)?;
        let mu: f64 = x.iter().zip(&self.beta).map(|(a, b)| a * b).sum();
        let k = self.beta.len();
        let cov = self.cov.as_ref().map(|c| {
            let fixed = self.family.fixed_scale();
            let m = if fixed { 1 } else { 2 };
            let mut out = Matrix::zeros(m, m);
            let mut var_mu = 0.0;
            for a in 0..k {
                for b in 0..k {
                    var_mu += x[a] * x[b] * c[(a, b)];
                }
            }
            out[(0, 0)] = var_mu;
            if !fixed {
                let cov_ml: f64 = (0..k).map(|a| x[a] * c[(a, k)]).sum();
                out[(0, 1)] = cov_ml;
                out[(1, 0)] = cov_ml;
                out[(1, 1)] = c[(k, k)];
            }
            out
        });
        Ok(LifeFit {
            family: self.family,
            mu,
            sigma: self.sigma,
            loglik: self.loglik,
            n: self.n,
            failures: self.failures,
            censored: self.n - self.failures,
            cov,
            converged: self.converged,
        })
    }

    /// Acceleration factor `t_p(use)/t_p(test)` (ratio of any percentile for log families).
    pub fn acceleration_factor(&self, use_stress: &[f64], test_stress: &[f64]) -> Result<f64> {
        let (a, b) = (self.at_stress(use_stress)?, self.at_stress(test_stress)?);
        Ok(if self.family.is_log() { (a.mu - b.mu).exp() } else { a.percentile(0.5) / b.percentile(0.5) })
    }
}
