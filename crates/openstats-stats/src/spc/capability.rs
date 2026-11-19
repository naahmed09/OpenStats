//! Process capability: normal (Cp, Cpk, Pp, Ppk, Cpm, Z.Bench, PPM), via a
//! Box–Cox or Johnson transformation, or via a fitted Weibull, lognormal or
//! gamma distribution.
//!
//! Definitions (Minitab's):
//!
//! * *Within* σ is estimated from the subgroup ranges/standard deviations or
//!   from the moving ranges of the individuals; *overall* σ is the ordinary
//!   sample standard deviation (n − 1).
//! * `Cp = (USL − LSL)/6σ_within`, `Cpk = min(CPL, CPU)` with
//!   `CPL = (μ − LSL)/3σ_within`, `CPU = (USL − μ)/3σ_within`; `Pp`, `Ppk` are
//!   the same with σ_overall; `Cpm = (USL − LSL)/(6√(σ_overall² + (μ − T)²))`.
//! * `Z.Bench = Φ⁻¹(1 − P(X < LSL) − P(X > USL))` (the normal deviate of the
//!   total expected out-of-spec probability), `Z.LSL = (μ − LSL)/σ`,
//!   `Z.USL = (USL − μ)/σ`.
//! * Non-normal capability uses the 0.135 % and 99.865 % points of the fitted
//!   distribution in place of μ ∓ 3σ, and the median in place of μ.
//! * Confidence intervals are given for Pp (χ²) and Ppk (Bissell's normal
//!   approximation `Ppk ± z·√(1/(9n) + Ppk²/(2(n−1)))`); none for Cp/Cpk.

use super::charts::{estimate_sigma, individuals_sigma_estimate, subgroup_stats, IndividualsSigma, SubgroupSigma};
use super::transform::Transformation;
use crate::error::{invalid, Result, StatsError};
use crate::graphical::{anderson_darling, AndersonDarling, Interval};
use openstats_math::dist::{
    ChiSquared, ContinuousDistribution, Exponential, Gamma, Logistic, Loglogistic, Lognormal, Normal, Weibull,
};
use openstats_math::roots::brent;
use openstats_math::special::erf::{norm_cdf, norm_isf, norm_ppf};
use openstats_math::special::gamma::digamma;

/// Specification limits and target.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Specs {
    /// Lower specification limit.
    pub lsl: Option<f64>,
    /// Target (for Cpm).
    pub target: Option<f64>,
    /// Upper specification limit.
    pub usl: Option<f64>,
}

impl Specs {
    /// Two-sided specification.
    pub fn two_sided(lsl: f64, usl: f64) -> Specs {
        Specs { lsl: Some(lsl), target: None, usl: Some(usl) }
    }

    /// Add a target.
    pub fn with_target(mut self, target: f64) -> Specs {
        self.target = Some(target);
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.lsl.is_none() && self.usl.is_none() {
            return invalid("at least one specification limit is required");
        }
        for v in [self.lsl, self.usl, self.target].into_iter().flatten() {
            if !v.is_finite() {
                return invalid("specification limits and the target must be finite");
            }
        }
        if let (Some(l), Some(u)) = (self.lsl, self.usl) {
            if l >= u {
                return invalid(format!("LSL ({l}) must be below USL ({u})"));
            }
        }
        Ok(())
    }
}

/// Options of a capability analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityOptions {
    /// Confidence level of the Pp / Ppk intervals (0.95).
    pub level: f64,
    /// σ estimate for individuals.
    pub individuals_sigma: IndividualsSigma,
    /// Moving-range span (2).
    pub span: usize,
    /// σ estimate for subgroups; `None` → R̄ when the largest subgroup has ≤ 8
    /// observations and s̄ otherwise.
    pub subgroup_sigma: Option<SubgroupSigma>,
}

impl Default for CapabilityOptions {
    fn default() -> Self {
        CapabilityOptions {
            level: 0.95,
            individuals_sigma: IndividualsSigma::AverageMovingRange,
            span: 2,
            subgroup_sigma: None,
        }
    }
}

/// Expected out-of-specification performance for a fitted distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectedPerformance {
    /// Z.Bench (`+∞` when nothing is expected out of spec).
    pub z_bench: f64,
    /// Z.LSL.
    pub z_lsl: Option<f64>,
    /// Z.USL.
    pub z_usl: Option<f64>,
    /// Expected PPM below LSL.
    pub ppm_below: f64,
    /// Expected PPM above USL.
    pub ppm_above: f64,
    /// Expected total PPM.
    pub ppm_total: f64,
}

/// Observed out-of-specification counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservedPerformance {
    /// Observations below LSL.
    pub n_below: usize,
    /// Observations above USL.
    pub n_above: usize,
    /// PPM below LSL.
    pub ppm_below: f64,
    /// PPM above USL.
    pub ppm_above: f64,
    /// Total PPM.
    pub ppm_total: f64,
}

pub(crate) fn observed(data: &[f64], specs: &Specs) -> ObservedPerformance {
    let n = data.len();
    let below = specs.lsl.map_or(0, |l| data.iter().filter(|&&x| x < l).count());
    let above = specs.usl.map_or(0, |u| data.iter().filter(|&&x| x > u).count());
    let ppm = |c: usize| c as f64 / n as f64 * 1e6;
    ObservedPerformance {
        n_below: below,
        n_above: above,
        ppm_below: ppm(below),
        ppm_above: ppm(above),
        ppm_total: ppm(below + above),
    }
}

pub(crate) fn expected_from(p_below: f64, p_above: f64, specs: &Specs) -> ExpectedPerformance {
    let total = p_below + p_above;
    ExpectedPerformance {
        z_bench: if total <= 0.0 { f64::INFINITY } else { norm_isf(total.min(1.0 - 1e-300)) },
        z_lsl: specs.lsl.map(|_| norm_isf(p_below.max(0.0))),
        z_usl: specs.usl.map(|_| norm_isf(p_above.max(0.0))),
        ppm_below: p_below * 1e6,
        ppm_above: p_above * 1e6,
        ppm_total: total * 1e6,
    }
}

/// Capability of a normal process.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalCapability {
    /// Number of observations.
    pub n: usize,
    /// Sample mean.
    pub mean: f64,
    /// Within-subgroup σ.
    pub sd_within: f64,
    /// Description of the within-σ estimate.
    pub within_method: String,
    /// Overall (sample) σ.
    pub sd_overall: f64,
    /// Specifications.
    pub specs: Specs,
    /// Cp.
    pub cp: Option<f64>,
    /// CPL.
    pub cpl: Option<f64>,
    /// CPU.
    pub cpu: Option<f64>,
    /// Cpk.
    pub cpk: f64,
    /// Pp.
    pub pp: Option<f64>,
    /// PPL.
    pub ppl: Option<f64>,
    /// PPU.
    pub ppu: Option<f64>,
    /// Ppk.
    pub ppk: f64,
    /// Cpm (needs both limits and a target).
    pub cpm: Option<f64>,
    /// Confidence interval for Pp.
    pub pp_ci: Option<Interval>,
    /// Confidence interval for Ppk.
    pub ppk_ci: Option<Interval>,
    /// Observed performance.
    pub observed: ObservedPerformance,
    /// Expected performance from the within σ.
    pub expected_within: ExpectedPerformance,
    /// Expected performance from the overall σ.
    pub expected_overall: ExpectedPerformance,
}

fn normal_capability(
    data: &[f64],
    sd_within: f64,
    within_method: String,
    specs: &Specs,
    level: f64,
) -> Result<NormalCapability> {
    specs.validate()?;
    if !(level > 0.0 && level < 1.0) {
        return invalid(format!("confidence level {level} must be in (0, 1)"));
    }
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "capability analysis", needed: 2, got: n });
    }
    let nf = n as f64;
    let mean = data.iter().sum::<f64>() / nf;
    let sd_overall = (data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0)).sqrt();
    if !(sd_within > 0.0 && sd_overall > 0.0) {
        return invalid("the process has no variation; capability is undefined");
    }
    let indices = |s: f64| {
        let cpl = specs.lsl.map(|l| (mean - l) / (3.0 * s));
        let cpu = specs.usl.map(|u| (u - mean) / (3.0 * s));
        let cp = match (specs.lsl, specs.usl) {
            (Some(l), Some(u)) => Some((u - l) / (6.0 * s)),
            _ => None,
        };
        let k = match (cpl, cpu) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => f64::NAN,
        };
        (cp, cpl, cpu, k)
    };
    let (cp, cpl, cpu, cpk) = indices(sd_within);
    let (pp, ppl, ppu, ppk) = indices(sd_overall);
    let cpm = match (specs.lsl, specs.usl, specs.target) {
        (Some(l), Some(u), Some(t)) => {
            Some((u - l) / (6.0 * (sd_overall * sd_overall + (mean - t) * (mean - t)).sqrt()))
        }
        _ => None,
    };
    let alpha = 1.0 - level;
    let pp_ci = match pp {
        Some(p) => {
            let c = ChiSquared::new(nf - 1.0)?;
            Some(Interval {
                lower: p * (c.quantile(alpha / 2.0) / (nf - 1.0)).sqrt(),
                upper: p * (c.quantile(1.0 - alpha / 2.0) / (nf - 1.0)).sqrt(),
                level,
            })
        }
        None => None,
    };
    let z = norm_ppf(1.0 - alpha / 2.0);
    let half = z * (1.0 / (9.0 * nf) + ppk * ppk / (2.0 * (nf - 1.0))).sqrt();
    let ppk_ci = Some(Interval { lower: ppk - half, upper: ppk + half, level });
    let tail = |s: f64| {
        let below = specs.lsl.map_or(0.0, |l| norm_cdf((l - mean) / s));
        let above = specs.usl.map_or(0.0, |u| norm_upper(u, mean, s));
        expected_from(below, above, specs)
    };
    Ok(NormalCapability {
        n,
        mean,
        sd_within,
        within_method,
        sd_overall,
        specs: *specs,
        cp,
        cpl,
        cpu,
        cpk,
        pp,
        ppl,
        ppu,
        ppk,
        cpm,
        pp_ci,
        ppk_ci,
        observed: observed(data, specs),
        expected_within: tail(sd_within),
        expected_overall: tail(sd_overall),
    })
}

fn norm_upper(u: f64, mean: f64, s: f64) -> f64 {
    openstats_math::special::erf::norm_sf((u - mean) / s)
}

/// Capability of individual observations in time order (within σ from moving ranges).
pub fn capability_individuals(data: &[f64], specs: &Specs, opts: &CapabilityOptions) -> Result<NormalCapability> {
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    if !(2..=25).contains(&opts.span) {
        return invalid("moving-range span must be in 2..=25");
    }
    let (sd, method) = individuals_sigma_estimate(data, opts.span, opts.individuals_sigma)?;
    let v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    normal_capability(&v, sd, method, specs, opts.level)
}

/// Capability of subgroup data (within σ from R̄, s̄ or the pooled standard deviation).
pub fn capability_subgroups(groups: &[Vec<f64>], specs: &Specs, opts: &CapabilityOptions) -> Result<NormalCapability> {
    let stats = subgroup_stats(groups)?;
    let max_n = stats.iter().map(|s| s.n).max().unwrap_or(0);
    let method = opts.subgroup_sigma.unwrap_or(if max_n <= 8 { SubgroupSigma::RBar } else { SubgroupSigma::SBar });
    let (sd, label) = estimate_sigma(&stats, method)?;
    let all: Vec<f64> = groups.iter().flatten().copied().filter(|x| !x.is_nan()).collect();
    normal_capability(&all, sd, label, specs, opts.level)
}

// --------------------------------------------------------- transformation

/// Capability after a normalising transformation (all quantities are on the
/// transformed scale; observed PPM is the same on both scales because the
/// transformations are increasing).
#[derive(Debug, Clone, PartialEq)]
pub struct TransformedCapability {
    /// The transformation applied.
    pub transformation: Transformation,
    /// Normal capability of the transformed data against the transformed limits.
    pub normal: NormalCapability,
    /// Specifications on the transformed scale.
    pub transformed_specs: Specs,
    /// Anderson–Darling test of normality of the transformed data.
    pub ad: AndersonDarling,
    /// Notes (e.g. a limit outside the support of a bounded Johnson curve).
    pub notes: Vec<String>,
}

fn transform_specs(t: &Transformation, specs: &Specs) -> Result<(Specs, Vec<String>)> {
    let mut notes = Vec::new();
    let mut out = Specs::default();
    if let Some(l) = specs.lsl {
        let v = t.forward(l);
        if v.is_nan() || v == f64::INFINITY {
            return invalid(
                "the lower specification limit cannot be transformed (outside the support of the transformation)",
            );
        }
        if v == f64::NEG_INFINITY {
            notes.push(
                "LSL lies below the support of the fitted transformation; no lower limit is applied.".to_string(),
            );
        } else {
            out.lsl = Some(v);
        }
    }
    if let Some(u) = specs.usl {
        let v = t.forward(u);
        if v.is_nan() || v == f64::NEG_INFINITY {
            return invalid(
                "the upper specification limit cannot be transformed (outside the support of the transformation)",
            );
        }
        if v == f64::INFINITY {
            notes.push(
                "USL lies above the support of the fitted transformation; no upper limit is applied.".to_string(),
            );
        } else {
            out.usl = Some(v);
        }
    }
    if let Some(g) = specs.target {
        let v = t.forward(g);
        if v.is_finite() {
            out.target = Some(v);
        }
    }
    if out.lsl.is_none() && out.usl.is_none() {
        return invalid("no specification limit remains inside the support of the transformation");
    }
    Ok((out, notes))
}

fn transform_data(t: &Transformation, data: &[f64]) -> Result<Vec<f64>> {
    let z: Vec<f64> = data.iter().map(|&x| if x.is_nan() { f64::NAN } else { t.forward(x) }).collect();
    if z.iter().any(|v| v.is_infinite() || (v.is_nan() && !data.iter().any(|x| x.is_nan()))) {
        return invalid("some observations lie outside the domain of the transformation");
    }
    if z.iter().zip(data).any(|(t, x)| t.is_nan() && !x.is_nan()) {
        return invalid("some observations lie outside the domain of the transformation");
    }
    Ok(z)
}

/// Transformed capability of individual observations.
pub fn capability_transformed_individuals(
    data: &[f64],
    transformation: &Transformation,
    specs: &Specs,
    opts: &CapabilityOptions,
) -> Result<TransformedCapability> {
    specs.validate()?;
    let z = transform_data(transformation, data)?;
    let (ts, notes) = transform_specs(transformation, specs)?;
    let normal = capability_individuals(&z, &ts, opts)?;
    let valid: Vec<f64> = z.iter().copied().filter(|v| !v.is_nan()).collect();
    Ok(TransformedCapability {
        transformation: *transformation,
        normal,
        transformed_specs: ts,
        ad: anderson_darling(&valid)?,
        notes,
    })
}

/// Transformed capability of subgroup data.
pub fn capability_transformed_subgroups(
    groups: &[Vec<f64>],
    transformation: &Transformation,
    specs: &Specs,
    opts: &CapabilityOptions,
) -> Result<TransformedCapability> {
    specs.validate()?;
    let zg: Vec<Vec<f64>> = groups.iter().map(|g| transform_data(transformation, g)).collect::<Result<_>>()?;
    let (ts, notes) = transform_specs(transformation, specs)?;
    let normal = capability_subgroups(&zg, &ts, opts)?;
    let valid: Vec<f64> = zg.iter().flatten().copied().filter(|v| !v.is_nan()).collect();
    Ok(TransformedCapability {
        transformation: *transformation,
        normal,
        transformed_specs: ts,
        ad: anderson_darling(&valid)?,
        notes,
    })
}

// ------------------------------------------------------ fitted distribution

/// Distribution families that can be fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Weibull (shape, scale, threshold).
    Weibull,
    /// Lognormal (location μ and scale σ of ln(x − threshold)).
    Lognormal,
    /// Gamma (shape, scale, threshold).
    Gamma,
    /// Two-parameter exponential (scale = mean − threshold, threshold).
    Exponential,
    /// Normal (mean, standard deviation; the maximum-likelihood σ divides by n).
    Normal,
    /// Logistic (location, scale).
    Logistic,
    /// Loglogistic: ln(x − threshold) is logistic (location, scale).
    Loglogistic,
}

impl Family {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Family::Weibull => "Weibull",
            Family::Lognormal => "Lognormal",
            Family::Gamma => "Gamma",
            Family::Exponential => "Exponential",
            Family::Normal => "Normal",
            Family::Logistic => "Logistic",
            Family::Loglogistic => "Loglogistic",
        }
    }

    /// Whether the family has a threshold parameter (the location-scale
    /// families Normal and Logistic do not; their threshold is reported as 0).
    pub fn uses_threshold(self) -> bool {
        !matches!(self, Family::Normal | Family::Logistic)
    }

    /// Labels of the two reported parameters. The exponential has one
    /// parameter; its second slot is `NaN`.
    pub fn parameter_labels(self) -> (&'static str, &'static str) {
        match self {
            Family::Weibull | Family::Gamma => ("Shape", "Scale"),
            Family::Lognormal | Family::Loglogistic | Family::Logistic => ("Location", "Scale"),
            Family::Normal => ("Mean", "StDev"),
            Family::Exponential => ("Scale", ""),
        }
    }
}

/// A distribution fitted by maximum likelihood (threshold fixed).
#[derive(Debug, Clone, PartialEq)]
pub struct FittedDistribution {
    /// Family.
    pub family: Family,
    /// Shape (Weibull, gamma) or location μ (lognormal).
    pub param1: f64,
    /// Scale (Weibull, gamma) or σ (lognormal).
    pub param2: f64,
    /// Threshold (shift).
    pub threshold: f64,
    /// Observations used.
    pub n: usize,
    /// Maximised log-likelihood.
    pub log_likelihood: f64,
    /// Anderson–Darling statistic against the fitted distribution.
    pub ad: f64,
}

impl FittedDistribution {
    fn dist(&self) -> Result<Box<dyn ContinuousDistribution>> {
        Ok(match self.family {
            Family::Weibull => Box::new(Weibull::new(self.param1, self.param2, self.threshold)?),
            Family::Lognormal => Box::new(Lognormal::new(self.param1, self.param2, self.threshold)?),
            Family::Gamma => Box::new(Gamma::new(self.param1, self.param2, self.threshold)?),
            Family::Exponential => Box::new(Exponential::new(self.param1, self.threshold)?),
            Family::Normal => Box::new(Normal::new(self.param1, self.param2)?),
            Family::Logistic => Box::new(Logistic::new(self.param1, self.param2)?),
            Family::Loglogistic => Box::new(Loglogistic::new(self.param1, self.param2, self.threshold)?),
        })
    }

    /// Cumulative distribution function.
    pub fn cdf(&self, x: f64) -> Result<f64> {
        Ok(self.dist()?.cdf(x))
    }

    /// Quantile function.
    pub fn quantile(&self, p: f64) -> Result<f64> {
        Ok(self.dist()?.quantile(p))
    }
}

/// Maximum-likelihood location and scale of a logistic sample. For a given
/// scale the location solves Σ tanh((xᵢ − μ)/2s) = 0 (monotone in μ); the scale
/// then solves Σ zᵢ tanh(zᵢ/2) = n (monotone in s), both by bracketed root finding.
pub(crate) fn logistic_mle(y: &[f64]) -> Result<(f64, f64)> {
    let n = y.len() as f64;
    let (lo, hi) =
        (y.iter().copied().fold(f64::INFINITY, f64::min), y.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let mean = y.iter().sum::<f64>() / n;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    if sd.is_nan() || sd <= 0.0 {
        return invalid("the data are all equal; no distribution can be fitted");
    }
    let location = |s: f64| -> Result<f64> {
        Ok(brent(|m| y.iter().map(|v| ((v - m) / (2.0 * s)).tanh()).sum::<f64>(), lo, hi, 1e-13 * sd, 500)?)
    };
    let g = |s: f64| -> f64 {
        match location(s) {
            Ok(m) => y.iter().map(|v| ((v - m) / s) * ((v - m) / (2.0 * s)).tanh()).sum::<f64>() - n,
            Err(_) => f64::NAN,
        }
    };
    let s = brent(g, sd * 1e-4, sd * 1e3, 1e-13 * sd, 500)?;
    Ok((location(s)?, s))
}

fn mean_log(y: &[f64]) -> f64 {
    y.iter().map(|v| v.ln()).sum::<f64>() / y.len() as f64
}

fn finish_fit(family: Family, p1: f64, p2: f64, thr: f64, sorted: &[f64]) -> Result<FittedDistribution> {
    let mut f = FittedDistribution {
        family,
        param1: p1,
        param2: p2,
        threshold: thr,
        n: sorted.len(),
        log_likelihood: 0.0,
        ad: 0.0,
    };
    let d = f.dist()?;
    f.log_likelihood = sorted.iter().map(|&x| d.ln_pdf(x)).sum();
    let n = sorted.len();
    let nf = n as f64;
    let mut s = 0.0;
    for i in 0..n {
        let lo = d.cdf(sorted[i]).clamp(1e-300, 1.0);
        let hi = d.sf(sorted[n - 1 - i]).clamp(1e-300, 1.0);
        s += (2.0 * (i as f64) + 1.0) * (lo.ln() + hi.ln());
    }
    f.ad = -nf - s / nf;
    Ok(f)
}

/// Maximum-likelihood fit with a fixed threshold (default 0).
pub fn fit_distribution(data: &[f64], family: Family, threshold: Option<f64>) -> Result<FittedDistribution> {
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let thr = if family.uses_threshold() { threshold.unwrap_or(0.0) } else { 0.0 };
    if !thr.is_finite() {
        return invalid("the threshold must be finite");
    }
    let mut x: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    if x.len() < 3 {
        return Err(StatsError::InsufficientData { what: "distribution fit", needed: 3, got: x.len() });
    }
    x.sort_by(|a, b| a.total_cmp(b));
    if family.uses_threshold() && x[0] <= thr {
        return invalid(format!("every observation must exceed the threshold {thr} (smallest is {})", x[0]));
    }
    if x[0] == x[x.len() - 1] {
        return invalid("the data are all equal; no distribution can be fitted");
    }
    let y: Vec<f64> = x.iter().map(|v| v - thr).collect();
    let n = y.len() as f64;
    let ly: Vec<f64> = if matches!(family, Family::Normal | Family::Logistic) {
        Vec::new()
    } else {
        y.iter().map(|v| v.ln()).collect()
    };
    let (p1, p2) = match family {
        Family::Exponential => (y.iter().sum::<f64>() / n, f64::NAN),
        Family::Normal => {
            let m = y.iter().sum::<f64>() / n;
            (m, (y.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt())
        }
        Family::Logistic => logistic_mle(&y)?,
        Family::Loglogistic => logistic_mle(&ly)?,
        Family::Weibull => {
            let lmax = ly[ly.len() - 1];
            let ml = ly.iter().sum::<f64>() / n;
            let g = |b: f64| {
                let (mut s0, mut s1) = (0.0, 0.0);
                for &l in &ly {
                    let w = (b * (l - lmax)).exp();
                    s0 += w;
                    s1 += w * l;
                }
                s1 / s0 - 1.0 / b - ml
            };
            let beta = brent(g, 1e-3, 1e3, 1e-14, 500)?;
            let s0: f64 = ly.iter().map(|&l| (beta * (l - lmax)).exp()).sum();
            (beta, (lmax + (s0 / n).ln() / beta).exp())
        }
        Family::Lognormal => {
            let mu = mean_log(&y);
            let sigma = (ly.iter().map(|l| (l - mu) * (l - mu)).sum::<f64>() / n).sqrt();
            (mu, sigma)
        }
        Family::Gamma => {
            let mean = y.iter().sum::<f64>() / n;
            let s = mean.ln() - mean_log(&y);
            let k = brent(|k| k.ln() - digamma(k) - s, 1e-4, 1e9, 1e-14, 500)?;
            (k, mean / k)
        }
    };
    finish_fit(family, p1, p2, thr, &x)
}

/// Fit the Weibull, lognormal and gamma families and order them by Anderson–Darling statistic (best first).
pub fn rank_distributions(data: &[f64], threshold: Option<f64>) -> Vec<FittedDistribution> {
    rank_families(data, threshold, &[Family::Weibull, Family::Lognormal, Family::Gamma])
}

/// Fit the listed families and order them by Anderson–Darling statistic (best first).
pub fn rank_families(data: &[f64], threshold: Option<f64>, families: &[Family]) -> Vec<FittedDistribution> {
    let mut fits: Vec<FittedDistribution> =
        families.iter().copied().filter_map(|f| fit_distribution(data, f, threshold).ok()).collect();
    fits.sort_by(|a, b| a.ad.total_cmp(&b.ad));
    fits
}

/// Capability from a fitted non-normal distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionCapability {
    /// The fitted distribution.
    pub fit: FittedDistribution,
    /// Specifications.
    pub specs: Specs,
    /// Median of the fitted distribution.
    pub median: f64,
    /// 0.135 percentile.
    pub x_low: f64,
    /// 99.865 percentile.
    pub x_high: f64,
    /// Pp.
    pub pp: Option<f64>,
    /// PPL.
    pub ppl: Option<f64>,
    /// PPU.
    pub ppu: Option<f64>,
    /// Ppk.
    pub ppk: f64,
    /// Observed performance.
    pub observed: ObservedPerformance,
    /// Expected overall performance.
    pub expected: ExpectedPerformance,
}

/// Fit `family` to the data and compute Pp, Ppk, Z.Bench and PPM from the fitted distribution.
pub fn capability_distribution(
    data: &[f64],
    family: Family,
    threshold: Option<f64>,
    specs: &Specs,
) -> Result<DistributionCapability> {
    specs.validate()?;
    let fit = fit_distribution(data, family, threshold)?;
    let d = fit.dist()?;
    let p3 = norm_cdf(-3.0);
    let (lo, med, hi) = (d.quantile(p3), d.quantile(0.5), d.quantile(1.0 - p3));
    let ppl = specs.lsl.map(|l| (med - l) / (med - lo));
    let ppu = specs.usl.map(|u| (u - med) / (hi - med));
    let pp = match (specs.lsl, specs.usl) {
        (Some(l), Some(u)) => Some((u - l) / (hi - lo)),
        _ => None,
    };
    let ppk = match (ppl, ppu) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => f64::NAN,
    };
    let below = specs.lsl.map_or(0.0, |l| d.cdf(l));
    let above = specs.usl.map_or(0.0, |u| d.sf(u));
    let valid: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    Ok(DistributionCapability {
        fit,
        specs: *specs,
        median: med,
        x_low: lo,
        x_high: hi,
        pp,
        ppl,
        ppu,
        ppk,
        observed: observed(&valid, specs),
        expected: expected_from(below, above, specs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_math::dist::Normal;

    fn normal_data(n: usize, mu: f64, sd: f64) -> Vec<f64> {
        let nd = Normal::standard();
        (1..=n).map(|i| mu + sd * nd.quantile((i as f64 - 0.5) / n as f64)).collect()
    }

    #[test]
    fn indices_from_the_definitions() {
        // Overall σ and mean known exactly from the data; within σ supplied directly.
        let data = vec![9.0, 10.0, 11.0, 10.0, 12.0, 8.0];
        let specs = Specs::two_sided(4.0, 16.5).with_target(10.0);
        let c = normal_capability(&data, 1.2, "test".into(), &specs, 0.95).unwrap();
        let mean = 10.0;
        let sd = (10.0f64 / 5.0).sqrt();
        assert!((c.mean - mean).abs() < 1e-14 && (c.sd_overall - sd).abs() < 1e-14);
        assert!((c.cp.unwrap() - 12.5 / (6.0 * 1.2)).abs() < 1e-14);
        assert!((c.cpl.unwrap() - 6.0 / 3.6).abs() < 1e-14 && (c.cpu.unwrap() - 6.5 / 3.6).abs() < 1e-14);
        assert!((c.cpk - 6.0 / 3.6).abs() < 1e-14);
        assert!((c.pp.unwrap() - 12.5 / (6.0 * sd)).abs() < 1e-14);
        assert!((c.ppk - 6.0 / (3.0 * sd)).abs() < 1e-14);
        assert!((c.cpm.unwrap() - 12.5 / (6.0 * sd)).abs() < 1e-14); // mean = target
                                                                     // Expected performance: Φ(−Z).
        let zl = 6.0 / 1.2;
        assert!((c.expected_within.z_lsl.unwrap() - zl).abs() < 1e-9);
        assert!((c.expected_within.ppm_below - norm_cdf(-zl) * 1e6).abs() < 1e-9);
        let total = norm_cdf(-zl) + openstats_math::special::erf::norm_sf(6.5 / 1.2);
        assert!((c.expected_within.z_bench - norm_isf(total)).abs() < 1e-9);
        assert_eq!(c.observed.n_below + c.observed.n_above, 0);
    }

    #[test]
    fn one_sided_specs_and_observed_counts() {
        let data = normal_data(1000, 50.0, 2.0);
        let c =
            normal_capability(&data, 2.0, "x".into(), &Specs { usl: Some(57.0), ..Default::default() }, 0.9).unwrap();
        assert!(c.cp.is_none() && c.pp.is_none() && c.cpl.is_none());
        assert!((c.cpk - (57.0 - 50.0) / 6.0).abs() < 1e-3);
        assert_eq!(c.observed.n_above, data.iter().filter(|&&x| x > 57.0).count());
        assert!((c.observed.ppm_above - c.observed.n_above as f64 * 1000.0).abs() < 1e-9);
        assert!(c.pp_ci.is_none() && c.ppk_ci.is_some());
    }

    #[test]
    fn pp_interval_and_validation() {
        let data = normal_data(50, 10.0, 1.0);
        let c = normal_capability(&data, 1.0, "x".into(), &Specs::two_sided(7.0, 13.0), 0.95).unwrap();
        let ci = c.pp_ci.unwrap();
        assert!(ci.lower < c.pp.unwrap() && c.pp.unwrap() < ci.upper);
        let chi = ChiSquared::new(49.0).unwrap();
        assert!((ci.lower - c.pp.unwrap() * (chi.quantile(0.025) / 49.0).sqrt()).abs() < 1e-12);
        assert!(normal_capability(&data, 1.0, "x".into(), &Specs::default(), 0.95).is_err());
        assert!(normal_capability(&data, 1.0, "x".into(), &Specs::two_sided(5.0, 5.0), 0.95).is_err());
        assert!(normal_capability(&[1.0, 1.0, 1.0], 1.0, "x".into(), &Specs::two_sided(0.0, 2.0), 0.95).is_err());
        assert!(normal_capability(&data, 1.0, "x".into(), &Specs::two_sided(7.0, 13.0), 1.0).is_err());
    }

    #[test]
    fn individuals_and_subgroups_entry_points() {
        let data: Vec<f64> = (0..40).map(|i| 10.0 + ((i * 7) % 11) as f64 * 0.1).collect();
        let specs = Specs::two_sided(9.0, 12.0);
        let c = capability_individuals(&data, &specs, &CapabilityOptions::default()).unwrap();
        assert!(c.within_method.contains("moving range") && c.n == 40);
        let groups: Vec<Vec<f64>> = data.chunks(5).map(<[f64]>::to_vec).collect();
        let s = capability_subgroups(&groups, &specs, &CapabilityOptions::default()).unwrap();
        assert!(s.within_method.starts_with("R̄"));
        let pooled = capability_subgroups(
            &groups,
            &specs,
            &CapabilityOptions { subgroup_sigma: Some(SubgroupSigma::Pooled), ..Default::default() },
        )
        .unwrap();
        assert!(pooled.within_method.contains("pooled"));
        assert_eq!(s.sd_overall, c.sd_overall);
    }

    #[test]
    fn distribution_fits_recover_parameters() {
        // Quantile samples from known distributions: the MLE should be close.
        let n = 2000;
        let w = Weibull::new(2.5, 10.0, 0.0).unwrap();
        let xs: Vec<f64> = (1..=n).map(|i| w.quantile((i as f64 - 0.5) / n as f64)).collect();
        let f = fit_distribution(&xs, Family::Weibull, None).unwrap();
        assert!((f.param1 - 2.5).abs() < 0.05 && (f.param2 - 10.0).abs() < 0.1, "{f:?}");
        let g = Gamma::new(4.0, 2.0, 0.0).unwrap();
        let xs: Vec<f64> = (1..=n).map(|i| g.quantile((i as f64 - 0.5) / n as f64)).collect();
        let f = fit_distribution(&xs, Family::Gamma, None).unwrap();
        assert!((f.param1 - 4.0).abs() < 0.1 && (f.param2 - 2.0).abs() < 0.05, "{f:?}");
        let l = Lognormal::new(1.0, 0.4, 3.0).unwrap();
        let xs: Vec<f64> = (1..=n).map(|i| l.quantile((i as f64 - 0.5) / n as f64)).collect();
        let f = fit_distribution(&xs, Family::Lognormal, Some(3.0)).unwrap();
        assert!((f.param1 - 1.0).abs() < 0.01 && (f.param2 - 0.4).abs() < 0.01, "{f:?}");
        // The true family has the smallest A² among the three.
        let ranked = rank_distributions(&xs, Some(3.0));
        assert_eq!(ranked[0].family, Family::Lognormal);
        assert!(fit_distribution(&[1.0, 2.0, 3.0], Family::Weibull, Some(1.0)).is_err());
        assert!(fit_distribution(&[2.0, 2.0, 2.0], Family::Gamma, None).is_err());
    }

    #[test]
    fn mle_satisfies_score_equations() {
        let w = Weibull::new(1.7, 5.0, 0.0).unwrap();
        let xs: Vec<f64> = (1..=300).map(|i| w.quantile((i as f64 - 0.5) / 300.0)).collect();
        let f = fit_distribution(&xs, Family::Weibull, None).unwrap();
        // Perturbing either parameter lowers the log-likelihood.
        let ll = |b: f64, s: f64| {
            let d = Weibull::new(b, s, 0.0).unwrap();
            xs.iter().map(|&x| d.ln_pdf(x)).sum::<f64>()
        };
        for (db, ds) in [(1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-3), (0.0, -1e-3)] {
            assert!(ll(f.param1 + db, f.param2 + ds) < f.log_likelihood + 1e-9);
        }
    }

    #[test]
    fn distribution_capability_matches_quantile_definitions() {
        let n = 1500;
        let g = Gamma::new(3.0, 1.0, 0.0).unwrap();
        let xs: Vec<f64> = (1..=n).map(|i| g.quantile((i as f64 - 0.5) / n as f64)).collect();
        let specs = Specs::two_sided(0.2, 12.0);
        let c = capability_distribution(&xs, Family::Gamma, None, &specs).unwrap();
        let d = c.fit.dist().unwrap();
        assert!((c.pp.unwrap() - (12.0 - 0.2) / (c.x_high - c.x_low)).abs() < 1e-12);
        assert!((c.ppu.unwrap() - (12.0 - c.median) / (c.x_high - c.median)).abs() < 1e-12);
        assert!((c.expected.ppm_above - d.sf(12.0) * 1e6).abs() < 1e-9);
        assert!((c.expected.z_bench - norm_isf(d.cdf(0.2) + d.sf(12.0))).abs() < 1e-9);
        assert!(c.ppk <= c.ppl.unwrap() && c.ppk <= c.ppu.unwrap());
    }

    #[test]
    fn transformed_capability_equals_normal_on_transformed_scale() {
        use super::super::transform::{BoxCox, LambdaChoice};
        let nd = Normal::standard();
        let xs: Vec<f64> = (1..=100).map(|i| (1.0 + 0.3 * nd.quantile((i as f64 - 0.5) / 100.0)).exp()).collect();
        // Deterministic shuffle so moving ranges are not trivially small.
        let data: Vec<f64> = (0..100).map(|i| xs[(i * 37) % 100]).collect();
        let b = BoxCox::fit(&data, LambdaChoice::Fixed(0.0)).unwrap();
        let t = Transformation::BoxCox(b);
        let specs = Specs::two_sided(0.8, 7.0);
        let c = capability_transformed_individuals(&data, &t, &specs, &CapabilityOptions::default()).unwrap();
        let z: Vec<f64> = data.iter().map(|x| x.ln()).collect();
        let direct =
            capability_individuals(&z, &Specs::two_sided(0.8f64.ln(), 7.0f64.ln()), &CapabilityOptions::default())
                .unwrap();
        assert!((c.normal.ppk - direct.ppk).abs() < 1e-12 && (c.normal.cpk - direct.cpk).abs() < 1e-12);
        assert!(c.ad.p_value > 0.5);
        assert!(capability_transformed_individuals(&[-1.0, 2.0, 3.0, 4.0], &t, &specs, &CapabilityOptions::default())
            .is_err());
    }
}
