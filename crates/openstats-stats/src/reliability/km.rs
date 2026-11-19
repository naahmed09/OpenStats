//! Kaplan–Meier estimation and comparison of survival curves.

use super::data::{LifeData, ObsKind};
use super::fit::zcrit;
use crate::error::{invalid, Result};
use openstats_math::dist::{ChiSquared, ContinuousDistribution};
use openstats_math::linalg::Cholesky;
use openstats_math::Matrix;

/// Confidence-interval method for the survival function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiMethod {
    /// `S ± z·se` (Greenwood), clamped to [0, 1].
    Plain,
    /// Complementary log-log transformation (stays inside [0, 1]).
    LogLog,
}

/// One time point of the estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KmStep {
    /// Time (a failure or a censoring time).
    pub time: f64,
    /// Units at risk just before `time`.
    pub at_risk: f64,
    /// Failures at `time`.
    pub failed: f64,
    /// Censored at `time`.
    pub censored: f64,
    /// Survival probability just after `time`.
    pub survival: f64,
    /// Greenwood standard error.
    pub se: f64,
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
    /// Nelson–Aalen cumulative hazard.
    pub cum_hazard: f64,
}

/// A Kaplan–Meier estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct KaplanMeier {
    /// Time points in increasing order.
    pub steps: Vec<KmStep>,
    /// Units.
    pub n: f64,
    /// Failures.
    pub failures: f64,
    /// Confidence level.
    pub level: f64,
    /// Interval method.
    pub method: CiMethod,
    /// Median survival time (`NaN` when the curve stays above 0.5).
    pub median: f64,
    /// Confidence limits of the median (`NaN` when not bounded by the data).
    pub median_ci: (f64, f64),
    /// First quartile (survival 0.75) with limits.
    pub q1: (f64, f64, f64),
    /// Third quartile (survival 0.25) with limits.
    pub q3: (f64, f64, f64),
    /// Mean survival time, restricted to the largest observation.
    pub mean: f64,
    /// Standard error of the restricted mean.
    pub mean_se: f64,
    /// `true` when the largest observation is censored, so the mean is a restricted mean.
    pub mean_restricted: bool,
}

/// Smallest time at which `f(step)` is `≤ target`.
fn first_below(steps: &[KmStep], target: f64, f: impl Fn(&KmStep) -> f64) -> f64 {
    steps.iter().find(|s| f(s) <= target).map_or(f64::NAN, |s| s.time)
}

/// Kaplan–Meier estimate from exact and right-censored data.
pub fn kaplan_meier(data: &LifeData, level: f64, method: CiMethod) -> Result<KaplanMeier> {
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    if data.obs.iter().any(|o| matches!(o.kind(), ObsKind::Left | ObsKind::Interval)) {
        return invalid(
            "Kaplan–Meier needs exact or right-censored times; use a parametric analysis for arbitrary censoring",
        );
    }
    let mut ev: Vec<(f64, f64, bool)> =
        data.obs.iter().map(|o| (o.lower, o.weight, o.kind() == ObsKind::Exact)).collect();
    ev.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = data.total();
    let z = zcrit(level);
    let (mut at_risk, mut s, mut green, mut h) = (total, 1.0, 0.0, 0.0);
    let mut steps = Vec::new();
    let mut i = 0;
    while i < ev.len() {
        let t = ev[i].0;
        let (mut d, mut c) = (0.0, 0.0);
        while i < ev.len() && ev[i].0 == t {
            if ev[i].2 {
                d += ev[i].1;
            } else {
                c += ev[i].1;
            }
            i += 1;
        }
        let n = at_risk;
        if d > 0.0 {
            s *= 1.0 - d / n;
            h += d / n;
            if n > d {
                green += d / (n * (n - d));
            } else {
                green = f64::INFINITY;
            }
        }
        let se = if s > 0.0 { s * green.sqrt() } else { 0.0 };
        let (lower, upper) = if s <= 0.0 || s >= 1.0 || !green.is_finite() {
            if s >= 1.0 {
                (1.0, 1.0)
            } else {
                (0.0, 0.0)
            }
        } else {
            match method {
                CiMethod::Plain => ((s - z * se).max(0.0), (s + z * se).min(1.0)),
                CiMethod::LogLog => {
                    let se_ll = green.sqrt() / s.ln().abs();
                    let lc = (-s.ln()).ln();
                    ((-(lc + z * se_ll).exp()).exp(), (-(lc - z * se_ll).exp()).exp())
                }
            }
        };
        steps.push(KmStep {
            time: t,
            at_risk: n,
            failed: d,
            censored: c,
            survival: s,
            se,
            lower,
            upper,
            cum_hazard: h,
        });
        at_risk -= d + c;
    }
    let tmax = steps.last().map_or(0.0, |s| s.time);
    // Restricted mean = ∫₀^tmax S(t) dt and its Greenwood-type standard error.
    let mut area = 0.0;
    let mut prev_t = 0.0;
    let mut prev_s = 1.0;
    let mut areas_after = vec![0.0; steps.len()];
    for st in &steps {
        area += prev_s * (st.time - prev_t);
        prev_t = st.time;
        prev_s = st.survival;
    }
    // A_i = ∫_{t_i}^{tmax} S dt
    let mut acc = 0.0;
    for k in (0..steps.len()).rev() {
        acc += steps[k].survival * (if k + 1 < steps.len() { steps[k + 1].time - steps[k].time } else { 0.0 });
        areas_after[k] = acc;
    }
    let mut var_mean = 0.0;
    for (k, st) in steps.iter().enumerate() {
        if st.failed > 0.0 && st.at_risk > st.failed {
            var_mean += areas_after[k].powi(2) * st.failed / (st.at_risk * (st.at_risk - st.failed));
        }
    }
    let last_censored = data.obs.iter().any(|o| o.kind() == ObsKind::Right && o.lower == tmax);
    let quant = |sv: f64| -> (f64, f64, f64) {
        (
            first_below(&steps, sv, |s| s.survival),
            first_below(&steps, sv, |s| s.upper),
            first_below(&steps, sv, |s| s.lower),
        )
    };
    let med = quant(0.5);
    Ok(KaplanMeier {
        n: total,
        failures: data.failures(),
        level,
        method,
        median: med.0,
        median_ci: (med.1, med.2),
        q1: quant(0.75),
        q3: quant(0.25),
        mean: area,
        mean_se: var_mean.sqrt(),
        mean_restricted: last_censored,
        steps,
    })
}

impl KaplanMeier {
    /// Survival probability at `t` (right-continuous step function).
    pub fn survival_at(&self, t: f64) -> f64 {
        self.steps.iter().take_while(|s| s.time <= t).last().map_or(1.0, |s| s.survival)
    }
}

/// Weights of the generalised log-rank family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    /// Log-rank (weight 1).
    LogRank,
    /// Wilcoxon / Breslow / Gehan (weight = number at risk).
    Wilcoxon,
    /// Tarone–Ware (weight = √ number at risk).
    TaroneWare,
}

/// Result of comparing survival curves.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupTest {
    /// Test used.
    pub kind: WeightKind,
    /// Observed failures per group.
    pub observed: Vec<f64>,
    /// Expected failures per group under equal survival.
    pub expected: Vec<f64>,
    /// χ² statistic.
    pub statistic: f64,
    /// Degrees of freedom (groups − 1).
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

/// Compare the survival curves of two or more groups.
pub fn compare_groups(groups: &[LifeData], kind: WeightKind) -> Result<GroupTest> {
    let g = groups.len();
    if g < 2 {
        return invalid("comparing survival curves needs at least two groups");
    }
    if groups.iter().flat_map(|d| &d.obs).any(|o| matches!(o.kind(), ObsKind::Left | ObsKind::Interval)) {
        return invalid("the log-rank family needs exact or right-censored times");
    }
    let mut times: Vec<f64> =
        groups.iter().flat_map(|d| d.obs.iter()).filter(|o| o.kind() == ObsKind::Exact).map(|o| o.lower).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    if times.is_empty() {
        return invalid("no failures: nothing to compare");
    }
    let (mut obs_g, mut exp_g) = (vec![0.0; g], vec![0.0; g]);
    let mut u = vec![0.0; g];
    let mut v = Matrix::zeros(g, g);
    for &t in &times {
        let mut nj = vec![0.0; g];
        let mut dj = vec![0.0; g];
        for (k, d) in groups.iter().enumerate() {
            for o in &d.obs {
                if o.lower >= t {
                    nj[k] += o.weight;
                }
                if o.kind() == ObsKind::Exact && o.lower == t {
                    dj[k] += o.weight;
                }
            }
        }
        let n: f64 = nj.iter().sum();
        let d: f64 = dj.iter().sum();
        let w = match kind {
            WeightKind::LogRank => 1.0,
            WeightKind::Wilcoxon => n,
            WeightKind::TaroneWare => n.sqrt(),
        };
        for k in 0..g {
            let e = d * nj[k] / n;
            obs_g[k] += dj[k];
            exp_g[k] += e;
            u[k] += w * (dj[k] - e);
        }
        if n > 1.0 {
            let c = w * w * d * (n - d) / (n - 1.0);
            for a in 0..g {
                for b in 0..g {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    v[(a, b)] += c * (nj[a] / n) * (delta - nj[b] / n);
                }
            }
        }
    }
    let m = g - 1;
    let mut vm = Matrix::zeros(m, m);
    for a in 0..m {
        for b in 0..m {
            vm[(a, b)] = v[(a, b)];
        }
    }
    let sol = Cholesky::new(&vm).and_then(|c| c.solve(&u[..m])).map_err(|_| {
        crate::error::StatsError::InvalidArgument("the groups do not have enough failures to compare".into())
    })?;
    let stat: f64 = u[..m].iter().zip(&sol).map(|(a, b)| a * b).sum();
    Ok(GroupTest {
        kind,
        observed: obs_g,
        expected: exp_g,
        statistic: stat,
        df: m,
        p_value: ChiSquared::new(m as f64)?.sf(stat),
    })
}
