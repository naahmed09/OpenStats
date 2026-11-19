//! Repairable systems: the Nelson mean cumulative function and the Crow-AMSAA (power-law
//! non-homogeneous Poisson process) reliability-growth model.

use super::fit::zcrit;
use crate::error::{invalid, Result};
use openstats_math::dist::{ChiSquared, ContinuousDistribution, Normal};

/// One repairable system: its observation window `(0, end]` and the times of its events.
#[derive(Debug, Clone, PartialEq)]
pub struct System {
    /// End of observation (retirement or end of test).
    pub end: f64,
    /// Event (repair / failure) times, any order.
    pub events: Vec<f64>,
}

/// Build systems from long-format data. `end` (when given) is the observation end of the row's
/// system (the largest value per system is used); otherwise each system ends at its last event.
pub fn systems_from_rows(system: &[String], time: &[f64], end: Option<&[f64]>) -> Result<Vec<System>> {
    if system.len() != time.len() || end.is_some_and(|e| e.len() != time.len()) {
        return invalid("the system, time and end columns must have the same length");
    }
    let mut ids: Vec<String> = Vec::new();
    let mut out: Vec<System> = Vec::new();
    for i in 0..time.len() {
        if system[i].is_empty() || time[i].is_nan() {
            continue;
        }
        if time[i] < 0.0 || time[i].is_infinite() {
            return invalid("event times must be non-negative and finite");
        }
        let k = match ids.iter().position(|s| *s == system[i]) {
            Some(k) => k,
            None => {
                ids.push(system[i].clone());
                out.push(System { end: 0.0, events: vec![] });
                ids.len() - 1
            }
        };
        let e = end.map_or(f64::NAN, |e| e[i]);
        // A row whose time equals the end value is a pure retirement marker, not an event.
        if e.is_nan() || time[i] < e {
            out[k].events.push(time[i]);
        }
        if !e.is_nan() {
            out[k].end = out[k].end.max(e);
        }
    }
    for s in &mut out {
        s.events.sort_by(f64::total_cmp);
        let last = s.events.last().copied().unwrap_or(0.0);
        if s.end < last {
            s.end = last;
        }
        if s.end <= 0.0 {
            return invalid("every system needs a positive observation time");
        }
    }
    if out.is_empty() {
        return invalid("no systems with data");
    }
    Ok(out)
}

fn check(systems: &[System]) -> Result<()> {
    if systems.is_empty() {
        return invalid("no systems");
    }
    for s in systems {
        if !(s.end > 0.0) || !s.end.is_finite() {
            return invalid("every system needs a positive observation time");
        }
        if s.events.iter().any(|&t| !(t > 0.0) || t > s.end) {
            return invalid("event times must lie in (0, end of observation]");
        }
    }
    if systems.iter().all(|s| s.events.is_empty()) {
        return invalid("no events were observed");
    }
    Ok(())
}

/// One step of the mean cumulative function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McfPoint {
    /// Event time.
    pub time: f64,
    /// Events at this time over all systems.
    pub events: f64,
    /// Systems still under observation.
    pub at_risk: f64,
    /// MCF estimate.
    pub mcf: f64,
    /// Standard error (Lawless–Nadeau).
    pub se: f64,
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
}

/// Nelson's non-parametric mean cumulative function with Lawless–Nadeau variances and
/// log-scale normal confidence limits.
pub fn mean_cumulative_function(systems: &[System], level: f64) -> Result<Vec<McfPoint>> {
    check(systems)?;
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    let mut times: Vec<f64> = systems.iter().flat_map(|s| s.events.iter().copied()).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let z = zcrit(level);
    // Per-system running sums of δ_ij (dN_ij − d_j/r_j)/r_j.
    let mut acc = vec![0.0; systems.len()];
    let mut mcf = 0.0;
    let mut out = Vec::with_capacity(times.len());
    for &t in &times {
        let r = systems.iter().filter(|s| s.end >= t).count() as f64;
        let dn: Vec<f64> = systems.iter().map(|s| s.events.iter().filter(|&&e| e == t).count() as f64).collect();
        let d: f64 = dn.iter().sum();
        mcf += d / r;
        for (i, s) in systems.iter().enumerate() {
            if s.end >= t {
                acc[i] += (dn[i] - d / r) / r;
            }
        }
        let var: f64 = acc.iter().map(|a| a * a).sum();
        let se = var.sqrt();
        let (lower, upper) = if mcf > 0.0 && se > 0.0 {
            let w = (z * se / mcf).exp();
            (mcf / w, mcf * w)
        } else {
            (mcf, mcf)
        };
        out.push(McfPoint { time: t, events: d, at_risk: r, mcf, se, lower, upper });
    }
    Ok(out)
}

/// How the observation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// Observation stopped at a fixed time.
    TimeTruncated,
    /// Observation stopped at the n-th failure (single system).
    FailureTruncated,
}

/// A Crow-AMSAA fit: intensity `λ β t^(β−1)`, expected events `λ t^β`.
#[derive(Debug, Clone)]
pub struct CrowAmsaa {
    /// Shape β (MLE).
    pub beta: f64,
    /// Scale λ.
    pub lambda: f64,
    /// Bias-corrected shape `(N−1)β̂/N`.
    pub beta_unbiased: f64,
    /// Confidence limits for β.
    pub beta_lower: f64,
    /// Upper confidence limit for β.
    pub beta_upper: f64,
    /// Events.
    pub n_events: f64,
    /// Systems.
    pub n_systems: usize,
    /// Total observation time.
    pub total_time: f64,
    /// Test of β = 1 (2N/β̂ against χ²): statistic.
    pub chi_stat: f64,
    /// Two-sided p-value of the β = 1 test.
    pub chi_p: f64,
    /// Laplace trend statistic (standard normal under a homogeneous Poisson process).
    pub laplace_u: f64,
    /// Two-sided p-value of the Laplace test.
    pub laplace_p: f64,
}

impl CrowAmsaa {
    /// Instantaneous intensity at `t`.
    pub fn intensity(&self, t: f64) -> f64 {
        self.lambda * self.beta * t.powf(self.beta - 1.0)
    }
    /// Expected cumulative events per system at `t`.
    pub fn expected(&self, t: f64) -> f64 {
        self.lambda * t.powf(self.beta)
    }
    /// Instantaneous mean time between failures at `t`.
    pub fn mtbf_instant(&self, t: f64) -> f64 {
        1.0 / self.intensity(t)
    }
    /// Cumulative mean time between failures `t / E[N(t)]`.
    pub fn mtbf_cumulative(&self, t: f64) -> f64 {
        t / self.expected(t)
    }
}

/// Fit the power-law process by maximum likelihood, pooling systems that share β and λ.
pub fn crow_amsaa(systems: &[System], termination: Termination, level: f64) -> Result<CrowAmsaa> {
    check(systems)?;
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be between 0 and 1");
    }
    if termination == Termination::FailureTruncated && systems.len() != 1 {
        return invalid("failure-truncated Crow-AMSAA applies to a single system");
    }
    let n: f64 = systems.iter().map(|s| s.events.len() as f64).sum();
    if n < 2.0 {
        return invalid("Crow-AMSAA needs at least two events");
    }
    let mut sum_ln = 0.0;
    for s in systems {
        let horizon = match termination {
            Termination::TimeTruncated => s.end,
            Termination::FailureTruncated => s.events.last().copied().unwrap_or(s.end),
        };
        sum_ln += s.events.iter().map(|&t| (horizon / t).ln()).sum::<f64>();
    }
    if !(sum_ln > 0.0) {
        return invalid("the events carry no information about the trend (all at the end time)");
    }
    let beta = n / sum_ln;
    let denom: f64 = systems
        .iter()
        .map(|s| {
            match termination {
                Termination::TimeTruncated => s.end,
                Termination::FailureTruncated => s.events.last().copied().unwrap_or(s.end),
            }
            .powf(beta)
        })
        .sum();
    let lambda = n / denom;
    let df = match termination {
        Termination::TimeTruncated => 2.0 * n,
        Termination::FailureTruncated => 2.0 * (n - 1.0),
    };
    let chi = ChiSquared::new(df)?;
    let a = (1.0 - level) / 2.0;
    // 2Nβ/β̂ ~ χ²(df)
    let beta_lower = beta * chi.quantile(a) / (2.0 * n);
    let beta_upper = beta * chi.quantile(1.0 - a) / (2.0 * n);
    let chi_stat = 2.0 * n / beta;
    let chi_n = ChiSquared::new(df)?;
    let pl = chi_n.cdf(chi_stat);
    let chi_p = (2.0 * pl.min(1.0 - pl)).min(1.0);
    // Laplace test, pooled over systems.
    let (mut s_t, mut mean_part, mut var_part) = (0.0, 0.0, 0.0);
    for s in systems {
        let horizon = match termination {
            Termination::TimeTruncated => s.end,
            Termination::FailureTruncated => s.events.last().copied().unwrap_or(s.end),
        };
        let m = match termination {
            Termination::TimeTruncated => s.events.len() as f64,
            Termination::FailureTruncated => s.events.len() as f64 - 1.0,
        };
        let sum: f64 = match termination {
            Termination::TimeTruncated => s.events.iter().sum(),
            Termination::FailureTruncated => s.events[..s.events.len() - 1].iter().sum(),
        };
        s_t += sum;
        mean_part += m * horizon / 2.0;
        var_part += m * horizon * horizon / 12.0;
    }
    let laplace_u = (s_t - mean_part) / var_part.sqrt();
    let laplace_p = 2.0 * Normal::standard().sf(laplace_u.abs());
    Ok(CrowAmsaa {
        beta,
        lambda,
        beta_unbiased: (n - 1.0) * beta / n,
        beta_lower,
        beta_upper,
        n_events: n,
        n_systems: systems.len(),
        total_time: systems.iter().map(|s| s.end).sum(),
        chi_stat,
        chi_p,
        laplace_u,
        laplace_p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(end: f64, ev: &[f64]) -> System {
        System { end, events: ev.to_vec() }
    }

    #[test]
    fn mcf_matches_hand_computation() {
        // two systems: A events at 2, 5 (end 8); B event at 5 (end 6)
        let s = [sys(8.0, &[2.0, 5.0]), sys(6.0, &[5.0])];
        let m = mean_cumulative_function(&s, 0.95).unwrap();
        assert_eq!(m.len(), 2);
        assert!((m[0].mcf - 0.5).abs() < 1e-12);
        assert!((m[1].mcf - 1.5).abs() < 1e-12);
        // variance at t=2: A: (1-0.5)/2 = .25, B: (0-.5)/2 = -.25 -> .125
        assert!((m[0].se - 0.125_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mcf_drops_retired_systems() {
        let s = [sys(3.0, &[1.0]), sys(10.0, &[1.0, 9.0])];
        let m = mean_cumulative_function(&s, 0.9).unwrap();
        assert_eq!(m[1].at_risk, 1.0);
        assert!((m[1].mcf - 2.0).abs() < 1e-12);
    }

    #[test]
    fn crow_single_system_formula() {
        let t = [5.0, 40.0, 43.0, 175.0, 389.0, 712.0, 747.0, 795.0, 1299.0, 1478.0];
        let c = crow_amsaa(&[sys(1500.0, &t)], Termination::TimeTruncated, 0.95).unwrap();
        let sl: f64 = t.iter().map(|x| (1500.0 / x).ln()).sum();
        assert!((c.beta - 10.0 / sl).abs() < 1e-12);
        assert!((c.lambda - 10.0 / 1500f64.powf(c.beta)).abs() < 1e-12);
        assert!(c.beta_lower < c.beta && c.beta < c.beta_upper);
        assert!((c.beta_unbiased - 0.9 * c.beta).abs() < 1e-12);
    }

    #[test]
    fn hpp_has_no_trend() {
        // evenly spread events: beta near 1, Laplace near 0
        let t: Vec<f64> = (1..=20).map(|i| i as f64 * 5.0 - 2.5).collect();
        let c = crow_amsaa(&[sys(100.0, &t)], Termination::TimeTruncated, 0.95).unwrap();
        assert!(c.laplace_u.abs() < 1e-9);
        assert!(c.laplace_p > 0.99);
        assert!(c.chi_p > 0.05);
    }

    #[test]
    fn errors() {
        assert!(crow_amsaa(&[sys(10.0, &[1.0])], Termination::TimeTruncated, 0.95).is_err());
        assert!(mean_cumulative_function(&[sys(10.0, &[])], 0.95).is_err());
        assert!(
            crow_amsaa(&[sys(10.0, &[1.0, 2.0]), sys(5.0, &[1.0, 2.0])], Termination::FailureTruncated, 0.95).is_err()
        );
    }

    #[test]
    fn rows_builder() {
        let id: Vec<String> = ["a", "a", "b", "b"].iter().map(|s| s.to_string()).collect();
        let t = [2.0, 8.0, 3.0, 6.0];
        let e = [8.0, 8.0, 6.0, 6.0];
        let s = systems_from_rows(&id, &t, Some(&e)).unwrap();
        assert_eq!(s[0].events, vec![2.0]);
        assert_eq!(s[1].events, vec![3.0]);
        assert_eq!(s[1].end, 6.0);
    }
}
