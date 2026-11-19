//! Wald–Wolfowitz runs test.

use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ContinuousDistribution, Normal};
use openstats_math::special::gamma::ln_factorial;

/// Result of the runs test.
#[derive(Debug, Clone, PartialEq)]
pub struct RunsTest {
    /// Observations used (values equal to the reference are omitted).
    pub n: usize,
    /// Reference value (mean, median or a given K).
    pub k: f64,
    /// Observations above and below the reference.
    pub above: usize,
    pub below: usize,
    /// Observations equal to the reference (omitted).
    pub equal: usize,
    /// Observed number of runs.
    pub runs: usize,
    /// Expected number of runs under randomness.
    pub expected: f64,
    /// Standardised statistic.
    pub z: f64,
    /// Two-sided p-value from the normal approximation.
    pub p_value: f64,
    /// Exact two-sided p-value (twice the smaller tail), when both counts are at most 100.
    pub exact_p_value: Option<f64>,
}

fn ln_choose(n: usize, k: usize) -> f64 {
    ln_factorial(n as u64) - ln_factorial(k as u64) - ln_factorial((n - k) as u64)
}

/// Exact distribution of the number of runs: P(R = r) for r = 2..=n1+n2.
fn runs_pmf(n1: usize, n2: usize) -> Vec<(usize, f64)> {
    let total = ln_choose(n1 + n2, n1);
    let mut out = Vec::new();
    for r in 2..=(n1 + n2).min(2 * n1.min(n2) + 1) {
        let p = if r % 2 == 0 {
            let k = r / 2;
            2.0 * (ln_choose(n1 - 1, k - 1) + ln_choose(n2 - 1, k - 1) - total).exp()
        } else {
            let k = r / 2;
            let a =
                if k < n1 && k >= 1 { (ln_choose(n1 - 1, k) + ln_choose(n2 - 1, k - 1) - total).exp() } else { 0.0 };
            let b =
                if k < n2 && k >= 1 { (ln_choose(n1 - 1, k - 1) + ln_choose(n2 - 1, k) - total).exp() } else { 0.0 };
            a + b
        };
        out.push((r, p));
    }
    out
}

/// Runs test about `k`. Missing values are dropped; the order of `x` is the order of the data.
pub fn runs_test(x: &[f64], k: f64) -> Result<RunsTest> {
    if !k.is_finite() {
        return invalid("the reference value must be finite");
    }
    let v = crate::clean(x)?;
    let equal = v.iter().filter(|&&a| a == k).count();
    let signs: Vec<bool> = v.iter().filter(|&&a| a != k).map(|&a| a > k).collect();
    let n = signs.len();
    let above = signs.iter().filter(|&&s| s).count();
    let below = n - above;
    if above == 0 || below == 0 {
        return Err(StatsError::InsufficientData {
            what: "the runs test (values on each side of the reference)",
            needed: 1,
            got: above.min(below),
        });
    }
    let runs = 1 + signs.windows(2).filter(|w| w[0] != w[1]).count();
    let (n1, n2, nf) = (above as f64, below as f64, n as f64);
    let expected = 1.0 + 2.0 * n1 * n2 / nf;
    let var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - nf) / (nf * nf * (nf - 1.0));
    if var <= 0.0 {
        return invalid("the runs statistic has zero variance (one value on a side)");
    }
    let z = (runs as f64 - expected) / var.sqrt();
    let nrm = Normal::new(0.0, 1.0)?;
    let p = (2.0 * nrm.sf(z.abs())).min(1.0);
    let exact = (above <= 100 && below <= 100).then(|| {
        let pm = runs_pmf(above, below);
        let lower: f64 = pm.iter().filter(|(r, _)| *r <= runs).map(|(_, p)| p).sum();
        let upper: f64 = pm.iter().filter(|(r, _)| *r >= runs).map(|(_, p)| p).sum();
        (2.0 * lower.min(upper)).min(1.0)
    });
    Ok(RunsTest { n, k, above, below, equal, runs, expected, z, p_value: p, exact_p_value: exact })
}
