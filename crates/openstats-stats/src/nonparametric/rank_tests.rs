//! Sign, Wilcoxon signed-rank and Mann–Whitney tests.

use super::tie_term;
use crate::correlation::average_ranks;
use crate::error::{invalid, Result, StatsError};
use crate::hypothesis::{check_level, Alternative};
use openstats_math::dist::{Binomial, ContinuousDistribution, DiscreteDistribution, Normal};

/// How a p-value is obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignRankMethod {
    /// Exact when there are no ties and the sample is small, otherwise normal.
    Auto,
    /// Exact null distribution (an error when there are ties or the sample is too large).
    Exact,
    /// Normal approximation with the tie correction.
    Normal,
}

const MAX_EXACT_SIGNRANK: usize = 50;
const MAX_EXACT_MW: usize = 50;
const MAX_PAIRS: usize = 5_000_000;

fn std_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal")
}

fn p_from_tails(lower: f64, upper: f64, alt: Alternative) -> f64 {
    match alt {
        Alternative::TwoSided => (2.0 * lower.min(upper)).min(1.0),
        Alternative::Less => lower,
        Alternative::Greater => upper,
    }
}

fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// ------------------------------------------------------------------ sign test

/// Result of the sign test.
#[derive(Debug, Clone, PartialEq)]
pub struct SignTest {
    /// Observations used (missing values dropped; those equal to the null median are not counted).
    pub n: usize,
    /// Values below the null median.
    pub below: usize,
    /// Values equal to the null median (excluded from the test).
    pub equal: usize,
    /// Values above the null median.
    pub above: usize,
    /// Sample median.
    pub median: f64,
    /// Null median.
    pub null_median: f64,
    /// Exact binomial p-value (two-sided: twice the smaller tail, at most 1).
    pub p_value: f64,
    /// Order-statistic interval for the median and its achieved confidence (`None` for n < 2).
    pub ci: Option<(f64, f64)>,
    /// Achieved confidence of `ci` (the largest level not above the target).
    pub achieved_level: Option<f64>,
    /// Lower and upper positions (1-based order statistics) of the interval.
    pub ci_ranks: Option<(usize, usize)>,
}

/// Sign test of `H₀: median = m0`. For paired data pass the differences.
pub fn sign_test(x: &[f64], m0: f64, alt: Alternative, level: f64) -> Result<SignTest> {
    check_level(level)?;
    let v = crate::clean_sorted(x)?;
    if v.is_empty() {
        return Err(StatsError::InsufficientData { what: "the sign test", needed: 1, got: 0 });
    }
    let below = v.iter().filter(|&&a| a < m0).count();
    let above = v.iter().filter(|&&a| a > m0).count();
    let equal = v.len() - below - above;
    let m = below + above;
    if m == 0 {
        return invalid("every observation equals the null median; the sign test has no information");
    }
    let b = Binomial::new(m as u64, 0.5)?;
    // P(X ≤ above) and P(X ≥ above) for X ~ Bin(m, ½).
    let lower = b.cdf(above as f64);
    let upper = if above == 0 { 1.0 } else { b.sf(above as f64 - 1.0) };
    let p = p_from_tails(lower, upper, alt);
    // Interval (x_(r), x_(n+1−r)) has confidence 1 − 2·P(B ≤ r − 1), B ~ Bin(n, ½).
    let n = v.len();
    let bn = Binomial::new(n as u64, 0.5)?;
    let mut best: Option<(usize, f64)> = None;
    for r in 1..=n.div_ceil(2) {
        let conf = 1.0 - 2.0 * bn.cdf(r as f64 - 1.0);
        if conf >= level - 1e-12 {
            best = Some((r, conf));
        } else {
            break;
        }
    }
    let (ci, achieved, ranks) = match best {
        Some((r, c)) if n >= 2 => (Some((v[r - 1], v[n - r])), Some(c), Some((r, n + 1 - r))),
        _ => (None, None, None),
    };
    Ok(SignTest {
        n: v.len(),
        below,
        equal,
        above,
        median: median_sorted(&v),
        null_median: m0,
        p_value: p,
        ci,
        achieved_level: achieved,
        ci_ranks: ranks,
    })
}

// ------------------------------------------------------------------ Wilcoxon signed rank

/// Result of the Wilcoxon signed-rank test.
#[derive(Debug, Clone, PartialEq)]
pub struct WilcoxonSignedRank {
    /// Non-missing observations.
    pub n: usize,
    /// Observations different from the null median (those used).
    pub n_used: usize,
    /// Sum of the ranks of the observations above the null median (Minitab's Wilcoxon statistic).
    pub w_plus: f64,
    /// Sum of the ranks of the observations below.
    pub w_minus: f64,
    /// Whether the p-value is exact.
    pub exact: bool,
    /// Standardised statistic (normal approximation; `None` when exact).
    pub z: Option<f64>,
    /// p-value.
    pub p_value: f64,
    /// Hodges–Lehmann estimate (median of the Walsh averages).
    pub estimate: f64,
    /// Confidence interval for the pseudo-median and its achieved level.
    pub ci: (f64, f64),
    /// Achieved confidence of `ci`.
    pub achieved_level: f64,
    /// Null median.
    pub null_median: f64,
    /// Whether ties among |d| were present.
    pub ties: bool,
}

/// Distribution of the sum of a random subset of `{1..n}` (each element in with probability ½).
fn signed_rank_pmf(n: usize) -> Vec<f64> {
    let top = n * (n + 1) / 2;
    let mut c = vec![0.0f64; top + 1];
    c[0] = 1.0;
    for k in 1..=n {
        for s in (k..=top).rev() {
            c[s] += c[s - k];
        }
    }
    let total = 2f64.powi(n as i32);
    c.iter().map(|x| x / total).collect()
}

fn cdf_of(pmf: &[f64], k: f64) -> f64 {
    if k < 0.0 {
        return 0.0;
    }
    let k = (k.floor() as usize).min(pmf.len() - 1);
    pmf[..=k].iter().sum::<f64>().min(1.0)
}

fn sf_of(pmf: &[f64], k: f64) -> f64 {
    // P(W ≥ k)
    if k <= 0.0 {
        return 1.0;
    }
    let k = k.ceil() as usize;
    if k >= pmf.len() {
        return 0.0;
    }
    pmf[k..].iter().sum::<f64>().min(1.0)
}

/// Wilcoxon signed-rank test of `H₀: median = m0` (for paired data pass the differences).
/// Zeros (observations equal to `m0`) are dropped before ranking. `correction` applies the
/// continuity correction in the normal approximation.
pub fn wilcoxon_signed_rank(
    x: &[f64],
    m0: f64,
    alt: Alternative,
    level: f64,
    method: SignRankMethod,
    correction: bool,
) -> Result<WilcoxonSignedRank> {
    check_level(level)?;
    let all = crate::clean(x)?;
    let n_all = all.len();
    let d: Vec<f64> = all.iter().map(|a| a - m0).filter(|a| *a != 0.0).collect();
    let n = d.len();
    if n < 1 {
        return Err(StatsError::InsufficientData { what: "the Wilcoxon signed-rank test", needed: 1, got: 0 });
    }
    if n_all > 3000 {
        return Err(StatsError::TooLarge(format!("the Hodges–Lehmann interval needs {n_all}(n+1)/2 Walsh averages")));
    }
    let abs: Vec<f64> = d.iter().map(|a| a.abs()).collect();
    let ranks = average_ranks(&abs)?;
    let w_plus: f64 = d.iter().zip(&ranks).filter(|(a, _)| **a > 0.0).map(|(_, r)| r).sum();
    let nf = n as f64;
    let total = nf * (nf + 1.0) / 2.0;
    let w_minus = total - w_plus;
    let tie = tie_term(&abs);
    let ties = tie > 0.0;
    let exact = match method {
        SignRankMethod::Exact => {
            if ties || n > MAX_EXACT_SIGNRANK {
                return invalid(format!(
                    "the exact signed-rank distribution needs no ties and at most {MAX_EXACT_SIGNRANK} nonzero differences"
                ));
            }
            true
        }
        SignRankMethod::Normal => false,
        SignRankMethod::Auto => !ties && n <= MAX_EXACT_SIGNRANK,
    };
    let pmf = (n <= MAX_EXACT_SIGNRANK).then(|| signed_rank_pmf(n));
    let mean = nf * (nf + 1.0) / 4.0;
    let var = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie / 48.0;
    let (p, z) = if exact {
        let pm = pmf.as_ref().expect("exact pmf");
        (p_from_tails(cdf_of(pm, w_plus), sf_of(pm, w_plus), alt), None)
    } else {
        if var <= 0.0 {
            return invalid("the signed-rank statistic has zero variance (all ranks tied)");
        }
        let sd = var.sqrt();
        let cc = if correction { 0.5 } else { 0.0 };
        let nrm = std_normal();
        match alt {
            Alternative::Greater => {
                let z = (w_plus - mean - cc) / sd;
                (nrm.sf(z), Some(z))
            }
            Alternative::Less => {
                let z = (w_plus - mean + cc) / sd;
                (nrm.cdf(z), Some(z))
            }
            Alternative::TwoSided => {
                let dev = (w_plus - mean).abs() - cc;
                let z = dev.max(0.0) / sd * (w_plus - mean).signum();
                ((2.0 * nrm.sf(z.abs())).min(1.0), Some(z))
            }
        }
    };
    // Hodges–Lehmann: Walsh averages of the observations themselves (shifted by nothing: the
    // estimate is for the pseudo-median of the sample).
    let mut walsh: Vec<f64> = Vec::with_capacity(n_all * (n_all + 1) / 2);
    for i in 0..n_all {
        for j in i..n_all {
            walsh.push(0.5 * (all[i] + all[j]));
        }
    }
    walsh.sort_by(|a, b| a.total_cmp(b));
    let m = walsh.len();
    let estimate = median_sorted(&walsh);
    // The interval uses the null distribution of the signed-rank statistic for n_all points.
    let na = n_all as f64;
    let alpha2 = (1.0 - level) / 2.0;
    let (c, achieved) = if n_all <= MAX_EXACT_SIGNRANK {
        let pm = signed_rank_pmf(n_all);
        let mut c: i64 = -1;
        let mut acc = 0.0;
        for (k, p) in pm.iter().enumerate() {
            acc += p;
            if acc <= alpha2 + 1e-12 {
                c = k as i64;
            } else {
                break;
            }
        }
        let tail = if c >= 0 { cdf_of(&pm, c as f64) } else { 0.0 };
        (c, 1.0 - 2.0 * tail)
    } else {
        let mu = na * (na + 1.0) / 4.0;
        let sd = (na * (na + 1.0) * (2.0 * na + 1.0) / 24.0).sqrt();
        let z = std_normal().quantile(1.0 - alpha2);
        let c = (mu - z * sd - 0.5).floor() as i64;
        (c, 1.0 - 2.0 * std_normal().cdf((c as f64 + 0.5 - mu) / sd))
    };
    let c = c.max(0) as usize;
    let ci = if c >= m / 2 { (walsh[0], walsh[m - 1]) } else { (walsh[c], walsh[m - 1 - c]) };
    Ok(WilcoxonSignedRank {
        n: n_all,
        n_used: n,
        w_plus,
        w_minus,
        exact,
        z,
        p_value: p,
        estimate,
        ci,
        achieved_level: achieved.clamp(0.0, 1.0),
        null_median: m0,
        ties,
    })
}

// ------------------------------------------------------------------ Mann–Whitney

/// Result of the Mann–Whitney test.
#[derive(Debug, Clone, PartialEq)]
pub struct MannWhitney {
    /// Sample sizes.
    pub n1: usize,
    pub n2: usize,
    /// Sample medians.
    pub median1: f64,
    pub median2: f64,
    /// Rank sum of the first sample in the pooled ranking (Minitab's W).
    pub w: f64,
    /// Mann–Whitney U of the first sample: `W − n₁(n₁+1)/2`.
    pub u: f64,
    /// Whether the p-value is exact.
    pub exact: bool,
    /// Standardised statistic (normal approximation; `None` when exact).
    pub z: Option<f64>,
    /// p-value for `H₁` about `η₁ − η₂` against 0.
    pub p_value: f64,
    /// Whether the pooled data contain ties.
    pub ties: bool,
    /// Hodges–Lehmann estimate of the difference `η₁ − η₂`.
    pub estimate: f64,
    /// Confidence interval for the difference.
    pub ci: (f64, f64),
    /// Achieved confidence of the interval.
    pub achieved_level: f64,
}

/// Null pmf of `U` for sample sizes `m` and `n` (no ties).
fn mw_pmf(m: usize, n: usize) -> Vec<f64> {
    let nn = m + n;
    let top = m * nn; // largest possible rank sum
    let mut f = vec![vec![0.0f64; top + 1]; m + 1];
    f[0][0] = 1.0;
    for e in 1..=nn {
        for k in (1..=m.min(e)).rev() {
            for s in (e..=top).rev() {
                let add = f[k - 1][s - e];
                f[k][s] += add;
            }
        }
    }
    let base = m * (m + 1) / 2;
    let mut pmf: Vec<f64> = (0..=m * n).map(|u| f[m][u + base]).collect();
    let tot: f64 = pmf.iter().sum();
    for p in &mut pmf {
        *p /= tot;
    }
    pmf
}

/// Mann–Whitney test of `H₀: η₁ = η₂`. Missing values are dropped from each sample.
pub fn mann_whitney(
    x: &[f64],
    y: &[f64],
    alt: Alternative,
    level: f64,
    method: SignRankMethod,
    correction: bool,
) -> Result<MannWhitney> {
    check_level(level)?;
    let xs = crate::clean_sorted(x)?;
    let ys = crate::clean_sorted(y)?;
    let (n1, n2) = (xs.len(), ys.len());
    if n1 < 1 || n2 < 1 {
        return Err(StatsError::InsufficientData { what: "each Mann–Whitney sample", needed: 1, got: n1.min(n2) });
    }
    if n1 * n2 > MAX_PAIRS {
        return Err(StatsError::TooLarge(format!("{n1} × {n2} pairwise differences")));
    }
    let pooled: Vec<f64> = xs.iter().chain(&ys).copied().collect();
    let ranks = average_ranks(&pooled)?;
    let w: f64 = ranks[..n1].iter().sum();
    let (n1f, n2f) = (n1 as f64, n2 as f64);
    let u = w - n1f * (n1f + 1.0) / 2.0;
    let tie = tie_term(&pooled);
    let ties = tie > 0.0;
    let exact = match method {
        SignRankMethod::Exact => {
            if ties || n1 > MAX_EXACT_MW || n2 > MAX_EXACT_MW {
                return invalid(format!(
                    "the exact Mann–Whitney distribution needs no ties and at most {MAX_EXACT_MW} per sample"
                ));
            }
            true
        }
        SignRankMethod::Normal => false,
        SignRankMethod::Auto => !ties && n1 <= MAX_EXACT_MW && n2 <= MAX_EXACT_MW,
    };
    let big_n = n1f + n2f;
    let mu = n1f * n2f / 2.0;
    let var = n1f * n2f / 12.0 * ((big_n + 1.0) - tie / (big_n * (big_n - 1.0)));
    let (p, z) = if exact {
        let pm = mw_pmf(n1, n2);
        (p_from_tails(cdf_of(&pm, u), sf_of(&pm, u), alt), None)
    } else {
        if var <= 0.0 {
            return invalid("all observations are tied; the Mann–Whitney statistic has zero variance");
        }
        let sd = var.sqrt();
        let cc = if correction { 0.5 } else { 0.0 };
        let nrm = std_normal();
        match alt {
            Alternative::Greater => {
                let z = (u - mu - cc) / sd;
                (nrm.sf(z), Some(z))
            }
            Alternative::Less => {
                let z = (u - mu + cc) / sd;
                (nrm.cdf(z), Some(z))
            }
            Alternative::TwoSided => {
                let z = ((u - mu).abs() - cc).max(0.0) / sd * (u - mu).signum();
                ((2.0 * nrm.sf(z.abs())).min(1.0), Some(z))
            }
        }
    };
    // Hodges–Lehmann shift: median of all pairwise differences x_i − y_j.
    let mut diffs: Vec<f64> = Vec::with_capacity(n1 * n2);
    for a in &xs {
        for b in &ys {
            diffs.push(a - b);
        }
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    let m = diffs.len();
    let estimate = median_sorted(&diffs);
    let alpha2 = (1.0 - level) / 2.0;
    let (c, achieved) = if n1 <= MAX_EXACT_MW && n2 <= MAX_EXACT_MW {
        let pm = mw_pmf(n1, n2);
        let mut c: i64 = -1;
        let mut acc = 0.0;
        for (k, p) in pm.iter().enumerate() {
            acc += p;
            if acc <= alpha2 + 1e-12 {
                c = k as i64;
            } else {
                break;
            }
        }
        let tail = if c >= 0 { cdf_of(&pm, c as f64) } else { 0.0 };
        (c, 1.0 - 2.0 * tail)
    } else {
        let sd = (n1f * n2f * (big_n + 1.0) / 12.0).sqrt();
        let z = std_normal().quantile(1.0 - alpha2);
        let c = (mu - z * sd - 0.5).floor() as i64;
        (c, 1.0 - 2.0 * std_normal().cdf((c as f64 + 0.5 - mu) / sd))
    };
    let c = c.max(0) as usize;
    let ci = if c >= m / 2 { (diffs[0], diffs[m - 1]) } else { (diffs[c], diffs[m - 1 - c]) };
    Ok(MannWhitney {
        n1,
        n2,
        median1: median_sorted(&xs),
        median2: median_sorted(&ys),
        w,
        u,
        exact,
        z,
        p_value: p,
        ties,
        estimate,
        ci,
        achieved_level: achieved.clamp(0.0, 1.0),
    })
}
