//! Two-sample Kolmogorov–Smirnov, Ansari–Bradley, Mood scale, van der Waerden normal scores and
//! the Mann–Kendall trend test.

use crate::correlation::average_ranks;
use crate::error::{invalid, Result, StatsError};
use crate::hypothesis::Alternative;
use openstats_math::dist::{ChiSquared, ContinuousDistribution, Normal};

const MAX_KS_CELLS: usize = 4_000_000;
const MAX_AB_EXACT: usize = 55;

fn std_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal")
}

fn clean(x: &[f64], what: &'static str, needed: usize) -> Result<Vec<f64>> {
    if x.iter().any(|v| v.is_infinite()) {
        return invalid("data contain infinite values");
    }
    let v: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    if v.len() < needed {
        return Err(StatsError::InsufficientData { what, needed, got: v.len() });
    }
    Ok(v)
}

fn has_ties(x: &[f64], y: &[f64]) -> bool {
    let mut v: Vec<f64> = x.iter().chain(y.iter()).copied().collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v.windows(2).any(|w| w[0] == w[1])
}

// ------------------------------------------------------------ Kolmogorov–Smirnov

/// Result of the two-sample Kolmogorov–Smirnov test.
#[derive(Debug, Clone, PartialEq)]
pub struct KsTwoSample {
    /// Sample sizes.
    pub n1: usize,
    /// Second sample size.
    pub n2: usize,
    /// `D⁺ = max(F₁ − F₂)`.
    pub d_plus: f64,
    /// `D⁻ = max(F₂ − F₁)`.
    pub d_minus: f64,
    /// The statistic used for the chosen alternative.
    pub d: f64,
    /// p-value.
    pub p_value: f64,
    /// Whether the p-value is exact.
    pub exact: bool,
}

/// Two-sample Kolmogorov–Smirnov test. `Greater` tests `F₁ > F₂` (sample 1 stochastically smaller).
pub fn ks_two_sample(x: &[f64], y: &[f64], alt: Alternative) -> Result<KsTwoSample> {
    let a = clean(x, "the two-sample KS test", 1)?;
    let b = clean(y, "the two-sample KS test", 1)?;
    let (n, m) = (a.len(), b.len());
    let mut pooled: Vec<(f64, bool)> = a.iter().map(|&v| (v, true)).chain(b.iter().map(|&v| (v, false))).collect();
    pooled.sort_by(|p, q| p.0.total_cmp(&q.0));
    let (mut i, mut j) = (0usize, 0usize);
    let (mut dp, mut dm) = (0.0f64, 0.0f64);
    let mut k = 0;
    while k < pooled.len() {
        let v = pooled[k].0;
        while k < pooled.len() && pooled[k].0 == v {
            if pooled[k].1 {
                i += 1;
            } else {
                j += 1;
            }
            k += 1;
        }
        let diff = i as f64 / n as f64 - j as f64 / m as f64;
        dp = dp.max(diff);
        dm = dm.max(-diff);
    }
    let d = match alt {
        Alternative::TwoSided => dp.max(dm),
        Alternative::Greater => dp,
        Alternative::Less => dm,
    };
    let exact = !has_ties(&a, &b) && n.saturating_mul(m) <= MAX_KS_CELLS;
    let p_value = if exact {
        exact_ks_p(n, m, d, alt == Alternative::TwoSided)
    } else {
        let en = (n as f64 * m as f64 / (n + m) as f64).sqrt();
        if alt == Alternative::TwoSided {
            crate::normality::ks_p_value((en * en).round().max(1.0) as usize, d)
        } else {
            (-2.0 * en * en * d * d).exp().min(1.0)
        }
    };
    Ok(KsTwoSample { n1: n, n2: m, d_plus: dp, d_minus: dm, d, p_value, exact })
}

/// `P(D ≥ d)` by counting lattice paths that stay strictly inside the band.
fn exact_ks_p(n: usize, m: usize, d: f64, two_sided: bool) -> f64 {
    let eps = 1e-10;
    let inside = |i: usize, j: usize| {
        let diff = i as f64 / n as f64 - j as f64 / m as f64;
        if two_sided {
            diff.abs() < d - eps
        } else {
            diff < d - eps
        }
    };
    // g(i, j) = (paths to (i, j) inside the band) / C(i + j, i)
    let mut prev = vec![0.0f64; m + 1];
    let mut cur = vec![0.0f64; m + 1];
    for i in 0..=n {
        for j in 0..=m {
            if !inside(i, j) {
                cur[j] = 0.0;
            } else if i == 0 && j == 0 {
                cur[j] = 1.0;
            } else {
                let s = (i + j) as f64;
                let mut v = 0.0;
                if i > 0 {
                    v += i as f64 / s * prev[j];
                }
                if j > 0 {
                    v += j as f64 / s * cur[j - 1];
                }
                cur[j] = v;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    (1.0 - prev[m]).clamp(0.0, 1.0)
}

// -------------------------------------------------------------- Ansari–Bradley

/// Result of the Ansari–Bradley scale test.
#[derive(Debug, Clone, PartialEq)]
pub struct AnsariBradley {
    /// Sample sizes.
    pub n1: usize,
    /// Second sample size.
    pub n2: usize,
    /// Sum of the Ansari–Bradley scores of sample 1.
    pub ab: f64,
    /// Null mean.
    pub mean: f64,
    /// Null variance (tie-adjusted).
    pub variance: f64,
    /// Standardized statistic (`None` for the exact test).
    pub z: Option<f64>,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Whether the p-value is exact.
    pub exact: bool,
}

/// Ansari–Bradley test of equal scale (two-sided). Exact for tie-free samples with both sizes below 55.
pub fn ansari_bradley(x: &[f64], y: &[f64]) -> Result<AnsariBradley> {
    let a = clean(x, "the Ansari-Bradley test", 2)?;
    let b = clean(y, "the Ansari-Bradley test", 2)?;
    let (m, n) = (a.len(), b.len());
    let big = m + n;
    let mut pooled: Vec<(f64, bool)> = a.iter().map(|&v| (v, true)).chain(b.iter().map(|&v| (v, false))).collect();
    pooled.sort_by(|p, q| p.0.total_cmp(&q.0));
    let base: Vec<f64> = (1..=big).map(|i| i.min(big + 1 - i) as f64).collect();
    let mut scores = base.clone();
    let mut i = 0;
    while i < big {
        let mut j = i;
        while j < big && pooled[j].0 == pooled[i].0 {
            j += 1;
        }
        let avg = base[i..j].iter().sum::<f64>() / (j - i) as f64;
        scores[i..j].iter_mut().for_each(|s| *s = avg);
        i = j;
    }
    let ab: f64 = pooled.iter().zip(&scores).filter(|(p, _)| p.1).map(|(_, s)| s).sum();
    let nn = big as f64;
    let abar = scores.iter().sum::<f64>() / nn;
    let mean = m as f64 * abar;
    let ss: f64 = scores.iter().map(|s| (s - abar) * (s - abar)).sum();
    let variance = m as f64 * n as f64 / (nn * (nn - 1.0)) * ss;
    let exact = !has_ties(&a, &b) && m < MAX_AB_EXACT && n < MAX_AB_EXACT;
    if exact {
        let ints: Vec<usize> = base.iter().map(|&v| v as usize).collect();
        let total: usize = ints.iter().sum();
        // ways[k][s]: subsets of size k with score sum s
        let mut ways = vec![vec![0.0f64; total + 1]; m + 1];
        ways[0][0] = 1.0;
        for &sc in &ints {
            for k in (1..=m).rev() {
                for s in (sc..=total).rev() {
                    let add = ways[k - 1][s - sc];
                    ways[k][s] += add;
                }
            }
        }
        let all: f64 = ways[m].iter().sum();
        let obs = ab.round() as usize;
        let lower: f64 = ways[m][..=obs].iter().sum::<f64>() / all;
        let upper: f64 = ways[m][obs..].iter().sum::<f64>() / all;
        let p = (2.0 * lower.min(upper)).min(1.0);
        return Ok(AnsariBradley { n1: m, n2: n, ab, mean, variance, z: None, p_value: p, exact });
    }
    let z = (ab - mean) / variance.sqrt();
    let p = (2.0 * std_normal().sf(z.abs())).min(1.0);
    Ok(AnsariBradley { n1: m, n2: n, ab, mean, variance, z: Some(z), p_value: p, exact })
}

// ------------------------------------------------------------------ Mood scale

/// Result of Mood's two-sample scale test.
#[derive(Debug, Clone, PartialEq)]
pub struct MoodScale {
    /// Sample sizes.
    pub n1: usize,
    /// Second sample size.
    pub n2: usize,
    /// Sum over sample 1 of the squared deviation of its rank from the centre.
    pub t: f64,
    /// Null mean.
    pub mean: f64,
    /// Null variance (tie-free formula).
    pub variance: f64,
    /// Standardized statistic.
    pub z: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

/// Mood's test of equal scale (asymptotic, two-sided). Ties use average ranks with the tie-free variance.
pub fn mood_scale(x: &[f64], y: &[f64]) -> Result<MoodScale> {
    let a = clean(x, "Mood's scale test", 1)?;
    let b = clean(y, "Mood's scale test", 1)?;
    let (m, n) = (a.len() as f64, b.len() as f64);
    let big = m + n;
    if big < 3.0 {
        return Err(StatsError::InsufficientData { what: "Mood's scale test", needed: 3, got: big as usize });
    }
    let all: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let r = average_ranks(&all)?;
    let c = (big + 1.0) / 2.0;
    let t: f64 = r[..a.len()].iter().map(|v| (v - c) * (v - c)).sum();
    let mean = m * (big * big - 1.0) / 12.0;
    let variance = m * n * (big + 1.0) * (big + 2.0) * (big - 2.0) / 180.0;
    let z = (t - mean) / variance.sqrt();
    Ok(MoodScale { n1: a.len(), n2: b.len(), t, mean, variance, z, p_value: (2.0 * std_normal().sf(z.abs())).min(1.0) })
}

// ---------------------------------------------------------------- normal scores

/// One group of the van der Waerden test.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalScoresGroup {
    /// Group label.
    pub label: String,
    /// Group size.
    pub n: usize,
    /// Average normal score.
    pub mean_score: f64,
}

/// Result of the van der Waerden normal-scores test.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalScores {
    /// Groups.
    pub groups: Vec<NormalScoresGroup>,
    /// Statistic.
    pub statistic: f64,
    /// Degrees of freedom.
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

/// van der Waerden normal-scores test for `k ≥ 2` groups.
pub fn normal_scores(groups: &[(&str, &[f64])]) -> Result<NormalScores> {
    if groups.len() < 2 {
        return invalid("the normal-scores test needs at least two groups");
    }
    let mut vals = Vec::new();
    let mut idx = Vec::new();
    for (g, (_, d)) in groups.iter().enumerate() {
        for &v in d.iter() {
            if v.is_infinite() {
                return invalid("data contain infinite values");
            }
            if !v.is_nan() {
                vals.push(v);
                idx.push(g);
            }
        }
    }
    let big = vals.len();
    if big < 3 {
        return Err(StatsError::InsufficientData { what: "the normal-scores test", needed: 3, got: big });
    }
    let r = average_ranks(&vals)?;
    let nd = std_normal();
    let a: Vec<f64> = r.iter().map(|&rk| nd.quantile(rk / (big as f64 + 1.0))).collect();
    let s2 = a.iter().map(|v| v * v).sum::<f64>() / (big as f64 - 1.0);
    let mut out = Vec::new();
    let mut stat = 0.0;
    for (g, (label, _)) in groups.iter().enumerate() {
        let sel: Vec<f64> = a.iter().zip(&idx).filter(|(_, &i)| i == g).map(|(v, _)| *v).collect();
        if sel.is_empty() {
            return invalid(format!("group '{label}' has no observations"));
        }
        let mean = sel.iter().sum::<f64>() / sel.len() as f64;
        stat += sel.len() as f64 * mean * mean / s2;
        out.push(NormalScoresGroup { label: (*label).to_string(), n: sel.len(), mean_score: mean });
    }
    let df = groups.len() - 1;
    Ok(NormalScores { groups: out, statistic: stat, df, p_value: ChiSquared::new(df as f64)?.sf(stat) })
}

// ------------------------------------------------------------------ Mann–Kendall

/// Result of the Mann–Kendall trend test.
#[derive(Debug, Clone, PartialEq)]
pub struct MannKendall {
    /// Observations.
    pub n: usize,
    /// Kendall's S.
    pub s: f64,
    /// Variance of S (tie-corrected).
    pub variance: f64,
    /// Standardized statistic (continuity-corrected).
    pub z: f64,
    /// p-value for the chosen alternative.
    pub p_value: f64,
    /// Kendall's tau between the series and time.
    pub tau: f64,
    /// Sen's slope per observation.
    pub sen_slope: f64,
}

/// Mann–Kendall test for a monotone trend in a series ordered in time. `Greater` tests an upward trend.
pub fn mann_kendall(x: &[f64], alt: Alternative) -> Result<MannKendall> {
    let v = clean(x, "the Mann-Kendall test", 3)?;
    let n = v.len();
    let mut s = 0.0;
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let d = v[j] - v[i];
            s += if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            slopes.push(d / (j - i) as f64);
        }
    }
    let nf = n as f64;
    let mut sorted = v.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut tie_sum = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        tie_sum += t * (t - 1.0) * (2.0 * t + 5.0);
        i = j;
    }
    let variance = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - tie_sum) / 18.0;
    let z = if s > 0.0 {
        (s - 1.0) / variance.sqrt()
    } else if s < 0.0 {
        (s + 1.0) / variance.sqrt()
    } else {
        0.0
    };
    let nd = std_normal();
    let p_value = match alt {
        Alternative::TwoSided => (2.0 * nd.sf(z.abs())).min(1.0),
        Alternative::Greater => nd.sf(z),
        Alternative::Less => nd.cdf(z),
    };
    slopes.sort_by(|a, b| a.total_cmp(b));
    let k = slopes.len();
    let sen = if k % 2 == 1 { slopes[k / 2] } else { 0.5 * (slopes[k / 2 - 1] + slopes[k / 2]) };
    let tau_den = (nf * (nf - 1.0) / 2.0).max(1.0);
    Ok(MannKendall { n, s, variance, z, p_value, tau: s / tau_den, sen_slope: sen })
}
