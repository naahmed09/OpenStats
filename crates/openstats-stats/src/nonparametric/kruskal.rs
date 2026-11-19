//! Kruskal–Wallis, Mood's median and Friedman tests.

use super::tie_term;
use crate::correlation::average_ranks;
use crate::error::{invalid, Result, StatsError};
use crate::quantile::{quantile_sorted, QuantileMethod};
use openstats_math::dist::{ChiSquared, ContinuousDistribution};

/// One group's row in a Kruskal–Wallis table.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupRankSummary {
    /// Observations in the group.
    pub n: usize,
    /// Group median.
    pub median: f64,
    /// Mean rank in the pooled ranking.
    pub mean_rank: f64,
    /// Standardised mean rank `(R̄ − (N+1)/2) / √((N+1)(N−n)/(12n))`.
    pub z: f64,
}

/// Result of the Kruskal–Wallis test.
#[derive(Debug, Clone, PartialEq)]
pub struct KruskalWallis {
    /// Per-group summaries, in input order.
    pub groups: Vec<GroupRankSummary>,
    /// Total observations.
    pub n: usize,
    /// Mean rank of all observations, `(N+1)/2`.
    pub overall_mean_rank: f64,
    /// H without the tie correction.
    pub h: f64,
    /// H adjusted for ties (equals `h` without ties).
    pub h_adjusted: f64,
    /// Degrees of freedom `k − 1`.
    pub df: usize,
    /// p-value of the unadjusted H (χ² approximation).
    pub p_value: f64,
    /// p-value of the tie-adjusted H.
    pub p_value_adjusted: f64,
    /// Whether any values are tied.
    pub ties: bool,
}

fn groups_clean(groups: &[Vec<f64>], what: &'static str) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for g in groups {
        let v = crate::clean_sorted(g)?;
        if !v.is_empty() {
            out.push(v);
        }
    }
    if out.len() < 2 {
        return Err(StatsError::InsufficientData { what, needed: 2, got: out.len() });
    }
    Ok(out)
}

fn median_of(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Kruskal–Wallis test. Missing values are dropped; groups left empty are ignored.
pub fn kruskal_wallis(groups: &[Vec<f64>]) -> Result<KruskalWallis> {
    let g = groups_clean(groups, "the Kruskal–Wallis test (groups with data)")?;
    let pooled: Vec<f64> = g.iter().flatten().copied().collect();
    let n = pooled.len();
    let nf = n as f64;
    let ranks = average_ranks(&pooled)?;
    let mut start = 0;
    let mut summaries = Vec::new();
    let mut sum_sq = 0.0;
    for v in &g {
        let ni = v.len();
        let rsum: f64 = ranks[start..start + ni].iter().sum();
        start += ni;
        let nif = ni as f64;
        let mean_rank = rsum / nif;
        sum_sq += rsum * rsum / nif;
        let se = ((nf + 1.0) * (nf - nif) / (12.0 * nif)).sqrt();
        let z = if se > 0.0 { (mean_rank - (nf + 1.0) / 2.0) / se } else { f64::NAN };
        summaries.push(GroupRankSummary { n: ni, median: median_of(v), mean_rank, z });
    }
    let h = 12.0 / (nf * (nf + 1.0)) * sum_sq - 3.0 * (nf + 1.0);
    let tie = tie_term(&pooled);
    let denom = 1.0 - tie / (nf * nf * nf - nf);
    if denom <= 0.0 {
        return invalid("all observations are tied; the Kruskal–Wallis statistic is undefined");
    }
    let h_adj = h / denom;
    let df = g.len() - 1;
    let chi = ChiSquared::new(df as f64)?;
    Ok(KruskalWallis {
        groups: summaries,
        n,
        overall_mean_rank: (nf + 1.0) / 2.0,
        h,
        h_adjusted: h_adj,
        df,
        p_value: chi.sf(h),
        p_value_adjusted: chi.sf(h_adj),
        ties: tie > 0.0,
    })
}

/// One group's row in Mood's median test.
#[derive(Debug, Clone, PartialEq)]
pub struct MoodGroup {
    /// Observations.
    pub n: usize,
    /// Observations ≤ the overall median.
    pub at_or_below: usize,
    /// Observations > the overall median.
    pub above: usize,
    /// Group median.
    pub median: f64,
    /// Q3 − Q1 (Minitab quartiles).
    pub iqr: f64,
}

/// Result of Mood's median test.
#[derive(Debug, Clone, PartialEq)]
pub struct MoodMedian {
    /// Groups in input order.
    pub groups: Vec<MoodGroup>,
    /// Overall median.
    pub overall_median: f64,
    /// Pearson χ² of the 2 × k table (no continuity correction).
    pub chi_square: f64,
    /// Degrees of freedom `k − 1`.
    pub df: usize,
    /// p-value.
    pub p_value: f64,
    /// Smallest expected count in the table (warn below 5).
    pub min_expected: f64,
}

/// Mood's median test: values equal to the overall median count as "at or below".
pub fn mood_median(groups: &[Vec<f64>]) -> Result<MoodMedian> {
    let g = groups_clean(groups, "Mood's median test (groups with data)")?;
    let pooled: Vec<f64> = {
        let mut p: Vec<f64> = g.iter().flatten().copied().collect();
        p.sort_by(|a, b| a.total_cmp(b));
        p
    };
    let n = pooled.len() as f64;
    let med = median_of(&pooled);
    let mut rows = Vec::new();
    let (mut tot_le, mut tot_gt) = (0usize, 0usize);
    for v in &g {
        let le = v.iter().filter(|&&a| a <= med).count();
        let gt = v.len() - le;
        tot_le += le;
        tot_gt += gt;
        let iqr = quantile_sorted(v, 0.75, QuantileMethod::Type6)? - quantile_sorted(v, 0.25, QuantileMethod::Type6)?;
        rows.push(MoodGroup { n: v.len(), at_or_below: le, above: gt, median: median_of(v), iqr });
    }
    if tot_gt == 0 || tot_le == 0 {
        return invalid("all observations fall on one side of the overall median; the test is undefined");
    }
    let mut chi = 0.0;
    let mut min_e = f64::INFINITY;
    for r in &rows {
        for (obs, tot) in [(r.at_or_below, tot_le), (r.above, tot_gt)] {
            let e = r.n as f64 * tot as f64 / n;
            min_e = min_e.min(e);
            chi += (obs as f64 - e).powi(2) / e;
        }
    }
    let df = rows.len() - 1;
    Ok(MoodMedian {
        groups: rows,
        overall_median: med,
        chi_square: chi,
        df,
        p_value: ChiSquared::new(df as f64)?.sf(chi),
        min_expected: min_e,
    })
}

/// Result of the Friedman test.
#[derive(Debug, Clone, PartialEq)]
pub struct Friedman {
    /// Blocks (complete rows) and treatments.
    pub blocks: usize,
    pub treatments: usize,
    /// Sum of ranks of each treatment.
    pub rank_sums: Vec<f64>,
    /// Estimated median of each treatment: treatment mean-adjusted for blocks, as in Minitab
    /// (observation − block median + grand median, then the median).
    pub estimated_medians: Vec<f64>,
    /// S statistic without the tie correction.
    pub statistic: f64,
    /// S adjusted for ties.
    pub statistic_adjusted: f64,
    /// Degrees of freedom `k − 1`.
    pub df: usize,
    /// p-value of the unadjusted statistic.
    pub p_value: f64,
    /// p-value of the tie-adjusted statistic.
    pub p_value_adjusted: f64,
}

/// Friedman test on a blocks × treatments matrix. Blocks with any missing value are dropped.
pub fn friedman(data: &[Vec<f64>]) -> Result<Friedman> {
    let k = data.first().map_or(0, Vec::len);
    if k < 2 {
        return Err(StatsError::InsufficientData { what: "the Friedman test (treatments)", needed: 2, got: k });
    }
    let mut rows: Vec<&Vec<f64>> = Vec::new();
    for r in data {
        if r.len() != k {
            return invalid("every block must have one value per treatment");
        }
        if r.iter().any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        if r.iter().all(|v| !v.is_nan()) {
            rows.push(r);
        }
    }
    let b = rows.len();
    if b < 2 {
        return Err(StatsError::InsufficientData { what: "the Friedman test (complete blocks)", needed: 2, got: b });
    }
    let (bf, kf) = (b as f64, k as f64);
    let mut rank_sums = vec![0.0; k];
    let mut tie_total = 0.0;
    for r in &rows {
        let rk = average_ranks(r)?;
        for (j, x) in rk.iter().enumerate() {
            rank_sums[j] += x;
        }
        tie_total += tie_term(r);
    }
    let stat = 12.0 / (bf * kf * (kf + 1.0)) * rank_sums.iter().map(|s| s * s).sum::<f64>() - 3.0 * bf * (kf + 1.0);
    let denom = 1.0 - tie_total / (bf * kf * (kf * kf - 1.0));
    if denom <= 0.0 {
        return invalid("every block is fully tied; the Friedman statistic is undefined");
    }
    let adj = stat / denom;
    // Estimated medians: adjust each observation by its block median, add the grand median.
    let mut all: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    all.sort_by(|a, b| a.total_cmp(b));
    let grand = median_of(&all);
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(b); k];
    for r in &rows {
        let mut s = (*r).clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let bm = median_of(&s);
        for (j, x) in r.iter().enumerate() {
            cols[j].push(x - bm + grand);
        }
    }
    let est: Vec<f64> = cols
        .iter_mut()
        .map(|c| {
            c.sort_by(|a, b| a.total_cmp(b));
            median_of(c)
        })
        .collect();
    let df = k - 1;
    let chi = ChiSquared::new(df as f64)?;
    Ok(Friedman {
        blocks: b,
        treatments: k,
        rank_sums,
        estimated_medians: est,
        statistic: stat,
        statistic_adjusted: adj,
        df,
        p_value: chi.sf(stat),
        p_value_adjusted: chi.sf(adj),
    })
}
