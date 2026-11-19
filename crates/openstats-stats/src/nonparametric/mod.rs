//! Nonparametric (rank-based) procedures — Minitab's Stat > Nonparametrics.
//!
//! * [`sign_test`] — one-sample / paired sign test with an exact binomial p-value and an
//!   order-statistic confidence interval for the median.
//! * [`wilcoxon_signed_rank`] — one-sample / paired Wilcoxon test with the exact null
//!   distribution (no ties, n ≤ 50) or the tie-corrected normal approximation, and the
//!   Hodges–Lehmann estimate with its confidence interval.
//! * [`mann_whitney`] — Mann–Whitney (Wilcoxon rank-sum) test, exact for small tie-free samples,
//!   with the Hodges–Lehmann shift estimate and confidence interval.
//! * [`kruskal_wallis`], [`mood_median`], [`friedman`] — k-sample and blocked designs.
//! * [`runs_test`] — Wald–Wolfowitz runs test for randomness about a value.
//!
//! Missing values are `NaN` and are dropped as documented on each function.

mod extra;
mod kruskal;
mod rank_tests;
mod report_impl;
mod runs;

pub use extra::{
    ansari_bradley, ks_two_sample, mann_kendall, mood_scale, normal_scores, AnsariBradley, KsTwoSample, MannKendall,
    MoodScale, NormalScores, NormalScoresGroup,
};
pub use kruskal::{
    friedman, kruskal_wallis, mood_median, Friedman, GroupRankSummary, KruskalWallis, MoodGroup, MoodMedian,
};
pub use rank_tests::{
    mann_whitney, sign_test, wilcoxon_signed_rank, MannWhitney, SignRankMethod, SignTest, WilcoxonSignedRank,
};
pub use runs::{runs_test, RunsTest};

/// Sum of `t³ − t` over the groups of tied values in `x` (any order).
pub(crate) fn tie_term(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let mut s = 0.0;
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j] == v[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        s += t * t * t - t;
        i = j;
    }
    s
}
