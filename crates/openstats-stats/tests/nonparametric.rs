//! Nonparametric tests against SciPy and brute-force enumeration (`tests/gen/np_ref.py`).

mod common;

use common::np_data as d;
use openstats_stats::hypothesis::Alternative::{Greater, Less, TwoSided};
use openstats_stats::nonparametric::*;
use openstats_stats::report::Reportable;

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{what}: {a} vs {b}");
}

#[test]
fn sign_test_matches_binomial_tails_and_interval() {
    let r = sign_test(d::SIGN_X, d::SIGN_M0, TwoSided, 0.95).unwrap();
    assert_eq!((r.above, r.below, r.equal), (d::SIGN_ABOVE, d::SIGN_BELOW, 0));
    close(r.p_value, d::SIGN_P_TWO, 1e-12, "two-sided");
    close(sign_test(d::SIGN_X, d::SIGN_M0, Greater, 0.95).unwrap().p_value, d::SIGN_P_GREATER, 1e-12, "greater");
    close(sign_test(d::SIGN_X, d::SIGN_M0, Less, 0.95).unwrap().p_value, d::SIGN_P_LESS, 1e-12, "less");
    let (lo, hi) = r.ci.unwrap();
    assert_eq!((lo, hi), (d::SIGN_CI_LO, d::SIGN_CI_HI));
    close(r.achieved_level.unwrap(), d::SIGN_CI_CONF, 1e-12, "achieved");
}

#[test]
fn sign_test_edge_cases() {
    // Values equal to the null median are excluded.
    let r = sign_test(&[1.0, 2.0, 3.0, 3.0, 4.0, f64::NAN, 9.0], 3.0, TwoSided, 0.9).unwrap();
    assert_eq!((r.n, r.below, r.equal, r.above), (6, 2, 2, 2));
    assert!(sign_test(&[2.0, 2.0], 2.0, TwoSided, 0.95).is_err());
    assert!(sign_test(&[], 0.0, TwoSided, 0.95).is_err());
    assert!(sign_test(&[1.0], 0.0, TwoSided, 1.5).is_err());
    // All on one side: two-sided p = 2 · (1/2)^n.
    let r = sign_test(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0, TwoSided, 0.95).unwrap();
    close(r.p_value, 2.0 / 32.0, 1e-15, "one-sided data");
}

#[test]
fn wilcoxon_exact_matches_scipy_and_enumeration() {
    let m = SignRankMethod::Exact;
    let r = wilcoxon_signed_rank(d::WSR_D, 0.0, TwoSided, 0.95, m, false).unwrap();
    assert!(r.exact && r.z.is_none() && !r.ties);
    close(r.w_plus, d::WSR_WPLUS, 1e-12, "W+");
    close(r.p_value, d::WSR_P_TWO, 1e-12, "two-sided");
    let g = wilcoxon_signed_rank(d::WSR_D, 0.0, Greater, 0.95, m, false).unwrap();
    close(g.p_value, d::WSR_P_GREATER, 1e-12, "greater");
    let l = wilcoxon_signed_rank(d::WSR_D, 0.0, Less, 0.95, m, false).unwrap();
    close(l.p_value, d::WSR_P_LESS, 1e-12, "less");
    close(r.estimate, d::WSR_HL, 1e-12, "Hodges-Lehmann");
    close(r.ci.0, d::WSR_CI_LO, 1e-12, "CI lower");
    close(r.ci.1, d::WSR_CI_HI, 1e-12, "CI upper");
    close(r.achieved_level, d::WSR_CI_ACH, 1e-12, "achieved level");
    // Auto picks the exact method here.
    let a = wilcoxon_signed_rank(d::WSR_D, 0.0, TwoSided, 0.95, SignRankMethod::Auto, false).unwrap();
    assert!(a.exact && a.p_value == r.p_value);
}

#[test]
fn wilcoxon_with_ties_and_zeros_uses_the_corrected_normal_approximation() {
    let n = SignRankMethod::Auto;
    let r = wilcoxon_signed_rank(d::WSR_T, 0.0, TwoSided, 0.95, n, false).unwrap();
    assert!(!r.exact && r.ties && r.n_used == 18 && r.n == 20);
    close(r.w_plus, d::WSR_T_WPLUS, 1e-12, "W+");
    close(r.p_value, d::WSR_T_P_NOCC, 1e-10, "no correction");
    let c = wilcoxon_signed_rank(d::WSR_T, 0.0, TwoSided, 0.95, n, true).unwrap();
    close(c.p_value, d::WSR_T_P_CC, 1e-10, "continuity");
    let g = wilcoxon_signed_rank(d::WSR_T, 0.0, Greater, 0.95, n, false).unwrap();
    close(g.p_value, d::WSR_T_PG_NOCC, 1e-10, "greater");
    let gc = wilcoxon_signed_rank(d::WSR_T, 0.0, Greater, 0.95, n, true).unwrap();
    close(gc.p_value, d::WSR_T_PG_CC, 1e-10, "greater, continuity");
    // Exact is refused with ties.
    assert!(wilcoxon_signed_rank(d::WSR_T, 0.0, TwoSided, 0.95, SignRankMethod::Exact, false).is_err());
    assert!(r.ci.0 <= r.estimate && r.estimate <= r.ci.1);
}

#[test]
fn mann_whitney_exact_matches_scipy_and_enumeration() {
    let m = SignRankMethod::Exact;
    let r = mann_whitney(d::MW_A, d::MW_B, TwoSided, 0.95, m, false).unwrap();
    assert!(r.exact);
    close(r.u, d::MW_U, 1e-12, "U");
    close(r.p_value, d::MW_P_TWO, 1e-12, "two-sided");
    close(mann_whitney(d::MW_A, d::MW_B, Greater, 0.95, m, false).unwrap().p_value, d::MW_P_GREATER, 1e-12, "greater");
    close(mann_whitney(d::MW_A, d::MW_B, Less, 0.95, m, false).unwrap().p_value, d::MW_P_LESS, 1e-12, "less");
    close(r.estimate, d::MW_HL, 1e-12, "HL shift");
    close(r.ci.0, d::MW_CI_LO, 1e-12, "CI lower");
    close(r.ci.1, d::MW_CI_HI, 1e-12, "CI upper");
    close(r.achieved_level, d::MW_CI_ACH, 1e-12, "achieved");
    // W = U + n1(n1+1)/2.
    close(r.w, r.u + 45.0, 1e-12, "W");
}

#[test]
fn mann_whitney_with_ties_matches_the_asymptotic_test() {
    let a = SignRankMethod::Auto;
    let r = mann_whitney(d::MW_TA, d::MW_TB, TwoSided, 0.95, a, false).unwrap();
    assert!(!r.exact && r.ties);
    close(r.u, d::MW_T_U, 1e-12, "U");
    close(r.p_value, d::MW_T_P_NOCC, 1e-10, "no correction");
    close(
        mann_whitney(d::MW_TA, d::MW_TB, TwoSided, 0.95, a, true).unwrap().p_value,
        d::MW_T_P_CC,
        1e-10,
        "correction",
    );
    close(mann_whitney(d::MW_TA, d::MW_TB, Less, 0.95, a, false).unwrap().p_value, d::MW_T_PL_NOCC, 1e-10, "less");
    close(
        mann_whitney(d::MW_TA, d::MW_TB, Less, 0.95, a, true).unwrap().p_value,
        d::MW_T_PL_CC,
        1e-10,
        "less, correction",
    );
    assert!(mann_whitney(d::MW_TA, d::MW_TB, TwoSided, 0.95, SignRankMethod::Exact, false).is_err());
}

#[test]
fn mann_whitney_symmetry_and_large_samples() {
    let a = mann_whitney(d::MW_A, d::MW_B, TwoSided, 0.9, SignRankMethod::Auto, false).unwrap();
    let b = mann_whitney(d::MW_B, d::MW_A, TwoSided, 0.9, SignRankMethod::Auto, false).unwrap();
    close(a.p_value, b.p_value, 1e-12, "swapping samples keeps the two-sided p");
    close(a.estimate, -b.estimate, 1e-12, "and negates the shift");
    // Samples above the exact limit use the normal approximation, which agrees closely.
    let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.01).collect();
    let y: Vec<f64> = (0..70).map(|i| (i as f64 * 0.53).cos() * 3.0 + 0.5 + i as f64 * 0.01).collect();
    let big = mann_whitney(&x, &y, TwoSided, 0.95, SignRankMethod::Auto, false).unwrap();
    assert!(!big.exact && big.z.is_some() && big.p_value > 0.0 && big.p_value <= 1.0);
    assert!(big.ci.0 < big.estimate && big.estimate < big.ci.1);
    assert!(mann_whitney(&[], &[1.0], TwoSided, 0.95, SignRankMethod::Auto, false).is_err());
    assert!(mann_whitney(&[1.0, 1.0], &[1.0, 1.0], TwoSided, 0.95, SignRankMethod::Auto, false).is_err());
}

#[test]
fn kruskal_wallis_matches_scipy() {
    let g = vec![d::KW_G1.to_vec(), d::KW_G2.to_vec(), d::KW_G3.to_vec(), d::KW_G4.to_vec()];
    let r = kruskal_wallis(&g).unwrap();
    assert!(r.ties && r.df == 3 && r.n == 25);
    close(r.h_adjusted, d::KW_H_ADJ, 1e-12, "H adjusted");
    close(r.p_value_adjusted, d::KW_P_ADJ, 1e-10, "p adjusted");
    close(r.h, d::KW_H, 1e-12, "H");
    close(r.p_value, d::KW_P, 1e-10, "p");
    for i in 0..4 {
        close(r.groups[i].mean_rank, d::KW_MEANRANK[i], 1e-12, "mean rank");
        close(r.groups[i].z, d::KW_Z[i], 1e-12, "z");
    }
    // Missing values and empty groups are ignored.
    let mut g2 = g.clone();
    g2[0].push(f64::NAN);
    g2.push(vec![f64::NAN]);
    assert_eq!(kruskal_wallis(&g2).unwrap(), r);
    assert!(kruskal_wallis(&g[..1]).is_err());
    assert!(kruskal_wallis(&[vec![1.0, 1.0], vec![1.0]]).is_err());
}

#[test]
fn moods_median_matches_scipy() {
    let g = vec![d::KW_G1.to_vec(), d::KW_G2.to_vec(), d::KW_G3.to_vec(), d::KW_G4.to_vec()];
    let r = mood_median(&g).unwrap();
    close(r.chi_square, d::MOOD_STAT, 1e-12, "chi-square");
    close(r.p_value, d::MOOD_P, 1e-10, "p");
    close(r.overall_median, d::MOOD_MED, 1e-12, "median");
    // SciPy's table is [above; at-or-below] row-major over the groups.
    for (j, grp) in r.groups.iter().enumerate() {
        assert_eq!(grp.above as f64, d::MOOD_TABLE[j], "above, group {j}");
        assert_eq!(grp.at_or_below as f64, d::MOOD_TABLE[4 + j], "at or below, group {j}");
    }
    assert!(mood_median(&[vec![1.0, 1.0, 1.0], vec![1.0, 1.0]]).is_err());
}

#[test]
fn friedman_matches_scipy() {
    let rows: Vec<Vec<f64>> = d::FR_DATA.chunks(4).map(<[f64]>::to_vec).collect();
    let r = friedman(&rows).unwrap();
    assert_eq!((r.blocks, r.treatments, r.df), (10, 4, 3));
    close(r.statistic_adjusted, d::FR_S_ADJ, 1e-12, "S adjusted");
    close(r.p_value_adjusted, d::FR_P_ADJ, 1e-10, "p adjusted");
    close(r.statistic, d::FR_S, 1e-12, "S");
    close(r.p_value, d::FR_P, 1e-10, "p");
    for j in 0..4 {
        close(r.rank_sums[j], d::FR_RANKSUM[j], 1e-12, "rank sum");
        close(r.estimated_medians[j], d::FR_EST[j], 1e-12, "estimated median");
    }
    // A block with a missing value is dropped.
    let mut with_gap = rows.clone();
    with_gap.push(vec![1.0, f64::NAN, 2.0, 3.0]);
    assert_eq!(friedman(&with_gap).unwrap(), r);
    assert!(friedman(&[vec![1.0], vec![2.0]]).is_err());
    assert!(friedman(&[vec![1.0, 2.0], vec![1.0]]).is_err());
}

#[test]
fn runs_test_matches_enumeration() {
    let r = runs_test(d::RUNS_X, d::RUNS_K).unwrap();
    assert_eq!((r.above, r.below, r.runs), (d::RUNS_N1, d::RUNS_N2, d::RUNS_R));
    close(r.z, d::RUNS_Z, 1e-12, "z");
    close(r.p_value, d::RUNS_P_NORM, 1e-12, "normal p");
    close(r.exact_p_value.unwrap(), d::RUNS_P_EXACT, 1e-10, "exact p");
    // Values equal to K are omitted and do not split runs.
    let r2 = runs_test(&[1.0, 5.0, 5.0, 1.0, 9.0, 9.0, 1.0, f64::NAN], 5.0).unwrap();
    assert_eq!((r2.n, r2.equal, r2.runs), (5, 2, 3));
    assert!(runs_test(&[1.0, 2.0, 3.0], 0.0).is_err());
}

#[test]
fn every_nonparametric_result_has_a_valid_report() {
    use openstats_stats::report::SessionReport;
    let mut rep = SessionReport::new("Nonparametrics");
    rep.push(sign_test(d::SIGN_X, d::SIGN_M0, TwoSided, 0.95).unwrap().report_section());
    rep.push(
        wilcoxon_signed_rank(d::WSR_D, 0.0, TwoSided, 0.95, SignRankMethod::Auto, false).unwrap().report_section(),
    );
    rep.push(mann_whitney(d::MW_TA, d::MW_TB, TwoSided, 0.95, SignRankMethod::Auto, false).unwrap().report_section());
    let g = vec![d::KW_G1.to_vec(), d::KW_G2.to_vec(), d::KW_G3.to_vec()];
    rep.push(kruskal_wallis(&g).unwrap().section_with_labels(&["A".into(), "B".into(), "C".into()]));
    rep.push(mood_median(&g).unwrap().report_section());
    let rows: Vec<Vec<f64>> = d::FR_DATA.chunks(4).map(<[f64]>::to_vec).collect();
    rep.push(friedman(&rows).unwrap().report_section());
    rep.push(runs_test(d::RUNS_X, d::RUNS_K).unwrap().report_section());
    rep.validate().unwrap();
    let text = rep.render_text();
    assert!(text.contains("Mann-Whitney Test") && text.contains("Hodges-Lehmann") && text.contains("Ave Rank"));
    let back = SessionReport::from_json(&rep.to_json()).unwrap();
    assert_eq!(back.render_text(), text);
}
