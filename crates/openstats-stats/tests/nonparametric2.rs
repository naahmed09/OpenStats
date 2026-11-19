//! KS two-sample, Ansari-Bradley, Mood scale, normal scores and Mann-Kendall against SciPy / brute force.

mod common;
use common::np2_data::*;
use openstats_stats::hypothesis::Alternative;
use openstats_stats::nonparametric::{ansari_bradley, ks_two_sample, mann_kendall, mood_scale, normal_scores};

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

#[test]
fn ks_exact_all_alternatives() {
    for (alt, st, p) in [
        (Alternative::TwoSided, KS_TWO_STAT, KS_TWO_P),
        (Alternative::Greater, KS_GT_STAT, KS_GT_P),
        (Alternative::Less, KS_LT_STAT, KS_LT_P),
    ] {
        let r = ks_two_sample(KS_X, KS_Y, alt).unwrap();
        assert!(r.exact);
        close(r.d, st, 1e-12, "D");
        close(r.p_value, p, 1e-6, &format!("p {alt:?}"));
    }
}

#[test]
fn ks_asymptotic_with_ties() {
    let r = ks_two_sample(KS_TX, KS_TY, Alternative::TwoSided).unwrap();
    assert!(!r.exact);
    close(r.d, KS_T_STAT, 1e-12, "D");
    close(r.p_value, KS_T_P, 1e-6, "p");
}

#[test]
fn ansari_exact_and_normal() {
    let r = ansari_bradley(AB_X, AB_Y).unwrap();
    assert!(r.exact);
    close(r.ab, AB_STAT, 1e-12, "AB");
    close(r.p_value, AB_P, 1e-6, "exact p");
    let r = ansari_bradley(AB2_X, AB2_Y).unwrap();
    assert!(r.exact);
    close(r.ab, AB2_STAT, 1e-12, "AB2");
    close(r.p_value, AB2_P, 1e-6, "exact p, larger N");
    let r = ansari_bradley(AB3_X, AB3_Y).unwrap();
    assert!(!r.exact);
    close(r.ab, AB3_STAT, 1e-12, "AB3");
    close(r.p_value, AB3_P, 1e-9, "normal p");
}

#[test]
fn mood_scale_matches_scipy() {
    let r = mood_scale(AB2_X, AB2_Y).unwrap();
    close(r.z, MOOD_Z, 1e-9, "z");
    close(r.p_value, MOOD_P, 1e-9, "p");
}

#[test]
fn normal_scores_match_brute_force() {
    let r = normal_scores(&[("a", NS_G1), ("b", NS_G2), ("c", NS_G3)]).unwrap();
    close(r.statistic, NS_STAT, 1e-9, "stat");
    close(r.p_value, NS_P, 1e-9, "p");
    assert_eq!(r.df, 2);
}

#[test]
fn mann_kendall_matches_brute_force() {
    let r = mann_kendall(MK_X, Alternative::TwoSided).unwrap();
    close(r.s, MK_S, 0.0, "S");
    close(r.variance, MK_VAR, 1e-12, "var");
    close(r.z, MK_Z, 1e-12, "z");
    close(r.p_value, MK_P, 1e-12, "p");
    close(r.sen_slope, MK_SEN, 1e-12, "sen");
}

#[test]
fn errors_and_reports() {
    use openstats_stats::report::{Reportable, SessionReport};
    assert!(ks_two_sample(&[], &[1.0], Alternative::TwoSided).is_err());
    assert!(ansari_bradley(&[1.0], &[1.0, 2.0]).is_err());
    assert!(mann_kendall(&[1.0, 2.0], Alternative::TwoSided).is_err());
    assert!(normal_scores(&[("a", &[1.0, 2.0])]).is_err());
    let mut rep = SessionReport::new("t");
    rep.push(ks_two_sample(KS_X, KS_Y, Alternative::TwoSided).unwrap().report_section());
    rep.push(ansari_bradley(AB_X, AB_Y).unwrap().report_section());
    rep.push(mood_scale(AB2_X, AB2_Y).unwrap().report_section());
    rep.push(mann_kendall(MK_X, Alternative::Greater).unwrap().report_section());
    rep.push(normal_scores(&[("a", NS_G1), ("b", NS_G2)]).unwrap().report_section());
    rep.validate().unwrap();
    assert!(rep.render_text().contains("Mann-Kendall"));
}
