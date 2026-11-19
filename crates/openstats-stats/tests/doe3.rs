//! Analysis of variability (Phase 5o-b). References: `tests/gen/doe3_ref.py` (numpy lstsq, scipy).

mod common;
use common::doe3_data::*;
use openstats_stats::doe::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1e-3), "{a} vs {b}");
}

#[test]
fn variability_matches_numpy_regression_of_ln_s() {
    let x: Vec<Vec<f64>> = VAR_X.chunks(3).map(|c| c.to_vec()).collect();
    let names: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
    let a = analyze_variability(&x, VAR_Y, &names, false).unwrap();
    close(a.constant, VAR_MAIN_COEF[0], 1e-9);
    for j in 0..3 {
        close(a.terms[j].coef, VAR_MAIN_COEF[j + 1], 1e-8);
        close(a.terms[j].se, VAR_MAIN_SE[j + 1], 1e-8);
        close(a.terms[j].p, VAR_MAIN_P[j + 1], 1e-7);
    }
    close(a.r2, VAR_MAIN_R2, 1e-9);
    let b = analyze_variability(&x, VAR_Y, &names, true).unwrap();
    assert_eq!(b.df_error, 1);
    for j in 0..6 {
        close(b.terms[j].coef, VAR_INT_COEF[j + 1], 1e-7);
        close(b.terms[j].se, VAR_INT_SE[j + 1], 1e-7);
        close(b.terms[j].p, VAR_INT_P[j + 1], 1e-6);
    }
    assert_eq!(b.terms[3].name, "A*B");
}

#[test]
fn variability_needs_replicates_and_error_df() {
    let x: Vec<Vec<f64>> = VAR_X.chunks(3).map(|c| c.to_vec()).collect();
    let names: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
    // singletons only → refused
    let once: Vec<Vec<f64>> = x.iter().step_by(3).cloned().collect();
    let y1: Vec<f64> = VAR_Y.iter().step_by(3).copied().collect();
    assert!(analyze_variability(&once, &y1, &names, false).is_err());
}
