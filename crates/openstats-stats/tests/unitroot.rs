//! ADF, KPSS, cross-correlation and automatic ARIMA against independent NumPy code.

mod common;
use common::ts2_data::*;
use openstats_stats::report::{Reportable, SessionReport};
use openstats_stats::timeseries::{
    adf_test, auto_arima, cross_correlation, kpss_test, AdfOptions, ArimaOptions, Deterministic,
};

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

#[test]
fn adf_statistics_match_numpy() {
    type AdfCase<'a> = (&'a [f64], [f64; 3], [f64; 3], [f64; 3]);
    let sets: [AdfCase; 3] = [
        (
            U_RW,
            [ADF_RW_N_FIX, ADF_RW_C_FIX, ADF_RW_T_FIX],
            [ADF_RW_N_AUTO, ADF_RW_C_AUTO, ADF_RW_T_AUTO],
            [ADF_RW_N_AUTOLAG, ADF_RW_C_AUTOLAG, ADF_RW_T_AUTOLAG],
        ),
        (
            U_AR,
            [ADF_AR_N_FIX, ADF_AR_C_FIX, ADF_AR_T_FIX],
            [ADF_AR_N_AUTO, ADF_AR_C_AUTO, ADF_AR_T_AUTO],
            [ADF_AR_N_AUTOLAG, ADF_AR_C_AUTOLAG, ADF_AR_T_AUTOLAG],
        ),
        (
            U_TR,
            [ADF_TR_N_FIX, ADF_TR_C_FIX, ADF_TR_T_FIX],
            [ADF_TR_N_AUTO, ADF_TR_C_AUTO, ADF_TR_T_AUTO],
            [ADF_TR_N_AUTOLAG, ADF_TR_C_AUTOLAG, ADF_TR_T_AUTOLAG],
        ),
    ];
    for (y, fix, auto, lag) in sets {
        for (k, det) in [Deterministic::None, Deterministic::Constant, Deterministic::Trend].into_iter().enumerate() {
            let f = adf_test(y, &AdfOptions { deterministic: det, lags: Some(2) }).unwrap();
            close(f.statistic, fix[k], 1e-8, "fixed lag");
            let a = adf_test(y, &AdfOptions { deterministic: det, lags: None }).unwrap();
            assert_eq!(a.lags as f64, lag[k], "chosen lag {det:?}");
            close(a.statistic, auto[k], 1e-8, "auto lag");
            assert!(a.critical[0] < a.critical[1] && a.critical[1] < a.critical[2]);
        }
    }
    // A stationary AR(1) rejects the unit root at 5 % with a constant; the random walk does not.
    let ar = adf_test(U_AR, &AdfOptions { deterministic: Deterministic::Constant, lags: Some(1) }).unwrap();
    assert!(ar.rejects(1));
    let rw = adf_test(U_RW, &AdfOptions::default()).unwrap();
    assert!(!rw.rejects(1));
}

#[test]
fn kpss_matches_numpy() {
    let sets = [
        (U_RW, [KPSS_RW_L_S, KPSS_RW_L_LONG, KPSS_RW_T_S, KPSS_RW_T_LONG]),
        (U_AR, [KPSS_AR_L_S, KPSS_AR_L_LONG, KPSS_AR_T_S, KPSS_AR_T_LONG]),
        (U_TR, [KPSS_TR_L_S, KPSS_TR_L_LONG, KPSS_TR_T_S, KPSS_TR_T_LONG]),
    ];
    for (y, want) in sets {
        close(kpss_test(y, false, Some(4)).unwrap().statistic, want[0], 1e-9, "level short");
        close(kpss_test(y, false, None).unwrap().statistic, want[1], 1e-9, "level long");
        close(kpss_test(y, true, Some(4)).unwrap().statistic, want[2], 1e-9, "trend short");
        close(kpss_test(y, true, None).unwrap().statistic, want[3], 1e-9, "trend long");
    }
    assert!(kpss_test(U_DR, false, None).unwrap().rejects(1));
    assert!(!kpss_test(U_AR, false, None).unwrap().rejects(1));
}

#[test]
fn cross_correlation_matches_numpy_and_finds_the_lead() {
    let c = cross_correlation(CCF_X, CCF_Y, 6, 0.95).unwrap();
    assert_eq!(c.lags.len(), 13);
    for (a, b) in c.ccf.iter().zip(CCF_REF) {
        close(*a, *b, 1e-12, "ccf");
    }
    let (imax, _) = c.ccf.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    assert_eq!(c.lags[imax], 3);
    close(c.limit, 1.959963984540054 / 80f64.sqrt(), 1e-9, "limit");
}

#[test]
fn auto_arima_ranks_and_picks_the_differencing() {
    let rw = auto_arima(U_DR, 2, 2, None, true, &ArimaOptions::default()).unwrap();
    assert_eq!(rw.d, 1);
    assert!(rw.ranked.windows(2).all(|w| w[0].aicc <= w[1].aicc));
    let ar = auto_arima(U_AR, 2, 1, None, true, &ArimaOptions::default()).unwrap();
    assert_eq!(ar.d, 0);
    assert!(ar.ranked[0].spec.p >= 1 || ar.ranked[0].spec.q >= 1);
    assert!(auto_arima(U_AR, 6, 1, None, true, &ArimaOptions::default()).is_err());
    assert!(auto_arima(U_AR, 1, 1, Some(3), true, &ArimaOptions::default()).is_err());
}

#[test]
fn reports_validate_and_errors() {
    let mut rep = SessionReport::new("t");
    rep.push(adf_test(U_RW, &AdfOptions::default()).unwrap().report_section());
    rep.push(kpss_test(U_RW, true, None).unwrap().report_section());
    rep.push(cross_correlation(CCF_X, CCF_Y, 4, 0.95).unwrap().report_section());
    rep.push(auto_arima(U_AR, 1, 1, Some(0), true, &ArimaOptions::default()).unwrap().report_section());
    rep.validate().unwrap();
    let t = rep.render_text();
    assert!(t.contains("Augmented Dickey-Fuller") && t.contains("KPSS") && t.contains("Cross Correlation"));
    assert!(adf_test(&[1.0; 5], &AdfOptions::default()).is_err());
    assert!(adf_test(&[1.0; 30], &AdfOptions::default()).is_err()); // constant series is singular
    assert!(kpss_test(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], false, None).is_err());
    assert!(cross_correlation(&[1.0, 2.0, 3.0], &[1.0, 2.0], 1, 0.95).is_err());
}
