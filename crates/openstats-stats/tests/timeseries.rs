//! Time series against independent NumPy/SciPy references (`common/ts_data.rs`, from
//! `gen/ts_ref.py`). ARIMA maximum likelihood is checked against the exact likelihood computed from
//! the full autocovariance matrix, which shares no algorithm with the Kalman filter.

mod common;
use common::ts_data::*;
use openstats_stats::report::Reportable;
use openstats_stats::timeseries::*;

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b} (tol {tol})");
}

fn all_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        close(*x, *y, tol, &format!("{what}[{i}]"));
    }
}

#[test]
fn acf_pacf_and_ljung_box_match_direct_computation() {
    let a = acf(ACF_X, Some(20), 0.95).unwrap();
    all_close(&a.acf, ACF_R, 1e-12, "acf");
    all_close(&a.se, ACF_SE, 1e-12, "bartlett se");
    all_close(&a.lbq, ACF_Q, 1e-10, "lbq");
    assert!((a.upper[0] - 1.959964 * a.se[0]).abs() < 1e-5);
    let p = pacf(ACF_X, Some(20), 0.95).unwrap();
    all_close(&p.pacf, PACF_R, 1e-10, "pacf");
    assert!((p.se - 1.0 / 80f64.sqrt()).abs() < 1e-15);
    let (q, pv) = ljung_box(&a.acf[..12], 80, 2);
    close(q, ACF_Q[11], 1e-10, "Q(12)");
    close(pv.unwrap(), LB_P_12_DF10, 1e-9, "Ljung-Box p");
    assert_eq!(ljung_box(&a.acf[..2], 80, 2).1, None);
}

#[test]
fn acf_input_errors_and_differencing() {
    assert!(acf(&[1.0, 2.0], None, 0.95).is_err());
    assert!(acf(&[3.0; 10], None, 0.95).is_err());
    assert!(acf(&[1.0, f64::NAN, 2.0, 3.0], None, 0.95).is_err());
    assert!(acf(ACF_X, Some(80), 0.95).is_err());
    assert!(acf(ACF_X, Some(5), 1.5).is_err());
    assert_eq!(difference(&[1.0, 4.0, 9.0, 16.0], 1, 1).unwrap(), vec![3.0, 5.0, 7.0]);
    assert_eq!(difference(&[1.0, 4.0, 9.0, 16.0], 1, 2).unwrap(), vec![2.0, 2.0]);
    assert_eq!(difference(&[1.0, 2.0, 5.0, 3.0, 4.0], 2, 1).unwrap(), vec![4.0, 1.0, -1.0]);
    assert!(difference(&[1.0, 2.0], 5, 1).is_err());
    assert!(difference(&[1.0, 2.0], 0, 1).is_err());
}

fn ml(y: &[f64], spec: ArimaSpec) -> ArimaFit {
    fit_arima(y, &spec, &ArimaOptions { method: ArimaMethod::Ml, ..Default::default() }).unwrap()
}

fn flat(f: &ArimaFit) -> Vec<f64> {
    // Reference order: AR, MA, SAR, SMA, mean.
    let mut v = Vec::new();
    v.extend(&f.phi);
    v.extend(&f.theta);
    v.extend(&f.sphi);
    v.extend(&f.stheta);
    if f.spec.constant {
        v.push(f.mean);
    }
    v
}

fn check_fit(f: &ArimaFit, params: &[f64], s2: f64, ll: f64, aic: f64, tol: f64, name: &str) {
    let got = flat(f);
    all_close(&got, params, tol, &format!("{name} params"));
    close(f.sigma2, s2, 5.0 * tol, &format!("{name} sigma2"));
    assert!((f.loglik - ll).abs() < 1e-3, "{name} loglik {} vs {}", f.loglik, ll);
    assert!((f.aic - aic).abs() < 2e-3, "{name} aic");
}

#[test]
fn exact_ml_matches_the_autocovariance_likelihood() {
    let f = ml(AR2_Y, ArimaSpec::new(2, 0, 0));
    check_fit(&f, AR2_PARAMS, AR2_S2, AR2_LL, AR2_AIC, 2e-3, "AR2");
    let f = ml(ARMA11_Y, ArimaSpec::new(1, 0, 1).with_constant(false));
    check_fit(&f, ARMA11_PARAMS, ARMA11_S2, ARMA11_LL, ARMA11_AIC, 3e-3, "ARMA11");
    let f = ml(MA1_Y, ArimaSpec::new(0, 0, 1));
    check_fit(&f, MA1_PARAMS, MA1_S2, MA1_LL, MA1_AIC, 2e-3, "MA1");
    let f = ml(ARIMA110_Y, ArimaSpec::new(1, 1, 0));
    check_fit(&f, ARIMA110_PARAMS, ARIMA110_S2, ARIMA110_LL, ARIMA110_AIC, 2e-3, "ARIMA110");
    assert!(f.converged);
}

#[test]
fn seasonal_airline_model_matches_the_exact_likelihood() {
    let spec = ArimaSpec::new(0, 1, 1).seasonal(0, 1, 1, 12).with_constant(false);
    let f = ml(AIR_Y, spec);
    check_fit(&f, AIR_PARAMS, AIR_S2, AIR_LL, AIR_AIC, 3e-3, "AIR");
    assert_eq!(f.spec.name(), "ARIMA(0,1,1)(0,1,1)12");
    // Full MA operator (1 - θB)(1 - ΘB¹²): lags 1, 12, 13.
    let m = f.ma_operator();
    assert_eq!(m.len(), 13);
    close(m[0], f.theta[0], 1e-12, "ma1");
    close(m[11], f.stheta[0], 1e-12, "ma12");
    close(m[12], -f.theta[0] * f.stheta[0], 1e-12, "ma13");
}

#[test]
fn forecasts_and_prediction_errors_match_exact_prediction() {
    // Exact best linear predictors from the full covariance matrix; the library truncates the
    // pre-sample innovations, which is accurate to well under 1e-3 here.
    for (name, y, spec, fc, se, tol) in [
        ("AR2", AR2_Y, ArimaSpec::new(2, 0, 0), AR2_FC, AR2_FCSE, 2e-3),
        ("ARMA11", ARMA11_Y, ArimaSpec::new(1, 0, 1).with_constant(false), ARMA11_FC, ARMA11_FCSE, 4e-3),
        ("MA1", MA1_Y, ArimaSpec::new(0, 0, 1), MA1_FC, MA1_FCSE, 2e-3),
        ("ARIMA110", ARIMA110_Y, ArimaSpec::new(1, 1, 0), ARIMA110_FC, ARIMA110_FCSE, 4e-3),
        ("AIR", AIR_Y, ArimaSpec::new(0, 1, 1).seasonal(0, 1, 1, 12).with_constant(false), AIR_FC, AIR_FCSE, 4e-3),
    ] {
        let f = ml(y, spec);
        let h = fc.len();
        let fcst = f.forecast(h, 0.95).unwrap();
        all_close(&fcst.point, fc, tol, &format!("{name} forecast"));
        all_close(&fcst.se, se, 5.0 * tol, &format!("{name} se"));
        assert_eq!(fcst.origin, y.len() - 1);
        for i in 0..h {
            assert!(fcst.lower[i] < fcst.point[i] && fcst.point[i] < fcst.upper[i]);
            close(fcst.upper[i] - fcst.point[i], 1.959964 * fcst.se[i], 1e-5, "interval half-width");
        }
        assert!(fcst.se.windows(2).all(|w| w[1] >= w[0] - 1e-12), "{name}: errors do not shrink");
    }
}

#[test]
fn css_for_a_pure_ar_model_is_least_squares() {
    // Conditional least squares of AR(2) with a mean is the regression of wₜ on its two lags:
    // intercept = μ(1 − φ₁ − φ₂).
    let f =
        fit_arima(AR2_Y, &ArimaSpec::new(2, 0, 0), &ArimaOptions { method: ArimaMethod::Css, ..Default::default() })
            .unwrap();
    close(f.phi[0], CSS_AR2_OLS[1], 1e-5, "phi1");
    close(f.phi[1], CSS_AR2_OLS[2], 1e-5, "phi2");
    close(f.constant, CSS_AR2_OLS[0], 1e-5, "intercept");
    // One-step residuals are the regression residuals.
    let n = AR2_Y.len();
    for t in 2..n {
        let e = AR2_Y[t] - CSS_AR2_OLS[0] - CSS_AR2_OLS[1] * AR2_Y[t - 1] - CSS_AR2_OLS[2] * AR2_Y[t - 2];
        close(f.residuals[t], e, 1e-4, "css residual");
    }
    assert!(f.residuals[0].is_nan() && f.residuals[1].is_nan());
}

#[test]
fn css_and_ml_agree_roughly_and_the_default_runs_both() {
    let css =
        fit_arima(MA1_Y, &ArimaSpec::new(0, 0, 1), &ArimaOptions { method: ArimaMethod::Css, ..Default::default() })
            .unwrap();
    let both = fit_arima(MA1_Y, &ArimaSpec::new(0, 0, 1), &ArimaOptions::default()).unwrap();
    assert_eq!(both.method, ArimaMethod::CssMl);
    check_fit(&both, MA1_PARAMS, MA1_S2, MA1_LL, MA1_AIC, 2e-3, "CSS-ML MA1");
    assert!((css.theta[0] - both.theta[0]).abs() < 0.05);
}

#[test]
fn coefficient_table_and_diagnostics() {
    let f = ml(AR2_Y, ArimaSpec::new(2, 0, 0));
    let labels: Vec<&str> = f.coefs.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(labels, ["AR 1", "AR 2", "Constant"]);
    // The strongly significant first AR term; the standard error matches 1/sqrt(n) roughly.
    let ar1 = &f.coefs[0];
    assert!(ar1.p.unwrap() < 0.001);
    let se = ar1.se.unwrap();
    assert!(se > 0.03 && se < 0.15, "se {se}");
    // Constant = mean · (1 − Σφ).
    close(f.constant, f.mean * (1.0 - f.phi.iter().sum::<f64>()), 1e-12, "constant");
    assert_eq!(f.ljung_box.last().unwrap().lag, 12);
    assert_eq!(f.ljung_box.last().unwrap().df, 10);
    assert!(f.ljung_box.last().unwrap().p.unwrap() > 0.01);
    assert!(f.aicc > f.aic && f.bic > f.aic - 10.0);
    // Fits and residuals partition the data.
    for t in 0..AR2_Y.len() {
        close(f.fitted[t] + f.residuals[t], AR2_Y[t], 1e-12, "fit + resid");
    }
    assert_eq!(f.accuracy.n, AR2_Y.len());
}

#[test]
fn arima_rejects_bad_input() {
    let o = ArimaOptions::default();
    assert!(fit_arima(&[1.0, 2.0, 3.0], &ArimaSpec::new(1, 0, 0), &o).is_err());
    assert!(fit_arima(AR2_Y, &ArimaSpec::new(1, 0, 0).seasonal(1, 0, 0, 1), &o).is_err());
    assert!(fit_arima(AR2_Y, &ArimaSpec::new(11, 0, 0), &o).is_err());
    let mut with_nan = AR2_Y.to_vec();
    with_nan[5] = f64::NAN;
    assert!(fit_arima(&with_nan, &ArimaSpec::new(1, 0, 0), &o).is_err());
    assert!(fit_arima(&[2.0; 40], &ArimaSpec::new(1, 0, 0), &o).is_err());
    let f = ml(AR2_Y, ArimaSpec::new(1, 0, 0));
    assert!(f.forecast(0, 0.95).is_err());
    assert!(f.forecast(3, 1.0).is_err());
}

#[test]
fn ranking_prefers_the_true_order() {
    let specs = [ArimaSpec::new(0, 0, 0), ArimaSpec::new(1, 0, 0), ArimaSpec::new(2, 0, 0), ArimaSpec::new(0, 0, 2)];
    let r = rank_arima(AR2_Y, &specs, &ArimaOptions { method: ArimaMethod::Ml, ..Default::default() }).unwrap();
    assert!(r.windows(2).all(|w| w[0].fit.aicc <= w[1].fit.aicc));
    assert_ne!(r[0].fit.spec, ArimaSpec::new(0, 0, 0));
}

// -------------------------------------------------------------------------------- smoothing

#[test]
fn holt_winters_with_given_weights_matches_an_independent_recursion() {
    let o = SmoothingOptions { alpha: Some(0.35), beta: Some(0.12), gamma: Some(0.25) };
    let f = fit_smoothing(ES_Y, &SmoothingSpec::winters(6, Seasonality::Additive), &o).unwrap();
    all_close(&f.fitted, HWA_FIT, 1e-10, "additive fits");
    close(f.sse, HWA_SSE, 1e-10, "additive SSE");
    let fc = f.forecast(12, 0.95).unwrap();
    all_close(&fc.point, HWA_FC, 1e-10, "additive forecasts");
    let f = fit_smoothing(ES_YM, &SmoothingSpec::winters(6, Seasonality::Multiplicative), &o).unwrap();
    all_close(&f.fitted, HWM_FIT, 1e-10, "multiplicative fits");
    close(f.sse, HWM_SSE, 1e-10, "multiplicative SSE");
    all_close(&f.forecast(12, 0.9).unwrap().point, HWM_FC, 1e-10, "multiplicative forecasts");
}

#[test]
fn optimised_weights_reach_the_grid_minimum() {
    let f =
        fit_smoothing(ES_Y, &SmoothingSpec::winters(6, Seasonality::Additive), &SmoothingOptions::default()).unwrap();
    assert!(f.sse <= HWA_BEST_SSE * (1.0 + 1e-3), "{} vs {}", f.sse, HWA_BEST_SSE);
    let f = fit_smoothing(ES_YM, &SmoothingSpec::winters(6, Seasonality::Multiplicative), &SmoothingOptions::default())
        .unwrap();
    assert!(f.sse <= HWM_BEST_SSE * (1.0 + 1e-3), "{} vs {}", f.sse, HWM_BEST_SSE);
    assert!(f.alpha > 0.0 && f.alpha < 1.0 && f.beta.unwrap() < 1.0 && f.gamma.unwrap() < 1.0);
}

#[test]
fn single_smoothing_matches_and_finds_the_optimal_alpha() {
    let o = SmoothingOptions { alpha: Some(0.4), ..Default::default() };
    let f = fit_smoothing(SES_Y, &SmoothingSpec::single(), &o).unwrap();
    all_close(&f.fitted, SES_FIT, 1e-12, "fits");
    close(*f.level.last().unwrap(), SES_LEVEL, 1e-12, "final level");
    close(f.sse, SES_SSE, 1e-12, "sse");
    let f = fit_smoothing(SES_Y, &SmoothingSpec::single(), &SmoothingOptions::default()).unwrap();
    close(f.alpha, SES_BEST_ALPHA, 1e-4, "alpha");
    close(f.sse, SES_BEST_SSE, 1e-8, "best sse");
    // Single smoothing forecasts are flat and the interval widens as √(1 + (h−1)α²).
    let fc = f.forecast(4, 0.95).unwrap();
    assert!(fc.point.iter().all(|&v| (v - fc.point[0]).abs() < 1e-12));
    for h in 1..=4usize {
        close(fc.se[h - 1], f.sigma * (1.0 + (h as f64 - 1.0) * f.alpha * f.alpha).sqrt(), 1e-12, "se");
    }
}

#[test]
fn double_smoothing_tracks_a_line_and_validates_input() {
    let y: Vec<f64> = (0..30).map(|t| 5.0 + 2.0 * t as f64).collect();
    let f = fit_smoothing(&y, &SmoothingSpec::double(), &SmoothingOptions::default()).unwrap();
    let fc = f.forecast(3, 0.95).unwrap();
    all_close(&fc.point, &[65.0, 67.0, 69.0], 1e-6, "linear extrapolation");
    assert!(f.sse < 1e-8);
    let bad = |o: SmoothingOptions| fit_smoothing(&y, &SmoothingSpec::double(), &o).is_err();
    assert!(bad(SmoothingOptions { alpha: Some(1.5), ..Default::default() }));
    assert!(bad(SmoothingOptions { beta: Some(0.0), ..Default::default() }));
    assert!(fit_smoothing(&[1.0, 2.0], &SmoothingSpec::single(), &SmoothingOptions::default()).is_err());
    assert!(fit_smoothing(
        &[1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0],
        &SmoothingSpec::winters(4, Seasonality::Multiplicative),
        &SmoothingOptions::default()
    )
    .is_err());
    assert!(fit_smoothing(&y, &SmoothingSpec::winters(1, Seasonality::Additive), &SmoothingOptions::default()).is_err());
    assert!(f.forecast(0, 0.95).is_err());
}

#[test]
fn smoothing_model_selection_is_ordered() {
    let r = select_smoothing(ES_Y, Some(6)).unwrap();
    assert!(r.len() >= 4);
    assert!(r.windows(2).all(|w| w[0].aicc <= w[1].aicc));
    // The data have a strong seasonal pattern; a seasonal model must beat single smoothing.
    assert_ne!(r[0].spec, SmoothingSpec::single());
    assert!(r[0].spec.seasonal != Seasonality::None);
}

// -------------------------------------------------------------------------------- decomposition

#[test]
fn multiplicative_linear_trend_decomposition_matches() {
    let d = decompose(DEC_Y, &DecomposeOptions::new(4)).unwrap();
    all_close(&d.coef, DEC_BETA, 1e-10, "trend coefficients");
    all_close(&d.indices, DEC_INDEX, 1e-10, "indices");
    all_close(&d.fitted, DEC_FIT, 1e-10, "fits");
    assert!((d.indices.iter().sum::<f64>() / 4.0 - 1.0).abs() < 1e-12);
    let fc = d.forecast(6, 0.95).unwrap();
    all_close(&fc.point, DEC_FC, 1e-10, "forecasts");
    assert!(fc.se[5] > fc.se[0] * 0.99, "leverage widens the interval");
    for i in 0..DEC_Y.len() {
        close(d.seasonal[i] * d.trend[i], d.fitted[i], 1e-12, "components multiply");
        close(d.irregular[i], DEC_Y[i] / d.fitted[i], 1e-12, "irregular ratio");
    }
}

#[test]
fn additive_moving_average_decomposition_matches() {
    let mut o = DecomposeOptions::new(4);
    o.model = DecompositionModel::Additive;
    o.trend = TrendKind::MovingAverage;
    let d = decompose(DECA_Y, &o).unwrap();
    for (i, (&t, &r)) in d.trend.iter().zip(DECA_TREND).enumerate() {
        if r == -999.0 {
            assert!(t.is_nan(), "trend[{i}] should be missing");
        } else {
            close(t, r, 1e-10, "ma trend");
        }
    }
    all_close(&d.indices, DECA_INDEX, 1e-10, "additive indices");
    assert!(d.indices.iter().sum::<f64>().abs() < 1e-12);
    let fc = d.forecast(4, 0.9).unwrap();
    assert_eq!(fc.point.len(), 4);
    assert!(fc.point.iter().all(|v| v.is_finite()));
}

#[test]
fn decomposition_options_and_errors() {
    let mut o = DecomposeOptions::new(4);
    o.trend = TrendKind::Exponential;
    let d = decompose(DEC_Y, &o).unwrap();
    assert!(d.trend.iter().all(|&t| t > 0.0));
    o.trend = TrendKind::Quadratic;
    assert_eq!(decompose(DEC_Y, &o).unwrap().coef.len(), 3);
    o.trend = TrendKind::Constant;
    close(decompose(DEC_Y, &o).unwrap().trend[3], DEC_Y.iter().sum::<f64>() / 48.0, 1e-12, "constant trend");
    let mut shifted = DecomposeOptions::new(4);
    shifted.first_season = 3;
    let d2 = decompose(DEC_Y, &shifted).unwrap();
    assert!((d2.seasonal[0] - d2.indices[2]).abs() < 1e-12);
    let mut bad = DecomposeOptions::new(4);
    bad.first_season = 5;
    assert!(decompose(DEC_Y, &bad).is_err());
    assert!(decompose(&[1.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &DecomposeOptions::new(2)).is_err());
    assert!(decompose(&[1.0, 2.0, 3.0], &DecomposeOptions::new(1)).is_err());
    let mut ma = DecomposeOptions::new(1);
    ma.trend = TrendKind::MovingAverage;
    assert!(decompose(DEC_Y, &ma).is_err());
}

// -------------------------------------------------------------------------------- reports

#[test]
fn reports_render_validate_and_round_trip_through_json() {
    use openstats_stats::report::SessionReport;
    let a = acf(ACF_X, Some(10), 0.95).unwrap();
    let p = pacf(ACF_X, Some(10), 0.95).unwrap();
    let hw =
        fit_smoothing(ES_Y, &SmoothingSpec::winters(6, Seasonality::Additive), &SmoothingOptions::default()).unwrap();
    let ar = ml(AR2_Y, ArimaSpec::new(2, 0, 0));
    let dec = decompose(DEC_Y, &DecomposeOptions::new(4)).unwrap();
    let mut rep = SessionReport::new("Time series");
    rep.push(a.report_section());
    rep.push(p.report_section());
    rep.push(hw.report_with_forecast(&hw.forecast(6, 0.95).unwrap()));
    rep.push(ar.report_with_forecast(&ar.forecast(6, 0.95).unwrap()));
    rep.push(dec.report_with_forecast(&dec.forecast(4, 0.95).unwrap()));
    rep.validate().unwrap();
    let text = rep.render_text();
    for needle in [
        "Autocorrelation Function",
        "Final Estimates of Parameters",
        "Seasonal Indices",
        "Forecasts (95% prediction limits)",
        "Ljung-Box",
    ] {
        assert!(text.contains(needle), "missing {needle}");
    }
    assert_eq!(SessionReport::from_json(&rep.to_json()).unwrap(), rep);
    assert!(rep.charts().len() >= 10);
}
