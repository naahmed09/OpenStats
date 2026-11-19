//! Time Series and Response Optimizer dialogs, run on a worksheet through the registry path.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::doe::{central_composite, CcdKind, CcdOptions};
use openstats_stats::report::{Block, Reportable, Section, SessionReport};
use openstats_stats::timeseries::{acf, fit_smoothing, SmoothingOptions, SmoothingSpec};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 72;

fn noise(i: usize) -> f64 {
    (((i * 7919 + 13) % 101) as f64 / 101.0 - 0.5) * 2.0
}

fn seasonal(i: usize) -> f64 {
    [0.9, 1.0, 1.15, 1.3, 1.1, 0.95, 0.85, 0.8, 0.95, 1.1, 1.2, 1.05][i % 12]
}

fn data() -> GridController {
    let mut wb = Workbook::new();
    let y: Vec<f64> = (0..N).map(|i| (100.0 + 1.5 * i as f64) * seasonal(i) + 2.0 * noise(i)).collect();
    // Same series with three trailing missing values and one with an interior gap.
    let mut trailing = y.clone();
    trailing.extend([f64::NAN; 3]);
    let mut gap = y.clone();
    gap[10] = f64::NAN;
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("sales", y)).unwrap();
    ws.add_column(Column::numeric("trailing", trailing)).unwrap();
    ws.add_column(Column::numeric("gap", gap)).unwrap();
    // A response-surface design with two responses (columns 3..6).
    let d = central_composite(2, &CcdOptions::new(CcdKind::Circumscribed))
        .unwrap()
        .with_levels(&[(100.0, 200.0), (10.0, 30.0)])
        .unwrap();
    let nat = d.natural();
    let yield_: Vec<f64> = d
        .coded
        .iter()
        .enumerate()
        .map(|(i, x)| {
            50.0 + 8.0 * x[0] + 3.0 * x[1] - 4.0 * x[0] * x[0] - 2.0 * x[1] * x[1] + x[0] * x[1] + 0.05 * noise(i)
        })
        .collect();
    let cost: Vec<f64> = d
        .coded
        .iter()
        .enumerate()
        .map(|(i, x)| 20.0 - 5.0 * x[0] + 2.0 * x[1] + 3.0 * x[1] * x[1] + 0.05 * noise(i + 3))
        .collect();
    ws.add_column(Column::numeric("temp", nat.iter().map(|r| r[0]).collect())).unwrap();
    ws.add_column(Column::numeric("time", nat.iter().map(|r| r[1]).collect())).unwrap();
    ws.add_column(Column::numeric("yield", yield_)).unwrap();
    ws.add_column(Column::numeric("cost", cost)).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

fn run(id: &str, set: &[(&str, Value)]) -> (GridController, Section) {
    try_run(id, set).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn try_run(id: &str, set: &[(&str, Value)]) -> Result<(GridController, Section), String> {
    let reg = Registry::standard();
    let mut g = data();
    let mut v: Values = reg.get(id).unwrap().spec.defaults();
    for (k, x) in set {
        v.set(k, x.clone());
    }
    let problems = validate(&reg.get(id).unwrap().spec, &v, g.worksheet());
    if !problems.is_empty() {
        return Err(format!("{problems:?}"));
    }
    reg.run(id, &mut g, &v).map(|s| (g, s)).map_err(|e| e.to_string())
}

fn vars(c: &[usize]) -> Value {
    Value::Vars(c.to_vec())
}

fn num(v: f64) -> Value {
    Value::Num(v)
}

fn on(b: bool) -> Value {
    Value::Bool(b)
}

fn column(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let c = ws.find(name).unwrap_or_else(|| panic!("column {name} not stored"));
    let col = ws.column(c).unwrap();
    (0..col.len()).map(|i| col.get(i).as_f64()).collect()
}

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

fn charts(s: &Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count()
}

fn y_data() -> Vec<f64> {
    column(&data(), "sales")
}

#[test]
fn menu_paths() {
    let reg = Registry::standard();
    for (id, last) in [
        ("stat.ts.acf", "Autocorrelation..."),
        ("stat.ts.pacf", "Partial Autocorrelation..."),
        ("stat.ts.ses", "Single Exp Smoothing..."),
        ("stat.ts.des", "Double Exp Smoothing..."),
        ("stat.ts.winters", "Winters' Method..."),
        ("stat.ts.arima", "ARIMA..."),
        ("stat.ts.decomp", "Decomposition..."),
    ] {
        let a = reg.get(id).unwrap();
        assert_eq!(a.menu, ["Stat", "Time Series", last]);
    }
    assert_eq!(reg.get("stat.doe.optimizer").unwrap().menu, ["Stat", "DOE", "Response Optimizer..."]);
}

#[test]
fn acf_and_pacf_store_what_the_library_computes() {
    let y = y_data();
    let (g, s) = run("stat.ts.acf", &[("series", vars(&[0])), ("lags", num(15.0)), ("store", on(true))]);
    let want = acf(&y, Some(15), 0.95).unwrap();
    let got = column(&g, "ACF");
    assert_eq!(got.len(), 15);
    for (a, b) in got.iter().zip(&want.acf) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(column(&g, "LBQ").len() == 15 && column(&g, "TSTAT").len() == 15);
    assert!(text(&s).contains("Autocorrelation Function: sales") && charts(&s) == 1);
    let (_, s) = run("stat.ts.acf", &[("series", vars(&[0])), ("plot", on(false))]);
    assert_eq!(charts(&s), 0);
    let (g, s) = run("stat.ts.pacf", &[("series", vars(&[0])), ("lags", num(10.0)), ("store", on(true))]);
    assert_eq!(column(&g, "PACF").len(), 10);
    assert!(text(&s).contains("Partial Autocorrelation Function: sales"));
}

#[test]
fn single_smoothing_stores_fits_residuals_and_forecasts() {
    let y = y_data();
    let (g, s) = run(
        "stat.ts.ses",
        &[
            ("series", vars(&[0])),
            ("alpha", num(0.3)),
            ("forecasts", num(5.0)),
            ("store_fits", on(true)),
            ("store_resid", on(true)),
            ("store_forecast", on(true)),
        ],
    );
    let want =
        fit_smoothing(&y, &SmoothingSpec::single(), &SmoothingOptions { alpha: Some(0.3), ..Default::default() })
            .unwrap();
    let fits = column(&g, "FITS");
    let resid = column(&g, "RESI");
    for i in 0..N {
        assert!((fits[i] - want.fitted[i]).abs() < 1e-12);
        assert!((fits[i] + resid[i] - y[i]).abs() < 1e-9);
    }
    let fc = want.forecast(5, 0.95).unwrap();
    let got = column(&g, "FORE");
    assert_eq!(got.len(), 5);
    for i in 0..5 {
        assert!((got[i] - fc.point[i]).abs() < 1e-12);
    }
    let (lo, hi) = (column(&g, "LOWER"), column(&g, "UPPER"));
    assert!((0..5).all(|i| lo[i] < got[i] && got[i] < hi[i]));
    let t = text(&s);
    assert!(t.contains("Single exponential smoothing for sales") && t.contains("Forecasts (95% prediction limits)"));
}

#[test]
fn double_smoothing_and_winters() {
    let (g, s) = run("stat.ts.des", &[("series", vars(&[0])), ("forecasts", num(3.0)), ("store_forecast", on(true))]);
    assert_eq!(column(&g, "FORE").len(), 3);
    assert!(text(&s).contains("Gamma (trend)"));
    let (g, s) = run(
        "stat.ts.winters",
        &[
            ("series", vars(&[0])),
            ("period", num(12.0)),
            ("forecasts", num(12.0)),
            ("store_fits", on(true)),
            ("store_forecast", on(true)),
        ],
    );
    assert_eq!(column(&g, "FORE").len(), 12);
    let y = y_data();
    let fits = column(&g, "FITS");
    // A seasonal model beats the naive mean by a wide margin on strongly seasonal data.
    let mad: f64 = (12..N).map(|i| (y[i] - fits[i]).abs()).sum::<f64>() / (N - 12) as f64;
    assert!(mad < 8.0, "MAD {mad}");
    assert!(text(&s).contains("Multiplicative Winters' method (period 12)") && text(&s).contains("Seasonal Terms"));
    // Additive choice and an explicit weight.
    let (_, s) = run(
        "stat.ts.winters",
        &[
            ("series", vars(&[0])),
            ("seasonal", Value::Choice(1)),
            ("alpha", num(0.4)),
            ("beta", num(0.1)),
            ("gamma", num(0.2)),
        ],
    );
    assert!(text(&s).contains("Additive Winters"));
}

#[test]
fn arima_stores_fits_and_forecasts_and_reports_diagnostics() {
    let (g, s) = run(
        "stat.ts.arima",
        &[
            ("series", vars(&[0])),
            ("p", num(1.0)),
            ("d", num(1.0)),
            ("q", num(1.0)),
            ("forecasts", num(6.0)),
            ("store_fits", on(true)),
            ("store_resid", on(true)),
            ("store_forecast", on(true)),
        ],
    );
    let y = y_data();
    let fits = column(&g, "FITS");
    let resid = column(&g, "RESI");
    assert!(fits[0].is_nan() && !fits[1].is_nan(), "one observation is lost to differencing");
    for i in 1..N {
        assert!((fits[i] + resid[i] - y[i]).abs() < 1e-9);
    }
    assert_eq!(column(&g, "FORE").len(), 6);
    let t = text(&s);
    for needle in [
        "ARIMA(1,1,1) for sales",
        "Final Estimates of Parameters",
        "AR 1",
        "MA 1",
        "Constant",
        "Ljung-Box",
        "Forecasts",
    ] {
        assert!(t.contains(needle), "missing {needle}");
    }
    // Seasonal model.
    let (g, s) = run(
        "stat.ts.arima",
        &[
            ("series", vars(&[0])),
            ("p", num(0.0)),
            ("d", num(1.0)),
            ("q", num(1.0)),
            ("sp", num(0.0)),
            ("sd", num(1.0)),
            ("sq", num(1.0)),
            ("period", num(12.0)),
            ("constant", on(false)),
            ("method", Value::Choice(1)),
            ("forecasts", num(12.0)),
            ("store_forecast", on(true)),
        ],
    );
    assert_eq!(column(&g, "FORE").len(), 12);
    assert!(text(&s).contains("ARIMA(0,1,1)(0,1,1)12 for sales") && text(&s).contains("SMA 12"));
    // Chart toggles.
    let (_, s) = run("stat.ts.arima", &[("series", vars(&[0])), ("plot", on(false)), ("acf_plot", on(false))]);
    assert_eq!(charts(&s), 0);
}

#[test]
fn decomposition_stores_components_that_recombine() {
    let (g, s) = run(
        "stat.ts.decomp",
        &[
            ("series", vars(&[0])),
            ("period", num(12.0)),
            ("forecasts", num(12.0)),
            ("store_trend", on(true)),
            ("store_seasonal", on(true)),
            ("store_fits", on(true)),
            ("store_resid", on(true)),
            ("store_detrend", on(true)),
            ("store_deseason", on(true)),
            ("store_forecast", on(true)),
        ],
    );
    let (tr, se, fi) = (column(&g, "TREND"), column(&g, "SEAS"), column(&g, "FITS"));
    let y = y_data();
    for i in 0..N {
        assert!((tr[i] * se[i] - fi[i]).abs() < 1e-9, "multiplicative fits row {i}");
        assert!((column(&g, "DETR")[i] - y[i] / tr[i]).abs() < 1e-9);
        assert!((column(&g, "DESEAS")[i] - y[i] / se[i]).abs() < 1e-9);
    }
    assert_eq!(column(&g, "FORE").len(), 12);
    let t = text(&s);
    assert!(t.contains("Multiplicative Time Series Decomposition") && t.contains("Seasonal Indices"));
    // The seasonal shape is recovered: December is the high season index of the generator's pattern.
    let idx: Vec<f64> = (0..12).map(|i| se[i]).collect();
    assert!(idx[3] > idx[7], "April index {} should exceed August {}", idx[3], idx[7]);
    let (_, s) = run(
        "stat.ts.decomp",
        &[("series", vars(&[0])), ("model", Value::Choice(1)), ("trend", Value::Choice(4)), ("period", num(12.0))],
    );
    assert!(text(&s).contains("Additive Time Series Decomposition") && text(&s).contains("moving average"));
}

#[test]
fn series_with_gaps_are_refused_and_trailing_blanks_are_ignored() {
    let e = try_run("stat.ts.acf", &[("series", vars(&[2]))]).unwrap_err();
    assert!(e.contains("missing value in row 11"), "{e}");
    let (g, _) = run("stat.ts.acf", &[("series", vars(&[1])), ("lags", num(5.0)), ("store", on(true))]);
    assert_eq!(column(&g, "ACF").len(), 5);
    // Bad orders and too-short seasons are reported, not panics.
    let e = try_run("stat.ts.winters", &[("series", vars(&[0])), ("period", num(40.0))]).unwrap_err();
    assert!(e.contains("smoothing") || e.contains("observations"), "{e}");
    assert!(try_run("stat.ts.arima", &[("series", vars(&[0])), ("p", num(20.0))]).is_err());
    assert!(try_run("stat.ts.acf", &[]).is_err());
}

#[test]
fn response_optimizer_reports_stores_and_validates() {
    let base = [
        ("factors", vars(&[3, 4])),
        ("responses", vars(&[5, 6])),
        ("goals", Value::Text("max 40 55; min 12 30".into())),
        ("importance", Value::Text("2 1".into())),
    ];
    let mut set: Vec<(&str, Value)> = base.to_vec();
    set.push(("store_fits", on(true)));
    set.push(("store_d", on(true)));
    let (g, s) = run("stat.doe.optimizer", &set);
    let t = text(&s);
    for needle in [
        "Response Optimization: yield, cost versus temp, time",
        "Model summary",
        "Solution 1: composite desirability",
        "Predicted responses",
        "Importance",
    ] {
        assert!(t.contains(needle), "missing {needle}\n{t}");
    }
    let d = column(&g, "COMPD");
    let runs = column(&g, "yield").len();
    assert_eq!(runs, 14, "CCD with two factors and default centre points");
    assert_eq!(d.len(), runs);
    assert!(d.iter().all(|v| (0.0..=1.0).contains(v)));
    let fy = column(&g, "FITS_yield");
    let y = column(&g, "yield");
    assert!((0..runs).all(|i| (fy[i] - y[i]).abs() < 0.2), "the quadratic model reproduces the response");
    assert!(charts(&s) >= 1);
    // Restricted range and a fixed factor.
    let mut set2: Vec<(&str, Value)> = base.to_vec();
    set2.push(("ranges", Value::Text("120 180; *".into())));
    set2.push(("fixed", Value::Text("* 20".into())));
    let (_, s2) = run("stat.doe.optimizer", &set2);
    assert!(text(&s2).contains("Solution 1"));
    // Validation messages.
    let bad = |over: (&'static str, Value)| {
        let mut v: Vec<(&str, Value)> = base.to_vec();
        v.retain(|(k, _)| *k != over.0);
        v.push(over);
        try_run("stat.doe.optimizer", &v).unwrap_err()
    };
    assert!(bad(("goals", Value::Text("max 40 55".into()))).contains("1 goals were entered for 2 responses"));
    assert!(bad(("goals", Value::Text("max 40 55; wiggle 1 2".into()))).contains("cost"));
    assert!(bad(("importance", Value::Text("1 2 3".into()))).contains("importance"));
    assert!(bad(("importance", Value::Text("0 1".into()))).contains("positive"));
    assert!(bad(("ranges", Value::Text("120 180".into()))).contains("one \"low high\" pair per factor"));
    assert!(bad(("ranges", Value::Text("0 180; *".into()))).contains("must lie within"));
    assert!(bad(("fixed", Value::Text("150".into()))).contains("one value"));
    assert!(bad(("goals", Value::Text("".into()))).contains("enter a goal"));
    assert!(bad(("responses", vars(&[3]))).contains("both a factor and a response"));
}

#[test]
fn time_series_sections_survive_json() {
    let (_, s) = run("stat.ts.arima", &[("series", vars(&[0])), ("d", num(1.0)), ("forecasts", num(4.0))]);
    let mut r = SessionReport::new("ts");
    r.push(s);
    r.validate().unwrap();
    assert_eq!(SessionReport::from_json(&r.to_json()).unwrap().to_json(), r.to_json());
    let _ = Reportable::report_section(&openstats_stats::timeseries::acf(&y_data(), Some(4), 0.9).unwrap());
}

/// The last number of the first table of a section.
fn last_num(s: &Section) -> f64 {
    use openstats_stats::report::Cell;
    for b in &s.blocks {
        if let Block::Table(t) = b {
            if let Some(Cell::Num(x)) = t.rows[0].last() {
                return *x;
            }
        }
    }
    panic!("no numeric cell");
}

#[test]
fn unit_root_ccf_and_auto_arima_dialogs() {
    use openstats_stats::timeseries::{adf_test, cross_correlation, kpss_test, AdfOptions, Deterministic};
    let y: Vec<f64> = (0..N).map(|i| (100.0 + 1.5 * i as f64) * seasonal(i) + 2.0 * noise(i)).collect();
    let (_, s) = run("stat.ts.adf", &[("series", vars(&[0])), ("det", Value::Choice(2)), ("lags", num(2.0))]);
    let want = adf_test(&y, &AdfOptions { deterministic: Deterministic::Trend, lags: Some(2) }).unwrap();
    let t = text(&s);
    assert!(t.contains("Augmented Dickey-Fuller Test: sales"), "{t}");
    assert_eq!(last_num(&s), want.statistic);
    // Blank lag count chooses by AIC.
    let (_, s) = run("stat.ts.adf", &[("series", vars(&[0]))]);
    assert!(text(&s).contains("Critical Values"));
    let (_, s) = run("stat.ts.kpss", &[("series", vars(&[0])), ("null", Value::Choice(1))]);
    let k = kpss_test(&y, true, None).unwrap();
    assert_eq!(last_num(&s), k.statistic);
    let (g, s) =
        run("stat.ts.ccf", &[("series", vars(&[0])), ("series2", vars(&[1])), ("lags", num(4.0)), ("store", on(true))]);
    assert!(text(&s).contains("Cross Correlation: sales and trailing") || text(&s).contains("Cross Correlation"));
    assert_eq!(column(&g, "CCF").len(), 9);
    assert!(try_run("stat.ts.ccf", &[("series", vars(&[0])), ("series2", vars(&[2]))]).is_err()); // gap
    let _ = cross_correlation(&y, &y, 2, 0.95).unwrap();
    let (_, s) = run(
        "stat.ts.autoarima",
        &[("series", vars(&[0])), ("max_p", num(1.0)), ("max_q", num(1.0)), ("d", Value::Choice(1))],
    );
    assert!(text(&s).contains("Candidate Models") && text(&s).contains("Differencing order d = 1"), "{}", text(&s));
    assert!(try_run("stat.ts.autoarima", &[("series", vars(&[2]))]).is_err());
}
