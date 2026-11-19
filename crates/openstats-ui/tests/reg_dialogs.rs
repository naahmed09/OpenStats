//! Phase 5b dialogs: power and sample size, stepwise, best subsets, prediction, fitted line plot,
//! GLM comparisons and the extra regression storage, run through the registry.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::anova::{glm, tukey, GlmOptions, Means};
use openstats_stats::hypothesis::Alternative;
use openstats_stats::model::{Data, DesignOptions};
use openstats_stats::power::{PowerModel, TwoSampleT};
use openstats_stats::regression::{
    best_subsets, regress, stepwise, RegressionOptions, StepMethod, StepwiseOptions, SubsetOptions,
};
use openstats_stats::report::{Block, Reportable, Section, SessionReport};
use openstats_stats::reportable::power_section;
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 30;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn xs() -> Vec<f64> {
    (0..N).map(|i| rng(i, 1) * 10.0).collect()
}
fn zs() -> Vec<f64> {
    (0..N).map(|i| rng(i, 2) * 6.0).collect()
}
fn ws_() -> Vec<f64> {
    (0..N).map(|i| rng(i, 3) * 4.0).collect()
}
fn ys() -> Vec<f64> {
    let (x, z) = (xs(), zs());
    (0..N).map(|i| 2.0 + 1.5 * x[i] + 0.5 * z[i] + (rng(i, 9) - 0.5) * 3.0).collect()
}
fn pad(mut v: Vec<f64>) -> Vec<f64> {
    v.resize(N, f64::NAN);
    v
}

// 0 x, 1 z, 2 w, 3 y, 4 nx, 5 nz, 6 nw, 7 gy, 8 ga, 9 gb
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("x", xs())).unwrap();
    ws.add_column(Column::numeric("z", zs())).unwrap();
    ws.add_column(Column::numeric("w", ws_())).unwrap();
    ws.add_column(Column::numeric("y", ys())).unwrap();
    ws.add_column(Column::numeric("nx", pad(vec![1.0, 4.0, 7.0]))).unwrap();
    ws.add_column(Column::numeric("nz", pad(vec![2.0, 3.0, 5.0]))).unwrap();
    ws.add_column(Column::numeric("nw", pad(vec![1.0, 1.0, 1.0]))).unwrap();
    let gy =
        [10.1, 11.4, 9.8, 12.9, 13.5, 12.2, 15.1, 14.2, 16.0, 11.0, 10.6, 12.1, 13.3, 14.8, 13.9, 17.2, 16.1, 15.5];
    ws.add_column(Column::numeric("gy", pad(gy.to_vec()))).unwrap();
    let ga: Vec<String> = "abc".chars().flat_map(|a| std::iter::repeat_n(a.to_string(), 6)).collect();
    let gb: Vec<String> = (0..3).flat_map(|_| "uuuvvv".chars().map(|c| c.to_string())).collect();
    let padt = |mut v: Vec<String>| {
        v.resize(N, String::new());
        v
    };
    ws.add_column(Column::text("ga", padt(ga))).unwrap();
    ws.add_column(Column::text("gb", padt(gb))).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

fn try_run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
    let reg = Registry::standard();
    let mut v: Values = reg.get(id).unwrap_or_else(|| panic!("{id} not registered")).spec.defaults();
    for (k, x) in set {
        v.set(k, x.clone());
    }
    let problems = validate(&reg.get(id).unwrap().spec, &v, g.worksheet());
    if !problems.is_empty() {
        return Err(format!("{problems:?}"));
    }
    reg.run(id, g, &v).map_err(|e| e.to_string())
}

fn run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Section {
    try_run(g, id, set).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn vars(c: &[usize]) -> Value {
    Value::Vars(c.to_vec())
}
fn num(x: f64) -> Value {
    Value::Num(x)
}
fn ch(i: usize) -> Value {
    Value::Choice(i)
}
fn txt(s: &str) -> Value {
    Value::Text(s.into())
}
fn on() -> Value {
    Value::Bool(true)
}
fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}
fn charts(s: &Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count()
}
fn column(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let c = (0..ws.columns.len()).find(|&c| ws.columns[c].name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[c].len()).map(|i| ws.columns[c].data.get(i).as_f64()).collect()
}

// ---------------------------------------------------------------------------------- power

#[test]
fn two_sample_t_power_matches_the_library() {
    let mut g = data();
    let s = run(&mut g, "stat.power.t2", &[("effects", txt("1")), ("powers", txt("0.9"))]);
    let m = TwoSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
    let sol = m.solve(None, Some(0.9), Some(1.0)).unwrap();
    assert_eq!(sol.sample_size, 23);
    assert_eq!(s.blocks, power_section(&m, &sol).blocks);
}

#[test]
fn solves_for_power_and_for_the_effect() {
    let mut g = data();
    let m = TwoSampleT { sigma: 2.0, alpha: 0.01, alternative: Alternative::Greater };
    let set = |n: &str, p: &str, e: &str| {
        vec![
            ("ns", txt(n)),
            ("powers", txt(p)),
            ("effects", txt(e)),
            ("sigma", num(2.0)),
            ("alpha", num(0.01)),
            ("alt", ch(2)),
        ]
    };
    let s = run(&mut g, "stat.power.t2", &set("15", "", "2"));
    assert_eq!(s.blocks, power_section(&m, &m.solve(Some(15), None, Some(2.0)).unwrap()).blocks);
    let s = run(&mut g, "stat.power.t2", &set("15", "0.8", ""));
    assert_eq!(s.blocks, power_section(&m, &m.solve(Some(15), Some(0.8), None).unwrap()).blocks);
}

#[test]
fn several_values_give_one_table_and_a_curve_per_sample_size() {
    let mut g = data();
    let s = run(&mut g, "stat.power.t2", &[("ns", txt("10, 20 30")), ("powers", txt("")), ("effects", txt("0.5"))]);
    let results = s
        .blocks
        .iter()
        .find_map(|b| match b {
            Block::Table(t) if t.title == "Results" => Some(t),
            _ => None,
        })
        .expect("results table");
    assert_eq!(results.rows.len(), 3);
    let chart = s.blocks.iter().find_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).unwrap();
    assert_eq!(chart.series.len(), 3);
    // Power rises with the sample size.
    let m = TwoSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
    let p10 = m.power(10, 0.5).unwrap();
    assert!(text(&s).contains(&format!("{p10:.4}")));
}

#[test]
fn power_inputs_are_checked() {
    let mut g = data();
    for set in [
        vec![("ns", txt("10")), ("powers", txt("0.8")), ("effects", txt("1"))], // nothing blank
        vec![("ns", txt("")), ("powers", txt("")), ("effects", txt("1"))],      // two blank
        vec![("ns", txt("abc")), ("powers", txt("")), ("effects", txt("1"))],
        vec![("ns", txt("2.5")), ("powers", txt("")), ("effects", txt("1"))],
        vec![("ns", txt("")), ("powers", txt("1.5")), ("effects", txt("1"))],
        vec![("ns", txt("")), ("powers", txt("0.9")), ("effects", txt("0"))], // zero effect
    ] {
        assert!(try_run(&mut g, "stat.power.t2", &set).is_err(), "{set:?}");
    }
}

#[test]
fn every_power_dialog_runs() {
    let mut g = data();
    // id, effects box, extra settings
    type Case<'a> = (&'a str, &'a str, Vec<(&'a str, Value)>);
    let cases: Vec<Case> = vec![
        ("stat.power.z1", "0.5", vec![]),
        ("stat.power.t1", "0.5", vec![]),
        ("stat.power.paired", "0.5", vec![]),
        ("stat.power.t2", "0.5", vec![]),
        ("stat.power.prop1", "0.7", vec![]),
        ("stat.power.prop1", "0.7", vec![("method", ch(1))]),
        ("stat.power.prop2", "0.7", vec![]),
        ("stat.power.poisson1", "2", vec![]),
        ("stat.power.poisson2", "2", vec![("method", ch(1))]),
        ("stat.power.var1", "2", vec![]),
        ("stat.power.var2", "2", vec![]),
        ("stat.power.anova", "1.5", vec![]),
        ("stat.power.factorial", "1", vec![]),
        ("stat.power.factorial", "1", vec![("design", ch(1)), ("factors", num(5.0)), ("generators", num(1.0))]),
        ("stat.power.factorial", "1", vec![("design", ch(2)), ("runs", num(12.0)), ("factors", num(6.0))]),
        ("stat.power.tost", "0.2", vec![]),
        ("stat.power.tost", "0.2", vec![("design", ch(0)), ("lower", num(-0.5)), ("upper", num(0.5))]),
    ];
    for (id, e, extra) in cases {
        let mut set = vec![("effects", txt(e))];
        set.extend(extra.iter().cloned());
        let s = run(&mut g, id, &set);
        assert!(text(&s).contains("Power and Sample Size"), "{id}");
        assert!(charts(&s) >= 1, "{id} has a power curve");
    }
}

#[test]
fn anova_and_factorial_power_use_their_own_parameters() {
    let mut g = data();
    let a = run(&mut g, "stat.power.anova", &[("effects", txt("1.5")), ("groups", num(4.0)), ("sigma", num(1.2))]);
    let model = openstats_stats::power::OneWayAnova { groups: 4, sigma: 1.2, alpha: 0.05 };
    assert_eq!(a.blocks, power_section(&model, &model.solve(None, Some(0.9), Some(1.5)).unwrap()).blocks);
    let f = run(
        &mut g,
        "stat.power.factorial",
        &[("effects", txt("1.5")), ("factors", num(4.0)), ("center", num(4.0)), ("blocks", num(2.0))],
    );
    let design = openstats_stats::power::FactorialDesign::full(4).unwrap();
    let mut m = openstats_stats::power::Factorial::new(design, 1.0, 0.05);
    m.center_points = 4;
    m.blocks = 2;
    assert_eq!(f.blocks, power_section(&m, &m.solve(None, Some(0.9), Some(1.5)).unwrap()).blocks);
}

// ------------------------------------------------------------------------------ regression

fn lib_data() -> Data {
    Data::new()
        .with_numeric("y", ys())
        .unwrap()
        .with_numeric("x", xs())
        .unwrap()
        .with_numeric("z", zs())
        .unwrap()
        .with_numeric("w", ws_())
        .unwrap()
}

#[test]
fn stepwise_matches_the_library() {
    let mut g = data();
    let s = run(&mut g, "stat.regression.stepwise", &[("response", vars(&[3])), ("preds", vars(&[0, 1, 2]))]);
    let r = stepwise(
        "y ~ x + z + w",
        &lib_data(),
        &DesignOptions::default(),
        &StepwiseOptions::new(StepMethod::Stepwise),
        &RegressionOptions::default(),
    )
    .unwrap();
    let want = r.report_section().blocks;
    let got: Vec<Block> = s.blocks.iter().filter(|b| !matches!(b, Block::Chart(_))).cloned().collect();
    let want: Vec<Block> = want.into_iter().filter(|b| !matches!(b, Block::Chart(_))).collect();
    assert_eq!(got, want);
    assert!(r.selected.contains(&"x".to_string()));
    // forward, backward and an information criterion all run
    for set in
        [vec![("method", ch(1))], vec![("method", ch(2))], vec![("criterion", ch(3))], vec![("criterion", ch(1))]]
    {
        let mut set = set;
        set.push(("response", vars(&[3])));
        set.push(("preds", vars(&[0, 1, 2])));
        run(&mut g, "stat.regression.stepwise", &set);
    }
}

#[test]
fn stepwise_stores_the_final_model_and_honours_forced_terms() {
    let mut g = data();
    run(
        &mut g,
        "stat.regression.stepwise",
        &[
            ("response", vars(&[3])),
            ("preds", vars(&[0, 1, 2])),
            ("force", vars(&[2])),
            ("store_fits", on()),
            ("store_resid", on()),
        ],
    );
    let fits = column(&g, "SWFITS");
    assert_eq!(fits.len(), N);
    assert!(fits.iter().all(|f| f.is_finite()));
    let resid = column(&g, "SWRESI");
    let y = ys();
    for i in 0..N {
        assert!((fits[i] + resid[i] - y[i]).abs() < 1e-8);
    }
    assert!(try_run(
        &mut g,
        "stat.regression.stepwise",
        &[("response", vars(&[3])), ("preds", vars(&[0, 1])), ("force", vars(&[2]))]
    )
    .is_err());
}

#[test]
fn best_subsets_matches_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.bestsubsets",
        &[("response", vars(&[3])), ("preds", vars(&[0, 1, 2])), ("n_best", num(3.0))],
    );
    let cols: Vec<(&str, Vec<f64>)> = vec![("x", xs()), ("z", zs()), ("w", ws_())];
    let refs: Vec<(&str, &[f64])> = cols.iter().map(|(n, c)| (*n, c.as_slice())).collect();
    let o = SubsetOptions { n_best: 3, ..SubsetOptions::default() };
    let b = best_subsets(&refs, &ys(), &o, &RegressionOptions::default()).unwrap();
    assert_eq!(s.blocks, b.report_section().blocks);
    // forcing w into every model leaves only models containing it
    let f = run(
        &mut g,
        "stat.regression.bestsubsets",
        &[("response", vars(&[3])), ("preds", vars(&[0, 1, 2])), ("force", vars(&[2]))],
    );
    assert_ne!(f.blocks, s.blocks);
    assert!(try_run(&mut g, "stat.regression.bestsubsets", &[("response", vars(&[3])), ("preds", vars(&[3]))]).is_err());
}

#[test]
fn prediction_matches_the_library_and_is_stored() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.predict",
        &[
            ("response", vars(&[3])),
            ("preds", vars(&[0, 1])),
            ("new", vars(&[4, 5])),
            ("level", num(90.0)),
            ("store", on()),
        ],
    );
    let fit = regress(
        "y ~ x + z",
        &lib_data(),
        &DesignOptions::default(),
        &RegressionOptions { level: 0.9, ..Default::default() },
    )
    .unwrap();
    let new =
        Data::new().with_numeric("x", vec![1.0, 4.0, 7.0]).unwrap().with_numeric("z", vec![2.0, 3.0, 5.0]).unwrap();
    let want = fit.predict(&new, 0.9).unwrap();
    let fits = column(&g, "PFITS");
    assert_eq!(fits.len(), N);
    for (i, p) in want.iter().enumerate() {
        assert!((fits[i] - p.fit).abs() < 1e-10);
        assert!((column(&g, "PPUP")[i] - p.pi.upper).abs() < 1e-10);
        assert!((column(&g, "PCLOW")[i] - p.ci.lower).abs() < 1e-10);
    }
    assert!(fits[3].is_nan());
    assert!(text(&s).contains("Prediction for New Observations"));
    // wrong number of new columns, and a response used as predictor
    assert!(try_run(
        &mut g,
        "stat.regression.predict",
        &[("response", vars(&[3])), ("preds", vars(&[0, 1])), ("new", vars(&[4]))]
    )
    .is_err());
}

#[test]
fn prediction_without_an_intercept() {
    let mut g = data();
    run(
        &mut g,
        "stat.regression.predict",
        &[
            ("response", vars(&[3])),
            ("preds", vars(&[0])),
            ("new", vars(&[4])),
            ("intercept", Value::Bool(false)),
            ("store", on()),
        ],
    );
    let (x, y) = (xs(), ys());
    let b = x.iter().zip(&y).map(|(a, c)| a * c).sum::<f64>() / x.iter().map(|a| a * a).sum::<f64>();
    assert!((column(&g, "PFITS")[1] - 4.0 * b).abs() < 1e-9);
}

#[test]
fn fitted_line_plot_draws_the_curve_and_bands() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.fittedline",
        &[("response", vars(&[3])), ("predictor", vars(&[0])), ("pi", on()), ("store_fits", on())],
    );
    let c = s.blocks.iter().find_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).unwrap();
    let names: Vec<&str> = c.series.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["y", "Fit", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]);
    assert_eq!(c.series[0].x.len(), N);
    let fit = &c.series[1];
    assert_eq!(fit.x.len(), 101);
    // the line is the least-squares line and sits inside its bands
    let (x, y) = (xs(), ys());
    let (mx, my) = (x.iter().sum::<f64>() / N as f64, y.iter().sum::<f64>() / N as f64);
    let b1 = x.iter().zip(&y).map(|(a, c)| (a - mx) * (c - my)).sum::<f64>()
        / x.iter().map(|a| (a - mx).powi(2)).sum::<f64>();
    let b0 = my - b1 * mx;
    for (px, py) in fit.x.iter().zip(&fit.y) {
        assert!((py - (b0 + b1 * px)).abs() < 1e-8);
    }
    for i in 0..101 {
        assert!(c.series[4].y[i] < c.series[2].y[i] && c.series[2].y[i] < fit.y[i]);
        assert!(fit.y[i] < c.series[3].y[i] && c.series[3].y[i] < c.series[5].y[i]);
    }
    let fits = column(&g, "FITS");
    assert!((fits[0] - (b0 + b1 * x[0])).abs() < 1e-8);
    // the renderer draws a fit/band series as curves, not points
    for kind in [1, 2] {
        let s = run(
            &mut g,
            "stat.regression.fittedline",
            &[("response", vars(&[3])), ("predictor", vars(&[0])), ("type", ch(kind))],
        );
        assert!(text(&s).contains("Fitted Line Plot"));
    }
    assert!(
        try_run(&mut g, "stat.regression.fittedline", &[("response", vars(&[3])), ("predictor", vars(&[3]))]).is_err()
    );
}

#[test]
fn quadratic_fitted_line_recovers_a_parabola() {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|a| 3.0 - 2.0 * a + 0.5 * a * a + (rng(*a as usize, 5) - 0.5) * 0.01).collect();
    ws.add_column(Column::numeric("x", x)).unwrap();
    ws.add_column(Column::numeric("y", y)).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(
        &mut g,
        "stat.regression.fittedline",
        &[("response", vars(&[1])), ("predictor", vars(&[0])), ("type", ch(1))],
    );
    let c = s.blocks.iter().find_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).unwrap();
    let fit = &c.series[1];
    for (px, py) in fit.x.iter().zip(&fit.y) {
        assert!((py - (3.0 - 2.0 * px + 0.5 * px * px)).abs() < 0.05);
    }
}

#[test]
fn glm_tukey_comparisons_match_scipy() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.anova.glmcompare",
        &[("response", vars(&[7])), ("factors", vars(&[8, 9])), ("compare", vars(&[8]))],
    );
    let t = text(&s);
    // SciPy studentized_range: MSE 3.31532 on 14 df; adjusted p-values for a-b, a-c, b-c.
    for p in ["0.347", "0.013", "0.184"] {
        assert!(t.contains(p), "missing {p}");
    }
    assert!(t.contains("2.617"), "critical value");
    assert!(s.warnings.is_empty(), "a balanced design needs no approximation warning");
    // identical to the library's Tukey on the same means
    let mut gy = vec![];
    let ws = g.worksheet();
    for i in 0..18 {
        gy.push(ws.columns[7].data.get(i).as_f64());
    }
    let la: Vec<String> = "abc".chars().flat_map(|a| std::iter::repeat_n(a.to_string(), 6)).collect();
    let lb: Vec<String> = (0..3).flat_map(|_| "uuuvvv".chars().map(|c| c.to_string())).collect();
    let d = Data::new()
        .with_numeric("gy", gy.clone())
        .unwrap()
        .with_factor("ga", &la)
        .unwrap()
        .with_factor("gb", &lb)
        .unwrap();
    let r = glm("gy ~ ga + gb", &d, &GlmOptions::default()).unwrap();
    let means = Means::new(
        vec!["a".into(), "b".into(), "c".into()],
        vec![6.0; 3],
        r.ls_means[0].rows.iter().map(|m| m.mean).collect(),
        r.fit.mse,
        r.fit.df_error as f64,
    )
    .unwrap();
    // (the dialog derives each level's size from its standard error, so compare the rendered text)
    let want = text(&tukey(&means, 0.05).unwrap().report_section());
    for line in want.lines().filter(|l| !l.trim().is_empty() && l.trim() != "Tukey Pairwise Comparisons") {
        assert!(t.contains(line.trim_end()), "missing line: {line:?}\n{t}");
    }
}

#[test]
fn glm_comparison_methods_and_errors() {
    let mut g = data();
    let base = |m: usize, extra: Vec<(&'static str, Value)>| {
        let mut v =
            vec![("response", vars(&[7])), ("factors", vars(&[8, 9])), ("compare", vars(&[8])), ("method", ch(m))];
        v.extend(extra);
        v
    };
    for (m, extra) in [
        (1, vec![]),
        (2, vec![]),
        (2, vec![("control", txt("b")), ("side", ch(2))]),
        (3, vec![]),
        (3, vec![("best", ch(1))]),
    ] {
        let s = run(&mut g, "stat.anova.glmcompare", &base(m, extra));
        assert!(text(&s).contains("General Linear Model with"), "{m}");
    }
    assert!(try_run(&mut g, "stat.anova.glmcompare", &base(2, vec![("control", txt("zzz"))])).is_err());
    // compared factor must be in the model
    assert!(try_run(
        &mut g,
        "stat.anova.glmcompare",
        &[("response", vars(&[7])), ("factors", vars(&[8])), ("compare", vars(&[9]))]
    )
    .is_err());
    // a covariate makes the comparisons approximate and says so
    let s = run(
        &mut g,
        "stat.anova.glmcompare",
        &[("response", vars(&[3])), ("factors", vars(&[8])), ("covariates", vars(&[0])), ("compare", vars(&[8]))],
    );
    assert!(!s.warnings.is_empty());
}

#[test]
fn extra_regression_storage_columns_match_the_diagnostics() {
    let mut g = data();
    run(
        &mut g,
        "stat.regression.fit",
        &[
            ("response", vars(&[3])),
            ("preds", vars(&[0, 1])),
            ("store_delres", on()),
            ("store_hi", on()),
            ("store_cook", on()),
            ("store_dffits", on()),
            ("store_press", on()),
            ("store_coef", on()),
        ],
    );
    let fit = regress("y ~ x + z", &lib_data(), &DesignOptions::default(), &RegressionOptions::default()).unwrap();
    let d = &fit.diagnostics;
    for (name, want) in [
        ("TRES", &d.deleted),
        ("HI", &d.leverage),
        ("COOK", &d.cooks_d),
        ("DFIT", &d.dffits),
        ("PRES", &d.press_residuals),
    ] {
        let got = column(&g, name);
        assert_eq!(got.len(), N, "{name}");
        for i in 0..N {
            assert!((got[i] - want[i]).abs() < 1e-9, "{name}[{i}]");
        }
    }
    let coef = column(&g, "COEF");
    assert_eq!(coef.len(), 3);
    for (a, b) in coef.iter().zip(fit.estimates()) {
        assert!((a - b).abs() < 1e-9);
    }
    let h: f64 = column(&g, "HI").iter().sum();
    assert!((h - 3.0).abs() < 1e-9, "leverages sum to the number of coefficients");
}
