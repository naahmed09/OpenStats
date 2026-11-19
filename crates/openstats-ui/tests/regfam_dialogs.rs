//! Phase 5f dialogs: logistic, Poisson / GLM, nonlinear, PLS and orthogonal regression, weights and
//! Box-Cox in the Regression dialog, run through the registry and compared with library output.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::glm::ordinal::{nominal_logistic, ordinal_logistic, CategoricalOptions};
use openstats_stats::glm::{glm, Family, GlmOptions, Link};
use openstats_stats::model::{Data, DesignOptions};
use openstats_stats::nonlinear::{nonlinear_regression, NonlinearOptions, Parameter};
use openstats_stats::orthogonal::orthogonal_regression;
use openstats_stats::regression::{regress_weighted, RegressionOptions};
use openstats_stats::report::{Block, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 120;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn sigmoid(e: f64) -> f64 {
    1.0 / (1.0 + (-e).exp())
}

fn xa() -> Vec<f64> {
    (0..N).map(|i| ((rng(i, 1) - 0.5) * 4.0 * 1000.0).round() / 1000.0).collect()
}
fn xb() -> Vec<f64> {
    (0..N).map(|i| ((rng(i, 2) * 3.0) * 1000.0).round() / 1000.0).collect()
}
fn grp() -> Vec<String> {
    (0..N).map(|i| ["p", "q", "r"][(rng(i, 3) * 3.0) as usize % 3].to_string()).collect()
}
fn yes() -> Vec<String> {
    let (a, b, g) = (xa(), xb(), grp());
    (0..N)
        .map(|i| {
            let e = -0.3 + 0.9 * a[i] - 0.6 * b[i] + if g[i] == "q" { 0.7 } else { 0.0 };
            if rng(i, 9) < sigmoid(e) { "yes" } else { "no" }.to_string()
        })
        .collect()
}
fn cnt() -> Vec<f64> {
    let a = xa();
    (0..N).map(|i| ((0.3 + 0.4 * a[i]).exp() * (0.4 + rng(i, 4))).round()).collect()
}
fn cat() -> Vec<String> {
    let a = xa();
    (0..N)
        .map(|i| {
            let z = a[i] + (rng(i, 5) / (1.0 - rng(i, 5))).ln().clamp(-3.0, 3.0);
            if z < -0.8 {
                "low"
            } else if z < 0.9 {
                "mid"
            } else {
                "high"
            }
            .to_string()
        })
        .collect()
}
fn nom() -> Vec<String> {
    let (a, b) = (xa(), xb());
    (0..N)
        .map(|i| {
            let s = [0.0, 0.9 * a[i] - 0.3 * b[i], -0.6 * a[i] + 0.8 * b[i]];
            let g = [rng(i, 6), rng(i, 7), rng(i, 8)].map(|u| -(-u.ln()).ln());
            let k = (0..3).max_by(|&p, &q| (s[p] + g[p]).total_cmp(&(s[q] + g[q]))).unwrap();
            ["red", "green", "blue"][k].to_string()
        })
        .collect()
}
fn gy() -> Vec<f64> {
    let b = xb();
    (0..N).map(|i| ((1.0 + 0.3 * b[i]).exp() * (0.5 + rng(i, 10)) * 1000.0).round() / 1000.0).collect()
}
fn tr() -> Vec<f64> {
    vec![20.0; N]
}
fn ev() -> Vec<f64> {
    let b = xb();
    (0..N).map(|i| (20.0 * sigmoid(-1.0 + 0.6 * b[i]) * (0.6 + 0.8 * rng(i, 11))).floor().min(20.0)).collect()
}
fn dose() -> Vec<f64> {
    (0..N).map(|i| i as f64 / 20.0).collect()
}
fn resp() -> Vec<f64> {
    let d = dose();
    (0..N)
        .map(|i| (((4.0 * (-0.45 * d[i]).exp() + 1.0 + 0.1 * (rng(i, 12) - 0.5)) * 10000.0).round()) / 10000.0)
        .collect()
}
fn wts() -> Vec<f64> {
    (0..N).map(|i| 1.0 + (rng(i, 13) * 4.0).floor()).collect()
}

// 0 yn, 1 xa, 2 xb, 3 grp, 4 cnt, 5 cat, 6 nom, 7 gy, 8 ev, 9 tr, 10 dose, 11 resp, 12 wt
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("yn", yes())).unwrap();
    ws.add_column(Column::numeric("xa", xa())).unwrap();
    ws.add_column(Column::numeric("xb", xb())).unwrap();
    ws.add_column(Column::text("grp", grp())).unwrap();
    ws.add_column(Column::numeric("cnt", cnt())).unwrap();
    ws.add_column(Column::text("cat", cat())).unwrap();
    ws.add_column(Column::text("nom", nom())).unwrap();
    ws.add_column(Column::numeric("gy", gy())).unwrap();
    ws.add_column(Column::numeric("ev", ev())).unwrap();
    ws.add_column(Column::numeric("tr", tr())).unwrap();
    ws.add_column(Column::numeric("dose", dose())).unwrap();
    ws.add_column(Column::numeric("resp", resp())).unwrap();
    ws.add_column(Column::numeric("wt", wts())).unwrap();
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
fn txt(s: &str) -> Value {
    Value::Text(s.into())
}
fn ch(i: usize) -> Value {
    Value::Choice(i)
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
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{a} vs {b}");
}

fn lib_data() -> Data {
    Data::new()
        .with_numeric("y_", yes().iter().map(|s| f64::from(u8::from(s == "yes"))).collect())
        .unwrap()
        .with_numeric("xa", xa())
        .unwrap()
        .with_numeric("xb", xb())
        .unwrap()
        .with_factor("grp", &grp())
        .unwrap()
}

#[test]
fn registry_holds_the_new_dialogs() {
    let reg = Registry::standard();
    for id in [
        "stat.regression.binary",
        "stat.regression.ordinal",
        "stat.regression.nominal",
        "stat.regression.poisson",
        "stat.regression.glmfamily",
        "stat.regression.nonlinear",
        "stat.regression.pls",
        "stat.regression.orthogonal",
    ] {
        assert!(reg.get(id).is_some(), "{id}");
    }
}

#[test]
fn binary_logistic_dialog_matches_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.binary",
        &[
            ("response", vars(&[0])),
            ("factors", vars(&[3])),
            ("covariates", vars(&[1, 2])),
            ("store_fits", on()),
            ("store_dres", on()),
            ("store_hi", on()),
            ("store_coef", on()),
            ("roc", on()),
            ("hl_table", on()),
        ],
    );
    let want = glm("y_ ~ grp + xa + xb", &lib_data(), &DesignOptions::default(), &GlmOptions::logistic()).unwrap();
    let fits = column(&g, "EPRO");
    assert_eq!(fits.len(), N);
    for (a, b) in fits.iter().zip(&want.fitted) {
        close(*a, *b, 1e-12);
    }
    let coef = column(&g, "COEF");
    close(coef[1], want.coefficients[1].estimate, 1e-12);
    assert_eq!(column(&g, "HI").len(), N);
    let t = text(&s);
    for needle in
        ["Binary Logistic Regression", "Odds Ratio", "Hosmer", "Classification Table", "ROC", "Measures of Association"]
    {
        assert!(t.contains(needle), "missing {needle}");
    }
    assert!(t.contains("event: yes"));
    assert!(charts(&s) >= 1);
}

#[test]
fn binary_logistic_event_choice_flips_the_signs() {
    let mut g = data();
    let base = [("response", vars(&[0])), ("covariates", vars(&[1])), ("store_coef", on())];
    run(&mut g, "stat.regression.binary", &base);
    let a = column(&g, "COEF");
    let mut g2 = data();
    let mut set = base.to_vec();
    set.push(("event", txt("no")));
    run(&mut g2, "stat.regression.binary", &set);
    let b = column(&g2, "COEF");
    close(a[1], -b[1], 1e-9);
    // Unknown event and a response with three values are refused.
    let mut set = base.to_vec();
    set.push(("event", txt("maybe")));
    assert!(try_run(&mut g, "stat.regression.binary", &set).unwrap_err().contains("not a value"));
    let e =
        try_run(&mut g, "stat.regression.binary", &[("response", vars(&[5])), ("covariates", vars(&[1]))]).unwrap_err();
    assert!(e.contains("exactly 2"), "{e}");
}

#[test]
fn binary_logistic_events_and_trials() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.binary",
        &[("response", vars(&[8])), ("trials", vars(&[9])), ("covariates", vars(&[2])), ("store_fits", on())],
    );
    let d = Data::new().with_numeric("e", ev()).unwrap().with_numeric("xb", xb()).unwrap();
    let o = GlmOptions { trials: Some(tr()), ..GlmOptions::logistic() };
    let want = glm("e ~ xb", &d, &DesignOptions::default(), &o).unwrap();
    for (a, b) in column(&g, "EPRO").iter().zip(&want.fitted) {
        close(*a, *b, 1e-12);
    }
    assert!(text(&s).contains("events in"));
}

#[test]
fn binary_logistic_stepwise_and_links() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.binary",
        &[("response", vars(&[0])), ("covariates", vars(&[1, 2])), ("factors", vars(&[3])), ("stepwise", ch(2))],
    );
    let t = text(&s);
    assert!(t.contains("Steps") && t.contains("Forward") && t.contains("Selected terms"), "{t}");
    let p =
        run(&mut g, "stat.regression.binary", &[("response", vars(&[0])), ("covariates", vars(&[1])), ("link", ch(1))]);
    assert!(text(&p).contains("Probit"));
    let bad = try_run(
        &mut g,
        "stat.regression.binary",
        &[("response", vars(&[0])), ("covariates", vars(&[1])), ("classification", on()), ("cutoff", num(1.5))],
    );
    assert!(bad.is_err());
}

#[test]
fn ordinal_and_nominal_dialogs_match_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.ordinal",
        &[
            ("response", vars(&[5])),
            ("covariates", vars(&[1])),
            ("order", txt("low, mid, high")),
            ("store_probs", on()),
        ],
    );
    let codes: Vec<f64> =
        cat().iter().map(|c| ["low", "mid", "high"].iter().position(|x| x == c).unwrap() as f64).collect();
    let d = Data::new().with_numeric("y_", codes).unwrap().with_numeric("xa", xa()).unwrap();
    let labels: Vec<String> = ["low", "mid", "high"].iter().map(|s| s.to_string()).collect();
    let want = ordinal_logistic("y_ ~ xa", &d, &DesignOptions::default(), &CategoricalOptions::new(labels)).unwrap();
    let p_mid = column(&g, "PROB_mid");
    for i in 0..N {
        close(p_mid[i], want.probabilities[(i, 1)], 1e-12);
    }
    let sum: Vec<f64> = (0..N).map(|i| column(&g, "PROB_low")[i] + p_mid[i] + column(&g, "PROB_high")[i]).collect();
    assert!(sum.iter().all(|v| (v - 1.0).abs() < 1e-12));
    let t = text(&s);
    assert!(t.contains("Ordinal Logistic Regression") && t.contains("low < mid < high") && t.contains("Const(1)"));
    // A category order that omits a value is refused.
    let e = try_run(
        &mut g,
        "stat.regression.ordinal",
        &[("response", vars(&[5])), ("covariates", vars(&[1])), ("order", txt("low, high"))],
    )
    .unwrap_err();
    assert!(e.contains("mid"), "{e}");

    let n = run(
        &mut g,
        "stat.regression.nominal",
        &[
            ("response", vars(&[6])),
            ("covariates", vars(&[1, 2])),
            ("order", txt("red, green, blue")),
            ("store_probs", on()),
        ],
    );
    let codes: Vec<f64> =
        nom().iter().map(|c| ["red", "green", "blue"].iter().position(|x| x == c).unwrap() as f64).collect();
    let d = Data::new()
        .with_numeric("y_", codes)
        .unwrap()
        .with_numeric("xa", xa())
        .unwrap()
        .with_numeric("xb", xb())
        .unwrap();
    let labels: Vec<String> = ["red", "green", "blue"].iter().map(|s| s.to_string()).collect();
    let want =
        nominal_logistic("y_ ~ xa + xb", &d, &DesignOptions::default(), &CategoricalOptions::new(labels)).unwrap();
    let pg = column(&g, "PROB_green");
    for i in 0..N {
        close(pg[i], want.probabilities[(i, 1)], 1e-10);
    }
    assert!(text(&n).contains("Logit green/red"));
    assert!(try_run(
        &mut g,
        "stat.regression.nominal",
        &[("response", vars(&[6])), ("covariates", vars(&[1])), ("reference", txt("purple"))]
    )
    .is_err());
}

#[test]
fn poisson_and_glm_dialogs_match_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.poisson",
        &[
            ("response", vars(&[4])),
            ("covariates", vars(&[1])),
            ("store_fits", on()),
            ("store_cook", on()),
            ("residual_plots", on()),
        ],
    );
    let d = Data::new().with_numeric("y", cnt()).unwrap().with_numeric("xa", xa()).unwrap();
    let want = glm("y ~ xa", &d, &DesignOptions::default(), &GlmOptions::poisson()).unwrap();
    for (a, b) in column(&g, "FITS").iter().zip(&want.fitted) {
        close(*a, *b, 1e-12);
    }
    assert!(column(&g, "COOK").iter().all(|v| *v >= 0.0));
    assert!(text(&s).contains("Poisson Regression") && charts(&s) == 1);
    let nb = run(
        &mut g,
        "stat.regression.poisson",
        &[("response", vars(&[4])), ("covariates", vars(&[1])), ("family", ch(1))],
    );
    assert!(text(&nb).contains("Negative Binomial") && text(&nb).contains("theta"));
    let mut g = data();
    let gm = run(
        &mut g,
        "stat.regression.glmfamily",
        &[("response", vars(&[7])), ("covariates", vars(&[2])), ("link", ch(1)), ("store_fits", on())],
    );
    let d = Data::new().with_numeric("y", gy()).unwrap().with_numeric("xb", xb()).unwrap();
    let o = GlmOptions { link: Link::Log, ..GlmOptions::new(Family::Gamma) };
    let want = glm("y ~ xb", &d, &DesignOptions::default(), &o).unwrap();
    for (a, b) in column(&g, "FITS").iter().zip(&want.fitted) {
        close(*a, *b, 1e-12);
    }
    assert!(text(&gm).contains("Gamma"));
    // A response outside the family is reported, not fitted.
    assert!(
        try_run(&mut g, "stat.regression.poisson", &[("response", vars(&[1])), ("covariates", vars(&[2]))]).is_err()
    );
}

#[test]
fn nonlinear_dialog_matches_the_library_and_draws_bands() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.nonlinear",
        &[
            ("response", vars(&[11])),
            ("equation", txt("a*exp(-b*dose) + c")),
            ("params", txt("a=3, b=0.3, c=0")),
            ("pi", on()),
            ("store_fits", on()),
            ("store_resid", on()),
        ],
    );
    let d = Data::new().with_numeric("response_", resp()).unwrap().with_numeric("dose", dose()).unwrap();
    let params = [Parameter::new("a", 3.0), Parameter::new("b", 0.3), Parameter::new("c", 0.0)];
    let want =
        nonlinear_regression("response_", "a*exp(-b*dose) + c", &params, &d, &NonlinearOptions::default()).unwrap();
    for (a, b) in column(&g, "FITS").iter().zip(&want.fitted) {
        close(*a, *b, 1e-9);
    }
    assert_eq!(charts(&s), 2, "residual plot and fitted curve");
    let t = text(&s);
    assert!(t.contains("Parameter Estimates") && t.contains("SE Estimate"));
    // Bounds are honoured and reported.
    let b = run(
        &mut g,
        "stat.regression.nonlinear",
        &[
            ("response", vars(&[11])),
            ("equation", txt("a*exp(-b*dose) + c")),
            ("params", txt("a=3, b=0.3, c=0")),
            ("upper", txt("a=3.5")),
        ],
    );
    assert!(text(&b).contains("bound"));
    // Errors: missing equation, unknown name, bound for a non-parameter.
    assert!(try_run(&mut g, "stat.regression.nonlinear", &[("response", vars(&[11])), ("params", txt("a=1"))]).is_err());
    let e = try_run(
        &mut g,
        "stat.regression.nonlinear",
        &[("response", vars(&[11])), ("equation", txt("a*exp(-b*zzz)")), ("params", txt("a=1, b=1"))],
    )
    .unwrap_err();
    assert!(e.contains("zzz"), "{e}");
    assert!(try_run(
        &mut g,
        "stat.regression.nonlinear",
        &[("response", vars(&[11])), ("equation", txt("a*dose")), ("params", txt("a=1")), ("lower", txt("q=0"))]
    )
    .is_err());
}

#[test]
fn pls_dialog_stores_fits_and_scores() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.pls",
        &[
            ("responses", vars(&[7, 4])),
            ("preds", vars(&[1, 2, 10])),
            ("components", num(2.0)),
            ("store_fits", on()),
            ("store_resid", on()),
            ("store_coef", on()),
            ("store_scores", on()),
            ("standardize", on()),
        ],
    );
    let t = text(&s);
    assert!(t.contains("Model Selection and Validation for gy") && t.contains("VIP"));
    let fits = column(&g, "FITS_gy");
    let res = column(&g, "RESI_gy");
    let y = gy();
    for i in 0..N {
        close(fits[i] + res[i], y[i], 1e-9);
    }
    assert_eq!(column(&g, "COEF_cnt").iter().filter(|v| v.is_finite()).count(), 4);
    assert_eq!(column(&g, "XSCORE2").len(), N);
    assert!(try_run(&mut g, "stat.regression.pls", &[("responses", vars(&[7])), ("preds", vars(&[7, 1]))]).is_err());
}

#[test]
fn orthogonal_dialog_matches_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.orthogonal",
        &[
            ("response", vars(&[11])),
            ("predictor", vars(&[10])),
            ("ratio", num(2.0)),
            ("store_fits", on()),
            ("store_dist", on()),
        ],
    );
    let want = orthogonal_regression("dose", &dose(), "resp", &resp(), 2.0, 0.95).unwrap();
    for (a, b) in column(&g, "FITS").iter().zip(&want.fitted) {
        close(*a, *b, 1e-12);
    }
    assert_eq!(column(&g, "ODIST").len(), N);
    assert!(text(&s).contains("Orthogonal Regression Analysis") && text(&s).contains("jackknife"));
    assert!(try_run(&mut g, "stat.regression.orthogonal", &[("response", vars(&[11])), ("predictor", vars(&[11]))])
        .is_err());
}

#[test]
fn regression_dialog_weights_and_box_cox() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.regression.fit",
        &[
            ("response", vars(&[7])),
            ("preds", vars(&[2])),
            ("weights", vars(&[12])),
            ("store_fits", on()),
            ("store_coef", on()),
        ],
    );
    let d = Data::new().with_numeric("y", gy()).unwrap().with_numeric("x", xb()).unwrap();
    let want = regress_weighted("y ~ x", &d, &DesignOptions::default(), &wts(), &RegressionOptions::default()).unwrap();
    let coef = column(&g, "COEF");
    close(coef[0], want.coefficients[0].estimate, 1e-9);
    close(coef[1], want.coefficients[1].estimate, 1e-9);
    for (a, b) in column(&g, "FITS").iter().zip(&want.diagnostics.fitted) {
        close(*a, *b, 1e-9);
    }
    assert!(text(&s).contains("Weighted least squares"));
    // Non-positive weights are refused.
    let mut g2 = data();
    assert!(try_run(
        &mut g2,
        "stat.regression.fit",
        &[("response", vars(&[7])), ("preds", vars(&[2])), ("weights", vars(&[1]))]
    )
    .is_err());
    // Box-Cox: the response is transformed and the fits are on the transformed scale.
    let b = run(
        &mut g,
        "stat.regression.fit",
        &[("response", vars(&[7])), ("preds", vars(&[2])), ("transform", ch(3)), ("lambda", num(0.0))],
    );
    assert!(text(&b).contains("Box-Cox transformed with lambda = 0.0000"));
    let e = try_run(
        &mut g,
        "stat.regression.fit",
        &[("response", vars(&[7])), ("preds", vars(&[2])), ("transform", ch(3))],
    )
    .unwrap_err();
    assert!(e.contains("lambda"), "{e}");
}
