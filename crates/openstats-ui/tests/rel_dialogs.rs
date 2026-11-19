//! Stat ▸ Reliability/Survival dialogs, run on a worksheet through the registry path.

use openstats_charts::{render_spec, PlotOptions};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::reliability::*;
use openstats_stats::report::{Block, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 60;

// Columns: 0 time, 1 status (1 = failure, 0 = censored), 2 group, 3 temperature, 4 frequency, 5 start,
// 6 ship, 7..10 returns.
fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

struct Fixture {
    time: Vec<f64>,
    status: Vec<f64>,
    group: Vec<String>,
    temp: Vec<f64>,
}

fn fixture() -> Fixture {
    let (mut time, mut status, mut group, mut temp) = (vec![], vec![], vec![], vec![]);
    for i in 0..N {
        let t_c = [80.0, 100.0, 120.0][i % 3];
        let x = 11604.5 / (t_c + 273.16);
        let mu = -14.9 + 0.7 * x;
        let t = (mu + 0.45 * (-(1.0 - rng(i, 1)).ln()).ln()).exp();
        let cut = 1500.0;
        let (v, s) = if t > cut { (cut, 0.0) } else { (t, 1.0) };
        time.push((v * 100.0).round() / 100.0);
        status.push(s);
        group.push(["A", "B", "C"][i % 3].to_string());
        temp.push(t_c);
    }
    Fixture { time, status, group, temp }
}

fn data() -> GridController {
    let f = fixture();
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("time", f.time.clone())).unwrap();
    ws.add_column(Column::numeric("status", f.status.clone())).unwrap();
    ws.add_column(Column::text("group", f.group.clone())).unwrap();
    ws.add_column(Column::numeric("temp", f.temp.clone())).unwrap();
    ws.add_column(Column::numeric("freq", vec![1.0; N])).unwrap();
    let start: Vec<f64> = (0..N).map(|i| if i < 6 { f.time[i] * 0.8 } else { f64::NAN }).collect();
    ws.add_column(Column::numeric("start", start)).unwrap();
    // warranty block: 5 shipment periods, returns observed 5, 4, 3, 2, 1 periods
    let ship = [1000.0, 1200.0, 900.0, 1500.0, 1100.0];
    let ages = [5usize, 4, 3, 2, 1];
    let mut ship_col = vec![f64::NAN; N];
    let mut rets = vec![vec![f64::NAN; N]; 5];
    for i in 0..5 {
        ship_col[i] = ship[i];
        for (l, col) in rets.iter_mut().enumerate().take(ages[i]) {
            col[i] = (ship[i] * (0.02 + 0.01 * l as f64)).round();
        }
    }
    ws.add_column(Column::numeric("ship", ship_col)).unwrap();
    for (l, c) in rets.into_iter().enumerate() {
        ws.add_column(Column::numeric(format!("r{}", l + 1), c)).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn try_run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
    let reg = Registry::standard();
    let mut v: Values = reg.get(id).unwrap().spec.defaults();
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

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

fn charts(s: &Section) -> Vec<String> {
    s.blocks
        .iter()
        .filter_map(|b| match b {
            Block::Chart(c) => Some(c.title.clone()),
            _ => None,
        })
        .collect()
}

fn life() -> LifeData {
    let f = fixture();
    LifeData::from_times(&f.time, Some(&f.status), 0.0, None).unwrap()
}

#[test]
fn menu_paths_are_under_reliability() {
    let reg = Registry::standard();
    for id in ["stat.rel.dist", "stat.rel.dist_id", "stat.rel.km", "stat.rel.alt", "stat.rel.warranty"] {
        let a = reg.get(id).unwrap_or_else(|| panic!("{id} not registered"));
        assert_eq!(&a.menu[..2], ["Stat", "Reliability/Survival"], "{id}");
    }
}

#[test]
fn distribution_analysis_matches_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.rel.dist",
        &[("time", vars(&[0])), ("censor", vars(&[1])), ("times", txt("200 400")), ("percents", txt("5 50"))],
    );
    let t = text(&s);
    for needle in [
        "Distribution Parameters",
        "Shape",
        "Scale",
        "Table of Percentiles",
        "Table of Survival Probabilities",
        "Mean(MTTF)",
    ] {
        assert!(t.contains(needle), "missing {needle}");
    }
    assert_eq!(charts(&s).len(), 3, "{:?}", charts(&s)); // survival (with the KM overlay), hazard, density
    let fit = fit_distribution(&life(), Family::Weibull).unwrap();
    let shape = fit.parameters(0.95)[0].1.est;
    assert!(t.contains(&format!("{shape:.4}")[..5]), "shape {shape} not in output");
    // every distribution runs, including interval-censored input
    for d in 0..6 {
        run(&mut g, "stat.rel.dist", &[("time", vars(&[0])), ("censor", vars(&[1])), ("dist", Value::Choice(d))]);
    }
    let s = run(&mut g, "stat.rel.dist", &[("time", vars(&[0])), ("start", vars(&[5]))]);
    assert!(text(&s).contains("failures"));
    // frequency column and confidence level
    run(
        &mut g,
        "stat.rel.dist",
        &[("time", vars(&[0])), ("censor", vars(&[1])), ("freq", vars(&[4])), ("level", num(90.0))],
    );
}

#[test]
fn distribution_analysis_reports_input_problems() {
    let mut g = data();
    let e = |g: &mut GridController, set: &[(&str, Value)]| try_run(g, "stat.rel.dist", set).unwrap_err();
    assert!(!e(&mut g, &[]).is_empty());
    assert!(e(&mut g, &[("time", vars(&[0])), ("start", vars(&[5])), ("censor", vars(&[1]))]).contains("not both"));
    assert!(e(&mut g, &[("time", vars(&[0])), ("percents", txt("5 abc"))]).contains("abc"));
    assert!(e(&mut g, &[("time", vars(&[0])), ("percents", txt("150"))]).contains("between 0 and 100"));
    assert!(e(&mut g, &[("time", vars(&[0])), ("times", txt("-5"))]).contains("positive"));
    // everything censored
    let all = vec![1.0; N];
    g.store_columns(vec![Column::numeric("allc", all)]).unwrap();
    let c = g.worksheet().find("allc").unwrap();
    let msg = e(&mut g, &[("time", vars(&[0])), ("censor", vars(&[c])), ("censor_value", num(1.0))]);
    assert!(msg.contains("failure"), "{msg}");
    // text column as time
    assert!(!e(&mut g, &[("time", vars(&[2]))]).is_empty());
}

#[test]
fn distribution_id_ranks_all_families() {
    let mut g = data();
    let s = run(&mut g, "stat.rel.dist_id", &[("time", vars(&[0])), ("censor", vars(&[1]))]);
    let t = text(&s);
    for d in ["Weibull", "Lognormal", "Exponential", "Normal", "Logistic", "Loglogistic"] {
        assert!(t.contains(d), "{d}");
    }
    assert_eq!(charts(&s), vec!["Survival Plot for the Candidate Distributions"]);
    let best = rank_distributions(&life())[0].family.name();
    let table = s.blocks.iter().find_map(|b| if let Block::Table(t) = b { Some(t) } else { None }).unwrap();
    assert!(format!("{:?}", table.rows[0][0]).contains(best));
}

#[test]
fn kaplan_meier_with_and_without_groups() {
    let mut g = data();
    let s = run(&mut g, "stat.rel.km", &[("time", vars(&[0])), ("censor", vars(&[1]))]);
    assert!(text(&s).contains("Kaplan-Meier Estimates"));
    assert_eq!(charts(&s), vec!["Survival Plot"]);
    let s = run(
        &mut g,
        "stat.rel.km",
        &[
            ("time", vars(&[0])),
            ("censor", vars(&[1])),
            ("group", vars(&[2])),
            ("wilcoxon", Value::Bool(true)),
            ("tarone", Value::Bool(true)),
            ("ci", Value::Choice(1)),
        ],
    );
    let t = text(&s);
    for needle in
        ["Log-Rank", "Wilcoxon", "Tarone-Ware", "Observed and Expected Failures", "Characteristics of Variable"]
    {
        assert!(t.contains(needle), "{needle}");
    }
    // p-value agrees with the library
    let f = fixture();
    let parts: Vec<LifeData> = ["A", "B", "C"]
        .iter()
        .map(|gname| {
            let idx: Vec<usize> = (0..N).filter(|&i| f.group[i] == *gname).collect();
            let tt: Vec<f64> = idx.iter().map(|&i| f.time[i]).collect();
            let cc: Vec<f64> = idx.iter().map(|&i| f.status[i]).collect();
            LifeData::from_times(&tt, Some(&cc), 0.0, None).unwrap()
        })
        .collect();
    let lr = compare_groups(&parts, WeightKind::LogRank).unwrap();
    assert!(t.contains(&format!("{:.3}", lr.p_value)) || lr.p_value < 0.0005);
    // errors: arbitrary censoring, one group
    assert!(try_run(&mut g, "stat.rel.km", &[("time", vars(&[0])), ("start", vars(&[5]))]).is_err());
}

#[test]
fn accelerated_life_testing_dialog() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.rel.alt",
        &[
            ("time", vars(&[0])),
            ("censor", vars(&[1])),
            ("stress", vars(&[3])),
            ("use_stress", txt("40")),
            ("times", txt("5000")),
        ],
    );
    let t = text(&s);
    for needle in [
        "Regression Table",
        "Intercept",
        "temp",
        "Table of Percentiles at stress 40",
        "Shape",
        "Table of Survival Probabilities",
    ] {
        assert!(t.contains(needle), "{needle}\n{t}");
    }
    // matches the library fit
    let f = fixture();
    let data = life();
    let stress = StressVariable { name: "temp".into(), values: f.temp.clone(), relationship: Relationship::Arrhenius };
    let fit = fit_alt(&data, &[stress], Family::Weibull).unwrap();
    assert!(t.contains(&format!("{:.3}", fit.beta[1])[..4]), "slope {} not found", fit.beta[1]);
    // other relationships and distributions run
    for rel in 1..4 {
        let r = try_run(
            &mut g,
            "stat.rel.alt",
            &[
                ("time", vars(&[0])),
                ("censor", vars(&[1])),
                ("stress", vars(&[3])),
                ("relationship", Value::Choice(rel)),
                ("dist", Value::Choice(1)),
            ],
        );
        // reciprocal/log/linear on temperatures are valid (all positive)
        r.unwrap_or_else(|e| panic!("relationship {rel}: {e}"));
    }
    assert!(try_run(
        &mut g,
        "stat.rel.alt",
        &[("time", vars(&[0])), ("stress", vars(&[3])), ("use_stress", txt("40 50"))]
    )
    .unwrap_err()
    .contains("one per"));
    assert!(try_run(&mut g, "stat.rel.alt", &[("time", vars(&[0])), ("stress", vars(&[3])), ("percents", txt("0"))])
        .is_err());
    // a constant stress column cannot be fitted
    g.store_columns(vec![Column::numeric("flat", vec![5.0; N])]).unwrap();
    let c = g.worksheet().find("flat").unwrap();
    assert!(try_run(&mut g, "stat.rel.alt", &[("time", vars(&[0])), ("stress", vars(&[c]))])
        .unwrap_err()
        .contains("two stress levels"));
}

#[test]
fn warranty_prediction_dialog() {
    let mut g = data();
    let ret: Vec<usize> = vec![7, 8, 9, 10, 11];
    let s = run(
        &mut g,
        "stat.rel.warranty",
        &[("ship", vars(&[6])), ("returns", vars(&ret)), ("periods", num(6.0)), ("future", num(1000.0))],
    );
    let t = text(&s);
    assert!(t.contains("Forecast of Returns") && t.contains("Expected Returns"));
    let table = s
        .blocks
        .iter()
        .filter_map(|b| if let Block::Table(t) = b { Some(t) } else { None })
        .find(|t| t.title == "Forecast of Returns")
        .unwrap();
    assert_eq!(table.rows.len(), 6);
    assert_eq!(charts(&s), vec!["Expected Returns per Period"]);
    // without future shipments the forecast still runs; other distributions too
    for d in 0..4 {
        run(&mut g, "stat.rel.warranty", &[("ship", vars(&[6])), ("returns", vars(&ret)), ("dist", Value::Choice(d))]);
    }
    // a gap in a row of returns is rejected
    let mut gap = vec![f64::NAN; N];
    gap[0] = 5.0;
    g.store_columns(vec![Column::numeric("gap", gap)]).unwrap();
    let c = g.worksheet().find("gap").unwrap();
    let e = try_run(&mut g, "stat.rel.warranty", &[("ship", vars(&[6])), ("returns", vars(&[c, 7]))]).unwrap_err();
    assert!(e.contains("without gaps"), "{e}");
}

#[test]
fn every_reliability_chart_renders() {
    let mut g = data();
    let sets: Vec<(&str, Vec<(&str, Value)>)> = vec![
        ("stat.rel.dist", vec![("time", vars(&[0])), ("censor", vars(&[1]))]),
        ("stat.rel.dist_id", vec![("time", vars(&[0])), ("censor", vars(&[1]))]),
        ("stat.rel.km", vec![("time", vars(&[0])), ("censor", vars(&[1])), ("group", vars(&[2]))]),
        (
            "stat.rel.alt",
            vec![("time", vars(&[0])), ("censor", vars(&[1])), ("stress", vars(&[3])), ("use_stress", txt("40"))],
        ),
        ("stat.rel.warranty", vec![("ship", vars(&[6])), ("returns", vars(&[7, 8, 9, 10, 11]))]),
    ];
    let mut n = 0;
    for (id, set) in sets {
        let s = run(&mut g, id, &set);
        for b in &s.blocks {
            if let Block::Chart(c) = b {
                let scene = render_spec(c, &PlotOptions::default());
                let svg = openstats_charts::render::svg::to_svg(&scene);
                assert!(svg.contains("<svg") && !svg.contains("NaN"), "{id}: {}", c.title);
                n += 1;
            }
        }
    }
    assert_eq!(n, 7);
}

#[test]
fn cox_regression_dialog_matches_the_library_and_stores_columns() {
    let f = fixture();
    let mut g = data();
    let s = run(
        &mut g,
        "stat.rel.cox",
        &[
            ("time", vars(&[0])),
            ("censor", vars(&[1])),
            ("covariates", vars(&[3])),
            ("censor_value", num(0.0)),
            ("store_lp", Value::Bool(true)),
        ],
    );
    let ev: Vec<bool> = f.status.iter().map(|&c| c != 0.0).collect();
    let want = cox_ph(&f.time, &ev, std::slice::from_ref(&f.temp), &["temp".to_string()], None, &CoxOptions::default())
        .unwrap();
    let t = text(&s);
    assert!(t.contains("Cox Regression: time") && t.contains("Hazard Ratio"), "{t}");
    assert!(t.contains(&format!("{:.3}", want.p_wald)) || t.contains("0.000"), "{t}");
    assert_eq!(g.worksheet().columns.len(), 12 + 1);
    // Factor + strata + Breslow ties.
    let s = run(
        &mut g,
        "stat.rel.cox",
        &[("time", vars(&[0])), ("censor", vars(&[1])), ("factors", vars(&[2])), ("ties", Value::Choice(1))],
    );
    assert!(text(&s).contains("Breslow"), "{}", text(&s));
    // Strata that determine the predictor leave nothing to estimate.
    let e =
        try_run(&mut g, "stat.rel.cox", &[("time", vars(&[0])), ("covariates", vars(&[3])), ("strata", vars(&[2]))]);
    assert!(e.is_err(), "{e:?}");
    // Errors.
    assert!(try_run(&mut g, "stat.rel.cox", &[("time", vars(&[0]))]).is_err());
    assert!(try_run(&mut g, "stat.rel.cox", &[("time", vars(&[0])), ("covariates", vars(&[0]))]).is_err());
    let _ = txt("");
}
