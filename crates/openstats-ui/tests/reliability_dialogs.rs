//! Reliability completion dialogs (Phase 5n), compared with the library routines they wrap.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::reliability::*;
use openstats_stats::report::{Block, Cell, ChartKind, ChartSpec, Section, SessionReport, Table};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn vars(v: &[usize]) -> Value {
    Value::Vars(v.to_vec())
}

fn grid(cols: Vec<Column>) -> GridController {
    let mut wb = Workbook::new();
    for c in cols {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn run_full(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<(Section, Vec<String>), String> {
    let reg = Registry::standard();
    let mut v: Values = reg.get(id).unwrap().spec.defaults();
    for (k, x) in set {
        v.set(k, x.clone());
    }
    let problems = validate(&reg.get(id).unwrap().spec, &v, g.worksheet());
    if !problems.is_empty() {
        return Err(format!("{problems:?}"));
    }
    let before: Vec<String> = g.worksheet().columns.iter().map(|c| c.name().to_string()).collect();
    let s = reg.run(id, g, &v).map_err(|e| e.to_string())?;
    let after: Vec<String> = g.worksheet().columns.iter().map(|c| c.name().to_string()).collect();
    let added = after.into_iter().filter(|n| !before.contains(n)).collect();
    Ok((s, added))
}

fn run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
    run_full(g, id, set).map(|r| r.0)
}

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

fn charts(s: &Section) -> Vec<&ChartSpec> {
    s.blocks.iter().filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).collect()
}

fn tables(s: &Section) -> Vec<&Table> {
    s.blocks.iter().filter_map(|b| if let Block::Table(t) = b { Some(t) } else { None }).collect()
}

fn table<'a>(s: &'a Section, title_start: &str) -> &'a Table {
    tables(s).into_iter().find(|t| t.title.starts_with(title_start)).unwrap_or_else(|| panic!("no table {title_start}"))
}

const SYS: [&str; 14] = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "d", "d", "e", "e", "e"];
const TIM: [f64; 14] = [10.0, 40.0, 90.0, 25.0, 70.0, 5.0, 30.0, 55.0, 95.0, 60.0, 85.0, 15.0, 45.0, 75.0];
const END: [f64; 14] = [100.0, 100.0, 100.0, 80.0, 80.0, 100.0, 100.0, 100.0, 100.0, 90.0, 90.0, 80.0, 80.0, 80.0];

fn rep_grid() -> GridController {
    grid(vec![
        Column::text("sys", SYS.to_vec()),
        Column::numeric("t", TIM.to_vec()),
        Column::numeric("end", END.to_vec()),
    ])
}

fn rep_systems() -> Vec<System> {
    let id: Vec<String> = SYS.iter().map(|s| s.to_string()).collect();
    systems_from_rows(&id, &TIM, Some(&END)).unwrap()
}

#[test]
fn mcf_dialog_matches_library() {
    let mut g = rep_grid();
    let (s, added) = run_full(
        &mut g,
        "stat.rel.mcf",
        &[("sys", vars(&[0])), ("time", vars(&[1])), ("end", vars(&[2])), ("store", Value::Bool(true))],
    )
    .unwrap();
    let m = mean_cumulative_function(&rep_systems(), 0.95).unwrap();
    let t = table(&s, "Mean Cumulative Function");
    assert_eq!(t.rows.len(), m.len());
    assert!(matches!(t.rows[m.len() - 1][3], Cell::Num(x) if (x - m[m.len() - 1].mcf).abs() < 1e-12));
    assert_eq!(charts(&s).len(), 1);
    assert_eq!(added, vec!["MCF_TIME", "MCF", "MCF_LOWER", "MCF_UPPER"]);
}

#[test]
fn crow_dialog_matches_library() {
    let mut g = rep_grid();
    let s = run(&mut g, "stat.rel.crow", &[("sys", vars(&[0])), ("time", vars(&[1])), ("end", vars(&[2]))]).unwrap();
    let c = crow_amsaa(&rep_systems(), Termination::TimeTruncated, 0.95).unwrap();
    let t = table(&s, "Parameter Estimates");
    assert!(matches!(t.rows[0][1], Cell::Num(x) if (x - c.beta).abs() < 1e-12));
    assert!(text(&s).contains("Laplace"));
    assert_eq!(charts(&s).len(), 2);
    assert!(run(&mut g, "stat.rel.crow", &[("time", vars(&[1]))]).is_err());
}

#[test]
fn demo_dialog_solves_both_ways() {
    let mut g = grid(vec![Column::numeric("x", vec![1.0])]);
    let s = run(
        &mut g,
        "stat.rel.demo",
        &[
            ("shape", n(1.0)),
            ("req_time", n(100.0)),
            ("reliability", n(0.9)),
            ("confidence", n(90.0)),
            ("test_time", n(100.0)),
        ],
    )
    .unwrap();
    let t = table(&s, "Plan");
    assert!(matches!(t.rows[0][4], Cell::Int(22)));
    assert!(table(&s, "Probability of Passing").rows.len() == 5);
    let s2 = run(
        &mut g,
        "stat.rel.demo",
        &[
            ("solve", Value::Choice(1)),
            ("shape", n(2.0)),
            ("req_time", n(500.0)),
            ("reliability", n(0.95)),
            ("confidence", n(90.0)),
            ("failures", n(1.0)),
            ("n", n(40.0)),
        ],
    )
    .unwrap();
    let p = plan_test_time(2.0, 500.0, 0.95, 0.9, 1, 40).unwrap();
    assert!(matches!(table(&s2, "Plan").rows[0][5], Cell::Num(x) if (x - p.test_time).abs() < 1e-9));
    assert!(run(&mut g, "stat.rel.demo", &[]).is_err());
}

fn life_grid() -> GridController {
    let t: Vec<f64> = (1..=30).map(|i| 100.0 + 50.0 * (-(1.0 - (i as f64 - 0.5) / 30.0).ln()).powf(0.55)).collect();
    let c: Vec<f64> = (0..30).map(|i| if i % 7 == 6 { 1.0 } else { 0.0 }).collect();
    grid(vec![Column::numeric("life", t), Column::numeric("cens", c)])
}

#[test]
fn turnbull_dialog_runs_and_plots() {
    let mut g = life_grid();
    let s = run(&mut g, "stat.rel.turnbull", &[("time", vars(&[0])), ("censor", vars(&[1])), ("censor_value", n(1.0))])
        .unwrap();
    let t = table(&s, "Turnbull Intervals");
    assert!(!t.rows.is_empty());
    assert_eq!(charts(&s).len(), 1);
}

#[test]
fn extended_distribution_dialog_covers_all_four() {
    let mut g = life_grid();
    for d in 0..4 {
        let s = run(
            &mut g,
            "stat.rel.dist_ext",
            &[
                ("time", vars(&[0])),
                ("censor", vars(&[1])),
                ("censor_value", n(1.0)),
                ("dist", Value::Choice(d)),
                ("times", Value::Text("120 150".into())),
            ],
        )
        .unwrap();
        assert!(!table(&s, "Parameter Estimates").rows.is_empty(), "{d}");
        assert_eq!(table(&s, "Survival Probabilities").rows.len(), 2);
        assert_eq!(table(&s, "Percentiles").rows.len(), 5);
    }
    let s = run(&mut g, "stat.rel.dist_ext", &[("time", vars(&[0])), ("dist", Value::Choice(3))]).unwrap();
    let d = LifeData::from_times(&numeric_col(&g, 0), None, 0.0, None).unwrap();
    let gm = fit_gamma(&d).unwrap();
    assert!(text(&s).contains(&format!("{:.6}", gm.mean())));
}

fn numeric_col(g: &GridController, c: usize) -> Vec<f64> {
    match &g.worksheet().columns[c].data {
        openstats_core::column::ColumnData::Numeric(v) => v.clone(),
        _ => panic!("not numeric"),
    }
}

#[test]
fn eyring_dialog_matches_library() {
    let temps = [40.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0, 60.0, 80.0, 80.0, 80.0, 80.0];
    let t = [900.0, 1100.0, 1400.0, 1800.0, 300.0, 380.0, 450.0, 600.0, 110.0, 130.0, 170.0, 220.0];
    let mut g = grid(vec![Column::numeric("t", t.to_vec()), Column::numeric("temp", temps.to_vec())]);
    let s = run(
        &mut g,
        "stat.rel.eyring",
        &[("time", vars(&[0])), ("temp", vars(&[1])), ("dist", Value::Choice(1)), ("use_temp", n(30.0))],
    )
    .unwrap();
    let d = LifeData::from_times(&t, None, 0.0, None).unwrap();
    let e = fit_eyring(&d, &temps, Family::Lognormal).unwrap();
    assert!(matches!(table(&s, "Regression Table").rows[0][1], Cell::Num(x) if (x - e.alt.beta[0]).abs() < 1e-9));
    assert_eq!(table(&s, "Percentiles at 30").rows.len(), 3);
    assert_eq!(table(&s, "Acceleration Factors").rows.len(), 3);
    assert!(run(&mut g, "stat.rel.eyring", &[("time", vars(&[0]))]).is_err());
}

#[test]
fn censored_probability_plot_has_fit_line() {
    let mut g = life_grid();
    let s = run(
        &mut g,
        "stat.rel.cprob",
        &[("time", vars(&[0])), ("censor", vars(&[1])), ("censor_value", n(1.0)), ("dist", Value::Choice(1))],
    )
    .unwrap();
    let c = charts(&s);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].kind, ChartKind::Scatter);
    assert!(c[0].series.iter().any(|x| x.name == "Fit" && x.x.len() == 2));
    assert!(c[0].series[0].x.len() >= 5);
}
