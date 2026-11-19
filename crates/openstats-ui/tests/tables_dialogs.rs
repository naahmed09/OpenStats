//! Tables, EDA, bootstrap, simulation and sampling dialogs (Phase 5l).

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::report::{Block, ChartKind, ChartSpec, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

fn grid(cols: Vec<Column>) -> GridController {
    let mut wb = Workbook::new();
    for c in cols {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
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

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

fn charts(s: &Section) -> Vec<&ChartSpec> {
    s.blocks.iter().filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).collect()
}

fn tables(s: &Section) -> Vec<&openstats_stats::report::Table> {
    s.blocks.iter().filter_map(|b| if let Block::Table(t) = b { Some(t) } else { None }).collect()
}

fn vars(v: &[usize]) -> Value {
    Value::Vars(v.to_vec())
}

fn col(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let i = ws.columns.iter().position(|c| c.name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[i].len()).map(|r| ws.columns[i].data.get(r).as_f64()).collect()
}

fn colour() -> Vec<&'static str> {
    vec!["red", "blue", "red", "green", "blue", "red", "red", "blue", "green", "red"]
}

#[test]
fn registry_lists_the_new_dialogs() {
    let reg = Registry::standard();
    for id in [
        "stat.tables.tally",
        "stat.tables.cells",
        "quality.pareto",
        "stat.eda.letter",
        "stat.eda.polish",
        "stat.eda.resistant",
        "stat.eda.rootogram",
        "stat.eda.symmetry",
        "stat.resample.bootstrap",
        "calc.simulate",
        "calc.sample",
    ] {
        assert!(reg.get(id).is_some(), "{id}");
    }
}

#[test]
fn tally_counts_and_percents() {
    let mut g = grid(vec![Column::text("c", colour())]);
    let s = run(
        &mut g,
        "stat.tables.tally",
        &[
            ("vars", vars(&[0])),
            ("percents", Value::Bool(true)),
            ("cum_counts", Value::Bool(true)),
            ("cum_percents", Value::Bool(true)),
        ],
    )
    .unwrap();
    let t = text(&s);
    // blue 3, green 2, red 5 (sorted by label); cumulative 3, 5, 10.
    assert!(t.contains("blue") && t.contains("30.00") && t.contains("50.00") && t.contains("100.00"), "{t}");
    let table = tables(&s)[0];
    let cols = table.formatted_columns();
    assert_eq!(cols[1][..3], ["3", "2", "5"]);
    assert_eq!(cols[3][..3], ["3", "5", "10"]);
    assert!(run(&mut g, "stat.tables.tally", &[("vars", vars(&[0])), ("counts", Value::Bool(false))]).is_err());
    // Numbers tally in numeric order; missing values are reported.
    let mut g = grid(vec![Column::numeric("n", vec![10.0, 2.0, 2.0, f64::NAN, 10.0, 10.0])]);
    let s = run(&mut g, "stat.tables.tally", &[("vars", vars(&[0]))]).unwrap();
    let c = tables(&s)[0].formatted_columns();
    assert_eq!(c[0][..2], ["2", "10"]);
    assert!(text(&s).contains("1 missing"));
}

#[test]
fn descriptive_table_cells() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let g2: Vec<&str> = vec!["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"];
    let mut g = grid(vec![Column::text("c", colour()), Column::text("g", g2), Column::numeric("y", y)]);
    let s = run(
        &mut g,
        "stat.tables.cells",
        &[("rows", vars(&[0])), ("cols", vars(&[1])), ("assoc", vars(&[2])), ("stat_sd", Value::Bool(true))],
    )
    .unwrap();
    let t = tables(&s)[0];
    let cols = t.formatted_columns();
    // Columns: c, g, Count, y N, y Mean, y StDev. Row order: blue a/b/All, green..., red..., then All.
    assert_eq!(cols[0][0], "blue");
    // blue rows are 2, 5, 8 (1-based) -> y = 2, 5, 8; groups b, a... row 2 is b, row 5 a, row 8 b.
    let find = |c: &str, gv: &str| (0..cols[0].len()).find(|&i| cols[0][i] == c && cols[1][i] == gv).unwrap();
    let r = find("blue", "All");
    assert_eq!(cols[2][r], "3");
    assert_eq!(cols[4][r].parse::<f64>().unwrap(), 5.0);
    assert_eq!(cols[5][r].parse::<f64>().unwrap(), 3.0);
    let r = find("red", "a");
    // red rows 1,3,6,7,10 -> a: rows 1,3,7 (y 1,3,7) -> mean 11/3.
    assert_eq!(cols[2][r], "3");
    assert!((cols[4][r].parse::<f64>().unwrap() - 11.0 / 3.0).abs() < 1e-4);
    let last = find("All", "All");
    assert_eq!(cols[2][last], "10");
    assert_eq!(cols[4][last].parse::<f64>().unwrap(), 5.5);
    assert!(run(
        &mut g,
        "stat.tables.cells",
        &[
            ("rows", vars(&[0])),
            ("assoc", vars(&[2])),
            ("stat_n", Value::Bool(false)),
            ("stat_mean", Value::Bool(false))
        ]
    )
    .is_err());
}

#[test]
fn pareto_sorts_and_accumulates() {
    let mut g = grid(vec![Column::text("d", vec!["scratch", "dent", "scratch", "crack", "scratch", "dent"])]);
    let s = run(&mut g, "quality.pareto", &[("defects", vars(&[0]))]).unwrap();
    let c = tables(&s)[0].formatted_columns();
    assert_eq!(c[0], ["scratch", "dent", "crack"]);
    assert_eq!(c[3], ["50.00", "83.33", "100.00"]);
    let ch = charts(&s)[0];
    assert_eq!(ch.kind, ChartKind::Pareto);
    // The chart draws.
    let scene = openstats_charts::plots::render_spec(ch, &openstats_charts::plots::PlotOptions::default());
    assert!(!scene.items.is_empty());
    // Counts column aggregates repeated names.
    let mut g = grid(vec![Column::text("d", vec!["a", "b", "a"]), Column::numeric("n", vec![1.0, 5.0, 2.0])]);
    let s = run(&mut g, "quality.pareto", &[("defects", vars(&[0])), ("counts", vars(&[1]))]).unwrap();
    assert_eq!(tables(&s)[0].formatted_columns()[0], ["b", "a"]);
    let mut g = grid(vec![Column::text("d", vec!["a"]), Column::numeric("n", vec![-1.0])]);
    assert!(run(&mut g, "quality.pareto", &[("defects", vars(&[0])), ("counts", vars(&[1]))]).is_err());
}

#[test]
fn eda_dialogs() {
    let x: Vec<f64> = (1..=9).map(f64::from).collect();
    let mut g = grid(vec![Column::numeric("x", x.clone())]);
    let s = run(&mut g, "stat.eda.letter", &[("vars", vars(&[0]))]).unwrap();
    let t = text(&s);
    assert!(t.contains("N = 9") && t.contains("1.5"), "{t}");
    // Median polish on an additive table stores zero residuals.
    let (mut yv, mut rv, mut cv) = (Vec::new(), Vec::new(), Vec::new());
    for (i, r) in ["r1", "r2", "r3"].iter().enumerate() {
        for (j, c) in ["c1", "c2", "c3", "c4"].iter().enumerate() {
            yv.push(10.0 + 2.0 * i as f64 + 3.0 * j as f64);
            rv.push(*r);
            cv.push(*c);
        }
    }
    let mut g = grid(vec![Column::numeric("y", yv.clone()), Column::text("r", rv), Column::text("c", cv)]);
    run(
        &mut g,
        "stat.eda.polish",
        &[
            ("y", vars(&[0])),
            ("row", vars(&[1])),
            ("col", vars(&[2])),
            ("store_res", Value::Bool(true)),
            ("store_fit", Value::Bool(true)),
        ],
    )
    .unwrap();
    assert!(col(&g, "RESI").iter().all(|r| r.abs() < 1e-12));
    assert_eq!(col(&g, "FITS"), yv);
    // Missing cell is refused.
    let mut g = grid(vec![
        Column::numeric("y", vec![1.0, 2.0, 3.0]),
        Column::text("r", vec!["a", "a", "b"]),
        Column::text("c", vec!["x", "y", "x"]),
    ]);
    assert!(run(&mut g, "stat.eda.polish", &[("y", vars(&[0])), ("row", vars(&[1])), ("col", vars(&[2]))]).is_err());
    // Resistant line recovers an exact line and stores fits.
    let xs: Vec<f64> = (1..=12).map(f64::from).collect();
    let ys: Vec<f64> = xs.iter().map(|v| 1.5 - 0.5 * v).collect();
    let mut g = grid(vec![Column::numeric("y", ys.clone()), Column::numeric("x", xs)]);
    let s =
        run(&mut g, "stat.eda.resistant", &[("y", vars(&[0])), ("x", vars(&[1])), ("store_fit", Value::Bool(true))])
            .unwrap();
    assert!(text(&s).contains("1.500000 + -0.500000"), "{}", text(&s));
    assert_eq!(col(&g, "FITS").len(), 12);
    assert!(charts(&s)[0].series.iter().any(|s| s.name == "Fit"));
    // Rootogram and symmetry plots.
    let data: Vec<f64> = (0..40).map(|i| (f64::from(i) * 0.7).sin() * 3.0 + f64::from(i % 5)).collect();
    let mut g = grid(vec![Column::numeric("v", data)]);
    let s = run(&mut g, "stat.eda.rootogram", &[("x", vars(&[0])), ("bins", Value::Num(6.0))]).unwrap();
    assert_eq!(tables(&s)[0].formatted_columns()[2].len(), 6);
    assert_eq!(charts(&s)[0].reference_lines.len(), 3);
    let s = run(&mut g, "stat.eda.symmetry", &[("x", vars(&[0]))]).unwrap();
    assert_eq!(charts(&s)[0].series[0].x.len(), 20);
}

#[test]
fn bootstrap_dialog() {
    let x: Vec<f64> = (0..30).map(|i| 10.0 + f64::from(i % 7) + f64::from(i) * 0.1).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, v)| 2.0 * v + f64::from((i * 5 % 9) as u32)).collect();
    let mut g = grid(vec![Column::numeric("x", x.clone()), Column::numeric("y", y.clone())]);
    let set = [("x", vars(&[0])), ("resamples", Value::Num(500.0)), ("store", Value::Bool(true))];
    let s = run(&mut g, "stat.resample.bootstrap", &set).unwrap();
    let t = text(&s);
    assert!(t.contains("Percentile") && t.contains("BCa") && t.contains("Normal") && t.contains("Basic"), "{t}");
    let reps = col(&g, "BootReps");
    assert_eq!(reps.len(), 500);
    let mean = x.iter().sum::<f64>() / 30.0;
    let rm = reps.iter().sum::<f64>() / 500.0;
    assert!((rm - mean).abs() < 0.3);
    assert_eq!(charts(&s)[0].kind, ChartKind::Histogram);
    // Same seed, same replicates; correlation and two-sample statistics.
    let mut g2 = grid(vec![Column::numeric("x", x.clone()), Column::numeric("y", y.clone())]);
    run(&mut g2, "stat.resample.bootstrap", &set).unwrap();
    assert_eq!(col(&g2, "BootReps"), reps);
    let s = run(
        &mut g,
        "stat.resample.bootstrap",
        &[("stat", Value::Choice(7)), ("x", vars(&[0])), ("y", vars(&[1])), ("resamples", Value::Num(300.0))],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("Correlation of x and y"), "{t}");
    let s = run(
        &mut g,
        "stat.resample.bootstrap",
        &[("stat", Value::Choice(8)), ("x", vars(&[0])), ("y", vars(&[1])), ("resamples", Value::Num(300.0))],
    )
    .unwrap();
    assert!(text(&s).contains("Difference of means"));
    assert!(run(&mut g, "stat.resample.bootstrap", &[("stat", Value::Choice(7)), ("x", vars(&[0]))]).is_err());
    let mut tiny = grid(vec![Column::numeric("x", vec![1.0])]);
    assert!(run(&mut tiny, "stat.resample.bootstrap", &[("x", vars(&[0]))]).is_err());
}

#[test]
fn simulation_dialog() {
    let mut g = grid(vec![Column::numeric("keep", vec![1.0])]);
    let s = run(
        &mut g,
        "calc.simulate",
        &[
            ("inputs", Value::Text("X = uniform(0, 1); Y = normal(10, 2)".into())),
            ("expr", Value::Text("2 * X + Y".into())),
            ("n", Value::Num(20000.0)),
            ("upper", Value::Num(12.0)),
            ("store", Value::Bool(true)),
        ],
    )
    .unwrap();
    let out = col(&g, "Output");
    assert_eq!(out.len(), 20000);
    let m = out.iter().sum::<f64>() / 20000.0;
    assert!((m - 11.0).abs() < 0.08, "{m}");
    // Output = 2X + Y is X*2 + Y with the stored inputs.
    let (xs, ys) = (col(&g, "X"), col(&g, "Y"));
    assert!((0..100).all(|i| (out[i] - (2.0 * xs[i] + ys[i])).abs() < 1e-9));
    assert!(xs.iter().all(|v| (0.0..1.0).contains(v)));
    // P(2X+Y > 12) from a normal approximation is 1 - Phi(1/sqrt(13/3)) = 0.316 (variance 4 + 1/3).
    let t = text(&s);
    assert!(t.contains("Probability of exceeding limits") && t.contains("> 12"), "{t}");
    let p = out.iter().filter(|&&v| v > 12.0).count() as f64 / 20000.0;
    assert!((p - 0.316).abs() < 0.02, "{p}");
    // Reproducible with the seed; function names are not mistaken for inputs.
    let mut g2 = grid(vec![Column::numeric("keep", vec![1.0])]);
    run(
        &mut g2,
        "calc.simulate",
        &[
            ("inputs", Value::Text("X = normal(0, 1)".into())),
            ("expr", Value::Text("SQRT(ABS(X))".into())),
            ("n", Value::Num(500.0)),
            ("store", Value::Bool(true)),
        ],
    )
    .unwrap();
    let mut g3 = grid(vec![Column::numeric("keep", vec![1.0])]);
    run(
        &mut g3,
        "calc.simulate",
        &[
            ("inputs", Value::Text("X = normal(0, 1)".into())),
            ("expr", Value::Text("SQRT(ABS(X))".into())),
            ("n", Value::Num(500.0)),
            ("store", Value::Bool(true)),
        ],
    )
    .unwrap();
    assert_eq!(col(&g2, "Output"), col(&g3, "Output"));
    // Errors: bad syntax, unknown distribution, unknown name in the equation, duplicate input.
    for (inputs, expr) in [
        ("X normal(0,1)", "X"),
        ("X = normal(0, one)", "X"),
        ("X = nosuchdist(1)", "X"),
        ("X = normal(0,1)", "X + Q"),
        ("X = normal(0,1); x = normal(0,1)", "X"),
        ("", "X"),
    ] {
        assert!(
            run(
                &mut g,
                "calc.simulate",
                &[("inputs", Value::Text(inputs.into())), ("expr", Value::Text(expr.into())), ("n", Value::Num(50.0))]
            )
            .is_err(),
            "{inputs} / {expr}"
        );
    }
}

#[test]
fn sampling_dialog() {
    let a: Vec<f64> = (1..=20).map(f64::from).collect();
    let b: Vec<f64> = a.iter().map(|v| v * 10.0).collect();
    let mut g = grid(vec![Column::numeric("a", a), Column::numeric("b", b)]);
    run(&mut g, "calc.sample", &[("vars", vars(&[0, 1])), ("n", Value::Num(8.0))]).unwrap();
    let (sa, sb) = (col(&g, "a_sample"), col(&g, "b_sample"));
    assert_eq!(sa.len(), 8);
    assert!(sa.iter().zip(&sb).all(|(x, y)| (x * 10.0 - y).abs() < 1e-12), "rows move together");
    let mut u = sa.clone();
    u.sort_by(|p, q| p.total_cmp(q));
    u.dedup();
    assert_eq!(u.len(), 8, "no repeats without replacement");
    assert!(run(&mut g, "calc.sample", &[("vars", vars(&[0])), ("n", Value::Num(25.0))]).is_err());
    let mut g2 = grid(vec![Column::numeric("a", (1..=5).map(f64::from).collect())]);
    run(&mut g2, "calc.sample", &[("vars", vars(&[0])), ("n", Value::Num(50.0)), ("replace", Value::Bool(true))])
        .unwrap();
    assert_eq!(col(&g2, "a_sample").len(), 50);
    // Reproducible.
    let mut g3 = grid(vec![Column::numeric("a", (1..=5).map(f64::from).collect())]);
    run(&mut g3, "calc.sample", &[("vars", vars(&[0])), ("n", Value::Num(50.0)), ("replace", Value::Bool(true))])
        .unwrap();
    assert_eq!(col(&g2, "a_sample"), col(&g3, "a_sample"));
}
