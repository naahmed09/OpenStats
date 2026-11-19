//! Carried Data/Calc/time-series dialogs (Phase 5k).

use openstats_core::column::{Column, ColumnData};
use openstats_core::{StoredMatrix, Workbook};
use openstats_stats::report::{Block, ChartKind, Section, SessionReport};
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

fn charts(s: &Section) -> Vec<&openstats_stats::report::ChartSpec> {
    s.blocks.iter().filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).collect()
}

fn vars(v: &[usize]) -> Value {
    Value::Vars(v.to_vec())
}

fn col(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let i = ws.columns.iter().position(|c| c.name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[i].len()).map(|r| ws.columns[i].data.get(r).as_f64()).collect()
}

fn nan(x: f64) -> bool {
    x.is_nan()
}

#[test]
fn replace_missing_all_methods() {
    let x = vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 9.0, 10.0];
    let mut g = grid(vec![Column::numeric("x", x)]);
    run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(1))]).unwrap();
    let m = col(&g, "x_filled");
    assert!((m[1] - 23.0 / 4.0).abs() < 1e-12);
    let mut g = grid(vec![Column::numeric("x", vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 9.0, 10.0])]);
    run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(2))]).unwrap();
    assert!((col(&g, "x_filled")[1] - 6.0).abs() < 1e-12); // median of 1,3,9,10 = 6
    let mut g = grid(vec![Column::numeric("x", vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 9.0, 10.0])]);
    run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(3))]).unwrap();
    assert_eq!(col(&g, "x_filled"), vec![1.0, 1.0, 3.0, 3.0, 3.0, 9.0, 10.0]);
    let mut g = grid(vec![Column::numeric("x", vec![f64::NAN, 2.0, f64::NAN, 4.0])]);
    run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(4))]).unwrap();
    let y = col(&g, "x_filled");
    assert_eq!(&y[..3], &[2.0, 2.0, 4.0]);
    let mut g = grid(vec![Column::numeric("x", vec![1.0, f64::NAN, f64::NAN, 4.0])]);
    let s = run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(5))]).unwrap();
    assert_eq!(col(&g, "x_filled"), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(text(&s).contains("Replaced 2 missing values"));
    let mut g = grid(vec![Column::numeric("x", vec![1.0, f64::NAN])]);
    run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(0)), ("value", Value::Num(7.0))])
        .unwrap();
    assert_eq!(col(&g, "x_filled"), vec![1.0, 7.0]);
    assert!(run(&mut g, "data.replace_missing", &[("vars", vars(&[0])), ("how", Value::Choice(0))]).is_err());
}

#[test]
fn unpivot_keeps_ids_and_names() {
    let mut g = grid(vec![
        Column::text("id", vec!["a", "b"]),
        Column::numeric("x", vec![1.0, 2.0]),
        Column::numeric("y", vec![3.0, 4.0]),
    ]);
    run(&mut g, "data.unpivot", &[("vars", vars(&[1, 2])), ("ids", vars(&[0]))]).unwrap();
    assert_eq!(col(&g, "Value"), vec![1.0, 2.0, 3.0, 4.0]);
    let ws = g.worksheet();
    let names = ws.columns.iter().find(|c| c.name() == "Variable").unwrap();
    assert_eq!(names.display(0), "x");
    assert_eq!(names.display(3), "y");
    let ids = ws.columns.iter().rev().find(|c| c.name() == "id" || c.name() == "id_1").unwrap();
    assert_eq!((ids.display(0), ids.display(1), ids.display(2)), ("a".into(), "b".into(), "a".into()));
    assert!(run(&mut g, "data.unpivot", &[("vars", vars(&[1])), ("ids", vars(&[1]))]).is_err());
}

#[test]
fn lag_difference_moving_average() {
    let x = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
    let mut g = grid(vec![Column::numeric("x", x)]);
    run(&mut g, "stat.ts.lag", &[("vars", vars(&[0])), ("lag", Value::Num(2.0))]).unwrap();
    let l = col(&g, "x_lag2");
    assert!(nan(l[0]) && nan(l[1]));
    assert_eq!(&l[2..], &[1.0, 4.0, 9.0, 16.0]);
    run(&mut g, "stat.ts.diff", &[("vars", vars(&[0])), ("times", Value::Num(2.0))]).unwrap();
    let d = col(&g, "x_diff1");
    assert!(nan(d[0]) && nan(d[1]));
    assert_eq!(&d[2..], &[2.0, 2.0, 2.0, 2.0]); // second differences of squares
    run(&mut g, "stat.ts.ma", &[("vars", vars(&[0])), ("length", Value::Num(3.0))]).unwrap();
    let m = col(&g, "x_MA3");
    assert!(nan(m[0]) && nan(m[1]));
    assert!((m[2] - 14.0 / 3.0).abs() < 1e-12 && (m[5] - 77.0 / 3.0).abs() < 1e-12);
    let mut g = grid(vec![Column::numeric("x", vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0])]);
    run(&mut g, "stat.ts.ma", &[("vars", vars(&[0])), ("length", Value::Num(3.0)), ("centered", Value::Bool(true))])
        .unwrap();
    let c = col(&g, "x_MA3");
    assert!(nan(c[0]) && nan(c[5]));
    assert!((c[1] - 14.0 / 3.0).abs() < 1e-12 && (c[4] - 77.0 / 3.0).abs() < 1e-12);
    // Even centred length: 2 x 4 moving average.
    let mut g = grid(vec![Column::numeric("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])]);
    run(&mut g, "stat.ts.ma", &[("vars", vars(&[0])), ("length", Value::Num(4.0)), ("centered", Value::Bool(true))])
        .unwrap();
    let e = col(&g, "x_MA4");
    assert!(nan(e[0]) && nan(e[1]) && !nan(e[2]) && !nan(e[4]) && nan(e[5]));
    assert!((e[2] - 3.0).abs() < 1e-12 && (e[4] - 5.0).abs() < 1e-12);
}

#[test]
fn trend_analysis_recovers_models() {
    let lin: Vec<f64> = (1..=20).map(|t| 5.0 + 2.0 * t as f64).collect();
    let mut g = grid(vec![Column::numeric("y", lin)]);
    let s = run(
        &mut g,
        "stat.ts.trend",
        &[
            ("y", vars(&[0])),
            ("forecasts", Value::Num(3.0)),
            ("store_fits", Value::Bool(true)),
            ("store_fc", Value::Bool(true)),
        ],
    )
    .unwrap();
    assert!(text(&s).contains("5.000000 + 2.000000 × t"), "{}", text(&s));
    let f = col(&g, "y_fcst");
    assert!((f[0] - 47.0).abs() < 1e-9 && (f[2] - 51.0).abs() < 1e-9);
    assert!(col(&g, "y_fits").iter().enumerate().all(|(i, v)| (v - (7.0 + 2.0 * i as f64)).abs() < 1e-9));
    let ch = s.blocks.iter().any(|b| matches!(b, Block::Chart(c) if c.kind == ChartKind::Line && c.series.len() == 3));
    assert!(ch);
    let quad: Vec<f64> = (1..=12).map(|t| 1.0 + 0.5 * t as f64 + 0.25 * (t * t) as f64).collect();
    let mut g = grid(vec![Column::numeric("y", quad)]);
    let s = run(&mut g, "stat.ts.trend", &[("y", vars(&[0])), ("model", Value::Choice(1))]).unwrap();
    assert!(text(&s).contains("0.250000 × t²"), "{}", text(&s));
    let exp: Vec<f64> = (1..=10).map(|t| 3.0 * (0.1 * t as f64).exp()).collect();
    let mut g = grid(vec![Column::numeric("y", exp)]);
    let s = run(&mut g, "stat.ts.trend", &[("y", vars(&[0])), ("model", Value::Choice(2))]).unwrap();
    assert!(text(&s).contains("3.000000 × (e^(0.100000 × t))"), "{}", text(&s));
    let mut g = grid(vec![Column::numeric("y", vec![1.0, -1.0, 2.0, 3.0])]);
    assert!(run(&mut g, "stat.ts.trend", &[("y", vars(&[0])), ("model", Value::Choice(2))]).is_err());
}

#[test]
fn time_series_plot_uses_calendar_axis() {
    use openstats_core::datetime::MICROS_PER_DAY;
    let days: Vec<i64> = (0..10).map(|i| (19000 + i * 30) * MICROS_PER_DAY).collect();
    let mut g = grid(vec![
        Column::new("when", ColumnData::DateTime(days)),
        Column::numeric("v", (0..10).map(|i| i as f64).collect()),
    ]);
    let s = run(&mut g, "graph.timeseries", &[("vars", vars(&[1])), ("time", vars(&[0]))]).unwrap();
    let c = charts(&s)[0];
    assert_eq!(c.kind, ChartKind::TimeLine);
    assert!((c.series[0].x[0] - (19000.0 + 25569.0)).abs() < 1e-9);
    // Without a time column: row index, ordinary line chart.
    let s = run(&mut g, "graph.timeseries", &[("vars", vars(&[1]))]).unwrap();
    assert_eq!(charts(&s)[0].kind, ChartKind::Line);
    // The chart's x axis is a calendar (time) scale.
    let spec = run(&mut g, "graph.timeseries", &[("vars", vars(&[1])), ("time", vars(&[0]))]).unwrap();
    let scene =
        openstats_charts::plots::render_spec(charts(&spec)[0], &openstats_charts::plots::PlotOptions::default());
    assert!(matches!(scene.axes[0].x, openstats_charts::core::scale::Scale::Time { .. }));
}

fn put(g: &mut GridController, k: usize, rows: usize, cols: usize, d: &[f64]) {
    g.set_matrix(k, Some(StoredMatrix { rows, cols, data: d.to_vec() })).unwrap();
}

fn mat(g: &GridController, k: usize) -> StoredMatrix {
    g.worksheet().matrices.get(&k).unwrap().clone()
}

#[test]
fn matrix_operations() {
    let mut g = grid(vec![Column::numeric("x", vec![1.0])]);
    put(&mut g, 1, 2, 2, &[4.0, 2.0, 2.0, 3.0]);
    put(&mut g, 2, 2, 2, &[1.0, 0.0, 0.0, 2.0]);
    fn a<'a>(op: usize, set: &[(&'a str, Value)]) -> Vec<(&'a str, Value)> {
        let mut v = vec![("op", Value::Choice(op))];
        v.extend(set.iter().cloned());
        v
    }
    run(&mut g, "calc.matrix", &a(1, &[("a", Value::Num(1.0)), ("result", Value::Num(3.0))])).unwrap();
    let inv = mat(&g, 3);
    let want = [3.0 / 8.0, -0.25, -0.25, 0.5];
    assert!(inv.data.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12));
    run(&mut g, "calc.matrix", &a(2, &[("a", Value::Num(1.0)), ("result", Value::Num(1.0))])).unwrap();
    assert!((g.worksheet().constants[&1].as_f64() - 8.0).abs() < 1e-12);
    run(&mut g, "calc.matrix", &a(3, &[("a", Value::Num(1.0)), ("result", Value::Num(2.0))])).unwrap();
    assert!((g.worksheet().constants[&2].as_f64() - 7.0).abs() < 1e-12);
    run(&mut g, "calc.matrix", &a(4, &[("a", Value::Num(1.0)), ("b", Value::Num(2.0)), ("result", Value::Num(4.0))]))
        .unwrap();
    assert_eq!(mat(&g, 4).data, vec![4.0, 4.0, 2.0, 6.0]);
    run(&mut g, "calc.matrix", &a(5, &[("a", Value::Num(1.0)), ("b", Value::Num(2.0)), ("result", Value::Num(5.0))]))
        .unwrap();
    assert_eq!(mat(&g, 5).data, vec![5.0, 2.0, 2.0, 5.0]);
    run(&mut g, "calc.matrix", &a(6, &[("a", Value::Num(1.0)), ("b", Value::Num(2.0)), ("result", Value::Num(6.0))]))
        .unwrap();
    assert_eq!(mat(&g, 6).data, vec![3.0, 2.0, 2.0, 1.0]);
    run(&mut g, "calc.matrix", &a(7, &[("a", Value::Num(1.0)), ("result", Value::Num(7.0))])).unwrap();
    let l = mat(&g, 7);
    assert!((l.data[0] - 2.0).abs() < 1e-12 && (l.data[2] - 1.0).abs() < 1e-12 && l.data[1].abs() < 1e-12);
    run(&mut g, "calc.matrix", &a(8, &[("a", Value::Num(1.0)), ("result", Value::Num(8.0))])).unwrap();
    let mut ev = mat(&g, 8).data;
    ev.sort_by(|a, b| a.total_cmp(b));
    let (t, d) = (7.0f64, 8.0f64);
    let r = (t * t / 4.0 - d).sqrt();
    assert!((ev[0] - (t / 2.0 - r)).abs() < 1e-10 && (ev[1] - (t / 2.0 + r)).abs() < 1e-10);
    assert_eq!(mat(&g, 9).rows, 2);
    // Transpose of a non-square matrix, and undo.
    put(&mut g, 10, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    run(&mut g, "calc.matrix", &a(0, &[("a", Value::Num(10.0)), ("result", Value::Num(11.0))])).unwrap();
    let t = mat(&g, 11);
    assert_eq!((t.rows, t.cols), (3, 2));
    assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    g.undo().unwrap();
    assert!(!g.worksheet().matrices.contains_key(&11));
    // Errors: singular, non-square, missing, dimension mismatch.
    put(&mut g, 12, 2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(run(&mut g, "calc.matrix", &a(1, &[("a", Value::Num(12.0)), ("result", Value::Num(13.0))])).is_err());
    assert!(run(&mut g, "calc.matrix", &a(1, &[("a", Value::Num(10.0)), ("result", Value::Num(13.0))])).is_err());
    assert!(run(&mut g, "calc.matrix", &a(0, &[("a", Value::Num(50.0)), ("result", Value::Num(13.0))])).is_err());
    assert!(run(
        &mut g,
        "calc.matrix",
        &a(4, &[("a", Value::Num(10.0)), ("b", Value::Num(10.0)), ("result", Value::Num(13.0))])
    )
    .is_err());
}

#[test]
fn columns_to_matrix_and_back() {
    let mut g = grid(vec![Column::numeric("p", vec![1.0, 2.0, 3.0]), Column::numeric("q", vec![4.0, 5.0, 6.0])]);
    run(&mut g, "calc.matrix", &[("op", Value::Choice(9)), ("cols", vars(&[0, 1])), ("result", Value::Num(1.0))])
        .unwrap();
    let m = mat(&g, 1);
    assert_eq!((m.rows, m.cols), (3, 2));
    assert_eq!(m.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    run(&mut g, "calc.matrix", &[("op", Value::Choice(9)), ("a", Value::Num(1.0))]).unwrap();
    assert_eq!(col(&g, "M1_1"), vec![1.0, 2.0, 3.0]);
    assert_eq!(col(&g, "M1_2"), vec![4.0, 5.0, 6.0]);
}

#[test]
fn project_properties_are_stored() {
    let mut g = grid(vec![Column::numeric("x", vec![1.0])]);
    run(
        &mut g,
        "file.properties",
        &[
            ("title", Value::Text("Study 7".into())),
            ("author", Value::Text("N. Ahmed".into())),
            ("description", Value::Text("Pilot run".into())),
        ],
    )
    .unwrap();
    assert_eq!(g.workbook().title, "Study 7");
    assert_eq!(g.workbook().metadata.get("author").map(String::as_str), Some("N. Ahmed"));
    assert_eq!(g.workbook().metadata.get("description").map(String::as_str), Some("Pilot run"));
    run(&mut g, "file.properties", &[("title", Value::Text(String::new()))]).unwrap();
    assert_eq!(g.workbook().title, "Study 7");
    assert!(!g.workbook().metadata.contains_key("author"));
}

#[test]
fn json_import_round_trips() {
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("x", vec![1.5, 2.5])).unwrap();
    wb.worksheets[0].add_column(Column::text("t", vec!["a", "b"])).unwrap();
    let bytes = openstats_core::io::json::workbook_to_json(&wb, true).into_bytes();
    let back = openstats_core::io::load_workbook(Some("a.json"), &bytes).unwrap();
    assert_eq!(back.worksheets[0].columns[0].data, wb.worksheets[0].columns[0].data);
    assert_eq!(back.worksheets[0].columns[1].display(1), "b");
}
