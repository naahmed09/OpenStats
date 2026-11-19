//! Phase 5e dialogs: Data and Calc menus run through the registry.

use openstats_core::column::{Column, ColumnData};
use openstats_core::{Workbook, Worksheet};
use openstats_stats::report::{Block, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

fn sheet(cols: Vec<Column>) -> GridController {
    let mut wb = Workbook::new();
    for c in cols {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn n(name: &str, v: &[f64]) -> Column {
    Column::numeric(name, v.to_vec())
}
fn t(name: &str, v: &[&str]) -> Column {
    Column::text(name, v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
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
fn on() -> Value {
    Value::Bool(true)
}
fn column(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let c = (0..ws.columns.len()).find(|&c| ws.columns[c].name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[c].len()).map(|i| ws.columns[c].data.get(i).as_f64()).collect()
}

fn txt(s: &str) -> Value {
    Value::Text(s.into())
}
fn rows(g: &GridController, i: usize) -> usize {
    g.workbook().worksheets[i].n_rows()
}
fn texts(g: &GridController, name: &str) -> Vec<String> {
    let ws = g.worksheet();
    let c = (0..ws.columns.len()).find(|&c| ws.columns[c].name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[c].len()).map(|i| ws.columns[c].display(i)).collect()
}

#[test]
fn stack_unstack_transpose_round_trip() {
    let mut g = sheet(vec![n("pa", &[1.0, 2.0, 3.0]), n("pb", &[4.0, 5.0, 6.0])]);
    run(&mut g, "data.stack", &[("vars", vars(&[0, 1])), ("name", txt("allv"))]);
    assert_eq!(column(&g, "allv"), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(texts(&g, "Subscripts"), vec!["pa", "pa", "pa", "pb", "pb", "pb"]);
    let before = g.worksheet().columns.len();
    run(&mut g, "data.unstack", &[("vars", vars(&[2])), ("by", vars(&[3]))]);
    let ws = g.worksheet();
    assert_eq!(ws.columns.len(), before + 2);
    assert_eq!(column(&g, "allv_pa"), vec![1.0, 2.0, 3.0]);
    assert_eq!(column(&g, "allv_pb"), vec![4.0, 5.0, 6.0]);
    let mut h = sheet(vec![n("ta", &[1.0, 2.0]), n("tb", &[3.0, 4.0]), n("tc", &[5.0, 6.0])]);
    run(&mut h, "data.transpose", &[("vars", vars(&[0, 1, 2]))]);
    let ws = h.worksheet();
    assert_eq!(ws.columns.len(), 3 + 3);
    assert_eq!(texts(&h, "Labels"), vec!["ta", "tb", "tc"]);
    assert!(run_err(&mut h, "data.transpose", &[]));
}

fn run_err(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> bool {
    try_run(g, id, set).is_err()
}

#[test]
fn copy_and_subset_select_rows() {
    let mut g = sheet(vec![n("xv", &[1.0, 2.0, 3.0, 4.0, 5.0]), t("grp", &["a", "b", "a", "b", "a"])]);
    run(&mut g, "data.copy", &[("vars", vars(&[0])), ("select", ch(1)), ("from_row", num(2.0)), ("to_row", num(4.0))]);
    assert_eq!(column(&g, "xv_1"), vec![2.0, 3.0, 4.0]);
    run(
        &mut g,
        "data.copy",
        &[
            ("vars", vars(&[0])),
            ("select", ch(3)),
            ("sel_col", vars(&[0])),
            ("lo", num(2.0)),
            ("hi", num(3.0)),
            ("exclude", on()),
        ],
    );
    assert_eq!(column(&g, "xv_2"), vec![1.0, 4.0, 5.0]);
    let sheets = g.workbook().worksheets.len();
    run(
        &mut g,
        "data.subset",
        &[("select", ch(2)), ("sel_col", vars(&[1])), ("sel_values", txt("a")), ("name", txt("only a"))],
    );
    assert_eq!(g.workbook().worksheets.len(), sheets + 1);
    assert_eq!(rows(&g, sheets), 3);
    assert_eq!(g.workbook().worksheets[sheets].name, "only a");
    g.undo().unwrap();
    assert_eq!(g.workbook().worksheets.len(), sheets);
    assert!(run_err(
        &mut g,
        "data.copy",
        &[("vars", vars(&[0])), ("select", ch(1)), ("from_row", num(4.0)), ("to_row", num(2.0))]
    ));
    assert!(run_err(&mut g, "data.copy", &[("vars", vars(&[0])), ("select", ch(2))]));
}

#[test]
fn split_and_merge_worksheets() {
    let mut g = sheet(vec![n("xv", &[1.0, 2.0, 3.0, 4.0]), t("grp", &["a", "b", "a", "b"])]);
    run(&mut g, "data.split", &[("by", vars(&[1]))]);
    assert_eq!(g.workbook().worksheets.len(), 3);
    assert_eq!(rows(&g, 1), 2);
    assert_eq!(rows(&g, 2), 2);

    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(t("key", &["k1", "k2", "k3"])).unwrap();
    wb.worksheets[0].add_column(n("lv", &[1.0, 2.0, 3.0])).unwrap();
    let mut other = Worksheet::new("Other");
    other.add_column(t("key", &["k2", "k3", "k4"])).unwrap();
    other.add_column(n("rv", &[20.0, 30.0, 40.0])).unwrap();
    wb.worksheets.push(other);
    let mut m = GridController::new(wb, 800.0, 400.0);
    run(&mut m, "data.merge", &[("other", txt("other")), ("name", txt("side"))]);
    assert_eq!(rows(&m, 2), 3);
    run(
        &mut m,
        "data.merge",
        &[
            ("other", txt("Other")),
            ("how", ch(1)),
            ("keys", vars(&[0])),
            ("other_keys", txt("1")),
            ("name", txt("inner")),
        ],
    );
    assert_eq!(rows(&m, 3), 2);
    run(
        &mut m,
        "data.merge",
        &[
            ("other", txt("Other")),
            ("how", ch(3)),
            ("keys", vars(&[0])),
            ("other_keys", txt("1")),
            ("name", txt("outer")),
        ],
    );
    assert_eq!(rows(&m, 4), 4);
    assert!(run_err(&mut m, "data.merge", &[("other", txt("nope"))]));
    assert!(run_err(
        &mut m,
        "data.merge",
        &[("other", txt("Other")), ("how", ch(1)), ("keys", vars(&[0])), ("other_keys", txt("0"))]
    ));
}

#[test]
fn code_rank_concatenate_and_types() {
    let mut g = sheet(vec![
        n("sc", &[1.0, 3.0, 7.0, f64::NAN, 12.0]),
        n("rk", &[10.0, 20.0, 20.0, 30.0, 5.0]),
        t("w1", &["a", "b", "c", "d", "e"]),
        t("nums", &["1", "2.5", "x", "4", "5"]),
    ]);
    run(&mut g, "data.code", &[("vars", vars(&[0])), ("rules", txt("1:5=1; 6:10=2; *=0")), ("unmatched", ch(1))]);
    assert_eq!(
        column(&g, "sc_coded").iter().map(|x| if x.is_nan() { -1.0 } else { *x }).collect::<Vec<_>>(),
        vec![1.0, 1.0, 2.0, 0.0, -1.0]
    );
    run(&mut g, "data.code", &[("vars", vars(&[2])), ("rules", txt("a=first; b=second")), ("to", ch(1))]);
    assert_eq!(texts(&g, "w1_coded"), vec!["first", "second", "c", "d", "e"]);
    assert!(run_err(&mut g, "data.code", &[("vars", vars(&[0])), ("rules", txt("nonsense"))]));
    assert!(run_err(&mut g, "data.code", &[("vars", vars(&[0])), ("rules", txt(""))]));
    run(&mut g, "data.rank", &[("vars", vars(&[1]))]);
    assert_eq!(column(&g, "Rank_rk"), vec![2.0, 3.5, 3.5, 5.0, 1.0]);
    run(&mut g, "data.rank", &[("vars", vars(&[1])), ("ties", ch(1)), ("order", ch(1))]);
    assert_eq!(column(&g, "Rank_rk_1"), vec![4.0, 2.0, 2.0, 1.0, 5.0]);
    run(&mut g, "data.concatenate", &[("vars", vars(&[2, 3])), ("sep", txt("-")), ("name", txt("joined"))]);
    assert_eq!(texts(&g, "joined"), vec!["a-1", "b-2.5", "c-x", "d-4", "e-5"]);
    run(&mut g, "data.change_type", &[("vars", vars(&[3]))]);
    let v = column(&g, "nums_numeric");
    assert_eq!(v[0], 1.0);
    assert!(v[2].is_nan());
    assert!(run_err(&mut g, "data.change_type", &[("vars", vars(&[3])), ("strict", on())]));
}

#[test]
fn extract_text_and_display() {
    let d = ColumnData::DateTime(vec![1_710_460_800_000_000, 1_710_460_800_000_000 + 86_400_000_000 * 365]);
    let mut dt = Column::new("when", d);
    dt.props.name = "when".into();
    let mut g = sheet(vec![dt, t("tx", &["  Hello ", "world"])]);
    run(&mut g, "data.extract", &[("vars", vars(&[0])), ("part", ch(0))]);
    assert_eq!(column(&g, "when_Year"), vec![2024.0, 2025.0]);
    run(&mut g, "data.extract", &[("vars", vars(&[0])), ("part", ch(2))]);
    assert_eq!(column(&g, "when_Month"), vec![3.0, 3.0]);
    run(&mut g, "data.extract", &[("vars", vars(&[0])), ("part", ch(11))]);
    assert_eq!(texts(&g, "when_Weekday")[0], "Friday");
    run(&mut g, "data.text", &[("vars", vars(&[1])), ("op", ch(4))]);
    assert_eq!(texts(&g, "tx_text"), vec!["  HELLO ", "WORLD"]);
    run(&mut g, "data.text", &[("vars", vars(&[1])), ("op", ch(7)), ("from", txt("o")), ("to", txt("0"))]);
    assert_eq!(texts(&g, "tx_text_1")[1], "w0rld");
    let s = run(&mut g, "data.display", &[("vars", vars(&[0, 1]))]);
    assert!(s.blocks.iter().any(|b| matches!(b, Block::Table(_))));
}

#[test]
fn random_data_and_probability_distributions() {
    let mut g = sheet(vec![n("xs", &[0.0, 1.96, f64::NAN])]);
    run(
        &mut g,
        "calc.random",
        &[("n", num(400.0)), ("params", txt("10 2")), ("store", txt("rnd")), ("seed", num(7.0))],
    );
    let r = column(&g, "rnd");
    assert_eq!(r.len(), 400);
    let mean = r.iter().sum::<f64>() / 400.0;
    assert!((mean - 10.0).abs() < 0.5, "{mean}");
    run(
        &mut g,
        "calc.random",
        &[("n", num(400.0)), ("params", txt("10 2")), ("store", txt("rnd2")), ("seed", num(7.0))],
    );
    assert_eq!(column(&g, "rnd2"), r);
    g.undo().unwrap();
    assert!(!g.worksheet().columns.iter().any(|c| c.name() == "rnd2"));
    assert!(run_err(&mut g, "calc.random", &[("n", num(5.0)), ("params", txt("a b"))]));
    run(&mut g, "calc.prob", &[("input", vars(&[0])), ("store", txt("cdfv"))]);
    let c = column(&g, "cdfv");
    assert!((c[0] - 0.5).abs() < 1e-12 && (c[1] - 0.9750021048517795).abs() < 1e-9, "{c:?}");
    run(&mut g, "calc.prob", &[("mode", ch(3)), ("constant", num(0.975)), ("store", txt("inv"))]);
    assert!((column(&g, "inv")[0] - 1.959963984540054).abs() < 1e-8);
    run(
        &mut g,
        "calc.prob",
        &[("mode", ch(0)), ("dist", ch(18)), ("params", txt("3")), ("constant", num(2.0)), ("store", txt("pm"))],
    );
    assert!((column(&g, "pm")[0] - 0.22404180765538775).abs() < 1e-12);
    assert!(run_err(&mut g, "calc.prob", &[("dist", ch(0))]));
}

#[test]
fn statistics_patterned_data_and_indicators() {
    let mut g = sheet(vec![
        n("sa", &[1.0, 2.0, 3.0, 4.0]),
        n("sb", &[10.0, f64::NAN, 30.0, 40.0]),
        t("lv", &["x", "y", "x", "y"]),
    ]);
    let s = run(&mut g, "calc.colstats", &[("vars", vars(&[0, 1]))]);
    let text = {
        let mut r = SessionReport::new("t");
        r.push(s.clone());
        r.render_text()
    };
    assert!(text.contains("sa") && text.contains("Median"), "{text}");
    run(&mut g, "calc.rowstats", &[("vars", vars(&[0, 1])), ("stat", ch(1)), ("store", txt("rmean"))]);
    assert_eq!(column(&g, "rmean"), vec![5.5, 2.0, 16.5, 22.0]);
    run(&mut g, "calc.rowstats", &[("vars", vars(&[0, 1])), ("stat", ch(7)), ("store", txt("rmiss"))]);
    assert_eq!(column(&g, "rmiss"), vec![0.0, 1.0, 0.0, 0.0]);
    run(
        &mut g,
        "calc.patterned",
        &[
            ("from", num(1.0)),
            ("to", num(5.0)),
            ("step", num(2.0)),
            ("each", num(2.0)),
            ("list", num(2.0)),
            ("store", txt("pat")),
        ],
    );
    assert_eq!(column(&g, "pat"), vec![1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0]);
    assert!(run_err(&mut g, "calc.patterned", &[("step", num(0.0))]));
    run(&mut g, "calc.indicator", &[("vars", vars(&[2]))]);
    assert_eq!(column(&g, "lv_x"), vec![1.0, 0.0, 1.0, 0.0]);
    assert_eq!(column(&g, "lv_y"), vec![0.0, 1.0, 0.0, 1.0]);
    let _ = Section::new("x");
}
