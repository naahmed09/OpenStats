//! Quality completion dialogs (Phase 5m), compared with the library routines they wrap.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::acceptance::{find_attribute_plan, find_variables_plan, LotModel};
use openstats_stats::attrcap::{binomial_capability, poisson_capability};
use openstats_stats::report::{Block, Cell, ChartKind, ChartSpec, Section, SessionReport, Table};
use openstats_stats::runchart::{run_chart, Center};
use openstats_stats::spc::capability::{capability_individuals, CapabilityOptions, Specs};
use openstats_stats::spc::transform::{box_cox_interval, BoxCox, LambdaChoice};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

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

fn num(t: &Table, row: usize, col: usize) -> f64 {
    match &t.rows[row][col] {
        Cell::Num(x) => *x,
        Cell::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

fn vars(v: &[usize]) -> Value {
    Value::Vars(v.to_vec())
}

fn n(x: f64) -> Value {
    Value::Num(x)
}

fn txt(s: &str) -> Value {
    Value::Text(s.into())
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

fn lognormalish(len: usize) -> Vec<f64> {
    // deterministic skewed positive data
    (0..len)
        .map(|i| ((i as f64 * 0.37).sin() * 0.6 + (i as f64 * 0.11).cos() * 0.4 + 1.0).exp() + (i % 7) as f64 * 0.01)
        .collect()
}

#[test]
fn registry_lists_the_quality_dialogs() {
    let reg = Registry::standard();
    for id in [
        "quality.accept.attr_create",
        "quality.accept.attr_compare",
        "quality.accept.double",
        "quality.accept.attr_decision",
        "quality.accept.var_create",
        "quality.accept.var_compare",
        "quality.accept.var_decision",
        "quality.multivari",
        "quality.runchart",
        "quality.gage_run",
        "quality.capability_binomial",
        "quality.capability_poisson",
        "quality.capability_multi",
        "quality.johnson",
        "quality.boxcox",
    ] {
        assert!(reg.get(id).is_some(), "{id}");
    }
}

#[test]
fn attribute_plan_dialog_equals_library() {
    let mut g = grid(vec![]);
    let s = run(&mut g, "quality.accept.attr_create", &[("aql", n(0.01)), ("rql", n(0.08))]).unwrap();
    let want = find_attribute_plan(0.01, 0.08, 0.05, 0.10, LotModel::Binomial, None).unwrap();
    let t = table(&s, "Sampling Plan");
    assert_eq!(num(t, 0, 1) as u64, want.plan.n);
    assert_eq!(num(t, 0, 2) as u64, want.plan.c);
    assert_eq!((want.plan.n, want.plan.c), (65, 2));
    assert_eq!(charts(&s).len(), 2, "OC and AOQ without a lot size");
    // with a lot size the ATI curve appears and the hypergeometric plan is found
    let s2 = run(
        &mut g,
        "quality.accept.attr_create",
        &[("aql", n(0.01)), ("rql", n(0.08)), ("lot", n(1000.0)), ("model", Value::Choice(2))],
    )
    .unwrap();
    assert_eq!(charts(&s2).len(), 3);
    assert_eq!(num(table(&s2, "Sampling Plan"), 0, 1) as u64, 64);
    assert!(text(&s2).contains("Sample 64 items"));
    // errors
    assert!(run(&mut g, "quality.accept.attr_create", &[("aql", n(0.1)), ("rql", n(0.05))]).is_err());
    assert!(run(&mut g, "quality.accept.attr_create", &[("model", Value::Choice(2))]).is_err());
}

#[test]
fn compare_double_and_decision_dialogs() {
    let mut g = grid(vec![]);
    let s = run(
        &mut g,
        "quality.accept.attr_compare",
        &[("ns", txt("50 80")), ("cs", txt("1 2")), ("aql", n(0.01)), ("rql", n(0.08))],
    )
    .unwrap();
    assert_eq!(charts(&s)[0].series.len(), 2);
    let t = table(&s, "Accept / Reject");
    assert_eq!(t.rows.len(), 4);
    assert!(run(&mut g, "quality.accept.attr_compare", &[("ns", txt("50 80")), ("cs", txt("1"))]).is_err());
    assert!(run(&mut g, "quality.accept.attr_compare", &[("ns", txt("50.5")), ("cs", txt("1"))]).is_err());

    let d = run(&mut g, "quality.accept.double", &[("lot", n(2000.0)), ("aql", n(0.01))]).unwrap();
    assert_eq!(charts(&d).len(), 4);
    let ops = table(&d, "Operating Characteristics");
    close(num(ops, 0, 3), 1.0, 1e-12);
    let level = table(&d, "Accept / Reject");
    let want = openstats_stats::acceptance::DoublePlan { n1: 50, c1: 1, r1: 4, n2: 100, c2: 4 }
        .outcome(LotModel::Binomial, 0.01)
        .unwrap();
    close(num(level, 0, 2), want.pa, 1e-12);
    assert!(run(&mut g, "quality.accept.double", &[("c1", n(5.0)), ("r1", n(3.0))]).is_err());

    let acc =
        run(&mut g, "quality.accept.attr_decision", &[("n", n(65.0)), ("c", n(2.0)), ("defectives", n(2.0))]).unwrap();
    assert!(text(&acc).contains("Accept lot"));
    let rej =
        run(&mut g, "quality.accept.attr_decision", &[("n", n(65.0)), ("c", n(2.0)), ("defectives", n(3.0))]).unwrap();
    assert!(text(&rej).contains("Reject lot"));
    assert!(
        run(&mut g, "quality.accept.attr_decision", &[("n", n(5.0)), ("c", n(2.0)), ("defectives", n(9.0))]).is_err()
    );
}

#[test]
fn variables_sampling_dialogs() {
    let mut g = grid(vec![Column::numeric("x", vec![10.0, 10.2, 9.9, 10.1, 9.8, 10.05, 9.95, 10.0])]);
    for (choice, known) in [(1usize, true), (0, false)] {
        let s = run(
            &mut g,
            "quality.accept.var_create",
            &[("sigma", Value::Choice(choice)), ("aql", n(0.01)), ("rql", n(0.08))],
        )
        .unwrap();
        let want = find_variables_plan(0.01, 0.08, 0.05, 0.10, known).unwrap();
        let t = table(&s, "Sampling Plan");
        assert_eq!(num(t, 0, 1) as u64, want.plan.n);
        close(num(t, 0, 2), want.plan.k, 1e-12);
        assert_eq!(charts(&s).len(), 1);
    }
    let c =
        run(&mut g, "quality.accept.var_compare", &[("ns", txt("10 20")), ("ks", txt("1.8 1.9")), ("aql", n(0.01))])
            .unwrap();
    assert_eq!(charts(&c)[0].series.len(), 2);
    assert!(run(&mut g, "quality.accept.var_compare", &[("ns", txt("1")), ("ks", txt("1.8"))]).is_err());
    let d =
        run(&mut g, "quality.accept.var_decision", &[("var", vars(&[0])), ("limit", n(11.0)), ("k", n(1.8))]).unwrap();
    assert!(text(&d).contains("Accept lot"));
    let d2 =
        run(&mut g, "quality.accept.var_decision", &[("var", vars(&[0])), ("limit", n(10.2)), ("k", n(1.8))]).unwrap();
    assert!(text(&d2).contains("Reject lot"));
    let lower = run(
        &mut g,
        "quality.accept.var_decision",
        &[("var", vars(&[0])), ("limit", n(9.0)), ("k", n(1.8)), ("side", Value::Choice(1))],
    )
    .unwrap();
    assert!(text(&lower).contains("Accept lot"));
    // known sigma needs a value
    assert!(run(
        &mut g,
        "quality.accept.var_decision",
        &[("var", vars(&[0])), ("limit", n(11.0)), ("sigma", Value::Choice(1))]
    )
    .is_err());
}

#[test]
fn multivari_and_gage_run_dialogs() {
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = Column::text("A", vec!["a", "a", "a", "a", "b", "b", "b", "b"]);
    let b = Column::text("B", vec!["x", "x", "y", "y", "x", "x", "y", "y"]);
    let mut g = grid(vec![Column::numeric("y", y), a, b]);
    let s = run(&mut g, "quality.multivari", &[("y", vars(&[0])), ("factors", vars(&[1, 2]))]).unwrap();
    let cells = table(&s, "Cell Statistics");
    assert_eq!(cells.rows.len(), 4);
    close(num(cells, 0, 4), 1.5, 1e-12);
    assert_eq!(charts(&s)[0].series.iter().filter(|x| x.name == "Fit").count(), 1);
    let lm = table(&s, "Means of Factor Levels");
    close(num(lm, 1, 3), 6.5, 1e-12);
    assert!(run(&mut g, "quality.multivari", &[("y", vars(&[0])), ("factors", vars(&[0]))]).is_err());
    assert!(run(&mut g, "quality.multivari", &[("y", vars(&[0])), ("factors", vars(&[1, 2, 1, 2, 1]))]).is_err());

    // gage run chart: 3 parts x 2 operators x 2 replicates
    let m = vec![10.0, 10.2, 20.0, 20.4, 30.0, 30.2, 10.4, 10.6, 20.2, 20.0, 30.4, 30.6];
    let part = Column::text("part", vec!["1", "1", "2", "2", "3", "3", "1", "1", "2", "2", "3", "3"]);
    let op = Column::text("op", vec!["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"]);
    let mut g2 = grid(vec![Column::numeric("m", m), part, op]);
    let r = run(
        &mut g2,
        "quality.gage_run",
        &[("measurement", vars(&[0])), ("part", vars(&[1])), ("operator", vars(&[2]))],
    )
    .unwrap();
    let t = table(&r, "Mean Measurement");
    assert_eq!(t.rows.len(), 3);
    close(num(t, 0, 1), 10.1, 1e-12);
    close(num(t, 0, 2), 10.5, 1e-12);
    close(num(t, 2, 2), 30.5, 1e-12);
    let ch = charts(&r)[0];
    assert_eq!(ch.series.len(), 4);
    assert!(ch.series.iter().any(|s| s.name == "Fit: B"));
    assert_eq!(ch.kind, ChartKind::Scatter);
    // without operators there is one group
    let r1 = run(&mut g2, "quality.gage_run", &[("measurement", vars(&[0])), ("part", vars(&[1]))]).unwrap();
    assert_eq!(charts(&r1)[0].series.len(), 2);
}

#[test]
fn run_chart_dialog_equals_library() {
    let x: Vec<f64> = (0..30).map(|i| ((i * 7 % 11) as f64) + (i as f64 * 0.1)).collect();
    let mut g = grid(vec![Column::numeric("x", x.clone())]);
    let s = run(&mut g, "quality.runchart", &[("var", vars(&[0]))]).unwrap();
    let want = run_chart(&x, Center::Median).unwrap();
    let t = table(&s, "Runs About");
    assert_eq!(num(t, 0, 0) as usize, want.runs_about);
    close(num(t, 0, 5), want.p_clustering, 1e-12);
    let ud = table(&s, "Runs Up or Down");
    assert_eq!(num(ud, 0, 0) as usize, want.runs_updown);
    assert_eq!(charts(&s)[0].reference_lines.len(), 1);
    // subgroup medians
    let s2 = run(&mut g, "quality.runchart", &[("var", vars(&[0])), ("size", n(5.0))]).unwrap();
    assert_eq!(charts(&s2)[0].series[0].y.len(), 6);
    // specified centre
    let s3 =
        run(&mut g, "quality.runchart", &[("var", vars(&[0])), ("center", Value::Choice(2)), ("center_value", n(5.0))])
            .unwrap();
    close(charts(&s3)[0].reference_lines[0].value, 5.0, 1e-15);
    assert!(run(&mut g, "quality.runchart", &[("var", vars(&[0])), ("center", Value::Choice(2))]).is_err());
    let mut tiny = grid(vec![Column::numeric("x", vec![1.0, 2.0])]);
    assert!(run(&mut tiny, "quality.runchart", &[("var", vars(&[0]))]).is_err());
}

#[test]
fn attribute_capability_dialogs_equal_library() {
    let d = vec![2.0, 3.0, 1.0, 4.0, 0.0, 3.0];
    let sizes = vec![100.0, 120.0, 100.0, 90.0, 110.0, 100.0];
    let mut g = grid(vec![Column::numeric("d", d.clone()), Column::numeric("n", sizes.clone())]);
    let s = run(
        &mut g,
        "quality.capability_binomial",
        &[("defectives", vars(&[0])), ("size_col", vars(&[1])), ("target", n(2.0))],
    )
    .unwrap();
    let want = binomial_capability(&d, &sizes, 0.95, Some(0.02)).unwrap();
    let t = table(&s, "Summary Statistics");
    close(num(t, 0, 1), 100.0 * want.p_bar, 1e-12);
    close(num(t, 0, 2), 100.0 * want.p_interval.0, 1e-12);
    close(num(t, 2, 1), want.z, 1e-12);
    let chi = table(&s, "Test of a Common");
    close(num(chi, 0, 0), want.chi_square.unwrap().0, 1e-12);
    assert_eq!(charts(&s).len(), 2);
    // constant sample size
    let s2 = run(&mut g, "quality.capability_binomial", &[("defectives", vars(&[0])), ("size", n(100.0))]).unwrap();
    let w2 = binomial_capability(&d, &[100.0; 6], 0.95, None).unwrap();
    close(num(table(&s2, "Summary Statistics"), 0, 1), 100.0 * w2.p_bar, 1e-12);
    assert!(run(&mut g, "quality.capability_binomial", &[("defectives", vars(&[0]))]).is_err());
    assert!(run(&mut g, "quality.capability_binomial", &[("defectives", vars(&[0])), ("size", n(2.0))]).is_err());

    let defects = vec![3.0, 5.0, 2.0, 4.0];
    let mut g2 = grid(vec![Column::numeric("def", defects.clone())]);
    let p = run(&mut g2, "quality.capability_poisson", &[("defects", vars(&[0])), ("units", n(10.0))]).unwrap();
    let pw = poisson_capability(&defects, &[10.0; 4], 0.95, None).unwrap();
    let pt = table(&p, "Summary Statistics");
    close(num(pt, 0, 1), pw.dpu, 1e-12);
    close(num(pt, 0, 3), pw.dpu_interval.1, 1e-12);
    assert_eq!(charts(&p).len(), 2);
    assert!(run(&mut g2, "quality.capability_poisson", &[("defects", vars(&[0])), ("units", n(0.0))]).is_err());
}

#[test]
fn multi_variable_capability_dialog() {
    let a: Vec<f64> = (0..40).map(|i| 10.0 + ((i * 13 % 17) as f64 - 8.0) * 0.05).collect();
    let b: Vec<f64> = (0..40).map(|i| 50.0 + ((i * 7 % 19) as f64 - 9.0) * 0.2).collect();
    let mut g = grid(vec![Column::numeric("a", a.clone()), Column::numeric("b", b.clone())]);
    let s = run(
        &mut g,
        "quality.capability_multi",
        &[("vars", vars(&[0, 1])), ("lsl", txt("9 48")), ("usl", txt("11 52"))],
    )
    .unwrap();
    let t = table(&s, "Capability Summary");
    assert_eq!(t.rows.len(), 2);
    let want = capability_individuals(
        &a,
        &Specs { lsl: Some(9.0), target: None, usl: Some(11.0) },
        &CapabilityOptions::default(),
    )
    .unwrap();
    close(num(t, 0, 5), want.cp.unwrap(), 1e-12);
    close(num(t, 0, 6), want.cpk, 1e-12);
    close(num(t, 0, 8), want.ppk, 1e-12);
    // one limit for all, the wrong number of limits, no limits
    let one = run(&mut g, "quality.capability_multi", &[("vars", vars(&[0, 1])), ("usl", txt("60"))]).unwrap();
    assert_eq!(table(&one, "Capability Summary").rows.len(), 2);
    assert!(run(&mut g, "quality.capability_multi", &[("vars", vars(&[0, 1])), ("lsl", txt("1 2 3"))]).is_err());
    assert!(run(&mut g, "quality.capability_multi", &[("vars", vars(&[0, 1]))]).is_err());
    // a variable without limits is skipped with a note
    let skip = run(&mut g, "quality.capability_multi", &[("vars", vars(&[0, 1])), ("usl", txt("11 *"))]).unwrap();
    assert_eq!(table(&skip, "Capability Summary").rows.len(), 1);
    assert!(text(&skip).contains("skipped"));
    // subgroups
    let sub = run(
        &mut g,
        "quality.capability_multi",
        &[("vars", vars(&[0])), ("lsl", txt("9")), ("usl", txt("11")), ("size", n(5.0))],
    )
    .unwrap();
    assert_eq!(table(&sub, "Capability Summary").rows.len(), 1);
}

#[test]
fn transformation_dialogs() {
    let x = lognormalish(80);
    let mut g = grid(vec![Column::numeric("x", x.clone())]);
    let (s, added) = run_full(&mut g, "quality.boxcox", &[("var", vars(&[0]))]).unwrap();
    assert_eq!(added, vec!["x_BC".to_string()]);
    let fit = BoxCox::fit(&x, LambdaChoice::Optimal).unwrap();
    let (lo, hi) = box_cox_interval(&x, 0.95).unwrap();
    let t = table(&s, "Lambda");
    close(num(t, 0, 0), fit.lambda, 1e-12);
    close(num(t, 0, 2), lo, 1e-12);
    close(num(t, 0, 3), hi, 1e-12);
    assert!(lo <= fit.lambda && fit.lambda <= hi);
    assert_eq!(charts(&s).len(), 3);
    let ad = table(&s, "Anderson-Darling");
    assert!(num(ad, 1, 2) > num(ad, 0, 2), "the transformed data should look more normal");
    let stored: Vec<f64> = {
        let ws = g.worksheet();
        let c = ws.columns.iter().position(|c| c.name() == "x_BC").unwrap();
        (0..ws.columns[c].len()).map(|r| ws.columns[c].data.get(r).as_f64()).collect()
    };
    close(stored[3], fit.forward(x[3]), 1e-12);
    // specified lambda 0 is the logarithm
    let mut g2 = grid(vec![Column::numeric("x", x.clone())]);
    let (_, added2) =
        run_full(&mut g2, "quality.boxcox", &[("var", vars(&[0])), ("method", Value::Choice(2)), ("lambda", n(0.0))])
            .unwrap();
    assert_eq!(added2.len(), 1);
    assert!(run(&mut g2, "quality.boxcox", &[("var", vars(&[0])), ("method", Value::Choice(2))]).is_err());
    let mut neg = grid(vec![Column::numeric("x", vec![-1.0, 2.0, 3.0, 4.0])]);
    assert!(run(&mut neg, "quality.boxcox", &[("var", vars(&[0]))]).is_err());

    let mut g3 = grid(vec![Column::numeric("x", x.clone())]);
    let (j, added3) =
        run_full(&mut g3, "quality.johnson", &[("var", vars(&[0])), ("lsl", n(1.0)), ("usl", n(20.0))]).unwrap();
    assert_eq!(added3, vec!["x_J".to_string()]);
    let jt = table(&j, "Johnson Transformation Function");
    assert_eq!(jt.rows.len(), 1);
    let jad = table(&j, "Anderson-Darling");
    assert!(num(jad, 1, 2) >= num(jad, 0, 2));
    assert_eq!(table(&j, "Transformed Specification").rows.len(), 2);
    let mut short = grid(vec![Column::numeric("x", vec![1.0, 2.0, 3.0])]);
    assert!(run(&mut short, "quality.johnson", &[("var", vars(&[0]))]).is_err());
}
