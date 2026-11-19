//! Every registered analysis, run on real data through the same path the dialogs use.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::report::Block;
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 40;

fn data() -> GridController {
    let mut wb = Workbook::new();
    let x: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let noise = |i: usize| ((i * 37 % 11) as f64 - 5.0) * 0.3;
    let y: Vec<f64> = (0..N).map(|i| 3.0 + 2.0 * i as f64 + noise(i)).collect();
    let z: Vec<f64> = (0..N).map(|i| 10.0 + noise(i * 3 + 1) + (i % 7) as f64).collect();
    let g: Vec<&str> = (0..N).map(|i| ["A", "B", "C", "D"][i % 4]).collect();
    for c in [Column::numeric("x", x), Column::numeric("y", y), Column::numeric("z", z), Column::text("g", g)] {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn vals(reg: &Registry, id: &str, set: &[(&str, Value)]) -> Values {
    let mut v = reg.get(id).unwrap().spec.defaults();
    for (k, x) in set {
        v.set(k, x.clone());
    }
    v
}

fn cols(c: &[usize]) -> Value {
    Value::Vars(c.to_vec())
}

fn run(id: &str, set: &[(&str, Value)]) -> (GridController, openstats_stats::report::Section) {
    let reg = Registry::standard();
    let mut g = data();
    let v = vals(&reg, id, set);
    let problems = validate(&reg.get(id).unwrap().spec, &v, g.worksheet());
    assert!(problems.is_empty(), "{id}: {problems:?}");
    let s = reg.run(id, &mut g, &v).unwrap_or_else(|e| panic!("{id}: {e}"));
    (g, s)
}

fn num(c: &openstats_core::Column, i: usize) -> f64 {
    c.get(i).as_f64()
}

fn charts(s: &openstats_stats::report::Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count()
}

fn tables(s: &openstats_stats::report::Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Table(_))).count()
}

#[test]
fn registry_has_menu_paths_for_every_analysis() {
    let reg = Registry::standard();
    assert_eq!(reg.all().len(), 246);
    for a in reg.all() {
        assert!(!a.menu.is_empty() && a.menu.len() >= 2, "{}", a.spec.id);
        assert_eq!(reg.get(&a.spec.id).unwrap().spec.id, a.spec.id);
    }
    let tops: std::collections::BTreeSet<&str> = reg.all().iter().map(|a| a.menu[0]).collect();
    for t in ["Stat", "Graph", "Calc", "Data"] {
        assert!(tops.contains(t), "{t}");
    }
}

#[test]
fn basic_statistics() {
    let (_, s) = run(
        "stat.basic.descriptive",
        &[("vars", cols(&[1, 2])), ("hist", Value::Bool(true)), ("box", Value::Bool(true))],
    );
    assert!(tables(&s) >= 1 && charts(&s) >= 1);
    let (_, s) = run("stat.basic.descriptive", &[("vars", cols(&[1])), ("by", cols(&[3]))]);
    assert!(tables(&s) >= 1);
    let (_, s) = run(
        "stat.basic.ttest1",
        &[("var", cols(&[2])), ("mu", Value::Num(10.0)), ("hist", Value::Bool(true)), ("box", Value::Bool(true))],
    );
    assert!(tables(&s) >= 1 && charts(&s) == 2);
    let (_, s) = run(
        "stat.basic.ttest2",
        &[("first", cols(&[1])), ("second", cols(&[2])), ("pooled", Value::Bool(true)), ("box", Value::Bool(true))],
    );
    assert!(tables(&s) >= 1 && charts(&s) == 1);
    let (_, s) = run("stat.basic.paired", &[("first", cols(&[1])), ("second", cols(&[2]))]);
    assert!(tables(&s) >= 1);
    for t in 0..5 {
        let (_, s) = run("stat.basic.normality", &[("var", cols(&[2])), ("test", Value::Choice(t))]);
        assert!(tables(&s) + charts(&s) >= 1, "test {t}");
    }
    for m in 0..3 {
        let (_, s) = run(
            "stat.basic.correlation",
            &[("vars", cols(&[0, 1, 2])), ("method", Value::Choice(m)), ("scatter", Value::Bool(true))],
        );
        assert!(tables(&s) >= 1 && charts(&s) == 3, "method {m}");
    }
    let (_, s) = run("stat.basic.graphsummary", &[("var", cols(&[1]))]);
    assert!(tables(&s) >= 1 && charts(&s) >= 1);
}

#[test]
fn alternatives_change_the_p_value() {
    let text = |alt: usize| {
        let sec =
            run("stat.basic.ttest1", &[("var", cols(&[2])), ("mu", Value::Num(10.0)), ("alt", Value::Choice(alt))]).1;
        let mut r = openstats_stats::report::SessionReport::new("t");
        r.push(sec);
        r.render_text()
    };
    let (a, b, c) = (text(0), text(1), text(2));
    assert_ne!(a, b);
    assert_ne!(b, c);
}

#[test]
fn regression_stores_fits_and_residuals() {
    let (g, s) = run(
        "stat.regression.fit",
        &[
            ("response", cols(&[1])),
            ("preds", cols(&[0])),
            ("store_fits", Value::Bool(true)),
            ("store_resid", Value::Bool(true)),
            ("store_stdres", Value::Bool(true)),
            ("residual_plots", Value::Bool(true)),
        ],
    );
    assert!(tables(&s) >= 2 && charts(&s) >= 1);
    let ws = g.worksheet();
    assert_eq!(ws.n_cols(), 7, "three stored columns appended");
    let names: Vec<&str> = (4..7).map(|c| ws.columns[c].name()).collect();
    assert!(names.iter().any(|n| n.contains("FITS")) && names.iter().any(|n| n.contains("RESI")), "{names:?}");
    let resid = ws.columns.iter().find(|c| c.name().starts_with("RESI")).unwrap();
    let sum: f64 = (0..N).map(|i| num(resid, i)).sum();
    assert!(sum.abs() < 1e-8, "least-squares residuals sum to zero, got {sum}");
    assert!(g.history().can_undo(), "storing is one undoable step");
}

#[test]
fn anova_spc_capability() {
    let base = [("response", cols(&[2])), ("factor", cols(&[3])), ("tukey", Value::Bool(false))];
    let (_, plain) = run("stat.anova.oneway", &base);
    let mut more = base.to_vec();
    more.extend([("fisher", Value::Bool(true)), ("interval", Value::Bool(true)), ("ivp", Value::Bool(true))]);
    let (_, s) = run("stat.anova.oneway", &more);
    assert!(tables(&s) > tables(&plain), "Fisher comparisons add tables");
    assert_eq!(charts(&s), charts(&plain) + 2, "interval and individual value plots add two charts");
    let (_, s) = run("stat.spc.imr", &[("var", cols(&[2])), ("all_tests", Value::Bool(true))]);
    assert!(charts(&s) >= 2);
    for sigma in 0..2 {
        let (_, s) =
            run("stat.spc.xbar", &[("var", cols(&[2])), ("size", Value::Num(5.0)), ("sigma", Value::Choice(sigma))]);
        assert!(charts(&s) >= 2, "sigma {sigma}");
    }
    let (_, s) = run(
        "stat.quality.capability",
        &[("var", cols(&[2])), ("lsl", Value::Num(2.0)), ("usl", Value::Num(20.0)), ("target", Value::Num(11.0))],
    );
    assert!(tables(&s) >= 1, "capability indices table");
    let (_, s) =
        run("stat.quality.capability", &[("var", cols(&[2])), ("size", Value::Num(4.0)), ("usl", Value::Num(20.0))]);
    assert!(tables(&s) >= 1);
}

#[test]
fn factorial_design_is_stored() {
    let (g, _) = run(
        "stat.doe.factorial",
        &[("factors", Value::Num(3.0)), ("replicates", Value::Num(2.0)), ("center", Value::Num(2.0))],
    );
    let ws = g.worksheet();
    assert!(ws.n_cols() > 4, "design columns appended");
    let (g2, _) = run(
        "stat.doe.factorial",
        &[("factors", Value::Num(2.0)), ("randomize", Value::Bool(true)), ("seed", Value::Num(7.0))],
    );
    assert!(g2.worksheet().n_cols() > 4);
    let (g3, s) = run("stat.doe.factorial", &[("factors", Value::Num(2.0)), ("store", Value::Bool(false))]);
    assert_eq!(g3.worksheet().n_cols(), 4);
    assert!(tables(&s) >= 1);
}

#[test]
fn graphs() {
    for (id, set) in [
        ("graph.histogram", vec![("vars", cols(&[1, 2]))]),
        ("graph.boxplot", vec![("vars", cols(&[1, 2]))]),
        ("graph.scatter", vec![("y", cols(&[1])), ("x", cols(&[0])), ("connect", Value::Bool(true))]),
        ("graph.probability", vec![("var", cols(&[2]))]),
        ("graph.timeseries", vec![("vars", cols(&[1, 2]))]),
    ] {
        let (_, s) = run(id, &set);
        assert!(charts(&s) >= 1, "{id}");
    }
}

#[test]
fn calc_and_data_commands_mutate_the_worksheet() {
    let (g, _) = run("calc.calculator", &[("target", Value::Num(5.0)), ("expr", Value::Text("x * 2 + 1".into()))]);
    assert_eq!(num(&g.worksheet().columns[4], 2), 5.0);
    assert!(g.history().can_undo());
    let (g, _) = run("calc.standardize", &[("vars", cols(&[1]))]);
    assert_eq!(g.worksheet().n_cols(), 5);
    let z: Vec<f64> = (0..N).map(|i| num(&g.worksheet().columns[4], i)).collect();
    let mean = z.iter().sum::<f64>() / z.len() as f64;
    assert!(mean.abs() < 1e-9);
    let var = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (z.len() - 1) as f64;
    assert!((var - 1.0).abs() < 1e-9);
    let (g, _) = run("data.sort", &[("by", cols(&[1])), ("order", Value::Choice(1))]);
    let first = num(&g.worksheet().columns[1], 0);
    let last = num(&g.worksheet().columns[1], N - 1);
    assert!(first > last);
    let x0 = num(&g.worksheet().columns[0], 0);
    assert_eq!(x0, 38.0, "rows travel together");
}

#[test]
fn bad_input_is_reported_not_panicked() {
    let reg = Registry::standard();
    let mut g = data();
    let v = vals(&reg, "stat.basic.ttest1", &[("var", cols(&[3]))]);
    assert!(
        !validate(&reg.get("stat.basic.ttest1").unwrap().spec, &v, g.worksheet()).is_empty(),
        "text column rejected"
    );
    let v = vals(&reg, "stat.basic.ttest1", &[]);
    assert!(!validate(&reg.get("stat.basic.ttest1").unwrap().spec, &v, g.worksheet()).is_empty(), "no variable chosen");
    let v = vals(&reg, "calc.calculator", &[("target", Value::Num(5.0)), ("expr", Value::Text("x +".into()))]);
    assert!(reg.run("calc.calculator", &mut g, &v).is_err());
    let v = vals(&reg, "calc.calculator", &[("target", Value::Num(5.0)), ("expr", Value::Text("nosuch + 1".into()))]);
    assert!(reg.run("calc.calculator", &mut g, &v).is_err());
    let v = vals(&reg, "stat.regression.fit", &[("response", cols(&[1])), ("preds", cols(&[1]))]);
    let _ = reg.run("stat.regression.fit", &mut g, &v);
    assert!(reg.run("no.such.analysis", &mut g, &Values::default()).is_err());
    assert_eq!(g.worksheet().n_cols(), 4, "failed runs leave the worksheet unchanged");
}

fn grid_of(cols: Vec<Column>) -> GridController {
    let mut wb = Workbook::new();
    for c in cols {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn run_on(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> openstats_stats::report::Section {
    let reg = Registry::standard();
    let v = vals(&reg, id, set);
    let problems = validate(&reg.get(id).unwrap().spec, &v, g.worksheet());
    assert!(problems.is_empty(), "{id}: {problems:?}");
    reg.run(id, g, &v).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn text_of(s: openstats_stats::report::Section) -> String {
    let mut r = openstats_stats::report::SessionReport::new("t");
    r.push(s);
    r.render_text()
}

#[test]
fn glm_with_factors_covariates_and_interactions() {
    let (_, s) =
        run("stat.anova.glm", &[("response", cols(&[2])), ("factors", cols(&[3])), ("covariates", cols(&[0]))]);
    let t = text_of(s);
    assert!(t.contains("General Linear Model") && t.contains("Analysis of Variance"), "{t}");
    let g4 = |order: f64| {
        let mut g = grid_of(vec![
            Column::numeric(
                "y",
                (0..24).map(|i| f64::from(i % 5) + f64::from(i / 6) * 0.7 + f64::from((i * 13) % 7) * 0.1).collect(),
            ),
            Column::text("A", (0..24).map(|i| ["a", "b"][i % 2]).collect()),
            Column::text("B", (0..24).map(|i| ["p", "q", "r"][i % 3]).collect()),
        ]);
        text_of(run_on(
            &mut g,
            "stat.anova.glm",
            &[("response", cols(&[0])), ("factors", cols(&[1, 2])), ("order", Value::Num(order))],
        ))
    };
    assert!(!g4(1.0).contains("A*B") && g4(2.0).contains("A*B"), "interactions appear only when asked for");
    // errors
    let reg = Registry::standard();
    let mut g = data();
    for set in [
        vec![("response", cols(&[1]))],
        vec![("response", cols(&[1])), ("factors", cols(&[1]))],
        vec![("response", cols(&[1])), ("factors", cols(&[3])), ("covariates", cols(&[3]))],
    ] {
        let v = vals(&reg, "stat.anova.glm", &set);
        assert!(
            reg.run("stat.anova.glm", &mut g, &v).is_err()
                || !validate(&reg.get("stat.anova.glm").unwrap().spec, &v, g.worksheet()).is_empty()
        );
    }
}

fn factorial_grid(noise: bool) -> GridController {
    let mut a = vec![];
    let mut b = vec![];
    let mut c = vec![];
    let mut y = vec![];
    for rep in 0..2 {
        for i in 0..8 {
            let (x1, x2, x3) = (
                if i & 1 == 0 { 10.0 } else { 20.0 },
                if i & 2 == 0 { 1.0 } else { 3.0 },
                if i & 4 == 0 { 0.0 } else { 5.0 },
            );
            a.push(x1);
            b.push(x2);
            c.push(x3);
            let lo = |v: f64, l: f64, h: f64| 2.0 * (v - l) / (h - l) - 1.0;
            let (ca, cb, cc) = (lo(x1, 10.0, 20.0), lo(x2, 1.0, 3.0), lo(x3, 0.0, 5.0));
            y.push(
                50.0 + 6.0 * ca - 2.0 * cb
                    + 1.5 * ca * cb
                    + if noise { f64::from((i + rep * 3) % 4) * 0.2 } else { 0.0 }
                    + cc * 0.0,
            );
        }
    }
    grid_of(vec![Column::numeric("A", a), Column::numeric("B", b), Column::numeric("C", c), Column::numeric("Y", y)])
}

#[test]
fn doe_analysis_recovers_the_effects() {
    let mut g = factorial_grid(false);
    let s = run_on(&mut g, "stat.doe.analyze", &[("factors", cols(&[0, 1, 2])), ("response", cols(&[3]))]);
    let t = text_of(s);
    assert!(t.contains("Factorial Fit: Y versus A, B, C"), "{t}");
    assert!(t.contains("12.0000") || t.contains("12.000"), "effect of A is 2 × 6: {t}");
    let mut g = factorial_grid(true);
    let t2 = text_of(run_on(
        &mut g,
        "stat.doe.analyze",
        &[("factors", cols(&[0, 1, 2])), ("response", cols(&[3])), ("maxorder", Value::Num(2.0))],
    ));
    assert!(t2.contains("A*B"));
    // a centre point is recognised as curvature
    let g = factorial_grid(true);
    let mut v = vec![];
    for c in 0..4 {
        let mut col: Vec<f64> = (0..16).map(|i| g.worksheet().columns[c].get(i).as_f64()).collect();
        col.push([15.0, 2.0, 2.5, 52.0][c]);
        v.push(Column::numeric(["A", "B", "C", "Y"][c], col));
    }
    let mut g = grid_of(v);
    let t3 = text_of(run_on(&mut g, "stat.doe.analyze", &[("factors", cols(&[0, 1, 2])), ("response", cols(&[3]))]));
    assert!(t3.contains("Curvature") || t3.contains("Ct Pt"), "{t3}");
    // not a two-level design
    let reg = Registry::standard();
    let mut g = data();
    let v = vals(&reg, "stat.doe.analyze", &[("factors", cols(&[0, 1])), ("response", cols(&[2]))]);
    assert!(reg.run("stat.doe.analyze", &mut g, &v).is_err());
}

fn gage_grid() -> GridController {
    let (mut m, mut p, mut o) = (vec![], vec![], vec![]);
    for op in ["Ann", "Bob", "Cy"] {
        for part in 0..6 {
            for rep in 0..3 {
                let bias = match op {
                    "Ann" => 0.0,
                    "Bob" => 0.15,
                    _ => -0.1,
                };
                m.push(10.0 + f64::from(part) * 0.8 + bias + f64::from((part * 7 + rep * 5) % 4) * 0.05);
                p.push(format!("P{part}"));
                o.push(op.to_string());
            }
        }
    }
    grid_of(vec![Column::numeric("Meas", m), Column::text("Part", p), Column::text("Op", o)])
}

#[test]
fn gage_rr_both_methods() {
    let mut g = gage_grid();
    let t = text_of(run_on(
        &mut g,
        "stat.quality.gage",
        &[("measurement", cols(&[0])), ("part", cols(&[1])), ("operator", cols(&[2]))],
    ));
    assert!(t.contains("Gage R&R") && t.contains("Repeatability") && t.contains("Reproducibility"), "{t}");
    let mut g = gage_grid();
    let t = text_of(run_on(
        &mut g,
        "stat.quality.gage",
        &[
            ("measurement", cols(&[0])),
            ("part", cols(&[1])),
            ("operator", cols(&[2])),
            ("method", Value::Choice(1)),
            ("tolerance", Value::Num(6.0)),
        ],
    ));
    assert!(t.contains("Xbar-R"), "{t}");
    let mut g = gage_grid();
    let t = text_of(run_on(
        &mut g,
        "stat.quality.gage",
        &[
            ("measurement", cols(&[0])),
            ("part", cols(&[1])),
            ("operator", cols(&[2])),
            ("interaction", Value::Choice(1)),
            ("tolerance", Value::Num(6.0)),
        ],
    ));
    assert!(t.contains("Tolerance"), "{t}");
    // unbalanced study is an error, not a panic
    let reg = Registry::standard();
    let mut g = gage_grid();
    g.enter_cell(openstats_ui::grid::CellRef { row: 0, col: 0 }, "").unwrap();
    let v =
        vals(&reg, "stat.quality.gage", &[("measurement", cols(&[0])), ("part", cols(&[1])), ("operator", cols(&[2]))]);
    assert!(reg.run("stat.quality.gage", &mut g, &v).is_err());
    let v =
        vals(&reg, "stat.quality.gage", &[("measurement", cols(&[0])), ("part", cols(&[0])), ("operator", cols(&[2]))]);
    assert!(reg.run("stat.quality.gage", &mut g, &v).is_err());
}

#[test]
fn hotelling_chart_individuals_and_subgroups() {
    let mut g = data();
    let s = run_on(&mut g, "stat.multivariate.hotelling", &[("vars", cols(&[1, 2]))]);
    assert!(charts(&s) >= 1 && tables(&s) + charts(&s) >= 2);
    assert!(text_of(s).contains("T²"));
    let mut g = data();
    let s = run_on(
        &mut g,
        "stat.multivariate.hotelling",
        &[("vars", cols(&[1, 2])), ("size", Value::Num(4.0)), ("cov", Value::Choice(1))],
    );
    assert!(charts(&s) >= 1);
    let reg = Registry::standard();
    let mut g = data();
    let v = vals(&reg, "stat.multivariate.hotelling", &[("vars", cols(&[1, 2])), ("size", Value::Num(30.0))]);
    assert!(reg.run("stat.multivariate.hotelling", &mut g, &v).is_err(), "fewer than two subgroups");
}
