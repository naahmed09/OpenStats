//! Summarized-data modes of the t and Z dialogs, with inputs and expected values from Minitab's published examples.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::report::{Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn empty() -> GridController {
    GridController::new(Workbook::new(), 800.0, 400.0)
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
fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

#[test]
fn one_sample_t_summarized_reproduces_the_minitab_example() {
    // N 35, mean 76.8, StDev 11.62, H0 mu = 72: SE 1.96, 95% CI (72.81, 80.79), T 2.44
    let mut g = empty();
    let s = try_run(
        &mut g,
        "stat.basic.ttest1",
        &[("mode", Value::Choice(1)), ("n", n(35.0)), ("mean", n(76.8)), ("sd", n(11.62)), ("mu", n(72.0))],
    )
    .unwrap();
    let t = text(&s);
    for want in ["72.8", "80.7", "2.44"] {
        assert!(t.contains(want), "missing {want} in\n{t}");
    }
    // column mode still works and still requires a column
    assert!(try_run(&mut g, "stat.basic.ttest1", &[]).is_err());
    let _ = Column::numeric("x", vec![1.0]);
}

#[test]
fn two_sample_and_paired_summarized_dialogs_run() {
    let mut g = empty();
    let s = try_run(
        &mut g,
        "stat.basic.ttest2",
        &[
            ("mode", Value::Choice(1)),
            ("n1", n(45.0)),
            ("mean1", n(140.0)),
            ("sd1", n(20.0)),
            ("n2", n(40.0)),
            ("mean2", n(150.0)),
            ("sd2", n(25.0)),
            ("alt", Value::Choice(1)),
        ],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("-2.02") || t.contains("-2.01"), "{t}");
    let p = try_run(
        &mut g,
        "stat.basic.paired",
        &[("mode", Value::Choice(1)), ("n", n(215.0)), ("mean", n(19.49)), ("sd", n(89.81))],
    )
    .unwrap();
    assert!(text(&p).contains("3.18"), "{}", text(&p));
}

#[test]
fn one_sample_z_dialog_reproduces_the_minitab_example() {
    // N 25, mean 80.94, known sigma 11.6 (SE 2.32), test mu = 85 vs <: Z -1.75, P 0.040
    let mut g = empty();
    let s = try_run(
        &mut g,
        "stat.basic.ztest1",
        &[
            ("mode", Value::Choice(1)),
            ("n", n(25.0)),
            ("mean", n(80.94)),
            ("sigma", n(11.6)),
            ("mu", n(85.0)),
            ("alt", Value::Choice(1)),
        ],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("-1.75"), "{t}");
    assert!(t.contains("84.7"), "{t}");
    // a column of data
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("x", vec![9.0, 11.0, 10.0, 12.0, 8.0])).unwrap();
    let mut g2 = GridController::new(wb, 800.0, 400.0);
    let s2 = try_run(&mut g2, "stat.basic.ztest1", &[("var", Value::Vars(vec![0])), ("sigma", n(2.0)), ("mu", n(9.0))])
        .unwrap();
    assert!(text(&s2).contains("Z-Value"), "{}", text(&s2));
}
