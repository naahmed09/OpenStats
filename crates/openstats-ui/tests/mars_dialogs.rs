//! Stat ▸ Predictive Analytics ▸ MARS Regression, run through the registry.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::report::{Block, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 90;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

// 0 y, 1 x1, 2 x2, 3 group (text), 4 text response
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let x1: Vec<f64> = (0..N).map(|i| rng(i, 1) * 10.0).collect();
    let x2: Vec<f64> = (0..N).map(|i| rng(i, 2) * 10.0).collect();
    let grp: Vec<String> = (0..N).map(|i| ["a", "b", "c"][i % 3].to_string()).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| {
            1.0 + 2.0 * (x1[i] - 4.0).max(0.0)
                + 1.5 * (6.0 - x2[i]).max(0.0)
                + [0.0, 2.0, -1.0][i % 3]
                + 0.05 * (rng(i, 3) - 0.5)
        })
        .collect();
    ws.add_column(Column::numeric("y", y)).unwrap();
    ws.add_column(Column::numeric("x1", x1)).unwrap();
    ws.add_column(Column::numeric("x2", x2)).unwrap();
    ws.add_column(Column::text("group", grp.clone())).unwrap();
    ws.add_column(Column::text("cls", grp)).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

fn try_run(g: &mut GridController, set: &[(&str, Value)]) -> Result<Section, String> {
    let reg = Registry::standard();
    let id = "stat.ml.mars";
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

fn charts(s: &Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count()
}

#[test]
fn fits_with_continuous_and_categorical_predictors() {
    let mut g = data();
    let s = try_run(
        &mut g,
        &[("response", Value::Vars(vec![0])), ("cont", Value::Vars(vec![1, 2])), ("cat", Value::Vars(vec![3]))],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("MARS: y versus"), "{t}");
    assert!(t.contains("group=b") || t.contains("group=c"));
    assert_eq!(charts(&s), 3);
}

#[test]
fn graph_options_and_storage() {
    let mut g = data();
    let s = try_run(
        &mut g,
        &[
            ("response", Value::Vars(vec![0])),
            ("cont", Value::Vars(vec![1, 2])),
            ("gcv_plot", Value::Bool(false)),
            ("importance_plot", Value::Bool(false)),
            ("fit_plot", Value::Bool(false)),
            ("store_fits", Value::Bool(true)),
            ("store_resid", Value::Bool(true)),
            ("degree", Value::Num(2.0)),
            ("penalty", Value::Text("2.5".into())),
        ],
    )
    .unwrap();
    assert_eq!(charts(&s), 0);
    let ws = g.worksheet();
    let names: Vec<String> = (0..ws.n_cols()).map(|c| ws.column(c).unwrap().name().to_string()).collect();
    assert!(names.iter().any(|n| n == "MARS_FITS") && names.iter().any(|n| n == "MARS_RESI"), "{names:?}");
}

#[test]
fn validation_and_errors() {
    let mut g = data();
    assert!(try_run(&mut g, &[]).is_err());
    assert!(try_run(&mut g, &[("response", Value::Vars(vec![0]))]).unwrap_err().contains("predictor"));
    assert!(try_run(&mut g, &[("response", Value::Vars(vec![0])), ("cont", Value::Vars(vec![0]))]).is_err());
    assert!(try_run(
        &mut g,
        &[("response", Value::Vars(vec![0])), ("cont", Value::Vars(vec![1])), ("cat", Value::Vars(vec![1]))]
    )
    .is_err());
}
