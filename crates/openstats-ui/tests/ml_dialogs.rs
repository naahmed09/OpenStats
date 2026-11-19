//! The Predictive Analytics dialogs, run on a worksheet through the same path the menu uses.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::report::Block;
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 60;

fn data() -> GridController {
    let mut wb = Workbook::new();
    let noise = |i: usize| ((i * 37 % 11) as f64 - 5.0) * 0.2;
    let x: Vec<f64> = (0..N).map(|i| (i % 20) as f64).collect();
    let w: Vec<f64> = (0..N).map(|i| ((i * 13) % 17) as f64).collect();
    let y: Vec<f64> = (0..N).map(|i| 2.0 * x[i] + noise(i)).collect();
    let g: Vec<&str> = (0..N).map(|i| ["lo", "mid", "hi"][(x[i] as usize) / 7]).collect();
    let k: Vec<&str> = (0..N).map(|i| ["u", "v"][i % 2]).collect();
    for c in [
        Column::numeric("x", x),
        Column::numeric("w", w),
        Column::numeric("y", y),
        Column::text("grp", g),
        Column::text("k", k),
    ] {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn run(id: &str, set: &[(&str, Value)]) -> (GridController, openstats_stats::report::Section) {
    let reg = Registry::standard();
    let mut g = data();
    let mut v: Values = reg.get(id).unwrap().spec.defaults();
    for (k, x) in set {
        v.set(k, x.clone());
    }
    let problems = validate(&reg.get(id).unwrap().spec, &v, g.worksheet());
    assert!(problems.is_empty(), "{id}: {problems:?}");
    let s = reg.run(id, &mut g, &v).unwrap_or_else(|e| panic!("{id}: {e}"));
    (g, s)
}

fn vars(c: &[usize]) -> Value {
    Value::Vars(c.to_vec())
}

fn find(g: &GridController, name: &str) -> Option<usize> {
    g.worksheet().find(name)
}

fn titles(s: &openstats_stats::report::Section) -> Vec<String> {
    s.blocks.iter().filter_map(|b| if let Block::Table(t) = b { Some(t.title.clone()) } else { None }).collect()
}

#[test]
fn menu_paths_are_under_predictive_analytics() {
    let reg = Registry::standard();
    for id in ["stat.ml.cart", "stat.ml.forest", "stat.ml.gbdt"] {
        let c = reg.get(id).unwrap();
        assert_eq!(c.menu[0], "Stat");
        assert_eq!(c.menu[1], "Predictive Analytics");
    }
}

#[test]
fn cart_regression_stores_fits_residuals_and_nodes() {
    let (g, s) = run(
        "stat.ml.cart",
        &[
            ("response", vars(&[2])),
            ("cont", vars(&[0, 1])),
            ("store_fits", Value::Bool(true)),
            ("store_resid", Value::Bool(true)),
            ("store_node", Value::Bool(true)),
        ],
    );
    let t = titles(&s);
    assert!(t.iter().any(|x| x.contains("Model Summary")), "{t:?}");
    assert!(t.iter().any(|x| x.contains("Variable Importance")), "{t:?}");
    let fits = find(&g, "FITS").expect("fits stored");
    let resid = find(&g, "RESI").expect("residuals stored");
    assert!(find(&g, "NODE").is_some());
    let ws = g.worksheet();
    for i in 0..N {
        let f = ws.column(fits).unwrap().get(i).as_f64();
        let r = ws.column(resid).unwrap().get(i).as_f64();
        let y = ws.column(2).unwrap().get(i).as_f64();
        assert!((f + r - y).abs() < 1e-9, "row {i}");
    }
}

#[test]
fn cart_classification_stores_probabilities_and_classes() {
    let (g, s) = run(
        "stat.ml.cart",
        &[
            ("response", vars(&[3])),
            ("cont", vars(&[0, 1])),
            ("store_fits", Value::Bool(true)),
            ("store_prob", Value::Bool(true)),
        ],
    );
    assert!(titles(&s).iter().any(|x| x.contains("Confusion")));
    assert!(find(&g, "PRED").is_some());
    for l in ["hi", "lo", "mid"] {
        assert!(find(&g, &format!("PROB_{l}")).is_some(), "probability column for {l}");
    }
    // Probabilities of each row sum to one.
    let ws = g.worksheet();
    let cols: Vec<usize> = ["hi", "lo", "mid"].iter().map(|l| find(&g, &format!("PROB_{l}")).unwrap()).collect();
    for i in 0..N {
        let sum: f64 = cols.iter().map(|&c| ws.column(c).unwrap().get(i).as_f64()).sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}

#[test]
fn cart_accepts_categorical_predictors_and_pruning_choices() {
    for pruning in 0..5 {
        let (_, s) = run(
            "stat.ml.cart",
            &[
                ("response", vars(&[2])),
                ("cont", vars(&[1])),
                ("cat", vars(&[4, 3])),
                ("pruning", Value::Choice(pruning)),
                ("folds", Value::Num(5.0)),
                ("max_leaves", Value::Num(4.0)),
            ],
        );
        assert!(!s.blocks.is_empty(), "pruning choice {pruning}");
    }
}

#[test]
fn forest_runs_with_oob_storage_and_intervals() {
    let (g, s) = run(
        "stat.ml.forest",
        &[
            ("response", vars(&[2])),
            ("cont", vars(&[0, 1])),
            ("trees", Value::Num(40.0)),
            ("intervals", Value::Bool(true)),
            ("level", Value::Num(90.0)),
            ("store_fits", Value::Bool(true)),
            ("store_resid", Value::Bool(true)),
            ("store_pi", Value::Bool(true)),
        ],
    );
    let t = titles(&s);
    assert!(t.iter().any(|x| x.contains("Out-of-Bag")), "{t:?}");
    assert!(t.iter().any(|x| x.contains("Variable Importance")));
    assert!(find(&g, "OOB_FITS").is_some() && find(&g, "OOB_RESI").is_some());
    let lo = find(&g, "PI_LOWER").expect("lower limit stored");
    let hi = find(&g, "PI_UPPER").expect("upper limit stored");
    let ws = g.worksheet();
    for i in 0..N {
        let (a, b) = (ws.column(lo).unwrap().get(i).as_f64(), ws.column(hi).unwrap().get(i).as_f64());
        assert!(a <= b, "row {i}");
    }
}

#[test]
fn forest_classification_and_in_sample_storage() {
    let (g, _) = run(
        "stat.ml.forest",
        &[
            ("response", vars(&[3])),
            ("cont", vars(&[0, 1])),
            ("trees", Value::Num(30.0)),
            ("stored", Value::Choice(1)),
            ("store_fits", Value::Bool(true)),
            ("store_prob", Value::Bool(true)),
        ],
    );
    assert!(find(&g, "PRED").is_some());
    assert!(find(&g, "PROB_hi").is_some());
}

#[test]
fn gbdt_regression_huber_and_classification() {
    let (g, s) = run(
        "stat.ml.gbdt",
        &[
            ("response", vars(&[2])),
            ("cont", vars(&[0, 1])),
            ("loss", Value::Choice(1)),
            ("trees", Value::Num(30.0)),
            ("validation", Value::Num(20.0)),
            ("early", Value::Num(5.0)),
            ("store_fits", Value::Bool(true)),
            ("store_resid", Value::Bool(true)),
        ],
    );
    assert!(titles(&s).iter().any(|x| x.contains("Model Summary")));
    assert!(find(&g, "FITS").is_some() && find(&g, "RESI").is_some());
    let (g, s) = run(
        "stat.ml.gbdt",
        &[
            ("response", vars(&[3])),
            ("cont", vars(&[0, 1])),
            ("trees", Value::Num(20.0)),
            ("store_prob", Value::Bool(true)),
        ],
    );
    assert!(titles(&s).iter().any(|x| x.contains("Confusion")));
    assert!(find(&g, "PROB_mid").is_some());
}

#[test]
fn missing_responses_are_dropped_and_missing_predictors_kept() {
    let reg = Registry::standard();
    let mut wb = Workbook::new();
    let y: Vec<f64> = (0..N).map(|i| if i % 10 == 3 { f64::NAN } else { i as f64 }).collect();
    let x: Vec<f64> = (0..N).map(|i| if i % 9 == 4 { f64::NAN } else { (i % 7) as f64 }).collect();
    wb.worksheets[0].add_column(Column::numeric("y", y)).unwrap();
    wb.worksheets[0].add_column(Column::numeric("x", x)).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let mut v = reg.get("stat.ml.cart").unwrap().spec.defaults();
    v.set("response", vars(&[0]));
    v.set("cont", vars(&[1]));
    v.set("store_fits", Value::Bool(true));
    reg.run("stat.ml.cart", &mut g, &v).unwrap();
    let fits = g.worksheet().find("FITS").unwrap();
    for i in 0..N {
        let cell = g.worksheet().column(fits).unwrap().get(i).as_f64();
        assert_eq!(cell.is_nan(), i % 10 == 3, "row {i}");
    }
}

#[test]
fn bad_selections_are_reported() {
    let reg = Registry::standard();
    let g = data();
    let spec = &reg.get("stat.ml.forest").unwrap().spec;
    let mut v = spec.defaults();
    v.set("response", vars(&[2]));
    let mut g = g;
    assert!(reg.run("stat.ml.forest", &mut g, &v).is_err(), "no predictors selected");
}
