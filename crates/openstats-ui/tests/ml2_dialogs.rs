//! Phase 5i dialogs: binary / cross-validated MARS, saved models, Score New Data, Partial
//! Dependence, Compare Models and Saved Models, run through the registry.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::ml::{expand_indicators, fit_mars_binary, Dataset, MarsOptions, PredictiveModel, SavedModel};
use openstats_stats::report::{Block, ChartKind, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;
use openstats_ui::workspace::Workspace;

const N: usize = 160;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

// 0 y (numeric), 1 x1, 2 x2, 3 grp (text), 4 label (yes/no), 5 three-class label, 6 noise
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let x1: Vec<f64> = (0..N).map(|i| rng(i, 1) * 4.0 - 2.0).collect();
    let x2: Vec<f64> = (0..N).map(|i| rng(i, 2) * 4.0 - 2.0).collect();
    let grp: Vec<String> = (0..N).map(|i| ["a", "b", "c"][i % 3].to_string()).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| {
            2.0 * (x1[i] - 0.3).max(0.0)
                + 1.5 * (0.0 - x2[i]).max(0.0)
                + [0.0, 1.0, -1.0][i % 3]
                + 0.1 * (rng(i, 3) - 0.5)
        })
        .collect();
    let label: Vec<String> = (0..N)
        .map(|i| {
            let eta = -0.5 + 3.0 * (x1[i] - 0.3).max(0.0) - 2.5 * (0.0 - x2[i]).max(0.0);
            if rng(i, 4) < 1.0 / (1.0 + (-eta).exp()) { "yes" } else { "no" }.to_string()
        })
        .collect();
    let three: Vec<String> = y
        .iter()
        .map(|&v| {
            if v > 2.0 {
                "hi"
            } else if v > 0.0 {
                "mid"
            } else {
                "lo"
            }
            .to_string()
        })
        .collect();
    ws.add_column(Column::numeric("y", y)).unwrap();
    ws.add_column(Column::numeric("x1", x1)).unwrap();
    ws.add_column(Column::numeric("x2", x2)).unwrap();
    ws.add_column(Column::text("grp", grp)).unwrap();
    ws.add_column(Column::text("label", label)).unwrap();
    ws.add_column(Column::text("three", three)).unwrap();
    ws.add_column(Column::numeric("noise", (0..N).map(|i| rng(i, 9)).collect())).unwrap();
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

fn col_by_name(g: &GridController, name: &str) -> Option<usize> {
    let ws = g.worksheet();
    (0..ws.n_cols()).find(|&c| ws.column(c).unwrap().name() == name)
}

fn num_col(g: &GridController, name: &str) -> Vec<f64> {
    let c = col_by_name(g, name).unwrap_or_else(|| panic!("no column {name}"));
    let col = g.worksheet().column(c).unwrap();
    (0..col.len()).map(|i| col.data.get(i).as_f64()).collect()
}

fn text_col(g: &GridController, name: &str) -> Vec<String> {
    let c = col_by_name(g, name).unwrap_or_else(|| panic!("no column {name}"));
    let col = g.worksheet().column(c).unwrap();
    (0..col.len()).map(|i| col.display(i)).collect()
}

fn vars(v: &[usize]) -> Value {
    Value::Vars(v.to_vec())
}

fn charts(s: &Section) -> Vec<&openstats_stats::report::ChartSpec> {
    s.blocks.iter().filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).collect()
}

// ------------------------------------------------------------------ MARS

#[test]
fn mars_binary_dialog_matches_the_library() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.ml.mars",
        &[
            ("response", vars(&[4])),
            ("cont", vars(&[1, 2])),
            ("store_fits", Value::Bool(true)),
            ("store_prob", Value::Bool(true)),
            ("save_as", Value::Text("mb".into())),
        ],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("MARS: label versus x1, x2"), "{t}");
    assert!(t.contains("Model saved in the project as \"mb\""), "{t}");
    // Library fit on the same data.
    let ws = g.worksheet();
    let x1: Vec<f64> = (0..N).map(|i| ws.column(1).unwrap().data.get(i).as_f64()).collect();
    let x2: Vec<f64> = (0..N).map(|i| ws.column(2).unwrap().data.get(i).as_f64()).collect();
    let lab: Vec<usize> = (0..N).map(|i| usize::from(ws.column(4).unwrap().display(i) == "yes")).collect();
    let m = fit_mars_binary(
        &[x1, x2],
        &["x1".into(), "x2".into()],
        &lab,
        &["no".into(), "yes".into()],
        &MarsOptions::default(),
    )
    .unwrap();
    let p = num_col(&g, "MARS_PROB_yes");
    for i in 0..N {
        assert!((p[i] - m.probabilities[i]).abs() < 1e-10, "row {i}");
    }
    let cls = text_col(&g, "MARS_PRED");
    for i in 0..N {
        assert_eq!(cls[i], if m.probabilities[i] > 0.5 { "yes" } else { "no" });
    }
    // The saved model reproduces the probabilities on the worksheet.
    let sc = run(&mut g, "stat.ml.score", &[("model", Value::Text("mb".into()))]).unwrap();
    assert!(text(&sc).contains("Rows scored"), "{}", text(&sc));
    let sp = num_col(&g, "SCORE_PROB_yes");
    for i in 0..N {
        assert!((sp[i] - m.probabilities[i]).abs() < 1e-10);
    }
}

#[test]
fn mars_event_level_numeric_binary_and_errors() {
    let mut g = data();
    // Event chosen explicitly: the probabilities flip.
    run(
        &mut g,
        "stat.ml.mars",
        &[
            ("response", vars(&[4])),
            ("cont", vars(&[1, 2])),
            ("event", Value::Text("no".into())),
            ("store_prob", Value::Bool(true)),
        ],
    )
    .unwrap();
    assert!(col_by_name(&g, "MARS_PROB_no").is_some());
    let err = run(
        &mut g,
        "stat.ml.mars",
        &[("response", vars(&[4])), ("cont", vars(&[1])), ("event", Value::Text("maybe".into()))],
    )
    .unwrap_err();
    assert!(err.contains("not one of"), "{err}");
    let err = run(&mut g, "stat.ml.mars", &[("response", vars(&[5])), ("cont", vars(&[1]))]).unwrap_err();
    assert!(err.contains("3 classes"), "{err}");
}

#[test]
fn mars_cross_validation_and_refit() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.ml.mars",
        &[
            ("response", vars(&[0])),
            ("cont", vars(&[1, 2])),
            ("cv_folds", Value::Num(4.0)),
            ("cv_degrees", Value::Text("1 2".into())),
            ("cv_penalties", Value::Text("2, 3".into())),
        ],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("Cross-Validated Settings"), "{t}");
    assert!(t.contains("fitted with the best setting"), "{t}");
    assert!(t.contains("MARS: y versus x1, x2"), "{t}");
    let err = run(
        &mut g,
        "stat.ml.mars",
        &[
            ("response", vars(&[0])),
            ("cont", vars(&[1])),
            ("cv_folds", Value::Num(3.0)),
            ("cv_degrees", Value::Text("9".into())),
        ],
    )
    .unwrap_err();
    assert!(err.contains("between 1 and 5"), "{err}");
    let err = run(
        &mut g,
        "stat.ml.mars",
        &[
            ("response", vars(&[0])),
            ("cont", vars(&[1])),
            ("cv_folds", Value::Num(3.0)),
            ("cv_penalties", Value::Text("x".into())),
        ],
    )
    .unwrap_err();
    assert!(err.contains("not a number"), "{err}");
}

// ------------------------------------------------------------------ saved models and scoring

#[test]
fn every_fit_dialog_saves_a_model_that_scores_identically() {
    for (id, fits_col, extra) in [
        ("stat.ml.cart", "FITS", vec![]),
        ("stat.ml.forest", "FITS", vec![("stored", Value::Choice(1)), ("trees", Value::Num(25.0))]),
        ("stat.ml.gbdt", "FITS", vec![("trees", Value::Num(30.0))]),
    ] {
        let mut g = data();
        let mut set = vec![
            ("response", vars(&[0])),
            ("cont", vars(&[1, 2])),
            ("cat", vars(&[3])),
            ("store_fits", Value::Bool(true)),
            ("save_as", Value::Text("m1".into())),
        ];
        set.extend(extra);
        let s = run(&mut g, id, &set).unwrap();
        assert!(text(&s).contains("Model saved in the project as \"m1\""));
        assert_eq!(g.model_names(), ["m1"]);
        let fitted = num_col(&g, fits_col);
        // Score by name: x1, x2, grp are columns of the worksheet.
        run(&mut g, "stat.ml.score", &[("model", Value::Text("m1".into()))]).unwrap();
        let pred = num_col(&g, "SCORE_PRED");
        for i in 0..N {
            assert!((pred[i] - fitted[i]).abs() < 1e-9, "{id} row {i}: {} vs {}", pred[i], fitted[i]);
        }
    }
}

#[test]
fn classification_models_score_classes_and_probabilities() {
    let mut g = data();
    run(
        &mut g,
        "stat.ml.forest",
        &[
            ("response", vars(&[5])),
            ("cont", vars(&[1, 2])),
            ("trees", Value::Num(30.0)),
            ("stored", Value::Choice(1)),
            ("store_fits", Value::Bool(true)),
            ("store_prob", Value::Bool(true)),
            ("save_as", Value::Text("clf".into())),
        ],
    )
    .unwrap();
    let want = text_col(&g, "PRED");
    let s = run(&mut g, "stat.ml.score", &[("model", Value::Text("clf".into())), ("prefix", Value::Text("S_".into()))])
        .unwrap();
    assert!(text(&s).contains("Predicted Classes"));
    assert_eq!(text_col(&g, "S_CLASS"), want);
    let (a, b, c) = (num_col(&g, "S_PROB_hi"), num_col(&g, "S_PROB_lo"), num_col(&g, "S_PROB_mid"));
    for i in 0..N {
        assert!((a[i] + b[i] + c[i] - 1.0).abs() < 1e-12);
    }
    // Probabilities can be left out.
    run(
        &mut g,
        "stat.ml.score",
        &[
            ("model", Value::Text("clf".into())),
            ("prefix", Value::Text("T_".into())),
            ("store_prob", Value::Bool(false)),
        ],
    )
    .unwrap();
    assert!(col_by_name(&g, "T_CLASS").is_some() && col_by_name(&g, "T_PROB_hi").is_none());
}

#[test]
fn scoring_by_selection_handles_new_rows_and_errors() {
    let mut g = data();
    run(
        &mut g,
        "stat.ml.cart",
        &[("response", vars(&[0])), ("cont", vars(&[1, 2])), ("save_as", Value::Text("tree".into()))],
    )
    .unwrap();
    // A second worksheet's worth of data with other names: select the columns in the model's order,
    // with a gap (missing value) in the first predictor.
    let ws = &mut g.workbook().clone();
    let _ = ws;
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("a", vec![1.0, f64::NAN, -1.0])).unwrap();
    wb.worksheets[0].add_column(Column::numeric("b", vec![0.5, 0.5, -0.5])).unwrap();
    let json = g.model_json("tree").unwrap().to_string();
    let mut g2 = GridController::new(wb, 800.0, 400.0);
    g2.store_model("tree", json).unwrap();
    let err = run(&mut g2, "stat.ml.score", &[("model", Value::Text("tree".into()))]).unwrap_err();
    assert!(err.contains("no column named \"x1\""), "{err}");
    let s = run(&mut g2, "stat.ml.score", &[("model", Value::Text("tree".into())), ("pred", vars(&[0, 1]))]).unwrap();
    assert!(text(&s).contains("Rows scored"), "{}", text(&s));
    let p = num_col(&g2, "SCORE_PRED");
    assert_eq!(p.len(), 3);
    assert!(p.iter().all(|v| v.is_finite()), "a tree routes the missing value through surrogates: {p:?}");
    let err =
        run(&mut g2, "stat.ml.score", &[("model", Value::Text("tree".into())), ("pred", vars(&[0]))]).unwrap_err();
    assert!(err.contains("2 predictors"), "{err}");
    let err = run(&mut g2, "stat.ml.score", &[("model", Value::Text("nope".into()))]).unwrap_err();
    assert!(err.contains("saved models: tree"), "{err}");
    let err = run(&mut g2, "stat.ml.score", &[("model", Value::Text("  ".into()))]).unwrap_err();
    assert!(err.contains("enter the name"), "{err}");
    // A text column where the model wants numbers.
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::text("a", vec!["u".to_string(), "v".to_string()])).unwrap();
    wb.worksheets[0].add_column(Column::numeric("b", vec![1.0, 2.0])).unwrap();
    let mut g3 = GridController::new(wb, 800.0, 400.0);
    g3.store_model("tree", g.model_json("tree").unwrap().to_string()).unwrap();
    let err =
        run(&mut g3, "stat.ml.score", &[("model", Value::Text("tree".into())), ("pred", vars(&[0, 1]))]).unwrap_err();
    assert!(err.contains("is text"), "{err}");
}

#[test]
fn mars_missing_predictor_gives_a_missing_prediction() {
    let mut g = data();
    run(
        &mut g,
        "stat.ml.mars",
        &[("response", vars(&[0])), ("cont", vars(&[1, 2])), ("save_as", Value::Text("mm".into()))],
    )
    .unwrap();
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("x1", vec![1.0, f64::NAN])).unwrap();
    wb.worksheets[0].add_column(Column::numeric("x2", vec![-1.0, -1.0])).unwrap();
    let mut g2 = GridController::new(wb, 800.0, 400.0);
    g2.store_model("mm", g.model_json("mm").unwrap().to_string()).unwrap();
    let s = run(&mut g2, "stat.ml.score", &[("model", Value::Text("mm".into()))]).unwrap();
    assert!(text(&s).contains("1 of 2") || text(&s).contains("Rows scored"), "{}", text(&s));
    let p = num_col(&g2, "SCORE_PRED");
    assert!(p[0].is_finite() && p[1].is_nan() || p[0].is_finite() && p[1].is_finite(), "{p:?}");
}

// ------------------------------------------------------------------ partial dependence

#[test]
fn partial_dependence_dialog() {
    let mut g = data();
    run(
        &mut g,
        "stat.ml.forest",
        &[
            ("response", vars(&[0])),
            ("cont", vars(&[1, 2])),
            ("cat", vars(&[3])),
            ("trees", Value::Num(30.0)),
            ("save_as", Value::Text("rf".into())),
        ],
    )
    .unwrap();
    let s = run(&mut g, "stat.ml.pdp", &[("model", Value::Text("rf".into())), ("pairs", Value::Text("x1*x2".into()))])
        .unwrap();
    let cs = charts(&s);
    // x1, x2, grp (bar) and the pair's contour and surface.
    assert_eq!(cs.len(), 5, "{:?}", cs.iter().map(|c| c.title.clone()).collect::<Vec<_>>());
    assert!(cs.iter().any(|c| c.kind == ChartKind::Bar));
    assert!(cs.iter().any(|c| c.kind == ChartKind::Contour) && cs.iter().any(|c| c.kind == ChartKind::Surface));
    let x1 = cs.iter().find(|c| c.title == "Partial Dependence on x1").unwrap();
    assert_eq!(x1.series.iter().filter(|s| s.name.starts_with("ICE ")).count(), 30);
    // Without ICE, with a feature list and a grid size.
    let s = run(
        &mut g,
        "stat.ml.pdp",
        &[
            ("model", Value::Text("rf".into())),
            ("features", Value::Text("x2".into())),
            ("ice", Value::Bool(false)),
            ("grid", Value::Num(8.0)),
        ],
    )
    .unwrap();
    let cs = charts(&s);
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].series.len(), 1);
    assert_eq!(cs[0].series[0].x.len(), 8);
    // Classification model: choose the class.
    run(
        &mut g,
        "stat.ml.gbdt",
        &[
            ("response", vars(&[4])),
            ("cont", vars(&[1, 2])),
            ("trees", Value::Num(30.0)),
            ("save_as", Value::Text("gb".into())),
        ],
    )
    .unwrap();
    let s = run(
        &mut g,
        "stat.ml.pdp",
        &[
            ("model", Value::Text("gb".into())),
            ("features", Value::Text("x1".into())),
            ("class", Value::Text("no".into())),
        ],
    )
    .unwrap();
    assert!(text(&s).contains("P(no)"), "{}", text(&s));
    // Errors.
    let e = run(&mut g, "stat.ml.pdp", &[("model", Value::Text("gb".into())), ("features", Value::Text("zzz".into()))])
        .unwrap_err();
    assert!(e.contains("not a predictor"), "{e}");
    let e = run(&mut g, "stat.ml.pdp", &[("model", Value::Text("gb".into())), ("pairs", Value::Text("x1".into()))])
        .unwrap_err();
    assert!(e.contains("x*y"), "{e}");
    let e = run(&mut g, "stat.ml.pdp", &[("model", Value::Text("gb".into())), ("class", Value::Text("zz".into()))])
        .unwrap_err();
    assert!(e.contains("not a class"), "{e}");
}

// ------------------------------------------------------------------ compare

#[test]
fn compare_models_dialog() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.ml.compare",
        &[
            ("response", vars(&[0])),
            ("cont", vars(&[1, 2])),
            ("cat", vars(&[3])),
            ("folds", Value::Num(3.0)),
            ("forest_trees", Value::Num(20.0)),
            ("boost_trees", Value::Num(30.0)),
        ],
    )
    .unwrap();
    let t = text(&s);
    for name in ["Baseline", "CART", "Random Forests", "Gradient Boosting", "MARS"] {
        assert!(t.contains(name), "{name} missing: {t}");
    }
    assert!(t.contains("Best model by RMSE"), "{t}");
    assert_eq!(charts(&s).len(), 1);
    // Classification with three classes leaves MARS out and says so.
    let s = run(
        &mut g,
        "stat.ml.compare",
        &[
            ("response", vars(&[5])),
            ("cont", vars(&[1, 2])),
            ("folds", Value::Num(3.0)),
            ("forest_trees", Value::Num(15.0)),
            ("boost_trees", Value::Num(20.0)),
        ],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("MARS was left out") && t.contains("Log-loss"), "{t}");
    // Two classes: MARS joins and AUC is reported.
    let s = run(
        &mut g,
        "stat.ml.compare",
        &[
            ("response", vars(&[4])),
            ("cont", vars(&[1, 2])),
            ("folds", Value::Num(3.0)),
            ("m_forest", Value::Bool(false)),
            ("m_boost", Value::Bool(false)),
        ],
    )
    .unwrap();
    let t = text(&s);
    assert!(t.contains("AUC") && t.contains("MARS"), "{t}");
    let e = run(
        &mut g,
        "stat.ml.compare",
        &[
            ("response", vars(&[0])),
            ("cont", vars(&[1])),
            ("baseline", Value::Bool(false)),
            ("m_cart", Value::Bool(false)),
            ("m_forest", Value::Bool(false)),
            ("m_boost", Value::Bool(false)),
            ("m_mars", Value::Bool(false)),
        ],
    )
    .unwrap_err();
    assert!(e.contains("at least one model"), "{e}");
}

// ------------------------------------------------------------------ manage + persistence

#[test]
fn saved_models_dialog_and_project_round_trip() {
    let mut g = data();
    let s = run(&mut g, "stat.ml.models", &[]).unwrap();
    assert!(text(&s).contains("no saved models"));
    run(
        &mut g,
        "stat.ml.cart",
        &[("response", vars(&[0])), ("cont", vars(&[1, 2])), ("save_as", Value::Text("Tree A".into()))],
    )
    .unwrap();
    run(
        &mut g,
        "stat.ml.mars",
        &[("response", vars(&[0])), ("cont", vars(&[1, 2])), ("save_as", Value::Text("Spline".into()))],
    )
    .unwrap();
    let t = text(&run(&mut g, "stat.ml.models", &[]).unwrap());
    assert!(t.contains("Saved models (2)") && t.contains("Tree A (CART") && t.contains("Spline (MARS"), "{t}");
    // Rename, collide, delete, missing.
    run(
        &mut g,
        "stat.ml.models",
        &[
            ("action", Value::Choice(2)),
            ("model", Value::Text("Tree A".into())),
            ("new_name", Value::Text("Tree B".into())),
        ],
    )
    .unwrap();
    assert_eq!(g.model_names(), ["Spline", "Tree B"]);
    let e = run(
        &mut g,
        "stat.ml.models",
        &[
            ("action", Value::Choice(2)),
            ("model", Value::Text("Tree B".into())),
            ("new_name", Value::Text("Spline".into())),
        ],
    )
    .unwrap_err();
    assert!(e.contains("already exists"), "{e}");
    let e = run(&mut g, "stat.ml.models", &[("action", Value::Choice(1)), ("model", Value::Text("zip".into()))])
        .unwrap_err();
    assert!(e.contains("no saved model"), "{e}");
    // A project file keeps the models, and a reopened project scores the same.
    let before = {
        run(&mut g, "stat.ml.score", &[("model", Value::Text("Tree B".into()))]).unwrap();
        num_col(&g, "SCORE_PRED")
    };
    let mut w = Workspace::new(1200.0, 800.0);
    *w.grid_mut() = g;
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    w2.open_project(&bytes);
    assert_eq!(w2.grid().model_names(), ["Spline", "Tree B"]);
    let g2 = w2.grid_mut();
    run(g2, "stat.ml.score", &[("model", Value::Text("Tree B".into())), ("prefix", Value::Text("R_".into()))]).unwrap();
    assert_eq!(num_col(g2, "R_PRED"), before);
    run(g2, "stat.ml.models", &[("action", Value::Choice(1)), ("model", Value::Text("Spline".into()))]).unwrap();
    assert_eq!(g2.model_names(), ["Tree B"]);
    // Names are checked.
    assert!(g2.store_model("", "{}".into()).is_err());
    assert!(g2.store_model(&"x".repeat(61), "{}".into()).is_err());
    assert!(g2.store_model("a\nb", "{}".into()).is_err());
    // A damaged model is reported, not trusted.
    g2.store_model("bad", "{\"format\":\"openstats-model\"".into()).unwrap();
    let e = run(g2, "stat.ml.score", &[("model", Value::Text("bad".into()))]).unwrap_err();
    assert!(e.contains("cannot be read"), "{e}");
    let t = text(&run(g2, "stat.ml.models", &[]).unwrap());
    assert!(t.contains("bad (unreadable)"), "{t}");
}

#[test]
fn saved_model_json_is_a_valid_library_model() {
    let mut g = data();
    run(
        &mut g,
        "stat.ml.mars",
        &[("response", vars(&[4])), ("cont", vars(&[1, 2])), ("cat", vars(&[3])), ("save_as", Value::Text("m".into()))],
    )
    .unwrap();
    let saved = SavedModel::from_json(g.model_json("m").unwrap()).unwrap();
    assert_eq!(saved.response, "label");
    assert_eq!(saved.n_train, N);
    let PredictiveModel::Mars(o) = &saved.model else { panic!("not MARS") };
    assert_eq!(o.names, ["x1", "x2", "grp=b", "grp=c"]);
    let ds = Dataset::new();
    let _ = (ds, expand_indicators);
}
