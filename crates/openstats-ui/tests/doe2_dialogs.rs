//! Phase 5o dialogs: mixture designs and analysis, optimal designs and split-plot designs,
//! compared with the library routines they wrap.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::doe::{
    fit_mixture, optimal_design, simplex_centroid, simplex_lattice, split_plot, Criterion, MixtureModel, OptModel,
    OptimalOptions,
};
use openstats_stats::report::{Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

fn num(x: f64) -> Value {
    Value::Num(x)
}
fn txt(s: &str) -> Value {
    Value::Text(s.into())
}
fn empty() -> GridController {
    GridController::new(Workbook::new(), 800.0, 400.0)
}
fn sheet(cols: Vec<(&str, Vec<f64>)>) -> GridController {
    let mut wb = Workbook::new();
    for (n, v) in cols {
        wb.worksheets[0].add_column(Column::numeric(n, v)).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
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
fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}
fn column(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let c = (0..ws.columns.len()).find(|&c| ws.columns[c].name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[c].len()).map(|i| ws.columns[c].data.get(i).as_f64()).collect()
}
fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn simplex_lattice_dialog_matches_the_library() {
    let mut g = empty();
    let s = run(&mut g, "stat.doe.mix_lattice", &[("names", txt("X Y Z")), ("degree", num(3.0))]);
    let lib = simplex_lattice(&names(&["X", "Y", "Z"]), 3, true, false, 1, None).unwrap();
    assert!(text(&s).contains(&format!("{} runs", lib.runs())), "{}", text(&s));
    for (j, c) in ["X", "Y", "Z"].iter().enumerate() {
        assert_eq!(column(&g, c), lib.points.iter().map(|p| p[j]).collect::<Vec<_>>());
    }
    let ty: Vec<f64> = lib.point_types.iter().map(|&t| t as f64).collect();
    assert_eq!(column(&g, "PtType"), ty);
}

#[test]
fn simplex_centroid_dialog_matches_the_library() {
    let mut g = empty();
    run(&mut g, "stat.doe.mix_centroid", &[("components", num(4.0)), ("axial", Value::Bool(true))]);
    let lib = simplex_centroid(&names(&["A", "B", "C", "D"]), true, 1, None).unwrap();
    assert_eq!(column(&g, "A").len(), lib.runs());
    assert_eq!(column(&g, "D"), lib.points.iter().map(|p| p[3]).collect::<Vec<_>>());
}

#[test]
fn extreme_vertices_dialog_checks_bounds_and_sums() {
    let mut g = empty();
    run(
        &mut g,
        "stat.doe.mix_vertices",
        &[("names", txt("A B C")), ("lower", txt("0.1 0.2 0")), ("upper", txt("0.6 0.7 0.5"))],
    );
    let (a, b, c) = (column(&g, "A"), column(&g, "B"), column(&g, "C"));
    assert!(!a.is_empty());
    for i in 0..a.len() {
        assert!((a[i] + b[i] + c[i] - 1.0).abs() < 1e-9);
        assert!(
            a[i] >= 0.1 - 1e-9 && a[i] <= 0.6 + 1e-9 && b[i] >= 0.2 - 1e-9 && b[i] <= 0.7 + 1e-9 && c[i] <= 0.5 + 1e-9
        );
    }
    let mut g2 = empty();
    let e = try_run(
        &mut g2,
        "stat.doe.mix_vertices",
        &[("names", txt("A B C")), ("lower", txt("0.1 0.2")), ("upper", txt("0.6 0.7 0.5"))],
    );
    assert!(e.is_err());
}

#[test]
fn mixture_analysis_matches_the_library_and_stores_fits() {
    let pts: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.6, 0.2, 0.2],
    ];
    let y = vec![11.0, 15.2, 9.1, 17.9, 12.4, 14.8, 16.1, 13.3];
    let mut g = sheet(vec![
        ("A", pts.iter().map(|p| p[0]).collect()),
        ("B", pts.iter().map(|p| p[1]).collect()),
        ("C", pts.iter().map(|p| p[2]).collect()),
        ("y", y.clone()),
    ]);
    let s = run(
        &mut g,
        "stat.doe.mix_analyze",
        &[
            ("response", Value::Vars(vec![3])),
            ("components", Value::Vars(vec![0, 1, 2])),
            ("model", Value::Choice(1)),
            ("store_fits", Value::Bool(true)),
            ("store_resid", Value::Bool(true)),
        ],
    );
    let x: Vec<Vec<f64>> = pts.iter().map(|p| p.to_vec()).collect();
    let fit = fit_mixture(&x, &y, &names(&["A", "B", "C"]), MixtureModel::Quadratic).unwrap();
    let t = text(&s);
    assert!(t.contains(&format!("{:.6}", fit.s)), "{t}");
    assert!(t.contains("A*B"), "{t}");
    let fits = column(&g, "FITS");
    for (a, b) in fits.iter().zip(&fit.fitted) {
        assert!((a - b).abs() < 1e-9);
    }
    assert_eq!(column(&g, "RESI").len(), 8);
}

#[test]
fn mixture_analysis_rejects_bad_proportions() {
    let mut g = sheet(vec![("A", vec![0.5, 0.6, 0.2]), ("B", vec![0.4, 0.3, 0.1]), ("y", vec![1.0, 2.0, 3.0])]);
    let e = try_run(
        &mut g,
        "stat.doe.mix_analyze",
        &[("response", Value::Vars(vec![2])), ("components", Value::Vars(vec![0, 1])), ("model", Value::Choice(0))],
    );
    assert!(e.is_err());
}

#[test]
fn optimal_dialog_matches_the_library() {
    let mut g = empty();
    let s = run(
        &mut g,
        "stat.doe.optimal",
        &[
            ("factors", num(3.0)),
            ("runs", num(10.0)),
            ("criterion", num(0.0)),
            ("model", Value::Choice(1)),
            ("seed", num(7.0)),
            ("restarts", num(5.0)),
        ],
    );
    let lib = optimal_design(&OptimalOptions {
        runs: 10,
        criterion: Criterion::D,
        model: OptModel::Interactions,
        levels: vec![2, 2, 2],
        restarts: 5,
        seed: 7,
    })
    .unwrap();
    assert_eq!(column(&g, "A"), lib.points.iter().map(|p| p[0]).collect::<Vec<_>>());
    assert_eq!(column(&g, "C"), lib.points.iter().map(|p| p[2]).collect::<Vec<_>>());
    assert!(text(&s).contains("D-efficiency"), "{}", text(&s));
}

#[test]
fn optimal_dialog_validates_levels() {
    let mut g = empty();
    assert!(try_run(&mut g, "stat.doe.optimal", &[("factors", num(3.0)), ("levels", txt("2 3"))]).is_err());
    assert!(try_run(
        &mut g,
        "stat.doe.optimal",
        &[("factors", num(2.0)), ("runs", num(3.0)), ("model", Value::Choice(2))]
    )
    .is_err());
}

#[test]
fn split_plot_dialog_matches_the_library() {
    let mut g = empty();
    let s = run(&mut g, "stat.doe.splitplot", &[("whole", num(2.0)), ("sub", num(1.0)), ("replicates", num(2.0))]);
    let lib = split_plot(2, 1, 2, None).unwrap();
    assert_eq!(column(&g, "WholePlot"), lib.runs.iter().map(|r| r.whole_plot as f64).collect::<Vec<_>>());
    assert_eq!(column(&g, "C"), lib.runs.iter().map(|r| r.levels[2]).collect::<Vec<_>>());
    assert!(text(&s).contains("Mixed Effects Model"));
    // subplot factors vary within a whole plot; whole-plot factors do not
    let wp = column(&g, "WholePlot");
    let a = column(&g, "A");
    for i in 1..wp.len() {
        if wp[i] == wp[i - 1] {
            assert_eq!(a[i], a[i - 1]);
        }
    }
}

#[test]
fn new_taguchi_arrays_are_offered() {
    let arrays = openstats_stats::doe::available_arrays();
    let idx = arrays.iter().position(|a| *a == "L16(4^5)").unwrap();
    let mut g = empty();
    run(&mut g, "stat.doe.taguchi_create", &[("array", Value::Choice(idx)), ("levels", txt("4 4 4 4 4"))]);
    assert_eq!(column(&g, "A").len(), 16);
    for c in ["A", "B", "C", "D", "E"] {
        let v = column(&g, c);
        for lev in 1..=4 {
            assert_eq!(v.iter().filter(|&&x| x as i64 == lev).count(), 4, "{c} level {lev}");
        }
    }
}

// ---- Phase 5o-b ------------------------------------------------------------------------------

#[test]
fn dsd_dialog_matches_the_library_and_converts_units() {
    let mut g = empty();
    run(
        &mut g,
        "stat.doe.dsd",
        &[("factors", num(6.0)), ("lows", txt("0 10 20 30 40 50")), ("highs", txt("2 12 22 32 42 52"))],
    );
    let lib = openstats_stats::doe::definitive_screening(6, 0, None).unwrap();
    let a = column(&g, "A");
    assert_eq!(a.len(), 13);
    for (i, r) in lib.runs.iter().enumerate() {
        assert!((a[i] - (1.0 + r[0])).abs() < 1e-12);
    }
    let f = column(&g, "F");
    assert!(f.iter().all(|&x| (x - 50.0).abs() < 1e-9 || (x - 51.0).abs() < 1e-9 || (x - 52.0).abs() < 1e-9));
    assert!(try_run(&mut g, "stat.doe.dsd", &[("factors", num(6.0)), ("lows", txt("0 1"))]).is_err());
}

#[test]
fn steepest_ascent_dialog_follows_the_fitted_gradient() {
    // y = 10 + 4a - 2b exactly on a 2^2 design with centre points
    let a = vec![10.0, 20.0, 10.0, 20.0, 15.0, 15.0];
    let b = vec![0.0, 0.0, 100.0, 100.0, 50.0, 50.0];
    let y: Vec<f64> =
        a.iter().zip(&b).map(|(a, b)| 10.0 + 4.0 * ((a - 15.0) / 5.0) - 2.0 * ((b - 50.0) / 50.0)).collect();
    let mut g = sheet(vec![("A", a), ("B", b), ("y", y)]);
    let s = run(
        &mut g,
        "stat.doe.steepest",
        &[
            ("factors", Value::Vars(vec![0, 1])),
            ("response", Value::Vars(vec![2])),
            ("step", num(0.5)),
            ("steps", num(4.0)),
            ("store", Value::Bool(true)),
        ],
    );
    assert!(text(&s).contains("Steepest Ascent"), "{}", text(&s));
    assert_eq!(column(&g, "PathStep"), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert!((column(&g, "PathA")[4] - 25.0).abs() < 1e-9);
    assert!((column(&g, "PathB")[4] - 0.0).abs() < 1e-9);
    assert!((column(&g, "PathFit")[4] - 20.0).abs() < 1e-9);
}

fn replicated_2cube() -> GridController {
    let (mut a, mut b, mut c, mut y) = (vec![], vec![], vec![], vec![]);
    let mut k = 0.0;
    for r in 0..8usize {
        for rep in 0..3 {
            k += 1.0;
            a.push(if r & 1 == 1 { 1.0 } else { -1.0 });
            b.push(if r >> 1 & 1 == 1 { 1.0 } else { -1.0 });
            c.push(if r >> 2 & 1 == 1 { 1.0 } else { -1.0 });
            let sd = if r & 1 == 1 { 2.0 } else { 1.0 };
            y.push(5.0 + [-1.0, 0.2, 0.8][rep] * sd + 0.001 * k);
        }
    }
    sheet(vec![("A", a), ("B", b), ("C", c), ("y", y)])
}

#[test]
fn analyze_variability_dialog_matches_the_library() {
    let mut g = replicated_2cube();
    let s = run(
        &mut g,
        "stat.doe.variability",
        &[("factors", Value::Vars(vec![0, 1, 2])), ("response", Value::Vars(vec![3]))],
    );
    let t = text(&s);
    assert!(t.contains("StDev Ratio"), "{t}");
    // factor A doubles the spread: ratio ~ 2
    let tab = s
        .blocks
        .iter()
        .find_map(|b| if let openstats_stats::report::Block::Table(t) = b { Some(t) } else { None })
        .unwrap();
    assert!(!tab.rows.is_empty());
    assert!(t.contains("1.99"), "{t}");
    let mut bad = sheet(vec![
        ("A", vec![1.0, -1.0, 1.0, -1.0]),
        ("B", vec![1.0, 1.0, -1.0, -1.0]),
        ("y", vec![1.0, 2.0, 3.0, 4.0]),
    ]);
    assert!(try_run(
        &mut bad,
        "stat.doe.variability",
        &[("factors", Value::Vars(vec![0, 1])), ("response", Value::Vars(vec![2]))]
    )
    .is_err());
}

#[test]
fn overlaid_contour_dialog_reports_the_feasible_region() {
    // y1 = A, y2 = B on a 2^2 design with centre points: feasible where both are above 0 (coded)
    let a = vec![0.0, 10.0, 0.0, 10.0, 5.0, 5.0];
    let b = vec![0.0, 0.0, 10.0, 10.0, 5.0, 5.0];
    let y1: Vec<f64> = a.iter().map(|v| v * 2.0).collect();
    let y2: Vec<f64> = b.iter().map(|v| v * 3.0 + 1.0).collect();
    let mut g = sheet(vec![("A", a), ("B", b), ("y1", y1), ("y2", y2)]);
    let s = run(
        &mut g,
        "stat.doe.overlay",
        &[
            ("factors", Value::Vars(vec![0, 1])),
            ("responses", Value::Vars(vec![2, 3])),
            ("bounds", txt("10 *; 16 *")),
            ("model", Value::Choice(0)),
        ],
    );
    let t = text(&s);
    // A >= 5 and B >= 5 → 25% of the square
    assert!(t.contains("25.0%") || t.contains("24.") || t.contains("25."), "{t}");
    let e = try_run(
        &mut g,
        "stat.doe.overlay",
        &[("factors", Value::Vars(vec![0, 1])), ("responses", Value::Vars(vec![2, 3])), ("bounds", txt("10 *"))],
    );
    assert!(e.is_err());
}
