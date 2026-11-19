//! Graph menu dialogs (Phase 5j), run through the registry and the workspace.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::graphical::ci_mean;
use openstats_stats::report::{Block, ChartKind, ChartSpec, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;
use openstats_ui::workspace::{NavAction, Outcome, Target, Workspace};

const N: usize = 60;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

// 0 a (numeric), 1 b, 2 c, 3 grp (text a/b/c), 4 size (positive), 5 colour (text), 6 value
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let a: Vec<f64> = (0..N).map(|i| 10.0 + 5.0 * rng(i, 1)).collect();
    let b: Vec<f64> = (0..N).map(|i| a[i] * 0.5 + rng(i, 2)).collect();
    let c: Vec<f64> = (0..N).map(|i| rng(i, 3) * 100.0 + 1.0).collect();
    let grp: Vec<String> = (0..N).map(|i| ["a", "b", "c"][i % 3].to_string()).collect();
    let colour: Vec<String> = (0..N).map(|i| if i % 4 == 0 { "red" } else { "blue" }.to_string()).collect();
    let mut size: Vec<f64> = (0..N).map(|i| 1.0 + (i % 7) as f64).collect();
    size[3] = 0.0;
    let value: Vec<f64> = (0..N).map(|i| (i % 5) as f64 + 1.0).collect();
    ws.add_column(Column::numeric("a", a)).unwrap();
    ws.add_column(Column::numeric("b", b)).unwrap();
    ws.add_column(Column::numeric("c", c)).unwrap();
    ws.add_column(Column::text("grp", grp)).unwrap();
    ws.add_column(Column::numeric("size", size)).unwrap();
    ws.add_column(Column::text("colour", colour)).unwrap();
    ws.add_column(Column::numeric("value", value)).unwrap();
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

fn one(s: &Section) -> &ChartSpec {
    let c = charts(s);
    assert_eq!(c.len(), 1, "{}", text(s));
    c[0]
}

fn vars(v: &[usize]) -> Value {
    Value::Vars(v.to_vec())
}

fn col(g: &GridController, c: usize) -> Vec<f64> {
    let col = g.worksheet().column(c).unwrap();
    (0..col.len()).map(|i| col.data.get(i).as_f64()).collect()
}

// ------------------------------------------------------------------ bar and pie

#[test]
fn bar_counts_groups_sums_and_means() {
    let mut g = data();
    let s = run(&mut g, "graph.bar", &[("cat", vars(&[5]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Bar);
    assert_eq!(c.categories, ["blue", "red"]);
    assert_eq!(c.series[0].y, [45.0, 15.0]);
    assert!(text(&s).contains("Bar chart data"));
    // Clustered and stacked by group; the counts are 30/30 split.
    let s = run(&mut g, "graph.bar", &[("cat", vars(&[5])), ("group", vars(&[3]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Bar);
    assert_eq!(c.series.len(), 3);
    let tot: f64 = c.series.iter().flat_map(|s| s.y.iter()).sum();
    assert_eq!(tot, 60.0);
    let s =
        run(&mut g, "graph.bar", &[("cat", vars(&[5])), ("group", vars(&[3])), ("multi", Value::Choice(1))]).unwrap();
    assert_eq!(one(&s).kind, ChartKind::StackedBar);
    // Sum and mean of a value variable.
    let value = col(&g, 6);
    let want_red: f64 = (0..N).filter(|i| i % 4 == 0).map(|i| value[i]).sum();
    let s =
        run(&mut g, "graph.bar", &[("cat", vars(&[5])), ("value", vars(&[6])), ("stat", Value::Choice(1))]).unwrap();
    assert!((one(&s).series[0].y[1] - want_red).abs() < 1e-12);
    let s =
        run(&mut g, "graph.bar", &[("cat", vars(&[5])), ("value", vars(&[6])), ("stat", Value::Choice(2))]).unwrap();
    assert!((one(&s).series[0].y[1] - want_red / 15.0).abs() < 1e-12);
    // Percent and decreasing order.
    let s =
        run(&mut g, "graph.bar", &[("cat", vars(&[5])), ("percent", Value::Bool(true)), ("order", Value::Choice(1))])
            .unwrap();
    let c = one(&s);
    assert_eq!(c.categories, ["blue", "red"]);
    assert!((c.series[0].y[0] - 75.0).abs() < 1e-12 && (c.series[0].y[1] - 25.0).abs() < 1e-12);
    // Errors.
    let e = run(&mut g, "graph.bar", &[("cat", vars(&[5])), ("stat", Value::Choice(1))]).unwrap_err();
    assert!(e.contains("value variable"), "{e}");
}

#[test]
fn bar_orders_numeric_labels_numerically() {
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("k", vec![10.0, 2.0, 2.0, 33.0, 10.0, 2.0])).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(&mut g, "graph.bar", &[("cat", vars(&[0]))]).unwrap();
    assert_eq!(one(&s).categories, ["2", "10", "33"]);
    assert_eq!(one(&s).series[0].y, [3.0, 2.0, 1.0]);
}

#[test]
fn pie_combines_small_slices() {
    let mut g = data();
    let s = run(&mut g, "graph.pie", &[("cat", vars(&[5]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Pie);
    assert_eq!(c.series[0].y, [45.0, 15.0]);
    let s = run(&mut g, "graph.pie", &[("cat", vars(&[5])), ("combine", Value::Num(30.0))]).unwrap();
    let c = one(&s);
    assert_eq!(c.categories, ["blue", "Other"]);
    assert_eq!(c.series[0].y, [45.0, 15.0]);
    // A negative value is refused.
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::text("k", vec!["a".to_string(), "b".to_string()])).unwrap();
    wb.worksheets[0].add_column(Column::numeric("v", vec![-1.0, 2.0])).unwrap();
    let mut g2 = GridController::new(wb, 800.0, 400.0);
    let e = run(&mut g2, "graph.pie", &[("cat", vars(&[0])), ("value", vars(&[1]))]).unwrap_err();
    assert!(e.contains("non-negative"), "{e}");
}

// ------------------------------------------------------------------ distribution plots

#[test]
fn dotplot_individual_interval_and_ecdf_by_columns_and_groups() {
    let mut g = data();
    let a = col(&g, 0);
    for (id, kind) in [
        ("graph.dotplot", ChartKind::Dotplot),
        ("graph.ivp", ChartKind::IndividualValue),
        ("graph.ecdf", ChartKind::Ecdf),
    ] {
        let s = run(&mut g, id, &[("vars", vars(&[0, 1]))]).unwrap();
        let c = one(&s);
        assert_eq!(c.kind, kind);
        assert_eq!(c.series.len(), 2, "{id}");
        let by = run(&mut g, id, &[("vars", vars(&[0])), ("by", vars(&[3]))]).unwrap();
        let c = one(&by);
        assert_eq!(c.series.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["a", "b", "c"], "{id}");
        let n: usize = c.series.iter().map(|s| s.x.len() + s.y.len()).sum();
        assert_eq!(n, N, "{id}");
        let e = run(&mut g, id, &[("vars", vars(&[0, 1])), ("by", vars(&[3]))]).unwrap_err();
        assert!(e.contains("exactly one"), "{e}");
    }
    // Interval plot: estimate and limits equal the library's t interval.
    let s = run(&mut g, "graph.interval", &[("vars", vars(&[0])), ("by", vars(&[3])), ("level", Value::Num(90.0))])
        .unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::IntervalPlot);
    let ga: Vec<f64> = (0..N).filter(|i| i % 3 == 0).map(|i| a[i]).collect();
    let ci = ci_mean(&ga, 0.90).unwrap();
    assert!((c.series[1].x[0] - ci.lower).abs() < 1e-12 && (c.series[2].x[0] - ci.upper).abs() < 1e-12);
    assert!((c.series[0].x[0] - ga.iter().sum::<f64>() / ga.len() as f64).abs() < 1e-12);
    assert!(c.title.contains("90% CI"));
    // A group with one value is skipped with a warning.
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("y", vec![1.0, 2.0, 3.0, 9.0])).unwrap();
    wb.worksheets[0]
        .add_column(Column::text("g", vec!["u".to_string(), "u".to_string(), "u".to_string(), "v".to_string()]))
        .unwrap();
    let mut g2 = GridController::new(wb, 800.0, 400.0);
    let s = run(&mut g2, "graph.interval", &[("vars", vars(&[0])), ("by", vars(&[1]))]).unwrap();
    assert_eq!(one(&s).categories, ["u"]);
    assert!(s.warnings.iter().any(|w| w.contains("left out")), "{:?}", s.warnings);
}

#[test]
fn stem_and_leaf_dialog() {
    let mut g = data();
    let s = run(&mut g, "graph.stem", &[("vars", vars(&[2]))]).unwrap();
    let t = text(&s);
    assert!(t.contains("Stem-and-Leaf") && t.contains("Leaf Unit = 1") && t.contains("N = 60"), "{t}");
    assert!(charts(&s).is_empty());
    let s = run(&mut g, "graph.stem", &[("vars", vars(&[2])), ("by", vars(&[3])), ("unit", Value::Text("1".into()))])
        .unwrap();
    let t = text(&s);
    assert!(t.contains("Display: a") && t.contains("Display: b") && t.contains("Display: c"), "{t}");
    assert!(t.contains("Leaf Unit = 1"), "{t}");
    let e = run(&mut g, "graph.stem", &[("vars", vars(&[2])), ("unit", Value::Text("0.3".into()))]).unwrap_err();
    assert!(e.contains("power of ten"), "{e}");
    let e = run(&mut g, "graph.stem", &[("vars", vars(&[2])), ("unit", Value::Text("abc".into()))]).unwrap_err();
    assert!(e.contains("not a number"), "{e}");
}

#[test]
fn probability_plot_distribution_choice() {
    let mut g = data();
    let s = run(&mut g, "graph.probability", &[("var", vars(&[2]))]).unwrap();
    assert_eq!(one(&s).kind, ChartKind::ProbabilityPlot);
    for (k, name) in [(1, "Lognormal"), (2, "Weibull")] {
        let s = run(&mut g, "graph.probability", &[("var", vars(&[2])), ("dist", Value::Choice(k))]).unwrap();
        let c = one(&s);
        assert_eq!(c.kind, ChartKind::DataProbability);
        assert_eq!(c.series[0].name, name);
        assert_eq!(c.series[0].x.len(), N);
    }
    // size has a zero: positive-only distributions refuse it.
    let e = run(&mut g, "graph.probability", &[("var", vars(&[4])), ("dist", Value::Choice(2))]).unwrap_err();
    assert!(e.contains("positive"), "{e}");
    assert!(run(&mut g, "graph.probability", &[("var", vars(&[4]))]).is_ok());
}

// ------------------------------------------------------------------ relationships

#[test]
fn area_stacked_and_plain() {
    let mut g = data();
    let (a, b) = (col(&g, 0), col(&g, 1));
    let s = run(&mut g, "graph.area", &[("vars", vars(&[0, 1]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Area);
    assert_eq!(c.series[0].y, a);
    assert_eq!(c.series[0].x[0], 1.0);
    let s = run(&mut g, "graph.area", &[("vars", vars(&[0, 1])), ("stacked", Value::Bool(true)), ("x", vars(&[2]))])
        .unwrap();
    let c = one(&s);
    // The first drawn series is the total (largest), the second the first variable alone.
    assert_eq!(c.series[0].name, "b");
    for i in 0..N {
        assert!((c.series[0].y[i] - (a[i] + b[i])).abs() < 1e-12);
        assert!((c.series[1].y[i] - a[i]).abs() < 1e-12);
    }
    assert_eq!(c.y_label, "Cumulative");
}

#[test]
fn bubble_matrix_marginal_heatmap() {
    let mut g = data();
    let s = run(&mut g, "graph.bubble", &[("y", vars(&[1])), ("x", vars(&[0])), ("size", vars(&[4]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Bubble);
    assert_eq!(c.series[0].x.len(), N - 1, "the zero size is dropped");
    assert_eq!(c.series[1].y.len(), N - 1);
    assert!(s.warnings.iter().any(|w| w.contains("non-positive")));
    let s = run(&mut g, "graph.matrix", &[("vars", vars(&[0, 1, 2]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Matrix);
    assert_eq!(c.categories, ["a", "b", "c"]);
    assert_eq!(c.series.len(), 6);
    let s01 = c.series.iter().find(|s| s.name == "0|1").unwrap();
    assert_eq!(s01.x, col(&g, 1), "series i|j has variable j on x and i on y");
    assert!(run(&mut g, "graph.matrix", &[("vars", vars(&[0]))]).is_err());
    let s = run(&mut g, "graph.marginal", &[("y", vars(&[1])), ("x", vars(&[0]))]).unwrap();
    assert_eq!(one(&s).kind, ChartKind::Marginal);
    let s = run(&mut g, "graph.heatmap", &[("vars", vars(&[0, 1, 2]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Heatmap);
    assert_eq!(c.series.len(), N);
    assert_eq!(c.series[5].x, [1.0, 2.0, 3.0]);
    assert_eq!(c.series[5].y, [col(&g, 0)[5], col(&g, 1)[5], col(&g, 2)[5]]);
    assert!(run(&mut g, "graph.heatmap", &[("vars", vars(&[0]))]).is_err());
}

// ------------------------------------------------------------------ 3-D

fn xyz_sheet(points: &[(f64, f64)]) -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("x", points.iter().map(|p| p.0).collect())).unwrap();
    ws.add_column(Column::numeric("y", points.iter().map(|p| p.1).collect())).unwrap();
    ws.add_column(Column::numeric("z", points.iter().map(|p| p.0 + p.1).collect())).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

fn pick3() -> Vec<(&'static str, Value)> {
    vec![("x", vars(&[0])), ("y", vars(&[1])), ("z", vars(&[2]))]
}

#[test]
fn scatter3d_surface_and_contour() {
    let mut g = data();
    let s = run(&mut g, "graph.scatter3d", &[("x", vars(&[0])), ("y", vars(&[1])), ("z", vars(&[2]))]).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Scatter3D);
    assert_eq!(c.series[1].name, "Z");
    assert_eq!(c.series[1].y, col(&g, 2));
    // A complete 3 x 4 grid is used as it is.
    let grid: Vec<(f64, f64)> = (0..4).flat_map(|j| (0..3).map(move |i| (f64::from(i), f64::from(j) * 2.0))).collect();
    let mut g = xyz_sheet(&grid);
    let s = run(&mut g, "graph.surface", &pick3()).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Surface);
    assert_eq!(c.series.len(), 4);
    assert_eq!(c.series[0].x, [0.0, 1.0, 2.0]);
    assert_eq!(c.series[3].name, "6");
    assert_eq!(c.series[3].y, [6.0, 7.0, 8.0]);
    assert!(!text(&s).contains("interpolated"));
    // Contours of the plane z = x + y: every vertex satisfies x + y = level.
    let pts: Vec<(f64, f64)> = (0..7).flat_map(|j| (0..7).map(move |i| (f64::from(i), f64::from(j)))).collect();
    let mut g = xyz_sheet(&pts);
    let mut set = pick3();
    set.push(("levels", Value::Num(5.0)));
    let s = run(&mut g, "graph.contour", &set).unwrap();
    let c = one(&s);
    assert_eq!(c.kind, ChartKind::Contour);
    assert!(c.series.len() >= 3);
    for sr in &c.series {
        let level: f64 = sr.name.parse().unwrap();
        let mut n = 0;
        for (x, y) in sr.x.iter().zip(&sr.y) {
            if x.is_finite() {
                assert!((x + y - level).abs() < 1e-9, "{x} + {y} != {level}");
                n += 1;
            }
        }
        assert!(n >= 2);
    }
}

#[test]
fn scattered_points_are_interpolated_and_constant_z_is_refused() {
    let pts: Vec<(f64, f64)> = (0..40).map(|i| (rng(i, 5) * 10.0, rng(i, 6) * 10.0)).collect();
    let mut g = xyz_sheet(&pts);
    let s = run(&mut g, "graph.surface", &pick3()).unwrap();
    assert!(text(&s).contains("interpolated"), "{}", text(&s));
    let c = one(&s);
    assert_eq!(c.series.len(), 25);
    assert!(c.series.iter().all(|r| r.x.len() == 25 && r.y.iter().all(|v| v.is_finite())));
    // Inverse-distance weighting never leaves the range of the data.
    let z: Vec<f64> = pts.iter().map(|p| p.0 + p.1).collect();
    let (lo, hi) =
        (z.iter().copied().fold(f64::INFINITY, f64::min), z.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    assert!(c.series.iter().flat_map(|r| r.y.iter()).all(|v| *v >= lo - 1e-9 && *v <= hi + 1e-9));
    // Constant Z: no contours.
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    for (n, v) in [("x", vec![0.0, 1.0, 2.0, 3.0]), ("y", vec![0.0, 1.0, 0.0, 1.0]), ("z", vec![5.0; 4])] {
        ws.add_column(Column::numeric(n, v)).unwrap();
    }
    let mut g = GridController::new(wb, 800.0, 400.0);
    let e = run(&mut g, "graph.contour", &pick3()).unwrap_err();
    assert!(e.contains("constant"), "{e}");
    let e = run(&mut xyz_sheet(&[(0.0, 0.0), (1.0, 1.0)]), "graph.surface", &pick3()).unwrap_err();
    assert!(e.contains("three complete rows"), "{e}");
}

// ------------------------------------------------------------------ workspace: graphs, PDF, editing

fn workspace() -> Workspace {
    let mut w = Workspace::new(1200.0, 800.0);
    *w.grid_mut() = data();
    w.set_size(1200.0, 800.0);
    w
}

fn fill(w: &mut Workspace, id: &str, set: &[(&str, Value)]) -> Outcome {
    w.open_dialog(id).unwrap();
    let d = w.dialog_mut().unwrap();
    let mut v = d.values().clone();
    for (k, x) in set {
        v.set(k, x.clone());
    }
    d.set_values(v);
    w.run_dialog()
}

#[test]
fn graphs_reach_the_navigator_export_as_pdf_and_can_be_edited() {
    let mut w = workspace();
    assert_eq!(fill(&mut w, "graph.bar", &[("cat", vars(&[5]))]), Outcome::None);
    assert_eq!(
        fill(&mut w, "graph.bubble", &[("y", vars(&[1])), ("x", vars(&[0])), ("size", vars(&[4]))]),
        Outcome::None
    );
    let names: Vec<String> = w.graphs().iter().map(|g| g.name.clone()).collect();
    assert_eq!(names, ["Bar Chart 1", "Bubble Plot 1"]);
    let id = w.graphs()[1].id;
    assert!(openstats_ui::workspace::Navigator::actions_for(Target::Graph(id)).contains(&NavAction::ExportPdf));
    match w.navigator_action(Target::Graph(id), NavAction::ExportPdf) {
        Outcome::File(f) => {
            assert_eq!(f.name, "Bubble Plot 1.pdf");
            assert_eq!(f.mime, "application/pdf");
            assert!(f.bytes.starts_with(b"%PDF-1.4") && f.bytes.ends_with(b"%%EOF\n"));
        }
        o => panic!("{o:?}"),
    }
    // Editing by the dialog changes the Navigator graph and the Session copy.
    let o = fill(
        &mut w,
        "graph.edit",
        &[
            ("graph", Value::Text("Bubble Plot 1".into())),
            ("title", Value::Text("My bubbles".into())),
            ("xlabel", Value::Text("Width".into())),
        ],
    );
    assert_eq!(o, Outcome::None);
    assert!(w.dialog().is_none());
    let spec = &w.graphs()[1].spec;
    assert_eq!((spec.title.as_str(), spec.x_label.as_str()), ("My bubbles", "Width"));
    assert_eq!(spec.y_label, "b", "a blank field keeps the old text");
    let (sid, sec, blk) = w.graphs()[1].source.unwrap();
    let si = w.sessions().iter().position(|s| s.id == sid).unwrap();
    let Block::Chart(c) = &w.sessions()[si].report.sections[sec].blocks[blk] else { panic!("not a chart") };
    assert_eq!(c.title, "My bubbles");
    // A blank name edits the latest graph; unknown names and an empty project are errors.
    assert_eq!(w.edit_graph("", "Latest", "", "").unwrap(), "Bubble Plot 1");
    let e = w.edit_graph("Nope", "x", "", "").unwrap_err();
    assert!(e.contains("Bar Chart 1, Bubble Plot 1"), "{e}");
    let o = fill(&mut w, "graph.edit", &[("graph", Value::Text("Nope".into())), ("title", Value::Text("x".into()))]);
    assert!(matches!(o, Outcome::Error(_)));
    assert!(w.dialog().is_some(), "the dialog stays open on an error");
    let empty = Workspace::new(800.0, 600.0);
    let mut empty = empty;
    assert!(empty.edit_graph("", "t", "", "").is_err());
}

#[test]
fn every_graph_dialog_draws_in_the_workspace() {
    let mut w = workspace();
    let cases: Vec<(&str, Vec<(&str, Value)>)> = vec![
        ("graph.bar", vec![("cat", vars(&[3]))]),
        ("graph.pie", vec![("cat", vars(&[3]))]),
        ("graph.dotplot", vec![("vars", vars(&[0]))]),
        ("graph.ivp", vec![("vars", vars(&[0]))]),
        ("graph.interval", vec![("vars", vars(&[0]))]),
        ("graph.ecdf", vec![("vars", vars(&[0]))]),
        ("graph.area", vec![("vars", vars(&[0]))]),
        ("graph.matrix", vec![("vars", vars(&[0, 1, 2]))]),
        ("graph.heatmap", vec![("vars", vars(&[0, 1]))]),
        ("graph.scatter3d", vec![("x", vars(&[0])), ("y", vars(&[1])), ("z", vars(&[2]))]),
        ("graph.surface", vec![("x", vars(&[0])), ("y", vars(&[1])), ("z", vars(&[2]))]),
        ("graph.contour", vec![("x", vars(&[0])), ("y", vars(&[1])), ("z", vars(&[2]))]),
    ];
    for (id, set) in &cases {
        assert_eq!(fill(&mut w, id, set), Outcome::None, "{id}");
    }
    assert_eq!(w.graphs().len(), cases.len());
    for g in w.graphs().to_vec() {
        // Every graph exports as SVG, PNG and PDF.
        for a in [NavAction::Export, NavAction::ExportPng, NavAction::ExportPdf] {
            assert!(matches!(w.navigator_action(Target::Graph(g.id), a), Outcome::File(_)), "{} {a:?}", g.name);
        }
    }
    // The session report with all the new chart kinds survives a JSON round trip.
    let text = w.sessions()[0].report.to_json();
    let back = SessionReport::from_json(&text).unwrap();
    assert_eq!(back.to_json(), text);
}
