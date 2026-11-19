//! Stat ▸ Multivariate dialogs, run on a worksheet through the registry path.

use openstats_charts::{render_spec, PlotOptions};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::multivariate::*;
use openstats_stats::report::{Block, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 45;

fn lcg(i: usize, k: usize) -> f64 {
    let mut x = ((i as u64) << 8 | k as u64).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn columns() -> Vec<Vec<f64>> {
    let centers = [[0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 2.0, -1.0], [-2.0, 4.0, 0.5, 2.0]];
    (0..4).map(|v| (0..N).map(|i| centers[i / 15][v] + 0.8 * lcg(i, v)).collect()).collect()
}

fn group() -> Vec<String> {
    (0..N).map(|i| ["A", "B", "C"][i / 15].to_string()).collect()
}

fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    for (v, c) in columns().into_iter().enumerate() {
        ws.add_column(Column::numeric(format!("x{}", v + 1), c)).unwrap();
    }
    ws.add_column(Column::text("group", group())).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

fn try_run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
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

fn run(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Section {
    try_run(g, id, set).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn vars(c: &[usize]) -> Value {
    Value::Vars(c.to_vec())
}

fn num(x: f64) -> Value {
    Value::Num(x)
}

fn on(b: bool) -> Value {
    Value::Bool(b)
}

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

fn charts(s: &Section) -> Vec<String> {
    s.blocks
        .iter()
        .filter_map(|b| match b {
            Block::Chart(c) => Some(c.title.clone()),
            _ => None,
        })
        .collect()
}

fn stored(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let c = ws.find(name).unwrap_or_else(|| panic!("column {name} not stored"));
    let col = ws.column(c).unwrap();
    (0..col.len()).map(|i| col.get(i).as_f64()).collect()
}

fn names() -> Vec<String> {
    (1..=4).map(|i| format!("x{i}")).collect()
}

const X: [usize; 4] = [0, 1, 2, 3];

#[test]
fn menu_paths_are_under_stat_multivariate() {
    let reg = Registry::standard();
    for id in [
        "stat.mv.pca",
        "stat.mv.factor",
        "stat.mv.discriminant",
        "stat.mv.cluster_obs",
        "stat.mv.cluster_vars",
        "stat.mv.kmeans",
        "stat.mv.manova",
        "stat.mv.item",
    ] {
        let a = reg.get(id).unwrap_or_else(|| panic!("{id} missing"));
        assert_eq!(&a.menu[..2], ["Stat", "Multivariate"], "{id}");
    }
}

#[test]
fn pca_stores_scores_and_honours_options() {
    let mut g = data();
    let s = run(&mut g, "stat.mv.pca", &[("vars", vars(&X)), ("store_scores", on(true)), ("ncomp", num(2.0))]);
    let t = text(&s);
    assert!(t.contains("Principal Component Analysis: x1, x2, x3, x4"), "{t}");
    let want =
        pca(&columns(), &names(), &PcaOptions { matrix: MatrixKind::Correlation, n_components: Some(2) }).unwrap();
    for j in 0..2 {
        let col = stored(&g, &format!("PC{}", j + 1));
        for i in 0..N {
            assert!((col[i] - want.scores[(i, j)]).abs() < 1e-10);
        }
    }
    assert!(g.worksheet().find("PC3").is_none());
    assert_eq!(charts(&s), vec!["Scree Plot", "Score Plot", "Loading Plot"]);
    // Graph switches and the covariance matrix.
    let mut g = data();
    let s = run(
        &mut g,
        "stat.mv.pca",
        &[
            ("vars", vars(&X)),
            ("matrix", Value::Choice(1)),
            ("scree", on(false)),
            ("score_plot", on(false)),
            ("loading_plot", on(false)),
        ],
    );
    assert!(charts(&s).is_empty());
    assert!(text(&s).contains("Eigenanalysis of the Covariance Matrix"));
    // A missing value leaves a missing score.
    {
        let mut cols = columns();
        cols[2][4] = f64::NAN;
        let mut wb = Workbook::new();
        for (v, c) in cols.into_iter().enumerate() {
            wb.worksheets[0].add_column(Column::numeric(format!("x{}", v + 1), c)).unwrap();
        }
        g = GridController::new(wb, 800.0, 400.0);
    }
    run(&mut g, "stat.mv.pca", &[("vars", vars(&X)), ("store_scores", on(true))]);
    let pc1 = stored(&g, "PC1");
    assert!(pc1[4].is_nan() && pc1[5].is_finite());
    // Errors.
    assert!(try_run(&mut data(), "stat.mv.pca", &[]).is_err());
    assert!(try_run(&mut data(), "stat.mv.pca", &[("vars", vars(&[0]))]).is_err());
    assert!(try_run(&mut data(), "stat.mv.pca", &[("vars", vars(&[0, 4]))]).is_err(), "text column");
    assert!(try_run(&mut data(), "stat.mv.pca", &[("vars", vars(&X)), ("ncomp", num(1.5))]).is_err());
}

#[test]
fn factor_analysis_dialog() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.mv.factor",
        &[("vars", vars(&X)), ("factors", num(2.0)), ("store_scores", on(true)), ("score_plot", on(true))],
    );
    let t = text(&s);
    assert!(t.contains("Rotated Factor Loadings") && t.contains("Varimax"), "{t}");
    let mut o = FactorOptions::new(2);
    o.scores = true;
    let want = factor_analysis(&columns(), &names(), &o).unwrap();
    let f1 = stored(&g, "FSCORE1");
    for i in 0..N {
        assert!((f1[i] - want.scores.as_ref().unwrap()[(i, 0)]).abs() < 1e-9);
    }
    assert_eq!(charts(&s), vec!["Scree Plot", "Loading Plot", "Score Plot"]);
    // Maximum likelihood without rotation reports the fit test when the degrees of freedom allow it.
    let s = run(
        &mut data(),
        "stat.mv.factor",
        &[("vars", vars(&X)), ("factors", num(1.0)), ("method", Value::Choice(1)), ("rotation", Value::Choice(0))],
    );
    let t = text(&s);
    assert!(t.contains("Maximum Likelihood") && t.contains("Likelihood Ratio Test") && !t.contains("Rotated"), "{t}");
    // Orthomax gamma and quartimax.
    run(&mut data(), "stat.mv.factor", &[("vars", vars(&X)), ("rotation", Value::Choice(4)), ("gamma", num(0.5))]);
    run(&mut data(), "stat.mv.factor", &[("vars", vars(&X)), ("rotation", Value::Choice(2)), ("kaiser", on(false))]);
    // Too many factors.
    assert!(try_run(&mut data(), "stat.mv.factor", &[("vars", vars(&X)), ("factors", num(5.0))]).is_err());
    assert!(try_run(
        &mut data(),
        "stat.mv.factor",
        &[("vars", vars(&X)), ("factors", num(4.0)), ("method", Value::Choice(1))]
    )
    .is_err());
}

#[test]
fn discriminant_dialog_stores_fits_probabilities_and_scores() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.mv.discriminant",
        &[
            ("groups", vars(&[4])),
            ("vars", vars(&X)),
            ("store_fits", on(true)),
            ("store_probs", on(true)),
            ("store_canonical", on(true)),
        ],
    );
    let t = text(&s);
    assert!(
        t.contains("Linear Discriminant Analysis") && t.contains("Cross-Validation") && t.contains("Canonical"),
        "{t}"
    );
    let want = discriminant(&columns(), &names(), &group(), &DiscriminantOptions::default()).unwrap();
    let ws = g.worksheet();
    let fits = ws.column(ws.find("DAFITS").unwrap()).unwrap();
    for i in 0..N {
        assert_eq!(fits.get(i).to_string(), want.groups[want.predicted[i]], "row {i}");
    }
    let pb = stored(&g, "DAPROB_B");
    for i in 0..N {
        assert!((pb[i] - want.posterior[(i, 1)]).abs() < 1e-12);
    }
    assert_eq!(stored(&g, "DASCORE1").len(), N);
    assert!(g.worksheet().find("DASCORE3").is_none(), "g - 1 = 2 canonical functions");
    assert!(charts(&s).contains(&"Canonical Scores".to_string()));
    // Quadratic with proportional priors; the canonical chart can be switched off.
    let s = run(
        &mut data(),
        "stat.mv.discriminant",
        &[
            ("groups", vars(&[4])),
            ("vars", vars(&X)),
            ("method", Value::Choice(1)),
            ("priors", Value::Choice(1)),
            ("canonical_plot", on(false)),
        ],
    );
    assert!(text(&s).contains("Quadratic Discriminant Analysis"));
    // Specified priors are parsed from text.
    let s = run(
        &mut data(),
        "stat.mv.discriminant",
        &[
            ("groups", vars(&[4])),
            ("vars", vars(&X)),
            ("priors", Value::Choice(2)),
            ("prior_values", Value::Text("0.5, 0.25 0.25".into())),
        ],
    );
    assert!(text(&s).contains("0.50000"), "{}", text(&s));
    // Errors.
    let bad = |set: &[(&str, Value)]| try_run(&mut data(), "stat.mv.discriminant", set).unwrap_err();
    assert!(bad(&[
        ("groups", vars(&[4])),
        ("vars", vars(&X)),
        ("priors", Value::Choice(2)),
        ("prior_values", Value::Text("a b".into()))
    ])
    .contains("prior"));
    assert!(bad(&[
        ("groups", vars(&[4])),
        ("vars", vars(&X)),
        ("priors", Value::Choice(2)),
        ("prior_values", Value::Text("0.5 0.5".into()))
    ])
    .contains("prior"));
    assert!(bad(&[("groups", vars(&[0])), ("vars", vars(&[0, 1]))]).contains("predictor"));
    assert!(try_run(&mut data(), "stat.mv.discriminant", &[("vars", vars(&X))]).is_err());
}

#[test]
fn hierarchical_cluster_dialogs() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.mv.cluster_obs",
        &[("vars", vars(&X)), ("linkage", Value::Choice(5)), ("k", num(3.0)), ("store_membership", on(true))],
    );
    let t = text(&s);
    assert!(t.contains("Ward Linkage") && t.contains("Amalgamation Steps") && t.contains("Final Partition"), "{t}");
    assert!(charts(&s).contains(&"Dendrogram".to_string()));
    let cl = stored(&g, "CLUSTER");
    // The three well separated blocks are recovered.
    for blk in 0..3 {
        let first = cl[blk * 15];
        assert!((0..15).all(|i| cl[blk * 15 + i] == first), "block {blk}");
    }
    assert!(cl[0] != cl[15] && cl[15] != cl[30] && cl[0] != cl[30]);
    // Similarity level, distance choices, no dendrogram, standardised data.
    let s = run(
        &mut data(),
        "stat.mv.cluster_obs",
        &[
            ("vars", vars(&X)),
            ("distance", Value::Choice(2)),
            ("standardize", on(true)),
            ("similarity", num(40.0)),
            ("dendrogram", on(false)),
        ],
    );
    assert!(!charts(&s).contains(&"Dendrogram".to_string()));
    assert!(text(&s).contains("Final Partition"));
    let none = run(&mut data(), "stat.mv.cluster_obs", &[("vars", vars(&X))]);
    assert!(!text(&none).contains("Final Partition"));
    // Errors.
    let e = |set: &[(&str, Value)]| try_run(&mut data(), "stat.mv.cluster_obs", set).unwrap_err();
    assert!(e(&[("vars", vars(&X)), ("k", num(3.0)), ("similarity", num(50.0))]).contains("not both"));
    assert!(e(&[("vars", vars(&X)), ("similarity", num(120.0))]).contains("between 0 and 100"));
    assert!(
        e(&[("vars", vars(&X)), ("store_membership", on(true))]).contains("number of clusters or a similarity level")
    );
    assert!(e(&[("vars", vars(&X)), ("k", num(500.0))]).contains("between 1 and"));
    // Variables.
    let s =
        run(&mut data(), "stat.mv.cluster_vars", &[("vars", vars(&X)), ("k", num(2.0)), ("measure", Value::Choice(1))]);
    let t = text(&s);
    assert!(t.contains("Cluster Analysis of Variables") && t.contains("Cluster Membership") && t.contains("x1"), "{t}");
    assert!(try_run(&mut data(), "stat.mv.cluster_vars", &[("vars", vars(&[0]))]).is_err());
}

#[test]
fn kmeans_dialog() {
    let mut g = data();
    let s = run(
        &mut g,
        "stat.mv.kmeans",
        &[("vars", vars(&X)), ("k", num(3.0)), ("store_membership", on(true)), ("store_distance", on(true))],
    );
    assert!(text(&s).contains("Final Partition") && text(&s).contains("Cluster Centroids"));
    let cl = stored(&g, "KCLUSTER");
    for blk in 0..3 {
        assert!((0..15).all(|i| cl[blk * 15 + i] == cl[blk * 15]), "block {blk}");
    }
    let d = stored(&g, "KDIST");
    assert!(d.iter().all(|x| *x >= 0.0 && x.is_finite()));
    // Initial partition from a column; random starts and standardisation.
    let part: Vec<f64> = (0..N).map(|i| (i % 3 + 1) as f64).collect();
    let mut g = data();
    g.store_columns(vec![Column::numeric("start", part.clone())]).unwrap();
    run(&mut g, "stat.mv.kmeans", &[("vars", vars(&X)), ("start", vars(&[5])), ("standardize", on(true))]);
    run(&mut data(), "stat.mv.kmeans", &[("vars", vars(&X)), ("init", Value::Choice(1)), ("seed", num(7.0))]);
    // A bad partition column is reported with its row.
    let mut bad = part;
    bad[3] = 9.0;
    let mut g = data();
    g.store_columns(vec![Column::numeric("start", bad)]).unwrap();
    let e = try_run(&mut g, "stat.mv.kmeans", &[("vars", vars(&X)), ("start", vars(&[5]))]).unwrap_err();
    assert!(e.contains("row 4"), "{e}");
    assert!(try_run(&mut data(), "stat.mv.kmeans", &[("vars", vars(&X)), ("k", num(1.0))]).is_err());
}

#[test]
fn manova_and_item_analysis_dialogs() {
    let mut g = data();
    let s = run(&mut g, "stat.mv.manova", &[("responses", vars(&[0, 1, 2])), ("factors", vars(&[4]))]);
    let t = text(&s);
    assert!(t.contains("General MANOVA: x1, x2, x3 versus group") && t.contains("Wilks") && t.contains("Roy"), "{t}");
    let d = openstats_stats::model::Data::new().with_factor("group", &group()).unwrap();
    let cols = columns();
    let want = manova(&cols[..3], &names()[..3], &d, "group", &ManovaOptions::default()).unwrap();
    assert!(t.contains(&format!("{:.3}", want.terms[0].tests[0].statistic)), "{t}");
    // A covariate and interactions.
    let mut g = data();
    run(
        &mut g,
        "stat.mv.manova",
        &[("responses", vars(&[0, 1])), ("factors", vars(&[4])), ("covariates", vars(&[2])), ("order", num(2.0))],
    );
    // Errors.
    let e = |set: &[(&str, Value)]| try_run(&mut data(), "stat.mv.manova", set).unwrap_err();
    assert!(e(&[("responses", vars(&[0, 1]))]).contains("factor or covariate"));
    assert!(e(&[("responses", vars(&[0, 1])), ("factors", vars(&[1]))]).contains("cannot also be"));
    assert!(!e(&[("responses", vars(&[0])), ("factors", vars(&[4]))]).is_empty());
    assert!(!e(&[("responses", vars(&[0, 1])), ("factors", vars(&[4])), ("covariates", vars(&[4]))]).is_empty());
    // Item analysis.
    let s = run(&mut data(), "stat.mv.item", &[("vars", vars(&X))]);
    let t = text(&s);
    assert!(t.contains("Cronbach's Alpha: x1, x2, x3, x4") && t.contains("Alpha if item omitted"), "{t}");
    let a = cronbach_alpha(&columns(), &names()).unwrap();
    assert!(t.contains(&format!("{:.4}", a.alpha)[..5]), "{t}");
    assert!(try_run(&mut data(), "stat.mv.item", &[("vars", vars(&[0]))]).is_err());
}

#[test]
fn every_multivariate_chart_renders() {
    let mut g = data();
    let mut all = Vec::new();
    for (id, set) in [
        ("stat.mv.pca", vec![("vars", vars(&X))]),
        ("stat.mv.factor", vec![("vars", vars(&X)), ("score_plot", on(true))]),
        ("stat.mv.discriminant", vec![("groups", vars(&[4])), ("vars", vars(&X))]),
        ("stat.mv.cluster_obs", vec![("vars", vars(&X)), ("k", num(3.0))]),
        ("stat.mv.cluster_vars", vec![("vars", vars(&X))]),
    ] {
        all.push(run(&mut g, id, &set));
    }
    let mut n = 0;
    for s in &all {
        for b in &s.blocks {
            if let Block::Chart(c) = b {
                let scene = render_spec(c, &PlotOptions::default());
                let svg = openstats_charts::render::svg::to_svg(&scene);
                assert!(svg.contains("<svg") && !svg.contains("NaN"), "{}", c.title);
                n += 1;
            }
        }
    }
    assert!(n >= 8, "{n} charts");
}

#[test]
fn correspondence_dialogs_and_promax() {
    // A 5 x 4 table of counts in four columns, with row names.
    let t = [
        [20.0, 4.0, 10.0, 2.0, 9.0],
        [5.0, 25.0, 8.0, 11.0, 4.0],
        [12.0, 9.0, 30.0, 5.0, 6.0],
        [7.0, 3.0, 6.0, 22.0, 8.0],
    ];
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    for (j, c) in t.iter().enumerate() {
        ws.add_column(Column::numeric(format!("cnt{}", j + 1), c.to_vec())).unwrap();
    }
    ws.add_column(Column::text("name", ["r1", "r2", "r3", "r4", "r5"].iter().map(|s| s.to_string()).collect()))
        .unwrap();
    ws.add_column(Column::text("f1", ["a", "b", "a", "b", "a"].iter().map(|s| s.to_string()).collect())).unwrap();
    ws.add_column(Column::text("f2", ["x", "x", "y", "y", "y"].iter().map(|s| s.to_string()).collect())).unwrap();
    ws.add_column(Column::text("f3", ["p", "q", "p", "q", "q"].iter().map(|s| s.to_string()).collect())).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(
        &mut g,
        "stat.mv.ca",
        &[("vars", vars(&[0, 1, 2, 3])), ("row_names", vars(&[4])), ("store_coord", Value::Bool(true))],
    );
    let text = SessionReport::new("t").tap(&s);
    assert!(
        text.contains("Simple Correspondence Analysis") && text.contains("r3") && text.contains("Symmetric Plot"),
        "{text}"
    );
    assert!(g.worksheet().find("RowCoord1").is_some());
    let table: Vec<Vec<f64>> = (0..5).map(|i| (0..4).map(|j| t[j][i]).collect()).collect();
    let want = correspondence_analysis(&table, &vec!["a".to_string(); 5], &vec!["b".to_string(); 4], None).unwrap();
    assert!(text.contains(&format!("{:.4}", want.chi_square)), "{text}");
    let s = run(&mut g, "stat.mv.mca", &[("vars", vars(&[5, 6, 7])), ("components", Value::Num(2.0))]);
    assert!(SessionReport::new("t").tap(&s).contains("Multiple Correspondence Analysis"));
    assert!(try_run(&mut g, "stat.mv.ca", &[("vars", vars(&[0]))]).is_err());
    assert!(try_run(&mut g, "stat.mv.mca", &[("vars", vars(&[5]))]).is_err());
    // Promax on the factor dialog.
    let mut g = data();
    let s = run(
        &mut g,
        "stat.mv.factor",
        &[("vars", vars(&X)), ("factors", Value::Num(2.0)), ("rotation", Value::Choice(5))],
    );
    let text = SessionReport::new("t").tap(&s);
    assert!(text.contains("Pattern Loadings") && text.contains("Factor Correlations"), "{text}");
    assert!(try_run(
        &mut g,
        "stat.mv.factor",
        &[("vars", vars(&X)), ("factors", Value::Num(1.0)), ("rotation", Value::Choice(5))]
    )
    .is_err());
}

trait Tap {
    fn tap(self, s: &Section) -> String;
}

impl Tap for SessionReport {
    fn tap(mut self, s: &Section) -> String {
        self.push(s.clone());
        self.render_text()
    }
}
