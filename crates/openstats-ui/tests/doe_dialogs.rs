//! Phase 5d dialogs: design creation (fractional factorial, Plackett-Burman, central composite,
//! Box-Behnken, Taguchi) and analysis (response surface, Taguchi, factorial plots).

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::doe::{
    analyze_taguchi, central_composite, fractional_factorial, full_factorial, orthogonal_array, CcdKind, CcdOptions,
    FactorialOptions, SignalToNoise, TaguchiDesign,
};
use openstats_stats::report::{Block, Reportable, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

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

fn txt(s: &str) -> Value {
    Value::Text(s.into())
}

#[test]
fn fractional_dialog_matches_the_library_and_stores_the_design() {
    let mut g = empty();
    let s =
        run(&mut g, "stat.doe.fractional", &[("factors", num(5.0)), ("fraction", num(1.0)), ("alias_order", num(2.0))]);
    let lib = fractional_factorial(5, 1, None, &FactorialOptions { replicates: 1, ..Default::default() }).unwrap();
    assert_eq!(s.blocks[0], lib.report_section().blocks[0]);
    assert!(text(&s).contains("I ="), "{}", text(&s));
    for j in 0..5 {
        assert_eq!(column(&g, ["A", "B", "C", "D", "E"][j]), lib.column(j));
    }
    assert_eq!(column(&g, "StdOrder").len(), 16);
    assert!(column(&g, "PtType").iter().all(|&p| p == 1.0));
}

#[test]
fn explicit_generators_blocks_names_and_levels() {
    let mut g = empty();
    run(
        &mut g,
        "stat.doe.fractional",
        &[
            ("factors", num(4.0)),
            ("generators", txt("D=ABC")),
            ("names", txt("Temp Time Pres Cat")),
            ("lows", txt("100 10 1 0.1")),
            ("highs", txt("200 20 3 0.5")),
            ("natural", on()),
            ("center", num(2.0)),
        ],
    );
    let (t, p, r, c) = (column(&g, "Temp"), column(&g, "Time"), column(&g, "Pres"), column(&g, "Cat"));
    assert_eq!(t.len(), 10);
    assert!(t.iter().all(|&x| x == 100.0 || x == 200.0 || x == 150.0));
    for i in 0..8 {
        // D = ABC in coded units: (T-150)/50 * (P-15)/5 * (R-2)/1
        let code = ((t[i] - 150.0) / 50.0) * ((p[i] - 15.0) / 5.0) * ((r[i] - 2.0) / 1.0);
        let cat = (c[i] - 0.3) / 0.2;
        assert!((code - cat).abs() < 1e-9, "run {i}");
    }
    assert!(try_run(&mut g, "stat.doe.fractional", &[("factors", num(4.0)), ("generators", txt("D=ABX"))]).is_err());
    assert!(try_run(&mut g, "stat.doe.fractional", &[("factors", num(4.0)), ("names", txt("a b"))]).is_err());
}

#[test]
fn full_factorial_dialog_gained_names_blocks_and_levels() {
    let mut g = empty();
    run(
        &mut g,
        "stat.doe.factorial",
        &[
            ("factors", num(3.0)),
            ("blocks", txt("ABC")),
            ("names", txt("x y z")),
            ("lows", txt("0 0 0")),
            ("highs", txt("1 2 4")),
            ("natural", on()),
        ],
    );
    let lib = full_factorial(
        3,
        &FactorialOptions { replicates: 1, block_generators: vec!["ABC".into()], ..Default::default() },
    )
    .unwrap();
    assert_eq!(column(&g, "Blocks"), lib.block.iter().map(|&b| b as f64).collect::<Vec<_>>());
    assert!(column(&g, "z").iter().all(|&x| x == 0.0 || x == 4.0));
}

#[test]
fn plackett_burman_design_is_orthogonal() {
    let mut g = empty();
    run(&mut g, "stat.doe.pb", &[("runs", num(12.0)), ("factors", num(7.0))]);
    let cols: Vec<Vec<f64>> = ["A", "B", "C", "D", "E", "F", "G"].iter().map(|n| column(&g, n)).collect();
    assert_eq!(cols[0].len(), 12);
    for a in 0..7 {
        for b in a + 1..7 {
            let dot: f64 = cols[a].iter().zip(&cols[b]).map(|(x, y)| x * y).sum();
            assert!(dot.abs() < 1e-12, "{a},{b}");
        }
    }
    assert!(try_run(&mut g, "stat.doe.pb", &[("runs", num(10.0)), ("factors", num(5.0))]).is_err());
}

#[test]
fn central_composite_and_box_behnken_dialogs() {
    let mut g = empty();
    run(
        &mut g,
        "stat.doe.ccd",
        &[("factors", num(3.0)), ("lows", txt("10 20 30")), ("highs", txt("20 40 50")), ("natural", on())],
    );
    let lib = central_composite(3, &CcdOptions::new(CcdKind::Circumscribed)).unwrap();
    let a = column(&g, "A");
    assert_eq!(a.len(), lib.n_runs());
    let alpha = 8f64.powf(0.25);
    let top = a.iter().cloned().fold(f64::MIN, f64::max);
    assert!((top - (15.0 + alpha * 5.0)).abs() < 1e-9, "{top}");
    assert!(column(&g, "PtType").contains(&-1.0));
    let mut g2 = empty();
    run(&mut g2, "stat.doe.ccd", &[("factors", num(3.0)), ("kind", ch(2)), ("blocked", on())]);
    let blocks = column(&g2, "Blocks");
    assert!(blocks.contains(&1.0) && blocks.contains(&2.0));
    assert!(column(&g2, "A").iter().all(|x| x.abs() <= 1.0 + 1e-12));
    assert!(try_run(&mut g2, "stat.doe.ccd", &[("factors", num(3.0)), ("alpha", ch(2))]).is_err());
    let mut g3 = empty();
    run(&mut g3, "stat.doe.bb", &[("factors", num(3.0)), ("center", num(3.0))]);
    assert_eq!(column(&g3, "A").len(), 15);
}

#[test]
fn taguchi_create_and_analyze() {
    let mut g = empty();
    let s = run(&mut g, "stat.doe.taguchi_create", &[("array", ch(2)), ("levels", txt("3 3 3 3"))]);
    assert!(text(&s).contains("9 runs"), "{}", text(&s));
    let names = openstats_stats::doe::available_arrays();
    assert_eq!(names[2], "L9");
    assert_eq!(column(&g, "A").len(), 9);
    assert!(try_run(&mut g, "stat.doe.taguchi_create", &[("array", ch(2)), ("levels", txt("3 3 3 3 3"))]).is_err());
    assert!(try_run(&mut g, "stat.doe.taguchi_create", &[("array", ch(2)), ("levels", txt("1 3"))]).is_err());

    // analysis of the stored design with two replicate columns
    let lib = TaguchiDesign::new(orthogonal_array("L9").unwrap(), &[("A", 3), ("B", 3), ("C", 3), ("D", 3)]).unwrap();
    let r1: Vec<f64> = (0..9).map(|i| 10.0 + (i * 7 % 5) as f64).collect();
    let r2: Vec<f64> = (0..9).map(|i| 15.5 + (i * 3 % 4) as f64).collect();
    let mut cols: Vec<(&str, Vec<f64>)> = Vec::new();
    for (j, n) in ["A", "B", "C", "D"].iter().enumerate() {
        cols.push((n, lib.levels.iter().map(|r| r[j] as f64).collect()));
    }
    cols.push(("y1", r1.clone()));
    cols.push(("y2", r2.clone()));
    let mut gw = sheet(cols);
    for sn in 0..4 {
        let s = run(
            &mut gw,
            "stat.doe.taguchi",
            &[("factors", vars(&[0, 1, 2, 3])), ("responses", vars(&[4, 5])), ("sn", ch(sn)), ("store_sn", on())],
        );
        let rows: Vec<Vec<f64>> = (0..9).map(|i| vec![r1[i], r2[i]]).collect();
        let kind = [
            SignalToNoise::SmallerIsBetter,
            SignalToNoise::LargerIsBetter,
            SignalToNoise::NominalIsBest,
            SignalToNoise::NominalIsBestVariance,
        ][sn];
        let want = analyze_taguchi(&lib, &rows, kind).unwrap();
        for (a, b) in column(&gw, "SNRA").iter().zip(&want.run_sn) {
            assert!((a - b).abs() < 1e-12);
        }
        for (a, b) in column(&gw, "MEAN").iter().zip(&want.run_mean) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(text(&s).contains("Taguchi Analysis"));
        assert!(s.blocks.iter().any(|b| matches!(b, Block::Chart(_))));
        gw = {
            // drop the stored columns by rebuilding the worksheet for the next signal-to-noise type
            let mut cols: Vec<(&str, Vec<f64>)> = Vec::new();
            for (j, n) in ["A", "B", "C", "D"].iter().enumerate() {
                cols.push((n, lib.levels.iter().map(|r| r[j] as f64).collect()));
            }
            cols.push(("y1", r1.clone()));
            cols.push(("y2", r2.clone()));
            sheet(cols)
        };
    }
    let mut bad = sheet(vec![("A", vec![0.0, 1.0, 2.0, 1.0]), ("y", vec![1.0, 2.0, 3.0, 4.0])]);
    assert!(try_run(&mut bad, "stat.doe.taguchi", &[("factors", vars(&[0])), ("responses", vars(&[1]))]).is_err());
}

#[test]
fn response_surface_dialog_recovers_an_exact_quadratic() {
    let d = central_composite(2, &CcdOptions::new(CcdKind::Circumscribed)).unwrap();
    let (x1, x2) = (d.column(0), d.column(1));
    let y: Vec<f64> = (0..d.n_runs())
        .map(|i| 5.0 + 2.0 * x1[i] - 3.0 * x2[i] + 1.5 * x1[i] * x1[i] + 0.5 * x2[i] * x2[i] + x1[i] * x2[i])
        .collect();
    let mut g = sheet(vec![("fa", x1), ("fb", x2), ("resp", y.clone())]);
    let s = run(
        &mut g,
        "stat.doe.surface",
        &[
            ("factors", vars(&[0, 1])),
            ("response", vars(&[2])),
            ("contour", on()),
            ("surface", on()),
            ("store_fits", on()),
        ],
    );
    assert!(text(&s).contains("Response Surface Regression"));
    assert!(s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count() >= 2);
    for (f, yv) in column(&g, "RSFITS").iter().zip(&y) {
        assert!((f - yv).abs() < 1e-8);
    }
    assert!(column(&g, "RSRESI").iter().all(|r| r.abs() < 1e-8));
    assert!(try_run(
        &mut g,
        "stat.doe.surface",
        &[("factors", vars(&[0, 1])), ("response", vars(&[2])), ("contour", on()), ("plot_y", num(1.0))]
    )
    .is_err());
    let lin =
        run(&mut g, "stat.doe.surface", &[("factors", vars(&[0, 1])), ("response", vars(&[2])), ("model", ch(0))]);
    assert!(!text(&lin).is_empty());
}

#[test]
fn factorial_analysis_draws_the_requested_plots() {
    let d = full_factorial(3, &FactorialOptions { replicates: 2, ..Default::default() }).unwrap();
    let y: Vec<f64> = (0..d.n_runs())
        .map(|i| 10.0 + 3.0 * d.coded[i][0] - 2.0 * d.coded[i][1] * d.coded[i][2] + (i % 3) as f64 * 0.3)
        .collect();
    let mut g = sheet(vec![("fa", d.column(0)), ("fb", d.column(1)), ("fc", d.column(2)), ("resp", y)]);
    let base = run(&mut g, "stat.doe.analyze", &[("factors", vars(&[0, 1, 2])), ("response", vars(&[3]))]);
    let all = run(
        &mut g,
        "stat.doe.analyze",
        &[
            ("factors", vars(&[0, 1, 2])),
            ("response", vars(&[3])),
            ("main_plot", on()),
            ("int_plot", on()),
            ("cube_plot", on()),
            ("pareto_plot", on()),
            ("normal_plot", on()),
        ],
    );
    let n = |s: &Section| s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count();
    assert!(n(&all) >= n(&base) + 5, "{} vs {}", n(&all), n(&base));
    let mut two = sheet(vec![
        ("fa", vec![-1.0, 1.0, -1.0, 1.0]),
        ("fb", vec![-1.0, -1.0, 1.0, 1.0]),
        ("resp", vec![1.0, 2.0, 3.0, 5.5]),
    ]);
    assert!(try_run(
        &mut two,
        "stat.doe.analyze",
        &[("factors", vars(&[0, 1])), ("response", vars(&[2])), ("cube_plot", on())]
    )
    .is_err());
    let _ = SessionReport::new("x");
}
