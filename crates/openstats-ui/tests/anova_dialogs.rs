//! Phase 5g dialogs: Balanced ANOVA, random factors in the GLM, Fully Nested ANOVA, Mixed Effects Model,
//! Repeated Measures ANOVA, Analysis of Covariance and the effects plots, run through the registry and
//! compared with the library.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::anova::{balanced_anova, glm, nested_anova, GlmOptions, RandomOptions};
use openstats_stats::mixed::{mixed_model, MixedOptions, RandomSpec};
use openstats_stats::model::Data;
use openstats_stats::report::{Block, Reportable, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

fn val(i: usize, k: u32) -> f64 {
    let mut x = (i as u64 * 2654435761 + k as u64 * 40503) & 0xFFFF_FFFF;
    x ^= x >> 13;
    x = (x * 0x5bd1e995) & 0xFFFF_FFFF;
    (x % 1000) as f64 / 1000.0
}

const NA: usize = 3;
const NB: usize = 4;
const REP: usize = 3;

fn design() -> (Vec<String>, Vec<String>, Vec<f64>) {
    let (mut a, mut b, mut y) = (vec![], vec![], vec![]);
    let mut k = 0;
    for i in 0..NA {
        for j in 0..NB {
            for _ in 0..REP {
                k += 1;
                a.push(format!("m{i}"));
                b.push(format!("o{j}"));
                y.push(
                    (100.0
                        + 4.0 * i as f64
                        + 3.0 * ((j * 5) % 4) as f64
                        + 5.0 * val(k, 1)
                        + 2.0 * ((i + j) % 2) as f64)
                        * 1000.0
                        / 1000.0,
                );
            }
        }
    }
    (a, b, y)
}

// 0 Machine, 1 Operator, 2 y
fn two_way() -> GridController {
    let (a, b, y) = design();
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("Machine", a)).unwrap();
    ws.add_column(Column::text("Operator", b)).unwrap();
    ws.add_column(Column::numeric("y", y)).unwrap();
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
fn ch(i: usize) -> Value {
    Value::Choice(i)
}
fn on() -> Value {
    Value::Bool(true)
}
fn off() -> Value {
    Value::Bool(false)
}
fn num(x: f64) -> Value {
    Value::Num(x)
}
fn txt(s: &str) -> Value {
    Value::Text(s.into())
}
fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}
fn charts(s: &Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count()
}

fn lib_data() -> Data {
    let (a, b, y) = design();
    Data::new().with_factor("Machine", &a).unwrap().with_factor("Operator", &b).unwrap().with_numeric("y", y).unwrap()
}

#[test]
fn balanced_anova_matches_library() {
    let mut g = two_way();
    let s = run(
        &mut g,
        "stat.anova.balanced",
        &[("response", vars(&[2])), ("factors", vars(&[0, 1])), ("random", vars(&[1])), ("order", num(2.0))],
    );
    let lib = balanced_anova(
        "y ~ (Machine + Operator)^2",
        &["Operator".into()],
        &lib_data(),
        &RandomOptions { restricted: true, ..Default::default() },
    )
    .unwrap();
    // The dialog uses the restricted form by default (checkbox on).
    let mut expected = lib.report_section();
    expected.heading = s.heading.clone();
    // The first block is the model text (the dialog writes the expanded formula).
    assert_eq!(s.blocks[1..], expected.blocks[1..]);
    let t = text(&s);
    assert!(
        t.contains("Expected Mean Squares") && t.contains("Variance Components") && t.contains("Machine*Operator"),
        "{t}"
    );
    assert!(s.heading.contains("Balanced ANOVA") && s.heading.contains("random: Operator"));
    // Unrestricted form changes the error term of the random factor.
    let u = run(
        &mut g,
        "stat.anova.balanced",
        &[
            ("response", vars(&[2])),
            ("factors", vars(&[0, 1])),
            ("random", vars(&[1])),
            ("order", num(2.0)),
            ("restricted", off()),
        ],
    );
    assert_ne!(text(&s), text(&u));
    // A typed model gives the same result as the order option.
    let m = run(
        &mut g,
        "stat.anova.balanced",
        &[
            ("response", vars(&[2])),
            ("factors", vars(&[0, 1])),
            ("random", vars(&[1])),
            ("model", txt("Machine Operator Machine*Operator")),
        ],
    );
    assert_eq!(m.blocks[1..], s.blocks[1..]);
}

#[test]
fn balanced_rejects_unbalanced_and_bad_input() {
    let (a, b, mut y) = design();
    y[5] = f64::NAN;
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("Machine", a)).unwrap();
    ws.add_column(Column::text("Operator", b)).unwrap();
    ws.add_column(Column::numeric("y", y)).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let e = try_run(
        &mut g,
        "stat.anova.balanced",
        &[("response", vars(&[2])), ("factors", vars(&[0, 1])), ("random", vars(&[1]))],
    )
    .unwrap_err();
    assert!(e.contains("balanced") || e.contains("missing"), "{e}");
    // The same data works in the general linear model.
    let s = run(
        &mut g,
        "stat.anova.glm",
        &[("response", vars(&[2])), ("factors", vars(&[0, 1])), ("random", vars(&[1])), ("order", num(2.0))],
    );
    assert!(text(&s).contains("unbalanced"));
    let mut g2 = two_way();
    assert!(try_run(
        &mut g2,
        "stat.anova.balanced",
        &[("response", vars(&[2])), ("factors", vars(&[0, 1])), ("random", vars(&[]))]
    )
    .is_err());
    assert!(try_run(
        &mut g2,
        "stat.anova.balanced",
        &[("response", vars(&[2])), ("factors", vars(&[0])), ("random", vars(&[1]))]
    )
    .is_err());
    let e = try_run(
        &mut g2,
        "stat.anova.balanced",
        &[
            ("response", vars(&[2])),
            ("factors", vars(&[0, 1])),
            ("random", vars(&[1])),
            ("model", txt("Machine Zebra")),
        ],
    )
    .unwrap_err();
    assert!(e.contains("Zebra"), "{e}");
}

#[test]
fn glm_dialog_without_random_factors_is_unchanged() {
    let mut g = two_way();
    let s = run(&mut g, "stat.anova.glm", &[("response", vars(&[2])), ("factors", vars(&[0, 1])), ("order", num(2.0))]);
    let lib = glm("y ~ (Machine + Operator)^2", &lib_data(), &GlmOptions::default()).unwrap();
    assert!(text(&s).contains("General Linear Model"));
    assert_eq!(s.blocks.len(), {
        let mut e = lib.report_section();
        e.heading = String::new();
        e.blocks.len()
    });
    assert!(!text(&s).contains("Expected Mean Squares"));
}

#[test]
fn nested_dialog_matches_library_with_repeated_labels() {
    // Labels 1, 2 repeat inside every parent: the dialog must treat them as nested.
    let (mut top, mut mid, mut low, mut y) = (vec![], vec![], vec![], vec![]);
    let mut k = 0;
    for t in 0..3 {
        for m in 0..2 {
            for l in 0..2 {
                for _ in 0..2 {
                    k += 1;
                    top.push(format!("T{t}"));
                    mid.push(format!("{m}"));
                    low.push(format!("{l}"));
                    y.push(50.0 + 2.0 * t as f64 + 1.5 * val(k, 3) + m as f64 * 0.7 + l as f64 * 0.4 + val(k, 9));
                }
            }
        }
    }
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("Top", top.clone())).unwrap();
    ws.add_column(Column::text("Mid", mid.clone())).unwrap();
    ws.add_column(Column::text("Low", low.clone())).unwrap();
    ws.add_column(Column::numeric("y", y.clone())).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(&mut g, "stat.anova.nested", &[("response", vars(&[3])), ("factors", vars(&[0, 1, 2]))]);
    let path = |k: usize| -> Vec<String> {
        (0..y.len())
            .map(|i| [&top, &mid, &low][..=k].iter().map(|c| c[i].as_str()).collect::<Vec<_>>().join("-"))
            .collect()
    };
    let (p0, p1, p2) = (path(0), path(1), path(2));
    let lib = nested_anova(&y, &[("Top", &p0), ("Mid", &p1), ("Low", &p2)]).unwrap();
    let t = text(&s);
    assert!(t.contains("Nested ANOVA") && t.contains("Variance Components"), "{t}");
    // Degrees of freedom: 2, 3, 6 and error 12.
    assert_eq!(lib.rows.iter().map(|r| r.df).collect::<Vec<_>>(), vec![2, 3, 6]);
    assert_eq!(lib.df_error, 12);
    // Number columns agree with the library (labels differ only by the path separator).
    for r in &lib.rows {
        assert!(t.contains(&r.source));
    }
    let lt = {
        let mut r = SessionReport::new("t");
        r.push(lib.report_section());
        r.render_text()
    };
    let digits = |s: &str| -> String { s.chars().filter(|c| c.is_ascii_digit()).collect() };
    let first_table =
        |s: &str| s.split("Expected Mean Squares").next().unwrap_or("").lines().skip(2).collect::<Vec<_>>().join("\n");
    assert_eq!(digits(&first_table(&t)), digits(&first_table(&lt)));
}

fn crossed() -> GridController {
    // 0 subj, 1 item, 2 T, 3 x, 4 y
    let (mut s, mut i_, mut t, mut x, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    let mut k = 0;
    for su in 0..6 {
        for it in 0..4 {
            for r in 0..2 {
                k += 1;
                if (su + it + r) % 5 == 4 {
                    continue; // unbalanced
                }
                s.push(format!("s{su}"));
                i_.push(format!("i{it}"));
                let tt = (su + r) % 2;
                t.push(["a", "b"][tt].to_string());
                let xv = (val(k, 5) * 2.0 - 1.0 * 1000.0).round() / 1000.0;
                x.push(xv);
                y.push(
                    5.0 + 0.7 * xv
                        + if tt == 0 { 0.4 } else { -0.4 }
                        + 1.1 * val(su, 7)
                        + 0.6 * val(it, 8)
                        + 0.5 * val(k, 2),
                );
            }
        }
    }
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("Subject", s)).unwrap();
    ws.add_column(Column::text("Item", i_)).unwrap();
    ws.add_column(Column::text("Treat", t)).unwrap();
    ws.add_column(Column::numeric("x", x)).unwrap();
    ws.add_column(Column::numeric("y", y)).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

#[test]
fn mixed_dialog_matches_library_and_stores_columns() {
    let mut g = crossed();
    let ws = g.worksheet().clone();
    let n = ws.column(4).unwrap().len();
    let col = |c: usize| -> Vec<String> { (0..n).map(|i| ws.column(c).unwrap().display(i)).collect() };
    let xs: Vec<f64> = (0..n).map(|i| ws.column(3).unwrap().data.get(i).as_f64()).collect();
    let ys: Vec<f64> = (0..n).map(|i| ws.column(4).unwrap().data.get(i).as_f64()).collect();
    let data = Data::new()
        .with_factor("Subject", &col(0))
        .unwrap()
        .with_factor("Item", &col(1))
        .unwrap()
        .with_factor("Treat", &col(2))
        .unwrap()
        .with_numeric("x", xs)
        .unwrap()
        .with_numeric("y", ys.clone())
        .unwrap();
    let lib = mixed_model(
        "y ~ Treat + x",
        &[RandomSpec::new(&["Subject"]), RandomSpec::new(&["Item"])],
        &data,
        &MixedOptions::default(),
    )
    .unwrap();
    let s = run(
        &mut g,
        "stat.anova.mixed",
        &[
            ("response", vars(&[4])),
            ("factors", vars(&[0, 1, 2])),
            ("covariates", vars(&[3])),
            ("random", vars(&[0, 1])),
            ("store_fits", on()),
            ("store_marginal", on()),
            ("store_resid", on()),
            ("resid_plot", on()),
            ("show_blups", on()),
        ],
    );
    let t = text(&s);
    assert!(
        t.contains("Mixed Effects Model")
            && t.contains("Satterthwaite")
            && t.contains("Predicted random effects: Subject"),
        "{t}"
    );
    assert_eq!(charts(&s), 1);
    // The variance table has the library's numbers.
    let lt = {
        let mut r = SessionReport::new("t");
        r.push(lib.report_section());
        r.render_text()
    };
    let variance_block =
        |s: &str| s.split("Variance Components").nth(1).unwrap_or("").lines().take(7).collect::<Vec<_>>().join("\n");
    assert_eq!(variance_block(&t), variance_block(&lt));
    // Stored columns equal the library fits.
    let reg = Registry::standard();
    let mut v = reg.get("stat.anova.mixed").unwrap().spec.defaults();
    for (k, x) in [
        ("response", vars(&[4])),
        ("factors", vars(&[0, 1, 2])),
        ("covariates", vars(&[3])),
        ("random", vars(&[0, 1])),
        ("store_fits", on()),
        ("store_resid", on()),
    ] {
        v.set(k, x);
    }
    let before = g.worksheet().columns.len();
    reg.run("stat.anova.mixed", &mut g, &v).unwrap();
    assert_eq!(g.worksheet().columns.len(), before + 2);
    let fits = g.worksheet().columns.iter().find(|c| c.name() == "FITS").expect("FITS stored");
    let resi = g.worksheet().columns.iter().find(|c| c.name() == "RESI").expect("RESI stored");
    for i in 0..n {
        let (f, r) = (fits.data.get(i).as_f64(), resi.data.get(i).as_f64());
        assert!((f - lib.fitted[i]).abs() < 1e-6, "{i}");
        assert!((f + r - ys[i]).abs() < 1e-9);
    }
}

#[test]
fn mixed_dialog_options_and_errors() {
    let mut g = crossed();
    // Maximum likelihood is selectable and reported.
    let s = run(
        &mut g,
        "stat.anova.mixed",
        &[("response", vars(&[4])), ("factors", vars(&[0, 2])), ("random", vars(&[0])), ("method", ch(1))],
    );
    assert!(text(&s).contains("Method: ML"));
    // A combined random term.
    let s = run(
        &mut g,
        "stat.anova.mixed",
        &[
            ("response", vars(&[4])),
            ("factors", vars(&[0, 1, 2])),
            ("random", vars(&[0])),
            ("random_terms", txt("Subject*Item")),
        ],
    );
    assert!(text(&s).contains("Subject*Item"));
    assert!(try_run(&mut g, "stat.anova.mixed", &[("response", vars(&[4])), ("factors", vars(&[2]))]).is_err());
    let e = try_run(
        &mut g,
        "stat.anova.mixed",
        &[("response", vars(&[4])), ("factors", vars(&[0, 2])), ("random_terms", txt("Nope"))],
    )
    .unwrap_err();
    assert!(e.contains("Nope"), "{e}");
    // A random factor that is not among the factors.
    assert!(try_run(
        &mut g,
        "stat.anova.mixed",
        &[("response", vars(&[4])), ("factors", vars(&[2])), ("random", vars(&[0]))]
    )
    .is_err());
}

#[test]
fn repeated_measures_tests_within_effect_against_residual() {
    // 8 subjects in two groups, 3 times.
    let (mut sub, mut grp, mut time, mut y) = (vec![], vec![], vec![], vec![]);
    let mut k = 0;
    for s in 0..8 {
        for t in 0..3 {
            k += 1;
            sub.push(format!("p{s}"));
            grp.push(if s < 4 { "ctl" } else { "trt" }.to_string());
            time.push(format!("t{t}"));
            y.push(
                10.0 + 1.2 * val(s, 4) * 3.0
                    + t as f64 * 1.5
                    + if s >= 4 { 0.8 * t as f64 } else { 0.0 }
                    + 0.4 * val(k, 6),
            );
        }
    }
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("Subject", sub.clone())).unwrap();
    ws.add_column(Column::text("Group", grp.clone())).unwrap();
    ws.add_column(Column::text("Time", time.clone())).unwrap();
    ws.add_column(Column::numeric("y", y.clone())).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(
        &mut g,
        "stat.anova.repeated",
        &[
            ("response", vars(&[3])),
            ("subject", vars(&[0])),
            ("between", vars(&[1])),
            ("within", vars(&[2])),
            ("order", num(2.0)),
        ],
    );
    let data = Data::new()
        .with_factor("Group", &grp)
        .unwrap()
        .with_factor("Time", &time)
        .unwrap()
        .with_factor("Subject", &sub)
        .unwrap()
        .with_numeric("y", y)
        .unwrap();
    let lib = openstats_stats::anova::glm_random(
        "y ~ (Group + Time)^2 + Subject(Group)",
        &["Subject".into()],
        &data,
        &RandomOptions { restricted: true, ..Default::default() },
    )
    .unwrap();
    // Group against Subject; Time and Group*Time against Error.
    // Terms: Group, Time, Subject(Group), Group*Time.
    assert_eq!(lib.rows[0].df, 1);
    assert_eq!(lib.rows[2].df, 6);
    assert_eq!(lib.rows[0].denominator, vec![(2, 1.0)]);
    assert_eq!(lib.rows[1].denominator, vec![(lib.rows.len(), 1.0)]);
    assert_eq!(lib.rows[3].denominator, vec![(lib.rows.len(), 1.0)]);
    let same = |a: &Section, b: &Section| a.blocks.iter().skip(1).zip(b.blocks.iter().skip(1)).all(|(x, y)| x == y);
    let mut lib_sec = lib.report_section();
    lib_sec.heading = s.heading.clone();
    assert!(same(&s, &lib_sec));
    let t = text(&s);
    assert!(t.contains("Repeated Measures ANOVA") && t.contains("Sphericity"), "{t}");
    assert!(t.contains("Subject(Group) (random)"), "{t}");
    // Subject cannot also be a factor.
    assert!(try_run(
        &mut g,
        "stat.anova.repeated",
        &[("response", vars(&[3])), ("subject", vars(&[0])), ("within", vars(&[0, 2]))]
    )
    .is_err());
    assert!(try_run(&mut g, "stat.anova.repeated", &[("response", vars(&[3])), ("subject", vars(&[0]))]).is_err());
}

#[test]
fn ancova_slope_test_and_plot() {
    // Two groups with different slopes.
    let (mut grp, mut x, mut y) = (vec![], vec![], vec![]);
    for i in 0..40 {
        let gi = i % 2;
        let xv = (i / 2) as f64 * 0.5 + val(i, 1);
        grp.push(["a", "b"][gi].to_string());
        x.push(xv);
        y.push(2.0 + if gi == 0 { 1.0 } else { 2.5 } * xv + 0.3 * val(i, 2));
    }
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("Group", grp.clone())).unwrap();
    ws.add_column(Column::numeric("x", x.clone())).unwrap();
    ws.add_column(Column::numeric("y", y.clone())).unwrap();
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(
        &mut g,
        "stat.anova.ancova",
        &[("response", vars(&[2])), ("factors", vars(&[0])), ("covariates", vars(&[1]))],
    );
    let t = text(&s);
    assert!(
        t.contains("Analysis of Covariance") && t.contains("Test for Equal Slopes") && t.contains("Group*x"),
        "{t}"
    );
    let base_charts = charts(&s) - 1; // the GLM report's own plots
                                      // The interaction F equals the library's nested-model F.
    let data =
        Data::new().with_factor("Group", &grp).unwrap().with_numeric("x", x).unwrap().with_numeric("y", y).unwrap();
    let red = glm("y ~ Group + x", &data, &GlmOptions::default()).unwrap();
    let full = glm("y ~ Group + x + Group*x", &data, &GlmOptions::default()).unwrap();
    let f = (red.fit.sse - full.fit.sse) / full.fit.mse;
    assert!(f > 50.0, "slopes differ strongly: {f}");
    let table = s
        .blocks
        .iter()
        .find_map(|b| if let Block::Table(t) = b { (t.title.starts_with("Test for Equal")).then_some(t) } else { None })
        .unwrap();
    match &table.rows[0][3] {
        openstats_stats::report::Cell::Num(v) => assert!((v - f).abs() < 1e-8 * (1.0 + f)),
        other => panic!("{other:?}"),
    }
    // Plot has marker series per level plus one fitted line per level.
    let chart = s
        .blocks
        .iter()
        .find_map(|b| if let Block::Chart(c) = b { (c.title.contains("by Group")).then_some(c) } else { None })
        .expect("the ANCOVA plot");
    assert_eq!(chart.series.len(), 4);
    assert!(chart.series.iter().any(|x| x.name == "Fit: a") && chart.series.iter().any(|x| x.name == "Fit: b"));
    // Options off: no slope table and no chart.
    let s2 = run(
        &mut g,
        "stat.anova.ancova",
        &[
            ("response", vars(&[2])),
            ("factors", vars(&[0])),
            ("covariates", vars(&[1])),
            ("slopes", off()),
            ("plot", off()),
        ],
    );
    assert!(!text(&s2).contains("Equal Slopes") && charts(&s2) == base_charts);
    assert!(try_run(&mut g, "stat.anova.ancova", &[("response", vars(&[2])), ("factors", vars(&[0]))]).is_err());
    // The fitted-line plot renders through the chart engine.
    let scene = openstats_charts::plots::render_spec(chart, &openstats_charts::plots::PlotOptions::default());
    assert!(!scene.items.is_empty());
}

#[test]
fn effects_plots() {
    let mut g = two_way();
    let s = run(&mut g, "stat.anova.maineffects", &[("response", vars(&[2])), ("factors", vars(&[0, 1]))]);
    assert_eq!(charts(&s), 2);
    let c = s.blocks.iter().find_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).unwrap();
    assert_eq!(c.categories, vec!["m0", "m1", "m2"]);
    let (_, _, y) = design();
    let m0: f64 = y[..NB * REP].iter().sum::<f64>() / (NB * REP) as f64;
    assert!((c.series[0].y[0] - m0).abs() < 1e-9);
    let s = run(&mut g, "stat.anova.interactions", &[("response", vars(&[2])), ("factors", vars(&[0, 1]))]);
    assert_eq!(charts(&s), 1);
    let c = s.blocks.iter().find_map(|b| if let Block::Chart(c) = b { Some(c) } else { None }).unwrap();
    assert_eq!(c.series.len(), NB);
    assert!(try_run(&mut g, "stat.anova.interactions", &[("response", vars(&[2])), ("factors", vars(&[0]))]).is_err());
    let _ = ch(0);
}
