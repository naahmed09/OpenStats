//! Stat ▸ Nonparametrics dialogs, run on a worksheet through the registry path.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::hypothesis::Alternative;
use openstats_stats::nonparametric::*;
use openstats_stats::report::{Reportable, Section, SessionReport};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const X: [f64; 12] = [4.1, 5.3, 2.2, 6.8, 7.7, 3.9, 5.0, 6.1, 8.4, 2.9, 5.5, 4.7];
const Y: [f64; 12] = [3.0, 5.9, 1.8, 6.1, 6.0, 4.4, 4.2, 5.2, 7.1, 3.5, 4.9, 4.9];
const GROUP: [&str; 12] = ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"];

fn data() -> GridController {
    data_with(vec![])
}

fn data_with(extra: Vec<Column>) -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("x", X.to_vec())).unwrap();
    ws.add_column(Column::numeric("y", Y.to_vec())).unwrap();
    ws.add_column(Column::text("group", GROUP.iter().map(|s| s.to_string()).collect())).unwrap();
    // A blocked design: 4 blocks × 3 treatments, stacked.
    let resp = [5.0, 7.0, 9.0, 4.0, 8.0, 8.5, 6.0, 7.5, 10.0, 5.5, 6.5, 9.5];
    let treat: Vec<String> = (0..12).map(|i| ["t1", "t2", "t3"][i % 3].to_string()).collect();
    let block: Vec<String> = (0..12).map(|i| format!("b{}", i / 3)).collect();
    ws.add_column(Column::numeric("resp", resp.to_vec())).unwrap();
    ws.add_column(Column::text("treat", treat)).unwrap();
    ws.add_column(Column::text("block", block)).unwrap();
    for c in extra {
        ws.add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn try_run(id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
    run_on(&mut data(), id, set)
}

fn run_on(g: &mut GridController, id: &str, set: &[(&str, Value)]) -> Result<Section, String> {
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

fn run(id: &str, set: &[(&str, Value)]) -> Section {
    try_run(id, set).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn vars(c: &[usize]) -> Value {
    Value::Vars(c.to_vec())
}

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

#[test]
fn menu_paths_are_under_stat_nonparametrics() {
    let reg = Registry::standard();
    let ids = [
        "stat.nonpar.sign",
        "stat.nonpar.wilcoxon",
        "stat.nonpar.mannwhitney",
        "stat.nonpar.kruskal",
        "stat.nonpar.mood",
        "stat.nonpar.friedman",
        "stat.nonpar.runs",
    ];
    for id in ids {
        let a = reg.get(id).unwrap_or_else(|| panic!("{id} missing"));
        assert_eq!(&a.menu[..2], ["Stat", "Nonparametrics"], "{id}");
    }
}

#[test]
fn sign_and_wilcoxon_agree_with_the_library_and_support_paired_data() {
    let s = run("stat.nonpar.sign", &[("var", vars(&[0])), ("median", Value::Num(5.0))]);
    let want = sign_test(&X, 5.0, Alternative::TwoSided, 0.95).unwrap().report_section();
    assert_eq!(text(&s).lines().skip(5).collect::<Vec<_>>(), text(&want).lines().skip(5).collect::<Vec<_>>());
    // Paired: x − y.
    let d: Vec<f64> = X.iter().zip(&Y).map(|(a, b)| a - b).collect();
    let s = run("stat.nonpar.wilcoxon", &[("var", vars(&[0])), ("minus", vars(&[1]))]);
    let w = wilcoxon_signed_rank(&d, 0.0, Alternative::TwoSided, 0.95, SignRankMethod::Auto, false).unwrap();
    let t = text(&s);
    assert!(t.contains("Wilcoxon Signed Rank Test: x - y"), "{t}");
    assert!(t.contains(&format!("{:.3}", w.p_value)), "{t}");
}

#[test]
fn mann_whitney_options_change_the_method() {
    // Tied samples fall back to the tie-adjusted normal approximation.
    let tied = run("stat.nonpar.mannwhitney", &[("first", vars(&[0])), ("second", vars(&[1]))]);
    assert!(text(&tied).contains("Normal approximation, adjusted for ties"), "{}", text(&tied));
    // Tie-free samples use the exact distribution under Auto.
    let a: Vec<f64> = vec![1.1, 2.3, 3.7, 4.2, 5.9, 6.4];
    let b: Vec<f64> = vec![2.9, 4.8, 5.1, 7.3, 8.8, 9.6];
    let mut g = data_with(vec![Column::numeric("a", a.clone()), Column::numeric("b", b.clone())]);
    // Columns are shorter than the worksheet's 12 rows; pad with missing.
    let s = run_on(&mut g, "stat.nonpar.mannwhitney", &[("first", vars(&[6])), ("second", vars(&[7]))])
        .unwrap_or_else(|e| panic!("{e}"));
    let w = mann_whitney(&a, &b, Alternative::TwoSided, 0.95, SignRankMethod::Auto, false).unwrap();
    assert!(text(&s).contains(&format!("{:.3}", w.p_value)), "{}", text(&s));
    let n = run_on(
        &mut g,
        "stat.nonpar.mannwhitney",
        &[("first", vars(&[6])), ("second", vars(&[7])), ("method", Value::Choice(2))],
    )
    .unwrap();
    assert!(text(&n).contains("Normal approximation"), "{}", text(&n));
}

#[test]
fn k_sample_tests_use_the_factor_to_form_groups() {
    let groups = |lvl: &str| -> Vec<f64> { (0..12).filter(|&i| GROUP[i] == lvl).map(|i| X[i]).collect() };
    let g = vec![groups("a"), groups("b"), groups("c")];
    let kw = kruskal_wallis(&g).unwrap();
    let s = run("stat.nonpar.kruskal", &[("response", vars(&[0])), ("factor", vars(&[2]))]);
    let t = text(&s);
    assert!(t.contains("Kruskal-Wallis Test: x versus group"), "{t}");
    assert!(t.contains(&format!("{:.3}", kw.h)), "{t}");
    for lvl in ["a", "b", "c"] {
        assert!(t.contains(lvl));
    }
    let s = run("stat.nonpar.mood", &[("response", vars(&[0])), ("factor", vars(&[2]))]);
    let t = text(&s);
    let mm = mood_median(&g).unwrap();
    assert!(t.contains("Mood Median Test") && t.contains(&format!("{:.3}", mm.chi_square)), "{t}");
}

#[test]
fn friedman_builds_the_block_matrix_and_rejects_bad_designs() {
    let s = run("stat.nonpar.friedman", &[("response", vars(&[3])), ("treatment", vars(&[4])), ("block", vars(&[5]))]);
    let t = text(&s);
    // Every block ranks t1 < t2 < t3 except one that ties 7.5 against... compare with the library.
    let rows: Vec<Vec<f64>> =
        [[5.0, 7.0, 9.0], [4.0, 8.0, 8.5], [6.0, 7.5, 10.0], [5.5, 6.5, 9.5]].iter().map(|r| r.to_vec()).collect();
    let f = friedman(&rows).unwrap();
    assert!(t.contains(&format!("{:.3}", f.statistic)) && t.contains("t3"), "{t}");
    // An incomplete design is refused with the missing cell named.
    let mut treat: Vec<String> = (0..12).map(|i| ["t1", "t2", "t3"][i % 3].to_string()).collect();
    treat[7] = "t1".into(); // block b2 now has t1 twice and no t2
    let mut g = data_with(vec![Column::text("treat2", treat)]);
    let e = run_on(
        &mut g,
        "stat.nonpar.friedman",
        &[("response", vars(&[3])), ("treatment", vars(&[6])), ("block", vars(&[5]))],
    )
    .unwrap_err();
    assert!(e.contains("more than one observation") || e.contains("no observation"), "{e}");
}

#[test]
fn runs_test_about_mean_median_and_k() {
    let mean = X.iter().sum::<f64>() / 12.0;
    let want = runs_test(&X, mean).unwrap();
    let s = run("stat.nonpar.runs", &[("var", vars(&[0]))]);
    assert!(text(&s).contains(&want.runs.to_string()) && text(&s).contains("Runs Test: x"), "{}", text(&s));
    let s = run("stat.nonpar.runs", &[("var", vars(&[0])), ("about", Value::Choice(2)), ("k", Value::Num(5.0))]);
    assert!(text(&s).contains("5.00000") || text(&s).contains('5'), "{}", text(&s));
    let s = run("stat.nonpar.runs", &[("var", vars(&[0])), ("about", Value::Choice(1))]);
    assert!(text(&s).contains("Runs Test"));
}

#[test]
fn input_errors_are_reported() {
    // No variable selected.
    assert!(try_run("stat.nonpar.sign", &[]).is_err());
    // A factor with one level only.
    let mut g = data_with(vec![Column::text("one", vec!["a".to_string(); 12])]);
    let e = run_on(&mut g, "stat.nonpar.kruskal", &[("response", vars(&[0])), ("factor", vars(&[6]))]).unwrap_err();
    assert!(e.contains("fewer than two levels"), "{e}");
    // Everything equals the null median.
    let mut g = data_with(vec![Column::numeric("flat", vec![2.0; 12])]);
    assert!(run_on(&mut g, "stat.nonpar.sign", &[("var", vars(&[6])), ("median", Value::Num(2.0))]).is_err());
}

#[test]
fn scale_trend_and_distribution_dialogs_match_the_library() {
    let ids = [
        "stat.nonpar.ks2",
        "stat.nonpar.ansari",
        "stat.nonpar.moodscale",
        "stat.nonpar.normalscores",
        "stat.nonpar.mannkendall",
    ];
    let reg = Registry::standard();
    for id in ids {
        assert_eq!(&reg.get(id).unwrap().menu[..2], ["Stat", "Nonparametrics"], "{id}");
    }
    let two = [("first", vars(&[0])), ("second", vars(&[1]))];
    let ks = ks_two_sample(&X, &Y, Alternative::TwoSided).unwrap();
    let t = text(&run("stat.nonpar.ks2", &two));
    assert!(t.contains("Two-Sample Kolmogorov-Smirnov: x vs y") && t.contains(&format!("{:.3}", ks.p_value)), "{t}");
    let ab = ansari_bradley(&X, &Y).unwrap();
    let t = text(&run("stat.nonpar.ansari", &two));
    assert!(t.contains(&format!("{:.3}", ab.p_value)), "{t}");
    let mo = mood_scale(&X, &Y).unwrap();
    let t = text(&run("stat.nonpar.moodscale", &two));
    assert!(t.contains(&format!("{:.3}", mo.p_value)), "{t}");
    let t = text(&run("stat.nonpar.mannkendall", &[("var", vars(&[0]))]));
    let mk = mann_kendall(&X, Alternative::TwoSided).unwrap();
    assert!(t.contains("Sen slope") && t.contains(&format!("{:.3}", mk.p_value)), "{t}");
    let t = text(&run("stat.nonpar.normalscores", &[("response", vars(&[0])), ("factor", vars(&[2]))]));
    let ns = normal_scores(&[
        ("a", &[X[0], X[3], X[6], X[9]][..]),
        ("b", &[X[1], X[4], X[7], X[10]][..]),
        ("c", &[X[2], X[5], X[8], X[11]][..]),
    ])
    .unwrap();
    assert!(t.contains(&format!("{:.3}", ns.p_value)), "{t}");
    assert!(try_run("stat.nonpar.ks2", &[("first", vars(&[0]))]).is_err());
}
