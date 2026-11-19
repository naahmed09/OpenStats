//! Stat ▸ Basic Statistics dialogs added in Phase 5a, run through the registry. Each result is
//! compared with the library routine it wraps.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::correlation::{covariance_matrix, partial_correlation, MissingData};
use openstats_stats::crosstab::{chi_square_gof, fisher_exact, pearson_chi_square, Crosstab, GofExpected};
use openstats_stats::equivalence::{tost_one_sample, tost_paired, tost_two_sample};
use openstats_stats::hypothesis::{Alternative, VarianceAssumption};
use openstats_stats::outlier::{generalized_esd, grubbs_test, OutlierSide};
use openstats_stats::poisson::{one_sample_poisson, two_sample_poisson};
use openstats_stats::proportion::{one_proportion, two_proportions};
use openstats_stats::report::{Reportable, Section, SessionReport};
use openstats_stats::tolerance::{nonparametric_tolerance, normal_tolerance, NormalMethod, ToleranceSides};
use openstats_stats::variance::{bonett_test, f_test, levene_test, Center};
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 24;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn pad(mut v: Vec<f64>) -> Vec<f64> {
    v.resize(N, f64::NAN);
    v
}

fn padt(mut v: Vec<String>) -> Vec<String> {
    v.resize(N, String::new());
    v
}

fn xs() -> Vec<f64> {
    (0..N).map(|i| rng(i, 1) * 10.0).collect()
}
fn ys() -> Vec<f64> {
    (0..N).map(|i| rng(i, 2) * 30.0 + 2.0).collect()
}
fn zs() -> Vec<f64> {
    (0..N).map(|i| rng(i, 3) * 5.0 + 0.4 * rng(i, 1) * 10.0).collect()
}
fn ws_() -> Vec<f64> {
    (0..N).map(|i| rng(i, 4) * 5.0 + 0.3 * rng(i, 2) * 30.0).collect()
}
fn outcome(yes: usize, no: usize) -> Vec<String> {
    padt((0..yes).map(|_| "yes".to_string()).chain((0..no).map(|_| "no".to_string())).collect())
}
fn val() -> Vec<f64> {
    let (x, y) = (xs(), ys());
    (0..N).map(|i| if i < 12 { x[i] } else { y[i] }).collect()
}
fn grp() -> Vec<String> {
    (0..N).map(|i| if i < 12 { "A" } else { "B" }.to_string()).collect()
}
fn outl() -> Vec<f64> {
    (0..N).map(|i| if i == 5 { 25.0 } else { rng(i, 7) }).collect()
}

// 0 outcome, 1 outcome2, 2 x, 3 y, 4 grp, 5 val, 6 c1, 7 c2, 8 t1, 9 t2, 10 rowcat, 11 colcat,
// 12 obs, 13 lab, 14 expp, 15 z, 16 w, 17 out, 18 g_with_three
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::text("outcome", outcome(12, 8))).unwrap();
    ws.add_column(Column::text("outcome2", outcome(9, 7))).unwrap();
    ws.add_column(Column::numeric("x", xs())).unwrap();
    ws.add_column(Column::numeric("y", ys())).unwrap();
    ws.add_column(Column::text("grp", grp())).unwrap();
    ws.add_column(Column::numeric("val", val())).unwrap();
    ws.add_column(Column::numeric("cnt_a", pad(vec![3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 4.0, 5.0, 2.0, 3.0]))).unwrap();
    ws.add_column(Column::numeric("cnt_b", pad(vec![6.0, 8.0, 5.0, 9.0, 7.0, 6.0, 8.0, 7.0]))).unwrap();
    ws.add_column(Column::numeric("tab_a", pad(vec![10.0, 20.0, 30.0]))).unwrap();
    ws.add_column(Column::numeric("tab_b", pad(vec![15.0, 15.0, 10.0]))).unwrap();
    ws.add_column(Column::text("rowcat", (0..N).map(|i| ["a", "b"][i % 2].to_string()).collect())).unwrap();
    ws.add_column(Column::text("colcat", (0..N).map(|i| ["u", "v", "w"][(i / 2 + i / 5) % 3].to_string()).collect()))
        .unwrap();
    ws.add_column(Column::numeric("obs", pad(vec![18.0, 25.0, 31.0, 26.0]))).unwrap();
    ws.add_column(Column::text("lab", padt(vec!["a".into(), "b".into(), "c".into(), "d".into()]))).unwrap();
    ws.add_column(Column::numeric("expp", pad(vec![0.4, 0.3, 0.2, 0.1]))).unwrap();
    ws.add_column(Column::numeric("z", zs())).unwrap();
    ws.add_column(Column::numeric("w", ws_())).unwrap();
    ws.add_column(Column::numeric("out", outl())).unwrap();
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
fn txt(s: &str) -> Value {
    Value::Text(s.into())
}
fn on() -> Value {
    Value::Bool(true)
}

fn text(s: &Section) -> String {
    let mut r = SessionReport::new("t");
    r.push(s.clone());
    r.render_text()
}

/// The blocks of a dialog result equal those of the library section (headings differ).
fn same_blocks(got: &Section, want: &Section) {
    assert_eq!(got.blocks, want.blocks);
}

#[test]
fn one_proportion_summarized_and_from_a_column() {
    let mut g = data();
    let want = one_proportion(12, 20, 0.5, Alternative::TwoSided, 0.95, false).unwrap().report_section();
    let a = run(&mut g, "stat.basic.prop1", &[("events", num(12.0)), ("trials", num(20.0))]);
    same_blocks(&a, &want);
    assert!(a.heading.contains("One Proportion"));
    // the event defaults to the last level in sorted order ("yes")
    let b = run(&mut g, "stat.basic.prop1", &[("mode", ch(1)), ("var", vars(&[0]))]);
    same_blocks(&b, &want);
    assert!(b.heading.contains("yes"));
    let c = run(
        &mut g,
        "stat.basic.prop1",
        &[
            ("mode", ch(1)),
            ("var", vars(&[0])),
            ("event", txt("no")),
            ("alt", ch(2)),
            ("cc", on()),
            ("p0", num(0.3)),
            ("level", num(90.0)),
        ],
    );
    same_blocks(&c, &one_proportion(8, 20, 0.3, Alternative::Greater, 0.9, true).unwrap().report_section());
    assert!(
        try_run(&mut g, "stat.basic.prop1", &[("mode", ch(1)), ("var", vars(&[0])), ("event", txt("maybe"))]).is_err()
    );
    assert!(try_run(&mut g, "stat.basic.prop1", &[("mode", ch(1))]).is_err());
    assert!(try_run(&mut g, "stat.basic.prop1", &[("events", num(30.0)), ("trials", num(20.0))]).is_err());
}

#[test]
fn two_proportions_in_every_data_layout() {
    let mut g = data();
    let want = two_proportions(12, 20, 9, 16, 0.0, Alternative::TwoSided, 0.95, false).unwrap().report_section();
    let a = run(
        &mut g,
        "stat.basic.prop2",
        &[("events1", num(12.0)), ("trials1", num(20.0)), ("events2", num(9.0)), ("trials2", num(16.0))],
    );
    same_blocks(&a, &want);
    let b = run(&mut g, "stat.basic.prop2", &[("mode", ch(1)), ("first", vars(&[0])), ("second", vars(&[1]))]);
    same_blocks(&b, &want);
    // stacked layout: build a worksheet with one outcome column and one sample column
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let o: Vec<String> = outcome(12, 8)
        .into_iter()
        .take(20)
        .chain(outcome(9, 7).into_iter().take(16))
        .chain(std::iter::repeat_n(String::new(), 0))
        .collect();
    let s: Vec<String> = (0..36).map(|i| if i < 20 { "s1" } else { "s2" }.to_string()).collect();
    ws.add_column(Column::text("o", o)).unwrap();
    ws.add_column(Column::text("s", s)).unwrap();
    let mut g2 = GridController::new(wb, 800.0, 400.0);
    let c = run(&mut g2, "stat.basic.prop2", &[("mode", ch(2)), ("resp", vars(&[0])), ("samples", vars(&[1]))]);
    same_blocks(&c, &want);
    assert!(
        try_run(&mut g, "stat.basic.prop2", &[("mode", ch(2)), ("resp", vars(&[0])), ("samples", vars(&[11]))])
            .is_err(),
        "three samples"
    );
}

#[test]
fn poisson_rates() {
    let mut g = data();
    let want = one_sample_poisson(33, 10.0 * 2.0, 1.5, Alternative::TwoSided, 0.95).unwrap().report_section();
    // c1 sums to 37; use summarized data for the check of the library path
    let a = run(&mut g, "stat.basic.poisson1", &[("total", num(33.0)), ("length", num(20.0)), ("rate0", num(1.5))]);
    same_blocks(&a, &want);
    let b = run(
        &mut g,
        "stat.basic.poisson1",
        &[("mode", ch(1)), ("var", vars(&[6])), ("length", num(2.0)), ("rate0", num(1.5)), ("alt", ch(1))],
    );
    same_blocks(&b, &one_sample_poisson(37, 20.0, 1.5, Alternative::Less, 0.95).unwrap().report_section());
    let w2 = two_sample_poisson(37, 10.0, 56, 8.0, 0.0, Alternative::TwoSided, 0.95).unwrap().report_section();
    let c = run(&mut g, "stat.basic.poisson2", &[("mode", ch(1)), ("first", vars(&[6])), ("second", vars(&[7]))]);
    same_blocks(&c, &w2);
    let d = run(
        &mut g,
        "stat.basic.poisson2",
        &[("total1", num(37.0)), ("length1", num(10.0)), ("total2", num(56.0)), ("length2", num(8.0))],
    );
    same_blocks(&d, &w2);
    assert!(try_run(&mut g, "stat.basic.poisson1", &[("mode", ch(1))]).is_err());
}

#[test]
fn variance_tests() {
    let mut g = data();
    let (x, y) = (xs(), ys());
    let f = f_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap().report_section();
    let lev = levene_test(&[x.clone(), y.clone()], Center::Median).unwrap().report_section();
    let a = run(&mut g, "stat.basic.var2", &[("first", vars(&[2])), ("second", vars(&[3]))]);
    assert_eq!(a.blocks[..f.blocks.len()], f.blocks[..]);
    assert_eq!(a.blocks[a.blocks.len() - lev.blocks.len()..], lev.blocks[..]);
    // stacked layout uses the two levels in sorted order
    let v = val();
    let b = run(&mut g, "stat.basic.var2", &[("mode", ch(1)), ("resp", vars(&[5])), ("samples", vars(&[4]))]);
    let fa = f_test(&v[..12], &v[12..], 1.0, Alternative::TwoSided, 0.95).unwrap().report_section();
    assert_eq!(b.blocks[..fa.blocks.len()], fa.blocks[..]);
    assert!(text(&b).contains("Levene"));
    // k groups
    let groups = vec![v[..12].to_vec(), v[12..].to_vec()];
    let e = run(&mut g, "stat.basic.eqvar", &[("resp", vars(&[5])), ("samples", vars(&[4]))]);
    let bon = bonett_test(&groups, 0.95).unwrap().report_section();
    assert_eq!(e.blocks[..bon.blocks.len()], bon.blocks[..]);
    assert!(try_run(&mut g, "stat.basic.eqvar", &[("resp", vars(&[5])), ("samples", vars(&[0]))]).is_ok());
    assert!(
        try_run(&mut g, "stat.basic.var2", &[("mode", ch(1)), ("resp", vars(&[5])), ("samples", vars(&[10]))]).is_ok()
    );
    assert!(
        try_run(&mut g, "stat.basic.var2", &[("mode", ch(1)), ("resp", vars(&[5])), ("samples", vars(&[11]))]).is_err(),
        "three levels"
    );
}

#[test]
fn outlier_tests() {
    let mut g = data();
    let o = outl();
    let a = run(&mut g, "stat.basic.outlier", &[("var", vars(&[17])), ("side", ch(1))]);
    same_blocks(&a, &grubbs_test(&o, OutlierSide::Larger).unwrap().report_section());
    assert!(text(&a).contains("25"));
    let b = run(
        &mut g,
        "stat.basic.outlier",
        &[("var", vars(&[17])), ("method", ch(2)), ("max", num(4.0)), ("alpha", num(0.01))],
    );
    same_blocks(&b, &generalized_esd(&o, 4, 0.01).unwrap().report_section());
    let c = run(&mut g, "stat.basic.outlier", &[("var", vars(&[17])), ("method", ch(1)), ("side", ch(1))]);
    assert!(text(&c).to_lowercase().contains("dixon"));
    assert!(try_run(&mut g, "stat.basic.outlier", &[]).is_err());
}

#[test]
fn equivalence_tests() {
    let mut g = data();
    let (x, y) = (xs(), ys());
    let a = run(
        &mut g,
        "stat.basic.equiv",
        &[("first", vars(&[2])), ("target", num(5.0)), ("lower", num(-2.0)), ("upper", num(2.0))],
    );
    same_blocks(&a, &tost_one_sample(&x, 5.0, -2.0, 2.0, 0.05).unwrap().report_section());
    let b = run(
        &mut g,
        "stat.basic.equiv",
        &[("kind", ch(1)), ("first", vars(&[2])), ("second", vars(&[3])), ("lower", num(-20.0)), ("upper", num(20.0))],
    );
    same_blocks(&b, &tost_paired(&x, &y, -20.0, 20.0, 0.05).unwrap().report_section());
    let c = run(
        &mut g,
        "stat.basic.equiv",
        &[
            ("kind", ch(2)),
            ("first", vars(&[2])),
            ("second", vars(&[3])),
            ("pooled", on()),
            ("lower", num(-30.0)),
            ("upper", num(30.0)),
            ("alpha", num(0.1)),
        ],
    );
    same_blocks(&c, &tost_two_sample(&x, &y, -30.0, 30.0, VarianceAssumption::Pooled, 0.1).unwrap().report_section());
    let d = run(
        &mut g,
        "stat.basic.equiv",
        &[("kind", ch(2)), ("first", vars(&[2])), ("second", vars(&[3])), ("lower", num(-30.0)), ("upper", num(30.0))],
    );
    same_blocks(&d, &tost_two_sample(&x, &y, -30.0, 30.0, VarianceAssumption::Welch, 0.05).unwrap().report_section());
    assert!(try_run(&mut g, "stat.basic.equiv", &[("kind", ch(1)), ("first", vars(&[2]))]).is_err());
    assert!(try_run(&mut g, "stat.basic.equiv", &[("first", vars(&[2])), ("lower", num(2.0)), ("upper", num(-2.0))])
        .is_err());
}

#[test]
fn tolerance_intervals() {
    let mut g = data();
    let x = xs();
    let a =
        run(&mut g, "stat.basic.tolerance", &[("var", vars(&[2])), ("coverage", num(95.0)), ("confidence", num(99.0))]);
    same_blocks(
        &a,
        &normal_tolerance(&x, 0.95, 0.99, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap().report_section(),
    );
    let b = run(
        &mut g,
        "stat.basic.tolerance",
        &[("var", vars(&[2])), ("method", ch(1)), ("sides", ch(1)), ("coverage", num(90.0))],
    );
    same_blocks(
        &b,
        &normal_tolerance(&x, 0.9, 0.95, ToleranceSides::Upper, NormalMethod::Howe).unwrap().report_section(),
    );
    let c = run(
        &mut g,
        "stat.basic.tolerance",
        &[("var", vars(&[2])), ("method", ch(2)), ("coverage", num(80.0)), ("confidence", num(90.0))],
    );
    same_blocks(&c, &nonparametric_tolerance(&x, 0.8, 0.9, ToleranceSides::TwoSided).unwrap().report_section());
    // 24 observations cannot give a 99 % / 99 % nonparametric interval
    assert!(try_run(
        &mut g,
        "stat.basic.tolerance",
        &[("var", vars(&[2])), ("method", ch(2)), ("coverage", num(99.0)), ("confidence", num(99.0))]
    )
    .is_err());
}

#[test]
fn cross_tabulation() {
    let mut g = data();
    let (r, c) = (
        (0..N).map(|i| ["a", "b"][i % 2].to_string()).collect::<Vec<_>>(),
        (0..N).map(|i| ["u", "v", "w"][(i / 2 + i / 5) % 3].to_string()).collect::<Vec<_>>(),
    );
    let ct = Crosstab::from_labels(&r, &c).unwrap();
    let a = run(
        &mut g,
        "stat.basic.crosstab",
        &[
            ("rowvar", vars(&[10])),
            ("colvar", vars(&[11])),
            ("expected", on()),
            ("row_pct", on()),
            ("col_pct", on()),
            ("tot_pct", on()),
            ("residuals", on()),
            ("lr", on()),
            ("trend", on()),
            ("assoc", on()),
        ],
    );
    let t = text(&a);
    for needle in [
        "Expected counts",
        "Row percentages",
        "Column percentages",
        "Total percentages",
        "Adjusted residuals",
        "Likelihood-Ratio",
        "Linear-by-Linear",
        "Cramer",
    ] {
        assert!(t.contains(needle), "missing {needle}\n{t}");
    }
    let chi = pearson_chi_square(&ct).unwrap().report_section();
    assert!(a.blocks.iter().any(|b| chi.blocks.contains(b)), "Pearson chi-square table present");
    // a table of counts: 3 rows, 2 columns; Fisher needs 2 x 2 handled by the library for r x c too
    let b = run(&mut g, "stat.basic.crosstab", &[("mode", ch(1)), ("counts", vars(&[8, 9])), ("fisher", on())]);
    let tab = Crosstab::from_counts(vec![vec![10, 15], vec![20, 15], vec![30, 10]]).unwrap();
    assert!(b.blocks.iter().any(|x| fisher_exact(&tab).unwrap().report_section().blocks.contains(x)));
    assert!(pearson_chi_square(&tab).is_ok());
    assert!(try_run(&mut g, "stat.basic.crosstab", &[("mode", ch(1)), ("counts", vars(&[8]))]).is_err());
    assert!(try_run(&mut g, "stat.basic.crosstab", &[("rowvar", vars(&[10]))]).is_err());
    assert!(
        try_run(&mut g, "stat.basic.crosstab", &[("rowvar", vars(&[10]),), ("colvar", vars(&[11])), ("mcnemar", on())])
            .is_err(),
        "McNemar needs a square 2 x 2"
    );
}

#[test]
fn goodness_of_fit() {
    let mut g = data();
    let obs = [18.0, 25.0, 31.0, 26.0];
    let labs: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
    let a = run(&mut g, "stat.basic.gof", &[("observed", vars(&[12])), ("names", vars(&[13]))]);
    same_blocks(&a, &chi_square_gof(&labs, &obs, &GofExpected::Equal, 0).unwrap().report_section());
    let b = run(
        &mut g,
        "stat.basic.gof",
        &[
            ("observed", vars(&[12])),
            ("names", vars(&[13])),
            ("expected", ch(1)),
            ("expcol", vars(&[14])),
            ("params", num(1.0)),
        ],
    );
    same_blocks(
        &b,
        &chi_square_gof(&labs, &obs, &GofExpected::Proportions(vec![0.4, 0.3, 0.2, 0.1]), 1).unwrap().report_section(),
    );
    let c = run(&mut g, "stat.basic.gof", &[("observed", vars(&[12])), ("expected", ch(2)), ("expcol", vars(&[14]))]);
    assert!(text(&c).contains("Goodness-of-Fit"));
    assert!(try_run(&mut g, "stat.basic.gof", &[("observed", vars(&[12])), ("expected", ch(1))]).is_err());
}

#[test]
fn covariance_and_partial_correlation() {
    let mut g = data();
    let cols = vec![xs(), ys(), zs()];
    let a = run(&mut g, "stat.basic.covariance", &[("vars", vars(&[2, 3, 15]))]);
    let cm = covariance_matrix(&cols, MissingData::Listwise).unwrap();
    let ncols = match &a.blocks[0] {
        openstats_stats::report::Block::Table(t) => t.columns.len(),
        _ => 0,
    };
    assert_eq!(ncols, 4);
    let want = cm.values[0][1];
    let rendered = text(&a);
    assert!(rendered.contains(&format!("{:.3}", want).trim_end_matches('0').to_string()[..4]), "{rendered}");
    let b = run(&mut g, "stat.basic.covariance", &[("vars", vars(&[2, 3])), ("missing", ch(1))]);
    assert!(text(&b).contains("Observations per pair"));
    // partial correlation of x and y controlling for z and w
    let p = run(&mut g, "stat.basic.partial", &[("vars", vars(&[2, 3])), ("controls", vars(&[15, 16]))]);
    let want = partial_correlation(&xs(), &ys(), &[zs(), ws_()], 0.95).unwrap();
    assert!(text(&p).contains(&format!("{:.4}", want.r)), "{}", text(&p));
    assert!(try_run(&mut g, "stat.basic.partial", &[("vars", vars(&[2, 3])), ("controls", vars(&[2]))]).is_err());
    assert!(try_run(&mut g, "stat.basic.partial", &[("vars", vars(&[2, 3]))]).is_err());
}
