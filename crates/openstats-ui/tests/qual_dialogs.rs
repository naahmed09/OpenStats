//! Phase 5c dialogs: control charts, capability, sixpack, distribution identification and
//! measurement-system studies, run through the registry.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_stats::report::{Block, Reportable, Section, SessionReport};
use openstats_stats::spc::attributes::{laney_p, p_chart_with, LimitMethod};
use openstats_stats::spc::charts::{i_mr, xbar_r, ChartOptions};
use openstats_stats::spc::rules::SpecialCauseTests;
use openstats_ui::dialogs::{validate, Registry, Value, Values};
use openstats_ui::grid::GridController;

const N: usize = 40;

fn rng(i: usize, k: u64) -> f64 {
    let mut x = ((i as u64) << 8 | k).wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn vals() -> Vec<f64> {
    (0..N).map(|i| 10.0 + (rng(i, 1) - 0.5) * 2.0 + if i >= 30 { 1.2 } else { 0.0 }).collect()
}
fn skew() -> Vec<f64> {
    (0..N).map(|i| 1.0 - rng(i, 5).ln()).collect()
}
fn counts() -> Vec<f64> {
    (0..N).map(|i| (rng(i, 2) * 8.0).floor() + 1.0).collect()
}
fn sizes() -> Vec<f64> {
    (0..N).map(|i| 100.0 + (rng(i, 3) * 20.0).floor()).collect()
}
fn stages() -> Vec<String> {
    (0..N).map(|i| if i < 20 { "a" } else { "b" }.to_string()).collect()
}

// 0 v, 1 skew, 2 cnt, 3 size, 4 stage, 5 m1, 6 m2
fn data() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("v", vals())).unwrap();
    ws.add_column(Column::numeric("skew", skew())).unwrap();
    ws.add_column(Column::numeric("cnt", counts())).unwrap();
    ws.add_column(Column::numeric("size", sizes())).unwrap();
    ws.add_column(Column::text("stage", stages())).unwrap();
    ws.add_column(Column::numeric("ma_", (0..N).map(|i| rng(i, 6)).collect())).unwrap();
    ws.add_column(Column::numeric("mb_", (0..N).map(|i| rng(i, 7) + 0.3 * rng(i, 6)).collect())).unwrap();
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
fn charts(s: &Section) -> usize {
    s.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count()
}
fn column(g: &GridController, name: &str) -> Vec<f64> {
    let ws = g.worksheet();
    let c = (0..ws.columns.len()).find(|&c| ws.columns[c].name() == name).unwrap_or_else(|| panic!("no column {name}"));
    (0..ws.columns[c].len()).map(|i| ws.columns[c].data.get(i).as_f64()).collect()
}

// ----------------------------------------------------------------------------- variables charts

#[test]
fn imr_dialog_matches_the_library_and_stores_flags() {
    let mut g = data();
    let s = run(&mut g, "stat.spc.imr", &[("var", vars(&[0])), ("store_chart", on())]);
    let want = i_mr(&vals(), &ChartOptions::default()).unwrap().report_section();
    assert_eq!(charts(&s), charts(&want));
    let ucl = column(&g, "UCL");
    assert_eq!(ucl.len(), N);
    assert!(ucl.iter().all(|x| (x - ucl[0]).abs() < 1e-12));
    assert!(column(&g, "FLAGGED").iter().all(|&x| x == 0.0 || x == 1.0));
    assert_eq!(column(&g, "UCL_MR").len(), N);
}

#[test]
fn imr_rules_choice_changes_the_tests() {
    let mut g = data();
    let one = run(&mut g, "stat.spc.imr", &[("var", vars(&[0]))]);
    let all = run(&mut g, "stat.spc.imr", &[("var", vars(&[0])), ("rules", ch(1))]);
    let o = ChartOptions { tests: SpecialCauseTests::nelson(), ..Default::default() };
    let want = i_mr(&vals(), &o).unwrap();
    let mut ws = want.report_section();
    ws.heading = "I-MR Chart of v".into();
    assert_eq!(text(&all), text(&ws));
    assert!(!text(&one).is_empty());
}

#[test]
fn imr_known_parameters_and_stages() {
    let mut g = data();
    let s = run(&mut g, "stat.spc.imr", &[("var", vars(&[0])), ("known_center", num(10.0)), ("known_sigma", num(0.5))]);
    assert!(text(&s).contains("10"));
    let st = run(&mut g, "stat.spc.imr", &[("var", vars(&[0])), ("stages", vars(&[4])), ("store_chart", on())]);
    assert!(text(&st).contains("by stage"));
    let ucl = column(&g, "UCL");
    assert!((ucl[0] - ucl[10]).abs() < 1e-12 && (ucl[0] - ucl[30]).abs() > 1e-9);
}

#[test]
fn xbar_dialog_matches_the_library() {
    let mut g = data();
    let s = run(&mut g, "stat.spc.xbar", &[("var", vars(&[0])), ("size", num(5.0))]);
    let groups: Vec<Vec<f64>> = vals().chunks(5).map(<[f64]>::to_vec).collect();
    let want = xbar_r(&groups, &ChartOptions::default()).unwrap();
    assert_eq!(charts(&s), charts(&want.report_section()));
    let sd = run(&mut g, "stat.spc.xbar", &[("var", vars(&[0])), ("size", num(5.0)), ("sigma", ch(1))]);
    assert!(text(&sd).contains("Xbar-S"));
}

#[test]
fn variables_chart_variants_run() {
    let mut g = data();
    for (id, set) in [
        ("stat.spc.between_within", vec![("var", vars(&[0])), ("size", num(5.0))]),
        ("stat.spc.between_within", vec![("var", vars(&[0])), ("size", num(5.0)), ("within", ch(1))]),
        ("stat.spc.zone", vec![("var", vars(&[0])), ("size", num(1.0))]),
        ("stat.spc.zone", vec![("var", vars(&[0])), ("size", num(5.0))]),
        ("stat.spc.zmr", vec![("var", vars(&[0])), ("parts", vars(&[4]))]),
        ("stat.spc.zmr", vec![("var", vars(&[0])), ("parts", vars(&[4])), ("sigma_by", ch(1))]),
        ("stat.spc.boxcox", vec![("var", vars(&[1])), ("size", num(1.0))]),
        ("stat.spc.boxcox", vec![("var", vars(&[1])), ("size", num(5.0))]),
    ] {
        let s = run(&mut g, id, &set);
        assert!(charts(&s) >= 1, "{id}");
    }
}

// ----------------------------------------------------------------------------- attribute charts

#[test]
fn p_chart_matches_the_library_in_both_limit_methods() {
    let mut g = data();
    for (m, lm) in [(0, LimitMethod::Normal), (1, LimitMethod::Exact)] {
        let s = run(&mut g, "stat.spc.p", &[("var", vars(&[2])), ("sizes", vars(&[3])), ("limits", ch(m))]);
        let want = p_chart_with(&counts(), &sizes(), &ChartOptions::default(), lm).unwrap();
        assert_eq!(charts(&s), charts(&want.report_section()));
        assert!(text(&s).contains("P Chart of cnt"));
    }
}

#[test]
fn laney_and_the_other_attribute_charts_run() {
    let mut g = data();
    let l = run(&mut g, "stat.spc.laney_p", &[("var", vars(&[2])), ("sizes", vars(&[3])), ("store_chart", on())]);
    let want = laney_p(&counts(), &sizes(), &ChartOptions::default()).unwrap();
    assert_eq!(charts(&l), charts(&want.report_section()));
    assert_eq!(column(&g, "ZSCORE").len(), N);
    for (id, set) in [
        ("stat.spc.np", vec![("var", vars(&[2])), ("size", num(150.0))]),
        ("stat.spc.c", vec![("var", vars(&[2]))]),
        ("stat.spc.u", vec![("var", vars(&[2])), ("sizes", vars(&[3]))]),
        ("stat.spc.laney_u", vec![("var", vars(&[2])), ("sizes", vars(&[3]))]),
        ("stat.spc.g", vec![("var", vars(&[2]))]),
        ("stat.spc.t", vec![("var", vars(&[1]))]),
        ("stat.spc.t", vec![("var", vars(&[1])), ("model", ch(1))]),
        ("stat.spc.p", vec![("var", vars(&[2])), ("sizes", vars(&[3])), ("stages", vars(&[4])), ("rules", ch(1))]),
        ("stat.spc.c", vec![("var", vars(&[2])), ("stages", vars(&[4]))]),
    ] {
        let s = run(&mut g, id, &set);
        assert!(charts(&s) >= 1, "{id}");
    }
}

#[test]
fn attribute_chart_input_errors() {
    let mut g = data();
    assert!(try_run(&mut g, "stat.spc.p", &[("var", vars(&[2])), ("size", num(f64::NAN))]).is_err());
    assert!(try_run(&mut g, "stat.spc.p", &[]).is_err());
}

// --------------------------------------------------------------------------- time-weighted

#[test]
fn time_weighted_charts_run_and_store() {
    let mut g = data();
    for (id, set) in [
        ("stat.spc.ma", vec![("var", vars(&[0])), ("length", num(4.0)), ("store_chart", on())]),
        ("stat.spc.ewma", vec![("var", vars(&[0])), ("lambda_w", num(0.3)), ("target", num(10.0))]),
        ("stat.spc.ewma", vec![("var", vars(&[0])), ("size", num(4.0)), ("asymptotic", on())]),
        ("stat.spc.cusum", vec![("var", vars(&[0])), ("target", num(10.0)), ("store_chart", on())]),
        ("stat.spc.cusum", vec![("var", vars(&[0])), ("target", num(10.0)), ("form", ch(1))]),
        ("stat.spc.cusum", vec![("var", vars(&[0])), ("sides", ch(1)), ("head_start", num(0.5))]),
    ] {
        let s = run(&mut g, id, &set);
        assert!(!text(&s).is_empty(), "{id}");
    }
    assert_eq!(column(&g, "CUSUMUPPER").len(), N);
    // the shifted last ten observations make CUSUM signal
    assert!(column(&g, "CUSUMFLAG").contains(&1.0));
}

#[test]
fn generalized_variance_needs_a_subgroup_size() {
    let mut g = data();
    let s = run(&mut g, "stat.spc.genvar", &[("vars", vars(&[5, 6])), ("size", num(5.0)), ("store_chart", on())]);
    assert!(charts(&s) >= 1);
    assert_eq!(column(&g, "GENVAR").len(), 8);
    assert!(try_run(&mut g, "stat.spc.genvar", &[("vars", vars(&[5, 6])), ("size", num(1.0))]).is_err());
}

// ------------------------------------------------------------------------------- capability

#[test]
fn nonnormal_capability_methods_run() {
    let mut g = data();
    for m in 0..9 {
        let s = try_run(
            &mut g,
            "stat.quality.capability_nonnormal",
            &[("var", vars(&[1])), ("usl", num(if m == 1 { 3.0 } else { 8.0 })), ("lsl", num(0.5)), ("method", ch(m))],
        )
        .unwrap_or_else(|e| panic!("method {m}: {e}"));
        assert!(text(&s).contains("Process Capability of skew"), "method {m}");
    }
    assert!(try_run(&mut g, "stat.quality.capability_nonnormal", &[("var", vars(&[1]))]).is_err());
}

#[test]
fn empirical_capability_and_distribution_identification() {
    let mut g = data();
    let r = try_run(
        &mut g,
        "stat.quality.capability_empirical",
        &[("var", vars(&[0])), ("usl", num(13.0)), ("extremes", on())],
    );
    assert!(r.is_ok(), "{r:?}");
    let d = run(&mut g, "stat.quality.distid", &[("var", vars(&[1]))]);
    assert!(text(&d).contains("Individual Distribution Identification of skew"));
}

#[test]
fn capability_intervals_and_sixpack() {
    let mut g = data();
    let plain = run(&mut g, "stat.quality.capability", &[("var", vars(&[0])), ("usl", num(13.0)), ("lsl", num(7.0))]);
    let ci = run(
        &mut g,
        "stat.quality.capability",
        &[("var", vars(&[0])), ("usl", num(13.0)), ("lsl", num(7.0)), ("size", num(5.0)), ("intervals", on())],
    );
    assert!(text(&ci).contains("Confidence Intervals"));
    assert!(!text(&plain).contains("Confidence Intervals"));
    for set in [
        vec![("var", vars(&[0])), ("usl", num(13.0)), ("lsl", num(7.0))],
        vec![("var", vars(&[0])), ("usl", num(13.0)), ("size", num(5.0))],
        vec![("var", vars(&[0])), ("usl", num(13.0)), ("lsl", num(7.0)), ("extended", on()), ("block", num(5.0))],
    ] {
        let s = run(&mut g, "stat.quality.sixpack", &set);
        assert!(text(&s).contains("Sixpack"), "{set:?}");
    }
}

// ------------------------------------------------------------------------------------- MSA

fn msa() -> GridController {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    // nested: operators A and B measure different parts, 3 repeats each
    let (mut m, mut p, mut o) = (Vec::new(), Vec::new(), Vec::new());
    for op in 0..2 {
        for part in 0..3 {
            for rep in 0..3 {
                let i = op * 9 + part * 3 + rep;
                m.push(5.0 + part as f64 + op as f64 * 0.3 + (rng(i, 8) - 0.5) * 0.4);
                p.push(format!("P{}", op * 3 + part));
                o.push(if op == 0 { "A" } else { "B" }.to_string());
            }
        }
    }
    ws.add_column(Column::numeric("meas", m)).unwrap();
    ws.add_column(Column::text("part", p)).unwrap();
    ws.add_column(Column::text("oper", o)).unwrap();
    // linearity: 5 references x 4 measurements with a linear bias
    let (mut r, mut x) = (Vec::new(), Vec::new());
    for (k, reference) in [2.0, 4.0, 6.0, 8.0, 10.0].iter().enumerate() {
        for rep in 0..4 {
            r.push(*reference);
            x.push(reference + 0.1 + 0.02 * reference + (rng(k * 4 + rep, 9) - 0.5) * 0.2);
        }
    }
    ws.add_column(Column::numeric("ref", r)).unwrap();
    ws.add_column(Column::numeric("obs", x)).unwrap();
    GridController::new(wb, 800.0, 400.0)
}

#[test]
fn nested_gage_and_linearity_studies_run() {
    let mut g = msa();
    let s = run(
        &mut g,
        "stat.quality.gage_nested",
        &[("measurement", vars(&[0])), ("part", vars(&[1])), ("operator", vars(&[2]))],
    );
    assert!(text(&s).contains("Nested"));
    let t = run(
        &mut g,
        "stat.quality.gage_nested",
        &[("measurement", vars(&[0])), ("part", vars(&[1])), ("operator", vars(&[2])), ("tolerance", num(4.0))],
    );
    assert!(text(&t).contains("Tolerance") || text(&t).contains("tolerance"));
    assert!(try_run(
        &mut g,
        "stat.quality.gage_nested",
        &[("measurement", vars(&[0])), ("part", vars(&[0])), ("operator", vars(&[2]))]
    )
    .is_err());
    let l = run(&mut g, "stat.quality.gage_linearity", &[("reference", vars(&[3])), ("measured", vars(&[4]))]);
    assert!(text(&l).contains("Linearity"));
}

#[test]
fn attribute_agreement_runs_with_a_standard() {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let (mut sample, mut app, mut rating, mut std) = (vec![], vec![], vec![], vec![]);
    for a in ["Amy", "Bob", "Cal"] {
        for s in 0..10 {
            for rep in 0..2 {
                let truth = s % 2 == 0;
                let flip = rng(s * 7 + rep + a.len(), 4) < 0.1;
                sample.push(format!("S{s}"));
                app.push(a.to_string());
                rating.push(if truth != flip { "pass" } else { "fail" }.to_string());
                std.push(if truth { "pass" } else { "fail" }.to_string());
            }
        }
    }
    for (n, c) in [("sample", sample), ("app", app), ("rating", rating), ("std", std)] {
        ws.add_column(Column::text(n, c)).unwrap();
    }
    let mut g = GridController::new(wb, 800.0, 400.0);
    let s = run(
        &mut g,
        "stat.quality.attribute_agreement",
        &[("sample", vars(&[0])), ("appraiser", vars(&[1])), ("rating", vars(&[2])), ("standard", vars(&[3]))],
    );
    assert!(text(&s).contains("Attribute Agreement Analysis for rating"));
}
