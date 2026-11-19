//! End-to-end runs of authentic-style Minitab macro files (`tests/fixtures/*.mac`).

use openstats_core::history::History;
use openstats_core::{Value, Workbook};
use openstats_script::machine::{Env, Library, Machine, MachineConfig, State};
use openstats_script::{parse, ErrorKind};
use openstats_stats::report::Section;
use std::collections::BTreeSet;

const GLOBAL_CLASSIC: &str = include_str!("fixtures/global_classic.mac");
const GLOBAL_END: &str = include_str!("fixtures/global_end.mac");
const LOCAL_MEND: &str = include_str!("fixtures/local_mend.mac");
const STRINGS: &str = include_str!("fixtures/strings.mac");
const CONTROL_FLOW: &str = include_str!("fixtures/control_flow.mac");
const SUBCOMMANDS: &str = include_str!("fixtures/subcommands.mac");
const NAMED_ARGS: &str = include_str!("fixtures/named_args.mac");
const IMPLICIT: &str = include_str!("fixtures/implicit.mac");

struct Rig {
    wb: Workbook,
    hist: History,
}

impl Rig {
    fn new() -> Rig {
        Rig { wb: Workbook::new(), hist: History::default() }
    }

    fn run_as(&mut self, name: &str, src: &str) -> (State, Vec<Section>) {
        let mut m = Machine::new(name, src, Library::default(), MachineConfig::default()).expect("macro compiles");
        let mut env = Env { wb: &mut self.wb, hist: &mut self.hist };
        let st = m.run(&mut env, &BTreeSet::new());
        if matches!(st, State::Failed(_)) {
            m.cleanup(&mut env);
        }
        (st, m.take_output())
    }

    fn ok(&mut self, src: &str) -> Vec<Section> {
        let (st, out) = self.run_as("test.mac", src);
        assert_eq!(st, State::Finished, "macro failed: {st:?}");
        out
    }

    fn k(&self, k: usize) -> Value {
        self.wb.active_sheet().unwrap().constants.get(&k).cloned().unwrap_or(Value::Missing)
    }

    fn kn(&self, k: usize) -> f64 {
        self.k(k).as_f64()
    }

    fn col(&self, c: usize) -> Vec<f64> {
        let ws = self.wb.active_sheet().unwrap();
        let col = ws.column(c).unwrap();
        (0..col.len()).map(|i| col.get(i).as_f64()).collect()
    }
}

fn text_of(sections: &[Section]) -> String {
    sections.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join("\n")
}

#[test]
fn classic_gmacro_with_name_on_the_next_line() {
    let mut r = Rig::new();
    r.ok(GLOBAL_CLASSIC);
    assert_eq!(r.col(0), vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    assert_eq!(r.kn(10), 55.0);
    let name = r.wb.active_sheet().unwrap().column(0).unwrap().props.name.clone();
    assert_eq!(name, "Square");
    // The MCONSTANT temporaries are gone again.
    assert!(r.wb.active_sheet().unwrap().constants.keys().all(|&k| k < 100), "no leaked macro variables");
}

#[test]
fn gmacro_closed_by_end_with_the_name_on_the_header_line() {
    let mut r = Rig::new();
    r.ok(GLOBAL_END);
    assert_eq!(r.col(1), vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0]);
    assert_eq!(r.kn(11), 56.0);
}

#[test]
fn local_macros_with_mend_and_positional_arguments() {
    let mut r = Rig::new();
    r.ok(LOCAL_MEND);
    assert_eq!(r.col(0), vec![15.0, 25.0, 35.0, 45.0]);
}

#[test]
fn string_variables_with_single_and_double_quotes() {
    let mut r = Rig::new();
    let out = r.ok(STRINGS);
    assert_eq!(r.k(20), Value::Text("Hello, Minitab!".into()));
    assert_eq!(r.kn(21), 1.0);
    let shown = text_of(&out);
    assert!(shown.contains("Hello, Minitab!"), "{shown}");
    // (The sections are shown with Debug formatting, which escapes the quotes.)
    assert!(shown.contains(r#"She said \"hi\""#), "{shown}");
}

#[test]
fn control_flow_loops_conditions_and_labels() {
    let mut r = Rig::new();
    r.ok(CONTROL_FLOW);
    assert_eq!(r.kn(30), 20.0, "5 outer x 4 inner iterations");
    assert_eq!(r.kn(31), 105.5);
    assert_eq!(r.kn(32), 4.0);
    assert_eq!(r.kn(33), 7.0);
    assert_eq!(r.kn(34), 47.0);
}

#[test]
fn subcommands_classic_one_line_and_modern_forms() {
    let mut r = Rig::new();
    let out = r.ok(SUBCOMMANDS);
    let (x, y) = ([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1]);
    let (mx, my) = (3.0, y.iter().sum::<f64>() / 5.0);
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let slope = sxy / sxx;
    let icpt = my - slope * mx;
    let fits = r.col(2);
    for (i, f) in fits.iter().enumerate() {
        assert!((f - (icpt + slope * x[i])).abs() < 1e-9);
    }
    assert_eq!(r.col(4), fits, "the one-line form gives the same fits");
    let r2 = r.kn(2);
    assert!(r2 > 0.99 && r2 < 1.0);
    assert!((r.kn(3) - r2).abs() < 1e-12 && (r.kn(4) - r2).abs() < 1e-12);
    assert!(out.len() >= 4, "three regressions and DESCRIBE report to the Session window");
}

#[test]
fn named_macro_arguments_through_subcommands() {
    let mut r = Rig::new();
    r.ok(NAMED_ARGS);
    assert_eq!(r.col(0), vec![16.0, 26.0, 36.0]);
    assert_eq!(r.kn(40), 1.0, "the optional argument was given by name");
}

#[test]
fn implicit_declarations_and_assignment_without_let() {
    let mut r = Rig::new();
    r.ok(IMPLICIT);
    assert_eq!(r.kn(1), 5.0);
    assert_eq!(r.kn(2), 11.0);
    assert_eq!(r.col(0), vec![10.0]);
    assert_eq!(r.col(1), vec![20.0]);
    assert_eq!(r.wb.active_sheet().unwrap().column(1).unwrap().props.name, "Doubled");
}

#[test]
fn every_fixture_parses_and_reparses_identically() {
    for (name, src) in [
        ("global_classic", GLOBAL_CLASSIC),
        ("global_end", GLOBAL_END),
        ("local_mend", LOCAL_MEND),
        ("strings", STRINGS),
        ("control_flow", CONTROL_FLOW),
        ("subcommands", SUBCOMMANDS),
        ("named_args", NAMED_ARGS),
        ("implicit", IMPLICIT),
    ] {
        let a = parse(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        // Windows line endings give the same program.
        let b = parse(&src.replace('\n', "\r\n")).unwrap_or_else(|e| panic!("{name} (CRLF): {e}"));
        assert_eq!(a, b, "{name}");
    }
}

#[test]
fn runtime_errors_name_the_line_and_the_problem() {
    let mut r = Rig::new();
    let src = "# header\nGMACRO\nbad\n  MCONSTANT n\n  Let n = 4\n  Let K1 = n / 0 + missing\nENDMACRO\n";
    let (st, _) = r.run_as("bad.mac", src);
    let State::Failed(e) = st else { panic!("expected a failure") };
    assert_eq!(e.kind, ErrorKind::Runtime);
    assert_eq!(e.line, 6);
    assert!(e.message.contains("missing"), "{e}");
    assert!(e.to_string().starts_with("line 6:"), "{e}");
}

#[test]
fn syntax_errors_report_exact_lines_and_show_the_source_text() {
    let cases: &[(&str, usize, &str)] = &[
        ("# c\n// c\nNOTE it's fine\nLet K1 = 1 +\n", 4, "expected a value"),
        ("MACRO\nm a\n  Let K1 = 1\n", 1, "is not closed"),
        ("GMACRO\ng\n  IF K1 = 1\nMEND\n", 4, "does not match"),
        ("Do K1 = 1:3\n  Let K2 = 1\nEnd\n", 3, "does not match"),
        ("K1 = \n", 1, "expected a value"),
        ("Goto nowhere\n", 1, "no MLABEL"),
        ("Regress C1;\n", 2, ""),
    ];
    for (src, line, frag) in cases {
        let Err(e) = parse(src).and_then(|p| {
            openstats_script::compile::compile_body("", &[], &p.main, 1)?;
            Ok(p)
        }) else {
            panic!("{src:?} should not parse");
        };
        assert_eq!(e.kind, ErrorKind::Syntax);
        if *line != 2 || !frag.is_empty() {
            assert_eq!(e.line, *line, "{src:?}: {e}");
        }
        assert!(e.message.contains(frag), "{src:?}: {e}");
    }
    let e = parse("# x\nLet K1 = (2\n").unwrap_err();
    assert!(e.to_string().contains("2 | Let K1 = (2"), "{e}");
}

#[test]
fn notes_may_contain_quotes_and_comments_work_everywhere() {
    let mut r = Rig::new();
    r.ok("NOTE It's a \"test\" with 'quotes\nLet K1 = 1 // trailing\n# whole line\n  // indented\nNote another one's\nK2 = K1 + 1 # x\n");
    assert_eq!(r.kn(2), 2.0);
}

#[test]
fn runtime_errors_from_nested_macros_carry_the_call_trace() {
    let mut r = Rig::new();
    let src = "MACRO\ninner x\n  MCONSTANT x\n  Let K1 = 1 / x\n  Let C1(0) = 1\nMEND\nMACRO\nouter\n  %inner 2\nMEND\n%outer\n";
    let (st, _) = r.run_as("t.mac", src);
    let State::Failed(e) = st else { panic!() };
    assert_eq!(e.line, 5);
    assert_eq!(e.trace.len(), 2);
    assert!(e.to_string().contains("in macro inner"), "{e}");
    assert!(e.to_string().contains("5 | Let C1(0) = 1"), "{e}");
}

#[test]
fn mstore_takes_any_kind_and_mtext_locals_are_erased() {
    let mut r = Rig::new();
    r.ok("MACRO\nm a\n  MSTORE a tmp\n  MTEXT t\n  t = 'x'\n  tmp = 3\n  a = tmp + 1\nMEND\n%m K5\nSET C1\n1 2\nEND\n%m C2\n");
    assert_eq!(r.kn(5), 4.0);
    assert_eq!(r.col(1), vec![4.0]);
}
