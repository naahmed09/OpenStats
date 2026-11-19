//! End-to-end behaviour of the interpreter against a real workbook.

use openstats_core::history::History;
use openstats_core::{Value, Workbook};
use openstats_script::machine::{Env, Library, Machine, MachineConfig, State};
use openstats_stats::report::Section;
use std::collections::BTreeSet;

struct Rig {
    wb: Workbook,
    hist: History,
}

impl Rig {
    fn new() -> Rig {
        Rig { wb: Workbook::new(), hist: History::default() }
    }

    fn run_lib(&mut self, src: &str, lib: Library) -> (State, Vec<Section>) {
        let mut m = Machine::new("test.mac", src, lib, MachineConfig::default()).expect("script compiles");
        let mut env = Env { wb: &mut self.wb, hist: &mut self.hist };
        let st = m.run(&mut env, &BTreeSet::new());
        if matches!(st, State::Failed(_)) {
            m.cleanup(&mut env);
        }
        (st, m.take_output())
    }

    fn run(&mut self, src: &str) -> State {
        self.run_lib(src, Library::default()).0
    }

    fn ok(&mut self, src: &str) {
        let st = self.run(src);
        assert_eq!(st, State::Finished, "script failed: {st:?}\n{src}");
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

    fn err(&mut self, src: &str) -> openstats_script::ScriptError {
        match self.run(src) {
            State::Failed(e) => e,
            s => panic!("expected a failure, got {s:?}"),
        }
    }
}

#[test]
fn let_arithmetic_and_functions() {
    let mut r = Rig::new();
    r.ok("LET K1 = 2 + 3 * 4\nLET K2 = SQRT(16) + ABS(-3)\nLET K3 = K1 ** 2 / 4\n");
    assert_eq!(r.kn(1), 14.0);
    assert_eq!(r.kn(2), 7.0);
    assert_eq!(r.kn(3), 49.0);
}

#[test]
fn column_arithmetic_and_elements() {
    let mut r = Rig::new();
    r.ok("SET C1\n1 2 3 4\nEND\nLET C2 = C1 * 10\nLET K1 = C2(3)\nLET C1(2) = 99\n");
    assert_eq!(r.col(1), vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(r.kn(1), 30.0);
    assert_eq!(r.col(0), vec![1.0, 99.0, 3.0, 4.0]);
}

#[test]
fn do_loop_counts_and_fills_a_column() {
    let mut r = Rig::new();
    r.ok("LET K2 = 0\nDO K1 = 1:5\n  LET K2 = K2 + K1\n  LET C1(K1) = K1 ** 2\nENDDO\n");
    assert_eq!(r.kn(2), 15.0);
    assert_eq!(r.col(0), vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    assert_eq!(r.kn(1), 5.0 + 1.0 - 1.0, "the DO variable keeps its last value");
}

#[test]
fn do_with_step_and_negative_step() {
    let mut r = Rig::new();
    r.ok("LET K2 = 0\nDO K1 = 10:1 / -3\n LET K2 = K2 * 10 + K1\nENDDO\nLET K3 = 0\nDO K4 = 1:2 BY 0.5\n LET K3 = K3 + 1\nENDDO\n");
    assert_eq!(r.kn(2), 10741.0);
    assert_eq!(r.kn(3), 3.0);
}

#[test]
fn empty_do_range_skips_the_body() {
    let mut r = Rig::new();
    r.ok("LET K2 = 7\nDO K1 = 5:1\n LET K2 = 0\nENDDO\n");
    assert_eq!(r.kn(2), 7.0);
}

#[test]
fn while_and_if_chain() {
    let mut r = Rig::new();
    r.ok("LET K1 = 1\nLET K2 = 0\nWHILE K1 < 100\n LET K1 = K1 * 2\n LET K2 = K2 + 1\nENDWHILE\n\
          IF K1 > 1000\n LET K3 = 1\nELSEIF K1 > 100\n LET K3 = 2\nELSE\n LET K3 = 3\nENDIF\n");
    assert_eq!(r.kn(1), 128.0);
    assert_eq!(r.kn(2), 7.0);
    assert_eq!(r.kn(3), 2.0);
}

#[test]
fn goto_and_labels() {
    let mut r = Rig::new();
    r.ok("LET K1 = 0\nMLABEL top\nLET K1 = K1 + 1\nIF K1 < 4\n GOTO top\nENDIF\nGOTO done\nLET K1 = -1\nMLABEL done\n");
    assert_eq!(r.kn(1), 4.0);
}

#[test]
fn text_constants_and_string_comparison() {
    let mut r = Rig::new();
    r.ok("LET K1 = \"hello\"\nIF K1 = \"hello\"\n LET K2 = 1\nELSE\n LET K2 = 0\nENDIF\n");
    assert_eq!(r.k(1), Value::Text("hello".into()));
    assert_eq!(r.kn(2), 1.0);
}

#[test]
fn macro_with_arguments_by_reference() {
    let mut r = Rig::new();
    r.ok("MACRO\nScale X Y F\nMCOLUMN X Y\nMCONSTANT F\nLET Y = X * F\nENDMACRO\n\
          SET C1\n1 2 3\nEND\n%Scale C1 C2 10\n");
    assert_eq!(r.col(1), vec![10.0, 20.0, 30.0]);
}

#[test]
fn macro_changes_constant_argument() {
    let mut r = Rig::new();
    r.ok("MACRO\nBump N\nMCONSTANT N\nLET N = N + 1\nENDMACRO\nLET K1 = 5\n%Bump K1\n%Bump K1\n");
    assert_eq!(r.kn(1), 7.0);
}

#[test]
fn macro_locals_are_erased_on_return() {
    let mut r = Rig::new();
    r.ok("MACRO\nSq X\nMCOLUMN X T\nMCONSTANT S\nLET T = X * X\nLET S = SUM(T)\nLET K1 = S\nENDMACRO\n\
          SET C1\n1 2 3\nEND\n%Sq C1\n");
    assert_eq!(r.kn(1), 14.0);
    let ws = r.wb.active_sheet().unwrap();
    assert_eq!(ws.constants.len(), 1, "only K1 remains: {:?}", ws.constants.keys().collect::<Vec<_>>());
    assert!(ws.columns.iter().skip(1).all(|c| c.is_empty()), "temporary columns are emptied");
}

#[test]
fn optional_args_nargs_and_isset() {
    let mut r = Rig::new();
    r.ok("MACRO\nM A\nMCONSTANT A B\nARGS B\nLET K1 = NARGS\nIF ISSET(B)\n LET K2 = B\nELSE\n LET K2 = -1\nENDIF\nENDMACRO\n%M 1\n%M 1 7\n");
    assert_eq!(r.kn(1), 2.0);
    assert_eq!(r.kn(2), 7.0);
    let mut r = Rig::new();
    r.ok("MACRO\nM A\nMCONSTANT A B\nARGS B\nLET K1 = NARGS\nIF ISSET(B)\n LET K2 = B\nELSE\n LET K2 = -1\nENDIF\nENDMACRO\n%M 1\n");
    assert_eq!(r.kn(1), 1.0);
    assert_eq!(r.kn(2), -1.0);
}

#[test]
fn recursion_and_the_depth_limit() {
    let mut r = Rig::new();
    r.ok("MACRO\nFact N R\nMCONSTANT N R\nMCONSTANT M S\nIF N <= 1\n LET R = 1\nELSE\n LET M = N - 1\n %Fact M S\n LET R = N * S\nENDIF\nENDMACRO\n%Fact 5 K1\n");
    assert_eq!(r.kn(1), 120.0);
    let e = r.err("MACRO\nLoop\n%Loop\nENDMACRO\n%Loop\n");
    assert_eq!(e.kind, openstats_script::ErrorKind::Limit);
}

#[test]
fn wrong_argument_count_is_reported() {
    let mut r = Rig::new();
    let e = r.err("MACRO\nTwo A B\nMCONSTANT A B\nENDMACRO\n%Two 1\n");
    assert!(e.message.contains("argument"), "{e}");
    assert_eq!(e.line, 5);
}

#[test]
fn runtime_errors_carry_line_numbers_and_a_trace() {
    let mut r = Rig::new();
    let e = r.err("LET K1 = 1\n\nLET K2 = NOSUCH + 1\n");
    assert_eq!(e.line, 3);
    assert!(e.message.contains("NOSUCH"), "{e}");
    let mut r = Rig::new();
    let e = r.err("MACRO\nBad X\nMCONSTANT X\nLET X = UNDEFINED(1)\nENDMACRO\n%Bad 1\n");
    assert_eq!(e.line, 4);
    assert_eq!(e.trace.len(), 1);
    assert_eq!(e.trace[0].1, 6);
}

#[test]
fn endless_loop_is_stopped_by_the_step_limit() {
    let mut r = Rig::new();
    let mut m = Machine::new(
        "t.mac",
        "LET K1 = 0\nWHILE 1 = 1\n LET K1 = K1 + 1\nENDWHILE\n",
        Library::default(),
        MachineConfig { max_total_steps: 5_000, steps_per_run: 100_000, ..MachineConfig::default() },
    )
    .unwrap();
    let mut env = Env { wb: &mut r.wb, hist: &mut r.hist };
    match m.run(&mut env, &BTreeSet::new()) {
        State::Failed(e) => assert_eq!(e.kind, openstats_script::ErrorKind::Limit),
        s => panic!("{s:?}"),
    }
}

#[test]
fn step_budget_pauses_and_run_resumes() {
    let mut r = Rig::new();
    let mut m = Machine::new(
        "t.mac",
        "DO K1 = 1:50\n LET K2 = K1\nENDDO\n",
        Library::default(),
        MachineConfig { steps_per_run: 20, ..MachineConfig::default() },
    )
    .unwrap();
    let mut env = Env { wb: &mut r.wb, hist: &mut r.hist };
    let mut pauses = 0;
    loop {
        match m.run(&mut env, &BTreeSet::new()) {
            State::Paused => pauses += 1,
            State::Finished => break,
            s => panic!("{s:?}"),
        }
        assert!(pauses < 100);
    }
    assert!(pauses >= 2);
    assert_eq!(env.wb.active_sheet().unwrap().constants[&2].as_f64(), 50.0);
}

#[test]
fn stepping_runs_one_line_at_a_time() {
    let mut r = Rig::new();
    let mut m = Machine::new(
        "t.mac",
        "LET K1 = 1\nLET K1 = K1 + 1\nLET K1 = K1 * 10\n",
        Library::default(),
        MachineConfig::default(),
    )
    .unwrap();
    let mut env = Env { wb: &mut r.wb, hist: &mut r.hist };
    assert_eq!(m.current_line(), Some(1));
    assert_eq!(m.step(&mut env), State::Paused);
    assert_eq!(m.current_line(), Some(2));
    assert_eq!(env.wb.active_sheet().unwrap().constants[&1].as_f64(), 1.0);
    m.step(&mut env);
    m.step(&mut env);
    let st = m.step(&mut env);
    assert_eq!(st, State::Finished);
    assert_eq!(env.wb.active_sheet().unwrap().constants[&1].as_f64(), 20.0);
}

#[test]
fn breakpoints_pause_before_the_line() {
    let mut r = Rig::new();
    let mut m =
        Machine::new("t.mac", "LET K1 = 1\nLET K1 = 2\nLET K1 = 3\n", Library::default(), MachineConfig::default())
            .unwrap();
    let mut env = Env { wb: &mut r.wb, hist: &mut r.hist };
    let bps: BTreeSet<usize> = [3].into_iter().collect();
    assert_eq!(m.run(&mut env, &bps), State::Paused);
    assert_eq!(m.current_line(), Some(3));
    assert!(m.pause_reason.contains("breakpoint"));
    assert_eq!(env.wb.active_sheet().unwrap().constants[&1].as_f64(), 2.0);
    assert_eq!(m.run(&mut env, &bps), State::Finished);
}

#[test]
fn stop_aborts_a_paused_machine() {
    let mut r = Rig::new();
    let mut m =
        Machine::new("t.mac", "LET K1 = 1\nLET K1 = 2\n", Library::default(), MachineConfig::default()).unwrap();
    let mut env = Env { wb: &mut r.wb, hist: &mut r.hist };
    m.step(&mut env);
    m.stop();
    assert_eq!(*m.state(), State::Stopped);
    assert_eq!(m.run(&mut env, &BTreeSet::new()), State::Stopped);
    assert_eq!(env.wb.active_sheet().unwrap().constants[&1].as_f64(), 1.0);
}

#[test]
fn stop_command_ends_the_script() {
    let mut r = Rig::new();
    r.ok("LET K1 = 1\nSTOP\nLET K1 = 2\n");
    assert_eq!(r.kn(1), 1.0);
}

#[test]
fn a_whole_run_is_one_undo_group() {
    let mut r = Rig::new();
    r.hist.begin_group("Run macro");
    r.ok("LET K1 = 1\nSET C1\n1 2 3\nEND\nLET C2 = C1 + 1\n");
    r.hist.end_group().unwrap();
    assert_eq!(r.col(1), vec![2.0, 3.0, 4.0]);
    r.hist.undo(&mut r.wb).unwrap();
    assert!(r.wb.active_sheet().unwrap().constants.is_empty());
    assert!(r.wb.active_sheet().unwrap().columns.iter().all(|c| c.is_empty()));
    assert!(!r.hist.can_undo());
}

// ----------------------------------------------------------------------------- commands

#[test]
fn set_read_name_erase_and_copy() {
    let mut r = Rig::new();
    r.ok("READ C1 C2\n1 10\n2 20\n3 *\nEND\nNAME C1 'x' C2 'y'\nCOPY C1 C3;\n USE 2:3.\nERASE C2\n");
    let ws = r.wb.active_sheet().unwrap();
    assert_eq!(ws.column(0).unwrap().name(), "x");
    assert_eq!(r.col(2), vec![2.0, 3.0]);
    assert_eq!(r.col(1).len(), 0);
}

#[test]
fn set_with_text_makes_a_text_column() {
    let mut r = Rig::new();
    r.ok("SET C1\n\"a\" \"b\" \"a\"\nEND\nNAME C1 'grp'\n");
    let ws = r.wb.active_sheet().unwrap();
    assert_eq!(ws.column(0).unwrap().get(1), Value::Text("b".into()));
    assert_eq!(ws.find("grp"), Some(0));
}

#[test]
fn column_names_can_be_used_in_expressions() {
    let mut r = Rig::new();
    r.ok("SET C1\n1 2 3\nEND\nNAME C1 'height'\nLET 'double' = 'height' * 2\nLET K1 = SUM('double')\n");
    assert_eq!(r.kn(1), 12.0);
    assert!(r.wb.active_sheet().unwrap().find("double").is_some());
}

#[test]
fn sort_by_key_and_descending() {
    let mut r = Rig::new();
    r.ok(
        "READ C1 C2\n3 30\n1 10\n2 20\nEND\nSORT C1 C2 C3 C4;\n BY C1.\nSORT C1 C2 C5 C6;\n BY C1;\n DESCENDING C1.\n",
    );
    assert_eq!(r.col(2), vec![1.0, 2.0, 3.0]);
    assert_eq!(r.col(3), vec![10.0, 20.0, 30.0]);
    assert_eq!(r.col(4), vec![3.0, 2.0, 1.0]);
    assert_eq!(r.col(5), vec![30.0, 20.0, 10.0]);
}

#[test]
fn column_statistics_store_into_constants() {
    let mut r = Rig::new();
    r.ok("SET C1\n2 4 4 4 5 5 7 9\nEND\nMEAN C1 K1\nSTDEV C1 K2\nSUM C1 K3\nMEDIAN C1 K4\nN C1 K5\nMINIMUM C1 K6\nMAXIMUM C1 K7\nRANGE C1 K8\nNMISS C1 K9\n");
    assert_eq!(r.kn(1), 5.0);
    assert!((r.kn(2) - 2.138089935299395).abs() < 1e-12);
    assert_eq!(r.kn(3), 40.0);
    assert_eq!(r.kn(4), 4.5);
    assert_eq!(r.kn(5), 8.0);
    assert_eq!(r.kn(6), 2.0);
    assert_eq!(r.kn(7), 9.0);
    assert_eq!(r.kn(8), 7.0);
    assert_eq!(r.kn(9), 0.0);
}

#[test]
fn describe_and_print_produce_session_output() {
    let mut r = Rig::new();
    let (st, out) =
        r.run_lib("SET C1\n1 2 3 4 5\nEND\nDESCRIBE C1\nPRINT C1\nLET K1 = 3\nPRINT K1\nNOECHO\n", Library::default());
    assert_eq!(st, State::Finished);
    assert_eq!(out.len(), 3, "{out:?}");
    assert!(out[0].heading.contains("Descriptive Statistics"));
    let text: String = out.iter().map(|s| format!("{s:?}")).collect();
    assert!(text.contains("K1"), "{text}");
}

#[test]
fn echo_prints_each_command_line() {
    let mut r = Rig::new();
    let (st, out) = r.run_lib("ECHO\nLET K1 = 1\nLET K2 = 2\n", Library::default());
    assert_eq!(st, State::Finished);
    let text: String = out.iter().map(|s| format!("{s:?}")).collect();
    assert!(text.contains("LET K1 = 1") && text.contains("LET K2 = 2"), "{text}");
}

#[test]
fn correlation_regression_ttest_and_oneway() {
    let mut r = Rig::new();
    let (st, out) = r.run_lib(
        "READ C1 C2\n1 2.1\n2 3.9\n3 6.2\n4 7.8\n5 10.1\n6 12.2\nEND\n\
         CORRELATION C1 C2\n\
         REGRESS C2 1 C1;\n FITS C3;\n RESIDUALS C4;\n MSE K1;\n RSQUARED K2.\n\
         TTEST 5 C2\n\
         SET C5\n1 1 1 2 2 2\nEND\n\
         ONEWAY C2 C5\n",
        Library::default(),
    );
    assert_eq!(st, State::Finished, "{st:?}");
    assert_eq!(out.len(), 4);
    assert!(out[0].heading.starts_with("Correlation"));
    assert!(out[1].heading.starts_with("Regression Analysis"));
    assert!(r.kn(2) > 0.99);
    assert_eq!(r.col(2).len(), 6);
    let fit0 = r.col(2)[0];
    assert!((fit0 - 2.1).abs() < 0.5, "{fit0}");
    let resid: f64 = r.col(3).iter().sum();
    assert!(resid.abs() < 1e-9);
}

#[test]
fn regression_with_wrong_predictor_count_fails() {
    let mut r = Rig::new();
    let e = r.err("READ C1 C2\n1 2\n2 4\n3 7\n4 8\nEND\nREGRESS C2 2 C1\n");
    assert!(e.message.contains("predictor"), "{e}");
}

#[test]
fn cart_forest_and_boost_fit_and_store() {
    let mut r = Rig::new();
    let mut src = String::from("READ C1 C2\n");
    for i in 0..40 {
        let x = i as f64;
        let y = if x < 20.0 { 1.0 } else { 5.0 } + (i % 3) as f64 * 0.1;
        src.push_str(&format!("{x} {y}\n"));
    }
    src.push_str("END\nCART C2 C1;\n FITS C3;\n NODE C4.\nFOREST C2 C1;\n TREES 25;\n SEED 3.\nBOOST C2 C1;\n TREES 30;\n DEPTH 2.\n");
    let (st, out) = r.run_lib(&src, Library::default());
    assert_eq!(st, State::Finished, "{st:?}");
    assert_eq!(out.len(), 3);
    assert!(out[0].heading.starts_with("Regression Tree"));
    assert!(out[1].heading.starts_with("Random Forest"));
    assert!(out[2].heading.starts_with("Gradient Boosting"));
    let fits = r.col(2);
    assert!((fits[0] - 1.1).abs() < 0.3 && (fits[39] - 5.1).abs() < 0.3, "{fits:?}");
    assert_eq!(r.col(3).len(), 40);
}

#[test]
fn classification_tree_on_a_text_response() {
    let mut r = Rig::new();
    let mut src = String::from("READ C1 C2\n");
    for i in 0..30 {
        src.push_str(&format!("{} \"{}\"\n", i, if i < 15 { "low" } else { "high" }));
    }
    src.push_str("END\nCART C2 C1;\n FITS C3.\n");
    let (st, out) = r.run_lib(&src, Library::default());
    assert_eq!(st, State::Finished, "{st:?}");
    assert!(out[0].heading.starts_with("Classification Tree"));
    let ws = r.wb.active_sheet().unwrap();
    assert_eq!(ws.column(2).unwrap().get(0), Value::Text("low".into()));
    assert_eq!(ws.column(2).unwrap().get(29), Value::Text("high".into()));
}

#[test]
fn matrices_copy_transpose_multiply_invert() {
    let mut r = Rig::new();
    r.ok("READ C1 C2\n2 0\n0 4\nEND\nCOPY C1 C2 M1\nINVERT M1 M2\nMULTIPLY M1 M2 M3\nTRANSPOSE M1 M4\nLET K1 = M2(2, 2)\nLET M1(1, 2) = 8\nLET K2 = M1(1, 2)\nCOPY M3 C5 C6\n");
    assert_eq!(r.kn(1), 0.25);
    assert_eq!(r.kn(2), 8.0);
    assert_eq!(r.col(4), vec![1.0, 0.0]);
    assert_eq!(r.col(5), vec![0.0, 1.0]);
}

#[test]
fn singular_matrix_is_an_error() {
    let mut r = Rig::new();
    let e = r.err("READ C1 C2\n1 2\n2 4\nEND\nCOPY C1 C2 M1\nINVERT M1 M2\n");
    assert!(e.message.contains("singular"), "{e}");
}

#[test]
fn unknown_command_is_a_runtime_error_with_its_line() {
    let mut r = Rig::new();
    let e = r.err("LET K1 = 1\nFROBNICATE C1\n");
    assert_eq!(e.line, 2);
}

#[test]
fn command_abbreviations_work() {
    let mut r = Rig::new();
    r.ok("SET C1\n1 2 3\nEND\nDESC C1\nMEAN C1 K1\n");
    assert_eq!(r.kn(1), 2.0);
    assert_eq!(openstats_script::commands::canonical("corr"), Some("CORRELATION"));
    assert_eq!(openstats_script::commands::canonical("zzzz"), None);
}

#[test]
fn macro_library_resolves_percent_calls() {
    let mut r = Rig::new();
    let lib = Library {
        scripts: vec![("helpers.mac".into(), "MACRO\nTriple X\nMCONSTANT X\nLET X = X * 3\nENDMACRO\n".into())],
    };
    let (st, _) = r.run_lib("LET K1 = 4\n%Triple K1\n", lib);
    assert_eq!(st, State::Finished);
    assert_eq!(r.kn(1), 12.0);
}

#[test]
fn library_script_with_the_called_name_runs_its_body() {
    let mut r = Rig::new();
    let lib = Library { scripts: vec![("setup".into(), "LET K5 = 55\n".into())] };
    let (st, _) = r.run_lib("%setup\n", lib);
    assert_eq!(st, State::Finished, "{st:?}");
    assert_eq!(r.kn(5), 55.0);
}

#[test]
fn missing_macro_is_reported() {
    let mut r = Rig::new();
    let e = r.err("%nothere\n");
    assert!(e.message.contains("nothere"), "{e}");
}

#[test]
fn a_script_made_only_of_a_macro_runs_it() {
    let mut r = Rig::new();
    let mut m =
        Machine::new("demo.mac", "GMACRO\nDemo\nLET K1 = 42\nENDMACRO\n", Library::default(), MachineConfig::default())
            .unwrap();
    let mut env = Env { wb: &mut r.wb, hist: &mut r.hist };
    assert_eq!(m.run(&mut env, &BTreeSet::new()), State::Finished);
    assert_eq!(env.wb.active_sheet().unwrap().constants[&1].as_f64(), 42.0);
}

#[test]
fn syntax_errors_are_reported_at_construction() {
    let e =
        Machine::new("x.mac", "DO K1 = 1:3\nLET K2 = 1\n", Library::default(), MachineConfig::default()).unwrap_err();
    assert_eq!(e.kind, openstats_script::ErrorKind::Syntax);
}

#[test]
fn missing_values_in_conditions_fail_clearly() {
    let mut r = Rig::new();
    let e = r.err("IF * = 1\n LET K1 = 1\nENDIF\n");
    assert!(e.message.contains("missing") || e.message.contains("condition"), "{e}");
}

#[test]
fn element_out_of_range_fails_clearly() {
    let mut r = Rig::new();
    let e = r.err("SET C1\n1 2\nEND\nLET K1 = C1(5)\n");
    assert!(e.message.contains("beyond"), "{e}");
}

#[test]
fn highlight_classifies_tokens() {
    use openstats_script::highlight::{highlight_line, Class};
    let spans = highlight_line("LET K1 = C2 * 3 # note");
    let classes: Vec<Class> = spans.iter().map(|s| s.class).collect();
    assert_eq!(
        classes,
        vec![
            Class::Keyword,
            Class::Variable,
            Class::Operator,
            Class::Variable,
            Class::Operator,
            Class::Number,
            Class::Comment
        ]
    );
    assert_eq!(highlight_line("# whole line")[0].class, Class::Comment);
    assert_eq!(highlight_line("NOTE this is a comment")[0].class, Class::Comment);
    let s = highlight_line("DESCRIBE 'my col' \"txt\" %macro");
    assert_eq!(s[0].class, Class::Command);
    assert_eq!(s[1].class, Class::Column);
    assert_eq!(s[2].class, Class::Text);
    assert_eq!(s[3].class, Class::Call);
}
