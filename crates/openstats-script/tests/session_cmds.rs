//! EXECUTE, STACK, UNSTACK and OUTFILE (Phase 5k).

use openstats_core::history::History;
use openstats_core::{Value, Workbook};
use openstats_script::machine::{Env, Library, Machine, MachineConfig, State};
use std::collections::BTreeSet;

struct Rig {
    wb: Workbook,
    hist: History,
}

struct Done {
    state: State,
    files: Vec<(String, String)>,
}

impl Rig {
    fn new() -> Rig {
        Rig { wb: Workbook::new(), hist: History::default() }
    }

    fn run(&mut self, src: &str, scripts: &[(&str, &str)]) -> Done {
        let lib = Library { scripts: scripts.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect() };
        let mut m = Machine::new("t.mac", src, lib, MachineConfig::default()).expect("compiles");
        let mut env = Env { wb: &mut self.wb, hist: &mut self.hist };
        let state = m.run(&mut env, &BTreeSet::new());
        Done { state, files: m.take_files() }
    }

    fn ok(&mut self, src: &str) -> Done {
        let d = self.run(src, &[]);
        assert_eq!(d.state, State::Finished, "{:?}\n{src}", d.state);
        d
    }

    fn col(&self, c: usize) -> Vec<Value> {
        let col = self.wb.active_sheet().unwrap().column(c).unwrap();
        (0..col.len()).map(|i| col.get(i)).collect()
    }

    fn nums(&self, c: usize) -> Vec<f64> {
        self.col(c).iter().map(Value::as_f64).collect()
    }
}

fn fail(r: &mut Rig, src: &str) -> String {
    match r.run(src, &[]).state {
        State::Failed(e) => e.message,
        s => panic!("expected failure, got {s:?}"),
    }
}

#[test]
fn stack_columns_with_subscripts() {
    let mut r = Rig::new();
    r.ok("SET C1\n1 2 3\nEND\nSET C2\n4 5 6\nEND\nSTACK C1 C2 INTO C10;\n  SUBSCRIPTS C11.\n");
    assert_eq!(r.nums(9), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(r.nums(10), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    // Blocks: four sources into two outputs.
    r.ok("SET C3\n7 8\nEND\nSET C4\n9 10\nEND\nSTACK C1 C2 C3 C4 INTO C20 C21\n");
    assert_eq!(r.nums(19).len(), 5);
    let e = fail(&mut r, "STACK C1 C2 C3 INTO C30 C31\n");
    assert!(e.contains("multiple"), "{e}");
    let e = fail(&mut r, "STACK C1 C2 C30\n");
    assert!(e.contains("INTO"), "{e}");
}

#[test]
fn unstack_by_subscripts() {
    let mut r = Rig::new();
    r.ok("SET C1\n10 20 30 40 50\nEND\nSET C2\n1 2 1 2 1\nEND\nUNSTACK C1 INTO C5 C6; SUBSCRIPTS C2.\n");
    assert_eq!(r.nums(4), [10.0, 30.0, 50.0]);
    assert_eq!(r.nums(5), [20.0, 40.0]);
    let e = fail(&mut r, "UNSTACK C1 INTO C7; SUBSCRIPTS C2.\n");
    assert!(e.contains("makes 2 columns"), "{e}");
    let e = fail(&mut r, "UNSTACK C1 INTO C7 C8\n");
    assert!(e.contains("SUBSCRIPTS"), "{e}");
}

#[test]
fn execute_runs_stored_scripts_repeatedly() {
    let mut r = Rig::new();
    let lib = [("addone", "K1 = K1 + 1\n")];
    let d = r.run("K1 = 0\nEXECUTE 'addone' 3\nEXECUTE 'addone.mac'\n", &lib);
    assert_eq!(d.state, State::Finished, "{:?}", d.state);
    assert_eq!(r.wb.active_sheet().unwrap().constants.get(&1).map(Value::as_f64), Some(4.0));
    let d = r.run("EXECUTE 'nothere'\n", &lib);
    assert!(matches!(d.state, State::Failed(_)));
    let m = Machine::new("t", "EXECUTE 'addone' 0\n", Library::default(), MachineConfig::default());
    assert!(m.is_err());
    let m = Machine::new("t", "EXECUTE 'a' 2.5\n", Library::default(), MachineConfig::default());
    assert!(m.is_err());
    let m = Machine::new("t", "EXECUTE\n", Library::default(), MachineConfig::default());
    assert!(m.is_err());
}

#[test]
fn outfile_captures_session_output() {
    let mut r = Rig::new();
    let d = r.ok(
        "SET C1\n1 2 3 4\nEND\nOUTFILE 'out.txt'\nMEAN C1 K1\nTEXT 'first'\nDESCRIBE C1\nNOOUTFILE\nTEXT 'not captured'\n",
    );
    assert_eq!(d.files.len(), 1);
    assert_eq!(d.files[0].0, "out.txt");
    let t = &d.files[0].1;
    assert!(t.contains("first") && t.contains("Descriptive"), "{t}");
    assert!(!t.contains("not captured"));
    // Reopening a file continues it, and a second file is separate.
    let d = r.ok("OUTFILE 'a'\nTEXT 'one'\nOUTFILE 'b'\nTEXT 'two'\nOUTFILE 'a'\nTEXT 'three'\n");
    assert_eq!(d.files.len(), 2);
    assert!(d.files[0].1.contains("one") && d.files[0].1.contains("three") && !d.files[0].1.contains("two"));
    assert!(d.files[1].1.contains("two"));
    assert!(fail(&mut r, "OUTFILE\n").contains("file name"));
}
