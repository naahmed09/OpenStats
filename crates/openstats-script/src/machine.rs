//! The interpreter.
//!
//! A [`Machine`] runs one script against an [`Env`] (a workbook and its undo history) one source
//! line at a time, so a host can offer Run, Step and Stop. Variables are the worksheet's own:
//! `C1…` are columns, `K1…` are stored constants, `M1…` are stored matrices, so a script and the
//! worksheet always agree and every change is undoable. Macro arguments are bound by reference
//! (`MCOLUMN` arguments alias the caller's column); `MCOLUMN`/`MCONSTANT`/`MMATRIX` names that
//! are not arguments are local temporaries that are erased when the macro returns.
//!
//! Expressions are lowered to formula-engine source (names resolved, elements read) and
//! evaluated by `openstats-core`; commands (see [`crate::commands`]) call `openstats-stats`.

use crate::ast::*;
use crate::compile::{compile_body, Body, Op};
use crate::error::{ErrorKind, ScriptError};
use crate::parser::parse;
use crate::Result;
use openstats_core::column::{Column, ColumnData};
use openstats_core::formula::{Arr, EvalCtx, FunctionRegistry};
use openstats_core::history::History;
use openstats_core::{Edit, StoredMatrix, Value, Workbook};
use openstats_stats::report::Section;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

/// Highest column number a script may address.
pub const MAX_COLS: usize = 4000;
/// Highest constant number.
pub const MAX_K: usize = 1000;

/// What a script runs against.
#[derive(Debug)]
pub struct Env<'a> {
    /// The project's workbook; the active worksheet is the one scripts see.
    pub wb: &'a mut Workbook,
    /// Its undo history. A host that wants one undo step per run opens a group around the run.
    pub hist: &'a mut History,
}

/// Scripts stored in the project, callable from a running script with `%name`.
#[derive(Debug, Clone, Default)]
pub struct Library {
    /// `(script name, source)`.
    pub scripts: Vec<(String, String)>,
}

/// Limits for one machine.
#[derive(Debug, Clone)]
pub struct MachineConfig {
    /// Instructions executed by one `run` call before it pauses.
    pub steps_per_run: usize,
    /// Instructions a machine may execute in total before it fails.
    pub max_total_steps: u64,
    /// Deepest macro nesting.
    pub max_depth: usize,
    /// Most report sections one machine may produce.
    pub max_sections: usize,
}

impl Default for MachineConfig {
    fn default() -> Self {
        MachineConfig { steps_per_run: 200_000, max_total_steps: 50_000_000, max_depth: 64, max_sections: 20_000 }
    }
}

/// Where a machine is.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    /// Not started.
    Ready,
    /// Stopped between instructions: after a step, at a breakpoint or after the step budget.
    Paused,
    /// The script ended (normally or with `STOP`).
    Finished,
    /// Aborted by [`Machine::stop`].
    Stopped,
    /// A statement failed.
    Failed(ScriptError),
}

impl State {
    /// `true` while the machine can still execute.
    pub fn is_active(&self) -> bool {
        matches!(self, State::Ready | State::Paused)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Binding {
    Col(usize),
    Const(usize),
    Mat(usize),
    Str(usize),
    Lit(Value),
}

/// A resolved name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Resolved {
    Col(usize),
    Const(usize),
    Mat(usize),
    /// A string variable `S1`…
    Str(usize),
    Lit(Value),
    Unknown(String),
}

#[derive(Debug)]
struct LoopState {
    end: f64,
    step: f64,
}

#[derive(Debug)]
pub(crate) struct Frame {
    body: Rc<Body>,
    pc: usize,
    src: usize,
    bind: HashMap<String, Binding>,
    loops: HashMap<usize, LoopState>,
    temp_cols: Vec<usize>,
    temp_consts: Vec<usize>,
    temp_mats: Vec<usize>,
    temp_strs: Vec<usize>,
    /// `MSTORE` locals that have not been given a value (and so a kind) yet.
    lazy: HashSet<String>,
    call_line: usize,
    nargs: usize,
}

#[derive(Debug)]
struct Loaded {
    macros: Vec<(String, Rc<Body>)>,
    main: Rc<Body>,
}

/// A running (or runnable) script.
#[derive(Debug)]
pub struct Machine {
    pub(crate) cfg: MachineConfig,
    macros: HashMap<String, (Rc<Body>, usize)>,
    library: Library,
    loaded: Vec<Option<Rc<Loaded>>>,
    pub(crate) sources: Vec<Vec<String>>,
    pub(crate) frames: Vec<Frame>,
    pub(crate) out: Vec<Section>,
    pub(crate) sections_made: usize,
    /// String variables `S1`… (text is not stored in the worksheet).
    pub(crate) strings: BTreeMap<usize, String>,
    state: State,
    steps_total: u64,
    pub(crate) echo: bool,
    registry: FunctionRegistry,
    /// Why the machine last paused.
    pub pause_reason: String,
    error_main_line: Option<usize>,
    /// Text files written by `OUTFILE` (name, contents), in the order they were opened.
    files: Vec<(String, String)>,
    /// Index into `files` of the file receiving the output now.
    pub(crate) outfile: Option<usize>,
}

fn rt<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| ScriptError::runtime(0, e.to_string()))
}

fn is_identifier(name: &str) -> bool {
    let mut ch = name.chars();
    ch.next().is_some_and(|c| c.is_alphabetic() || c == '_') && ch.all(|c| c.is_alphanumeric() || c == '_' || c == '$')
}

fn stem(name: &str) -> String {
    let n = name.trim();
    let n = n.rsplit(['/', '\\']).next().unwrap_or(n);
    let lower = n.to_ascii_lowercase();
    for ext in [".mac", ".mtb", ".txt"] {
        if lower.ends_with(ext) {
            return n[..n.len() - ext.len()].to_string();
        }
    }
    n.to_string()
}

impl Machine {
    /// Parse and compile `source` (a script called `name`) and get it ready to run.
    ///
    /// The entry point is the top-level statements; a script with none runs the macro named like
    /// the script, else its first `GMACRO`, else its only `MACRO` (which must not need arguments).
    pub fn new(name: &str, source: &str, library: Library, cfg: MachineConfig) -> Result<Machine> {
        let prog = parse(source)?;
        let main = Rc::new(compile_body("", &[], &prog.main, 1)?);
        let mut macros = HashMap::new();
        for m in &prog.macros {
            let b = Rc::new(compile_body(&m.name, &m.params, &m.body, m.line)?);
            macros.insert(m.name.to_ascii_uppercase(), (b, 0));
        }
        let entry: Rc<Body> = if !prog.main.is_empty() {
            main
        } else {
            let want = stem(name).to_ascii_uppercase();
            let pick = prog
                .macros
                .iter()
                .find(|m| m.name.to_ascii_uppercase() == want)
                .or_else(|| prog.macros.iter().find(|m| m.global))
                .or_else(|| prog.macros.first());
            let Some(m) = pick else {
                return Err(ScriptError::syntax(1, 1, "the script has no commands and defines no macro"));
            };
            if !m.params.is_empty() {
                return Err(ScriptError::syntax(
                    m.line,
                    1,
                    format!("the macro `{}` needs arguments; call it with %{} from another script", m.name, m.name),
                ));
            }
            Rc::clone(&macros[&m.name.to_ascii_uppercase()].0)
        };
        let lines: Vec<String> = source.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
        let n_lib = library.scripts.len();
        let mut m = Machine {
            cfg,
            macros,
            library,
            loaded: vec![None; n_lib],
            sources: vec![lines],
            frames: Vec::new(),
            out: Vec::new(),
            sections_made: 0,
            strings: BTreeMap::new(),
            state: State::Ready,
            steps_total: 0,
            echo: false,
            registry: FunctionRegistry::standard(),
            pause_reason: String::new(),
            error_main_line: None,
            files: Vec::new(),
            outfile: None,
        };
        m.frames.push(Frame::new(entry, 0, 0, 0));
        Ok(m)
    }

    /// Current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Instructions executed so far.
    pub fn steps(&self) -> u64 {
        self.steps_total
    }

    /// The source line that will execute next (in the script's own text), if any.
    pub fn current_line(&self) -> Option<usize> {
        if !self.state.is_active() {
            return None;
        }
        let f = self.frames.last()?;
        f.body.code.get(f.pc).map(|i| i.line)
    }

    /// The line of the script's own text that is executing or about to: the current line when the
    /// code is in the script itself, else the line from which the running stored script was
    /// called.
    pub fn current_line_in_script(&self) -> Option<usize> {
        if !self.state.is_active() {
            return None;
        }
        let mut inner: Option<usize> = None;
        for (i, f) in self.frames.iter().enumerate().rev() {
            let here = if i + 1 == self.frames.len() { f.body.code.get(f.pc).map(|c| c.line) } else { inner };
            if f.src == 0 {
                return here;
            }
            inner = Some(f.call_line);
        }
        None
    }

    /// The macro call stack, outermost first: `(macro name or "" for the script, current line)`.
    pub fn call_stack(&self) -> Vec<(String, usize)> {
        self.frames.iter().map(|f| (f.body.name.clone(), f.body.code.get(f.pc).map_or(0, |i| i.line))).collect()
    }

    /// Take the report sections produced since the last call.
    pub fn take_output(&mut self) -> Vec<Section> {
        std::mem::take(&mut self.out)
    }

    /// Abort the run. Temporaries of the active macros are erased on the next `Env` access by
    /// the host calling [`Machine::cleanup`].
    pub fn stop(&mut self) {
        if self.state.is_active() {
            self.state = State::Stopped;
        }
    }

    /// Erase the temporary variables of every active macro (after a stop or a failure).
    pub fn cleanup(&mut self, env: &mut Env) {
        while let Some(f) = self.frames.pop() {
            let _ = self.erase_temps(env, &f);
        }
    }

    /// Execute one source line: instructions run until the line changes or a macro is entered or
    /// left. Returns the new state.
    pub fn step(&mut self, env: &mut Env) -> State {
        if !self.state.is_active() {
            return self.state.clone();
        }
        let start_line = self.current_line();
        let depth = self.frames.len();
        self.state = State::Paused;
        self.pause_reason = "step".into();
        let mut n = 0usize;
        loop {
            self.exec_one(env);
            n += 1;
            if !self.state.is_active() || self.frames.len() != depth || self.current_line() != start_line {
                break;
            }
            if n >= self.cfg.steps_per_run {
                break;
            }
        }
        self.state.clone()
    }

    /// Run until the script ends, fails, reaches one of `breakpoints` (lines of the script's own
    /// text) or has executed `steps_per_run` instructions.
    pub fn run(&mut self, env: &mut Env, breakpoints: &BTreeSet<usize>) -> State {
        if !self.state.is_active() {
            return self.state.clone();
        }
        let mut n = 0usize;
        let mut first = true;
        self.state = State::Paused;
        loop {
            if !first && self.frames.last().is_some_and(|f| f.src == 0) {
                if let Some(l) = self.current_line() {
                    if breakpoints.contains(&l) {
                        self.pause_reason = format!("breakpoint at line {l}");
                        return self.state.clone();
                    }
                }
            }
            first = false;
            self.exec_one(env);
            if !self.state.is_active() {
                return self.state.clone();
            }
            n += 1;
            if n >= self.cfg.steps_per_run {
                self.pause_reason = format!("paused after {n} instructions; run again to continue");
                return self.state.clone();
            }
        }
    }

    // ------------------------------------------------------------------ execution

    fn fail(&mut self, mut e: ScriptError, line: usize) {
        if e.line == 0 {
            e.line = line;
        }
        self.error_main_line = {
            // The line of the script's own text: the error line if the failing code is in it,
            // else the line where the call chain left it.
            let mut inner_line = e.line;
            let mut found = None;
            for f in self.frames.iter().rev() {
                if f.src == 0 {
                    found = Some(inner_line);
                    break;
                }
                inner_line = f.call_line;
            }
            found
        };
        // The text of the failing line comes from the source of the body that was running.
        if let Some(f) = self.frames.last() {
            if let Some(lines) = self.sources.get(f.src) {
                e = e.with_snippet(lines);
            }
        }
        if e.trace.is_empty() {
            // Innermost first: each active macro with the line it was called from.
            for f in self.frames.iter().skip(1).rev() {
                e.trace.push((f.body.name.clone(), f.call_line));
            }
            // The error's own line belongs to the innermost body; the trace tells the callers.
        }
        self.state = State::Failed(e);
    }

    fn exec_one(&mut self, env: &mut Env) {
        if self.steps_total >= self.cfg.max_total_steps {
            let line = self.current_line().unwrap_or(0);
            self.fail(ScriptError::limit(line, "the script executed too many instructions (is a loop endless?)"), line);
            return;
        }
        self.steps_total += 1;
        let Some(frame) = self.frames.last_mut() else {
            self.state = State::Finished;
            return;
        };
        let body = Rc::clone(&frame.body);
        let pc = frame.pc;
        if pc >= body.code.len() {
            if let Err(e) = self.leave_frame(env) {
                let line = body.code.last().map_or(0, |i| i.line);
                self.fail(e, line);
            }
            return;
        }
        frame.pc = pc + 1;
        let instr = &body.code[pc];
        let line = instr.line;
        if let Err(e) = self.dispatch(env, &instr.op, line) {
            if !matches!(self.state, State::Failed(_)) {
                self.fail(e, line);
            }
        }
    }

    fn leave_frame(&mut self, env: &mut Env) -> Result<()> {
        let Some(f) = self.frames.pop() else { return Ok(()) };
        self.erase_temps(env, &f)?;
        if self.frames.is_empty() {
            self.state = State::Finished;
        }
        Ok(())
    }

    fn erase_temps(&mut self, env: &mut Env, f: &Frame) -> Result<()> {
        let sheet = env.wb.active;
        for &c in &f.temp_cols {
            if env.wb.sheet(sheet).is_ok_and(|w| c < w.columns.len()) {
                self.edit(
                    env,
                    "Erase macro column",
                    Edit::ReplaceColumn { sheet, col: c, column: Column::numeric("", Vec::new()) },
                )?;
            }
        }
        for &k in &f.temp_consts {
            if env.wb.sheet(sheet).is_ok_and(|w| w.constants.contains_key(&k)) {
                self.edit(env, "Erase macro constant", Edit::SetConstant { sheet, k, value: None })?;
            }
        }
        for n in &f.temp_strs {
            self.strings.remove(n);
        }
        for &m in &f.temp_mats {
            if env.wb.sheet(sheet).is_ok_and(|w| w.matrices.contains_key(&m)) {
                self.edit(env, "Erase macro matrix", Edit::SetMatrix { sheet, m, matrix: None })?;
            }
        }
        Ok(())
    }

    fn dispatch(&mut self, env: &mut Env, op: &Op, line: usize) -> Result<()> {
        match op {
            Op::Nop => Ok(()),
            Op::Jump(t) => {
                self.top_mut().pc = *t;
                Ok(())
            }
            Op::JumpIfNot(cond, t) => {
                if !self.eval_condition(env, cond)? {
                    self.top_mut().pc = *t;
                }
                Ok(())
            }
            Op::Let { target, value } => {
                self.echo_line(line);
                self.assign(env, target, value)
            }
            Op::Command(c) => {
                self.echo_line(line);
                crate::commands::run_command(self, env, c, line)
            }
            Op::Declare { kind, names } => self.declare(env, *kind, names),
            Op::Args(_) => Ok(()),
            Op::Call { name, args, subs } => {
                self.echo_line(line);
                self.call_macro(env, name, args, subs, line)
            }
            Op::DoInit { id, var, from, to, step, exit } => {
                let a = self.eval_number(env, from, "the start of DO")?;
                let b = self.eval_number(env, to, "the end of DO")?;
                let s = match step {
                    Some(e) => self.eval_number(env, e, "the step of DO")?,
                    None => 1.0,
                };
                if s == 0.0 {
                    return Err(ScriptError::runtime(line, "the DO step cannot be 0"));
                }
                if (s > 0.0 && a > b) || (s < 0.0 && a < b) {
                    self.top_mut().pc = *exit;
                    return Ok(());
                }
                self.set_scalar(env, var, Value::Numeric(a))?;
                self.top_mut().loops.insert(*id, LoopState { end: b, step: s });
                Ok(())
            }
            Op::DoNext { id } => {
                let Some(l) = self.top_mut().loops.get(id).map(|l| (l.end, l.step)) else {
                    return Err(ScriptError::runtime(line, "ENDDO reached without its DO"));
                };
                let Op::DoInit { var, .. } = &self.top().body.code[*id].op else { unreachable!() };
                let var = var.clone();
                let cur = self.eval_number(env, &var, "the DO variable")?;
                let next = cur + l.1;
                if (l.1 > 0.0 && next <= l.0 + 1e-12 * l.0.abs().max(1.0))
                    || (l.1 < 0.0 && next >= l.0 - 1e-12 * l.0.abs().max(1.0))
                {
                    self.set_scalar(env, &var, Value::Numeric(next))?;
                    self.top_mut().pc = *id + 1;
                } else {
                    self.top_mut().loops.remove(id);
                }
                Ok(())
            }
            Op::Return => self.leave_frame(env),
            Op::Stop => {
                while let Some(f) = self.frames.pop() {
                    self.erase_temps(env, &f)?;
                }
                self.state = State::Finished;
                Ok(())
            }
        }
    }

    pub(crate) fn top(&self) -> &Frame {
        self.frames.last().expect("a frame is active")
    }

    fn top_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("a frame is active")
    }

    fn echo_line(&mut self, line: usize) {
        if !self.echo {
            return;
        }
        let src = self.top().src;
        let text = self.sources.get(src).and_then(|s| s.get(line.wrapping_sub(1))).cloned().unwrap_or_default();
        let mut s = Section::new("");
        s.text(format!("MTB > {}", text.trim()));
        self.push_section(s);
    }

    /// Queue a report section, enforcing the output limit.
    /// Start (or continue) sending the output text to the file `name` (`OUTFILE`).
    pub(crate) fn open_outfile(&mut self, name: &str) {
        let i = match self.files.iter().position(|(n, _)| n == name) {
            Some(i) => i,
            None => {
                self.files.push((name.to_string(), String::new()));
                self.files.len() - 1
            }
        };
        self.outfile = Some(i);
    }

    /// Take the files written by `OUTFILE` since the last call: `(name, text)`.
    pub fn take_files(&mut self) -> Vec<(String, String)> {
        self.outfile = None;
        std::mem::take(&mut self.files)
    }

    pub(crate) fn push_section(&mut self, s: Section) {
        if let Some(i) = self.outfile {
            let mut r = openstats_stats::report::SessionReport::new("");
            r.push(s.clone());
            let t = r.render_text();
            self.files[i].1.push_str(&t);
            self.files[i].1.push('\n');
        }
        self.sections_made += 1;
        if self.sections_made <= self.cfg.max_sections {
            self.out.push(s);
        }
    }

    // ------------------------------------------------------------------ edits

    pub(crate) fn edit(&mut self, env: &mut Env, label: &str, e: Edit) -> Result<()> {
        rt(env.hist.execute(env.wb, label, e))
    }

    pub(crate) fn sheet(&self, env: &Env) -> usize {
        env.wb.active
    }

    // ------------------------------------------------------------------ name resolution

    pub(crate) fn resolve(&self, env: &Env, name: &str) -> Result<Resolved> {
        let up = name.to_ascii_uppercase();
        if let Some(b) = self.frames.last().and_then(|f| f.bind.get(&up)) {
            return Ok(match b {
                Binding::Col(i) => Resolved::Col(*i),
                Binding::Const(k) => Resolved::Const(*k),
                Binding::Mat(m) => Resolved::Mat(*m),
                Binding::Str(n) => Resolved::Str(*n),
                Binding::Lit(v) => Resolved::Lit(v.clone()),
            });
        }
        let num = |rest: &str| -> Option<usize> {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                rest.parse().ok()
            } else {
                None
            }
        };
        if let Some(n) = up.strip_prefix('C').and_then(num) {
            if n == 0 || n > MAX_COLS {
                return Err(ScriptError::runtime(0, format!("`{name}` is not a valid column (C1 to C{MAX_COLS})")));
            }
            return Ok(Resolved::Col(n - 1));
        }
        if let Some(n) = up.strip_prefix('K').and_then(num) {
            if n == 0 || n > MAX_K {
                return Err(ScriptError::runtime(0, format!("`{name}` is not a valid constant (K1 to K{MAX_K})")));
            }
            return Ok(Resolved::Const(n));
        }
        if let Some(n) = up.strip_prefix('M').and_then(num) {
            if n == 0 || n > MAX_K {
                return Err(ScriptError::runtime(0, format!("`{name}` is not a valid matrix (M1 to M{MAX_K})")));
            }
            return Ok(Resolved::Mat(n));
        }
        if let Some(n) = up.strip_prefix('S').and_then(num) {
            if n == 0 || n > MAX_K {
                return Err(ScriptError::runtime(0, format!("`{name}` is not a valid string (S1 to S{MAX_K})")));
            }
            return Ok(Resolved::Str(n));
        }
        if let Ok(ws) = env.wb.active_sheet() {
            if let Some(i) = ws.find(name) {
                return Ok(Resolved::Col(i));
            }
        }
        Ok(Resolved::Unknown(name.to_string()))
    }

    /// Resolve a column name given in quotes.
    pub(crate) fn resolve_quoted(&self, env: &Env, name: &str) -> Resolved {
        match env.wb.active_sheet().ok().and_then(|ws| ws.find(name)) {
            Some(i) => Resolved::Col(i),
            None => Resolved::Unknown(name.to_string()),
        }
    }

    pub(crate) fn resolve_expr(&self, env: &Env, e: &Expr) -> Result<Resolved> {
        match e {
            Expr::Word(w) => self.resolve(env, w),
            Expr::Quoted(q) => Ok(self.resolve_quoted(env, q)),
            Expr::Num(x) => Ok(Resolved::Lit(Value::Numeric(*x))),
            Expr::Text(s) => Ok(Resolved::Lit(Value::Text(s.clone()))),
            _ => Err(ScriptError::runtime(0, format!("`{e}` is not a variable"))),
        }
    }

    // ------------------------------------------------------------------ expressions

    fn lit(v: &Value) -> String {
        match v {
            Value::Numeric(x) if x.is_nan() => "*".into(),
            Value::Numeric(x) if x.is_infinite() => "*".into(),
            Value::Numeric(x) if *x < 0.0 => format!("({x})"),
            Value::Numeric(x) => format!("{x}"),
            Value::Text(s) => format!("\"{}\"", s.replace('"', "\"\"")),
            Value::DateTime(t) => format!("{}", openstats_core::datetime::to_excel_serial(*t)),
            Value::Missing => "*".into(),
        }
    }

    fn element(&self, env: &Env, base: &Resolved, idx: &[Value]) -> Result<Value> {
        let ws = env.wb.active_sheet().map_err(|e| ScriptError::runtime(0, e.to_string()))?;
        let index = |v: &Value, what: &str| -> Result<usize> {
            let x = v.as_f64();
            if !x.is_finite() || x < 1.0 || x.fract() != 0.0 {
                return Err(ScriptError::runtime(
                    0,
                    format!("{what} must be a whole number of at least 1, not {}", Self::lit(v)),
                ));
            }
            Ok(x as usize)
        };
        match base {
            Resolved::Col(c) => {
                if idx.len() != 1 {
                    return Err(ScriptError::runtime(0, "a column element needs one row number, as in C1(3)"));
                }
                let r = index(&idx[0], "the row number")?;
                let col =
                    ws.column(*c).map_err(|_| ScriptError::runtime(0, format!("column C{} does not exist", c + 1)))?;
                if r > col.len() {
                    return Err(ScriptError::runtime(
                        0,
                        format!("row {r} is beyond the end of C{} ({} rows)", c + 1, col.len()),
                    ));
                }
                Ok(col.get(r - 1))
            }
            Resolved::Mat(m) => {
                if idx.len() != 2 {
                    return Err(ScriptError::runtime(0, "a matrix element needs a row and a column, as in M1(2, 3)"));
                }
                let (r, c) = (index(&idx[0], "the row number")?, index(&idx[1], "the column number")?);
                let mat =
                    ws.matrices.get(m).ok_or_else(|| ScriptError::runtime(0, format!("matrix M{m} does not exist")))?;
                if r > mat.rows || c > mat.cols {
                    return Err(ScriptError::runtime(
                        0,
                        format!("M{m} is {} by {}; element ({r}, {c}) does not exist", mat.rows, mat.cols),
                    ));
                }
                Ok(Value::Numeric(mat.data[(r - 1) * mat.cols + (c - 1)]))
            }
            _ => Err(ScriptError::runtime(0, "only columns and matrices have elements")),
        }
    }

    /// Source text for the formula engine: names resolved, elements read.
    fn lower(&self, env: &Env, e: &Expr) -> Result<String> {
        Ok(match e {
            Expr::Num(x) => Self::lit(&Value::Numeric(*x)),
            Expr::Text(s) => Self::lit(&Value::Text(s.clone())),
            Expr::Missing => "*".into(),
            Expr::Quoted(q) => match self.resolve_quoted(env, q) {
                Resolved::Col(i) => format!("C{}", i + 1),
                // `'text'` that names no column is a string, so `S1 = 'hello'` works.
                _ => Self::lit(&Value::Text(q.clone())),
            },
            Expr::Word(w) => {
                if w.eq_ignore_ascii_case("NARGS") && !self.top().bind.contains_key("NARGS") {
                    return Ok(self.top().nargs.to_string());
                }
                match self.resolve(env, w)? {
                    Resolved::Col(i) => format!("C{}", i + 1),
                    Resolved::Const(k) => format!("K{k}"),
                    Resolved::Lit(v) => Self::lit(&v),
                    Resolved::Str(n) => Self::lit(&Value::Text(self.strings.get(&n).cloned().unwrap_or_default())),
                    Resolved::Mat(_) => {
                        return Err(ScriptError::runtime(
                            0,
                            format!("the matrix {w} cannot be used as a number; use an element such as {w}(1, 1)"),
                        ))
                    }
                    Resolved::Unknown(n) => {
                        if self.top().lazy.contains(&n.to_ascii_uppercase()) {
                            return Err(ScriptError::runtime(
                                0,
                                format!("`{n}` was declared with MSTORE but has not been given a value yet"),
                            ));
                        }
                        return Err(ScriptError::runtime(
                            0,
                            format!("`{n}` is not a column, constant, string or macro argument"),
                        ));
                    }
                }
            }
            Expr::Index(b, args) => Self::lit(&self.eval_index(env, b, args)?),
            Expr::Unary(UnOp::Neg, x) => format!("(-{})", self.lower(env, x)?),
            Expr::Unary(UnOp::Plus, x) => format!("(+{})", self.lower(env, x)?),
            Expr::Unary(UnOp::Not, x) => format!("(NOT {})", self.lower(env, x)?),
            Expr::Binary(BinOp::Add, a, b) if self.is_text(env, a) || self.is_text(env, b) => {
                // `+` joins text, as in `S1 + " items"`.
                format!("CONCATENATE({}, {})", self.lower(env, a)?, self.lower(env, b)?)
            }
            Expr::Binary(op, a, b) => format!("({} {} {})", self.lower(env, a)?, op.symbol(), self.lower(env, b)?),
            Expr::Call(name, args) => {
                if name.eq_ignore_ascii_case("ISSET") {
                    let [Expr::Word(w)] = args.as_slice() else {
                        return Err(ScriptError::runtime(0, "ISSET takes one macro argument name"));
                    };
                    return Ok(if self.top().bind.contains_key(&w.to_ascii_uppercase()) { "1" } else { "0" }.into());
                }
                // A column or matrix argument followed by (…) is element access.
                if let Ok(r @ (Resolved::Col(_) | Resolved::Mat(_))) = self.resolve(env, name) {
                    if self.top().bind.contains_key(&name.to_ascii_uppercase()) {
                        let vals = self.eval_values(env, args)?;
                        return Ok(Self::lit(&self.element(env, &r, &vals)?));
                    }
                }
                let parts: Result<Vec<String>> = args.iter().map(|a| self.lower(env, a)).collect();
                format!("{name}({})", parts?.join(", "))
            }
        })
    }

    /// Does `e` obviously produce text? (A string, a string variable, a text constant or a
    /// text function.)
    fn is_text(&self, env: &Env, e: &Expr) -> bool {
        match e {
            Expr::Text(_) => true,
            Expr::Quoted(q) => matches!(self.resolve_quoted(env, q), Resolved::Unknown(_)),
            Expr::Word(w) => match self.resolve(env, w) {
                Ok(Resolved::Str(_)) => true,
                Ok(Resolved::Lit(Value::Text(_))) => true,
                Ok(Resolved::Const(k)) => env
                    .wb
                    .active_sheet()
                    .ok()
                    .and_then(|ws| ws.constants.get(&k))
                    .is_some_and(|v| matches!(v, Value::Text(_))),
                _ => false,
            },
            Expr::Binary(BinOp::Add, a, b) => self.is_text(env, a) || self.is_text(env, b),
            Expr::Call(n, _) => {
                matches!(
                    n.to_ascii_uppercase().as_str(),
                    "CONCATENATE"
                        | "UPPER"
                        | "LOWER"
                        | "TRIM"
                        | "LEFT"
                        | "RIGHT"
                        | "MID"
                        | "TEXT"
                        | "PROPER"
                        | "SUBSTITUTE"
                )
            }
            _ => false,
        }
    }

    fn eval_values(&self, env: &Env, args: &[Expr]) -> Result<Vec<Value>> {
        args.iter().map(|a| self.eval_value(env, a)).collect()
    }

    fn eval_index(&self, env: &Env, base: &Expr, args: &[Expr]) -> Result<Value> {
        let r = self.resolve_expr(env, base)?;
        let vals = self.eval_values(env, args)?;
        self.element(env, &r, &vals)
    }

    /// Evaluate to an array (a column or a scalar).
    pub(crate) fn eval_arr(&self, env: &Env, e: &Expr) -> Result<Arr> {
        let src = self.lower(env, e)?;
        let ws = env.wb.active_sheet().map_err(|x| ScriptError::runtime(0, x.to_string()))?;
        let ctx = EvalCtx::new(ws, &self.registry);
        rt(ctx.eval_str(&src))
    }

    /// Evaluate to one value.
    pub(crate) fn eval_value(&self, env: &Env, e: &Expr) -> Result<Value> {
        let a = self.eval_arr(env, e)?;
        if a.len() != 1 {
            return Err(ScriptError::runtime(
                0,
                format!("`{e}` gives {} values where a single value is needed", a.len()),
            ));
        }
        Ok(a.first())
    }

    pub(crate) fn eval_number(&self, env: &Env, e: &Expr, what: &str) -> Result<f64> {
        match self.eval_value(env, e)? {
            Value::Numeric(x) if x.is_finite() => Ok(x),
            Value::Numeric(_) | Value::Missing => {
                Err(ScriptError::runtime(0, format!("{what} is missing or infinite")))
            }
            Value::Text(_) => Err(ScriptError::runtime(0, format!("{what} is text where a number is needed"))),
            Value::DateTime(t) => Ok(openstats_core::datetime::to_excel_serial(t)),
        }
    }

    fn eval_condition(&self, env: &Env, e: &Expr) -> Result<bool> {
        match self.eval_value(env, e)? {
            Value::Numeric(x) => match openstats_core::formula::eval::truth(x) {
                Some(b) => Ok(b),
                None => Err(ScriptError::runtime(0, format!("the condition `{e}` is missing (*)"))),
            },
            Value::Text(_) => Err(ScriptError::runtime(0, format!("the condition `{e}` is text, not a comparison"))),
            _ => Err(ScriptError::runtime(0, format!("the condition `{e}` is not a number"))),
        }
    }

    // ------------------------------------------------------------------ assignment

    pub(crate) fn set_scalar(&mut self, env: &mut Env, target: &Expr, v: Value) -> Result<()> {
        match self.resolve_expr(env, target)? {
            Resolved::Const(k) => self.set_const(env, k, v),
            Resolved::Lit(_) => Err(ScriptError::runtime(0, "a number or text passed to a macro cannot be changed")),
            _ => Err(ScriptError::runtime(0, format!("`{target}` is not a constant"))),
        }
    }

    pub(crate) fn set_const(&mut self, env: &mut Env, k: usize, v: Value) -> Result<()> {
        let sheet = self.sheet(env);
        self.edit(env, "Set constant", Edit::SetConstant { sheet, k, value: Some(v) })
    }

    /// Replace (or create) column `idx` with `data`, keeping its name and format.
    pub(crate) fn write_column(
        &mut self,
        env: &mut Env,
        idx: usize,
        data: ColumnData,
        name: Option<&str>,
    ) -> Result<()> {
        if idx >= MAX_COLS {
            return Err(ScriptError::runtime(0, format!("a worksheet has at most {MAX_COLS} columns")));
        }
        let sheet = self.sheet(env);
        let mut col = match env.wb.sheet(sheet).ok().and_then(|w| w.columns.get(idx)) {
            Some(c) => c.clone(),
            None => Column::new("", ColumnData::Numeric(Vec::new())),
        };
        col.data = data;
        col.props.formula = None;
        if let Some(n) = name {
            col.props.name = n.to_string();
        }
        let e = rt(Edit::store_column(env.wb, sheet, idx, col))?;
        self.edit(env, "Script: store column", e)
    }

    fn assign(&mut self, env: &mut Env, target: &Expr, value: &Expr) -> Result<()> {
        // Element assignment: C1(3) = …, M1(2, 3) = …, x(3) = … for a column argument x.
        let (base, idx): (Option<Expr>, Option<Vec<Expr>>) = match target {
            Expr::Index(b, a) => (Some((**b).clone()), Some(a.clone())),
            Expr::Call(n, a) => (Some(Expr::Word(n.clone())), Some(a.clone())),
            _ => (None, None),
        };
        if let (Some(base), Some(idx)) = (base, idx) {
            let vals = self.eval_values(env, &idx)?;
            let v = self.eval_value(env, value)?;
            return self.assign_element(env, &base, &vals, v);
        }
        match self.resolve_expr(env, target)? {
            Resolved::Const(k) => {
                let v = self.eval_value(env, value)?;
                self.set_const(env, k, v)
            }
            Resolved::Str(n) => {
                let v = self.eval_value(env, value)?;
                let text = match v {
                    Value::Text(s) => s,
                    Value::Numeric(x) if x.is_finite() => format!("{x}"),
                    Value::DateTime(t) => format!("{}", openstats_core::datetime::to_excel_serial(t)),
                    _ => String::new(),
                };
                self.strings.insert(n, text);
                Ok(())
            }
            Resolved::Col(c) => {
                let arr = self.eval_arr(env, value)?;
                let existing = env.wb.active_sheet().ok().and_then(|w| w.columns.get(c)).map_or(0, Column::len);
                let data = if arr.scalar || arr.len() == 1 { arr.broadcast(existing.max(1)).data } else { arr.data };
                self.write_column(env, c, data, None)
            }
            Resolved::Mat(_) => Err(ScriptError::runtime(
                0,
                "a whole matrix is assigned with COPY, MULTIPLY, INVERT or TRANSPOSE; use M1(r, c) for one element",
            )),
            Resolved::Lit(_) => Err(ScriptError::runtime(0, "a number or text passed to a macro cannot be changed")),
            Resolved::Unknown(name) if matches!(target, Expr::Word(_)) && is_identifier(&name) => {
                // Implicit declaration: the first assignment to a new name creates it, as a
                // string, a constant or a column according to what is assigned.
                let arr = self.eval_arr(env, value)?;
                let kind = if arr.scalar || arr.len() == 1 {
                    if matches!(arr.first(), Value::Text(_)) {
                        DeclKind::Text
                    } else {
                        DeclKind::Constant
                    }
                } else {
                    DeclKind::Column
                };
                let persistent = self.frames.len() == 1 && self.top().call_line == 0;
                let b = self.alloc(env, kind, !persistent)?;
                self.top_mut().lazy.remove(&name.to_ascii_uppercase());
                self.top_mut().bind.insert(name.to_ascii_uppercase(), b);
                self.assign(env, target, value)
            }
            Resolved::Unknown(name) => {
                if matches!(target, Expr::Quoted(_)) {
                    let arr = self.eval_arr(env, value)?;
                    let at = env.wb.active_sheet().map_or(0, |w| w.next_free_column());
                    let data = if arr.scalar { arr.broadcast(1).data } else { arr.data };
                    self.write_column(env, at, data, Some(&name))
                } else {
                    Err(ScriptError::runtime(0, format!("`{name}` is not a column, constant or macro argument")))
                }
            }
        }
    }

    fn assign_element(&mut self, env: &mut Env, base: &Expr, idx: &[Value], v: Value) -> Result<()> {
        let r = self.resolve_expr(env, base)?;
        let sheet = self.sheet(env);
        let whole = |v: &Value, what: &str| -> Result<usize> {
            let x = v.as_f64();
            if !x.is_finite() || x < 1.0 || x.fract() != 0.0 {
                return Err(ScriptError::runtime(0, format!("{what} must be a whole number of at least 1")));
            }
            Ok(x as usize)
        };
        match r {
            Resolved::Col(c) => {
                if idx.len() != 1 {
                    return Err(ScriptError::runtime(0, "a column element needs one row number"));
                }
                let row = whole(&idx[0], "the row number")?;
                if c >= env.wb.sheet(sheet).map_or(0, |w| w.columns.len()) {
                    self.write_column(env, c, ColumnData::Numeric(Vec::new()), None)?;
                }
                self.edit(env, "Script: set cell", Edit::SetCell { sheet, col: c, row: row - 1, value: v })
            }
            Resolved::Mat(m) => {
                if idx.len() != 2 {
                    return Err(ScriptError::runtime(0, "a matrix element needs a row and a column"));
                }
                let (row, col) = (whole(&idx[0], "the row number")?, whole(&idx[1], "the column number")?);
                let mut mat = env
                    .wb
                    .sheet(sheet)
                    .ok()
                    .and_then(|w| w.matrices.get(&m).cloned())
                    .ok_or_else(|| ScriptError::runtime(0, format!("matrix M{m} does not exist")))?;
                if row > mat.rows || col > mat.cols {
                    return Err(ScriptError::runtime(
                        0,
                        format!("M{m} is {} by {}; element ({row}, {col}) does not exist", mat.rows, mat.cols),
                    ));
                }
                let x = match v {
                    Value::Numeric(x) => x,
                    Value::Missing => f64::NAN,
                    _ => return Err(ScriptError::runtime(0, "a matrix element must be a number")),
                };
                mat.data[(row - 1) * mat.cols + (col - 1)] = x;
                self.edit(env, "Script: set matrix element", Edit::SetMatrix { sheet, m, matrix: Some(mat) })
            }
            _ => Err(ScriptError::runtime(0, "only columns and matrices have elements")),
        }
    }

    // ------------------------------------------------------------------ declarations and temporaries

    fn declare(&mut self, env: &mut Env, kind: DeclKind, names: &[String]) -> Result<()> {
        let optional: Vec<String> = self
            .top()
            .body
            .code
            .iter()
            .filter_map(|i| {
                if let Op::Args(ns) = &i.op {
                    Some(ns.iter().map(|n| n.to_ascii_uppercase()).collect::<Vec<_>>())
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        for n in names {
            let up = n.to_ascii_uppercase();
            if self.top().bind.contains_key(&up) || optional.contains(&up) {
                continue;
            }
            if kind == DeclKind::Store {
                // Its kind is decided by the first assignment.
                self.top_mut().lazy.insert(up);
                continue;
            }
            let b = self.alloc(env, kind, true)?;
            self.top_mut().bind.insert(up, b);
        }
        Ok(())
    }

    /// Reserve a column, constant number, matrix number or string number for a macro variable.
    /// With `temp` the frame erases it when the macro returns.
    fn alloc(&mut self, env: &Env, kind: DeclKind, temp: bool) -> Result<Binding> {
        Ok(match kind {
            DeclKind::Column => {
                let used = self.used_columns(env);
                let mut c = env.wb.active_sheet().map_or(0, |w| w.next_free_column());
                while used.contains(&c) {
                    c += 1;
                }
                if c >= MAX_COLS {
                    return Err(ScriptError::runtime(0, "no free column for a macro variable"));
                }
                if temp {
                    self.top_mut().temp_cols.push(c);
                }
                Binding::Col(c)
            }
            DeclKind::Constant | DeclKind::Store => {
                let k = self.free_number(env, |w| &w.constants, |f| &f.temp_consts)?;
                if temp {
                    self.top_mut().temp_consts.push(k);
                }
                Binding::Const(k)
            }
            DeclKind::Matrix => {
                let m = self.free_number(env, |w| &w.matrices, |f| &f.temp_mats)?;
                if temp {
                    self.top_mut().temp_mats.push(m);
                }
                Binding::Mat(m)
            }
            DeclKind::Text => {
                let n = (1..=MAX_K)
                    .rev()
                    .find(|n| !self.strings.contains_key(n) && !self.frames.iter().any(|f| f.temp_strs.contains(n)))
                    .ok_or_else(|| ScriptError::runtime(0, "no free string number for a macro variable"))?;
                self.strings.insert(n, String::new());
                if temp {
                    self.top_mut().temp_strs.push(n);
                }
                Binding::Str(n)
            }
        })
    }

    fn used_columns(&self, env: &Env) -> BTreeSet<usize> {
        let mut s: BTreeSet<usize> = BTreeSet::new();
        for f in &self.frames {
            s.extend(f.temp_cols.iter().copied());
            for b in f.bind.values() {
                if let Binding::Col(c) = b {
                    s.insert(*c);
                }
            }
        }
        let _ = env;
        s
    }

    fn free_number<T>(
        &self,
        env: &Env,
        map: impl Fn(&openstats_core::Worksheet) -> &std::collections::BTreeMap<usize, T>,
        temps: impl Fn(&Frame) -> &Vec<usize>,
    ) -> Result<usize> {
        let ws = env.wb.active_sheet().map_err(|e| ScriptError::runtime(0, e.to_string()))?;
        let m = map(ws);
        for k in (1..=MAX_K).rev() {
            if !m.contains_key(&k) && !self.frames.iter().any(|f| temps(f).contains(&k)) {
                return Ok(k);
            }
        }
        Err(ScriptError::runtime(0, "no free constant or matrix number for a macro variable"))
    }

    // ------------------------------------------------------------------ macro calls

    fn load_script(&mut self, idx: usize) -> Result<Rc<Loaded>> {
        if let Some(l) = &self.loaded[idx] {
            return Ok(Rc::clone(l));
        }
        let (name, src) = self.library.scripts[idx].clone();
        let prog = parse(&src).map_err(|e| ScriptError::runtime(0, format!("in the stored script `{name}`: {e}")))?;
        let src_id = self.sources.len();
        self.sources.push(src.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect());
        let mut macros = Vec::new();
        for m in &prog.macros {
            let b = compile_body(&m.name, &m.params, &m.body, m.line)
                .map_err(|e| ScriptError::runtime(0, format!("in the stored script `{name}`: {e}")))?;
            macros.push((m.name.clone(), Rc::new(b)));
        }
        let main = compile_body("", &[], &prog.main, 1)
            .map_err(|e| ScriptError::runtime(0, format!("in the stored script `{name}`: {e}")))?;
        let l = Rc::new(Loaded { macros, main: Rc::new(main) });
        self.loaded[idx] = Some(Rc::clone(&l));
        // Remember which source the bodies came from.
        for (n, b) in &l.macros {
            self.macros.entry(n.to_ascii_uppercase()).or_insert((Rc::clone(b), src_id));
        }
        Ok(l)
    }

    /// Find the body to run for `%name`: macros of the script, then stored scripts.
    fn find_callee(&mut self, name: &str) -> Result<(Rc<Body>, usize)> {
        let want = stem(name).to_ascii_uppercase();
        if let Some((b, s)) = self.macros.get(&want) {
            return Ok((Rc::clone(b), *s));
        }
        for i in 0..self.library.scripts.len() {
            let sname = self.library.scripts[i].0.clone();
            if stem(&sname).to_ascii_uppercase() == want || self.library_defines(i, &want) {
                let l = self.load_script(i)?;
                if let Some((b, s)) = self.macros.get(&want) {
                    return Ok((Rc::clone(b), *s));
                }
                if !l.main.code.is_empty() {
                    let sid = self.source_of(i);
                    return Ok((Rc::clone(&l.main), sid));
                }
                if let Some((_, b)) = l.macros.first() {
                    let sid = self.source_of(i);
                    return Ok((Rc::clone(b), sid));
                }
            }
        }
        Err(ScriptError::runtime(0, format!("there is no macro or stored script named `{name}`")))
    }

    fn library_defines(&self, i: usize, want_upper: &str) -> bool {
        match parse(&self.library.scripts[i].1) {
            Ok(p) => p.macros.iter().any(|m| m.name.to_ascii_uppercase() == want_upper),
            Err(_) => false,
        }
    }

    fn source_of(&self, lib_idx: usize) -> usize {
        // Sources are pushed in load order after the main source; find this script's text.
        let text: Vec<String> =
            self.library.scripts[lib_idx].1.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
        self.sources.iter().position(|s| *s == text).unwrap_or(0)
    }

    fn call_macro(&mut self, env: &mut Env, name: &str, args: &[Arg], subs: &[Sub], line: usize) -> Result<()> {
        if self.frames.len() >= self.cfg.max_depth {
            return Err(ScriptError::limit(
                line,
                format!("macros are nested more than {} deep (recursion without an end?)", self.cfg.max_depth),
            ));
        }
        let (body, src) = self.find_callee(name)?;
        let decl: HashMap<String, DeclKind> = body
            .code
            .iter()
            .filter_map(|i| if let Op::Declare { kind, names } = &i.op { Some((*kind, names)) } else { None })
            .flat_map(|(k, ns)| ns.iter().map(move |n| (n.to_ascii_uppercase(), k)))
            .collect();
        let optional: Vec<String> = body
            .code
            .iter()
            .filter_map(|i| if let Op::Args(ns) = &i.op { Some(ns.clone()) } else { None })
            .flatten()
            .collect();
        let all: Vec<String> = body.params.iter().cloned().chain(optional).collect();
        let required = body.params.len();
        if args.len() > all.len() || (subs.is_empty() && args.len() < required) {
            let range =
                if all.len() == required { format!("{required}") } else { format!("{required} to {}", all.len()) };
            return Err(ScriptError::runtime(
                line,
                format!("the macro `{}` takes {range} argument(s) but was given {}", body.name, args.len()),
            ));
        }
        // Positional arguments first, then the ones given by name (`%m C1; K 3.`).
        let mut given: Vec<Option<&Arg>> = vec![None; all.len()];
        for (i, a) in args.iter().enumerate() {
            given[i] = Some(a);
        }
        for sub in subs {
            let Some(pos) = all.iter().position(|p| p.eq_ignore_ascii_case(&sub.name)) else {
                return Err(ScriptError::runtime(
                    sub.line,
                    format!("the macro `{}` has no argument named `{}`", body.name, sub.name),
                ));
            };
            if given[pos].is_some() {
                return Err(ScriptError::runtime(
                    sub.line,
                    format!("the argument `{}` of macro `{}` is given twice", all[pos], body.name),
                ));
            }
            let [a] = sub.args.as_slice() else {
                return Err(ScriptError::runtime(
                    sub.line,
                    format!("the argument `{}` takes exactly one value", sub.name),
                ));
            };
            given[pos] = Some(a);
        }
        if let Some(missing) = (0..required).find(|&i| given[i].is_none()) {
            return Err(ScriptError::runtime(
                line,
                format!("the macro `{}` needs a value for its argument `{}`", body.name, all[missing]),
            ));
        }
        let mut bind = HashMap::new();
        let mut count = 0;
        for (p, a) in all.iter().zip(&given) {
            let Some(a) = a else { continue };
            count += 1;
            let up = p.to_ascii_uppercase();
            let b = self.bind_arg(env, a, decl.get(&up).copied(), p)?;
            bind.insert(up, b);
        }
        let mut f = Frame::new(body, src, line, count);
        f.bind = bind;
        self.frames.push(f);
        Ok(())
    }

    fn bind_arg(&self, env: &Env, a: &Arg, kind: Option<DeclKind>, param: &str) -> Result<Binding> {
        let resolved = match a {
            Arg::Word(w) => self.resolve(env, w)?,
            Arg::Quoted(q) => match self.resolve_quoted(env, q) {
                // `'text'` that names no column can still be passed as a string.
                Resolved::Unknown(_) if !matches!(kind, Some(DeclKind::Column)) => {
                    Resolved::Lit(Value::Text(q.clone()))
                }
                r => r,
            },
            Arg::Num(x) => Resolved::Lit(Value::Numeric(*x)),
            Arg::Text(s) => Resolved::Lit(Value::Text(s.clone())),
            Arg::Missing => Resolved::Lit(Value::Missing),
            Arg::Paren(e) => Resolved::Lit(self.eval_value(env, e)?),
            Arg::Range(..) => return Err(ScriptError::runtime(0, format!("argument `{param}` cannot be a range"))),
        };
        let bad = |what: &str| ScriptError::runtime(0, format!("argument `{param}` must be {what}"));
        match (kind, resolved) {
            (_, Resolved::Unknown(n)) => {
                Err(ScriptError::runtime(0, format!("`{n}` is not a column, constant or matrix (argument `{param}`)")))
            }
            (Some(DeclKind::Column | DeclKind::Store) | None, Resolved::Col(c)) => Ok(Binding::Col(c)),
            (Some(DeclKind::Constant | DeclKind::Store) | None, Resolved::Const(k)) => Ok(Binding::Const(k)),
            (Some(DeclKind::Matrix | DeclKind::Store) | None, Resolved::Mat(m)) => Ok(Binding::Mat(m)),
            (Some(DeclKind::Text | DeclKind::Store) | None, Resolved::Str(n)) => Ok(Binding::Str(n)),
            (Some(DeclKind::Constant | DeclKind::Store) | None, Resolved::Lit(v)) => Ok(Binding::Lit(v)),
            (Some(DeclKind::Text), Resolved::Lit(Value::Numeric(x))) => Ok(Binding::Lit(Value::Text(format!("{x}")))),
            (Some(DeclKind::Text), Resolved::Lit(v @ Value::Text(_))) => Ok(Binding::Lit(v)),
            (Some(DeclKind::Column), _) => Err(bad("a column")),
            (Some(DeclKind::Constant), _) => Err(bad("a constant or a number")),
            (Some(DeclKind::Matrix), _) => Err(bad("a matrix")),
            (Some(DeclKind::Text), _) => Err(bad("text or a string variable")),
        }
    }

    // ------------------------------------------------------------------ helpers for commands

    /// The stored matrix `m` of the active worksheet.
    pub(crate) fn matrix(&self, env: &Env, m: usize) -> Result<StoredMatrix> {
        env.wb
            .active_sheet()
            .ok()
            .and_then(|w| w.matrices.get(&m).cloned())
            .ok_or_else(|| ScriptError::runtime(0, format!("matrix M{m} does not exist")))
    }

    /// After a failure: the line of the script's own text that was executing, or from which the
    /// failing stored script was called (`None` once the frames have been cleaned up).
    pub fn error_line_in_script(&self) -> Option<usize> {
        self.error_main_line
    }

    /// `true` for a state that ended with an error.
    pub fn failed(&self) -> Option<&ScriptError> {
        if let State::Failed(e) = &self.state {
            Some(e)
        } else {
            None
        }
    }

    /// The kind of error, when the machine failed.
    pub fn error_kind(&self) -> Option<ErrorKind> {
        self.failed().map(|e| e.kind)
    }
}

impl Frame {
    fn new(body: Rc<Body>, src: usize, call_line: usize, nargs: usize) -> Frame {
        Frame {
            body,
            pc: 0,
            src,
            bind: HashMap::new(),
            loops: HashMap::new(),
            temp_cols: Vec::new(),
            temp_consts: Vec::new(),
            temp_mats: Vec::new(),
            temp_strs: Vec::new(),
            lazy: HashSet::new(),
            call_line,
            nargs,
        }
    }
}
