//! Compiler: statement trees → flat, jump-based instruction lists.
//!
//! Flattening makes `GOTO` (even out of nested blocks), single-stepping and breakpoints exact:
//! the interpreter only needs a program counter. Loops compile to `DoInit … DoNext` pairs and
//! `WHILE` / `IF` to conditional jumps. Labels are local to a body (the top level of a script or
//! one macro); a `GOTO` to a missing label is an error at compile time.

use crate::ast::*;
use crate::error::ScriptError;
use crate::Result;
use std::collections::HashMap;

/// One instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `LET`.
    Let {
        /// Assignment target.
        target: Expr,
        /// Value expression.
        value: Expr,
    },
    /// A command.
    Command(Command),
    /// A macro call.
    Call {
        /// Macro or file name.
        name: String,
        /// Positional arguments.
        args: Vec<Arg>,
        /// Arguments given by name.
        subs: Vec<Sub>,
    },
    /// Variable declaration.
    Declare {
        /// Kind.
        kind: DeclKind,
        /// Names.
        names: Vec<String>,
    },
    /// Extra optional arguments.
    Args(Vec<String>),
    /// Unconditional jump.
    Jump(usize),
    /// Jump unless the condition is true.
    JumpIfNot(Expr, usize),
    /// Start a `DO` loop; jump to `exit` when it runs zero times.
    DoInit {
        /// Loop id (index of this instruction).
        id: usize,
        /// Loop variable.
        var: Expr,
        /// First value.
        from: Expr,
        /// Last value.
        to: Expr,
        /// Increment.
        step: Option<Expr>,
        /// Where to continue after the loop.
        exit: usize,
    },
    /// End of a loop body: increment and jump back, or fall through.
    DoNext {
        /// The id of the matching `DoInit`.
        id: usize,
    },
    /// Leave the current macro.
    Return,
    /// End the run.
    Stop,
    /// Does nothing (labels keep a line to stop on).
    Nop,
}

/// An instruction with its source line.
#[derive(Debug, Clone, PartialEq)]
pub struct Instr {
    /// The operation.
    pub op: Op,
    /// 1-based source line.
    pub line: usize,
}

/// A compiled body: the top level of a script or one macro.
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    /// Macro name (empty for the top level).
    pub name: String,
    /// Declared argument names.
    pub params: Vec<String>,
    /// Instructions.
    pub code: Vec<Instr>,
    /// Source line where the body starts.
    pub line: usize,
}

/// Compile a statement list.
pub fn compile_body(name: &str, params: &[String], stmts: &[Stmt], line: usize) -> Result<Body> {
    let mut c = Compiler { code: Vec::new(), labels: HashMap::new(), fixups: Vec::new() };
    c.stmts(stmts)?;
    for (at, label, line) in std::mem::take(&mut c.fixups) {
        let Some(&target) = c.labels.get(&label.to_ascii_uppercase()) else {
            return Err(ScriptError::syntax(line, 1, format!("GOTO {label}: there is no MLABEL {label}")));
        };
        c.code[at].op = Op::Jump(target);
    }
    Ok(Body { name: name.to_string(), params: params.to_vec(), code: c.code, line })
}

struct Compiler {
    code: Vec<Instr>,
    labels: HashMap<String, usize>,
    fixups: Vec<(usize, String, usize)>,
}

impl Compiler {
    fn emit(&mut self, op: Op, line: usize) -> usize {
        self.code.push(Instr { op, line });
        self.code.len() - 1
    }

    fn stmts(&mut self, list: &[Stmt]) -> Result<()> {
        for s in list {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<()> {
        let line = s.line;
        match &s.kind {
            StmtKind::Let { target, value } => {
                self.emit(Op::Let { target: target.clone(), value: value.clone() }, line);
            }
            StmtKind::Command(c) => {
                let up = c.name.to_ascii_uppercase();
                if up.len() >= 4 && "EXECUTE".starts_with(&up) {
                    // EXECUTE 'name' [k]: run a stored script k times (a macro call, repeated).
                    let name = match c.args.first() {
                        Some(Arg::Quoted(n)) | Some(Arg::Word(n)) | Some(Arg::Text(n)) => n.clone(),
                        _ => return Err(ScriptError::syntax(line, 1, "EXECUTE needs the name of a stored script")),
                    };
                    let times = match c.args.get(1) {
                        None => 1,
                        Some(Arg::Num(k)) if *k >= 1.0 && *k <= 10_000.0 && k.fract() == 0.0 => *k as usize,
                        Some(_) => {
                            return Err(ScriptError::syntax(
                                line,
                                1,
                                "EXECUTE repeats a whole number of times (1 to 10000)",
                            ))
                        }
                    };
                    let name =
                        name.strip_suffix(".mac").or_else(|| name.strip_suffix(".MAC")).unwrap_or(&name).to_string();
                    for _ in 0..times {
                        self.emit(Op::Call { name: name.clone(), args: Vec::new(), subs: Vec::new() }, line);
                    }
                } else {
                    self.emit(Op::Command(c.clone()), line);
                }
            }
            StmtKind::Call { name, args, subs } => {
                self.emit(Op::Call { name: name.clone(), args: args.clone(), subs: subs.clone() }, line);
            }
            StmtKind::Declare { kind, names } => {
                self.emit(Op::Declare { kind: *kind, names: names.clone() }, line);
            }
            StmtKind::Args(names) => {
                self.emit(Op::Args(names.clone()), line);
            }
            StmtKind::Return => {
                self.emit(Op::Return, line);
            }
            StmtKind::Stop => {
                self.emit(Op::Stop, line);
            }
            StmtKind::Label(l) => {
                let key = l.to_ascii_uppercase();
                if self.labels.contains_key(&key) {
                    return Err(ScriptError::syntax(line, 1, format!("the label `{l}` is defined twice")));
                }
                let at = self.emit(Op::Nop, line);
                self.labels.insert(key, at);
            }
            StmtKind::Goto(l) => {
                let at = self.emit(Op::Nop, line);
                self.fixups.push((at, l.clone(), line));
            }
            StmtKind::While { cond, body } => {
                let top = self.emit(Op::Nop, line);
                self.code[top].op = Op::JumpIfNot(cond.clone(), usize::MAX);
                self.stmts(body)?;
                self.emit(Op::Jump(top), line);
                let end = self.code.len();
                self.code[top].op = Op::JumpIfNot(cond.clone(), end);
            }
            StmtKind::If { arms, otherwise } => {
                let mut end_jumps = Vec::new();
                for (cond, body) in arms {
                    let test = self.emit(Op::Nop, line);
                    self.stmts(body)?;
                    end_jumps.push(self.emit(Op::Nop, line));
                    let next = self.code.len();
                    self.code[test].op = Op::JumpIfNot(cond.clone(), next);
                }
                if let Some(body) = otherwise {
                    self.stmts(body)?;
                }
                let end = self.code.len();
                for j in end_jumps {
                    self.code[j].op = Op::Jump(end);
                }
            }
            StmtKind::Do { var, from, to, step, body } => {
                let id = self.emit(Op::Nop, line);
                self.stmts(body)?;
                self.emit(Op::DoNext { id }, line);
                let exit = self.code.len();
                self.code[id].op =
                    Op::DoInit { id, var: var.clone(), from: from.clone(), to: to.clone(), step: step.clone(), exit };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn body(src: &str) -> Body {
        let p = parse(src).unwrap();
        compile_body("", &[], &p.main, 1).unwrap()
    }

    #[test]
    fn while_and_if_jump_targets() {
        let b = body("WHILE K1 < 3\nLET K1 = K1 + 1\nENDWHILE\nLET K2 = 1\n");
        // 0: JumpIfNot -> 3, 1: Let, 2: Jump 0, 3: Let
        assert!(matches!(b.code[0].op, Op::JumpIfNot(_, 3)));
        assert_eq!(b.code[2].op, Op::Jump(0));
        assert!(matches!(b.code[3].op, Op::Let { .. }));

        let b = body("IF K1 = 1\nLET K2 = 1\nELSEIF K1 = 2\nLET K2 = 2\nELSE\nLET K2 = 3\nENDIF\n");
        // 0: test1 ->3, 1: Let, 2: Jump end(7), 3: test2 ->6, 4: Let, 5: Jump 7, 6: Let(else)
        assert!(matches!(b.code[0].op, Op::JumpIfNot(_, 3)));
        assert_eq!(b.code[2].op, Op::Jump(7));
        assert!(matches!(b.code[3].op, Op::JumpIfNot(_, 6)));
        assert_eq!(b.code[5].op, Op::Jump(7));
        assert_eq!(b.code.len(), 7);
    }

    #[test]
    fn do_loops_and_goto_out_of_nested_blocks() {
        let b = body("DO K1 = 1:3\nDO K2 = 1:3\nIF K2 = 2\nGOTO out\nENDIF\nENDDO\nENDDO\nMLABEL out\nLET K3 = 1\n");
        let inits: Vec<_> = b.code.iter().enumerate().filter(|(_, i)| matches!(i.op, Op::DoInit { .. })).collect();
        assert_eq!(inits.len(), 2);
        let out = b.code.iter().position(|i| matches!(i.op, Op::Nop) && i.line == 8).unwrap();
        assert!(b.code.iter().any(|i| i.op == Op::Jump(out)));
        // Each DoInit exits just after its DoNext.
        for (idx, i) in inits {
            let Op::DoInit { id, exit, .. } = &i.op else { unreachable!() };
            assert_eq!(*id, idx);
            assert!(matches!(b.code[exit - 1].op, Op::DoNext { id: x } if x == idx));
        }
    }

    #[test]
    fn label_errors() {
        let p = parse("GOTO nowhere\n").unwrap();
        let e = compile_body("", &[], &p.main, 1).unwrap_err();
        assert!(e.message.contains("no MLABEL"));
        let p = parse("MLABEL a\nMLABEL A\n").unwrap();
        let e = compile_body("", &[], &p.main, 1).unwrap_err();
        assert!(e.message.contains("defined twice") && e.line == 2);
    }
}
