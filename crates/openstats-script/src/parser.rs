//! Recursive-descent parser: tokens → [`Program`].
//!
//! ```text
//! program   := ( macro | statement )*
//! macro     := ('GMACRO' | 'MACRO') [name params…] EOL [name params… EOL] statement*
//!              ('ENDMACRO' | 'MEND' | 'END')
//! statement := let | do | while | if | goto | label | return | stop | args | declare
//!            | label | call | command
//! label     := ('MLABEL' | 'LABEL') name | name ':'
//! let       := ['LET'] target '=' expr          (`LET` is optional: `K1 = 5`, `C1 = C2 + C3`)
//! do        := 'DO' var '=' expr ':' expr [ ('BY' | '/') expr ] block 'ENDDO'
//! while     := 'WHILE' expr block 'ENDWHILE'
//! if        := 'IF' expr block ('ELSEIF' expr block)* ['ELSE' block] 'ENDIF'
//! call      := '%' name args… [ ';' (name args… ';')* name args… '.' ]   (named arguments)
//! command   := word args… [ ';' [EOL] (word args… ';' [EOL])* word args… ('.' | EOL) ]
//! ```
//! `NOTE …` lines, `#` and `//` are comments. `SET` and `READ` are followed by data lines up to `END`.
//! Expressions follow the formula engine's grammar (`or > and > not > comparison > + - > * / > unary
//! > **`).

use crate::ast::*;
use crate::error::ScriptError;
use crate::lexer::{lex, Tok, Token};
use crate::Result;

/// Parse `src` into a [`Program`].
pub fn parse(src: &str) -> Result<Program> {
    let lines: Vec<String> = src.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
    let toks = lex(src, false).map_err(|e| e.with_snippet(&lines))?;
    let mut p = Parser { toks, pos: 0, stop_slash: false };
    p.program().map_err(|e| e.with_snippet(&lines))
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// Inside a `DO` upper bound a top-level `/` introduces the step, not a division.
    stop_slash: bool,
}

const BLOCK_ENDS: &[&str] = &["ENDDO", "ENDWHILE", "ENDIF", "ELSE", "ELSEIF", "ENDMACRO", "MEND"];

/// Words that end a macro definition.
const MACRO_ENDS: &[&str] = &["ENDMACRO", "MEND", "END"];

/// Words that begin a statement of their own, so `word = …` is never an assignment to them.
const STATEMENT_WORDS: &[&str] = &[
    "NOTE",
    "LET",
    "DO",
    "WHILE",
    "IF",
    "GOTO",
    "MLABEL",
    "LABEL",
    "RETURN",
    "STOP",
    "ARGS",
    "MCOLUMN",
    "MCONSTANT",
    "MMATRIX",
    "MTEXT",
    "MSTORE",
    "GMACRO",
    "MACRO",
];

fn is_ref(w: &str, letter: char) -> bool {
    let mut c = w.chars();
    c.next().is_some_and(|x| x.eq_ignore_ascii_case(&letter))
        && !w[1..].is_empty()
        && w[1..].bytes().all(|b| b.is_ascii_digit())
}

impl Parser {
    // ------------------------------------------------------------------ token helpers

    fn peek(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    fn peek_at(&self, n: usize) -> &Token {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)]
    }

    fn next(&mut self) -> Token {
        let t = self.peek().clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at_eol(&self) -> bool {
        matches!(self.peek().tok, Tok::Eol | Tok::Eof)
    }

    fn at_sym(&self, s: &str) -> bool {
        matches!(&self.peek().tok, Tok::Sym(x) if *x == s)
    }

    fn at_word(&self, w: &str) -> bool {
        matches!(&self.peek().tok, Tok::Word(x) if x.eq_ignore_ascii_case(w))
    }

    fn err<T>(&self, t: &Token, msg: impl Into<String>) -> Result<T> {
        Err(ScriptError::syntax(t.line, t.col, msg))
    }

    fn describe(t: &Token) -> String {
        match &t.tok {
            Tok::Word(w) => format!("`{w}`"),
            Tok::Num(n) => format!("`{n}`"),
            Tok::Str(s) => format!("\"{s}\""),
            Tok::Quoted(s) => format!("'{s}'"),
            Tok::Sym(s) => format!("`{s}`"),
            Tok::Eol => "the end of the line".into(),
            Tok::Eof => "the end of the script".into(),
        }
    }

    fn expect_sym(&mut self, s: &str) -> Result<Token> {
        if self.at_sym(s) {
            Ok(self.next())
        } else {
            let t = self.peek().clone();
            self.err(&t, format!("expected `{s}` but found {}", Self::describe(&t)))
        }
    }

    fn end_of_statement(&mut self) -> Result<()> {
        if self.at_eol() {
            self.next();
            Ok(())
        } else {
            let t = self.peek().clone();
            self.err(&t, format!("unexpected {} at the end of the statement", Self::describe(&t)))
        }
    }

    fn skip_eols(&mut self) {
        while matches!(self.peek().tok, Tok::Eol) {
            self.next();
        }
    }

    fn skip_line(&mut self) {
        while !self.at_eol() {
            self.next();
        }
        if matches!(self.peek().tok, Tok::Eol) {
            self.next();
        }
    }

    fn word(&mut self, what: &str) -> Result<(String, Token)> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Word(w) => {
                self.next();
                Ok((w.clone(), t))
            }
            _ => self.err(&t, format!("expected {what} but found {}", Self::describe(&t))),
        }
    }

    fn words_to_eol(&mut self) -> Result<Vec<String>> {
        let mut v = Vec::new();
        while !self.at_eol() {
            let (w, _) = self.word("a name")?;
            v.push(w);
        }
        self.next();
        Ok(v)
    }

    // ------------------------------------------------------------------ program and macros

    fn program(&mut self) -> Result<Program> {
        let mut prog = Program::default();
        loop {
            self.skip_eols();
            if matches!(self.peek().tok, Tok::Eof) {
                break;
            }
            if self.at_word("GMACRO") || self.at_word("MACRO") {
                let m = self.macro_def()?;
                if prog.macros.iter().any(|x| x.name.eq_ignore_ascii_case(&m.name)) {
                    return Err(ScriptError::syntax(m.line, 1, format!("the macro `{}` is defined twice", m.name)));
                }
                prog.macros.push(m);
            } else if let Some(s) = self.statement()? {
                prog.main.push(s);
            }
        }
        Ok(prog)
    }

    fn macro_def(&mut self) -> Result<MacroDef> {
        let head = self.next();
        let global = matches!(&head.tok, Tok::Word(w) if w.eq_ignore_ascii_case("GMACRO"));
        if self.at_eol() {
            self.skip_eols();
        }
        let (name, nt) = self.word("the macro name")?;
        if BLOCK_ENDS.iter().any(|k| name.eq_ignore_ascii_case(k)) {
            return self.err(&nt, format!("`{name}` cannot be a macro name"));
        }
        let params = self.words_to_eol()?;
        if global && !params.is_empty() {
            return self.err(&nt, "a GMACRO takes no arguments; use MACRO to declare arguments");
        }
        let (body, _) = self.block(MACRO_ENDS, head.line)?;
        let end_line = self.toks[self.pos.saturating_sub(2).min(self.toks.len() - 1)].line;
        Ok(MacroDef { name, global, params, body, line: head.line, end_line: end_line.max(head.line) })
    }

    /// Statements up to (and including) one of `terminators`; returns which one ended the block.
    fn block(&mut self, terminators: &[&str], opened_at: usize) -> Result<(Vec<Stmt>, String)> {
        let mut body = Vec::new();
        loop {
            self.skip_eols();
            let t = self.peek().clone();
            match &t.tok {
                Tok::Eof => {
                    return Err(ScriptError::syntax(
                        opened_at,
                        1,
                        format!("the block opened here is not closed (expected {})", terminators.join(" or ")),
                    ));
                }
                Tok::Word(w) => {
                    let up = w.to_ascii_uppercase();
                    if terminators.contains(&up.as_str()) {
                        self.next();
                        if up == "ELSE" || up == "ELSEIF" {
                            return Ok((body, up));
                        }
                        // ENDDO / ENDWHILE / ENDIF / ENDMACRO / MEND / END may be followed by a comment only.
                        self.end_of_statement()?;
                        return Ok((body, up));
                    }
                    if BLOCK_ENDS.contains(&up.as_str()) || (up == "END" && !terminators.contains(&"END")) {
                        return self.err(&t, format!("`{w}` does not match an open block"));
                    }
                    if up == "MACRO" || up == "GMACRO" {
                        return self.err(&t, "macros cannot be defined inside other blocks");
                    }
                }
                _ => {}
            }
            if let Some(s) = self.statement()? {
                body.push(s);
            }
        }
    }

    // ------------------------------------------------------------------ statements

    fn statement(&mut self) -> Result<Option<Stmt>> {
        let t = self.peek().clone();
        let line = t.line;
        let Tok::Word(w) = &t.tok else {
            if matches!(t.tok, Tok::Sym("%")) {
                self.next();
                return self.macro_call(line).map(Some);
            }
            // `'Sales' = C1 * 2` and `'Sales'(3) = 9`: assignment to a named column.
            if matches!(t.tok, Tok::Quoted(_)) && self.looks_like_assignment() {
                let kind = self.assignment(&t)?;
                return Ok(Some(Stmt { line, kind }));
            }
            return self.err(&t, format!("expected a command but found {}", Self::describe(&t)));
        };
        let up = w.to_ascii_uppercase();
        if BLOCK_ENDS.contains(&up.as_str()) || up == "END" {
            return self.err(&t, format!("`{w}` does not match an open block"));
        }
        if !STATEMENT_WORDS.contains(&up.as_str()) {
            // `name:` is a label.
            if matches!(self.peek_at(1).tok, Tok::Sym(":")) && matches!(self.peek_at(2).tok, Tok::Eol | Tok::Eof) {
                self.next();
                self.next();
                self.end_of_statement()?;
                return Ok(Some(Stmt { line, kind: StmtKind::Label(w.clone()) }));
            }
            // `K1 = 5`, `C1 = C2 + C3`, `C1(3) = 9`: `LET` is optional.
            if self.looks_like_assignment() {
                let kind = self.assignment(&t)?;
                return Ok(Some(Stmt { line, kind }));
            }
        }
        let kind = match up.as_str() {
            "NOTE" => {
                self.skip_line();
                return Ok(None);
            }
            "LET" => {
                self.next();
                self.assignment(&t)?
            }
            "DO" => {
                self.next();
                let var = self.primary()?;
                if !matches!(var, Expr::Word(_)) {
                    return self.err(&t, "DO needs a variable such as K1");
                }
                self.expect_sym("=")?;
                let from = self.expr()?;
                self.expect_sym(":")?;
                self.stop_slash = true;
                let to = self.expr();
                self.stop_slash = false;
                let to = to?;
                let step = if self.at_word("BY") || self.at_sym("/") {
                    self.next();
                    Some(self.expr()?)
                } else {
                    None
                };
                self.end_of_statement()?;
                let (body, _) = self.block(&["ENDDO"], line)?;
                StmtKind::Do { var, from, to, step, body }
            }
            "WHILE" => {
                self.next();
                let cond = self.expr()?;
                self.end_of_statement()?;
                let (body, _) = self.block(&["ENDWHILE"], line)?;
                StmtKind::While { cond, body }
            }
            "IF" => {
                self.next();
                let mut arms = Vec::new();
                let mut otherwise = None;
                let mut cond = self.expr()?;
                self.end_of_statement()?;
                loop {
                    let (body, ended) = self.block(&["ELSEIF", "ELSE", "ENDIF"], line)?;
                    arms.push((cond.clone(), body));
                    match ended.as_str() {
                        "ELSEIF" => {
                            cond = self.expr()?;
                            self.end_of_statement()?;
                        }
                        "ELSE" => {
                            self.end_of_statement()?;
                            let (body, e2) = self.block(&["ENDIF"], line)?;
                            debug_assert_eq!(e2, "ENDIF");
                            otherwise = Some(body);
                            break;
                        }
                        _ => break,
                    }
                }
                StmtKind::If { arms, otherwise }
            }
            "GOTO" => {
                self.next();
                let l = self.label_name()?;
                self.end_of_statement()?;
                StmtKind::Goto(l)
            }
            "MLABEL" | "LABEL" => {
                self.next();
                let l = self.label_name()?;
                self.end_of_statement()?;
                StmtKind::Label(l)
            }
            "RETURN" => {
                self.next();
                self.end_of_statement()?;
                StmtKind::Return
            }
            "STOP" => {
                self.next();
                self.end_of_statement()?;
                StmtKind::Stop
            }
            "ARGS" => {
                self.next();
                StmtKind::Args(self.words_to_eol()?)
            }
            "MCOLUMN" | "MCONSTANT" | "MMATRIX" | "MTEXT" | "MSTORE" => {
                self.next();
                let kind = match up.as_str() {
                    "MCOLUMN" => DeclKind::Column,
                    "MCONSTANT" => DeclKind::Constant,
                    "MMATRIX" => DeclKind::Matrix,
                    "MTEXT" => DeclKind::Text,
                    _ => DeclKind::Store,
                };
                StmtKind::Declare { kind, names: self.words_to_eol()? }
            }
            _ => StmtKind::Command(self.command()?),
        };
        Ok(Some(Stmt { line, kind }))
    }

    /// A label: a word or a whole number.
    fn label_name(&mut self) -> Result<String> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Word(w) => {
                self.next();
                Ok(w.clone())
            }
            Tok::Num(x) if x.fract() == 0.0 && *x >= 0.0 => {
                self.next();
                Ok(format!("{x}"))
            }
            _ => self.err(&t, format!("expected a label name but found {}", Self::describe(&t))),
        }
    }

    /// At a word or a quoted name that is followed by `=` (possibly after an `(index)`).
    fn looks_like_assignment(&self) -> bool {
        let mut n = 1;
        if matches!(self.peek_at(n).tok, Tok::Sym("(")) && !self.peek_at(n).space_before {
            let mut depth = 0usize;
            loop {
                match &self.peek_at(n).tok {
                    Tok::Sym("(") => depth += 1,
                    Tok::Sym(")") => {
                        depth -= 1;
                        if depth == 0 {
                            n += 1;
                            break;
                        }
                    }
                    Tok::Eol | Tok::Eof => return false,
                    _ => {}
                }
                n += 1;
            }
        }
        matches!(self.peek_at(n).tok, Tok::Sym("="))
    }

    /// `target = expr` (after an optional `LET`).
    fn assignment(&mut self, head: &Token) -> Result<StmtKind> {
        let target = self.postfix()?;
        if !matches!(target, Expr::Word(_) | Expr::Quoted(_) | Expr::Index(..) | Expr::Call(..)) {
            return self.err(head, "LET needs a column, constant, matrix or element on its left");
        }
        self.expect_sym("=")?;
        let value = self.expr()?;
        self.end_of_statement()?;
        Ok(StmtKind::Let { target, value })
    }

    /// The `; sub args ; sub args .` tail of a command or macro call, or just its terminator.
    ///
    /// Subcommands may follow the `;` on the same line or start on the next one; a lone `.`
    /// after a `;` ends the statement.
    fn subcommands(&mut self) -> Result<Vec<Sub>> {
        let mut subs = Vec::new();
        if !self.at_sym(";") {
            if self.at_sym(".") {
                self.next();
            }
            self.end_of_statement()?;
            return Ok(subs);
        }
        let mut semi = self.next();
        loop {
            self.skip_eols();
            if self.at_sym(".") {
                self.next();
                self.end_of_statement()?;
                break;
            }
            let t = self.peek().clone();
            if matches!(t.tok, Tok::Eof) {
                return self.err(&semi, "the `;` is not followed by a subcommand");
            }
            let Tok::Word(sname) = &t.tok else {
                return self.err(&t, format!("expected a subcommand but found {}", Self::describe(&t)));
            };
            self.next();
            let sargs = self.args()?;
            subs.push(Sub { line: t.line, name: sname.clone(), args: sargs });
            if self.at_sym(";") {
                semi = self.next();
                continue;
            }
            if self.at_sym(".") {
                self.next();
            }
            self.end_of_statement()?;
            break;
        }
        Ok(subs)
    }

    fn macro_call(&mut self, line: usize) -> Result<Stmt> {
        let t = self.peek().clone();
        let mut name = match &t.tok {
            Tok::Word(w) => {
                self.next();
                w.clone()
            }
            Tok::Quoted(q) | Tok::Str(q) => {
                self.next();
                q.clone()
            }
            _ => return self.err(&t, "expected a macro name after `%`"),
        };
        // `%file.mac`: keep the extension in the name.
        while self.at_sym(".")
            && !self.peek().space_before
            && matches!(self.peek_at(1).tok, Tok::Word(_))
            && !self.peek_at(1).space_before
        {
            self.next();
            let (ext, _) = self.word("a file extension")?;
            name.push('.');
            name.push_str(&ext);
        }
        let args = self.args()?;
        let subs = self.subcommands()?;
        Ok(Stmt { line, kind: StmtKind::Call { name, args, subs } })
    }

    fn command(&mut self) -> Result<Command> {
        let (name, nt) = self.word("a command")?;
        let args = self.args()?;
        let subs = self.subcommands()?;
        let up = name.to_ascii_uppercase();
        let mut data = Vec::new();
        let from_file = subs.iter().any(|s| s.name.eq_ignore_ascii_case("FILE"));
        if (up == "SET" || up == "READ") && !from_file {
            data = self.data_lines(&nt)?;
        }
        Ok(Command { name, args, subs, data })
    }

    fn data_lines(&mut self, head: &Token) -> Result<Vec<DataLine>> {
        let mut out = Vec::new();
        loop {
            if matches!(self.peek().tok, Tok::Eof) {
                return Err(ScriptError::syntax(head.line, head.col, "the data is not closed by END"));
            }
            let first = self.peek().clone();
            if self.at_word("END") && matches!(self.peek_at(1).tok, Tok::Eol | Tok::Eof) {
                self.next();
                self.end_of_statement()?;
                return Ok(out);
            }
            let mut items = Vec::new();
            while !self.at_eol() {
                let t = self.next();
                match t.tok {
                    Tok::Num(x) => items.push(Datum::Num(x)),
                    Tok::Str(s) | Tok::Quoted(s) => items.push(Datum::Text(s)),
                    Tok::Word(s) => items.push(Datum::Text(s)),
                    Tok::Sym("*") => items.push(Datum::Missing),
                    Tok::Sym("-") if matches!(self.peek().tok, Tok::Num(_)) && !self.peek().space_before => {
                        let Tok::Num(x) = self.next().tok else { unreachable!() };
                        items.push(Datum::Num(-x));
                    }
                    Tok::Sym(",") => {}
                    _ => return self.err(&t, format!("unexpected {} in the data", Self::describe(&t))),
                }
            }
            self.next();
            if !items.is_empty() {
                out.push(DataLine { line: first.line, items });
            }
        }
    }

    // ------------------------------------------------------------------ command arguments

    /// Arguments up to the end of the line, a `;` or a `.`.
    fn args(&mut self) -> Result<Vec<Arg>> {
        let mut out = Vec::new();
        while !self.at_eol() && !self.at_sym(";") && !self.at_sym(".") {
            if self.at_sym(",") {
                self.next();
                continue;
            }
            out.push(self.arg()?);
        }
        Ok(out)
    }

    fn arg_atom(&mut self) -> Result<Arg> {
        let t = self.next();
        match t.tok {
            Tok::Num(x) => Ok(Arg::Num(x)),
            Tok::Str(s) => Ok(Arg::Text(s)),
            Tok::Quoted(s) => Ok(Arg::Quoted(s)),
            Tok::Word(w) => Ok(Arg::Word(w)),
            Tok::Sym("*") => Ok(Arg::Missing),
            Tok::Sym("-") if matches!(self.peek().tok, Tok::Num(_)) && !self.peek().space_before => {
                let Tok::Num(x) = self.next().tok else { unreachable!() };
                Ok(Arg::Num(-x))
            }
            Tok::Sym("(") => {
                let saved = std::mem::replace(&mut self.stop_slash, false);
                let e = self.expr()?;
                self.stop_slash = saved;
                self.expect_sym(")")?;
                Ok(Arg::Paren(e))
            }
            _ => self.err(&t, format!("unexpected {} in the arguments", Self::describe(&t))),
        }
    }

    fn arg(&mut self) -> Result<Arg> {
        let first = self.arg_atom()?;
        let is_col = |a: &Arg| {
            matches!(a, Arg::Word(w) if is_ref(w, 'C') || is_ref(w, 'K') || is_ref(w, 'M'))
                || matches!(a, Arg::Quoted(_))
        };
        if self.at_sym("-") && !self.peek().space_before && is_col(&first) {
            let nxt = self.peek_at(1).clone();
            if !nxt.space_before && matches!(nxt.tok, Tok::Word(_) | Tok::Quoted(_)) {
                self.next();
                let last = self.arg_atom()?;
                return Ok(Arg::Range(Box::new(first), Box::new(last)));
            }
        }
        if self.at_sym(":") && matches!(first, Arg::Word(_) | Arg::Num(_) | Arg::Quoted(_)) {
            self.next();
            let last = self.arg_atom()?;
            return Ok(Arg::Range(Box::new(first), Box::new(last)));
        }
        Ok(first)
    }

    // ------------------------------------------------------------------ expressions

    fn expr(&mut self) -> Result<Expr> {
        self.or()
    }

    fn or(&mut self) -> Result<Expr> {
        let mut a = self.and()?;
        while self.at_word("OR") || self.at_sym("|") {
            self.next();
            let b = self.and()?;
            a = Expr::Binary(BinOp::Or, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn and(&mut self) -> Result<Expr> {
        let mut a = self.not()?;
        while self.at_word("AND") || self.at_sym("&") {
            self.next();
            let b = self.not()?;
            a = Expr::Binary(BinOp::And, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn not(&mut self) -> Result<Expr> {
        if self.at_word("NOT") || self.at_sym("~") {
            self.next();
            let e = self.not()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(e)));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<Expr> {
        let a = self.add()?;
        let op = match &self.peek().tok {
            Tok::Sym("=") | Tok::Sym("==") => Some(BinOp::Eq),
            Tok::Sym("<>") | Tok::Sym("!=") => Some(BinOp::Ne),
            Tok::Sym("<") => Some(BinOp::Lt),
            Tok::Sym("<=") => Some(BinOp::Le),
            Tok::Sym(">") => Some(BinOp::Gt),
            Tok::Sym(">=") => Some(BinOp::Ge),
            Tok::Word(w) => match w.to_ascii_uppercase().as_str() {
                "EQ" => Some(BinOp::Eq),
                "NE" => Some(BinOp::Ne),
                "LT" => Some(BinOp::Lt),
                "LE" => Some(BinOp::Le),
                "GT" => Some(BinOp::Gt),
                "GE" => Some(BinOp::Ge),
                _ => None,
            },
            _ => None,
        };
        if let Some(op) = op {
            self.next();
            let b = self.add()?;
            return Ok(Expr::Binary(op, Box::new(a), Box::new(b)));
        }
        Ok(a)
    }

    fn add(&mut self) -> Result<Expr> {
        let mut a = self.mul()?;
        loop {
            let op = if self.at_sym("+") {
                BinOp::Add
            } else if self.at_sym("-") {
                BinOp::Sub
            } else {
                break;
            };
            self.next();
            let b = self.mul()?;
            a = Expr::Binary(op, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn mul(&mut self) -> Result<Expr> {
        let mut a = self.unary()?;
        loop {
            let op = if self.at_sym("*") {
                BinOp::Mul
            } else if self.at_sym("/") && !self.stop_slash {
                BinOp::Div
            } else {
                break;
            };
            self.next();
            let b = self.unary()?;
            a = Expr::Binary(op, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.at_sym("-") {
            self.next();
            let e = self.unary()?;
            return Ok(match e {
                Expr::Num(x) => Expr::Num(-x),
                e => Expr::Unary(UnOp::Neg, Box::new(e)),
            });
        }
        if self.at_sym("+") {
            self.next();
            return self.unary();
        }
        self.pow()
    }

    fn pow(&mut self) -> Result<Expr> {
        let base = self.postfix()?;
        if self.at_sym("**") || self.at_sym("^") {
            self.next();
            let e = self.unary()?;
            return Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(e)));
        }
        Ok(base)
    }

    /// A primary, followed by `(i, j)` element access when it is a column or matrix reference.
    fn postfix(&mut self) -> Result<Expr> {
        let e = self.primary()?;
        let indexable = match &e {
            Expr::Quoted(_) => true,
            Expr::Word(w) => is_ref(w, 'C') || is_ref(w, 'M'),
            _ => false,
        };
        if indexable && self.at_sym("(") && !self.peek().space_before {
            let args = self.call_args()?;
            return Ok(Expr::Index(Box::new(e), args));
        }
        Ok(e)
    }

    fn call_args(&mut self) -> Result<Vec<Expr>> {
        self.expect_sym("(")?;
        let saved = std::mem::replace(&mut self.stop_slash, false);
        let mut args = Vec::new();
        if !self.at_sym(")") {
            loop {
                args.push(self.expr()?);
                if self.at_sym(",") {
                    self.next();
                } else {
                    break;
                }
            }
        }
        self.stop_slash = saved;
        self.expect_sym(")")?;
        Ok(args)
    }

    fn primary(&mut self) -> Result<Expr> {
        let t = self.next();
        match t.tok {
            Tok::Num(x) => Ok(Expr::Num(x)),
            Tok::Str(s) => Ok(Expr::Text(s)),
            Tok::Quoted(s) => Ok(Expr::Quoted(s)),
            Tok::Sym("*") => Ok(Expr::Missing),
            Tok::Word(w) => {
                if self.at_sym("(") && !self.peek().space_before && !is_ref(&w, 'C') && !is_ref(&w, 'M') {
                    let args = self.call_args()?;
                    return Ok(Expr::Call(w, args));
                }
                Ok(Expr::Word(w))
            }
            Tok::Sym("(") => {
                let saved = std::mem::replace(&mut self.stop_slash, false);
                let e = self.expr()?;
                self.stop_slash = saved;
                self.expect_sym(")")?;
                Ok(e)
            }
            _ => self.err(&t, format!("expected a value but found {}", Self::describe(&t))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stmt(src: &str) -> StmtKind {
        parse(src).unwrap().main.remove(0).kind
    }

    fn e(src: &str) -> String {
        let StmtKind::Let { value, .. } = stmt(&format!("LET K1 = {src}")) else { panic!() };
        value.to_string()
    }

    #[test]
    fn expression_precedence_and_associativity() {
        assert_eq!(e("1 + 2 * 3"), "(1 + (2 * 3))");
        assert_eq!(e("(1 + 2) * 3"), "((1 + 2) * 3)");
        assert_eq!(e("2 ** 3 ** 2"), "(2 ** (3 ** 2))");
        assert_eq!(e("-2 ** 2"), "(-(2 ** 2))");
        assert_eq!(e("K1 > 2 AND K2 <= 3 OR NOT K3"), "(((K1 > 2) AND (K2 <= 3)) OR (NOT K3))");
        assert_eq!(e("K1 EQ 2"), "(K1 = 2)");
        assert_eq!(e("a - b - c"), "((a - b) - c)");
        assert_eq!(e("SQRT(K1) + MEAN(C1)"), "(SQRT(K1) + MEAN(C1))");
        assert_eq!(e("C1(3) + M2(1, 2)"), "(C1(3) + M2(1, 2))");
        assert_eq!(e("'Weight (g)' * 2"), "('Weight (g)' * 2)");
        assert_eq!(e("* + 1"), "(* + 1)");
        assert_eq!(e("\"a\" = \"b\""), "(\"a\" = \"b\")");
    }

    #[test]
    fn let_do_while_if() {
        let p = parse(
            "LET K1 = 0\nDO K2 = 1:10 BY 2\n  LET K1 = K1 + K2\nENDDO\nWHILE K1 > 0\n  LET K1 = K1 - 1\nENDWHILE\n\
             IF K1 = 0\n  LET K3 = 1\nELSEIF K1 = 1\n  LET K3 = 2\nELSE\n  LET K3 = 3\nENDIF\n",
        )
        .unwrap();
        assert_eq!(p.main.len(), 4);
        let StmtKind::Do { step, body, .. } = &p.main[1].kind else { panic!() };
        assert!(step.is_some());
        assert_eq!(body.len(), 1);
        let StmtKind::If { arms, otherwise } = &p.main[3].kind else { panic!() };
        assert_eq!(arms.len(), 2);
        assert!(otherwise.is_some());
        assert_eq!(p.main[3].line, 8);
    }

    #[test]
    fn do_step_by_slash_and_division_inside_parentheses() {
        let StmtKind::Do { to, step, .. } = stmt("DO K1 = 1:(10 / 2) / 2\nENDDO") else { panic!() };
        assert_eq!(to.to_string(), "(10 / 2)");
        assert_eq!(step.unwrap().to_string(), "2");
    }

    #[test]
    fn commands_subcommands_data_and_ranges() {
        let p = parse("REGRESS C1 2 C2 C3;\n  RESIDUALS C4;\n  FITS 'my fits'.\nDESCRIBE C1-C3 -5 \"t\" *\nSET C5\n 1 2 3\n -4 *\nEND\nPRINT C5\n").unwrap();
        let StmtKind::Command(c) = &p.main[0].kind else { panic!() };
        assert_eq!(c.args.len(), 4);
        assert_eq!(c.subs.len(), 2);
        assert_eq!(c.subs[1].args, vec![Arg::Quoted("my fits".into())]);
        let StmtKind::Command(d) = &p.main[1].kind else { panic!() };
        assert_eq!(
            d.args,
            vec![
                Arg::Range(Box::new(Arg::Word("C1".into())), Box::new(Arg::Word("C3".into()))),
                Arg::Num(-5.0),
                Arg::Text("t".into()),
                Arg::Missing
            ]
        );
        let StmtKind::Command(s) = &p.main[2].kind else { panic!() };
        assert_eq!(s.data.len(), 2);
        assert_eq!(s.data[1].items, vec![Datum::Num(-4.0), Datum::Missing]);
        assert!(matches!(&p.main[3].kind, StmtKind::Command(c) if c.name == "PRINT"));
    }

    #[test]
    fn macros_goto_and_calls() {
        let p = parse(
            "GMACRO\nrun\n  MLABEL top\n  LET K1 = K1 + 1\n  IF K1 < 3\n    GOTO top\n  ENDIF\nENDMACRO\n\
             MACRO\nscale c k\n  MCOLUMN c\n  MCONSTANT k\n  ARGS extra\n  LET c = c * k\nENDMACRO\n%run\n%scale C1 2\n%lib.mac C2 3\nNOTE this is a comment\n",
        )
        .unwrap();
        assert_eq!(p.macros.len(), 2);
        assert!(p.macros[0].global);
        assert_eq!(p.macros[1].params, vec!["c", "k"]);
        assert!(matches!(&p.macros[1].body[2].kind, StmtKind::Args(a) if a == &["extra"]));
        let names: Vec<&str> = p
            .main
            .iter()
            .filter_map(|s| if let StmtKind::Call { name, .. } = &s.kind { Some(name.as_str()) } else { None })
            .collect();
        assert_eq!(names, ["run", "scale", "lib.mac"]);
        assert_eq!(p.main.len(), 3, "NOTE lines produce no statement");
    }

    #[test]
    fn syntax_errors_have_positions() {
        let cases: &[(&str, usize, &str)] = &[
            ("LET K1 = \n", 1, "expected a value"),
            ("DO K1 = 1:3\nLET K2 = 1\n", 1, "not closed"),
            ("ENDDO\n", 1, "does not match"),
            ("IF K1\nENDWHILE\n", 2, "does not match"),
            ("LET 5 = 1\n", 1, "LET needs"),
            ("PRINT C1 )\n", 1, "unexpected"),
            ("GMACRO\nm x\nENDMACRO\n", 2, "takes no arguments"),
            ("MACRO\nm\nMACRO\nn\nENDMACRO\nENDMACRO\n", 3, "inside other blocks"),
            ("SET C1\n1 2\n", 1, "not closed by END"),
            ("%m C1;\n", 1, "not followed by a subcommand"),
            ("GMACRO\na\nENDMACRO\nGMACRO\nA\nENDMACRO\n", 4, "defined twice"),
            ("LET K1 = (1 + 2\n", 1, "expected `)`"),
        ];
        for (src, line, frag) in cases {
            let e = parse(src).unwrap_err();
            assert_eq!(e.line, *line, "{src:?}: {e}");
            assert!(e.message.contains(frag), "{src:?}: {e}");
        }
    }

    #[test]
    fn comments_blank_lines_and_trailing_end_markers() {
        let p = parse("# header\n\nLET K1 = 1 # trailing\nIF K1 = 1 # c\n  NOTE x\nENDIF # done\n").unwrap();
        assert_eq!(p.main.len(), 2);
    }

    #[test]
    fn minitab_forms_assignment_without_let_labels_and_terminators() {
        assert!(matches!(stmt("K1 = 5"), StmtKind::Let { .. }));
        assert!(matches!(stmt("C1(3) = 5"), StmtKind::Let { .. }));
        assert!(matches!(stmt("'Sales' = C1 * 2"), StmtKind::Let { .. }));
        assert!(matches!(stmt("done:"), StmtKind::Label(l) if l == "done"));
        assert!(matches!(stmt("Label 10"), StmtKind::Label(l) if l == "10"));
        assert!(matches!(stmt("MTEXT a b"), StmtKind::Declare { kind: DeclKind::Text, .. }));
        assert!(matches!(stmt("MSTORE a"), StmtKind::Declare { kind: DeclKind::Store, .. }));
        for end in ["ENDMACRO", "MEND", "END"] {
            let p = parse(&format!("MACRO m a\n  K1 = 1\n{end}\n")).unwrap();
            assert_eq!(p.macros[0].body.len(), 1, "{end}");
            assert_eq!(p.macros[0].end_line, 3);
        }
        // A command whose first argument is a name followed by `=` is still a command.
        assert!(matches!(stmt("PRINT C1\n"), StmtKind::Command(_)));
    }

    #[test]
    fn one_line_subcommands_and_named_macro_arguments() {
        let StmtKind::Command(c) = stmt("REGRESS C1 1 C2; FITS C3; RESIDUALS C4.") else { panic!() };
        assert_eq!(c.subs.len(), 2);
        let StmtKind::Call { args, subs, .. } = stmt("%m C1; Factor 3; Mode 'x'.") else { panic!() };
        assert_eq!((args.len(), subs.len()), (1, 2));
        let StmtKind::Command(c) = stmt("DESCRIBE C1;\n.\n") else { panic!() };
        assert!(c.subs.is_empty());
    }
}
