//! Numeric transformation expressions: the contents of `I(...)` and of the
//! function terms `log(x)`, `sqrt(x)`, …

use super::lexer::{Tok, Token};
use crate::error::{invalid, Result};

/// Functions allowed in transformation terms.
pub(crate) const FUNCTIONS: &[&str] = &["log", "ln", "log10", "log2", "exp", "sqrt", "abs", "sin", "cos", "tan", "I"];

/// Arithmetic expression over numeric columns.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal.
    Num(f64),
    /// Numeric column.
    Var(String),
    /// Negation.
    Neg(Box<Expr>),
    /// Binary operation.
    Bin(char, Box<Expr>, Box<Expr>),
    /// Function call (`log`, `exp`, `sqrt`, …).
    Func(String, Box<Expr>),
}

impl Expr {
    /// Canonical text, used as the term label.
    pub fn text(&self) -> String {
        match self {
            Expr::Num(v) => format!("{v}"),
            Expr::Var(n) => n.clone(),
            Expr::Neg(e) => format!("-{}", e.text_atom()),
            Expr::Bin(op, a, b) => format!("{}{}{}", a.text_atom(), op, b.text_atom()),
            Expr::Func(f, e) => format!("{f}({})", e.text()),
        }
    }

    fn text_atom(&self) -> String {
        match self {
            Expr::Bin(..) => format!("({})", self.text()),
            _ => self.text(),
        }
    }

    /// Names of the columns the expression reads.
    pub fn variables(&self) -> Vec<String> {
        fn walk(e: &Expr, out: &mut Vec<String>) {
            match e {
                Expr::Var(n) => {
                    if !out.contains(n) {
                        out.push(n.clone());
                    }
                }
                Expr::Neg(a) | Expr::Func(_, a) => walk(a, out),
                Expr::Bin(_, a, b) => {
                    walk(a, out);
                    walk(b, out);
                }
                Expr::Num(_) => {}
            }
        }
        let mut v = Vec::new();
        walk(self, &mut v);
        v
    }

    /// Evaluate at one row; `lookup` returns the value of a column at that row.
    pub(crate) fn eval(&self, lookup: &dyn Fn(&str) -> f64) -> f64 {
        match self {
            Expr::Num(v) => *v,
            Expr::Var(n) => lookup(n),
            Expr::Neg(a) => -a.eval(lookup),
            Expr::Bin(op, a, b) => {
                let (x, y) = (a.eval(lookup), b.eval(lookup));
                match op {
                    '+' => x + y,
                    '-' => x - y,
                    '*' => x * y,
                    '/' => x / y,
                    _ => x.powf(y),
                }
            }
            Expr::Func(f, a) => {
                let x = a.eval(lookup);
                match f.as_str() {
                    "log" | "ln" => x.ln(),
                    "log10" => x.log10(),
                    "log2" => x.log2(),
                    "exp" => x.exp(),
                    "sqrt" => x.sqrt(),
                    "abs" => x.abs(),
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    _ => x, // I(...)
                }
            }
        }
    }
}

/// Recursive-descent parser over a token slice.
pub(crate) struct ExprParser<'a> {
    pub toks: &'a [Token],
    pub i: usize,
}

impl ExprParser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|t| &t.tok)
    }

    pub fn parse(&mut self) -> Result<Expr> {
        self.sum()
    }

    fn sum(&mut self) -> Result<Expr> {
        let mut l = self.product()?;
        while let Some(Tok::Op(c @ ('+' | '-'))) = self.peek() {
            let c = *c;
            self.i += 1;
            let r = self.product()?;
            l = Expr::Bin(c, Box::new(l), Box::new(r));
        }
        Ok(l)
    }

    fn product(&mut self) -> Result<Expr> {
        let mut l = self.unary()?;
        while let Some(Tok::Op(c @ ('*' | '/'))) = self.peek() {
            let c = *c;
            self.i += 1;
            let r = self.unary()?;
            l = Expr::Bin(c, Box::new(l), Box::new(r));
        }
        Ok(l)
    }

    fn unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Tok::Op('-')) => {
                self.i += 1;
                Ok(Expr::Neg(Box::new(self.unary()?)))
            }
            Some(Tok::Op('+')) => {
                self.i += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr> {
        let base = self.primary()?;
        if let Some(Tok::Op('^')) = self.peek() {
            self.i += 1;
            let exp = self.unary()?; // right-associative, allows x^-1
            return Ok(Expr::Bin('^', Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Expr> {
        let Some(t) = self.toks.get(self.i) else {
            return invalid("formula: unexpected end of expression");
        };
        match &t.tok {
            Tok::Num(v) => {
                self.i += 1;
                Ok(Expr::Num(*v))
            }
            Tok::Ident(name) => {
                self.i += 1;
                if let Some(Tok::Op('(')) = self.peek() {
                    if FUNCTIONS.contains(&name.as_str()) {
                        self.i += 1;
                        let inner = self.sum()?;
                        match self.peek() {
                            Some(Tok::Op(')')) => self.i += 1,
                            _ => return invalid(format!("formula: missing ')' after {name}(")),
                        }
                        return Ok(Expr::Func(name.clone(), Box::new(inner)));
                    }
                    return invalid(format!("formula: unknown function '{name}' in an expression"));
                }
                Ok(Expr::Var(name.clone()))
            }
            Tok::Op('(') => {
                self.i += 1;
                let e = self.sum()?;
                match self.peek() {
                    Some(Tok::Op(')')) => {
                        self.i += 1;
                        Ok(e)
                    }
                    _ => invalid("formula: missing ')'"),
                }
            }
            other => invalid(format!("formula: unexpected {other:?} in expression at position {}", t.pos)),
        }
    }
}

/// Parse a complete token slice as one expression.
pub(crate) fn parse_expr(toks: &[Token]) -> Result<Expr> {
    let mut p = ExprParser { toks, i: 0 };
    let e = p.parse()?;
    if p.i != toks.len() {
        return invalid(format!("formula: unexpected token at position {}", toks[p.i].pos));
    }
    Ok(e)
}

impl Expr {
    /// Parse a stand-alone arithmetic expression such as `b0 * exp(b1 * x)`.
    /// Identifiers are columns or parameters; `+ - * / ^`, parentheses and
    /// the functions `log ln log10 log2 exp sqrt abs sin cos tan` are allowed.
    pub fn parse(src: &str) -> Result<Expr> {
        let toks = super::lexer::tokenize(src)?;
        parse_expr(&toks)
    }

    /// Evaluate with `lookup` supplying the value of every identifier.
    pub fn evaluate(&self, lookup: &dyn Fn(&str) -> f64) -> f64 {
        self.eval(lookup)
    }
}
