//! Recursive-descent formula parser.

use super::ast::{BinOp, Expr, UnOp};
use super::lexer::{tokenize, Tok, Token};
use crate::error::{CoreError, Result};

/// Maximum expression nesting (guards against stack exhaustion in Wasm).
pub const MAX_NESTING: usize = 200;

struct P {
    t: Vec<Token>,
    i: usize,
    end: usize,
    depth: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i).map(|t| &t.tok)
    }
    fn pos(&self) -> usize {
        self.t.get(self.i).map_or(self.end, |t| t.pos)
    }
    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(CoreError::parse("formula", msg, Some(self.pos())))
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn nest<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return self.err("expression nested too deeply");
        }
        let r = f(self);
        self.depth -= 1;
        r
    }

    fn expr(&mut self) -> Result<Expr> {
        self.nest(|p| p.or())
    }
    fn or(&mut self) -> Result<Expr> {
        let mut l = self.and()?;
        while self.eat(&Tok::Or) {
            let r = self.and()?;
            l = Expr::Binary(BinOp::Or, Box::new(l), Box::new(r));
        }
        Ok(l)
    }
    fn and(&mut self) -> Result<Expr> {
        let mut l = self.not()?;
        while self.eat(&Tok::And) {
            let r = self.not()?;
            l = Expr::Binary(BinOp::And, Box::new(l), Box::new(r));
        }
        Ok(l)
    }
    fn not(&mut self) -> Result<Expr> {
        if self.eat(&Tok::Not) {
            let e = self.nest(|p| p.not())?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(e)));
        }
        self.cmp()
    }
    fn cmp(&mut self) -> Result<Expr> {
        let l = self.add()?;
        let op = match self.peek() {
            Some(Tok::Eq) => BinOp::Eq,
            Some(Tok::Ne) => BinOp::Ne,
            Some(Tok::Lt) => BinOp::Lt,
            Some(Tok::Le) => BinOp::Le,
            Some(Tok::Gt) => BinOp::Gt,
            Some(Tok::Ge) => BinOp::Ge,
            _ => return Ok(l),
        };
        self.i += 1;
        let r = self.add()?;
        if matches!(self.peek(), Some(Tok::Eq | Tok::Ne | Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge)) {
            return self.err("comparisons cannot be chained; combine them with AND");
        }
        Ok(Expr::Binary(op, Box::new(l), Box::new(r)))
    }
    fn add(&mut self) -> Result<Expr> {
        let mut l = self.mul()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => BinOp::Add,
                Some(Tok::Minus) => BinOp::Sub,
                _ => return Ok(l),
            };
            self.i += 1;
            let r = self.mul()?;
            l = Expr::Binary(op, Box::new(l), Box::new(r));
        }
    }
    fn mul(&mut self) -> Result<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => BinOp::Mul,
                Some(Tok::Slash) => BinOp::Div,
                _ => return Ok(l),
            };
            self.i += 1;
            let r = self.unary()?;
            l = Expr::Binary(op, Box::new(l), Box::new(r));
        }
    }
    fn unary(&mut self) -> Result<Expr> {
        if self.eat(&Tok::Minus) {
            let e = self.nest(|p| p.unary())?;
            // Fold literal negation so "-3" stays a number.
            return Ok(match e {
                Expr::Number(x) => Expr::Number(-x),
                e => Expr::Unary(UnOp::Neg, Box::new(e)),
            });
        }
        if self.eat(&Tok::Plus) {
            let e = self.nest(|p| p.unary())?;
            return Ok(Expr::Unary(UnOp::Plus, Box::new(e)));
        }
        self.pow()
    }
    fn pow(&mut self) -> Result<Expr> {
        let base = self.primary()?;
        if self.eat(&Tok::Pow) {
            let exp = self.nest(|p| p.unary())?;
            return Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }
    /// Argument list after the opening parenthesis.
    fn call(&mut self, name: String, pos: usize) -> Result<Expr> {
        let mut args = Vec::new();
        if !self.eat(&Tok::RParen) {
            loop {
                args.push(self.expr()?);
                if self.eat(&Tok::RParen) {
                    break;
                }
                if !self.eat(&Tok::Comma) {
                    return self.err("expected ',' or ')' in argument list");
                }
            }
        }
        Ok(Expr::Call { name, args, pos })
    }

    fn primary(&mut self) -> Result<Expr> {
        let pos = self.pos();
        let Some(tok) = self.peek().cloned() else {
            return self.err("unexpected end of formula");
        };
        self.i += 1;
        match tok {
            Tok::Number(x) => Ok(Expr::Number(x)),
            Tok::Text(s) => Ok(Expr::Text(s)),
            Tok::Star => Ok(Expr::Missing),
            Tok::QuotedName(n) => Ok(Expr::ColumnName { name: n, pos }),
            Tok::LParen => {
                let e = self.expr()?;
                if !self.eat(&Tok::RParen) {
                    return self.err("expected ')'");
                }
                Ok(e)
            }
            // AND(…) / OR(…) / NOT(…) in operand position are function calls.
            Tok::And | Tok::Or | Tok::Not if self.peek() == Some(&Tok::LParen) => {
                let name = match tok {
                    Tok::And => "AND",
                    Tok::Or => "OR",
                    _ => "NOT",
                };
                self.i += 1;
                self.call(name.to_string(), pos)
            }
            Tok::Ident(id) => {
                if self.eat(&Tok::LParen) {
                    return self.call(id.to_ascii_uppercase(), pos);
                }
                Ok(classify_ident(&id, pos))
            }
            other => {
                self.i -= 1;
                self.err(format!("unexpected {}", describe(&other)))
            }
        }
    }
}

fn classify_ident(id: &str, pos: usize) -> Expr {
    let up = id.to_ascii_uppercase();
    let num = |p: char| -> Option<usize> {
        let rest = up.strip_prefix(p)?;
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        rest.parse().ok().filter(|&n| n > 0)
    };
    if let Some(n) = num('C') {
        Expr::ColumnNumber { n, pos }
    } else if let Some(k) = num('K') {
        Expr::Constant { k, pos }
    } else {
        Expr::ColumnName { name: id.to_string(), pos }
    }
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::RParen => "')'".into(),
        Tok::Comma => "','".into(),
        Tok::Slash => "'/'".into(),
        Tok::Pow => "'**'".into(),
        Tok::Eq | Tok::Ne | Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge => "comparison operator".into(),
        Tok::And | Tok::Or => "logical operator".into(),
        other => format!("{other:?}"),
    }
}

/// Parse a formula into an expression tree.
pub fn parse(src: &str) -> Result<Expr> {
    let t = tokenize(src)?;
    if t.is_empty() {
        return Err(CoreError::parse("formula", "empty formula", Some(0)));
    }
    let mut p = P { t, i: 0, end: src.len(), depth: 0 };
    let e = p.expr()?;
    if p.i != p.t.len() {
        return p.err("unexpected text after end of expression");
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure() {
        assert_eq!(parse("1 + 2 * 3").unwrap().to_string(), "(1 + (2 * 3))");
        assert_eq!(parse("-x ** 2").unwrap().to_string(), "(-(x ** 2))");
        assert_eq!(parse("a OR b AND NOT c = 1").unwrap().to_string(), "(a OR (b AND (NOT (c = 1))))");
        assert_eq!(parse("mean(C1) - K2").unwrap().to_string(), "(MEAN(C1) - K2)");
        assert_eq!(parse("IF(x, *, 'Long name')").unwrap().to_string(), "IF(x, *, 'Long name')");
        assert_eq!(parse("-3").unwrap(), Expr::Number(-3.0));
        assert_eq!(parse("2 ** -3").unwrap().to_string(), "(2 ** -3)");
    }

    #[test]
    fn display_roundtrips() {
        for s in ["(a + b) * c / -d", "IF(C1 > 2, \"x\"\"y\", *)", "NOT a OR b <> 3", "x ** y ** z", "PI()"] {
            let e = parse(s).unwrap();
            assert_eq!(parse(&e.to_string()).unwrap().to_string(), e.to_string());
        }
    }

    #[test]
    fn errors_and_limits() {
        for bad in ["", "1 +", "(1", "f(1,", "1 2", "a < b < c", ")", "1 , 2"] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        let deep = "(".repeat(MAX_NESTING + 5) + "1" + &")".repeat(MAX_NESTING + 5);
        assert!(parse(&deep).is_err());
        let ok = "(".repeat(50) + "1" + &")".repeat(50);
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn refs() {
        let e = parse("C3 + 'Weight' * K1 - MEAN(c3)").unwrap();
        let r: Vec<_> = e.column_refs().into_iter().collect();
        assert_eq!(
            r,
            vec![super::super::ast::ColumnRef::Number(3), super::super::ast::ColumnRef::Name("Weight".into())]
        );
        assert_eq!(e.constant_refs().into_iter().collect::<Vec<_>>(), vec![1]);
    }
}
