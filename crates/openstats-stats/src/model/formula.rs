//! Model-formula parser (R / Minitab hybrid).
//!
//! ```text
//! formula  := [ response "~" ] terms
//! terms    := item { ("+" | "-" | whitespace) item }
//! item     := cross
//! cross    := inter { ("*" | "|" | "/") inter }      crossing / nesting
//! inter    := power { (":" | ".") power }            interaction only
//! power    := primary [ "^" k ]                      (group)^k: all interactions up to order k
//! primary  := name | name(nested-in)  | I(expr) | fn(expr) | C(name)
//!           | offset(expr) | "(" terms ")" | 0 | 1
//! ```
//!
//! * `A*B` and `A|B` are the crossing A + B + A:B (so `X|X` is `X + X²` and
//!   `X|X|X` is `X + X² + X³`); `A:B` and `A.B` are the interaction alone.
//! * `A(B)` is A nested within B; `A/B` is `A + B(A)`.
//! * `I(X^2)`, `log(x)`, `sqrt(x)`, … are transformations of numeric columns.
//! * `poly(x, k)` is `x + x² + … + x^k` (raw polynomial).
//! * `0`, `-1` remove and `1` restores the intercept; `offset(z)` adds a
//!   column with coefficient fixed at 1.
//! * `C(x)` / `factor(x)` treats a numeric column as categorical.
//! * Names with dots or spaces are quoted with backticks, `"` or `'`.
//! * Terms are ordered by interaction order, then by appearance.

use super::expr::{parse_expr, Expr, FUNCTIONS};
use super::lexer::{tokenize, Tok, Token};
use crate::error::{invalid, Result};

/// What an atom of a term refers to.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomKind {
    /// A worksheet column (numeric or categorical, decided by the data).
    Var(String),
    /// A numeric transformation.
    Expr(Expr),
    /// A numeric column treated as a factor.
    AsFactor(String),
}

/// One variable of a term.
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Display / identity label (`x`, `I(x^2)`, `C(dose)`).
    pub label: String,
    /// Meaning.
    pub kind: AtomKind,
}

/// A model term: a product of atoms (with powers), optionally nested.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// Atoms with their powers, in order of appearance.
    pub factors: Vec<(Atom, u32)>,
    /// Atoms the term is nested within (`A(B)` → `[B]`).
    pub nested_in: Vec<Atom>,
}

impl Term {
    fn single(a: Atom) -> Term {
        Term { factors: vec![(a, 1)], nested_in: Vec::new() }
    }

    /// Interaction order (sum of powers).
    pub fn order(&self) -> u32 {
        self.factors.iter().map(|(_, p)| *p).sum()
    }

    /// Display label: `A`, `A*B`, `x^2`, `A(B)`.
    pub fn label(&self) -> String {
        let body = self
            .factors
            .iter()
            .map(|(a, p)| if *p == 1 { a.label.clone() } else { format!("{}^{}", a.label, p) })
            .collect::<Vec<_>>()
            .join("*");
        if self.nested_in.is_empty() {
            body
        } else {
            format!("{}({})", body, self.nested_in.iter().map(|a| a.label.clone()).collect::<Vec<_>>().join(" "))
        }
    }

    /// Order-independent identity.
    fn key(&self) -> String {
        let mut f: Vec<String> = self.factors.iter().map(|(a, p)| format!("{}^{}", a.label, p)).collect();
        f.sort();
        let mut n: Vec<String> = self.nested_in.iter().map(|a| a.label.clone()).collect();
        n.sort();
        format!("{}|{}", f.join("*"), n.join(" "))
    }
}

/// A parsed model formula.
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    /// Response expression (`None` for a right-hand side only).
    pub response: Option<Expr>,
    /// Model terms in order (by interaction order, then appearance).
    pub terms: Vec<Term>,
    /// Whether the model has an intercept.
    pub intercept: bool,
    /// Offset expressions.
    pub offsets: Vec<Expr>,
    /// The source text.
    pub source: String,
}

impl Formula {
    /// Parse a formula string.
    pub fn parse(src: &str) -> Result<Formula> {
        let toks = tokenize(src)?;
        if toks.is_empty() {
            return invalid("formula is empty");
        }
        let mut depth = 0i32;
        let mut tilde = None;
        for (i, t) in toks.iter().enumerate() {
            match t.tok {
                Tok::Op('(') => depth += 1,
                Tok::Op(')') => depth -= 1,
                Tok::Op('~') if depth == 0 => {
                    if tilde.is_some() {
                        return invalid("formula has more than one '~'");
                    }
                    tilde = Some(i);
                }
                _ => {}
            }
        }
        let (response, rhs) = match tilde {
            Some(i) => {
                if i == 0 {
                    return invalid("formula: nothing before '~'");
                }
                (Some(parse_expr(&toks[..i])?), &toks[i + 1..])
            }
            None => (None, &toks[..]),
        };
        if rhs.is_empty() {
            return invalid("formula: nothing after '~'");
        }
        let mut p = Parser { toks: rhs, i: 0 };
        let e = p.sum()?;
        if p.i != rhs.len() {
            return invalid(format!("formula: unexpected token at position {}", rhs[p.i].pos));
        }
        let mut terms = e.terms;
        // Stable sort by interaction order.
        terms.sort_by_key(Term::order);
        let intercept = e.intercept.unwrap_or(true);
        if terms.is_empty() && !intercept && e.offsets.is_empty() {
            return invalid("formula has no terms and no intercept");
        }
        Ok(Formula { response, terms, intercept, offsets: e.offsets, source: src.to_string() })
    }

    /// Labels of the terms.
    pub fn term_labels(&self) -> Vec<String> {
        self.terms.iter().map(Term::label).collect()
    }
}

#[derive(Debug, Clone, Default)]
struct Expansion {
    terms: Vec<Term>,
    intercept: Option<bool>,
    offsets: Vec<Expr>,
}

impl Expansion {
    fn push_unique(&mut self, t: Term) {
        let k = t.key();
        if !self.terms.iter().any(|x| x.key() == k) {
            self.terms.push(t);
        }
    }
}

/// Merge two terms; powers add (or, with `as_set`, atoms are united).
fn merge(a: &Term, b: &Term, as_set: bool) -> Term {
    let mut factors = a.factors.clone();
    for (atom, p) in &b.factors {
        if let Some(f) = factors.iter_mut().find(|(x, _)| x.label == atom.label) {
            if !as_set {
                f.1 += p;
            }
        } else {
            factors.push((atom.clone(), *p));
        }
    }
    let mut nested = a.nested_in.clone();
    for n in &b.nested_in {
        if !nested.iter().any(|x| x.label == n.label) {
            nested.push(n.clone());
        }
    }
    Term { factors, nested_in: nested }
}

fn product(a: &Expansion, b: &Expansion, as_set: bool) -> Expansion {
    let mut out = Expansion::default();
    for x in &a.terms {
        for y in &b.terms {
            out.push_unique(merge(x, y, as_set));
        }
    }
    out
}

fn union(a: &Expansion, b: &Expansion) -> Expansion {
    let mut out = Expansion { terms: Vec::new(), intercept: None, offsets: Vec::new() };
    for t in a.terms.iter().chain(&b.terms) {
        out.push_unique(t.clone());
    }
    out.offsets = a.offsets.iter().chain(&b.offsets).cloned().collect();
    out
}

struct Parser<'a> {
    toks: &'a [Token],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|t| &t.tok)
    }

    fn expect_close(&mut self, what: &str) -> Result<()> {
        match self.peek() {
            Some(Tok::Op(')')) => {
                self.i += 1;
                Ok(())
            }
            _ => invalid(format!("formula: missing ')' after {what}")),
        }
    }

    fn sum(&mut self) -> Result<Expansion> {
        let mut acc = Expansion::default();
        let mut sign = '+';
        match self.peek() {
            Some(Tok::Op('+')) => self.i += 1,
            Some(Tok::Op('-')) => {
                sign = '-';
                self.i += 1;
            }
            _ => {}
        }
        loop {
            let (item, is_num) = self.item()?;
            if is_num {
                let v = item.intercept.unwrap_or(true);
                acc.intercept = Some(if sign == '+' { v } else { !v });
            } else if sign == '+' {
                let prev = acc.intercept;
                acc = union(&acc, &item);
                acc.intercept = item.intercept.or(prev);
            } else {
                let keys: Vec<String> = item.terms.iter().map(Term::key).collect();
                acc.terms.retain(|t| !keys.contains(&t.key()));
            }
            match self.peek() {
                Some(Tok::Op('+')) => {
                    sign = '+';
                    self.i += 1;
                }
                Some(Tok::Op('-')) => {
                    sign = '-';
                    self.i += 1;
                }
                Some(Tok::Ident(_) | Tok::Num(_) | Tok::Op('(')) => sign = '+',
                _ => break,
            }
        }
        Ok(acc)
    }

    /// One item; the flag says it was a bare `0` / `1`.
    fn item(&mut self) -> Result<(Expansion, bool)> {
        if let Some(Tok::Num(v)) = self.peek() {
            let v = *v;
            if v != 0.0 && v != 1.0 {
                return invalid(format!("formula: the number {v} may only appear as 0 or 1 (intercept control)"));
            }
            self.i += 1;
            if matches!(self.peek(), Some(Tok::Op('*' | ':' | '|' | '/' | '^' | '.'))) {
                return invalid("formula: 0 and 1 cannot be used as operands");
            }
            return Ok((Expansion { intercept: Some(v == 1.0), ..Expansion::default() }, true));
        }
        Ok((self.cross()?, false))
    }

    fn cross(&mut self) -> Result<Expansion> {
        let mut l = self.inter()?;
        while let Some(Tok::Op(c @ ('*' | '|' | '/'))) = self.peek() {
            let c = *c;
            self.i += 1;
            let r = self.inter()?;
            l = if c == '/' {
                // A/B = A + B(A): every term of B nested within all atoms of A.
                let mut ctx: Vec<Atom> = Vec::new();
                for t in &l.terms {
                    for (a, _) in &t.factors {
                        if !ctx.iter().any(|x| x.label == a.label) {
                            ctx.push(a.clone());
                        }
                    }
                }
                let mut out = l.clone();
                for t in &r.terms {
                    let mut nt = t.clone();
                    for a in &ctx {
                        if !nt.nested_in.iter().any(|x| x.label == a.label) {
                            nt.nested_in.push(a.clone());
                        }
                    }
                    out.push_unique(nt);
                }
                out
            } else {
                let mut out = union(&l, &r);
                for t in product(&l, &r, false).terms {
                    out.push_unique(t);
                }
                out
            };
        }
        Ok(l)
    }

    fn inter(&mut self) -> Result<Expansion> {
        let mut l = self.power()?;
        while let Some(Tok::Op(':' | '.')) = self.peek() {
            self.i += 1;
            let r = self.power()?;
            l = product(&l, &r, false);
        }
        Ok(l)
    }

    fn power(&mut self) -> Result<Expansion> {
        let (e, is_group) = self.primary()?;
        if let Some(Tok::Op('^')) = self.peek() {
            self.i += 1;
            if !is_group {
                return invalid("formula: '^' needs a parenthesised group, e.g. (A+B)^2; for a squared numeric term write I(X^2) or X|X");
            }
            let k = match self.peek() {
                Some(Tok::Num(v)) if *v >= 1.0 && v.fract() == 0.0 && *v <= 8.0 => *v as usize,
                _ => return invalid("formula: the exponent of a group must be an integer from 1 to 8"),
            };
            self.i += 1;
            let mut acc = e.clone();
            for _ in 1..k {
                let mut next = union(&acc, &e);
                for t in product(&acc, &e, true).terms {
                    next.push_unique(t);
                }
                acc = next;
            }
            return Ok(acc);
        }
        Ok(e)
    }

    fn atom_expansion(a: Atom) -> Expansion {
        Expansion { terms: vec![Term::single(a)], intercept: None, offsets: Vec::new() }
    }

    /// Tokens between the `(` at `self.i` and its matching `)`; advances past it.
    fn balanced(&mut self) -> Result<&[Token]> {
        let start = self.i + 1;
        let mut depth = 0;
        let mut j = self.i;
        while j < self.toks.len() {
            match self.toks[j].tok {
                Tok::Op('(') => depth += 1,
                Tok::Op(')') => {
                    depth -= 1;
                    if depth == 0 {
                        let inner = &self.toks[start..j];
                        self.i = j + 1;
                        return Ok(inner);
                    }
                }
                _ => {}
            }
            j += 1;
        }
        invalid("formula: unbalanced parentheses")
    }

    fn primary(&mut self) -> Result<(Expansion, bool)> {
        let Some(t) = self.toks.get(self.i) else {
            return invalid("formula: unexpected end");
        };
        let pos = t.pos;
        match &t.tok {
            Tok::Op('(') => {
                self.i += 1;
                let e = self.sum()?;
                self.expect_close("group")?;
                Ok((e, true))
            }
            Tok::Ident(name) => {
                let name = name.clone();
                self.i += 1;
                let adjacent_paren =
                    matches!(self.toks.get(self.i), Some(Token { tok: Tok::Op('('), space_before: false, .. }));
                if !adjacent_paren {
                    return Ok((Self::atom_expansion(Atom { label: name.clone(), kind: AtomKind::Var(name) }), false));
                }
                match name.as_str() {
                    "offset" => {
                        let inner = self.balanced()?.to_vec();
                        let e = parse_expr(&inner)?;
                        Ok((Expansion { offsets: vec![e], ..Expansion::default() }, false))
                    }
                    "C" | "factor" => {
                        let inner = self.balanced()?.to_vec();
                        match inner.as_slice() {
                            [Token { tok: Tok::Ident(v), .. }] => Ok((
                                Self::atom_expansion(Atom {
                                    label: format!("{name}({v})"),
                                    kind: AtomKind::AsFactor(v.clone()),
                                }),
                                false,
                            )),
                            _ => invalid(format!("formula: {name}(...) takes one column name")),
                        }
                    }
                    "poly" => {
                        let inner = self.balanced()?.to_vec();
                        let comma = inner.iter().position(|t| t.tok == Tok::Op(','));
                        let (var, k) = match (comma, inner.first()) {
                            (Some(1), Some(Token { tok: Tok::Ident(v), .. })) => match inner.get(2) {
                                Some(Token { tok: Tok::Num(k), .. })
                                    if inner.len() == 3 && k.fract() == 0.0 && (1.0..=10.0).contains(k) =>
                                {
                                    (v.clone(), *k as u32)
                                }
                                _ => return invalid("formula: poly(x, k) needs an integer degree from 1 to 10"),
                            },
                            _ => return invalid("formula: poly(x, k) takes a column name and a degree"),
                        };
                        let atom = Atom { label: var.clone(), kind: AtomKind::Var(var) };
                        let terms =
                            (1..=k).map(|p| Term { factors: vec![(atom.clone(), p)], nested_in: Vec::new() }).collect();
                        Ok((Expansion { terms, ..Expansion::default() }, false))
                    }
                    f if FUNCTIONS.contains(&f) => {
                        let inner = self.balanced()?.to_vec();
                        let e = parse_expr(&inner)?;
                        let (label, expr) = if f == "I" {
                            (format!("I({})", e.text()), e)
                        } else {
                            (format!("{f}({})", e.text()), Expr::Func(f.to_string(), Box::new(e)))
                        };
                        Ok((Self::atom_expansion(Atom { label, kind: AtomKind::Expr(expr) }), false))
                    }
                    _ => {
                        // A(B C ...): A nested within the cells of B, C, ...
                        self.i += 1; // '('
                        let ctx = self.sum()?;
                        self.expect_close(&format!("{name}("))?;
                        let mut atoms: Vec<Atom> = Vec::new();
                        for t in &ctx.terms {
                            if !t.nested_in.is_empty() {
                                return invalid("formula: nesting inside a nesting list is not supported");
                            }
                            for (a, _) in &t.factors {
                                if !atoms.iter().any(|x| x.label == a.label) {
                                    atoms.push(a.clone());
                                }
                            }
                        }
                        if atoms.is_empty() {
                            return invalid(format!("formula: {name}() has an empty nesting list"));
                        }
                        let main = Atom { label: name.clone(), kind: AtomKind::Var(name) };
                        Ok((
                            Expansion {
                                terms: vec![Term { factors: vec![(main, 1)], nested_in: atoms }],
                                ..Expansion::default()
                            },
                            false,
                        ))
                    }
                }
            }
            Tok::Op('.') => invalid(format!(
                "formula: unexpected '.' at position {pos}; quote names containing dots with backticks"
            )),
            other => invalid(format!("formula: unexpected {other:?} at position {pos}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(s: &str) -> Vec<String> {
        Formula::parse(s).unwrap().term_labels()
    }

    #[test]
    fn main_effects_and_response() {
        let f = Formula::parse("y ~ A + B + x").unwrap();
        assert_eq!(f.term_labels(), ["A", "B", "x"]);
        assert!(f.intercept && f.response.as_ref().unwrap().text() == "y");
        // Minitab style: whitespace separated, no response.
        let g = Formula::parse("A B x").unwrap();
        assert_eq!(g.term_labels(), ["A", "B", "x"]);
        assert!(g.response.is_none());
    }

    #[test]
    fn crossing_and_interaction() {
        assert_eq!(labels("y ~ A*B"), ["A", "B", "A*B"]);
        assert_eq!(labels("y ~ A|B"), ["A", "B", "A*B"]);
        assert_eq!(labels("y ~ A:B"), ["A*B"]);
        assert_eq!(labels("y ~ A.B"), ["A*B"]);
        assert_eq!(labels("y ~ A*B*C"), ["A", "B", "C", "A*B", "A*C", "B*C", "A*B*C"]);
        assert_eq!(labels("y ~ (A+B+C)^2"), ["A", "B", "C", "A*B", "A*C", "B*C"]);
        assert_eq!(labels("y ~ A*B - A:B"), ["A", "B"]);
        // Duplicates and ordering by interaction order.
        assert_eq!(labels("y ~ A:B + A + B + A"), ["A", "B", "A*B"]);
        assert_eq!(labels("y ~ B:A + A*B"), ["A", "B", "B*A"]);
    }

    #[test]
    fn polynomials() {
        assert_eq!(labels("y ~ X|X"), ["X", "X^2"]);
        assert_eq!(labels("y ~ X|X|X"), ["X", "X^2", "X^3"]);
        assert_eq!(labels("y ~ x + I(x^2) + I(x^3)"), ["x", "I(x^2)", "I(x^3)"]);
        assert_eq!(labels("y ~ poly(x, 3)"), ["x", "x^2", "x^3"]);
        assert_eq!(labels("y ~ log(x) + sqrt(z)"), ["log(x)", "sqrt(z)"]);
        assert_eq!(labels("y ~ I(x*z + 1/w)"), ["I((x*z)+(1/w))"]);
        assert!(Formula::parse("y ~ x^2").is_err());
        assert!(Formula::parse("y ~ I(x^2").is_err());
        assert!(Formula::parse("y ~ I(x +)").is_err());
        assert!(Formula::parse("y ~ I(foo(x))").is_err());
    }

    #[test]
    fn nesting() {
        assert_eq!(labels("y ~ B + A(B)"), ["B", "A(B)"]);
        assert_eq!(labels("y ~ B A(B)"), ["B", "A(B)"]);
        assert_eq!(labels("y ~ B/A"), ["B", "A(B)"]);
        assert_eq!(labels("y ~ A(B C)"), ["A(B C)"]);
        assert_eq!(labels("y ~ A (B)"), ["A", "B"]); // space: two terms, not nesting
    }

    #[test]
    fn intercept_and_offsets() {
        assert!(!Formula::parse("y ~ 0 + x").unwrap().intercept);
        assert!(!Formula::parse("y ~ x - 1").unwrap().intercept);
        assert!(!Formula::parse("y ~ x + 0").unwrap().intercept);
        assert!(Formula::parse("y ~ 0 + x + 1").unwrap().intercept);
        assert!(Formula::parse("y ~ x - 0").unwrap().intercept);
        let f = Formula::parse("y ~ x + offset(log(t))").unwrap();
        assert_eq!(f.term_labels(), ["x"]);
        assert_eq!(f.offsets.len(), 1);
        assert_eq!(f.offsets[0].variables(), ["t"]);
        let i = Formula::parse("y ~ 1").unwrap();
        assert!(i.terms.is_empty() && i.intercept);
        assert!(Formula::parse("y ~ 0").is_err());
        assert!(Formula::parse("y ~ 2 + x").is_err());
    }

    #[test]
    fn factors_and_quotes() {
        assert_eq!(labels("y ~ C(dose) + `my var`"), ["C(dose)", "my var"]);
        assert_eq!(labels("y ~ factor(g)*x"), ["factor(g)", "x", "factor(g)*x"]);
        assert_eq!(labels("log(y) ~ x").len(), 1);
        assert_eq!(Formula::parse("log(y) ~ x").unwrap().response.unwrap().text(), "log(y)");
    }

    #[test]
    fn syntax_errors() {
        for bad in [
            "",
            "y ~",
            "~ x",
            "y ~ x ~ z",
            "y ~ (x",
            "y ~ x)",
            "y ~ x +* z",
            "y ~ A:",
            "y ~ .",
            "y ~ x $ z",
            "y ~ (A+B)^0",
            "y ~ 1:x",
        ] {
            assert!(Formula::parse(bad).is_err(), "{bad:?} should fail");
        }
    }
}
