//! Formula syntax tree.

use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Plus,
    Not,
}

/// Expression node. `pos` fields are byte offsets into the source.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Text(String),
    Missing,
    /// `C#` (1-based).
    ColumnNumber {
        n: usize,
        pos: usize,
    },
    /// Column by name (bare or quoted).
    ColumnName {
        name: String,
        pos: usize,
    },
    /// `K#` (1-based).
    Constant {
        k: usize,
        pos: usize,
    },
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call {
        name: String,
        args: Vec<Expr>,
        pos: usize,
    },
}

/// A column reference found in an expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ColumnRef {
    Number(usize),
    Name(String),
}

impl Expr {
    /// Column references used by the expression (for dependency tracking).
    pub fn column_refs(&self) -> BTreeSet<ColumnRef> {
        let mut out = BTreeSet::new();
        self.walk(&mut |e| match e {
            Expr::ColumnNumber { n, .. } => {
                out.insert(ColumnRef::Number(*n));
            }
            Expr::ColumnName { name, .. } => {
                out.insert(ColumnRef::Name(name.clone()));
            }
            _ => {}
        });
        out
    }

    /// Constants (`K#`) used by the expression.
    pub fn constant_refs(&self) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        self.walk(&mut |e| {
            if let Expr::Constant { k, .. } = e {
                out.insert(*k);
            }
        });
        out
    }

    fn walk(&self, f: &mut dyn FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Unary(_, a) => a.walk(f),
            Expr::Binary(_, a, b) => {
                a.walk(f);
                b.walk(f);
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| a.walk(f)),
            _ => {}
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "**",
            BinOp::Eq => "=",
            BinOp::Ne => "<>",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "AND",
            BinOp::Or => "OR",
        })
    }
}

/// Canonical, fully parenthesised source text (round-trips through the parser).
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(x) => write!(f, "{}", crate::value::format_number(*x)),
            Expr::Text(s) => write!(f, "\"{}\"", s.replace('"', "\"\"")),
            Expr::Missing => f.write_str("*"),
            Expr::ColumnNumber { n, .. } => write!(f, "C{n}"),
            Expr::ColumnName { name, .. } => f.write_str(&crate::refs::quote_name(name)),
            Expr::Constant { k, .. } => write!(f, "K{k}"),
            Expr::Unary(UnOp::Neg, a) => write!(f, "(-{a})"),
            Expr::Unary(UnOp::Plus, a) => write!(f, "(+{a})"),
            Expr::Unary(UnOp::Not, a) => write!(f, "(NOT {a})"),
            Expr::Binary(op, a, b) => write!(f, "({a} {op} {b})"),
            Expr::Call { name, args, .. } => {
                write!(f, "{name}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{a}")?;
                }
                f.write_str(")")
            }
        }
    }
}
