//! Syntax tree of a script.

use std::fmt;

/// Binary operators, in the formula engine's vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `**` or `^`
    Pow,
    /// `=`, `==`, `EQ`
    Eq,
    /// `<>`, `!=`, `NE`
    Ne,
    /// `<`, `LT`
    Lt,
    /// `<=`, `LE`
    Le,
    /// `>`, `GT`
    Gt,
    /// `>=`, `GE`
    Ge,
    /// `AND`, `&`
    And,
    /// `OR`, `|`
    Or,
}

impl BinOp {
    /// The operator as the formula engine spells it.
    pub fn symbol(self) -> &'static str {
        match self {
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
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// `-x`
    Neg,
    /// `+x`
    Plus,
    /// `NOT x`, `~x`
    Not,
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A number.
    Num(f64),
    /// `"text"`.
    Text(String),
    /// `*`, the missing value.
    Missing,
    /// A word: `C3`, `K1`, `M2`, a macro argument or an unquoted column name.
    Word(String),
    /// `'column name'`.
    Quoted(String),
    /// Element access: `C1(5)`, `M1(2, 3)`, `'Sales'(4)`.
    Index(Box<Expr>, Vec<Expr>),
    /// Unary operation.
    Unary(UnOp, Box<Expr>),
    /// Binary operation.
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// `NAME(args)`: a function call, or element access when `NAME` turns out to be a column or
    /// matrix (decided when the script runs, because a macro argument can stand for either).
    Call(String, Vec<Expr>),
}

fn quote(s: &str, q: char) -> String {
    let doubled = s.replace(q, &format!("{q}{q}"));
    format!("{q}{doubled}{q}")
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(x) => write!(f, "{x}"),
            Expr::Text(s) => f.write_str(&quote(s, '"')),
            Expr::Missing => f.write_str("*"),
            Expr::Word(w) => f.write_str(w),
            Expr::Quoted(s) => f.write_str(&quote(s, '\'')),
            Expr::Index(b, a) => {
                write!(f, "{b}(")?;
                list(f, a)?;
                f.write_str(")")
            }
            Expr::Unary(UnOp::Neg, e) => write!(f, "(-{e})"),
            Expr::Unary(UnOp::Plus, e) => write!(f, "(+{e})"),
            Expr::Unary(UnOp::Not, e) => write!(f, "(NOT {e})"),
            Expr::Binary(op, a, b) => write!(f, "({a} {} {b})", op.symbol()),
            Expr::Call(n, a) => {
                write!(f, "{n}(")?;
                list(f, a)?;
                f.write_str(")")
            }
        }
    }
}

fn list(f: &mut fmt::Formatter<'_>, items: &[Expr]) -> fmt::Result {
    for (i, e) in items.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{e}")?;
    }
    Ok(())
}

/// One command argument. Arguments are separated by white space, not commas.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// A number (a leading `-` is part of it).
    Num(f64),
    /// `"text"`.
    Text(String),
    /// `'column name'`.
    Quoted(String),
    /// A word: `C1`, `K2`, `M1`, a keyword, or a macro argument name.
    Word(String),
    /// `*`.
    Missing,
    /// `C1-C5`, `C1:C5`, `1:10`.
    Range(Box<Arg>, Box<Arg>),
    /// `( expression )`, evaluated when the command runs.
    Paren(Expr),
}

/// A `;` subcommand with its arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct Sub {
    /// 1-based source line.
    pub line: usize,
    /// Subcommand name as typed.
    pub name: String,
    /// Arguments.
    pub args: Vec<Arg>,
}

/// A value on an inline data line (`SET` / `READ`).
#[derive(Debug, Clone, PartialEq)]
pub enum Datum {
    /// A number.
    Num(f64),
    /// Text.
    Text(String),
    /// `*`.
    Missing,
}

/// One line of inline data.
#[derive(Debug, Clone, PartialEq)]
pub struct DataLine {
    /// 1-based source line.
    pub line: usize,
    /// The values on it.
    pub items: Vec<Datum>,
}

/// A command with subcommands and inline data.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    /// The command word as typed.
    pub name: String,
    /// Arguments.
    pub args: Vec<Arg>,
    /// Subcommands, in order.
    pub subs: Vec<Sub>,
    /// Inline data between the command and `END`.
    pub data: Vec<DataLine>,
}

/// Kinds of macro-variable declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKind {
    /// `MCOLUMN`
    Column,
    /// `MCONSTANT`
    Constant,
    /// `MMATRIX`
    Matrix,
    /// `MTEXT`: a text (string) variable, `S1` style.
    Text,
    /// `MSTORE`: a variable of any kind. As an argument it takes whatever the caller passes
    /// (column, constant, matrix, string or a literal); as a local it takes its kind from the
    /// first value assigned to it.
    Store,
}

/// A statement with its source line.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    /// 1-based source line of the statement's first token.
    pub line: usize,
    /// What it is.
    pub kind: StmtKind,
}

/// The kinds of statement.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    /// `LET target = value`.
    Let {
        /// A constant, column, matrix or element (`C1(3)`).
        target: Expr,
        /// The expression.
        value: Expr,
    },
    /// `DO K1 = from:to [BY step] … ENDDO`.
    Do {
        /// The loop variable (a `K` constant or a constant argument).
        var: Expr,
        /// First value.
        from: Expr,
        /// Last value (inclusive).
        to: Expr,
        /// Increment (default 1).
        step: Option<Expr>,
        /// Loop body.
        body: Vec<Stmt>,
    },
    /// `WHILE cond … ENDWHILE`.
    While {
        /// Loop condition.
        cond: Expr,
        /// Loop body.
        body: Vec<Stmt>,
    },
    /// `IF … ELSEIF … ELSE … ENDIF`.
    If {
        /// `(condition, body)` for the `IF` and each `ELSEIF`.
        arms: Vec<(Expr, Vec<Stmt>)>,
        /// The `ELSE` body.
        otherwise: Option<Vec<Stmt>>,
    },
    /// `GOTO label`.
    Goto(String),
    /// `MLABEL label`.
    Label(String),
    /// `RETURN`: leave the current macro.
    Return,
    /// `STOP`: end the whole run.
    Stop,
    /// `ARGS names…`: further (optional) macro arguments.
    Args(Vec<String>),
    /// `MCOLUMN` / `MCONSTANT` / `MMATRIX` / `MTEXT` / `MSTORE` `names…`.
    Declare {
        /// Kind of variable.
        kind: DeclKind,
        /// Names declared.
        names: Vec<String>,
    },
    /// `%name args…`: call a macro. `;` subcommands give arguments by name (`%m C1; K 3.`).
    Call {
        /// Macro (or file) name, without the `%`.
        name: String,
        /// Positional arguments.
        args: Vec<Arg>,
        /// Arguments given by name, one subcommand each.
        subs: Vec<Sub>,
    },
    /// Any other command.
    Command(Command),
}

/// A macro definition.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroDef {
    /// Name (as typed; matched case-insensitively).
    pub name: String,
    /// Defined with `GMACRO` (no arguments) rather than `MACRO`.
    pub global: bool,
    /// Argument names from the header.
    pub params: Vec<String>,
    /// Statements.
    pub body: Vec<Stmt>,
    /// Line of `MACRO` / `GMACRO`.
    pub line: usize,
    /// Line of `ENDMACRO`.
    pub end_line: usize,
}

/// A parsed script: macro definitions and top-level (exec-file style) statements.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    /// Macros defined in the script.
    pub macros: Vec<MacroDef>,
    /// Statements outside any macro.
    pub main: Vec<Stmt>,
}
