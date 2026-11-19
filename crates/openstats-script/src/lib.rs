//! OpenStats macro and exec-file engine (std only).
//!
//! A Minitab-style scripting language for automating OpenStats:
//!
//! * [`lexer`] turns source text into tokens (`#` / `//` / `NOTE` comments, `&` continuation
//!   lines, `'column names'` and `'text'`, `"text"`, `K1`, `C1`, `M1`, `S1`, operators);
//! * [`parser`] is a recursive-descent parser producing the [`ast`]: macro definitions
//!   (`GMACRO` / `MACRO`, closed by `ENDMACRO`, `MEND` or `END`; `ARGS`, `MCOLUMN`, `MCONSTANT`,
//!   `MMATRIX`, `MTEXT`, `MSTORE`), control flow (`DO`, `WHILE`, `IF / ELSEIF / ELSE`, `GOTO` with
//!   `MLABEL` / `LABEL` / `name:` labels), assignments with or without `LET` (implicit
//!   declarations, string variables `S1`), commands with `;` subcommands ended by `.` and inline
//!   data (`SET`, `READ` … `END`), and macro calls (`%name args`, named arguments `; arg value`);
//! * [`compile`] flattens a body into jump-based [`compile::Op`]s, which makes `GOTO`, stepping
//!   and breakpoints exact;
//! * [`machine`] is the interpreter. A [`machine::Machine`] advances one source line at a time
//!   against an [`machine::Env`] (a workbook and its undo history), so a host can offer Run,
//!   Step and Stop. Arithmetic is evaluated by the formula engine of `openstats-core`; commands
//!   call `openstats-stats` (descriptive statistics, correlation, regression, t tests, one-way
//!   ANOVA, CART, random forests, gradient boosting) and their output is returned as report
//!   [`openstats_stats::report::Section`]s for the Session window;
//! * [`highlight`] classifies the tokens of a line for syntax colouring in an editor.
//!
//! Every change a script makes to a workbook goes through the workbook's [`History`], so a
//! whole run can be undone as one step by the host.
//!
//! [`History`]: openstats_core::history::History

pub mod ast;
pub mod commands;
pub mod compile;
pub mod error;
pub mod highlight;
pub mod lexer;
pub mod machine;
pub mod parser;

pub use error::{ErrorKind, ScriptError};
pub use machine::{Env, Library, Machine, MachineConfig, State};
pub use parser::parse;

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, ScriptError>;
