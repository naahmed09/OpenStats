//! Model formulas and design matrices.
//!
//! * [`Formula`] — parser for R / Minitab model formulas (main effects,
//!   crossing `A*B` / `A|B`, interactions `A:B` / `A.B`, nesting `A(B)`,
//!   polynomials `I(X^2)` / `X|X` / `poly(x, k)`, transformations, offsets,
//!   intercept control).
//! * [`Data`], [`ModelSpec`], [`DesignMatrix`] — encoding of worksheet
//!   columns into numeric design matrices with treatment, sum-to-zero,
//!   Helmert, orthogonal-polynomial or custom [`Contrast`] codings.

mod design;
mod expr;
mod formula;
mod lexer;

pub(crate) use design::level_sort;
pub use design::{build_design, Contrast, Data, DesignMatrix, DesignOptions, ModelSpec, Reference, TermColumns};
pub use expr::Expr;
pub use formula::{Atom, AtomKind, Formula, Term};
