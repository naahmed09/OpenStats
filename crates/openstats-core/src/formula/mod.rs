//! Formula calculator (Minitab **Calc > Calculator** and column formulas).
//!
//! ```text
//! expr    := or
//! or      := and   ( (OR | '|') and )*
//! and     := not   ( (AND | '&') not )*
//! not     := (NOT | '~') not | cmp
//! cmp     := add   ( ('=' | '==' | '<>' | '!=' | '~=' | '<' | '<=' | '>' | '>=' | EQ NE LT LE GT GE) add )?
//! add     := mul   ( ('+' | '-') mul )*
//! mul     := unary ( ('*' | '/') unary )*
//! unary   := ('-' | '+') unary | pow
//! pow     := primary ( ('**' | '^') unary )?          (right-associative)
//! primary := number | "text" | '*' (missing) | C# | K# | name | 'quoted name'
//!          | FUNCTION '(' [expr (',' expr)*] ')' | '(' expr ')'
//! ```
//!
//! Evaluation is vectorised: columns are arrays, constants and column
//! statistics are scalars broadcast against them. Columns combined
//! element-wise must have equal length. Missing values (`*`) propagate through
//! arithmetic, comparisons and functions; `AND`/`OR` use three-valued logic.
//! Logical results are numeric 1/0. Date/time ± number shifts by days, date −
//! date gives days, and a date compared with text parses the text as a date.
//!
//! Functions live in a [`FunctionRegistry`]; downstream crates (e.g. the
//! distribution functions in `openstats-math`) register more.

pub mod array;
pub mod ast;
pub mod eval;
pub mod functions;
pub mod lexer;
pub mod parser;
pub mod recalc;

pub use array::Arr;
pub use ast::{BinOp, Expr, UnOp};
pub use eval::EvalCtx;
pub use functions::{FunctionDef, FunctionRegistry};

use crate::error::Result;
use crate::worksheet::Worksheet;

/// Parse and evaluate `formula` against `ws` with the standard functions.
pub fn evaluate(formula: &str, ws: &Worksheet) -> Result<Arr> {
    let reg = FunctionRegistry::standard();
    let ctx = EvalCtx::new(ws, &reg);
    ctx.eval_str(formula)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{Column, ColumnData};
    use crate::value::Value;

    fn ws() -> Worksheet {
        let mut ws = Worksheet::with_columns(
            "s",
            vec![
                Column::numeric("x", vec![1.0, 2.0, 3.0, f64::NAN]),
                Column::numeric("y", vec![10.0, 20.0, 30.0, 40.0]),
                Column::text("g", vec!["a", "b", "a", ""]),
                Column::datetime(
                    "d",
                    vec![
                        crate::datetime::parse("2026-01-01").unwrap(),
                        crate::datetime::parse("2026-01-15").unwrap(),
                        crate::datetime::parse("2026-02-01").unwrap(),
                        crate::missing::DATETIME_MISSING,
                    ],
                ),
                Column::numeric("Weight (g)", vec![5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();
        ws.constants.insert(1, Value::Numeric(100.0));
        ws
    }

    fn num(f: &str) -> Vec<f64> {
        evaluate(f, &ws()).unwrap().data.to_f64_vec()
    }

    fn eqv(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| crate::column::f64_eq(*x, *y) || (x - y).abs() < 1e-12)
    }

    #[test]
    fn arithmetic_precedence_and_missing() {
        assert!(eqv(&num("x + y * 2"), &[21.0, 42.0, 63.0, f64::NAN]));
        assert!(eqv(&num("-2 ** 2"), &[-4.0]));
        assert!(eqv(&num("2 ** 3 ** 2"), &[512.0]));
        assert!(eqv(&num("2 ^ -1"), &[0.5]));
        assert!(eqv(&num("(x + 1) / 0"), &[f64::NAN; 4]));
        assert!(eqv(&num("C2 - K1"), &[-90.0, -80.0, -70.0, -60.0]));
        assert!(eqv(&num("'Weight (g)' * 2"), &[10.0, 12.0, 14.0, 16.0]));
        assert!(eqv(&num("IF(x > 1, x, *)"), &[f64::NAN, 2.0, 3.0, f64::NAN]));
        assert!(eqv(&num("x * *"), &[f64::NAN; 4]));
        assert!(eqv(&num("1e3 + .5"), &[1000.5]));
    }

    #[test]
    fn comparisons_and_logic() {
        assert!(eqv(&num("x >= 2"), &[0.0, 1.0, 1.0, f64::NAN]));
        assert!(eqv(&num("g = \"a\""), &[1.0, 0.0, 1.0, f64::NAN]));
        assert!(eqv(&num("x > 1 AND y < 30"), &[0.0, 1.0, 0.0, 0.0]));
        assert!(eqv(&num("y < 15 | x > 99"), &[1.0, 0.0, 0.0, f64::NAN]));
        assert!(eqv(&num("x = 99 & y > 0"), &[0.0, 0.0, 0.0, f64::NAN]));
        assert!(eqv(&num("x > 99 AND *"), &[0.0, 0.0, 0.0, f64::NAN]));
        assert!(eqv(&num("1 OR *"), &[1.0]));
        assert!(eqv(&num("NOT (x <> 2)"), &[0.0, 1.0, 0.0, f64::NAN]));
        assert!(eqv(&num("x LT 2 OR x GE 3"), &[1.0, 0.0, 1.0, f64::NAN]));
        assert!(eqv(&num("d > \"1/10/2026\""), &[0.0, 1.0, 1.0, f64::NAN]));
    }

    #[test]
    fn dates() {
        let r = evaluate("d + 1.5", &ws()).unwrap();
        assert_eq!(crate::datetime::format(r.data.as_datetime().unwrap()[0], "yyyy-MM-dd HH"), "2026-01-02 12");
        assert!(eqv(&num("d - d"), &[0.0, 0.0, 0.0, f64::NAN]));
        assert!(eqv(&num("C4 - DATE(2026, 1, 1)"), &[0.0, 14.0, 31.0, f64::NAN]));
    }

    #[test]
    fn errors() {
        let w = ws();
        for bad in ["x +", "nope + 1", "FOO(1)", "g + 1", "SQRT()", "(1", "C99", "x = = 1", "\"open", "MEAN(x, y, x)"] {
            assert!(evaluate(bad, &w).is_err(), "{bad} should fail");
        }
        let short =
            Worksheet::with_columns("s", vec![Column::numeric("a", vec![1.0]), Column::numeric("b", vec![1.0, 2.0])])
                .unwrap();
        assert!(evaluate("a + b", &short).is_err());
        assert_eq!(evaluate("MEAN(a) + b", &short).unwrap().data, ColumnData::Numeric(vec![2.0, 3.0]));
    }
}
