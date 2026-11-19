//! Vectorised expression evaluator.

use super::array::{at, at_i, at_s, common_len, dt_missing, fin, Arr};
use super::ast::{BinOp, Expr, UnOp};
use super::functions::FunctionRegistry;
use super::parser::parse;
use crate::column::{ColumnData, ColumnKind};
use crate::datetime::{self, MICROS_PER_DAY};
use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;
use crate::worksheet::Worksheet;
use std::cmp::Ordering;

/// Evaluation context: worksheet, functions and the current time (for
/// `NOW()`/`TODAY()`; `None` makes those functions fail, which is the default
/// on `wasm32` where the host must supply the clock).
#[derive(Debug, Clone, Copy)]
pub struct EvalCtx<'a> {
    pub ws: &'a Worksheet,
    pub registry: &'a FunctionRegistry,
    /// Current local time, µs since epoch.
    pub now: Option<i64>,
}

/// System clock in µs since epoch (UTC), when the platform has one.
pub fn system_now() -> Option<i64> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok()?;
        Some(d.as_micros() as i64)
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

impl<'a> EvalCtx<'a> {
    pub fn new(ws: &'a Worksheet, registry: &'a FunctionRegistry) -> Self {
        EvalCtx { ws, registry, now: system_now() }
    }

    pub fn with_now(mut self, now: Option<i64>) -> Self {
        self.now = now;
        self
    }

    /// Parse and evaluate.
    pub fn eval_str(&self, src: &str) -> Result<Arr> {
        self.eval(&parse(src)?)
    }

    pub fn eval(&self, e: &Expr) -> Result<Arr> {
        match e {
            Expr::Number(x) => Ok(Arr::num(*x)),
            Expr::Text(s) => Ok(Arr::text(s.clone())),
            Expr::Missing => Ok(Arr::missing()),
            Expr::ColumnNumber { n, .. } => {
                let c = self.ws.columns.get(n - 1).ok_or_else(|| CoreError::NotFound(format!("column C{n}")))?;
                Ok(Arr::vector(c.data.clone()))
            }
            Expr::ColumnName { name, .. } => {
                let i = self.ws.find(name).ok_or_else(|| CoreError::NotFound(format!("column {name:?}")))?;
                Ok(Arr::vector(self.ws.columns[i].data.clone()))
            }
            Expr::Constant { k, .. } => {
                let v = self.ws.constant(*k).ok_or_else(|| CoreError::NotFound(format!("constant K{k}")))?;
                Ok(Arr::from_value(v))
            }
            Expr::Unary(op, a) => {
                let a = self.eval(a)?;
                unary(*op, &a)
            }
            Expr::Binary(op, a, b) => {
                let a = self.eval(a)?;
                let b = self.eval(b)?;
                binary(*op, &a, &b)
            }
            Expr::Call { name, args, .. } => {
                let def = self.registry.get(name).ok_or_else(|| CoreError::NotFound(format!("function {name}")))?;
                if args.len() < def.min_args || def.max_args.is_some_and(|m| args.len() > m) {
                    let expect = match def.max_args {
                        Some(m) if m == def.min_args => format!("{m}"),
                        Some(m) => format!("{}–{m}", def.min_args),
                        None => format!("at least {}", def.min_args),
                    };
                    return Err(CoreError::invalid(format!("{name} takes {expect} argument(s), {} given", args.len())));
                }
                let vals = args.iter().map(|a| self.eval(a)).collect::<Result<Vec<_>>>()?;
                (def.f)(&vals, self).map_err(|e| match e {
                    CoreError::InvalidArgument(m) if !m.starts_with(name.as_str()) => {
                        CoreError::InvalidArgument(format!("{name}: {m}"))
                    }
                    other => other,
                })
            }
        }
    }
}

/// Truth value of a number: missing → None.
#[inline]
pub fn truth(x: f64) -> Option<bool> {
    if x.is_nan() {
        None
    } else {
        Some(x != 0.0)
    }
}

#[inline]
fn b2f(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

fn unary(op: UnOp, a: &Arr) -> Result<Arr> {
    let v = a.to_nums("unary operator")?;
    if a.kind() == ColumnKind::DateTime && op != UnOp::Plus {
        return Err(CoreError::invalid("a date/time cannot be negated or used as a logical value"));
    }
    let out: Vec<f64> = match op {
        UnOp::Neg => v.iter().map(|x| -x).collect(),
        UnOp::Plus => {
            if a.kind() == ColumnKind::DateTime {
                return Ok(a.clone());
            }
            v
        }
        UnOp::Not => v.iter().map(|&x| truth(x).map_or(f64::NAN, |t| b2f(!t))).collect(),
    };
    Ok(Arr::nums(out, a.scalar))
}

/// Apply a binary operator element-wise.
pub fn binary(op: BinOp, a: &Arr, b: &Arr) -> Result<Arr> {
    let (n, scalar) = common_len(&[a, b], &format!("operator {op}"))?;
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow => arith(op, a, b, n, scalar),
        BinOp::And | BinOp::Or => {
            let x = a.to_nums("logical operator")?;
            let y = b.to_nums("logical operator")?;
            let out = (0..n)
                .map(|i| {
                    let (p, q) = (truth(at(&x, i)), truth(at(&y, i)));
                    match op {
                        BinOp::And => match (p, q) {
                            (Some(false), _) | (_, Some(false)) => 0.0,
                            (Some(true), Some(true)) => 1.0,
                            _ => f64::NAN,
                        },
                        _ => match (p, q) {
                            (Some(true), _) | (_, Some(true)) => 1.0,
                            (Some(false), Some(false)) => 0.0,
                            _ => f64::NAN,
                        },
                    }
                })
                .collect();
            Ok(Arr::nums(out, scalar))
        }
        _ => compare(op, a, b, n, scalar),
    }
}

fn arith(op: BinOp, a: &Arr, b: &Arr, n: usize, scalar: bool) -> Result<Arr> {
    use ColumnKind::*;
    let is_text = |x: &Arr| x.kind() == Text && !x.all_missing();
    if is_text(a) || is_text(b) {
        return Err(CoreError::invalid(format!(
            "text cannot be used with operator {op} (use CONCATENATE to join text)"
        )));
    }
    let (ka, kb) = (if a.all_missing() && a.kind() != DateTime { Numeric } else { a.kind() }, b.kind());
    // Date arithmetic.
    match (ka, kb, op) {
        (DateTime, Numeric, BinOp::Add | BinOp::Sub) | (Numeric, DateTime, BinOp::Add) => {
            let (d, x, sign) = if ka == DateTime {
                (a.to_datetimes("date")?, b.to_nums("days")?, if op == BinOp::Sub { -1.0 } else { 1.0 })
            } else {
                (b.to_datetimes("date")?, a.to_nums("days")?, 1.0)
            };
            let out = (0..n)
                .map(|i| {
                    let (t, days) = (at_i(&d, i), at(&x, i));
                    if dt_missing(t) || !days.is_finite() {
                        DATETIME_MISSING
                    } else {
                        t.saturating_add((sign * days * MICROS_PER_DAY as f64).round() as i64)
                    }
                })
                .collect();
            return Ok(Arr { data: ColumnData::DateTime(out), scalar });
        }
        (DateTime, DateTime, BinOp::Sub) => {
            let (x, y) = (a.to_datetimes("date")?, b.to_datetimes("date")?);
            let out = (0..n).map(|i| datetime::diff_days(at_i(&x, i), at_i(&y, i))).collect();
            return Ok(Arr::nums(out, scalar));
        }
        _ => {}
    }
    let x = a.to_nums("arithmetic")?;
    let y = b.to_nums("arithmetic")?;
    let f: fn(f64, f64) -> f64 = match op {
        BinOp::Add => |p, q| p + q,
        BinOp::Sub => |p, q| p - q,
        BinOp::Mul => |p, q| p * q,
        BinOp::Div => |p, q| p / q,
        _ => pow,
    };
    let out = if x.len() == n && y.len() == n {
        x.iter().zip(&y).map(|(&p, &q)| fin(f(p, q))).collect()
    } else {
        (0..n).map(|i| fin(f(at(&x, i), at(&y, i)))).collect()
    };
    Ok(Arr::nums(out, scalar))
}

/// Minitab-style power: 0**0 = 1, negative base needs an integer exponent.
pub fn pow(b: f64, e: f64) -> f64 {
    if b.is_nan() || e.is_nan() {
        return f64::NAN;
    }
    if b < 0.0 && e.fract() != 0.0 {
        return f64::NAN;
    }
    if e.fract() == 0.0 && e.abs() <= i32::MAX as f64 {
        b.powi(e as i32)
    } else {
        b.powf(e)
    }
}

fn compare(op: BinOp, a: &Arr, b: &Arr, n: usize, scalar: bool) -> Result<Arr> {
    use ColumnKind::*;
    let decide = |o: Option<Ordering>| -> f64 {
        match o {
            None => f64::NAN,
            Some(o) => b2f(match op {
                BinOp::Eq => o == Ordering::Equal,
                BinOp::Ne => o != Ordering::Equal,
                BinOp::Lt => o == Ordering::Less,
                BinOp::Le => o != Ordering::Greater,
                BinOp::Gt => o == Ordering::Greater,
                _ => o != Ordering::Less,
            }),
        }
    };
    let out: Vec<f64> = match (a.kind(), b.kind()) {
        (Text, Text) => {
            let (x, y) = (a.to_texts(), b.to_texts());
            (0..n)
                .map(|i| {
                    let (p, q) = (at_s(&x, i), at_s(&y, i));
                    decide(if p.is_empty() || q.is_empty() { None } else { Some(p.cmp(q)) })
                })
                .collect()
        }
        (DateTime, DateTime) | (DateTime, Text) | (Text, DateTime) => {
            let (x, y) = (a.to_datetimes("comparison")?, b.to_datetimes("comparison")?);
            (0..n)
                .map(|i| {
                    let (p, q) = (at_i(&x, i), at_i(&y, i));
                    decide(if dt_missing(p) || dt_missing(q) { None } else { Some(p.cmp(&q)) })
                })
                .collect()
        }
        (Text, _) | (_, Text) => {
            if a.all_missing() || b.all_missing() {
                vec![f64::NAN; n]
            } else {
                return Err(CoreError::invalid("cannot compare text with numbers"));
            }
        }
        _ => {
            let (x, y) = (a.to_nums("comparison")?, b.to_nums("comparison")?);
            (0..n).map(|i| decide(at(&x, i).partial_cmp(&at(&y, i)))).collect()
        }
    };
    Ok(Arr::nums(out, scalar))
}
