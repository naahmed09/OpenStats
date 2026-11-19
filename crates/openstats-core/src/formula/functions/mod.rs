//! Calculator function registry.
//!
//! Every function receives fully evaluated arguments ([`Arr`]) and the
//! evaluation context. Categories:
//!
//! * [`math`] — element-wise arithmetic and transcendental functions
//! * [`stats`] — column statistics (scalar results) and column transforms
//! * [`row`] — row statistics across several columns
//! * [`logic`] — `IF`, `AND`, `OR`, missing-value tests
//! * [`text`] — string functions
//! * [`date`] — date/time construction and extraction

pub mod date;
pub mod logic;
pub mod math;
pub mod row;
pub mod stats;
pub mod text;

use super::array::{at, common_len, fin, Arr};
use super::eval::EvalCtx;
use crate::column::ColumnKind;
use crate::error::{CoreError, Result};
use std::collections::BTreeMap;

/// Function implementation signature.
pub type FnImpl = fn(&[Arr], &EvalCtx) -> Result<Arr>;

/// Function category (for the calculator dialog's function list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    Arithmetic,
    Trigonometry,
    Statistics,
    RowStatistics,
    Logical,
    Text,
    DateTime,
    Probability,
}

/// A registered function.
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: &'static str,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub category: Category,
    /// Signature and one-line description shown in the calculator dialog.
    pub help: &'static str,
    pub f: FnImpl,
}

/// Name → function table (names are upper-case).
#[derive(Debug, Clone, Default)]
pub struct FunctionRegistry {
    map: BTreeMap<String, FunctionDef>,
}

impl FunctionRegistry {
    /// An empty registry.
    pub fn empty() -> Self {
        Self::default()
    }

    /// All built-in functions.
    pub fn standard() -> Self {
        let mut r = Self::empty();
        math::register(&mut r);
        stats::register(&mut r);
        row::register(&mut r);
        logic::register(&mut r);
        text::register(&mut r);
        date::register(&mut r);
        r
    }

    /// Add or replace a function.
    pub fn register(&mut self, def: FunctionDef) {
        self.map.insert(def.name.to_ascii_uppercase(), def);
    }

    /// Register `alias` as another name for an existing function.
    pub fn alias(&mut self, alias: &'static str, target: &str) {
        let mut d = self.map.get(target).cloned().unwrap_or_else(|| panic!("alias target {target} not registered"));
        d.name = alias;
        self.map.insert(alias.to_string(), d);
    }

    pub fn get(&self, name: &str) -> Option<&FunctionDef> {
        self.map.get(&name.to_ascii_uppercase())
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Functions sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = &FunctionDef> {
        self.map.values()
    }
}

/// Shorthand used by the category modules.
pub(crate) fn def(
    name: &'static str,
    min_args: usize,
    max_args: Option<usize>,
    category: Category,
    help: &'static str,
    f: FnImpl,
) -> FunctionDef {
    FunctionDef { name, min_args, max_args, category, help, f }
}

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

/// Element-wise unary numeric map.
pub(crate) fn map1(a: &Arr, f: impl Fn(f64) -> f64) -> Result<Arr> {
    let v = a.to_nums("argument")?;
    Ok(Arr::nums(v.into_iter().map(|x| fin(f(x))).collect(), a.scalar))
}

/// Element-wise binary numeric map with broadcasting.
pub(crate) fn map2(a: &Arr, b: &Arr, f: impl Fn(f64, f64) -> f64) -> Result<Arr> {
    let (n, scalar) = common_len(&[a, b], "arguments")?;
    let (x, y) = (a.to_nums("argument")?, b.to_nums("argument")?);
    Ok(Arr::nums((0..n).map(|i| fin(f(at(&x, i), at(&y, i)))).collect(), scalar))
}

/// Optional second argument (default when absent).
pub(crate) fn map1_opt(args: &[Arr], default: f64, f: impl Fn(f64, f64) -> f64) -> Result<Arr> {
    match args.get(1) {
        Some(b) => map2(&args[0], b, f),
        None => map1(&args[0], |x| f(x, default)),
    }
}

/// A scalar argument as f64.
pub(crate) fn scalar_num(a: &Arr, what: &str) -> Result<f64> {
    if !a.scalar && a.len() != 1 {
        return Err(CoreError::invalid(format!("{what} must be a single value, not a column")));
    }
    Ok(a.to_nums(what)?.first().copied().unwrap_or(f64::NAN))
}

/// A scalar integer argument.
pub(crate) fn scalar_int(a: &Arr, what: &str) -> Result<i64> {
    let x = scalar_num(a, what)?;
    if !x.is_finite() || x.fract() != 0.0 {
        return Err(CoreError::invalid(format!("{what} must be an integer")));
    }
    Ok(x as i64)
}

/// Non-missing numeric values of a column argument.
pub(crate) fn present(a: &Arr) -> Result<Vec<f64>> {
    Ok(a.to_nums("column")?.into_iter().filter(|x| !x.is_nan()).collect())
}

/// Require a text argument.
pub(crate) fn texts(a: &Arr, what: &str) -> Result<Vec<String>> {
    match a.kind() {
        ColumnKind::Text => Ok(a.to_texts()),
        _ if a.all_missing() => Ok(vec![String::new(); a.len()]),
        _ => Err(CoreError::invalid(format!("{what} must be text"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_size_and_lookup() {
        let r = FunctionRegistry::standard();
        assert!(r.len() >= 150, "only {} functions registered", r.len());
        assert!(r.get("mean").is_some());
        for f in r.iter() {
            assert!(!f.help.is_empty(), "{} lacks help", f.name);
            assert!(f.max_args.is_none_or(|m| m >= f.min_args), "{} arity", f.name);
        }
    }
}
