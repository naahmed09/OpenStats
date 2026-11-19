//! Evaluation values: typed arrays that may be scalars (broadcast).

use crate::column::{ColumnData, ColumnKind};
use crate::datetime;
use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;
use crate::value::Value;

/// A typed vector; `scalar` marks single values that broadcast.
#[derive(Debug, Clone, PartialEq)]
pub struct Arr {
    pub data: ColumnData,
    pub scalar: bool,
}

impl Arr {
    pub fn num(x: f64) -> Self {
        Arr { data: ColumnData::Numeric(vec![x]), scalar: true }
    }
    pub fn text(s: impl Into<String>) -> Self {
        Arr { data: ColumnData::Text(vec![s.into()]), scalar: true }
    }
    pub fn datetime(t: i64) -> Self {
        Arr { data: ColumnData::DateTime(vec![t]), scalar: true }
    }
    pub fn missing() -> Self {
        Self::num(f64::NAN)
    }
    pub fn vector(data: ColumnData) -> Self {
        Arr { data, scalar: false }
    }
    pub fn nums(v: Vec<f64>, scalar: bool) -> Self {
        Arr { data: ColumnData::Numeric(v), scalar }
    }
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Numeric(x) => Self::num(*x),
            Value::Text(s) => Self::text(s.clone()),
            Value::DateTime(t) => Self::datetime(*t),
            Value::Missing => Self::missing(),
        }
    }

    pub fn kind(&self) -> ColumnKind {
        self.data.kind()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `true` if every element is missing (numeric missing can be coerced to
    /// any type — this is how the untyped `*` literal behaves).
    pub fn all_missing(&self) -> bool {
        self.data.missing_count() == self.data.len()
    }

    /// The scalar value (first element).
    pub fn first(&self) -> Value {
        self.data.get(0)
    }

    /// Numeric view: numbers as-is, date/times as spreadsheet serials, text is
    /// an error (Minitab does not coerce text in arithmetic).
    pub fn to_nums(&self, what: &str) -> Result<Vec<f64>> {
        match &self.data {
            ColumnData::Numeric(v) => Ok(v.clone()),
            ColumnData::DateTime(v) => Ok(v.iter().map(|&t| datetime::to_excel_serial(t)).collect()),
            ColumnData::Text(v) => {
                if v.iter().all(|s| s.is_empty()) {
                    Ok(vec![f64::NAN; v.len()])
                } else {
                    Err(CoreError::TypeMismatch { expected: "numeric", found: "text" })
                        .map_err(|e| CoreError::invalid(format!("{what}: {e}")))
                }
            }
        }
    }

    /// Text view (numbers/dates formatted, missing → empty).
    pub fn to_texts(&self) -> Vec<String> {
        match &self.data {
            ColumnData::Text(v) => v.clone(),
            other => (0..other.len())
                .map(|i| {
                    let v = other.get(i);
                    if v.is_missing() {
                        String::new()
                    } else {
                        v.to_text()
                    }
                })
                .collect(),
        }
    }

    /// Date/time view: dates as-is, numbers as spreadsheet serials, text parsed.
    pub fn to_datetimes(&self, what: &str) -> Result<Vec<i64>> {
        match &self.data {
            ColumnData::DateTime(v) => Ok(v.clone()),
            ColumnData::Numeric(v) => Ok(v.iter().map(|&x| datetime::from_excel_serial(x)).collect()),
            ColumnData::Text(v) => {
                v.iter().map(|s| datetime::parse(s).map_err(|e| CoreError::invalid(format!("{what}: {e}")))).collect()
            }
        }
    }

    /// Convert to `kind`, allowing all-missing arrays to change type freely.
    pub fn coerce_to(&self, kind: ColumnKind, what: &str) -> Result<Arr> {
        if self.kind() == kind {
            return Ok(self.clone());
        }
        if self.all_missing() {
            return Ok(Arr { data: ColumnData::missing(kind, self.len()), scalar: self.scalar });
        }
        let data = match kind {
            ColumnKind::Numeric => ColumnData::Numeric(self.to_nums(what)?),
            ColumnKind::Text => ColumnData::Text(self.to_texts()),
            ColumnKind::DateTime => ColumnData::DateTime(self.to_datetimes(what)?),
        };
        Ok(Arr { data, scalar: self.scalar })
    }

    /// Repeat a scalar to length `n` (vectors are returned unchanged).
    pub fn broadcast(&self, n: usize) -> Arr {
        if !self.scalar {
            return self.clone();
        }
        let idx = vec![0usize; n];
        Arr { data: self.data.take(&idx), scalar: n == 1 && self.scalar }
    }
}

/// Common length of `args` for element-wise operations: all vectors must have
/// the same length; returns (len, all_scalar).
pub fn common_len(args: &[&Arr], what: &str) -> Result<(usize, bool)> {
    let mut n: Option<usize> = None;
    for a in args {
        if a.scalar {
            continue;
        }
        match n {
            None => n = Some(a.len()),
            Some(m) if m != a.len() => {
                return Err(CoreError::invalid(format!(
                    "{what}: arguments have different lengths ({m} and {})",
                    a.len()
                )))
            }
            _ => {}
        }
    }
    Ok(match n {
        Some(m) => (m, false),
        None => (1, true),
    })
}

/// Normalise a numeric result: non-finite values become missing.
#[inline]
pub fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NAN
    }
}

/// Element `i` of a numeric slice with scalar broadcast.
#[inline]
pub fn at(v: &[f64], i: usize) -> f64 {
    if v.len() == 1 {
        v[0]
    } else {
        v[i]
    }
}

/// Element `i` of a date slice with scalar broadcast.
#[inline]
pub fn at_i(v: &[i64], i: usize) -> i64 {
    if v.len() == 1 {
        v[0]
    } else {
        v[i]
    }
}

/// Element `i` of a string slice with scalar broadcast.
#[inline]
pub fn at_s(v: &[String], i: usize) -> &str {
    if v.len() == 1 {
        &v[0]
    } else {
        &v[i]
    }
}

/// Missing-aware date check.
#[inline]
pub fn dt_missing(t: i64) -> bool {
    t == DATETIME_MISSING
}
