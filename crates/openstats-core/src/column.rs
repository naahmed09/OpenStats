//! Typed worksheet columns.

use crate::datetime;
use crate::error::{CoreError, Result};
use crate::missing::{DATETIME_MISSING, MISSING_TOKEN};
use crate::value::{format_number, parse_number, Value};
use std::collections::{BTreeSet, HashSet};

/// Storage type of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnKind {
    Numeric,
    Text,
    DateTime,
}

impl ColumnKind {
    pub fn name(self) -> &'static str {
        match self {
            ColumnKind::Numeric => "numeric",
            ColumnKind::Text => "text",
            ColumnKind::DateTime => "date/time",
        }
    }
    /// One-letter tag shown in the grid header (`T` text, `D` date/time).
    pub fn tag(self) -> &'static str {
        match self {
            ColumnKind::Numeric => "",
            ColumnKind::Text => "T",
            ColumnKind::DateTime => "D",
        }
    }
}

/// Display format for numeric columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericFormat {
    /// Shortest round-trip representation.
    #[default]
    Automatic,
    /// Fixed number of decimal places.
    Fixed(u8),
    /// Scientific notation with the given mantissa decimals.
    Scientific(u8),
    /// Percentage with the given decimals (value 0.25 → `25.0%`).
    Percent(u8),
}

impl NumericFormat {
    pub fn format(self, x: f64) -> String {
        if x.is_nan() {
            return MISSING_TOKEN.to_string();
        }
        match self {
            NumericFormat::Automatic => format_number(x),
            NumericFormat::Fixed(d) => format!("{:.*}", d as usize, x),
            NumericFormat::Scientific(d) => {
                let s = format!("{:.*e}", d as usize, x);
                let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
                let e: i32 = exp.parse().unwrap_or(0);
                format!("{mant}E{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
            }
            NumericFormat::Percent(d) => format!("{:.*}%", d as usize, x * 100.0),
        }
    }
}

/// The raw column vector.
///
/// Equality treats all numeric missing values (NaN) as equal to each other,
/// so that missing-value semantics survive comparisons (e.g. undo checks).
#[derive(Debug, Clone)]
pub enum ColumnData {
    Numeric(Vec<f64>),
    Text(Vec<String>),
    DateTime(Vec<i64>),
}

/// NaN-aware float equality: equal values, or both missing.
#[inline]
pub fn f64_eq(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

impl PartialEq for ColumnData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ColumnData::Numeric(a), ColumnData::Numeric(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| f64_eq(x, y))
            }
            (ColumnData::Text(a), ColumnData::Text(b)) => a == b,
            (ColumnData::DateTime(a), ColumnData::DateTime(b)) => a == b,
            _ => false,
        }
    }
}

impl ColumnData {
    /// An empty vector of the given kind.
    pub fn empty(kind: ColumnKind) -> Self {
        Self::missing(kind, 0)
    }

    /// `n` missing values of the given kind.
    pub fn missing(kind: ColumnKind, n: usize) -> Self {
        match kind {
            ColumnKind::Numeric => ColumnData::Numeric(vec![f64::NAN; n]),
            ColumnKind::Text => ColumnData::Text(vec![String::new(); n]),
            ColumnKind::DateTime => ColumnData::DateTime(vec![DATETIME_MISSING; n]),
        }
    }

    pub fn kind(&self) -> ColumnKind {
        match self {
            ColumnData::Numeric(_) => ColumnKind::Numeric,
            ColumnData::Text(_) => ColumnKind::Text,
            ColumnData::DateTime(_) => ColumnKind::DateTime,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnData::Numeric(v) => v.len(),
            ColumnData::Text(v) => v.len(),
            ColumnData::DateTime(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value at row `i`; rows past the end read as missing.
    pub fn get(&self, i: usize) -> Value {
        match self {
            ColumnData::Numeric(v) => v.get(i).map_or(Value::Missing, |&x| Value::Numeric(x)),
            ColumnData::Text(v) => v.get(i).map_or(Value::Missing, |s| Value::Text(s.clone())),
            ColumnData::DateTime(v) => v.get(i).map_or(Value::Missing, |&t| Value::DateTime(t)),
        }
    }

    /// `true` if row `i` is missing (rows past the end are missing).
    pub fn is_missing(&self, i: usize) -> bool {
        match self {
            ColumnData::Numeric(v) => v.get(i).is_none_or(|x| x.is_nan()),
            ColumnData::Text(v) => v.get(i).is_none_or(|s| s.is_empty()),
            ColumnData::DateTime(v) => v.get(i).is_none_or(|&t| t == DATETIME_MISSING),
        }
    }

    pub fn missing_count(&self) -> usize {
        (0..self.len()).filter(|&i| self.is_missing(i)).count()
    }

    /// Convert a value to this column's native representation.
    fn coerce(&self, v: &Value) -> Result<Value> {
        let mismatch = |found: &'static str| CoreError::TypeMismatch { expected: self.kind().name(), found };
        Ok(match (self, v) {
            (_, Value::Missing) => match self.kind() {
                ColumnKind::Numeric => Value::Numeric(f64::NAN),
                ColumnKind::Text => Value::Text(String::new()),
                ColumnKind::DateTime => Value::DateTime(DATETIME_MISSING),
            },
            (ColumnData::Numeric(_), Value::Numeric(x)) => Value::Numeric(*x),
            (ColumnData::Numeric(_), Value::Text(s)) => Value::Numeric(parse_number(s).map_err(|_| mismatch("text"))?),
            (ColumnData::Numeric(_), Value::DateTime(_)) => return Err(mismatch("date/time")),
            (ColumnData::Text(_), Value::Numeric(x)) => {
                Value::Text(if x.is_nan() { String::new() } else { format_number(*x) })
            }
            (ColumnData::Text(_), Value::Text(s)) => Value::Text(s.clone()),
            (ColumnData::Text(_), Value::DateTime(t)) => {
                Value::Text(if *t == DATETIME_MISSING { String::new() } else { datetime::format_default(*t) })
            }
            (ColumnData::DateTime(_), Value::DateTime(t)) => Value::DateTime(*t),
            (ColumnData::DateTime(_), Value::Text(s)) => {
                Value::DateTime(datetime::parse(s).map_err(|_| mismatch("text"))?)
            }
            (ColumnData::DateTime(_), Value::Numeric(x)) if x.is_nan() => Value::DateTime(DATETIME_MISSING),
            (ColumnData::DateTime(_), Value::Numeric(_)) => return Err(mismatch("numeric")),
        })
    }

    /// Set row `i`, extending the column with missing values if needed.
    /// Returns the previous value (Missing if the row did not exist).
    pub fn set(&mut self, i: usize, v: &Value) -> Result<Value> {
        let v = self.coerce(v)?;
        if i >= self.len() {
            self.resize(i + 1);
        }
        let old = self.get(i);
        match (self, v) {
            (ColumnData::Numeric(c), Value::Numeric(x)) => c[i] = x,
            (ColumnData::Text(c), Value::Text(s)) => c[i] = s,
            (ColumnData::DateTime(c), Value::DateTime(t)) => c[i] = t,
            _ => unreachable!("coerce returns the column's own type"),
        }
        Ok(old)
    }

    /// Append a value.
    pub fn push(&mut self, v: &Value) -> Result<()> {
        let n = self.len();
        self.set(n, v).map(|_| ())
    }

    /// Grow (with missing) or truncate to `n` rows.
    pub fn resize(&mut self, n: usize) {
        match self {
            ColumnData::Numeric(v) => v.resize(n, f64::NAN),
            ColumnData::Text(v) => v.resize(n, String::new()),
            ColumnData::DateTime(v) => v.resize(n, DATETIME_MISSING),
        }
    }

    /// Remove trailing missing values (Minitab's notion of column length).
    pub fn trim_trailing_missing(&mut self) {
        let mut n = self.len();
        while n > 0 && self.is_missing(n - 1) {
            n -= 1;
        }
        self.resize(n);
    }

    /// Insert `n` missing rows at `at` (padding first if `at` is past the end).
    pub fn insert_missing(&mut self, at: usize, n: usize) {
        let seg = ColumnData::missing(self.kind(), n);
        self.insert_data(at, seg).expect("same kind");
    }

    /// Insert a segment of the same kind at `at`.
    pub fn insert_data(&mut self, at: usize, seg: ColumnData) -> Result<()> {
        if seg.kind() != self.kind() {
            return Err(CoreError::TypeMismatch { expected: self.kind().name(), found: seg.kind().name() });
        }
        if at > self.len() {
            self.resize(at);
        }
        match (self, seg) {
            (ColumnData::Numeric(v), ColumnData::Numeric(s)) => {
                v.splice(at..at, s);
            }
            (ColumnData::Text(v), ColumnData::Text(s)) => {
                v.splice(at..at, s);
            }
            (ColumnData::DateTime(v), ColumnData::DateTime(s)) => {
                v.splice(at..at, s);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Remove rows `at..at+n` (clamped to the column) and return them.
    pub fn remove_range(&mut self, at: usize, n: usize) -> ColumnData {
        let len = self.len();
        let s = at.min(len);
        let e = at.saturating_add(n).min(len);
        match self {
            ColumnData::Numeric(v) => ColumnData::Numeric(v.drain(s..e).collect()),
            ColumnData::Text(v) => ColumnData::Text(v.drain(s..e).collect()),
            ColumnData::DateTime(v) => ColumnData::DateTime(v.drain(s..e).collect()),
        }
    }

    /// New vector with rows gathered by index; out-of-range indices give missing.
    pub fn take(&self, idx: &[usize]) -> ColumnData {
        match self {
            ColumnData::Numeric(v) => {
                ColumnData::Numeric(idx.iter().map(|&i| v.get(i).copied().unwrap_or(f64::NAN)).collect())
            }
            ColumnData::Text(v) => {
                ColumnData::Text(idx.iter().map(|&i| v.get(i).cloned().unwrap_or_default()).collect())
            }
            ColumnData::DateTime(v) => {
                ColumnData::DateTime(idx.iter().map(|&i| v.get(i).copied().unwrap_or(DATETIME_MISSING)).collect())
            }
        }
    }

    /// Append another vector of the same kind.
    pub fn extend(&mut self, other: &ColumnData) -> Result<()> {
        let n = self.len();
        self.insert_data(n, other.clone())
    }

    pub fn as_numeric(&self) -> Option<&[f64]> {
        if let ColumnData::Numeric(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_text(&self) -> Option<&[String]> {
        if let ColumnData::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_datetime(&self) -> Option<&[i64]> {
        if let ColumnData::DateTime(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Numeric view of any column: numeric as-is, date/time as Excel serial,
    /// text parsed (unparsable → missing).
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            ColumnData::Numeric(v) => v.clone(),
            _ => (0..self.len()).map(|i| self.get(i).as_f64()).collect(),
        }
    }

    /// Display strings for every row (missing numeric/date shown as `*`,
    /// missing text as empty).
    pub fn to_text_vec(&self) -> Vec<String> {
        (0..self.len())
            .map(|i| match self.get(i) {
                Value::Text(s) => s,
                v => v.to_string(),
            })
            .collect()
    }

    /// Convert to another kind. With `strict`, a non-missing value that cannot
    /// be converted is an error; otherwise it becomes missing.
    pub fn convert(&self, kind: ColumnKind, strict: bool) -> Result<ColumnData> {
        if kind == self.kind() {
            return Ok(self.clone());
        }
        let mut out = ColumnData::missing(kind, self.len());
        for i in 0..self.len() {
            let v = self.get(i);
            if v.is_missing() {
                continue;
            }
            let converted = match (kind, &v) {
                // Date/time → numeric uses the spreadsheet serial (Minitab-compatible).
                (ColumnKind::Numeric, Value::DateTime(t)) => Ok(Value::Numeric(datetime::to_excel_serial(*t))),
                (ColumnKind::DateTime, Value::Numeric(x)) => Ok(Value::DateTime(datetime::from_excel_serial(*x))),
                _ => out.coerce(&v),
            };
            match converted {
                Ok(c) => {
                    out.set(i, &c)?;
                }
                Err(e) if strict => {
                    return Err(CoreError::invalid(format!("row {}: {e}", i + 1)));
                }
                Err(_) => {}
            }
        }
        Ok(out)
    }
}

/// Column metadata.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColumnProps {
    pub name: String,
    pub description: String,
    pub numeric_format: NumericFormat,
    /// Display pattern for date/time columns (see [`crate::datetime`]).
    pub datetime_format: Option<String>,
    /// Custom value order for categorical use (text values in order). `None`
    /// means natural order (numeric ascending, text alphabetical, dates
    /// chronological).
    pub value_order: Option<Vec<String>>,
    /// Column formula (see [`crate::formula::recalc`]); `None` for plain data.
    pub formula: Option<String>,
    /// Text shown in place of values: `(value as typed, label)`. The data keep their values; only the worksheet
    /// display changes. Numbers are matched by their full-precision text (`1`, `2.5`).
    pub value_labels: Vec<(String, String)>,
}

/// A named, typed column.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub props: ColumnProps,
    pub data: ColumnData,
}

impl Column {
    pub fn new(name: impl Into<String>, data: ColumnData) -> Self {
        Column { props: ColumnProps { name: name.into(), ..Default::default() }, data }
    }
    pub fn numeric(name: impl Into<String>, v: Vec<f64>) -> Self {
        Self::new(name, ColumnData::Numeric(v))
    }
    pub fn text<S: Into<String>>(name: impl Into<String>, v: Vec<S>) -> Self {
        Self::new(name, ColumnData::Text(v.into_iter().map(Into::into).collect()))
    }
    pub fn datetime(name: impl Into<String>, v: Vec<i64>) -> Self {
        Self::new(name, ColumnData::DateTime(v))
    }

    pub fn name(&self) -> &str {
        &self.props.name
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
    pub fn get(&self, i: usize) -> Value {
        self.data.get(i)
    }

    /// Formatted cell text honoring the column's display format.
    pub fn display(&self, i: usize) -> String {
        self.display_with(i, &crate::locale::Locale::default())
    }

    /// The label of the value in row `i`, when the column has a label for it.
    pub fn label_for(&self, i: usize) -> Option<&str> {
        if self.props.value_labels.is_empty() || self.data.is_missing(i) {
            return None;
        }
        let key = match &self.data {
            ColumnData::Numeric(v) => crate::value::format_number(*v.get(i)?),
            ColumnData::Text(v) => v.get(i)?.clone(),
            ColumnData::DateTime(_) => return None,
        };
        self.props.value_labels.iter().find(|(k, _)| *k == key).map(|(_, l)| l.as_str())
    }

    /// [`Column::display`] with a locale's decimal mark and default date order; value labels replace values.
    pub fn display_with(&self, i: usize, loc: &crate::locale::Locale) -> String {
        if let Some(l) = self.label_for(i) {
            return l.to_string();
        }
        match &self.data {
            ColumnData::Numeric(v) => {
                loc.localize_number(&self.props.numeric_format.format(v.get(i).copied().unwrap_or(f64::NAN)))
            }
            ColumnData::Text(v) => v.get(i).cloned().unwrap_or_default(),
            ColumnData::DateTime(v) => {
                let t = v.get(i).copied().unwrap_or(DATETIME_MISSING);
                match &self.props.datetime_format {
                    Some(p) => datetime::format(t, p),
                    None if t == DATETIME_MISSING => datetime::format_default(t),
                    None => {
                        let (d, dt) = loc.date_patterns();
                        datetime::format(t, if datetime::DateTime::from_timestamp(t).is_midnight() { d } else { dt })
                    }
                }
            }
        }
    }

    /// Distinct non-missing values in the column's value order. Custom value
    /// orders list their entries first (only those present), followed by any
    /// values not mentioned, in natural order.
    pub fn levels(&self) -> Vec<Value> {
        let natural: Vec<Value> = match &self.data {
            ColumnData::Numeric(v) => {
                let mut u: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
                u.sort_by(|a, b| a.total_cmp(b));
                u.dedup_by(|a, b| a == b);
                u.into_iter().map(Value::Numeric).collect()
            }
            ColumnData::Text(v) => {
                let set: BTreeSet<&String> = v.iter().filter(|s| !s.is_empty()).collect();
                set.into_iter().map(|s| Value::Text(s.clone())).collect()
            }
            ColumnData::DateTime(v) => {
                let set: BTreeSet<i64> = v.iter().copied().filter(|&t| t != DATETIME_MISSING).collect();
                set.into_iter().map(Value::DateTime).collect()
            }
        };
        match &self.props.value_order {
            None => natural,
            Some(order) => {
                let present: HashSet<String> = natural.iter().map(|v| v.to_text()).collect();
                let mut out: Vec<Value> = Vec::with_capacity(natural.len());
                let mut used: HashSet<String> = HashSet::new();
                for o in order {
                    if present.contains(o) && used.insert(o.clone()) {
                        out.push(natural.iter().find(|v| &v.to_text() == o).cloned().expect("present"));
                    }
                }
                out.extend(natural.into_iter().filter(|v| !used.contains(&v.to_text())));
                out
            }
        }
    }

    /// Level index (0-based, per [`Column::levels`]) for each row; `None` for missing.
    pub fn level_codes(&self) -> (Vec<Value>, Vec<Option<usize>>) {
        let levels = self.levels();
        let keys: Vec<String> = levels.iter().map(level_key).collect();
        let map: std::collections::HashMap<&str, usize> =
            keys.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
        let codes = (0..self.len())
            .map(|i| {
                let v = self.get(i);
                if v.is_missing() {
                    None
                } else {
                    map.get(level_key(&v).as_str()).copied()
                }
            })
            .collect();
        (levels, codes)
    }
}

/// Exact identity key for a value (bit pattern for numerics, so that grouping
/// never merges distinct floats that happen to print the same).
pub fn level_key(v: &Value) -> String {
    match v {
        Value::Numeric(x) => {
            let x = if *x == 0.0 { 0.0 } else { *x }; // fold -0.0 into 0.0
            format!("n{:016x}", x.to_bits())
        }
        Value::Text(s) => format!("t{s}"),
        Value::DateTime(t) => format!("d{t}"),
        Value::Missing => "m".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_coerce() {
        let mut c = ColumnData::Numeric(vec![1.0]);
        c.set(3, &Value::Text("2.5".into())).unwrap();
        assert_eq!(c.len(), 4);
        assert!(c.is_missing(1));
        assert_eq!(c.get(3), Value::Numeric(2.5));
        assert!(c.set(0, &Value::Text("abc".into())).is_err());
        let mut t = ColumnData::Text(vec![]);
        t.push(&Value::Numeric(3.0)).unwrap();
        t.push(&Value::Numeric(f64::NAN)).unwrap();
        assert_eq!(t.as_text().unwrap(), &["3".to_string(), String::new()]);
        let mut d = ColumnData::DateTime(vec![]);
        d.push(&Value::Text("2026-01-02".into())).unwrap();
        assert!(d.push(&Value::Numeric(1.0)).is_err());
    }

    #[test]
    fn insert_remove_take() {
        let mut c = ColumnData::Numeric(vec![1.0, 2.0, 3.0]);
        c.insert_missing(1, 2);
        assert_eq!(c.missing_count(), 2);
        let removed = c.remove_range(1, 2);
        assert_eq!(removed.len(), 2);
        assert_eq!(c, ColumnData::Numeric(vec![1.0, 2.0, 3.0]));
        let t = c.take(&[2, 0, 9]);
        assert_eq!(t.get(0), Value::Numeric(3.0));
        assert!(t.is_missing(2));
        let mut c2 = ColumnData::Numeric(vec![1.0, f64::NAN, f64::NAN]);
        c2.trim_trailing_missing();
        assert_eq!(c2.len(), 1);
    }

    #[test]
    fn conversions() {
        let t = ColumnData::Text(vec!["1".into(), "x".into(), "".into()]);
        assert!(t.convert(ColumnKind::Numeric, true).is_err());
        let n = t.convert(ColumnKind::Numeric, false).unwrap();
        assert_eq!(n.get(0), Value::Numeric(1.0));
        assert!(n.is_missing(1) && n.is_missing(2));
        let d = ColumnData::DateTime(vec![datetime::parse("2026-11-16").unwrap()]);
        let num = d.convert(ColumnKind::Numeric, true).unwrap();
        assert_eq!(num.get(0), Value::Numeric(46342.0));
        assert_eq!(num.convert(ColumnKind::DateTime, true).unwrap(), d);
    }

    #[test]
    fn levels_and_value_order() {
        let mut c = Column::text("Shift", vec!["Night", "Day", "", "Swing", "Day"]);
        assert_eq!(c.levels().iter().map(|v| v.to_text()).collect::<Vec<_>>(), ["Day", "Night", "Swing"]);
        c.props.value_order = Some(vec!["Swing".into(), "Day".into(), "Absent".into()]);
        let (lv, codes) = c.level_codes();
        assert_eq!(lv.iter().map(|v| v.to_text()).collect::<Vec<_>>(), ["Swing", "Day", "Night"]);
        assert_eq!(codes, vec![Some(2), Some(1), None, Some(0), Some(1)]);
        let n = Column::numeric("x", vec![3.0, 1.0, 3.0, f64::NAN, -0.0, 0.0]);
        assert_eq!(n.levels().len(), 3);
    }

    #[test]
    fn display_formats() {
        let mut c = Column::numeric("x", vec![1234.5678, f64::NAN]);
        c.props.numeric_format = NumericFormat::Fixed(2);
        assert_eq!(c.display(0), "1234.57");
        assert_eq!(c.display(1), "*");
        c.props.numeric_format = NumericFormat::Scientific(3);
        assert_eq!(c.display(0), "1.235E+03");
        c.props.numeric_format = NumericFormat::Percent(1);
        assert_eq!(Column { data: ColumnData::Numeric(vec![0.256]), ..c.clone() }.display(0), "25.6%");
    }
}
