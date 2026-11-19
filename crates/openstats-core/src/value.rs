//! Single cell values.

use crate::datetime;
use crate::error::{CoreError, Result};
use crate::missing::{self, DATETIME_MISSING};
use std::cmp::Ordering;
use std::fmt;

/// A single cell value. `Missing` is type-less; when written into a typed
/// column it becomes that column's missing representation.
///
/// Equality is missing-aware: `Numeric(NaN) == Numeric(NaN)`.
#[derive(Debug, Clone)]
pub enum Value {
    Numeric(f64),
    Text(String),
    /// Microseconds since 1970-01-01T00:00:00Z.
    DateTime(i64),
    Missing,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Numeric(a), Value::Numeric(b)) => crate::column::f64_eq(*a, *b),
            (Value::Text(a), Value::Text(b)) => a == b,
            (Value::DateTime(a), Value::DateTime(b)) => a == b,
            (Value::Missing, Value::Missing) => true,
            _ => false,
        }
    }
}

impl Value {
    /// `true` for `Missing`, NaN numerics, missing date/times and empty text.
    pub fn is_missing(&self) -> bool {
        match self {
            Value::Missing => true,
            Value::Numeric(x) => x.is_nan(),
            Value::DateTime(t) => *t == DATETIME_MISSING,
            Value::Text(s) => s.is_empty(),
        }
    }

    /// Human-readable type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Numeric(_) => "numeric",
            Value::Text(_) => "text",
            Value::DateTime(_) => "date/time",
            Value::Missing => "missing",
        }
    }

    /// Numeric view: numbers as-is, date/times as Excel-style serial days
    /// (Minitab treats dates as numbers in arithmetic), parsable text as its
    /// number, everything else NaN.
    pub fn as_f64(&self) -> f64 {
        match self {
            Value::Numeric(x) => *x,
            Value::DateTime(t) => datetime::to_excel_serial(*t),
            Value::Text(s) => parse_number(s).unwrap_or(f64::NAN),
            Value::Missing => f64::NAN,
        }
    }

    /// Text view used for display and text-column coercion. Missing numerics
    /// and date/times display as `*`; missing text and `Missing` as `""`.
    pub fn to_text(&self) -> String {
        match self {
            Value::Numeric(x) => format_number(*x),
            Value::Text(s) => s.clone(),
            Value::DateTime(t) => datetime::format_default(*t),
            Value::Missing => String::new(),
        }
    }

    /// Total ordering used for sorting: Minitab places missing values last;
    /// numeric < date/time < text when mixing types.
    pub fn total_cmp(&self, other: &Value) -> Ordering {
        fn rank(v: &Value) -> u8 {
            if v.is_missing() {
                return 3;
            }
            match v {
                Value::Numeric(_) => 0,
                Value::DateTime(_) => 1,
                Value::Text(_) => 2,
                Value::Missing => 3,
            }
        }
        let (ra, rb) = (rank(self), rank(other));
        if ra != rb {
            return ra.cmp(&rb);
        }
        match (self, other) {
            (Value::Numeric(a), Value::Numeric(b)) if ra == 0 => a.total_cmp(b),
            (Value::DateTime(a), Value::DateTime(b)) if ra == 1 => a.cmp(b),
            (Value::Text(a), Value::Text(b)) if ra == 2 => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Missing => f.write_str(missing::MISSING_TOKEN),
            other => f.write_str(&other.to_text()),
        }
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Value::Numeric(x)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Text(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Text(s)
    }
}

/// Format a number the way it is shown in a worksheet when no explicit format
/// is set: shortest round-trip representation, `*` for missing, no trailing
/// `.0` for integers, and scientific notation for very large/small magnitudes.
pub fn format_number(x: f64) -> String {
    if x.is_nan() {
        return missing::MISSING_TOKEN.to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return "0".into();
    }
    let ax = x.abs();
    if (1e-5..1e15).contains(&ax) {
        let s = format!("{x}");
        s
    } else {
        // Shortest exponent form, e.g. 1.5E+20, 2E-07
        let s = format!("{x:e}");
        let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = exp.parse().unwrap_or(0);
        format!("{mant}E{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

/// Parse a number from user text. Accepts leading/trailing whitespace, a
/// leading `+`, thousands separators (`1,234.5`), exponents, and `inf`.
/// Missing tokens return `Ok(NaN)`. Returns an error if the text is not
/// numeric.
pub fn parse_number(s: &str) -> Result<f64> {
    parse_number_with(s, '.')
}

/// [`parse_number`] with a configurable decimal separator (`.` or `,`).
/// When the decimal separator is `,`, `.` and spaces are treated as
/// thousands separators.
pub fn parse_number_with(s: &str, decimal: char) -> Result<f64> {
    let t = s.trim();
    if missing::is_missing_token(t) && t != "." {
        return Ok(f64::NAN);
    }
    let thousands = if decimal == ',' { '.' } else { ',' };
    let mut cleaned = String::with_capacity(t.len());
    let mut seen_digit = false;
    let mut in_exponent = false;
    for (i, c) in t.char_indices() {
        if c == thousands && !in_exponent {
            // Only accept as a grouping separator between digits.
            let next_is_digit = t[i + c.len_utf8()..].chars().next().is_some_and(|n| n.is_ascii_digit());
            if seen_digit && next_is_digit && decimal != thousands {
                continue;
            }
            return Err(CoreError::parse("number", format!("unexpected {c:?} in {s:?}"), Some(i)));
        }
        if decimal == ',' && c == ' ' {
            continue;
        }
        if c == decimal {
            cleaned.push('.');
            continue;
        }
        if c == 'e' || c == 'E' {
            in_exponent = true;
        }
        if c.is_ascii_digit() {
            seen_digit = true;
        }
        cleaned.push(c);
    }
    let lc = cleaned.to_ascii_lowercase();
    match lc.as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => return Ok(f64::INFINITY),
        "-inf" | "-infinity" => return Ok(f64::NEG_INFINITY),
        _ => {}
    }
    if !seen_digit || lc.contains("nan") || lc.contains("inf") {
        return Err(CoreError::parse("number", format!("not a number: {s:?}"), None));
    }
    cleaned.parse::<f64>().map_err(|_| CoreError::parse("number", format!("not a number: {s:?}"), None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formatting() {
        assert_eq!(format_number(3.0), "3");
        assert_eq!(format_number(-0.125), "-0.125");
        assert_eq!(format_number(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(format_number(1.5e20), "1.5E+20");
        assert_eq!(format_number(2e-7), "2E-07");
        assert_eq!(format_number(f64::NAN), "*");
    }

    #[test]
    fn number_parsing() {
        assert_eq!(parse_number(" 1,234.5 ").unwrap(), 1234.5);
        assert_eq!(parse_number("+3e2").unwrap(), 300.0);
        assert_eq!(parse_number_with("1.234,5", ',').unwrap(), 1234.5);
        assert!(parse_number("*").unwrap().is_nan());
        assert!(parse_number("abc").is_err());
        assert!(parse_number("1,,2").is_err());
        assert!(parse_number(".").is_err());
        assert!(parse_number("nan").unwrap().is_nan());
        assert_eq!(parse_number("-inf").unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn ordering_missing_last() {
        let mut v = [
            Value::Missing,
            Value::Numeric(2.0),
            Value::Text("a".into()),
            Value::Numeric(f64::NAN),
            Value::Numeric(1.0),
        ];
        v.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(v[0], Value::Numeric(1.0));
        assert_eq!(v[1], Value::Numeric(2.0));
        assert_eq!(v[2], Value::Text("a".into()));
        assert!(v[3].is_missing() && v[4].is_missing());
    }
}
