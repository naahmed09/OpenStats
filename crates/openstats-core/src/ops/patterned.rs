//! Calc > Make Patterned Data.

use crate::column::ColumnData;
use crate::datetime;
use crate::error::{CoreError, Result};

/// Upper bound on generated rows, to protect against runaway patterns.
pub const MAX_PATTERN_ROWS: usize = 50_000_000;

/// Repeat each element `each` times, then the whole sequence `list` times.
fn repeat<T: Clone>(seq: &[T], each: usize, list: usize) -> Result<Vec<T>> {
    if each == 0 || list == 0 {
        return Err(CoreError::invalid("repeat counts must be at least 1"));
    }
    let total = seq.len().checked_mul(each).and_then(|x| x.checked_mul(list)).unwrap_or(usize::MAX);
    if total > MAX_PATTERN_ROWS {
        return Err(CoreError::invalid(format!("pattern would create {total} rows")));
    }
    let mut out = Vec::with_capacity(total);
    for _ in 0..list {
        for v in seq {
            for _ in 0..each {
                out.push(v.clone());
            }
        }
    }
    Ok(out)
}

/// Simple set of numbers `from, from+step, … ≤ to` (or ≥ to when stepping
/// down). Values are computed as `from + k*step` to avoid drift, and the
/// endpoint is included when within 1e-9·|step|.
pub fn numbers(from: f64, to: f64, step: f64, each: usize, list: usize) -> Result<ColumnData> {
    if !from.is_finite() || !to.is_finite() || !step.is_finite() || step == 0.0 {
        return Err(CoreError::invalid("from/to must be finite and step non-zero"));
    }
    if (to - from) * step < 0.0 {
        return Err(CoreError::invalid("step has the wrong sign for this range"));
    }
    let count = ((to - from) / step + 1e-9).floor() as usize + 1;
    if count > MAX_PATTERN_ROWS {
        return Err(CoreError::invalid(format!("pattern would create {count} values")));
    }
    let seq: Vec<f64> = (0..count).map(|k| from + k as f64 * step).collect();
    Ok(ColumnData::Numeric(repeat(&seq, each, list)?))
}

/// Arbitrary set of values (any column kind) with repeats.
pub fn values(seq: &ColumnData, each: usize, list: usize) -> Result<ColumnData> {
    Ok(match seq {
        ColumnData::Numeric(v) => ColumnData::Numeric(repeat(v, each, list)?),
        ColumnData::Text(v) => ColumnData::Text(repeat(v, each, list)?),
        ColumnData::DateTime(v) => ColumnData::DateTime(repeat(v, each, list)?),
    })
}

/// Unit for date/time patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

/// Date/time step `k` from `start` (calendar-aware for months/years).
pub fn step_time(start: i64, k: i64, amount: i64, unit: TimeUnit) -> i64 {
    use datetime::*;
    match unit {
        TimeUnit::Second => start + k * amount * MICROS_PER_SECOND,
        TimeUnit::Minute => start + k * amount * MICROS_PER_MINUTE,
        TimeUnit::Hour => start + k * amount * MICROS_PER_HOUR,
        TimeUnit::Day => start + k * amount * MICROS_PER_DAY,
        TimeUnit::Week => start + k * amount * 7 * MICROS_PER_DAY,
        TimeUnit::Month => add_months(start, k * amount),
        TimeUnit::Quarter => add_months(start, k * amount * 3),
        TimeUnit::Year => add_months(start, k * amount * 12),
    }
}

/// Patterned date/times from `start` to `end` inclusive.
pub fn dates(start: i64, end: i64, amount: i64, unit: TimeUnit, each: usize, list: usize) -> Result<ColumnData> {
    if amount == 0 {
        return Err(CoreError::invalid("step must be non-zero"));
    }
    if (end - start).signum() * amount.signum() < 0 {
        return Err(CoreError::invalid("step has the wrong sign for this range"));
    }
    let mut seq = Vec::new();
    let mut k = 0;
    loop {
        // Month arithmetic is computed from `start` each time so that
        // Jan 31 + 1, 2, 3 months gives Feb 28, Mar 31, Apr 30.
        let t = step_time(start, k, amount, unit);
        if (amount > 0 && t > end) || (amount < 0 && t < end) {
            break;
        }
        seq.push(t);
        k += 1;
        if seq.len() > MAX_PATTERN_ROWS {
            return Err(CoreError::invalid("pattern too long"));
        }
    }
    Ok(ColumnData::DateTime(repeat(&seq, each, list)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_patterns() {
        assert_eq!(
            numbers(1.0, 3.0, 1.0, 2, 2).unwrap().as_numeric().unwrap(),
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
        let v = numbers(0.0, 1.0, 0.1, 1, 1).unwrap();
        assert_eq!(v.len(), 11);
        assert_eq!(v.as_numeric().unwrap()[3], 0.30000000000000004);
        assert_eq!(numbers(5.0, 1.0, -2.0, 1, 1).unwrap().as_numeric().unwrap(), [5.0, 3.0, 1.0]);
        assert!(numbers(1.0, 5.0, -1.0, 1, 1).is_err());
        assert!(numbers(1.0, 5.0, 1.0, 0, 1).is_err());
        let t = values(&ColumnData::Text(vec!["a".into(), "b".into()]), 1, 3).unwrap();
        assert_eq!(t.len(), 6);
    }

    #[test]
    fn date_patterns() {
        let s = datetime::parse("2024-01-31").unwrap();
        let e = datetime::parse("2024-05-31").unwrap();
        let d = dates(s, e, 1, TimeUnit::Month, 1, 1).unwrap();
        let f: Vec<String> = d.as_datetime().unwrap().iter().map(|&t| datetime::format(t, "MM-dd")).collect();
        assert_eq!(f, ["01-31", "02-29", "03-31", "04-30", "05-31"]);
        let w = dates(s, s + 20 * datetime::MICROS_PER_DAY, 1, TimeUnit::Week, 1, 1).unwrap();
        assert_eq!(w.len(), 3);
    }
}
