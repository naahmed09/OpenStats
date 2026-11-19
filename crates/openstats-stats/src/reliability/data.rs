//! Life data with exact, right-, left- and interval-censored observations.

use crate::error::{invalid, Result, StatsError};

/// One observation: the failure time lies in `[lower, upper]` (equal for an exact time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obs {
    /// Lower bound (`0` or `−∞` for a left-censored time).
    pub lower: f64,
    /// Upper bound (`+∞` for a right-censored time).
    pub upper: f64,
    /// Number of units this record stands for.
    pub weight: f64,
    /// Index of the worksheet row (within the original columns).
    pub row: usize,
}

/// Kinds of observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsKind {
    /// Failure time observed.
    Exact,
    /// Survived beyond `lower`.
    Right,
    /// Failed before `upper`.
    Left,
    /// Failed between `lower` and `upper`.
    Interval,
}

impl Obs {
    /// Classify the record.
    pub fn kind(&self) -> ObsKind {
        if self.lower == self.upper {
            ObsKind::Exact
        } else if self.upper == f64::INFINITY {
            ObsKind::Right
        } else if self.lower <= 0.0 || self.lower == f64::NEG_INFINITY {
            ObsKind::Left
        } else {
            ObsKind::Interval
        }
    }
}

/// A set of life-time observations.
#[derive(Debug, Clone, PartialEq)]
pub struct LifeData {
    /// Observations (rows with missing times are not included).
    pub obs: Vec<Obs>,
    /// Number of input rows that were skipped because of a missing time, frequency or censor code.
    pub n_skipped: usize,
}

impl LifeData {
    /// Failure/suspension times with an optional censoring column and frequencies.
    ///
    /// A row is right-censored when `censor[row] == censored_value`; any other value is a
    /// failure. Missing (`NaN`) times or frequencies skip the row. Frequencies must be
    /// positive whole numbers.
    pub fn from_times(
        time: &[f64],
        censor: Option<&[f64]>,
        censored_value: f64,
        freq: Option<&[f64]>,
    ) -> Result<LifeData> {
        let n = time.len();
        check_len(n, censor.map(|c| c.len()), freq.map(|f| f.len()))?;
        let mut obs = Vec::with_capacity(n);
        let mut skipped = 0;
        for i in 0..n {
            let w = weight(freq, i)?;
            let t = time[i];
            if t.is_nan() || w.is_nan() || censor.is_some_and(|c| c[i].is_nan()) {
                skipped += 1;
                continue;
            }
            if t.is_infinite() {
                return Err(StatsError::NonFinite);
            }
            if t < 0.0 {
                return invalid(format!("row {}: times must not be negative", i + 1));
            }
            let right = censor.is_some_and(|c| c[i] == censored_value);
            obs.push(Obs { lower: t, upper: if right { f64::INFINITY } else { t }, weight: w, row: i });
        }
        finish(obs, skipped)
    }

    /// Arbitrary censoring from start and end columns: a missing start is a left-censored time
    /// (failed before `end`), a missing end is right-censored (survived past `start`), equal
    /// values are an exact time, and `start < end` is an interval.
    pub fn from_intervals(start: &[f64], end: &[f64], freq: Option<&[f64]>) -> Result<LifeData> {
        let n = start.len();
        check_len(n, Some(end.len()), freq.map(|f| f.len()))?;
        let mut obs = Vec::with_capacity(n);
        let mut skipped = 0;
        for i in 0..n {
            let w = weight(freq, i)?;
            let (s, e) = (start[i], end[i]);
            if w.is_nan() || (s.is_nan() && e.is_nan()) {
                skipped += 1;
                continue;
            }
            if s.is_infinite() || e.is_infinite() {
                return Err(StatsError::NonFinite);
            }
            let lower = if s.is_nan() { 0.0 } else { s };
            let upper = if e.is_nan() { f64::INFINITY } else { e };
            if lower < 0.0 || upper < lower {
                return invalid(format!("row {}: need 0 ≤ start ≤ end", i + 1));
            }
            obs.push(Obs { lower, upper, weight: w, row: i });
        }
        finish(obs, skipped)
    }

    /// Total number of units.
    pub fn total(&self) -> f64 {
        self.obs.iter().map(|o| o.weight).sum()
    }

    /// Units with a failure observed (exact, left or interval).
    pub fn failures(&self) -> f64 {
        self.obs.iter().filter(|o| o.kind() != ObsKind::Right).map(|o| o.weight).sum()
    }

    /// Right-censored units.
    pub fn right_censored(&self) -> f64 {
        self.total() - self.failures()
    }
}

fn check_len(n: usize, a: Option<usize>, b: Option<usize>) -> Result<()> {
    if a.is_some_and(|m| m != n) || b.is_some_and(|m| m != n) {
        return invalid("the time, censoring and frequency columns must have the same length");
    }
    Ok(())
}

fn weight(freq: Option<&[f64]>, i: usize) -> Result<f64> {
    match freq {
        None => Ok(1.0),
        Some(f) => {
            let w = f[i];
            if w.is_nan() {
                Ok(f64::NAN)
            } else if w < 0.0 || w.fract() != 0.0 || w.is_infinite() {
                invalid(format!("row {}: frequencies must be non-negative whole numbers", i + 1))
            } else {
                Ok(w)
            }
        }
    }
}

fn finish(mut obs: Vec<Obs>, n_skipped: usize) -> Result<LifeData> {
    obs.retain(|o| o.weight > 0.0);
    if obs.is_empty() {
        return Err(StatsError::InsufficientData { what: "life data", needed: 1, got: 0 });
    }
    Ok(LifeData { obs, n_skipped })
}
