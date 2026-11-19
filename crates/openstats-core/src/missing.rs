//! Missing-value semantics.
//!
//! Minitab conventions are followed exactly:
//!
//! * **Numeric** columns store missing as `NaN`; every NaN bit pattern is
//!   treated as missing and is displayed as `*`.
//! * **Date/time** columns store missing as [`DATETIME_MISSING`] (`i64::MIN`),
//!   also displayed as `*`.
//! * **Text** columns store missing as the empty string.
//!
//! Arithmetic on numeric values propagates missing automatically through IEEE
//! NaN semantics; the helpers here make the intent explicit and implement the
//! case-deletion rules used by statistical procedures.

/// Canonical numeric missing value.
pub const NUMERIC_MISSING: f64 = f64::NAN;
/// Canonical date/time missing value.
pub const DATETIME_MISSING: i64 = i64::MIN;
/// Display token for numeric / date-time missing values.
pub const MISSING_TOKEN: &str = "*";

/// `true` if `x` is a numeric missing value (any NaN).
#[inline]
pub fn is_missing_f64(x: f64) -> bool {
    x.is_nan()
}

/// `true` if `t` is a date/time missing value.
#[inline]
pub fn is_missing_dt(t: i64) -> bool {
    t == DATETIME_MISSING
}

/// `true` if `s` is a text missing value (empty string).
#[inline]
pub fn is_missing_text(s: &str) -> bool {
    s.is_empty()
}

/// Tokens that are read as missing when parsing numeric or date/time text.
/// `*` is Minitab's own token; the others are common exports from other tools.
pub fn is_missing_token(s: &str) -> bool {
    let t = s.trim();
    t.is_empty()
        || t == "*"
        || t.eq_ignore_ascii_case("na")
        || t.eq_ignore_ascii_case("nan")
        || t.eq_ignore_ascii_case("n/a")
        || t == "."
        || t.eq_ignore_ascii_case("null")
}

/// Row-wise "complete cases" mask across several numeric slices
/// (listwise deletion). Slices of unequal length treat absent rows as missing.
pub fn complete_cases(columns: &[&[f64]]) -> Vec<bool> {
    let n = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    (0..n).map(|i| columns.iter().all(|c| i < c.len() && !c[i].is_nan())).collect()
}

/// Extract the rows of `columns` for which `mask` is `true`.
pub fn apply_mask(columns: &[&[f64]], mask: &[bool]) -> Vec<Vec<f64>> {
    columns
        .iter()
        .map(|c| c.iter().zip(mask.iter()).filter_map(|(&v, &keep)| if keep { Some(v) } else { None }).collect())
        .collect()
}

/// Listwise deletion: keep only rows where every column is non-missing.
pub fn listwise(columns: &[&[f64]]) -> Vec<Vec<f64>> {
    let mask = complete_cases(columns);
    apply_mask(columns, &mask)
}

/// Pairwise deletion for two columns: rows where both values are present.
pub fn pairwise(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().min(y.len());
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            a.push(x[i]);
            b.push(y[i]);
        }
    }
    (a, b)
}

/// Remove missing values from a numeric slice.
pub fn drop_missing(x: &[f64]) -> Vec<f64> {
    x.iter().copied().filter(|v| !v.is_nan()).collect()
}

/// Count missing values in a numeric slice.
pub fn count_missing(x: &[f64]) -> usize {
    x.iter().filter(|v| v.is_nan()).count()
}

/// Binary numeric operation with explicit missing propagation. IEEE arithmetic
/// already propagates NaN, but operations such as `max`/`min` and comparisons
/// do not, so they must go through this helper.
#[inline]
pub fn propagate2(a: f64, b: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        NUMERIC_MISSING
    } else {
        f(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens() {
        for t in ["", " ", "*", "NA", "nan", "N/A", ".", "NULL"] {
            assert!(is_missing_token(t), "{t:?}");
        }
        assert!(!is_missing_token("0"));
        assert!(!is_missing_token("x"));
    }

    #[test]
    fn listwise_and_pairwise() {
        let a = [1.0, f64::NAN, 3.0, 4.0];
        let b = [1.0, 2.0, f64::NAN];
        assert_eq!(complete_cases(&[&a, &b]), vec![true, false, false, false]);
        assert_eq!(listwise(&[&a, &b]), vec![vec![1.0], vec![1.0]]);
        let (x, y) = pairwise(&a, &b);
        assert_eq!((x, y), (vec![1.0], vec![1.0]));
        assert_eq!(count_missing(&a), 1);
        assert_eq!(drop_missing(&a), vec![1.0, 3.0, 4.0]);
    }

    #[test]
    fn propagation() {
        assert!(propagate2(f64::NAN, 1.0, f64::max).is_nan());
        assert_eq!(propagate2(2.0, 1.0, f64::max), 2.0);
    }
}
