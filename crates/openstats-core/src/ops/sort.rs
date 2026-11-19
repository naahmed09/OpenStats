//! Data > Sort.

use crate::column::{level_key, Column, ColumnKind};
use crate::error::{CoreError, Result};
use crate::worksheet::Worksheet;
use std::cmp::Ordering;
use std::collections::HashMap;

/// One sort key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortKey {
    pub col: usize,
    pub descending: bool,
}

impl SortKey {
    pub fn asc(col: usize) -> Self {
        SortKey { col, descending: false }
    }
    pub fn desc(col: usize) -> Self {
        SortKey { col, descending: true }
    }
}

/// Stable row permutation that sorts by `keys` (first key most significant)
/// over `n` rows. Missing values sort last regardless of direction. Text
/// columns with a custom value order sort by that order; values not listed
/// follow in alphabetical order.
pub fn sort_permutation(ws: &Worksheet, keys: &[SortKey], n: usize) -> Result<Vec<usize>> {
    if keys.is_empty() {
        return Err(CoreError::invalid("sort needs at least one key"));
    }
    // Precompute per-key comparable ranks: f64 rank for numeric/datetime,
    // ordinal rank for text (respecting value order).
    enum KeyData<'a> {
        Num(&'a [f64]),
        Time(&'a [i64]),
        Ord(Vec<Option<usize>>),
    }
    let mut kd = Vec::with_capacity(keys.len());
    for k in keys {
        let c = ws.column(k.col)?;
        kd.push(match c.kind() {
            ColumnKind::Numeric => KeyData::Num(c.data.as_numeric().expect("numeric")),
            ColumnKind::DateTime => KeyData::Time(c.data.as_datetime().expect("datetime")),
            ColumnKind::Text => {
                let (levels, codes) = c.level_codes();
                let _ = levels;
                let mut codes = codes;
                codes.resize(n, None);
                KeyData::Ord(codes)
            }
        });
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        for (k, d) in keys.iter().zip(kd.iter()) {
            let (ma, mb, ord) = match d {
                KeyData::Num(v) => {
                    let x = v.get(a).copied().unwrap_or(f64::NAN);
                    let y = v.get(b).copied().unwrap_or(f64::NAN);
                    (x.is_nan(), y.is_nan(), x.partial_cmp(&y).unwrap_or(Ordering::Equal))
                }
                KeyData::Time(v) => {
                    let x = v.get(a).copied().unwrap_or(crate::missing::DATETIME_MISSING);
                    let y = v.get(b).copied().unwrap_or(crate::missing::DATETIME_MISSING);
                    (crate::missing::is_missing_dt(x), crate::missing::is_missing_dt(y), x.cmp(&y))
                }
                KeyData::Ord(c) => (c[a].is_none(), c[b].is_none(), c[a].cmp(&c[b])),
            };
            let o = match (ma, mb) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => {
                    if k.descending {
                        ord.reverse()
                    } else {
                        ord
                    }
                }
            };
            if o != Ordering::Equal {
                return o;
            }
        }
        Ordering::Equal
    });
    Ok(idx)
}

/// Sort the `carry` columns by `keys`, returning reordered copies. The number
/// of rows is the longest of the carried and key columns.
pub fn sort_columns(ws: &Worksheet, carry: &[usize], keys: &[SortKey]) -> Result<Vec<Column>> {
    let key_cols: Vec<usize> = keys.iter().map(|k| k.col).collect();
    let n = super::rows_spanned(ws, carry)?.max(super::rows_spanned(ws, &key_cols)?);
    let perm = sort_permutation(ws, keys, n)?;
    carry
        .iter()
        .map(|&c| {
            let src = ws.column(c)?;
            let mut nc = src.clone();
            nc.data = src.data.take(&perm);
            nc.data.trim_trailing_missing();
            Ok(nc)
        })
        .collect()
}

/// Rows grouped by identical key values, in first-appearance order (helper
/// for "by-group" processing elsewhere).
pub fn group_rows(ws: &Worksheet, by: &[usize]) -> Result<Vec<Vec<usize>>> {
    let n = super::rows_spanned(ws, by)?;
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for r in 0..n {
        let mut parts = Vec::with_capacity(by.len());
        for &c in by {
            parts.push(level_key(&ws.columns[c].get(r)));
        }
        let k = parts.join("\u{1f}");
        map.entry(k.clone())
            .or_insert_with(|| {
                order.push(k);
                Vec::new()
            })
            .push(r);
    }
    Ok(order.into_iter().map(|k| map.remove(&k).expect("present")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    #[test]
    fn multi_key_stable_missing_last() {
        let mut ws = Worksheet::with_columns(
            "s",
            vec![
                Column::text("g", vec!["b", "a", "b", "a", ""]),
                Column::numeric("x", vec![2.0, f64::NAN, 1.0, 3.0, 0.0]),
                Column::numeric("id", vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            ],
        )
        .unwrap();
        let out = sort_columns(&ws, &[2], &[SortKey::asc(0), SortKey::desc(1)]).unwrap();
        assert_eq!(out[0].data, ColumnData::Numeric(vec![3.0, 1.0, 0.0, 2.0, 4.0]));
        ws.columns[0].props.value_order = Some(vec!["b".into(), "a".into()]);
        let out = sort_columns(&ws, &[2], &[SortKey::asc(0), SortKey::asc(1)]).unwrap();
        assert_eq!(out[0].data, ColumnData::Numeric(vec![2.0, 0.0, 3.0, 1.0, 4.0]));
        let g = group_rows(&ws, &[0]).unwrap();
        assert_eq!(g, vec![vec![0, 2], vec![1, 3], vec![4]]);
    }
}
