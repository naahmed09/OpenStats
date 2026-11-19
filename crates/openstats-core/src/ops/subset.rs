//! Data > Subset Worksheet and Data > Copy > Columns to Columns.

use crate::column::Column;
use crate::error::{CoreError, Result};
use crate::value::Value;
use crate::worksheet::Worksheet;

/// Which rows to keep (or exclude).
#[derive(Debug, Clone, PartialEq)]
pub enum RowSelector {
    All,
    /// Explicit 0-based row indices (duplicates and order are honored).
    Rows(Vec<usize>),
    /// Half-open range `start..end` (0-based).
    Range(usize, usize),
    /// Boolean mask; rows past the mask's end are not selected.
    Mask(Vec<bool>),
    /// Rows whose value in `col` is one of `values`.
    ValuesIn {
        col: usize,
        values: Vec<Value>,
    },
    /// Rows whose numeric value in `col` lies in `[lo, hi]`.
    Between {
        col: usize,
        lo: f64,
        hi: f64,
    },
    /// Rows where `col` is (or is not, with `NotMissing`) missing.
    Missing(usize),
    NotMissing(usize),
}

/// Resolve a selector to ascending row indices within `n` rows (except
/// `Rows`, which keeps its given order). With `exclude`, the complement is
/// returned in ascending order.
pub fn select_rows(ws: &Worksheet, sel: &RowSelector, exclude: bool) -> Result<Vec<usize>> {
    let n = ws.n_rows();
    let picked: Vec<usize> = match sel {
        RowSelector::All => (0..n).collect(),
        RowSelector::Rows(r) => {
            if let Some(&bad) = r.iter().find(|&&i| i >= n) {
                return Err(CoreError::IndexOutOfRange { what: "row", index: bad, len: n });
            }
            r.clone()
        }
        RowSelector::Range(s, e) => (*s.min(&n)..*e.min(&n)).collect(),
        RowSelector::Mask(m) => (0..n.min(m.len())).filter(|&i| m[i]).collect(),
        RowSelector::ValuesIn { col, values } => {
            let c = ws.column(*col)?;
            let keys: std::collections::HashSet<String> = values
                .iter()
                .map(|v| {
                    // Coerce probe values to the column type so "3" matches 3.0.
                    let mut tmp = crate::column::ColumnData::empty(c.kind());
                    match tmp.push(v) {
                        Ok(()) => crate::column::level_key(&tmp.get(0)),
                        Err(_) => crate::column::level_key(v),
                    }
                })
                .collect();
            (0..n)
                .filter(|&i| {
                    let v = c.get(i);
                    !v.is_missing() && keys.contains(&crate::column::level_key(&v))
                })
                .collect()
        }
        RowSelector::Between { col, lo, hi } => {
            let c = ws.column(*col)?;
            (0..n)
                .filter(|&i| {
                    let x = c.get(i).as_f64();
                    !x.is_nan() && x >= *lo && x <= *hi
                })
                .collect()
        }
        RowSelector::Missing(col) => {
            let c = ws.column(*col)?;
            (0..n).filter(|&i| c.data.is_missing(i)).collect()
        }
        RowSelector::NotMissing(col) => {
            let c = ws.column(*col)?;
            (0..n).filter(|&i| !c.data.is_missing(i)).collect()
        }
    };
    if exclude {
        let mut keep = vec![true; n];
        for i in picked {
            keep[i] = false;
        }
        Ok((0..n).filter(|&i| keep[i]).collect())
    } else {
        Ok(picked)
    }
}

/// New worksheet holding the selected rows of every column.
pub fn subset_worksheet(ws: &Worksheet, sel: &RowSelector, exclude: bool, name: &str) -> Result<Worksheet> {
    let rows = select_rows(ws, sel, exclude)?;
    let mut out = Worksheet::new(name);
    out.description = ws.description.clone();
    out.constants = ws.constants.clone();
    out.matrices = ws.matrices.clone();
    for c in &ws.columns {
        let mut nc = c.clone();
        nc.data = c.data.take(&rows);
        nc.data.trim_trailing_missing();
        out.columns.push(nc);
    }
    Ok(out)
}

/// Copy selected rows of selected columns (output columns keep names and
/// formats; store them with [`super::store_columns`]).
pub fn copy_columns(ws: &Worksheet, cols: &[usize], sel: &RowSelector, exclude: bool) -> Result<Vec<Column>> {
    let rows = select_rows(ws, sel, exclude)?;
    cols.iter()
        .map(|&c| {
            let src = ws.column(c)?;
            let mut nc = src.clone();
            nc.data = src.data.take(&rows);
            nc.data.trim_trailing_missing();
            Ok(nc)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn ws() -> Worksheet {
        Worksheet::with_columns(
            "Data",
            vec![Column::numeric("x", vec![5.0, f64::NAN, 7.0, 8.0]), Column::text("g", vec!["a", "b", "a", "c"])],
        )
        .unwrap()
    }

    #[test]
    fn selectors() {
        let w = ws();
        assert_eq!(select_rows(&w, &RowSelector::Range(1, 9), false).unwrap(), vec![1, 2, 3]);
        assert_eq!(select_rows(&w, &RowSelector::Missing(0), false).unwrap(), vec![1]);
        assert_eq!(select_rows(&w, &RowSelector::Missing(0), true).unwrap(), vec![0, 2, 3]);
        assert_eq!(select_rows(&w, &RowSelector::Between { col: 0, lo: 6.0, hi: 8.0 }, false).unwrap(), vec![2, 3]);
        let sel = RowSelector::ValuesIn { col: 1, values: vec!["a".into(), "c".into()] };
        assert_eq!(select_rows(&w, &sel, false).unwrap(), vec![0, 2, 3]);
        let sel = RowSelector::ValuesIn { col: 0, values: vec!["7".into()] };
        assert_eq!(select_rows(&w, &sel, false).unwrap(), vec![2]);
        assert!(select_rows(&w, &RowSelector::Rows(vec![9]), false).is_err());
        assert_eq!(select_rows(&w, &RowSelector::Mask(vec![true, false, true]), false).unwrap(), vec![0, 2]);
    }

    #[test]
    fn subset_and_copy() {
        let w = ws();
        let s = subset_worksheet(&w, &RowSelector::Rows(vec![3, 0]), false, "Sub").unwrap();
        assert_eq!(s.columns[0].data, ColumnData::Numeric(vec![8.0, 5.0]));
        let c = copy_columns(&w, &[1], &RowSelector::NotMissing(0), false).unwrap();
        assert_eq!(c[0].data.as_text().unwrap(), ["a", "a", "c"]);
    }
}
