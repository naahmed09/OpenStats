//! Data > Unstack.

use crate::column::{level_key, Column, ColumnData};
use crate::error::{CoreError, Result};
use crate::value::Value;
use crate::worksheet::Worksheet;
use std::collections::HashMap;

/// Unstack options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnstackOptions {
    /// Treat missing subscript values as their own group (named `*`).
    pub include_missing_group: bool,
}

/// Unstack `data_cols` by the combined levels of `by` columns. For every
/// group (in value order of the subscripts, first `by` column slowest) one
/// output column per data column is produced, named `Data_Level[_Level2…]`.
pub fn unstack(ws: &Worksheet, data_cols: &[usize], by: &[usize], opts: UnstackOptions) -> Result<Vec<Column>> {
    if data_cols.is_empty() || by.is_empty() {
        return Err(CoreError::invalid("unstack needs data and subscript columns"));
    }
    let n = super::rows_spanned(ws, data_cols)?.max(super::rows_spanned(ws, by)?);
    // Levels per subscript column, in value order.
    let mut level_lists: Vec<Vec<Value>> = Vec::new();
    let mut code_lists: Vec<Vec<Option<usize>>> = Vec::new();
    for &b in by {
        let (mut lv, mut codes) = ws.column(b)?.level_codes();
        codes.resize(n, None);
        if opts.include_missing_group && codes.iter().any(|c| c.is_none()) {
            let miss = lv.len();
            lv.push(Value::Missing);
            for c in codes.iter_mut() {
                if c.is_none() {
                    *c = Some(miss);
                }
            }
        }
        level_lists.push(lv);
        code_lists.push(codes);
    }
    // Groups that actually occur, ordered lexicographically by level codes.
    let mut group_rows: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for r in 0..n {
        let key: Option<Vec<usize>> = code_lists.iter().map(|c| c[r]).collect();
        if let Some(k) = key {
            group_rows.entry(k).or_default().push(r);
        }
    }
    let mut groups: Vec<Vec<usize>> = group_rows.keys().cloned().collect();
    groups.sort();
    let mut out = Vec::new();
    for g in &groups {
        let rows = &group_rows[g];
        let label: Vec<String> = g
            .iter()
            .zip(level_lists.iter())
            .map(|(&i, lv)| match &lv[i] {
                Value::Missing => "*".to_string(),
                v => v.to_text(),
            })
            .collect();
        for &d in data_cols {
            let src = ws.column(d)?;
            let data: ColumnData = src.data.take(rows);
            let base = ws.display_name(d);
            let mut col = Column::new(format!("{base}_{}", label.join("_")), data);
            col.props.numeric_format = src.props.numeric_format;
            col.props.datetime_format = src.props.datetime_format.clone();
            out.push(col);
        }
    }
    Ok(out)
}

/// Key string for the combined subscripts of a row (used by split/merge).
pub(crate) fn row_key(ws: &Worksheet, cols: &[usize], r: usize) -> Option<String> {
    let mut parts = Vec::with_capacity(cols.len());
    for &c in cols {
        let v = ws.columns[c].get(r);
        if v.is_missing() {
            return None;
        }
        parts.push(level_key(&v));
    }
    Some(parts.join("\u{1f}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unstack_two_way() {
        let ws = Worksheet::with_columns(
            "s",
            vec![
                Column::numeric("Y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Column::text("Shift", vec!["B", "A", "B", "A", "", "A"]),
                Column::numeric("Line", vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0]),
            ],
        )
        .unwrap();
        let out = unstack(&ws, &[0], &[1, 2], UnstackOptions::default()).unwrap();
        let names: Vec<_> = out.iter().map(|c| c.name().to_string()).collect();
        assert_eq!(names, ["Y_A_1", "Y_A_2", "Y_B_1", "Y_B_2"]);
        assert_eq!(out[0].data, ColumnData::Numeric(vec![2.0, 6.0]));
        let with_missing = unstack(&ws, &[0], &[1], UnstackOptions { include_missing_group: true }).unwrap();
        assert_eq!(with_missing.last().unwrap().name(), "Y_*");
        assert_eq!(with_missing.last().unwrap().data, ColumnData::Numeric(vec![5.0]));
    }
}
