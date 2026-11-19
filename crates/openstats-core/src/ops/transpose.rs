//! Data > Transpose Columns.

use crate::column::{Column, ColumnData, ColumnKind};
use crate::error::{CoreError, Result};
use crate::worksheet::Worksheet;

/// Transpose `cols` (each becomes a row). Output column *j* holds row *j* of
/// every source column. If every source column is numeric the outputs are
/// numeric, otherwise text. With `labels`, a leading text column `Labels`
/// holds the source column names. With `names_from`, output columns are named
/// from that column's values (which is then excluded from the transposed data).
pub fn transpose(ws: &Worksheet, cols: &[usize], labels: bool, names_from: Option<usize>) -> Result<Vec<Column>> {
    if cols.is_empty() {
        return Err(CoreError::invalid("nothing to transpose"));
    }
    let data_cols: Vec<usize> = cols.iter().copied().filter(|c| Some(*c) != names_from).collect();
    let n = super::rows_spanned(ws, &data_cols)?;
    let all_numeric = data_cols.iter().all(|&c| ws.columns[c].kind() == ColumnKind::Numeric);
    let mut out = Vec::with_capacity(n + 1);
    if labels {
        out.push(Column::text("Labels", data_cols.iter().map(|&c| ws.display_name(c)).collect::<Vec<_>>()));
    }
    for r in 0..n {
        let name = match names_from {
            Some(nc) => ws.column(nc)?.display(r),
            None => String::new(),
        };
        let data = if all_numeric {
            ColumnData::Numeric(data_cols.iter().map(|&c| ws.columns[c].get(r).as_f64()).collect())
        } else {
            ColumnData::Text(
                data_cols
                    .iter()
                    .map(|&c| {
                        let col = &ws.columns[c];
                        if col.data.is_missing(r) {
                            String::new()
                        } else {
                            col.display(r)
                        }
                    })
                    .collect(),
            )
        };
        out.push(Column::new(if name == "*" { String::new() } else { name }, data));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    #[test]
    fn transpose_numeric_and_mixed() {
        let ws = Worksheet::with_columns(
            "s",
            vec![
                Column::text("id", vec!["r1", "r2"]),
                Column::numeric("a", vec![1.0, 2.0]),
                Column::numeric("b", vec![3.0]),
            ],
        )
        .unwrap();
        let t = transpose(&ws, &[0, 1, 2], true, Some(0)).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].data.as_text().unwrap(), ["a", "b"]);
        assert_eq!(t[1].name(), "r1");
        assert_eq!(t[1].data, ColumnData::Numeric(vec![1.0, 3.0]));
        assert!(t[2].data.is_missing(1));
        let m = transpose(&ws, &[0, 1], false, None).unwrap();
        assert_eq!(m[0].get(1), Value::Text("1".into()));
    }
}
