//! Data > Split Worksheet.

use crate::error::{CoreError, Result};
use crate::value::Value;
use crate::worksheet::Worksheet;

/// Split a worksheet into one worksheet per combination of levels of the
/// `by` columns (rows with a missing `by` value are dropped unless
/// `include_missing` is set). Worksheets are named
/// `Sheet(Var1 = level1, Var2 = level2)` and ordered by value order.
pub fn split_worksheet(ws: &Worksheet, by: &[usize], include_missing: bool) -> Result<Vec<Worksheet>> {
    if by.is_empty() {
        return Err(CoreError::invalid("split needs at least one By column"));
    }
    let n = ws.n_rows();
    let mut levels = Vec::new();
    let mut codes = Vec::new();
    for &b in by {
        let (mut lv, mut c) = ws.column(b)?.level_codes();
        c.resize(n, None);
        if include_missing && c.iter().any(|x| x.is_none()) {
            let m = lv.len();
            lv.push(Value::Missing);
            c.iter_mut().filter(|x| x.is_none()).for_each(|x| *x = Some(m));
        }
        levels.push(lv);
        codes.push(c);
    }
    let mut groups: std::collections::BTreeMap<Vec<usize>, Vec<usize>> = std::collections::BTreeMap::new();
    for r in 0..n {
        if let Some(k) = codes.iter().map(|c| c[r]).collect::<Option<Vec<usize>>>() {
            groups.entry(k).or_default().push(r);
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for (key, rows) in groups {
        let desc: Vec<String> = key
            .iter()
            .zip(by.iter())
            .zip(levels.iter())
            .map(|((&i, &b), lv)| {
                let lvl = match &lv[i] {
                    Value::Missing => "*".to_string(),
                    v => v.to_text(),
                };
                format!("{} = {}", ws.display_name(b), lvl)
            })
            .collect();
        let mut part = Worksheet::new(format!("{}({})", ws.name, desc.join(", ")));
        part.description = ws.description.clone();
        for c in &ws.columns {
            let mut nc = c.clone();
            nc.data = c.data.take(&rows);
            nc.data.trim_trailing_missing();
            part.columns.push(nc);
        }
        part.constants = ws.constants.clone();
        out.push(part);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{Column, ColumnData};

    #[test]
    fn split_by_text() {
        let ws = Worksheet::with_columns(
            "Data",
            vec![Column::text("G", vec!["b", "a", "b", ""]), Column::numeric("Y", vec![1.0, 2.0, 3.0, 4.0])],
        )
        .unwrap();
        let parts = split_worksheet(&ws, &[0], false).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "Data(G = a)");
        assert_eq!(parts[1].columns[1].data, ColumnData::Numeric(vec![1.0, 3.0]));
        let parts = split_worksheet(&ws, &[0], true).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2].name, "Data(G = *)");
    }
}
