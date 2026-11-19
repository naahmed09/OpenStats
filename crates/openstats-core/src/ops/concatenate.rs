//! Data > Combine/Concatenate text columns.

use crate::column::Column;
use crate::error::{CoreError, Result};
use crate::worksheet::Worksheet;

/// Concatenate the display text of `cols` row by row, joined by `sep`.
/// Missing pieces contribute an empty string; with `skip_missing` they (and
/// their separator) are omitted. Leading/trailing whitespace of each piece is
/// trimmed when `trim` is set.
pub fn concatenate(
    ws: &Worksheet,
    cols: &[usize],
    sep: &str,
    skip_missing: bool,
    trim: bool,
    name: &str,
) -> Result<Column> {
    if cols.is_empty() {
        return Err(CoreError::invalid("nothing to concatenate"));
    }
    let n = super::rows_spanned(ws, cols)?;
    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        let mut parts: Vec<String> = Vec::with_capacity(cols.len());
        for &c in cols {
            let col = &ws.columns[c];
            if col.data.is_missing(r) {
                if !skip_missing {
                    parts.push(String::new());
                }
                continue;
            }
            let s = col.display(r);
            parts.push(if trim { s.trim().to_string() } else { s });
        }
        out.push(parts.join(sep));
    }
    Ok(Column::text(name, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat() {
        let ws = Worksheet::with_columns(
            "s",
            vec![Column::text("a", vec!["Lot ", "B", ""]), Column::numeric("n", vec![1.0, f64::NAN, 3.0])],
        )
        .unwrap();
        let c = concatenate(&ws, &[0, 1], "-", false, true, "id").unwrap();
        assert_eq!(c.data.as_text().unwrap(), ["Lot-1", "B-", "-3"]);
        let c = concatenate(&ws, &[0, 1], "-", true, false, "id").unwrap();
        assert_eq!(c.data.as_text().unwrap(), ["Lot -1", "B", "3"]);
    }
}
