//! Data > Stack.

use crate::column::{Column, ColumnData, ColumnKind};
use crate::error::{CoreError, Result};
use crate::worksheet::Worksheet;

/// How the subscript column identifies the source of each stacked row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subscripts {
    /// No subscript column.
    None,
    /// Text column holding the source column names.
    Names,
    /// Numeric column holding 1, 2, … for the first, second, … source.
    Numbers,
}

/// Common kind for stacking: identical kinds are kept, mixtures become text.
fn common_kind(kinds: impl Iterator<Item = ColumnKind>) -> ColumnKind {
    let mut k: Option<ColumnKind> = None;
    for x in kinds {
        k = match k {
            None => Some(x),
            Some(prev) if prev == x => Some(prev),
            Some(_) => Some(ColumnKind::Text),
        };
    }
    k.unwrap_or(ColumnKind::Numeric)
}

/// Stack several columns into one. Returns `[stacked, subscripts?]`.
/// The stacked column is named `name`.
pub fn stack_columns(ws: &Worksheet, cols: &[usize], subs: Subscripts, name: &str) -> Result<Vec<Column>> {
    let blocks: Vec<Vec<usize>> = cols.iter().map(|&c| vec![c]).collect();
    let mut out = stack_blocks(ws, &blocks, subs, &[name.to_string()])?;
    if let Some(s) = out.get_mut(1) {
        s.props.name = "Subscripts".into();
    }
    Ok(out)
}

/// Stack blocks of columns: block *i* contributes its *j*-th column to output
/// column *j*. All blocks must have the same width. Within a block, the block
/// length is the longest column in it (shorter columns padded with missing).
/// Subscripts use the first column of each block.
pub fn stack_blocks(ws: &Worksheet, blocks: &[Vec<usize>], subs: Subscripts, names: &[String]) -> Result<Vec<Column>> {
    let width = blocks.first().map(|b| b.len()).ok_or_else(|| CoreError::invalid("nothing to stack"))?;
    if width == 0 {
        return Err(CoreError::invalid("empty block"));
    }
    for b in blocks {
        if b.len() != width {
            return Err(CoreError::LengthMismatch { expected: width, found: b.len() });
        }
    }
    let mut outputs: Vec<ColumnData> = Vec::with_capacity(width);
    for j in 0..width {
        let mut kinds = Vec::new();
        for b in blocks {
            kinds.push(ws.column(b[j])?.kind());
        }
        outputs.push(ColumnData::empty(common_kind(kinds.into_iter())));
    }
    let mut sub_names: Vec<String> = Vec::new();
    let mut sub_nums: Vec<f64> = Vec::new();
    for (bi, b) in blocks.iter().enumerate() {
        let block_len = b.iter().map(|&c| ws.columns[c].len()).max().unwrap_or(0);
        for (j, &c) in b.iter().enumerate() {
            let mut d = ws.columns[c].data.convert(outputs[j].kind(), false)?;
            d.resize(block_len);
            outputs[j].extend(&d)?;
        }
        let label = ws.display_name(b[0]);
        for _ in 0..block_len {
            sub_names.push(label.clone());
            sub_nums.push((bi + 1) as f64);
        }
    }
    let mut cols: Vec<Column> = outputs
        .into_iter()
        .enumerate()
        .map(|(j, d)| Column::new(names.get(j).cloned().unwrap_or_default(), d))
        .collect();
    match subs {
        Subscripts::None => {}
        Subscripts::Names => cols.push(Column::text("Subscripts", sub_names)),
        Subscripts::Numbers => cols.push(Column::numeric("Subscripts", sub_nums)),
    }
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    fn ws() -> Worksheet {
        Worksheet::with_columns(
            "s",
            vec![
                Column::numeric("A", vec![1.0, 2.0]),
                Column::numeric("B", vec![3.0]),
                Column::text("C", vec!["x", "y", "z"]),
                Column::numeric("D", vec![10.0, 20.0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn stack_simple() {
        let out = stack_columns(&ws(), &[0, 1], Subscripts::Names, "Data").unwrap();
        assert_eq!(out[0].data, ColumnData::Numeric(vec![1.0, 2.0, 3.0]));
        assert_eq!(out[1].data.as_text().unwrap(), ["A", "A", "B"]);
        let mixed = stack_columns(&ws(), &[0, 2], Subscripts::Numbers, "Data").unwrap();
        assert_eq!(mixed[0].kind(), ColumnKind::Text);
        assert_eq!(mixed[0].get(4), Value::Text("z".into()));
        assert_eq!(mixed[1].data.as_numeric().unwrap(), [1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn stack_block_pairs() {
        let out = stack_blocks(&ws(), &[vec![0, 1], vec![3, 0]], Subscripts::Names, &["P".into(), "Q".into()]).unwrap();
        assert_eq!(out[0].data, ColumnData::Numeric(vec![1.0, 2.0, 10.0, 20.0]));
        assert!(out[1].data.is_missing(1));
        assert_eq!(out[1].get(2), Value::Numeric(1.0));
        assert_eq!(out[2].data.as_text().unwrap(), ["A", "A", "D", "D"]);
        assert!(stack_blocks(&ws(), &[vec![0, 1], vec![2]], Subscripts::None, &[]).is_err());
    }
}
