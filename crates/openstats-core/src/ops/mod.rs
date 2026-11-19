//! Data-manipulation operations (Minitab's **Data** and part of **Calc** menus).
//!
//! Operations are pure functions from worksheet data to new columns or
//! worksheets. To make the result undoable, turn it into an [`Edit`] with
//! [`store_columns`] (or [`add_worksheets`]) and run it through
//! [`crate::History::execute`].

pub mod change_type;
pub mod concatenate;
pub mod extract;
pub mod merge;
pub mod patterned;
pub mod rank;
pub mod recode;
pub mod sort;
pub mod split;
pub mod stack;
pub mod subset;
pub mod transpose;
pub mod unstack;

use crate::column::Column;
use crate::edit::Edit;
use crate::error::Result;
use crate::workbook::Workbook;
use crate::worksheet::Worksheet;
use std::collections::HashSet;

/// Where an operation's output columns go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// After the last non-empty column of the sheet (Minitab default).
    Append,
    /// Starting at a specific 0-based column index, overwriting existing
    /// columns (names of overwritten columns are replaced).
    At(usize),
    /// Into explicitly listed 0-based column indices (one per output column).
    Columns(Vec<usize>),
}

/// Return `base` or `base_1`, `base_2`, … so that it does not collide with any
/// name in `taken` (case-insensitive). Empty names are left empty.
pub fn unique_name(base: &str, taken: &HashSet<String>) -> String {
    if base.is_empty() || !taken.contains(&base.to_lowercase()) {
        return base.to_string();
    }
    let mut k = 1;
    loop {
        let cand = format!("{base}_{k}");
        if !taken.contains(&cand.to_lowercase()) {
            return cand;
        }
        k += 1;
    }
}

/// Build an [`Edit`] that writes `cols` into worksheet `sheet` at `target`.
/// Output names that would collide with columns that are *not* being
/// overwritten get a `_1`, `_2`, … suffix; names longer than 31 characters are
/// truncated.
pub fn store_columns(wb: &Workbook, sheet: usize, cols: Vec<Column>, target: Target) -> Result<Edit> {
    let ws = wb.sheet(sheet)?;
    let positions: Vec<usize> = match target {
        Target::Append => {
            let s = ws.next_free_column();
            (s..s + cols.len()).collect()
        }
        Target::At(s) => (s..s + cols.len()).collect(),
        Target::Columns(v) => {
            if v.len() != cols.len() {
                return Err(crate::CoreError::LengthMismatch { expected: cols.len(), found: v.len() });
            }
            v
        }
    };
    let overwritten: HashSet<usize> = positions.iter().copied().collect();
    let mut taken: HashSet<String> = ws
        .columns
        .iter()
        .enumerate()
        .filter(|(i, c)| !overwritten.contains(i) && !c.name().is_empty())
        .map(|(_, c)| c.name().to_lowercase())
        .collect();
    let mut n = ws.n_cols();
    let mut edits = Vec::new();
    for (mut col, &pos) in cols.into_iter().zip(positions.iter()) {
        let mut name: String = col.props.name.chars().take(crate::worksheet::MAX_NAME_LEN).collect();
        name = name.trim().to_string();
        name = unique_name(&name, &taken);
        if name.chars().count() > crate::worksheet::MAX_NAME_LEN {
            // Suffix pushed it over the limit: shorten the base and retry.
            let base: String = col.props.name.chars().take(crate::worksheet::MAX_NAME_LEN - 4).collect();
            name = unique_name(base.trim(), &taken);
        }
        if !name.is_empty() {
            taken.insert(name.to_lowercase());
        }
        col.props.name = name;
        while n < pos {
            edits.push(Edit::InsertColumn { sheet, at: n, column: Column::numeric("", vec![]) });
            n += 1;
        }
        if pos < n {
            edits.push(Edit::ReplaceColumn { sheet, col: pos, column: col });
        } else {
            edits.push(Edit::InsertColumn { sheet, at: pos, column: col });
            n += 1;
        }
    }
    // Two-phase naming: clear names of overwritten columns first so swapping
    // names between output columns cannot trip duplicate detection.
    let mut clear = Vec::new();
    for &pos in &positions {
        if pos < ws.n_cols() && !ws.columns[pos].name().is_empty() {
            let mut props = ws.columns[pos].props.clone();
            props.name.clear();
            clear.push(Edit::SetColumnProps { sheet, col: pos, props });
        }
    }
    clear.extend(edits);
    Ok(Edit::Batch(clear))
}

/// Build an [`Edit`] that appends worksheets to the workbook, making their
/// names unique.
pub fn add_worksheets(wb: &Workbook, sheets: Vec<Worksheet>) -> Result<Edit> {
    let mut names: Vec<String> = wb.worksheets.iter().map(|w| w.name.clone()).collect();
    let mut edits = Vec::new();
    let first = wb.worksheets.len();
    for (offset, mut s) in sheets.into_iter().enumerate() {
        let mut cand = s.name.clone();
        let mut k = 2;
        while names.iter().any(|n| n.eq_ignore_ascii_case(&cand)) {
            cand = format!("{} ({k})", s.name);
            k += 1;
        }
        s.name = cand.clone();
        names.push(cand);
        edits.push(Edit::InsertWorksheet { at: first + offset, sheet: Box::new(s) });
    }
    Ok(Edit::Batch(edits))
}

/// Number of rows spanned by a set of columns (longest column).
pub(crate) fn rows_spanned(ws: &Worksheet, cols: &[usize]) -> Result<usize> {
    let mut n = 0;
    for &c in cols {
        n = n.max(ws.column(c)?.len());
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::History;

    #[test]
    fn store_append_and_overwrite() {
        let mut wb = Workbook::new();
        {
            let ws = wb.sheet_mut(0).unwrap();
            ws.add_column(Column::numeric("x", vec![1.0])).unwrap();
            ws.add_column(Column::numeric("y", vec![2.0])).unwrap();
        }
        let mut h = History::default();
        let e =
            store_columns(&wb, 0, vec![Column::numeric("x", vec![3.0]), Column::numeric("", vec![])], Target::Append)
                .unwrap();
        h.execute(&mut wb, "append", e).unwrap();
        let ws = wb.sheet(0).unwrap();
        assert_eq!(ws.n_cols(), 4);
        assert_eq!(ws.columns[2].name(), "x_1");

        // Swap names via overwrite at explicit positions.
        let e = store_columns(
            &wb,
            0,
            vec![Column::numeric("y", vec![]), Column::numeric("x", vec![])],
            Target::Columns(vec![0, 1]),
        )
        .unwrap();
        h.execute(&mut wb, "swap", e).unwrap();
        assert_eq!(wb.sheet(0).unwrap().columns[0].name(), "y");

        let e = store_columns(&wb, 0, vec![Column::numeric("far", vec![])], Target::At(7)).unwrap();
        h.execute(&mut wb, "far", e).unwrap();
        assert_eq!(wb.sheet(0).unwrap().n_cols(), 8);
        h.undo(&mut wb).unwrap();
        h.undo(&mut wb).unwrap();
        h.undo(&mut wb).unwrap();
        assert_eq!(wb.sheet(0).unwrap().n_cols(), 2);
        assert_eq!(wb.sheet(0).unwrap().columns[0].name(), "x");
    }

    #[test]
    fn worksheets_unique() {
        let wb = Workbook::new();
        let e = add_worksheets(&wb, vec![Worksheet::new("Worksheet 1"), Worksheet::new("Worksheet 1")]).unwrap();
        let mut wb2 = wb.clone();
        e.apply(&mut wb2).unwrap();
        let names: Vec<_> = wb2.worksheets.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["Worksheet 1", "Worksheet 1 (2)", "Worksheet 1 (3)"]);
    }
}
