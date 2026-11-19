//! Reversible edits.
//!
//! Every mutation of a [`Workbook`] that should be undoable is expressed as an
//! [`Edit`]. Applying an edit returns its exact inverse, so the undo stack
//! simply stores inverses (and redo stores the inverses of inverses). Compound
//! operations are [`Edit::Batch`]es, which are applied atomically: if any step
//! fails, the steps already applied are rolled back before the error is
//! returned, leaving the workbook unchanged.

use crate::column::{Column, ColumnData, ColumnProps};
use crate::error::{CoreError, Result};
use crate::value::Value;
use crate::workbook::Workbook;
use crate::worksheet::{StoredMatrix, Worksheet};

#[derive(Debug, Clone, PartialEq)]
pub enum Edit {
    /// Set one cell (extending the column with missing values if needed).
    SetCell {
        sheet: usize,
        col: usize,
        row: usize,
        value: Value,
    },
    /// Replace a column wholesale (data and properties).
    ReplaceColumn {
        sheet: usize,
        col: usize,
        column: Column,
    },
    /// Insert a column at position `at` (shifting later columns right).
    InsertColumn {
        sheet: usize,
        at: usize,
        column: Column,
    },
    /// Remove the column at `at`.
    RemoveColumn {
        sheet: usize,
        at: usize,
    },
    /// Change a column's name/format/value order/description.
    SetColumnProps {
        sheet: usize,
        col: usize,
        props: ColumnProps,
    },
    /// Insert data segments at row `at` in the listed columns. All-missing
    /// segments are skipped for columns that do not reach row `at`.
    InsertRows {
        sheet: usize,
        at: usize,
        segments: Vec<(usize, ColumnData)>,
    },
    /// Remove `count` rows starting at `at` from the listed columns
    /// (columns that do not reach `at` are unaffected).
    RemoveRows {
        sheet: usize,
        at: usize,
        count: usize,
        cols: Vec<usize>,
    },
    /// Grow (with missing) or truncate a column.
    Resize {
        sheet: usize,
        col: usize,
        len: usize,
    },
    /// Set or clear stored constant `K{k}`.
    SetConstant {
        sheet: usize,
        k: usize,
        value: Option<Value>,
    },
    /// Set or clear stored matrix `M{m}`.
    SetMatrix {
        sheet: usize,
        m: usize,
        matrix: Option<StoredMatrix>,
    },
    InsertWorksheet {
        at: usize,
        sheet: Box<Worksheet>,
    },
    RemoveWorksheet {
        at: usize,
    },
    RenameWorksheet {
        at: usize,
        name: String,
    },
    SetActiveWorksheet {
        at: usize,
    },
    /// Apply several edits atomically, in order.
    Batch(Vec<Edit>),
}

impl Edit {
    /// Convenience: insert `count` blank rows at `at` in every column of a sheet.
    pub fn insert_blank_rows(wb: &Workbook, sheet: usize, at: usize, count: usize) -> Result<Edit> {
        let ws = wb.sheet(sheet)?;
        let segments = ws.columns.iter().enumerate().map(|(i, c)| (i, ColumnData::missing(c.kind(), count))).collect();
        Ok(Edit::InsertRows { sheet, at, segments })
    }

    /// Convenience: remove rows from every column of a sheet.
    pub fn remove_rows_all(wb: &Workbook, sheet: usize, at: usize, count: usize) -> Result<Edit> {
        let n = wb.sheet(sheet)?.n_cols();
        Ok(Edit::RemoveRows { sheet, at, count, cols: (0..n).collect() })
    }

    /// Convenience: rename a column.
    pub fn rename_column(wb: &Workbook, sheet: usize, col: usize, name: &str) -> Result<Edit> {
        let mut props = wb.sheet(sheet)?.column(col)?.props.clone();
        props.name = name.to_string();
        Ok(Edit::SetColumnProps { sheet, col, props })
    }

    /// Store `column` at position `col`, inserting empty columns as needed if
    /// `col` is past the end, otherwise replacing the existing column.
    pub fn store_column(wb: &Workbook, sheet: usize, col: usize, column: Column) -> Result<Edit> {
        let n = wb.sheet(sheet)?.n_cols();
        if col < n {
            Ok(Edit::ReplaceColumn { sheet, col, column })
        } else {
            let mut edits: Vec<Edit> =
                (n..col).map(|at| Edit::InsertColumn { sheet, at, column: Column::numeric("", vec![]) }).collect();
            edits.push(Edit::InsertColumn { sheet, at: col, column });
            Ok(Edit::Batch(edits))
        }
    }

    /// Approximate heap footprint (used by the history memory budget).
    pub fn approx_bytes(&self) -> usize {
        fn col_bytes(c: &ColumnData) -> usize {
            match c {
                ColumnData::Numeric(v) => v.len() * 8,
                ColumnData::DateTime(v) => v.len() * 8,
                ColumnData::Text(v) => v.iter().map(|s| s.len() + 24).sum(),
            }
        }
        let base = std::mem::size_of::<Edit>();
        base + match self {
            Edit::SetCell { value: Value::Text(s), .. } => s.len(),
            Edit::ReplaceColumn { column, .. } | Edit::InsertColumn { column, .. } => col_bytes(&column.data),
            Edit::InsertRows { segments, .. } => segments.iter().map(|(_, s)| col_bytes(s)).sum(),
            Edit::SetMatrix { matrix: Some(m), .. } => m.data.len() * 8,
            Edit::InsertWorksheet { sheet, .. } => sheet.columns.iter().map(|c| col_bytes(&c.data)).sum(),
            Edit::Batch(v) => v.iter().map(Edit::approx_bytes).sum(),
            _ => 0,
        }
    }

    /// Apply the edit and return its inverse.
    pub fn apply(self, wb: &mut Workbook) -> Result<Edit> {
        match self {
            Edit::SetCell { sheet, col, row, value } => {
                let column = wb.sheet_mut(sheet)?.column_mut(col)?;
                let old_len = column.data.len();
                let old = column.data.set(row, &value)?;
                let undo_set = Edit::SetCell { sheet, col, row, value: old };
                if row >= old_len {
                    // The write extended the column; the inverse also shrinks it back.
                    Ok(Edit::Batch(vec![undo_set, Edit::Resize { sheet, col, len: old_len }]))
                } else {
                    Ok(undo_set)
                }
            }
            Edit::ReplaceColumn { sheet, col, column } => {
                let ws = wb.sheet_mut(sheet)?;
                ws.column(col)?;
                ws.validate_name(column.name(), Some(col))?;
                let old = std::mem::replace(&mut ws.columns[col], column);
                Ok(Edit::ReplaceColumn { sheet, col, column: old })
            }
            Edit::InsertColumn { sheet, at, column } => {
                let ws = wb.sheet_mut(sheet)?;
                if at > ws.columns.len() {
                    return Err(CoreError::IndexOutOfRange { what: "column", index: at, len: ws.columns.len() });
                }
                ws.validate_name(column.name(), None)?;
                ws.columns.insert(at, column);
                Ok(Edit::RemoveColumn { sheet, at })
            }
            Edit::RemoveColumn { sheet, at } => {
                let ws = wb.sheet_mut(sheet)?;
                ws.column(at)?;
                let column = ws.columns.remove(at);
                Ok(Edit::InsertColumn { sheet, at, column })
            }
            Edit::SetColumnProps { sheet, col, props } => {
                let ws = wb.sheet_mut(sheet)?;
                ws.column(col)?;
                ws.validate_name(&props.name, Some(col))?;
                let old = std::mem::replace(&mut ws.columns[col].props, props);
                Ok(Edit::SetColumnProps { sheet, col, props: old })
            }
            Edit::InsertRows { sheet, at, segments } => {
                let ws = wb.sheet_mut(sheet)?;
                for (c, seg) in &segments {
                    let col = ws.column(*c)?;
                    if col.kind() != seg.kind() {
                        return Err(CoreError::TypeMismatch { expected: col.kind().name(), found: seg.kind().name() });
                    }
                }
                let mut inverse = Vec::new();
                for (c, seg) in segments {
                    let col = &mut ws.columns[c].data;
                    let len = col.len();
                    if len < at {
                        if seg.missing_count() == seg.len() {
                            continue;
                        }
                        // Padding is required: the inverse must also drop the padding.
                        let n = seg.len();
                        col.insert_data(at, seg)?;
                        inverse.push(Edit::RemoveRows { sheet, at, count: n, cols: vec![c] });
                        inverse.push(Edit::Resize { sheet, col: c, len });
                    } else {
                        let n = seg.len();
                        col.insert_data(at, seg)?;
                        inverse.push(Edit::RemoveRows { sheet, at, count: n, cols: vec![c] });
                    }
                }
                Ok(Edit::Batch(inverse))
            }
            Edit::RemoveRows { sheet, at, count, cols } => {
                let ws = wb.sheet_mut(sheet)?;
                for &c in &cols {
                    ws.column(c)?;
                }
                let mut segments = Vec::new();
                for c in cols {
                    let col = &mut ws.columns[c].data;
                    if col.len() > at {
                        segments.push((c, col.remove_range(at, count)));
                    }
                }
                Ok(Edit::InsertRows { sheet, at, segments })
            }
            Edit::Resize { sheet, col, len } => {
                let data = &mut wb.sheet_mut(sheet)?.column_mut(col)?.data;
                let old = data.len();
                if len < old {
                    let removed = data.remove_range(len, old - len);
                    Ok(Edit::InsertRows { sheet, at: len, segments: vec![(col, removed)] })
                } else {
                    data.resize(len);
                    Ok(Edit::Resize { sheet, col, len: old })
                }
            }
            Edit::SetConstant { sheet, k, value } => {
                if k == 0 {
                    return Err(CoreError::invalid("constants are numbered from K1"));
                }
                let ws = wb.sheet_mut(sheet)?;
                let old = match value {
                    Some(v) => ws.constants.insert(k, v),
                    None => ws.constants.remove(&k),
                };
                Ok(Edit::SetConstant { sheet, k, value: old })
            }
            Edit::SetMatrix { sheet, m, matrix } => {
                if m == 0 {
                    return Err(CoreError::invalid("matrices are numbered from M1"));
                }
                let ws = wb.sheet_mut(sheet)?;
                let old = match matrix {
                    Some(x) => ws.matrices.insert(m, x),
                    None => ws.matrices.remove(&m),
                };
                Ok(Edit::SetMatrix { sheet, m, matrix: old })
            }
            Edit::InsertWorksheet { at, sheet } => {
                if at > wb.worksheets.len() {
                    return Err(CoreError::IndexOutOfRange { what: "worksheet", index: at, len: wb.worksheets.len() });
                }
                if wb.find_sheet(&sheet.name).is_some() {
                    return Err(CoreError::Duplicate(sheet.name.clone()));
                }
                let had_sheets = !wb.worksheets.is_empty();
                wb.worksheets.insert(at, *sheet);
                if had_sheets && at <= wb.active {
                    wb.active += 1;
                }
                Ok(Edit::RemoveWorksheet { at })
            }
            Edit::RemoveWorksheet { at } => {
                wb.sheet(at)?;
                let old_active = wb.active;
                let sheet = wb.worksheets.remove(at);
                if wb.active > at || (wb.active == at && wb.active >= wb.worksheets.len() && wb.active > 0) {
                    wb.active -= 1;
                }
                Ok(Edit::Batch(vec![
                    Edit::InsertWorksheet { at, sheet: Box::new(sheet) },
                    Edit::SetActiveWorksheet { at: old_active },
                ]))
            }
            Edit::RenameWorksheet { at, name } => {
                if name.trim().is_empty() {
                    return Err(CoreError::invalid("worksheet name may not be empty"));
                }
                if let Some(j) = wb.find_sheet(&name) {
                    if j != at {
                        return Err(CoreError::Duplicate(name));
                    }
                }
                let ws = wb.sheet_mut(at)?;
                let old = std::mem::replace(&mut ws.name, name);
                Ok(Edit::RenameWorksheet { at, name: old })
            }
            Edit::SetActiveWorksheet { at } => {
                let old = wb.active;
                wb.set_active(at)?;
                Ok(Edit::SetActiveWorksheet { at: old })
            }
            Edit::Batch(edits) => {
                let mut inverses: Vec<Edit> = Vec::with_capacity(edits.len());
                for e in edits {
                    match e.apply(wb) {
                        Ok(inv) => inverses.push(inv),
                        Err(err) => {
                            while let Some(inv) = inverses.pop() {
                                // Inverses of successfully applied edits cannot fail.
                                inv.apply(wb).expect("rollback of applied edit failed");
                            }
                            return Err(err);
                        }
                    }
                }
                inverses.reverse();
                Ok(Edit::Batch(inverses))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wb() -> Workbook {
        let mut wb = Workbook::new();
        let ws = wb.sheet_mut(0).unwrap();
        ws.add_column(Column::numeric("x", vec![1.0, 2.0, 3.0])).unwrap();
        ws.add_column(Column::text("g", vec!["a", "b"])).unwrap();
        wb
    }

    /// Apply, check, undo, check original, redo, check again.
    fn roundtrip(edit: Edit, check: impl Fn(&Workbook)) {
        let mut w = wb();
        let original = w.clone();
        let inv = edit.apply(&mut w).unwrap();
        check(&w);
        let after = w.clone();
        let redo = inv.apply(&mut w).unwrap();
        assert_eq!(w, original, "undo did not restore original");
        redo.apply(&mut w).unwrap();
        assert_eq!(w, after, "redo did not restore edited state");
    }

    #[test]
    fn set_cell_extending() {
        roundtrip(Edit::SetCell { sheet: 0, col: 0, row: 5, value: Value::Numeric(9.0) }, |w| {
            assert_eq!(w.sheet(0).unwrap().columns[0].len(), 6);
        });
    }

    #[test]
    fn column_edits() {
        roundtrip(Edit::InsertColumn { sheet: 0, at: 1, column: Column::numeric("z", vec![7.0]) }, |w| {
            assert_eq!(w.sheet(0).unwrap().columns[1].name(), "z");
        });
        roundtrip(Edit::RemoveColumn { sheet: 0, at: 0 }, |w| assert_eq!(w.sheet(0).unwrap().n_cols(), 1));
        roundtrip(Edit::ReplaceColumn { sheet: 0, col: 1, column: Column::numeric("g2", vec![]) }, |_| {});
        let w = wb();
        roundtrip(Edit::rename_column(&w, 0, 0, "Weight").unwrap(), |w| {
            assert_eq!(w.sheet(0).unwrap().find("weight"), Some(0));
        });
        roundtrip(Edit::store_column(&w, 0, 4, Column::numeric("far", vec![1.0])).unwrap(), |w| {
            assert_eq!(w.sheet(0).unwrap().n_cols(), 5);
        });
    }

    #[test]
    fn row_edits() {
        let w = wb();
        roundtrip(Edit::insert_blank_rows(&w, 0, 1, 2).unwrap(), |w| {
            let ws = w.sheet(0).unwrap();
            assert_eq!(ws.columns[0].len(), 5);
            assert_eq!(ws.columns[1].len(), 4);
        });
        roundtrip(Edit::remove_rows_all(&w, 0, 0, 2).unwrap(), |w| {
            let ws = w.sheet(0).unwrap();
            assert_eq!(ws.columns[0].data, ColumnData::Numeric(vec![3.0]));
            assert_eq!(ws.columns[1].len(), 0);
        });
        roundtrip(Edit::InsertRows { sheet: 0, at: 4, segments: vec![(1, ColumnData::Text(vec!["q".into()]))] }, |w| {
            assert_eq!(w.sheet(0).unwrap().columns[1].len(), 5);
        });
        roundtrip(Edit::Resize { sheet: 0, col: 0, len: 1 }, |w| assert_eq!(w.sheet(0).unwrap().columns[0].len(), 1));
        roundtrip(Edit::Resize { sheet: 0, col: 0, len: 10 }, |_| {});
    }

    #[test]
    fn sheet_edits() {
        roundtrip(Edit::InsertWorksheet { at: 0, sheet: Box::new(Worksheet::new("New")) }, |w| {
            assert_eq!(w.worksheets[0].name, "New");
            assert_eq!(w.active, 1);
        });
        roundtrip(Edit::RenameWorksheet { at: 0, name: "Main".into() }, |_| {});
        roundtrip(Edit::SetConstant { sheet: 0, k: 1, value: Some(Value::Numeric(2.0)) }, |_| {});
        roundtrip(
            Edit::SetMatrix { sheet: 0, m: 2, matrix: Some(StoredMatrix::new(1, 1, vec![1.0]).unwrap()) },
            |_| {},
        );
        let mut w = wb();
        Edit::InsertWorksheet { at: 1, sheet: Box::new(Worksheet::new("Two")) }.apply(&mut w).unwrap();
        w.active = 1;
        let original = w.clone();
        let inv = Edit::RemoveWorksheet { at: 1 }.apply(&mut w).unwrap();
        assert_eq!(w.active, 0);
        inv.apply(&mut w).unwrap();
        assert_eq!(w, original);
    }

    #[test]
    fn batch_is_atomic() {
        let mut w = wb();
        let original = w.clone();
        let bad = Edit::Batch(vec![
            Edit::SetCell { sheet: 0, col: 0, row: 0, value: Value::Numeric(100.0) },
            Edit::InsertColumn { sheet: 0, at: 0, column: Column::numeric("y", vec![]) },
            Edit::SetCell { sheet: 0, col: 2, row: 0, value: Value::Numeric(1.0) }, // text col: "1" ok
            Edit::RemoveColumn { sheet: 0, at: 99 },
        ]);
        assert!(bad.apply(&mut w).is_err());
        assert_eq!(w, original);
        let dup = Edit::InsertColumn { sheet: 0, at: 0, column: Column::numeric("X", vec![]) };
        assert!(matches!(dup.apply(&mut w), Err(CoreError::Duplicate(_))));
    }
}
