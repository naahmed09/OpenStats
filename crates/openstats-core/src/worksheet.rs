//! Worksheets: positional columns (C1, C2, …), stored constants (K1, …) and
//! stored matrices (M1, …).

use crate::column::{Column, ColumnData, ColumnKind};
use crate::error::{CoreError, Result};
use crate::refs::{parse_ref, split_ref_list, Ref};
use crate::value::Value;
use std::collections::BTreeMap;

/// Maximum column-name length (Minitab limit).
pub const MAX_NAME_LEN: usize = 31;

/// Dense row-major matrix stored in a worksheet. Numerical work converts this
/// to the `openstats-math` matrix type; here it is plain storage.
#[derive(Debug, Clone)]
pub struct StoredMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl PartialEq for StoredMatrix {
    fn eq(&self, o: &Self) -> bool {
        self.rows == o.rows
            && self.cols == o.cols
            && self.data.iter().zip(&o.data).all(|(&a, &b)| crate::column::f64_eq(a, b))
    }
}

impl StoredMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(CoreError::LengthMismatch { expected: rows * cols, found: data.len() });
        }
        Ok(StoredMatrix { rows, cols, data })
    }
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
}

/// One worksheet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Worksheet {
    pub name: String,
    pub description: String,
    pub columns: Vec<Column>,
    pub constants: BTreeMap<usize, Value>,
    pub matrices: BTreeMap<usize, StoredMatrix>,
}

impl Worksheet {
    pub fn new(name: impl Into<String>) -> Self {
        Worksheet { name: name.into(), ..Default::default() }
    }

    /// Build a worksheet from columns, validating names.
    pub fn with_columns(name: impl Into<String>, columns: Vec<Column>) -> Result<Self> {
        let mut ws = Worksheet::new(name);
        for c in columns {
            ws.add_column(c)?;
        }
        Ok(ws)
    }

    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Number of rows = length of the longest column.
    pub fn n_rows(&self) -> usize {
        self.columns.iter().map(|c| c.len()).max().unwrap_or(0)
    }

    pub fn column(&self, i: usize) -> Result<&Column> {
        let len = self.columns.len();
        self.columns.get(i).ok_or(CoreError::IndexOutOfRange { what: "column", index: i, len })
    }

    pub fn column_mut(&mut self, i: usize) -> Result<&mut Column> {
        let len = self.columns.len();
        self.columns.get_mut(i).ok_or(CoreError::IndexOutOfRange { what: "column", index: i, len })
    }

    /// Header label: `C3`, `C3-T` (text) or `C3-D` (date/time).
    pub fn column_label(&self, i: usize) -> String {
        match self.columns.get(i) {
            Some(c) if c.kind() != ColumnKind::Numeric => format!("C{}-{}", i + 1, c.kind().tag()),
            _ => format!("C{}", i + 1),
        }
    }

    /// Display name: the column name, or its `C#` label if unnamed.
    pub fn display_name(&self, i: usize) -> String {
        match self.columns.get(i) {
            Some(c) if !c.name().is_empty() => c.name().to_string(),
            _ => format!("C{}", i + 1),
        }
    }

    /// Case-insensitive lookup by name.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| !c.name().is_empty() && c.name().eq_ignore_ascii_case(name))
    }

    /// Validate a candidate name for column `exclude` (pass `None` for a new column).
    pub fn validate_name(&self, name: &str, exclude: Option<usize>) -> Result<()> {
        if name.is_empty() {
            return Ok(()); // unnamed columns are allowed
        }
        if name.chars().count() > MAX_NAME_LEN {
            return Err(CoreError::invalid(format!("column name {name:?} exceeds {MAX_NAME_LEN} characters")));
        }
        if name.trim() != name || name.starts_with('\'') || name.starts_with('*') || name.contains('#') {
            return Err(CoreError::invalid(format!(
                "column name {name:?} may not start/end with spaces, start with ' or *, or contain #"
            )));
        }
        if matches!(parse_ref(name), Ok(Ref::Column(_) | Ref::ColumnRange(..) | Ref::Constant(_) | Ref::Matrix(_))) {
            return Err(CoreError::invalid(format!("column name {name:?} looks like a reference")));
        }
        if let Some(j) = self.find(name) {
            if Some(j) != exclude {
                return Err(CoreError::Duplicate(name.to_string()));
            }
        }
        Ok(())
    }

    /// Append a column, returning its index.
    pub fn add_column(&mut self, col: Column) -> Result<usize> {
        self.validate_name(col.name(), None)?;
        self.columns.push(col);
        Ok(self.columns.len() - 1)
    }

    /// Ensure at least `n` columns exist, padding with empty numeric columns.
    pub fn ensure_columns(&mut self, n: usize) {
        while self.columns.len() < n {
            self.columns.push(Column::new("", ColumnData::empty(ColumnKind::Numeric)));
        }
    }

    /// Resolve a single column reference (`C3`, name, `'quoted name'`).
    pub fn resolve(&self, r: &str) -> Result<usize> {
        match parse_ref(r)? {
            Ref::Column(n) => {
                if n <= self.columns.len() {
                    Ok(n - 1)
                } else {
                    Err(CoreError::NotFound(format!("column C{n}")))
                }
            }
            Ref::Name(name) => self.find(&name).ok_or_else(|| CoreError::NotFound(format!("column {name:?}"))),
            other => Err(CoreError::invalid(format!("{other:?} is not a single column"))),
        }
    }

    /// Resolve a list of column references, expanding ranges.
    pub fn resolve_list(&self, s: &str) -> Result<Vec<usize>> {
        let mut out = Vec::new();
        for tok in split_ref_list(s)? {
            match parse_ref(&tok)? {
                Ref::ColumnRange(a, b) => {
                    let (lo, hi) = (a.min(b), a.max(b));
                    if hi > self.columns.len() {
                        return Err(CoreError::NotFound(format!("column C{hi}")));
                    }
                    if a <= b {
                        out.extend((lo - 1)..hi);
                    } else {
                        out.extend(((lo - 1)..hi).rev());
                    }
                }
                _ => out.push(self.resolve(&tok)?),
            }
        }
        Ok(out)
    }

    /// Column data by reference.
    pub fn data(&self, r: &str) -> Result<&ColumnData> {
        Ok(&self.columns[self.resolve(r)?].data)
    }

    /// Cell value (Missing beyond the column end).
    pub fn cell(&self, col: usize, row: usize) -> Result<Value> {
        Ok(self.column(col)?.get(row))
    }

    pub fn constant(&self, k: usize) -> Option<&Value> {
        self.constants.get(&k)
    }

    pub fn matrix(&self, m: usize) -> Option<&StoredMatrix> {
        self.matrices.get(&m)
    }

    /// First column index after the last non-empty column (where Minitab
    /// places new output by default).
    pub fn next_free_column(&self) -> usize {
        let mut n = self.columns.len();
        while n > 0 && self.columns[n - 1].is_empty() && self.columns[n - 1].name().is_empty() {
            n -= 1;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet() -> Worksheet {
        Worksheet::with_columns(
            "Data",
            vec![
                Column::numeric("Weight", vec![1.0, 2.0, 3.0]),
                Column::text("Batch No", vec!["A", "B"]),
                Column::numeric("", vec![]),
                Column::numeric("y", vec![5.0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn resolution() {
        let ws = sheet();
        assert_eq!(ws.n_rows(), 3);
        assert_eq!(ws.resolve("weight").unwrap(), 0);
        assert_eq!(ws.resolve("'Batch No'").unwrap(), 1);
        assert_eq!(ws.resolve("C4").unwrap(), 3);
        assert!(ws.resolve("C9").is_err());
        assert_eq!(ws.resolve_list("C1-C3 y").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(ws.resolve_list("C3-C1").unwrap(), vec![2, 1, 0]);
        assert_eq!(ws.column_label(1), "C2-T");
        assert_eq!(ws.display_name(2), "C3");
        assert_eq!(ws.next_free_column(), 4);
    }

    #[test]
    fn name_validation() {
        let mut ws = sheet();
        assert!(matches!(ws.add_column(Column::numeric("WEIGHT", vec![])), Err(CoreError::Duplicate(_))));
        assert!(ws.add_column(Column::numeric("C7", vec![])).is_err());
        assert!(ws.add_column(Column::numeric(" x", vec![])).is_err());
        assert!(ws.add_column(Column::numeric("a#b", vec![])).is_err());
        assert!(ws.add_column(Column::numeric("x".repeat(32), vec![])).is_err());
        assert!(ws.add_column(Column::numeric("", vec![])).is_ok());
        assert!(ws.validate_name("Weight", Some(0)).is_ok());
    }

    #[test]
    fn matrices() {
        assert!(StoredMatrix::new(2, 2, vec![1.0; 3]).is_err());
        let m = StoredMatrix::new(2, 3, (0..6).map(|x| x as f64).collect()).unwrap();
        assert_eq!(m.get(1, 2), 5.0);
    }
}
