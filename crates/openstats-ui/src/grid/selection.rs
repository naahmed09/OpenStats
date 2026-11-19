//! Cells, ranges and multi-range selections.

/// A cell address (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CellRef {
    /// Row.
    pub row: usize,
    /// Column.
    pub col: usize,
}

impl CellRef {
    /// Construct.
    pub const fn new(row: usize, col: usize) -> CellRef {
        CellRef { row, col }
    }
}

/// A rectangular block of cells, inclusive. `usize::MAX` as `r1` / `c1` means "to the end"
/// (a whole-column or whole-row selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellRange {
    /// First row.
    pub r0: usize,
    /// First column.
    pub c0: usize,
    /// Last row (inclusive).
    pub r1: usize,
    /// Last column (inclusive).
    pub c1: usize,
}

impl CellRange {
    /// Range spanning two corners in any order.
    pub fn spanning(a: CellRef, b: CellRef) -> CellRange {
        CellRange { r0: a.row.min(b.row), c0: a.col.min(b.col), r1: a.row.max(b.row), c1: a.col.max(b.col) }
    }

    /// A single cell.
    pub fn cell(c: CellRef) -> CellRange {
        CellRange::spanning(c, c)
    }

    /// Whole columns `c0 ..= c1`.
    pub fn columns(c0: usize, c1: usize) -> CellRange {
        CellRange { r0: 0, c0: c0.min(c1), r1: usize::MAX, c1: c0.max(c1) }
    }

    /// Whole rows `r0 ..= r1`.
    pub fn rows(r0: usize, r1: usize) -> CellRange {
        CellRange { r0: r0.min(r1), c0: 0, r1: r0.max(r1), c1: usize::MAX }
    }

    /// Whether the cell lies inside.
    pub fn contains(&self, c: CellRef) -> bool {
        (self.r0..=self.r1).contains(&c.row) && (self.c0..=self.c1).contains(&c.col)
    }

    /// Whether this range covers entire columns.
    pub fn is_whole_columns(&self) -> bool {
        self.r1 == usize::MAX
    }

    /// Whether this range covers entire rows.
    pub fn is_whole_rows(&self) -> bool {
        self.c1 == usize::MAX
    }

    /// Clamp the open ends to a data extent of `rows` × `cols` cells (at least one each).
    pub fn clamp(&self, rows: usize, cols: usize) -> CellRange {
        CellRange { r0: self.r0, c0: self.c0, r1: self.r1.min(rows.max(1) - 1), c1: self.c1.min(cols.max(1) - 1) }
    }

    /// Number of rows (after clamping `usize::MAX`).
    pub fn n_rows(&self) -> usize {
        self.r1.saturating_sub(self.r0).saturating_add(1)
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.c1.saturating_sub(self.c0).saturating_add(1)
    }

    /// Number of cells (saturating).
    pub fn n_cells(&self) -> usize {
        self.n_rows().saturating_mul(self.n_cols())
    }
}

/// The selection: one or more ranges, the anchor of the range being extended and the active
/// cell (where typing goes).
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    ranges: Vec<CellRange>,
    anchor: CellRef,
    active: CellRef,
}

impl Default for Selection {
    fn default() -> Selection {
        Selection::new()
    }
}

impl Selection {
    /// Cell `(0, 0)` selected.
    pub fn new() -> Selection {
        Selection {
            ranges: vec![CellRange::cell(CellRef::default())],
            anchor: CellRef::default(),
            active: CellRef::default(),
        }
    }

    /// All ranges (the last one is the one being extended).
    pub fn ranges(&self) -> &[CellRange] {
        &self.ranges
    }

    /// The range being extended.
    pub fn current(&self) -> CellRange {
        *self.ranges.last().unwrap_or(&CellRange::cell(self.active))
    }

    /// Active cell.
    pub fn active(&self) -> CellRef {
        self.active
    }

    /// Anchor of the current range.
    pub fn anchor(&self) -> CellRef {
        self.anchor
    }

    /// Select only `c`.
    pub fn select_cell(&mut self, c: CellRef) {
        self.ranges = vec![CellRange::cell(c)];
        self.anchor = c;
        self.active = c;
    }

    /// Extend the current range from its anchor to `c` (the active cell moves to `c`).
    pub fn extend_to(&mut self, c: CellRef) {
        let r = CellRange::spanning(self.anchor, c);
        match self.ranges.last_mut() {
            Some(last) => *last = r,
            None => self.ranges.push(r),
        }
        self.active = c;
    }

    /// Start a new range at `c` while keeping the others (Ctrl+click).
    pub fn add_cell(&mut self, c: CellRef) {
        self.ranges.push(CellRange::cell(c));
        self.anchor = c;
        self.active = c;
    }

    /// Select whole columns; with `extend`, from the anchor column to `c`.
    pub fn select_column(&mut self, c: usize, extend: bool, add: bool) {
        let from = if extend { self.anchor.col } else { c };
        let r = CellRange::columns(from, c);
        if !extend {
            self.anchor = CellRef::new(0, c);
        }
        self.place(r, add);
        self.active = CellRef::new(0, c);
        if !extend {
            self.anchor = CellRef::new(0, c);
        }
    }

    /// Select whole rows; with `extend`, from the anchor row to `r`.
    pub fn select_row(&mut self, r: usize, extend: bool, add: bool) {
        let from = if extend { self.anchor.row } else { r };
        let range = CellRange::rows(from, r);
        if !extend {
            self.anchor = CellRef::new(r, 0);
        }
        self.place(range, add);
        self.active = CellRef::new(r, 0);
    }

    /// Select everything.
    pub fn select_all(&mut self) {
        self.ranges = vec![CellRange { r0: 0, c0: 0, r1: usize::MAX, c1: usize::MAX }];
        self.anchor = CellRef::default();
        self.active = CellRef::default();
    }

    fn place(&mut self, r: CellRange, add: bool) {
        if add {
            self.ranges.push(r);
        } else {
            self.ranges = vec![r];
        }
    }

    /// Whether `c` is selected.
    pub fn contains(&self, c: CellRef) -> bool {
        self.ranges.iter().any(|r| r.contains(c))
    }

    /// Whether the selection is a single cell.
    pub fn is_single(&self) -> bool {
        self.ranges.len() == 1 && self.ranges[0].n_cells() == 1
    }

    /// Whether any selected range touches column `c`.
    pub fn touches_column(&self, c: usize) -> bool {
        self.ranges.iter().any(|r| (r.c0..=r.c1).contains(&c))
    }

    /// Whether any selected range touches row `r`.
    pub fn touches_row(&self, r: usize) -> bool {
        self.ranges.iter().any(|x| (x.r0..=x.r1).contains(&r))
    }

    /// Move the active cell without changing the selection extent logic: select only `c`.
    pub fn move_to(&mut self, c: CellRef) {
        self.select_cell(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        let r = CellRange::spanning(CellRef::new(5, 3), CellRef::new(2, 7));
        assert_eq!((r.r0, r.c0, r.r1, r.c1), (2, 3, 5, 7));
        assert!(r.contains(CellRef::new(3, 4)));
        assert!(!r.contains(CellRef::new(6, 4)));
        assert_eq!(r.n_cells(), 4 * 5);
        let c = CellRange::columns(4, 2);
        assert!(c.is_whole_columns() && !c.is_whole_rows());
        assert_eq!(c.clamp(100, 10), CellRange { r0: 0, c0: 2, r1: 99, c1: 4 });
        assert_eq!(c.clamp(0, 0).r1, 0);
        let w = CellRange::rows(1, 3);
        assert!(w.is_whole_rows());
        assert_eq!(w.n_rows(), 3);
        assert!(CellRange::columns(0, 0).contains(CellRef::new(9_999_999, 0)));
    }

    #[test]
    fn extend_add_and_axes() {
        let mut s = Selection::new();
        s.select_cell(CellRef::new(2, 2));
        s.extend_to(CellRef::new(4, 5));
        assert_eq!(s.current(), CellRange { r0: 2, c0: 2, r1: 4, c1: 5 });
        assert_eq!(s.active(), CellRef::new(4, 5));
        assert_eq!(s.anchor(), CellRef::new(2, 2));
        s.extend_to(CellRef::new(0, 0));
        assert_eq!(s.current(), CellRange { r0: 0, c0: 0, r1: 2, c1: 2 });
        s.add_cell(CellRef::new(10, 10));
        assert_eq!(s.ranges().len(), 2);
        assert!(s.contains(CellRef::new(1, 1)) && s.contains(CellRef::new(10, 10)) && !s.contains(CellRef::new(5, 5)));
        assert!(!s.is_single());
        s.select_cell(CellRef::new(1, 1));
        assert!(s.is_single());
        s.select_column(3, false, false);
        assert!(s.current().is_whole_columns() && s.touches_column(3) && !s.touches_column(4));
        s.select_column(6, true, false);
        assert_eq!((s.current().c0, s.current().c1), (3, 6));
        s.select_column(9, false, true);
        assert_eq!(s.ranges().len(), 2);
        s.select_row(4, false, false);
        s.select_row(1, true, false);
        assert_eq!((s.current().r0, s.current().r1), (1, 4));
        assert!(s.touches_row(2) && !s.touches_row(5));
        s.select_all();
        assert!(s.contains(CellRef::new(123, 456)));
        s.move_to(CellRef::new(3, 3));
        assert!(s.is_single());
    }
}
