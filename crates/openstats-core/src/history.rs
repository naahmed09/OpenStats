//! Undo/redo stack with transaction grouping, depth and memory limits, and
//! save-point tracking.

use crate::edit::Edit;
use crate::error::{CoreError, Result};
use crate::workbook::Workbook;

#[derive(Debug, Clone)]
struct Entry {
    id: u64,
    label: String,
    inverse: Edit,
    bytes: usize,
}

#[derive(Debug, Clone)]
struct Group {
    label: String,
    /// Inverses in application order.
    inverses: Vec<Edit>,
    depth: usize,
}

/// The undo/redo history of a workbook.
#[derive(Debug, Clone)]
pub struct History {
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    group: Option<Group>,
    max_entries: usize,
    max_bytes: usize,
    next_id: u64,
    saved_id: Option<u64>,
    /// Id of the state at the bottom of the undo stack (the last trimmed entry).
    base_id: u64,
}

impl Default for History {
    fn default() -> Self {
        History::new(1000, 512 * 1024 * 1024)
    }
}

impl History {
    /// `max_entries` undo steps, `max_bytes` approximate memory budget.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        History {
            undo: Vec::new(),
            redo: Vec::new(),
            group: None,
            max_entries: max_entries.max(1),
            max_bytes,
            next_id: 1,
            saved_id: Some(0),
            base_id: 0,
        }
    }

    /// Change the limits: at most `max_entries` undo steps and about `max_bytes` of memory. Older steps beyond
    /// the new limits are dropped at once.
    pub fn set_limits(&mut self, max_entries: usize, max_bytes: usize) {
        self.max_entries = max_entries.max(1);
        self.max_bytes = max_bytes;
        self.enforce_limits();
    }

    /// The current `(max_entries, max_bytes)`.
    pub fn limits(&self) -> (usize, usize) {
        (self.max_entries, self.max_bytes)
    }

    /// Apply `edit` to `wb` and record it under `label`.
    pub fn execute(&mut self, wb: &mut Workbook, label: impl Into<String>, edit: Edit) -> Result<()> {
        let inverse = edit.apply(wb)?;
        if let Some(g) = self.group.as_mut() {
            g.inverses.push(inverse);
            return Ok(());
        }
        self.push_undo(label.into(), inverse);
        Ok(())
    }

    fn push_undo(&mut self, label: String, inverse: Edit) {
        let bytes = inverse.approx_bytes();
        let id = self.next_id;
        self.next_id += 1;
        self.undo.push(Entry { id, label, inverse, bytes });
        self.redo.clear();
        self.enforce_limits();
    }

    fn enforce_limits(&mut self) {
        while self.undo.len() > self.max_entries {
            self.base_id = self.undo.remove(0).id;
        }
        let mut total: usize = self.undo.iter().map(|e| e.bytes).sum();
        // Always keep the most recent step even if it alone exceeds the budget.
        while total > self.max_bytes && self.undo.len() > 1 {
            let e = self.undo.remove(0);
            self.base_id = e.id;
            total -= e.bytes;
        }
    }

    /// Start a transaction: subsequent `execute` calls become a single undo
    /// step. Transactions nest; only the outermost label is kept.
    pub fn begin_group(&mut self, label: impl Into<String>) {
        match self.group.as_mut() {
            Some(g) => g.depth += 1,
            None => self.group = Some(Group { label: label.into(), inverses: Vec::new(), depth: 1 }),
        }
    }

    /// Finish the current transaction.
    pub fn end_group(&mut self) -> Result<()> {
        let g = self.group.as_mut().ok_or_else(|| CoreError::invalid("end_group without begin_group"))?;
        g.depth -= 1;
        if g.depth > 0 {
            return Ok(());
        }
        let g = self.group.take().expect("checked");
        if !g.inverses.is_empty() {
            let mut inv = g.inverses;
            inv.reverse();
            self.push_undo(g.label, Edit::Batch(inv));
        }
        Ok(())
    }

    /// Abort the current transaction, rolling back everything executed in it.
    pub fn cancel_group(&mut self, wb: &mut Workbook) -> Result<()> {
        let g = self.group.take().ok_or_else(|| CoreError::invalid("cancel_group without begin_group"))?;
        for inv in g.inverses.into_iter().rev() {
            inv.apply(wb)?;
        }
        Ok(())
    }

    pub fn in_group(&self) -> bool {
        self.group.is_some()
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() && self.group.is_none()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty() && self.group.is_none()
    }

    /// Label of the step that `undo` would revert.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|e| e.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|e| e.label.as_str())
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Revert the most recent step; returns its label.
    pub fn undo(&mut self, wb: &mut Workbook) -> Result<String> {
        if self.group.is_some() {
            return Err(CoreError::invalid("cannot undo inside an open transaction"));
        }
        let e = self.undo.pop().ok_or(CoreError::NothingToUndo)?;
        match e.inverse.clone().apply(wb) {
            Ok(redo_inv) => {
                let label = e.label.clone();
                let bytes = redo_inv.approx_bytes();
                self.redo.push(Entry { id: e.id, label: e.label, inverse: redo_inv, bytes });
                Ok(label)
            }
            Err(err) => {
                self.undo.push(e);
                Err(err)
            }
        }
    }

    /// Re-apply the most recently undone step; returns its label.
    pub fn redo(&mut self, wb: &mut Workbook) -> Result<String> {
        if self.group.is_some() {
            return Err(CoreError::invalid("cannot redo inside an open transaction"));
        }
        let e = self.redo.pop().ok_or(CoreError::NothingToRedo)?;
        match e.inverse.clone().apply(wb) {
            Ok(undo_inv) => {
                let label = e.label.clone();
                let bytes = undo_inv.approx_bytes();
                self.undo.push(Entry { id: e.id, label: e.label, inverse: undo_inv, bytes });
                Ok(label)
            }
            Err(err) => {
                self.redo.push(e);
                Err(err)
            }
        }
    }

    /// Forget all history (e.g. after loading a file).
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.group = None;
        self.saved_id = Some(0);
        self.base_id = 0;
    }

    fn current_id(&self) -> u64 {
        self.undo.last().map_or(self.base_id, |e| e.id)
    }

    /// Record that the current state has been saved.
    pub fn mark_saved(&mut self) {
        self.saved_id = Some(self.current_id());
    }

    /// `true` if the workbook differs from the last saved state.
    pub fn is_dirty(&self) -> bool {
        if self.group.as_ref().is_some_and(|g| !g.inverses.is_empty()) {
            return true;
        }
        self.saved_id != Some(self.current_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::value::Value;

    fn setup() -> (Workbook, History) {
        let mut wb = Workbook::new();
        wb.sheet_mut(0).unwrap().add_column(Column::numeric("x", vec![1.0, 2.0])).unwrap();
        (wb, History::default())
    }

    fn set(row: usize, v: f64) -> Edit {
        Edit::SetCell { sheet: 0, col: 0, row, value: Value::Numeric(v) }
    }

    #[test]
    fn undo_redo_cycle() {
        let (mut wb, mut h) = setup();
        let s0 = wb.clone();
        h.execute(&mut wb, "a", set(0, 10.0)).unwrap();
        let s1 = wb.clone();
        h.execute(&mut wb, "b", set(5, 20.0)).unwrap();
        let s2 = wb.clone();
        assert_eq!(h.undo_label(), Some("b"));
        assert_eq!(h.undo(&mut wb).unwrap(), "b");
        assert_eq!(wb, s1);
        assert_eq!(h.undo(&mut wb).unwrap(), "a");
        assert_eq!(wb, s0);
        assert!(matches!(h.undo(&mut wb), Err(CoreError::NothingToUndo)));
        h.redo(&mut wb).unwrap();
        h.redo(&mut wb).unwrap();
        assert_eq!(wb, s2);
        h.undo(&mut wb).unwrap();
        h.execute(&mut wb, "c", set(1, 3.0)).unwrap();
        assert!(!h.can_redo(), "new edit clears redo");
    }

    #[test]
    fn groups() {
        let (mut wb, mut h) = setup();
        let s0 = wb.clone();
        h.begin_group("paste");
        h.execute(&mut wb, "", set(0, 1.5)).unwrap();
        h.begin_group("inner");
        h.execute(&mut wb, "", set(1, 2.5)).unwrap();
        h.end_group().unwrap();
        assert!(!h.can_undo());
        h.end_group().unwrap();
        assert_eq!(h.undo_len(), 1);
        assert_eq!(h.undo_label(), Some("paste"));
        h.undo(&mut wb).unwrap();
        assert_eq!(wb, s0);

        h.begin_group("aborted");
        h.execute(&mut wb, "", set(0, 99.0)).unwrap();
        h.cancel_group(&mut wb).unwrap();
        assert_eq!(wb, s0);
        assert!(h.end_group().is_err());
    }

    #[test]
    fn limits_and_dirty() {
        let (mut wb, _) = setup();
        let mut h = History::new(3, usize::MAX);
        assert!(!h.is_dirty());
        for i in 0..5 {
            h.execute(&mut wb, format!("e{i}"), set(0, i as f64)).unwrap();
        }
        assert_eq!(h.undo_len(), 3);
        assert!(h.is_dirty());
        h.mark_saved();
        assert!(!h.is_dirty());
        h.undo(&mut wb).unwrap();
        assert!(h.is_dirty());
        h.redo(&mut wb).unwrap();
        assert!(!h.is_dirty());

        let mut tiny = History::new(100, 1);
        tiny.execute(&mut wb, "a", set(0, 1.0)).unwrap();
        tiny.execute(&mut wb, "b", set(0, 2.0)).unwrap();
        assert_eq!(tiny.undo_len(), 1);
    }

    #[test]
    fn failed_edit_not_recorded() {
        let (mut wb, mut h) = setup();
        assert!(h.execute(&mut wb, "bad", Edit::RemoveColumn { sheet: 0, at: 7 }).is_err());
        assert_eq!(h.undo_len(), 0);
    }
}
