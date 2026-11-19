//! Workbooks (Minitab "projects"): an ordered set of worksheets.

use crate::error::{CoreError, Result};
use crate::worksheet::Worksheet;
use std::collections::BTreeMap;

/// A project containing one or more worksheets.
#[derive(Debug, Clone, PartialEq)]
pub struct Workbook {
    pub title: String,
    pub worksheets: Vec<Worksheet>,
    /// Index of the active worksheet.
    pub active: usize,
    /// Free-form metadata (author, created, application version, …).
    pub metadata: BTreeMap<String, String>,
}

impl Default for Workbook {
    fn default() -> Self {
        Workbook::new()
    }
}

impl Workbook {
    /// A new project with one empty worksheet named `Worksheet 1`.
    pub fn new() -> Self {
        Workbook {
            title: "Untitled".into(),
            worksheets: vec![Worksheet::new("Worksheet 1")],
            active: 0,
            metadata: BTreeMap::new(),
        }
    }

    /// A project with no worksheets (used by loaders).
    pub fn empty() -> Self {
        Workbook { title: "Untitled".into(), worksheets: Vec::new(), active: 0, metadata: BTreeMap::new() }
    }

    pub fn sheet(&self, i: usize) -> Result<&Worksheet> {
        let len = self.worksheets.len();
        self.worksheets.get(i).ok_or(CoreError::IndexOutOfRange { what: "worksheet", index: i, len })
    }

    pub fn sheet_mut(&mut self, i: usize) -> Result<&mut Worksheet> {
        let len = self.worksheets.len();
        self.worksheets.get_mut(i).ok_or(CoreError::IndexOutOfRange { what: "worksheet", index: i, len })
    }

    pub fn active_sheet(&self) -> Result<&Worksheet> {
        self.sheet(self.active)
    }

    pub fn active_sheet_mut(&mut self) -> Result<&mut Worksheet> {
        let a = self.active;
        self.sheet_mut(a)
    }

    /// Case-insensitive lookup by worksheet name.
    pub fn find_sheet(&self, name: &str) -> Option<usize> {
        self.worksheets.iter().position(|w| w.name.eq_ignore_ascii_case(name))
    }

    /// A worksheet name not yet in use: `Worksheet N`.
    pub fn next_sheet_name(&self) -> String {
        let mut n = self.worksheets.len() + 1;
        loop {
            let cand = format!("Worksheet {n}");
            if self.find_sheet(&cand).is_none() {
                return cand;
            }
            n += 1;
        }
    }

    /// Make a worksheet name unique by appending ` (2)`, ` (3)`, ….
    pub fn unique_sheet_name(&self, base: &str) -> String {
        if self.find_sheet(base).is_none() {
            return base.to_string();
        }
        let mut k = 2;
        loop {
            let cand = format!("{base} ({k})");
            if self.find_sheet(&cand).is_none() {
                return cand;
            }
            k += 1;
        }
    }

    pub fn set_active(&mut self, i: usize) -> Result<()> {
        self.sheet(i)?;
        self.active = i;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naming() {
        let mut wb = Workbook::new();
        assert_eq!(wb.next_sheet_name(), "Worksheet 2");
        wb.worksheets.push(Worksheet::new("Data"));
        assert_eq!(wb.unique_sheet_name("data"), "data (2)");
        assert_eq!(wb.find_sheet("DATA"), Some(1));
        assert!(wb.set_active(5).is_err());
        wb.set_active(1).unwrap();
        assert_eq!(wb.active_sheet().unwrap().name, "Data");
    }
}
