//! The analysis registry: every menu command that opens a dialog, with the routine that the
//! dialog's values are bound to.

use super::spec::{DialogSpec, Values};
use crate::grid::GridController;
use openstats_core::column::{Column, ColumnData};
use openstats_core::{CoreError, Worksheet};
use openstats_stats::report::Section;
use openstats_stats::StatsError;
use std::fmt;

/// An error shown to the user in the Session window or the dialog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogError(pub String);

impl fmt::Display for DialogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DialogError {}

impl From<CoreError> for DialogError {
    fn from(e: CoreError) -> Self {
        DialogError(e.to_string())
    }
}

impl From<StatsError> for DialogError {
    fn from(e: StatsError) -> Self {
        DialogError(e.to_string())
    }
}

impl From<String> for DialogError {
    fn from(e: String) -> Self {
        DialogError(e)
    }
}

impl From<&str> for DialogError {
    fn from(e: &str) -> Self {
        DialogError(e.to_string())
    }
}

/// Result alias for analyses.
pub type Res<T> = Result<T, DialogError>;

/// What a report-producing analysis returns.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Output {
    /// The Session-window section (tables, text and charts).
    pub section: Section,
    /// Columns to store in the worksheet (after the last used column).
    pub stored: Vec<Column>,
    /// Models to save in the project, as `(name, JSON)`.
    pub models: Vec<(String, String)>,
}

/// How an analysis runs.
#[derive(Debug, Clone, Copy)]
pub enum Runner {
    /// Reads the worksheet and produces output for the Session window.
    Report(fn(&Worksheet, &Values) -> Res<Output>),
    /// Changes the worksheet and returns a one-line message for the Session window.
    Mutate(fn(&mut GridController, &Values) -> Res<String>),
    /// Reads the whole project (worksheet and saved models) and produces output like `Report`.
    Project(fn(&GridController, &Values) -> Res<Output>),
}

/// A dialog plus the routine behind it.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// The dialog.
    pub spec: DialogSpec,
    /// Menu path, e.g. `["Stat", "Basic Statistics", "1-Sample t..."]`.
    pub menu: Vec<&'static str>,
    /// The routine.
    pub runner: Runner,
}

/// All analyses, in menu order.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    items: Vec<Analysis>,
}

impl Registry {
    /// Every analysis OpenStats ships with.
    pub fn standard() -> Registry {
        let mut r = Registry::default();
        super::analyses::register_all(&mut r);
        r
    }

    /// Add an analysis. Panics are avoided: a duplicate id replaces the old entry.
    pub fn add(&mut self, a: Analysis) {
        match self.items.iter().position(|x| x.spec.id == a.spec.id) {
            Some(i) => self.items[i] = a,
            None => self.items.push(a),
        }
    }

    /// The analysis with this id.
    pub fn get(&self, id: &str) -> Option<&Analysis> {
        self.items.iter().find(|a| a.spec.id == id)
    }

    /// All analyses in registration order.
    pub fn all(&self) -> &[Analysis] {
        &self.items
    }

    /// Run the analysis `id` on `grid`, returning the section for the Session window. Report
    /// analyses store their columns in the worksheet.
    pub fn run(&self, id: &str, grid: &mut GridController, values: &Values) -> Res<Section> {
        let a = self.get(id).ok_or_else(|| DialogError(format!("unknown command {id}")))?;
        match a.runner {
            Runner::Report(f) => finish(grid, f(grid.worksheet(), values)?),
            Runner::Project(f) => finish(grid, f(grid, values)?),
            Runner::Mutate(f) => {
                let msg = f(grid, values)?;
                let mut s = Section::new(a.spec.title.trim_end_matches("...").to_string());
                s.text(msg);
                Ok(s)
            }
        }
    }
}

/// Store an analysis's columns and models, and return its section.
fn finish(grid: &mut GridController, out: Output) -> Res<Section> {
    let mut section = out.section;
    if !out.stored.is_empty() {
        let names: Vec<String> = out.stored.iter().map(|c| c.name().to_string()).collect();
        let first = grid.store_columns(out.stored)?;
        let labels: Vec<String> = (0..names.len()).map(|i| grid.worksheet().column_label(first + i)).collect();
        section.text(format!(
            "Stored in the worksheet: {}.",
            labels.iter().zip(&names).map(|(l, n)| format!("{l} ({n})")).collect::<Vec<_>>().join(", ")
        ));
    }
    for (name, json) in out.models {
        grid.store_model(&name, json).map_err(DialogError)?;
        section.text(format!("Model saved in the project as \"{name}\"."));
    }
    Ok(section)
}

// ------------------------------------------------------------------ helpers shared by analyses

/// Column title: its name, or `C3` when it has none.
pub fn col_title(ws: &Worksheet, c: usize) -> String {
    match ws.columns.get(c) {
        Some(col) if !col.name().is_empty() => col.name().to_string(),
        _ => format!("C{}", c + 1),
    }
}

/// Values of a numeric (or date/time) column as `f64`, missing as `NaN`.
pub fn numeric(ws: &Worksheet, c: usize) -> Res<Vec<f64>> {
    let col = ws.column(c)?;
    Ok(match &col.data {
        ColumnData::Numeric(v) => v.clone(),
        ColumnData::Text(_) => return Err(DialogError(format!("{} is a text column", col_title(ws, c)))),
        _ => (0..col.len()).map(|i| col.data.get(i).as_f64()).collect(),
    })
}

/// Values of any column as text labels (missing as the empty string).
pub fn labels(ws: &Worksheet, c: usize) -> Res<Vec<String>> {
    let col = ws.column(c)?;
    Ok(match &col.data {
        ColumnData::Text(v) => v.clone(),
        _ => (0..col.len()).map(|i| if col.data.is_missing(i) { String::new() } else { col.display(i) }).collect(),
    })
}

/// Confidence level as a fraction from a percent field.
pub fn level(v: &Values) -> Res<f64> {
    let l = v.num("level") / 100.0;
    if l > 0.0 && l < 1.0 {
        Ok(l)
    } else {
        Err(DialogError("the confidence level must be between 0 and 100".into()))
    }
}

/// Rows where every listed column is present, and the values in those rows.
pub fn complete_rows(cols: &[&[f64]]) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    let rows: Vec<usize> = (0..n).filter(|&i| cols.iter().all(|c| c.get(i).is_some_and(|x| !x.is_nan()))).collect();
    let data = cols.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect();
    (rows, data)
}

/// A stored numeric column of length `n` holding `values` at `rows` and missing elsewhere.
pub fn expand(name: &str, n: usize, rows: &[usize], values: &[f64]) -> Column {
    let mut v = vec![f64::NAN; n];
    for (&r, &x) in rows.iter().zip(values) {
        v[r] = x;
    }
    Column::numeric(name, v)
}
