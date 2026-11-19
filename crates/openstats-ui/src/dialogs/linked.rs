//! Graphs linked to their data (Phase 6a): re-running a graph dialog on changed data and mapping
//! the points of a graph back to worksheet rows (for brushing).

use super::analysis::{complete_rows, numeric, Registry, Res, Runner};
use super::spec::Values;
use openstats_core::{ColumnData, Worksheet};
use openstats_stats::report::{Block, ChartSpec};

/// Graph dialogs the workspace handles itself (they do not draw from worksheet columns).
const WORKSPACE_OWNED: [&str; 5] = ["graph.edit", "graph.update", "graph.format", "graph.annotate", "graph.brush"];

impl Registry {
    /// Whether dialog `id` is a plain graph dialog whose charts can be rebuilt from the worksheet
    /// alone (it stores no columns and saves no models).
    pub fn is_linkable(&self, id: &str) -> bool {
        id.starts_with("graph.")
            && !WORKSPACE_OWNED.contains(&id)
            && matches!(self.get(id).map(|a| a.runner), Some(Runner::Report(_)))
    }

    /// The charts dialog `id` draws for `values` on `ws`, in order; `None` when `id` is not linkable.
    pub fn charts(&self, id: &str, ws: &Worksheet, values: &Values) -> Option<Res<Vec<ChartSpec>>> {
        if !self.is_linkable(id) {
            return None;
        }
        let Some(Runner::Report(f)) = self.get(id).map(|a| a.runner) else { return None };
        Some(f(ws, values).map(|out| {
            out.section
                .blocks
                .into_iter()
                .filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None })
                .collect()
        }))
    }

    /// For each chart of dialog `id`, the worksheet row behind every point of its first series
    /// (`None` when the dialog's points cannot be mapped to rows). Supported: Scatterplot,
    /// Time Series Plot, Marginal Plot and Bubble Plot.
    pub fn brush_rows(&self, id: &str, ws: &Worksheet, v: &Values) -> Option<Vec<Vec<usize>>> {
        if !self.is_linkable(id) {
            return None;
        }
        match id {
            "graph.scatter" | "graph.marginal" => {
                let (y, x) = (v.var("y")?, v.var("x")?);
                let (xd, yd) = (numeric(ws, x).ok()?, numeric(ws, y).ok()?);
                Some(vec![complete_rows(&[&xd, &yd]).0])
            }
            "graph.bubble" => {
                let (y, x, s) = (v.var("y")?, v.var("x")?, v.var("size")?);
                let (xd, yd, sd) = (numeric(ws, x).ok()?, numeric(ws, y).ok()?, numeric(ws, s).ok()?);
                let (rows, d) = complete_rows(&[&xd, &yd, &sd]);
                Some(vec![(0..rows.len()).filter(|&i| d[2][i] > 0.0).map(|i| rows[i]).collect()])
            }
            "graph.timeseries" => {
                let finite: Option<Vec<bool>> = match v.var("time") {
                    Some(t) => {
                        let col = ws.column(t).ok()?;
                        Some(match &col.data {
                            ColumnData::DateTime(d) => {
                                d.iter().map(|&m| m != openstats_core::missing::DATETIME_MISSING).collect()
                            }
                            _ => numeric(ws, t).ok()?.iter().map(|x| x.is_finite()).collect(),
                        })
                    }
                    None => None,
                };
                let mut out = Vec::new();
                for &c in v.vars("vars") {
                    let d = numeric(ws, c).ok()?;
                    let n = finite.as_ref().map_or(d.len(), |f| f.len().min(d.len()));
                    out.push((0..n).filter(|&i| finite.as_ref().is_none_or(|f| f[i])).collect());
                }
                Some(out)
            }
            _ => None,
        }
    }
}
