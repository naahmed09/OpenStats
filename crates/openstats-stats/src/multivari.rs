//! Multi-vari analysis: how a response varies across up to four (nested or crossed) factors.
//!
//! The cells are the combinations of factor levels that occur in the data, ordered with the first
//! factor varying slowest. For every cell the individual values, the mean and the standard
//! deviation are kept, and for every factor the means of its levels, so the report shows which
//! factor moves the response most (the factor with the largest range of level means).

use crate::error::{invalid, Result, StatsError};
use crate::report::{Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Section, Series, Table};
use std::collections::BTreeMap;

/// One combination of factor levels.
#[derive(Debug, Clone, PartialEq)]
pub struct MvCell {
    /// Level label of each factor.
    pub labels: Vec<String>,
    /// Observations in the cell.
    pub values: Vec<f64>,
    /// Cell mean.
    pub mean: f64,
    /// Cell standard deviation (`None` for a single observation).
    pub sd: Option<f64>,
}

/// Mean of one level of one factor.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelMean {
    /// Level label.
    pub level: String,
    /// Observations at the level.
    pub n: usize,
    /// Mean at the level.
    pub mean: f64,
}

/// Result of a multi-vari analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiVari {
    /// Factor names.
    pub names: Vec<String>,
    /// Cells in display order.
    pub cells: Vec<MvCell>,
    /// For each factor, its level means in level order.
    pub level_means: Vec<Vec<LevelMean>>,
    /// Mean of all observations.
    pub overall_mean: f64,
}

/// Analyse `response` against 1–4 factors. `factors[j][i]` is the level of row `i` for factor `j`
/// (an empty label is a missing level), and `levels[j]` gives that factor's levels in display
/// order. Rows with a missing response or level are dropped.
pub fn multi_vari(
    response: &[f64],
    factors: &[Vec<String>],
    names: &[String],
    levels: &[Vec<String>],
) -> Result<MultiVari> {
    if factors.is_empty() || factors.len() > 4 {
        return invalid("a multi-vari chart takes between 1 and 4 factors");
    }
    if names.len() != factors.len() || levels.len() != factors.len() {
        return invalid("each factor needs a name and a level list");
    }
    if factors.iter().any(|f| f.len() != response.len()) {
        return invalid("the response and every factor must have the same length");
    }
    if response.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let mut groups: BTreeMap<Vec<usize>, Vec<f64>> = BTreeMap::new();
    let mut used = 0usize;
    'rows: for (i, &y) in response.iter().enumerate() {
        if y.is_nan() {
            continue;
        }
        let mut key = Vec::with_capacity(factors.len());
        for (j, f) in factors.iter().enumerate() {
            match levels[j].iter().position(|l| *l == f[i]) {
                Some(k) => key.push(k),
                None => continue 'rows,
            }
        }
        groups.entry(key).or_default().push(y);
        used += 1;
    }
    if used < 2 {
        return Err(StatsError::InsufficientData { what: "multi-vari analysis", needed: 2, got: used });
    }
    let mut cells = Vec::new();
    for (key, values) in &groups {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sd = if values.len() > 1 {
            Some((values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt())
        } else {
            None
        };
        let labels = key.iter().enumerate().map(|(j, &k)| levels[j][k].clone()).collect();
        cells.push(MvCell { labels, values: values.clone(), mean, sd });
    }
    let total: f64 = cells.iter().flat_map(|c| c.values.iter()).sum();
    let overall_mean = total / used as f64;
    let mut level_means = Vec::new();
    for j in 0..factors.len() {
        let mut v = Vec::new();
        for lev in &levels[j] {
            let vals: Vec<f64> =
                cells.iter().filter(|c| c.labels[j] == *lev).flat_map(|c| c.values.iter().copied()).collect();
            if !vals.is_empty() {
                v.push(LevelMean {
                    level: lev.clone(),
                    n: vals.len(),
                    mean: vals.iter().sum::<f64>() / vals.len() as f64,
                });
            }
        }
        level_means.push(v);
    }
    Ok(MultiVari { names: names.to_vec(), cells, level_means, overall_mean })
}

impl MultiVari {
    /// Range of the level means of factor `j` (largest minus smallest).
    pub fn range_of_means(&self, j: usize) -> f64 {
        let m = &self.level_means[j];
        let hi = m.iter().map(|l| l.mean).fold(f64::NEG_INFINITY, f64::max);
        let lo = m.iter().map(|l| l.mean).fold(f64::INFINITY, f64::min);
        hi - lo
    }

    /// Chart: every observation at its cell position, the cell means joined by a line and the
    /// overall mean as a reference.
    pub fn chart_spec(&self, response: &str) -> ChartSpec {
        let (mut ox, mut oy, mut mx, mut my) = (vec![], vec![], vec![], vec![]);
        for (i, c) in self.cells.iter().enumerate() {
            let pos = (i + 1) as f64;
            for &v in &c.values {
                ox.push(pos);
                oy.push(v);
            }
            mx.push(pos);
            my.push(c.mean);
        }
        ChartSpec {
            kind: ChartKind::Scatter,
            title: format!("Multi-Vari Chart for {response} by {}", self.names.join(", ")),
            x_label: "Cell (factor levels in the table order)".into(),
            y_label: response.into(),
            categories: Vec::new(),
            series: vec![
                Series { name: "Observations".into(), x: ox, y: oy },
                Series { name: "Fit".into(), x: mx, y: my },
            ],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: self.overall_mean, label: "Mean".into() }],
        }
    }

    /// Session section.
    pub fn section(&self, response: &str) -> Section {
        let mut sec = Section::new(format!("Multi-Vari Chart for {response}"));
        let mut cols = vec![Column::new("Cell", ColumnKind::Integer)];
        for n in &self.names {
            cols.push(Column::new(n.clone(), ColumnKind::Text));
        }
        cols.push(Column::new("N", ColumnKind::Integer));
        cols.push(Column::new("Mean", ColumnKind::Significant(6)));
        cols.push(Column::new("StDev", ColumnKind::Significant(6)));
        let mut t = Table::new("Cell Statistics", cols);
        for (i, c) in self.cells.iter().enumerate() {
            let mut row = vec![Cell::Int(i as i64 + 1)];
            row.extend(c.labels.iter().map(Cell::text));
            row.push(Cell::Int(c.values.len() as i64));
            row.push(Cell::Num(c.mean));
            row.push(c.sd.map_or(Cell::Missing, Cell::Num));
            t.row(row);
        }
        sec.table(t);
        let mut lm = Table::new(
            "Means of Factor Levels",
            vec![
                Column::new("Factor", ColumnKind::Text),
                Column::new("Level", ColumnKind::Text),
                Column::new("N", ColumnKind::Integer),
                Column::new("Mean", ColumnKind::Significant(6)),
            ],
        );
        for (j, v) in self.level_means.iter().enumerate() {
            for l in v {
                lm.row(vec![
                    Cell::text(&self.names[j]),
                    Cell::text(&l.level),
                    Cell::Int(l.n as i64),
                    Cell::Num(l.mean),
                ]);
            }
        }
        sec.table(lm);
        let mut rg = Table::new(
            "Range of the Level Means",
            vec![Column::new("Factor", ColumnKind::Text), Column::new("Range", ColumnKind::Significant(6))],
        );
        for (j, n) in self.names.iter().enumerate() {
            rg.row(vec![Cell::text(n), Cell::Num(self.range_of_means(j))]);
        }
        sec.table(rg);
        sec.chart(self.chart_spec(response));
        sec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cells_and_means() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = s(&["a", "a", "a", "a", "b", "b", "b", "b"]);
        let b = s(&["x", "x", "y", "y", "x", "x", "y", "y"]);
        let mv = multi_vari(&y, &[a, b], &s(&["A", "B"]), &[s(&["a", "b"]), s(&["x", "y"])]).unwrap();
        assert_eq!(mv.cells.len(), 4);
        assert_eq!(mv.cells[0].labels, s(&["a", "x"]));
        assert!((mv.cells[0].mean - 1.5).abs() < 1e-15);
        assert!((mv.overall_mean - 4.5).abs() < 1e-15);
        assert!((mv.level_means[0][1].mean - 6.5).abs() < 1e-15);
        assert!((mv.range_of_means(0) - 4.0).abs() < 1e-15);
        assert!((mv.range_of_means(1) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn errors() {
        let y = [1.0, 2.0, 3.0];
        let a = s(&["a", "a", "b"]);
        assert!(multi_vari(&y, &[], &[], &[]).is_err());
        assert!(multi_vari(&y, std::slice::from_ref(&a), &s(&["A"]), &[s(&["z"])]).is_err());
        assert!(multi_vari(&y, &[s(&["a", "b"])], &s(&["A"]), &[s(&["a", "b"])]).is_err());
    }
}
