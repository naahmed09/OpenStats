//! Run chart tests for clustering, mixtures, trends and oscillation.
//!
//! Two sets of runs are counted. *Runs about the centre* are maximal blocks of points on the
//! same side of the centre line (points exactly on it are skipped); too few runs point to
//! clustering and too many to mixtures. *Runs up or down* are maximal blocks of increasing or of
//! decreasing consecutive points (equal neighbours are skipped); too few point to trends and too
//! many to oscillation. P-values use the usual normal approximation of the number of runs.

use crate::error::{invalid, Result, StatsError};
use crate::report::{Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Section, Series, Table};
use openstats_math::special::erf::{norm_cdf, norm_sf};

/// Where the centre line is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Center {
    /// Median of the plotted points.
    Median,
    /// Mean of the plotted points.
    Mean,
    /// A given value.
    Value(f64),
}

/// Statistic plotted for each subgroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupStat {
    /// Subgroup mean.
    Mean,
    /// Subgroup median.
    Median,
}

/// Result of a run chart analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct RunChart {
    /// The plotted points (observations or subgroup statistics).
    pub points: Vec<f64>,
    /// Centre line.
    pub center: f64,
    /// Points above the centre.
    pub above: usize,
    /// Points below the centre.
    pub below: usize,
    /// Points on the centre (not counted in the runs).
    pub on: usize,
    /// Number of runs about the centre.
    pub runs_about: usize,
    /// Expected number of runs about the centre.
    pub expected_about: f64,
    /// Longest run about the centre.
    pub longest_about: usize,
    /// P-value for clustering (small when there are too few runs); `NaN` when undefined.
    pub p_clustering: f64,
    /// P-value for mixtures (small when there are too many runs); `NaN` when undefined.
    pub p_mixtures: f64,
    /// Number of runs up or down.
    pub runs_updown: usize,
    /// Expected number of runs up or down.
    pub expected_updown: f64,
    /// Longest run up or down.
    pub longest_updown: usize,
    /// P-value for trends (small when there are too few runs).
    pub p_trends: f64,
    /// P-value for oscillation (small when there are too many runs).
    pub p_oscillation: f64,
}

/// Subgroup statistics of consecutive blocks of `size` observations (a short last block is used
/// as it is; missing values are dropped inside each block, and a block with none is skipped).
pub fn subgroup_points(data: &[f64], size: usize, stat: SubgroupStat) -> Result<Vec<f64>> {
    if size == 0 {
        return invalid("the subgroup size must be at least 1");
    }
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let mut out = Vec::new();
    for block in data.chunks(size) {
        let mut v: Vec<f64> = block.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.is_empty() {
            continue;
        }
        out.push(match stat {
            SubgroupStat::Mean => v.iter().sum::<f64>() / v.len() as f64,
            SubgroupStat::Median => {
                v.sort_by(|a, b| a.total_cmp(b));
                let m = v.len() / 2;
                if v.len() % 2 == 1 {
                    v[m]
                } else {
                    0.5 * (v[m - 1] + v[m])
                }
            }
        });
    }
    Ok(out)
}

fn normal_p(runs: f64, mean: f64, var: f64) -> (f64, f64) {
    if var <= 0.0 || !var.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let z = (runs - mean) / var.sqrt();
    (norm_cdf(z), norm_sf(z))
}

/// Analyse plotted points (missing values dropped) about the chosen centre.
pub fn run_chart(points: &[f64], center: Center) -> Result<RunChart> {
    if points.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let pts: Vec<f64> = points.iter().copied().filter(|x| !x.is_nan()).collect();
    if pts.len() < 3 {
        return Err(StatsError::InsufficientData { what: "run chart", needed: 3, got: pts.len() });
    }
    let c = match center {
        Center::Value(v) => {
            if !v.is_finite() {
                return invalid("the centre line must be finite");
            }
            v
        }
        Center::Mean => pts.iter().sum::<f64>() / pts.len() as f64,
        Center::Median => {
            let mut s = pts.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            let m = s.len() / 2;
            if s.len() % 2 == 1 {
                s[m]
            } else {
                0.5 * (s[m - 1] + s[m])
            }
        }
    };
    let (mut above, mut below, mut on) = (0usize, 0usize, 0usize);
    let (mut runs_about, mut longest_about, mut cur, mut last_side) = (0usize, 0usize, 0usize, 0i8);
    for &x in &pts {
        let side = if x > c {
            above += 1;
            1
        } else if x < c {
            below += 1;
            -1
        } else {
            on += 1;
            0
        };
        if side == 0 {
            continue;
        }
        if side == last_side {
            cur += 1;
        } else {
            runs_about += 1;
            cur = 1;
            last_side = side;
        }
        longest_about = longest_about.max(cur);
    }
    let (n1, n2) = (above as f64, below as f64);
    let n = n1 + n2;
    let expected_about = if n > 0.0 { 1.0 + 2.0 * n1 * n2 / n } else { f64::NAN };
    let var_about = if n > 1.0 { 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0)) } else { 0.0 };
    let (p_clustering, p_mixtures) = normal_p(runs_about as f64, expected_about, var_about);

    let (mut runs_ud, mut longest_ud, mut cur, mut last_dir) = (0usize, 0usize, 0usize, 0i8);
    for w in pts.windows(2) {
        let d = w[1] - w[0];
        let dir = if d > 0.0 {
            1
        } else if d < 0.0 {
            -1
        } else {
            0
        };
        if dir == 0 {
            continue;
        }
        if dir == last_dir {
            cur += 1;
        } else {
            runs_ud += 1;
            cur = 1;
            last_dir = dir;
        }
        longest_ud = longest_ud.max(cur);
    }
    let m = pts.len() as f64;
    let expected_updown = (2.0 * m - 1.0) / 3.0;
    let var_ud = (16.0 * m - 29.0) / 90.0;
    let (p_trends, p_oscillation) = normal_p(runs_ud as f64, expected_updown, var_ud);
    Ok(RunChart {
        points: pts,
        center: c,
        above,
        below,
        on,
        runs_about,
        expected_about,
        longest_about,
        p_clustering,
        p_mixtures,
        runs_updown: runs_ud,
        expected_updown,
        longest_updown: longest_ud,
        p_trends,
        p_oscillation,
    })
}

fn pcell(p: f64) -> Cell {
    if p.is_nan() {
        Cell::Missing
    } else {
        Cell::Num(p)
    }
}

impl RunChart {
    /// The chart: points joined by a line with the centre line as a reference.
    pub fn chart_spec(&self, title: &str, center_label: &str) -> ChartSpec {
        let x: Vec<f64> = (1..=self.points.len()).map(|i| i as f64).collect();
        ChartSpec {
            kind: ChartKind::Line,
            title: title.into(),
            x_label: "Observation".into(),
            y_label: "Value".into(),
            categories: Vec::new(),
            series: vec![Series { name: "Data".into(), x, y: self.points.clone() }],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: self.center, label: center_label.into() }],
        }
    }

    /// Session section with the two tests and the chart.
    pub fn section(&self, name: &str, center_label: &str) -> Section {
        let mut sec = Section::new(format!("Run Chart of {name}"));
        let mut a = Table::new(
            format!("Runs About {center_label} ({:.6})", self.center),
            vec![
                Column::new("Number of Runs", ColumnKind::Integer),
                Column::new("Expected Runs", ColumnKind::Significant(6)),
                Column::new("Longest Run", ColumnKind::Integer),
                Column::new("Points Above", ColumnKind::Integer),
                Column::new("Points Below", ColumnKind::Integer),
                Column::new("P (Clustering)", ColumnKind::PValue),
                Column::new("P (Mixtures)", ColumnKind::PValue),
            ],
        );
        a.row(vec![
            Cell::Int(self.runs_about as i64),
            Cell::Num(self.expected_about),
            Cell::Int(self.longest_about as i64),
            Cell::Int(self.above as i64),
            Cell::Int(self.below as i64),
            pcell(self.p_clustering),
            pcell(self.p_mixtures),
        ]);
        sec.table(a);
        let mut b = Table::new(
            "Runs Up or Down",
            vec![
                Column::new("Number of Runs", ColumnKind::Integer),
                Column::new("Expected Runs", ColumnKind::Significant(6)),
                Column::new("Longest Run", ColumnKind::Integer),
                Column::new("P (Trends)", ColumnKind::PValue),
                Column::new("P (Oscillation)", ColumnKind::PValue),
            ],
        );
        b.row(vec![
            Cell::Int(self.runs_updown as i64),
            Cell::Num(self.expected_updown),
            Cell::Int(self.longest_updown as i64),
            pcell(self.p_trends),
            pcell(self.p_oscillation),
        ]);
        sec.table(b);
        if self.on > 0 {
            sec.text(format!("{} point(s) fall on the center line and are not counted in the runs about it.", self.on));
        }
        sec.chart(self.chart_spec(&format!("Run Chart of {name}"), center_label));
        sec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating_series_is_all_runs() {
        let x: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
        let r = run_chart(&x, Center::Median).unwrap();
        assert_eq!(r.runs_about, 20);
        assert_eq!(r.longest_about, 1);
        assert_eq!(r.runs_updown, 19);
        assert!(r.p_mixtures < 0.01 && r.p_oscillation < 0.01);
    }

    #[test]
    fn trend_has_one_run() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = run_chart(&x, Center::Median).unwrap();
        assert_eq!(r.runs_about, 2);
        assert_eq!(r.runs_updown, 1);
        assert_eq!(r.longest_updown, 19);
        assert!(r.p_trends < 0.001 && r.p_clustering < 0.001);
    }

    #[test]
    fn subgroup_medians_and_errors() {
        let p = subgroup_points(&[1.0, 5.0, 3.0, 9.0, 7.0, f64::NAN], 3, SubgroupStat::Median).unwrap();
        assert_eq!(p, vec![3.0, 8.0]);
        assert!(subgroup_points(&[1.0], 0, SubgroupStat::Mean).is_err());
        assert!(run_chart(&[1.0, 2.0], Center::Mean).is_err());
    }
}
