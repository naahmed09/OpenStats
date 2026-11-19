//! Stem-and-leaf display (Tukey), with Minitab-style depth column and leaf unit.

use crate::error::{invalid, Result};
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

/// One printed line of the display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StemLine {
    /// Depth: cumulative count towards the median line, `(count)` on the median line, empty
    /// for an empty line beyond the median that has no leaves.
    pub depth: String,
    /// The stem as printed (`-0`, `0`, `12`, ...).
    pub stem: String,
    /// The leaf digits, sorted, one character each.
    pub leaves: String,
}

/// A stem-and-leaf display.
#[derive(Debug, Clone, PartialEq)]
pub struct StemLeaf {
    /// Leaf unit (a leaf digit `d` stands for `d * leaf_unit`).
    pub leaf_unit: f64,
    /// Values per line, in leaf units (2, 5 or 10).
    pub line_width: u32,
    /// The lines from the smallest to the largest value.
    pub lines: Vec<StemLine>,
    /// Number of observations.
    pub n: usize,
    /// Observations with a missing value, dropped.
    pub n_missing: usize,
}

/// Build the display. `leaf_unit` of `None` chooses the smallest power of ten (with the
/// narrowest line width among 2, 5, 10 leaf units) that gives at most about `10·log10(n)` lines
/// (never fewer than 6 allowed). Values are truncated, not rounded, to the leaf unit. Needs
/// at least 2 non-missing finite values.
pub fn stem_and_leaf(data: &[f64], leaf_unit: Option<f64>) -> Result<StemLeaf> {
    let n_all = data.len();
    let mut v: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if v.iter().any(|x| x.abs() > 1e15) {
        return invalid("stem-and-leaf: values larger than 1e15 in magnitude are not supported");
    }
    if v.len() < 2 {
        return invalid("stem-and-leaf needs at least 2 values");
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let (lo, hi) = (v[0], v[n - 1]);
    let max_lines = ((10.0 * (n as f64).log10()).floor() as usize).max(6);
    let widths = [2u32, 5, 10];
    let (unit, width) = match leaf_unit {
        Some(u) => {
            if u.is_nan() || u <= 0.0 || !u.is_finite() {
                return invalid("stem-and-leaf: the leaf unit must be positive");
            }
            let p = u.log10();
            if (p - p.round()).abs() > 1e-9 {
                return invalid("stem-and-leaf: the leaf unit must be a power of ten");
            }
            let mut best = (u, 10);
            for w in widths {
                if (hi - lo) / (f64::from(w) * u) + 1.0 <= max_lines as f64 {
                    best = (u, w);
                    break;
                }
            }
            best
        }
        None => {
            let mut m = ((hi - lo).max(1e-300).log10().floor() as i32) - 3;
            loop {
                let u = 10f64.powi(m);
                if let Some(w) = widths.iter().find(|&&w| (hi - lo) / (f64::from(w) * u) + 1.0 <= max_lines as f64) {
                    break (u, *w);
                }
                m += 1;
            }
        }
    };
    // Truncate to integers in leaf units.
    let t: Vec<i64> = v.iter().map(|&x| (x / unit + if x >= 0.0 { 1e-9 } else { -1e-9 }).trunc() as i64).collect();
    let w = i64::from(width);
    let line_of = |t: i64| if t >= 0 { t / w } else { -((-t) / w) - 1 };
    let (l0, l1) = (line_of(t[0]), line_of(t[n - 1]));
    let nl = (l1 - l0 + 1) as usize;
    if nl > 5000 {
        return invalid("stem-and-leaf: too many lines for this leaf unit");
    }
    let mut leaves: Vec<Vec<u8>> = vec![Vec::new(); nl];
    let mut counts = vec![0usize; nl];
    for &ti in &t {
        let k = (line_of(ti) - l0) as usize;
        counts[k] += 1;
        leaves[k].push((ti.abs() % 10) as u8);
    }
    for l in &mut leaves {
        l.sort_unstable();
    }
    // Negative lines list their leaves from the one closest to zero outwards (largest first).
    // The median line is where the cumulative count first reaches (n+1)/2.
    let half = n.div_ceil(2);
    let mut acc = 0usize;
    let mut med = 0usize;
    for (k, &c) in counts.iter().enumerate() {
        acc += c;
        if acc >= half {
            med = k;
            break;
        }
    }
    let mut lines = Vec::with_capacity(nl);
    let mut below = 0usize;
    for k in 0..nl {
        let key = l0 + k as i64;
        let stem_abs = if key >= 0 { (key * w) / 10 } else { ((-key - 1) * w) / 10 };
        let stem = if key >= 0 { format!("{stem_abs}") } else { format!("-{stem_abs}") };
        let depth = if k < med {
            below += counts[k];
            if counts[k] == 0 && below == 0 {
                String::new()
            } else {
                format!("{below}")
            }
        } else if k == med {
            format!("({})", counts[k])
        } else {
            let above: usize = counts[k..].iter().sum();
            format!("{above}")
        };
        let digits: String = leaves[k].iter().map(|d| char::from(b'0' + d)).collect();
        lines.push(StemLine { depth, stem, leaves: digits });
    }
    Ok(StemLeaf { leaf_unit: unit, line_width: width, lines, n, n_missing: n_all - n })
}

impl Reportable for StemLeaf {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Stem-and-Leaf Display");
        s.text(format!("Leaf Unit = {}", self.leaf_unit));
        s.text(format!(
            "N = {}{}",
            self.n,
            if self.n_missing > 0 { format!(" ({} missing ignored)", self.n_missing) } else { String::new() }
        ));
        let mut t = Table::new(
            "",
            vec![
                Column::new("Depth", ColumnKind::Text),
                Column::new("Stem", ColumnKind::Text),
                Column::new("Leaves", ColumnKind::Text),
            ],
        );
        for l in &self.lines {
            t.push_row(vec![Cell::text(&l.depth), Cell::text(&l.stem), Cell::text(&l.leaves)]).expect("three cells");
        }
        s.table(t);
        s
    }
}
