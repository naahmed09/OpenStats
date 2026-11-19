//! Data > Merge Worksheets.

use crate::column::{Column, ColumnData};
use crate::error::{CoreError, Result};
use crate::worksheet::Worksheet;
use std::collections::{HashMap, HashSet};

/// Join type when merging by key columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    /// Only rows whose keys appear in both worksheets.
    Inner,
    /// All rows of the left worksheet.
    Left,
    /// All rows of both worksheets.
    Outer,
}

/// Place two worksheets side by side (row *i* of each aligned).
pub fn merge_side_by_side(left: &Worksheet, right: &Worksheet, name: &str) -> Result<Worksheet> {
    let mut out = Worksheet::new(name);
    let mut taken: HashSet<String> = HashSet::new();
    for c in left.columns.iter().chain(right.columns.iter()) {
        let mut nc = c.clone();
        nc.props.name = super::unique_name(c.name(), &taken);
        if !nc.props.name.is_empty() {
            taken.insert(nc.props.name.to_lowercase());
        }
        out.columns.push(nc);
    }
    Ok(out)
}

/// Merge by matching key columns (`left_keys[i]` pairs with `right_keys[i]`).
/// Duplicate keys produce every pairing (as a relational join). Rows with a
/// missing key never match but are kept by `Left`/`Outer` joins. Key columns
/// appear once; when only one side has the row, the key comes from that side.
/// Output rows follow left order, then unmatched right rows in right order.
pub fn merge_by_keys(
    left: &Worksheet,
    right: &Worksheet,
    left_keys: &[usize],
    right_keys: &[usize],
    join: JoinKind,
    name: &str,
) -> Result<Worksheet> {
    if left_keys.is_empty() || left_keys.len() != right_keys.len() {
        return Err(CoreError::invalid("merge needs the same non-zero number of key columns on each side"));
    }
    for (&l, &r) in left_keys.iter().zip(right_keys) {
        let (lk, rk) = (left.column(l)?.kind(), right.column(r)?.kind());
        if lk != rk {
            return Err(CoreError::TypeMismatch { expected: lk.name(), found: rk.name() });
        }
    }
    let nl = left.n_rows();
    let nr = right.n_rows();
    let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
    for r in 0..nr {
        if let Some(k) = super::unstack::row_key(right, right_keys, r) {
            right_index.entry(k).or_default().push(r);
        }
    }
    // Row pairs: (Some(left_row) | None, Some(right_row) | None)
    let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let mut right_used = vec![false; nr];
    for l in 0..nl {
        let matches = super::unstack::row_key(left, left_keys, l).and_then(|k| right_index.get(&k));
        match matches {
            Some(rs) => {
                for &r in rs {
                    pairs.push((Some(l), Some(r)));
                    right_used[r] = true;
                }
            }
            None => {
                if join != JoinKind::Inner {
                    pairs.push((Some(l), None));
                }
            }
        }
    }
    if join == JoinKind::Outer {
        for r in 0..nr {
            if !right_used[r] {
                pairs.push((None, Some(r)));
            }
        }
    }
    let li: Vec<usize> = pairs.iter().map(|p| p.0.unwrap_or(usize::MAX)).collect();
    let ri: Vec<usize> = pairs.iter().map(|p| p.1.unwrap_or(usize::MAX)).collect();

    let mut out = Worksheet::new(name);
    let mut taken: HashSet<String> = HashSet::new();
    let mut push = |out: &mut Worksheet, mut c: Column| {
        c.props.name = super::unique_name(c.name(), &taken);
        if !c.props.name.is_empty() {
            taken.insert(c.props.name.to_lowercase());
        }
        out.columns.push(c);
    };
    for (ci, c) in left.columns.iter().enumerate() {
        let mut nc = c.clone();
        nc.data = c.data.take(&li);
        if let Some(kpos) = left_keys.iter().position(|&k| k == ci) {
            // Fill keys of right-only rows from the right worksheet.
            let rcol = &right.columns[right_keys[kpos]].data;
            for (row, p) in pairs.iter().enumerate() {
                if p.0.is_none() {
                    let v = rcol.get(p.1.expect("right-only row"));
                    nc.data.set(row, &v)?;
                }
            }
        }
        push(&mut out, nc);
    }
    let rkeys: HashSet<usize> = right_keys.iter().copied().collect();
    for (ci, c) in right.columns.iter().enumerate() {
        if rkeys.contains(&ci) {
            continue;
        }
        let mut nc = c.clone();
        let d: ColumnData = c.data.take(&ri);
        nc.data = d;
        push(&mut out, nc);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    fn sheets() -> (Worksheet, Worksheet) {
        let l = Worksheet::with_columns(
            "L",
            vec![Column::text("Part", vec!["a", "b", "c", ""]), Column::numeric("Qty", vec![1.0, 2.0, 3.0, 4.0])],
        )
        .unwrap();
        let r = Worksheet::with_columns(
            "R",
            vec![Column::numeric("Qty", vec![9.0, 8.0, 7.0]), Column::text("Part", vec!["b", "d", "b"])],
        )
        .unwrap();
        (l, r)
    }

    #[test]
    fn joins() {
        let (l, r) = sheets();
        let inner = merge_by_keys(&l, &r, &[0], &[1], JoinKind::Inner, "m").unwrap();
        assert_eq!(inner.columns[0].data.as_text().unwrap(), ["b", "b"]);
        assert_eq!(inner.columns[2].name(), "Qty_1");
        assert_eq!(inner.columns[2].data, ColumnData::Numeric(vec![9.0, 7.0]));
        let left = merge_by_keys(&l, &r, &[0], &[1], JoinKind::Left, "m").unwrap();
        assert_eq!(left.n_rows(), 5);
        let outer = merge_by_keys(&l, &r, &[0], &[1], JoinKind::Outer, "m").unwrap();
        assert_eq!(outer.n_rows(), 6);
        assert_eq!(outer.columns[0].get(5), Value::Text("d".into()));
        assert!(outer.columns[1].data.is_missing(5));
        assert!(merge_by_keys(&l, &r, &[0], &[0], JoinKind::Inner, "m").is_err());
    }

    #[test]
    fn side_by_side() {
        let (l, r) = sheets();
        let m = merge_side_by_side(&l, &r, "m").unwrap();
        let names: Vec<_> = m.columns.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["Part", "Qty", "Qty_1", "Part_1"]);
    }
}
