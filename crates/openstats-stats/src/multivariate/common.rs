//! Shared helpers: listwise deletion, moments and matrix utilities.

use crate::error::{invalid, Result, StatsError};
use openstats_math::Matrix;

/// Which matrix an analysis works on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixKind {
    /// Standardized variables (Minitab's default).
    Correlation,
    /// Raw deviations from the mean.
    Covariance,
}

/// Columns with every row that has a missing value removed.
pub(crate) struct Complete {
    pub rows: Vec<usize>,
    pub cols: Vec<Vec<f64>>,
}

pub(crate) fn listwise(cols: &[Vec<f64>]) -> Result<Complete> {
    if cols.is_empty() {
        return invalid("no variables were given");
    }
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return invalid("the variables have different lengths");
    }
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut ok = true;
        for c in cols {
            let v = c[i];
            if v.is_infinite() {
                return Err(StatsError::NonFinite);
            }
            if v.is_nan() {
                ok = false;
            }
        }
        if ok {
            rows.push(i);
        }
    }
    let kept = cols.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect();
    Ok(Complete { rows, cols: kept })
}

pub(crate) fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample covariance matrix (divisor n − 1) of complete columns.
pub(crate) fn covariance(cols: &[Vec<f64>]) -> Matrix {
    let p = cols.len();
    let n = cols[0].len();
    let means: Vec<f64> = cols.iter().map(|c| mean(c)).collect();
    let mut s = Matrix::zeros(p, p);
    for a in 0..p {
        for b in a..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += (cols[a][i] - means[a]) * (cols[b][i] - means[b]);
            }
            let v = acc / (n as f64 - 1.0);
            s[(a, b)] = v;
            s[(b, a)] = v;
        }
    }
    s
}

pub(crate) fn corr_from_cov(s: &Matrix) -> Result<Matrix> {
    let p = s.rows();
    let sd: Vec<f64> = (0..p).map(|i| s[(i, i)].sqrt()).collect();
    if sd.iter().any(|&d| !(d > 0.0)) {
        return invalid("a variable is constant, so its correlations are undefined");
    }
    Ok(Matrix::from_fn(p, p, |i, j| if i == j { 1.0 } else { s[(i, j)] / (sd[i] * sd[j]) }))
}

/// Descending eigen-decomposition of a symmetric matrix: (values, vectors as columns).
pub(crate) fn eig_desc(a: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    let e = openstats_math::linalg::SymmetricEigen::new(a)?;
    let p = e.values.len();
    let values: Vec<f64> = (0..p).map(|i| e.values[p - 1 - i]).collect();
    let vectors = Matrix::from_fn(p, p, |i, j| e.vectors[(i, p - 1 - j)]);
    Ok((values, vectors))
}

/// Flip each column's sign so its largest-magnitude entry is positive (first wins ties).
pub(crate) fn fix_signs(m: &mut Matrix) {
    for j in 0..m.cols() {
        let mut best = 0usize;
        for i in 1..m.rows() {
            if m[(i, j)].abs() > m[(best, j)].abs() * (1.0 + 1e-12) {
                best = i;
            }
        }
        if m[(best, j)] < 0.0 {
            for i in 0..m.rows() {
                m[(i, j)] = -m[(i, j)];
            }
        }
    }
}

pub(crate) fn names_or_default(names: &[String], p: usize, prefix: &str) -> Vec<String> {
    (0..p).map(|i| names.get(i).cloned().unwrap_or_else(|| format!("{prefix}{}", i + 1))).collect()
}

/// Group labels (empty string = missing) into level names (sorted by first appearance order of the
/// sorted labels) and per-row indices.
pub(crate) fn levels_of(labels: &[String]) -> (Vec<String>, Vec<Option<usize>>) {
    let mut names: Vec<String> = labels.iter().filter(|s| !s.is_empty()).cloned().collect();
    let numeric = names.iter().all(|s| s.parse::<f64>().is_ok());
    if numeric {
        names.sort_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse::<f64>().unwrap()));
    } else {
        names.sort();
    }
    names.dedup();
    let idx = labels.iter().map(|s| if s.is_empty() { None } else { names.iter().position(|n| n == s) }).collect();
    (names, idx)
}

/// Indices of the rows that have no missing value in any of `cols`.
pub fn complete_rows(cols: &[Vec<f64>]) -> Vec<usize> {
    let n = cols.first().map_or(0, Vec::len);
    (0..n).filter(|&i| cols.iter().all(|c| c.get(i).is_some_and(|x| !x.is_nan()))).collect()
}
