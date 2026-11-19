//! Simple and multiple correspondence analysis, and promax (oblique) rotation of factor loadings.

use crate::error::{invalid, Result, StatsError};
use openstats_math::linalg::{Cholesky, Matrix, Svd};

/// Result of a correspondence analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrespondenceAnalysis {
    /// Row labels.
    pub row_labels: Vec<String>,
    /// Column labels.
    pub col_labels: Vec<String>,
    /// Grand total (observations for multiple correspondence analysis).
    pub total: f64,
    /// Row masses and column masses.
    pub row_mass: Vec<f64>,
    /// Column masses.
    pub col_mass: Vec<f64>,
    /// Singular values.
    pub singular_values: Vec<f64>,
    /// Principal inertias (squared singular values).
    pub inertia: Vec<f64>,
    /// Total inertia (`χ² / n`).
    pub total_inertia: f64,
    /// Pearson chi-square statistic `n · total inertia` (meaningful for two-way tables).
    pub chi_square: f64,
    /// Row principal coordinates (`rows × components`).
    pub row_coord: Matrix,
    /// Column principal coordinates (`columns × components`).
    pub col_coord: Matrix,
    /// Row contribution to each component (`mass · coord² / inertia`).
    pub row_contrib: Matrix,
    /// Column contribution to each component.
    pub col_contrib: Matrix,
    /// Row quality: squared correlation with each component (`coord² / squared distance to the centroid`).
    pub row_quality: Matrix,
    /// Column squared correlations with each component.
    pub col_quality: Matrix,
    /// Row inertia (mass times squared distance to the centroid), as a share of the total.
    pub row_inertia: Vec<f64>,
    /// Column inertia shares.
    pub col_inertia: Vec<f64>,
}

/// Simple correspondence analysis of a table of non-negative counts (`rows × columns`, row-major).
pub fn correspondence_analysis(
    table: &[Vec<f64>],
    row_labels: &[String],
    col_labels: &[String],
    n_components: Option<usize>,
) -> Result<CorrespondenceAnalysis> {
    let r = table.len();
    let c = table.first().map_or(0, Vec::len);
    if r < 2 || c < 2 {
        return invalid("correspondence analysis needs a table with at least two rows and two columns");
    }
    if table.iter().any(|row| row.len() != c) || row_labels.len() != r || col_labels.len() != c {
        return invalid("the table and its labels must have matching shapes");
    }
    if table.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
        return invalid("counts must be finite and non-negative");
    }
    let total: f64 = table.iter().flatten().sum();
    if total <= 0.0 {
        return invalid("the table has no counts");
    }
    let rs: Vec<f64> = table.iter().map(|row| row.iter().sum::<f64>()).collect();
    let cs: Vec<f64> = (0..c).map(|j| table.iter().map(|row| row[j]).sum()).collect();
    if rs.contains(&0.0) || cs.contains(&0.0) {
        return invalid("a row or column of the table is empty; remove it first");
    }
    let rm: Vec<f64> = rs.iter().map(|v| v / total).collect();
    let cm: Vec<f64> = cs.iter().map(|v| v / total).collect();
    let s = Matrix::from_fn(r, c, |i, j| (table[i][j] / total - rm[i] * cm[j]) / (rm[i] * cm[j]).sqrt());
    let svd = Svd::new(&s)?;
    let rank_max = (r.min(c) - 1).max(1);
    let tol = 1e-12 * svd.s.first().copied().unwrap_or(0.0).max(1e-300);
    let k = n_components.unwrap_or(rank_max).min(rank_max).min(svd.s.iter().filter(|&&v| v > tol).count().max(1));
    let sv: Vec<f64> = svd.s[..k].to_vec();
    let row_coord = Matrix::from_fn(r, k, |i, a| svd.u[(i, a)] * sv[a] / rm[i].sqrt());
    let col_coord = Matrix::from_fn(c, k, |j, a| svd.v[(j, a)] * sv[a] / cm[j].sqrt());
    let inertia: Vec<f64> = sv.iter().map(|v| v * v).collect();
    let total_inertia: f64 = s.as_slice().iter().map(|v| v * v).sum();
    let contrib = |coord: &Matrix, mass: &[f64]| {
        Matrix::from_fn(coord.rows(), k, |i, a| {
            if inertia[a] > 0.0 {
                mass[i] * coord[(i, a)].powi(2) / inertia[a]
            } else {
                0.0
            }
        })
    };
    // Squared distance to the centroid on the full space, per row / column.
    let row_d2: Vec<f64> = (0..r).map(|i| (0..c).map(|j| s[(i, j)].powi(2)).sum::<f64>() / rm[i]).collect();
    let col_d2: Vec<f64> = (0..c).map(|j| (0..r).map(|i| s[(i, j)].powi(2)).sum::<f64>() / cm[j]).collect();
    let quality = |coord: &Matrix, d2: &[f64]| {
        Matrix::from_fn(coord.rows(), k, |i, a| if d2[i] > 0.0 { coord[(i, a)].powi(2) / d2[i] } else { 0.0 })
    };
    let row_inertia =
        (0..r).map(|i| if total_inertia > 0.0 { rm[i] * row_d2[i] / total_inertia } else { 0.0 }).collect();
    let col_inertia =
        (0..c).map(|j| if total_inertia > 0.0 { cm[j] * col_d2[j] / total_inertia } else { 0.0 }).collect();
    Ok(CorrespondenceAnalysis {
        row_labels: row_labels.to_vec(),
        col_labels: col_labels.to_vec(),
        total,
        row_contrib: contrib(&row_coord, &rm),
        col_contrib: contrib(&col_coord, &cm),
        row_quality: quality(&row_coord, &row_d2),
        col_quality: quality(&col_coord, &col_d2),
        row_mass: rm,
        col_mass: cm,
        singular_values: sv,
        inertia,
        total_inertia,
        chi_square: total * total_inertia,
        row_coord,
        col_coord,
        row_inertia,
        col_inertia,
    })
}

/// Multiple correspondence analysis of categorical variables: simple correspondence analysis of the
/// indicator matrix (observations in rows, one column per category). Rows with a missing (empty)
/// label are dropped.
pub fn multiple_correspondence(
    columns: &[Vec<String>],
    names: &[String],
    n_components: Option<usize>,
) -> Result<CorrespondenceAnalysis> {
    if columns.len() < 2 {
        return invalid("multiple correspondence analysis needs at least two categorical variables");
    }
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) || names.len() != columns.len() {
        return invalid("the variables must have the same length");
    }
    let rows: Vec<usize> = (0..n).filter(|&i| columns.iter().all(|c| !c[i].is_empty())).collect();
    if rows.len() < 3 {
        return Err(StatsError::InsufficientData {
            what: "multiple correspondence analysis",
            needed: 3,
            got: rows.len(),
        });
    }
    let mut cats: Vec<(usize, String)> = Vec::new();
    for (k, col) in columns.iter().enumerate() {
        let mut levels: Vec<&String> = rows.iter().map(|&i| &col[i]).collect();
        levels.sort();
        levels.dedup();
        if levels.len() < 2 {
            return invalid(format!("variable '{}' has fewer than two categories", names[k]));
        }
        cats.extend(levels.into_iter().map(|l| (k, l.clone())));
    }
    let table: Vec<Vec<f64>> =
        rows.iter().map(|&i| cats.iter().map(|(k, l)| f64::from(columns[*k][i] == *l)).collect()).collect();
    let row_labels: Vec<String> = rows.iter().map(|&i| (i + 1).to_string()).collect();
    let col_labels: Vec<String> = cats.iter().map(|(k, l)| format!("{}={}", names[*k], l)).collect();
    // Components beyond (categories − variables) carry no inertia.
    let max = cats.len() - columns.len();
    correspondence_analysis(&table, &row_labels, &col_labels, Some(n_components.unwrap_or(max).min(max)))
}

/// Promax rotation: an oblique rotation obtained by raising a varimax solution to a power.
#[derive(Debug, Clone, PartialEq)]
pub struct Promax {
    /// Power (commonly 4).
    pub power: f64,
    /// Pattern loadings (`variables × factors`).
    pub pattern: Matrix,
    /// Structure loadings (`pattern · Φ`).
    pub structure: Matrix,
    /// Factor correlation matrix Φ.
    pub factor_correlation: Matrix,
    /// Transformation applied to the unrotated loadings (`pattern = unrotated · transform`).
    pub transform: Matrix,
}

/// Promax rotation of `varimax` loadings. `rotation` is the orthogonal matrix that produced them from
/// `unrotated` (`varimax = unrotated · rotation`).
pub fn promax(unrotated: &Matrix, varimax: &Matrix, rotation: &Matrix, power: f64) -> Result<Promax> {
    let (p, m) = varimax.shape();
    if m < 2 {
        return invalid("promax needs at least two factors");
    }
    if !(power > 1.0 && power.is_finite()) {
        return invalid("the promax power must exceed 1");
    }
    if unrotated.shape() != (p, m) || rotation.shape() != (m, m) {
        return invalid("loadings and rotation matrices have inconsistent shapes");
    }
    let q = Matrix::from_fn(p, m, |i, j| {
        let x = varimax[(i, j)];
        x * x.abs().powf(power - 1.0)
    });
    // Least squares U = (X'X)^-1 X'Q
    let xtx = varimax.gram();
    let xtq = varimax.t_matmul(&q)?;
    let u0 = Cholesky::new(&xtx)?.solve_mat(&xtq)?;
    let utu_inv = Cholesky::new(&u0.gram())?.inverse()?;
    let u = Matrix::from_fn(m, m, |i, j| u0[(i, j)] * utu_inv[(j, j)].sqrt());
    let transform = rotation.matmul(&u)?;
    let pattern = unrotated.matmul(&transform)?;
    let phi = Cholesky::new(&transform.gram())?.inverse()?;
    let structure = pattern.matmul(&phi)?;
    Ok(Promax { power, pattern, structure, factor_correlation: phi, transform })
}
