//! Principal component analysis.

use super::common::{corr_from_cov, covariance, eig_desc, fix_signs, listwise, mean, names_or_default, MatrixKind};
use crate::error::{invalid, Result, StatsError};
use openstats_math::Matrix;

/// Options of [`pca`].
#[derive(Debug, Clone)]
pub struct PcaOptions {
    /// Correlation (default) or covariance matrix.
    pub matrix: MatrixKind,
    /// Number of components whose scores are computed (default: all).
    pub n_components: Option<usize>,
}

impl Default for PcaOptions {
    fn default() -> Self {
        PcaOptions { matrix: MatrixKind::Correlation, n_components: None }
    }
}

/// Result of [`pca`].
#[derive(Debug, Clone)]
pub struct Pca {
    pub names: Vec<String>,
    pub matrix: MatrixKind,
    /// Observations used and their worksheet rows.
    pub n: usize,
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    pub means: Vec<f64>,
    /// Standard deviations of the variables (used for standardising under the correlation matrix).
    pub sds: Vec<f64>,
    /// All p eigenvalues, descending.
    pub eigenvalues: Vec<f64>,
    pub proportion: Vec<f64>,
    pub cumulative: Vec<f64>,
    /// p × p eigenvectors (columns); the largest-magnitude entry of each column is positive.
    pub loadings: Matrix,
    /// n × k component scores of the first k components.
    pub scores: Matrix,
    pub total_variance: f64,
}

impl Pca {
    /// Scores of new observations (rows of columns `cols`, same variable order); `NaN` where a value is missing.
    pub fn project(&self, cols: &[Vec<f64>], k: usize) -> Result<Matrix> {
        if cols.len() != self.names.len() {
            return invalid("the number of variables differs from the fitted model");
        }
        let n = cols[0].len();
        let k = k.min(self.names.len());
        let mut out = Matrix::filled(n, k, f64::NAN);
        for i in 0..n {
            if cols.iter().any(|c| c[i].is_nan()) {
                continue;
            }
            for c in 0..k {
                let mut acc = 0.0;
                for (v, col) in cols.iter().enumerate() {
                    let z = (col[i] - self.means[v]) / self.scale(v);
                    acc += z * self.loadings[(v, c)];
                }
                out[(i, c)] = acc;
            }
        }
        Ok(out)
    }

    fn scale(&self, v: usize) -> f64 {
        match self.matrix {
            MatrixKind::Correlation => self.sds[v],
            MatrixKind::Covariance => 1.0,
        }
    }

    /// Correlations between the variables and the components: loading · √eigenvalue / sd (correlation matrix: loading · √eigenvalue).
    pub fn variable_component_correlations(&self) -> Matrix {
        let p = self.names.len();
        Matrix::from_fn(p, p, |v, c| {
            let sd = match self.matrix {
                MatrixKind::Correlation => 1.0,
                MatrixKind::Covariance => self.sds[v],
            };
            self.loadings[(v, c)] * self.eigenvalues[c].max(0.0).sqrt() / sd
        })
    }
}

/// Principal components of `cols` (one `Vec` per variable; `NaN` = missing, dropped listwise).
pub fn pca(cols: &[Vec<f64>], names: &[String], opts: &PcaOptions) -> Result<Pca> {
    let p = cols.len();
    if p < 2 {
        return invalid("principal components need at least two variables");
    }
    let c = listwise(cols)?;
    let n = c.rows.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "principal components", needed: 2, got: n });
    }
    let means: Vec<f64> = c.cols.iter().map(|x| mean(x)).collect();
    let s = covariance(&c.cols);
    let sds: Vec<f64> = (0..p).map(|i| s[(i, i)].sqrt()).collect();
    let work = match opts.matrix {
        MatrixKind::Correlation => corr_from_cov(&s)?,
        MatrixKind::Covariance => s.clone(),
    };
    let (mut values, mut vectors) = eig_desc(&work)?;
    for v in &mut values {
        if *v < 0.0 && *v > -1e-12 * values_scale(&work) {
            *v = 0.0;
        }
    }
    fix_signs(&mut vectors);
    let total: f64 = values.iter().sum();
    if !(total > 0.0) {
        return invalid("the variables have no variation");
    }
    let proportion: Vec<f64> = values.iter().map(|v| v / total).collect();
    let mut cumulative = Vec::with_capacity(p);
    let mut acc = 0.0;
    for q in &proportion {
        acc += q;
        cumulative.push(acc);
    }
    let k = opts.n_components.unwrap_or(p).clamp(1, p);
    let mut scores = Matrix::zeros(n, k);
    for i in 0..n {
        for j in 0..k {
            let mut a = 0.0;
            for v in 0..p {
                let scale = if opts.matrix == MatrixKind::Correlation { sds[v] } else { 1.0 };
                a += (c.cols[v][i] - means[v]) / scale * vectors[(v, j)];
            }
            scores[(i, j)] = a;
        }
    }
    Ok(Pca {
        names: names_or_default(names, p, "C"),
        matrix: opts.matrix,
        n,
        n_dropped: cols[0].len() - n,
        rows_used: c.rows,
        means,
        sds,
        eigenvalues: values,
        proportion,
        cumulative,
        loadings: vectors,
        scores,
        total_variance: total,
    })
}

fn values_scale(m: &Matrix) -> f64 {
    m.norm_max().max(1.0)
}
