//! Factor analysis: principal-component and maximum-likelihood extraction, orthomax rotations,
//! regression scores.

use super::common::{corr_from_cov, covariance, eig_desc, listwise, mean, names_or_default, MatrixKind};
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ChiSquared, ContinuousDistribution};
use openstats_math::linalg::{Cholesky, Svd, SymmetricEigen};
use openstats_math::optim::{lbfgs_bounded, Options};
use openstats_math::Matrix;

/// Extraction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorMethod {
    /// Eigen-decomposition of the matrix (principal component method).
    PrincipalComponents,
    /// Gaussian maximum likelihood (Jöreskog's profile likelihood over the uniquenesses).
    MaximumLikelihood,
}

/// Orthogonal rotation of the loadings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rotation {
    None,
    Varimax,
    Quartimax,
    /// Equamax: orthomax with γ = m / 2.
    Equamax,
    /// Orthomax with the given γ (0 = quartimax, 1 = varimax).
    Orthomax(f64),
}

/// Options of [`factor_analysis`].
#[derive(Debug, Clone)]
pub struct FactorOptions {
    pub method: FactorMethod,
    pub n_factors: usize,
    /// Correlation (default) or covariance matrix.
    pub matrix: MatrixKind,
    pub rotation: Rotation,
    /// Kaiser row normalisation during rotation (default true).
    pub kaiser: bool,
    /// Compute regression factor scores.
    pub scores: bool,
}

impl FactorOptions {
    pub fn new(n_factors: usize) -> FactorOptions {
        FactorOptions {
            method: FactorMethod::PrincipalComponents,
            n_factors,
            matrix: MatrixKind::Correlation,
            rotation: Rotation::Varimax,
            kaiser: true,
            scores: false,
        }
    }
}

/// Likelihood-ratio test of "m factors are sufficient".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorFitTest {
    pub chi_square: f64,
    pub df: usize,
    pub p_value: f64,
}

/// Result of [`factor_analysis`].
#[derive(Debug, Clone)]
pub struct FactorAnalysis {
    pub names: Vec<String>,
    pub method: FactorMethod,
    pub matrix: MatrixKind,
    pub rotation: Rotation,
    pub n: usize,
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    /// Eigenvalues of the analysed matrix (descending).
    pub eigenvalues: Vec<f64>,
    /// p × m loadings before rotation.
    pub unrotated: Matrix,
    /// p × m loadings after rotation (equal to `unrotated` without rotation).
    pub loadings: Matrix,
    /// m × m orthogonal rotation: `loadings = unrotated · rotation_matrix`.
    pub rotation_matrix: Matrix,
    pub communalities: Vec<f64>,
    pub uniquenesses: Vec<f64>,
    /// Sum of squared loadings of each (rotated) factor.
    pub variance: Vec<f64>,
    /// `variance` over the total variance of the analysed matrix.
    pub proportion: Vec<f64>,
    pub total_variance: f64,
    /// ML only: discrepancy at the optimum, iterations used, convergence flag and the fit test.
    pub discrepancy: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub fit_test: Option<FactorFitTest>,
    /// p × m regression coefficients (`R⁻¹Λ`) and, when requested, the n × m scores.
    pub score_coefficients: Matrix,
    pub scores: Option<Matrix>,
}

pub fn factor_analysis(cols: &[Vec<f64>], names: &[String], opts: &FactorOptions) -> Result<FactorAnalysis> {
    let p = cols.len();
    let m = opts.n_factors;
    if p < 2 {
        return invalid("factor analysis needs at least two variables");
    }
    if m == 0 || m > p {
        return invalid(format!("the number of factors must be between 1 and {p}"));
    }
    if opts.method == FactorMethod::MaximumLikelihood && m >= p {
        return invalid("maximum likelihood needs fewer factors than variables");
    }
    let c = listwise(cols)?;
    let n = c.rows.len();
    if n < p + 1 {
        return Err(StatsError::InsufficientData { what: "factor analysis", needed: p + 1, got: n });
    }
    let means: Vec<f64> = c.cols.iter().map(|x| mean(x)).collect();
    let s = covariance(&c.cols);
    let sds: Vec<f64> = (0..p).map(|i| s[(i, i)].sqrt()).collect();
    let work = match opts.matrix {
        MatrixKind::Correlation => corr_from_cov(&s)?,
        MatrixKind::Covariance => {
            if sds.iter().any(|&d| !(d > 0.0)) {
                return invalid("a variable is constant");
            }
            s.clone()
        }
    };
    let total = work.trace();
    let (eigenvalues, vectors) = eig_desc(&work)?;

    let mut unrotated;
    let mut discrepancy = None;
    let mut iterations = 0;
    let mut converged = true;
    match opts.method {
        FactorMethod::PrincipalComponents => {
            unrotated = Matrix::zeros(p, m);
            for j in 0..m {
                let l = eigenvalues[j].max(0.0).sqrt();
                for i in 0..p {
                    unrotated[(i, j)] = vectors[(i, j)] * l;
                }
            }
        }
        FactorMethod::MaximumLikelihood => {
            let fit = ml_extract(&work, m)?;
            unrotated = fit.loadings;
            discrepancy = Some(fit.f);
            iterations = fit.iterations;
            converged = fit.converged;
        }
    }
    orient(&mut unrotated);

    let gamma = match opts.rotation {
        Rotation::None => None,
        Rotation::Varimax => Some(1.0),
        Rotation::Quartimax => Some(0.0),
        Rotation::Equamax => Some(m as f64 / 2.0),
        Rotation::Orthomax(g) => {
            if !g.is_finite() || !(0.0..=m as f64).contains(&g) {
                return invalid(format!("orthomax gamma must lie in [0, {m}]"));
            }
            Some(g)
        }
    };
    let (loadings, rot) = match gamma {
        Some(g) if m > 1 => rotate(&unrotated, g, opts.kaiser)?,
        _ => (unrotated.clone(), Matrix::identity(m)),
    };

    let communalities: Vec<f64> = (0..p).map(|i| (0..m).map(|j| loadings[(i, j)].powi(2)).sum()).collect();
    let uniquenesses: Vec<f64> = (0..p).map(|i| work[(i, i)] - communalities[i]).collect();
    let variance: Vec<f64> = (0..m).map(|j| (0..p).map(|i| loadings[(i, j)].powi(2)).sum()).collect();
    let proportion = variance.iter().map(|v| v / total).collect();

    let fit_test = if opts.method == FactorMethod::MaximumLikelihood {
        let df2 = (p as i64 - m as i64).pow(2) - (p as i64 + m as i64);
        if df2 > 0 {
            let f = discrepancy.unwrap_or(0.0);
            let stat = (n as f64 - 1.0 - (2.0 * p as f64 + 5.0) / 6.0 - 2.0 * m as f64 / 3.0) * f;
            let df = (df2 / 2) as usize;
            Some(FactorFitTest { chi_square: stat, df, p_value: ChiSquared::new(df as f64)?.sf(stat) })
        } else {
            None
        }
    } else {
        None
    };

    // Regression scores: coefficients R⁻¹Λ applied to the (standardised) data.
    let chol = Cholesky::new(&work).map_err(|_| StatsError::InvalidArgument("the matrix is singular".into()))?;
    let score_coefficients = chol.solve_mat(&loadings)?;
    let scores = if opts.scores {
        let mut f = Matrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut a = 0.0;
                for v in 0..p {
                    let scale = if opts.matrix == MatrixKind::Correlation { sds[v] } else { 1.0 };
                    a += (c.cols[v][i] - means[v]) / scale * score_coefficients[(v, j)];
                }
                f[(i, j)] = a;
            }
        }
        Some(f)
    } else {
        None
    };

    Ok(FactorAnalysis {
        names: names_or_default(names, p, "C"),
        method: opts.method,
        matrix: opts.matrix,
        rotation: opts.rotation,
        n,
        n_dropped: cols[0].len() - n,
        rows_used: c.rows,
        eigenvalues,
        unrotated,
        loadings,
        rotation_matrix: rot,
        communalities,
        uniquenesses,
        variance,
        proportion,
        total_variance: total,
        discrepancy,
        iterations,
        converged,
        fit_test,
        score_coefficients,
        scores,
    })
}

/// Make each column's sum positive.
fn orient(a: &mut Matrix) {
    for j in 0..a.cols() {
        let s: f64 = (0..a.rows()).map(|i| a[(i, j)]).sum();
        if s < 0.0 {
            for i in 0..a.rows() {
                a[(i, j)] = -a[(i, j)];
            }
        }
    }
}

struct MlFit {
    loadings: Matrix,
    f: f64,
    iterations: usize,
    converged: bool,
}

/// Loadings and discrepancy of the ML solution for given uniquenesses.
fn ml_given_psi(s: &Matrix, m: usize, psi: &[f64]) -> Result<(Matrix, f64, Vec<f64>)> {
    let p = psi.len();
    let d: Vec<f64> = psi.iter().map(|x| x.sqrt()).collect();
    let sstar = Matrix::from_fn(p, p, |i, j| s[(i, j)] / (d[i] * d[j]));
    let e = SymmetricEigen::new(&sstar)?; // ascending
    let mut f = 0.0;
    for j in 0..p - m {
        let th = e.values[j].max(1e-300);
        f += th - th.ln() - 1.0;
    }
    let mut lam = Matrix::zeros(p, m);
    for k in 0..m {
        let col = p - 1 - k;
        let w = (e.values[col] - 1.0).max(0.0).sqrt();
        for i in 0..p {
            lam[(i, k)] = d[i] * e.vectors[(i, col)] * w;
        }
    }
    let grad = (0..p)
        .map(|i| {
            let sig: f64 = (0..m).map(|k| lam[(i, k)].powi(2)).sum::<f64>() + psi[i];
            (sig - s[(i, i)]) / (psi[i] * psi[i])
        })
        .collect();
    Ok((lam, f, grad))
}

fn ml_extract(s: &Matrix, m: usize) -> Result<MlFit> {
    let p = s.rows();
    let inv = Cholesky::new(s).and_then(|c| c.inverse()).map_err(|_| {
        StatsError::InvalidArgument("the matrix is singular, so maximum likelihood cannot be used".into())
    })?;
    let start: Vec<f64> =
        (0..p).map(|i| ((1.0 - 0.5 * m as f64 / p as f64) / inv[(i, i)]).clamp(0.005 * s[(i, i)], s[(i, i)])).collect();
    let lower: Vec<f64> = (0..p).map(|i| 0.005 * s[(i, i)]).collect();
    let upper: Vec<f64> = (0..p).map(|i| s[(i, i)]).collect();
    let mut failed = false;
    let mut obj = |x: &[f64]| match ml_given_psi(s, m, x) {
        Ok((_, f, _)) => f,
        Err(_) => {
            failed = true;
            f64::MAX
        }
    };
    let mut grad = |x: &[f64]| match ml_given_psi(s, m, x) {
        Ok((_, _, g)) => g,
        Err(_) => vec![0.0; x.len()],
    };
    let opts = Options { max_iter: 5000, gtol: 1e-9, ftol: 1e-16, xtol: 1e-14, memory: 10 };
    let r = lbfgs_bounded(&mut obj, Some(&mut grad), &start, &lower, &upper, &opts)?;
    if failed {
        return invalid("the eigen-decomposition failed during maximum likelihood estimation");
    }
    let (lam, f, _) = ml_given_psi(s, m, &r.x)?;
    Ok(MlFit { loadings: lam, f, iterations: r.iterations, converged: r.converged })
}

/// Orthomax rotation by the SVD iteration; returns the rotated loadings (columns ordered by variance,
/// positive column sums) and the rotation matrix.
fn rotate(a: &Matrix, gamma: f64, kaiser: bool) -> Result<(Matrix, Matrix)> {
    let (p, m) = a.shape();
    let h: Vec<f64> = (0..p)
        .map(|i| {
            let r = (0..m).map(|j| a[(i, j)].powi(2)).sum::<f64>().sqrt();
            if kaiser && r > 1e-12 {
                r
            } else {
                1.0
            }
        })
        .collect();
    let an = Matrix::from_fn(p, m, |i, j| a[(i, j)] / h[i]);
    let mut t = Matrix::identity(m);
    let mut d = 0.0;
    for _ in 0..2000 {
        let b = an.matmul(&t)?;
        let colsq: Vec<f64> = (0..m).map(|j| (0..p).map(|i| b[(i, j)].powi(2)).sum()).collect();
        let g = Matrix::from_fn(p, m, |i, j| b[(i, j)].powi(3) - gamma / p as f64 * b[(i, j)] * colsq[j]);
        let q = an.t_matmul(&g)?;
        let svd = Svd::new(&q)?;
        t = svd.u.matmul(&svd.v.t())?;
        let dn: f64 = svd.s.iter().sum();
        if dn < d * (1.0 + 1e-12) {
            break;
        }
        d = dn;
    }
    let mut rotated = an.matmul(&t)?;
    for i in 0..p {
        for j in 0..m {
            rotated[(i, j)] *= h[i];
        }
    }
    // Order by explained variance and orient.
    let mut order: Vec<usize> = (0..m).collect();
    let var: Vec<f64> = (0..m).map(|j| (0..p).map(|i| rotated[(i, j)].powi(2)).sum()).collect();
    order.sort_by(|&x, &y| var[y].total_cmp(&var[x]));
    let mut r2 = Matrix::zeros(p, m);
    let mut t2 = Matrix::zeros(m, m);
    for (nj, &oj) in order.iter().enumerate() {
        let sum: f64 = (0..p).map(|i| rotated[(i, oj)]).sum();
        let sg = if sum < 0.0 { -1.0 } else { 1.0 };
        for i in 0..p {
            r2[(i, nj)] = sg * rotated[(i, oj)];
        }
        for i in 0..m {
            t2[(i, nj)] = sg * t[(i, oj)];
        }
    }
    Ok((r2, t2))
}
