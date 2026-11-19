//! Ordinary least squares.

use super::summary::{sum, two_prod, two_sum};
use crate::error::{MathError, Result};
use crate::linalg::{Matrix, Qr};

/// How R² is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R2Kind {
    /// 1 − SSE/Σ(y − ȳ)²: for models with an intercept.
    Centered,
    /// 1 − SSE/Σy²: for models through the origin (NIST's and Minitab's definition).
    Uncentered,
}

/// Result of [`ols`].
#[derive(Debug, Clone)]
pub struct OlsFit {
    pub coef: Vec<f64>,
    pub std_err: Vec<f64>,
    /// s²·(XᵀX)⁻¹.
    pub cov: Matrix,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub sse: f64,
    /// Residual degrees of freedom n − p.
    pub dof: usize,
    /// √(SSE/dof).
    pub residual_sd: f64,
    pub r_squared: f64,
}

/// Residual r = y − Xb with the products and sums carried in double-double
/// precision, so that b can be refined to nearly full accuracy.
fn accurate_residual(x: &Matrix, y: &[f64], b: &[f64]) -> Vec<f64> {
    (0..x.rows())
        .map(|i| {
            let (mut hi, mut lo) = (y[i], 0.0f64);
            for (j, bj) in b.iter().enumerate() {
                let (p, pe) = two_prod(-x[(i, j)], *bj);
                let (s, se) = two_sum(hi, p);
                hi = s;
                lo += se + pe;
            }
            hi + lo
        })
        .collect()
}

/// Least squares fit of `y` on the columns of `x` (include a column of ones
/// for an intercept). The solution is computed by Householder QR and then
/// improved by iterative refinement with extended-precision residuals, which
/// recovers several digits on ill-conditioned designs such as Longley's.
/// Fails if X is rank deficient (a zero on the diagonal of R) or n ≤ p.
pub fn ols(x: &Matrix, y: &[f64], r2: R2Kind) -> Result<OlsFit> {
    let (n, p) = x.shape();
    if y.len() != n {
        return Err(MathError::DimensionMismatch { op: "ols", left: (n, p), right: (y.len(), 1) });
    }
    if n <= p || p == 0 {
        return Err(MathError::InvalidArgument(format!(
            "ols: need more observations than coefficients ({n} observations, {p} coefficients)"
        )));
    }
    if x.as_slice().iter().chain(y).any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: "ols" });
    }
    let qr = Qr::new(x)?;
    let rd = qr.r_diag();
    let rmax = rd.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if let Some(k) = rd.iter().position(|v| v.abs() <= rmax * f64::EPSILON * n as f64) {
        return Err(MathError::Singular { pivot: k });
    }
    let mut b = qr.solve_least_squares(y)?;
    for _ in 0..4 {
        let r = accurate_residual(x, y, &b);
        let d = qr.solve_least_squares(&r)?;
        let step = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (bi, di) in b.iter_mut().zip(&d) {
            *bi += di;
        }
        if step <= 1e-17 * scale {
            break;
        }
    }
    let residuals = accurate_residual(x, y, &b);
    let fitted: Vec<f64> = y.iter().zip(&residuals).map(|(yi, ri)| yi - ri).collect();
    let sse = sum(residuals.iter().map(|r| r * r));
    let dof = n - p;
    let s2 = sse / dof as f64;
    // (XᵀX)⁻¹ = R⁻¹R⁻ᵀ with R the p×p upper-triangular factor.
    let r = qr.r();
    let mut rinv = Matrix::zeros(p, p);
    for j in 0..p {
        rinv[(j, j)] = 1.0 / r[(j, j)];
        for i in (0..j).rev() {
            let s: f64 = (i + 1..=j).map(|k| r[(i, k)] * rinv[(k, j)]).sum();
            rinv[(i, j)] = -s / r[(i, i)];
        }
    }
    let mut cov = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..=i {
            let v: f64 = (i.max(j)..p).map(|k| rinv[(i, k)] * rinv[(j, k)]).sum::<f64>() * s2;
            cov[(i, j)] = v;
            cov[(j, i)] = v;
        }
    }
    let std_err = (0..p).map(|i| cov[(i, i)].sqrt()).collect();
    let total = match r2 {
        R2Kind::Centered => crate::stats::summary::sum_squared_deviations(y)?,
        R2Kind::Uncentered => sum(y.iter().map(|v| v * v)),
    };
    let r_squared = if total > 0.0 { 1.0 - sse / total } else { f64::NAN };
    Ok(OlsFit { coef: b, std_err, cov, fitted, residuals, sse, dof, residual_sd: s2.sqrt(), r_squared })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_line_and_errors() {
        let x = Matrix::from_fn(6, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let y: Vec<f64> = (0..6).map(|i| 3.0 + 2.0 * i as f64).collect();
        let f = ols(&x, &y, R2Kind::Centered).unwrap();
        assert!((f.coef[0] - 3.0).abs() < 1e-14 && (f.coef[1] - 2.0).abs() < 1e-14);
        assert!(f.sse < 1e-24 && (f.r_squared - 1.0).abs() < 1e-14);
        assert_eq!(f.dof, 4);
        // Noisy line: hand-computed values (x = 1..5, y = 2.1, 3.9, 6.2, 7.8, 10.1).
        let x = Matrix::from_fn(5, 2, |i, j| if j == 0 { 1.0 } else { (i + 1) as f64 });
        let f = ols(&x, &[2.1, 3.9, 6.2, 7.8, 10.1], R2Kind::Centered).unwrap();
        assert!((f.coef[1] - 1.99).abs() < 1e-13 && (f.coef[0] - 0.05).abs() < 1e-13);
        // Rank deficiency and shape errors.
        let dup = Matrix::from_fn(5, 2, |i, _| (i + 1) as f64);
        assert!(ols(&dup, &[1.0, 2.0, 3.0, 4.0, 5.0], R2Kind::Centered).is_err());
        assert!(ols(&x, &[1.0, 2.0], R2Kind::Centered).is_err());
        assert!(ols(&Matrix::zeros(2, 2), &[1.0, 2.0], R2Kind::Centered).is_err());
    }
}
