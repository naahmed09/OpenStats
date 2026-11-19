//! Cholesky (A = L·Lᵀ), LDLᵀ (A = L·D·Lᵀ, unit L) and modified Cholesky
//! (A + τI = L·Lᵀ with the smallest τ found by doubling).
//!
//! All routines read only the lower triangle of the input.

use super::matrix::Matrix;
use crate::error::{MathError, Result};

/// Cholesky factor of a symmetric positive-definite matrix.
#[derive(Debug, Clone)]
pub struct Cholesky {
    l: Matrix,
}

impl Cholesky {
    pub fn new(a: &Matrix) -> Result<Self> {
        let n = a.require_square()?;
        a.require_finite("Cholesky")?;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let s = a[(i, j)] - super::blas::dot(&l.row(i)[..j], &l.row(j)[..j]);
                if i == j {
                    if s <= 0.0 || !s.is_finite() {
                        return Err(MathError::NotPositiveDefinite { pivot: i });
                    }
                    l[(i, i)] = s.sqrt();
                } else {
                    l[(i, j)] = s / l[(j, j)];
                }
            }
        }
        Ok(Cholesky { l })
    }

    pub fn l(&self) -> &Matrix {
        &self.l
    }

    pub fn dim(&self) -> usize {
        self.l.rows()
    }

    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let mut x = b.to_vec();
        super::triangular::solve_lower(&self.l, &mut x, false)?;
        super::triangular::solve_lower_t(&self.l, &mut x, false)?;
        Ok(x)
    }

    pub fn solve_mat(&self, b: &Matrix) -> Result<Matrix> {
        let mut out = Matrix::zeros(b.rows(), b.cols());
        for j in 0..b.cols() {
            out.set_col(j, &self.solve(&b.col(j))?);
        }
        Ok(out)
    }

    /// A⁻¹ (symmetric).
    pub fn inverse(&self) -> Result<Matrix> {
        let n = self.dim();
        // A⁻¹ = L⁻ᵀ L⁻¹; compute L⁻¹ by forward substitution on I.
        let mut linv = Matrix::identity(n);
        super::triangular::solve_lower_mat(&self.l, &mut linv, false)?;
        let inv = linv.t_matmul(&linv)?;
        Ok(inv.symmetrize())
    }

    pub fn det(&self) -> f64 {
        self.l.diagonal().iter().map(|d| d * d).product()
    }

    /// ln det(A) = 2 Σ ln L_ii (no overflow).
    pub fn log_det(&self) -> f64 {
        2.0 * self.l.diagonal().iter().map(|d| d.ln()).sum::<f64>()
    }
}

/// LDLᵀ factorisation without pivoting (for symmetric matrices with
/// non-zero leading minors, including indefinite ones).
#[derive(Debug, Clone)]
pub struct Ldlt {
    l: Matrix,
    d: Vec<f64>,
}

impl Ldlt {
    pub fn new(a: &Matrix) -> Result<Self> {
        let n = a.require_square()?;
        a.require_finite("LDLt")?;
        let mut l = Matrix::identity(n);
        let mut d = vec![0.0; n];
        let mut w = vec![0.0; n]; // w[k] = L_jk · d_k for the current row j
        for j in 0..n {
            for k in 0..j {
                w[k] = l[(j, k)] * d[k];
            }
            let dj = a[(j, j)] - (0..j).map(|k| l[(j, k)] * w[k]).sum::<f64>();
            if dj == 0.0 || !dj.is_finite() {
                return Err(MathError::Singular { pivot: j });
            }
            d[j] = dj;
            for i in j + 1..n {
                let s = a[(i, j)] - (0..j).map(|k| l[(i, k)] * w[k]).sum::<f64>();
                l[(i, j)] = s / dj;
            }
        }
        Ok(Ldlt { l, d })
    }

    pub fn l(&self) -> &Matrix {
        &self.l
    }
    pub fn d(&self) -> &[f64] {
        &self.d
    }

    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let mut x = b.to_vec();
        super::triangular::solve_lower(&self.l, &mut x, true)?;
        for (xi, di) in x.iter_mut().zip(&self.d) {
            *xi /= di;
        }
        super::triangular::solve_lower_t(&self.l, &mut x, true)?;
        Ok(x)
    }

    pub fn det(&self) -> f64 {
        self.d.iter().product()
    }

    /// (positive, negative, zero) eigenvalue counts (Sylvester's law of inertia).
    pub fn inertia(&self) -> (usize, usize, usize) {
        let p = self.d.iter().filter(|&&v| v > 0.0).count();
        let n = self.d.iter().filter(|&&v| v < 0.0).count();
        (p, n, self.d.len() - p - n)
    }
}

/// Modified Cholesky (Nocedal & Wright, Algorithm 3.3): find τ ≥ 0 such that
/// A + τI is positive definite, starting from τ = 0 if the diagonal is
/// positive and doubling otherwise. Returns the factor and τ. Used to make
/// Newton steps in optimizers descent directions.
pub fn modified_cholesky(a: &Matrix) -> Result<(Cholesky, f64)> {
    let n = a.require_square()?;
    a.require_finite("modified Cholesky")?;
    let scale = a.norm_fro().max(1.0);
    let beta = 1e-3 * scale;
    let min_diag = (0..n).map(|i| a[(i, i)]).fold(f64::INFINITY, f64::min);
    let mut tau = if min_diag > 0.0 { 0.0 } else { beta - min_diag };
    for _ in 0..200 {
        let mut shifted = a.clone();
        for i in 0..n {
            shifted[(i, i)] += tau;
        }
        if let Ok(c) = Cholesky::new(&shifted) {
            return Ok((c, tau));
        }
        tau = (2.0 * tau).max(beta);
    }
    Err(MathError::NoConvergence { algorithm: "modified Cholesky", iterations: 200 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    #[test]
    fn cholesky_spd() {
        for n in [1, 3, 25, 80] {
            let a = spd(n, n as u64);
            let c = Cholesky::new(&a).unwrap();
            assert!(max_abs_diff(&c.l().matmul(&c.l().t()).unwrap(), &a) < 1e-11 * n as f64);
            let x: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
            let sol = c.solve(&a.matvec(&x).unwrap()).unwrap();
            assert!(sol.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-10));
            let inv = c.inverse().unwrap();
            assert!(max_abs_diff(&a.matmul(&inv).unwrap(), &Matrix::identity(n)) < 1e-11);
            let lu = super::super::lu::Lu::new(&a).unwrap();
            assert!((c.log_det() - lu.log_abs_det().0).abs() < 1e-9);
        }
        // Hilbert(10) is SPD but very ill-conditioned: Cholesky still succeeds.
        assert!(Cholesky::new(&hilbert(10)).is_ok());
        let indefinite = Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap();
        assert!(matches!(Cholesky::new(&indefinite), Err(MathError::NotPositiveDefinite { pivot: 1 })));
    }

    #[test]
    fn ldlt_indefinite() {
        let a = Matrix::from_rows(&[vec![4.0, 2.0, -2.0], vec![2.0, -3.0, 1.0], vec![-2.0, 1.0, 5.0]]).unwrap();
        let f = Ldlt::new(&a).unwrap();
        let d = Matrix::diag(f.d());
        let rec = f.l().matmul(&d).unwrap().matmul(&f.l().t()).unwrap();
        assert!(max_abs_diff(&rec, &a) < 1e-14);
        assert_eq!(f.inertia(), (2, 1, 0));
        let x = [1.0, 2.0, 3.0];
        let sol = f.solve(&a.matvec(&x).unwrap()).unwrap();
        assert!(sol.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-13));
        let lu = super::super::lu::Lu::new(&a).unwrap();
        assert!((f.det() - lu.det()).abs() < 1e-12);
        assert!(Ldlt::new(&Matrix::from_rows(&[vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap()).is_err());
    }

    #[test]
    fn modified_cholesky_shifts_only_when_needed() {
        let a = spd(6, 9);
        let (_, tau) = modified_cholesky(&a).unwrap();
        assert_eq!(tau, 0.0);
        let indefinite = Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap(); // eigenvalues 3, −1
        let (c, tau) = modified_cholesky(&indefinite).unwrap();
        assert!(tau > 1.0, "needs a shift beyond the most negative eigenvalue: {tau}");
        let rec = c.l().matmul(&c.l().t()).unwrap();
        assert!((rec[(0, 0)] - (1.0 + tau)).abs() < 1e-12);
    }
}
