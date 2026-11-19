//! LU factorisation with partial (row) pivoting: P·A = L·U.

use super::matrix::Matrix;
use crate::error::{MathError, Result};

/// LU factors of a square matrix.
#[derive(Debug, Clone)]
pub struct Lu {
    /// Unit-lower L (strictly below the diagonal) and U (on and above).
    lu: Matrix,
    /// Row `i` of P·A is row `piv[i]` of A.
    piv: Vec<usize>,
    sign: f64,
    /// First zero pivot, if any.
    singular_at: Option<usize>,
    norm1: f64,
}

impl Lu {
    /// Factorise. Never fails on singular input (that is reported by
    /// [`Lu::is_singular`] and by the solvers); fails on non-square or
    /// non-finite input.
    pub fn new(a: &Matrix) -> Result<Self> {
        let n = a.require_square()?;
        a.require_finite("LU")?;
        let norm1 = a.norm_1();
        let mut lu = a.clone();
        let mut piv: Vec<usize> = (0..n).collect();
        let mut sign = 1.0;
        let mut singular_at = None;
        for k in 0..n {
            // Pivot: largest |a_ik| for i ≥ k.
            let mut p = k;
            let mut best = lu[(k, k)].abs();
            for i in k + 1..n {
                let v = lu[(i, k)].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if p != k {
                lu.swap_rows(p, k);
                piv.swap(p, k);
                sign = -sign;
            }
            let pivot = lu[(k, k)];
            if pivot == 0.0 {
                singular_at.get_or_insert(k);
                continue;
            }
            let (top, rest) = lu.as_mut_slice().split_at_mut((k + 1) * n);
            let prow = &top[k * n + k + 1..k * n + n];
            for i in 0..n - k - 1 {
                let row = &mut rest[i * n..(i + 1) * n];
                let f = row[k] / pivot;
                row[k] = f;
                if f != 0.0 {
                    for (rj, &pj) in row[k + 1..].iter_mut().zip(prow) {
                        *rj -= f * pj;
                    }
                }
            }
        }
        Ok(Lu { lu, piv, sign, singular_at, norm1 })
    }

    pub fn dim(&self) -> usize {
        self.lu.rows()
    }

    pub fn is_singular(&self) -> bool {
        self.singular_at.is_some()
    }

    /// Row permutation: row `i` of P·A is row `perm()[i]` of A.
    pub fn perm(&self) -> &[usize] {
        &self.piv
    }

    pub fn l(&self) -> Matrix {
        let n = self.dim();
        Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
            std::cmp::Ordering::Greater => self.lu[(i, j)],
            std::cmp::Ordering::Equal => 1.0,
            _ => 0.0,
        })
    }

    pub fn u(&self) -> Matrix {
        let n = self.dim();
        Matrix::from_fn(n, n, |i, j| if i <= j { self.lu[(i, j)] } else { 0.0 })
    }

    /// Permutation matrix P.
    pub fn p(&self) -> Matrix {
        let n = self.dim();
        let mut p = Matrix::zeros(n, n);
        for (i, &r) in self.piv.iter().enumerate() {
            p[(i, r)] = 1.0;
        }
        p
    }

    pub fn det(&self) -> f64 {
        if self.is_singular() {
            return 0.0;
        }
        self.sign * self.lu.diagonal().iter().product::<f64>()
    }

    /// ln|det| and the sign of the determinant (robust for large matrices).
    pub fn log_abs_det(&self) -> (f64, f64) {
        if self.is_singular() {
            return (f64::NEG_INFINITY, 0.0);
        }
        let mut s = self.sign;
        let mut l = 0.0;
        for d in self.lu.diagonal() {
            if d < 0.0 {
                s = -s;
            }
            l += d.abs().ln();
        }
        (l, s)
    }

    fn check(&self, len: usize, op: &'static str) -> Result<()> {
        if let Some(p) = self.singular_at {
            return Err(MathError::Singular { pivot: p });
        }
        if len != self.dim() {
            return Err(MathError::DimensionMismatch { op, left: self.lu.shape(), right: (len, 1) });
        }
        Ok(())
    }

    /// Solve A·x = b.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        self.check(b.len(), "LU solve")?;
        let n = self.dim();
        let mut x: Vec<f64> = self.piv.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let row = self.lu.row(i);
            let mut s = x[i];
            for j in 0..i {
                s -= row[j] * x[j];
            }
            x[i] = s;
        }
        for i in (0..n).rev() {
            let row = self.lu.row(i);
            let mut s = x[i];
            for j in i + 1..n {
                s -= row[j] * x[j];
            }
            x[i] = s / row[i];
        }
        Ok(x)
    }

    /// Solve Aᵀ·x = b.
    pub fn solve_transpose(&self, b: &[f64]) -> Result<Vec<f64>> {
        self.check(b.len(), "LU solve_transpose")?;
        let n = self.dim();
        // Aᵀ = Uᵀ·Lᵀ·P  ⇒ solve Uᵀw = b, Lᵀv = w, x = Pᵀv.
        let mut w = b.to_vec();
        for i in 0..n {
            w[i] /= self.lu[(i, i)];
            let wi = w[i];
            let row = self.lu.row(i);
            for j in i + 1..n {
                w[j] -= row[j] * wi;
            }
        }
        for i in (0..n).rev() {
            let wi = w[i];
            let row = self.lu.row(i);
            for j in 0..i {
                w[j] -= row[j] * wi;
            }
        }
        let mut x = vec![0.0; n];
        for (i, &p) in self.piv.iter().enumerate() {
            x[p] = w[i];
        }
        Ok(x)
    }

    /// Solve A·X = B for every column of B.
    pub fn solve_mat(&self, b: &Matrix) -> Result<Matrix> {
        let mut out = Matrix::zeros(b.rows(), b.cols());
        for j in 0..b.cols() {
            out.set_col(j, &self.solve(&b.col(j))?);
        }
        Ok(out)
    }

    pub fn inverse(&self) -> Result<Matrix> {
        self.solve_mat(&Matrix::identity(self.dim()))
    }

    /// Reciprocal 1-norm condition number estimate (Hager / Higham), as in
    /// LAPACK `dgecon`: 1 / (‖A‖₁ · est‖A⁻¹‖₁). Returns 0 for singular input.
    pub fn rcond(&self) -> f64 {
        let n = self.dim();
        if self.is_singular() || n == 0 {
            return if n == 0 { 1.0 } else { 0.0 };
        }
        if self.norm1 == 0.0 {
            return 0.0;
        }
        let mut x = vec![1.0 / n as f64; n];
        let mut est = 0.0;
        for iter in 0..5 {
            let y = self.solve(&x).expect("nonsingular");
            let new_est: f64 = y.iter().map(|v| v.abs()).sum();
            if iter > 0 && new_est <= est {
                break;
            }
            est = new_est;
            let xi: Vec<f64> = y.iter().map(|&v| if v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let z = self.solve_transpose(&xi).expect("nonsingular");
            let (j, zmax) = z
                .iter()
                .enumerate()
                .fold((0, 0.0f64), |(bj, bm), (i, &v)| if v.abs() > bm { (i, v.abs()) } else { (bj, bm) });
            let zx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
            if zmax <= zx {
                break;
            }
            x = vec![0.0; n];
            x[j] = 1.0;
        }
        // Higham's alternative estimate guards against bad cases.
        let alt: Vec<f64> = (0..n)
            .map(|i| (if i % 2 == 0 { 1.0 } else { -1.0 }) * (1.0 + i as f64 / (n as f64 - 1.0).max(1.0)))
            .collect();
        let y = self.solve(&alt).expect("nonsingular");
        let alt_est = 2.0 * y.iter().map(|v| v.abs()).sum::<f64>() / (3.0 * n as f64);
        let inv_norm = est.max(alt_est);
        1.0 / (self.norm1 * inv_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    #[test]
    fn factor_and_solve() {
        for n in [1, 2, 5, 40, 120] {
            let a = random(n, n, n as u64);
            let lu = Lu::new(&a).unwrap();
            let pa = lu.p().matmul(&a).unwrap();
            assert!(max_abs_diff(&pa, &lu.l().matmul(&lu.u()).unwrap()) < 1e-12 * n as f64);
            let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let b = a.matvec(&x).unwrap();
            let sol = lu.solve(&b).unwrap();
            assert!(sol.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-9), "n={n}");
            let bt = a.t_matvec(&x).unwrap();
            let solt = lu.solve_transpose(&bt).unwrap();
            assert!(solt.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-9), "transpose n={n}");
            let inv = lu.inverse().unwrap();
            assert!(max_abs_diff(&a.matmul(&inv).unwrap(), &Matrix::identity(n)) < 1e-9);
        }
    }

    #[test]
    fn determinant_and_condition() {
        let a = Matrix::from_rows(&[vec![2.0, 1.0, 1.0], vec![4.0, -6.0, 0.0], vec![-2.0, 7.0, 2.0]]).unwrap();
        let lu = Lu::new(&a).unwrap();
        assert!((lu.det() - -16.0).abs() < 1e-13);
        let (l, s) = lu.log_abs_det();
        assert!((s * l.exp() - -16.0).abs() < 1e-12);
        // rcond of the identity is 1; of Hilbert(8) ≈ 1/(3.387e10) (1-norm).
        assert!((Lu::new(&Matrix::identity(5)).unwrap().rcond() - 1.0).abs() < 1e-15);
        let h = Lu::new(&hilbert(8)).unwrap().rcond();
        let exact = 1.0 / 3.386_872_183e10;
        assert!(h > exact / 3.0 && h < exact * 3.0, "rcond {h:e}");
    }

    #[test]
    fn singular_and_errors() {
        let s = Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
        let lu = Lu::new(&s).unwrap();
        assert!(lu.is_singular());
        assert_eq!(lu.det(), 0.0);
        assert_eq!(lu.rcond(), 0.0);
        assert!(matches!(lu.solve(&[1.0, 1.0]), Err(MathError::Singular { .. })));
        assert!(Lu::new(&Matrix::zeros(2, 3)).is_err());
        assert!(Lu::new(&Matrix::filled(2, 2, f64::NAN)).is_err());
    }
}
