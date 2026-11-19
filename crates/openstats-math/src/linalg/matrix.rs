//! Dense row-major matrix.

use super::blas;
use crate::error::{MathError, Result};
use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// A dense `rows × cols` matrix stored row-major.
#[derive(Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn filled(rows: usize, cols: usize, v: f64) -> Self {
        Matrix { rows, cols, data: vec![v; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// From row-major data.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidArgument(format!("{} values for a {rows}×{cols} matrix", data.len())));
        }
        Ok(Matrix { rows, cols, data })
    }

    /// From a slice of rows (all of equal length).
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let r = rows.len();
        let c = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != c) {
            return Err(MathError::InvalidArgument("rows have different lengths".into()));
        }
        Ok(Matrix { rows: r, cols: c, data: rows.concat() })
    }

    /// From column vectors (all of equal length).
    pub fn from_columns(cols: &[Vec<f64>]) -> Result<Self> {
        let c = cols.len();
        let r = cols.first().map_or(0, Vec::len);
        if cols.iter().any(|col| col.len() != r) {
            return Err(MathError::InvalidArgument("columns have different lengths".into()));
        }
        Ok(Self::from_fn(r, c, |i, j| cols[j][i]))
    }

    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Matrix { rows, cols, data }
    }

    pub fn diag(d: &[f64]) -> Self {
        let mut m = Self::zeros(d.len(), d.len());
        for (i, &v) in d.iter().enumerate() {
            m[(i, i)] = v;
        }
        m
    }

    /// Column vector.
    pub fn column(v: &[f64]) -> Self {
        Matrix { rows: v.len(), cols: 1, data: v.to_vec() }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }
    pub fn set_col(&mut self, j: usize, v: &[f64]) {
        assert_eq!(v.len(), self.rows);
        for (i, &x) in v.iter().enumerate() {
            self.data[i * self.cols + j] = x;
        }
    }
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.rows.min(self.cols)).map(|i| self[(i, i)]).collect()
    }

    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            for j in 0..self.cols {
                self.data.swap(a * self.cols + j, b * self.cols + j);
            }
        }
    }
    pub fn swap_cols(&mut self, a: usize, b: usize) {
        if a != b {
            for i in 0..self.rows {
                self.data.swap(i * self.cols + a, i * self.cols + b);
            }
        }
    }

    /// Transpose (cache-blocked copy).
    pub fn t(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        const B: usize = 32;
        for ib in (0..self.rows).step_by(B) {
            for jb in (0..self.cols).step_by(B) {
                for i in ib..(ib + B).min(self.rows) {
                    for j in jb..(jb + B).min(self.cols) {
                        out.data[j * self.rows + i] = self.data[i * self.cols + j];
                    }
                }
            }
        }
        out
    }

    /// Rows `r0..r1`, columns `c0..c1`.
    pub fn submatrix(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Matrix {
        assert!(r0 <= r1 && r1 <= self.rows && c0 <= c1 && c1 <= self.cols, "submatrix out of range");
        Matrix::from_fn(r1 - r0, c1 - c0, |i, j| self[(r0 + i, c0 + j)])
    }

    /// Selected rows and columns, in the given order.
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> Matrix {
        Matrix::from_fn(rows.len(), cols.len(), |i, j| self[(rows[i], cols[j])])
    }

    /// Write `block` with its top-left corner at (r0, c0).
    pub fn set_submatrix(&mut self, r0: usize, c0: usize, block: &Matrix) {
        assert!(r0 + block.rows <= self.rows && c0 + block.cols <= self.cols, "block out of range");
        for i in 0..block.rows {
            self.row_mut(r0 + i)[c0..c0 + block.cols].copy_from_slice(block.row(i));
        }
    }

    /// `[self | other]`
    pub fn hstack(&self, other: &Matrix) -> Result<Matrix> {
        if self.rows != other.rows {
            return Err(MathError::DimensionMismatch { op: "hstack", left: self.shape(), right: other.shape() });
        }
        let mut m = Matrix::zeros(self.rows, self.cols + other.cols);
        m.set_submatrix(0, 0, self);
        m.set_submatrix(0, self.cols, other);
        Ok(m)
    }

    /// `[self ; other]`
    pub fn vstack(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.cols {
            return Err(MathError::DimensionMismatch { op: "vstack", left: self.shape(), right: other.shape() });
        }
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        Ok(Matrix { rows: self.rows + other.rows, cols: self.cols, data })
    }

    /// Matrix product (cache-blocked GEMM).
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch { op: "matmul", left: self.shape(), right: other.shape() });
        }
        let mut c = Matrix::zeros(self.rows, other.cols);
        blas::gemm(1.0, self, false, other, false, 0.0, &mut c)?;
        Ok(c)
    }

    /// `selfᵀ · other` without forming the transpose explicitly.
    pub fn t_matmul(&self, other: &Matrix) -> Result<Matrix> {
        if self.rows != other.rows {
            return Err(MathError::DimensionMismatch { op: "t_matmul", left: self.shape(), right: other.shape() });
        }
        let mut c = Matrix::zeros(self.cols, other.cols);
        blas::gemm(1.0, self, true, other, false, 0.0, &mut c)?;
        Ok(c)
    }

    /// Matrix–vector product.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if self.cols != x.len() {
            return Err(MathError::DimensionMismatch { op: "matvec", left: self.shape(), right: (x.len(), 1) });
        }
        let mut y = vec![0.0; self.rows];
        blas::gemv(1.0, self, false, x, 0.0, &mut y);
        Ok(y)
    }

    /// `selfᵀ · x`.
    pub fn t_matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if self.rows != x.len() {
            return Err(MathError::DimensionMismatch { op: "t_matvec", left: self.shape(), right: (x.len(), 1) });
        }
        let mut y = vec![0.0; self.cols];
        blas::gemv(1.0, self, true, x, 0.0, &mut y);
        Ok(y)
    }

    /// Gram matrix `selfᵀ · self` (symmetric, computed once per pair).
    pub fn gram(&self) -> Matrix {
        blas::syrk_t(self)
    }

    pub fn scale(&self, s: f64) -> Matrix {
        Matrix { rows: self.rows, cols: self.cols, data: self.data.iter().map(|x| x * s).collect() }
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Matrix {
        Matrix { rows: self.rows, cols: self.cols, data: self.data.iter().map(|&x| f(x)).collect() }
    }

    fn zip_with(&self, o: &Matrix, op: &'static str, f: impl Fn(f64, f64) -> f64) -> Result<Matrix> {
        if self.shape() != o.shape() {
            return Err(MathError::DimensionMismatch { op, left: self.shape(), right: o.shape() });
        }
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(&o.data).map(|(&a, &b)| f(a, b)).collect(),
        })
    }

    pub fn try_add(&self, o: &Matrix) -> Result<Matrix> {
        self.zip_with(o, "add", |a, b| a + b)
    }
    pub fn try_sub(&self, o: &Matrix) -> Result<Matrix> {
        self.zip_with(o, "sub", |a, b| a - b)
    }
    /// Element-wise (Hadamard) product.
    pub fn hadamard(&self, o: &Matrix) -> Result<Matrix> {
        self.zip_with(o, "hadamard", |a, b| a * b)
    }

    pub fn trace(&self) -> f64 {
        self.diagonal().iter().sum()
    }

    /// Frobenius norm (overflow-safe).
    pub fn norm_fro(&self) -> f64 {
        blas::nrm2(&self.data)
    }
    /// Maximum absolute column sum.
    pub fn norm_1(&self) -> f64 {
        (0..self.cols).map(|j| (0..self.rows).map(|i| self[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max)
    }
    /// Maximum absolute row sum.
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows).map(|i| self.row(i).iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max)
    }
    /// Largest absolute entry.
    pub fn norm_max(&self) -> f64 {
        self.data.iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    /// `true` if symmetric within `tol · max|a_ij|`.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        let scale = self.norm_max().max(f64::MIN_POSITIVE);
        (0..self.rows).all(|i| (0..i).all(|j| (self[(i, j)] - self[(j, i)]).abs() <= tol * scale))
    }

    /// Symmetrise: (A + Aᵀ)/2.
    pub fn symmetrize(&self) -> Matrix {
        Matrix::from_fn(self.rows, self.cols, |i, j| 0.5 * (self[(i, j)] + self[(j, i)]))
    }

    pub fn approx_eq(&self, o: &Matrix, tol: f64) -> bool {
        self.shape() == o.shape() && self.data.iter().zip(&o.data).all(|(a, b)| (a - b).abs() <= tol)
    }

    pub(crate) fn require_square(&self) -> Result<usize> {
        if self.is_square() {
            Ok(self.rows)
        } else {
            Err(MathError::NotSquare { rows: self.rows, cols: self.cols })
        }
    }

    pub(crate) fn require_finite(&self, op: &'static str) -> Result<()> {
        if self.is_finite() {
            Ok(())
        } else {
            Err(MathError::NonFinite { op })
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        debug_assert!(i < self.rows && j < self.cols, "index ({i}, {j}) out of range for {}×{}", self.rows, self.cols);
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        debug_assert!(i < self.rows && j < self.cols, "index ({i}, {j}) out of range for {}×{}", self.rows, self.cols);
        &mut self.data[i * self.cols + j]
    }
}

/// Panicking operator forms (use the `try_*` methods for fallible versions).
impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, o: &Matrix) -> Matrix {
        self.try_add(o).expect("matrix add")
    }
}
impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, o: &Matrix) -> Matrix {
        self.try_sub(o).expect("matrix sub")
    }
}
impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, o: &Matrix) -> Matrix {
        self.matmul(o).expect("matrix mul")
    }
}
impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, s: f64) -> Matrix {
        self.scale(s)
    }
}
impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        self.scale(-1.0)
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix {}×{} [", self.rows, self.cols)?;
        for i in 0..self.rows.min(12) {
            let row: Vec<String> = self.row(i).iter().take(12).map(|x| format!("{x:12.6e}")).collect();
            writeln!(f, "  {}{}", row.join(" "), if self.cols > 12 { " …" } else { "" })?;
        }
        if self.rows > 12 {
            writeln!(f, "  …")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    #[test]
    fn construction_and_access() {
        let m = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m[(1, 2)], 6.0);
        assert_eq!(m.col(1), vec![2.0, 5.0]);
        assert_eq!(m.t().row(2), &[3.0, 6.0]);
        assert_eq!(Matrix::from_columns(&[vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]).unwrap(), m);
        assert!(Matrix::from_rows(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        assert!(Matrix::from_vec(2, 2, vec![1.0]).is_err());
        assert_eq!(m.submatrix(0, 2, 1, 3).as_slice(), &[2.0, 3.0, 5.0, 6.0]);
        assert_eq!(m.select(&[1], &[2, 0]).as_slice(), &[6.0, 4.0]);
        let h = m.hstack(&Matrix::identity(2)).unwrap();
        assert_eq!(h.shape(), (2, 5));
        assert_eq!(m.vstack(&m).unwrap().rows(), 4);
        assert!(m.hstack(&Matrix::zeros(3, 1)).is_err());
    }

    #[test]
    fn arithmetic_and_norms() {
        let a = random(37, 53, 1);
        let b = random(53, 29, 2);
        let c = a.matmul(&b).unwrap();
        let naive = Matrix::from_fn(37, 29, |i, j| (0..53).map(|k| a[(i, k)] * b[(k, j)]).sum());
        assert!(max_abs_diff(&c, &naive) < 1e-13);
        assert!(max_abs_diff(&a.t_matmul(&a).unwrap(), &a.gram()) < 1e-13);
        assert!(a.gram().is_symmetric(0.0));
        let x: Vec<f64> = (0..53).map(|i| i as f64).collect();
        let y = a.matvec(&x).unwrap();
        assert!((y[5] - (0..53).map(|k| a[(5, k)] * x[k]).sum::<f64>()).abs() < 1e-12);
        let z = a.t_matvec(&y).unwrap();
        assert_eq!(z.len(), 53);
        let m = Matrix::from_rows(&[vec![1.0, -2.0], vec![-3.0, 4.0]]).unwrap();
        assert_eq!(m.norm_1(), 6.0);
        assert_eq!(m.norm_inf(), 7.0);
        assert_eq!(m.norm_max(), 4.0);
        assert!((m.norm_fro() - 30f64.sqrt()).abs() < 1e-15);
        assert_eq!(m.trace(), 5.0);
        assert_eq!((&m + &m)[(1, 1)], 8.0);
        assert_eq!((&m * &Matrix::identity(2)), m);
        assert!(a.matmul(&a).is_err());
        assert!(Matrix::filled(2, 2, 1e300).norm_fro().is_finite());
    }
}
