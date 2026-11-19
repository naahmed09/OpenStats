//! Triangular solves.

use super::matrix::Matrix;
use crate::error::{MathError, Result};

/// Solve L·x = b in place for lower-triangular L (unit diagonal if `unit`).
pub fn solve_lower(l: &Matrix, b: &mut [f64], unit: bool) -> Result<()> {
    let n = l.require_square()?;
    if b.len() != n {
        return Err(MathError::DimensionMismatch { op: "solve_lower", left: l.shape(), right: (b.len(), 1) });
    }
    for i in 0..n {
        let row = l.row(i);
        let mut s = b[i];
        for j in 0..i {
            s -= row[j] * b[j];
        }
        if unit {
            b[i] = s;
        } else {
            if row[i] == 0.0 {
                return Err(MathError::Singular { pivot: i });
            }
            b[i] = s / row[i];
        }
    }
    Ok(())
}

/// Solve U·x = b in place for upper-triangular U.
pub fn solve_upper(u: &Matrix, b: &mut [f64], unit: bool) -> Result<()> {
    let n = u.require_square()?;
    if b.len() != n {
        return Err(MathError::DimensionMismatch { op: "solve_upper", left: u.shape(), right: (b.len(), 1) });
    }
    for i in (0..n).rev() {
        let row = u.row(i);
        let mut s = b[i];
        for j in i + 1..n {
            s -= row[j] * b[j];
        }
        if unit {
            b[i] = s;
        } else {
            if row[i] == 0.0 {
                return Err(MathError::Singular { pivot: i });
            }
            b[i] = s / row[i];
        }
    }
    Ok(())
}

/// Solve Lᵀ·x = b in place for lower-triangular L (column-oriented on L).
pub fn solve_lower_t(l: &Matrix, b: &mut [f64], unit: bool) -> Result<()> {
    let n = l.require_square()?;
    if b.len() != n {
        return Err(MathError::DimensionMismatch { op: "solve_lower_t", left: l.shape(), right: (b.len(), 1) });
    }
    for i in (0..n).rev() {
        if !unit {
            if l[(i, i)] == 0.0 {
                return Err(MathError::Singular { pivot: i });
            }
            b[i] /= l[(i, i)];
        }
        let bi = b[i];
        let row = l.row(i);
        for j in 0..i {
            b[j] -= row[j] * bi;
        }
    }
    Ok(())
}

/// Solve for every column of `b` (overwritten): L·X = B.
pub fn solve_lower_mat(l: &Matrix, b: &mut Matrix, unit: bool) -> Result<()> {
    for j in 0..b.cols() {
        let mut col = b.col(j);
        solve_lower(l, &mut col, unit)?;
        b.set_col(j, &col);
    }
    Ok(())
}

/// Solve for every column of `b` (overwritten): U·X = B.
pub fn solve_upper_mat(u: &Matrix, b: &mut Matrix, unit: bool) -> Result<()> {
    for j in 0..b.cols() {
        let mut col = b.col(j);
        solve_upper(u, &mut col, unit)?;
        b.set_col(j, &col);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutions() {
        let l = Matrix::from_rows(&[vec![2.0, 0.0, 0.0], vec![1.0, 3.0, 0.0], vec![-1.0, 2.0, 4.0]]).unwrap();
        let x = [1.0, -2.0, 3.0];
        let mut b = l.matvec(&x).unwrap();
        solve_lower(&l, &mut b, false).unwrap();
        assert!(b.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-15));
        let u = l.t();
        let mut b = u.matvec(&x).unwrap();
        solve_upper(&u, &mut b, false).unwrap();
        assert!(b.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-15));
        let mut b = u.matvec(&x).unwrap();
        solve_lower_t(&l, &mut b, false).unwrap();
        assert!(b.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-15));
        let mut sing = l.clone();
        sing[(1, 1)] = 0.0;
        assert!(matches!(solve_lower(&sing, &mut [1.0, 1.0, 1.0], false), Err(MathError::Singular { pivot: 1 })));
        let mut bm = Matrix::identity(3);
        solve_lower_mat(&l, &mut bm, false).unwrap();
        assert!(l.matmul(&bm).unwrap().approx_eq(&Matrix::identity(3), 1e-15));
    }
}
