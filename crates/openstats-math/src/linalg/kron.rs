//! Kronecker product and vectorisation.

use super::matrix::Matrix;

/// A ⊗ B.
pub fn kron(a: &Matrix, b: &Matrix) -> Matrix {
    let (p, q) = b.shape();
    Matrix::from_fn(a.rows() * p, a.cols() * q, |i, j| a[(i / p, j / q)] * b[(i % p, j % q)])
}

/// Stack columns into a vector (column-major vec operator).
pub fn vec(a: &Matrix) -> Vec<f64> {
    (0..a.cols()).flat_map(|j| a.col(j)).collect()
}

/// Inverse of [`vec()`].
pub fn unvec(v: &[f64], rows: usize, cols: usize) -> Matrix {
    assert_eq!(v.len(), rows * cols, "unvec length");
    Matrix::from_fn(rows, cols, |i, j| v[j * rows + i])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    #[test]
    fn mixed_product_property() {
        // (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD), and vec(AXB) = (Bᵀ ⊗ A) vec(X).
        let (a, b, c, d) = (random(2, 3, 1), random(3, 2, 2), random(3, 2, 3), random(2, 4, 4));
        let lhs = kron(&a, &b).matmul(&kron(&c, &d)).unwrap();
        let rhs = kron(&a.matmul(&c).unwrap(), &b.matmul(&d).unwrap());
        assert!(max_abs_diff(&lhs, &rhs) < 1e-13);
        let x = random(3, 3, 5);
        let b2 = random(3, 2, 6);
        let axb = a.matmul(&x).unwrap().matmul(&b2).unwrap();
        let v = kron(&b2.t(), &a).matvec(&vec(&x)).unwrap();
        assert!(vec(&axb).iter().zip(&v).all(|(p, q)| (p - q).abs() < 1e-13));
        assert_eq!(unvec(&vec(&x), 3, 3), x);
    }
}
