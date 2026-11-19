//! Dense and sparse linear algebra.
//!
//! | module | contents |
//! |---|---|
//! | [`matrix`] | row-major [`Matrix`] type, arithmetic, norms, slicing |
//! | [`blas`] | vector kernels, matrix–vector products, cache-blocked GEMM, `AᵀA` |
//! | [`triangular`] | forward/back substitution |
//! | [`householder`] | Householder reflectors (shared by QR, tridiagonal and Hessenberg reductions) |
//! | [`lu`] | LU with partial pivoting: solve, determinant, inverse, condition estimate |
//! | [`qr`] | Householder QR with optional column pivoting, rank, least squares |
//! | [`cholesky`] | Cholesky LLᵀ, LDLᵀ, modified Cholesky |
//! | [`eigen_sym`] | symmetric eigensolvers: tridiagonal QL (default) and cyclic Jacobi |
//! | [`eigen_gen`] | general real eigensolver: Hessenberg + Francis double-shift QR |
//! | [`svd`] | Golub–Kahan–Reinsch SVD, pseudo-inverse, minimum-norm least squares, rank, condition number |
//! | [`sparse`] | triplet/CSR/CSC matrices, minimum-degree sparse Cholesky, preconditioned CG |
//! | [`kron`] | Kronecker product, vec/unvec |

pub mod blas;
pub mod cholesky;
pub mod eigen_gen;
pub mod eigen_sym;
pub mod householder;
pub mod kron;
pub mod lu;
pub mod matrix;
pub mod qr;
pub mod sparse;
pub mod svd;
pub mod triangular;

pub use cholesky::{Cholesky, Ldlt};
pub use eigen_gen::Eigen;
pub use eigen_sym::SymmetricEigen;
pub use lu::Lu;
pub use matrix::Matrix;
pub use qr::Qr;
pub use svd::Svd;

/// Deterministic test matrices shared by the unit tests.
#[cfg(test)]
pub(crate) mod testutil {
    use super::Matrix;

    /// xorshift64* uniform values in [-1, 1).
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> f64 {
            let mut s = self.0;
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            self.0 = s;
            let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
            v as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    pub fn random(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        Matrix::from_fn(rows, cols, |_, _| r.next())
    }

    /// Random symmetric positive-definite matrix: BᵀB + n·I.
    pub fn spd(n: usize, seed: u64) -> Matrix {
        let b = random(n, n, seed);
        let mut a = b.t().matmul(&b).unwrap();
        for i in 0..n {
            a[(i, i)] += n as f64;
        }
        a
    }

    pub fn hilbert(n: usize) -> Matrix {
        Matrix::from_fn(n, n, |i, j| 1.0 / (i + j + 1) as f64)
    }

    pub fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
        assert_eq!(a.shape(), b.shape());
        a.as_slice().iter().zip(b.as_slice()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }
}
