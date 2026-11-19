//! BLAS-style kernels: vector operations, matrix–vector products and a
//! cache-blocked general matrix multiply.

use super::matrix::Matrix;
use crate::error::{MathError, Result};

/// Dot product with four independent accumulators (better pipelining and
/// slightly better rounding behaviour than a single running sum).
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len().min(y.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
    let chunks = n / 4;
    for k in 0..chunks {
        let i = 4 * k;
        s0 += x[i] * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for i in 4 * chunks..n {
        s += x[i] * y[i];
    }
    s
}

/// y ← a·x + y
#[inline]
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// x ← a·x
pub fn scal(a: f64, x: &mut [f64]) {
    x.iter_mut().for_each(|v| *v *= a);
}

/// Euclidean norm without overflow or harmful underflow (scaled sum of
/// squares, as in LAPACK's `dnrm2`).
pub fn nrm2(x: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &v in x {
        if v != 0.0 {
            let a = v.abs();
            if scale < a {
                ssq = 1.0 + ssq * (scale / a) * (scale / a);
                scale = a;
            } else {
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// Index of the entry with the largest absolute value (first on ties).
pub fn iamax(x: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in x.iter().enumerate() {
        if best.is_none_or(|(_, b)| v.abs() > b) {
            best = Some((i, v.abs()));
        }
    }
    best.map(|(i, _)| i)
}

/// y ← α·op(A)·x + β·y
pub fn gemv(alpha: f64, a: &Matrix, trans: bool, x: &[f64], beta: f64, y: &mut [f64]) {
    let (m, n) = a.shape();
    if beta != 1.0 {
        if beta == 0.0 {
            y.iter_mut().for_each(|v| *v = 0.0);
        } else {
            scal(beta, y);
        }
    }
    if !trans {
        for i in 0..m {
            y[i] += alpha * dot(a.row(i), x);
        }
    } else {
        for i in 0..m {
            axpy(alpha * x[i], a.row(i), &mut y[..n]);
        }
    }
}

/// Block sizes for GEMM (rows of A, shared dimension, columns of B). Chosen
/// so that a KC×NC panel of B (256×128 doubles = 256 KiB) fits in L2.
const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 128;

/// C ← α·op(A)·op(B) + β·C, cache-blocked. Transposed operands are packed
/// once into row-major panels so the inner kernel always streams contiguous
/// rows (`c[i, j..] += a[i, k] · b[k, j..]`), which the compiler vectorises.
pub fn gemm(alpha: f64, a: &Matrix, ta: bool, b: &Matrix, tb: bool, beta: f64, c: &mut Matrix) -> Result<()> {
    let (m, ka) = if ta { (a.cols(), a.rows()) } else { a.shape() };
    let (kb, n) = if tb { (b.cols(), b.rows()) } else { b.shape() };
    if ka != kb || c.shape() != (m, n) {
        return Err(MathError::DimensionMismatch { op: "gemm", left: (m, ka), right: (kb, n) });
    }
    let k = ka;
    if beta != 1.0 {
        let cs = c.as_mut_slice();
        if beta == 0.0 {
            cs.iter_mut().for_each(|v| *v = 0.0);
        } else {
            scal(beta, cs);
        }
    }
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return Ok(());
    }
    let a_owned;
    let a = if ta {
        a_owned = a.t();
        &a_owned
    } else {
        a
    };
    let b_owned;
    let b = if tb {
        b_owned = b.t();
        &b_owned
    } else {
        b
    };
    let (ad, bd) = (a.as_slice(), b.as_slice());
    let cd = c.as_mut_slice();
    for jc in (0..n).step_by(NC) {
        let je = (jc + NC).min(n);
        for pc in (0..k).step_by(KC) {
            let pe = (pc + KC).min(k);
            for ic in (0..m).step_by(MC) {
                let ie = (ic + MC).min(m);
                for i in ic..ie {
                    let crow = &mut cd[i * n + jc..i * n + je];
                    let arow = &ad[i * k..(i + 1) * k];
                    for p in pc..pe {
                        let aip = alpha * arow[p];
                        if aip == 0.0 {
                            continue;
                        }
                        let brow = &bd[p * n + jc..p * n + je];
                        for (cv, &bv) in crow.iter_mut().zip(brow) {
                            *cv += aip * bv;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Gram matrix AᵀA for an m×n matrix, computing only the upper triangle and
/// mirroring it (half the flops of a general product).
pub fn syrk_t(a: &Matrix) -> Matrix {
    let (m, n) = a.shape();
    let mut g = Matrix::zeros(n, n);
    for r in 0..m {
        let row = a.row(r);
        for i in 0..n {
            let ri = row[i];
            if ri == 0.0 {
                continue;
            }
            let gi = g.row_mut(i);
            for j in i..n {
                gi[j] += ri * row[j];
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            g[(i, j)] = g[(j, i)];
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    #[test]
    fn vector_kernels() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(dot(&x, &x), 55.0);
        let mut y = [1.0; 5];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, [3.0, 5.0, 7.0, 9.0, 11.0]);
        assert_eq!(nrm2(&[3.0, 4.0]), 5.0);
        assert_eq!(nrm2(&[1e300, 1e300]), 1e300 * 2f64.sqrt());
        assert!((nrm2(&[1e-300, 1e-300]) - 1e-300 * 2f64.sqrt()).abs() < 1e-315);
        assert_eq!(nrm2(&[]), 0.0);
        assert_eq!(iamax(&[1.0, -7.0, 7.0]), Some(1));
        assert_eq!(iamax(&[]), None);
    }

    #[test]
    fn gemm_all_transposes_and_scalars() {
        let a = random(70, 300, 3); // spans several KC blocks
        let b = random(300, 150, 4); // spans NC blocks
        let reference = Matrix::from_fn(70, 150, |i, j| (0..300).map(|k| a[(i, k)] * b[(k, j)]).sum());
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            let aa = if ta { a.t() } else { a.clone() };
            let bb = if tb { b.t() } else { b.clone() };
            let mut c = Matrix::filled(70, 150, 1.0);
            gemm(2.0, &aa, ta, &bb, tb, -1.0, &mut c).unwrap();
            let expect = Matrix::from_fn(70, 150, |i, j| 2.0 * reference[(i, j)] - 1.0);
            assert!(max_abs_diff(&c, &expect) < 1e-11, "ta={ta} tb={tb}");
        }
        let mut wrong = Matrix::zeros(3, 3);
        assert!(gemm(1.0, &a, false, &b, false, 0.0, &mut wrong).is_err());
        let mut y = vec![1.0; 70];
        gemv(1.0, &a, false, &vec![1.0; 300], 2.0, &mut y);
        assert!((y[0] - (2.0 + a.row(0).iter().sum::<f64>())).abs() < 1e-12);
    }
}
