//! Symmetric eigendecomposition A = V·diag(λ)·Vᵀ.
//!
//! * [`SymmetricEigen::new`]: Householder tridiagonalisation followed by the
//!   implicit QL algorithm with Wilkinson-type shifts (EISPACK `tred2`/`tql2`)
//!   — O(n³), the default.
//! * [`SymmetricEigen::jacobi`]: cyclic Jacobi rotations with threshold
//!   (Rutishauser) — slower but delivers small eigenvalues to high relative
//!   accuracy for well-scaled positive-definite matrices.
//!
//! Eigenvalues are returned in ascending order with orthonormal eigenvectors
//! in the corresponding columns of V.

use super::matrix::Matrix;
use crate::error::{MathError, Result};

#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    pub values: Vec<f64>,
    pub vectors: Matrix,
}

impl SymmetricEigen {
    /// Tridiagonal QL (reads the lower triangle; the matrix is symmetrised).
    pub fn new(a: &Matrix) -> Result<Self> {
        let n = a.require_square()?;
        a.require_finite("symmetric eigen")?;
        let mut v = a.symmetrize();
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        if n == 0 {
            return Ok(SymmetricEigen { values: d, vectors: v });
        }
        tred2(&mut v, &mut d, &mut e);
        tql2(&mut v, &mut d, &mut e)?;
        Ok(sorted(d, v))
    }

    /// Cyclic Jacobi.
    pub fn jacobi(a: &Matrix) -> Result<Self> {
        let n = a.require_square()?;
        a.require_finite("Jacobi eigen")?;
        let mut m = a.symmetrize();
        let mut v = Matrix::identity(n);
        const MAX_SWEEPS: usize = 100;
        for sweep in 0..MAX_SWEEPS {
            let off: f64 = (0..n).flat_map(|i| (0..i).map(move |j| (i, j))).map(|(i, j)| m[(i, j)] * m[(i, j)]).sum();
            let diag: f64 = (0..n).map(|i| m[(i, i)] * m[(i, i)]).sum();
            if off <= f64::EPSILON * f64::EPSILON * diag.max(f64::MIN_POSITIVE) || off == 0.0 {
                let d = (0..n).map(|i| m[(i, i)]).collect();
                return Ok(sorted(d, v));
            }
            // Threshold: skip tiny rotations in the first sweeps.
            let thresh = if sweep < 3 { 0.2 * off.sqrt() / (n * n) as f64 } else { 0.0 };
            for p in 0..n {
                for q in p + 1..n {
                    let apq = m[(p, q)];
                    if apq.abs() <= thresh || apq == 0.0 {
                        continue;
                    }
                    let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / t.hypot(1.0);
                    let s = t * c;
                    for k in 0..n {
                        let (mkp, mkq) = (m[(k, p)], m[(k, q)]);
                        m[(k, p)] = c * mkp - s * mkq;
                        m[(k, q)] = s * mkp + c * mkq;
                    }
                    for k in 0..n {
                        let (mpk, mqk) = (m[(p, k)], m[(q, k)]);
                        m[(p, k)] = c * mpk - s * mqk;
                        m[(q, k)] = s * mpk + c * mqk;
                    }
                    m[(p, q)] = 0.0;
                    m[(q, p)] = 0.0;
                    for k in 0..n {
                        let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
                        v[(k, p)] = c * vkp - s * vkq;
                        v[(k, q)] = s * vkp + c * vkq;
                    }
                }
            }
        }
        Err(MathError::NoConvergence { algorithm: "Jacobi eigen", iterations: MAX_SWEEPS })
    }

    /// Reconstruct V·diag(λ)·Vᵀ.
    pub fn reconstruct(&self) -> Matrix {
        let n = self.values.len();
        let vd = Matrix::from_fn(n, n, |i, j| self.vectors[(i, j)] * self.values[j]);
        vd.matmul(&self.vectors.t()).expect("square")
    }

    /// Smallest and largest eigenvalue magnitude ratio (2-norm condition
    /// number for SPD matrices).
    pub fn condition(&self) -> f64 {
        let max = self.values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let min = self.values.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
        max / min
    }
}

fn sorted(d: Vec<f64>, v: Matrix) -> SymmetricEigen {
    let n = d.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let values = idx.iter().map(|&i| d[i]).collect();
    let mut vectors = Matrix::from_fn(n, n, |r, c| v[(r, idx[c])]);
    // Deterministic signs: largest-magnitude component of each vector positive.
    for c in 0..n {
        let col = vectors.col(c);
        let big = col.iter().fold(0.0f64, |b, &x| if x.abs() > b.abs() { x } else { b });
        if big < 0.0 {
            for r in 0..n {
                vectors[(r, c)] = -vectors[(r, c)];
            }
        }
    }
    SymmetricEigen { values, vectors }
}

/// Householder reduction to tridiagonal form (EISPACK tred2). On exit `v`
/// holds the orthogonal transformation, `d` the diagonal and `e[1..]` the
/// sub-diagonal.
fn tred2(v: &mut Matrix, d: &mut [f64], e: &mut [f64]) {
    let n = d.len();
    for j in 0..n {
        d[j] = v[(n - 1, j)];
    }
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[(i - 1, j)];
                v[(i, j)] = 0.0;
                v[(j, i)] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in 0..i {
                e[j] = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[(j, i)] = f;
                g = e[j] + v[(j, j)] * f;
                for k in j + 1..i {
                    g += v[(k, j)] * d[k];
                    e[k] += v[(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1, j)];
                v[(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate transformations.
    for i in 0..n - 1 {
        v[(n - 1, i)] = v[(i, i)];
        v[(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[(k, i + 1)] * v[(k, j)];
                }
                for k in 0..=i {
                    v[(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[(n - 1, j)];
        v[(n - 1, j)] = 0.0;
    }
    v[(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

/// Implicit QL on the tridiagonal matrix (EISPACK tql2).
fn tql2(v: &mut Matrix, d: &mut [f64], e: &mut [f64]) -> Result<()> {
    let n = d.len();
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1 = 0.0f64;
    let eps = f64::EPSILON;
    let max_iter = 60 * n.max(1);
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        if m > l {
            let mut iter = 0;
            loop {
                iter += 1;
                if iter > max_iter {
                    return Err(MathError::NoConvergence { algorithm: "tridiagonal QL", iterations: max_iter });
                }
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().skip(l + 2) {
                    *di -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        let hk = v[(k, i + 1)];
                        v[(k, i + 1)] = s * v[(k, i)] + c * hk;
                        v[(k, i)] = c * v[(k, i)] - s * hk;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    fn check(a: &Matrix, e: &SymmetricEigen, tol: f64) {
        let n = a.rows();
        let v = &e.vectors;
        assert!(max_abs_diff(&v.t_matmul(v).unwrap(), &Matrix::identity(n)) < tol, "orthogonality");
        assert!(max_abs_diff(&e.reconstruct(), &a.symmetrize()) < tol * a.norm_max().max(1.0), "reconstruction");
        assert!(e.values.windows(2).all(|w| w[0] <= w[1]), "ascending");
    }

    #[test]
    fn known_spectra() {
        // Second-difference matrix: λ_k = 2 − 2cos(kπ/(n+1)).
        let n = 30;
        let t = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });
        for e in [SymmetricEigen::new(&t).unwrap(), SymmetricEigen::jacobi(&t).unwrap()] {
            for k in 0..n {
                let exact = 2.0 - 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
                assert!((e.values[k] - exact).abs() < 1e-13, "k={k}");
            }
            check(&t, &e, 1e-12);
        }
        // Repeated eigenvalues: J − I has eigenvalues −1 (×n−1) and n−1.
        let j = Matrix::from_fn(6, 6, |i, k| if i == k { 0.0 } else { 1.0 });
        let e = SymmetricEigen::new(&j).unwrap();
        assert!(e.values[..5].iter().all(|v| (v + 1.0).abs() < 1e-13));
        assert!((e.values[5] - 5.0).abs() < 1e-13);
        check(&j, &e, 1e-13);
    }

    #[test]
    fn random_and_degenerate() {
        for n in [1, 2, 7, 60] {
            let a = random(n, n, 40 + n as u64).symmetrize();
            let ql = SymmetricEigen::new(&a).unwrap();
            let jac = SymmetricEigen::jacobi(&a).unwrap();
            check(&a, &ql, 1e-12);
            check(&a, &jac, 1e-12);
            assert!(ql.values.iter().zip(&jac.values).all(|(p, q)| (p - q).abs() < 1e-12));
        }
        let z = SymmetricEigen::new(&Matrix::zeros(4, 4)).unwrap();
        assert!(z.values.iter().all(|v| *v == 0.0));
        let d = SymmetricEigen::new(&Matrix::diag(&[3.0, -1.0, 2.0])).unwrap();
        assert_eq!(d.values, vec![-1.0, 2.0, 3.0]);
        assert!(SymmetricEigen::new(&Matrix::zeros(2, 3)).is_err());
    }

    #[test]
    fn jacobi_relative_accuracy_on_graded_spd() {
        // Graded diagonal SPD with tiny eigenvalues: D·H·D, H = Hilbert-like
        // well-conditioned core. Jacobi recovers the smallest eigenvalue to
        // high relative accuracy.
        let n = 5;
        let dscale = [1.0, 1e-3, 1e-6, 1e-9, 1e-12];
        let core = Matrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.1 });
        let a = Matrix::from_fn(n, n, |i, j| dscale[i] * core[(i, j)] * dscale[j]);
        let jac = SymmetricEigen::jacobi(&a).unwrap();
        let smallest = jac.values[0];
        // det(A) = det(core)·Π d_i² ⇒ check product of eigenvalues relative to det.
        let det_core = super::super::lu::Lu::new(&core).unwrap().det();
        let det = det_core * dscale.iter().map(|d| d * d).product::<f64>();
        let prod: f64 = jac.values.iter().product();
        assert!(((prod - det) / det).abs() < 1e-8, "relative det error {}", ((prod - det) / det).abs());
        assert!(smallest > 0.0);
    }
}
