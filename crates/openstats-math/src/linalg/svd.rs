//! Singular value decomposition A = U·diag(σ)·Vᵀ by Householder
//! bidiagonalisation and the Golub–Kahan–Reinsch implicit-shift QR
//! iteration, plus the pseudo-inverse, minimum-norm least squares, rank,
//! 2-norm and condition number built on it.

use super::matrix::Matrix;
use crate::error::{MathError, Result};

/// Thin SVD: for an m×n matrix with k = min(m, n), U is m×k, σ has k
/// non-negative entries in decreasing order and V is n×k.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Matrix,
    pub s: Vec<f64>,
    pub v: Matrix,
}

impl Svd {
    pub fn new(a: &Matrix) -> Result<Self> {
        a.require_finite("SVD")?;
        let (m, n) = a.shape();
        if m == 0 || n == 0 {
            return Ok(Svd { u: Matrix::zeros(m, 0), s: vec![], v: Matrix::zeros(n, 0) });
        }
        if m >= n {
            gkr(a)
        } else {
            // A = U S Vᵀ ⇔ Aᵀ = V S Uᵀ.
            let t = gkr(&a.t())?;
            Ok(Svd { u: t.v, s: t.s, v: t.u })
        }
    }

    pub fn rank_tol(&self) -> f64 {
        let (m, n) = (self.u.rows(), self.v.rows());
        m.max(n) as f64 * f64::EPSILON * self.s.first().copied().unwrap_or(0.0)
    }

    /// Number of singular values above `tol` (default max(m,n)·ε·σ₁).
    pub fn rank(&self, tol: Option<f64>) -> usize {
        let t = tol.unwrap_or_else(|| self.rank_tol());
        self.s.iter().filter(|&&x| x > t).count()
    }

    /// Spectral norm σ₁.
    pub fn norm2(&self) -> f64 {
        self.s.first().copied().unwrap_or(0.0)
    }

    /// 2-norm condition number σ₁/σₖ (∞ if rank-deficient).
    pub fn cond(&self) -> f64 {
        match (self.s.first(), self.s.last()) {
            (Some(&a), Some(&b)) if b > 0.0 => a / b,
            (Some(_), Some(_)) => f64::INFINITY,
            _ => 0.0,
        }
    }

    /// Moore–Penrose pseudo-inverse A⁺ = V·diag(1/σ)·Uᵀ, ignoring σ ≤ tol.
    pub fn pinv(&self, tol: Option<f64>) -> Matrix {
        let t = tol.unwrap_or_else(|| self.rank_tol());
        let (m, n, k) = (self.u.rows(), self.v.rows(), self.s.len());
        let mut out = Matrix::zeros(n, m);
        for l in 0..k {
            if self.s[l] <= t {
                continue;
            }
            let inv = 1.0 / self.s[l];
            for i in 0..n {
                let vil = self.v[(i, l)] * inv;
                if vil == 0.0 {
                    continue;
                }
                let row = out.row_mut(i);
                for j in 0..m {
                    row[j] += vil * self.u[(j, l)];
                }
            }
        }
        out
    }

    /// Minimum-norm least-squares solution of A·x ≈ b.
    pub fn solve_least_squares(&self, b: &[f64], tol: Option<f64>) -> Result<Vec<f64>> {
        let (m, n) = (self.u.rows(), self.v.rows());
        if b.len() != m {
            return Err(MathError::DimensionMismatch { op: "SVD least squares", left: (m, n), right: (b.len(), 1) });
        }
        let t = tol.unwrap_or_else(|| self.rank_tol());
        let mut x = vec![0.0; n];
        for l in 0..self.s.len() {
            if self.s[l] <= t {
                continue;
            }
            let c: f64 = (0..m).map(|i| self.u[(i, l)] * b[i]).sum::<f64>() / self.s[l];
            for i in 0..n {
                x[i] += c * self.v[(i, l)];
            }
        }
        Ok(x)
    }

    /// Orthonormal basis of the null space of A (columns of V for σ ≤ tol;
    /// complete only when m ≥ n).
    pub fn null_space(&self, tol: Option<f64>) -> Matrix {
        let t = tol.unwrap_or_else(|| self.rank_tol());
        let cols: Vec<usize> = (0..self.s.len()).filter(|&l| self.s[l] <= t).collect();
        Matrix::from_fn(self.v.rows(), cols.len(), |i, j| self.v[(i, cols[j])])
    }

    pub fn reconstruct(&self) -> Matrix {
        let us = Matrix::from_fn(self.u.rows(), self.s.len(), |i, j| self.u[(i, j)] * self.s[j]);
        us.matmul(&self.v.t()).expect("conformable")
    }
}

/// Golub–Kahan–Reinsch for m ≥ n (after the classic LINPACK/JAMA layout).
#[allow(clippy::many_single_char_names)]
fn gkr(a_in: &Matrix) -> Result<Svd> {
    let (m, n) = a_in.shape();
    let mut a = a_in.clone();
    let nu = m.min(n);
    let mut s = vec![0.0f64; (m + 1).min(n)];
    let mut u = Matrix::zeros(m, nu);
    let mut v = Matrix::zeros(n, n);
    let mut e = vec![0.0f64; n];
    let mut work = vec![0.0; m];

    // Reduce A to bidiagonal form, storing the diagonal in s and the
    // super-diagonal in e.
    let nct = (m - 1).min(n);
    let nrt = n.saturating_sub(2).min(m);
    for k in 0..nct.max(nrt) {
        if k < nct {
            s[k] = 0.0;
            for i in k..m {
                s[k] = s[k].hypot(a[(i, k)]);
            }
            if s[k] != 0.0 {
                if a[(k, k)] < 0.0 {
                    s[k] = -s[k];
                }
                for i in k..m {
                    a[(i, k)] /= s[k];
                }
                a[(k, k)] += 1.0;
            }
            s[k] = -s[k];
        }
        for j in k + 1..n {
            if k < nct && s[k] != 0.0 {
                let mut t = 0.0;
                for i in k..m {
                    t += a[(i, k)] * a[(i, j)];
                }
                t = -t / a[(k, k)];
                for i in k..m {
                    a[(i, j)] += t * a[(i, k)];
                }
            }
            e[j] = a[(k, j)];
        }
        if k < nct {
            for i in k..m {
                u[(i, k)] = a[(i, k)];
            }
        }
        if k < nrt {
            e[k] = 0.0;
            for i in k + 1..n {
                e[k] = e[k].hypot(e[i]);
            }
            if e[k] != 0.0 {
                if e[k + 1] < 0.0 {
                    e[k] = -e[k];
                }
                for i in k + 1..n {
                    e[i] /= e[k];
                }
                e[k + 1] += 1.0;
            }
            e[k] = -e[k];
            if k + 1 < m && e[k] != 0.0 {
                for w in work.iter_mut().take(m).skip(k + 1) {
                    *w = 0.0;
                }
                for j in k + 1..n {
                    for i in k + 1..m {
                        work[i] += e[j] * a[(i, j)];
                    }
                }
                for j in k + 1..n {
                    let t = -e[j] / e[k + 1];
                    for i in k + 1..m {
                        a[(i, j)] += t * work[i];
                    }
                }
            }
            for i in k + 1..n {
                v[(i, k)] = e[i];
            }
        }
    }

    // Set up the final bidiagonal matrix of order p.
    let mut p = n.min(m + 1);
    if nct < n {
        s[nct] = a[(nct, nct)];
    }
    if m < p {
        s[p - 1] = 0.0;
    }
    if nrt + 1 < p {
        e[nrt] = a[(nrt, p - 1)];
    }
    e[p - 1] = 0.0;

    // Generate U.
    for j in nct..nu {
        for i in 0..m {
            u[(i, j)] = 0.0;
        }
        u[(j, j)] = 1.0;
    }
    for k in (0..nct).rev() {
        if s[k] != 0.0 {
            for j in k + 1..nu {
                let mut t = 0.0;
                for i in k..m {
                    t += u[(i, k)] * u[(i, j)];
                }
                t = -t / u[(k, k)];
                for i in k..m {
                    u[(i, j)] += t * u[(i, k)];
                }
            }
            for i in k..m {
                u[(i, k)] = -u[(i, k)];
            }
            u[(k, k)] += 1.0;
            for i in 0..k.saturating_sub(1) {
                u[(i, k)] = 0.0;
            }
        } else {
            for i in 0..m {
                u[(i, k)] = 0.0;
            }
            u[(k, k)] = 1.0;
        }
    }

    // Generate V.
    for k in (0..n).rev() {
        if k < nrt && e[k] != 0.0 {
            for j in k + 1..nu {
                let mut t = 0.0;
                for i in k + 1..n {
                    t += v[(i, k)] * v[(i, j)];
                }
                t = -t / v[(k + 1, k)];
                for i in k + 1..n {
                    v[(i, j)] += t * v[(i, k)];
                }
            }
        }
        for i in 0..n {
            v[(i, k)] = 0.0;
        }
        v[(k, k)] = 1.0;
    }

    // Main iteration loop for the singular values.
    let pp = p - 1;
    let eps = f64::EPSILON;
    let tiny = 2f64.powi(-966);
    let mut iter = 0usize;
    let max_iter = 75 * n.max(1);
    while p > 0 {
        // Inspect for negligible elements in s and e.
        // kase 1: s(p) and e[k-1] negligible, k < p
        // kase 2: s(k) negligible, k < p
        // kase 3: e[k-1] negligible, k < p, s(k..p) not negligible (QR step)
        // kase 4: e(p-1) negligible (convergence)
        let pi = p as isize;
        let mut k: isize = pi - 2;
        while k >= 0 {
            let ku = k as usize;
            if e[ku].abs() <= tiny + eps * (s[ku].abs() + s[ku + 1].abs()) {
                e[ku] = 0.0;
                break;
            }
            k -= 1;
        }
        let kase;
        if k == pi - 2 {
            kase = 4;
        } else {
            let mut ks: isize = pi - 1;
            while ks > k {
                let ksu = ks as usize;
                let t =
                    (if ks != pi { e[ksu].abs() } else { 0.0 }) + (if ks != k + 1 { e[ksu - 1].abs() } else { 0.0 });
                if s[ksu].abs() <= tiny + eps * t {
                    s[ksu] = 0.0;
                    break;
                }
                ks -= 1;
            }
            if ks == k {
                kase = 3;
            } else if ks == pi - 1 {
                kase = 1;
            } else {
                kase = 2;
                k = ks;
            }
        }
        k += 1;
        let k = k as usize;

        match kase {
            1 => {
                // Deflate negligible s(p).
                let mut f = e[p - 2];
                e[p - 2] = 0.0;
                for j in (k..=p - 2).rev() {
                    let t = s[j].hypot(f);
                    let cs = s[j] / t;
                    let sn = f / t;
                    s[j] = t;
                    if j != k {
                        f = -sn * e[j - 1];
                        e[j - 1] *= cs;
                    }
                    for i in 0..n {
                        let t = cs * v[(i, j)] + sn * v[(i, p - 1)];
                        v[(i, p - 1)] = -sn * v[(i, j)] + cs * v[(i, p - 1)];
                        v[(i, j)] = t;
                    }
                }
            }
            2 => {
                // Split at negligible s(k).
                let mut f = e[k - 1];
                e[k - 1] = 0.0;
                for j in k..p {
                    let t = s[j].hypot(f);
                    let cs = s[j] / t;
                    let sn = f / t;
                    s[j] = t;
                    f = -sn * e[j];
                    e[j] *= cs;
                    for i in 0..m {
                        let t = cs * u[(i, j)] + sn * u[(i, k - 1)];
                        u[(i, k - 1)] = -sn * u[(i, j)] + cs * u[(i, k - 1)];
                        u[(i, j)] = t;
                    }
                }
            }
            3 => {
                // One QR step with the Wilkinson-type shift.
                let scale = s[p - 1].abs().max(s[p - 2].abs()).max(e[p - 2].abs()).max(s[k].abs()).max(e[k].abs());
                let sp = s[p - 1] / scale;
                let spm1 = s[p - 2] / scale;
                let epm1 = e[p - 2] / scale;
                let sk = s[k] / scale;
                let ek = e[k] / scale;
                let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                let c = (sp * epm1) * (sp * epm1);
                let mut shift = 0.0;
                if b != 0.0 || c != 0.0 {
                    shift = (b * b + c).sqrt();
                    if b < 0.0 {
                        shift = -shift;
                    }
                    shift = c / (b + shift);
                }
                let mut f = (sk + sp) * (sk - sp) + shift;
                let mut g = sk * ek;
                for j in k..p - 1 {
                    let mut t = f.hypot(g);
                    let mut cs = f / t;
                    let mut sn = g / t;
                    if j != k {
                        e[j - 1] = t;
                    }
                    f = cs * s[j] + sn * e[j];
                    e[j] = cs * e[j] - sn * s[j];
                    g = sn * s[j + 1];
                    s[j + 1] *= cs;
                    for i in 0..n {
                        t = cs * v[(i, j)] + sn * v[(i, j + 1)];
                        v[(i, j + 1)] = -sn * v[(i, j)] + cs * v[(i, j + 1)];
                        v[(i, j)] = t;
                    }
                    t = f.hypot(g);
                    cs = f / t;
                    sn = g / t;
                    s[j] = t;
                    f = cs * e[j] + sn * s[j + 1];
                    s[j + 1] = -sn * e[j] + cs * s[j + 1];
                    g = sn * e[j + 1];
                    e[j + 1] *= cs;
                    if j < m - 1 {
                        for i in 0..m {
                            t = cs * u[(i, j)] + sn * u[(i, j + 1)];
                            u[(i, j + 1)] = -sn * u[(i, j)] + cs * u[(i, j + 1)];
                            u[(i, j)] = t;
                        }
                    }
                }
                e[p - 2] = f;
                iter += 1;
                if iter > max_iter {
                    return Err(MathError::NoConvergence { algorithm: "SVD", iterations: max_iter });
                }
            }
            _ => {
                // Convergence: make the singular value positive and order them.
                let mut k = k;
                if s[k] <= 0.0 {
                    s[k] = if s[k] < 0.0 { -s[k] } else { 0.0 };
                    for i in 0..=pp {
                        v[(i, k)] = -v[(i, k)];
                    }
                }
                while k < pp {
                    if s[k] >= s[k + 1] {
                        break;
                    }
                    s.swap(k, k + 1);
                    if k < n - 1 {
                        v.swap_cols(k, k + 1);
                    }
                    if k < m - 1 {
                        u.swap_cols(k, k + 1);
                    }
                    k += 1;
                }
                iter = 0;
                p -= 1;
            }
        }
    }
    s.truncate(nu);
    Ok(Svd { u, s, v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    fn check(a: &Matrix, f: &Svd) {
        let (m, n) = a.shape();
        let k = m.min(n);
        assert_eq!(f.s.len(), k);
        assert!(f.s.windows(2).all(|w| w[0] >= w[1]) && f.s.iter().all(|&x| x >= 0.0));
        assert!(max_abs_diff(&f.u.t_matmul(&f.u).unwrap(), &Matrix::identity(k)) < 1e-12, "UᵀU");
        assert!(max_abs_diff(&f.v.t_matmul(&f.v).unwrap(), &Matrix::identity(k)) < 1e-12, "VᵀV");
        assert!(max_abs_diff(&f.reconstruct(), a) < 1e-12 * a.norm_max().max(1.0), "reconstruction {m}x{n}");
    }

    #[test]
    fn shapes_and_reconstruction() {
        for (m, n) in [(1, 1), (5, 3), (3, 5), (40, 40), (100, 7), (7, 100), (2, 1), (1, 4)] {
            let a = random(m, n, (31 * m + n) as u64);
            check(&a, &Svd::new(&a).unwrap());
        }
    }

    #[test]
    fn known_singular_values() {
        // σ of [[3, 0], [4, 5]] are √45 and √5.
        let a = Matrix::from_rows(&[vec![3.0, 0.0], vec![4.0, 5.0]]).unwrap();
        let f = Svd::new(&a).unwrap();
        assert!((f.s[0] - 45f64.sqrt()).abs() < 1e-14 && (f.s[1] - 5f64.sqrt()).abs() < 1e-14);
        // Singular values of a symmetric matrix are |eigenvalues|.
        let s = random(12, 12, 77).symmetrize();
        let mut ev: Vec<f64> =
            super::super::eigen_sym::SymmetricEigen::new(&s).unwrap().values.iter().map(|v| v.abs()).collect();
        ev.sort_by(|a, b| b.total_cmp(a));
        let sv = Svd::new(&s).unwrap().s;
        assert!(ev.iter().zip(&sv).all(|(p, q)| (p - q).abs() < 1e-12));
        // Hilbert(8): σ_max ≈ 1.6959389969219498, cond₂ ≈ 1.5257575566e10.
        let h = Svd::new(&hilbert(8)).unwrap();
        assert!((h.s[0] - 1.695_938_996_921_949_8).abs() < 1e-13);
        assert!((h.cond() / 1.525_757_556_7e10 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rank_deficient_pinv_and_min_norm() {
        // Rank 2 matrix, 6×4.
        let b = random(6, 2, 5);
        let c = random(2, 4, 6);
        let a = b.matmul(&c).unwrap();
        let f = Svd::new(&a).unwrap();
        check(&a, &f);
        assert_eq!(f.rank(None), 2);
        assert_eq!(f.null_space(None).cols(), 2);
        let ns = f.null_space(None);
        assert!(a.matmul(&ns).unwrap().norm_max() < 1e-12);
        let p = f.pinv(None);
        // Penrose conditions.
        assert!(max_abs_diff(&a.matmul(&p).unwrap().matmul(&a).unwrap(), &a) < 1e-12);
        assert!(max_abs_diff(&p.matmul(&a).unwrap().matmul(&p).unwrap(), &p) < 1e-10);
        let ap = a.matmul(&p).unwrap();
        assert!(max_abs_diff(&ap, &ap.t()) < 1e-12);
        // Minimum-norm least squares equals pinv·b and is orthogonal to the null space.
        let rhs: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
        let x = f.solve_least_squares(&rhs, None).unwrap();
        let xp = p.matvec(&rhs).unwrap();
        assert!(x.iter().zip(&xp).all(|(u, w)| (u - w).abs() < 1e-11));
        let proj = ns.t_matvec(&x).unwrap();
        assert!(proj.iter().all(|v| v.abs() < 1e-11));
        let zero = Svd::new(&Matrix::zeros(3, 2)).unwrap();
        assert_eq!(zero.s, vec![0.0, 0.0]);
        assert_eq!(zero.rank(None), 0);
        assert!(zero.cond().is_infinite());
    }
}
