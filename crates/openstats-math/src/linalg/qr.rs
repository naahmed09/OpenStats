//! Householder QR factorisation, optionally with column pivoting
//! (rank-revealing): A·P = Q·R.

use super::blas::nrm2;
use super::householder::{apply_left, make_reflector};
use super::matrix::Matrix;
use crate::error::{MathError, Result};

/// QR factors of an m×n matrix.
#[derive(Debug, Clone)]
pub struct Qr {
    /// R on and above the diagonal; Householder vectors below it.
    qr: Matrix,
    tau: Vec<f64>,
    /// Column `j` of A·P is column `perm[j]` of A.
    perm: Vec<usize>,
    pivoted: bool,
}

impl Qr {
    /// Unpivoted Householder QR.
    pub fn new(a: &Matrix) -> Result<Self> {
        Self::factor(a, false)
    }

    /// QR with column pivoting (largest remaining column norm first), with
    /// the LAPACK `dlaqp2` norm-downdating safeguard.
    pub fn new_pivoted(a: &Matrix) -> Result<Self> {
        Self::factor(a, true)
    }

    fn factor(a: &Matrix, pivot: bool) -> Result<Self> {
        a.require_finite("QR")?;
        let (m, n) = a.shape();
        let mut qr = a.clone();
        let k = m.min(n);
        let mut tau = vec![0.0; k];
        let mut perm: Vec<usize> = (0..n).collect();
        let mut vn1: Vec<f64> = (0..n).map(|j| nrm2(&qr.col(j))).collect();
        let mut vn2 = vn1.clone();
        let tol3z = f64::EPSILON.sqrt();
        let mut x = vec![0.0; m];
        for i in 0..k {
            if pivot {
                let p = (i..n).fold(i, |b, j| if vn1[j] > vn1[b] { j } else { b });
                if p != i {
                    qr.swap_cols(p, i);
                    perm.swap(p, i);
                    vn1.swap(p, i);
                    vn2.swap(p, i);
                }
            }
            let len = m - i;
            for r in 0..len {
                x[r] = qr[(i + r, i)];
            }
            tau[i] = make_reflector(&mut x[..len]);
            for r in 0..len {
                qr[(i + r, i)] = x[r];
            }
            if i + 1 < n {
                apply_left(&x[1..len], tau[i], qr.as_mut_slice(), n, i, i + 1, n);
            }
            if pivot {
                for j in i + 1..n {
                    if vn1[j] != 0.0 {
                        let t = 1.0 - (qr[(i, j)].abs() / vn1[j]).powi(2);
                        let t = t.max(0.0);
                        let t2 = t * (vn1[j] / vn2[j]).powi(2);
                        if t2 <= tol3z {
                            let tail: Vec<f64> = (i + 1..m).map(|r| qr[(r, j)]).collect();
                            vn1[j] = nrm2(&tail);
                            vn2[j] = vn1[j];
                        } else {
                            vn1[j] *= t.sqrt();
                        }
                    }
                }
            }
        }
        Ok(Qr { qr, tau, perm, pivoted: pivot })
    }

    pub fn shape(&self) -> (usize, usize) {
        self.qr.shape()
    }

    pub fn is_pivoted(&self) -> bool {
        self.pivoted
    }

    /// Column permutation (identity when unpivoted).
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Permutation matrix P with A·P = Q·R.
    pub fn p(&self) -> Matrix {
        let n = self.perm.len();
        let mut p = Matrix::zeros(n, n);
        for (j, &c) in self.perm.iter().enumerate() {
            p[(c, j)] = 1.0;
        }
        p
    }

    /// Thin R: min(m,n) × n upper triangular (trapezoidal).
    pub fn r(&self) -> Matrix {
        let (m, n) = self.shape();
        Matrix::from_fn(m.min(n), n, |i, j| if i <= j { self.qr[(i, j)] } else { 0.0 })
    }

    /// Diagonal of R.
    pub fn r_diag(&self) -> Vec<f64> {
        self.qr.diagonal()
    }

    /// x ← Qᵀ·x (length m).
    pub fn apply_qt(&self, x: &mut [f64]) -> Result<()> {
        let (m, _) = self.shape();
        if x.len() != m {
            return Err(MathError::DimensionMismatch { op: "QR apply_qt", left: self.shape(), right: (x.len(), 1) });
        }
        for (i, &t) in self.tau.iter().enumerate() {
            self.reflect(i, t, x);
        }
        Ok(())
    }

    /// x ← Q·x (length m).
    pub fn apply_q(&self, x: &mut [f64]) -> Result<()> {
        let (m, _) = self.shape();
        if x.len() != m {
            return Err(MathError::DimensionMismatch { op: "QR apply_q", left: self.shape(), right: (x.len(), 1) });
        }
        for (i, &t) in self.tau.iter().enumerate().rev() {
            self.reflect(i, t, x);
        }
        Ok(())
    }

    fn reflect(&self, i: usize, tau: f64, x: &mut [f64]) {
        if tau == 0.0 {
            return;
        }
        let m = self.qr.rows();
        let mut s = x[i];
        for r in i + 1..m {
            s += self.qr[(r, i)] * x[r];
        }
        s *= tau;
        x[i] -= s;
        for r in i + 1..m {
            x[r] -= s * self.qr[(r, i)];
        }
    }

    /// Thin Q: m × min(m,n) with orthonormal columns.
    pub fn q(&self) -> Matrix {
        let (m, n) = self.shape();
        self.q_cols(m.min(n))
    }

    /// Full orthogonal Q: m × m.
    pub fn q_full(&self) -> Matrix {
        self.q_cols(self.shape().0)
    }

    fn q_cols(&self, k: usize) -> Matrix {
        let m = self.shape().0;
        let mut q = Matrix::zeros(m, k);
        let mut e = vec![0.0; m];
        for j in 0..k {
            e.iter_mut().for_each(|v| *v = 0.0);
            e[j] = 1.0;
            self.apply_q(&mut e).expect("length m");
            q.set_col(j, &e);
        }
        q
    }

    /// Numerical rank: number of |R_ii| > tol. Default tolerance
    /// `max(m,n) · ε · |R_00|` (meaningful for the pivoted factorisation).
    pub fn rank(&self, tol: Option<f64>) -> usize {
        let (m, n) = self.shape();
        let d = self.r_diag();
        let r0 = d.first().map_or(0.0, |v| v.abs());
        let tol = tol.unwrap_or(m.max(n) as f64 * f64::EPSILON * r0);
        if self.pivoted {
            d.iter().take_while(|v| v.abs() > tol).count()
        } else {
            d.iter().filter(|v| v.abs() > tol).count()
        }
    }

    /// Least-squares solution of min ‖A·x − b‖₂. For a rank-deficient matrix
    /// (detected with the pivoted factorisation) this is the *basic* solution:
    /// variables beyond the rank are set to zero. Errors if an unpivoted
    /// factorisation hits a zero diagonal.
    pub fn solve_least_squares(&self, b: &[f64]) -> Result<Vec<f64>> {
        let (m, n) = self.shape();
        let mut y = b.to_vec();
        self.apply_qt(&mut y)?;
        let r = if self.pivoted { self.rank(None) } else { m.min(n) };
        let mut z = vec![0.0; n];
        for i in (0..r).rev() {
            let mut s = y[i];
            for j in i + 1..r {
                s -= self.qr[(i, j)] * z[j];
            }
            let d = self.qr[(i, i)];
            if d == 0.0 {
                return Err(MathError::Singular { pivot: i });
            }
            z[i] = s / d;
        }
        let mut x = vec![0.0; n];
        for (j, &c) in self.perm.iter().enumerate() {
            x[c] = z[j];
        }
        Ok(x)
    }

    /// Residual sum of squares of the least-squares fit: ‖(Qᵀb)[rank..]‖².
    pub fn residual_ss(&self, b: &[f64]) -> Result<f64> {
        let mut y = b.to_vec();
        self.apply_qt(&mut y)?;
        let r = if self.pivoted { self.rank(None) } else { self.shape().0.min(self.shape().1) };
        Ok(y[r..].iter().map(|v| v * v).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    #[test]
    fn reconstruction_and_orthogonality() {
        for (m, n) in [(6, 4), (4, 6), (50, 50), (120, 30)] {
            let a = random(m, n, (m * 7 + n) as u64);
            for piv in [false, true] {
                let f = if piv { Qr::new_pivoted(&a) } else { Qr::new(&a) }.unwrap();
                let q = f.q();
                let k = m.min(n);
                assert!(max_abs_diff(&q.t_matmul(&q).unwrap(), &Matrix::identity(k)) < 1e-13);
                let qr = q.matmul(&f.r()).unwrap();
                let ap = a.matmul(&f.p()).unwrap();
                assert!(max_abs_diff(&qr, &ap) < 1e-12, "{m}x{n} piv={piv}");
                let qf = f.q_full();
                assert!(max_abs_diff(&qf.t_matmul(&qf).unwrap(), &Matrix::identity(m)) < 1e-13);
                if piv {
                    let d = f.r_diag();
                    assert!(
                        d.windows(2).all(|w| w[0].abs() >= w[1].abs() * (1.0 - 1e-12)),
                        "pivoted R diag decreasing"
                    );
                }
            }
        }
    }

    #[test]
    fn least_squares_and_rank() {
        // Overdetermined consistent system recovers x exactly.
        let a = random(30, 5, 11);
        let x = [1.0, -2.0, 0.5, 3.0, -1.0];
        let b = a.matvec(&x).unwrap();
        let sol = Qr::new(&a).unwrap().solve_least_squares(&b).unwrap();
        assert!(sol.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-12));
        // Fitting a line to noisy points: matches the normal-equation solution.
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys = [1.1, 2.9, 5.2, 7.1, 8.8, 11.2, 13.1, 14.8, 17.2, 18.9];
        let design = Matrix::from_fn(10, 2, |i, j| if j == 0 { 1.0 } else { xs[i] });
        let f = Qr::new(&design).unwrap();
        let beta = f.solve_least_squares(&ys).unwrap();
        let (sx, sy, sxx, sxy) =
            (45.0, ys.iter().sum::<f64>(), 285.0, xs.iter().zip(&ys).map(|(a, b)| a * b).sum::<f64>());
        let slope = (10.0 * sxy - sx * sy) / (10.0 * sxx - sx * sx);
        assert!((beta[1] - slope).abs() < 1e-13);
        assert!((beta[0] - (sy - slope * sx) / 10.0).abs() < 1e-13);
        let rss: f64 = (0..10).map(|i| (ys[i] - beta[0] - beta[1] * xs[i]).powi(2)).sum();
        assert!((f.residual_ss(&ys).unwrap() - rss).abs() < 1e-12);
        // Rank-deficient: third column = first + second.
        let mut d = random(20, 3, 12);
        for i in 0..20 {
            d[(i, 2)] = d[(i, 0)] + d[(i, 1)];
        }
        let p = Qr::new_pivoted(&d).unwrap();
        assert_eq!(p.rank(None), 2);
        let bb = d.matvec(&[1.0, 1.0, 1.0]).unwrap();
        let basic = p.solve_least_squares(&bb).unwrap();
        let fit = d.matvec(&basic).unwrap();
        assert!(fit.iter().zip(&bb).all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(basic.iter().filter(|v| **v == 0.0).count(), 1);
    }

    #[test]
    fn ill_conditioned_vandermonde() {
        // Degree-8 polynomial fit on [0, 1]: QR keeps full accuracy where the
        // normal equations lose about half the digits.
        let n = 40;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let coef: Vec<f64> = (0..9).map(|k| 1.0 / (k + 1) as f64).collect();
        let a = Matrix::from_fn(n, 9, |i, j| t[i].powi(j as i32));
        let b = a.matvec(&coef).unwrap();
        let sol = Qr::new_pivoted(&a).unwrap().solve_least_squares(&b).unwrap();
        let err = sol.iter().zip(&coef).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(err < 1e-6, "max coefficient error {err:e}");
    }
}
