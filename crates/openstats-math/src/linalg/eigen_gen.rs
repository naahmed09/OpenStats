//! Eigendecomposition of a general real square matrix.
//!
//! Householder reduction to upper Hessenberg form (EISPACK `orthes`), then
//! the Francis double-shift QR algorithm with exceptional shifts and
//! back-substitution for eigenvectors (EISPACK `hqr2`). Complex eigenvalues
//! come in conjugate pairs; their eigenvectors are stored packed in two real
//! columns (real part, imaginary part).

use super::matrix::Matrix;
use crate::complex::Complex;
use crate::error::{MathError, Result};

#[derive(Debug, Clone)]
pub struct Eigen {
    /// Eigenvalues (complex pairs adjacent, positive imaginary part first).
    pub values: Vec<Complex>,
    /// Packed real eigenvectors: for a real eigenvalue column j is its
    /// eigenvector; for a pair (j, j+1) the eigenvector of `values[j]` is
    /// `V[:, j] + i·`V[:, j+1]`` and of `values[j+1]` its conjugate.
    pub vectors: Matrix,
}

impl Eigen {
    pub fn new(a: &Matrix) -> Result<Self> {
        let n = a.require_square()?;
        a.require_finite("eigen")?;
        if a.is_symmetric(0.0) {
            // Symmetric input: use the faster, orthogonal symmetric solver.
            let s = super::eigen_sym::SymmetricEigen::new(a)?;
            return Ok(Eigen { values: s.values.into_iter().map(Complex::real).collect(), vectors: s.vectors });
        }
        let mut h = a.clone();
        let mut v = Matrix::identity(n);
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        if n > 0 {
            orthes(&mut h, &mut v);
            hqr2(&mut h, &mut v, &mut d, &mut e)?;
        }
        Ok(Eigen { values: d.iter().zip(&e).map(|(&r, &i)| Complex::new(r, i)).collect(), vectors: v })
    }

    pub fn is_real(&self) -> bool {
        self.values.iter().all(|z| z.im == 0.0)
    }

    /// Unit-norm complex eigenvector for `values[j]`.
    pub fn eigenvector(&self, j: usize) -> Vec<Complex> {
        let n = self.vectors.rows();
        let z = self.values[j];
        let raw: Vec<Complex> = if z.im == 0.0 {
            (0..n).map(|i| Complex::real(self.vectors[(i, j)])).collect()
        } else if z.im > 0.0 {
            (0..n).map(|i| Complex::new(self.vectors[(i, j)], self.vectors[(i, j + 1)])).collect()
        } else {
            (0..n).map(|i| Complex::new(self.vectors[(i, j - 1)], -self.vectors[(i, j)])).collect()
        };
        let norm = raw.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f64>().sqrt();
        if norm == 0.0 {
            raw
        } else {
            raw.into_iter().map(|c| c * (1.0 / norm)).collect()
        }
    }

    /// Eigenvalues sorted by decreasing modulus (convenience for spectral
    /// radius / stability checks).
    pub fn values_by_modulus(&self) -> Vec<Complex> {
        let mut v = self.values.clone();
        v.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        v
    }
}

/// Householder reduction to Hessenberg form, accumulating the transform in V.
fn orthes(h: &mut Matrix, v: &mut Matrix) {
    let n = h.rows();
    let (low, high) = (0usize, n - 1);
    let mut ort = vec![0.0; n];
    for m in low + 1..high {
        let mut scale = 0.0;
        for i in m..=high {
            scale += h[(i, m - 1)].abs();
        }
        if scale != 0.0 {
            let mut hh = 0.0;
            for i in (m..=high).rev() {
                ort[i] = h[(i, m - 1)] / scale;
                hh += ort[i] * ort[i];
            }
            let mut g = hh.sqrt();
            if ort[m] > 0.0 {
                g = -g;
            }
            hh -= ort[m] * g;
            ort[m] -= g;
            for j in m..n {
                let mut f = 0.0;
                for i in (m..=high).rev() {
                    f += ort[i] * h[(i, j)];
                }
                f /= hh;
                for i in m..=high {
                    h[(i, j)] -= f * ort[i];
                }
            }
            for i in 0..=high {
                let mut f = 0.0;
                for j in (m..=high).rev() {
                    f += ort[j] * h[(i, j)];
                }
                f /= hh;
                for j in m..=high {
                    h[(i, j)] -= f * ort[j];
                }
            }
            ort[m] *= scale;
            h[(m, m - 1)] = scale * g;
        }
    }
    for m in (low + 1..high).rev() {
        if h[(m, m - 1)] != 0.0 {
            for i in m + 1..=high {
                ort[i] = h[(i, m - 1)];
            }
            for j in m..=high {
                let mut g = 0.0;
                for i in m..=high {
                    g += ort[i] * v[(i, j)];
                }
                // Double division avoids possible underflow.
                g = (g / ort[m]) / h[(m, m - 1)];
                for i in m..=high {
                    v[(i, j)] += g * ort[i];
                }
            }
        }
    }
}

/// Complex scalar division (xr + i·xi)/(yr + i·yi).
fn cdiv(xr: f64, xi: f64, yr: f64, yi: f64) -> (f64, f64) {
    let q = Complex::new(xr, xi) / Complex::new(yr, yi);
    (q.re, q.im)
}

/// Francis double-shift QR on the Hessenberg matrix with eigenvectors.
#[allow(clippy::many_single_char_names)]
fn hqr2(h: &mut Matrix, v: &mut Matrix, d: &mut [f64], e: &mut [f64]) -> Result<()> {
    let nn = h.rows();
    let mut n = nn as isize - 1;
    let low: isize = 0;
    let high: isize = nn as isize - 1;
    let eps = f64::EPSILON;
    let mut exshift = 0.0;
    let (mut p, mut q, mut r, mut s, mut z) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut t, mut w, mut x, mut y);
    let at = |i: isize, j: isize| (i as usize, j as usize);

    let mut norm = 0.0;
    for i in 0..nn {
        for j in i.saturating_sub(1)..nn {
            norm += h[(i, j)].abs();
        }
    }
    let mut iter = 0usize;
    let mut total_iter = 0usize;
    let max_total = 100 * nn.max(1);
    while n >= low {
        // Look for a single small sub-diagonal element.
        let mut l = n;
        while l > low {
            s = h[at(l - 1, l - 1)].abs() + h[at(l, l)].abs();
            if s == 0.0 {
                s = norm;
            }
            if h[at(l, l - 1)].abs() < eps * s {
                break;
            }
            l -= 1;
        }
        if l == n {
            // One root found.
            h[at(n, n)] += exshift;
            d[n as usize] = h[at(n, n)];
            e[n as usize] = 0.0;
            n -= 1;
            iter = 0;
        } else if l == n - 1 {
            // Two roots found.
            w = h[at(n, n - 1)] * h[at(n - 1, n)];
            p = (h[at(n - 1, n - 1)] - h[at(n, n)]) / 2.0;
            q = p * p + w;
            z = q.abs().sqrt();
            h[at(n, n)] += exshift;
            h[at(n - 1, n - 1)] += exshift;
            x = h[at(n, n)];
            if q >= 0.0 {
                z = if p >= 0.0 { p + z } else { p - z };
                d[(n - 1) as usize] = x + z;
                d[n as usize] = d[(n - 1) as usize];
                if z != 0.0 {
                    d[n as usize] = x - w / z;
                }
                e[(n - 1) as usize] = 0.0;
                e[n as usize] = 0.0;
                x = h[at(n, n - 1)];
                s = x.abs() + z.abs();
                p = x / s;
                q = z / s;
                r = (p * p + q * q).sqrt();
                p /= r;
                q /= r;
                for j in (n - 1)..nn as isize {
                    z = h[at(n - 1, j)];
                    h[at(n - 1, j)] = q * z + p * h[at(n, j)];
                    h[at(n, j)] = q * h[at(n, j)] - p * z;
                }
                for i in 0..=n {
                    z = h[at(i, n - 1)];
                    h[at(i, n - 1)] = q * z + p * h[at(i, n)];
                    h[at(i, n)] = q * h[at(i, n)] - p * z;
                }
                for i in low..=high {
                    z = v[at(i, n - 1)];
                    v[at(i, n - 1)] = q * z + p * v[at(i, n)];
                    v[at(i, n)] = q * v[at(i, n)] - p * z;
                }
            } else {
                d[(n - 1) as usize] = x + p;
                d[n as usize] = x + p;
                e[(n - 1) as usize] = z;
                e[n as usize] = -z;
            }
            n -= 2;
            iter = 0;
        } else {
            // No convergence yet: form shift.
            x = h[at(n, n)];
            y = 0.0;
            w = 0.0;
            if l < n {
                y = h[at(n - 1, n - 1)];
                w = h[at(n, n - 1)] * h[at(n - 1, n)];
            }
            // Wilkinson's original ad hoc shift.
            if iter == 10 {
                exshift += x;
                for i in low..=n {
                    h[at(i, i)] -= x;
                }
                s = h[at(n, n - 1)].abs() + h[at(n - 1, n - 2)].abs();
                x = 0.75 * s;
                y = x;
                w = -0.4375 * s * s;
            }
            // MATLAB's ad hoc shift.
            if iter == 30 {
                s = (y - x) / 2.0;
                s = s * s + w;
                if s > 0.0 {
                    s = s.sqrt();
                    if y < x {
                        s = -s;
                    }
                    s = x - w / ((y - x) / 2.0 + s);
                    for i in low..=n {
                        h[at(i, i)] -= s;
                    }
                    exshift += s;
                    x = 0.964;
                    y = x;
                    w = x;
                }
            }
            iter += 1;
            total_iter += 1;
            if total_iter > max_total {
                return Err(MathError::NoConvergence { algorithm: "Hessenberg QR", iterations: max_total });
            }
            // Look for two consecutive small sub-diagonal elements.
            let mut m = n - 2;
            while m >= l {
                z = h[at(m, m)];
                r = x - z;
                s = y - z;
                p = (r * s - w) / h[at(m + 1, m)] + h[at(m, m + 1)];
                q = h[at(m + 1, m + 1)] - z - r - s;
                r = h[at(m + 2, m + 1)];
                s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                if h[at(m, m - 1)].abs() * (q.abs() + r.abs())
                    < eps * (p.abs() * (h[at(m - 1, m - 1)].abs() + z.abs() + h[at(m + 1, m + 1)].abs()))
                {
                    break;
                }
                m -= 1;
            }
            for i in m + 2..=n {
                h[at(i, i - 2)] = 0.0;
                if i > m + 2 {
                    h[at(i, i - 3)] = 0.0;
                }
            }
            // Double QR step involving rows l:n and columns m:n.
            let mut k = m;
            while k < n {
                let notlast = k != n - 1;
                if k != m {
                    p = h[at(k, k - 1)];
                    q = h[at(k + 1, k - 1)];
                    r = if notlast { h[at(k + 2, k - 1)] } else { 0.0 };
                    x = p.abs() + q.abs() + r.abs();
                    if x == 0.0 {
                        k += 1;
                        continue;
                    }
                    p /= x;
                    q /= x;
                    r /= x;
                }
                s = (p * p + q * q + r * r).sqrt();
                if p < 0.0 {
                    s = -s;
                }
                if s != 0.0 {
                    if k != m {
                        h[at(k, k - 1)] = -s * x;
                    } else if l != m {
                        h[at(k, k - 1)] = -h[at(k, k - 1)];
                    }
                    p += s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q /= p;
                    r /= p;
                    for j in k..nn as isize {
                        p = h[at(k, j)] + q * h[at(k + 1, j)];
                        if notlast {
                            p += r * h[at(k + 2, j)];
                            h[at(k + 2, j)] -= p * z;
                        }
                        h[at(k, j)] -= p * x;
                        h[at(k + 1, j)] -= p * y;
                    }
                    for i in 0..=n.min(k + 3) {
                        p = x * h[at(i, k)] + y * h[at(i, k + 1)];
                        if notlast {
                            p += z * h[at(i, k + 2)];
                            h[at(i, k + 2)] -= p * r;
                        }
                        h[at(i, k)] -= p;
                        h[at(i, k + 1)] -= p * q;
                    }
                    for i in low..=high {
                        p = x * v[at(i, k)] + y * v[at(i, k + 1)];
                        if notlast {
                            p += z * v[at(i, k + 2)];
                            v[at(i, k + 2)] -= p * r;
                        }
                        v[at(i, k)] -= p;
                        v[at(i, k + 1)] -= p * q;
                    }
                }
                k += 1;
            }
        }
    }

    // Back-substitute to find vectors of the upper triangular form.
    if norm == 0.0 {
        return Ok(());
    }
    for n in (0..nn as isize).rev() {
        p = d[n as usize];
        q = e[n as usize];
        if q == 0.0 {
            // Real vector.
            let mut l = n;
            h[at(n, n)] = 1.0;
            for i in (0..n).rev() {
                w = h[at(i, i)] - p;
                r = 0.0;
                for j in l..=n {
                    r += h[at(i, j)] * h[at(j, n)];
                }
                if e[i as usize] < 0.0 {
                    z = w;
                    s = r;
                } else {
                    l = i;
                    if e[i as usize] == 0.0 {
                        h[at(i, n)] = if w != 0.0 { -r / w } else { -r / (eps * norm) };
                    } else {
                        x = h[at(i, i + 1)];
                        y = h[at(i + 1, i)];
                        q = (d[i as usize] - p) * (d[i as usize] - p) + e[i as usize] * e[i as usize];
                        t = (x * s - z * r) / q;
                        h[at(i, n)] = t;
                        h[at(i + 1, n)] = if x.abs() > z.abs() { (-r - w * t) / x } else { (-s - y * t) / z };
                    }
                    // Overflow control.
                    t = h[at(i, n)].abs();
                    if (eps * t) * t > 1.0 {
                        for j in i..=n {
                            h[at(j, n)] /= t;
                        }
                    }
                }
            }
        } else if q < 0.0 {
            // Complex vector.
            let mut l = n - 1;
            if h[at(n, n - 1)].abs() > h[at(n - 1, n)].abs() {
                h[at(n - 1, n - 1)] = q / h[at(n, n - 1)];
                h[at(n - 1, n)] = -(h[at(n, n)] - p) / h[at(n, n - 1)];
            } else {
                let (cr, ci) = cdiv(0.0, -h[at(n - 1, n)], h[at(n - 1, n - 1)] - p, q);
                h[at(n - 1, n - 1)] = cr;
                h[at(n - 1, n)] = ci;
            }
            h[at(n, n - 1)] = 0.0;
            h[at(n, n)] = 1.0;
            for i in (0..n - 1).rev() {
                let mut ra = 0.0;
                let mut sa = 0.0;
                for j in l..=n {
                    ra += h[at(i, j)] * h[at(j, n - 1)];
                    sa += h[at(i, j)] * h[at(j, n)];
                }
                w = h[at(i, i)] - p;
                if e[i as usize] < 0.0 {
                    z = w;
                    r = ra;
                    s = sa;
                } else {
                    l = i;
                    if e[i as usize] == 0.0 {
                        let (cr, ci) = cdiv(-ra, -sa, w, q);
                        h[at(i, n - 1)] = cr;
                        h[at(i, n)] = ci;
                    } else {
                        x = h[at(i, i + 1)];
                        y = h[at(i + 1, i)];
                        let di = d[i as usize] - p;
                        let mut vr = di * di + e[i as usize] * e[i as usize] - q * q;
                        let vi = di * 2.0 * q;
                        if vr == 0.0 && vi == 0.0 {
                            vr = eps * norm * (w.abs() + q.abs() + x.abs() + y.abs() + z.abs());
                        }
                        let (cr, ci) = cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
                        h[at(i, n - 1)] = cr;
                        h[at(i, n)] = ci;
                        if x.abs() > z.abs() + q.abs() {
                            h[at(i + 1, n - 1)] = (-ra - w * h[at(i, n - 1)] + q * h[at(i, n)]) / x;
                            h[at(i + 1, n)] = (-sa - w * h[at(i, n)] - q * h[at(i, n - 1)]) / x;
                        } else {
                            let (cr, ci) = cdiv(-r - y * h[at(i, n - 1)], -s - y * h[at(i, n)], z, q);
                            h[at(i + 1, n - 1)] = cr;
                            h[at(i + 1, n)] = ci;
                        }
                    }
                    // Overflow control.
                    t = h[at(i, n - 1)].abs().max(h[at(i, n)].abs());
                    if (eps * t) * t > 1.0 {
                        for j in i..=n {
                            h[at(j, n - 1)] /= t;
                            h[at(j, n)] /= t;
                        }
                    }
                }
            }
        }
    }
    // Back-transformation to eigenvectors of the original matrix.
    for j in (low..nn as isize).rev() {
        for i in low..=high {
            z = 0.0;
            for k in low..=j.min(high) {
                z += v[at(i, k)] * h[at(k, j)];
            }
            v[at(i, j)] = z;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    /// max_j ‖A v_j − λ_j v_j‖ for unit eigenvectors.
    fn residual(a: &Matrix, e: &Eigen) -> f64 {
        let n = a.rows();
        let mut worst = 0.0f64;
        for j in 0..n {
            let v = e.eigenvector(j);
            let lam = e.values[j];
            for i in 0..n {
                let mut av = Complex::default();
                for k in 0..n {
                    av = av + v[k] * a[(i, k)];
                }
                worst = worst.max((av - lam * v[i]).abs());
            }
        }
        worst
    }

    fn sorted_values(e: &Eigen) -> Vec<Complex> {
        let mut v = e.values.clone();
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
        v
    }

    #[test]
    fn rotation_and_companion() {
        // Rotation by θ: eigenvalues e^{±iθ}.
        let th = 0.7f64;
        let rot = Matrix::from_rows(&[vec![th.cos(), -th.sin()], vec![th.sin(), th.cos()]]).unwrap();
        let e = Eigen::new(&rot).unwrap();
        assert!(!e.is_real());
        assert!((e.values[0].re - th.cos()).abs() < 1e-15 && (e.values[0].im.abs() - th.sin()).abs() < 1e-15);
        assert!(residual(&rot, &e) < 1e-14);
        // Companion matrix of (x−1)(x−2)(x−3)(x²+1): roots 1, 2, 3, ±i.
        // p(x) = x⁵ − 6x⁴ + 12x³ − 12x² + 11x − 6
        let c = [-6.0, 11.0, -12.0, 12.0, -6.0]; // coefficients a0..a4 of the monic poly
        let comp = Matrix::from_fn(5, 5, |i, j| {
            if j == 4 {
                -c[i]
            } else if i == j + 1 {
                1.0
            } else {
                0.0
            }
        });
        let e = Eigen::new(&comp).unwrap();
        let v = sorted_values(&e);
        let expect = [
            Complex::new(0.0, -1.0),
            Complex::new(0.0, 1.0),
            Complex::real(1.0),
            Complex::real(2.0),
            Complex::real(3.0),
        ];
        for (got, want) in v.iter().zip(&expect) {
            assert!((*got - *want).abs() < 1e-10, "{got} vs {want}");
        }
        assert!(residual(&comp, &e) < 1e-10);
    }

    #[test]
    fn random_general_matrices() {
        for n in [1, 3, 10, 40] {
            let a = random(n, n, 900 + n as u64);
            let e = Eigen::new(&a).unwrap();
            assert!(residual(&a, &e) < 1e-10 * n as f64, "n={n} residual {}", residual(&a, &e));
            // Trace = Σλ, det = Πλ.
            let sum = e.values.iter().fold(Complex::default(), |s, &z| s + z);
            assert!((sum.re - a.trace()).abs() < 1e-10 && sum.im.abs() < 1e-10);
            let prod = e.values.iter().fold(Complex::real(1.0), |s, &z| s * z);
            let det = super::super::lu::Lu::new(&a).unwrap().det();
            assert!((prod.re - det).abs() < 1e-8 * det.abs().max(1.0) && prod.im.abs() < 1e-8 * det.abs().max(1.0));
        }
    }

    #[test]
    fn defective_triangular_and_symmetric_path() {
        // Jordan block: eigenvalue 2 (×3), one eigenvector direction.
        let j = Matrix::from_rows(&[vec![2.0, 1.0, 0.0], vec![0.0, 2.0, 1.0], vec![0.0, 0.0, 2.0]]).unwrap();
        let e = Eigen::new(&j).unwrap();
        assert!(e.values.iter().all(|z| (z.re - 2.0).abs() < 1e-5 && z.im.abs() < 1e-5));
        let t = Matrix::from_rows(&[vec![1.0, 5.0, -3.0], vec![0.0, 4.0, 2.0], vec![0.0, 0.0, -2.0]]).unwrap();
        let v = sorted_values(&Eigen::new(&t).unwrap());
        assert!((v[0].re + 2.0).abs() < 1e-14 && (v[1].re - 1.0).abs() < 1e-14 && (v[2].re - 4.0).abs() < 1e-14);
        let s = spd(5, 3);
        let e = Eigen::new(&s).unwrap();
        assert!(e.is_real() && residual(&s, &e) < 1e-11);
        assert_eq!(Eigen::new(&Matrix::zeros(3, 3)).unwrap().values, vec![Complex::real(0.0); 3]);
    }
}
