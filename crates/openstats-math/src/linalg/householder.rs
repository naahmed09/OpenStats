//! Householder reflectors H = I − τ·v·vᵀ with `v[0]` = 1 (LAPACK `dlarfg`
//! convention).

use super::blas::nrm2;

/// Compute a reflector that maps `x` to `β·e₁`. On return `x[0] = β` and
/// `x[1..]` holds `v[1..]` (`v[0]` = 1 implicitly). Returns τ (0 if `x` is
/// already a multiple of e₁).
pub fn make_reflector(x: &mut [f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return 0.0;
    }
    let alpha = x[0];
    let xnorm = nrm2(&x[1..]);
    if xnorm == 0.0 {
        return 0.0;
    }
    let beta = -alpha.signum() * alpha.hypot(xnorm);
    let beta = if alpha == 0.0 { -alpha.hypot(xnorm) } else { beta };
    let tau = (beta - alpha) / beta;
    let scale = 1.0 / (alpha - beta);
    for v in &mut x[1..] {
        *v *= scale;
    }
    x[0] = beta;
    tau
}

/// Apply H = I − τ·v·vᵀ (`v[0]` = 1, `v_tail` = `v[1..]`) from the left to the
/// rows `r0..r0+len(v)` and columns `c0..` of the row-major block `a`
/// (`stride` columns per row).
pub fn apply_left(v_tail: &[f64], tau: f64, a: &mut [f64], stride: usize, r0: usize, c0: usize, c1: usize) {
    if tau == 0.0 {
        return;
    }
    let mut w = vec![0.0; c1 - c0];
    // w = vᵀ·A
    w.copy_from_slice(&a[r0 * stride + c0..r0 * stride + c1]);
    for (k, &vk) in v_tail.iter().enumerate() {
        let row = &a[(r0 + 1 + k) * stride + c0..(r0 + 1 + k) * stride + c1];
        for (wj, &aj) in w.iter_mut().zip(row) {
            *wj += vk * aj;
        }
    }
    // A -= τ·v·wᵀ
    for (wj, aj) in w.iter().zip(&mut a[r0 * stride + c0..r0 * stride + c1]) {
        *aj -= tau * wj;
    }
    for (k, &vk) in v_tail.iter().enumerate() {
        let f = tau * vk;
        let row = &mut a[(r0 + 1 + k) * stride + c0..(r0 + 1 + k) * stride + c1];
        for (aj, &wj) in row.iter_mut().zip(&w) {
            *aj -= f * wj;
        }
    }
}

/// Apply H from the right to columns `c0..c0+len(v)` of rows `r0..r1`.
pub fn apply_right(v_tail: &[f64], tau: f64, a: &mut [f64], stride: usize, r0: usize, r1: usize, c0: usize) {
    if tau == 0.0 {
        return;
    }
    for i in r0..r1 {
        let row = &mut a[i * stride + c0..i * stride + c0 + 1 + v_tail.len()];
        let mut s = row[0];
        for (k, &vk) in v_tail.iter().enumerate() {
            s += vk * row[1 + k];
        }
        s *= tau;
        row[0] -= s;
        for (k, &vk) in v_tail.iter().enumerate() {
            row[1 + k] -= s * vk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflector_annihilates() {
        for x0 in [[3.0, 4.0, 0.0, 12.0], [-3.0, 4.0, 0.0, 12.0], [0.0, 1.0, 1.0, 1.0]] {
            let mut x = x0;
            let tau = make_reflector(&mut x);
            let beta = x[0];
            assert!((beta.abs() - nrm2(&x0)).abs() < 1e-14);
            // Apply H to the original vector (as a 4×1 matrix).
            let mut a = x0.to_vec();
            apply_left(&x[1..], tau, &mut a, 1, 0, 0, 1);
            assert!((a[0] - beta).abs() < 1e-14);
            assert!(a[1..].iter().all(|v| v.abs() < 1e-14));
        }
        let mut e1 = [5.0, 0.0, 0.0];
        assert_eq!(make_reflector(&mut e1), 0.0);
    }
}
