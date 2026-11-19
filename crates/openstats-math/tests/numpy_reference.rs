//! Cross-checks against reference results computed independently with
//! NumPy/LAPACK (see `tests/reference/numpy.rs`).

#[rustfmt::skip]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
mod reference {
    include!("reference/numpy.rs");
}

use openstats_math::linalg::{Cholesky, Eigen, Lu, Matrix, Qr, Svd, SymmetricEigen};
use openstats_math::Complex;
use reference::*;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

#[test]
fn general_eigenvalues_match_lapack() {
    let a = Matrix::from_vec(6, 6, A.to_vec()).unwrap();
    let mut ev = Eigen::new(&a).unwrap().values;
    ev.sort_by(|p, q| p.re.total_cmp(&q.re).then(p.im.total_cmp(&q.im)));
    for k in 0..6 {
        assert!(
            (ev[k] - Complex::new(EIG_RE[k], EIG_IM[k])).abs() < 1e-12,
            "{k}: {} vs {}+{}i",
            ev[k],
            EIG_RE[k],
            EIG_IM[k]
        );
    }
    assert!(rel(Lu::new(&a).unwrap().det(), DET_A) < 1e-13);
}

#[test]
fn symmetric_eigenvalues_match_lapack() {
    let a = Matrix::from_vec(6, 6, A.to_vec()).unwrap();
    let s = a.try_add(&a.t()).unwrap();
    for e in [SymmetricEigen::new(&s).unwrap(), SymmetricEigen::jacobi(&s).unwrap()] {
        for k in 0..6 {
            assert!(rel(e.values[k], SYM_EIG[k]) < 1e-13, "{k}: {} vs {}", e.values[k], SYM_EIG[k]);
        }
    }
}

#[test]
fn svd_lstsq_pinv_match_lapack() {
    let b = Matrix::from_vec(7, 5, B.to_vec()).unwrap();
    let f = Svd::new(&b).unwrap();
    for k in 0..5 {
        assert!(rel(f.s[k], SV_B[k]) < 1e-13, "σ{k}");
    }
    assert!(rel(f.cond(), COND2_B) < 1e-12);
    let x_svd = f.solve_least_squares(&Y, None).unwrap();
    let x_qr = Qr::new(&b).unwrap().solve_least_squares(&Y).unwrap();
    let x_qrp = Qr::new_pivoted(&b).unwrap().solve_least_squares(&Y).unwrap();
    for k in 0..5 {
        assert!((x_svd[k] - LSTSQ[k]).abs() < 1e-13, "svd lstsq {k}");
        assert!((x_qr[k] - LSTSQ[k]).abs() < 1e-13, "qr lstsq {k}");
        assert!((x_qrp[k] - LSTSQ[k]).abs() < 1e-13, "pivoted qr lstsq {k}");
    }
    let r = Matrix::from_vec(4, 3, R.to_vec()).unwrap();
    let p = Svd::new(&r).unwrap().pinv(None);
    assert!(p.as_slice().iter().zip(PINV_R.iter()).all(|(u, v)| (u - v).abs() < 1e-14));
}

#[test]
fn hilbert_inverse_matches_lapack() {
    let h = Matrix::from_fn(6, 6, |i, j| 1.0 / (i + j + 1) as f64);
    // κ(H6) ≈ 1.5e7, so expect ~1e-9 relative agreement in the inverse.
    let via_lu = Lu::new(&h).unwrap().inverse().unwrap();
    let via_chol = Cholesky::new(&h).unwrap().inverse().unwrap();
    for (k, &want) in HINV6.iter().enumerate() {
        let (i, j) = (k / 6, k % 6);
        assert!(rel(via_lu[(i, j)], want) < 1e-8, "LU H⁻¹[{i},{j}]");
        assert!(rel(via_chol[(i, j)], want) < 1e-8, "Cholesky H⁻¹[{i},{j}]");
    }
    // The exact inverse has integer entries: H⁻¹[0][0] = 36.
    assert!((via_chol[(0, 0)] - 36.0).abs() < 1e-7);
}
