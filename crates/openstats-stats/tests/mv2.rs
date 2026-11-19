//! Correspondence analysis, multiple correspondence analysis and promax against NumPy.

mod common;
use common::mv2_data::*;
use openstats_math::Matrix;
use openstats_stats::multivariate::{correspondence_analysis, multiple_correspondence, promax};
use openstats_stats::report::{Reportable, SessionReport};

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn table() -> Vec<Vec<f64>> {
    CA_TABLE.chunks(4).map(<[f64]>::to_vec).collect()
}

fn labels(p: &str, n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("{p}{i}")).collect()
}

/// Orient every axis so the first row coordinate is positive (the sign of an axis is arbitrary).
fn oriented(row: &Matrix, col: &Matrix) -> (Matrix, Matrix) {
    let (mut r, mut c) = (row.clone(), col.clone());
    for a in 0..r.cols() {
        if r[(0, a)] < 0.0 {
            for i in 0..r.rows() {
                r[(i, a)] = -r[(i, a)];
            }
            for j in 0..c.rows() {
                c[(j, a)] = -c[(j, a)];
            }
        }
    }
    (r, c)
}

#[test]
fn simple_correspondence_matches_numpy() {
    let ca = correspondence_analysis(&table(), &labels("r", 5), &labels("c", 4), None).unwrap();
    assert_eq!(ca.singular_values.len(), 3);
    for (a, b) in ca.singular_values.iter().zip(CA_SV) {
        close(*a, *b, 1e-10, "singular value");
    }
    close(ca.total_inertia, CA_TOTAL_INERTIA, 1e-12, "inertia");
    close(ca.chi_square, CA_CHI2, 1e-10, "chi-square");
    let (row, col) = oriented(&ca.row_coord, &ca.col_coord);
    for i in 0..5 {
        for a in 0..3 {
            close(row[(i, a)], CA_ROW[i * 3 + a], 1e-9, "row coordinate");
            close(ca.row_contrib[(i, a)], CA_ROW_CTR[i * 3 + a], 1e-9, "row contribution");
            close(ca.row_quality[(i, a)], CA_ROW_COS[i * 3 + a], 1e-9, "row cos2");
        }
        close(ca.row_mass[i], CA_ROW_MASS[i], 1e-12, "mass");
    }
    for j in 0..4 {
        for a in 0..3 {
            close(col[(j, a)], CA_COL[j * 3 + a], 1e-9, "column coordinate");
        }
    }
    // Contributions of each axis add to one; inertia shares add to one.
    for a in 0..3 {
        close((0..5).map(|i| ca.row_contrib[(i, a)]).sum::<f64>(), 1.0, 1e-12, "row ctr sum");
        close((0..4).map(|j| ca.col_contrib[(j, a)]).sum::<f64>(), 1.0, 1e-12, "col ctr sum");
    }
    close(ca.row_inertia.iter().sum::<f64>(), 1.0, 1e-12, "row inertia");
    let two = correspondence_analysis(&table(), &labels("r", 5), &labels("c", 4), Some(2)).unwrap();
    assert_eq!(two.row_coord.cols(), 2);
}

#[test]
fn multiple_correspondence_matches_numpy() {
    let v = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let cols = vec![v(MCA_V0), v(MCA_V1), v(MCA_V2)];
    let names = vec!["v0".to_string(), "v1".to_string(), "v2".to_string()];
    let m = multiple_correspondence(&cols, &names, None).unwrap();
    let kk = MCA_NCOL as usize - 3;
    assert_eq!(m.col_coord.rows(), MCA_NCOL as usize);
    assert_eq!(m.singular_values.len(), kk);
    for (a, b) in m.singular_values.iter().zip(MCA_SV) {
        close(*a, *b, 1e-9, "sv");
    }
    close(m.total_inertia, MCA_TOTAL_INERTIA, 1e-10, "inertia");
    let (_, col) = oriented(&m.row_coord, &m.col_coord);
    for j in 0..MCA_NCOL as usize {
        for a in 0..kk.min(2) {
            close(col[(j, a)].abs(), MCA_COL[j * kk + a].abs(), 1e-8, "category coordinate");
        }
    }
    assert!(m.col_labels[0].starts_with("v0="));
}

#[test]
fn promax_matches_numpy() {
    let a = Matrix::from_fn(8, 3, |i, j| PM_A[i * 3 + j]);
    let q = Matrix::from_fn(3, 3, |i, j| PM_Q[i * 3 + j]);
    let v = a.matmul(&q).unwrap();
    let r = promax(&a, &v, &q, 4.0).unwrap();
    for i in 0..8 {
        for j in 0..3 {
            close(r.pattern[(i, j)], PM_PATTERN[i * 3 + j], 1e-9, "pattern");
            close(r.structure[(i, j)], PM_STRUCT[i * 3 + j], 1e-9, "structure");
        }
    }
    for i in 0..3 {
        close(r.factor_correlation[(i, i)], 1.0, 1e-9, "unit diagonal");
        for j in 0..3 {
            close(r.factor_correlation[(i, j)], PM_PHI[i * 3 + j], 1e-9, "phi");
        }
    }
    let mut rep = SessionReport::new("t");
    rep.push(r.report_with_names(&labels("x", 8)));
    rep.validate().unwrap();
}

#[test]
fn errors_and_reports() {
    let t = table();
    assert!(correspondence_analysis(&t[..1], &labels("r", 1), &labels("c", 4), None).is_err());
    let mut neg = t.clone();
    neg[0][0] = -1.0;
    assert!(correspondence_analysis(&neg, &labels("r", 5), &labels("c", 4), None).is_err());
    let mut empty = t.clone();
    empty[2] = vec![0.0; 4];
    assert!(correspondence_analysis(&empty, &labels("r", 5), &labels("c", 4), None).is_err());
    assert!(correspondence_analysis(&t, &labels("r", 4), &labels("c", 4), None).is_err());
    assert!(multiple_correspondence(&[vec!["a".into()]], &["x".into()], None).is_err());
    let a = Matrix::identity(3);
    assert!(promax(&a, &a, &a, 1.0).is_err());
    let ca = correspondence_analysis(&t, &labels("r", 5), &labels("c", 4), None).unwrap();
    let mut rep = SessionReport::new("t");
    rep.push(ca.report_section());
    rep.validate().unwrap();
    assert!(rep.render_text().contains("Column Contributions"));
    assert_eq!(rep.charts().len(), 1);
}
