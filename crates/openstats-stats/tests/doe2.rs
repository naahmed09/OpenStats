//! DOE completion (Phase 5o): mixture, optimal and split-plot designs, GF(4) arrays.
//! References: `tests/gen/doe2_ref.py` (numpy, exhaustive search, scipy HalfspaceIntersection).

mod common;
use common::doe2_data::*;
use openstats_stats::doe::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

fn names(q: usize) -> Vec<String> {
    (0..q).map(|i| format!("X{}", i + 1)).collect()
}

#[test]
fn scheffe_special_cubic_matches_lstsq() {
    let x: Vec<Vec<f64>> = MIX_X.chunks(3).map(|c| c.to_vec()).collect();
    let f = fit_mixture(&x, MIX_Y, &names(3), MixtureModel::SpecialCubic).unwrap();
    for (i, b) in MIX_COEF.iter().enumerate() {
        close(f.coef[i], *b, 1e-8);
        close(f.se[i], MIX_SE[i], 1e-7);
    }
    close(f.r2, MIX_R2, 1e-10);
    close(f.r2_adj, MIX_R2ADJ, 1e-10);
    let row = |s: &str| f.anova.iter().find(|r| r.source == s).unwrap();
    close(row("Linear").ss, MIX_SS_LIN, 1e-8);
    close(row("Quadratic").ss, MIX_SS_QUAD, 1e-8);
    close(row("Special Cubic").ss, MIX_SS_CUBIC, 1e-8);
    close(row("Residual Error").ss, MIX_SSE, 1e-8);
    close(row("Regression").p, MIX_P_REG, 1e-8);
    assert_eq!(row("Linear").df, 2);
    assert_eq!(row("Quadratic").df, 3);
    assert_eq!(row("Special Cubic").df, 1);
}

#[test]
fn mixture_errors() {
    let x = vec![vec![0.5, 0.5], vec![0.4, 0.4]];
    assert!(fit_mixture(&x, &[1.0, 2.0], &names(2), MixtureModel::Linear).is_err());
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert!(fit_mixture(&x, &[1.0, 2.0], &names(2), MixtureModel::Linear).is_err());
    // n <= p
}

#[test]
fn extreme_vertices_match_halfspace_intersection() {
    let d = extreme_vertices(&names(4), EV_LO, EV_HI, false, 1, None).unwrap();
    assert_eq!(d.runs(), EV_N);
    let mut got = d.points.clone();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (i, row) in got.iter().enumerate() {
        for j in 0..4 {
            close(row[j], EV_VERTS[i * 4 + j], 1e-8);
        }
    }
}

#[test]
fn lattice_sizes_follow_the_binomial_formula() {
    for (q, m, n) in [(3usize, 3usize, 10usize), (4, 2, 10), (5, 3, 35), (6, 2, 21)] {
        assert_eq!(simplex_lattice(&names(q), m, false, false, 1, None).unwrap().runs(), n, "q={q} m={m}");
    }
}

#[test]
fn optimal_designs_reach_the_exhaustive_optimum() {
    for (n, ld, ia) in [(7usize, OPT7_LOGDET, OPT7_IAVG), (9, OPT9_LOGDET, OPT9_IAVG)] {
        let mk = |c| OptimalOptions {
            runs: n,
            criterion: c,
            model: OptModel::Quadratic,
            levels: vec![3, 3],
            restarts: 30,
            seed: 11,
        };
        let d = optimal_design(&mk(Criterion::D)).unwrap();
        close(d.log_det, ld, 1e-9);
        let i = optimal_design(&mk(Criterion::I)).unwrap();
        close(i.avg_prediction_variance, ia, 1e-8);
    }
}

#[test]
fn gf4_arrays_are_orthogonal_arrays() {
    for (name, runs, cols) in [("L16(4^5)", 16usize, 5usize), ("L64(4^21)", 64, 21)] {
        let a = orthogonal_array(name).unwrap();
        assert_eq!((a.runs(), a.columns()), (runs, cols));
        assert!(a.is_orthogonal(), "{name}");
        assert!(a.levels.iter().all(|&l| l == 4));
    }
    let d = TaguchiDesign::new(orthogonal_array("L16(4^5)").unwrap(), &[("A", 4), ("B", 4), ("C", 4)]).unwrap();
    assert_eq!(d.runs(), 16);
}
