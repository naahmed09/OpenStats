//! Multivariate analysis against NumPy / SciPy / scikit-learn references (tests/gen/mv_ref.py).

mod common;
use common::mv_data::*;
use openstats_stats::model::Data;
use openstats_stats::multivariate::*;
use openstats_stats::report::{Reportable, SessionReport};

fn cols(flat: &[f64], n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..p).map(|j| (0..n).map(|i| flat[i * p + j]).collect()).collect()
}

fn names(p: usize) -> Vec<String> {
    (1..=p).map(|i| format!("V{i}")).collect()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

/// Compare matrices (row-major reference) column by column up to sign.
fn same_up_to_sign(m: &openstats_math::Matrix, reference: &[f64], ncols_ref: usize, k: usize, tol: f64, what: &str) {
    for j in 0..k {
        let mut dot = 0.0;
        for i in 0..m.rows() {
            dot += m[(i, j)] * reference[i * ncols_ref + j];
        }
        let sg = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..m.rows() {
            close(sg * m[(i, j)], reference[i * ncols_ref + j], tol, &format!("{what}[{i},{j}]"));
        }
    }
}

fn exact(m: &openstats_math::Matrix, reference: &[f64], tol: f64, what: &str) {
    assert_eq!(m.rows() * m.cols(), reference.len(), "{what} size");
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            close(m[(i, j)], reference[i * m.cols() + j], tol, &format!("{what}[{i},{j}]"));
        }
    }
}

fn iris() -> Vec<Vec<f64>> {
    cols(IRIS_X, 150, 4)
}

fn labels(s: &[&str]) -> Vec<String> {
    s.iter().map(|x| x.to_string()).collect()
}

// ------------------------------------------------------------------ PCA

#[test]
fn pca_correlation_matches_numpy() {
    let r = pca(&iris(), &names(4), &PcaOptions::default()).unwrap();
    for i in 0..4 {
        close(r.eigenvalues[i], PCA_CORR_EIG[i], 1e-10, "eigenvalue");
    }
    // The famous iris values.
    close(r.eigenvalues[0], 2.91849782, 1e-7, "iris λ1");
    close(r.eigenvalues[3], 0.02071484, 1e-6, "iris λ4");
    close(r.cumulative[1], 0.9581320720, 1e-9, "cumulative");
    close(r.total_variance, 4.0, 1e-12, "total");
    same_up_to_sign(&r.loadings, PCA_CORR_VEC, 4, 4, 1e-9, "loadings");
    // Scores follow the loadings' sign convention.
    for j in 0..4 {
        let mut dot = 0.0;
        for i in 0..4 {
            dot += r.loadings[(i, j)] * PCA_CORR_VEC[i * 4 + j];
        }
        let sg = dot.signum();
        for i in 0..150 {
            close(r.scores[(i, j)], sg * PCA_CORR_SCORES[i * 4 + j], 1e-9, "score");
        }
    }
    // Sign convention: the largest |loading| of each component is positive.
    for j in 0..4 {
        let mx = (0..4).map(|i| r.loadings[(i, j)]).fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a });
        assert!(mx > 0.0);
    }
    // Eigenvectors are orthonormal and scores are uncorrelated with variance λ.
    for a in 0..4 {
        let var: f64 = (0..150).map(|i| r.scores[(i, a)].powi(2)).sum::<f64>() / 149.0;
        close(var, r.eigenvalues[a], 1e-9, "score variance");
    }
}

#[test]
fn pca_covariance_matches_numpy() {
    let opts = PcaOptions { matrix: MatrixKind::Covariance, n_components: Some(2) };
    let r = pca(&iris(), &names(4), &opts).unwrap();
    for i in 0..4 {
        close(r.eigenvalues[i], PCA_COV_EIG[i], 1e-10, "cov eigenvalue");
    }
    same_up_to_sign(&r.loadings, PCA_COV_VEC, 4, 4, 1e-9, "cov loadings");
    assert_eq!(r.scores.cols(), 2);
    let mut dot = 0.0;
    for i in 0..4 {
        dot += r.loadings[(i, 0)] * PCA_COV_VEC[i * 4];
    }
    for i in 0..150 {
        close(r.scores[(i, 0)], dot.signum() * PCA_COV_SCORES[i * 4], 1e-9, "cov score");
    }
    close(r.total_variance, 4.572957046979866, 1e-9, "total variance = trace");
}

#[test]
fn pca_missing_values_project_and_errors() {
    let mut c = iris();
    c[1][3] = f64::NAN;
    c[2][10] = f64::NAN;
    let r = pca(&c, &names(4), &PcaOptions::default()).unwrap();
    assert_eq!((r.n, r.n_dropped), (148, 2));
    assert!(!r.rows_used.contains(&3) && !r.rows_used.contains(&10));
    // Equal to the analysis of the complete rows.
    let kept: Vec<Vec<f64>> = c.iter().map(|col| r.rows_used.iter().map(|&i| col[i]).collect()).collect();
    let r2 = pca(&kept, &names(4), &PcaOptions::default()).unwrap();
    for i in 0..4 {
        close(r.eigenvalues[i], r2.eigenvalues[i], 1e-12, "listwise");
    }
    // project() reproduces the scores and leaves incomplete rows missing.
    let proj = r.project(&c, 4).unwrap();
    assert!(proj[(3, 0)].is_nan());
    for (pos, &row) in r.rows_used.iter().enumerate().take(20) {
        close(proj[(row, 1)], r.scores[(pos, 1)], 1e-10, "project");
    }
    assert!(pca(&[vec![1.0, 2.0, 3.0]], &names(1), &PcaOptions::default()).is_err());
    assert!(
        pca(&[vec![1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]], &names(2), &PcaOptions::default()).is_err(),
        "constant variable"
    );
    assert!(pca(&[vec![1.0, 2.0], vec![1.0]], &names(2), &PcaOptions::default()).is_err(), "ragged");
    assert!(pca(&[vec![f64::INFINITY, 2.0], vec![1.0, 2.0]], &names(2), &PcaOptions::default()).is_err());
}

// ------------------------------------------------------------------ Factor analysis

fn fa_cols() -> Vec<Vec<f64>> {
    cols(FA_X, 200, 6)
}

fn fa(method: FactorMethod, m: usize, rotation: Rotation, kaiser: bool) -> FactorAnalysis {
    let mut o = FactorOptions::new(m);
    o.method = method;
    o.rotation = rotation;
    o.kaiser = kaiser;
    o.scores = true;
    factor_analysis(&fa_cols(), &names(6), &o).unwrap()
}

#[test]
fn factor_principal_components_and_rotations() {
    let r = fa(FactorMethod::PrincipalComponents, 2, Rotation::None, true);
    exact(&r.loadings, FA_PC_L2, 1e-9, "PC loadings");
    for i in 0..6 {
        close(r.communalities[i], (0..2).map(|j| FA_PC_L2[i * 2 + j].powi(2)).sum(), 1e-9, "communality");
        close(r.uniquenesses[i], 1.0 - r.communalities[i], 1e-12, "uniqueness");
    }
    close(r.total_variance, 6.0, 1e-12, "total");
    let v = fa(FactorMethod::PrincipalComponents, 2, Rotation::Varimax, true);
    exact(&v.loadings, FA_PC_VARIMAX, 1e-7, "varimax");
    // Rotation is orthogonal: communalities unchanged, loadings = unrotated · T.
    for i in 0..6 {
        close(v.communalities[i], r.communalities[i], 1e-10, "rotation keeps communality");
    }
    let back = v.unrotated.matmul(&v.rotation_matrix).unwrap();
    exact(&back, v.loadings.as_slice(), 1e-10, "L·T");
    let tt = v.rotation_matrix.t_matmul(&v.rotation_matrix).unwrap();
    close(tt[(0, 1)], 0.0, 1e-9, "T orthogonal");
    close(tt[(0, 0)], 1.0, 1e-9, "T orthogonal");
    exact(
        &fa(FactorMethod::PrincipalComponents, 2, Rotation::Varimax, false).loadings,
        FA_PC_VARIMAX_RAW,
        1e-7,
        "varimax raw",
    );
    exact(
        &fa(FactorMethod::PrincipalComponents, 2, Rotation::Quartimax, true).loadings,
        FA_PC_QUARTIMAX,
        1e-7,
        "quartimax",
    );
    let e3 = fa(FactorMethod::PrincipalComponents, 3, Rotation::Equamax, true);
    exact(&e3.loadings, FA_PC3_EQUAMAX, 1e-6, "equamax");
    let o3 = fa(FactorMethod::PrincipalComponents, 3, Rotation::Orthomax(0.5), false);
    exact(&o3.loadings, FA_PC3_ORTHOMAX, 1e-6, "orthomax 0.5");
    // Explained variance is ordered and sums to the communalities.
    assert!(v.variance[0] >= v.variance[1]);
    close(v.variance.iter().sum::<f64>(), v.communalities.iter().sum(), 1e-10, "variance sum");
    // Regression scores.
    exact(&v.score_coefficients, FA_PC_SCORE_COEF, 1e-8, "score coefficients");
    exact(v.scores.as_ref().unwrap(), FA_PC_SCORES, 1e-8, "scores");
}

#[test]
fn factor_maximum_likelihood_matches_scipy() {
    for m in [1usize, 2] {
        let r = fa(FactorMethod::MaximumLikelihood, m, Rotation::None, true);
        assert!(r.converged);
        let (lref, psi, f, chi, df, pv) = if m == 1 {
            (FA_ML1_L, FA_ML1_PSI, FA_ML1_F, FA_ML1_CHI, FA_ML1_DF, FA_ML1_P)
        } else {
            (FA_ML2_L, FA_ML2_PSI, FA_ML2_F, FA_ML2_CHI, FA_ML2_DF, FA_ML2_P)
        };
        exact(&r.loadings, lref, 2e-4, "ML loadings");
        for i in 0..6 {
            close(r.uniquenesses[i], psi[i], 2e-4, "ML uniqueness");
        }
        close(r.discrepancy.unwrap(), f, 1e-6, "ML discrepancy");
        let t = r.fit_test.expect("fit test");
        close(t.chi_square, chi, 1e-4, "chi-square");
        assert_eq!(t.df as f64, df);
        close(t.p_value, pv, 1e-3, "chi-square p");
        // The model reproduces the correlation matrix better than chance: uniqueness + communality = 1.
        for i in 0..6 {
            close(r.communalities[i] + r.uniquenesses[i], 1.0, 1e-9, "sum to 1");
        }
    }
    let rot = fa(FactorMethod::MaximumLikelihood, 2, Rotation::Varimax, true);
    exact(&rot.loadings, FA_ML2_VARIMAX, 5e-4, "ML varimax");
    // m = 2 with 6 variables has df = 4 > 0; m = 3 has df = 0 so no test is reported.
    let m3 = fa(FactorMethod::MaximumLikelihood, 3, Rotation::None, true);
    assert!(m3.fit_test.is_none());
    // Covariance-matrix ML.
    let mut o = FactorOptions::new(1);
    o.method = FactorMethod::MaximumLikelihood;
    o.matrix = MatrixKind::Covariance;
    o.rotation = Rotation::None;
    let c = factor_analysis(&fa_cols(), &names(6), &o).unwrap();
    exact(&c.loadings, FA_MLCOV1_L, 5e-4, "cov ML loadings");
    for i in 0..6 {
        close(c.uniquenesses[i], FA_MLCOV1_PSI[i], 5e-4, "cov ML uniqueness");
    }
}

#[test]
fn factor_input_errors() {
    let c = fa_cols();
    let o = |m| FactorOptions::new(m);
    assert!(factor_analysis(&c, &names(6), &o(0)).is_err());
    assert!(factor_analysis(&c, &names(6), &o(7)).is_err());
    let mut ml = o(6);
    ml.method = FactorMethod::MaximumLikelihood;
    assert!(factor_analysis(&c, &names(6), &ml).is_err(), "ML needs m < p");
    let mut bad = o(2);
    bad.rotation = Rotation::Orthomax(5.0);
    assert!(factor_analysis(&c, &names(6), &bad).is_err());
    assert!(factor_analysis(&c[..1], &names(1), &o(1)).is_err());
    let few: Vec<Vec<f64>> = c.iter().map(|x| x[..4].to_vec()).collect();
    assert!(factor_analysis(&few, &names(6), &o(1)).is_err());
    // Perfectly collinear variables: ML cannot invert the matrix.
    let mut col = c.clone();
    col[5] = col[0].iter().map(|v| 2.0 * v).collect();
    let mut ml2 = o(1);
    ml2.method = FactorMethod::MaximumLikelihood;
    assert!(factor_analysis(&col, &names(6), &ml2).is_err());
}

// ------------------------------------------------------------------ Hierarchical clustering

fn hc_cols() -> Vec<Vec<f64>> {
    cols(HC_X, 30, 3)
}

fn check_linkage(l: &Linkage, z: &[f64], what: &str) {
    let n = l.n;
    assert_eq!(l.steps.len(), n - 1);
    for (i, st) in l.steps.iter().enumerate() {
        let (a, b) = (z[i * 4] as usize, z[i * 4 + 1] as usize);
        let mut mine = [st.left, st.right];
        mine.sort();
        let mut theirs = [a, b];
        theirs.sort();
        assert_eq!(mine, theirs, "{what} step {i} pair");
        close(st.distance, z[i * 4 + 2], 1e-9, &format!("{what} step {i} height"));
        assert_eq!(st.size as f64, z[i * 4 + 3], "{what} step {i} size");
    }
}

fn partition_of(labels: &[usize]) -> Vec<usize> {
    let mut seen: Vec<usize> = Vec::new();
    labels
        .iter()
        .map(|l| match seen.iter().position(|x| x == l) {
            Some(p) => p,
            None => {
                seen.push(*l);
                seen.len() - 1
            }
        })
        .collect()
}

#[test]
fn hierarchical_matches_scipy_for_every_method() {
    let c = hc_cols();
    for (m, z, tag) in [
        (LinkageMethod::Single, HC_SINGLE, "single"),
        (LinkageMethod::Complete, HC_COMPLETE, "complete"),
        (LinkageMethod::Average, HC_AVERAGE, "average"),
        (LinkageMethod::Centroid, HC_CENTROID, "centroid"),
        (LinkageMethod::Median, HC_MEDIAN, "median"),
        (LinkageMethod::Ward, HC_WARD, "ward"),
        (LinkageMethod::McQuitty, HC_MCQUITTY, "mcquitty"),
    ] {
        let l = hierarchical(&c, m, Distance::Euclidean, false).unwrap();
        check_linkage(&l, z, tag);
    }
    // Other metrics (Lance–Williams applied to the distances as given).
    check_linkage(
        &hierarchical(&c, LinkageMethod::Average, Distance::SquaredEuclidean, false).unwrap(),
        HC_SQEUC_AVG,
        "sqeuc avg",
    );
    check_linkage(
        &hierarchical(&c, LinkageMethod::Single, Distance::SquaredEuclidean, false).unwrap(),
        HC_SQEUC_SINGLE,
        "sqeuc single",
    );
    check_linkage(
        &hierarchical(&c, LinkageMethod::Complete, Distance::Manhattan, false).unwrap(),
        HC_CITY_COMPLETE,
        "city",
    );
    check_linkage(&hierarchical(&c, LinkageMethod::Average, Distance::Pearson, false).unwrap(), HC_CORR_AVG, "pearson");
    check_linkage(&hierarchical(&c, LinkageMethod::Ward, Distance::Euclidean, true).unwrap(), HC_STD_WARD, "std ward");
}

#[test]
fn hierarchical_cuts_similarity_and_dendrogram() {
    let l = hierarchical(&hc_cols(), LinkageMethod::Average, Distance::Euclidean, false).unwrap();
    for (k, r) in [(2usize, HC_AVERAGE_CUT2), (3, HC_AVERAGE_CUT3), (5, HC_AVERAGE_CUT5)] {
        let mine = l.cut_clusters(k).unwrap();
        let theirs: Vec<usize> = r.to_vec();
        assert_eq!(partition_of(&mine), partition_of(&theirs), "cut {k}");
        assert_eq!(*mine.iter().max().unwrap(), k);
    }
    assert_eq!(l.cut_clusters(1).unwrap(), vec![1; 30]);
    assert_eq!(l.cut_clusters(30).unwrap(), (1..=30).collect::<Vec<_>>());
    assert!(l.cut_clusters(0).is_err() && l.cut_clusters(31).is_err());
    // Similarity: 100 at zero distance, 0 at the largest step; cutting at a level matches cut by count.
    let last = l.steps.last().unwrap();
    close(last.similarity, 0.0, 1e-12, "last similarity");
    assert!(l.steps[0].similarity > 80.0);
    let k = l.clusters_at_similarity(l.steps[26].similarity - 1e-9);
    assert_eq!(l.cut_similarity(l.steps[26].similarity - 1e-9).unwrap(), l.cut_clusters(k).unwrap());
    assert_eq!(l.clusters_at_similarity(-1.0), 1);
    assert_eq!(l.clusters_at_similarity(101.0), 30);
    // Leaf order is a permutation and every dendrogram path joins existing positions.
    let mut o = l.order.clone();
    o.sort();
    assert_eq!(o, (0..30).collect::<Vec<_>>());
    let paths = l.dendrogram_paths();
    assert_eq!(paths.len(), 29);
    for (x, y) in &paths {
        assert!(x.iter().chain(y).all(|v| v.is_finite()));
    }
    // Labels: the cluster formed is named after its smallest observation.
    for st in &l.steps {
        assert_eq!(st.label_new, st.label_left.min(st.label_right));
        assert!(st.label_left <= st.label_right);
    }
}

#[test]
fn variable_clustering_matches_scipy() {
    let c = fa_cols();
    check_linkage(&cluster_variables(&c, &names(6), LinkageMethod::Average, false).unwrap(), HC_VAR_CORR, "1-r");
    check_linkage(&cluster_variables(&c, &names(6), LinkageMethod::Average, true).unwrap(), HC_VAR_ABS, "1-|r|");
    assert!(cluster_variables(&c[..1], &names(1), LinkageMethod::Average, false).is_err());
}

#[test]
fn hierarchical_errors_and_missing_rows() {
    let mut c = hc_cols();
    c[1][4] = f64::NAN;
    let l = hierarchical(&c, LinkageMethod::Single, Distance::Euclidean, false).unwrap();
    assert_eq!((l.n, l.n_dropped), (29, 1));
    assert!(!l.rows_used.contains(&4));
    assert!(hierarchical(&[vec![1.0]], LinkageMethod::Single, Distance::Euclidean, false).is_err());
    // A constant variable cannot be standardised.
    let k = vec![vec![1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]];
    assert!(hierarchical(&k, LinkageMethod::Single, Distance::Euclidean, true).is_err());
    // A constant row has no Pearson distance.
    let flat = vec![vec![1.0, 2.0, 3.0], vec![1.0, 5.0, 3.0], vec![1.0, 7.0, 3.0]];
    assert!(hierarchical(&flat, LinkageMethod::Single, Distance::Pearson, false).is_err());
    // Equal distances do not break the algorithm.
    let ties = vec![vec![0.0, 1.0, 2.0, 3.0]];
    let t = hierarchical(&ties, LinkageMethod::Complete, Distance::Euclidean, false).unwrap();
    assert_eq!(t.steps.len(), 3);
}

// ------------------------------------------------------------------ K-means

#[test]
fn kmeans_matches_scikit_learn_on_separated_clusters() {
    let c = cols(KM_X, 75, 2);
    for init in [KMeansInit::PlusPlus(3), KMeansInit::Random(9), KMeansInit::Rows(vec![0, 1, 2])] {
        let mut o = KMeansOptions::new(3);
        o.init = init;
        let r = kmeans(&c, &o).unwrap();
        // The global optimum for well separated clusters equals scikit-learn's best of 20 starts.
        if r.summary.total_within_ss > KM_INERTIA * (1.0 + 1e-9) {
            continue; // a poor local optimum from this start; the next must still be valid below
        }
        close(r.summary.total_within_ss, KM_INERTIA, 1e-9, "inertia");
        assert_eq!(partition_of(&r.labels), partition_of(&KM_LABELS.iter().map(|&l| l + 1).collect::<Vec<_>>()));
        assert!(r.converged);
        // Centroids equal scikit-learn's up to order.
        for k in 0..3 {
            let found =
                (0..3).any(|q| (0..2).all(|v| (r.summary.centroids[(k, v)] - KM_CENTERS[q * 2 + v]).abs() < 1e-9));
            assert!(found, "centroid {k}");
        }
    }
    // At least the k-means++ start reaches the optimum.
    let r = kmeans(&c, &KMeansOptions::new(3)).unwrap();
    close(r.summary.total_within_ss, KM_INERTIA, 1e-9, "default inertia");
    // Statistics are internally consistent.
    assert_eq!(r.summary.sizes.iter().sum::<usize>(), 75);
    let tot: f64 = r.summary.within_ss.iter().sum();
    close(tot, r.summary.total_within_ss, 1e-12, "sum of SS");
    for k in 0..3 {
        assert!(r.summary.maximum_distance[k] >= r.summary.average_distance[k]);
    }
    close(r.summary.centroid_distances[(0, 0)], 0.0, 1e-12, "self distance");
    close(r.summary.centroid_distances[(0, 1)], r.summary.centroid_distances[(1, 0)], 1e-12, "symmetry");
    // Distances recorded per observation.
    assert_eq!(r.distances.len(), 75);
}

#[test]
fn kmeans_local_optimality_and_options() {
    let c = cols(KM_Y, 60, 3);
    let mut best = f64::INFINITY;
    for seed in 0..20u64 {
        let mut o = KMeansOptions::new(4);
        o.init = KMeansInit::Random(seed);
        let r = kmeans(&c, &o).unwrap();
        best = best.min(r.summary.total_within_ss);
        // No single-observation transfer can lower the SS (Hartigan optimality).
        let k = r.k;
        let mut cent = vec![[0.0; 3]; k];
        for i in 0..60 {
            for v in 0..3 {
                cent[r.labels[i] - 1][v] += c[v][i] / r.summary.sizes[r.labels[i] - 1] as f64;
            }
        }
        for i in 0..60 {
            let a = r.labels[i] - 1;
            if r.summary.sizes[a] <= 1 {
                continue;
            }
            let d = |q: usize| (0..3).map(|v| (c[v][i] - cent[q][v]).powi(2)).sum::<f64>();
            let na = r.summary.sizes[a] as f64;
            let remove = na / (na - 1.0) * d(a);
            for b in 0..k {
                if b != a {
                    let nb = r.summary.sizes[b] as f64;
                    assert!(
                        nb / (nb + 1.0) * d(b) >= remove * (1.0 - 1e-9),
                        "seed {seed}: observation {i} should move"
                    );
                }
            }
        }
    }
    // The optimal-transfer refinement does at least as well as scikit-learn's best of 30 Lloyd starts.
    assert!(best <= KM_Y_BEST_INERTIA * (1.0 + 1e-9), "{best} vs {KM_Y_BEST_INERTIA}");
    // Reproducible for a seed; different seeds may differ.
    let mut o = KMeansOptions::new(3);
    o.init = KMeansInit::Random(5);
    assert_eq!(kmeans(&c, &o).unwrap().labels, kmeans(&c, &o).unwrap().labels);
    // Starting partition, standardisation, missing rows.
    let part: Vec<usize> = (0..60).map(|i| i % 3 + 1).collect();
    let mut o = KMeansOptions::new(3);
    o.init = KMeansInit::Partition(part);
    o.standardize = true;
    let r = kmeans(&c, &o).unwrap();
    assert!(r.standardized);
    assert_eq!(r.summary.sizes.iter().sum::<usize>(), 60);
    let mut m = c.clone();
    m[0][2] = f64::NAN;
    let r = kmeans(&m, &KMeansOptions::new(3)).unwrap();
    assert_eq!((r.n, r.n_dropped), (59, 1));
    // Errors.
    assert!(kmeans(&c, &KMeansOptions::new(1)).is_err());
    assert!(kmeans(&c, &KMeansOptions::new(61)).is_err());
    let mut bad = KMeansOptions::new(3);
    bad.init = KMeansInit::Rows(vec![0, 1]);
    assert!(kmeans(&c, &bad).is_err());
    bad.init = KMeansInit::Partition(vec![1; 60]);
    assert!(kmeans(&c, &bad).is_err(), "empty starting clusters");
    bad.init = KMeansInit::Partition(vec![4; 60]);
    assert!(kmeans(&c, &bad).is_err());
}

// ------------------------------------------------------------------ Discriminant analysis

fn species() -> Vec<String> {
    labels(IRIS_SPECIES)
}

fn da(quadratic: bool, priors: Priors) -> Discriminant {
    let o = DiscriminantOptions { quadratic, priors, cross_validate: true };
    discriminant(&iris(), &names(4), &species(), &o).unwrap()
}

fn check_da(d: &Discriminant, post: &[f64], pred: &[usize], loo: &[usize], what: &str) {
    for i in 0..150 {
        for k in 0..3 {
            close(d.posterior[(i, k)], post[i * 3 + k], 1e-8, &format!("{what} posterior"));
        }
        assert_eq!(d.predicted[i], pred[i], "{what} prediction {i}");
        assert_eq!(d.loo_predicted.as_ref().unwrap()[i], loo[i], "{what} loo {i}");
    }
}

#[test]
fn linear_discriminant_matches_numpy() {
    for (tag, priors, coef, cst, post, pred, loo, loop_) in [
        ("equal", Priors::Equal, DA_EQ_COEF, DA_EQ_CONST, DA_EQ_POST, DA_EQ_PRED, DA_EQ_LOO, DA_EQ_LOOPOST),
        (
            "prop",
            Priors::Proportional,
            DA_PROP_COEF,
            DA_PROP_CONST,
            DA_PROP_POST,
            DA_PROP_PRED,
            DA_PROP_LOO,
            DA_PROP_LOOPOST,
        ),
        (
            "given",
            Priors::Given(vec![0.5, 0.3, 0.2]),
            DA_GIVEN_COEF,
            DA_GIVEN_CONST,
            DA_GIVEN_POST,
            DA_GIVEN_PRED,
            DA_GIVEN_LOO,
            DA_GIVEN_LOOPOST,
        ),
    ] {
        let d = da(false, priors);
        exact(&d.coefficients, coef, 1e-9, &format!("{tag} coefficients"));
        for k in 0..3 {
            close(d.constants[k], cst[k], 1e-9, &format!("{tag} constant"));
        }
        check_da(&d, post, pred, loo, tag);
        let lp = d.loo_posterior.as_ref().unwrap();
        for i in 0..150 {
            for k in 0..3 {
                close(lp[(i, k)], loop_[i * 3 + k], 1e-7, &format!("{tag} loo posterior"));
            }
        }
        // Confusion matrix agrees with the predictions.
        for t in 0..3 {
            for q in 0..3 {
                let c = (0..150).filter(|&i| d.true_group[i] == t && d.predicted[i] == q).count();
                assert_eq!(d.confusion[t][q], c);
            }
        }
    }
    let d = da(false, Priors::Equal);
    assert_eq!(d.groups, labels(&["setosa", "versicolor", "virginica"]));
    assert_eq!(d.counts, vec![50, 50, 50]);
    exact(&d.means, DA_MEANS, 1e-12, "means");
    exact(&d.pooled_covariance, DA_POOLED, 1e-10, "pooled");
    exact(&d.squared_distances, DA_MAHAL, 1e-9, "Mahalanobis");
    // The famous 3 resubstitution errors, and 3 under leave-one-out for iris with equal priors.
    close(d.accuracy(), 147.0 / 150.0, 1e-12, "accuracy");
    close(d.loo_accuracy().unwrap(), 1.0 - 3.0 / 150.0, 1e-12, "loo accuracy");
    // Canonical functions: eigenvalues of W⁻¹B, unit pooled within variance, uncorrelated scores.
    let c = d.canonical.as_ref().unwrap();
    assert_eq!(c.eigenvalues.len(), 2);
    for k in 0..2 {
        close(c.eigenvalues[k], DA_CANON_EIG[k], 1e-9, "canonical eigenvalue");
    }
    close(c.proportion.iter().sum::<f64>(), 1.0, 1e-12, "proportions");
    close(c.eigenvalues[0], 32.19192920, 1e-7, "iris λ1");
    for k in 0..2 {
        let w: f64 = (0..150).map(|i| c.scores[(i, k)].powi(2)).sum::<f64>();
        let mut within = 0.0;
        for i in 0..150 {
            within += (c.scores[(i, k)] - c.centroids[(d.true_group[i], k)]).powi(2);
        }
        close(within / 147.0, 1.0, 1e-9, "unit within-group variance");
        assert!(w > 0.0);
    }
    // classify() agrees with the stored predictions.
    let x: Vec<f64> = (0..4).map(|v| iris()[v][100]).collect();
    let (g, post) = d.classify(&x).unwrap();
    assert_eq!(g, d.predicted[100]);
    close(post.iter().sum::<f64>(), 1.0, 1e-12, "posterior sum");
    assert!(d.classify(&[1.0, 2.0]).is_err());
}

#[test]
fn quadratic_discriminant_matches_scikit_learn() {
    for (tag, priors, post, pred, loo) in [
        ("equal", Priors::Equal, DA_QEQ_POST, DA_QEQ_PRED, DA_QEQ_LOO),
        ("given", Priors::Given(vec![0.5, 0.3, 0.2]), DA_QGIVEN_POST, DA_QGIVEN_PRED, DA_QGIVEN_LOO),
    ] {
        let d = da(true, priors);
        assert!(d.quadratic && d.canonical.is_none());
        check_da(&d, post, pred, loo, tag);
        assert_eq!(d.group_covariances.len(), 3);
    }
}

#[test]
fn discriminant_missing_rows_and_errors() {
    let mut c = iris();
    c[0][5] = f64::NAN;
    let mut g = species();
    g[7] = String::new();
    let d = discriminant(&c, &names(4), &g, &DiscriminantOptions::default()).unwrap();
    assert_eq!((d.n, d.n_dropped), (148, 2));
    assert!(!d.rows_used.contains(&5) && !d.rows_used.contains(&7));
    // One group only.
    let one = vec!["a".to_string(); 150];
    assert!(discriminant(&iris(), &names(4), &one, &DiscriminantOptions::default()).is_err());
    // Mismatched lengths.
    assert!(discriminant(&iris(), &names(4), &g[..10], &DiscriminantOptions::default()).is_err());
    // Bad priors.
    let bad = DiscriminantOptions { priors: Priors::Given(vec![0.5, 0.5]), ..Default::default() };
    assert!(discriminant(&iris(), &names(4), &species(), &bad).is_err());
    let neg = DiscriminantOptions { priors: Priors::Given(vec![0.5, -0.1, 0.6]), ..Default::default() };
    assert!(discriminant(&iris(), &names(4), &species(), &neg).is_err());
    // A predictor that is a multiple of another makes the pooled covariance singular.
    let mut sing = iris();
    sing[3] = sing[2].iter().map(|v| 3.0 * v).collect();
    assert!(discriminant(&sing, &names(4), &species(), &DiscriminantOptions::default()).is_err());
    // Quadratic needs enough observations in each group.
    let few: Vec<Vec<f64>> = iris().iter().map(|c| c[..7].to_vec()).collect();
    let g7 = labels(&["a", "a", "a", "b", "b", "b", "b"]);
    let q = DiscriminantOptions { quadratic: true, ..Default::default() };
    assert!(discriminant(&few, &names(4), &g7, &q).is_err());
    // Numeric group labels sort numerically.
    let gn: Vec<String> = (0..150).map(|i| ((i / 50) * 5 + 1).to_string()).collect();
    let dn =
        discriminant(&iris(), &names(4), &gn, &DiscriminantOptions { cross_validate: false, ..Default::default() })
            .unwrap();
    assert_eq!(dn.groups, labels(&["1", "6", "11"]));
    assert!(dn.loo_predicted.is_none());
}

// ------------------------------------------------------------------ MANOVA

fn check_tests(t: &ManovaTerm, r: &[f64], what: &str) {
    assert_eq!(t.tests.len(), 4);
    for (i, x) in t.tests.iter().enumerate() {
        let b = i * 5;
        close(x.statistic, r[b], 1e-8, &format!("{what} {} statistic", x.name));
        close(x.f, r[b + 1], 1e-7, &format!("{what} {} F", x.name));
        close(x.df1, r[b + 2], 1e-9, &format!("{what} {} df1", x.name));
        close(x.df2, r[b + 3], 1e-9, &format!("{what} {} df2", x.name));
        close(x.p_value, r[b + 4], 1e-6, &format!("{what} {} p", x.name));
    }
}

#[test]
fn manova_one_way_on_iris() {
    let sp = species();
    let data = Data::new().with_factor("Species", &sp).unwrap();
    let m = manova(&iris(), &names(4), &data, "Species", &ManovaOptions::default()).unwrap();
    assert_eq!((m.n, m.error_df), (150, 147));
    assert_eq!(m.terms.len(), 1);
    let t = &m.terms[0];
    assert_eq!(t.label, "Species");
    assert_eq!(t.df, 2);
    exact(&t.hypothesis, MV1_H, 1e-8, "H");
    exact(&m.error_sscp, MV1_E, 1e-8, "E");
    for k in 0..2 {
        close(t.eigenvalues[k], MV1_THETA[k], 1e-8, "theta");
    }
    check_tests(t, MV1_STATS, "iris");
    // Published values (R summary.manova): Wilks 0.02344, approx F 199.15 (8, 288); Pillai 1.1919 / 53.466 (8, 290);
    // Hotelling-Lawley 32.477 / 580.53 (8, 286); Roy 32.192 / 1167.0 (4, 145).
    let w = &t.tests[0];
    close(w.statistic, 0.0234386, 1e-4, "Wilks");
    close(w.f, 199.145, 1e-4, "Wilks F");
    assert_eq!((w.df1, w.df2), (8.0, 288.0));
    assert!(w.p_value < 1e-30);
    close(t.tests[2].statistic, 1.19190, 1e-4, "Pillai");
    close(t.tests[2].f, 53.4665, 1e-4, "Pillai F");
    close(t.tests[1].f, 580.532, 1e-4, "HL F");
    close(t.tests[3].f, 1166.96, 1e-4, "Roy F");
    // Treatment coding gives the same tests.
    let tc = manova(&iris(), &names(4), &data, "Species", &ManovaOptions { treatment_coding: true }).unwrap();
    check_tests(&tc.terms[0], MV1_STATS, "iris treatment");
}

#[test]
fn manova_two_groups_equals_hotelling_t2() {
    let sub: Vec<usize> = MV2G_SUB.to_vec();
    let rows: Vec<usize> = (0..150).filter(|&i| sub[i] == 1).collect();
    let c: Vec<Vec<f64>> = iris().iter().map(|col| rows.iter().map(|&i| col[i]).collect()).collect();
    let g: Vec<String> = rows.iter().map(|&i| species()[i].clone()).collect();
    let data = Data::new().with_factor("G", &g).unwrap();
    let m = manova(&c, &names(4), &data, "G", &ManovaOptions::default()).unwrap();
    for x in &m.terms[0].tests {
        assert!(x.exact, "{} exact for one df", x.name);
        close(x.f, MV2G_F, 1e-8, &format!("{} F = Hotelling F", x.name));
        close(x.p_value, MV2G_P, 1e-6, &format!("{} p", x.name));
        close(x.df1, 4.0, 1e-12, "df1");
        close(x.df2, 95.0, 1e-9, "df2");
    }
}

#[test]
fn manova_two_factor_unbalanced_adjusted() {
    let a = labels(MV3_A);
    let b = labels(MV3_B);
    let y = cols(MV3_Y, a.len(), 3);
    let data = Data::new().with_factor("A", &a).unwrap().with_factor("B", &b).unwrap();
    let m = manova(&y, &names(3), &data, "A + B + A*B", &ManovaOptions::default()).unwrap();
    assert_eq!(m.error_df as f64, MV3_EDF);
    exact(&m.error_sscp, MV3_E, 1e-8, "E");
    assert_eq!(m.terms.len(), 3);
    for (t, tag, stats, h, theta, df) in [
        (&m.terms[0], "A", MV3_A_STATS, MV3_A_H, MV3_A_THETA, MV3_A_DF),
        (&m.terms[1], "B", MV3_B_STATS, MV3_B_H, MV3_B_THETA, MV3_B_DF),
        (&m.terms[2], "A*B", MV3_AB_STATS, MV3_AB_H, MV3_AB_THETA, MV3_AB_DF),
    ] {
        assert_eq!(t.label, tag);
        assert_eq!(t.df as f64, df);
        exact(&t.hypothesis, h, 1e-7, &format!("H {tag}"));
        for k in 0..theta.len().min(t.eigenvalues.len()) {
            if theta[k] > 1e-9 {
                close(t.eigenvalues[k], theta[k], 1e-7, &format!("theta {tag}"));
            }
        }
        check_tests(t, stats, tag);
    }
    // `A | B` is the same model.
    let m2 = manova(&y, &names(3), &data, "A*B", &ManovaOptions::default()).unwrap();
    assert_eq!(m2.terms.len(), 3);
    check_tests(&m2.terms[0], MV3_A_STATS, "A via A*B");
}

#[test]
fn manova_missing_values_and_errors() {
    let sp = species();
    let mut c = iris();
    c[2][3] = f64::NAN;
    let data = Data::new().with_factor("Species", &sp).unwrap();
    let m = manova(&c, &names(4), &data, "Species", &ManovaOptions::default()).unwrap();
    assert_eq!((m.n, m.n_dropped), (149, 1));
    assert!(!m.rows_used.contains(&3));
    // A missing factor level drops the row as well.
    let mut sp2 = sp.clone();
    sp2[10] = String::new();
    let data2 = Data::new().with_factor("Species", &sp2).unwrap();
    assert_eq!(manova(&iris(), &names(4), &data2, "Species", &ManovaOptions::default()).unwrap().n, 149);
    assert!(manova(&iris()[..1], &names(1), &data, "Species", &ManovaOptions::default()).is_err(), "one response");
    assert!(manova(&iris(), &names(4), &data, "", &ManovaOptions::default()).is_err(), "no terms");
    assert!(manova(&iris(), &names(4), &data, "Nope", &ManovaOptions::default()).is_err(), "unknown term");
    // Linearly dependent responses.
    let mut dep = iris();
    dep[3] = dep[0].iter().zip(&dep[1]).map(|(a, b)| a + b).collect();
    assert!(manova(&dep, &names(4), &data, "Species", &ManovaOptions::default()).is_err());
    // Too few observations for the error degrees of freedom.
    let tiny: Vec<Vec<f64>> = iris().iter().map(|c| c[..5].to_vec()).collect();
    let d5 = Data::new().with_factor("G", &labels(&["a", "a", "b", "b", "c"])).unwrap();
    assert!(manova(&tiny, &names(4), &d5, "G", &ManovaOptions::default()).is_err());
}

// ------------------------------------------------------------------ Item analysis

#[test]
fn cronbach_alpha_matches_numpy() {
    let c = cols(CA_X, 60, 5);
    let r = cronbach_alpha(&c, &names(5)).unwrap();
    assert_eq!((r.n, r.n_dropped), (58, 2));
    close(r.alpha, CA_ALPHA, 1e-10, "alpha");
    close(r.standardized_alpha, CA_STD_ALPHA, 1e-10, "standardized alpha");
    for (i, it) in r.items.iter().enumerate() {
        let b = i * 5;
        close(it.adjusted_total_mean, CA_ITEMS[b], 1e-10, "adj mean");
        close(it.adjusted_total_sd, CA_ITEMS[b + 1], 1e-10, "adj sd");
        close(it.item_total_correlation, CA_ITEMS[b + 2], 1e-10, "item-total");
        close(it.squared_multiple_correlation, CA_ITEMS[b + 3], 1e-10, "smc");
        close(it.alpha_if_omitted, CA_ITEMS[b + 4], 1e-10, "alpha if omitted");
    }
    assert!(cronbach_alpha(&c[..1], &names(1)).is_err());
    // Two items: alpha if omitted is undefined.
    let two = cronbach_alpha(&c[..2], &names(2)).unwrap();
    assert!(two.items[0].alpha_if_omitted.is_nan());
}

// ------------------------------------------------------------------ Reports

#[test]
fn every_result_reports_validates_and_round_trips() {
    let mut r = SessionReport::new("multivariate");
    r.push(pca(&iris(), &names(4), &PcaOptions::default()).unwrap().report_section());
    let mut fo = FactorOptions::new(2);
    fo.method = FactorMethod::MaximumLikelihood;
    fo.scores = true;
    r.push(factor_analysis(&fa_cols(), &names(6), &fo).unwrap().report_section());
    r.push(factor_analysis(&fa_cols(), &names(6), &FactorOptions::new(2)).unwrap().report_section());
    r.push(da(false, Priors::Equal).report_section());
    r.push(da(true, Priors::Proportional).report_section());
    r.push(hierarchical(&hc_cols(), LinkageMethod::Ward, Distance::Euclidean, false).unwrap().report_section());
    r.push(cluster_variables(&fa_cols(), &names(6), LinkageMethod::Complete, true).unwrap().report_section());
    r.push(kmeans(&cols(KM_X, 75, 2), &KMeansOptions::new(3)).unwrap().report_section());
    let sp = species();
    let data = Data::new().with_factor("Species", &sp).unwrap();
    r.push(manova(&iris(), &names(4), &data, "Species", &ManovaOptions::default()).unwrap().report_section());
    r.push(cronbach_alpha(&cols(CA_X, 60, 5), &names(5)).unwrap().report_section());
    r.validate().unwrap();
    let text = r.render_text();
    for needle in [
        "Principal Component Analysis",
        "Eigenanalysis of the Correlation Matrix",
        "Likelihood Ratio Test",
        "Rotated Factor Loadings",
        "Linear Discriminant Function for Groups",
        "Summary of Classification with Cross-Validation",
        "Amalgamation Steps",
        "Final Partition",
        "Wilks",
        "Cronbach's Alpha",
    ] {
        assert!(text.contains(needle), "missing {needle}");
    }
    let back = SessionReport::from_json(&r.to_json()).unwrap();
    assert_eq!(back.render_text(), text);
    assert!(r.charts().len() >= 10);
}
