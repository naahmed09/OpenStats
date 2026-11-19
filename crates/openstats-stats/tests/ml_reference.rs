//! Tree models against scikit-learn 1.9 (values in `common/ml_data.rs`, produced by
//! `gen/ml_ref.py`). With continuous predictors and no ties the greedy split search is unique, so
//! OpenStats's trees must be the same trees, node for node.

mod common;
use common::ml_data::*;
use openstats_stats::ml::{
    fit_cart, fit_gbdt, CartOptions, Criterion, Dataset, GbdtLoss, GbdtOptions, Pruning, Response,
};

fn data() -> Dataset {
    Dataset::new()
        .with_numeric("x1", X1.to_vec())
        .unwrap()
        .with_numeric("x2", X2.to_vec())
        .unwrap()
        .with_numeric("x3", X3.to_vec())
        .unwrap()
        .with_numeric("x4", X4.to_vec())
        .unwrap()
}

fn classes(c: &[usize]) -> Response {
    let labels: Vec<String> = c.iter().map(|v| v.to_string()).collect();
    Response::from_labels(&labels).unwrap()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: got {a}, want {b}");
}

fn cart_opts(max_depth: usize, min_leaf: usize, criterion: Option<Criterion>) -> CartOptions {
    CartOptions {
        criterion,
        max_depth,
        min_split: 2 * min_leaf,
        min_leaf,
        cp: 0.0,
        max_surrogates: 0,
        pruning: Pruning::None,
        cv_folds: 0,
        seed: 1,
    }
}

fn normalised(v: &[f64]) -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|x| x / s).collect()
}

#[test]
fn regression_tree_matches_sklearn() {
    let m = fit_cart(&data(), &Response::numeric(Y.to_vec()).unwrap(), &cart_opts(4, 3, None)).unwrap();
    assert_eq!(m.tree.n_leaves(), REG_TREE_LEAVES);
    for (i, want) in REG_TREE_PRED.iter().enumerate() {
        close(m.fitted.values[i], *want, 1e-10, &format!("prediction {i}"));
    }
    let imp = normalised(&m.importance);
    for (j, want) in REG_TREE_IMP.iter().enumerate() {
        close(imp[j], *want, 1e-9, &format!("importance {j}"));
    }
}

#[test]
fn full_regression_tree_and_pruning_path_match_sklearn() {
    let o = cart_opts(100, 3, None);
    let m = fit_cart(&data(), &Response::numeric(Y.to_vec()).unwrap(), &o).unwrap();
    assert_eq!(m.full_leaves, REG_FULL_LEAVES);
    for (i, want) in REG_FULL_PRED.iter().enumerate() {
        close(m.fitted.values[i], *want, 1e-10, &format!("prediction {i}"));
    }
    // sklearn's alpha is on the impurity scale; ours is relative to the root's error.
    assert_eq!(m.sequence.len(), REG_CCP_ALPHAS.len(), "number of subtrees");
    for (k, row) in m.sequence.iter().enumerate() {
        close(row.alpha * REG_ROOT_IMPURITY, REG_CCP_ALPHAS[k], 1e-9, &format!("alpha {k}"));
        assert_eq!(row.leaves, REG_CCP_LEAVES[k], "leaves of subtree {k}");
    }
}

#[test]
fn classification_trees_match_sklearn() {
    for (crit, proba, leaves, imp) in [
        (Criterion::Gini, CLS3_GINI_PROBA, CLS3_GINI_LEAVES, CLS3_GINI_IMP),
        (Criterion::Entropy, CLS3_ENTROPY_PROBA, CLS3_ENTROPY_LEAVES, CLS3_ENTROPY_IMP),
    ] {
        let m = fit_cart(&data(), &classes(CLS3), &cart_opts(3, 5, Some(crit))).unwrap();
        assert_eq!(m.tree.n_leaves(), leaves, "{crit:?} leaves");
        for (i, p) in m.fitted.probabilities.iter().enumerate() {
            for (c, q) in p.iter().enumerate() {
                close(*q, proba[i * 3 + c], 1e-10, &format!("{crit:?} proba {i},{c}"));
            }
        }
        let got = normalised(&m.importance);
        for (j, want) in imp.iter().enumerate() {
            close(got[j], *want, 1e-9, &format!("{crit:?} importance {j}"));
        }
    }
}

fn boost(loss: Option<GbdtLoss>, trees: usize, lr: f64) -> GbdtOptions {
    GbdtOptions {
        n_trees: trees,
        learning_rate: lr,
        max_depth: 2,
        min_leaf: 1,
        subsample: 1.0,
        loss,
        huber_alpha: 0.9,
        validation_fraction: 0.0,
        early_stopping_rounds: 0,
        seed: 1,
    }
}

#[test]
fn gradient_boosting_squared_error_matches_sklearn() {
    let m = fit_gbdt(&data(), &Response::numeric(Y.to_vec()).unwrap(), &boost(None, 30, 0.1)).unwrap();
    for (i, want) in GB_REG_PRED.iter().enumerate() {
        close(m.fitted.values[i], *want, 1e-9, &format!("prediction {i}"));
    }
}

#[test]
fn gradient_boosting_huber_matches_sklearn() {
    let y = Response::numeric(Y.to_vec()).unwrap();
    // One stage is identical: same initial median, delta (inverted-CDF 90th percentile of |r|),
    // clipped pseudo-residuals, trees and leaf values.
    let one = fit_gbdt(&data(), &y, &boost(Some(GbdtLoss::Huber), 1, 0.1)).unwrap();
    for (i, want) in GB_HUBER1_PRED.iter().enumerate() {
        close(one.fitted.values[i], *want, 1e-10, &format!("one-stage prediction {i}"));
    }
    // Over 30 stages the fits stay close but are not bit-identical: scikit-learn takes the lower of
    // the two middle residuals as the "median" of a terminal node with an even number of
    // observations, where OpenStats (like Friedman 2001) averages them. (With the lower middle
    // substituted, all 30 stages agree with scikit-learn to rounding error; that was checked once.)
    let m = fit_gbdt(&data(), &y, &boost(Some(GbdtLoss::Huber), 30, 0.1)).unwrap();
    let worst = GB_HUBER_PRED.iter().zip(&m.fitted.values).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 0.3, "largest difference from scikit-learn {worst}");
}

#[test]
fn gradient_boosting_binomial_deviance_matches_sklearn() {
    let m = fit_gbdt(&data(), &classes(CLS2), &boost(None, 30, 0.2)).unwrap();
    for (i, p) in m.fitted.probabilities.iter().enumerate() {
        for (c, q) in p.iter().enumerate() {
            close(*q, GB_BIN_PROBA[i * 2 + c], 1e-9, &format!("proba {i},{c}"));
        }
    }
}

#[test]
fn gradient_boosting_multinomial_deviance_matches_sklearn() {
    let m = fit_gbdt(&data(), &classes(CLS3), &boost(None, 25, 0.2)).unwrap();
    for (i, p) in m.fitted.probabilities.iter().enumerate() {
        for (c, q) in p.iter().enumerate() {
            close(*q, GB_MULTI_PROBA[i * 3 + c], 1e-9, &format!("proba {i},{c}"));
        }
    }
}
