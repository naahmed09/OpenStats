//! Tree-based machine learning (Phase 4a): classification and regression trees with
//! cost-complexity pruning, random forests and gradient boosted trees.
//!
//! * [`cart`] — CART (Breiman, Friedman, Olshen & Stone 1984): recursive binary partitioning with
//!   variance (regression), Gini or entropy (classification) criteria, numeric and categorical
//!   predictors, surrogate splits for missing predictor values, weakest-link cost-complexity
//!   pruning R_α(T) = R(T) + α|T̃|, K-fold cross-validated tree size (minimum or one-standard-
//!   error rule) and variable importance.
//! * [`random_forest`] — Breiman (2001): bootstrap aggregation with a random subset of `mtry`
//!   predictors per split, out-of-bag error, permutation and impurity importance, class
//!   probability averaging and (regression) quantile-forest prediction intervals.
//! * [`gbdt`] — Friedman (2001): gradient boosting of small regression trees with shrinkage and
//!   stochastic subsampling for squared-error and Huber regression and for binomial and
//!   multinomial deviance classification, with validation-based early stopping.
//! * [`tree`] — the tree itself, the split search and the surrogate machinery they share.
//!
//! Predictors are a [`Dataset`] (numeric columns with `NaN` for missing, categorical columns with
//! labels); the response is a [`Response`]. All randomness comes from a seeded
//! [`openstats_math::random::Pcg64`], so a seed reproduces a fit exactly.

pub mod cart;
pub mod compare;
pub mod data;
pub mod gbdt;
pub mod mars;
pub mod mars_ext;
pub mod pdp;
pub mod random_forest;
mod report_impl;
pub mod store;
pub mod tree;

pub use cart::{fit_cart, CartModel, CartOptions, CpRow, Pruning};
pub use compare::{compare_models, fit_spec, Comparison, ComparisonRow, ModelSpec};
pub use data::{Dataset, Feature, Response};
pub use gbdt::{fit_gbdt, GbdtLoss, GbdtModel, GbdtOptions};
pub use mars::{fit_mars, Factor, MarsModel, MarsOptions, PruneRow, Term};
pub use mars_ext::{
    fit_mars_binary, logistic_on_basis, mars_cross_validate, random_folds, CvResponse, LogitBasisFit, LogitCoefficient,
    MarsBinaryModel, MarsCv, MarsCvRow,
};
pub use pdp::{partial_dependence, partial_dependence_2d, PartialDependence, PartialDependence2D, PdOptions};
pub use random_forest::{fit_forest, ForestModel, ForestOptions};
pub use store::{expand_indicators, MarsLink, MarsObject, PredictiveModel, SavedModel};
pub use tree::{Criterion, Tree};

/// Predictions of a fitted model for a set of observations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Predictions {
    /// Regression: the predicted values (empty for classification).
    pub values: Vec<f64>,
    /// Classification: the predicted class index of every observation (empty for regression).
    pub classes: Vec<usize>,
    /// Classification: one probability per class and observation (empty for regression).
    pub probabilities: Vec<Vec<f64>>,
}

/// Goodness of fit of a regression model.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionFit {
    /// Observations used.
    pub n: usize,
    /// 1 − SSE/SST (`NaN` when the response is constant).
    pub r2: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
}

/// Goodness of fit of a classification model.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassFit {
    /// Observations used.
    pub n: usize,
    /// Class labels.
    pub levels: Vec<String>,
    /// `confusion[actual][predicted]`.
    pub confusion: Vec<Vec<usize>>,
    /// Share of observations classified correctly.
    pub accuracy: f64,
    /// Area under the ROC curve (two classes; the second level is the event).
    pub auc: Option<f64>,
    /// Mean negative log probability of the true class.
    pub log_loss: Option<f64>,
}

/// R², RMSE and MAE of `pred` against `y`.
pub fn regression_fit(y: &[f64], pred: &[f64]) -> RegressionFit {
    let n = y.len().min(pred.len());
    if n == 0 {
        return RegressionFit { n: 0, r2: f64::NAN, rmse: f64::NAN, mae: f64::NAN };
    }
    let mean = y[..n].iter().sum::<f64>() / n as f64;
    let (mut sse, mut sst, mut sae) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let e = y[i] - pred[i];
        sse += e * e;
        sae += e.abs();
        sst += (y[i] - mean) * (y[i] - mean);
    }
    RegressionFit {
        n,
        r2: if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN },
        rmse: (sse / n as f64).sqrt(),
        mae: sae / n as f64,
    }
}

pub(crate) fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// Area under the ROC curve for scores of the event class, with ties counted half.
pub fn auc(scores: &[f64], event: &[bool]) -> Option<f64> {
    let pos = event.iter().filter(|&&e| e).count();
    let neg = event.len() - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &t in &idx[i..=j] {
            if event[t] {
                rank_sum += avg_rank;
            }
        }
        i = j + 1;
    }
    Some((rank_sum - pos as f64 * (pos as f64 + 1.0) / 2.0) / (pos as f64 * neg as f64))
}

/// Confusion matrix, accuracy, AUC (two classes) and log-loss of class probabilities.
pub fn class_fit(codes: &[usize], probs: &[Vec<f64>], levels: &[String]) -> ClassFit {
    let k = levels.len();
    let n = codes.len().min(probs.len());
    let mut confusion = vec![vec![0usize; k]; k];
    let mut correct = 0usize;
    let mut ll = 0.0;
    for i in 0..n {
        let pred = argmax(&probs[i]);
        confusion[codes[i]][pred] += 1;
        if pred == codes[i] {
            correct += 1;
        }
        ll -= probs[i][codes[i]].max(1e-15).ln();
    }
    let auc = if k == 2 {
        let scores: Vec<f64> = probs[..n].iter().map(|p| p[1]).collect();
        let event: Vec<bool> = codes[..n].iter().map(|&c| c == 1).collect();
        auc(&scores, &event)
    } else {
        None
    };
    ClassFit {
        n,
        levels: levels.to_vec(),
        confusion,
        accuracy: if n > 0 { correct as f64 / n as f64 } else { f64::NAN },
        auc,
        log_loss: (n > 0).then(|| ll / n as f64),
    }
}

/// `100 × v / max(v)` (zeros when nothing is positive): importance scaled so the best is 100.
pub fn relative_importance(v: &[f64]) -> Vec<f64> {
    let max = v.iter().copied().fold(0.0_f64, f64::max);
    v.iter().map(|&x| if max > 0.0 { 100.0 * x.max(0.0) / max } else { 0.0 }).collect()
}

/// Seeded generator used by every model.
pub(crate) fn rng_for(seed: u64, stream: u64) -> openstats_math::random::Pcg64 {
    openstats_math::random::Pcg64::with_stream(seed, stream)
}

/// Shuffle a slice with `rng` (Fisher–Yates).
pub(crate) fn shuffle<T>(v: &mut [T], rng: &mut dyn openstats_math::random::Rng) {
    for i in (1..v.len()).rev() {
        let j = usize::try_from(rng.below(i as u64 + 1)).unwrap_or(0);
        v.swap(i, j);
    }
}
