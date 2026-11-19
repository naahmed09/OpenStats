//! Random forests (Breiman 2001).
//!
//! Each tree is grown on a bootstrap sample with a random subset of `mtry` predictors examined at
//! every split and no pruning. The observations a tree did not see (out of bag) give honest error
//! estimates and the permutation importance: the drop in a tree's out-of-bag accuracy (or rise in
//! its squared error) when one predictor's values are shuffled, averaged over trees. Class
//! probabilities average the trees' terminal-node class proportions. Regression prediction
//! intervals are quantile-forest intervals (Meinshausen 2006): the response values that share a
//! terminal node with the new observation, weighted 1/(node size) in every tree, form an empirical
//! distribution whose quantiles bound the response; for the training data only the trees that did
//! not see an observation contribute, so the intervals are out-of-bag.

use super::data::{Dataset, Presort, Response};
use super::tree::{grow, Criterion, GrowParams, Target, Tree};
use super::{argmax, class_fit, regression_fit, rng_for, shuffle, ClassFit, Predictions, RegressionFit};
use crate::error::{invalid, Result, StatsError};
use openstats_math::random::Rng;

/// Controls of [`fit_forest`].
#[derive(Debug, Clone, PartialEq)]
pub struct ForestOptions {
    /// Number of trees.
    pub n_trees: usize,
    /// Predictors examined per split; `None` is ⌊√p⌋ for classification and ⌊p/3⌋ for regression.
    pub mtry: Option<usize>,
    /// Smallest child node; `None` is 1 for classification and 5 for regression.
    pub min_leaf: Option<usize>,
    /// Maximum depth of a tree.
    pub max_depth: usize,
    /// Splitting criterion; `None` is variance (numeric) or Gini (classes).
    pub criterion: Option<Criterion>,
    /// Bootstrap sample size as a fraction of the observations.
    pub sample_fraction: f64,
    /// Compute permutation importance.
    pub importance: bool,
    /// Regression only: coverage level (0–1) of out-of-bag quantile-forest prediction intervals.
    pub interval_level: Option<f64>,
    /// Seed.
    pub seed: u64,
}

impl Default for ForestOptions {
    fn default() -> Self {
        ForestOptions {
            n_trees: 500,
            mtry: None,
            min_leaf: None,
            max_depth: 100,
            criterion: None,
            sample_fraction: 1.0,
            importance: true,
            interval_level: None,
            seed: 1,
        }
    }
}

/// Permutation importance of every predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct PermutationImportance {
    /// Mean over trees of the out-of-bag loss in accuracy (classification) or increase in mean
    /// squared error (regression) caused by permuting the predictor.
    pub mean: Vec<f64>,
    /// Standard error of the mean over trees.
    pub se: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct IntervalData {
    train_y: Vec<f64>,
    /// `leaf_rows[tree][node]`: the in-bag training observations of a terminal node.
    leaf_rows: Vec<Vec<Vec<u32>>>,
}

/// A fitted random forest.
#[derive(Debug, Clone, PartialEq)]
pub struct ForestModel {
    /// The trees.
    pub trees: Vec<Tree>,
    /// Predictor names and levels.
    pub schema: Dataset,
    /// Class labels (empty for regression).
    pub levels: Vec<String>,
    /// The options used.
    pub options: ForestOptions,
    /// The criterion used.
    pub criterion: Criterion,
    /// Observations.
    pub n: usize,
    /// Predictors tried per split.
    pub mtry: usize,
    /// Smallest child node.
    pub min_leaf: usize,
    /// Number of trees that had each observation out of bag.
    pub oob_count: Vec<usize>,
    /// Out-of-bag predictions (observations never out of bag have none).
    pub oob: Predictions,
    /// Whether each observation has an out-of-bag prediction.
    pub has_oob: Vec<bool>,
    /// Out-of-bag misclassification rate or mean squared error.
    pub oob_error: f64,
    /// Out-of-bag error after 1, 2, … trees.
    pub oob_curve: Vec<f64>,
    /// Regression: out-of-bag R² (1 − MSE / variance of the response).
    pub oob_r2: Option<f64>,
    /// Classification: out-of-bag confusion matrix and fit (observations with an OOB prediction).
    pub oob_class_fit: Option<ClassFit>,
    /// Regression: out-of-bag fit statistics.
    pub oob_regression_fit: Option<RegressionFit>,
    /// Permutation importance, if requested.
    pub permutation: Option<PermutationImportance>,
    /// Mean decrease in node deviance per predictor (summed over trees, divided by their number).
    pub impurity_importance: Vec<f64>,
    /// Predictions of the whole forest for the training observations (optimistic).
    pub fitted: Predictions,
    /// Regression: out-of-bag prediction interval of every observation (`NaN` when unavailable).
    pub oob_interval: Vec<(f64, f64)>,
    /// Regression: share of observations whose response lies inside its out-of-bag interval.
    pub interval_coverage: Option<f64>,
    intervals: Option<IntervalData>,
}

fn weighted_quantiles(mut pairs: Vec<(f64, f64)>, lo: f64, hi: f64) -> (f64, f64) {
    if pairs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    let pick = |q: f64| -> f64 {
        let target = q * total;
        let mut cum = 0.0;
        for &(y, w) in &pairs {
            cum += w;
            if cum >= target - 1e-12 * total {
                return y;
            }
        }
        pairs[pairs.len() - 1].0
    };
    (pick(lo), pick(hi))
}

/// Fit a random forest.
pub fn fit_forest(ds: &Dataset, y: &Response, o: &ForestOptions) -> Result<ForestModel> {
    ds.validate()?;
    let n = ds.n_rows();
    let p = ds.n_features();
    if y.len() != n {
        return invalid(format!("the response has {} values but the predictors have {n}", y.len()));
    }
    if o.n_trees == 0 {
        return invalid("the forest needs at least one tree");
    }
    if !(o.sample_fraction > 0.0 && o.sample_fraction <= 1.0) {
        return invalid("the sample fraction must be in (0, 1]");
    }
    let criterion = match (o.criterion, y.is_class()) {
        (None, false) => Criterion::Variance,
        (None, true) => Criterion::Gini,
        (Some(Criterion::Variance), false) => Criterion::Variance,
        (Some(c), true) if c.is_classification() => c,
        _ => return invalid("the splitting criterion does not match the type of the response"),
    };
    let (target, k, levels) = match y {
        Response::Numeric(v) => (Target::Reg(v), 0, Vec::new()),
        Response::Class { codes, levels } => {
            let present = (0..levels.len()).filter(|&c| codes.contains(&c)).count();
            if present < 2 {
                return invalid("the response has only one class");
            }
            (Target::Class { codes, k: levels.len() }, levels.len(), levels.clone())
        }
    };
    let mtry = o.mtry.unwrap_or(if k > 0 { (p as f64).sqrt().floor() as usize } else { p / 3 }).clamp(1, p);
    if o.mtry.is_some_and(|m| m == 0 || m > p) {
        return invalid(format!("mtry must be between 1 and the number of predictors ({p})"));
    }
    let min_leaf = o.min_leaf.unwrap_or(if k > 0 { 1 } else { 5 }).max(1);
    let want_intervals = match (o.interval_level, y) {
        (Some(l), Response::Numeric(_)) => {
            if !(l > 0.0 && l < 1.0) {
                return invalid("the interval level must be between 0 and 1");
            }
            if n.saturating_mul(o.n_trees) > 50_000_000 {
                return Err(StatsError::TooLarge("prediction intervals for this many observations and trees".into()));
            }
            true
        }
        (Some(_), Response::Class { .. }) => return invalid("prediction intervals are for a numeric response"),
        (None, _) => false,
    };

    let presort = Presort::new(ds);
    let mut params = GrowParams::new(criterion);
    params.max_depth = o.max_depth;
    params.min_leaf = min_leaf;
    params.min_split = 2 * min_leaf;
    params.mtry = Some(mtry);
    params.keep_leaf_rows = want_intervals;
    let sample = ((o.sample_fraction * n as f64).round() as usize).max(1);

    let width = if k > 0 { k } else { 1 };
    let mut oob_sum = vec![vec![0.0; width]; n];
    let mut oob_count = vec![0usize; n];
    let mut trees = Vec::with_capacity(o.n_trees);
    let mut curve = Vec::with_capacity(o.n_trees);
    let mut imp_sum = vec![0.0; p];
    let mut imp_sq = vec![0.0; p];
    let mut imp_cnt = vec![0usize; p];
    let mut impurity = vec![0.0; p];
    let mut leaf_store: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut inbag: Vec<Vec<u16>> = Vec::new();

    let loss_now = |oob_sum: &[Vec<f64>], oob_count: &[usize]| -> f64 {
        let (mut bad, mut m) = (0.0, 0usize);
        for i in 0..n {
            if oob_count[i] == 0 {
                continue;
            }
            m += 1;
            let c = oob_count[i] as f64;
            match y {
                Response::Numeric(v) => {
                    let e = v[i] - oob_sum[i][0] / c;
                    bad += e * e;
                }
                Response::Class { codes, .. } => {
                    if argmax(&oob_sum[i]) != codes[i] {
                        bad += 1.0;
                    }
                }
            }
        }
        if m == 0 {
            f64::NAN
        } else {
            bad / m as f64
        }
    };

    // Grow every tree (and its permutation-importance deltas) in parallel; each tree depends only on its own
    // random stream, so the merge below, done in tree order, reproduces the sequential result exactly.
    struct Grown {
        tree: crate::ml::tree::Tree,
        rows: Vec<Vec<u32>>,
        counts: Vec<u32>,
        deltas: Option<Vec<f64>>,
    }
    let grown: Vec<Grown> = crate::parallel::par_map(o.n_trees, |b| {
        let mut rng = rng_for(o.seed, 1000 + b as u64);
        let mut counts = vec![0u32; n];
        for _ in 0..sample {
            counts[usize::try_from(rng.below(n as u64)).unwrap_or(0)] += 1;
        }
        let (tree, rows) = grow(ds, &presort, target, &counts, &params, &mut rng);
        let oob_rows: Vec<usize> = (0..n).filter(|&i| counts[i] == 0).collect();
        let mut deltas = None;
        if o.importance && !oob_rows.is_empty() {
            let m = oob_rows.len() as f64;
            let loss = |leaf_of: &dyn Fn(usize) -> usize| -> f64 {
                let mut s = 0.0;
                for &i in &oob_rows {
                    let leaf = leaf_of(i);
                    match y {
                        Response::Numeric(v) => {
                            let e = v[i] - tree.nodes[leaf].value;
                            s += e * e;
                        }
                        Response::Class { codes, .. } => {
                            if tree.class_of(leaf) != codes[i] {
                                s += 1.0;
                            }
                        }
                    }
                }
                s / m
            };
            let base = loss(&|i| tree.leaf(ds, i));
            let mut d = Vec::with_capacity(p);
            for f in 0..p {
                let mut src = oob_rows.clone();
                shuffle(&mut src, &mut rng);
                let pos: std::collections::HashMap<usize, usize> =
                    oob_rows.iter().enumerate().map(|(a, &r)| (r, src[a])).collect();
                d.push(loss(&|i| tree.leaf_permuted(ds, i, f, pos[&i])) - base);
            }
            deltas = Some(d);
        }
        Grown { tree, rows, counts, deltas }
    });
    for g in grown {
        let Grown { tree, rows, counts, deltas } = g;
        let oob_rows: Vec<usize> = (0..n).filter(|&i| counts[i] == 0).collect();
        for &i in &oob_rows {
            let leaf = tree.leaf(ds, i);
            oob_count[i] += 1;
            if k > 0 {
                for (s, q) in oob_sum[i].iter_mut().zip(tree.proportions(leaf)) {
                    *s += q;
                }
            } else {
                oob_sum[i][0] += tree.nodes[leaf].value;
            }
        }
        if let Some(d) = deltas {
            for (f, x) in d.into_iter().enumerate() {
                imp_sum[f] += x;
                imp_sq[f] += x * x;
                imp_cnt[f] += 1;
            }
        }
        tree.add_primary_importance(&mut impurity);
        if want_intervals {
            leaf_store.push(rows);
            inbag.push(counts.iter().map(|&c| u16::try_from(c).unwrap_or(u16::MAX)).collect());
        }
        trees.push(tree);
        curve.push(loss_now(&oob_sum, &oob_count));
    }
    for v in &mut impurity {
        *v /= o.n_trees as f64;
    }

    // Out-of-bag predictions and fit.
    let has_oob: Vec<bool> = oob_count.iter().map(|&c| c > 0).collect();
    let mut oob = Predictions::default();
    for i in 0..n {
        let c = oob_count[i].max(1) as f64;
        match y {
            Response::Numeric(_) => oob.values.push(if has_oob[i] { oob_sum[i][0] / c } else { f64::NAN }),
            Response::Class { .. } => {
                let probs: Vec<f64> = oob_sum[i].iter().map(|s| s / c).collect();
                oob.classes.push(if has_oob[i] { argmax(&probs) } else { 0 });
                oob.probabilities.push(if has_oob[i] { probs } else { vec![f64::NAN; k] });
            }
        }
    }
    let used: Vec<usize> = (0..n).filter(|&i| has_oob[i]).collect();
    let (mut oob_r2, mut oob_class_fit, mut oob_regression_fit) = (None, None, None);
    let oob_error = curve.last().copied().unwrap_or(f64::NAN);
    match y {
        Response::Numeric(v) => {
            let yy: Vec<f64> = used.iter().map(|&i| v[i]).collect();
            let pp: Vec<f64> = used.iter().map(|&i| oob.values[i]).collect();
            let f = regression_fit(&yy, &pp);
            oob_r2 = Some(f.r2);
            oob_regression_fit = Some(f);
        }
        Response::Class { codes, .. } => {
            let cc: Vec<usize> = used.iter().map(|&i| codes[i]).collect();
            let pp: Vec<Vec<f64>> = used.iter().map(|&i| oob.probabilities[i].clone()).collect();
            oob_class_fit = Some(class_fit(&cc, &pp, &levels));
        }
    }

    let permutation = o.importance.then(|| {
        let mut mean = vec![0.0; p];
        let mut se = vec![0.0; p];
        for f in 0..p {
            let c = imp_cnt[f] as f64;
            if c > 0.0 {
                mean[f] = imp_sum[f] / c;
                let var = (imp_sq[f] / c - mean[f] * mean[f]).max(0.0) * c / (c - 1.0).max(1.0);
                se[f] = (var / c).sqrt();
            }
        }
        // For classification the loss is the error rate, so a positive value is a fall in accuracy.
        PermutationImportance { mean, se }
    });

    let mut model = ForestModel {
        trees,
        schema: ds.schema(),
        levels,
        options: o.clone(),
        criterion,
        n,
        mtry,
        min_leaf,
        oob_count,
        oob,
        has_oob,
        oob_error,
        oob_curve: curve,
        oob_r2,
        oob_class_fit,
        oob_regression_fit,
        permutation,
        impurity_importance: impurity,
        fitted: Predictions::default(),
        oob_interval: Vec::new(),
        interval_coverage: None,
        intervals: None,
    };
    model.fitted = model.predict_dataset(ds);
    if want_intervals {
        let Response::Numeric(yv) = y else { return invalid("prediction intervals are for a numeric response") };
        let level = o.interval_level.unwrap_or(0.95);
        let (qlo, qhi) = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0);
        let mut ivs = Vec::with_capacity(n);
        for i in 0..n {
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for (t, tree) in model.trees.iter().enumerate() {
                if inbag[t][i] != 0 {
                    continue;
                }
                let leaf = tree.leaf(ds, i);
                let rows = &leaf_store[t][leaf];
                if rows.is_empty() {
                    continue;
                }
                let w = 1.0 / rows.len() as f64;
                pairs.extend(rows.iter().map(|&r| (yv[r as usize], w)));
            }
            ivs.push(weighted_quantiles(pairs, qlo, qhi));
        }
        let covered = (0..n).filter(|&i| ivs[i].0.is_finite() && yv[i] >= ivs[i].0 && yv[i] <= ivs[i].1).count();
        let usable = ivs.iter().filter(|iv| iv.0.is_finite()).count();
        model.interval_coverage = (usable > 0).then(|| covered as f64 / usable as f64);
        model.oob_interval = ivs;
        model.intervals = Some(IntervalData { train_y: yv.clone(), leaf_rows: leaf_store });
    }
    Ok(model)
}

impl ForestModel {
    fn predict_dataset(&self, d: &Dataset) -> Predictions {
        let n = d.n_rows();
        let t = self.trees.len() as f64;
        let mut out = Predictions::default();
        for i in 0..n {
            if self.levels.is_empty() {
                let s: f64 = self.trees.iter().map(|tr| tr.predict_value(d, i)).sum();
                out.values.push(s / t);
            } else {
                let mut probs = vec![0.0; self.levels.len()];
                for tr in &self.trees {
                    for (a, q) in probs.iter_mut().zip(tr.proportions(tr.leaf(d, i))) {
                        *a += q;
                    }
                }
                for a in &mut probs {
                    *a /= t;
                }
                out.classes.push(argmax(&probs));
                out.probabilities.push(probs);
            }
        }
        out
    }

    /// Predict observations described by `ds` (same predictors, in the same order, as the fit).
    pub fn predict(&self, ds: &Dataset) -> Result<Predictions> {
        Ok(self.predict_dataset(&ds.remap_to(&self.schema)?))
    }

    /// Quantile-forest prediction intervals with coverage `level` for new observations
    /// (regression forests fitted with `interval_level` set).
    pub fn predict_interval(&self, ds: &Dataset, level: f64) -> Result<Vec<(f64, f64)>> {
        let data = self
            .intervals
            .as_ref()
            .ok_or_else(|| StatsError::InvalidArgument("the forest was fitted without prediction intervals".into()))?;
        if !(level > 0.0 && level < 1.0) {
            return invalid("the interval level must be between 0 and 1");
        }
        let d = ds.remap_to(&self.schema)?;
        let (qlo, qhi) = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0);
        let mut out = Vec::with_capacity(d.n_rows());
        for i in 0..d.n_rows() {
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for (t, tree) in self.trees.iter().enumerate() {
                let rows = &data.leaf_rows[t][tree.leaf(&d, i)];
                if rows.is_empty() {
                    continue;
                }
                let w = 1.0 / rows.len() as f64;
                pairs.extend(rows.iter().map(|&r| (data.train_y[r as usize], w)));
            }
            out.push(weighted_quantiles(pairs, qlo, qhi));
        }
        Ok(out)
    }

    /// Whether the forest classifies.
    pub fn is_classification(&self) -> bool {
        !self.levels.is_empty()
    }
}
