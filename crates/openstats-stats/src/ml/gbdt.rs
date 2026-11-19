//! Gradient boosted decision trees (Friedman 2001, 2002).
//!
//! The model is F(x) = F₀ + ν Σₘ hₘ(x), built stage-wise: each stage fits a small regression tree
//! hₘ to the negative gradient of the loss at the current F (the "pseudo-residuals"), then
//! replaces every terminal-node value by the one-dimensional optimum for the loss:
//!
//! | loss | F₀ | pseudo-residual | terminal-node value |
//! |---|---|---|---|
//! | squared error | mean y | y − F | mean of the residuals |
//! | Huber(δ) | median y | y − F clipped to ±δ | median r̃ + mean of clip(r − r̃, ±δ) |
//! | binomial deviance | log-odds | y − p | Σ(y − p) / Σ p(1 − p) (one Newton step) |
//! | multinomial deviance | log priors | y_k − p_k | (K−1)/K · Σ r / Σ \|r\|(1 − \|r\|) |
//!
//! δ is the `huber_alpha` quantile of |y − F| over the training rows at each stage. `subsample`
//! < 1 fits each tree on a random subset of the training rows (stochastic gradient boosting);
//! `validation_fraction` holds rows out to track the validation loss, choose the number of trees
//! and (with `early_stopping_rounds`) stop early.

use super::data::{Dataset, Presort, Response};
use super::tree::{grow, Criterion, GrowParams, Target, Tree};
use super::{argmax, class_fit, regression_fit, rng_for, shuffle, ClassFit, Predictions, RegressionFit};
use crate::error::{invalid, Result};

/// Loss minimised by boosting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbdtLoss {
    /// Squared error (regression).
    SquaredError,
    /// Huber loss, robust to outliers (regression).
    Huber,
    /// Binomial (two classes) or multinomial (more) deviance, i.e. log-loss (classification).
    Deviance,
}

impl GbdtLoss {
    /// Name used in reports.
    pub fn label(self, classes: usize) -> &'static str {
        match (self, classes) {
            (GbdtLoss::SquaredError, _) => "Squared error",
            (GbdtLoss::Huber, _) => "Huber",
            (GbdtLoss::Deviance, 2) => "Binomial deviance (log-loss)",
            (GbdtLoss::Deviance, _) => "Multinomial deviance (log-loss)",
        }
    }
}

/// Controls of [`fit_gbdt`].
#[derive(Debug, Clone, PartialEq)]
pub struct GbdtOptions {
    /// Maximum number of boosting stages.
    pub n_trees: usize,
    /// Shrinkage ν in (0, 1].
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// Smallest child node.
    pub min_leaf: usize,
    /// Fraction of the training rows used for each tree, in (0, 1].
    pub subsample: f64,
    /// Loss; `None` is squared error for a numeric and deviance for a class response.
    pub loss: Option<GbdtLoss>,
    /// Huber: the quantile of |residual| used as δ.
    pub huber_alpha: f64,
    /// Share of the observations held out for validation, in [0, 0.5].
    pub validation_fraction: f64,
    /// Stop after this many stages without a better validation loss (0 = never).
    pub early_stopping_rounds: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for GbdtOptions {
    fn default() -> Self {
        GbdtOptions {
            n_trees: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_leaf: 5,
            subsample: 0.5,
            loss: None,
            huber_alpha: 0.9,
            validation_fraction: 0.0,
            early_stopping_rounds: 0,
            seed: 1,
        }
    }
}

/// A fitted gradient boosting model.
#[derive(Debug, Clone, PartialEq)]
pub struct GbdtModel {
    /// Initial value(s) F₀ (one per class for multinomial, else one).
    pub init: Vec<f64>,
    /// The trees kept: `trees[stage][dimension]`.
    pub trees: Vec<Vec<Tree>>,
    /// Shrinkage used.
    pub learning_rate: f64,
    /// Predictor names and levels.
    pub schema: Dataset,
    /// Class labels (empty for regression).
    pub levels: Vec<String>,
    /// The options used.
    pub options: GbdtOptions,
    /// The loss used.
    pub loss: GbdtLoss,
    /// Observations.
    pub n: usize,
    /// Rows the trees were fitted on.
    pub n_train: usize,
    /// Rows held out for validation.
    pub n_valid: usize,
    /// Whether each observation was held out.
    pub is_validation: Vec<bool>,
    /// Training loss after each stage fitted.
    pub train_loss: Vec<f64>,
    /// Validation loss after each stage fitted (empty without validation).
    pub valid_loss: Vec<f64>,
    /// Stages kept (the one with the lowest validation loss, or all of them).
    pub best_iteration: usize,
    /// Relative influence of each predictor (sum of split improvements over all trees, raw).
    pub importance: Vec<f64>,
    /// Predictions of the final model for every observation.
    pub fitted: Predictions,
    /// Regression fit on the training rows.
    pub train_regression: Option<RegressionFit>,
    /// Regression fit on the validation rows.
    pub valid_regression: Option<RegressionFit>,
    /// Classification fit on the training rows.
    pub train_class: Option<ClassFit>,
    /// Classification fit on the validation rows.
    pub valid_class: Option<ClassFit>,
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    // The inverted empirical CDF: the smallest value with at least a share q of the data at or below it.
    let idx = ((v.len() as f64 * q).ceil() as usize).saturating_sub(1);
    v[idx.min(v.len() - 1)]
}

fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
}

fn softmax(f: &[f64]) -> Vec<f64> {
    let m = f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = f.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn huber(r: f64, delta: f64) -> f64 {
    if r.abs() <= delta {
        0.5 * r * r
    } else {
        delta * (r.abs() - 0.5 * delta)
    }
}

/// Fit a gradient boosting model.
pub fn fit_gbdt(ds: &Dataset, y: &Response, o: &GbdtOptions) -> Result<GbdtModel> {
    ds.validate()?;
    let n = ds.n_rows();
    if y.len() != n {
        return invalid(format!("the response has {} values but the predictors have {n}", y.len()));
    }
    if o.n_trees == 0 {
        return invalid("boosting needs at least one tree");
    }
    if !(o.learning_rate > 0.0 && o.learning_rate <= 1.0) {
        return invalid("the learning rate must be in (0, 1]");
    }
    if !(o.subsample > 0.0 && o.subsample <= 1.0) {
        return invalid("the subsample fraction must be in (0, 1]");
    }
    if !(0.0..=0.5).contains(&o.validation_fraction) {
        return invalid("the validation fraction must be between 0 and 0.5");
    }
    if o.max_depth == 0 || o.min_leaf == 0 {
        return invalid("tree depth and minimum node size must be at least 1");
    }
    if !(o.huber_alpha > 0.0 && o.huber_alpha < 1.0) {
        return invalid("the Huber quantile must be between 0 and 1");
    }
    let loss = match (o.loss, y.is_class()) {
        (None, false) => GbdtLoss::SquaredError,
        (None, true) => GbdtLoss::Deviance,
        (Some(l @ (GbdtLoss::SquaredError | GbdtLoss::Huber)), false) => l,
        (Some(GbdtLoss::Deviance), true) => GbdtLoss::Deviance,
        _ => return invalid("the loss does not match the type of the response"),
    };
    let (k, levels) = match y {
        Response::Class { codes, levels } => {
            let present = (0..levels.len()).filter(|&c| codes.contains(&c)).count();
            if present < 2 {
                return invalid("the response has only one class");
            }
            (levels.len(), levels.clone())
        }
        Response::Numeric(_) => (0, Vec::new()),
    };
    let dims = if k > 2 { k } else { 1 };

    // Training / validation split.
    let mut rng = rng_for(o.seed, 1);
    let mut perm: Vec<usize> = (0..n).collect();
    shuffle(&mut perm, &mut rng);
    let n_valid = (o.validation_fraction * n as f64).floor() as usize;
    let valid: Vec<usize> = perm[..n_valid].to_vec();
    let mut train: Vec<usize> = perm[n_valid..].to_vec();
    train.sort_unstable();
    let mut valid_sorted = valid;
    valid_sorted.sort_unstable();
    let n_train = train.len();
    if n_train < 2 * o.min_leaf {
        return invalid(format!("{n_train} training observations are too few for nodes of at least {}", o.min_leaf));
    }
    let mut is_validation = vec![false; n];
    for &i in &valid_sorted {
        is_validation[i] = true;
    }

    // Initial values.
    let mut init = vec![0.0; dims];
    match y {
        Response::Numeric(v) => {
            let mut t: Vec<f64> = train.iter().map(|&i| v[i]).collect();
            init[0] = match loss {
                GbdtLoss::Huber => median(&mut t),
                _ => t.iter().sum::<f64>() / n_train as f64,
            };
        }
        Response::Class { codes, .. } => {
            let mut c = vec![0.0_f64; k];
            for &i in &train {
                c[codes[i]] += 1.0;
            }
            // Class frequencies; a class absent from the training rows gets half an observation.
            let nt = n_train as f64;
            if k == 2 {
                let p = c[1].clamp(0.5, nt - 0.5) / nt;
                init[0] = (p / (1.0 - p)).ln();
            } else {
                let total: f64 = c.iter().map(|x| x.max(0.5)).sum();
                for (d, cnt) in c.iter().enumerate() {
                    init[d] = (cnt.max(0.5) / total).ln();
                }
            }
        }
    }
    let mut f: Vec<Vec<f64>> = (0..n).map(|_| init.clone()).collect();

    let presort = Presort::new(ds);
    let mut params = GrowParams::new(Criterion::Variance);
    params.max_depth = o.max_depth;
    params.min_leaf = o.min_leaf;
    params.min_split = 2 * o.min_leaf;
    params.keep_leaf_rows = true;
    let sub_n = ((o.subsample * n_train as f64).round() as usize).clamp(2 * o.min_leaf, n_train);

    let mut trees: Vec<Vec<Tree>> = Vec::new();
    let mut train_loss = Vec::new();
    let mut valid_loss = Vec::new();
    let mut importance = vec![0.0; ds.n_features()];
    let mut stage_importance: Vec<Vec<f64>> = Vec::new();
    let mut best = (f64::INFINITY, 0usize);
    let mut resid = vec![vec![0.0; n]; dims];
    let mut delta = 0.0;

    for m in 0..o.n_trees {
        // Pseudo-residuals at the current F (and, for Huber, the current δ).
        match y {
            Response::Numeric(v) => {
                if loss == GbdtLoss::Huber {
                    let mut a: Vec<f64> = train.iter().map(|&i| (v[i] - f[i][0]).abs()).collect();
                    delta = quantile(&mut a, o.huber_alpha).max(1e-12);
                }
                for &i in &train {
                    let r = v[i] - f[i][0];
                    resid[0][i] = if loss == GbdtLoss::Huber { r.clamp(-delta, delta) } else { r };
                }
            }
            Response::Class { codes, .. } => {
                for &i in &train {
                    if k == 2 {
                        resid[0][i] = f64::from(u8::from(codes[i] == 1)) - sigmoid(f[i][0]);
                    } else {
                        let p = softmax(&f[i]);
                        for d in 0..k {
                            resid[d][i] = f64::from(u8::from(codes[i] == d)) - p[d];
                        }
                    }
                }
            }
        }
        // This stage's subsample.
        let mut pool = train.clone();
        shuffle(&mut pool, &mut rng);
        let mut counts = vec![0u32; n];
        for &i in pool.iter().take(sub_n) {
            counts[i] = 1;
        }
        let mut stage = Vec::with_capacity(dims);
        let mut gains = vec![0.0; ds.n_features()];
        for d in 0..dims {
            let (mut tree, leaf_rows) = grow(ds, &presort, Target::Reg(&resid[d]), &counts, &params, &mut rng);
            for id in 0..tree.nodes.len() {
                if !tree.nodes[id].is_leaf() || leaf_rows[id].is_empty() {
                    continue;
                }
                let rows = &leaf_rows[id];
                let value = match y {
                    Response::Numeric(v) => match loss {
                        GbdtLoss::Huber => {
                            let mut res: Vec<f64> = rows.iter().map(|&r| v[r as usize] - f[r as usize][0]).collect();
                            let med = median(&mut res.clone());
                            let adj: f64 = res.iter_mut().map(|r| (*r - med).clamp(-delta, delta)).sum::<f64>()
                                / rows.len() as f64;
                            med + adj
                        }
                        _ => rows.iter().map(|&r| resid[0][r as usize]).sum::<f64>() / rows.len() as f64,
                    },
                    Response::Class { .. } => {
                        let num: f64 = rows.iter().map(|&r| resid[d][r as usize]).sum();
                        if k == 2 {
                            let den: f64 = rows
                                .iter()
                                .map(|&r| {
                                    let p = sigmoid(f[r as usize][0]);
                                    p * (1.0 - p)
                                })
                                .sum();
                            if den < 1e-12 {
                                0.0
                            } else {
                                (num / den).clamp(-20.0, 20.0)
                            }
                        } else {
                            let den: f64 = rows
                                .iter()
                                .map(|&r| {
                                    let a = resid[d][r as usize].abs();
                                    a * (1.0 - a)
                                })
                                .sum();
                            if den < 1e-12 {
                                0.0
                            } else {
                                ((k as f64 - 1.0) / k as f64 * num / den).clamp(-20.0, 20.0)
                            }
                        }
                    }
                };
                tree.nodes[id].value = value;
            }
            tree.add_primary_importance(&mut gains);
            for i in 0..n {
                f[i][d] += o.learning_rate * tree.predict_value(ds, i);
            }
            stage.push(tree);
        }
        trees.push(stage);
        stage_importance.push(gains);

        // Loss after this stage.
        let eval = |rows: &[usize]| -> f64 {
            if rows.is_empty() {
                return f64::NAN;
            }
            let mut s = 0.0;
            for &i in rows {
                s += match y {
                    Response::Numeric(v) => {
                        let r = v[i] - f[i][0];
                        if loss == GbdtLoss::Huber {
                            huber(r, delta)
                        } else {
                            r * r
                        }
                    }
                    Response::Class { codes, .. } => {
                        let p = if k == 2 {
                            let p1 = sigmoid(f[i][0]);
                            if codes[i] == 1 {
                                p1
                            } else {
                                1.0 - p1
                            }
                        } else {
                            softmax(&f[i])[codes[i]]
                        };
                        -p.max(1e-15).ln()
                    }
                };
            }
            s / rows.len() as f64
        };
        let tl = eval(&train);
        train_loss.push(tl);
        if n_valid > 0 {
            let vl = eval(&valid_sorted);
            valid_loss.push(vl);
            if vl < best.0 {
                best = (vl, m + 1);
            }
            if o.early_stopping_rounds > 0 && m + 1 >= best.1 + o.early_stopping_rounds {
                break;
            }
        }
    }
    let best_iteration = if n_valid > 0 { best.1.max(1) } else { trees.len() };
    trees.truncate(best_iteration);
    for g in stage_importance.iter().take(best_iteration) {
        for (a, b) in importance.iter_mut().zip(g) {
            *a += b;
        }
    }

    let mut model = GbdtModel {
        init,
        trees,
        learning_rate: o.learning_rate,
        schema: ds.schema(),
        levels,
        options: o.clone(),
        loss,
        n,
        n_train,
        n_valid,
        is_validation,
        train_loss,
        valid_loss,
        best_iteration,
        importance,
        fitted: Predictions::default(),
        train_regression: None,
        valid_regression: None,
        train_class: None,
        valid_class: None,
    };
    model.fitted = model.predict_dataset(ds);
    let pick_r = |rows: &[usize], v: &[f64], p: &[f64]| -> RegressionFit {
        regression_fit(&rows.iter().map(|&i| v[i]).collect::<Vec<_>>(), &rows.iter().map(|&i| p[i]).collect::<Vec<_>>())
    };
    match y {
        Response::Numeric(v) => {
            model.train_regression = Some(pick_r(&train, v, &model.fitted.values));
            if n_valid > 0 {
                model.valid_regression = Some(pick_r(&valid_sorted, v, &model.fitted.values));
            }
        }
        Response::Class { codes, levels } => {
            let pick_c = |rows: &[usize]| -> ClassFit {
                class_fit(
                    &rows.iter().map(|&i| codes[i]).collect::<Vec<_>>(),
                    &rows.iter().map(|&i| model.fitted.probabilities[i].clone()).collect::<Vec<_>>(),
                    levels,
                )
            };
            model.train_class = Some(pick_c(&train));
            if n_valid > 0 {
                model.valid_class = Some(pick_c(&valid_sorted));
            }
        }
    }
    Ok(model)
}

impl GbdtModel {
    fn raw(&self, d: &Dataset, row: usize) -> Vec<f64> {
        let mut f = self.init.clone();
        for stage in &self.trees {
            for (dim, tree) in stage.iter().enumerate() {
                f[dim] += self.learning_rate * tree.predict_value(d, row);
            }
        }
        f
    }

    fn predict_dataset(&self, d: &Dataset) -> Predictions {
        let mut out = Predictions::default();
        for i in 0..d.n_rows() {
            let f = self.raw(d, i);
            if self.levels.is_empty() {
                out.values.push(f[0]);
            } else {
                let probs = if self.levels.len() == 2 {
                    let p = sigmoid(f[0]);
                    vec![1.0 - p, p]
                } else {
                    softmax(&f)
                };
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

    /// Whether the model classifies.
    pub fn is_classification(&self) -> bool {
        !self.levels.is_empty()
    }
}
