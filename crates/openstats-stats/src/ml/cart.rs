//! Classification and regression trees (CART) with cost-complexity pruning.
//!
//! The tree is grown to its largest size (nodes of at least `min_split` observations, children of
//! at least `min_leaf`), then pruned back along the weakest-link sequence: with the resubstitution
//! error R(t) of a node taken as a leaf (sum of squares for regression, misclassified count for
//! classification) divided by the error of the root, the subtree T_α minimising
//! R_α(T) = R(T) + α·|T̃| (|T̃| = terminal nodes) is found for every α ≥ 0 by repeatedly collapsing
//! the node with the smallest g(t) = (R(t) − R(T_t)) / (|T̃_t| − 1). The tree size is chosen by
//! K-fold cross-validation: the minimum cross-validated error, or the smallest tree within one
//! standard error of it.

use super::data::{Dataset, Presort, Response};
use super::tree::{grow, Criterion, GrowParams, Target, Tree};
use super::{class_fit, regression_fit, rng_for, shuffle, ClassFit, Predictions, RegressionFit};
use crate::error::{invalid, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// How the final tree size is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pruning {
    /// Keep the largest tree.
    None,
    /// The subtree with the smallest cross-validated error.
    MinCv,
    /// The smallest subtree whose cross-validated error is within one standard error of the minimum.
    OneSe,
    /// The optimal subtree for this complexity parameter α (relative to the root's error).
    Alpha(f64),
    /// The best subtree with at most this many terminal nodes.
    Leaves(usize),
}

/// Controls of [`fit_cart`].
#[derive(Debug, Clone, PartialEq)]
pub struct CartOptions {
    /// Splitting criterion; `None` chooses variance for a numeric and Gini for a class response.
    pub criterion: Option<Criterion>,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Smallest node that is split.
    pub min_split: usize,
    /// Smallest child node.
    pub min_leaf: usize,
    /// A split must reduce the root deviance by at least this fraction.
    pub cp: f64,
    /// Surrogate splits kept per node.
    pub max_surrogates: usize,
    /// How the tree size is chosen.
    pub pruning: Pruning,
    /// Cross-validation folds (0 = none; required > 1 for the cross-validated rules).
    pub cv_folds: usize,
    /// Seed for the fold assignment.
    pub seed: u64,
}

impl Default for CartOptions {
    fn default() -> Self {
        CartOptions {
            criterion: None,
            max_depth: 30,
            min_split: 10,
            min_leaf: 3,
            cp: 0.0,
            max_surrogates: 5,
            pruning: Pruning::MinCv,
            cv_folds: 10,
            seed: 1,
        }
    }
}

/// One row of the cost-complexity table: the subtree that is optimal from `alpha` up to the next row.
#[derive(Debug, Clone, PartialEq)]
pub struct CpRow {
    /// Complexity parameter at which this subtree becomes optimal (relative to the root's error).
    pub alpha: f64,
    /// Terminal nodes.
    pub leaves: usize,
    /// Resubstitution error relative to the root.
    pub error: f64,
    /// Cross-validated error relative to the root.
    pub cv_error: Option<f64>,
    /// Standard error of the cross-validated error.
    pub cv_se: Option<f64>,
}

/// Cross-validated performance of the selected tree.
#[derive(Debug, Clone, PartialEq)]
pub struct CvSummary {
    /// Number of folds.
    pub folds: usize,
    /// Cross-validated R² (regression) or accuracy (classification).
    pub score: f64,
    /// Standard error of the score.
    pub se: f64,
}

/// A fitted classification or regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct CartModel {
    /// The final tree.
    pub tree: Tree,
    /// Predictor names and levels the model was fitted on.
    pub schema: Dataset,
    /// Class labels (empty for regression).
    pub levels: Vec<String>,
    /// The options used.
    pub options: CartOptions,
    /// The criterion used.
    pub criterion: Criterion,
    /// Observations.
    pub n: usize,
    /// Missing predictor cells in the training data.
    pub n_missing_cells: usize,
    /// Terminal nodes of the largest tree.
    pub full_leaves: usize,
    /// The cost-complexity table, largest tree first.
    pub sequence: Vec<CpRow>,
    /// Index into `sequence` of the chosen tree.
    pub selected: usize,
    /// Variable importance (raw, one per predictor).
    pub importance: Vec<f64>,
    /// Cross-validated score of the chosen tree.
    pub cv: Option<CvSummary>,
    /// Terminal node (1-based, left to right) of every training observation.
    pub terminal_node: Vec<usize>,
    /// Predictions for the training observations.
    pub fitted: Predictions,
    /// Regression fit on the training data.
    pub regression: Option<RegressionFit>,
    /// Classification fit on the training data.
    pub classification: Option<ClassFit>,
}

// ------------------------------------------------------------------ pruning

#[derive(PartialEq)]
struct Entry {
    g: f64,
    id: usize,
    ver: u32,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is a max-heap: reverse so the smallest g (then smallest id) pops first.
        o.g.total_cmp(&self.g).then(o.id.cmp(&self.id))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// One subtree of the nested weakest-link sequence.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PruneStep {
    pub alpha: f64,
    pub ops: usize,
    pub leaves: usize,
    pub error: f64,
}

/// The nested sequence of optimal subtrees of a tree.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PruneSequence {
    /// Nodes collapsed, in order.
    pub ops: Vec<usize>,
    /// Subtrees from the largest (step 0) to the root alone.
    pub steps: Vec<PruneStep>,
}

impl PruneSequence {
    /// Number of collapses that give the optimal subtree for complexity `alpha`.
    pub(crate) fn ops_for_alpha(&self, alpha: f64) -> usize {
        let mut ops = 0;
        for s in &self.steps {
            if s.alpha <= alpha * (1.0 + 1e-12) + 1e-300 {
                ops = s.ops;
            }
        }
        ops
    }

    /// Leaf flags after `n_ops` collapses.
    pub(crate) fn mask(&self, n_nodes: usize, n_ops: usize) -> Vec<bool> {
        let mut m = vec![false; n_nodes];
        for &id in self.ops.iter().take(n_ops) {
            m[id] = true;
        }
        m
    }
}

pub(crate) fn prune_sequence(tree: &Tree) -> PruneSequence {
    let nn = tree.nodes.len();
    let scale = tree.nodes[0].error;
    let r: Vec<f64> = tree.nodes.iter().map(|n| if scale > 0.0 { n.error / scale } else { 0.0 }).collect();
    let mut leaves = vec![1usize; nn];
    let mut sub = r.clone();
    for i in (0..nn).rev() {
        if let Some((l, rr)) = tree.nodes[i].children {
            leaves[i] = leaves[l] + leaves[rr];
            sub[i] = sub[l] + sub[rr];
        }
    }
    let mut total_leaves = leaves[0];
    let mut total_err = sub[0];
    let mut steps = vec![PruneStep { alpha: 0.0, ops: 0, leaves: total_leaves, error: total_err }];
    let mut ops = Vec::new();
    if scale <= 0.0 || nn == 1 {
        return PruneSequence { ops, steps };
    }
    let gval = |leaves: &[usize], sub: &[f64], i: usize| ((r[i] - sub[i]).max(0.0)) / (leaves[i] - 1).max(1) as f64;
    let mut version = vec![0u32; nn];
    let mut heap = BinaryHeap::new();
    for i in 0..nn {
        if tree.nodes[i].children.is_some() {
            heap.push(Entry { g: gval(&leaves, &sub, i), id: i, ver: 0 });
        }
    }
    let mut dead = vec![false; nn];
    let mut collapsed = vec![false; nn];
    let mut prev_alpha = 0.0_f64;
    while let Some(e) = heap.pop() {
        if dead[e.id] || collapsed[e.id] || e.ver != version[e.id] {
            continue;
        }
        let id = e.id;
        let alpha = prev_alpha.max(e.g);
        prev_alpha = alpha;
        collapsed[id] = true;
        // The subtree below `id` no longer exists.
        let mut stack: Vec<usize> = tree.nodes[id].children.iter().flat_map(|&(l, r)| [l, r]).collect();
        while let Some(d) = stack.pop() {
            dead[d] = true;
            if let Some((l, rr)) = tree.nodes[d].children {
                stack.push(l);
                stack.push(rr);
            }
        }
        let lost_leaves = leaves[id] - 1;
        let lost_sub = sub[id] - r[id];
        leaves[id] = 1;
        sub[id] = r[id];
        total_leaves -= lost_leaves;
        total_err -= lost_sub;
        let mut a = tree.nodes[id].parent;
        while let Some(p) = a {
            leaves[p] -= lost_leaves;
            sub[p] -= lost_sub;
            version[p] += 1;
            heap.push(Entry { g: gval(&leaves, &sub, p), id: p, ver: version[p] });
            a = tree.nodes[p].parent;
        }
        ops.push(id);
        let step = PruneStep { alpha, ops: ops.len(), leaves: total_leaves, error: total_err.max(0.0) };
        let same = steps.last().is_some_and(|s| s.ops > 0 && (s.alpha - alpha).abs() <= 1e-12 * (1.0 + alpha));
        if same {
            if let Some(last) = steps.last_mut() {
                *last = step;
            }
        } else {
            steps.push(step);
        }
    }
    PruneSequence { ops, steps }
}

// ------------------------------------------------------------------ cross-validation

fn assign_folds(y: &Response, folds: usize, seed: u64) -> Vec<usize> {
    let n = y.len();
    let mut rng = rng_for(seed, 7);
    let mut fold = vec![0usize; n];
    match y {
        Response::Numeric(_) => {
            let mut perm: Vec<usize> = (0..n).collect();
            shuffle(&mut perm, &mut rng);
            for (i, &r) in perm.iter().enumerate() {
                fold[r] = i % folds;
            }
        }
        Response::Class { codes, levels } => {
            let mut counter = 0usize;
            for c in 0..levels.len() {
                let mut rows: Vec<usize> = (0..n).filter(|&i| codes[i] == c).collect();
                shuffle(&mut rows, &mut rng);
                for r in rows {
                    fold[r] = counter % folds;
                    counter += 1;
                }
            }
        }
    }
    fold
}

fn target_of(y: &Response) -> Target<'_> {
    match y {
        Response::Numeric(v) => Target::Reg(v),
        Response::Class { codes, levels } => Target::Class { codes, k: levels.len() },
    }
}

/// Held-out loss of every fold at every complexity in `betas`: squared error (regression) or the
/// number of misclassified observations (classification).
fn cv_losses(
    ds: &Dataset,
    presort: &Presort,
    y: &Response,
    params: &GrowParams,
    betas: &[f64],
    folds: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n = ds.n_rows();
    let fold = assign_folds(y, folds, seed);
    let target = target_of(y);
    let mut out = Vec::with_capacity(folds);
    let mut rng = rng_for(seed, 11);
    for f in 0..folds {
        let counts: Vec<u32> = (0..n).map(|i| u32::from(fold[i] != f)).collect();
        let (tree, _) = grow(ds, presort, target, &counts, params, &mut rng);
        let seq = prune_sequence(&tree);
        let held: Vec<usize> = (0..n).filter(|&i| fold[i] == f).collect();
        let mut losses = Vec::with_capacity(betas.len());
        for &b in betas {
            let mask = seq.mask(tree.nodes.len(), seq.ops_for_alpha(b));
            let mut loss = 0.0;
            for &i in &held {
                let leaf = tree.leaf_masked(ds, i, &mask);
                loss += match y {
                    Response::Numeric(v) => {
                        let e = v[i] - tree.nodes[leaf].value;
                        e * e
                    }
                    Response::Class { codes, .. } => f64::from(u8::from(tree.class_of(leaf) != codes[i])),
                };
            }
            losses.push(loss);
        }
        out.push(losses);
    }
    out
}

// ------------------------------------------------------------------ fitting

/// Terminal node numbers (1-based, left to right) of every node (0 for internal nodes).
pub(crate) fn terminal_numbers(tree: &Tree) -> Vec<usize> {
    let mut k = 0;
    tree.nodes
        .iter()
        .map(|n| {
            if n.is_leaf() {
                k += 1;
                k
            } else {
                0
            }
        })
        .collect()
}

/// Grow and prune a classification or regression tree.
pub fn fit_cart(ds: &Dataset, y: &Response, o: &CartOptions) -> Result<CartModel> {
    ds.validate()?;
    let n = ds.n_rows();
    if y.len() != n {
        return invalid(format!("the response has {} values but the predictors have {n}", y.len()));
    }
    let criterion = match (o.criterion, y.is_class()) {
        (None, false) => Criterion::Variance,
        (None, true) => Criterion::Gini,
        (Some(Criterion::Variance), false) => Criterion::Variance,
        (Some(c), true) if c.is_classification() => c,
        _ => return invalid("the splitting criterion does not match the type of the response"),
    };
    if let Response::Class { codes, levels } = y {
        let present = (0..levels.len()).filter(|&c| codes.contains(&c)).count();
        if present < 2 {
            return invalid("the response has only one class");
        }
    }
    if o.min_leaf == 0 || o.min_split == 0 {
        return invalid("minimum node sizes must be at least 1");
    }
    if !(0.0..1.0).contains(&o.cp) {
        return invalid("cp must be in [0, 1)");
    }
    let needs_cv = matches!(o.pruning, Pruning::MinCv | Pruning::OneSe);
    if needs_cv && o.cv_folds < 2 {
        return invalid("cross-validated pruning needs at least 2 folds");
    }
    if let Pruning::Alpha(a) = o.pruning {
        if !(a >= 0.0 && a.is_finite()) {
            return invalid("alpha must be non-negative");
        }
    }
    let target = target_of(y);
    let presort = Presort::new(ds);
    let mut params = GrowParams::new(criterion);
    params.max_depth = o.max_depth;
    params.min_split = o.min_split;
    params.min_leaf = o.min_leaf;
    params.cp = o.cp;
    params.max_surrogates = o.max_surrogates;
    let mut rng = rng_for(o.seed, 1);
    let counts = vec![1u32; n];
    let (full, _) = grow(ds, &presort, target, &counts, &params, &mut rng);
    let seq = prune_sequence(&full);
    let root_err = full.nodes[0].error;

    // Cross-validation over the complexities between consecutive steps (geometric means).
    let folds = o.cv_folds.min(n);
    let steps = &seq.steps;
    let (mut cv_error, mut cv_se): (Vec<Option<f64>>, Vec<Option<f64>>) =
        (vec![None; steps.len()], vec![None; steps.len()]);
    if o.cv_folds >= 2 {
        let betas: Vec<f64> = (0..steps.len())
            .map(|i| if i + 1 < steps.len() { (steps[i].alpha * steps[i + 1].alpha).sqrt() } else { steps[i].alpha })
            .collect();
        let losses = cv_losses(ds, &presort, y, &params, &betas, folds, o.seed);
        for k in 0..steps.len() {
            if root_err <= 0.0 {
                cv_error[k] = Some(0.0);
                cv_se[k] = Some(0.0);
                continue;
            }
            let per_fold: Vec<f64> = losses.iter().map(|l| l[k] * folds as f64 / root_err).collect();
            let mean = per_fold.iter().sum::<f64>() / folds as f64;
            let var = per_fold.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / (folds as f64 - 1.0).max(1.0);
            cv_error[k] = Some(mean);
            cv_se[k] = Some((var / folds as f64).sqrt());
        }
    }

    // Choose the tree.
    let last = steps.len() - 1;
    let min_cv = || -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (k, e) in cv_error.iter().enumerate() {
            if let Some(e) = e {
                if best.is_none_or(|(_, b)| *e < b - 1e-12) {
                    best = Some((k, *e));
                } else if best.is_some_and(|(_, b)| (*e - b).abs() <= 1e-12) {
                    best = Some((k, *e)); // ties go to the smaller tree
                }
            }
        }
        best
    };
    let selected = match o.pruning {
        Pruning::None => 0,
        Pruning::MinCv => min_cv().map_or(0, |b| b.0),
        Pruning::OneSe => match min_cv() {
            Some((k, e)) => {
                let limit = e + cv_se[k].unwrap_or(0.0);
                (0..=last).rev().find(|&j| cv_error[j].is_some_and(|x| x <= limit + 1e-12)).unwrap_or(k)
            }
            None => 0,
        },
        Pruning::Alpha(a) => {
            let ops = seq.ops_for_alpha(a);
            steps.iter().rposition(|s| s.ops <= ops).unwrap_or(0)
        }
        Pruning::Leaves(m) => steps.iter().position(|s| s.leaves <= m.max(1)).unwrap_or(last),
    };
    let mask = seq.mask(full.nodes.len(), steps[selected].ops);
    let tree = full.pruned(&mask);

    let sequence: Vec<CpRow> = steps
        .iter()
        .enumerate()
        .map(|(k, s)| CpRow {
            alpha: s.alpha,
            leaves: s.leaves,
            error: s.error,
            cv_error: cv_error[k],
            cv_se: cv_se[k],
        })
        .collect();
    let cv = cv_error[selected].map(|e| {
        let se = cv_se[selected].unwrap_or(0.0);
        match y {
            Response::Numeric(_) => CvSummary { folds, score: 1.0 - e, se },
            Response::Class { .. } => {
                let scale = root_err / n as f64;
                CvSummary { folds, score: 1.0 - e * scale, se: se * scale }
            }
        }
    });

    let model = finish(ds, y, tree, o, criterion, full.n_leaves(), sequence, selected, cv)?;
    Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    ds: &Dataset,
    y: &Response,
    tree: Tree,
    o: &CartOptions,
    criterion: Criterion,
    full_leaves: usize,
    sequence: Vec<CpRow>,
    selected: usize,
    cv: Option<CvSummary>,
) -> Result<CartModel> {
    let n = ds.n_rows();
    let numbers = terminal_numbers(&tree);
    let mut terminal_node = Vec::with_capacity(n);
    let mut fitted = Predictions::default();
    for i in 0..n {
        let leaf = tree.leaf(ds, i);
        terminal_node.push(numbers[leaf]);
        match y {
            Response::Numeric(_) => fitted.values.push(tree.nodes[leaf].value),
            Response::Class { .. } => {
                fitted.classes.push(tree.class_of(leaf));
                fitted.probabilities.push(tree.proportions(leaf));
            }
        }
    }
    let (levels, regression, classification) = match y {
        Response::Numeric(v) => (Vec::new(), Some(regression_fit(v, &fitted.values)), None),
        Response::Class { codes, levels } => {
            (levels.clone(), None, Some(class_fit(codes, &fitted.probabilities, levels)))
        }
    };
    let importance = tree.importance(ds.n_features());
    Ok(CartModel {
        importance,
        schema: ds.schema(),
        levels,
        options: o.clone(),
        criterion,
        n,
        n_missing_cells: ds.missing_cells(),
        full_leaves,
        sequence,
        selected,
        cv,
        terminal_node,
        fitted,
        regression,
        classification,
        tree,
    })
}

impl CartModel {
    /// Predict observations described by `ds` (same predictors, in the same order, as the fit).
    pub fn predict(&self, ds: &Dataset) -> Result<Predictions> {
        let d = ds.remap_to(&self.schema)?;
        let mut out = Predictions::default();
        for i in 0..d.n_rows() {
            let leaf = self.tree.leaf(&d, i);
            if self.levels.is_empty() {
                out.values.push(self.tree.nodes[leaf].value);
            } else {
                out.classes.push(self.tree.class_of(leaf));
                out.probabilities.push(self.tree.proportions(leaf));
            }
        }
        Ok(out)
    }

    /// Terminal nodes of the final tree.
    pub fn n_leaves(&self) -> usize {
        self.tree.n_leaves()
    }

    /// Whether the response was categorical.
    pub fn is_classification(&self) -> bool {
        !self.levels.is_empty()
    }
}
