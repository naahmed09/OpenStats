//! The binary decision tree shared by CART, random forests and gradient boosting: split search
//! (variance, Gini and entropy criteria; numeric and categorical predictors), surrogate splits
//! for missing values, prediction, importance and rule text.
//!
//! Trees are grown iteratively over pre-sorted index lists: every node keeps, for each numeric
//! predictor, its observations in sorted order and splits them by a stable partition, so no sort
//! happens below the root and growth is O(p · n · depth). An observation may appear several
//! times (a bootstrap sample); every copy counts as one observation.

use super::data::{Dataset, Feature, Presort};
use openstats_math::random::Rng;

/// Impurity measure a tree is grown with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Variance (sum of squares) reduction: regression.
    Variance,
    /// Gini impurity 1 − Σp²: classification.
    Gini,
    /// Shannon entropy −Σ p log₂ p: classification.
    Entropy,
}

impl Criterion {
    /// Whether the criterion is for a categorical response.
    pub fn is_classification(self) -> bool {
        !matches!(self, Criterion::Variance)
    }

    /// Name used in reports.
    pub fn label(self) -> &'static str {
        match self {
            Criterion::Variance => "Least squares (variance reduction)",
            Criterion::Gini => "Gini impurity",
            Criterion::Entropy => "Shannon entropy",
        }
    }
}

/// A yes/no question about one predictor. "Yes" sends the observation to the left child.
#[derive(Debug, Clone, PartialEq)]
pub enum Test {
    /// Numeric predictor `feature` is at most `threshold`.
    LessEq {
        /// Predictor index.
        feature: usize,
        /// Cut point (midway between two observed values).
        threshold: f64,
    },
    /// Categorical predictor `feature` is one of the levels flagged in `left`.
    InSet {
        /// Predictor index.
        feature: usize,
        /// One flag per level of the predictor.
        left: Vec<bool>,
    },
}

impl Test {
    /// The predictor this test reads.
    pub fn feature(&self) -> usize {
        match self {
            Test::LessEq { feature, .. } | Test::InSet { feature, .. } => *feature,
        }
    }

    /// The answer for `row`, or `None` when the predictor is missing (or the level is unknown).
    /// With `perm = Some((f, src))` predictor `f` is read from observation `src` instead.
    fn answer(&self, ds: &Dataset, row: usize, perm: Option<(usize, usize)>) -> Option<bool> {
        let f = self.feature();
        let r = match perm {
            Some((pf, src)) if pf == f => src,
            _ => row,
        };
        match (self, ds.features.get(f)?) {
            (Test::LessEq { threshold, .. }, Feature::Numeric(v)) => {
                let x = *v.get(r)?;
                (!x.is_nan()).then_some(x <= *threshold)
            }
            (Test::InSet { left, .. }, Feature::Categorical { codes, .. }) => {
                let c = usize::try_from(*codes.get(r)?).ok()?;
                left.get(c).copied()
            }
            _ => None,
        }
    }
}

/// A rule that stands in for the primary split of a node when the primary predictor is missing.
#[derive(Debug, Clone, PartialEq)]
pub struct Surrogate {
    /// The substitute test.
    pub test: Test,
    /// When set the test's "yes" goes right instead of left.
    pub reversed: bool,
    /// Fraction of observations (with both predictors present) the rule sends where the primary
    /// split sends them.
    pub agreement: f64,
    /// Agreement beyond what always following the majority child achieves, scaled to [0, 1].
    pub adjusted: f64,
}

/// The split at an internal node.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The primary test.
    pub test: Test,
    /// Improvement in deviance (variance: sum of squares; classification: n × impurity), scaled
    /// by the share of observations that had the predictor.
    pub gain: f64,
    /// Surrogate tests, best agreement first.
    pub surrogates: Vec<Surrogate>,
    /// Where an observation goes when the primary test and every surrogate are unavailable.
    pub default_left: bool,
    /// Training observations sent left.
    pub n_left: usize,
    /// Training observations sent right.
    pub n_right: usize,
}

impl Split {
    /// Whether `row` goes to the left child.
    pub fn goes_left(&self, ds: &Dataset, row: usize) -> bool {
        self.goes_left_perm(ds, row, None)
    }

    fn goes_left_perm(&self, ds: &Dataset, row: usize, perm: Option<(usize, usize)>) -> bool {
        if let Some(a) = self.test.answer(ds, row, perm) {
            return a;
        }
        for s in &self.surrogates {
            if let Some(a) = s.test.answer(ds, row, perm) {
                return a != s.reversed;
            }
        }
        self.default_left
    }
}

/// One node of a [`Tree`].
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Parent node (`None` for the root).
    pub parent: Option<usize>,
    /// Left and right children (`None` for a terminal node).
    pub children: Option<(usize, usize)>,
    /// Depth (root = 0).
    pub depth: usize,
    /// Training observations in the node.
    pub n: usize,
    /// Regression: the mean response (boosting overwrites it with the leaf value). Classification:
    /// the predicted class.
    pub value: f64,
    /// Class counts (empty for regression).
    pub counts: Vec<f64>,
    /// Deviance of the node: sum of squares (regression) or n × impurity (classification).
    pub deviance: f64,
    /// Resubstitution error of the node as a leaf: sum of squares or misclassified count.
    pub error: f64,
    /// The split (internal nodes only).
    pub split: Option<Split>,
}

impl Node {
    /// Whether the node is terminal.
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

/// A grown decision tree. Node 0 is the root and every child has a larger index than its parent.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    /// All nodes.
    pub nodes: Vec<Node>,
    /// The criterion it was grown with.
    pub criterion: Criterion,
    /// Number of classes (0 for regression).
    pub n_classes: usize,
}

/// What a tree predicts.
#[derive(Clone, Copy)]
pub(crate) enum Target<'a> {
    Reg(&'a [f64]),
    Class { codes: &'a [usize], k: usize },
}

/// Growth controls.
#[derive(Debug, Clone)]
pub struct GrowParams {
    /// Impurity measure.
    pub criterion: Criterion,
    /// Maximum depth of the tree.
    pub max_depth: usize,
    /// Minimum observations in a node for it to be split.
    pub min_split: usize,
    /// Minimum observations in a child.
    pub min_leaf: usize,
    /// A split must improve the deviance by at least `cp` × the root deviance.
    pub cp: f64,
    /// Predictors tried at each split (all when `None`).
    pub mtry: Option<usize>,
    /// Surrogate splits kept per node (0 = none).
    pub max_surrogates: usize,
    /// Return the training rows of every terminal node.
    pub keep_leaf_rows: bool,
}

impl GrowParams {
    /// Defaults for `criterion`: unrestricted depth, nodes of at least two observations split.
    pub fn new(criterion: Criterion) -> GrowParams {
        GrowParams {
            criterion,
            max_depth: 100,
            min_split: 2,
            min_leaf: 1,
            cp: 0.0,
            mtry: None,
            max_surrogates: 0,
            keep_leaf_rows: false,
        }
    }
}

// ------------------------------------------------------------------ node statistics

/// Aggregated response statistics of a set of observations.
#[derive(Clone)]
struct Agg {
    n: f64,
    sum: f64,
    counts: Vec<f64>,
}

impl Agg {
    fn new(k: usize) -> Agg {
        Agg { n: 0.0, sum: 0.0, counts: vec![0.0; k] }
    }

    fn add(&mut self, t: &Target, row: u32, centre: f64, sign: f64) {
        self.n += sign;
        match t {
            Target::Reg(y) => self.sum += sign * (y[row as usize] - centre),
            Target::Class { codes, .. } => self.counts[codes[row as usize]] += sign,
        }
    }
}

fn class_deviance(n: f64, counts: impl Iterator<Item = f64>, crit: Criterion) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    match crit {
        Criterion::Gini => n - counts.map(|c| c * c).sum::<f64>() / n,
        Criterion::Entropy => -counts.filter(|&c| c > 0.0).map(|c| c * (c / n).log2()).sum::<f64>(),
        Criterion::Variance => 0.0,
    }
}

/// Deviance reduction from splitting `total` into `left` and the remainder.
fn gain_of(total: &Agg, left: &Agg, crit: Criterion) -> f64 {
    let nr = total.n - left.n;
    if left.n <= 0.0 || nr <= 0.0 {
        return 0.0;
    }
    match crit {
        Criterion::Variance => {
            let sr = total.sum - left.sum;
            left.sum * left.sum / left.n + sr * sr / nr - total.sum * total.sum / total.n
        }
        _ => {
            let parent = class_deviance(total.n, total.counts.iter().copied(), crit);
            let l = class_deviance(left.n, left.counts.iter().copied(), crit);
            let r = class_deviance(nr, total.counts.iter().zip(&left.counts).map(|(t, l)| t - l), crit);
            parent - l - r
        }
    }
}

fn make_node(rows: &[u32], target: &Target, crit: Criterion, parent: Option<usize>, depth: usize) -> Node {
    let n = rows.len();
    match *target {
        Target::Reg(y) => {
            let sum: f64 = rows.iter().map(|&r| y[r as usize]).sum();
            let mean = if n > 0 { sum / n as f64 } else { 0.0 };
            let sse: f64 = rows
                .iter()
                .map(|&r| {
                    let d = y[r as usize] - mean;
                    d * d
                })
                .sum();
            Node {
                parent,
                children: None,
                depth,
                n,
                value: mean,
                counts: Vec::new(),
                deviance: sse,
                error: sse,
                split: None,
            }
        }
        Target::Class { codes, k } => {
            let mut counts = vec![0.0; k];
            for &r in rows {
                counts[codes[r as usize]] += 1.0;
            }
            let (best, max) =
                counts.iter().enumerate().fold((0, -1.0), |(bi, bm), (i, &c)| if c > bm { (i, c) } else { (bi, bm) });
            let dev = class_deviance(n as f64, counts.iter().copied(), crit);
            Node {
                parent,
                children: None,
                depth,
                n,
                value: best as f64,
                counts,
                deviance: dev,
                error: n as f64 - max.max(0.0),
                split: None,
            }
        }
    }
}

// ------------------------------------------------------------------ split search

struct Cand {
    gain: f64,
    test: Test,
}

fn fnum(ds: &Dataset, f: usize) -> &[f64] {
    match &ds.features[f] {
        Feature::Numeric(v) => v,
        Feature::Categorical { .. } => &[],
    }
}

fn mid(a: f64, b: f64) -> f64 {
    let m = a + (b - a) / 2.0;
    if m < b {
        m
    } else {
        a
    }
}

struct NodeCtx<'a> {
    ds: &'a Dataset,
    target: Target<'a>,
    params: &'a GrowParams,
    k: usize,
    centre: f64,
}

impl NodeCtx<'_> {
    fn agg_of(&self, rows: &[u32]) -> Agg {
        let mut a = Agg::new(self.k);
        for &r in rows {
            a.add(&self.target, r, self.centre, 1.0);
        }
        a
    }

    fn best_numeric(&self, f: usize, sorted: &[u32], node_n: usize) -> Option<Cand> {
        let m = sorted.len();
        let p = self.params;
        if m < 2 * p.min_leaf.max(1) {
            return None;
        }
        let x = fnum(self.ds, f);
        let total = self.agg_of(sorted);
        let mut left = Agg::new(self.k);
        let mut best: Option<(f64, f64)> = None;
        for i in 0..m - 1 {
            left.add(&self.target, sorted[i], self.centre, 1.0);
            let (xi, xn) = (x[sorted[i] as usize], x[sorted[i + 1] as usize]);
            let nl = i + 1;
            if xi < xn && nl >= p.min_leaf && m - nl >= p.min_leaf {
                let g = gain_of(&total, &left, p.criterion);
                if best.is_none_or(|(bg, _)| g > bg) {
                    best = Some((g, mid(xi, xn)));
                }
            }
        }
        let (g, thr) = best?;
        Some(Cand { gain: g * m as f64 / node_n as f64, test: Test::LessEq { feature: f, threshold: thr } })
    }

    fn best_categorical(&self, f: usize, rows: &[u32], node_n: usize) -> Option<Cand> {
        let Feature::Categorical { codes, levels } = &self.ds.features[f] else { return None };
        let nl = levels.len();
        let p = self.params;
        let mut stats: Vec<Agg> = (0..nl).map(|_| Agg::new(self.k)).collect();
        let mut total = Agg::new(self.k);
        for &r in rows {
            if let Ok(c) = usize::try_from(codes[r as usize]) {
                stats[c].add(&self.target, r, self.centre, 1.0);
                total.add(&self.target, r, self.centre, 1.0);
            }
        }
        let m = total.n as usize;
        let mut present: Vec<usize> = (0..nl).filter(|&l| stats[l].n > 0.0).collect();
        if present.len() < 2 || m < 2 * p.min_leaf.max(1) {
            return None;
        }
        let mut best: Option<(f64, Vec<usize>)> = None;
        let consider = |left: &Agg, set: &[usize], best: &mut Option<(f64, Vec<usize>)>| {
            let (l, r) = (left.n as usize, m - left.n as usize);
            if l >= p.min_leaf && r >= p.min_leaf && l > 0 && r > 0 {
                let g = gain_of(&total, left, p.criterion);
                if best.as_ref().is_none_or(|(bg, _)| g > *bg) {
                    *best = Some((g, set.to_vec()));
                }
            }
        };
        let exhaustive = self.k > 2 && p.criterion.is_classification() && present.len() <= 10;
        if exhaustive {
            // Every grouping of the present levels, the last level pinned right (mirror images
            // are the same split).
            let q = present.len() - 1;
            for mask in 1u32..(1u32 << q) {
                let mut left = Agg::new(self.k);
                let mut set = Vec::new();
                for (b, &l) in present.iter().take(q).enumerate() {
                    if mask >> b & 1 == 1 {
                        set.push(l);
                        left.n += stats[l].n;
                        left.sum += stats[l].sum;
                        for (a, s) in left.counts.iter_mut().zip(&stats[l].counts) {
                            *a += s;
                        }
                    }
                }
                consider(&left, &set, &mut best);
            }
        } else {
            // Levels ordered by mean response (regression), by the share of class 1 (two classes)
            // or by the share of the node's majority class (more classes, > 10 levels), then every
            // prefix is a candidate: optimal for regression and two classes (Breiman et al.).
            let key = |s: &Agg, majority: usize| -> f64 {
                match self.target {
                    Target::Reg(_) => s.sum / s.n,
                    Target::Class { k, .. } => s.counts[if k == 2 { 1 } else { majority }] / s.n,
                }
            };
            let majority = total
                .counts
                .iter()
                .enumerate()
                .fold((0, -1.0), |(bi, bm), (i, &c)| if c > bm { (i, c) } else { (bi, bm) })
                .0;
            present.sort_by(|&a, &b| key(&stats[a], majority).total_cmp(&key(&stats[b], majority)).then(a.cmp(&b)));
            let mut left = Agg::new(self.k);
            let mut set = Vec::new();
            for &l in present.iter().take(present.len() - 1) {
                set.push(l);
                left.n += stats[l].n;
                left.sum += stats[l].sum;
                for (a, s) in left.counts.iter_mut().zip(&stats[l].counts) {
                    *a += s;
                }
                consider(&left, &set, &mut best);
            }
        }
        let (g, set) = best?;
        let mut flags = vec![false; nl];
        for l in set {
            flags[l] = true;
        }
        Some(Cand { gain: g * m as f64 / node_n as f64, test: Test::InSet { feature: f, left: flags } })
    }
}

/// Fisher–Yates shuffle prefix: the first `take` entries become a uniform random subset.
fn partial_shuffle(v: &mut [usize], take: usize, rng: &mut dyn Rng) {
    let n = v.len();
    for i in 0..take.min(n) {
        let j = i + usize::try_from(rng.below((n - i) as u64)).unwrap_or(0);
        v.swap(i, j);
    }
}

/// Surrogate splits for a node whose primary split is `primary`. `dir[r]` is 1 (left) or 2
/// (right) for node rows where the primary predictor is present, 0 where it is missing.
fn find_surrogates(
    ctx: &NodeCtx,
    primary: &Test,
    rows: &[u32],
    sorted: &[Vec<u32>],
    dir: &[u8],
    max: usize,
) -> Vec<Surrogate> {
    let pf = primary.feature();
    let ds = ctx.ds;
    let mut out: Vec<(f64, f64, usize, Surrogate)> = Vec::new();
    for f in 0..ds.n_features() {
        if f == pf {
            continue;
        }
        match &ds.features[f] {
            Feature::Numeric(x) => {
                let pairs: Vec<(f64, bool)> = sorted[f]
                    .iter()
                    .filter(|&&r| dir[r as usize] != 0)
                    .map(|&r| (x[r as usize], dir[r as usize] == 1))
                    .collect();
                let both = pairs.len();
                if both < 2 {
                    continue;
                }
                let n_left_lab = pairs.iter().filter(|p| p.1).count();
                let maj = n_left_lab.max(both - n_left_lab);
                // agreement of "x <= t goes left" is (left labels with x <= t) + (right labels above t)
                let (mut left_lab_le, mut right_lab_le) = (0usize, 0usize);
                let mut best: Option<(usize, f64, bool)> = None;
                for i in 0..both - 1 {
                    if pairs[i].1 {
                        left_lab_le += 1;
                    } else {
                        right_lab_le += 1;
                    }
                    if pairs[i].0 < pairs[i + 1].0 {
                        let right_lab_total = both - n_left_lab;
                        let agree = left_lab_le + (right_lab_total - right_lab_le);
                        let (a, rev) = if agree >= both - agree { (agree, false) } else { (both - agree, true) };
                        if best.is_none_or(|(ba, _, _)| a > ba) {
                            best = Some((a, mid(pairs[i].0, pairs[i + 1].0), rev));
                        }
                    }
                }
                if let Some((agree, thr, rev)) = best {
                    if agree > maj {
                        out.push((
                            agree as f64 / both as f64,
                            (agree - maj) as f64 / (both - maj).max(1) as f64,
                            f,
                            Surrogate {
                                test: Test::LessEq { feature: f, threshold: thr },
                                reversed: rev,
                                agreement: agree as f64 / both as f64,
                                adjusted: (agree - maj) as f64 / (both - maj).max(1) as f64,
                            },
                        ));
                    }
                }
            }
            Feature::Categorical { codes, levels } => {
                let nl = levels.len();
                let mut lab = vec![(0usize, 0usize); nl];
                let (mut both, mut n_left_lab) = (0usize, 0usize);
                for &r in rows {
                    let d = dir[r as usize];
                    if d == 0 {
                        continue;
                    }
                    if let Ok(c) = usize::try_from(codes[r as usize]) {
                        both += 1;
                        if d == 1 {
                            lab[c].0 += 1;
                            n_left_lab += 1;
                        } else {
                            lab[c].1 += 1;
                        }
                    }
                }
                if both < 2 {
                    continue;
                }
                let maj = n_left_lab.max(both - n_left_lab);
                let flags: Vec<bool> = lab.iter().map(|&(l, r)| l > r).collect();
                let agree: usize = lab.iter().map(|&(l, r)| l.max(r)).sum();
                if agree > maj {
                    let adj = (agree - maj) as f64 / (both - maj).max(1) as f64;
                    out.push((
                        agree as f64 / both as f64,
                        adj,
                        f,
                        Surrogate {
                            test: Test::InSet { feature: f, left: flags },
                            reversed: false,
                            agreement: agree as f64 / both as f64,
                            adjusted: adj,
                        },
                    ));
                }
            }
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)));
    out.into_iter().take(max).map(|t| t.3).collect()
}

struct Work {
    id: usize,
    rows: Vec<u32>,
    sorted: Vec<Vec<u32>>,
}

fn partition(list: Vec<u32>, left: &[bool]) -> (Vec<u32>, Vec<u32>) {
    let mut l = Vec::with_capacity(list.len() / 2 + 1);
    let mut r = Vec::with_capacity(list.len() / 2 + 1);
    for x in list {
        if left[x as usize] {
            l.push(x);
        } else {
            r.push(x);
        }
    }
    (l, r)
}

/// Grow a tree on the observations selected by `counts` (observation `i` appears `counts[i]`
/// times). Returns the tree and, when requested, the training rows of every terminal node
/// (indexed by node number).
pub(crate) fn grow(
    ds: &Dataset,
    presort: &Presort,
    target: Target,
    counts: &[u32],
    params: &GrowParams,
    rng: &mut dyn Rng,
) -> (Tree, Vec<Vec<u32>>) {
    let n_all = ds.n_rows();
    let p = ds.n_features();
    let k = match target {
        Target::Reg(_) => 0,
        Target::Class { k, .. } => k,
    };
    let mut rows0: Vec<u32> = Vec::new();
    for (i, &c) in counts.iter().enumerate().take(n_all) {
        for _ in 0..c {
            rows0.push(u32::try_from(i).unwrap_or(0));
        }
    }
    let sorted0: Vec<Vec<u32>> = (0..p)
        .map(|f| {
            let mut v = Vec::new();
            for &r in &presort.order[f] {
                for _ in 0..counts[r as usize] {
                    v.push(r);
                }
            }
            v
        })
        .collect();
    let mut nodes = vec![make_node(&rows0, &target, params.criterion, None, 0)];
    let root_dev = nodes[0].deviance;
    let mut leaf_rows: Vec<Vec<u32>> = Vec::new();
    let mut stack = vec![Work { id: 0, rows: rows0, sorted: sorted0 }];
    let mut left_flag = vec![false; n_all];
    let mut dir = vec![0u8; n_all];
    let mut order: Vec<usize> = (0..p).collect();
    let min_leaf = params.min_leaf.max(1);

    while let Some(w) = stack.pop() {
        let (n, depth, dev, centre) = {
            let nd = &nodes[w.id];
            (nd.n, nd.depth, nd.deviance, nd.value)
        };
        let finish = |nodes_len: usize, rows: Vec<u32>, leaf_rows: &mut Vec<Vec<u32>>, id: usize| {
            if params.keep_leaf_rows {
                if leaf_rows.len() < nodes_len {
                    leaf_rows.resize(nodes_len, Vec::new());
                }
                leaf_rows[id] = rows;
            }
        };
        let splittable = depth < params.max_depth
            && n >= params.min_split.max(2 * min_leaf)
            && dev > 1e-12 * root_dev
            && root_dev > 0.0;
        if !splittable {
            finish(nodes.len(), w.rows, &mut leaf_rows, w.id);
            continue;
        }
        let ctx =
            NodeCtx { ds, target, params, k, centre: if matches!(target, Target::Reg(_)) { centre } else { 0.0 } };
        // Predictors to examine, in random order when mtry is set.
        let mut best: Option<Cand> = None;
        let tries = params.mtry.map_or(p, |m| m.clamp(1, p));
        if params.mtry.is_some() {
            partial_shuffle(&mut order, p, rng);
        }
        for (examined, &f) in order.iter().enumerate() {
            if examined >= tries && best.is_some() {
                break;
            }
            let c = match &ds.features[f] {
                Feature::Numeric(_) => ctx.best_numeric(f, &w.sorted[f], n),
                Feature::Categorical { .. } => ctx.best_categorical(f, &w.rows, n),
            };
            if let Some(c) = c {
                if best.as_ref().is_none_or(|b| c.gain > b.gain) {
                    best = Some(c);
                }
            }
        }
        let Some(best) = best else {
            finish(nodes.len(), w.rows, &mut leaf_rows, w.id);
            continue;
        };
        if best.gain <= 1e-12 * root_dev || best.gain < params.cp * root_dev {
            finish(nodes.len(), w.rows, &mut leaf_rows, w.id);
            continue;
        }
        // Direction of every observation that has the primary predictor, then surrogates.
        let (mut nl, mut nr) = (0usize, 0usize);
        for &r in &w.rows {
            dir[r as usize] = match best.test.answer(ds, r as usize, None) {
                Some(true) => {
                    nl += 1;
                    1
                }
                Some(false) => {
                    nr += 1;
                    2
                }
                None => 0,
            };
        }
        let surrogates = if params.max_surrogates > 0 {
            find_surrogates(&ctx, &best.test, &w.rows, &w.sorted, &dir, params.max_surrogates)
        } else {
            Vec::new()
        };
        let mut split =
            Split { test: best.test, gain: best.gain, surrogates, default_left: nl >= nr, n_left: 0, n_right: 0 };
        for &r in &w.rows {
            left_flag[r as usize] = split.goes_left(ds, r as usize);
        }
        let Work { id, rows, sorted } = w;
        let (lrows, rrows) = partition(rows, &left_flag);
        if lrows.is_empty() || rrows.is_empty() {
            let mut rows = lrows;
            rows.extend(rrows);
            finish(nodes.len(), rows, &mut leaf_rows, id);
            continue;
        }
        split.n_left = lrows.len();
        split.n_right = rrows.len();
        let mut lsorted = Vec::with_capacity(p);
        let mut rsorted = Vec::with_capacity(p);
        for list in sorted {
            let (l, r) = partition(list, &left_flag);
            lsorted.push(l);
            rsorted.push(r);
        }
        let lid = nodes.len();
        nodes.push(make_node(&lrows, &target, params.criterion, Some(id), depth + 1));
        let rid = nodes.len();
        nodes.push(make_node(&rrows, &target, params.criterion, Some(id), depth + 1));
        nodes[id].children = Some((lid, rid));
        nodes[id].split = Some(split);
        stack.push(Work { id: rid, rows: rrows, sorted: rsorted });
        stack.push(Work { id: lid, rows: lrows, sorted: lsorted });
    }
    leaf_rows.resize(nodes.len(), Vec::new());
    (Tree { nodes, criterion: params.criterion, n_classes: k }, leaf_rows)
}

// ------------------------------------------------------------------ using a tree

impl Tree {
    /// Number of terminal nodes.
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Depth of the deepest node.
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    fn descend(&self, ds: &Dataset, row: usize, perm: Option<(usize, usize)>, stop: Option<&[bool]>) -> usize {
        let mut id = 0;
        loop {
            let node = &self.nodes[id];
            let Some((l, r)) = node.children else { return id };
            if stop.is_some_and(|s| s[id]) {
                return id;
            }
            let Some(split) = &node.split else { return id };
            id = if split.goes_left_perm(ds, row, perm) { l } else { r };
        }
    }

    /// The terminal node observation `row` of `ds` falls in.
    pub fn leaf(&self, ds: &Dataset, row: usize) -> usize {
        self.descend(ds, row, None, None)
    }

    /// The terminal node of `row` when predictor `feature` is read from observation `source`.
    pub(crate) fn leaf_permuted(&self, ds: &Dataset, row: usize, feature: usize, source: usize) -> usize {
        self.descend(ds, row, Some((feature, source)), None)
    }

    /// The node `row` stops at when the nodes flagged in `is_leaf` are treated as terminal.
    pub(crate) fn leaf_masked(&self, ds: &Dataset, row: usize, is_leaf: &[bool]) -> usize {
        self.descend(ds, row, None, Some(is_leaf))
    }

    /// Predicted value of the node reached by `row` (regression mean, or the class index).
    pub fn predict_value(&self, ds: &Dataset, row: usize) -> f64 {
        self.nodes[self.leaf(ds, row)].value
    }

    /// Class proportions of node `id`.
    pub fn proportions(&self, id: usize) -> Vec<f64> {
        let c = &self.nodes[id].counts;
        let n: f64 = c.iter().sum();
        if n > 0.0 {
            c.iter().map(|x| x / n).collect()
        } else {
            vec![0.0; c.len()]
        }
    }

    /// Predicted class of node `id` (the most frequent class; ties go to the lowest index).
    pub fn class_of(&self, id: usize) -> usize {
        self.nodes[id].value as usize
    }

    /// A compact copy in which the nodes flagged in `is_leaf` have lost their subtrees.
    pub(crate) fn pruned(&self, is_leaf: &[bool]) -> Tree {
        let mut map = vec![usize::MAX; self.nodes.len()];
        let mut keep = Vec::new();
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            map[i] = keep.len();
            keep.push(i);
            if let Some((l, r)) = self.nodes[i].children {
                if !is_leaf[i] {
                    stack.push(r);
                    stack.push(l);
                }
            }
        }
        let nodes = keep
            .iter()
            .map(|&i| {
                let mut nd = self.nodes[i].clone();
                nd.parent = nd.parent.map(|p| map[p]);
                if is_leaf[i] {
                    nd.children = None;
                    nd.split = None;
                } else if let Some((l, r)) = nd.children {
                    nd.children = Some((map[l], map[r]));
                }
                nd
            })
            .collect();
        Tree { nodes, criterion: self.criterion, n_classes: self.n_classes }
    }

    /// Variable importance: the improvement of every split credited to its predictor, plus each
    /// surrogate's adjusted agreement times the improvement of the split it stands in for.
    pub fn importance(&self, n_features: usize) -> Vec<f64> {
        let mut imp = vec![0.0; n_features];
        for node in &self.nodes {
            if let Some(s) = &node.split {
                if let Some(v) = imp.get_mut(s.test.feature()) {
                    *v += s.gain;
                }
                for sur in &s.surrogates {
                    if let Some(v) = imp.get_mut(sur.test.feature()) {
                        *v += s.gain * sur.adjusted;
                    }
                }
            }
        }
        imp
    }

    /// Variable importance from primary splits only (used by ensembles, where no surrogates grow).
    pub(crate) fn add_primary_importance(&self, imp: &mut [f64]) {
        for node in &self.nodes {
            if let Some(s) = &node.split {
                if let Some(v) = imp.get_mut(s.test.feature()) {
                    *v += s.gain;
                }
            }
        }
    }

    /// Plain-text description of a test, e.g. `Age <= 31.5` or `Colour in {Red, Blue}`.
    pub fn describe_test(&self, t: &Test, schema: &Dataset, left: bool) -> String {
        let name = schema.names.get(t.feature()).map_or("?", String::as_str);
        match t {
            Test::LessEq { threshold, .. } => {
                format!("{name} {} {}", if left { "<=" } else { ">" }, fmt_num(*threshold))
            }
            Test::InSet { feature, left: flags } => {
                let levels = schema.levels(*feature).unwrap_or(&[]);
                let set: Vec<&str> =
                    levels.iter().zip(flags).filter(|(_, &f)| f == left).map(|(l, _)| l.as_str()).collect();
                format!("{name} {} {{{}}}", if left { "in" } else { "not in" }, set.join(", "))
            }
        }
    }

    /// The conditions on the path from the root to node `id`.
    pub fn path_conditions(&self, id: usize, schema: &Dataset) -> Vec<String> {
        let mut conds = Vec::new();
        let mut cur = id;
        while let Some(p) = self.nodes[cur].parent {
            if let (Some((l, _)), Some(split)) = (self.nodes[p].children, &self.nodes[p].split) {
                conds.push(self.describe_test(&split.test, schema, l == cur));
            }
            cur = p;
        }
        conds.reverse();
        conds
    }
}

/// A number with at most six significant digits, no trailing zeros.
pub(crate) fn fmt_num(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let a = x.abs();
    if (1e-4..1e7).contains(&a) {
        let mag = a.log10().floor() as i32;
        let decimals = usize::try_from((5 - mag).max(0)).unwrap_or(0);
        let s = format!("{x:.decimals$}");
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    } else {
        format!("{x:.5e}")
    }
}
