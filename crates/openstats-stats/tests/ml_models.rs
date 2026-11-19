//! Behavioural tests of the tree models: exact small cases worked by hand, structural invariants of
//! the pruning sequence, surrogate splits, categorical splits against brute force, out-of-bag and
//! interval calibration of random forests, and boosting dynamics.

use openstats_math::random::variates::std_normal;
use openstats_math::random::{Pcg64, Rng};
use openstats_stats::ml::tree::Test;
use openstats_stats::ml::{
    fit_cart, fit_forest, fit_gbdt, relative_importance, CartOptions, Criterion, Dataset, ForestOptions, GbdtLoss,
    GbdtOptions, Pruning, Response,
};
use openstats_stats::report::Reportable;

fn rng(seed: u64) -> Pcg64 {
    Pcg64::with_stream(seed, 0)
}

fn nums(v: &[f64]) -> Response {
    Response::numeric(v.to_vec()).unwrap()
}

fn labels(v: &[&str]) -> Response {
    Response::from_labels(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
}

fn ds1(x: &[f64]) -> Dataset {
    Dataset::new().with_numeric("x", x.to_vec()).unwrap()
}

fn plain() -> CartOptions {
    CartOptions {
        min_split: 2,
        min_leaf: 1,
        pruning: Pruning::None,
        cv_folds: 0,
        max_surrogates: 0,
        ..Default::default()
    }
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{a} vs {b}");
}

/// `n` observations of a noisy linear signal and two noise predictors.
fn linear_data(n: usize, noise: f64, seed: u64) -> (Dataset, Vec<f64>) {
    let mut r = rng(seed);
    let (mut x1, mut x2, mut x3, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let a = r.next_f64() * 10.0;
        x1.push(a);
        x2.push(r.next_f64());
        x3.push(r.next_f64());
        y.push(2.0 * a + noise * std_normal(&mut r));
    }
    let ds = Dataset::new()
        .with_numeric("signal", x1)
        .unwrap()
        .with_numeric("noise1", x2)
        .unwrap()
        .with_numeric("noise2", x3)
        .unwrap();
    (ds, y)
}

/// Two well separated classes in the plane plus a noise predictor.
fn blobs(n: usize, seed: u64) -> (Dataset, Vec<String>) {
    let mut r = rng(seed);
    let (mut x, mut y, mut z, mut c) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let k = i % 3;
        x.push([0.0, 5.0, 0.0][k] + std_normal(&mut r) * 0.6);
        y.push([0.0, 0.0, 5.0][k] + std_normal(&mut r) * 0.6);
        z.push(std_normal(&mut r));
        c.push(["A", "B", "C"][k].to_string());
    }
    (Dataset::new().with_numeric("x", x).unwrap().with_numeric("y", y).unwrap().with_numeric("z", z).unwrap(), c)
}

// ------------------------------------------------------------------------------------- CART

#[test]
fn step_function_is_one_split() {
    let x: Vec<f64> = (1..=8).map(f64::from).collect();
    let y = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
    let m = fit_cart(&ds1(&x), &nums(&y), &plain()).unwrap();
    assert_eq!(m.tree.n_leaves(), 2);
    let Some(split) = &m.tree.nodes[0].split else { panic!("no split") };
    assert_eq!(split.test, Test::LessEq { feature: 0, threshold: 4.5 });
    assert_eq!(m.fitted.values, vec![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]);
    close(m.regression.as_ref().unwrap().r2, 1.0, 1e-12);
    // The root deviance is the total sum of squares 8 * 4 = 32 and the split removes all of it.
    close(m.tree.nodes[0].deviance, 32.0, 1e-12);
    close(split.gain, 32.0, 1e-12);
}

#[test]
fn impurities_by_hand() {
    // Six of class a and four of b: Gini 1 - .36 - .16 = .48 (deviance 4.8), entropy 0.97095 bits.
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    let c = ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"];
    let gini = fit_cart(&ds1(&x), &labels(&c), &CartOptions { criterion: Some(Criterion::Gini), ..plain() }).unwrap();
    close(gini.tree.nodes[0].deviance, 4.8, 1e-12);
    let ent = fit_cart(&ds1(&x), &labels(&c), &CartOptions { criterion: Some(Criterion::Entropy), ..plain() }).unwrap();
    close(ent.tree.nodes[0].deviance, 10.0 * 0.970_950_594_454_668_6, 1e-12);
    // Perfectly separable: one split, accuracy 1, split at 5.5.
    assert_eq!(gini.tree.n_leaves(), 2);
    assert_eq!(gini.classification.as_ref().unwrap().accuracy, 1.0);
    let Some(s) = &gini.tree.nodes[0].split else { panic!() };
    assert_eq!(s.test, Test::LessEq { feature: 0, threshold: 5.5 });
}

#[test]
fn xor_needs_depth_two() {
    let (mut a, mut b, mut c) = (vec![], vec![], vec![]);
    for i in 0..40 {
        let (p, q) = (i % 2 == 0, (i / 2) % 2 == 0);
        a.push(if p { 0.0 } else { 1.0 } + (i as f64) * 1e-3);
        b.push(if q { 0.0 } else { 1.0 } + (i as f64) * 1e-3);
        c.push(if p != q { "odd" } else { "even" });
    }
    let ds = Dataset::new().with_numeric("a", a).unwrap().with_numeric("b", b).unwrap();
    let m = fit_cart(&ds, &labels(&c), &plain()).unwrap();
    // Greedy splitting cannot see XOR in one step (every first split has ~zero gain), but a full tree
    // still separates the classes exactly.
    assert_eq!(m.classification.as_ref().unwrap().accuracy, 1.0);
    assert!(m.tree.depth() >= 2 && m.tree.n_leaves() >= 4);
}

/// Best Gini gain over every split of a categorical predictor into two non-empty level groups.
fn brute_force_gain(levels: &[usize], classes: &[usize], k: usize, nl: usize) -> f64 {
    let dev = |counts: &[f64]| {
        let n: f64 = counts.iter().sum();
        if n == 0.0 {
            0.0
        } else {
            n - counts.iter().map(|c| c * c).sum::<f64>() / n
        }
    };
    let mut all = vec![0.0; k];
    for &c in classes {
        all[c] += 1.0;
    }
    let mut best = f64::NEG_INFINITY;
    for mask in 1u32..(1 << nl) - 1 {
        let (mut l, mut r) = (vec![0.0; k], vec![0.0; k]);
        for (i, &lv) in levels.iter().enumerate() {
            if mask >> lv & 1 == 1 {
                l[classes[i]] += 1.0;
            } else {
                r[classes[i]] += 1.0;
            }
        }
        if l.iter().sum::<f64>() > 0.0 && r.iter().sum::<f64>() > 0.0 {
            best = best.max(dev(&all) - dev(&l) - dev(&r));
        }
    }
    best
}

#[test]
fn categorical_split_is_the_optimal_grouping() {
    // Two classes: the sorted-by-proportion shortcut must find the brute-force optimum.
    let mut r = rng(5);
    let nl = 6;
    let levels: Vec<usize> = (0..120).map(|_| usize::try_from(r.below(nl as u64)).unwrap()).collect();
    let p1 = [0.1, 0.8, 0.3, 0.9, 0.2, 0.6];
    let classes: Vec<usize> = levels.iter().map(|&l| usize::from(r.next_f64() < p1[l])).collect();
    for k_classes in [2usize, 3] {
        let cls: Vec<usize> = if k_classes == 2 {
            classes.clone()
        } else {
            classes
                .iter()
                .zip(&levels)
                .map(|(&c, &l)| (c + usize::from(l % 2 == 0 && r.next_f64() < 0.5)) % 3)
                .collect()
        };
        let lab: Vec<String> = levels.iter().map(|l| format!("L{l}")).collect();
        let ds = Dataset::new().with_categorical("g", &lab).unwrap();
        let resp = Response::from_labels(&cls.iter().map(|c| format!("c{c}")).collect::<Vec<_>>()).unwrap();
        let m = fit_cart(&ds, &resp, &CartOptions { max_depth: 1, ..plain() }).unwrap();
        let got = m.tree.nodes[0].split.as_ref().unwrap().gain;
        // Re-index levels by the order of first appearance in the dataset's own (sorted) coding.
        let sorted_levels = ds.levels(0).unwrap().to_vec();
        let idx: Vec<usize> = lab.iter().map(|l| sorted_levels.iter().position(|s| s == l).unwrap()).collect();
        let kk = m.levels.len();
        let want = brute_force_gain(&idx, &cls_by_label(&resp, kk), kk, sorted_levels.len());
        close(got, want, 1e-9);
    }
}

fn cls_by_label(resp: &Response, _k: usize) -> Vec<usize> {
    match resp {
        Response::Class { codes, .. } => codes.clone(),
        Response::Numeric(_) => unreachable!(),
    }
}

#[test]
fn categorical_regression_groups_levels_by_mean() {
    let g = ["a", "b", "c", "d", "e"];
    let means = [1.0, 9.0, 2.0, 8.0, 3.0];
    let (mut lab, mut y) = (vec![], vec![]);
    for rep in 0..6 {
        for (i, l) in g.iter().enumerate() {
            lab.push(l.to_string());
            y.push(means[i] + f64::from(rep) * 0.001);
        }
    }
    let ds = Dataset::new().with_categorical("g", &lab).unwrap();
    let m = fit_cart(&ds, &nums(&y), &CartOptions { max_depth: 1, ..plain() }).unwrap();
    let Some(s) = &m.tree.nodes[0].split else { panic!() };
    let Test::InSet { left, .. } = &s.test else { panic!("expected a set split") };
    // {a, c, e} (low means) against {b, d}: either side may be "left".
    assert_eq!(left, &vec![true, false, true, false, true]);
    let text = m.tree.describe_test(&s.test, &m.schema, true);
    assert_eq!(text, "g in {a, c, e}");
}

#[test]
fn surrogates_route_missing_values() {
    // x2 is a perfect copy of x1 (the primary splitter); a third column is noise.
    let n = 60;
    let x1: Vec<f64> = (0..n).map(|i| f64::from(i) / 6.0).collect();
    let y: Vec<f64> = x1.iter().map(|&a| if a < 5.0 { 0.0 } else { 10.0 }).collect();
    let mut r = rng(3);
    let noise: Vec<f64> = (0..n).map(|_| r.next_f64()).collect();
    let mut x1m = x1.clone();
    let mut x2m: Vec<f64> = x1.iter().map(|a| a * 2.0 + 1.0).collect();
    for i in 0..n as usize {
        // Rows next to the class boundary (29, 30, 31) keep both predictors.
        if i % 5 == 2 {
            x1m[i] = f64::NAN;
        } else if i % 5 == 3 {
            x2m[i] = f64::NAN;
        }
    }
    let ds = Dataset::new()
        .with_numeric("x1", x1m.clone())
        .unwrap()
        .with_numeric("x2", x2m)
        .unwrap()
        .with_numeric("noise", noise)
        .unwrap();
    let with = fit_cart(&ds, &nums(&y), &CartOptions { max_depth: 1, max_surrogates: 3, ..plain() }).unwrap();
    let split = with.tree.nodes[0].split.as_ref().unwrap();
    let primary = split.test.feature();
    assert!(primary < 2, "x1 or its perfect copy x2 splits");
    let sur = &split.surrogates;
    assert!(!sur.is_empty());
    assert_eq!(sur[0].test.feature(), 1 - primary, "the copy is the best surrogate");
    close(sur[0].agreement, 1.0, 1e-12);
    close(sur[0].adjusted, 1.0, 1e-12);
    // Every observation, including those with x1 missing, lands on the right side.
    for i in 0..n as usize {
        assert_eq!(with.fitted.values[i], y[i], "observation {i}");
    }
    // Importance: the copy earns credit as a perfect surrogate, noise none.
    assert!(with.importance[0] > 0.0 && with.importance[1] > 0.0);
    assert!(with.importance[2] < 0.5 * with.importance[0].min(with.importance[1]), "{:?}", with.importance);
    // Without surrogates the missing rows take the majority branch and some are misrouted.
    let without = fit_cart(&ds, &nums(&y), &CartOptions { max_depth: 1, max_surrogates: 0, ..plain() }).unwrap();
    let wrong = (0..n as usize).filter(|&i| without.fitted.values[i] != y[i]).count();
    assert!(wrong >= 3, "only {wrong} misrouted without surrogates");
}

#[test]
fn missing_values_everywhere_do_not_break_fitting_or_prediction() {
    let (ds0, y) = linear_data(120, 0.5, 11);
    let mut cols = Vec::new();
    for f in 0..3 {
        let openstats_stats::ml::Feature::Numeric(v) = &ds0.features[f] else { unreachable!() };
        let mut v = v.clone();
        for (i, a) in v.iter_mut().enumerate() {
            if (i + f) % 4 == 0 {
                *a = f64::NAN;
            }
        }
        cols.push(v);
    }
    let ds = Dataset::new()
        .with_numeric("a", cols[0].clone())
        .unwrap()
        .with_numeric("b", cols[1].clone())
        .unwrap()
        .with_numeric("c", cols[2].clone())
        .unwrap();
    let m = fit_cart(&ds, &nums(&y), &CartOptions { min_split: 10, min_leaf: 4, ..Default::default() }).unwrap();
    assert!(m.n_missing_cells > 0);
    assert!(m.fitted.values.iter().all(|v| v.is_finite()));
    // A row with every predictor missing still gets a prediction (the majority branches).
    let blank = Dataset::new()
        .with_numeric("a", vec![f64::NAN])
        .unwrap()
        .with_numeric("b", vec![f64::NAN])
        .unwrap()
        .with_numeric("c", vec![f64::NAN])
        .unwrap();
    assert!(m.predict(&blank).unwrap().values[0].is_finite());
}

#[test]
fn pruning_sequence_is_nested_monotone_and_optimal() {
    let (ds, y) = linear_data(150, 3.0, 21);
    let m = fit_cart(
        &ds,
        &nums(&y),
        &CartOptions { min_split: 4, min_leaf: 2, pruning: Pruning::None, cv_folds: 0, ..Default::default() },
    )
    .unwrap();
    let s = &m.sequence;
    assert!(s.len() > 5);
    assert_eq!(s[0].leaves, m.full_leaves);
    assert_eq!(s.last().unwrap().leaves, 1, "the sequence ends at the root");
    for w in s.windows(2) {
        assert!(w[1].alpha >= w[0].alpha - 1e-15, "alpha must not decrease");
        assert!(w[1].leaves < w[0].leaves, "trees must shrink");
        assert!(w[1].error >= w[0].error - 1e-12, "resubstitution error cannot fall as the tree shrinks");
    }
    close(s.last().unwrap().error, 1.0, 1e-9);
    // Optimality: between two consecutive alphas the step's tree minimises R + alpha * leaves.
    for k in 0..s.len() - 1 {
        let a = 0.5 * (s[k].alpha + s[k + 1].alpha);
        let cost = |r: &openstats_stats::ml::CpRow| r.error + a * r.leaves as f64;
        let mine = cost(&s[k]);
        for other in s {
            assert!(mine <= cost(other) + 1e-9, "step {k} is not optimal at alpha {a}");
        }
    }
    // Selection rules honour their constraint.
    let big = fit_cart(
        &ds,
        &nums(&y),
        &CartOptions { pruning: Pruning::Alpha(10.0), cv_folds: 0, min_split: 4, min_leaf: 2, ..Default::default() },
    )
    .unwrap();
    assert_eq!(big.tree.n_leaves(), 1);
    let few = fit_cart(
        &ds,
        &nums(&y),
        &CartOptions { pruning: Pruning::Leaves(5), cv_folds: 0, min_split: 4, min_leaf: 2, ..Default::default() },
    )
    .unwrap();
    assert!(few.tree.n_leaves() <= 5 && few.tree.n_leaves() >= 2);
}

#[test]
fn cross_validation_picks_a_smaller_tree_and_one_se_no_larger() {
    let (ds, y) = linear_data(300, 4.0, 8);
    let base = CartOptions { min_split: 6, min_leaf: 3, cv_folds: 10, seed: 4, ..Default::default() };
    let full = fit_cart(&ds, &nums(&y), &CartOptions { pruning: Pruning::None, ..base.clone() }).unwrap();
    let min = fit_cart(&ds, &nums(&y), &CartOptions { pruning: Pruning::MinCv, ..base.clone() }).unwrap();
    let one = fit_cart(&ds, &nums(&y), &CartOptions { pruning: Pruning::OneSe, ..base.clone() }).unwrap();
    assert!(min.tree.n_leaves() < full.tree.n_leaves(), "noise should be pruned away");
    assert!(one.tree.n_leaves() <= min.tree.n_leaves());
    assert!(min.cv.as_ref().unwrap().score > 0.55, "cv R2 {:?}", min.cv);
    // The selected row has the lowest CV error.
    let best = min.sequence.iter().filter_map(|r| r.cv_error).fold(f64::INFINITY, f64::min);
    assert!((min.sequence[min.selected].cv_error.unwrap() - best).abs() < 1e-12);
    // The one-SE choice is within one SE of the best.
    let limit = best + min.sequence[min.selected].cv_se.unwrap();
    assert!(one.sequence[one.selected].cv_error.unwrap() <= limit + 1e-12);
    // Reproducible, and a different seed changes the folds.
    let again = fit_cart(&ds, &nums(&y), &CartOptions { pruning: Pruning::MinCv, ..base.clone() }).unwrap();
    assert_eq!(min, again);
    let other = fit_cart(&ds, &nums(&y), &CartOptions { pruning: Pruning::MinCv, seed: 99, ..base }).unwrap();
    assert_ne!(min.sequence, other.sequence);
}

#[test]
fn classification_cv_reports_accuracy() {
    let (ds, c) = blobs(150, 2);
    let resp = Response::from_labels(&c).unwrap();
    let m = fit_cart(&ds, &resp, &CartOptions { min_split: 6, min_leaf: 3, ..Default::default() }).unwrap();
    let cv = m.cv.as_ref().unwrap();
    assert!(cv.score > 0.9, "cv accuracy {}", cv.score);
    assert!(m.classification.as_ref().unwrap().accuracy > 0.95);
    // The confusion matrix rows add up to the class sizes.
    let conf = &m.classification.as_ref().unwrap().confusion;
    for row in conf {
        assert_eq!(row.iter().sum::<usize>(), 50);
    }
    assert_eq!(m.tree.n_leaves(), m.tree.nodes.iter().filter(|n| n.is_leaf()).count());
    // z is pure noise.
    let rel = relative_importance(&m.importance);
    assert!(rel[2] < 25.0, "noise importance {}", rel[2]);
}

#[test]
fn structural_limits_are_respected() {
    let (ds, y) = linear_data(200, 1.0, 5);
    let m = fit_cart(
        &ds,
        &nums(&y),
        &CartOptions {
            max_depth: 3,
            min_split: 20,
            min_leaf: 8,
            pruning: Pruning::None,
            cv_folds: 0,
            max_surrogates: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(m.tree.depth() <= 3);
    for node in &m.tree.nodes {
        assert!(node.n >= 8, "terminal or internal node of {} observations", node.n);
        if !node.is_leaf() {
            assert!(node.n >= 20);
        }
    }
    // Children add up to the parent.
    for node in &m.tree.nodes {
        if let Some((l, r)) = node.children {
            assert_eq!(m.tree.nodes[l].n + m.tree.nodes[r].n, node.n);
            assert_eq!(m.tree.nodes[l].parent, Some(m.tree.nodes.iter().position(|n| n == node).unwrap()));
        }
    }
}

#[test]
fn predict_remaps_categorical_levels() {
    let lab: Vec<String> = ["red", "blue", "green", "red", "blue", "green", "red", "blue", "green", "red"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let y = [9.0, 1.0, 5.0, 9.1, 1.1, 5.1, 8.9, 0.9, 4.9, 9.0];
    let ds = Dataset::new().with_categorical("colour", &lab).unwrap();
    let m = fit_cart(&ds, &nums(&y), &plain()).unwrap();
    // New data with a different level order and an unseen colour.
    let new: Vec<String> = ["green", "red", "purple", "blue"].iter().map(|s| s.to_string()).collect();
    let p = m.predict(&Dataset::new().with_categorical("colour", &new).unwrap()).unwrap();
    close(p.values[0], 5.0, 0.05);
    close(p.values[1], 9.0, 0.05);
    close(p.values[3], 1.0, 0.05);
    assert!(p.values[2].is_finite(), "an unseen level is treated as missing");
    // Wrong number or type of predictors is an error, not a panic.
    // A numeric column where the model was fitted on a categorical one is an error, not a panic.
    assert!(m.predict(&ds1(&[1.0, 2.0])).is_err());
    let two = Dataset::new().with_categorical("a", &new).unwrap().with_numeric("b", vec![1.0; 4]).unwrap();
    assert!(m.predict(&two).is_err());
}

#[test]
fn rules_and_reports_render() {
    let (ds, c) = blobs(60, 3);
    let m = fit_cart(
        &ds,
        &Response::from_labels(&c).unwrap(),
        &CartOptions { min_split: 6, min_leaf: 3, ..Default::default() },
    )
    .unwrap();
    let report = m.to_report("CART");
    report.validate().unwrap();
    let text = report.render_text();
    for want in [
        "Model Summary",
        "Confusion Matrix",
        "Cost-Complexity Pruning Sequence",
        "Variable Importance",
        "Terminal Node Rules",
        "Splits",
    ] {
        assert!(text.contains(want), "report lacks {want}:\n{text}");
    }
    assert!(text.contains(" <= ") || text.contains(" > "));
    // JSON round trip.
    let json = report.to_json();
    assert_eq!(openstats_stats::report::SessionReport::from_json(&json).unwrap(), report);
}

#[test]
fn input_errors_are_reported_not_panicked() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert!(Response::numeric(vec![1.0, f64::NAN]).is_err());
    assert!(Response::numeric(vec![1.0, f64::INFINITY]).is_err());
    assert!(Dataset::new().with_numeric("x", vec![1.0, f64::INFINITY]).is_err());
    assert!(Dataset::new().with_numeric("x", vec![1.0]).unwrap().with_numeric("y", vec![1.0, 2.0]).is_err());
    assert!(fit_cart(&Dataset::new(), &nums(&[1.0, 2.0]), &plain()).is_err(), "no predictors");
    assert!(fit_cart(&ds1(&x), &nums(&[1.0, 2.0]), &plain()).is_err(), "length mismatch");
    assert!(fit_cart(&ds1(&x), &labels(&["a"; 6]), &plain()).is_err(), "one class");
    assert!(
        fit_cart(&ds1(&x), &nums(&[1.0; 6]), &CartOptions { pruning: Pruning::MinCv, cv_folds: 1, ..plain() }).is_err()
    );
    assert!(fit_cart(&ds1(&x), &nums(&x), &CartOptions { criterion: Some(Criterion::Gini), ..plain() }).is_err());
    assert!(fit_cart(
        &ds1(&x),
        &labels(&["a", "b", "a", "b", "a", "b"]),
        &CartOptions { criterion: Some(Criterion::Variance), ..plain() }
    )
    .is_err());
    assert!(fit_cart(&ds1(&x), &nums(&x), &CartOptions { min_leaf: 0, ..plain() }).is_err());
    // A constant response grows no tree but is not an error.
    let m = fit_cart(&ds1(&x), &nums(&[3.0; 6]), &plain()).unwrap();
    assert_eq!(m.tree.n_leaves(), 1);
    // An all-missing predictor yields a single node.
    let m = fit_cart(&ds1(&[f64::NAN; 6]), &nums(&x), &plain()).unwrap();
    assert_eq!(m.tree.n_leaves(), 1);
}

// ------------------------------------------------------------------------------- random forest

#[test]
fn forest_is_reproducible_and_seed_sensitive() {
    let (ds, y) = linear_data(100, 1.0, 1);
    let o = ForestOptions { n_trees: 25, seed: 5, ..Default::default() };
    let a = fit_forest(&ds, &nums(&y), &o).unwrap();
    let b = fit_forest(&ds, &nums(&y), &o).unwrap();
    assert_eq!(a, b);
    let c = fit_forest(&ds, &nums(&y), &ForestOptions { seed: 6, ..o }).unwrap();
    assert_ne!(a.oob.values, c.oob.values);
}

#[test]
fn regression_forest_oob_and_importance() {
    let (ds, y) = linear_data(250, 1.0, 2);
    let m = fit_forest(&ds, &nums(&y), &ForestOptions { n_trees: 150, seed: 3, ..Default::default() }).unwrap();
    assert!(m.oob_r2.unwrap() > 0.9, "OOB R2 {:?}", m.oob_r2);
    assert_eq!(m.oob_curve.len(), 150);
    assert_eq!(*m.oob_curve.last().unwrap(), m.oob_error);
    assert!(m.oob_curve[0] > m.oob_error, "more trees reduce the out-of-bag error");
    // OOB error exceeds the optimistic in-sample error.
    let fit = openstats_stats::ml::regression_fit(&y, &m.fitted.values);
    assert!(fit.rmse < m.oob_regression_fit.as_ref().unwrap().rmse);
    let imp = m.permutation.as_ref().unwrap();
    assert!(imp.mean[0] > 20.0 * imp.mean[1].abs().max(imp.mean[2].abs()).max(1e-9), "{imp:?}");
    assert!(imp.se.iter().all(|s| *s >= 0.0));
    assert!(m.impurity_importance[0] > 10.0 * m.impurity_importance[1]);
    // About 37% of the observations are out of bag for any one tree.
    let one = fit_forest(&ds, &nums(&y), &ForestOptions { n_trees: 1, ..Default::default() }).unwrap();
    let frac = one.oob_count.iter().sum::<usize>() as f64 / 250.0;
    assert!((0.30..0.45).contains(&frac), "{frac}");
}

#[test]
fn classification_forest_probabilities() {
    let (ds, c) = blobs(150, 4);
    let resp = Response::from_labels(&c).unwrap();
    let m = fit_forest(&ds, &resp, &ForestOptions { n_trees: 80, seed: 2, ..Default::default() }).unwrap();
    assert!(m.oob_error < 0.04, "OOB error {}", m.oob_error);
    assert_eq!(m.mtry, 1); // floor(sqrt(3))
    for (i, p) in m.fitted.probabilities.iter().enumerate() {
        close(p.iter().sum::<f64>(), 1.0, 1e-12);
        assert_eq!(m.fitted.classes[i], p.iter().enumerate().fold(0, |b, (j, &v)| if v > p[b] { j } else { b }));
    }
    let conf = &m.oob_class_fit.as_ref().unwrap().confusion;
    assert_eq!(conf.iter().flatten().sum::<usize>(), m.has_oob.iter().filter(|&&b| b).count());
    // The noise predictor is the least important.
    let pm = &m.permutation.as_ref().unwrap().mean;
    assert!(pm[2] < pm[0] && pm[2] < pm[1], "{pm:?}");
    // predict() on the training data reproduces the stored fit.
    assert_eq!(m.predict(&ds).unwrap(), m.fitted);
}

#[test]
fn quantile_forest_intervals_are_calibrated() {
    // y = x + noise(sd 1): 90% out-of-bag intervals should cover about 90%.
    let mut r = rng(77);
    let n = 500;
    let x: Vec<f64> = (0..n).map(|_| r.next_f64() * 10.0).collect();
    let y: Vec<f64> = x.iter().map(|a| a + std_normal(&mut r)).collect();
    let ds = ds1(&x);
    let o =
        ForestOptions { n_trees: 200, min_leaf: Some(20), interval_level: Some(0.9), seed: 1, ..Default::default() };
    let m = fit_forest(&ds, &nums(&y), &o).unwrap();
    let cov = m.interval_coverage.unwrap();
    assert!((0.84..0.96).contains(&cov), "coverage {cov}");
    assert!(m.oob_interval.iter().all(|(lo, hi)| lo <= hi));
    // New observations: wider level gives wider intervals, and they bracket the truth roughly.
    let new = ds1(&[2.0, 5.0, 8.0]);
    let narrow = m.predict_interval(&new, 0.5).unwrap();
    let wide = m.predict_interval(&new, 0.95).unwrap();
    for i in 0..3 {
        assert!(wide[i].0 <= narrow[i].0 && wide[i].1 >= narrow[i].1);
        let centre = [2.0, 5.0, 8.0][i];
        assert!(wide[i].0 < centre && wide[i].1 > centre, "{:?} around {centre}", wide[i]);
    }
    // A forest fitted without intervals refuses.
    let plain = fit_forest(&ds, &nums(&y), &ForestOptions { n_trees: 5, ..Default::default() }).unwrap();
    assert!(plain.predict_interval(&new, 0.9).is_err());
}

#[test]
fn forest_option_errors() {
    let (ds, y) = linear_data(40, 1.0, 9);
    let r = nums(&y);
    assert!(fit_forest(&ds, &r, &ForestOptions { n_trees: 0, ..Default::default() }).is_err());
    assert!(fit_forest(&ds, &r, &ForestOptions { mtry: Some(0), ..Default::default() }).is_err());
    assert!(fit_forest(&ds, &r, &ForestOptions { mtry: Some(9), ..Default::default() }).is_err());
    assert!(fit_forest(&ds, &r, &ForestOptions { sample_fraction: 0.0, ..Default::default() }).is_err());
    assert!(
        fit_forest(&ds, &r, &ForestOptions { interval_level: Some(1.5), n_trees: 5, ..Default::default() }).is_err()
    );
    let (dc, c) = blobs(30, 1);
    assert!(fit_forest(
        &dc,
        &Response::from_labels(&c).unwrap(),
        &ForestOptions { interval_level: Some(0.9), ..Default::default() }
    )
    .is_err());
    assert!(fit_forest(&ds, &r, &ForestOptions { criterion: Some(Criterion::Gini), ..Default::default() }).is_err());
}

#[test]
fn forest_report_renders() {
    let (ds, y) = linear_data(60, 1.0, 12);
    let m = fit_forest(&ds, &nums(&y), &ForestOptions { n_trees: 20, interval_level: Some(0.9), ..Default::default() })
        .unwrap();
    let r = m.to_report("Forest");
    r.validate().unwrap();
    let text = r.render_text();
    for want in
        ["Model Summary", "Out-of-Bag Performance", "Variable Importance", "Permutation", "prediction intervals"]
    {
        assert!(text.contains(want), "{want}:\n{text}");
    }
    assert_eq!(openstats_stats::report::SessionReport::from_json(&r.to_json()).unwrap(), r);
}

// ------------------------------------------------------------------------------------ boosting

fn boost() -> GbdtOptions {
    GbdtOptions { n_trees: 60, learning_rate: 0.1, max_depth: 2, min_leaf: 2, subsample: 1.0, ..Default::default() }
}

#[test]
fn boosting_training_loss_falls_every_stage() {
    let (ds, y) = linear_data(120, 1.0, 6);
    let m = fit_gbdt(&ds, &nums(&y), &boost()).unwrap();
    assert_eq!(m.train_loss.len(), 60);
    for w in m.train_loss.windows(2) {
        assert!(w[1] < w[0], "squared-error training loss must fall: {} then {}", w[0], w[1]);
    }
    assert!(m.train_regression.as_ref().unwrap().r2 > 0.95);
    // Importance finds the signal.
    let rel = relative_importance(&m.importance);
    assert_eq!(rel[0], 100.0);
    assert!(rel[1] < 10.0 && rel[2] < 10.0, "{rel:?}");
    // predict() reproduces the stored fits.
    assert_eq!(m.predict(&ds).unwrap(), m.fitted);
}

#[test]
fn classification_boosting_separates_and_probabilities_are_proper() {
    let (ds, c) = blobs(150, 7);
    let resp = Response::from_labels(&c).unwrap();
    let m = fit_gbdt(&ds, &resp, &boost()).unwrap();
    assert!(m.train_class.as_ref().unwrap().accuracy > 0.97);
    for p in &m.fitted.probabilities {
        close(p.iter().sum::<f64>(), 1.0, 1e-12);
        assert!(p.iter().all(|&q| (0.0..=1.0).contains(&q)));
    }
    for w in m.train_loss.windows(2) {
        assert!(w[1] < w[0] + 1e-12, "deviance must not rise");
    }
    // Binary case.
    let two: Vec<String> = c.iter().map(|s| if s == "A" { "no".into() } else { "yes".into() }).collect();
    let b = fit_gbdt(&ds, &Response::from_labels(&two).unwrap(), &boost()).unwrap();
    assert!(b.train_class.as_ref().unwrap().auc.unwrap() > 0.99);
    assert_eq!(b.init.len(), 1);
    assert_eq!(m.init.len(), 3);
}

#[test]
fn validation_choice_and_early_stopping() {
    // Pure noise with a big learning rate: validation loss bottoms out almost at once.
    let mut r = rng(31);
    let n = 120;
    let ds = Dataset::new()
        .with_numeric("a", (0..n).map(|_| r.next_f64()).collect())
        .unwrap()
        .with_numeric("b", (0..n).map(|_| r.next_f64()).collect())
        .unwrap();
    let y: Vec<f64> = (0..n).map(|_| std_normal(&mut r)).collect();
    let o = GbdtOptions {
        n_trees: 200,
        learning_rate: 0.5,
        max_depth: 3,
        min_leaf: 2,
        subsample: 1.0,
        validation_fraction: 0.25,
        early_stopping_rounds: 8,
        seed: 3,
        ..Default::default()
    };
    let m = fit_gbdt(&ds, &nums(&y), &o).unwrap();
    assert_eq!(m.n_valid, 30);
    assert_eq!(m.n_train, 90);
    assert_eq!(m.is_validation.iter().filter(|&&b| b).count(), 30);
    assert!(m.train_loss.len() < 200, "stopped after {}", m.train_loss.len());
    assert_eq!(m.valid_loss.len(), m.train_loss.len());
    let best = m.valid_loss.iter().cloned().fold(f64::INFINITY, f64::min);
    assert_eq!(m.valid_loss[m.best_iteration - 1], best);
    assert_eq!(m.trees.len(), m.best_iteration);
    assert!(m.valid_regression.is_some());
    // Without early stopping the full run is kept in the loss history but only the best trees are used.
    let full = fit_gbdt(&ds, &nums(&y), &GbdtOptions { early_stopping_rounds: 0, ..o }).unwrap();
    assert_eq!(full.train_loss.len(), 200);
    assert!(full.best_iteration < 200);
}

#[test]
fn huber_resists_outliers() {
    let (ds, clean) = linear_data(150, 0.3, 14);
    let mut y = clean.clone();
    for i in (0..150).step_by(15) {
        y[i] += 80.0;
    }
    let sq = fit_gbdt(&ds, &nums(&y), &boost()).unwrap();
    let hu =
        fit_gbdt(&ds, &nums(&y), &GbdtOptions { loss: Some(GbdtLoss::Huber), huber_alpha: 0.8, ..boost() }).unwrap();
    let err = |p: &[f64]| -> f64 {
        let ok: Vec<usize> = (0..150).filter(|i| i % 15 != 0).collect();
        (ok.iter().map(|&i| (p[i] - clean[i]).powi(2)).sum::<f64>() / ok.len() as f64).sqrt()
    };
    assert!(
        err(&hu.fitted.values) < 0.6 * err(&sq.fitted.values),
        "huber {} squared {}",
        err(&hu.fitted.values),
        err(&sq.fitted.values)
    );
}

#[test]
fn stochastic_boosting_is_reproducible() {
    let (ds, y) = linear_data(100, 1.0, 17);
    let o = GbdtOptions { subsample: 0.5, seed: 8, ..boost() };
    let a = fit_gbdt(&ds, &nums(&y), &o).unwrap();
    assert_eq!(a, fit_gbdt(&ds, &nums(&y), &o).unwrap());
    assert_ne!(a.fitted.values, fit_gbdt(&ds, &nums(&y), &GbdtOptions { seed: 9, ..o }).unwrap().fitted.values);
}

#[test]
fn boosting_option_errors_and_report() {
    let (ds, y) = linear_data(40, 1.0, 19);
    let r = nums(&y);
    for bad in [
        GbdtOptions { n_trees: 0, ..boost() },
        GbdtOptions { learning_rate: 0.0, ..boost() },
        GbdtOptions { learning_rate: 1.5, ..boost() },
        GbdtOptions { subsample: 0.0, ..boost() },
        GbdtOptions { validation_fraction: 0.8, ..boost() },
        GbdtOptions { max_depth: 0, ..boost() },
        GbdtOptions { huber_alpha: 1.0, ..boost() },
        GbdtOptions { loss: Some(GbdtLoss::Deviance), ..boost() },
        GbdtOptions { min_leaf: 30, ..boost() },
    ] {
        assert!(fit_gbdt(&ds, &r, &bad).is_err(), "{bad:?}");
    }
    let (dc, c) = blobs(30, 1);
    assert!(fit_gbdt(
        &dc,
        &Response::from_labels(&c).unwrap(),
        &GbdtOptions { loss: Some(GbdtLoss::Huber), ..boost() }
    )
    .is_err());
    let m = fit_gbdt(&ds, &r, &GbdtOptions { validation_fraction: 0.2, ..boost() }).unwrap();
    let rep = m.to_report("Boosting");
    rep.validate().unwrap();
    let text = rep.render_text();
    for want in ["Model Summary", "Model Fit (training rows)", "Model Fit (validation rows)", "Variable Importance"] {
        assert!(text.contains(want), "{want}");
    }
    assert_eq!(openstats_stats::report::SessionReport::from_json(&rep.to_json()).unwrap(), rep);
}
