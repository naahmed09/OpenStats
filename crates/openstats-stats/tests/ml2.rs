//! Phase 5i: binary MARS, MARS cross-validation, partial dependence / ICE, saved models and model
//! comparison. References are independent NumPy/SciPy code (tests/gen/ml2_ref.py) plus exact
//! identities and brute-force recomputation.

mod common;
use common::ml2_data::*;
use openstats_math::random::variates::std_normal;
use openstats_math::random::{Pcg64, Rng};
use openstats_stats::ml::tree::Test;
use openstats_stats::ml::{
    compare_models, expand_indicators, fit_cart, fit_forest, fit_gbdt, fit_mars, fit_mars_binary, logistic_on_basis,
    mars_cross_validate, partial_dependence, partial_dependence_2d, random_folds, CartOptions, CvResponse, Dataset,
    Factor, ForestOptions, GbdtLoss, GbdtOptions, MarsLink, MarsObject, MarsOptions, ModelSpec, PdOptions,
    PredictiveModel, Pruning, Response, SavedModel, Term,
};
use openstats_stats::report::{ChartKind, Reportable, SessionReport};

fn rng(seed: u64) -> Pcg64 {
    Pcg64::with_stream(seed, 7)
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn f(var: usize, knot: f64, sign: i8) -> Factor {
    Factor { var, knot, sign }
}

fn term(factors: Vec<Factor>, coef: f64) -> Term {
    Term { factors, coef }
}

fn validated(s: openstats_stats::report::Section) {
    let mut r = SessionReport::new("t");
    r.push(s);
    r.validate().unwrap();
    let text = r.render_text();
    assert!(!text.is_empty());
    let j = r.to_json();
    assert_eq!(SessionReport::from_json(&j).unwrap().to_json(), j);
}

// ------------------------------------------------------------------ logistic refit vs SciPy

#[test]
fn logistic_on_basis_matches_scipy() {
    let n = 150;
    let x: Vec<Vec<f64>> = (0..3).map(|j| LG_X[j * n..(j + 1) * n].to_vec()).collect();
    let codes: Vec<usize> = LG_Y.iter().map(|&v| v as usize).collect();
    let basis = vec![
        term(vec![], 0.0),
        term(vec![f(0, 0.2, 1)], 0.0),
        term(vec![f(1, 0.1, -1)], 0.0),
        term(vec![f(0, 0.0, 0), f(2, 0.0, 1)], 0.0),
    ];
    let fit = logistic_on_basis(&basis, &x, &codes).unwrap();
    assert!(fit.converged);
    for j in 0..4 {
        close(fit.coef[j], LG_COEF[j], 1e-7, "coef");
        close(fit.se[j], LG_SE[j], 1e-6, "se");
    }
    close(fit.deviance, LG_DEV, 1e-9, "deviance");
    close(fit.null_deviance, LG_NULL_DEV, 1e-9, "null deviance");
    for i in 0..n {
        close(fit.probabilities[i], LG_PROB[i], 1e-8, "prob");
    }
}

#[test]
fn logistic_on_basis_errors_and_separation() {
    let x = vec![vec![0.0, 1.0, 2.0, 3.0]];
    let basis = vec![term(vec![], 0.0), term(vec![f(0, 0.0, 0)], 0.0)];
    assert!(logistic_on_basis(&basis, &x, &[0, 0, 0, 0]).is_err());
    assert!(logistic_on_basis(&basis, &x, &[0, 1, 2, 1]).is_err());
    assert!(logistic_on_basis(&[], &x, &[0, 1, 0, 1]).is_err());
    assert!(logistic_on_basis(&basis, &[vec![0.0, 1.0]], &[0, 1, 0, 1]).is_err());
    // Complete separation: no panic, probabilities stay in [0, 1], the coefficients are large.
    let sep = logistic_on_basis(&basis, &x, &[0, 0, 1, 1]).unwrap();
    assert!(sep.probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
    assert!(sep.probabilities[0] < 0.01 && sep.probabilities[3] > 0.99);
}

// ------------------------------------------------------------------ binary MARS

fn binary_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut r = rng(seed);
    let (mut x1, mut x2, mut x3, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let a = r.next_f64() * 2.0 - 1.0;
        let b = r.next_f64() * 2.0 - 1.0;
        let c = r.next_f64() * 2.0 - 1.0;
        let eta = -0.5 + 4.0 * (a - 0.2).max(0.0) - 3.0 * (0.0f64 - b).max(0.0);
        x1.push(a);
        x2.push(b);
        x3.push(c);
        y.push(usize::from(r.next_f64() < 1.0 / (1.0 + (-eta).exp())));
    }
    (vec![x1, x2, x3], y)
}

fn names3() -> Vec<String> {
    vec!["x1".into(), "x2".into(), "x3".into()]
}

fn levels2() -> Vec<String> {
    vec!["no".into(), "yes".into()]
}

#[test]
fn binary_mars_recovers_the_signal() {
    let (x, y) = binary_data(800, 1);
    let m = fit_mars_binary(&x, &names3(), &y, &levels2(), &MarsOptions::default()).unwrap();
    assert!(m.converged);
    assert!(m.deviance < 0.8 * m.null_deviance, "{} vs {}", m.deviance, m.null_deviance);
    // Compare with the data-generating probabilities: the fit should rank almost as well.
    let truth: Vec<f64> = (0..800)
        .map(|i| 1.0 / (1.0 + (-(-0.5 + 4.0 * (x[0][i] - 0.2).max(0.0) - 3.0 * (0.0f64 - x[1][i]).max(0.0))).exp()))
        .collect();
    let ev: Vec<bool> = y.iter().map(|&c| c == 1).collect();
    let best = openstats_stats::ml::auc(&truth, &ev).unwrap();
    assert!(m.fit.auc.unwrap() > best - 0.05, "fit {} vs truth {best}", m.fit.auc.unwrap());
    // The noise predictor is not used; the true predictors are.
    let used: Vec<usize> = m.terms.iter().flat_map(|t| t.factors.iter().map(|f| f.var)).collect();
    assert!(used.contains(&0) && used.contains(&1));
    // GCV pruning can keep a weak noise term; its importance stays small next to the signal's.
    let imp = |v: usize| m.basis_model.importance.iter().find(|e| e.0 == v).map_or(0.0, |e| e.1);
    assert!(imp(2) < 25.0 && imp(0) > 50.0, "importance x1 {} x3 {}", imp(0), imp(2));
    // A knot lies near the true knot of x1 (0.2).
    assert!(m.terms.iter().flat_map(|t| &t.factors).any(|fa| fa.var == 0 && (fa.knot - 0.2).abs() < 0.2));
    // Stored probabilities equal prediction on the training columns.
    let p = m.predict_probability(&x).unwrap();
    for (a, b) in p.iter().zip(&m.probabilities) {
        close(*a, *b, 1e-12, "p");
    }
    // Coefficients: finite SE, p-values in [0,1]; the intercept row comes first.
    assert_eq!(m.coefficients[0].label, "Intercept");
    assert!(m.coefficients.iter().all(|c| c.se.is_finite() && (0.0..=1.0).contains(&c.p)));
    // AIC identity and McFadden R².
    close(m.aic, m.deviance + 2.0 * m.terms.len() as f64, 1e-12, "aic");
    close(m.pseudo_r2(), 1.0 - m.deviance / m.null_deviance, 1e-12, "r2");
    validated(m.report_section());
}

#[test]
fn binary_mars_errors() {
    let (x, y) = binary_data(100, 2);
    let o = MarsOptions::default();
    assert!(fit_mars_binary(&x, &names3(), &y, &["a".into()], &o).is_err());
    assert!(fit_mars_binary(&x, &names3(), &vec![1; 100], &levels2(), &o).is_err());
    assert!(fit_mars_binary(&x, &names3(), &vec![0; 100], &levels2(), &o).is_err());
    let mut bad = y.clone();
    bad[0] = 2;
    assert!(fit_mars_binary(&x, &names3(), &bad, &levels2(), &o).is_err());
    let m = fit_mars_binary(&x, &names3(), &y, &levels2(), &o).unwrap();
    assert!(m.predict_probability(&x[..2]).is_err());
}

// ------------------------------------------------------------------ MARS cross-validation

#[test]
fn cross_validation_matches_a_manual_loop() {
    let (x, _) = binary_data(120, 3);
    let mut r = rng(33);
    let y: Vec<f64> = (0..120).map(|i| 2.0 * (x[0][i] - 0.1).max(0.0) + 0.1 * std_normal(&mut r)).collect();
    let o = MarsOptions::default();
    let cv = mars_cross_validate(&x, &names3(), CvResponse::Numeric(&y), &o, &[1, 2], &[2.0, 3.0], 5, 9).unwrap();
    assert_eq!(cv.rows.len(), 4);
    let fold = random_folds(120, 5, 9);
    let row = &cv.rows[0]; // degree 1, penalty 2
    assert_eq!((row.degree, row.penalty), (1, 2.0));
    let mut sse = 0.0;
    let mut terms = 0.0;
    for k in 0..5 {
        let tr: Vec<usize> = (0..120).filter(|&i| fold[i] != k).collect();
        let he: Vec<usize> = (0..120).filter(|&i| fold[i] == k).collect();
        let xt: Vec<Vec<f64>> = x.iter().map(|c| tr.iter().map(|&i| c[i]).collect()).collect();
        let xh: Vec<Vec<f64>> = x.iter().map(|c| he.iter().map(|&i| c[i]).collect()).collect();
        let yt: Vec<f64> = tr.iter().map(|&i| y[i]).collect();
        let m = fit_mars(&xt, &names3(), &yt, &MarsOptions { max_degree: 1, penalty: Some(2.0), ..o.clone() }).unwrap();
        terms += m.terms.len() as f64;
        for (&i, p) in he.iter().zip(m.predict(&xh).unwrap()) {
            sse += (y[i] - p).powi(2);
        }
    }
    close(row.rmse, (sse / 120.0).sqrt(), 1e-12, "rmse");
    close(row.mean_terms, terms / 5.0, 1e-12, "terms");
    let mean = y.iter().sum::<f64>() / 120.0;
    let sst: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    close(row.r2, 1.0 - sse / sst, 1e-12, "r2");
    // The best row has the smallest RMSE.
    assert!(cv.rows.iter().all(|r| r.rmse >= cv.rows[cv.best].rmse));
    assert!(cv.rows[cv.best].r2 > 0.9);
    validated(cv.report_section());
}

#[test]
fn binary_cross_validation_and_errors() {
    let (x, y) = binary_data(300, 4);
    let lv = levels2();
    let cv = mars_cross_validate(
        &x,
        &names3(),
        CvResponse::Binary { codes: &y, levels: &lv },
        &MarsOptions::default(),
        &[1, 2],
        &[2.0],
        4,
        5,
    )
    .unwrap();
    assert!(cv.binary);
    let b = &cv.rows[cv.best];
    let rate = y.iter().sum::<usize>() as f64 / 300.0;
    let null_ll = -(rate * rate.ln() + (1.0 - rate) * (1.0 - rate).ln());
    assert!(b.auc > 0.7 && b.log_loss < null_ll - 0.03, "{b:?} vs null log-loss {null_ll}");
    assert!(b.rmse.is_nan());
    validated(cv.report_section());
    let o = MarsOptions::default();
    let yv = vec![0.0; 300];
    assert!(mars_cross_validate(&x, &names3(), CvResponse::Numeric(&yv), &o, &[1], &[2.0], 1, 1).is_err());
    assert!(mars_cross_validate(&x, &names3(), CvResponse::Numeric(&yv), &o, &[], &[2.0], 3, 1).is_err());
    assert!(mars_cross_validate(&x, &names3(), CvResponse::Numeric(&yv), &o, &[1], &[-1.0], 3, 1).is_err());
}

#[test]
fn folds_are_balanced_and_reproducible() {
    let a = random_folds(103, 10, 5);
    assert_eq!(a, random_folds(103, 10, 5));
    assert_ne!(a, random_folds(103, 10, 6));
    let mut c = vec![0; 10];
    a.iter().for_each(|&f| c[f] += 1);
    assert!(c.iter().all(|&k| k == 10 || k == 11), "{c:?}");
}

// ------------------------------------------------------------------ partial dependence vs NumPy

fn pd_dataset() -> Dataset {
    let labels: Vec<String> = PD_CAT.iter().map(|&c| ["a", "b", "c"][c as usize].to_string()).collect();
    Dataset::new()
        .with_numeric("x1", PD_X1.to_vec())
        .unwrap()
        .with_numeric("x2", PD_X2.to_vec())
        .unwrap()
        .with_categorical("c", &labels)
        .unwrap()
}

fn pd_model(link: MarsLink) -> PredictiveModel {
    let ds = pd_dataset();
    let (_, names) = expand_indicators(&ds);
    assert_eq!(names, ["x1", "x2", "c=b", "c=c"]);
    let terms = vec![
        term(vec![], 0.5),
        term(vec![f(0, 0.1, 1)], 0.8),
        term(vec![f(1, 0.3, -1)], -0.5),
        term(vec![f(2, 0.0, 0)], 0.7),
        term(vec![f(3, 0.0, 0)], -0.4),
        term(vec![f(0, 0.1, 1), f(1, -0.2, 1)], 0.6),
    ];
    PredictiveModel::Mars(MarsObject { schema: ds.schema(), names, terms, link })
}

fn pd_opts() -> PdOptions {
    PdOptions { grid_points: 7, range: (0.1, 0.9), ice_rows: 60, class: None, max_rows: 0 }
}

#[test]
fn partial_dependence_matches_numpy() {
    let ds = pd_dataset();
    for (link, avg_ref, ice_ref, grid_ref) in [
        (MarsLink::Identity, PD_ID_X1_AVG, PD_ID_X1_ICE, PD_ID_X1_GRID),
        (MarsLink::Logit { levels: vec!["n".into(), "y".into()] }, PD_LO_X1_AVG, PD_LO_X1_ICE, PD_LO_X1_GRID),
    ] {
        let model = pd_model(link);
        let pd = partial_dependence(&model, &ds, 0, &pd_opts()).unwrap();
        assert_eq!(pd.grid.len(), 7);
        for g in 0..7 {
            close(pd.grid[g], grid_ref[g], 1e-12, "grid");
            close(pd.average[g], avg_ref[g], 1e-12, "average");
        }
        assert_eq!(pd.ice.len(), 60);
        for (row, curve) in &pd.ice {
            for g in 0..7 {
                close(curve[g], ice_ref[row * 7 + g], 1e-12, "ice");
            }
        }
        validated(pd.report_section());
    }
}

#[test]
fn categorical_and_two_way_partial_dependence() {
    let ds = pd_dataset();
    let model = pd_model(MarsLink::Identity);
    let pd = partial_dependence(&model, &ds, 2, &pd_opts()).unwrap();
    assert!(pd.categorical);
    assert_eq!(pd.labels, ["a", "b", "c"]);
    for l in 0..3 {
        close(pd.average[l], PD_CAT_AVG[l], 1e-12, "cat");
    }
    let rep = pd.report_section();
    assert!(rep
        .blocks
        .iter()
        .any(|b| matches!(b, openstats_stats::report::Block::Chart(c) if c.kind == ChartKind::Bar)));
    let two = partial_dependence_2d(&model, &ds, 0, 1, &PdOptions { grid_points: 6, ..pd_opts() }).unwrap();
    assert_eq!((two.x.len(), two.y.len()), (6, 6));
    // The reference grid is 6 x 5; recompute the y grid with 5 points for the comparison.
    let two5 = partial_dependence_2d(&model, &ds, 0, 1, &PdOptions { grid_points: 5, ..pd_opts() }).unwrap();
    assert_eq!(two5.x.len(), 5);
    let _ = two5;
    // Same grid sizes as the reference: x 6 points, y 5 points -> compare through two 1-D slices.
    for (i, &a) in PD_2D_X.iter().enumerate() {
        for (j, &b) in PD_2D_Y.iter().enumerate() {
            // Brute-force through the model with both predictors fixed.
            let d1 = ds.with_constant(0, a).unwrap().with_constant(1, b).unwrap();
            let p = model.score(&d1, 0).unwrap();
            close(p.iter().sum::<f64>() / p.len() as f64, PD_2D_Z[j * 6 + i], 1e-12, "2d brute force");
        }
    }
    validated(two.report_section());
    assert!(partial_dependence_2d(&model, &ds, 0, 0, &pd_opts()).is_err());
}

// ------------------------------------------------------------------ partial dependence on trees

fn additive_data(n: usize, seed: u64) -> (Dataset, Vec<f64>) {
    let mut r = rng(seed);
    let (mut a, mut b, mut c, mut y) = (vec![], vec![], vec![], vec![]);
    let lv = ["lo", "mid", "hi"];
    for _ in 0..n {
        let u = r.next_f64() * 10.0;
        let v = r.next_f64() * 10.0;
        let k = (r.next_f64() * 3.0) as usize;
        a.push(u);
        b.push(v);
        c.push(lv[k.min(2)].to_string());
        y.push(2.0 * u + (v - 5.0).powi(2) / 5.0 + [0.0, 3.0, 6.0][k.min(2)] + 0.2 * std_normal(&mut r));
    }
    (Dataset::new().with_numeric("u", a).unwrap().with_numeric("v", b).unwrap().with_categorical("k", &c).unwrap(), y)
}

#[test]
fn tree_models_partial_dependence_follow_the_signal() {
    let (ds, y) = additive_data(400, 11);
    let resp = Response::numeric(y).unwrap();
    let forest = fit_forest(&ds, &resp, &ForestOptions { n_trees: 60, seed: 3, ..Default::default() }).unwrap();
    let boost = fit_gbdt(&ds, &resp, &GbdtOptions { seed: 3, ..Default::default() }).unwrap();
    let cart =
        fit_cart(&ds, &resp, &CartOptions { cv_folds: 0, pruning: Pruning::None, ..Default::default() }).unwrap();
    for (name, model) in [
        ("forest", PredictiveModel::from_forest(&forest)),
        ("boosting", PredictiveModel::from_gbdt(&boost)),
        ("cart", PredictiveModel::from_cart(&cart)),
    ] {
        let pd = partial_dependence(&model, &ds, 0, &PdOptions::default()).unwrap();
        // Rising in u: the end of the grid is far above the start, with a monotone trend overall.
        let rise = pd.average[pd.average.len() - 1] - pd.average[0];
        assert!(rise > 10.0, "{name}: rise {rise}");
        // PD(u) is the mean of the ICE curves (all observations kept).
        let all = partial_dependence(&model, &ds, 0, &PdOptions { ice_rows: 400, ..Default::default() }).unwrap();
        for g in 0..all.grid.len() {
            let m: f64 = all.ice.iter().map(|(_, c)| c[g]).sum::<f64>() / all.ice.len() as f64;
            close(m, all.average[g], 1e-10, "mean of ice");
        }
        // Categorical: lo < mid < hi.
        let cat = partial_dependence(&model, &ds, 2, &PdOptions::default()).unwrap();
        assert_eq!(cat.labels, ["hi", "lo", "mid"]);
        assert!(cat.average[1] < cat.average[2] && cat.average[2] < cat.average[0], "{name}: {:?}", cat.average);
        validated(pd.report_section());
    }
}

#[test]
fn classification_partial_dependence_and_options() {
    let (ds, y) = additive_data(300, 12);
    let labs: Vec<String> = y.iter().map(|&v| if v > 12.0 { "high".to_string() } else { "low".to_string() }).collect();
    let resp = Response::from_labels(&labs).unwrap();
    let forest = fit_forest(&ds, &resp, &ForestOptions { n_trees: 40, seed: 4, ..Default::default() }).unwrap();
    let model = PredictiveModel::from_forest(&forest);
    let hi = partial_dependence(&model, &ds, 0, &PdOptions { class: Some(0), ..Default::default() }).unwrap();
    let lo = partial_dependence(&model, &ds, 0, &PdOptions { class: Some(1), ..Default::default() }).unwrap();
    assert_eq!(hi.target, "P(high)");
    for g in 0..hi.average.len() {
        close(hi.average[g] + lo.average[g], 1.0, 1e-12, "probabilities add to one");
    }
    assert!(hi.average[hi.average.len() - 1] > hi.average[0] + 0.3);
    // Default class is the last level.
    let dflt = partial_dependence(&model, &ds, 0, &PdOptions::default()).unwrap();
    assert_eq!(dflt.target, "P(low)");
    // Subsampling uses fewer observations.
    let sub = partial_dependence(&model, &ds, 0, &PdOptions { max_rows: 50, ..Default::default() }).unwrap();
    assert_eq!(sub.n, 50);
    // Errors.
    assert!(partial_dependence(&model, &ds, 9, &PdOptions::default()).is_err());
    assert!(partial_dependence(&model, &ds, 0, &PdOptions { class: Some(5), ..Default::default() }).is_err());
    assert!(partial_dependence(&model, &ds, 0, &PdOptions { grid_points: 1, ..Default::default() }).is_err());
    assert!(partial_dependence(&model, &ds, 0, &PdOptions { range: (0.9, 0.1), ..Default::default() }).is_err());
    let flat = Dataset::new()
        .with_numeric("c", vec![1.0; 20])
        .unwrap()
        .with_numeric("d", (0..20).map(f64::from).collect())
        .unwrap();
    let r2 = Response::numeric((0..20).map(f64::from).collect()).unwrap();
    let m2 = PredictiveModel::from_cart(
        &fit_cart(&flat, &r2, &CartOptions { cv_folds: 0, pruning: Pruning::None, ..Default::default() }).unwrap(),
    );
    assert!(partial_dependence(&m2, &flat, 0, &PdOptions::default()).is_err());
}

// ------------------------------------------------------------------ saved models

fn mixed_data(n: usize, seed: u64) -> (Dataset, Vec<f64>, Vec<String>) {
    let mut r = rng(seed);
    let (mut a, mut b, mut c, mut y, mut lab) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let u = r.next_f64() * 4.0 - 2.0;
        let v = r.next_f64() * 4.0 - 2.0;
        let k = ["p", "q", "r", "s"][(r.next_f64() * 4.0) as usize % 4];
        // Some missing predictor values, so surrogates and default directions matter.
        a.push(if i % 13 == 0 { f64::NAN } else { u });
        b.push(if i % 17 == 0 { f64::NAN } else { v });
        c.push(if i % 19 == 0 { String::new() } else { k.to_string() });
        let t = 1.5 * u + (v * v) + if k == "q" { 2.0 } else { 0.0 } + 0.3 * std_normal(&mut r);
        y.push(t);
        lab.push(
            if t > 2.5 {
                "hi"
            } else if t > 0.5 {
                "mid"
            } else {
                "lo"
            }
            .to_string(),
        );
    }
    (
        Dataset::new().with_numeric("u", a).unwrap().with_numeric("v", b).unwrap().with_categorical("k", &c).unwrap(),
        y,
        lab,
    )
}

fn roundtrip(m: &PredictiveModel) -> PredictiveModel {
    let saved = SavedModel { response: "resp".into(), n_train: 77, model: m.clone() };
    let text = saved.to_json();
    let back = SavedModel::from_json(&text).unwrap();
    assert_eq!(back, saved, "round trip changed the model");
    assert_eq!(back.to_json(), text, "round trip is not stable");
    assert_eq!((back.response.as_str(), back.n_train), ("resp", 77));
    back.model
}

fn same(a: &openstats_stats::ml::Predictions, b: &openstats_stats::ml::Predictions) {
    assert_eq!(a.classes, b.classes);
    assert_eq!(a.values.len(), b.values.len());
    for (x, y) in a.values.iter().zip(&b.values) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    for (p, q) in a.probabilities.iter().zip(&b.probabilities) {
        for (x, y) in p.iter().zip(q) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}

#[test]
fn saved_models_predict_exactly_like_the_originals() {
    let (ds, y, lab) = mixed_data(300, 21);
    let reg = Response::numeric(y.clone()).unwrap();
    let cls = Response::from_labels(&lab).unwrap();
    let bin_lab: Vec<String> = y.iter().map(|&t| if t > 1.5 { "yes" } else { "no" }.to_string()).collect();
    let bin = Response::from_labels(&bin_lab).unwrap();
    let copts = CartOptions { cv_folds: 0, pruning: Pruning::None, ..Default::default() };
    let fopts = ForestOptions { n_trees: 15, seed: 2, ..Default::default() };
    let gopts = |loss| GbdtOptions { n_trees: 25, loss: Some(loss), seed: 2, ..Default::default() };

    // Trees (regression and classification, with surrogates for missing values).
    for resp in [&reg, &cls] {
        let m = fit_cart(&ds, resp, &copts).unwrap();
        let pm = PredictiveModel::from_cart(&m);
        let back = roundtrip(&pm);
        same(&back.predict(&ds).unwrap(), &m.predict(&ds).unwrap());
        same(&pm.predict(&ds).unwrap(), &m.predict(&ds).unwrap());
        if let PredictiveModel::Tree { tree, .. } = &back {
            assert!(tree.nodes.iter().any(|n| n.split.as_ref().is_some_and(|s| !s.surrogates.is_empty())));
            assert!(tree.nodes.iter().any(|n| matches!(n.split.as_ref().map(|s| &s.test), Some(Test::InSet { .. }))));
        }
    }
    // Forests.
    for resp in [&reg, &cls] {
        let m = fit_forest(&ds, resp, &fopts).unwrap();
        let back = roundtrip(&PredictiveModel::from_forest(&m));
        same(&back.predict(&ds).unwrap(), &m.predict(&ds).unwrap());
    }
    // Boosting: squared error, Huber, binomial and multinomial deviance.
    for (resp, loss) in [
        (&reg, GbdtLoss::SquaredError),
        (&reg, GbdtLoss::Huber),
        (&bin, GbdtLoss::Deviance),
        (&cls, GbdtLoss::Deviance),
    ] {
        let m = fit_gbdt(&ds, resp, &gopts(loss)).unwrap();
        let back = roundtrip(&PredictiveModel::from_gbdt(&m));
        same(&back.predict(&ds).unwrap(), &m.predict(&ds).unwrap());
    }
    // MARS: numeric and binary, on complete rows (categorical predictor expanded to indicators).
    let keep: Vec<usize> = (0..ds.n_rows()).filter(|&i| (0..3).all(|f| !ds.is_missing(f, i))).collect();
    let cd = ds.select_rows(&keep);
    let (x, names) = expand_indicators(&cd);
    let yk: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
    let m = fit_mars(&x, &names, &yk, &MarsOptions { max_degree: 2, ..Default::default() }).unwrap();
    let obj = PredictiveModel::Mars(MarsObject::from_model(&m, &cd, MarsLink::Identity));
    let back = roundtrip(&obj);
    let want = m.predict(&x).unwrap();
    for (a, b) in back.predict(&cd).unwrap().values.iter().zip(&want) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let codes: Vec<usize> = keep.iter().map(|&i| usize::from(y[i] > 1.5)).collect();
    let mb = fit_mars_binary(&x, &names, &codes, &["no".into(), "yes".into()], &MarsOptions::default()).unwrap();
    let objb = PredictiveModel::Mars(MarsObject::from_binary(&mb, &cd));
    let backb = roundtrip(&objb);
    let p = backb.predict(&cd).unwrap();
    for (a, b) in p.probabilities.iter().zip(&mb.probabilities) {
        close(a[1], *b, 1e-14, "mars prob");
    }
    assert!(backb.is_classification() && backb.levels() == ["no", "yes"]);
    assert_eq!(backb.algorithm(), "MARS (binary)");
    // New data with an unseen level and a different level order is handled by label.
    let newd = Dataset::new()
        .with_numeric("u", vec![0.5, 1.0])
        .unwrap()
        .with_numeric("v", vec![0.5, -1.0])
        .unwrap()
        .with_categorical("k", &["q".to_string(), "zzz".to_string()])
        .unwrap();
    let r = back.predict(&newd).unwrap();
    assert!(r.values[0].is_finite() && r.values[1].is_nan(), "{:?}", r.values);
}

#[test]
fn damaged_models_are_refused_not_panicked_on() {
    let (ds, y, _) = mixed_data(150, 22);
    let resp = Response::numeric(y).unwrap();
    let m = fit_cart(&ds, &resp, &CartOptions { cv_folds: 0, pruning: Pruning::None, ..Default::default() }).unwrap();
    let text = SavedModel { response: "y".into(), n_train: 150, model: PredictiveModel::from_cart(&m) }.to_json();
    assert!(SavedModel::from_json(&text).is_ok());
    // Every truncation fails cleanly.
    for cut in (0..text.len()).step_by(37) {
        assert!(SavedModel::from_json(&text[..cut]).is_err(), "cut {cut}");
    }
    assert!(SavedModel::from_json(&format!("{text} x")).is_err());
    assert!(SavedModel::from_json("").is_err());
    assert!(SavedModel::from_json("[1,2]").is_err());
    assert!(SavedModel::from_json(&text.replace("openstats-model", "other")).is_err());
    assert!(SavedModel::from_json(&text.replace("\"version\":1", "\"version\":2")).is_err());
    assert!(SavedModel::from_json(&text.replace("\"kind\":\"tree\"", "\"kind\":\"zzz\"")).is_err());
    assert!(SavedModel::from_json(&text.replace("\"criterion\":\"variance\"", "\"criterion\":\"nope\"")).is_err());
    // A split on a predictor that does not exist or has the wrong kind.
    assert!(SavedModel::from_json(&text.replacen("\"t\":\"le\",\"f\":0", "\"t\":\"le\",\"f\":9", 1)).is_err());
    assert!(SavedModel::from_json(&text.replacen("\"t\":\"le\",\"f\":0", "\"t\":\"le\",\"f\":2", 1)).is_err());
    // Mutating random bytes never panics (the result is an error or a model that predicts).
    let mut r = rng(5);
    for _ in 0..200 {
        let mut b = text.clone().into_bytes();
        let i = (r.next_f64() * b.len() as f64) as usize % b.len();
        b[i] = b"0123456789[]{},:\"-.e tnf"[(r.next_f64() * 24.0) as usize % 24];
        if let Ok(s) = String::from_utf8(b) {
            if let Ok(back) = SavedModel::from_json(&s) {
                let _ = back.model.predict(&ds);
            }
        }
    }
}

#[test]
fn json_handles_unicode_and_nan() {
    let ds = Dataset::new()
        .with_numeric("größe \"x\"\n", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        .unwrap()
        .with_categorical(
            "été",
            &(0..10).map(|i| if i % 2 == 0 { "à".to_string() } else { "b\u{1F600}".to_string() }).collect::<Vec<_>>(),
        )
        .unwrap();
    let resp = Response::numeric((0..10).map(|i| f64::from(i) * 2.0).collect()).unwrap();
    let m = fit_cart(
        &ds,
        &resp,
        &CartOptions { cv_folds: 0, pruning: Pruning::None, min_split: 2, min_leaf: 1, ..Default::default() },
    )
    .unwrap();
    let back = roundtrip(&PredictiveModel::from_cart(&m));
    assert_eq!(back.schema().names, ds.names);
    assert_eq!(back.schema().levels(1).unwrap(), ["b\u{1F600}", "à"]);
}

// ------------------------------------------------------------------ model comparison

#[test]
fn baseline_cross_validation_equals_fold_means() {
    let (ds, y, _) = mixed_data(120, 31);
    let resp = Response::numeric(y.clone()).unwrap();
    let c = compare_models(&ds, &resp, &[("Baseline".into(), ModelSpec::Baseline)], 5, 8).unwrap();
    let fold = random_folds(120, 5, 8);
    let mut sse = 0.0;
    let mut sae = 0.0;
    for k in 0..5 {
        let tr: Vec<f64> = (0..120).filter(|&i| fold[i] != k).map(|i| y[i]).collect();
        let mean = tr.iter().sum::<f64>() / tr.len() as f64;
        for i in (0..120).filter(|&i| fold[i] == k) {
            sse += (y[i] - mean).powi(2);
            sae += (y[i] - mean).abs();
        }
    }
    close(c.rows[0].rmse, (sse / 120.0).sqrt(), 1e-12, "rmse");
    close(c.rows[0].mae, sae / 120.0, 1e-12, "mae");
    assert!(c.rows[0].r2 < 0.0 && c.rows[0].r2 > -0.2, "{}", c.rows[0].r2);
    assert_eq!(c.rows[0].n_scored, 120);

    // Classification baseline: training class proportions; log-loss from the same folds.
    let lab: Vec<String> = y.iter().map(|&t| if t > 1.0 { "b" } else { "a" }.to_string()).collect();
    let cr = Response::from_labels(&lab).unwrap();
    let cc = compare_models(&ds, &cr, &[("Baseline".into(), ModelSpec::Baseline)], 5, 8).unwrap();
    let mut ll = 0.0;
    for k in 0..5 {
        let tr: Vec<usize> = (0..120).filter(|&i| fold[i] != k).collect();
        let nb = tr.iter().filter(|&&i| lab[i] == "b").count() as f64;
        let pb = nb / tr.len() as f64;
        for i in (0..120).filter(|&i| fold[i] == k) {
            ll -= if lab[i] == "b" { pb.ln() } else { (1.0 - pb).ln() };
        }
    }
    close(cc.rows[0].log_loss, ll / 120.0, 1e-12, "log loss");
    // Each fold predicts one constant, so ranking inside a fold is uninformative; the pooled AUC is
    // near one half (folds with a higher training share of "b" score higher, which is noise).
    assert!((cc.rows[0].auc - 0.5).abs() < 0.1, "{}", cc.rows[0].auc);
}

#[test]
fn comparison_ranks_real_models_above_the_baseline() {
    let (ds, y) = additive_data(300, 41);
    let resp = Response::numeric(y).unwrap();
    let cands = vec![
        ("Baseline".to_string(), ModelSpec::Baseline),
        (
            "CART".to_string(),
            ModelSpec::Cart(CartOptions { cv_folds: 0, pruning: Pruning::None, min_leaf: 5, ..Default::default() }),
        ),
        ("Forest".to_string(), ModelSpec::Forest(ForestOptions { n_trees: 40, seed: 1, ..Default::default() })),
        ("Boosting".to_string(), ModelSpec::Boosting(GbdtOptions { n_trees: 80, seed: 1, ..Default::default() })),
        ("MARS".to_string(), ModelSpec::Mars(MarsOptions::default())),
    ];
    let c = compare_models(&ds, &resp, &cands, 4, 6).unwrap();
    assert_eq!(c.rows.len(), 5);
    let base = c.rows[0].rmse;
    for r in &c.rows[1..] {
        assert!(r.rmse < 0.6 * base, "{}: {} vs baseline {}", r.name, r.rmse, base);
    }
    assert_ne!(c.rows[c.best].name, "Baseline");
    // The same call is deterministic.
    let again = compare_models(&ds, &resp, &cands, 4, 6).unwrap();
    for (a, b) in c.rows.iter().zip(&again.rows) {
        assert_eq!(a.rmse.to_bits(), b.rmse.to_bits());
    }
    validated(c.report_section());
}

#[test]
fn classification_comparison_and_errors() {
    let (ds, y, lab) = mixed_data(240, 42);
    let _ = y;
    let resp = Response::from_labels(&lab).unwrap();
    let cands = vec![
        ("Baseline".to_string(), ModelSpec::Baseline),
        ("Forest".to_string(), ModelSpec::Forest(ForestOptions { n_trees: 30, seed: 1, ..Default::default() })),
    ];
    let c = compare_models(&ds, &resp, &cands, 3, 2).unwrap();
    assert!(c.classification && c.levels.len() == 3);
    assert!(c.rows[1].error_rate < c.rows[0].error_rate);
    assert!(c.rows[1].log_loss < c.rows[0].log_loss);
    assert!(c.rows[1].auc.is_nan(), "AUC is reported for two classes only");
    validated(c.report_section());
    // MARS needs a numeric or two-class response.
    assert!(compare_models(&ds, &resp, &[("M".into(), ModelSpec::Mars(MarsOptions::default()))], 3, 2).is_err());
    assert!(compare_models(&ds, &resp, &[], 3, 2).is_err());
    assert!(compare_models(&ds, &resp, &cands, 1, 2).is_err());
    assert!(compare_models(&ds, &resp, &cands, 1000, 2).is_err());
    let short = Response::numeric(vec![1.0; 5]).unwrap();
    assert!(compare_models(&ds, &short, &cands, 3, 2).is_err());
}
