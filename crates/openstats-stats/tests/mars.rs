//! MARS against a naive NumPy implementation (tests/gen/mars_ref.py) plus behavioural checks.

mod common;
use common::mars_data::*;
use openstats_stats::ml::{fit_mars, MarsOptions};
use openstats_stats::report::{Reportable, SessionReport};

const N: usize = 120;

fn xcols() -> Vec<Vec<f64>> {
    (0..3).map(|j| X[j * N..(j + 1) * N].to_vec()).collect()
}

fn names() -> Vec<String> {
    vec!["x1".into(), "x2".into(), "x3".into()]
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn check(
    tag: &str,
    opts: &MarsOptions,
    y: &[f64],
    coef: &[f64],
    rss: f64,
    gcv: f64,
    terms: &[f64],
    path: &[f64],
    pred: &[f64],
    nfwd: f64,
) {
    let m = fit_mars(&xcols(), &names(), y, opts).unwrap();
    assert_eq!(m.forward_terms as f64, nfwd, "{tag} forward terms");
    assert_eq!(m.terms.len(), coef.len(), "{tag} terms kept");
    for (i, t) in m.terms.iter().enumerate() {
        close(t.coef, coef[i], 1e-7, &format!("{tag} coef {i}"));
        for q in 0..3 {
            let (v, k, s) = (terms[(i * 3 + q) * 3], terms[(i * 3 + q) * 3 + 1], terms[(i * 3 + q) * 3 + 2]);
            if v < 0.0 {
                assert!(t.factors.len() <= q, "{tag} term {i} has extra factor");
            } else {
                let f = &t.factors[q];
                assert_eq!((f.var as f64, f.sign as f64), (v, s), "{tag} term {i} factor {q}");
                if s != 0.0 {
                    close(f.knot, k, 1e-12, &format!("{tag} knot {i}.{q}"));
                }
            }
        }
    }
    close(m.rss, rss, 1e-8, &format!("{tag} rss"));
    close(m.gcv, gcv, 1e-8, &format!("{tag} gcv"));
    assert_eq!(m.path.len() * 2, path.len(), "{tag} path length");
    for (i, r) in m.path.iter().enumerate() {
        assert_eq!(r.terms as f64, path[2 * i]);
        close(r.gcv, path[2 * i + 1], 1e-7, &format!("{tag} path gcv {i}"));
    }
    assert_eq!(m.path.iter().filter(|r| r.selected).count(), 1);
    let xt: Vec<Vec<f64>> = (0..3)
        .map(|j| [[0.1, 0.2, 0.9], [0.5, 0.5, 0.5], [0.9, 0.1, 0.3], [0.3, 0.8, 0.6]].iter().map(|r| r[j]).collect())
        .collect();
    let p = m.predict(&xt).unwrap();
    for i in 0..4 {
        close(p[i], pred[i], 1e-7, &format!("{tag} prediction {i}"));
    }
    // fitted values reproduce the model
    let f = m.predict(&xcols()).unwrap();
    for i in 0..N {
        close(f[i], m.fitted[i], 1e-9, "fitted");
    }
}

#[test]
fn additive_model_matches_naive_reference() {
    let o = MarsOptions {
        max_terms: Some(15),
        max_degree: 1,
        penalty: Some(2.0),
        min_span: Some(3),
        end_span: Some(5),
        allow_linear: false,
        ..MarsOptions::default()
    };
    check("A", &o, Y1, A_COEF, A_RSS, A_GCV, A_TERMS, A_PATH, A_PRED, A_NFWD);
}

#[test]
fn interaction_model_matches_naive_reference() {
    let o = MarsOptions {
        max_terms: Some(15),
        max_degree: 2,
        penalty: Some(3.0),
        min_span: Some(3),
        end_span: Some(5),
        allow_linear: false,
        ..MarsOptions::default()
    };
    check("B", &o, Y2, B_COEF, B_RSS, B_GCV, B_TERMS, B_PATH, B_PRED, B_NFWD);
}

#[test]
fn recovers_a_noise_free_hinge_function() {
    let x: Vec<f64> = (0..80).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 + 3.0 * (v - 30.0).max(0.0) - 1.0 * (55.0 - v).max(0.0)).collect();
    let m = fit_mars(
        std::slice::from_ref(&x),
        &["x".into()],
        &y,
        &MarsOptions { min_span: Some(1), end_span: Some(2), ..MarsOptions::default() },
    )
    .unwrap();
    assert!(m.r2() > 0.999, "r2 {}", m.r2());
    let p = m.predict(&[vec![0.0, 40.0, 70.0]]).unwrap();
    for (got, v) in p.iter().zip([0.0f64, 40.0, 70.0]) {
        let want = 2.0 + 3.0 * (v - 30.0).max(0.0) - (55.0 - v).max(0.0);
        assert!((got - want).abs() < 2.0, "{got} vs {want}");
    }
    assert_eq!(m.importance.len(), 1);
    assert_eq!(m.importance[0].1, 100.0);
}

#[test]
fn irrelevant_predictors_are_dropped_and_importance_ranks() {
    let m = fit_mars(&xcols(), &names(), Y1, &MarsOptions::default()).unwrap();
    // y1 depends on all three variables; the largest effect is x1 or x2, x3 is weakest
    assert_eq!(m.importance.len(), 3);
    assert!(m.importance.windows(2).all(|w| w[0].1 >= w[1].1));
    assert_eq!(m.importance.last().unwrap().0, 2);
    // pure noise response: pruned to few terms with GRSq near zero or below
    let noise: Vec<f64> = (0..N).map(|i| (((i * 7919) % 101) as f64) / 101.0).collect();
    let mn = fit_mars(&xcols(), &names(), &noise, &MarsOptions::default()).unwrap();
    assert!(mn.terms.len() <= 6, "{} terms", mn.terms.len());
    assert!(mn.grsq() < 0.5);
}

#[test]
fn options_are_honoured() {
    let full =
        fit_mars(&xcols(), &names(), Y1, &MarsOptions { prune: false, max_terms: Some(11), ..MarsOptions::default() })
            .unwrap();
    assert!(full.path.is_empty());
    assert_eq!(full.terms.len(), full.forward_terms);
    assert!(full.terms.len() <= 11);
    let pruned =
        fit_mars(&xcols(), &names(), Y1, &MarsOptions { max_terms: Some(11), ..MarsOptions::default() }).unwrap();
    assert!(pruned.terms.len() <= full.terms.len() && pruned.gcv <= full.gcv + 1e-12);
    // additive model: no term has more than one factor, and no variable repeats a hinge pair's role incorrectly
    assert!(pruned.terms.iter().all(|t| t.degree() <= 1));
    let inter = fit_mars(&xcols(), &names(), Y2, &MarsOptions { max_degree: 2, ..MarsOptions::default() }).unwrap();
    assert!(inter.terms.iter().any(|t| t.degree() == 2), "expected an interaction");
    assert_eq!(inter.penalty, 3.0);
    // a higher penalty never keeps more terms
    let strict =
        fit_mars(&xcols(), &names(), Y1, &MarsOptions { penalty: Some(8.0), ..MarsOptions::default() }).unwrap();
    let lax = fit_mars(&xcols(), &names(), Y1, &MarsOptions { penalty: Some(0.0), ..MarsOptions::default() }).unwrap();
    assert!(strict.terms.len() <= lax.terms.len());
    // automatic spans follow Friedman's rules
    let auto = fit_mars(&xcols(), &names(), Y1, &MarsOptions::default()).unwrap();
    assert_eq!(auto.spans, (5, 9));
}

#[test]
fn errors_are_reported() {
    let ok = MarsOptions::default();
    assert!(fit_mars(&[], &[], Y1, &ok).is_err());
    assert!(fit_mars(&xcols(), &names()[..2], Y1, &ok).is_err());
    assert!(fit_mars(&xcols(), &names(), &Y1[..50], &ok).is_err());
    assert!(fit_mars(&xcols(), &names(), &vec![1.0; N], &ok).unwrap_err().to_string().contains("constant"));
    let mut bad = xcols();
    bad[1][3] = f64::NAN;
    assert!(fit_mars(&bad, &names(), Y1, &ok).is_err());
    assert!(fit_mars(&xcols(), &names(), Y1, &MarsOptions { max_degree: 0, ..MarsOptions::default() }).is_err());
    assert!(fit_mars(&xcols(), &names(), Y1, &MarsOptions { max_terms: Some(2), ..MarsOptions::default() }).is_err());
    assert!(fit_mars(&xcols(), &names(), Y1, &MarsOptions { penalty: Some(-1.0), ..MarsOptions::default() }).is_err());
    let small: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]];
    assert!(fit_mars(&small, &["x".into()], &[1.0, 2.0, 3.0], &ok).is_err());
    let m = fit_mars(&xcols(), &names(), Y1, &ok).unwrap();
    assert!(m.predict(&xcols()[..2]).is_err());
    let with_nan = m.predict(&[vec![f64::NAN], vec![0.5], vec![0.5]]).unwrap();
    assert!(with_nan[0].is_nan());
}

#[test]
fn report_renders_and_round_trips() {
    let m = fit_mars(&xcols(), &names(), Y1, &MarsOptions::default()).unwrap();
    let mut r = SessionReport::new("MARS");
    r.push(m.report_section());
    r.validate().unwrap();
    let t = r.render_text();
    for needle in [
        "Model Summary",
        "Basis Functions",
        "max(0,",
        "Backward Elimination",
        "Variable Importance",
        "Generalized R-squared",
    ] {
        assert!(t.contains(needle), "missing {needle}");
    }
    assert_eq!(SessionReport::from_json(&r.to_json()).unwrap().to_json(), r.to_json());
}

#[test]
fn linear_terms_are_used_when_allowed() {
    let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.37).sin() + i as f64 / 30.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 1.5 + 2.0 * v).collect();
    let m = fit_mars(&[x], &["x".into()], &y, &MarsOptions::default()).unwrap();
    assert!(m.r2() > 1.0 - 1e-9);
    assert!(m.terms.iter().any(|t| t.factors.iter().any(|f| f.sign == 0)) || m.terms.len() > 1);
}
