//! Cox regression against an independent brute-force partial-likelihood maximiser.

mod common;
use common::cox_data::*;
use openstats_stats::model::{Data, DesignOptions};
use openstats_stats::reliability::{cox_formula, cox_ph, CoxOptions, CoxTies};
use openstats_stats::report::{Reportable, SessionReport};

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn run(ties: CoxTies, strata: bool) -> openstats_stats::reliability::CoxFit {
    let event: Vec<bool> = COX_D.iter().map(|&d| d > 0.5).collect();
    let st: Vec<usize> = COX_G.iter().map(|&g| g as usize).collect();
    cox_ph(
        COX_T,
        &event,
        &[COX_X1.to_vec(), COX_X2.to_vec()],
        &["x1".to_string(), "x2".to_string()],
        if strata { Some(&st) } else { None },
        &CoxOptions { ties, ..CoxOptions::default() },
    )
    .unwrap()
}

#[test]
fn efron_and_breslow_match_brute_force() {
    let cases = [
        (CoxTies::Efron, false, COX_EFRON_PLAIN_BETA, COX_EFRON_PLAIN_SE, COX_EFRON_PLAIN_LL, COX_EFRON_PLAIN_LL0),
        (CoxTies::Efron, true, COX_EFRON_STRATA_BETA, COX_EFRON_STRATA_SE, COX_EFRON_STRATA_LL, COX_EFRON_STRATA_LL0),
        (
            CoxTies::Breslow,
            false,
            COX_BRESLOW_PLAIN_BETA,
            COX_BRESLOW_PLAIN_SE,
            COX_BRESLOW_PLAIN_LL,
            COX_BRESLOW_PLAIN_LL0,
        ),
        (
            CoxTies::Breslow,
            true,
            COX_BRESLOW_STRATA_BETA,
            COX_BRESLOW_STRATA_SE,
            COX_BRESLOW_STRATA_LL,
            COX_BRESLOW_STRATA_LL0,
        ),
    ];
    for (ties, strata, beta, se, ll, ll0) in cases {
        let f = run(ties, strata);
        assert!(f.converged, "{ties:?} {strata}");
        for j in 0..2 {
            close(f.coefficients[j].coef, beta[j], 1e-6, "beta");
            close(f.coefficients[j].se, se[j], 1e-4, "se");
            close(f.coefficients[j].hazard_ratio, beta[j].exp(), 1e-6, "hr");
        }
        close(f.loglik, ll, 1e-9, "loglik");
        close(f.loglik_null, ll0, 1e-9, "null loglik");
        close(f.lr, 2.0 * (ll - ll0), 1e-8, "LR");
        assert_eq!(f.n_strata, if strata { 3 } else { 1 });
        assert!(f.concordance > 0.5 && f.concordance < 1.0);
        assert!(f.p_lr < 1.0 && f.p_wald < 1.0 && f.p_score < 1.0);
    }
}

#[test]
fn formula_interface_codes_factors_and_drops_missing() {
    let g: Vec<&str> = COX_X2.iter().map(|&v| if v > 0.5 { "b" } else { "a" }).collect();
    let data = Data::new().with_numeric("x1", COX_X1.to_vec()).unwrap().with_factor("grp", &g).unwrap();
    let mut time = COX_T.to_vec();
    time[3] = f64::NAN;
    let event: Vec<bool> = COX_D.iter().map(|&d| d > 0.5).collect();
    let f =
        cox_formula("x1 + grp", &data, &time, &event, None, &DesignOptions::default(), &CoxOptions::default()).unwrap();
    assert_eq!(f.n, 69);
    assert_eq!(f.coefficients.len(), 2);
    assert!(f.coefficients[1].name.contains("grp"), "{}", f.coefficients[1].name);
    // The same model on complete data reproduces the numeric fit (treatment coding of a 0/1 factor).
    let full =
        cox_formula("x1 + grp", &data, COX_T, &event, None, &DesignOptions::default(), &CoxOptions::default()).unwrap();
    let want = run(CoxTies::Efron, false);
    close(full.coefficients[0].coef, want.coefficients[0].coef, 1e-8, "x1");
    close(full.coefficients[1].coef, want.coefficients[1].coef, 1e-8, "grp");
    let mut rep = SessionReport::new("cox");
    rep.push(full.report_section());
    rep.validate().unwrap();
    assert!(rep.render_text().contains("Hazard Ratio"));
}

#[test]
fn errors() {
    let ev = [true, false, true];
    let o = CoxOptions::default();
    assert!(cox_ph(&[1.0, 2.0, 3.0], &ev, &[vec![1.0, 2.0, 3.0]], &["x".into()], None, &o).is_err()); // events <= p+? two events, one coef
    assert!(cox_ph(&[1.0, 2.0], &ev, &[vec![1.0, 2.0, 3.0]], &["x".into()], None, &o).is_err());
    assert!(cox_ph(&[1.0, 2.0, 3.0], &ev, &[], &[], None, &o).is_err());
}
