//! Reliability completion (Phase 5n): repairable systems, demonstration plans, gamma, threshold
//! Weibull, Turnbull and Eyring. References come from `tests/gen/rel2_ref.py`.

mod common;
use common::rel2_data::*;
use openstats_stats::reliability::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

fn systems() -> Vec<System> {
    let ids: Vec<String> = SYS_ID.iter().map(|x| format!("{x}")).collect();
    // end time per row = that system's end
    let ends: Vec<f64> = SYS_ID.iter().map(|&i| SYS_END[i as usize]).collect();
    let mut s = systems_from_rows(&ids, SYS_TIME, Some(&ends)).unwrap();
    // a system with no events is absent from the long format; restore it from the end vector
    for (i, e) in SYS_END.iter().enumerate() {
        if !SYS_ID.contains(&(i as f64)) {
            s.push(System { end: *e, events: vec![] });
        }
    }
    s
}

#[test]
fn mcf_matches_reference() {
    let m = mean_cumulative_function(&systems(), 0.95).unwrap();
    assert_eq!(m.len(), MCF_T.len());
    for (i, p) in m.iter().enumerate() {
        close(p.time, MCF_T[i], 1e-12);
        close(p.mcf, MCF_EST[i], 1e-12);
        close(p.se, MCF_SE[i], 1e-10);
    }
}

#[test]
fn crow_amsaa_matches_reference() {
    let c = crow_amsaa(&systems(), Termination::TimeTruncated, 0.95).unwrap();
    close(c.beta, CROW_BETA, 1e-12);
    close(c.lambda, CROW_LAMBDA, 1e-10);
    close(c.beta_lower, CROW_BLO, 1e-9);
    close(c.beta_upper, CROW_BHI, 1e-9);
    close(c.chi_stat, CROW_CHI, 1e-10);
    close(c.chi_p, CROW_CHIP, 1e-8);
    close(c.laplace_u, CROW_U, 1e-10);
    close(c.laplace_p, CROW_UP, 1e-10);
}

#[test]
fn demonstration_plans_match_scipy() {
    for &(sh, t, r, c, f, tt, n) in DEMO_N {
        let p = plan_sample_size(sh, t, r, c, f, tt).unwrap();
        assert_eq!(p.n, n, "shape {sh}");
    }
    for &(sh, t, r, c, f, n, tt) in DEMO_T {
        let p = plan_test_time(sh, t, r, c, f, n).unwrap();
        close(p.test_time, tt, 1e-8);
    }
}

#[test]
fn gamma_matches_scipy() {
    let d = LifeData::from_times(GAM_T, Some(GAM_C), 1.0, None).unwrap();
    let g = fit_gamma(&d).unwrap();
    close(g.loglik, GAM_LL, 1e-7);
    close(g.shape, GAM_SHAPE, 1e-3);
    close(g.scale, GAM_SCALE, 1e-3);
}

#[test]
fn threshold_weibull_matches_profile() {
    let d = LifeData::from_times(THR_T, None, 1.0, None).unwrap();
    let f = fit_threshold(&d, Family::Weibull, 0.95).unwrap();
    close(f.loglik, THR_LL, 1e-6);
    assert!((f.threshold - THR_GAMMA).abs() < 0.05, "{} vs {}", f.threshold, THR_GAMMA);
}

#[test]
fn turnbull_loglik_matches_numpy_em() {
    let r: Vec<f64> = TB_R.iter().map(|&x| if x < 0.0 { f64::NAN } else { x }).collect();
    let d = LifeData::from_intervals(TB_L, &r, None).unwrap();
    let t = turnbull(&d, 1e-13, 500_000).unwrap();
    close(t.loglik, TB_LL, 1e-6);
    close(t.mass.iter().sum::<f64>(), 1.0, 1e-9);
}

#[test]
fn eyring_matches_direct_mle() {
    let d = LifeData::from_times(EY_T, None, 1.0, None).unwrap();
    let e = fit_eyring(&d, EY_TEMP, Family::Lognormal).unwrap();
    close(e.loglik, EY_LL, 1e-6);
    close(e.alt.beta[0], EY_A, 1e-3);
    close(e.alt.beta[1], EY_B, 1e-3);
    close(e.alt.sigma, EY_SIGMA, 1e-4);
}
