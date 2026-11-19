//! Reliability and survival analysis against NumPy / SciPy references (tests/gen/rel_ref.py).

mod common;
use common::rel_data::*;
use openstats_stats::reliability::report_impl::{life_section, AltReport, KmGroups, LifeReportOptions};
use openstats_stats::reliability::*;
use openstats_stats::report::{Block, Reportable, SessionReport};

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn mix() -> LifeData {
    let obs = (0..MIX_LO.len()).map(|i| Obs { lower: MIX_LO[i], upper: MIX_UP[i], weight: MIX_W[i], row: i }).collect();
    LifeData { obs, n_skipped: 0 }
}

fn km_data() -> LifeData {
    LifeData::from_times(KM_T, Some(KM_C), 1.0, None).unwrap()
}

fn groups() -> Vec<LifeData> {
    (0..3)
        .map(|k| {
            let idx: Vec<usize> = (0..GRP_G.len()).filter(|&i| GRP_G[i] == k as f64).collect();
            let t: Vec<f64> = idx.iter().map(|&i| GRP_T[i]).collect();
            let c: Vec<f64> = idx.iter().map(|&i| GRP_C[i]).collect();
            LifeData::from_times(&t, Some(&c), 1.0, None).unwrap()
        })
        .collect()
}

struct Ref {
    fam: Family,
    theta: &'static [f64],
    ll: f64,
    cov: &'static [f64],
    pct: &'static [f64],
    mean: f64,
    sd: f64,
    sf: &'static [f64],
    pdf: &'static [f64],
    pct_ci: &'static [f64],
    sf_ci: &'static [f64],
}

fn refs() -> Vec<Ref> {
    vec![
        Ref {
            fam: Family::Weibull,
            theta: WEIBULL_THETA,
            ll: WEIBULL_LL,
            cov: WEIBULL_COV,
            pct: WEIBULL_PCT,
            mean: WEIBULL_MEAN,
            sd: WEIBULL_SD,
            sf: WEIBULL_SF,
            pdf: WEIBULL_PDF,
            pct_ci: WEIBULL_PCT_CI,
            sf_ci: WEIBULL_SF_CI,
        },
        Ref {
            fam: Family::Lognormal,
            theta: LOGNORMAL_THETA,
            ll: LOGNORMAL_LL,
            cov: LOGNORMAL_COV,
            pct: LOGNORMAL_PCT,
            mean: LOGNORMAL_MEAN,
            sd: LOGNORMAL_SD,
            sf: LOGNORMAL_SF,
            pdf: LOGNORMAL_PDF,
            pct_ci: LOGNORMAL_PCT_CI,
            sf_ci: LOGNORMAL_SF_CI,
        },
        Ref {
            fam: Family::Exponential,
            theta: EXPONENTIAL_THETA,
            ll: EXPONENTIAL_LL,
            cov: EXPONENTIAL_COV,
            pct: EXPONENTIAL_PCT,
            mean: EXPONENTIAL_MEAN,
            sd: EXPONENTIAL_SD,
            sf: EXPONENTIAL_SF,
            pdf: EXPONENTIAL_PDF,
            pct_ci: EXPONENTIAL_PCT_CI,
            sf_ci: EXPONENTIAL_SF_CI,
        },
        Ref {
            fam: Family::Normal,
            theta: NORMAL_THETA,
            ll: NORMAL_LL,
            cov: NORMAL_COV,
            pct: NORMAL_PCT,
            mean: NORMAL_MEAN,
            sd: NORMAL_SD,
            sf: NORMAL_SF,
            pdf: NORMAL_PDF,
            pct_ci: NORMAL_PCT_CI,
            sf_ci: NORMAL_SF_CI,
        },
        Ref {
            fam: Family::Logistic,
            theta: LOGISTIC_THETA,
            ll: LOGISTIC_LL,
            cov: LOGISTIC_COV,
            pct: LOGISTIC_PCT,
            mean: LOGISTIC_MEAN,
            sd: LOGISTIC_SD,
            sf: LOGISTIC_SF,
            pdf: LOGISTIC_PDF,
            pct_ci: LOGISTIC_PCT_CI,
            sf_ci: LOGISTIC_SF_CI,
        },
        Ref {
            fam: Family::Loglogistic,
            theta: LOGLOGISTIC_THETA,
            ll: LOGLOGISTIC_LL,
            cov: LOGLOGISTIC_COV,
            pct: LOGLOGISTIC_PCT,
            mean: LOGLOGISTIC_MEAN,
            sd: LOGLOGISTIC_SD,
            sf: LOGLOGISTIC_SF,
            pdf: LOGLOGISTIC_PDF,
            pct_ci: LOGLOGISTIC_PCT_CI,
            sf_ci: LOGLOGISTIC_SF_CI,
        },
    ]
}

#[test]
fn censored_fits_match_scipy_likelihood() {
    let data = mix();
    for r in refs() {
        let name = r.fam.name();
        let fit = fit_distribution(&data, r.fam).unwrap();
        close(fit.loglik, r.ll, 1e-8, &format!("{name} loglik"));
        close(fit.mu, r.theta[0], 1e-4, &format!("{name} mu"));
        if r.fam != Family::Exponential {
            close(fit.sigma.ln(), r.theta[1], 1e-4, &format!("{name} ln sigma"));
        }
        let cov = fit.cov.as_ref().unwrap_or_else(|| panic!("{name}: no covariance"));
        let k = fit.n_params();
        for i in 0..k {
            for j in 0..k {
                close(cov[(i, j)], r.cov[i * k + j], 2e-3, &format!("{name} cov[{i}][{j}]"));
            }
        }
        for (i, p) in [0.01, 0.1, 0.5, 0.9].iter().enumerate() {
            close(fit.percentile(*p), r.pct[i], 1e-5, &format!("{name} pct {p}"));
            let ci = fit.percentile_ci(*p, 0.95);
            close(ci.lower, r.pct_ci[2 * i], 2e-3, &format!("{name} pct lo {p}"));
            close(ci.upper, r.pct_ci[2 * i + 1], 2e-3, &format!("{name} pct hi {p}"));
        }
        if r.mean.is_finite() {
            close(fit.mean(), r.mean, 1e-5, &format!("{name} mean"));
        } else {
            assert!(fit.mean().is_nan());
        }
        if r.sd.is_finite() {
            close(fit.std_dev(), r.sd, 1e-5, &format!("{name} sd"));
        }
        for (i, t) in [20.0, 60.0, 120.0].iter().enumerate() {
            close(fit.survival(*t), r.sf[i], 1e-6, &format!("{name} sf {t}"));
            close(fit.pdf(*t), r.pdf[i], 1e-5, &format!("{name} pdf {t}"));
            let ci = fit.reliability_ci(*t, 0.95);
            close(ci.lower, r.sf_ci[2 * i], 2e-3, &format!("{name} sf lo {t}"));
            close(ci.upper, r.sf_ci[2 * i + 1], 2e-3, &format!("{name} sf hi {t}"));
        }
        // hazard = pdf / survival
        close(fit.hazard(60.0), fit.pdf(60.0) / fit.survival(60.0), 1e-9, &format!("{name} hazard"));
        assert_eq!(fit.failures + fit.censored, fit.n);
    }
}

#[test]
fn parameter_tables_use_minitab_parameterisation() {
    let fit = fit_distribution(&mix(), Family::Weibull).unwrap();
    let p = fit.parameters(0.95);
    assert_eq!(p[0].0, "Shape");
    assert_eq!(p[1].0, "Scale");
    close(p[0].1.est, 1.0 / fit.sigma, 1e-12, "shape");
    close(p[1].1.est, fit.mu.exp(), 1e-12, "scale");
    assert!(p[0].1.lower < p[0].1.est && p[0].1.est < p[0].1.upper);
    assert!(p[1].1.lower < p[1].1.est && p[1].1.est < p[1].1.upper);
    let n = fit_distribution(&mix(), Family::Normal).unwrap().parameters(0.9);
    assert_eq!((n[0].0.as_str(), n[1].0.as_str()), ("Mean", "StDev"));
    assert_eq!(fit_distribution(&mix(), Family::Exponential).unwrap().parameters(0.95)[0].0, "Mean");
}

#[test]
fn exponential_matches_total_time_on_test() {
    let data = LifeData::from_times(EXPO_T, Some(EXPO_CENS), 1.0, None).unwrap();
    let fit = fit_distribution(&data, Family::Exponential).unwrap();
    close(fit.mu.exp(), EXPO_MEAN, 1e-6, "mean");
    assert_eq!(fit.failures, EXPO_NFAIL);
    // Fisher information: var(mean) = mean² / failures.
    let se = fit.parameters(0.95)[0].1.se;
    close(se, EXPO_MEAN / EXPO_NFAIL.sqrt(), 5e-4, "se");
    close(fit.survival(30.0), (-30.0 / EXPO_MEAN).exp(), 1e-6, "survival");
}

#[test]
fn frequencies_equal_repeated_rows() {
    let t = [10.0, 20.0, 35.0, 50.0, 80.0, 80.0];
    let c = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let f = [1.0, 3.0, 2.0, 1.0, 1.0, 1.0];
    let a = LifeData::from_times(&t, Some(&c), 1.0, Some(&f)).unwrap();
    let (mut tt, mut cc) = (vec![], vec![]);
    for i in 0..t.len() {
        for _ in 0..f[i] as usize {
            tt.push(t[i]);
            cc.push(c[i]);
        }
    }
    let b = LifeData::from_times(&tt, Some(&cc), 1.0, None).unwrap();
    let (fa, fb) = (fit_distribution(&a, Family::Weibull).unwrap(), fit_distribution(&b, Family::Weibull).unwrap());
    close(fa.mu, fb.mu, 1e-6, "mu");
    close(fa.sigma, fb.sigma, 1e-6, "sigma");
    close(fa.loglik, fb.loglik, 1e-9, "loglik");
    assert_eq!(a.total(), 9.0);
    // a missing time skips its row
    let d = LifeData::from_times(&[1.0, f64::NAN, 3.0], None, 0.0, None).unwrap();
    assert_eq!((d.obs.len(), d.n_skipped), (2, 1));
}

#[test]
fn interval_data_input_forms() {
    let start = [f64::NAN, 5.0, 8.0, 12.0, 9.0];
    let end = [4.0, 5.0, 10.0, f64::NAN, 9.5];
    let d = LifeData::from_intervals(&start, &end, None).unwrap();
    let kinds: Vec<ObsKind> = d.obs.iter().map(|o| o.kind()).collect();
    assert_eq!(kinds, vec![ObsKind::Left, ObsKind::Exact, ObsKind::Interval, ObsKind::Right, ObsKind::Interval]);
    assert_eq!((d.failures(), d.right_censored()), (4.0, 1.0));
    assert!(LifeData::from_intervals(&[5.0], &[3.0], None).is_err());
}

#[test]
fn kaplan_meier_matches_scipy_ecdf_and_loops() {
    let data = km_data();
    let km = kaplan_meier(&data, 0.95, CiMethod::LogLog).unwrap();
    let ev: Vec<&KmStep> = km.steps.iter().filter(|s| s.failed > 0.0).collect();
    assert!(ev.iter().all(|s| KM_TIMES.contains(&s.time)));
    for s in ev.iter() {
        let i = KM_TIMES.iter().position(|&t| t == s.time).unwrap();
        close(s.survival, KM_SF[i], 1e-12, "survival");
        // scipy leaves the interval undefined where it cannot be computed; compare when finite.
        if KM_LL_LO[i].is_finite() && KM_LL_HI[i].is_finite() && s.survival > 0.0 {
            close(s.lower, KM_LL_LO[i], 1e-8, "loglog lower");
            close(s.upper, KM_LL_HI[i], 1e-8, "loglog upper");
        }
    }
    let plain = kaplan_meier(&data, 0.95, CiMethod::Plain).unwrap();
    for s in plain.steps.iter().filter(|s| s.failed > 0.0) {
        let i = KM_TIMES.iter().position(|&t| t == s.time).unwrap();
        if KM_PL_LO[i].is_finite() && KM_PL_HI[i].is_finite() {
            close(s.lower, KM_PL_LO[i], 1e-8, "plain lower vs scipy");
            close(s.upper, KM_PL_HI[i], 1e-8, "plain upper vs scipy");
        }
    }
    assert_eq!(plain.steps.len(), KM_STEP_T.len());
    for (i, s) in plain.steps.iter().enumerate() {
        close(s.time, KM_STEP_T[i], 1e-12, "step time");
        close(s.at_risk, KM_STEP_N[i], 1e-12, "at risk");
        close(s.failed, KM_STEP_D[i], 1e-12, "failed");
        close(s.survival, KM_STEP_S[i], 1e-12, "S");
        if KM_STEP_SE[i].is_finite() {
            close(s.se, KM_STEP_SE[i], 1e-10, "Greenwood se");
            close(s.lower, (s.survival - 1.959963984540054 * KM_STEP_SE[i]).max(0.0), 1e-9, "plain lower");
        }
        close(s.cum_hazard, KM_STEP_H[i], 1e-12, "Nelson-Aalen");
    }
    close(plain.mean, KM_MEAN, 1e-10, "restricted mean");
    close(plain.mean_se, KM_MEAN_SE, 1e-8, "restricted mean se");
    close(plain.median, KM_MEDIAN, 0.0, "median");
    assert_eq!(plain.survival_at(0.5), 1.0);
    close(plain.survival_at(KM_STEP_T[3]), KM_STEP_S[3], 1e-12, "survival_at");
}

#[test]
fn group_comparisons_match_numpy_and_scipy_logrank() {
    let g = groups();
    for (kind, st, p) in [
        (WeightKind::LogRank, GRP_LR_STAT, GRP_LR_P),
        (WeightKind::Wilcoxon, GRP_WIL_STAT, GRP_WIL_P),
        (WeightKind::TaroneWare, GRP_TW_STAT, GRP_TW_P),
    ] {
        let t = compare_groups(&g, kind).unwrap();
        close(t.statistic, st, 1e-9, "statistic");
        close(t.p_value, p, 1e-9, "p");
        assert_eq!(t.df, 2);
        for k in 0..3 {
            close(t.observed[k], GRP_OBS[k], 1e-12, "observed");
            close(t.expected[k], GRP_EXP[k], 1e-9, "expected");
        }
    }
    // two groups against scipy.stats.logrank (z² = chi-square)
    let idx0: Vec<usize> = (0..GRP2_G.len()).filter(|&i| GRP2_G[i] == 0.0).collect();
    let idx1: Vec<usize> = (0..GRP2_G.len()).filter(|&i| GRP2_G[i] == 1.0).collect();
    let mk = |idx: &[usize]| {
        let t: Vec<f64> = idx.iter().map(|&i| GRP2_T[i]).collect();
        let c: Vec<f64> = idx.iter().map(|&i| GRP2_C[i]).collect();
        LifeData::from_times(&t, Some(&c), 1.0, None).unwrap()
    };
    let t = compare_groups(&[mk(&idx0), mk(&idx1)], WeightKind::LogRank).unwrap();
    close(t.statistic, LR2_Z * LR2_Z, 1e-9, "2-group statistic");
    close(t.p_value, LR2_P, 1e-9, "2-group p");
    assert_eq!(t.df, 1);
}

#[test]
fn accelerated_life_arrhenius_weibull() {
    let data = LifeData::from_times(ALT_T, Some(ALT_C), 1.0, None).unwrap();
    let stress =
        StressVariable { name: "Temp".into(), values: ALT_TEMP.to_vec(), relationship: Relationship::Arrhenius };
    let fit = fit_alt(&data, &[stress], Family::Weibull).unwrap();
    close(fit.loglik, ALT_LL, 1e-8, "loglik");
    close(fit.beta[0], ALT_THETA[0], 1e-3, "b0");
    close(fit.beta[1], ALT_THETA[1], 1e-3, "b1");
    close(fit.sigma.ln(), ALT_THETA[2], 1e-3, "ln sigma");
    let cov = fit.cov.as_ref().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            close(cov[(i, j)], ALT_COV[i * 3 + j], 1e-2, "cov");
        }
    }
    let life = fit.at_stress(&[40.0]).unwrap();
    close(life.mu, ALT_MU_USE, 1e-3, "mu at use");
    for (i, p) in [0.01, 0.1, 0.5].iter().enumerate() {
        close(life.percentile(*p), ALT_USE_PCT[i], 1e-3, "use percentile");
        let ci = life.percentile_ci(*p, 0.95);
        close(ci.lower, ALT_USE_PCT_CI[2 * i], 2e-2, "use pct lo");
        close(ci.upper, ALT_USE_PCT_CI[2 * i + 1], 2e-2, "use pct hi");
    }
    // life is longer at lower stress; acceleration factor consistent with the percentile ratio
    let af = fit.acceleration_factor(&[40.0], &[120.0]).unwrap();
    assert!(af > 1.0);
    let hot = fit.at_stress(&[120.0]).unwrap();
    close(af, life.percentile(0.1) / hot.percentile(0.1), 1e-9, "acceleration factor");
    let coefs = fit.coefficients(0.95);
    assert_eq!(coefs[0].0, "Intercept");
    assert_eq!(coefs[1].0, "Temp");
    assert!(coefs[1].3 < 0.05, "slope should be significant");
}

#[test]
fn warranty_forecast_matches_reference() {
    let mut returns: Vec<Vec<f64>> = Vec::new();
    let mut cur = Vec::new();
    for &v in WAR_RET_FLAT {
        if v < 0.0 {
            returns.push(std::mem::take(&mut cur));
        } else {
            cur.push(v);
        }
    }
    let input = WarrantyInput { shipments: WAR_SHIP.to_vec(), returns, future_shipments: WAR_FUTURE.to_vec() };
    let w = warranty_forecast(&input, Family::Weibull, 8).unwrap();
    close(w.fit.mu, WAR_THETA[0], 1e-4, "mu");
    close(w.fit.sigma.ln(), WAR_THETA[1], 1e-4, "ln sigma");
    for (i, e) in w.expected.iter().enumerate() {
        close(*e, WAR_EXPECTED[i], 1e-4, "expected returns");
    }
    close(w.cumulative[7], WAR_EXPECTED.iter().sum::<f64>(), 1e-4, "cumulative");
    assert_eq!(w.returned + w.in_service, WAR_SHIP.iter().sum::<f64>());
    // validation
    let mut bad = input.clone();
    bad.returns[0][0] = 1e9;
    assert!(warranty_forecast(&bad, Family::Weibull, 4).is_err());
    assert!(warranty_forecast(&input, Family::Normal, 4).is_err());
    assert!(warranty_forecast(&input, Family::Weibull, 0).is_err());
}

#[test]
fn ranking_orders_by_aic() {
    let r = rank_distributions(&mix());
    assert_eq!(r.len(), 6);
    assert!(r.windows(2).all(|w| w[0].aic() <= w[1].aic()));
    // data were generated from a Weibull, which should rank near the top
    assert!(r.iter().position(|f| f.family == Family::Weibull).unwrap() <= 1);
}

#[test]
fn input_errors() {
    // everything censored
    let all = LifeData::from_times(&[5.0, 6.0, 7.0], Some(&[1.0, 1.0, 1.0]), 1.0, None).unwrap();
    assert!(fit_distribution(&all, Family::Weibull).unwrap_err().to_string().contains("failure"));
    // zero failure time for a log family, fine for the normal
    let z = LifeData::from_times(&[0.0, 2.0, 3.0, 4.0], None, 0.0, None).unwrap();
    assert!(fit_distribution(&z, Family::Weibull).is_err());
    assert!(fit_distribution(&z, Family::Normal).is_ok());
    // negative time, mismatched columns, bad frequency
    assert!(LifeData::from_times(&[-1.0], None, 0.0, None).is_err());
    assert!(LifeData::from_times(&[1.0, 2.0], Some(&[0.0]), 0.0, None).is_err());
    assert!(LifeData::from_times(&[1.0, 2.0], None, 0.0, Some(&[1.5, 1.0])).is_err());
    assert!(LifeData::from_times(&[f64::NAN], None, 0.0, None).is_err());
    // Kaplan-Meier refuses arbitrary censoring and bad levels
    let arb = LifeData::from_intervals(&[f64::NAN, 2.0], &[3.0, 4.0], None).unwrap();
    assert!(kaplan_meier(&arb, 0.95, CiMethod::Plain).is_err());
    assert!(kaplan_meier(&km_data(), 1.5, CiMethod::Plain).is_err());
    assert!(compare_groups(&[km_data()], WeightKind::LogRank).is_err());
    // accelerated life: one stress level, missing stress, bad Arrhenius value
    let d = LifeData::from_times(&[5.0, 6.0, 7.0, 9.0], None, 0.0, None).unwrap();
    let sv = |v: Vec<f64>, r| StressVariable { name: "S".into(), values: v, relationship: r };
    assert!(fit_alt(&d, &[sv(vec![1.0; 4], Relationship::Linear)], Family::Weibull)
        .unwrap_err()
        .to_string()
        .contains("two stress levels"));
    assert!(fit_alt(&d, &[sv(vec![1.0, 2.0, f64::NAN, 3.0], Relationship::Linear)], Family::Weibull).is_err());
    assert!(fit_alt(&d, &[sv(vec![-300.0, 20.0, 30.0, 40.0], Relationship::Arrhenius)], Family::Weibull).is_err());
    assert!(fit_alt(&d, &[], Family::Weibull).is_err());
}

#[test]
fn reports_render_and_round_trip() {
    let data = mix();
    let km = kaplan_meier(&km_data(), 0.95, CiMethod::Plain).unwrap();
    let fit = fit_distribution(&data, Family::Weibull).unwrap();
    let opts = LifeReportOptions { times: vec![50.0, 100.0], ..LifeReportOptions::default() };
    let sec = life_section(&fit, &opts, Some(&km), "Distribution Analysis: Weibull");
    let charts = sec.blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count();
    assert_eq!(charts, 3);
    let g = groups();
    let curves: Vec<KaplanMeier> = g.iter().map(|d| kaplan_meier(d, 0.95, CiMethod::Plain).unwrap()).collect();
    let tests =
        vec![compare_groups(&g, WeightKind::LogRank).unwrap(), compare_groups(&g, WeightKind::Wilcoxon).unwrap()];
    let kg = KmGroups { names: vec!["A".into(), "B".into(), "C".into()], curves, tests };
    let at = LifeData::from_times(ALT_T, Some(ALT_C), 1.0, None).unwrap();
    let stress =
        StressVariable { name: "Temp".into(), values: ALT_TEMP.to_vec(), relationship: Relationship::Arrhenius };
    let alt = AltReport {
        fit: fit_alt(&at, &[stress], Family::Weibull).unwrap(),
        level: 0.95,
        use_stress: vec![40.0],
        percents: vec![1.0, 10.0],
        times: vec![1000.0],
    };
    let mut rep = SessionReport::new("Reliability");
    rep.push(sec);
    rep.push(fit.report_section());
    rep.push(km.report_section());
    rep.push(kg.report_section());
    rep.push(alt.report_section());
    rep.validate().unwrap();
    let text = rep.render_text();
    for needle in [
        "Distribution Parameters",
        "Table of Percentiles",
        "Kaplan-Meier Estimates",
        "Comparison of Survival Curves",
        "Regression Table",
        "Shape",
    ] {
        assert!(text.contains(needle), "missing {needle}");
    }
    let back = SessionReport::from_json(&rep.to_json()).unwrap();
    assert_eq!(back.to_json(), rep.to_json());
}
