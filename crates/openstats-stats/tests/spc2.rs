//! Phase 2g quality tools — Western Electric/Nelson rule sets, staged limits, I-MR-R/S, zone,
//! Z-MR, Box–Cox charts, Laney P′/U′, G/T charts, MA/EWMA/CUSUM (tabular and V-mask) and Hotelling T² —
//! checked against Montgomery's CUSUM/EWMA example (data transcription verified in
//! `gen/stats_ref7.py`) and against independent NumPy/SciPy computations (`common/data7.rs`).

mod common;
use common::data7::*;
use openstats_math::linalg::Matrix;
use openstats_stats::report::{Reportable, SessionReport};
use openstats_stats::spc::attributes::*;
use openstats_stats::spc::charts::*;
use openstats_stats::spc::multivariate::*;
use openstats_stats::spc::rules::*;
use openstats_stats::spc::stages::*;
use openstats_stats::spc::time_weighted::*;
use openstats_stats::spc::transform::LambdaChoice;
use openstats_stats::spc::variables_ext::*;

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let ok = if got == want {
        true
    } else if want == 0.0 {
        got.abs() <= rel
    } else {
        ((got - want) / want).abs() <= rel
    };
    assert!(ok, "{what}: got {got:e}, want {want:e}");
}

fn close_all(got: &[f64], want: &[f64], rel: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, rel, &format!("{what}[{i}]"));
    }
}

fn same_json(r: &SessionReport) {
    r.validate().unwrap();
    assert_eq!(SessionReport::from_json(&r.to_json()).unwrap().to_json(), r.to_json());
    assert!(!r.render_text().is_empty());
}

fn tw(values: &[f64], sizes: &[f64], target: f64, sigma: f64) -> TimeWeightedInput {
    TimeWeightedInput {
        values: values.to_vec(),
        sizes: sizes.iter().map(|&s| s as usize).collect(),
        target,
        sigma,
        sigma_method: "known".into(),
    }
}

fn groups(g: &[&[f64]]) -> Vec<Vec<f64>> {
    g.iter().map(|x| x.to_vec()).collect()
}

fn rows(g: &[&[f64]]) -> Vec<Vec<f64>> {
    groups(g)
}

// ------------------------------------------------------------ rule sets

#[test]
fn western_electric_rules_flag_the_documented_patterns() {
    // 4 of 5 beyond 1σ on one side completes at the fourth qualifying point.
    let z = [1.2, 0.3, 1.4, 1.1, 1.3];
    let v = evaluate(&z, &SpecialCauseTests::western_electric()).unwrap();
    assert_eq!(v.len(), 1);
    assert_eq!((v[0].test, v[0].points.clone()), (6, vec![4]));
    // 2 of 3 beyond 2σ
    let v = evaluate(&[2.5, 0.1, 2.2], &SpecialCauseTests::western_electric()).unwrap();
    assert_eq!((v[0].test, v[0].points.clone()), (5, vec![2]));
    // 3σ
    let v = evaluate(&[0.0, -3.4], &SpecialCauseTests::western_electric()).unwrap();
    assert_eq!((v[0].test, v[0].points.clone()), (1, vec![1]));
    // the supplementary set adds trends and stratification
    let trend: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
    let v = evaluate(&trend, &SpecialCauseTests::western_electric_supplementary()).unwrap();
    assert!(v.iter().any(|t| t.test == 3));
    let strat = vec![0.2; 15];
    let v = evaluate(&strat, &SpecialCauseTests::western_electric_supplementary()).unwrap();
    assert!(v.iter().any(|t| t.test == 7));
    assert!(evaluate(&strat, &SpecialCauseTests::western_electric()).unwrap().iter().all(|t| t.test != 7));
}

#[test]
fn rule_thresholds_are_configurable() {
    let t = SpecialCauseTests { beyond_sigma: Some(2.5), same_side: Some(5), ..SpecialCauseTests::none() };
    let v = evaluate(&[0.1, 0.2, 0.3, 0.4, 0.5, 2.6], &t).unwrap();
    assert_eq!(v.iter().find(|x| x.test == 2).unwrap().points, vec![4, 5]);
    assert_eq!(v.iter().find(|x| x.test == 1).unwrap().points, vec![5]);
    assert!(evaluate(&[0.0], &SpecialCauseTests { same_side: Some(1), ..SpecialCauseTests::none() }).is_err());
    let nelson = SpecialCauseTests::nelson();
    assert_eq!(nelson, SpecialCauseTests::all());
}

#[test]
fn charts_accept_the_western_electric_preset() {
    let data: Vec<f64> = (0..30).map(|i| 10.0 + ((i * 7) % 11) as f64 * 0.1).collect();
    let opts = ChartOptions { tests: SpecialCauseTests::western_electric(), ..ChartOptions::default() };
    let c = i_mr(&data, &opts).unwrap();
    assert_eq!(c.location.tests, SpecialCauseTests::western_electric());
}

// ------------------------------------------------------------ staged limits

#[test]
fn staged_xbar_matches_separate_charts() {
    let g = groups(BW_GROUPS);
    let labels: Vec<&str> = (0..g.len()).map(|i| if i < 8 { "before" } else { "after" }).collect();
    let st = staged_xbar(&g, &labels, true, &ChartOptions::default()).unwrap();
    let a = xbar_r(&g[..8], &ChartOptions::default()).unwrap();
    let b = xbar_r(&g[8..], &ChartOptions::default()).unwrap();
    assert_eq!(st.stages.len(), 2);
    close(st.location.ucl[0], a.location.ucl[0], 1e-15, "stage 1 UCL");
    close(st.location.ucl[8], b.location.ucl[0], 1e-15, "stage 2 UCL");
    close(st.dispersion.center[15], b.dispersion.center[0], 1e-15, "stage 2 R centre");
    assert_eq!(st.location.values.len(), g.len());
    same_json(&st.to_report("Staged"));
}

#[test]
fn staged_attribute_charts_step_at_each_stage() {
    let d = LP_D;
    let n = LP_N;
    let labels: Vec<&str> = (0..d.len()).map(|i| if i < 10 { "s1" } else { "s2" }).collect();
    let st = staged_p(d, n, &labels, &ChartOptions::default()).unwrap();
    let a = p_chart(&d[..10], &n[..10], &ChartOptions::default()).unwrap();
    let b = p_chart(&d[10..], &n[10..], &ChartOptions::default()).unwrap();
    close(st.panel.center[0], a.center, 1e-15, "centre 1");
    close(st.panel.center[19], b.center, 1e-15, "centre 2");
    close(st.panel.ucl[12], b.panel.ucl[2], 1e-15, "ucl 2");
    let su = staged_u(
        LU_C,
        LU_N,
        &(0..LU_C.len()).map(|i| (i / 6).to_string()).collect::<Vec<_>>(),
        &ChartOptions::default(),
    )
    .unwrap();
    assert_eq!(su.stages.len(), 3);
    let sc = staged_c(
        &[3.0, 4.0, 2.0, 5.0, 9.0, 11.0, 10.0, 12.0],
        &["a", "a", "a", "a", "b", "b", "b", "b"],
        &ChartOptions::default(),
    )
    .unwrap();
    assert!(sc.panel.center[0] < sc.panel.center[7]);
    let snp = staged_np(
        &[3.0, 4.0, 2.0, 5.0, 9.0, 11.0, 10.0, 12.0],
        50.0,
        &[1, 1, 1, 1, 2, 2, 2, 2].map(|x: i32| x.to_string()),
        &ChartOptions::default(),
    )
    .unwrap();
    assert_eq!(snp.charts.len(), 2);
    same_json(&st.to_report("Staged"));
    assert!(staged_p(d, n, &["a"], &ChartOptions::default()).is_err());
}

// ------------------------------------------------------------ I-MR-R/S

#[test]
fn between_within_matches_reference() {
    let g = groups(BW_GROUPS);
    for (tag, w, r) in [("R", WithinChart::Range, BW_REF_R), ("S", WithinChart::StDev, BW_REF_S)] {
        let c = i_mr_r_s(&g, w, &ChartOptions::default()).unwrap();
        close(c.mean, r[0], 1e-12, &format!("{tag} mean"));
        close(c.sigma_within, r[1], 1e-11, &format!("{tag} σ within"));
        close(c.sigma_between, r[2], 1e-9, &format!("{tag} σ between"));
        close(c.means.sigma[0], r[3], 1e-9, &format!("{tag} se"));
        close(c.means.ucl[3], r[4], 1e-10, &format!("{tag} UCL"));
        close(c.means.lcl[3], r[5], 1e-10, &format!("{tag} LCL"));
        close(c.moving_range.center[2], r[6], 1e-9, &format!("{tag} MR centre"));
        close(c.moving_range.ucl[2], r[7], 1e-8, &format!("{tag} MR UCL"));
        close(c.moving_range.lcl[2], r[8], 1e-8, &format!("{tag} MR LCL"));
        close(c.within.center[0], r[9], 1e-10, &format!("{tag} within centre"));
        close(c.within.ucl[0], r[10], 1e-8, &format!("{tag} within UCL"));
        close(c.within.lcl[0], r[11], 1e-8, &format!("{tag} within LCL"));
        same_json(&c.to_report(tag));
    }
}

#[test]
fn between_within_errors() {
    let g = groups(BW_GROUPS);
    assert!(i_mr_r_s(&g[..2], WithinChart::Range, &ChartOptions::default()).is_err());
    assert!(i_mr_r_s(&[vec![1.0], vec![2.0], vec![3.0]], WithinChart::Range, &ChartOptions::default()).is_err());
}

// ------------------------------------------------------------ zone chart

#[test]
fn zone_chart_matches_reference_loop() {
    let c = zone_chart_individuals(ZONE_DATA, &ChartOptions::default(), &ZoneOptions::default()).unwrap();
    close(c.panel.center[0], ZONE_SIGMA[0], 1e-12, "mean");
    close(c.panel.sigma[0], ZONE_SIGMA[1], 1e-11, "sigma");
    close_all(&c.scores, ZONE_SCORES, 1e-12, "scores");
    let want: Vec<usize> = ZONE_SIGNALS.iter().map(|&x| x as usize).collect();
    assert_eq!(c.signals, want);
    same_json(&c.to_report("Zone"));
    let g = zone_chart_subgroups(&groups(BW_GROUPS), &ChartOptions::default(), &ZoneOptions::default()).unwrap();
    assert_eq!(g.scores.len(), 16);
    assert!(zone_chart_individuals(
        ZONE_DATA,
        &ChartOptions::default(),
        &ZoneOptions { weights: [0.0, -1.0, 4.0, 8.0], signal_at: 8.0 }
    )
    .is_err());
}

// ------------------------------------------------------------ Z-MR

#[test]
fn z_mr_matches_reference() {
    let p = ZMR_PARTS;
    let pooled = z_mr(ZMR_DATA, p, None, &ShortRunSigma::Pooled, &ChartOptions::default()).unwrap();
    assert_eq!(pooled.parts, vec!["A", "B", "C"]);
    close_all(&pooled.targets, &ZMR_REF_POOLED[..3], 1e-12, "targets");
    close_all(&pooled.sigmas, &[ZMR_REF_POOLED[3]; 3], 1e-11, "pooled σ");
    close_all(&pooled.z, &ZMR_REF_POOLED[4..], 1e-11, "z");
    assert_eq!(pooled.chart.location.ucl[0], 3.0);
    assert_eq!(pooled.chart.location.center[0], 0.0);
    let bp = z_mr(ZMR_DATA, p, None, &ShortRunSigma::ByPart, &ChartOptions::default()).unwrap();
    close_all(&bp.sigmas, &ZMR_REF_BYPART[..3], 1e-11, "σ by part");
    close_all(&bp.z, &ZMR_REF_BYPART[3..], 1e-11, "z by part");
    same_json(&bp.to_report("Z-MR"));
}

#[test]
fn z_mr_known_parameters_and_errors() {
    let p = ZMR_PARTS;
    let c = z_mr(
        ZMR_DATA,
        p,
        Some(&[50.0, 120.0, 8.0]),
        &ShortRunSigma::Known(vec![1.0, 2.4, 0.3]),
        &ChartOptions::default(),
    )
    .unwrap();
    close(c.z[0], (ZMR_DATA[0] - 50.0) / 1.0, 1e-14, "z0");
    close(c.z[1], (ZMR_DATA[1] - 120.0) / 2.4, 1e-14, "z1");
    assert!(z_mr(ZMR_DATA, &p[..5], None, &ShortRunSigma::Pooled, &ChartOptions::default()).is_err());
    assert!(z_mr(ZMR_DATA, p, Some(&[1.0]), &ShortRunSigma::Pooled, &ChartOptions::default()).is_err());
    assert!(z_mr(ZMR_DATA, p, None, &ShortRunSigma::Known(vec![1.0]), &ChartOptions::default()).is_err());
    assert!(z_mr(&[1.0, 1.0, 1.0, 1.0], &["a", "a", "b", "b"], None, &ShortRunSigma::Pooled, &ChartOptions::default())
        .is_err());
}

// ------------------------------------------------------------ Box-Cox charts

#[test]
fn boxcox_i_mr_matches_scipy() {
    let c = boxcox_i_mr(BC_DATA, LambdaChoice::Optimal, &ChartOptions::default()).unwrap();
    assert!((c.transform.lambda - BC_REF[0]).abs() < 1e-5, "λ {} vs {}", c.transform.lambda, BC_REF[0]);
    close(c.chart.mean, BC_REF[1], 1e-4, "mean of transformed");
    close(c.chart.sigma, BC_REF[2], 1e-4, "σ of transformed");
    close(c.center[0], BC_REF[3], 1e-4, "centre, original units");
    close(c.ucl[0], BC_REF[4], 1e-4, "UCL, original units");
    close(c.lcl[0], BC_REF[5], 1e-4, "LCL, original units");
    same_json(&c.to_report("Box-Cox"));
    let fixed = boxcox_i_mr(BC_DATA, LambdaChoice::Fixed(0.0), &ChartOptions::default()).unwrap();
    assert_eq!(fixed.transform.lambda, 0.0);
    assert!(boxcox_i_mr(&[1.0, -2.0, 3.0, 4.0], LambdaChoice::Optimal, &ChartOptions::default()).is_err());
}

#[test]
fn boxcox_xbar_runs_on_subgroups() {
    let g: Vec<Vec<f64>> = BC_DATA.chunks(5).map(<[f64]>::to_vec).collect();
    let r = boxcox_xbar(&g, true, LambdaChoice::Optimal, &ChartOptions::default()).unwrap();
    let s = boxcox_xbar(&g, false, LambdaChoice::Rounded, &ChartOptions::default()).unwrap();
    assert_eq!(r.chart.location.values.len(), 8);
    assert_eq!(s.transform.lambda, (s.transform.optimal_lambda * 2.0).round() / 2.0);
    assert!(r.ucl[0] > r.center[0] && r.center[0] > r.lcl[0]);
}

// ------------------------------------------------------------ Laney

#[test]
fn laney_p_matches_reference() {
    let l = laney_p(LP_D, LP_N, &ChartOptions::default()).unwrap();
    close(l.chart.center, LP_REF[0], 1e-14, "p-bar");
    close(l.sigma_z, LP_REF[1], 1e-11, "σ_z");
    close_all(&l.z, &LP_REF[2..22], 1e-11, "z");
    close_all(&l.chart.panel.ucl, &LP_REF[22..42], 1e-11, "UCL");
    close_all(&l.chart.panel.lcl, &LP_REF[42..62], 1e-10, "LCL");
    same_json(&l.to_report("Laney P'"));
}

#[test]
fn laney_u_matches_reference() {
    let l = laney_u(LU_C, LU_N, &ChartOptions::default()).unwrap();
    close(l.chart.center, LU_REF[0], 1e-14, "u-bar");
    close(l.sigma_z, LU_REF[1], 1e-11, "σ_z");
    close_all(&l.z, &LU_REF[2..20], 1e-11, "z");
    close_all(&l.chart.panel.ucl, &LU_REF[20..38], 1e-11, "UCL");
    close_all(&l.chart.panel.lcl, &LU_REF[38..56], 1e-10, "LCL");
    same_json(&l.to_report("Laney U'"));
}

#[test]
fn laney_rejects_degenerate_input() {
    assert!(laney_p(&[0.0, 0.0, 0.0], &[10.0, 10.0, 10.0], &ChartOptions::default()).is_err());
    assert!(laney_p(&[1.0, 2.0], &[10.0, 10.0], &ChartOptions::default()).is_err());
    assert!(laney_u(&[3.0, 4.0, 5.0], &[1.0, 1.0], &ChartOptions::default()).is_err());
    assert!(laney_p(&[11.0, 2.0, 3.0], &[10.0, 10.0, 10.0], &ChartOptions::default()).is_err());
}

// ------------------------------------------------------------ G and T

#[test]
fn g_chart_matches_reference() {
    let c = g_chart(G_DATA, &ChartOptions::default()).unwrap();
    close(c.param1, G_REF[0], 1e-14, "p");
    close(c.panel.lcl[0], G_REF[1], 1e-12, "LCL");
    close(c.panel.center[0], G_REF[2], 1e-12, "median");
    close(c.panel.ucl[0], G_REF[3], 1e-12, "UCL");
    // SciPy's integer-valued geometric quantiles bracket the continuous limits.
    assert_eq!((c.panel.ucl[0] + 1.0).ceil(), G_REF[4]);
    assert_eq!((c.panel.lcl[0] + 1.0).ceil().max(1.0), G_REF[5]);
    assert_eq!((c.panel.center[0] + 1.0).ceil(), G_REF[6]);
    assert!(c.panel.flagged().is_empty()); // the largest count, 120, is inside the limit of about 232.7
    let big = g_chart(&[3.0, 5.0, 4.0, 6.0, 2.0, 5.0, 4.0, 3.0, 150.0], &ChartOptions::default()).unwrap();
    assert_eq!(big.panel.flagged(), vec![8]);
    same_json(&c.to_report("G"));
}

#[test]
fn t_chart_matches_reference() {
    let c = t_chart(T_DATA, &ChartOptions::default()).unwrap();
    close(c.param1, T_REF[0], 1e-9, "shape");
    close(c.param2, T_REF[1], 1e-9, "scale");
    close(c.panel.lcl[0], T_REF[2], 1e-8, "LCL");
    close(c.panel.center[0], T_REF[3], 1e-9, "median");
    close(c.panel.ucl[0], T_REF[4], 1e-9, "UCL");
    close(c.panel.ucl[0], T_REF[5], 1e-9, "UCL (SciPy)");
    same_json(&c.to_report("T"));
    assert!(t_chart(&[1.0, -2.0, 3.0], &ChartOptions::default()).is_err());
}

// ------------------------------------------------------------ CUSUM / EWMA / MA

#[test]
fn montgomery_cusum_example() {
    let inp = tw(MONT_X, &[1.0; 30], 10.0, 1.0);
    let c = cusum_tabular(&inp, &CusumOptions { k: 0.5, h: 5.0, head_start: 0.0 }).unwrap();
    close_all(&c.upper, MONT_CP, 1e-12, "C+");
    close_all(&c.lower, MONT_CM, 1e-12, "C-");
    let np: Vec<usize> = MONT_NP.iter().map(|&x| x as usize).collect();
    let nm: Vec<usize> = MONT_NM.iter().map(|&x| x as usize).collect();
    assert_eq!(c.n_upper, np);
    assert_eq!(c.n_lower, nm);
    // published: the upper CUSUM exceeds H = 5 at observation 29 with C+ = 5.28
    assert_eq!(c.signals_upper[0], 28);
    assert!((c.upper[28] - 5.28).abs() < 0.005);
    assert!(c.signals_lower.is_empty());
    let (cp, _, h) = c.in_data_units().unwrap();
    assert_eq!((cp[28], h), (c.upper[28], 5.0));
    same_json(&c.to_report("CUSUM"));
}

#[test]
fn vmask_equals_tabular_on_montgomery_and_random_data() {
    for data in [MONT_X.to_vec(), TW_MEANS.to_vec()] {
        let n = data.len();
        let ones = vec![1.0; n];
        let inp = tw(&data, &ones, if n == 30 { 10.0 } else { 5.0 }, 1.0);
        let tab = cusum_tabular(&inp, &CusumOptions { k: 0.5, h: 5.0, head_start: 0.0 }).unwrap();
        let vm = cusum_vmask(&inp, 0.5, 5.0).unwrap();
        assert_eq!(vm.signals_upper, tab.signals_upper);
        assert_eq!(vm.signals_lower, tab.signals_lower);
    }
    let vm = cusum_vmask(&tw(MONT_X, &[1.0; 30], 10.0, 1.0), 0.5, 5.0).unwrap();
    assert_eq!(vm.lead_distance, 10.0);
    close(vm.half_angle, 0.5f64.atan(), 1e-15, "angle");
    same_json(&vm.to_report("V-mask"));
}

#[test]
fn montgomery_ewma_example() {
    let inp = tw(MONT_X, &[1.0; 30], 10.0, 1.0);
    let c = ewma(&inp, 0.1, 2.7, true).unwrap();
    close_all(&c.panel.values, &MONT_EWMA[..30], 1e-12, "EWMA");
    close_all(&c.panel.ucl, &MONT_EWMA[30..60], 1e-12, "UCL");
    close_all(&c.panel.lcl, &MONT_EWMA[60..], 1e-12, "LCL");
    assert_eq!(c.signals[0], 28);
    let a = ewma(&inp, 0.1, 2.7, false).unwrap();
    close_all(&a.panel.ucl, MONT_EWMA_ASYM_UCL, 1e-12, "asymptotic UCL");
    same_json(&c.to_report("EWMA"));
}

#[test]
fn unequal_subgroup_sizes_cusum_ewma_ma() {
    let inp = tw(TW_MEANS, TW_SIZES, 5.0, 0.4);
    let c = cusum_tabular(&inp, &CusumOptions { k: 0.5, h: 4.0, head_start: 1.0 }).unwrap();
    close_all(&c.upper, &TW_CUSUM[..24], 1e-12, "C+");
    close_all(&c.lower, &TW_CUSUM[24..], 1e-12, "C-");
    assert!(c.in_data_units().is_err());
    let e = ewma(&inp, 0.25, 3.0, true).unwrap();
    close_all(&e.panel.values, &TW_EWMA[..24], 1e-12, "EWMA");
    close_all(&e.panel.sigma, &TW_EWMA[24..], 1e-12, "EWMA σ");
    let m = moving_average(&inp, 4, 3.0).unwrap();
    close_all(&m.panel.values, &TW_MA[..24], 1e-12, "MA");
    close_all(&m.panel.sigma, &TW_MA[24..], 1e-12, "MA σ");
    same_json(&m.to_report("MA"));
}

#[test]
fn time_weighted_inputs_estimate_sigma() {
    let i = TimeWeightedInput::individuals(MONT_X, None, None).unwrap();
    assert_eq!(i.sizes, vec![1; 30]);
    assert!(i.sigma > 0.0 && (i.target - 10.2).abs() < 0.3);
    let g = TimeWeightedInput::subgroups(&groups(BW_GROUPS), None, None, None).unwrap();
    assert_eq!(g.sizes, vec![4; 16]);
    assert!(g.sigma_method.contains("pooled"));
    let g2 = TimeWeightedInput::subgroups(&groups(BW_GROUPS), Some(50.0), Some(1.2), None).unwrap();
    assert_eq!((g2.target, g2.sigma), (50.0, 1.2));
    assert!(TimeWeightedInput::individuals(&[], None, None).is_err());
}

// ------------------------------------------------------------ Hotelling T²

#[test]
fn hotelling_individuals_match_reference() {
    let d = rows(HT_DATA);
    let c = hotelling_individuals(&d, &HotellingOptions::default()).unwrap();
    close(c.ucl, HT_REF[0], 1e-10, "Phase I UCL");
    close_all(&c.t2, &HT_REF[3..43], 1e-9, "T²");
    assert!(c.flagged().contains(&27));
    let sd = hotelling_individuals(
        &d,
        &HotellingOptions { covariance: CovarianceEstimate::SuccessiveDifferences, ..HotellingOptions::default() },
    )
    .unwrap();
    close(sd.ucl, HT_REF[1], 1e-10, "successive-difference UCL");
    close_all(&sd.t2, &HT_REF[43..], 1e-9, "successive-difference T²");
    same_json(&c.to_report("T2"));
}

#[test]
fn hotelling_known_and_historical_parameters() {
    let d = rows(HT_DATA);
    let mean = HT_MU0S0[..3].to_vec();
    let cov = Matrix::from_vec(3, 3, HT_MU0S0[3..].to_vec()).unwrap();
    let hist = hotelling_individuals(
        &d,
        &HotellingOptions {
            reference: HotellingReference::Historical { mean: mean.clone(), covariance: cov.clone(), m: 60 },
            ..HotellingOptions::default()
        },
    )
    .unwrap();
    close(hist.ucl, HT_REF_KNOWN[0], 1e-10, "Phase II UCL");
    close_all(&hist.t2, &HT_REF_KNOWN[1..], 1e-9, "T² about known parameters");
    let known = hotelling_individuals(
        &d,
        &HotellingOptions {
            reference: HotellingReference::Known { mean, covariance: cov },
            ..HotellingOptions::default()
        },
    )
    .unwrap();
    close(known.ucl, HT_REF[2], 1e-10, "χ² UCL");
    assert!(known.limit_description.contains("χ²"));
}

#[test]
fn hotelling_subgroups_match_reference() {
    let all = rows(HTS_DATA);
    let g: Vec<Vec<Vec<f64>>> = all.chunks(4).map(<[Vec<f64>]>::to_vec).collect();
    let c = hotelling_subgroups(&g, &HotellingOptions::default()).unwrap();
    close(c.ucl, HTS_REF[0], 1e-10, "Phase I UCL");
    close_all(&c.t2, &HTS_REF[2..], 1e-9, "T²");
    close_all(&c.mean, HTS_GRAND, 1e-12, "grand mean");
    close_all(c.covariance.as_slice(), HTS_COV, 1e-11, "pooled covariance");
    assert!(c.flagged().contains(&9));
    let p2 = hotelling_subgroups(
        &g,
        &HotellingOptions {
            reference: HotellingReference::Historical {
                mean: HTS_GRAND.to_vec(),
                covariance: Matrix::from_vec(3, 3, HTS_COV.to_vec()).unwrap(),
                m: 12,
            },
            ..HotellingOptions::default()
        },
    )
    .unwrap();
    close(p2.ucl, HTS_REF[1], 1e-10, "Phase II UCL");
    same_json(&c.to_report("T2 subgroups"));
}

#[test]
fn hotelling_errors() {
    let d = rows(HT_DATA);
    assert!(hotelling_individuals(&d[..4], &HotellingOptions::default()).is_err());
    assert!(hotelling_individuals(&d, &HotellingOptions { alpha: 0.0, ..HotellingOptions::default() }).is_err());
    let bad = HotellingOptions {
        reference: HotellingReference::Known { mean: vec![0.0, 0.0], covariance: Matrix::identity(2) },
        ..HotellingOptions::default()
    };
    assert!(hotelling_individuals(&d, &bad).is_err());
    let singular = HotellingOptions {
        reference: HotellingReference::Known { mean: vec![0.0; 3], covariance: Matrix::zeros(3, 3) },
        ..HotellingOptions::default()
    };
    assert!(hotelling_individuals(&d, &singular).is_err());
    let g = vec![vec![vec![1.0, 2.0], vec![2.0, 1.0]], vec![vec![1.0, 2.0]]];
    assert!(hotelling_subgroups(&g, &HotellingOptions::default()).is_err());
}
