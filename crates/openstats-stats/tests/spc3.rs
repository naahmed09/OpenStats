//! Phase 2h: exact binomial/Poisson limits, one-sided CUSUM, exponential T chart, rules on smoothing
//! charts, staging of every new chart, Hotelling T² decomposition and the generalized-variance chart,
//! checked against independent NumPy/SciPy computations (`common/data8.rs`, `gen/stats_ref8.py`).

mod common;
use common::data7::{HT_DATA, LP_D, LP_N, TW_MEANS};
use common::data8::*;
use openstats_math::linalg::Matrix;
use openstats_stats::report::{Reportable, SessionReport};
use openstats_stats::spc::attributes::*;
use openstats_stats::spc::charts::*;
use openstats_stats::spc::multivariate::*;
use openstats_stats::spc::rules::*;
use openstats_stats::spc::stages::*;
use openstats_stats::spc::time_weighted::*;

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

fn idx(v: &[f64]) -> Vec<usize> {
    v.iter().map(|&x| x as usize).collect()
}

fn rows(g: &[&[f64]]) -> Vec<Vec<f64>> {
    g.iter().map(|x| x.to_vec()).collect()
}

fn unit(values: &[f64]) -> TimeWeightedInput {
    TimeWeightedInput {
        values: values.to_vec(),
        sizes: vec![1; values.len()],
        target: 0.0,
        sigma: 1.0,
        sigma_method: "known".into(),
    }
}

// ------------------------------------------------------------ exact limits

#[test]
fn exact_p_chart_has_stair_step_binomial_limits() {
    let c = p_chart_with(EX_P_D, EX_P_N, &ChartOptions::default(), LimitMethod::Exact).unwrap();
    close(c.center, EX_P_REF[0], 1e-14, "p-bar");
    close_all(&c.panel.lcl, &EX_P_REF[1..23], 1e-12, "LCL");
    close_all(&c.panel.ucl, &EX_P_REF[23..], 1e-12, "UCL");
    assert_eq!(c.panel.flagged(), idx(EX_P_FLAG));
    // limits differ between samples of different size (stair-step)
    assert!(c.panel.ucl.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-9));
    // exact lower limit flags a zero count that the normal LCL (0) cannot
    let normal = p_chart_with(EX_P_D, EX_P_N, &ChartOptions::default(), LimitMethod::Normal).unwrap();
    assert!(!normal.panel.flagged().contains(&14));
    same_json(&c.to_report("exact p"));
}

#[test]
fn exact_np_c_u_charts_match_reference() {
    let np = np_chart_with(&EX_NP[..25], 50.0, &ChartOptions::default(), LimitMethod::Exact).unwrap();
    close(np.center, EX_NP[25], 1e-14, "np centre");
    assert_eq!((np.panel.lcl[0], np.panel.ucl[0]), (EX_NP[26], EX_NP[27]));
    assert_eq!(np.panel.flagged(), idx(EX_NP_FLAG));
    let c = c_chart_with(&EX_C[..30], &ChartOptions::default(), LimitMethod::Exact).unwrap();
    close(c.center, EX_C[30], 1e-14, "c centre");
    assert_eq!((c.panel.lcl[0], c.panel.ucl[0]), (EX_C[31], EX_C[32]));
    assert_eq!(c.panel.flagged(), idx(EX_C_FLAG));
    let u = u_chart_with(EX_U_C, EX_U_N, &ChartOptions::default(), LimitMethod::Exact).unwrap();
    close(u.center, EX_U_REF[0], 1e-14, "u centre");
    close_all(&u.panel.lcl, &EX_U_REF[1..21], 1e-12, "u LCL");
    close_all(&u.panel.ucl, &EX_U_REF[21..], 1e-12, "u UCL");
    assert_eq!(u.panel.flagged(), idx(EX_U_FLAG));
    same_json(&u.to_report("exact u"));
}

#[test]
fn exact_limits_keep_tests_two_to_four_and_k() {
    let opts = ChartOptions { tests: SpecialCauseTests::first_four(), ..ChartOptions::default() };
    let mut counts = vec![3.0; 10];
    counts.extend([1.0; 10]);
    let c = c_chart_with(&counts, &opts, LimitMethod::Exact).unwrap();
    assert!(c.panel.violations.iter().any(|v| v.test == 2), "20 points on one side of the centre");
    // wider k → wider exact limits
    let wide = c_chart_with(&EX_C[..30], &ChartOptions { k_sigma: 4.0, ..ChartOptions::default() }, LimitMethod::Exact)
        .unwrap();
    let base = c_chart_with(&EX_C[..30], &ChartOptions::default(), LimitMethod::Exact).unwrap();
    assert!(wide.panel.ucl[0] >= base.panel.ucl[0]);
    // zero defects: degenerate limits, no panic
    let z = c_chart_with(&[0.0, 0.0, 0.0], &ChartOptions::default(), LimitMethod::Exact).unwrap();
    assert_eq!((z.panel.lcl[0], z.panel.ucl[0]), (0.0, 0.0));
    assert!(p_chart_with(&[1.0], &[5.0, 6.0], &ChartOptions::default(), LimitMethod::Exact).is_err());
}

// ------------------------------------------------------------ CUSUM sides, exponential T

#[test]
fn one_sided_and_fir_cusum_match_reference() {
    let inp = unit(OS_X);
    for (tag, k, h, hs, r) in [("A", 0.5, 4.0, 0.0, OS_A), ("B", 0.75, 3.5, 1.5, OS_B)] {
        let o = CusumOptions { k, h, head_start: hs };
        let both = cusum_tabular_sided(&inp, &o, CusumSides::Both).unwrap();
        close_all(&both.upper, &r[..40], 1e-12, &format!("{tag} C+"));
        close_all(&both.lower, &r[40..], 1e-12, &format!("{tag} C-"));
        let up = cusum_tabular_sided(&inp, &o, CusumSides::UpperOnly).unwrap();
        close_all(&up.upper, &r[..40], 1e-12, &format!("{tag} upper-only"));
        assert!(up.lower.iter().all(|&v| v == 0.0) && up.signals_lower.is_empty());
        assert_eq!(up.signals_upper, both.signals_upper);
        let lo = cusum_tabular_sided(&inp, &o, CusumSides::LowerOnly).unwrap();
        close_all(&lo.lower, &r[40..], 1e-12, &format!("{tag} lower-only"));
        assert!(lo.upper.iter().all(|&v| v == 0.0) && lo.signals_upper.is_empty());
        assert_eq!(lo.signals_lower, both.signals_lower);
    }
    let a = cusum_tabular(&inp, &CusumOptions::default()).unwrap();
    assert!(!a.signals_upper.is_empty() && !a.signals_lower.is_empty());
    assert_eq!(a.sides, CusumSides::Both);
    // FIR makes the chart signal no later than without head start
    let fir = cusum_tabular(&inp, &CusumOptions { head_start: 2.0, ..CusumOptions::default() }).unwrap();
    assert!(fir.signals_upper[0] <= a.signals_upper[0]);
}

#[test]
fn exponential_t_chart_matches_closed_form() {
    let c = t_chart_with(EXP_T, &ChartOptions::default(), TimeModel::Exponential).unwrap();
    close(c.param2, EXP_REF[0], 1e-14, "scale");
    assert_eq!(c.param1, 1.0);
    close(c.panel.lcl[0], EXP_REF[1], 1e-12, "LCL");
    close(c.panel.center[0], EXP_REF[2], 1e-12, "median");
    close(c.panel.ucl[0], EXP_REF[3], 1e-12, "UCL");
    assert_eq!(c.model, "exponential");
    same_json(&c.to_report("T exp"));
    assert!(t_chart_with(&[1.0, -1.0, 2.0], &ChartOptions::default(), TimeModel::Exponential).is_err());
}

// ------------------------------------------------------------ rules on smoothing charts

#[test]
fn nelson_and_western_electric_rules_apply_to_ewma_and_ma() {
    let inp = unit(&[0.2; 12]);
    let we = SpecialCauseTests::western_electric();
    let e = ewma_with_tests(&inp, 0.2, 3.0, true, Some(&we)).unwrap();
    let t2 = e.panel.violations.iter().find(|v| v.test == 2).unwrap();
    assert_eq!(t2.points.first(), Some(&7));
    let m = moving_average_with_tests(&inp, 3, 3.0, Some(&SpecialCauseTests::nelson())).unwrap();
    assert_eq!(m.panel.violations.iter().find(|v| v.test == 2).unwrap().points.first(), Some(&8));
    let nel = ewma_with_tests(&unit(&[0.2; 12]), 0.2, 3.0, true, Some(&SpecialCauseTests::nelson())).unwrap();
    assert_eq!(nel.panel.violations.iter().find(|v| v.test == 2).unwrap().points.first(), Some(&8));
    // default behaviour unchanged: test 1 at k
    let d = ewma(&inp, 0.2, 3.0, true).unwrap();
    assert!(d.panel.violations.is_empty());
}

// ------------------------------------------------------------ staging

#[test]
fn staged_ewma_and_ma_restart_per_stage() {
    let mut data: Vec<f64> = TW_MEANS[..12].to_vec();
    data.extend(TW_MEANS[12..].iter().map(|v| v + 3.0));
    let labels: Vec<&str> = (0..data.len()).map(|i| if i < 12 { "a" } else { "b" }).collect();
    let st = staged_smoothing_individuals(&data, &labels, None, None, |i| ewma(i, 0.2, 3.0, true)).unwrap();
    let a = ewma(&TimeWeightedInput::individuals(&data[..12], None, None).unwrap(), 0.2, 3.0, true).unwrap();
    let b = ewma(&TimeWeightedInput::individuals(&data[12..], None, None).unwrap(), 0.2, 3.0, true).unwrap();
    assert_eq!(st.panel.values[..12], a.panel.values[..]);
    assert_eq!(st.panel.values[12..], b.panel.values[..]);
    assert!(st.panel.center[0] < st.panel.center[12]);
    // a fixed target and σ apply to every stage
    let fixed =
        staged_smoothing_individuals(&data, &labels, Some(5.0), Some(0.4), |i| moving_average(i, 3, 3.0)).unwrap();
    assert!(fixed.panel.center.iter().all(|&c| c == 5.0));
    assert!(!fixed.panel.flagged().is_empty());
    same_json(&st.to_report("staged ewma"));
    let groups: Vec<Vec<f64>> = (0..8).map(|i| vec![1.0 + i as f64 * 0.1, 1.2, 0.9, 1.1 - i as f64 * 0.02]).collect();
    let g =
        staged_smoothing_subgroups(&groups, &[1, 1, 1, 1, 2, 2, 2, 2].map(|x: i32| x.to_string()), None, None, |i| {
            ewma(i, 0.3, 3.0, false)
        })
        .unwrap();
    assert_eq!(g.charts.len(), 2);
    assert!(staged_smoothing_individuals(&data, &["a"], None, None, |i| ewma(i, 0.2, 3.0, true)).is_err());
}

#[test]
fn staged_cusum_restarts_and_offsets_signals() {
    let mut data = vec![0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3];
    data.extend([5.0, 5.2, 4.9, 5.1, 5.3, 4.8, 5.0, 5.2, 7.5, 7.9]);
    let labels: Vec<&str> = (0..data.len()).map(|i| if i < 8 { "x" } else { "y" }).collect();
    let o = CusumOptions { k: 0.5, h: 4.0, head_start: 0.0 };
    let st = staged_cusum_individuals(&data, &labels, None, Some(0.3), &o, CusumSides::Both).unwrap();
    let alone = cusum_tabular(&TimeWeightedInput::individuals(&data[8..], None, Some(0.3)).unwrap(), &o).unwrap();
    assert_eq!(st.upper[8..], alone.upper[..]);
    assert_eq!(st.signals_upper, alone.signals_upper.iter().map(|i| i + 8).collect::<Vec<_>>());
    assert!(!st.signals_upper.is_empty());
    same_json(&st.to_report("staged cusum"));
}

#[test]
fn staged_laney_rare_event_and_hotelling() {
    let labels: Vec<&str> = (0..LP_D.len()).map(|i| if i < 10 { "1" } else { "2" }).collect();
    let l = staged_laney_p(LP_D, LP_N, &labels, &ChartOptions::default()).unwrap();
    let a = laney_p(&LP_D[..10], &LP_N[..10], &ChartOptions::default()).unwrap();
    assert_eq!(l.panel.ucl[..10], a.chart.panel.ucl[..]);
    assert_eq!(l.charts.len(), 2);
    same_json(&l.to_report("staged laney"));
    let lu = staged_laney_u(
        EX_U_C,
        EX_U_N,
        &(0..20).map(|i| (i / 10).to_string()).collect::<Vec<_>>(),
        &ChartOptions::default(),
    )
    .unwrap();
    assert_eq!(lu.stages.len(), 2);

    let g = [12.0, 30.0, 8.0, 45.0, 20.0, 15.0, 60.0, 9.0, 100.0, 140.0, 90.0, 210.0, 160.0, 120.0];
    let gl: Vec<&str> = (0..14).map(|i| if i < 7 { "a" } else { "b" }).collect();
    let sg = staged_g(&g, &gl, &ChartOptions::default()).unwrap();
    assert!(sg.panel.center[0] < sg.panel.center[13]);
    same_json(&sg.to_report("staged g"));
    let t = [5.0, 7.0, 4.0, 6.0, 8.0, 5.5, 6.5, 40.0, 70.0, 45.0, 62.0, 80.0, 55.0, 66.0];
    let st = staged_t(&t, &gl, &ChartOptions::default(), TimeModel::Exponential).unwrap();
    assert_eq!(st.charts.len(), 2);
    same_json(&st.to_report("staged t"));

    let d = rows(HT_DATA);
    let hl: Vec<&str> = (0..d.len()).map(|i| if i < 20 { "p1" } else { "p2" }).collect();
    let sh = staged_hotelling_individuals(&d, &hl, &HotellingOptions::default()).unwrap();
    let h2 = hotelling_individuals(&d[20..], &HotellingOptions::default()).unwrap();
    assert_eq!(sh.ucl[0], sh.charts[0].ucl);
    assert_eq!(sh.ucl[39], h2.ucl);
    assert_eq!(sh.t2[20..], h2.t2[..]);
    assert!(sh.flagged.iter().all(|&i| i < 40));
    same_json(&sh.to_report("staged T2"));
}

// ------------------------------------------------------------ decomposition

#[test]
fn contributions_add_up_and_match_reference() {
    let d = rows(DC_DATA);
    let dec = hotelling_decomposition_individuals(&d, &HotellingOptions::default()).unwrap();
    close_all(&dec.chart.t2, &DC_REF[..30], 1e-9, "T²");
    for i in 0..30 {
        for j in 0..3 {
            close(dec.contributions[(i, j)], DC_REF[30 + i * 3 + j], 1e-8, &format!("contribution {i},{j}"));
        }
        let s: f64 = (0..3).map(|j| dec.contributions[(i, j)]).sum();
        close(s, dec.chart.t2[i], 1e-10, "row sum");
    }
    assert_eq!(dec.unconditional.len(), 3);
    for (j, u) in dec.unconditional.iter().enumerate() {
        close(u.ucl, DC_UNC[0], 1e-10, "unconditional UCL");
        for i in 0..30 {
            close(u.t2[i], DC_UNC[1 + i * 3 + j], 1e-9, &format!("unconditional {i},{j}"));
        }
    }
    assert!(dec.chart.flagged().contains(&19));
    // every variable's unconditional chart is a valid chart on its own
    assert!(dec.unconditional.iter().all(|u| u.p == 1 && u.t2.len() == 30));
    same_json(&dec.to_report("T2 decomposition"));
}

#[test]
fn subgroup_decomposition_matches_reference() {
    let all = rows(DCS_DATA);
    let g: Vec<Vec<Vec<f64>>> = all.chunks(5).map(<[Vec<f64>]>::to_vec).collect();
    let dec = hotelling_decomposition_subgroups(&g, &HotellingOptions::default()).unwrap();
    for i in 0..10 {
        for j in 0..3 {
            close(dec.contributions[(i, j)], DCS_REF[i * 3 + j], 1e-8, &format!("contribution {i},{j}"));
            close(dec.unconditional[j].t2[i], DCS_UNC[i * 3 + j], 1e-9, &format!("unconditional {i},{j}"));
        }
    }
    close(dec.unconditional[0].ucl, DCS_UNC_UCL[0], 1e-10, "unconditional UCL");
    assert!(dec.chart.flagged().contains(&6));
}

#[test]
fn decomposition_with_known_parameters_and_errors() {
    let d = rows(DC_DATA);
    let opts = HotellingOptions {
        reference: HotellingReference::Known {
            mean: vec![4.0, 8.0, 2.0],
            covariance: Matrix::from_rows(&[vec![1.0, 0.6, 0.1], vec![0.6, 1.0, 0.5], vec![0.1, 0.5, 0.7]]).unwrap(),
        },
        ..HotellingOptions::default()
    };
    let dec = hotelling_decomposition_individuals(&d, &opts).unwrap();
    assert!(dec.unconditional.iter().all(|u| u.limit_description.contains("χ²")));
    assert!(dec.chart.contributions(&d[..3]).is_err());
}

// ------------------------------------------------------------ generalized variance

#[test]
fn generalized_variance_constants_match_reference() {
    for (n, p, r) in [(5, 2, GV_B_5_2), (8, 3, GV_B_8_3), (4, 3, GV_B_4_3)] {
        let (b1, b2) = generalized_variance_constants(n, p).unwrap();
        close(b1, r[0], 1e-13, "b1");
        close(b2, r[1], 1e-12, "b2");
    }
    // p = 1: |S| is s², whose mean is σ² (b1 = 1) and variance 2σ⁴/(n−1) (b2 = 2/(n−1))
    let (b1, b2) = generalized_variance_constants(6, 1).unwrap();
    close(b1, 1.0, 1e-14, "b1, p=1");
    close(b2, 2.0 / 5.0, 1e-14, "b2, p=1");
    assert!(generalized_variance_constants(3, 3).is_err());
}

#[test]
fn generalized_variance_chart_matches_reference() {
    let all = rows(GV_DATA);
    let g: Vec<Vec<Vec<f64>>> = all.chunks(6).map(<[Vec<f64>]>::to_vec).collect();
    let c = generalized_variance(&g, None, 3.0).unwrap();
    close(c.sigma_det, GV_REF[0], 1e-10, "|Σ|");
    close(c.center, GV_REF[1], 1e-10, "centre");
    close(c.ucl, GV_REF[2], 1e-10, "UCL");
    close(c.lcl, GV_REF[3], 1e-10, "LCL");
    close_all(&c.det, &GV_REF[4..], 1e-9, "|S|");
    assert!(c.flagged().contains(&11), "the inflated subgroup is flagged");
    let s0 = Matrix::from_vec(3, 3, GV_S0.to_vec()).unwrap();
    let k = generalized_variance(&g, Some(&s0), 3.0).unwrap();
    close(k.sigma_det, GV_KNOWN[0], 1e-12, "known |Σ|");
    close(k.center, GV_KNOWN[1], 1e-12, "known centre");
    close(k.ucl, GV_KNOWN[2], 1e-12, "known UCL");
    close(k.lcl, GV_KNOWN[3], 1e-9, "known LCL");
    same_json(&c.to_report("|S|"));
    assert!(generalized_variance(&g[..1], None, 3.0).is_err());
    assert!(generalized_variance(&g, None, 0.0).is_err());
    assert!(generalized_variance(&g, Some(&Matrix::identity(2)), 3.0).is_err());
    let small: Vec<Vec<Vec<f64>>> = g.iter().map(|x| x[..3].to_vec()).collect();
    assert!(generalized_variance(&small, None, 3.0).is_err());
}
