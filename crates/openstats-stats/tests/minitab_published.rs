//! Checks against output printed by Minitab itself: the support.minitab.com examples and the Penn State
//! (STAT 200/415/500) course notes, which show Minitab session output. Inputs are the summary statistics or
//! counts printed with each example, so agreement is limited by the rounding of the published figures; the
//! tolerance on every comparison is half a unit of the last printed digit (plus a little for inputs that were
//! themselves rounded). No Minitab installation was used.

use openstats_stats::crosstab::{likelihood_ratio_chi_square, pearson_chi_square, Crosstab};
use openstats_stats::hypothesis::{Alternative, VarianceAssumption};
use openstats_stats::proportion::one_proportion;
use openstats_stats::ttest::*;
use Alternative::*;

fn near(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: got {got}, Minitab prints {want} (tol {tol})");
}

// ---- 1-sample t (summarized data) --------------------------------------------------------------

#[test]
fn one_sample_t_minitab_support_example_1() {
    // support.minitab.com 1-Sample t example: N 25, mean 330.6, StDev 154.2, SE 30.8, 95% CI (266.9, 394.2), H0 mu = 200
    let r = one_sample_t_summary(25, 330.6, 154.2, 200.0, TwoSided, 0.95).unwrap();
    // the printed mean and StDev are rounded, so allow 0.1 on the limits
    near(r.se, 30.8, 0.05, "SE Mean");
    near(r.ci.lower, 266.9, 0.1, "CI lower");
    near(r.ci.upper, 394.2, 0.1, "CI upper");
}

#[test]
fn one_sample_t_psu_examples() {
    // (n, mean, sd, mu0, SE, CI lower, CI upper) as printed by Minitab
    let cases = [
        (35, 76.8, 11.62, 72.0, 1.96, 72.81, 80.79),
        (226, 3.2311, 0.5104, 3.0, 0.0340, 3.1642, 3.2980),
        (525, 67.009, 4.462, 66.0, 0.195, 66.627, 67.392),
        (30, 28.000, 4.400, 27.0, 0.803, 26.357, 29.643),
        (25, 14.300, 2.367, 15.0, 0.473, 13.323, 15.277),
    ];
    for (n, m, s, mu0, se, lo, hi) in cases {
        let r = one_sample_t_summary(n, m, s, mu0, TwoSided, 0.95).unwrap();
        near(r.se, se, 0.5e-3_f64.max(se * 0.006), "SE");
        near(r.ci.lower, lo, 0.6e-3_f64.max(lo.abs() * 1e-4), &format!("n={n} lower"));
        near(r.ci.upper, hi, 0.6e-3_f64.max(hi.abs() * 1e-4), &format!("n={n} upper"));
    }
    let r = one_sample_t_summary(35, 76.8, 11.62, 72.0, TwoSided, 0.95).unwrap();
    near(r.t, 2.44, 0.005, "T-Value");
}

#[test]
fn one_sample_t_one_sided_lower_bound() {
    // PSU STAT 500: test of mu = 10 vs > 10, N 40, mean 11.000, StDev 3.000, SE 0.474, 95% lower bound 10.201, T 2.11, P 0.021
    let r = one_sample_t_summary(40, 11.0, 3.0, 10.0, Greater, 0.95).unwrap();
    near(r.se, 0.474, 0.0005, "SE");
    near(r.ci.lower, 10.201, 0.0006, "lower bound");
    near(r.t, 2.11, 0.005, "T");
    near(r.p_value, 0.021, 0.0006, "P");
}

#[test]
fn paired_t_psu_sat_example() {
    // Paired t: SATM, SATV, n 215, mean difference 19.49, StDev 89.81, SE 6.12, 95% CI (7.42, 31.56), T 3.18, P 0.002
    let r = paired_t_summary(215, 19.49, 89.81, 0.0, TwoSided, 0.95).unwrap();
    near(r.se, 6.12, 0.005, "SE");
    near(r.ci.lower, 7.42, 0.006, "lower");
    near(r.ci.upper, 31.56, 0.006, "upper");
    near(r.t, 3.18, 0.005, "T");
    near(r.p_value, 0.002, 0.0006, "P");
}

#[test]
fn one_sample_z_psu_example() {
    // N 25, mean 80.94, SE 2.32 (sigma 11.6), 95% upper bound 84.76, Z -1.75, P 0.040 (test of mu = 85 vs < 85)
    let r = z_test_summary(25, 80.94, 85.0, 11.6, Less, 0.95).unwrap();
    near(r.se, 2.32, 0.005, "SE");
    near(r.ci.upper, 84.76, 0.006, "upper bound");
    near(r.z, -1.75, 0.005, "Z");
    near(r.p_value, 0.040, 0.0006, "P");
}

// ---- 2-sample t (summarized data) --------------------------------------------------------------

#[test]
fn two_sample_t_welch_psu_examples() {
    // STAT 200: N 45 mean 140.0 sd 20.0 vs N 40 mean 150.0 sd 25.0; H1: mu1 - mu2 < 0.
    // Minitab: difference -10.00, 95% upper bound -1.75, T -2.02, DF 74, P 0.024
    let r =
        two_sample_t_summary((45, 140.0, 20.0), (40, 150.0, 25.0), 0.0, VarianceAssumption::Welch, Less, 0.95).unwrap();
    near(r.estimate, -10.0, 1e-9, "difference");
    near(r.ci.upper, -1.75, 0.006, "upper bound");
    near(r.t, -2.02, 0.005, "T");
    assert_eq!(r.df.floor(), 74.0, "DF {}", r.df);
    near(r.p_value, 0.024, 0.0006, "P");
    // Height by sex: 126 65.62 6.53 vs 99 70.24 3.63 (less): T -6.73, DF 202
    let r = two_sample_t_summary((126, 65.62, 6.53), (99, 70.24, 3.63), 0.0, VarianceAssumption::Welch, Less, 0.95)
        .unwrap();
    near(r.t, -6.73, 0.006, "T");
    assert_eq!(r.df.floor(), 202.0, "DF {}", r.df);
    assert!(r.p_value < 0.0005);
    // Spiders: 10 10.26 2.51 vs 10 9.02 1.90 → estimate 1.240, 95% CI (-0.852, 3.332) (printed under "pooled": t(18) · 0.9955)
    let r = two_sample_t_summary((10, 10.26, 2.51), (10, 9.02, 1.90), 0.0, VarianceAssumption::Pooled, TwoSided, 0.95)
        .unwrap();
    near(r.estimate, 1.240, 1e-9, "estimate");
    near(r.ci.lower, -0.852, 0.006, "lower");
    near(r.ci.upper, 3.332, 0.006, "upper");
}

#[test]
fn two_sample_t_pooled_examples() {
    // 40 45.42 7.68 vs 40 39.15 9.21 (pooled): estimate 6.27, 95% CI (2.49, 10.04), 95% lower bound 3.11, T 3.30, DF 78
    let r = two_sample_t_summary((40, 45.42, 7.68), (40, 39.15, 9.21), 0.0, VarianceAssumption::Pooled, TwoSided, 0.95)
        .unwrap();
    near(r.estimate, 6.27, 1e-9, "estimate");
    near(r.ci.lower, 2.49, 0.01, "lower");
    near(r.ci.upper, 10.04, 0.01, "upper");
    near(r.t, 3.30, 0.01, "T");
    assert_eq!(r.df, 78.0);
    let g = two_sample_t_summary((40, 45.42, 7.68), (40, 39.15, 9.21), 0.0, VarianceAssumption::Pooled, Greater, 0.95)
        .unwrap();
    near(g.ci.lower, 3.11, 0.01, "lower bound");
    // 34 105.5 20.1 vs 29 90.9 12.2 (pooled): T 3.42, DF 61
    let r = two_sample_t_summary((34, 105.5, 20.1), (29, 90.9, 12.2), 0.0, VarianceAssumption::Pooled, TwoSided, 0.95)
        .unwrap();
    near(r.t, 3.42, 0.01, "T");
    assert_eq!(r.df, 61.0);
}

// ---- 1 proportion ----------------------------------------------------------------------------------

#[test]
fn one_proportion_normal_approximation_psu() {
    // (N, events, Sample p, CI lower, CI upper) printed by Minitab with the normal approximation
    let cases = [
        (524, 184, 0.351145, 0.310276, 0.392015),
        (100, 70, 0.700000, 0.610183, 0.789817),
        (226, 99, 0.438053, 0.373368, 0.502738),
    ];
    for (n, x, ph, lo, hi) in cases {
        let r = one_proportion(x, n, 0.5, TwoSided, 0.95, false).unwrap();
        near(r.p_hat, ph, 5e-7, "Sample p");
        near(r.ci_normal.lower, lo, 5e-7, &format!("n={n} lower"));
        near(r.ci_normal.upper, hi, 5e-7, &format!("n={n} upper"));
    }
    // N 226, 99 events, H0 p = 0.5: Z = -1.86, P = 0.0625
    let r = one_proportion(99, 226, 0.5, TwoSided, 0.95, false).unwrap();
    near(r.z, -1.86, 0.005, "Z");
    near(r.p_value_normal, 0.0625, 0.00006, "P");
    // N 129, 37 events, H0 p = 0.4, H1 p < 0.4: 95% upper bound 0.352321
    let r = one_proportion(37, 129, 0.4, Less, 0.95, false).unwrap();
    near(r.p_hat, 0.286822, 5e-7, "Sample p");
    near(r.ci_normal.upper, 0.352321, 5e-7, "upper bound");
}

#[test]
fn one_proportion_minitab_support_example() {
    // support.minitab.com 1 Proportion example: N 1000, Event 87, Sample p 0.087000, 95% CI (0.070617, 0.106130), H0 p = 0.065, P 0.007
    let r = one_proportion(87, 1000, 0.065, TwoSided, 0.95, false).unwrap();
    near(r.p_hat, 0.087, 1e-9, "Sample p");
    // Minitab's exact interval is Blaker's (not Clopper-Pearson, which would be (0.070268, 0.106208))
    near(r.ci_blaker.lower, 0.070617, 5e-7, "Blaker CI lower");
    near(r.ci_blaker.upper, 0.106130, 5e-7, "Blaker CI upper");
    near(r.ci_exact.lower, 0.070268, 5e-7, "Clopper-Pearson lower");
    near(r.p_value_exact, 0.007, 0.0006, "exact P");
}

// ---- chi-square -------------------------------------------------------------------------------------

#[test]
fn chi_square_minitab_support_example() {
    // support.minitab.com Chi-Square Test for Association: machine x shift
    let t = Crosstab::from_counts(vec![vec![48, 47, 48], vec![76, 47, 32], vec![36, 40, 34]]).unwrap();
    let p = pearson_chi_square(&t).unwrap();
    near(p.statistic, 11.788, 0.0006, "Pearson");
    assert_eq!(p.df, 4);
    near(p.p_value, 0.019, 0.0006, "Pearson P");
    let g = likelihood_ratio_chi_square(&t).unwrap();
    near(g.statistic, 11.816, 0.0006, "LR");
    near(g.p_value, 0.019, 0.0006, "LR P");
    let e = t.expected();
    let printed = [[56.08, 46.97, 39.96], [60.78, 50.91, 43.31], [43.14, 36.13, 30.74]];
    let contrib = [[1.1637, 0.0000, 1.6195], [3.8088, 0.2998, 2.9530], [1.1809, 0.4151, 0.3468]];
    let res = t.pearson_residuals();
    for i in 0..3 {
        for j in 0..3 {
            near(e[i][j], printed[i][j], 0.006, "expected");
            near(res[i][j] * res[i][j], contrib[i][j], 0.00006, "contribution");
        }
    }
}

#[test]
fn chi_square_psu_2x2_and_3x2() {
    // Likes tea: [[30, 25], [10, 35]] → Pearson 10.774, LR 11.138, DF 1, P 0.001
    let t = Crosstab::from_counts(vec![vec![30, 25], vec![10, 35]]).unwrap();
    near(pearson_chi_square(&t).unwrap().statistic, 10.774, 0.0006, "Pearson");
    near(likelihood_ratio_chi_square(&t).unwrap().statistic, 11.138, 0.0006, "LR");
    // Seating by cheating: Back 24 8, Front 38 8, Middle 109 39 (expected 24.21 7.79 / 34.81 11.19 / 111.98 36.02); chi-square 0.743
    let t = Crosstab::from_counts(vec![vec![24, 8], vec![38, 8], vec![109, 39]]).unwrap();
    let e = t.expected();
    near(e[0][0], 24.21, 0.006, "expected");
    near(e[1][0], 34.81, 0.006, "expected");
    near(e[2][1], 36.02, 0.006, "expected");
    near(pearson_chi_square(&t).unwrap().statistic, 1.539, 0.0006, "Pearson");
}

// ---- power and sample size ---------------------------------------------------------------------------

#[test]
fn power_two_sample_t_minitab_support_example() {
    // support.minitab.com Power and Sample Size for 2-Sample t: difference 5, StDev 10, alpha 0.05, target power 0.9
    // → sample size 86 per group, actual power 0.903230
    use openstats_stats::power::PowerModel;
    let m = openstats_stats::power::TwoSampleT {
        sigma: 10.0,
        alpha: 0.05,
        alternative: openstats_stats::hypothesis::Alternative::TwoSided,
    };
    assert_eq!(m.sample_size(5.0, 0.9).unwrap(), 86);
    near(m.power(86, 5.0).unwrap(), 0.903230, 5e-7, "actual power");
}

// ---- one-way ANOVA -------------------------------------------------------------------------------------

#[test]
fn one_way_anova_psu_503_cotton_example() {
    // Montgomery Example 3.1 (tensile strength by cotton weight %), Minitab output printed in PSU STAT 503 Lesson 3:
    // Cotton Weight % DF 4 SS 475.76 MS 118.94 F 14.76 P 0.000; Error DF 20 SS 161.20 MS 8.06; Total DF 24 SS 636.96
    use openstats_stats::anova::{one_way_stacked, OneWayOptions};
    let data: [(&str, [f64; 5]); 5] = [
        ("15", [7.0, 7.0, 15.0, 11.0, 9.0]),
        ("20", [12.0, 17.0, 12.0, 18.0, 18.0]),
        ("25", [14.0, 18.0, 18.0, 19.0, 19.0]),
        ("30", [19.0, 25.0, 22.0, 19.0, 23.0]),
        ("35", [7.0, 10.0, 11.0, 15.0, 11.0]),
    ];
    let mut y = Vec::new();
    let mut f = Vec::new();
    for (l, v) in data {
        for x in v {
            y.push(x);
            f.push(l.to_string());
        }
    }
    let a = one_way_stacked(&y, &f, &OneWayOptions::default()).unwrap();
    assert_eq!((a.df_factor, a.df_error), (4, 20));
    near(a.ss_factor, 475.76, 0.006, "SS factor");
    near(a.ms_factor, 118.94, 0.006, "MS factor");
    near(a.ss_error, 161.20, 0.006, "SS error");
    near(a.ms_error, 8.06, 0.006, "MS error");
    near(a.ss_total, 636.96, 0.006, "SS total");
    near(a.f.unwrap(), 14.76, 0.006, "F");
    assert!(a.p_value.unwrap() < 0.0005);
}
