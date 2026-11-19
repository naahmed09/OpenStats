//! Exploratory data analysis and bootstrap (Phase 5l), checked against independent NumPy/SciPy code.

use openstats_stats::bootstrap::{bootstrap, intervals_from_replicates, BootstrapOptions, IntervalMethod};
use openstats_stats::eda::{letter_values, median_polish, resistant_line, rootogram, symmetry_points};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

const LV_X: [f64; 27] = [
    3.12, 4.563, 9.971, 3.724, 6.275, 2.776, 6.024, 5.065, 3.531, 3.326, 5.13, 6.786, 1.566, 2.729, 2.133, 5.211,
    2.591, 2.504, 6.102, 10.141, 1.108, 2.484, 5.477, 4.607, 9.133, 8.125, 4.323,
];
const LV_EXPECT: [(f64, f64, f64); 6] = [
    (14.0, 4.563, 4.563),
    (7.5, 2.7525, 6.063000000000001),
    (4.0, 2.484, 8.125),
    (2.5, 1.8495, 9.552),
    (1.5, 1.3370000000000002, 10.056000000000001),
    (1.0, 1.108, 10.141),
];

#[test]
fn letter_values_match_independent_code() {
    let lv = letter_values(&LV_X).unwrap();
    assert_eq!(lv.len(), LV_EXPECT.len());
    for (l, e) in lv.iter().zip(LV_EXPECT) {
        close(l.depth, e.0, 1e-12);
        close(l.lower, e.1, 1e-12);
        close(l.upper, e.2, 1e-12);
        close(l.mid, (e.1 + e.2) / 2.0, 1e-12);
    }
    let labels: Vec<&str> = lv.iter().map(|l| l.label.as_str()).collect();
    assert_eq!(labels, ["M", "H", "E", "D", "C", "1"]);
    // Tukey's textbook rule on 1..9: hinges 3 and 7, eighths 2 and 8, D at depth 1.5.
    let nine: Vec<f64> = (1..=9).map(f64::from).collect();
    let l = letter_values(&nine).unwrap();
    assert_eq!((l[1].lower, l[1].upper, l[2].lower, l[2].upper), (3.0, 7.0, 2.0, 8.0));
    assert_eq!((l[3].depth, l[3].lower, l[3].upper), (1.5, 1.5, 8.5));
    assert!(letter_values(&[1.0]).is_err());
}

const MP_T: [f64; 20] = [
    14.35, 11.7, 17.3, 11.93, 12.53, 12.52, 8.18, 9.79, 14.05, 8.81, 10.57, 9.94, 11.83, 8.91, 9.54, 10.73, 10.31,
    7.41, 12.69, 6.1,
];

#[test]
fn median_polish_matches_r_algorithm() {
    let mp = median_polish(&MP_T, 4, 5, 10, 0.01).unwrap();
    close(mp.overall, 10.8125, 1e-12);
    for (a, b) in mp.row.iter().zip([3.2575, -0.36749999999999927, 0.3624999999999994, -0.3624999999999994]) {
        close(*a, b, 1e-12);
    }
    for (a, b) in mp.col.iter().zip([0.28000000000000025, -1.75, 0.0, 0.04999999999999982, -1.6349999999999998]) {
        close(*a, b, 1e-12);
    }
    for (a, b) in mp.residuals.iter().zip([
        0.0,
        -0.6200000000000001,
        3.2300000000000013,
        -2.1899999999999995,
        0.09499999999999975,
        1.794999999999999,
        -0.5150000000000006,
        -0.6550000000000011,
        3.5550000000000006,
        0.0,
        -0.8849999999999989,
        0.5150000000000006,
        0.6550000000000011,
        -2.3149999999999986,
        0.0,
        0.0,
        1.6100000000000003,
        -3.04,
        2.1899999999999995,
        -2.7150000000000007,
    ]) {
        close(*a, b, 1e-12);
    }
    assert_eq!(mp.iterations, 3);
    // The fit reproduces the table.
    for i in 0..4 {
        for j in 0..5 {
            close(mp.overall + mp.row[i] + mp.col[j] + mp.residuals[i * 5 + j], MP_T[i * 5 + j], 1e-12);
        }
    }
}

#[test]
fn median_polish_recovers_additive_table() {
    let (rows, cols) = ([1.0, 4.0, -2.0], [0.0, 3.0, 5.0, -1.0]);
    let mut t = Vec::new();
    for r in rows {
        for c in cols {
            t.push(10.0 + r + c);
        }
    }
    let mp = median_polish(&t, 3, 4, 10, 0.01).unwrap();
    assert!(mp.residuals.iter().all(|r| r.abs() < 1e-12));
    close(mp.row[1] - mp.row[0], 3.0, 1e-12);
    close(mp.col[2] - mp.col[0], 5.0, 1e-12);
    assert!(median_polish(&[1.0, f64::NAN, 3.0, 4.0], 2, 2, 10, 0.01).is_err());
    assert!(median_polish(&[1.0, 2.0], 1, 2, 10, 0.01).is_err());
}

#[test]
fn resistant_line_is_exact_and_resists_outliers() {
    let x: Vec<f64> = (1..=15).map(f64::from).collect();
    let mut y: Vec<f64> = x.iter().map(|v| 2.0 + 3.0 * v).collect();
    let fit = resistant_line(&x, &y, 10).unwrap();
    close(fit.slope, 3.0, 1e-12);
    close(fit.intercept, 2.0, 1e-12);
    // Gross outliers in the middle and an end group change nothing material.
    y[7] += 500.0;
    y[1] -= 300.0;
    let fit = resistant_line(&x, &y, 10).unwrap();
    close(fit.slope, 3.0, 1e-9);
    close(fit.intercept, 2.0, 1e-8);
    assert_eq!(fit.residuals.len(), 15);
    assert!(resistant_line(&[1.0, 2.0], &[1.0, 2.0], 5).is_err());
    assert!(resistant_line(&[1.0; 9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 5).is_err());
}

#[test]
fn rootogram_counts_and_symmetry() {
    let bins = rootogram(&LV_X, 5).unwrap();
    assert_eq!(bins.len(), 5);
    assert_eq!(bins.iter().map(|b| b.observed).sum::<f64>(), 27.0);
    close(bins.iter().map(|b| b.expected).sum::<f64>(), 27.0, 1e-12);
    for b in &bins {
        close(b.hanging, b.root_expected - b.root_observed, 1e-12);
        close(b.double_root, (2.0 + 4.0 * b.observed).sqrt() - (1.0 + 4.0 * b.expected).sqrt(), 1e-12);
    }
    assert!(rootogram(&[1.0, 1.0, 1.0, 1.0, 1.0], 3).is_err());
    let sym = symmetry_points(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(sym, vec![(3.5, 3.5), (2.5, 2.5), (1.5, 1.5), (0.5, 0.5)]);
    let skew = symmetry_points(&[1.0, 2.0, 3.0, 4.0, 20.0, 30.0]).unwrap();
    assert!(skew.iter().all(|p| p.1 >= p.0));
}

const REPS: [f64; 400] = [
    3.798884, 3.717508, 5.966972, 4.639392, 4.028964, 3.863979, 5.421131, 3.945159, 3.727922, 5.613993, 3.803292,
    4.677562, 4.993238, 4.554665, 4.945906, 6.338773, 4.483106, 3.740693, 3.163254, 4.795234, 4.647743, 5.265091,
    4.535755, 4.521362, 4.278684, 4.48024, 5.160227, 4.619647, 5.100442, 6.901187, 5.479118, 3.42391, 6.73353, 5.34781,
    4.058587, 5.907049, 5.017656, 4.384781, 4.366491, 4.006568, 5.048119, 6.068817, 4.674948, 5.420824, 6.875646,
    3.785409, 5.257522, 4.693563, 3.940744, 3.974153, 4.984715, 5.433863, 4.46841, 5.054391, 5.722209, 5.296561,
    5.86867, 5.461396, 5.488006, 6.826634, 5.623172, 5.129471, 5.709944, 4.080876, 4.662295, 5.793221, 5.630787,
    6.548362, 5.010473, 3.537672, 6.947247, 6.092893, 3.941263, 6.375824, 5.032403, 3.186071, 4.579646, 4.491096,
    6.590848, 4.207931, 4.746383, 4.723239, 4.621027, 4.085926, 5.219087, 6.076805, 5.623978, 4.072501, 3.850184,
    5.118974, 4.29342, 4.369834, 3.320305, 6.950492, 5.916619, 4.026093, 5.90821, 6.342487, 2.610447, 4.451055,
    4.612027, 5.648254, 4.878535, 4.769569, 4.941629, 6.853326, 7.15998, 4.475174, 4.073756, 7.692553, 4.020255,
    4.426598, 5.036582, 5.483246, 6.029036, 5.390977, 4.127617, 5.506382, 5.249918, 6.876919, 4.985, 3.663027, 3.95498,
    6.450153, 4.459869, 2.895534, 4.4193, 5.000015, 6.188831, 3.985532, 5.666683, 5.795299, 4.300612, 4.81241, 6.76945,
    6.720485, 5.855522, 5.331946, 6.13831, 4.859323, 4.904907, 4.13964, 5.005563, 4.917983, 7.773632, 4.807157,
    6.270818, 6.320497, 4.813521, 6.169397, 2.823186, 5.094927, 5.857261, 2.602135, 3.840829, 6.056074, 4.742989,
    3.901801, 4.626404, 4.457412, 5.724447, 5.453886, 4.719772, 4.334657, 4.946124, 6.325156, 5.349026, 5.638946,
    4.849318, 3.708184, 4.289156, 5.651237, 5.103216, 4.09616, 3.38213, 5.110682, 4.660469, 4.64947, 4.67055, 5.283905,
    5.294448, 5.962129, 5.928422, 6.332093, 5.806363, 4.664392, 4.945731, 5.494215, 4.689708, 5.549439, 6.153609,
    4.198754, 3.229616, 6.193655, 5.01776, 5.70867, 3.667695, 6.732993, 5.73862, 4.873926, 4.648012, 5.113896,
    4.833269, 4.72131, 3.8804, 6.262809, 4.960422, 5.074636, 5.092916, 2.641783, 5.527141, 4.670845, 4.710772,
    5.069788, 4.027524, 5.843381, 4.304135, 4.096342, 4.13781, 5.158753, 5.745779, 4.404947, 3.489813, 6.062491,
    3.934916, 6.19483, 4.945704, 4.30165, 4.35996, 5.461866, 5.889372, 4.80183, 5.946934, 7.086978, 4.791692, 3.850553,
    3.9456, 3.609665, 4.360335, 4.276221, 3.674897, 5.530184, 2.388661, 3.679201, 7.015252, 6.037802, 5.522792,
    5.457259, 6.770937, 5.53563, 5.077577, 3.76984, 4.926333, 4.266116, 4.293825, 4.285761, 4.955353, 4.670289,
    5.871111, 3.673819, 4.83053, 6.859497, 5.038897, 6.198615, 6.136863, 6.316648, 5.738119, 3.619554, 4.732612,
    4.619664, 3.451145, 4.536118, 3.74992, 3.334449, 5.210896, 4.956409, 4.312185, 4.049414, 4.458727, 4.60591,
    4.59189, 4.947292, 4.564459, 5.386972, 5.522871, 5.35787, 4.336896, 4.383494, 5.434864, 4.802957, 4.956848,
    2.508005, 2.299896, 5.337148, 5.329559, 6.393432, 5.721163, 5.221739, 5.016362, 3.57326, 5.883176, 4.396414,
    5.502484, 5.200042, 5.272097, 5.892675, 4.510134, 4.612765, 4.144139, 4.622848, 6.665941, 4.636681, 6.185616,
    5.826011, 5.729282, 5.803015, 3.985723, 5.373448, 6.588412, 4.304204, 4.952893, 4.310841, 5.317012, 3.92147,
    6.60968, 4.623165, 3.457338, 6.790791, 5.970523, 5.518511, 5.532588, 4.642202, 4.776704, 5.724019, 5.917625,
    5.932612, 5.212551, 4.533612, 4.807913, 6.1096, 6.439918, 4.658688, 4.522696, 6.24755, 5.723649, 5.697258,
    3.687666, 4.626243, 4.485004, 5.207905, 3.315058, 6.271753, 5.300132, 5.434837, 3.81586, 4.97358, 6.33465,
    3.430304, 5.305768, 5.698805, 4.308053, 5.814741, 4.669931, 6.229767, 5.527512, 5.224366, 7.008916, 5.831896,
    6.308636, 3.81048, 4.353393, 5.836569, 5.905612, 4.94426, 5.779727, 3.941646, 5.213835, 3.852716, 5.340571,
    4.513828, 4.359002, 5.056299, 6.81746, 3.988238, 4.398249, 6.067664, 4.576433, 6.769264, 5.22253, 5.090632,
    5.356884, 6.74118, 5.655362, 5.516712, 5.251858, 3.437581, 4.114728, 6.483344, 5.821878, 5.506065,
];
const JACK: [f64; 30] = [
    5.227413, 4.991281, 5.167593, 4.919841, 5.178272, 4.957173, 5.057236, 5.190683, 5.034642, 5.229516, 5.191208,
    5.171892, 5.364882, 4.805223, 4.780744, 5.223248, 5.288159, 5.502328, 5.251251, 5.266043, 4.923384, 5.250524,
    5.234768, 4.888827, 4.920736, 5.077794, 5.157651, 4.511043, 5.114061, 5.146171,
];

#[test]
fn intervals_match_scipy_formulas() {
    let iv = intervals_from_replicates(5.1, &REPS, &JACK, 0.95);
    let get = |m: IntervalMethod| iv.iter().find(|i| i.method == m).copied().unwrap();
    let p = get(IntervalMethod::Percentile);
    close(p.lower, 3.187159625, 1e-12);
    close(p.upper, 6.876887175, 1e-12);
    let b = get(IntervalMethod::Basic);
    close(b.lower, 2.0 * 5.1 - 6.876887175, 1e-12);
    close(b.upper, 2.0 * 5.1 - 3.187159625, 1e-12);
    let n = get(IntervalMethod::Normal);
    close(n.lower, 3.1964642639385175, 1e-9);
    close(n.upper, 7.003535736061481, 1e-9);
    let c = get(IntervalMethod::Bca);
    close(c.lower, 3.4791197475921987, 1e-9);
    close(c.upper, 7.0838129234155955, 1e-9);
    let mut sorted = REPS.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let (z0, acc) = openstats_stats::bootstrap::bca_constants(5.1, &sorted, &JACK).unwrap();
    close(z0, 0.12566134685507416, 1e-9);
    close(acc, 0.0244109944415687, 1e-9);
}

fn data() -> Vec<f64> {
    LV_X.to_vec()
}

#[test]
fn bootstrap_mean_matches_theory_and_is_reproducible() {
    let x = data();
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let opts = BootstrapOptions { resamples: 4000, level: 0.95, seed: 7 };
    let stat = |s: &[Vec<usize>]| s[0].iter().map(|&i| x[i]).sum::<f64>() / n;
    let b = bootstrap(&[x.len()], stat, &opts).unwrap();
    close(b.estimate, mean, 1e-12);
    // Bootstrap SE is close to the plug-in SE sd*sqrt((n-1)/n)/sqrt(n).
    let plug = sd * ((n - 1.0) / n).sqrt() / n.sqrt();
    assert!((b.std_error / plug - 1.0).abs() < 0.06, "{} vs {plug}", b.std_error);
    assert!(b.bias.abs() < 0.1 * plug * 3.0);
    assert_eq!(b.intervals.len(), 4);
    let bca = b.intervals.iter().find(|i| i.method == IntervalMethod::Bca).unwrap();
    assert!(bca.lower < mean && mean < bca.upper);
    let again = bootstrap(&[x.len()], stat, &opts).unwrap();
    assert_eq!(b.replicates, again.replicates);
    let other = bootstrap(&[x.len()], stat, &BootstrapOptions { seed: 8, ..opts }).unwrap();
    assert_ne!(b.replicates, other.replicates);
}

#[test]
fn bootstrap_coverage_is_near_nominal() {
    // Skewed population: exponential(1) via inverse cdf from a simple LCG; true mean 1.
    let mut state = 12345u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let (mut hit_pct, mut hit_bca) = (0, 0);
    let trials = 120;
    for t in 0..trials {
        let x: Vec<f64> = (0..30).map(|_| -next().ln()).collect();
        let m = x.len() as f64;
        let b = bootstrap(
            &[30],
            |s| s[0].iter().map(|&i| x[i]).sum::<f64>() / m,
            &BootstrapOptions { resamples: 400, level: 0.90, seed: t + 1 },
        )
        .unwrap();
        for i in &b.intervals {
            if i.lower <= 1.0 && 1.0 <= i.upper {
                match i.method {
                    IntervalMethod::Percentile => hit_pct += 1,
                    IntervalMethod::Bca => hit_bca += 1,
                    _ => {}
                }
            }
        }
    }
    // Nominal 90 % with 120 trials: accept 78 %..97 %.
    for h in [hit_pct, hit_bca] {
        let r = f64::from(h) / f64::from(trials as u32);
        assert!(r > 0.78 && r < 0.97, "coverage {r}");
    }
}

#[test]
fn bootstrap_errors() {
    let ok = BootstrapOptions::default();
    assert!(bootstrap(&[1], |_| 1.0, &ok).is_err());
    assert!(bootstrap(&[5], |_| 1.0, &BootstrapOptions { resamples: 5, ..ok }).is_err());
    assert!(bootstrap(&[5], |_| 1.0, &BootstrapOptions { level: 1.2, ..ok }).is_err());
    assert!(bootstrap(&[5], |_| f64::NAN, &ok).is_err());
    // A constant statistic has no BCa interval but the other three exist.
    let b = bootstrap(&[8], |_| 2.0, &ok).unwrap();
    assert_eq!(b.intervals.len(), 3);
    assert!(b.bca_constants.is_none());
}
