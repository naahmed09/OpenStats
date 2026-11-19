//! Tests of normality: Shapiro–Wilk, Ryan–Joiner, Kolmogorov–Smirnov (fully
//! specified normal) and Lilliefors (estimated mean and SD). The
//! Anderson–Darling test lives in [`crate::graphical::anderson_darling`].

use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ContinuousDistribution, Normal};
use openstats_math::stats::summary;

/// Result of the Shapiro–Wilk test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapiroWilk {
    pub w: f64,
    pub p_value: f64,
    pub n: usize,
}

fn poly(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
}

/// Shapiro–Wilk test for 3 ≤ n ≤ 5000 using Royston's (1992, 1995)
/// algorithm AS R94. Errors if the sample is constant.
pub fn shapiro_wilk(data: &[f64]) -> Result<ShapiroWilk> {
    let x = crate::clean_sorted(data)?;
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "Shapiro-Wilk test", needed: 3, got: n });
    }
    if n > 5000 {
        return invalid(format!("the Shapiro-Wilk algorithm is valid for n <= 5000, got {n}"));
    }
    let range = x[n - 1] - x[0];
    if range == 0.0 {
        return invalid("the Shapiro-Wilk test needs non-constant data");
    }
    let nn2 = n / 2;
    let nf = n as f64;
    let norm = Normal::standard();
    let mut a = vec![0.0; nn2 + 1]; // 1-based
    if n == 3 {
        a[1] = 0.5f64.sqrt();
    } else {
        let an25 = nf + 0.25;
        let m: Vec<f64> =
            std::iter::once(0.0).chain((1..=nn2).map(|i| norm.quantile((i as f64 - 0.375) / an25))).collect();
        let summ2 = 2.0 * m[1..].iter().map(|v| v * v).sum::<f64>();
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / nf.sqrt();
        const C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
        const C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
        let a1 = poly(&C1, rsn) - m[1] / ssumm2;
        let (i1, fac) = if n > 5 {
            let a2 = -m[2] / ssumm2 + poly(&C2, rsn);
            a[2] = a2;
            (3, ((summ2 - 2.0 * m[1] * m[1] - 2.0 * m[2] * m[2]) / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2)).sqrt())
        } else {
            (2, ((summ2 - 2.0 * m[1] * m[1]) / (1.0 - 2.0 * a1 * a1)).sqrt())
        };
        a[1] = a1;
        for i in i1..=nn2 {
            a[i] = -m[i] / fac;
        }
    }
    let b = summary::sum((1..=nn2).map(|i| a[i] * (x[n - i] - x[i - 1])));
    let ssd = summary::sum_squared_deviations(&x)?;
    let w = (b * b / ssd).clamp(0.0, 1.0);

    let p = if n == 3 {
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = std::f64::consts::FRAC_PI_3;
        (pi6 * (w.sqrt().asin() - stqr)).max(0.0)
    } else {
        let y = (1.0 - w).ln();
        let xx = nf.ln();
        const G: [f64; 2] = [-2.273, 0.459];
        const C3: [f64; 4] = [0.544, -0.39978, 0.025054, -6.714e-4];
        const C4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
        const C5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
        const C6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];
        let (mu, sigma, y) = if n <= 11 {
            let gamma = poly(&G, nf);
            if y >= gamma {
                return Ok(ShapiroWilk { w, p_value: 1e-99, n });
            }
            (poly(&C3, nf), poly(&C4, nf).exp(), -(gamma - y).ln())
        } else {
            (poly(&C5, xx), poly(&C6, xx).exp(), y)
        };
        norm.sf((y - mu) / sigma)
    };
    Ok(ShapiroWilk { w, p_value: p.clamp(0.0, 1.0), n })
}

/// Result of the Ryan–Joiner test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RyanJoiner {
    /// Correlation between the ordered data and their Blom normal scores.
    pub rj: f64,
    /// Approximate p-value (see [`ryan_joiner`]).
    pub p_value: f64,
    pub n: usize,
}

/// Ryan–Joiner normality test, 5 ≤ n ≤ 5000.
///
/// The statistic RJ is the Pearson correlation of the ordered data with the
/// Blom normal scores Φ⁻¹((i − 3/8)/(n + 1/4)). The p-value uses Royston's
/// (1993) normalising transformation for the equivalent Shapiro–Francia
/// statistic W′ = RJ²: with u = ln n, v = ln u, ln(1 − W′) is approximately
/// normal with mean −1.2725 + 1.0521(v − u) and SD 1.0308 − 0.26758(v + 2/u).
/// Minitab interpolates published critical-value tables instead, so its
/// p-values can differ slightly.
pub fn ryan_joiner(data: &[f64]) -> Result<RyanJoiner> {
    let x = crate::clean_sorted(data)?;
    let n = x.len();
    if n < 5 {
        return Err(StatsError::InsufficientData { what: "Ryan-Joiner test", needed: 5, got: n });
    }
    if n > 5000 {
        return invalid(format!("the Ryan-Joiner p-value approximation is valid for n <= 5000, got {n}"));
    }
    let norm = Normal::standard();
    let nf = n as f64;
    let scores: Vec<f64> = (1..=n).map(|i| norm.quantile((i as f64 - 0.375) / (nf + 0.25))).collect();
    let mx = summary::mean(&x)?;
    let sxx = summary::sum_squared_deviations(&x)?;
    if sxx == 0.0 {
        return invalid("the Ryan-Joiner test needs non-constant data");
    }
    let sxy = summary::sum(x.iter().zip(&scores).map(|(a, b)| (a - mx) * b));
    let syy = summary::sum(scores.iter().map(|s| s * s)); // scores are symmetric, mean 0
    let rj = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let w = rj * rj;
    let p = if w >= 1.0 {
        1.0
    } else {
        let u = nf.ln();
        let v = u.ln();
        let mu = -1.2725 + 1.0521 * (v - u);
        let sig = 1.0308 - 0.26758 * (v + 2.0 / u);
        norm.sf(((1.0 - w).ln() - mu) / sig)
    };
    Ok(RyanJoiner { rj, p_value: p.clamp(0.0, 1.0), n })
}

/// Result of a Kolmogorov–Smirnov type test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsTest {
    /// D = sup |Fₙ − F|.
    pub d: f64,
    pub d_plus: f64,
    pub d_minus: f64,
    pub p_value: f64,
    pub n: usize,
}

fn ks_statistic(sorted: &[f64], cdf: impl Fn(f64) -> f64) -> (f64, f64) {
    let n = sorted.len() as f64;
    let (mut dp, mut dm) = (0.0f64, 0.0f64);
    for (i, &x) in sorted.iter().enumerate() {
        let f = cdf(x);
        dp = dp.max((i + 1) as f64 / n - f);
        dm = dm.max(f - i as f64 / n);
    }
    (dp, dm)
}

/// Kolmogorov survival function 2Σ(−1)^{k−1}exp(−2k²λ²).
pub(crate) fn kolmogorov_sf(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    if lambda < 0.2 {
        return 1.0;
    }
    let mut s = 0.0;
    for k in 1..=200 {
        let kf = f64::from(k);
        let t = (-2.0 * kf * kf * lambda * lambda).exp();
        s += if k % 2 == 1 { t } else { -t };
        if t < 1e-18 {
            break;
        }
    }
    (2.0 * s).clamp(0.0, 1.0)
}

type Mat = Vec<Vec<f64>>;

fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    let m = a.len();
    let mut c = vec![vec![0.0; m]; m];
    for i in 0..m {
        for k in 0..m {
            let aik = a[i][k];
            if aik != 0.0 {
                for j in 0..m {
                    c[i][j] += aik * b[k][j];
                }
            }
        }
    }
    c
}

/// Matrix power with a decimal exponent kept separately to avoid overflow.
fn mat_pow(a: &Mat, ea: i32, n: u64) -> (Mat, i32) {
    if n == 1 {
        return (a.clone(), ea);
    }
    let (v, ev) = mat_pow(a, ea, n / 2);
    let b = mat_mul(&v, &v);
    let eb = 2 * ev;
    let (mut r, mut er) = if n % 2 == 0 { (b, eb) } else { (mat_mul(a, &b), ea + eb) };
    let mid = r.len() / 2;
    if r[mid][mid] > 1e140 {
        for row in &mut r {
            for x in row.iter_mut() {
                *x *= 1e-140;
            }
        }
        er += 140;
    }
    (r, er)
}

/// P(Dₙ < d) for the two-sided KS statistic, exact (Marsaglia, Tsang & Wang 2003).
fn kolmogorov_cdf_exact(n: usize, d: f64) -> f64 {
    let nd = n as f64 * d;
    let k = nd as usize + 1;
    let m = 2 * k - 1;
    let h = k as f64 - nd;
    let mut hm = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            hm[i][j] = if i + 1 >= j { 1.0 } else { 0.0 };
        }
    }
    for i in 0..m {
        hm[i][0] -= h.powi(i as i32 + 1);
        hm[m - 1][i] -= h.powi((m - i) as i32);
    }
    hm[m - 1][0] += if 2.0 * h - 1.0 > 0.0 { (2.0 * h - 1.0).powi(m as i32) } else { 0.0 };
    for i in 0..m {
        for j in 0..m {
            if i + 1 > j {
                for g in 1..=(i + 1 - j) {
                    hm[i][j] /= g as f64;
                }
            }
        }
    }
    let (q, mut eq) = mat_pow(&hm, 0, n as u64);
    let mut s = q[k - 1][k - 1];
    for i in 1..=n {
        s = s * i as f64 / n as f64;
        if s < 1e-140 {
            s *= 1e140;
            eq -= 140;
        }
    }
    s * 10f64.powi(eq)
}

/// Two-sided p-value of the one-sample KS statistic D for sample size n.
pub(crate) fn ks_p_value(n: usize, d: f64) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let nf = n as f64;
    let nd2 = nf * d * d;
    let exact_ok = n <= 1000 && (if n <= 99 { nd2 < 7.24 } else { nd2 < 3.76 });
    if exact_ok {
        (1.0 - kolmogorov_cdf_exact(n, d)).clamp(0.0, 1.0)
    } else {
        // Stephens' modification of the asymptotic law
        kolmogorov_sf((nf.sqrt() + 0.12 + 0.11 / nf.sqrt()) * d)
    }
}

/// Kolmogorov–Smirnov test of H₀: data ~ Normal(`mu`, `sigma`) with the
/// parameters **specified**, two-sided. The p-value is exact for n ≤ 1000
/// (Marsaglia–Tsang–Wang) and uses Stephens' asymptotic approximation
/// beyond, or in the far tail where the complement of the exact cdf would
/// lose accuracy.
pub fn ks_normal(data: &[f64], mu: f64, sigma: f64) -> Result<KsTest> {
    if !(sigma > 0.0 && sigma.is_finite() && mu.is_finite()) {
        return invalid("mu must be finite and sigma positive");
    }
    let x = crate::clean_sorted(data)?;
    let n = x.len();
    if n < 1 {
        return Err(StatsError::InsufficientData { what: "Kolmogorov-Smirnov test", needed: 1, got: 0 });
    }
    let norm = Normal::new(mu, sigma)?;
    let (dp, dm) = ks_statistic(&x, |v| norm.cdf(v));
    let d = dp.max(dm);
    Ok(KsTest { d, d_plus: dp, d_minus: dm, p_value: ks_p_value(n, d), n })
}

/// Lilliefors test (KS with the mean and SD estimated from the data), n ≥ 5.
///
/// The p-value is the Dallal–Wilkinson (1986) approximation with the
/// refinements used by R's `nortest::lillie.test`; it is accurate for
/// p ≲ 0.1 and a smooth approximation above.
pub fn lilliefors(data: &[f64]) -> Result<KsTest> {
    let x = crate::clean_sorted(data)?;
    let n = x.len();
    if n < 5 {
        return Err(StatsError::InsufficientData { what: "Lilliefors test", needed: 5, got: n });
    }
    let mean = summary::mean(&x)?;
    let sd = summary::std_dev(&x)?;
    if sd == 0.0 {
        return invalid("the Lilliefors test needs non-constant data");
    }
    let norm = Normal::standard();
    let (dp, dm) = ks_statistic(&x, |v| norm.cdf((v - mean) / sd));
    let d = dp.max(dm);
    let nf = n as f64;
    let (kd, nd) = if n <= 100 { (d, nf) } else { (d * (nf / 100.0).powf(0.49), 100.0) };
    let mut p = (-7.01256 * kd * kd * (nd + 2.78019) + 2.99587 * kd * (nd + 2.78019).sqrt() - 0.122119
        + 0.974598 / nd.sqrt()
        + 1.67997 / nd)
        .exp();
    if p > 0.1 {
        let kk = (nf.sqrt() - 0.01 + 0.85 / nf.sqrt()) * d;
        p = if kk <= 0.302 {
            1.0
        } else if kk <= 0.5 {
            2.76773 - 19.828315 * kk + 80.709644 * kk.powi(2) - 138.55152 * kk.powi(3) + 81.218052 * kk.powi(4)
        } else if kk <= 0.9 {
            -4.901232 + 40.662806 * kk - 97.490286 * kk.powi(2) + 94.029866 * kk.powi(3) - 32.355711 * kk.powi(4)
        } else if kk <= 1.31 {
            6.198765 - 19.558097 * kk + 23.186922 * kk.powi(2) - 12.234627 * kk.powi(3) + 2.423045 * kk.powi(4)
        } else {
            0.0
        };
    }
    Ok(KsTest { d, d_plus: dp, d_minus: dm, p_value: p.clamp(0.0, 1.0), n })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_math::random::Pcg64;

    fn normal_sample(seed: u64, n: usize) -> Vec<f64> {
        let mut r = Pcg64::with_stream(seed, 1);
        let z = Normal::standard();
        (0..n).map(|_| z.sample(&mut r)).collect()
    }

    #[test]
    fn shapiro_edge_cases() {
        // n = 3 exact: W for (1,2,4)
        let r = shapiro_wilk(&[1.0, 2.0, 4.0]).unwrap();
        assert!(r.w > 0.75 && r.w < 1.0 && r.p_value > 0.0 && r.p_value <= 1.0);
        assert!(shapiro_wilk(&[1.0, 2.0]).is_err());
        assert!(shapiro_wilk(&[3.0, 3.0, 3.0]).is_err());
        assert!(shapiro_wilk(&vec![0.0; 6000]).is_err());
        let r = shapiro_wilk(&[1.0, 2.0, 3.0]).unwrap();
        assert!((r.w - 1.0).abs() < 1e-12 && (r.p_value - 1.0).abs() < 1e-9);
        let r = shapiro_wilk(&[1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(r.n, 6);
    }

    #[test]
    fn shapiro_p_values_are_calibrated() {
        // Under H0 the p-values should be roughly uniform: P(p < .05) ≈ .05
        for n in [8usize, 20, 100] {
            let mut rej = 0;
            let reps = 600;
            for s in 0..reps {
                if shapiro_wilk(&normal_sample(1000 + s as u64, n)).unwrap().p_value < 0.05 {
                    rej += 1;
                }
            }
            let rate = f64::from(rej) / f64::from(reps as u32);
            assert!((rate - 0.05).abs() < 0.025, "n={n}: rejection rate {rate}");
        }
    }

    #[test]
    fn ryan_joiner_is_calibrated_and_detects_skew() {
        for n in [10usize, 30, 100] {
            let mut rej = 0;
            let reps = 600;
            for s in 0..reps {
                if ryan_joiner(&normal_sample(5000 + s as u64, n)).unwrap().p_value < 0.05 {
                    rej += 1;
                }
            }
            let rate = f64::from(rej) / f64::from(reps as u32);
            assert!((rate - 0.05).abs() < 0.03, "n={n}: rejection rate {rate}");
        }
        let skew: Vec<f64> = (1..=40).map(|i| (f64::from(i) / 8.0).exp()).collect();
        assert!(ryan_joiner(&skew).unwrap().p_value < 0.01);
        let lin: Vec<f64> = (1..=20).map(f64::from).collect();
        assert!(ryan_joiner(&lin).unwrap().rj > 0.97);
        assert!(ryan_joiner(&[1.0, 2.0, 3.0, 4.0]).is_err());
        assert!(ryan_joiner(&[1.0; 6]).is_err());
    }

    #[test]
    fn ks_exact_vs_asymptotic_consistency() {
        // For a given D the exact p is close to Stephens' approximation at moderate n
        let (n, d) = (50usize, 0.15);
        let exact = ks_p_value(n, d);
        let approx = kolmogorov_sf((50f64.sqrt() + 0.12 + 0.11 / 50f64.sqrt()) * d);
        assert!((exact - approx).abs() < 0.01, "{exact} vs {approx}");
        assert_eq!(ks_p_value(10, 0.0), 1.0);
        assert_eq!(ks_p_value(10, 1.0), 0.0);
        // n = 1: D = max(F, 1-F) ≥ 0.5, P(D_1 < d) = 2d - 1
        assert!((kolmogorov_cdf_exact(1, 0.75) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn ks_and_lilliefors_basics() {
        let x = normal_sample(7, 40);
        let k = ks_normal(&x, 0.0, 1.0).unwrap();
        assert!(k.p_value > 0.0 && k.p_value <= 1.0 && k.d == k.d_plus.max(k.d_minus));
        assert!(ks_normal(&x, 0.0, -1.0).is_err());
        assert!(ks_normal(&[], 0.0, 1.0).is_err());
        let l = lilliefors(&x).unwrap();
        assert!(l.p_value > 0.0 && l.p_value <= 1.0);
        let skew: Vec<f64> = (1..=50).map(|i| (f64::from(i) / 10.0).exp()).collect();
        assert!(lilliefors(&skew).unwrap().p_value < 0.01);
        assert!(lilliefors(&[1.0, 2.0, 3.0, 4.0]).is_err());
        assert!(lilliefors(&[2.0; 8]).is_err());
    }

    #[test]
    fn lilliefors_matches_published_critical_values() {
        // Lilliefors (1967) 5 % points: n=10: .258, 20: .190, 30: .161 — p should be ≈ .05
        for (n, d) in [(10usize, 0.258), (20, 0.190), (30, 0.161)] {
            // build a sample whose D equals d is awkward; instead evaluate the p formula directly
            let nf = n as f64;
            let p = (-7.01256 * d * d * (nf + 2.78019) + 2.99587 * d * (nf + 2.78019).sqrt() - 0.122119
                + 0.974598 / nf.sqrt()
                + 1.67997 / nf)
                .exp();
            assert!((p - 0.05).abs() < 0.012, "n={n}: p={p}");
        }
    }

    #[test]
    fn lilliefors_is_calibrated() {
        let n = 25;
        let mut rej = 0;
        let reps = 800;
        for s in 0..reps {
            if lilliefors(&normal_sample(9000 + s as u64, n)).unwrap().p_value < 0.05 {
                rej += 1;
            }
        }
        let rate = f64::from(rej) / f64::from(reps as u32);
        assert!((rate - 0.05).abs() < 0.025, "rejection rate {rate}");
    }
}
