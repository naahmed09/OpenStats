//! Goodness-of-fit and moment checks for every sampler.
//!
//! Continuous distributions: one-sample Kolmogorov–Smirnov test against the
//! exact cdf plus a z-test of the sample mean. Discrete distributions:
//! Pearson χ² over pooled cells (expected count ≥ 10). Seeds are fixed, so a
//! run is deterministic; the acceptance level is p > 1e-3 per test.

use openstats_math::dist::*;
use openstats_math::random::multivariate::{dirichlet, multinomial, wishart, MultivariateNormal};
use openstats_math::random::{Mt19937, Mt19937_64, Pcg64, Rng, SeedableRng, Xoshiro256StarStar};
use openstats_math::Matrix;
use std::collections::BTreeMap;

/// P(√n·D > t) for the Kolmogorov distribution.
fn ks_pvalue(d: f64, n: usize) -> f64 {
    let t = d * ((n as f64).sqrt() + 0.12 + 0.11 / (n as f64).sqrt());
    if t < 0.2 {
        return 1.0;
    }
    let mut s = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * t * t).exp();
        s += if k % 2 == 1 { term } else { -term };
        if term < 1e-18 {
            break;
        }
    }
    (2.0 * s).clamp(0.0, 1.0)
}

fn ks_test(d: &dyn ContinuousDistribution, rng: &mut dyn Rng, n: usize, label: &str) {
    let mut x = d.sample_n(rng, n);
    assert!(x.iter().all(|v| !v.is_nan()), "{label}: NaN sample");
    x.sort_by(f64::total_cmp);
    let mut dmax = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        let f = d.cdf(v);
        dmax = dmax.max(f - i as f64 / n as f64).max((i + 1) as f64 / n as f64 - f);
    }
    let p = ks_pvalue(dmax, n);
    assert!(p > 1e-3, "{label}: KS D = {dmax:.5}, p = {p:.2e}");
    let (m, v) = (d.mean(), d.variance());
    if m.is_finite() && v.is_finite() && v > 0.0 {
        let mean = x.iter().sum::<f64>() / n as f64;
        let z = (mean - m) / (v / n as f64).sqrt();
        // Heavy tails make the mean's distribution non-normal; use 6 sigma.
        assert!(z.abs() < 6.0, "{label}: mean {mean} vs {m} (z = {z:.2})");
    }
}

fn chi_square_test(d: &dyn DiscreteDistribution, rng: &mut dyn Rng, n: usize, label: &str) {
    let mut counts: BTreeMap<i64, u64> = BTreeMap::new();
    let mut sum = 0.0;
    for _ in 0..n {
        let v = d.sample(rng);
        assert!(v == v.floor(), "{label}: non-integer sample {v}");
        let (lo, hi) = d.support();
        assert!(v >= lo && v <= hi, "{label}: sample {v} outside support");
        sum += v;
        *counts.entry(v as i64).or_insert(0) += 1;
    }
    let nf = n as f64;
    let start = d.quantile(1e-13).max(d.support().0) as i64;
    let max_obs = *counts.keys().next_back().unwrap();
    let mut cells: Vec<(f64, f64)> = Vec::new();
    let (mut e_acc, mut o_acc) = (nf * d.cdf((start - 1) as f64), 0.0);
    for (&k, &c) in counts.range(..start) {
        let _ = k;
        o_acc += c as f64;
    }
    let mut k = start;
    loop {
        e_acc += nf * d.pmf(k as f64);
        o_acc += counts.get(&k).copied().unwrap_or(0) as f64;
        if e_acc >= 10.0 {
            cells.push((o_acc, e_acc));
            e_acc = 0.0;
            o_acc = 0.0;
        }
        let tail = nf * d.sf(k as f64);
        if (k >= max_obs && tail < 1e-3) || k as f64 >= d.support().1 {
            break;
        }
        k += 1;
    }
    // Remaining mass above k, and anything left unpooled, join the last cell.
    let tail = nf * d.sf(k as f64);
    let tail_obs: f64 = counts.range(k + 1..).map(|(_, &c)| c as f64).sum();
    e_acc += tail;
    o_acc += tail_obs;
    if e_acc > 0.0 || o_acc > 0.0 {
        if let Some(last) = cells.last_mut() {
            last.0 += o_acc;
            last.1 += e_acc;
        } else {
            cells.push((o_acc, e_acc));
        }
    }
    assert!(cells.len() >= 2, "{label}: degenerate cell structure ({} cells)", cells.len());
    let stat: f64 = cells.iter().map(|&(o, e)| (o - e) * (o - e) / e).sum();
    let p = ChiSquared::new(cells.len() as f64 - 1.0).unwrap().sf(stat);
    assert!(p > 1e-3, "{label}: χ² = {stat:.1} on {} cells, p = {p:.2e}", cells.len() - 1);
    let (m, v) = (d.mean(), d.variance());
    let mean = sum / nf;
    let z = (mean - m) / (v / nf).sqrt();
    assert!(z.abs() < 5.0, "{label}: mean {mean} vs {m} (z = {z:.2})");
}

#[test]
fn continuous_samplers_pass_ks() {
    let mut rng = Pcg64::seed_from_u64(20261117);
    let n = 40_000;
    let cases: Vec<(&str, Box<dyn ContinuousDistribution>)> = vec![
        ("normal", Box::new(Normal::new(1.5, 2.0).unwrap())),
        ("lognormal", Box::new(Lognormal::new(0.3, 0.7, 1.0).unwrap())),
        ("halfnormal", Box::new(HalfNormal::new(2.0, 0.5).unwrap())),
        ("foldednormal", Box::new(FoldedNormal::new(1.0, 2.0).unwrap())),
        ("johnsonsu", Box::new(Johnson::new(JohnsonFamily::SU, 0.5, 1.3, 2.0, 1.5).unwrap())),
        ("johnsonsb", Box::new(Johnson::new(JohnsonFamily::SB, -0.3, 0.9, 1.0, 4.0).unwrap())),
        ("johnsonsl", Box::new(Johnson::new(JohnsonFamily::SL, 0.2, 1.1, 0.0, 2.0).unwrap())),
        ("gamma 0.05", Box::new(Gamma::new(0.05, 1.0, 0.0).unwrap())),
        ("gamma 0.5", Box::new(Gamma::new(0.5, 2.0, 0.0).unwrap())),
        ("gamma 1", Box::new(Gamma::new(1.0, 1.0, 0.0).unwrap())),
        ("gamma 2.5", Box::new(Gamma::new(2.5, 3.0, 1.0).unwrap())),
        ("gamma 40", Box::new(Gamma::new(40.0, 0.2, 0.0).unwrap())),
        ("chisq 1", Box::new(ChiSquared::new(1.0).unwrap())),
        ("chisq 7", Box::new(ChiSquared::new(7.0).unwrap())),
        ("exponential", Box::new(Exponential::new(2.0, 1.0).unwrap())),
        ("weibull", Box::new(Weibull::new(0.6, 1.0, 0.0).unwrap())),
        ("beta .3 .4", Box::new(Beta::new(0.3, 0.4).unwrap())),
        ("beta 2 5", Box::new(Beta::new(2.0, 5.0).unwrap())),
        ("beta 50 60", Box::new(Beta::new(50.0, 60.0).unwrap())),
        ("beta .01 3", Box::new(Beta::new(0.01, 3.0).unwrap())),
        ("t 1.5", Box::new(StudentT::new(1.5).unwrap())),
        ("t 5", Box::new(StudentT::new(5.0).unwrap())),
        ("t 30", Box::new(StudentT::new(30.0).unwrap())),
        ("t inf", Box::new(StudentT::new(f64::INFINITY).unwrap())),
        ("f 3 12", Box::new(FisherF::new(3.0, 12.0).unwrap())),
        ("f 0.5 0.7", Box::new(FisherF::new(0.5, 0.7).unwrap())),
        ("logistic", Box::new(Logistic::new(1.0, 2.0).unwrap())),
        ("loglogistic", Box::new(Loglogistic::new(0.5, 0.4, 1.0).unwrap())),
        ("cauchy", Box::new(Cauchy::new(-1.0, 0.5).unwrap())),
        ("laplace", Box::new(Laplace::new(1.0, 2.0).unwrap())),
        ("uniform", Box::new(Uniform::new(-2.0, 3.0).unwrap())),
        ("triangular", Box::new(Triangular::new(0.0, 1.0, 4.0).unwrap())),
        ("sev", Box::new(SmallestExtremeValue::new(10.0, 2.0).unwrap())),
        ("lev", Box::new(LargestExtremeValue::new(10.0, 2.0).unwrap())),
    ];
    for (name, d) in &cases {
        ks_test(d.as_ref(), &mut rng, n, name);
    }
}

#[test]
fn non_central_samplers_pass_ks() {
    let mut rng = Pcg64::seed_from_u64(77);
    let n = 6_000;
    let cases: Vec<(&str, Box<dyn ContinuousDistribution>)> = vec![
        ("ncchisq 3 2", Box::new(NonCentralChiSquared::new(3.0, 2.0).unwrap())),
        ("ncchisq 0.5 40", Box::new(NonCentralChiSquared::new(0.5, 40.0).unwrap())),
        ("ncchisq 10 500", Box::new(NonCentralChiSquared::new(10.0, 500.0).unwrap())),
        ("ncbeta 2 5 3", Box::new(NonCentralBeta::new(2.0, 5.0, 3.0).unwrap())),
        ("ncbeta .5 .5 10", Box::new(NonCentralBeta::new(0.5, 0.5, 10.0).unwrap())),
        ("ncf 5 10 4", Box::new(NonCentralF::new(5.0, 10.0, 4.0).unwrap())),
        ("ncf 3 50 40", Box::new(NonCentralF::new(3.0, 50.0, 40.0).unwrap())),
        ("nct 10 1.5", Box::new(NonCentralT::new(10.0, 1.5).unwrap())),
        ("nct 10 -1.5", Box::new(NonCentralT::new(10.0, -1.5).unwrap())),
        ("nct 30 5", Box::new(NonCentralT::new(30.0, 5.0).unwrap())),
    ];
    for (name, d) in &cases {
        ks_test(d.as_ref(), &mut rng, n, name);
    }
}

#[test]
fn discrete_samplers_pass_chi_square() {
    let mut rng = Pcg64::seed_from_u64(424242);
    let n = 200_000;
    let cases: Vec<(&str, Box<dyn DiscreteDistribution>)> = vec![
        ("poisson 0.3", Box::new(Poisson::new(0.3).unwrap())),
        ("poisson 3", Box::new(Poisson::new(3.0).unwrap())),
        ("poisson 9.9", Box::new(Poisson::new(9.9).unwrap())),
        ("poisson 10", Box::new(Poisson::new(10.0).unwrap())),
        ("poisson 50", Box::new(Poisson::new(50.0).unwrap())),
        ("poisson 1e4", Box::new(Poisson::new(1e4).unwrap())),
        ("binomial 10 .3", Box::new(Binomial::new(10, 0.3).unwrap())),
        ("binomial 100 .5", Box::new(Binomial::new(100, 0.5).unwrap())),
        ("binomial 1000 .9", Box::new(Binomial::new(1000, 0.9).unwrap())),
        ("binomial 1e6 .3", Box::new(Binomial::new(1_000_000, 0.3).unwrap())),
        ("binomial 1e6 1e-5", Box::new(Binomial::new(1_000_000, 1e-5).unwrap())),
        ("hypergeom 50 20 10", Box::new(Hypergeometric::new(50, 20, 10).unwrap())),
        ("hypergeom 1000 300 100", Box::new(Hypergeometric::new(1000, 300, 100).unwrap())),
        ("hypergeom 1e5 3e4 2e3", Box::new(Hypergeometric::new(100_000, 30_000, 2_000).unwrap())),
        ("hypergeom 20 18 15", Box::new(Hypergeometric::new(20, 18, 15).unwrap())),
        ("negbin 3 .4", Box::new(NegativeBinomial::new(3, 0.4).unwrap())),
        ("negbin 50 .05", Box::new(NegativeBinomial::new(50, 0.05).unwrap())),
        ("geometric .01", Box::new(Geometric::new(0.01).unwrap())),
        ("geometric .7", Box::new(Geometric::new(0.7).unwrap())),
        ("uniform -3 4", Box::new(DiscreteUniform::new(-3, 4).unwrap())),
        ("uniform 0 999", Box::new(DiscreteUniform::new(0, 999).unwrap())),
    ];
    for (name, d) in &cases {
        chi_square_test(d.as_ref(), &mut rng, n, name);
    }
}

#[test]
fn user_defined_discrete_sampler() {
    let d = Discrete::new(&[1.0, 2.5, 7.0, 9.0], &[0.1, 0.4, 0.0, 0.5]).unwrap();
    let mut rng = Pcg64::seed_from_u64(8);
    let n = 200_000;
    let mut c = [0u64; 4];
    for _ in 0..n {
        let v = d.sample(&mut rng);
        let i = [1.0, 2.5, 7.0, 9.0].iter().position(|&x| x == v).expect("sample is a listed value");
        c[i] += 1;
    }
    assert_eq!(c[2], 0, "zero-probability value drawn");
    let stat: f64 = [(c[0], 0.1), (c[1], 0.4), (c[3], 0.5)]
        .iter()
        .map(|&(o, p)| (o as f64 - n as f64 * p).powi(2) / (n as f64 * p))
        .sum();
    assert!(ChiSquared::new(2.0).unwrap().sf(stat) > 1e-3, "χ² = {stat}");
}

#[test]
fn degenerate_parameters() {
    let mut rng = Pcg64::seed_from_u64(1);
    assert_eq!(Binomial::new(0, 0.5).unwrap().sample(&mut rng), 0.0);
    assert_eq!(Binomial::new(7, 1.0).unwrap().sample(&mut rng), 7.0);
    assert_eq!(Poisson::new(0.0).unwrap().sample(&mut rng), 0.0);
    assert_eq!(Geometric::new(1.0).unwrap().sample(&mut rng), 1.0);
    assert_eq!(NegativeBinomial::new(5, 1.0).unwrap().sample(&mut rng), 5.0);
    assert_eq!(Hypergeometric::new(10, 10, 4).unwrap().sample(&mut rng), 4.0);
    assert_eq!(Hypergeometric::new(10, 0, 4).unwrap().sample(&mut rng), 0.0);
    assert_eq!(DiscreteUniform::new(5, 5).unwrap().sample(&mut rng), 5.0);
    let full = DiscreteUniform::new(i64::MIN, i64::MAX).unwrap();
    let _ = full.sample(&mut rng);
}

#[test]
fn generators_are_uniform_and_reproducible() {
    fn check(rng: &mut dyn Rng, label: &str) {
        let n = 400_000;
        let u: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)), "{label}");
        let mean = u.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 5.0 * (1.0 / 12.0 / n as f64).sqrt(), "{label}: mean {mean}");
        // Lag-1 serial correlation.
        let c: f64 = u.windows(2).map(|w| (w[0] - 0.5) * (w[1] - 0.5)).sum::<f64>() / (n - 1) as f64 / (1.0 / 12.0);
        assert!(c.abs() < 5.0 / (n as f64).sqrt(), "{label}: lag-1 correlation {c}");
        // 100-cell χ² on the unit interval.
        let mut cells = [0u32; 100];
        for &v in &u {
            cells[(v * 100.0) as usize] += 1;
        }
        let e = n as f64 / 100.0;
        let stat: f64 = cells.iter().map(|&c| (c as f64 - e).powi(2) / e).sum();
        let p = ChiSquared::new(99.0).unwrap().sf(stat);
        assert!(p > 1e-3, "{label}: χ² p = {p}");
        // Bit balance of the 64-bit output.
        let ones: u64 = (0..10_000).map(|_| u64::from(rng.next_u64().count_ones())).sum();
        assert!((ones as f64 - 320_000.0).abs() < 5.0 * (160_000.0f64).sqrt(), "{label}: bit balance {ones}");
        // Open-interval and bounded integers.
        for _ in 0..10_000 {
            let o = rng.next_f64_open();
            assert!(o > 0.0 && o < 1.0);
            assert!(rng.below(7) < 7);
        }
    }
    check(&mut Pcg64::seed_from_u64(1), "pcg64");
    check(&mut Mt19937::seed_from_u64(1), "mt19937");
    check(&mut Mt19937_64::seed_from_u64(1), "mt19937-64");
    check(&mut Xoshiro256StarStar::seed_from_u64(1), "xoshiro256**");

    // Reproducible, seed- and stream-sensitive.
    let take = |mut g: Pcg64| (0..4).map(|_| g.next_u64()).collect::<Vec<_>>();
    assert_eq!(take(Pcg64::seed_from_u64(9)), take(Pcg64::seed_from_u64(9)));
    assert_ne!(take(Pcg64::seed_from_u64(9)), take(Pcg64::seed_from_u64(10)));
    assert_ne!(take(Pcg64::with_stream(9, 0)), take(Pcg64::with_stream(9, 1)));
    let (s, i) = Pcg64::seed_from_u64(3).raw();
    assert_eq!(take(Pcg64::from_raw(s, i)), take(Pcg64::seed_from_u64(3)));

    // `below` is unbiased: every residue of a non-power-of-two modulus is hit equally.
    let mut g = Pcg64::seed_from_u64(5);
    let mut c = [0u32; 6];
    for _ in 0..600_000 {
        c[g.below(6) as usize] += 1;
    }
    let stat: f64 = c.iter().map(|&x| (x as f64 - 100_000.0).powi(2) / 100_000.0).sum();
    assert!(ChiSquared::new(5.0).unwrap().sf(stat) > 1e-3);
}

#[test]
fn different_generators_drive_the_samplers() {
    let d = Gamma::new(2.5, 1.0, 0.0).unwrap();
    ks_test(&d, &mut Mt19937::seed_from_u64(11), 30_000, "gamma/mt19937");
    ks_test(&d, &mut Mt19937_64::seed_from_u64(11), 30_000, "gamma/mt19937-64");
    ks_test(&d, &mut Xoshiro256StarStar::seed_from_u64(11), 30_000, "gamma/xoshiro");
}

#[test]
fn multivariate_normal_matches_mean_and_covariance() {
    let cov = Matrix::from_rows(&[vec![4.0, 1.2, -0.8], vec![1.2, 2.0, 0.5], vec![-0.8, 0.5, 1.0]]).unwrap();
    let mean = [1.0, -2.0, 0.5];
    let mvn = MultivariateNormal::new(&mean, &cov).unwrap();
    let mut rng = Pcg64::seed_from_u64(31);
    let n = 200_000;
    let x = mvn.sample_n(&mut rng, n);
    let mut m = [0.0; 3];
    for i in 0..n {
        for j in 0..3 {
            m[j] += x[(i, j)] / n as f64;
        }
    }
    for j in 0..3 {
        assert!((m[j] - mean[j]).abs() < 5.0 * (cov[(j, j)] / n as f64).sqrt(), "mean {j}: {}", m[j]);
        for k in 0..3 {
            let c: f64 = (0..n).map(|i| (x[(i, j)] - m[j]) * (x[(i, k)] - m[k])).sum::<f64>() / (n - 1) as f64;
            let se = ((cov[(j, j)] * cov[(k, k)] + cov[(j, k)] * cov[(j, k)]) / n as f64).sqrt();
            assert!((c - cov[(j, k)]).abs() < 5.0 * se, "cov {j}{k}: {c} vs {}", cov[(j, k)]);
        }
    }
    // Singular (rank-1) covariance: both coordinates move together.
    let sing = Matrix::from_rows(&[vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap();
    let s = MultivariateNormal::new(&[0.0, 0.0], &sing).unwrap();
    for _ in 0..100 {
        let v = s.sample(&mut rng);
        assert!((v[0] - v[1]).abs() < 1e-12, "{v:?}");
    }
    assert!(
        MultivariateNormal::new(&[0.0, 0.0], &Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap()).is_err()
    );
}

#[test]
fn wishart_dirichlet_multinomial_moments() {
    let mut rng = Pcg64::seed_from_u64(99);
    // Wishart: E[W] = df·Σ, Var[W_ij] = df (Σ_ij² + Σ_ii Σ_jj).
    let sigma = Matrix::from_rows(&[vec![2.0, 0.6], vec![0.6, 1.0]]).unwrap();
    let df = 6.5;
    let n = 60_000;
    let mut mean = Matrix::zeros(2, 2);
    let mut m2 = Matrix::zeros(2, 2);
    for _ in 0..n {
        let w = wishart(&mut rng, df, &sigma).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                mean[(i, j)] += w[(i, j)] / n as f64;
                m2[(i, j)] += w[(i, j)] * w[(i, j)] / n as f64;
            }
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let var = df * (sigma[(i, j)].powi(2) + sigma[(i, i)] * sigma[(j, j)]);
            assert!((mean[(i, j)] - df * sigma[(i, j)]).abs() < 5.0 * (var / n as f64).sqrt(), "wishart mean {i}{j}");
            let sample_var = m2[(i, j)] - mean[(i, j)].powi(2);
            assert!((sample_var / var - 1.0).abs() < 0.05, "wishart var {i}{j}: {sample_var} vs {var}");
        }
    }
    assert!(wishart(&mut rng, 1.0, &sigma).is_err());

    // Dirichlet: E = α/α₀, Var = E(1−E)/(α₀+1), and the components sum to 1.
    for (alpha, tol) in [(vec![2.0, 3.0, 5.0], 0.08), (vec![0.1, 0.2, 0.05], 0.08), (vec![1e-3, 1.0], 0.6)] {
        let a0: f64 = alpha.iter().sum();
        let n = 80_000;
        let mut m = vec![0.0; alpha.len()];
        let mut q = vec![0.0; alpha.len()];
        for _ in 0..n {
            let x = dirichlet(&mut rng, &alpha).unwrap();
            assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for i in 0..x.len() {
                m[i] += x[i] / n as f64;
                q[i] += x[i] * x[i] / n as f64;
            }
        }
        for i in 0..alpha.len() {
            let e = alpha[i] / a0;
            let v = e * (1.0 - e) / (a0 + 1.0);
            assert!((m[i] - e).abs() < 5.0 * (v / n as f64).sqrt(), "dirichlet mean {alpha:?}[{i}]: {} vs {e}", m[i]);
            assert!(((q[i] - m[i] * m[i]) / v - 1.0).abs() < tol, "dirichlet var {alpha:?}[{i}]");
        }
    }
    assert!(dirichlet(&mut rng, &[1.0, 0.0]).is_err());

    // Multinomial: counts sum to n; E = n p, Cov = n(diag p − ppᵀ).
    let p = [0.2, 0.5, 0.05, 0.25];
    let trials = 1000u64;
    let reps = 40_000;
    let mut m = [0.0; 4];
    let mut c01 = 0.0;
    for _ in 0..reps {
        let x = multinomial(&mut rng, trials, &p).unwrap();
        assert_eq!(x.iter().sum::<u64>(), trials);
        for i in 0..4 {
            m[i] += x[i] as f64 / reps as f64;
        }
        c01 += (x[0] as f64 - 200.0) * (x[1] as f64 - 500.0) / reps as f64;
    }
    for i in 0..4 {
        let v = trials as f64 * p[i] * (1.0 - p[i]);
        assert!((m[i] - trials as f64 * p[i]).abs() < 5.0 * (v / reps as f64).sqrt(), "multinomial mean {i}");
    }
    let want = -(trials as f64) * p[0] * p[1];
    assert!((c01 - want).abs() < 0.05 * want.abs(), "multinomial cov: {c01} vs {want}");
    // Unnormalised weights and zero-probability categories.
    let x = multinomial(&mut rng, 100, &[0.0, 3.0, 0.0, 1.0]).unwrap();
    assert_eq!((x[0], x[2], x[1] + x[3]), (0, 0, 100));
    assert!(multinomial(&mut rng, 5, &[0.0, 0.0]).is_err());
}
