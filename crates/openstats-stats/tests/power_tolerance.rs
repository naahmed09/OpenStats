//! Phase 2c reference tests: power / sample size and tolerance intervals
//! against SciPy and independently coded brute-force references
//! (`gen/stats_ref3.py`).

mod common;
use common::data3::*;
use openstats_stats::hypothesis::Alternative;
use openstats_stats::power::*;
use openstats_stats::tolerance::*;

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{what}: got {a}, expected {b}");
}

fn alt(s: &str) -> Alternative {
    match s {
        "two" => Alternative::TwoSided,
        "less" => Alternative::Less,
        _ => Alternative::Greater,
    }
}

#[test]
fn t_test_power_matches_scipy_nct() {
    for &(kind, n, eff, sig, alpha, two, want) in T_POWER {
        let a = if two == 1 { Alternative::TwoSided } else { Alternative::Greater };
        let got = match kind {
            1 => OneSampleT { sigma: sig, alpha, alternative: a }.power(n, eff),
            2 => TwoSampleT { sigma: sig, alpha, alternative: a }.power(n, eff),
            _ => PairedT { sigma_diff: sig, alpha, alternative: a }.power(n, eff),
        }
        .unwrap();
        close(got, want, 1e-10, &format!("t power kind {kind} n {n}"));
    }
}

#[test]
fn anova_power_matches_scipy_ncf() {
    for &(k, n, delta, sig, alpha, want) in ANOVA_POWER {
        let got = OneWayAnova { groups: k, sigma: sig, alpha }.power(n, delta).unwrap();
        close(got, want, 1e-10, &format!("anova k {k} n {n}"));
    }
}

#[test]
fn variance_power_matches_scipy() {
    for &(kind, n, ratio, alpha, a, want) in VAR_POWER {
        let got = if kind == 1 {
            OneVariance { alpha, alternative: alt(a), direction: Direction::Increase }.power(n, ratio)
        } else {
            TwoVariances { alpha, alternative: alt(a), direction: Direction::Increase }.power(n, ratio)
        }
        .unwrap();
        close(got, want, 1e-11, &format!("variance kind {kind} n {n}"));
    }
}

#[test]
fn exact_one_proportion_power_matches_binomtest_region() {
    for &(n, p0, p1, alpha, a, want) in PROP1_EXACT {
        let m = OneProportion::new(p0, alpha, alt(a), DiscreteMethod::Exact);
        close(m.power(n, p1).unwrap(), want, 1e-11, &format!("exact 1-prop n {n}"));
    }
    for &(p0, p1, alpha, a, target, n) in PROP1_EXACT_N {
        let m = OneProportion::new(p0, alpha, alt(a), DiscreteMethod::Exact);
        assert_eq!(m.sample_size(p1, target).unwrap(), n, "p0 {p0} p1 {p1}");
    }
}

#[test]
fn normal_two_proportion_power_matches_formula() {
    for &(n, p1, p2, alpha, a, want) in PROP2_NORMAL {
        let m = TwoProportions::new(p2, alpha, alt(a), DiscreteMethod::NormalApproximation);
        close(m.power(n, p1).unwrap(), want, 1e-12, &format!("2-prop n {n}"));
    }
}

#[test]
fn fisher_power_matches_scipy_fisher_exact_enumeration() {
    for &(n, p1, p2, alpha, a, want) in FISHER_POWER {
        let m = TwoProportions::new(p2, alpha, alt(a), DiscreteMethod::Exact);
        close(m.power(n, p1).unwrap(), want, 1e-11, &format!("Fisher n {n}"));
    }
}

#[test]
fn poisson_exact_power_matches_reference() {
    for &(n, len, l0, l1, alpha, a, want) in POIS1_EXACT {
        let mut m = OnePoisson::new(l0, alpha, alt(a), DiscreteMethod::Exact);
        m.length = len;
        close(m.power(n, l1).unwrap(), want, 1e-10, &format!("1-Poisson n {n}"));
    }
    for &(n, len, l2, l1, alpha, a, want) in POIS2_EXACT {
        let mut m = TwoPoisson::new(l2, alpha, alt(a), DiscreteMethod::Exact);
        m.length = len;
        close(m.power(n, l1).unwrap(), want, 1e-9, &format!("2-Poisson n {n}"));
    }
}

#[test]
fn factorial_power_matches_scipy_nct() {
    for &(base, reps, cp, blocks, omit, sig, alpha, eff, df, want) in FACTORIAL {
        let design = if base.is_power_of_two() {
            FactorialDesign { kind: FactorialKind::Full, factors: base.trailing_zeros() as usize, base_runs: base }
        } else {
            FactorialDesign::plackett_burman(base, 5).unwrap()
        };
        let m = Factorial { design, sigma: sig, alpha, center_points: cp, blocks, omitted_terms: omit };
        assert_eq!(m.error_df(reps).unwrap(), df);
        close(m.power(reps, eff).unwrap(), want, 1e-10, &format!("factorial base {base}"));
    }
}

#[test]
fn tost_power_matches_independent_integration_order() {
    for &(design, n, d, lo, hi, sig, alpha, want) in TOST_POWER {
        let design = match design {
            1 => TostDesign::OneSample,
            2 => TostDesign::TwoSample,
            _ => TostDesign::Paired,
        };
        let got = Tost::new(design, lo, hi, sig, alpha).power(n, d).unwrap();
        close(got, want, 1e-9, &format!("TOST n {n}"));
    }
    for &(design, d, lo, hi, sig, alpha, target, n) in TOST_N {
        let design = if design == 1 { TostDesign::OneSample } else { TostDesign::TwoSample };
        let m = Tost::new(design, lo, hi, sig, alpha);
        assert_eq!(m.sample_size(d, target).unwrap(), n);
    }
}

#[test]
fn one_sided_tolerance_factors_match_scipy_nct() {
    for &(n, p, g, want) in TOL_ONE_SIDED {
        for sides in [ToleranceSides::Upper, ToleranceSides::Lower] {
            let k = normal_tolerance_factor(n, p, g, sides, NormalMethod::Exact).unwrap();
            close(k, want, 1e-9, &format!("one-sided k n {n}"));
        }
    }
}

#[test]
fn two_sided_tolerance_factors_match_independent_integration() {
    for &(n, p, g, want) in TOL_TWO_SIDED {
        let k = normal_tolerance_factor(n, p, g, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
        close(k, want, 5e-9, &format!("two-sided k n {n} p {p}"));
        // Howe's approximation is within a few tenths of a percent for n ≥ 10.
        if n >= 10 {
            let h = normal_tolerance_factor(n, p, g, ToleranceSides::TwoSided, NormalMethod::Howe).unwrap();
            assert!((h - k).abs() / k < 0.01, "Howe n {n}: {h} vs {k}");
        }
    }
}

#[test]
fn nonparametric_confidence_matches_scipy_binomial() {
    for &(n, p, l, u, want) in NP_CONF {
        close(nonparametric_confidence(n, p, Some(l), Some(u)).unwrap(), want, 1e-12, &format!("np conf n {n}"));
    }
}
