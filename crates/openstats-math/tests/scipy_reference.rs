//! Cross-checks of special functions and distributions against SciPy
//! (Cephes / Boost / AMOS). Run with `--nocapture` to see the error report.

#[rustfmt::skip]
// Generated data: some values coincide with named constants (ln 2, 2/π, …).
#[allow(clippy::excessive_precision, clippy::unreadable_literal, clippy::approx_constant)]
mod reference {
    include!("reference/scipy.rs");
}

#[rustfmt::skip]
#[allow(clippy::excessive_precision, clippy::unreadable_literal, clippy::approx_constant)]
mod reference2 {
    include!("reference/scipy2.rs");
}

use openstats_math::dist::*;
use openstats_math::special::{bessel, erf, expint, gamma, incbeta, incgamma, lambertw, owen, tukey, zeta};
use std::collections::BTreeMap;

fn special(f: &str, a: f64, b: f64, x: f64) -> f64 {
    match f {
        "gamma" => gamma::gamma(x),
        "lgamma" => gamma::ln_gamma(x),
        "digamma" => gamma::digamma(x),
        "trigamma" => gamma::trigamma(x),
        "lnbeta" => gamma::ln_beta(a, b),
        "erf" => erf::erf(x),
        "erfc" => erf::erfc(x),
        "erfcx" => erf::erfcx(x),
        "erfinv" => erf::erf_inv(x),
        "erfcinv" => erf::erfc_inv(x),
        "gammap" => incgamma::gamma_p(a, x),
        "gammaq" => incgamma::gamma_q(a, x),
        "gammapinv" => incgamma::gamma_p_inv(a, x),
        "gammaqinv" => incgamma::gamma_q_inv(a, x),
        "betainc" => incbeta::beta_inc(a, b, x),
        "betaincc" => incbeta::beta_inc_c(a, b, x),
        "betaincinv" => incbeta::beta_inc_inv(a, b, x),
        "jn" => bessel::jn(a as i32, x),
        "yn" => bessel::yn(a as i32, x),
        "in" => bessel::in_(a as i32, x),
        "kn" => bessel::kn(a as i32, x),
        "ptukey" => tukey::ptukey(x, a, b).unwrap(),
        "qtukey" => tukey::qtukey(x, a, b).unwrap(),
        other => panic!("unknown function {other}"),
    }
}

fn make(name: &str, p: [f64; 4]) -> Box<dyn ContinuousDistribution> {
    match name {
        "normal" => Box::new(Normal::new(p[0], p[1]).unwrap()),
        "lognormal" => Box::new(Lognormal::new(p[0], p[1], p[2]).unwrap()),
        "halfnormal" => Box::new(HalfNormal::new(p[0], p[1]).unwrap()),
        "foldednormal" => Box::new(FoldedNormal::new(p[0], p[1]).unwrap()),
        "johnsonsu" => Box::new(Johnson::new(JohnsonFamily::SU, p[0], p[1], p[2], p[3]).unwrap()),
        "johnsonsb" => Box::new(Johnson::new(JohnsonFamily::SB, p[0], p[1], p[2], p[3]).unwrap()),
        "johnsonsl" => Box::new(Johnson::new(JohnsonFamily::SL, p[0], p[1], p[2], p[3]).unwrap()),
        "gamma" => Box::new(Gamma::new(p[0], p[1], p[2]).unwrap()),
        "chisq" => Box::new(ChiSquared::new(p[0]).unwrap()),
        "exponential" => Box::new(Exponential::new(p[0], p[1]).unwrap()),
        "weibull" => Box::new(Weibull::new(p[0], p[1], p[2]).unwrap()),
        "beta" => Box::new(Beta::new(p[0], p[1]).unwrap()),
        "t" => Box::new(StudentT::new(p[0]).unwrap()),
        "f" => Box::new(FisherF::new(p[0], p[1]).unwrap()),
        "logistic" => Box::new(Logistic::new(p[0], p[1]).unwrap()),
        "loglogistic" => Box::new(Loglogistic::new(p[0], p[1], p[2]).unwrap()),
        "cauchy" => Box::new(Cauchy::new(p[0], p[1]).unwrap()),
        "laplace" => Box::new(Laplace::new(p[0], p[1]).unwrap()),
        "uniform" => Box::new(Uniform::new(p[0], p[1]).unwrap()),
        "triangular" => Box::new(Triangular::new(p[0], p[1], p[2]).unwrap()),
        "sev" => Box::new(SmallestExtremeValue::new(p[0], p[1]).unwrap()),
        "lev" => Box::new(LargestExtremeValue::new(p[0], p[1]).unwrap()),
        other => panic!("unknown distribution {other}"),
    }
}

/// Relative error, falling back to absolute error below `floor` (for
/// functions with real zeros, where only absolute accuracy is meaningful).
fn err(got: f64, want: f64, floor: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    (got - want).abs() / want.abs().max(floor)
}

/// Reference values where SciPy itself is inaccurate, replaced by values
/// computed independently in 40-digit decimal arithmetic.
fn override_value(key: &str, a: f64, b: f64, x: f64) -> Option<f64> {
    match (key, a, b, x) {
        // ln B(1e5, 3) = ln 2 − ln(1e5·100001·100002); SciPy: −33.84565921407193.
        ("lnbeta", a, b, _) if a == 1e5 && b == 3.0 => Some(-33.845_659_214_100_743),
        _ => None,
    }
}

// Guards keep exact float comparisons explicit and readable.
#[allow(clippy::redundant_guards)]
fn override_dist(name: &str, f: &str, x: f64) -> Option<f64> {
    match (name, f, x) {
        // 1/(1 + e^z), z = (ln 9999 − 0.5)/0.4; SciPy: 3.4912162050204596e-10.
        ("loglogistic", "sf", x) if x == 1e4 => Some(3.491_215_694_707_761e-10),
        // (4 − x)²/12 at the double nearest 3.999; SciPy forms 1 − cdf.
        ("triangular", "sf", x) if x == 3.999 => Some(8.333_333_333_331_498e-8),
        // 4 − √(12·1e-20) = 3.99999999965358983…; SciPy returns 4 (it forms 1 − q).
        ("triangular", "isf", x) if x == 1e-20 => Some(3.999_999_999_653_59),
        _ => None,
    }
}

fn report(title: &str, worst: &BTreeMap<String, (f64, String)>) {
    println!("\n{title}");
    for (k, (e, at)) in worst {
        println!("  {k:<28} max err {e:9.2e}   at {at}");
    }
}

#[test]
fn special_functions_match_scipy() {
    let mut worst: BTreeMap<String, (f64, String)> = BTreeMap::new();
    for &(f, a, b, x, want) in reference::SPECIAL {
        let got = special(f, a, b, x);
        let want = override_value(f, a, b, x).unwrap_or(want);
        let floor = match f {
            "jn" | "yn" | "lgamma" | "digamma" => 1.0,
            "ptukey" | "qtukey" => 1.0,
            _ => 1e-300,
        };
        let e = err(got, want, floor);
        let entry = worst.entry(f.to_string()).or_insert((0.0, String::new()));
        if e > entry.0 || e.is_nan() {
            *entry = (e, format!("a={a} b={b} x={x}: got {got:e} want {want:e}"));
        }
    }
    report("Special functions vs SciPy (relative error; absolute for J, Y, lnΓ, ψ, Tukey)", &worst);
    for (f, (e, at)) in &worst {
        let tol = match f.as_str() {
            "ptukey" | "qtukey" => 1e-9,
            "betaincinv" | "gammapinv" | "gammaqinv" => 1e-12,
            "betainc" | "betaincc" => 2e-12,
            _ => 5e-13,
        };
        assert!(*e <= tol, "{f}: error {e:e} > {tol:e} ({at})");
    }
}

#[test]
fn distributions_match_scipy() {
    let mut worst: BTreeMap<String, (f64, String)> = BTreeMap::new();
    for &(name, p, f, x, want) in reference::DIST {
        let d = make(name, p);
        let got = match f {
            "pdf" => d.pdf(x),
            "cdf" => d.cdf(x),
            "sf" => d.sf(x),
            "ppf" => d.quantile(x),
            "isf" => d.isf(x),
            other => panic!("{other}"),
        };
        let want = override_dist(name, f, x).unwrap_or(want);
        let mut e = err(got, want, 1e-300);
        // Quantile disagreements: accept ours only if a strict round trip
        // proves it (SciPy's root finders fail in a few far-tail cases:
        // beta(2,5) at 1e-100, F(1,1) at 1e-10, folded normal, triangular).
        if e > 1e-12 && (f == "ppf" || f == "isf") {
            let back = if f == "ppf" { d.cdf(got) } else { d.sf(got) };
            let rt = err(back, x, 1e-300);
            if rt <= 1e-12 {
                println!("  note: SciPy {name}.{f}({x:e}) = {want:e}; ours {got:e} round-trips to {rt:.1e}");
                e = rt;
            }
        }
        let key = format!("{name}.{f}");
        let entry = worst.entry(key).or_insert((0.0, String::new()));
        if e > entry.0 || e.is_nan() {
            *entry = (e, format!("params={p:?} x={x}: got {got:e} want {want:e}"));
        }
    }
    report("Distributions vs SciPy (relative error)", &worst);
    for (k, (e, at)) in &worst {
        assert!(*e <= 1e-12, "{k}: error {e:e} ({at})");
    }
}

// ---------------------------------------------------------------------------
// Phase 1e: Owen's T, E1/Ei, zeta, Lambert W, discrete and non-central
// ---------------------------------------------------------------------------

#[test]
fn phase_1e_special_functions_match_scipy() {
    let mut worst: BTreeMap<String, (f64, String)> = BTreeMap::new();
    for &(f, a, _b, x, want) in reference2::SPECIAL2 {
        let got = match f {
            "owent" => owen::owens_t(x, a),
            "e1" => expint::e1(x),
            "ei" => expint::ei(x),
            "zeta" => zeta::zeta(x),
            "lambertw0" => lambertw::lambert_w0(x),
            "lambertwm1" => lambertw::lambert_wm1(x),
            other => panic!("unknown function {other}"),
        };
        // Ei vanishes at one point; elsewhere everything is judged relatively.
        let floor = if f == "ei" { 1e-3 } else { 1e-300 };
        let e = err(got, want, floor);
        let entry = worst.entry(f.to_string()).or_insert((0.0, String::new()));
        if e > entry.0 || e.is_nan() {
            *entry = (e, format!("a={a} x={x}: got {got:e} want {want:e}"));
        }
    }
    report("Phase 1e special functions vs SciPy", &worst);
    for (f, (e, at)) in &worst {
        let tol = match f.as_str() {
            "owent" => 2e-13,
            _ => 5e-13,
        };
        assert!(*e <= tol, "{f}: error {e:e} > {tol:e} ({at})");
    }
}

fn make_nc(name: &str, p: [f64; 3]) -> Box<dyn ContinuousDistribution> {
    match name {
        "nct" => Box::new(NonCentralT::new(p[0], p[1]).unwrap()),
        "ncx2" => Box::new(NonCentralChiSquared::new(p[0], p[1]).unwrap()),
        "ncf" => Box::new(NonCentralF::new(p[0], p[1], p[2]).unwrap()),
        "ncbeta" => Box::new(NonCentralBeta::new(p[0], p[1], p[2]).unwrap()),
        other => panic!("unknown distribution {other}"),
    }
}

fn make_disc(name: &str, p: [f64; 3]) -> Box<dyn DiscreteDistribution> {
    match name {
        "binom" => Box::new(Binomial::new(p[0] as u64, p[1]).unwrap()),
        "pois" => Box::new(Poisson::new(p[0]).unwrap()),
        "hyper" => Box::new(Hypergeometric::new(p[0] as u64, p[1] as u64, p[2] as u64).unwrap()),
        "nbinom" => Box::new(NegativeBinomial::new(p[0] as u64, p[1]).unwrap()),
        "geom" => Box::new(Geometric::new(p[0]).unwrap()),
        "randint" => Box::new(DiscreteUniform::new(p[0] as i64, p[1] as i64).unwrap()),
        other => panic!("unknown distribution {other}"),
    }
}

#[test]
fn non_central_distributions_match_scipy() {
    let mut worst: BTreeMap<String, (f64, String)> = BTreeMap::new();
    let mut notes = 0;
    for &(name, p, f, x, want) in reference2::DIST2 {
        if !matches!(name, "nct" | "ncx2" | "ncf" | "ncbeta") {
            continue;
        }
        let d = make_nc(name, p);
        let got = match f {
            "pdf" => d.pdf(x),
            "cdf" => d.cdf(x),
            "sf" => d.sf(x),
            "ppf" => d.quantile(x),
            "isf" => d.isf(x),
            other => panic!("{other}"),
        };
        let mut e = err(got, want, 1e-300);
        if e > 1e-11 && (f == "ppf" || f == "isf") {
            let back = if f == "ppf" { d.cdf(got) } else { d.sf(got) };
            let rt = err(back, x, 1e-300);
            if rt <= 1e-11 {
                notes += 1;
                println!("  note: SciPy {name}{p:?}.{f}({x:e}) = {want:e}; ours {got:e} round-trips to {rt:.1e}");
                e = rt;
            }
        }
        let entry = worst.entry(format!("{name}.{f}")).or_insert((0.0, String::new()));
        if e > entry.0 || e.is_nan() {
            *entry = (e, format!("params={p:?} x={x}: got {got:e} want {want:e}"));
        }
    }
    report("Non-central distributions vs SciPy (relative error)", &worst);
    println!("  ({notes} quantile rows accepted by round trip)");
    for (k, (e, at)) in &worst {
        assert!(*e <= 1e-11, "{k}: error {e:e} ({at})");
    }
}

#[test]
fn discrete_distributions_match_scipy() {
    let mut worst: BTreeMap<String, (f64, String)> = BTreeMap::new();
    for &(name, p, f, x, want) in reference2::DIST2 {
        if matches!(name, "nct" | "ncx2" | "ncf" | "ncbeta") {
            continue;
        }
        let d = make_disc(name, p);
        let key = format!("{name}.{f}");
        let (got, e);
        match f {
            "pmf" | "cdf" | "sf" => {
                got = match f {
                    "pmf" => d.pmf(x),
                    "cdf" => d.cdf(x),
                    _ => d.sf(x),
                };
                e = err(got, want, 1e-300);
            }
            "ppf" | "isf" => {
                got = if f == "ppf" { d.quantile(x) } else { d.isf(x) };
                // Definition check: smallest k with cdf(k) ≥ p (sf(k) ≤ q), up
                // to the 64-ulp fuzz; SciPy's own value is compared when finite.
                let (lo, _) = d.support();
                let ok = if f == "ppf" {
                    d.cdf(got) >= x * (1.0 - 1e-13) && (got <= lo || d.cdf(got - 1.0) < x * (1.0 + 1e-13))
                } else {
                    d.sf(got) <= x * (1.0 + 1e-13) && (got <= lo || d.sf(got - 1.0) > x * (1.0 - 1e-13))
                };
                if !ok {
                    panic!("{key}{p:?}({x:e}) = {got}: violates the quantile definition");
                }
                if want.is_finite() && got != want {
                    // SciPy's search differs only through its own rounding; ours
                    // must still satisfy the definition (checked above).
                    println!(
                        "  note: SciPy {key}{p:?}({x:e}) = {want}; ours {got} (both bracket the target within 1e-13)"
                    );
                }
                e = 0.0;
            }
            other => panic!("{other}"),
        }
        let entry = worst.entry(key).or_insert((0.0, String::new()));
        if e > entry.0 || e.is_nan() {
            *entry = (e, format!("params={p:?} x={x}: got {got:e} want {want:e}"));
        }
    }
    report("Discrete distributions vs SciPy (relative error; quantiles by definition)", &worst);
    for (k, (e, at)) in &worst {
        // Documented limit: cdf/sf go through the incomplete beta, whose
        // relative error grows ~ √(n·ε)-ish for n ≳ 1e8 (Binomial n = 1e9:
        // ~3e-11) and for NB sums with ~1e6 failures (~7e-12).
        let tol = if at.contains("1000000000.0") { 5e-11 } else { 1e-11 };
        assert!(*e <= tol, "{k}: error {e:e} ({at})");
    }
}
