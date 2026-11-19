//! Calculator functions backed by `openstats-math` (feature `formula`):
//!
//! * `NSCORES(c)` — normal scores, Blom's method Φ⁻¹((r − 3/8)/(n + 1/4)),
//!   tied values receive the average of their scores, missing stays missing.
//! * `PDF(x, "dist", p1, …)`, `CDF(x, "dist", p1, …)`, `INVCDF(p, "dist", p1, …)`
//!   and `SFCDF(x, "dist", p1, …)` (upper tail) for every distribution in
//!   [`crate::dist`]. Parameters are scalars and follow Minitab's order;
//!   trailing parameters may be omitted where defaults exist.
//!
//! | name | parameters (defaults) |
//! |---|---|
//! | normal | mean (0), stdev (1) |
//! | t | df |
//! | chisquare | df |
//! | f | df1, df2 |
//! | weibull | shape, scale (1), threshold (0) |
//! | lognormal | location (0), scale (1), threshold (0) |
//! | gamma | shape, scale (1), threshold (0) |
//! | exponential | scale (1), threshold (0) |
//! | logistic / cauchy / laplace / sev / lev | location (0), scale (1) |
//! | loglogistic | location (0), scale (1), threshold (0) |
//! | beta | a, b |
//! | uniform | lower (0), upper (1) |
//! | triangular | lower, mode, upper |
//! | halfnormal | scale (1), threshold (0) |
//! | foldednormal | mean, stdev |
//! | johnsonsu / johnsonsb / johnsonsl | γ, δ, ξ (0), λ (1) |
//! | nct / noncentralt | df (∞ allowed), non-centrality δ |
//! | ncchisquare / noncentralchisquare / ncx2 | df, non-centrality λ (0) |
//! | ncf / noncentralf | df1, df2, non-centrality λ (0) |
//! | ncbeta / noncentralbeta | a, b, non-centrality λ (0) |
//! | binomial | n (trials), p (event probability) |
//! | poisson | mean |
//! | hypergeometric | N (population), M (events in population), n (sample) |
//! | negativebinomial | r (number of events), p — counts *trials*, support r, r+1, … |
//! | geometric | p — counts trials, support 1, 2, … |
//! | integer / discreteuniform | minimum, maximum |
//!
//! ## Random data
//!
//! * `RANDOM(n, "dist", p1, …)` — a column of `n` random draws from any
//!   distribution in the table above (same names and parameter order),
//!   from a *contextual* generator: a per-thread PCG64 stream that is seeded
//!   from the clock on first use, or from `BASE`.
//! * `BASE(seed)` — set the contextual generator's seed (Minitab's `BASE`
//!   command); the following `RANDOM` calls are then reproducible. Returns
//!   the seed.
//! * `RANDOMSEED(seed, n, "dist", p1, …)` — like `RANDOM` but with an
//!   explicit seed: the same arguments always give the same column, on every
//!   platform, independent of any other calls.
//!
//! `n` must be a whole number from 0 to 100 000 000 and the seed a whole
//! number in 0..2⁵³. Discrete distributions return integers; `RANDOM` with
//! `"integer"` is Minitab's `RANDOM; INTEGER` (inclusive limits).
//!
//! For the discrete distributions `PDF` returns the probability mass P(X = x)
//! (0 for non-integers), `CDF` uses ⌊x⌋, and `INVCDF(p)` is the smallest
//! integer k with P(X ≤ k) ≥ p, as in Minitab.

use crate::dist::*;
use crate::random::{Pcg64, Rng, SeedableRng};
use crate::special::erf::norm_ppf;
use openstats_core::column::ColumnData;
use openstats_core::error::{CoreError, Result as CoreResult};
use openstats_core::formula::array::{at, common_len, Arr};
use openstats_core::formula::functions::{Category, FunctionDef, FunctionRegistry};
use openstats_core::formula::EvalCtx;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Register NSCORES, PDF, CDF, SFCDF and INVCDF.
pub fn register(r: &mut FunctionRegistry) {
    r.register(FunctionDef {
        name: "NSCORES",
        min_args: 1,
        max_args: Some(1),
        category: Category::Probability,
        help: "NSCORES(c): normal scores (Blom), ties averaged",
        f: nscores,
    });
    for (name, help, f) in [
        ("PDF", "PDF(x, \"dist\", params…): probability density", pdf as fn(&[Arr], &EvalCtx) -> CoreResult<Arr>),
        ("CDF", "CDF(x, \"dist\", params…): cumulative probability P(X ≤ x)", cdf),
        ("SFCDF", "SFCDF(x, \"dist\", params…): upper-tail probability P(X > x)", sfcdf),
        ("INVCDF", "INVCDF(p, \"dist\", params…): inverse cumulative probability", invcdf),
    ] {
        r.register(FunctionDef { name, min_args: 2, max_args: Some(6), category: Category::Probability, help, f });
    }
    r.register(FunctionDef {
        name: "RANDOM",
        min_args: 2,
        max_args: Some(6),
        category: Category::Probability,
        help: "RANDOM(n, \"dist\", params…): n random draws from the contextual generator (see BASE)",
        f: random,
    });
    r.register(FunctionDef {
        name: "RANDOMSEED",
        min_args: 3,
        max_args: Some(7),
        category: Category::Probability,
        help: "RANDOMSEED(seed, n, \"dist\", params…): n reproducible random draws for the given seed",
        f: random_seeded,
    });
    r.register(FunctionDef {
        name: "BASE",
        min_args: 1,
        max_args: Some(1),
        category: Category::Probability,
        help: "BASE(seed): seed the contextual random generator used by RANDOM",
        f: base,
    });
}

/// Registry with the core functions plus these.
pub fn full_registry() -> FunctionRegistry {
    let mut r = FunctionRegistry::standard();
    register(&mut r);
    r
}

fn bad(msg: impl Into<String>) -> CoreError {
    CoreError::InvalidArgument(msg.into())
}

/// Blom normal scores with averaged ties.
pub fn normal_scores(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).filter(|&i| !x[i].is_nan()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let n = idx.len() as f64;
    let mut out = vec![f64::NAN; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i..j).map(|r| norm_ppf((r as f64 + 1.0 - 0.375) / (n + 0.25))).sum::<f64>() / (j - i) as f64;
        for &k in &idx[i..j] {
            out[k] = avg;
        }
        i = j;
    }
    out
}

fn nscores(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    let x = a[0].to_nums("NSCORES")?;
    Ok(Arr::nums(normal_scores(&x), a[0].scalar))
}

/// Build a distribution from its name and scalar parameters.
pub fn make_distribution(name: &str, p: &[f64]) -> std::result::Result<Box<dyn ContinuousDistribution>, String> {
    let g = |i: usize, default: Option<f64>| -> std::result::Result<f64, String> {
        match p.get(i) {
            Some(&v) => Ok(v),
            None => default.ok_or_else(|| format!("distribution {name:?} needs parameter {}", i + 1)),
        }
    };
    let e = |r: crate::Result<Box<dyn ContinuousDistribution>>| r.map_err(|e| e.to_string());
    let lname = name.trim().to_ascii_lowercase().replace([' ', '-', '_'], "");
    let d: Box<dyn ContinuousDistribution> = match lname.as_str() {
        "normal" => e(Normal::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?,
        "t" | "student" | "studentt" => e(StudentT::new(g(0, None)?).map(|d| Box::new(d) as _))?,
        "chisquare" | "chisq" | "chi2" => e(ChiSquared::new(g(0, None)?).map(|d| Box::new(d) as _))?,
        "f" => e(FisherF::new(g(0, None)?, g(1, None)?).map(|d| Box::new(d) as _))?,
        "weibull" => e(Weibull::new(g(0, None)?, g(1, Some(1.0))?, g(2, Some(0.0))?).map(|d| Box::new(d) as _))?,
        "lognormal" => {
            e(Lognormal::new(g(0, Some(0.0))?, g(1, Some(1.0))?, g(2, Some(0.0))?).map(|d| Box::new(d) as _))?
        }
        "gamma" => e(Gamma::new(g(0, None)?, g(1, Some(1.0))?, g(2, Some(0.0))?).map(|d| Box::new(d) as _))?,
        "exponential" => e(Exponential::new(g(0, Some(1.0))?, g(1, Some(0.0))?).map(|d| Box::new(d) as _))?,
        "logistic" => e(Logistic::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?,
        "loglogistic" => {
            e(Loglogistic::new(g(0, Some(0.0))?, g(1, Some(1.0))?, g(2, Some(0.0))?).map(|d| Box::new(d) as _))?
        }
        "beta" => e(Beta::new(g(0, None)?, g(1, None)?).map(|d| Box::new(d) as _))?,
        "uniform" => e(Uniform::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?,
        "cauchy" => e(Cauchy::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?,
        "laplace" => e(Laplace::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?,
        "triangular" => e(Triangular::new(g(0, None)?, g(1, None)?, g(2, None)?).map(|d| Box::new(d) as _))?,
        "sev" | "smallestextremevalue" => {
            e(SmallestExtremeValue::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?
        }
        "lev" | "largestextremevalue" | "gumbel" => {
            e(LargestExtremeValue::new(g(0, Some(0.0))?, g(1, Some(1.0))?).map(|d| Box::new(d) as _))?
        }
        "halfnormal" => e(HalfNormal::new(g(0, Some(1.0))?, g(1, Some(0.0))?).map(|d| Box::new(d) as _))?,
        "foldednormal" => e(FoldedNormal::new(g(0, None)?, g(1, None)?).map(|d| Box::new(d) as _))?,
        "johnsonsu" | "johnsonsb" | "johnsonsl" => {
            let fam = match &lname[7..] {
                "su" => JohnsonFamily::SU,
                "sb" => JohnsonFamily::SB,
                _ => JohnsonFamily::SL,
            };
            e(Johnson::new(fam, g(0, None)?, g(1, None)?, g(2, Some(0.0))?, g(3, Some(1.0))?)
                .map(|d| Box::new(d) as _))?
        }
        "nct" | "noncentralt" | "noncentralstudent" => {
            e(NonCentralT::new(g(0, None)?, g(1, None)?).map(|d| Box::new(d) as _))?
        }
        "ncchisquare" | "noncentralchisquare" | "ncchisq" | "ncchi2" | "ncx2" => {
            e(NonCentralChiSquared::new(g(0, None)?, g(1, Some(0.0))?).map(|d| Box::new(d) as _))?
        }
        "ncf" | "noncentralf" => {
            e(NonCentralF::new(g(0, None)?, g(1, None)?, g(2, Some(0.0))?).map(|d| Box::new(d) as _))?
        }
        "ncbeta" | "noncentralbeta" => {
            e(NonCentralBeta::new(g(0, None)?, g(1, None)?, g(2, Some(0.0))?).map(|d| Box::new(d) as _))?
        }
        _ => return Err(format!("unknown distribution {name:?}")),
    };
    Ok(d)
}

/// Build a discrete distribution from its name and scalar parameters.
pub fn make_discrete(name: &str, p: &[f64]) -> std::result::Result<Box<dyn DiscreteDistribution>, String> {
    let get = |i: usize| -> std::result::Result<f64, String> {
        p.get(i).copied().ok_or_else(|| format!("distribution {name:?} needs parameter {}", i + 1))
    };
    // Count-like parameters must be whole numbers (n, N, M, r).
    let count = |i: usize, what: &str| -> std::result::Result<u64, String> {
        let v = get(i)?;
        if v >= 0.0 && v == v.floor() && v < 1.8e19 {
            Ok(v as u64)
        } else {
            Err(format!("{name}: {what} must be a non-negative whole number (got {v})"))
        }
    };
    let whole = |i: usize, what: &str| -> std::result::Result<i64, String> {
        let v = get(i)?;
        if v == v.floor() && v.abs() < 9.0e18 {
            Ok(v as i64)
        } else {
            Err(format!("{name}: {what} must be a whole number (got {v})"))
        }
    };
    let lname = name.trim().to_ascii_lowercase().replace([' ', '-', '_'], "");
    let e = |r: crate::Result<Box<dyn DiscreteDistribution>>| r.map_err(|e| e.to_string());
    match lname.as_str() {
        "binomial" | "binom" => e(Binomial::new(count(0, "number of trials")?, get(1)?).map(|d| Box::new(d) as _)),
        "poisson" | "pois" => e(Poisson::new(get(0)?).map(|d| Box::new(d) as _)),
        "hypergeometric" | "hyper" => e(Hypergeometric::new(
            count(0, "population size")?,
            count(1, "number of events in the population")?,
            count(2, "sample size")?,
        )
        .map(|d| Box::new(d) as _)),
        "negativebinomial" | "negbinomial" | "nbinom" => {
            e(NegativeBinomial::new(count(0, "number of events")?, get(1)?).map(|d| Box::new(d) as _))
        }
        "geometric" | "geom" => e(Geometric::new(get(0)?).map(|d| Box::new(d) as _)),
        "integer" | "discreteuniform" | "randint" => {
            e(DiscreteUniform::new(whole(0, "minimum")?, whole(1, "maximum")?).map(|d| Box::new(d) as _))
        }
        _ => Err(format!("unknown distribution {name:?}")),
    }
}

/// A continuous or discrete distribution, as named in a calculator call.
#[allow(missing_debug_implementations)]
pub enum AnyDistribution {
    Continuous(Box<dyn ContinuousDistribution>),
    Discrete(Box<dyn DiscreteDistribution>),
}

impl AnyDistribution {
    /// Resolve a distribution name (continuous names first, then discrete).
    pub fn from_name(name: &str, p: &[f64]) -> std::result::Result<Self, String> {
        match make_distribution(name, p) {
            Ok(d) => Ok(AnyDistribution::Continuous(d)),
            Err(e) if !e.starts_with("unknown distribution") => Err(e),
            Err(_) => make_discrete(name, p).map(AnyDistribution::Discrete),
        }
    }
    /// Density (continuous) or probability mass (discrete).
    pub fn pdf(&self, x: f64) -> f64 {
        match self {
            AnyDistribution::Continuous(d) => d.pdf(x),
            AnyDistribution::Discrete(d) => d.pmf(x),
        }
    }
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            AnyDistribution::Continuous(d) => d.cdf(x),
            AnyDistribution::Discrete(d) => d.cdf(x),
        }
    }
    pub fn sf(&self, x: f64) -> f64 {
        match self {
            AnyDistribution::Continuous(d) => d.sf(x),
            AnyDistribution::Discrete(d) => d.sf(x),
        }
    }
    pub fn quantile(&self, p: f64) -> f64 {
        match self {
            AnyDistribution::Continuous(d) => d.quantile(p),
            AnyDistribution::Discrete(d) => d.quantile(p),
        }
    }
    /// One random variate.
    pub fn sample(&self, rng: &mut dyn Rng) -> f64 {
        match self {
            AnyDistribution::Continuous(d) => d.sample(rng),
            AnyDistribution::Discrete(d) => d.sample(rng),
        }
    }
}

fn dist_from_args(a: &[Arr]) -> CoreResult<AnyDistribution> {
    dist_from_slice(&a[1], &a[2..])
}

fn dist_from_slice(name_arg: &Arr, param_args: &[Arr]) -> CoreResult<AnyDistribution> {
    let name = match &name_arg.data {
        ColumnData::Text(v) if name_arg.scalar || v.len() == 1 => v[0].clone(),
        _ => return Err(bad("the distribution must be given by name in quotes, e.g. \"normal\"")),
    };
    let mut params = Vec::new();
    for (i, arg) in param_args.iter().enumerate() {
        if !arg.scalar && arg.len() != 1 {
            return Err(bad(format!("distribution parameter {} must be a single value", i + 1)));
        }
        params.push(arg.to_nums("parameter")?[0]);
    }
    AnyDistribution::from_name(&name, &params).map_err(bad)
}

fn apply(a: &[Arr], f: impl Fn(&AnyDistribution, f64) -> f64) -> CoreResult<Arr> {
    let d = dist_from_args(a)?;
    let (n, scalar) = common_len(&[&a[0]], "distribution function")?;
    let x = a[0].to_nums("x")?;
    Ok(Arr::nums((0..n).map(|i| at(&x, i)).map(|v| if v.is_nan() { f64::NAN } else { f(&d, v) }).collect(), scalar))
}

static STREAM: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static CONTEXT_RNG: RefCell<Option<Pcg64>> = const { RefCell::new(None) };
}

/// Set the contextual generator used by `RANDOM` on this thread (Minitab's
/// `BASE`).
pub fn set_base(seed: u64) {
    CONTEXT_RNG.with(|c| *c.borrow_mut() = Some(Pcg64::seed_from_u64(seed)));
}

fn scalar_whole(a: &Arr, what: &str, max: f64) -> CoreResult<u64> {
    if !a.scalar && a.len() != 1 {
        return Err(bad(format!("{what} must be a single value")));
    }
    let v = a.to_nums(what)?[0];
    if !(v >= 0.0 && v <= max && v == v.floor()) {
        return Err(bad(format!("{what} must be a whole number from 0 to {max} (got {v})")));
    }
    Ok(v as u64)
}

fn draw_column(n: u64, d: &AnyDistribution, rng: &mut dyn Rng) -> Arr {
    Arr::nums((0..n).map(|_| d.sample(rng)).collect(), false)
}

fn random(a: &[Arr], ctx: &EvalCtx) -> CoreResult<Arr> {
    let n = scalar_whole(&a[0], "the number of values n", 1e8)?;
    let d = dist_from_slice(&a[1], &a[2..])?;
    Ok(CONTEXT_RNG.with(|c| {
        let mut slot = c.borrow_mut();
        let rng = slot.get_or_insert_with(|| {
            // Unseeded: mix the clock (when the host provides one) with a
            // process-wide counter so threads and calls never share a stream.
            let clock = ctx.now.unwrap_or(0) as u64;
            let k = STREAM.fetch_add(1, Ordering::Relaxed);
            Pcg64::with_stream(clock ^ 0x5DEE_CE66_D1CE_4E5B, k)
        });
        draw_column(n, &d, rng)
    }))
}

fn random_seeded(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    let seed = scalar_whole(&a[0], "the seed", 9_007_199_254_740_992.0)?;
    let n = scalar_whole(&a[1], "the number of values n", 1e8)?;
    let d = dist_from_slice(&a[2], &a[3..])?;
    let mut rng = Pcg64::seed_from_u64(seed);
    Ok(draw_column(n, &d, &mut rng))
}

fn base(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    let seed = scalar_whole(&a[0], "the seed", 9_007_199_254_740_992.0)?;
    set_base(seed);
    Ok(Arr::num(seed as f64))
}

fn pdf(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    apply(a, |d, x| d.pdf(x))
}
fn cdf(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    apply(a, |d, x| d.cdf(x))
}
fn sfcdf(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    apply(a, |d, x| d.sf(x))
}
fn invcdf(a: &[Arr], _: &EvalCtx) -> CoreResult<Arr> {
    apply(a, |d, p| d.quantile(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_core::column::Column;
    use openstats_core::worksheet::Worksheet;

    fn eval(f: &str, ws: &Worksheet) -> CoreResult<Arr> {
        let reg = full_registry();
        EvalCtx::new(ws, &reg).eval_str(f)
    }

    #[test]
    fn calculator_functions() {
        let ws = Worksheet::with_columns("s", vec![Column::numeric("x", vec![3.0, 1.0, 2.0, 2.0, f64::NAN])]).unwrap();
        let ns = eval("NSCORES(x)", &ws).unwrap().data.to_f64_vec();
        // n = 4: scores for ranks 1..4 are Φ⁻¹((r − 3/8)/4.25); ranks 2 and 3 tie.
        let s = |r: f64| norm_ppf((r - 0.375) / 4.25);
        assert!((ns[0] - s(4.0)).abs() < 1e-15 && (ns[1] - s(1.0)).abs() < 1e-15);
        assert!((ns[2] - 0.5 * (s(2.0) + s(3.0))).abs() < 1e-15 && ns[2] == ns[3]);
        assert!(ns[4].is_nan());
        let v = |f: &str| eval(f, &ws).unwrap().data.to_f64_vec()[0];
        assert!((v("CDF(1.96, \"normal\")") - 0.975_002_104_851_779_6).abs() < 1e-15);
        assert!((v("INVCDF(0.975, \"t\", 10)") - 2.228_138_851_986_274).abs() < 1e-13);
        assert!((v("SFCDF(30, \"Chisquare\", 10)") - 8.566_412_107_753_01e-4).abs() < 1e-15);
        assert!((v("PDF(0, \"normal\", 0, 2)") - 0.199_471_140_200_716_35).abs() < 1e-16);
        assert!((v("INVCDF(0.5, \"weibull\", 2, 3)") - 3.0 * 2f64.ln().sqrt()).abs() < 1e-15);
        assert_eq!(eval("CDF(x, \"exponential\", 1)", &ws).unwrap().len(), 5);
        assert!(eval("CDF(1, \"nope\")", &ws).is_err());
        assert!(eval("CDF(1, \"t\")", &ws).is_err(), "missing df");
        assert!(eval("CDF(1, \"normal\", x)", &ws).is_err(), "column parameter");
        assert!(eval("CDF(1, \"normal\", 0, -1)", &ws).is_err(), "invalid sigma");
    }

    #[test]
    fn discrete_and_non_central_functions() {
        let ws = Worksheet::with_columns("s", vec![Column::numeric("x", vec![0.0, 1.0, 2.0, 2.5, f64::NAN])]).unwrap();
        let v = |f: &str| eval(f, &ws).unwrap().data.to_f64_vec()[0];
        // Binomial(10, 0.3): pmf(3) = 120·0.3³·0.7⁷ = 0.266827932, cdf(3) = 0.6496107184.
        assert!((v("PDF(3, \"binomial\", 10, 0.3)") - 0.266_827_932).abs() < 1e-12);
        assert!((v("CDF(3, \"binomial\", 10, 0.3)") - 0.649_610_718_4).abs() < 1e-12);
        assert!((v("SFCDF(3, \"binomial\", 10, 0.3)") - 0.350_389_281_6).abs() < 1e-12);
        assert_eq!(v("INVCDF(0.5, \"binomial\", 10, 0.3)"), 3.0);
        assert!((v("CDF(2, \"Poisson\", 2)") - 5.0 * (-2.0f64).exp()).abs() < 1e-15);
        assert!((v("PDF(2, \"hypergeometric\", 20, 7, 5)") - 6006.0 / 15504.0).abs() < 1e-14);
        assert!((v("PDF(5, \"negative binomial\", 3, 0.5)") - 0.1875).abs() < 1e-15);
        assert!((v("CDF(3, \"geometric\", 0.25)") - 0.578_125).abs() < 1e-15);
        assert!((v("PDF(4, \"integer\", 1, 6)") - 1.0 / 6.0).abs() < 1e-16);
        assert_eq!(v("INVCDF(0.3, \"discreteuniform\", 1, 10)"), 3.0);
        // Non-integers carry no mass and the cdf steps at integers.
        assert_eq!(v("PDF(2.5, \"poisson\", 3)"), 0.0);
        assert_eq!(v("CDF(2.5, \"poisson\", 3)"), v("CDF(2, \"poisson\", 3)"));
        // Element-wise over a column, missing stays missing.
        let col = eval("PDF(x, \"poisson\", 1)", &ws).unwrap().data.to_f64_vec();
        assert!((col[1] - (-1.0f64).exp()).abs() < 1e-16 && col[3] == 0.0 && col[4].is_nan());
        // Non-central: λ = 0 equals the central distribution; known SciPy values otherwise.
        assert!((v("CDF(1.5, \"nct\", 10, 2)") - 0.304_785_447_376_042_1).abs() < 1e-13);
        assert!((v("CDF(5, \"ncchisquare\", 3, 2)") - 0.593_405_180_083_155_5).abs() < 1e-13);
        assert!((v("CDF(2, \"ncf\", 4, 10, 3)") - 0.598_630_606_918_366_4).abs() < 1e-13);
        assert!((v("CDF(0.3, \"ncbeta\", 2, 2, 0)") - v("CDF(0.3, \"beta\", 2, 2)")).abs() < 1e-15);
        let q = v("INVCDF(0.9, \"ncf\", 4, 10, 3)");
        assert!((v(&format!("CDF({q}, \"ncf\", 4, 10, 3)")) - 0.9).abs() < 1e-12);
        // Parameter validation.
        assert!(eval("CDF(1, \"binomial\", 10.5, 0.3)", &ws).is_err(), "n must be whole");
        assert!(eval("CDF(1, \"binomial\", 10, 1.5)", &ws).is_err());
        assert!(eval("CDF(1, \"binomial\", 10)", &ws).is_err(), "missing p");
        assert!(eval("CDF(1, \"hypergeometric\", 5, 7, 2)", &ws).is_err());
        assert!(eval("CDF(1, \"integer\", 5, 2)", &ws).is_err());
        assert!(eval("CDF(1, \"ncf\", 4)", &ws).is_err());
        assert!(eval("CDF(1, \"nct\", 0, 1)", &ws).is_err());
        // Direct API.
        assert!(AnyDistribution::from_name("poisson", &[-1.0]).is_err());
        assert!(matches!(AnyDistribution::from_name("normal", &[]).unwrap(), AnyDistribution::Continuous(_)));
        assert!(matches!(AnyDistribution::from_name("geom", &[0.5]).unwrap(), AnyDistribution::Discrete(_)));
        assert!(AnyDistribution::from_name("bogus", &[]).is_err());
    }

    #[test]
    fn random_data_functions() {
        let ws = Worksheet::with_columns("s", vec![Column::numeric("x", vec![1.0])]).unwrap();
        let col = |f: &str| eval(f, &ws).unwrap().data.to_f64_vec();
        // Explicit seed: reproducible, seed-sensitive, correct length.
        let a = col("RANDOMSEED(42, 5, \"normal\", 10, 2)");
        assert_eq!(a.len(), 5);
        assert_eq!(a, col("RANDOMSEED(42, 5, \"normal\", 10, 2)"));
        assert_ne!(a, col("RANDOMSEED(43, 5, \"normal\", 10, 2)"));
        // A longer column extends the shorter one (same stream).
        assert_eq!(&col("RANDOMSEED(42, 8, \"normal\", 10, 2)")[..5], &a[..]);
        // BASE makes the contextual generator reproducible.
        assert_eq!(col("BASE(7)"), vec![7.0]);
        let b1 = col("RANDOM(6, \"poisson\", 4)");
        col("BASE(7)");
        assert_eq!(b1, col("RANDOM(6, \"poisson\", 4)"));
        assert!(b1.iter().all(|v| *v >= 0.0 && *v == v.floor()));
        // Moments of a large draw: Normal(10, 2) and Integer(1, 6).
        let big = col("RANDOMSEED(1, 100000, \"normal\", 10, 2)");
        let mean = big.iter().sum::<f64>() / big.len() as f64;
        let var = big.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / big.len() as f64;
        assert!((mean - 10.0).abs() < 0.03 && (var - 4.0).abs() < 0.08, "mean {mean} var {var}");
        let dice = col("RANDOMSEED(2, 60000, \"integer\", 1, 6)");
        for face in 1..=6 {
            let c = dice.iter().filter(|v| **v == face as f64).count() as f64;
            assert!((c - 10_000.0).abs() < 400.0, "face {face}: {c}");
        }
        assert!(dice.iter().all(|v| (1.0..=6.0).contains(v)));
        // Unseeded contextual draws differ between consecutive calls.
        let u1 = col("RANDOM(3, \"uniform\", 0, 1)");
        let u2 = col("RANDOM(3, \"uniform\", 0, 1)");
        assert_ne!(u1, u2);
        assert_eq!(col("RANDOM(0, \"normal\")").len(), 0);
        // Every distribution name accepted by PDF is accepted by RANDOM.
        for spec in [
            "\"t\", 5",
            "\"chisquare\", 3",
            "\"f\", 3, 7",
            "\"weibull\", 2, 3",
            "\"lognormal\", 0, 1",
            "\"gamma\", 2, 3",
            "\"exponential\", 2",
            "\"logistic\", 0, 1",
            "\"loglogistic\", 0, 1",
            "\"cauchy\", 0, 1",
            "\"laplace\", 0, 1",
            "\"sev\", 0, 1",
            "\"lev\", 0, 1",
            "\"beta\", 2, 3",
            "\"uniform\", 0, 1",
            "\"triangular\", 0, 1, 3",
            "\"halfnormal\", 1",
            "\"foldednormal\", 1, 2",
            "\"johnsonsu\", 0.5, 1.3, 2, 1.5",
            "\"nct\", 10, 1",
            "\"ncchisquare\", 3, 2",
            "\"ncf\", 4, 10, 3",
            "\"ncbeta\", 2, 2, 1",
            "\"binomial\", 10, 0.3",
            "\"poisson\", 3",
            "\"hypergeometric\", 20, 7, 5",
            "\"negativebinomial\", 3, 0.5",
            "\"geometric\", 0.25",
        ] {
            let v = col(&format!("RANDOMSEED(3, 20, {spec})"));
            assert!(v.len() == 20 && v.iter().all(|x| x.is_finite()), "{spec}");
        }
        // Validation.
        assert!(eval("RANDOM(2.5, \"normal\")", &ws).is_err());
        assert!(eval("RANDOM(-1, \"normal\")", &ws).is_err());
        assert!(eval("RANDOM(5, \"nosuch\")", &ws).is_err());
        assert!(eval("RANDOM(5, \"binomial\", 10)", &ws).is_err());
        assert!(eval("RANDOMSEED(1.5, 5, \"normal\")", &ws).is_err());
        assert!(eval("BASE(-3)", &ws).is_err());
    }
}
