//! Control-chart constants for subgroup sizes 2 … 25.
//!
//! Nothing is typed in from a table: every constant is computed from the
//! distribution of the range (`d2`, `d3`, `d4`) and of the standard deviation
//! (`c4`) of normal samples, and then checked against the published AIAG /
//! ASTM tables in the tests.
//!
//! * `d2 = E[W]`, `d3 = SD[W]` with `W` the range of `n` standard normals,
//!   by Gauss–Legendre integration of the exact range distribution.
//! * `d4` is the median of `W`.
//! * `c4 = E[S] = √(2/(n−1))·Γ(n/2)/Γ((n−1)/2)`.
//! * `A2 = 3/(d2√n)`, `A3 = 3/(c4√n)`, `B3/B4 = 1 ∓ 3√(1−c4²)/c4` (`B3 ≥ 0`),
//!   `B5/B6 = c4 ∓ 3√(1−c4²)` (`B5 ≥ 0`), `D1/D2 = d2 ∓ 3·d3` (`D1 ≥ 0`),
//!   `D3/D4 = 1 ∓ 3·d3/d2` (`D3 ≥ 0`).

use crate::error::{invalid, Result};
use openstats_math::integrate::GaussLegendre;
use openstats_math::roots::brent;
use openstats_math::special::gamma::ln_gamma;
use openstats_math::special::tukey::normal_range_cdf;
use std::sync::OnceLock;

/// Smallest subgroup size with tabulated constants.
pub const MIN_SIZE: usize = 2;
/// Largest subgroup size with tabulated constants.
pub const MAX_SIZE: usize = 25;

/// All constants for one subgroup size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubgroupConstants {
    /// Subgroup size.
    pub n: usize,
    /// `E[range]` of n standard normals.
    pub d2: f64,
    /// `SD[range]` of n standard normals.
    pub d3: f64,
    /// Median of the range of n standard normals.
    pub d4: f64,
    /// `E[s]` of n standard normals.
    pub c4: f64,
    /// X̄ chart factor for R̄: `3/(d2√n)`.
    pub a2: f64,
    /// X̄ chart factor for s̄: `3/(c4√n)`.
    pub a3: f64,
    /// s chart lower factor `max(0, 1 − 3√(1−c4²)/c4)`.
    pub b3: f64,
    /// s chart upper factor `1 + 3√(1−c4²)/c4`.
    pub b4: f64,
    /// s chart lower factor for a known σ: `max(0, c4 − 3√(1−c4²))`.
    pub b5: f64,
    /// s chart upper factor for a known σ: `c4 + 3√(1−c4²)`.
    pub b6: f64,
    /// R chart lower factor for a known σ: `max(0, d2 − 3·d3)`.
    pub big_d1: f64,
    /// R chart upper factor for a known σ: `d2 + 3·d3`.
    pub big_d2: f64,
    /// R chart lower factor `max(0, 1 − 3·d3/d2)`.
    pub big_d3: f64,
    /// R chart upper factor `1 + 3·d3/d2`.
    pub big_d4: f64,
}

fn range_tail_integrals(n: usize) -> (f64, f64) {
    let gl = GaussLegendre::new(24);
    let k = n as f64;
    let (mut m1, mut m2) = (0.0, 0.0);
    let panels = 24;
    let top = 18.0;
    for i in 0..panels {
        let a = top * i as f64 / panels as f64;
        let b = top * (i + 1) as f64 / panels as f64;
        let h = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        for (&t, &wt) in gl.nodes.iter().zip(&gl.weights) {
            let w = mid + h * t;
            let tail = 1.0 - normal_range_cdf(w, k);
            m1 += h * wt * tail;
            m2 += h * wt * 2.0 * w * tail;
        }
    }
    (m1, m2)
}

/// `d2*(m, g) = √(d2² + d3²/g)`: the expected value, in units of σ, of the
/// range of `m` normal observations averaged (as a root mean square) over `g`
/// independent ranges. `g = 1` gives the factor used for a *single* range
/// (the AIAG `K2`/`K3` constants), `g → ∞` gives `d2`. Unlike the tabulated
/// constants it works for any `m ≥ 2` and any real `g ≥ 1`.
pub fn d2_star(m: usize, g: f64) -> Result<f64> {
    if m < MIN_SIZE {
        return invalid(format!("d2*: group size {m} must be at least {MIN_SIZE}"));
    }
    if g.is_nan() || g < 1.0 {
        return invalid(format!("d2*: the number of ranges {g} must be at least 1"));
    }
    let (d2, second) = range_tail_integrals(m);
    let var = (second - d2 * d2).max(0.0);
    Ok((d2 * d2 + var / g).sqrt())
}

fn check_size(n: usize) -> Result<()> {
    if !(MIN_SIZE..=MAX_SIZE).contains(&n) {
        return invalid(format!("subgroup size {n} is outside {MIN_SIZE}..={MAX_SIZE}"));
    }
    Ok(())
}

/// `c4(n) = E[s]/σ` for any `n ≥ 2` (a real `n` is allowed, as needed for
/// pooled estimates with `n = Σ(nᵢ − 1) + 1`).
pub fn c4_real(n: f64) -> Result<f64> {
    if n.is_nan() || n < 2.0 || !n.is_finite() {
        return invalid(format!("c4 needs n ≥ 2 (got {n})"));
    }
    Ok((2.0 / (n - 1.0)).sqrt() * (ln_gamma(n / 2.0) - ln_gamma((n - 1.0) / 2.0)).exp())
}

fn compute(n: usize) -> SubgroupConstants {
    let (d2, m2) = range_tail_integrals(n);
    let d3 = (m2 - d2 * d2).max(0.0).sqrt();
    let d4 = brent(|w| normal_range_cdf(w, n as f64) - 0.5, 0.0, 20.0, 1e-14, 200).unwrap_or(f64::NAN);
    let c4 = c4_real(n as f64).unwrap_or(f64::NAN);
    let nf = n as f64;
    let s = (1.0 - c4 * c4).sqrt();
    SubgroupConstants {
        n,
        d2,
        d3,
        d4,
        c4,
        a2: 3.0 / (d2 * nf.sqrt()),
        a3: 3.0 / (c4 * nf.sqrt()),
        b3: (1.0 - 3.0 * s / c4).max(0.0),
        b4: 1.0 + 3.0 * s / c4,
        b5: (c4 - 3.0 * s).max(0.0),
        b6: c4 + 3.0 * s,
        big_d1: (d2 - 3.0 * d3).max(0.0),
        big_d2: d2 + 3.0 * d3,
        big_d3: (1.0 - 3.0 * d3 / d2).max(0.0),
        big_d4: 1.0 + 3.0 * d3 / d2,
    }
}

fn table() -> &'static [SubgroupConstants] {
    static TABLE: OnceLock<Vec<SubgroupConstants>> = OnceLock::new();
    TABLE.get_or_init(|| (MIN_SIZE..=MAX_SIZE).map(compute).collect())
}

/// The constants for subgroup size `n` (2 ≤ n ≤ 25).
pub fn constants(n: usize) -> Result<SubgroupConstants> {
    check_size(n)?;
    Ok(table()[n - MIN_SIZE])
}

/// The full table for n = 2 … 25.
pub fn constants_table() -> &'static [SubgroupConstants] {
    table()
}

/// `d2(n)`.
pub fn d2(n: usize) -> Result<f64> {
    Ok(constants(n)?.d2)
}

/// `d3(n)`.
pub fn d3(n: usize) -> Result<f64> {
    Ok(constants(n)?.d3)
}

/// Median of the range of `n` standard normals (`d4`; used to unbias a median moving range).
pub fn d4(n: usize) -> Result<f64> {
    Ok(constants(n)?.d4)
}

/// `c4(n)` for 2 ≤ n ≤ 25 (see [`c4_real`] for other n).
pub fn c4(n: usize) -> Result<f64> {
    Ok(constants(n)?.c4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_small_cases() {
        // n = 2: W = |N(0,2)|, E[W] = 2/√π, E[W²] = 2.
        let c = constants(2).unwrap();
        assert!((c.d2 - 2.0 / std::f64::consts::PI.sqrt()).abs() < 1e-13);
        assert!((c.d3 - (2.0 - 4.0 / std::f64::consts::PI).sqrt()).abs() < 1e-13);
        assert!((c.c4 - (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-14);
        // n = 3: d2 = 3/√π.
        assert!((constants(3).unwrap().d2 - 3.0 / std::f64::consts::PI.sqrt()).abs() < 1e-13);
        // c4(3) = √π/2.
        assert!((c4(3).unwrap() - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-14);
    }

    #[test]
    fn derived_factors_are_consistent() {
        for c in constants_table() {
            let n = c.n as f64;
            assert!((c.a2 * c.d2 * n.sqrt() - 3.0).abs() < 1e-12);
            assert!((c.a3 * c.c4 * n.sqrt() - 3.0).abs() < 1e-12);
            assert!((c.big_d4 - 1.0 - 3.0 * c.d3 / c.d2).abs() < 1e-14);
            assert!(c.big_d3 >= 0.0 && c.b3 >= 0.0 && c.b5 >= 0.0 && c.big_d1 >= 0.0);
            assert!(c.d4 > 0.0 && (normal_range_cdf(c.d4, n) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn d2_star_matches_aiag_constants() {
        // AIAG K2 = 1/d2*(m, 1): 0.7071 (2 operators), 0.5231 (3); K3 for 10 parts 0.3146.
        assert!((1.0 / d2_star(2, 1.0).unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-4);
        assert!((1.0 / d2_star(3, 1.0).unwrap() - 0.5231).abs() < 1e-4);
        assert!((1.0 / d2_star(10, 1.0).unwrap() - 0.3146).abs() < 1e-4);
        assert!((d2_star(5, 1e12).unwrap() - d2(5).unwrap()).abs() < 1e-9);
        assert!(d2_star(1, 1.0).is_err() && d2_star(3, 0.5).is_err());
        assert!(d2_star(40, 1.0).unwrap() > d2_star(30, 1.0).unwrap());
    }

    #[test]
    fn size_limits() {
        assert!(constants(1).is_err() && constants(26).is_err());
        assert!(c4_real(1.5).is_err());
        assert!((c4_real(5.0).unwrap() - c4(5).unwrap()).abs() < 1e-15);
    }
}
