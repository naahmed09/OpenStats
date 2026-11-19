//! Studentized range distribution (Tukey's HSD and related procedures).
//!
//! For k means and ν error degrees of freedom,
//!
//! ```text
//! P(Q ≤ q) = ∫₀^∞ f_ν(s) · W(q·s) ds,
//! W(w)     = k ∫ φ(z) [Φ(z) − Φ(z − w)]^{k−1} dz        (range of k normals)
//! f_ν(s)   = 2 (ν/2)^{ν/2} s^{ν−1} e^{−νs²/2} / Γ(ν/2)  (density of √(χ²_ν/ν))
//! ```
//!
//! Both integrals use composite Gauss–Legendre rules: the inner one over
//! z ∈ [−8.5, 8.5] (φ is below 1e-16 outside), the outer one over the
//! 1e-17 … 1 − 1e-17 quantile range of f_ν. Accuracy is about 1e-12 absolute.

use super::erf::{norm_cdf, norm_pdf};
use super::gamma::ln_gamma;
use super::incgamma::{gamma_p_inv, gamma_q_inv};
use crate::error::{MathError, Result};
use crate::integrate::GaussLegendre;
use std::sync::OnceLock;

fn rule32() -> &'static GaussLegendre {
    static R: OnceLock<GaussLegendre> = OnceLock::new();
    R.get_or_init(|| GaussLegendre::new(32))
}

/// Nodes/weights of a composite rule on [a, b] with `panels` equal panels.
pub(crate) fn composite(a: f64, b: f64, panels: usize) -> Vec<(f64, f64)> {
    let r = rule32();
    let w = (b - a) / panels as f64;
    let mut out = Vec::with_capacity(panels * r.nodes.len());
    for p in 0..panels {
        let lo = a + p as f64 * w;
        let (h, c) = (0.5 * w, lo + 0.5 * w);
        for (x, wt) in r.nodes.iter().zip(&r.weights) {
            out.push((c + h * x, wt * h));
        }
    }
    out
}

/// Inner quadrature points over z ∈ [−8.5, 8.5] with φ(z) folded into the weight.
pub(crate) fn normal_points() -> &'static [(f64, f64, f64)] {
    static P: OnceLock<Vec<(f64, f64, f64)>> = OnceLock::new();
    P.get_or_init(|| composite(-8.5, 8.5, 17).into_iter().map(|(z, w)| (z, w * norm_pdf(z), norm_cdf(z))).collect())
}

/// ∫₀^∞ f_ν(s) g(s) ds; ν = ∞ (or > 1e7) evaluates g(1).
pub(crate) fn mix_over_scaled_chi(df: f64, mut g: impl FnMut(f64) -> f64) -> f64 {
    if !df.is_finite() || df > 1e7 {
        return g(1.0);
    }
    let a = df / 2.0;
    let lo = (2.0 * gamma_p_inv(a, 1e-17) / df).sqrt();
    let hi = (2.0 * gamma_q_inv(a, 1e-17) / df).sqrt();
    let lnc = std::f64::consts::LN_2 + a * a.ln() - ln_gamma(a);
    let panels = if df < 3.0 { 24 } else { 12 };
    composite(lo, hi, panels)
        .into_iter()
        .map(|(s, w)| {
            let dens = (lnc + (df - 1.0) * s.ln() - a * s * s).exp();
            w * dens * g(s)
        })
        .sum()
}

/// CDF of the range of k independent standard normals.
pub fn normal_range_cdf(w: f64, k: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let s: f64 = normal_points()
        .iter()
        .map(|&(z, wt, cz)| {
            let d = cz - norm_cdf(z - w);
            if d <= 0.0 {
                0.0
            } else {
                wt * d.powf(k - 1.0)
            }
        })
        .sum();
    (k * s).clamp(0.0, 1.0)
}

fn check(k: f64, df: f64) -> Result<()> {
    if k.is_nan() || k < 2.0 || df.is_nan() || df < 1.0 {
        return Err(MathError::InvalidArgument(format!(
            "studentized range needs k ≥ 2 and df ≥ 1 (got k={k}, df={df})"
        )));
    }
    Ok(())
}

/// P(Q ≤ q) for the studentized range with k groups and `df` degrees of
/// freedom (`df = ∞` allowed).
pub fn ptukey(q: f64, k: f64, df: f64) -> Result<f64> {
    check(k, df)?;
    if q.is_nan() {
        return Ok(f64::NAN);
    }
    if q <= 0.0 {
        return Ok(0.0);
    }
    if q.is_infinite() {
        return Ok(1.0);
    }
    Ok(mix_over_scaled_chi(df, |s| normal_range_cdf(q * s, k)).clamp(0.0, 1.0))
}

/// Quantile: q with P(Q ≤ q) = p.
pub fn qtukey(p: f64, k: f64, df: f64) -> Result<f64> {
    check(k, df)?;
    if !(0.0..=1.0).contains(&p) || p.is_nan() {
        return Err(MathError::InvalidArgument(format!("probability {p} outside [0, 1]")));
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    let f = |q: f64| ptukey(q, k, df).unwrap_or(f64::NAN) - p;
    let (lo, hi) = crate::roots::expand_bracket(f, 1e-3, 5.0, 0.0, 1e6)?;
    crate::roots::brent(f, lo, hi, 1e-12, 200)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_groups_reduce_to_normal_and_t() {
        // k = 2: range of two normals = |Z₁ − Z₂| = √2|Z| ⇒ W(w) = 2Φ(w/√2) − 1.
        for &w in &[0.3, 1.0, 2.77, 5.0] {
            let exact = 2.0 * norm_cdf(w / 2f64.sqrt()) - 1.0;
            assert!((normal_range_cdf(w, 2.0) - exact).abs() < 1e-13, "w={w}");
        }
        // With df: P(Q ≤ q) = P(|T_ν| ≤ q/√2).
        for &(q, df) in &[(3.0, 5.0), (2.5, 30.0), (4.0, 1.0)] {
            let t = q / 2f64.sqrt();
            let x = df / (df + t * t);
            let exact = 1.0 - super::super::incbeta::beta_inc(df / 2.0, 0.5, x);
            let got = ptukey(q, 2.0, df).unwrap();
            assert!((got - exact).abs() < 1e-11, "q={q} df={df}: {got} vs {exact}");
        }
    }

    #[test]
    fn quantile_roundtrip() {
        let q = qtukey(0.95, 3.0, 10.0).unwrap();
        assert!((ptukey(q, 3.0, 10.0).unwrap() - 0.95).abs() < 1e-11);
        assert!(qtukey(0.95, 1.0, 10.0).is_err());
        assert_eq!(ptukey(-1.0, 4.0, 10.0).unwrap(), 0.0);
    }
}
