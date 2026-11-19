//! Outlier tests for (approximately) normal data: Grubbs' test, the
//! generalized ESD test (Rosner) for several outliers, and Dixon's Q test.

use crate::error::{invalid, Result, StatsError};
use crate::hypothesis::check_level;
use openstats_math::dist::{ContinuousDistribution, Normal, StudentT};
use openstats_math::integrate::GaussLegendre;
use openstats_math::roots::brent;
use openstats_math::stats::summary;

/// Which end(s) of the sample may hold the outlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierSide {
    /// The most extreme value at either end.
    Either,
    /// Test the maximum.
    Larger,
    /// Test the minimum.
    Smaller,
}

/// Non-missing values with their positions in `data`.
fn with_index(data: &[f64]) -> Result<Vec<(f64, usize)>> {
    let mut v = Vec::new();
    for (i, &x) in data.iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        if x.is_infinite() {
            return Err(StatsError::NonFinite);
        }
        v.push((x, i));
    }
    Ok(v)
}

/// Result of Grubbs' test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grubbs {
    pub n: usize,
    pub mean: f64,
    pub std_dev: f64,
    /// Position of the suspect value in the input slice.
    pub index: usize,
    pub value: f64,
    /// G = |suspect − mean|/s.
    pub g: f64,
    /// p-value: n·P(T_{n−2} > t) for one end, doubled (capped at 1) for `Either`.
    pub p_value: f64,
}

fn grubbs_t(n: f64, g: f64) -> f64 {
    let denom = (n - 1.0) * (n - 1.0) - n * g * g;
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        (n * (n - 2.0) * g * g / denom).sqrt()
    }
}

/// Grubbs' test for a single outlier (n ≥ 3).
pub fn grubbs_test(data: &[f64], side: OutlierSide) -> Result<Grubbs> {
    let v = with_index(data)?;
    let n = v.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "Grubbs' test", needed: 3, got: n });
    }
    let vals: Vec<f64> = v.iter().map(|p| p.0).collect();
    let mean = summary::mean(&vals)?;
    let sd = summary::std_dev(&vals)?;
    if sd == 0.0 {
        return invalid("Grubbs' test needs non-constant data");
    }
    let dev = |p: &(f64, usize)| match side {
        OutlierSide::Either => (p.0 - mean).abs(),
        OutlierSide::Larger => p.0 - mean,
        OutlierSide::Smaller => mean - p.0,
    };
    let s = v.iter().copied().max_by(|a, b| dev(a).total_cmp(&dev(b))).expect("n >= 3");
    let g = dev(&s) / sd;
    let nf = n as f64;
    let tail = if g <= 0.0 { 0.5 } else { StudentT::new(nf - 2.0)?.sf(grubbs_t(nf, g)) };
    let mult = if side == OutlierSide::Either { 2.0 } else { 1.0 };
    Ok(Grubbs { n, mean, std_dev: sd, index: s.1, value: s.0, g, p_value: (mult * nf * tail).min(1.0) })
}

/// Critical value of G at significance `alpha`.
pub fn grubbs_critical(n: usize, alpha: f64, side: OutlierSide) -> Result<f64> {
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "Grubbs' critical value", needed: 3, got: n });
    }
    check_level(1.0 - alpha)?;
    let nf = n as f64;
    let p = if side == OutlierSide::Either { alpha / (2.0 * nf) } else { alpha / nf };
    let t = StudentT::new(nf - 2.0)?.isf(p);
    Ok((nf - 1.0) / nf.sqrt() * (t * t / (nf - 2.0 + t * t)).sqrt())
}

/// One step of the generalized ESD test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsdStep {
    /// Number of outliers tested so far (1-based).
    pub i: usize,
    /// Test statistic Rᵢ = max|x − x̄|/s on the remaining sample.
    pub r: f64,
    /// Critical value λᵢ.
    pub lambda: f64,
    /// Position of the removed value in the input slice, and the value.
    pub index: usize,
    pub value: f64,
}

/// Result of the generalized ESD test.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedEsd {
    pub n: usize,
    pub steps: Vec<EsdStep>,
    /// Number of outliers: the largest i with Rᵢ > λᵢ.
    pub n_outliers: usize,
    /// Positions (in the input slice) of the declared outliers, most extreme first.
    pub outlier_indices: Vec<usize>,
}

/// Rosner's generalized ESD test for up to `max_outliers` outliers at
/// significance `alpha`. It is the proper multi-outlier extension of
/// Grubbs' test: all `max_outliers` candidates are removed one at a time
/// and the *largest* i with Rᵢ > λᵢ gives the count, which avoids masking.
/// Requires `max_outliers ≤ n − 2`.
pub fn generalized_esd(data: &[f64], max_outliers: usize, alpha: f64) -> Result<GeneralizedEsd> {
    check_level(1.0 - alpha)?;
    let mut v = with_index(data)?;
    let n = v.len();
    if max_outliers == 0 || max_outliers + 2 > n {
        return invalid(format!("max_outliers must be between 1 and n - 2 = {}", n.saturating_sub(2)));
    }
    let nf = n as f64;
    let mut steps = Vec::new();
    for i in 1..=max_outliers {
        let vals: Vec<f64> = v.iter().map(|p| p.0).collect();
        let mean = summary::mean(&vals)?;
        let sd = summary::std_dev(&vals)?;
        if sd == 0.0 {
            break;
        }
        let (pos, &(val, idx)) = v
            .iter()
            .enumerate()
            .max_by(|a, b| (a.1 .0 - mean).abs().total_cmp(&(b.1 .0 - mean).abs()))
            .expect("non-empty");
        let r = (val - mean).abs() / sd;
        let p = 1.0 - alpha / (2.0 * (nf - i as f64 + 1.0));
        let df = nf - i as f64 - 1.0;
        let t = StudentT::new(df)?.quantile(p);
        let lambda = (nf - i as f64) * t / ((df + t * t) * (nf - i as f64 + 1.0)).sqrt();
        steps.push(EsdStep { i, r, lambda, index: idx, value: val });
        v.remove(pos);
    }
    let n_out = steps.iter().filter(|s| s.r > s.lambda).map(|s| s.i).max().unwrap_or(0);
    Ok(GeneralizedEsd {
        n,
        outlier_indices: steps[..n_out].iter().map(|s| s.index).collect(),
        steps,
        n_outliers: n_out,
    })
}

/// Largest n for which Dixon's Q p-values are computed.
pub const DIXON_MAX_N: usize = 30;

/// P(r₁₀ > q) for the *high* end: r₁₀ = (x₍ₙ₎ − x₍ₙ₋₁₎)/(x₍ₙ₎ − x₍₁₎) for a
/// normal sample, by 2-D Gauss–Legendre integration of the joint density of
/// (x₍₁₎, x₍ₙ₋₁₎, x₍ₙ₎) with the largest value integrated out analytically.
fn dixon_tail(n: usize, q: f64) -> f64 {
    if q <= 0.0 {
        return 1.0;
    }
    if q >= 1.0 {
        return 0.0;
    }
    let norm = Normal::standard();
    let gl = GaussLegendre::new(16);
    const L: f64 = 8.6;
    const PANELS: usize = 28;
    let panels = |lo: f64, hi: f64, mut f: &mut dyn FnMut(f64) -> f64| -> f64 {
        let w = (hi - lo) / PANELS as f64;
        (0..PANELS).map(|k| gl.integrate(lo + w * k as f64, lo + w * (k + 1) as f64, &mut f)).sum()
    };
    let nn = (n * (n - 1) * (n - 2)) as f64;
    let mut outer = |b: f64| -> f64 {
        let (phib, cb) = (norm.pdf(b), norm.cdf(b));
        let c_star = |a: f64| (b - q * a) / (1.0 - q);
        let mut inner = |a: f64| norm.pdf(a) * (cb - norm.cdf(a)).max(0.0).powi(n as i32 - 3) * norm.sf(c_star(a));
        phib * panels(-L, b, &mut inner)
    };
    (nn * panels(-L, L, &mut outer)).clamp(0.0, 1.0)
}

/// Result of Dixon's Q test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DixonQ {
    pub n: usize,
    /// Q = gap/range of the suspect value.
    pub q: f64,
    /// Position of the suspect value in the input slice.
    pub index: usize,
    pub value: f64,
    /// p-value from the normal-theory null distribution of Q (one end), or
    /// doubled (capped at 1) for [`OutlierSide::Either`].
    pub p_value: f64,
}

/// Dixon's Q test (r₁₀: gap ÷ range) for 3 ≤ n ≤ [`DIXON_MAX_N`].
///
/// The p-value is computed by numerical integration of the exact null
/// distribution for normal data (accurate to about 1e-9), not interpolated
/// from a table. For `Either` the end with the larger gap is tested and the
/// one-end p-value doubled; for Q ≤ ½ this is an upper bound.
pub fn dixon_q_test(data: &[f64], side: OutlierSide) -> Result<DixonQ> {
    let mut v = with_index(data)?;
    let n = v.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "Dixon's Q test", needed: 3, got: n });
    }
    if n > DIXON_MAX_N {
        return invalid(format!("Dixon's Q test is defined for n <= {DIXON_MAX_N}, got {n}"));
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let range = v[n - 1].0 - v[0].0;
    if range == 0.0 {
        return invalid("Dixon's Q test needs non-constant data");
    }
    let q_hi = (v[n - 1].0 - v[n - 2].0) / range;
    let q_lo = (v[1].0 - v[0].0) / range;
    let (q, suspect, mult) = match side {
        OutlierSide::Larger => (q_hi, v[n - 1], 1.0),
        OutlierSide::Smaller => (q_lo, v[0], 1.0),
        OutlierSide::Either => {
            if q_hi >= q_lo {
                (q_hi, v[n - 1], 2.0)
            } else {
                (q_lo, v[0], 2.0)
            }
        }
    };
    Ok(DixonQ { n, q, index: suspect.1, value: suspect.0, p_value: (mult * dixon_tail(n, q)).min(1.0) })
}

/// Critical value of Q: the q with P(Q > q) = `alpha` for one end
/// (`one_sided`) or `alpha`/2 per end (two-sided, i.e. either end).
pub fn dixon_q_critical(n: usize, alpha: f64, one_sided: bool) -> Result<f64> {
    if !(3..=DIXON_MAX_N).contains(&n) {
        return invalid(format!("Dixon's Q critical values are available for 3 <= n <= {DIXON_MAX_N}, got {n}"));
    }
    check_level(1.0 - alpha)?;
    let target = if one_sided { alpha } else { alpha / 2.0 };
    Ok(brent(|q| dixon_tail(n, q) - target, 1e-9, 1.0 - 1e-12, 1e-12, 200)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grubbs_by_hand_and_sides() {
        let x = [2.1, 2.3, 2.2, 2.4, 2.2, 2.3, 9.0, 2.1];
        let e = grubbs_test(&x, OutlierSide::Either).unwrap();
        assert_eq!((e.index, e.value), (6, 9.0));
        let l = grubbs_test(&x, OutlierSide::Larger).unwrap();
        assert!((l.p_value * 2.0 - e.p_value).abs() < 1e-12 || e.p_value == 1.0);
        assert!(e.p_value < 0.001);
        let s = grubbs_test(&x, OutlierSide::Smaller).unwrap();
        assert_eq!(s.value, 2.1);
        assert!(s.p_value > 0.5);
        // G ≤ (n−1)/√n always
        assert!(e.g <= 7.0 / 8f64.sqrt() + 1e-12);
        // critical value consistent with the p-value: p(Gcrit) = alpha
        let gc = grubbs_critical(8, 0.05, OutlierSide::Either).unwrap();
        let nf = 8.0;
        let t = grubbs_t(nf, gc);
        let p = 2.0 * nf * StudentT::new(nf - 2.0).unwrap().sf(t);
        assert!((p - 0.05).abs() < 1e-10);
        assert!(grubbs_test(&[1.0, 2.0], OutlierSide::Either).is_err());
        assert!(grubbs_test(&[1.0, 1.0, 1.0], OutlierSide::Either).is_err());
        assert!(grubbs_critical(2, 0.05, OutlierSide::Either).is_err());
        // missing values keep original positions
        let y = [f64::NAN, 2.1, 2.3, 2.2, 2.4, 2.2, 2.3, 9.0, 2.1];
        assert_eq!(grubbs_test(&y, OutlierSide::Either).unwrap().index, 7);
    }

    #[test]
    fn esd_masking_and_limits() {
        let mut x = vec![10.0, 10.2, 9.9, 10.1, 10.3, 9.8, 10.0, 10.2, 9.95, 10.05, 10.15, 9.85];
        x.extend([25.0, 26.0]); // two masking outliers
        let r = generalized_esd(&x, 4, 0.05).unwrap();
        assert_eq!(r.n_outliers, 2);
        let mut idx = r.outlier_indices.clone();
        idx.sort_unstable();
        assert_eq!(idx, vec![12, 13]);
        assert!(generalized_esd(&x, 0, 0.05).is_err());
        assert!(generalized_esd(&x, 13, 0.05).is_err());
        let clean = generalized_esd(&[10.0, 10.1, 9.9, 10.05, 9.95, 10.02, 9.98, 10.0], 2, 0.05).unwrap();
        assert_eq!(clean.n_outliers, 0);
        // constant data after removals stops early
        let c = generalized_esd(&[1.0, 1.0, 1.0, 1.0, 1.0], 2, 0.05).unwrap();
        assert_eq!(c.n_outliers, 0);
    }

    #[test]
    fn dixon_tail_basic_properties() {
        // n = 3 closed form: Q = (x3-x2)/(x3-x1); for the high end P(Q > q) is 1 - ... check monotone & range
        let (a, b, c) = (dixon_tail(3, 0.5), dixon_tail(3, 0.9), dixon_tail(3, 0.99));
        assert!(1.0 > a && a > b && b > c && c > 0.0);
        // symmetry of the two ends under negation
        let x = [1.0, 1.2, 1.5, 1.4, 6.0, 1.1];
        let neg: Vec<f64> = x.iter().map(|v| -v).collect();
        let up = dixon_q_test(&x, OutlierSide::Larger).unwrap();
        let dn = dixon_q_test(&neg, OutlierSide::Smaller).unwrap();
        assert!((up.p_value - dn.p_value).abs() < 1e-12 && (up.q - dn.q).abs() < 1e-12);
        assert!(up.p_value < 0.01);
        assert!(dixon_q_test(&[1.0, 2.0], OutlierSide::Either).is_err());
        assert!(dixon_q_test(&[1.0, 1.0, 1.0], OutlierSide::Either).is_err());
        assert!(dixon_q_test(&vec![1.0; 31], OutlierSide::Either).is_err());
        assert!(dixon_q_critical(2, 0.05, true).is_err());
        let e = dixon_q_test(&x, OutlierSide::Either).unwrap();
        assert!((e.p_value - 2.0 * up.p_value).abs() < 1e-12);
    }
}
