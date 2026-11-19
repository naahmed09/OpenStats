//! Covariance and correlation.
//!
//! * [`covariance_matrix`], [`correlation_matrix`] — Pearson, with pairwise or
//!   listwise deletion of missing values (`NaN`), p-values and Fisher-z
//!   confidence intervals.
//! * [`spearman`] — Pearson correlation of average ranks, p-value from the t
//!   approximation with n−2 df.
//! * [`kendall`] — τ-a, τ-b, τ-c from concordant/discordant pair counts
//!   (Knight's O(n log n) algorithm). The p-value tests S = C − D: exact when
//!   there are no ties and n ≤ [`KENDALL_EXACT_MAX_N`], otherwise normal with
//!   Kendall's tie-corrected variance.
//! * [`partial_correlation`] — correlation of two variables after removing
//!   the linear effect of control variables.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use openstats_math::dist::{ContinuousDistribution, Normal, StudentT};
use openstats_math::linalg::Lu;
use openstats_math::stats::summary;
use openstats_math::Matrix;

/// How missing values are handled when several variables are involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingData {
    /// Each pair of variables uses all rows where both are present.
    Pairwise,
    /// Only rows complete in every variable are used.
    Listwise,
}

/// Largest n for which Kendall's exact null distribution is used.
pub const KENDALL_EXACT_MAX_N: usize = 60;

fn check_cols(cols: &[Vec<f64>]) -> Result<usize> {
    let n = cols.first().map_or(0, Vec::len);
    if cols.is_empty() {
        return invalid("at least one variable is required");
    }
    if cols.iter().any(|c| c.len() != n) {
        return invalid("variables have different lengths");
    }
    if cols.iter().flatten().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    Ok(n)
}

fn check_pair(x: &[f64], y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return invalid(format!("variables have different lengths ({} and {})", x.len(), y.len()));
    }
    if x.iter().chain(y).any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    Ok(())
}

/// Rows where both are present.
fn complete_pairs(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    x.iter().zip(y).filter(|(a, b)| !a.is_nan() && !b.is_nan()).map(|(a, b)| (*a, *b)).unzip()
}

/// Sums of squares and cross-products about the means of complete pairs.
fn sxy(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    let (mx, my) = (summary::mean(x)?, summary::mean(y)?);
    Ok((
        summary::sum(x.iter().map(|a| (a - mx) * (a - mx))),
        summary::sum(y.iter().map(|b| (b - my) * (b - my))),
        summary::sum(x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my))),
    ))
}

/// Pairwise or listwise covariance matrix (divisor n − 1) with the number of
/// observations behind each entry. Entries with fewer than 2 observations are
/// `NaN`.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceMatrix {
    pub values: Vec<Vec<f64>>,
    pub n: Vec<Vec<usize>>,
}

/// Covariance matrix of the columns.
pub fn covariance_matrix(cols: &[Vec<f64>], missing: MissingData) -> Result<CovarianceMatrix> {
    let rows = check_cols(cols)?;
    let p = cols.len();
    let usable: Vec<bool> = (0..rows).map(|i| cols.iter().all(|c| !c[i].is_nan())).collect();
    let mut values = vec![vec![f64::NAN; p]; p];
    let mut ns = vec![vec![0usize; p]; p];
    for a in 0..p {
        for b in a..p {
            let (x, y): (Vec<f64>, Vec<f64>) = match missing {
                MissingData::Pairwise => complete_pairs(&cols[a], &cols[b]),
                MissingData::Listwise => (0..rows).filter(|&i| usable[i]).map(|i| (cols[a][i], cols[b][i])).unzip(),
            };
            ns[a][b] = x.len();
            ns[b][a] = x.len();
            if x.len() >= 2 {
                let (_, _, c) = sxy(&x, &y)?;
                values[a][b] = c / (x.len() - 1) as f64;
                values[b][a] = values[a][b];
            }
        }
    }
    Ok(CovarianceMatrix { values, n: ns })
}

/// Result of a Pearson correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pearson {
    pub r: f64,
    pub n: usize,
    /// t = r√(n−2)/√(1−r²); `None` for n < 3.
    pub t: Option<f64>,
    /// Two-sided p-value from t with n−2 df (0 when |r| = 1); `None` for n < 3.
    pub p_value: Option<f64>,
    /// Fisher-z interval; `None` for n < 4.
    pub ci: Option<Interval>,
}

fn fisher_interval(r: f64, n_eff: f64, level: f64) -> Option<Interval> {
    if n_eff <= 3.0 {
        return None;
    }
    if r.abs() >= 1.0 {
        return Some(Interval { lower: r, upper: r, level });
    }
    let z = r.atanh();
    let q = Normal::standard().isf((1.0 - level) / 2.0);
    let se = 1.0 / (n_eff - 3.0).sqrt();
    Some(Interval { lower: (z - q * se).tanh(), upper: (z + q * se).tanh(), level })
}

fn t_p(r: f64, df: f64) -> Result<(f64, f64)> {
    if r.abs() >= 1.0 {
        return Ok((f64::INFINITY.copysign(r), 0.0));
    }
    let t = r * (df / (1.0 - r * r)).sqrt();
    let p = 2.0 * StudentT::new(df)?.sf(t.abs());
    Ok((t, p.min(1.0)))
}

fn check_level(level: f64) -> Result<()> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        invalid(format!("confidence level {level} must be in (0, 1)"))
    }
}

/// Pearson correlation of the complete pairs of `x` and `y`.
///
/// Errors when fewer than 2 pairs remain or either variable is constant.
pub fn pearson(x: &[f64], y: &[f64], level: f64) -> Result<Pearson> {
    check_level(level)?;
    check_pair(x, y)?;
    let (x, y) = complete_pairs(x, y);
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "correlation", needed: 2, got: n });
    }
    let (sxx, syy, sxy_) = sxy(&x, &y)?;
    if sxx == 0.0 || syy == 0.0 {
        return invalid("correlation is undefined when a variable is constant");
    }
    let r = (sxy_ / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let (t, p_value) = if n >= 3 {
        let (t, p) = t_p(r, (n - 2) as f64)?;
        (Some(t), Some(p))
    } else {
        (None, None)
    };
    Ok(Pearson { r, n, t, p_value, ci: fisher_interval(r, n as f64, level) })
}

/// Pearson correlation matrix with p-values and confidence limits. Undefined
/// entries (fewer than 3 observations, constant variable) are `NaN`; the
/// diagonal is 1 with p-value `NaN`.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationMatrix {
    pub r: Vec<Vec<f64>>,
    pub n: Vec<Vec<usize>>,
    pub p_value: Vec<Vec<f64>>,
    pub ci_lower: Vec<Vec<f64>>,
    pub ci_upper: Vec<Vec<f64>>,
}

/// Pearson correlation matrix of the columns.
pub fn correlation_matrix(cols: &[Vec<f64>], missing: MissingData, level: f64) -> Result<CorrelationMatrix> {
    check_level(level)?;
    let rows = check_cols(cols)?;
    let p = cols.len();
    let usable: Vec<bool> = (0..rows).map(|i| cols.iter().all(|c| !c[i].is_nan())).collect();
    let nan = vec![vec![f64::NAN; p]; p];
    let mut m = CorrelationMatrix {
        r: nan.clone(),
        n: vec![vec![0; p]; p],
        p_value: nan.clone(),
        ci_lower: nan.clone(),
        ci_upper: nan,
    };
    for a in 0..p {
        let na = match missing {
            MissingData::Pairwise => cols[a].iter().filter(|v| !v.is_nan()).count(),
            MissingData::Listwise => usable.iter().filter(|&&u| u).count(),
        };
        m.n[a][a] = na;
        let (sxx, _, _) = {
            let v: Vec<f64> = (0..rows)
                .filter(|&i| match missing {
                    MissingData::Pairwise => !cols[a][i].is_nan(),
                    MissingData::Listwise => usable[i],
                })
                .map(|i| cols[a][i])
                .collect();
            if v.len() >= 2 {
                sxy(&v, &v)?
            } else {
                (0.0, 0.0, 0.0)
            }
        };
        if sxx > 0.0 {
            m.r[a][a] = 1.0;
        }
        for b in a + 1..p {
            let (x, y): (Vec<f64>, Vec<f64>) = match missing {
                MissingData::Pairwise => complete_pairs(&cols[a], &cols[b]),
                MissingData::Listwise => (0..rows).filter(|&i| usable[i]).map(|i| (cols[a][i], cols[b][i])).unzip(),
            };
            m.n[a][b] = x.len();
            m.n[b][a] = x.len();
            if let Ok(pr) = pearson(&x, &y, level) {
                for (i, j) in [(a, b), (b, a)] {
                    m.r[i][j] = pr.r;
                    m.p_value[i][j] = pr.p_value.unwrap_or(f64::NAN);
                    m.ci_lower[i][j] = pr.ci.map_or(f64::NAN, |c| c.lower);
                    m.ci_upper[i][j] = pr.ci.map_or(f64::NAN, |c| c.upper);
                }
            }
        }
    }
    Ok(m)
}

/// Fill a symmetric matrix by applying a pairwise statistic to the complete
/// pairs (or listwise-complete rows) of every pair of columns; the diagonal is
/// 1. `stat` returns `None` where the statistic is undefined.
pub fn pairwise_matrix(
    cols: &[Vec<f64>],
    missing: MissingData,
    stat: impl Fn(&[f64], &[f64]) -> Option<f64>,
) -> Result<Vec<Vec<f64>>> {
    let rows = check_cols(cols)?;
    let p = cols.len();
    let usable: Vec<bool> = (0..rows).map(|i| cols.iter().all(|c| !c[i].is_nan())).collect();
    let mut out = vec![vec![f64::NAN; p]; p];
    for a in 0..p {
        out[a][a] = 1.0;
        for b in a + 1..p {
            let (x, y): (Vec<f64>, Vec<f64>) = match missing {
                MissingData::Pairwise => complete_pairs(&cols[a], &cols[b]),
                MissingData::Listwise => (0..rows).filter(|&i| usable[i]).map(|i| (cols[a][i], cols[b][i])).unzip(),
            };
            let v = stat(&x, &y).unwrap_or(f64::NAN);
            out[a][b] = v;
            out[b][a] = v;
        }
    }
    Ok(out)
}

/// Average ranks (1-based; ties share the mean of their ranks).
pub fn average_ranks(x: &[f64]) -> Result<Vec<f64>> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite);
    }
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + 1 + j) as f64 / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    Ok(ranks)
}

/// Result of a Spearman rank correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spearman {
    pub rho: f64,
    pub n: usize,
    /// Two-sided p-value from the t approximation (n−2 df); `None` for n < 3.
    pub p_value: Option<f64>,
}

/// Spearman's ρ: the Pearson correlation of the average ranks of the
/// complete pairs. Errors if either variable is constant.
pub fn spearman(x: &[f64], y: &[f64]) -> Result<Spearman> {
    check_pair(x, y)?;
    let (x, y) = complete_pairs(x, y);
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "Spearman correlation", needed: 2, got: n });
    }
    let (rx, ry) = (average_ranks(&x)?, average_ranks(&y)?);
    let (sxx, syy, sxy_) = sxy(&rx, &ry)?;
    if sxx == 0.0 || syy == 0.0 {
        return invalid("Spearman correlation is undefined when a variable is constant");
    }
    let rho = (sxy_ / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let p_value = if n >= 3 { Some(t_p(rho, (n - 2) as f64)?.1) } else { None };
    Ok(Spearman { rho, n, p_value })
}

/// Result of Kendall's rank correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kendall {
    pub n: usize,
    pub concordant: u64,
    pub discordant: u64,
    /// Pairs tied in x only / y only / both.
    pub ties_x: u64,
    pub ties_y: u64,
    pub ties_both: u64,
    /// (C − D)/(n(n−1)/2).
    pub tau_a: f64,
    /// (C − D)/√((n₀ − n₁)(n₀ − n₂)); `NaN` when a variable is constant.
    pub tau_b: f64,
    /// 2(C − D)/(n²(m−1)/m), m = min(distinct x, distinct y) (Stuart's τ-c);
    /// `NaN` when m = 1.
    pub tau_c: f64,
    /// Standardised S = C − D under the tie-corrected normal approximation.
    pub z: f64,
    /// Two-sided p-value for H₀: independence, from S.
    pub p_value: f64,
    /// Whether `p_value` came from the exact null distribution.
    pub exact: bool,
}

/// Strict inversions in `v` by merge sort.
fn inversions(v: &mut [f64]) -> u64 {
    let n = v.len();
    if n < 2 {
        return 0;
    }
    let mut buf = v.to_vec();
    let mut width = 1;
    let mut count = 0u64;
    let (mut src, dst) = (v.to_vec(), &mut buf[..]);
    while width < n {
        let mut i = 0;
        while i < n {
            let (mid, hi) = ((i + width).min(n), (i + 2 * width).min(n));
            let (mut a, mut b, mut k) = (i, mid, i);
            while a < mid && b < hi {
                if src[b] < src[a] {
                    count += (mid - a) as u64;
                    dst[k] = src[b];
                    b += 1;
                } else {
                    dst[k] = src[a];
                    a += 1;
                }
                k += 1;
            }
            while a < mid {
                dst[k] = src[a];
                a += 1;
                k += 1;
            }
            while b < hi {
                dst[k] = src[b];
                b += 1;
                k += 1;
            }
            i += 2 * width;
        }
        src.copy_from_slice(dst);
        width *= 2;
    }
    v.copy_from_slice(&src);
    count
}

/// Group sizes of equal values in a sorted slice.
fn tie_groups(sorted: &[f64]) -> Vec<u64> {
    let mut g = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        g.push((j - i) as u64);
        i = j;
    }
    g
}

/// Exact P(D ≤ d) and P(D ≥ d) for the number of inversions D of a uniformly
/// random permutation of n items.
fn inversion_tail_probs(n: usize, d: u64) -> (f64, f64) {
    let maxd = n * (n - 1) / 2;
    let mut p = vec![0.0f64; maxd + 1];
    p[0] = 1.0;
    let mut cur = 0usize;
    for m in 2..=n {
        let nmax = cur + (m - 1);
        let mut q = vec![0.0f64; maxd + 1];
        // q[k] = (1/m) Σ_{j=0}^{m-1} p[k-j]; sliding window
        let mut win = 0.0;
        for k in 0..=nmax {
            if k <= cur {
                win += p[k];
            }
            if k >= m {
                win -= p[k - m];
            }
            q[k] = win / m as f64;
        }
        p = q;
        cur = nmax;
    }
    let d = d as usize;
    let le: f64 = p[..=d.min(maxd)].iter().sum();
    let ge: f64 = p[d.min(maxd)..].iter().sum();
    (le.min(1.0), ge.min(1.0))
}

/// Kendall's rank correlation of the complete pairs.
pub fn kendall(x: &[f64], y: &[f64]) -> Result<Kendall> {
    check_pair(x, y)?;
    let (x, y) = complete_pairs(x, y);
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "Kendall correlation", needed: 2, got: n });
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(y[a].total_cmp(&y[b])));
    let (xs, mut ys): (Vec<f64>, Vec<f64>) = idx.iter().map(|&i| (x[i], y[i])).unzip();
    let pairs = |g: u64| g * (g - 1) / 2;
    let tx = tie_groups(&xs);
    let n1: u64 = tx.iter().map(|&g| pairs(g)).sum();
    // joint ties: runs of equal (x, y) in the (x, y)-sorted order
    let mut n3 = 0u64;
    let mut run = 1u64;
    for i in 1..n {
        if xs[i] == xs[i - 1] && ys[i] == ys[i - 1] {
            run += 1;
        } else {
            n3 += pairs(run);
            run = 1;
        }
    }
    n3 += pairs(run);
    let d = inversions(&mut ys); // ys is now sorted
    let ty = tie_groups(&ys);
    let n2: u64 = ty.iter().map(|&g| pairs(g)).sum();
    let n0 = pairs(n as u64);
    let c = n0 + n3 - n1 - n2 - d;
    let s = c as f64 - d as f64;
    let nf = n as f64;
    let tau_a = s / n0 as f64;
    let denom = ((n0 - n1) as f64 * (n0 - n2) as f64).sqrt();
    let tau_b = if denom > 0.0 { s / denom } else { f64::NAN };
    let m = tx.len().min(ty.len()) as f64;
    let tau_c = if m > 1.0 { 2.0 * s / (nf * nf * (m - 1.0) / m) } else { f64::NAN };

    // Tie-corrected null variance of S.
    let sum = |g: &[u64], f: &dyn Fn(f64) -> f64| g.iter().map(|&t| f(t as f64)).sum::<f64>();
    let (v0, vt, vu) = (
        nf * (nf - 1.0) * (2.0 * nf + 5.0),
        sum(&tx, &|t| t * (t - 1.0) * (2.0 * t + 5.0)),
        sum(&ty, &|t| t * (t - 1.0) * (2.0 * t + 5.0)),
    );
    let (t1, u1) = (sum(&tx, &|t| t * (t - 1.0)), sum(&ty, &|t| t * (t - 1.0)));
    let (t2, u2) = (sum(&tx, &|t| t * (t - 1.0) * (t - 2.0)), sum(&ty, &|t| t * (t - 1.0) * (t - 2.0)));
    let mut var = (v0 - vt - vu) / 18.0 + t1 * u1 / (2.0 * nf * (nf - 1.0));
    if n >= 3 {
        var += t2 * u2 / (9.0 * nf * (nf - 1.0) * (nf - 2.0));
    }
    let z = if var > 0.0 { s / var.sqrt() } else { f64::NAN };
    let no_ties = n1 == 0 && n2 == 0;
    let (p_value, exact) = if no_ties && n <= KENDALL_EXACT_MAX_N {
        let (le, ge) = inversion_tail_probs(n, d);
        ((2.0 * le.min(ge)).min(1.0), true)
    } else if var > 0.0 {
        ((2.0 * Normal::standard().sf(z.abs())).min(1.0), false)
    } else {
        (f64::NAN, false)
    };
    Ok(Kendall {
        n,
        concordant: c,
        discordant: d,
        ties_x: n1 - n3,
        ties_y: n2 - n3,
        ties_both: n3,
        tau_a,
        tau_b,
        tau_c,
        z,
        p_value,
        exact,
    })
}

/// Partial correlation of two variables controlling for others.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartialCorrelation {
    pub r: f64,
    /// Rows used (listwise-complete over all variables).
    pub n: usize,
    /// Degrees of freedom n − 2 − k.
    pub df: usize,
    pub t: f64,
    pub p_value: f64,
    /// Fisher-z interval with n − k observations; `None` when n − k ≤ 3.
    pub ci: Option<Interval>,
}

/// Partial correlation of `x` and `y` given `controls`, from the inverse of
/// the covariance matrix of all variables (listwise deletion). Errors if the
/// variables are collinear or too few rows remain (need n − k − 2 ≥ 1).
pub fn partial_correlation(x: &[f64], y: &[f64], controls: &[Vec<f64>], level: f64) -> Result<PartialCorrelation> {
    check_level(level)?;
    let mut cols: Vec<Vec<f64>> = vec![x.to_vec(), y.to_vec()];
    cols.extend(controls.iter().cloned());
    let rows = check_cols(&cols)?;
    let keep: Vec<usize> = (0..rows).filter(|&i| cols.iter().all(|c| !c[i].is_nan())).collect();
    let (n, k) = (keep.len(), controls.len());
    if n < k + 3 {
        return Err(StatsError::InsufficientData { what: "partial correlation", needed: k + 3, got: n });
    }
    let used: Vec<Vec<f64>> = cols.iter().map(|c| keep.iter().map(|&i| c[i]).collect()).collect();
    let cov = covariance_matrix(&used, MissingData::Listwise)?;
    let p = used.len();
    for i in 0..p {
        if cov.values[i][i] <= 0.0 {
            return invalid("partial correlation is undefined when a variable is constant");
        }
    }
    let m = Matrix::from_rows(&cov.values)?;
    let lu = Lu::new(&m)?;
    if lu.is_singular() || lu.rcond() < 1e-13 {
        return invalid("the variables are (numerically) collinear");
    }
    let inv = lu.inverse()?;
    let r = (-inv[(0, 1)] / (inv[(0, 0)] * inv[(1, 1)]).sqrt()).clamp(-1.0, 1.0);
    let df = n - 2 - k;
    let (t, p_value) = t_p(r, df as f64)?;
    Ok(PartialCorrelation { r, n, df, t, p_value, ci: fisher_interval(r, (n - k) as f64, level) })
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    const Y: [f64; 8] = [2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 8.0, 6.0];

    #[test]
    fn covariance_modes() {
        let a = vec![1.0, 2.0, 3.0, f64::NAN, 5.0];
        let b = vec![2.0, 4.0, f64::NAN, 8.0, 10.0];
        let c = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let pw = covariance_matrix(&[a.clone(), b.clone(), c.clone()], MissingData::Pairwise).unwrap();
        assert_eq!(pw.n[0][1], 3);
        assert!((pw.values[0][1] - 26.0 / 3.0).abs() < 1e-13); // (1,2),(2,4),(5,10)
        assert_eq!(pw.n[0][0], 4);
        let lw = covariance_matrix(&[a, b, c], MissingData::Listwise).unwrap();
        assert_eq!(lw.n[0][0], 3);
        assert_eq!(lw.n[1][2], 3);
        assert!(covariance_matrix(&[vec![1.0, 2.0], vec![1.0]], MissingData::Pairwise).is_err());
        assert!(covariance_matrix(&[], MissingData::Pairwise).is_err());
        let one = covariance_matrix(&[vec![1.0, f64::NAN]], MissingData::Pairwise).unwrap();
        assert!(one.values[0][0].is_nan());
    }

    #[test]
    fn pearson_basic_and_edges() {
        let p = pearson(&X, &Y, 0.95).unwrap();
        // SciPy 1.18: r, p, Fisher-z CI
        assert!((p.r - 0.8333333333333331).abs() < 1e-14);
        assert!((p.p_value.unwrap() - 0.010175540123456842).abs() < 1e-12);
        assert_eq!(p.n, 8);
        let ci = p.ci.unwrap();
        assert!((ci.lower - 0.31169805718344).abs() < 1e-11 && (ci.upper - 0.968989208891987).abs() < 1e-11);
        let perfect = pearson(&X, &X, 0.95).unwrap();
        assert_eq!(perfect.r, 1.0);
        assert_eq!(perfect.p_value, Some(0.0));
        assert!(pearson(&X, &[1.0; 8], 0.95).is_err());
        assert!(pearson(&[1.0], &[1.0], 0.95).is_err());
        assert!(pearson(&X, &Y[..7], 0.95).is_err());
        let two = pearson(&[1.0, 2.0], &[3.0, 5.0], 0.95).unwrap();
        assert!(two.p_value.is_none() && two.ci.is_none());
        let three = pearson(&[1.0, 2.0, 4.0], &[3.0, 5.0, 4.0], 0.95).unwrap();
        assert!(three.p_value.is_some() && three.ci.is_none());
        assert!(pearson(&X, &Y, 1.0).is_err());
    }

    #[test]
    fn matrix_layout() {
        let cols = vec![X.to_vec(), Y.to_vec(), vec![1.0; 8]];
        let m = correlation_matrix(&cols, MissingData::Pairwise, 0.95).unwrap();
        assert_eq!(m.r[0][0], 1.0);
        assert!(m.r[2][2].is_nan() && m.r[0][2].is_nan()); // constant variable
        assert!((m.r[0][1] - m.r[1][0]).abs() < 1e-15);
        assert!(m.p_value[0][1] > 0.0 && m.p_value[0][1] < 0.05);
        let sp = pairwise_matrix(&cols, MissingData::Pairwise, |a, b| spearman(a, b).ok().map(|s| s.rho)).unwrap();
        assert!((sp[0][1] - 0.8333333333333335).abs() < 1e-14);
    }

    #[test]
    fn ranks_with_ties() {
        assert_eq!(average_ranks(&[10.0, 20.0, 10.0, 30.0]).unwrap(), vec![1.5, 3.0, 1.5, 4.0]);
        assert!(average_ranks(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn spearman_and_kendall_small() {
        let s = spearman(&X, &Y).unwrap();
        assert!((s.rho - 0.8333333333333335).abs() < 1e-14);
        assert!((s.p_value.unwrap() - 0.010175540123456752).abs() < 1e-12);
        assert!(spearman(&X, &[2.0; 8]).is_err());
        let k = kendall(&X, &Y).unwrap();
        assert_eq!((k.concordant, k.discordant), (23, 5));
        assert!((k.tau_b - 0.6428571428571428).abs() < 1e-14 && (k.tau_a - k.tau_b).abs() < 1e-15);
        assert!((k.tau_c - 0.6428571428571428).abs() < 1e-14);
        assert!(k.exact);
        assert!((k.p_value - 0.03115079365079365).abs() < 1e-12); // SciPy exact
                                                                  // monotone → tau 1, exact p = 2/n!
        let k = kendall(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!((k.tau_a, k.tau_b, k.discordant), (1.0, 1.0, 0));
        assert!((k.p_value - 2.0 / 24.0).abs() < 1e-14);
    }

    #[test]
    fn kendall_ties_counts() {
        let x = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0];
        let y = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0];
        let k = kendall(&x, &y).unwrap();
        // brute force
        let (mut c, mut d, mut tx, mut ty, mut tb) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for i in 0..6 {
            for j in i + 1..6 {
                let (a, b) = (x[i] - x[j], y[i] - y[j]);
                match (a == 0.0, b == 0.0) {
                    (true, true) => tb += 1,
                    (true, false) => tx += 1,
                    (false, true) => ty += 1,
                    _ => {
                        if a * b > 0.0 {
                            c += 1
                        } else {
                            d += 1
                        }
                    }
                }
            }
        }
        assert_eq!((k.concordant, k.discordant, k.ties_x, k.ties_y, k.ties_both), (c, d, tx, ty, tb));
        assert!(!k.exact);
        assert!(kendall(&x, &[1.0; 6]).unwrap().tau_b.is_nan());
        assert!(kendall(&[1.0], &[1.0]).is_err());
    }

    #[test]
    fn inversion_distribution_sums_to_one() {
        let (le, ge) = inversion_tail_probs(10, 0);
        assert!((le - 1.0 / 3628800.0).abs() < 1e-18 && (ge - 1.0).abs() < 1e-12);
        // n = 4: counts of permutations by inversions 1,3,5,6,5,3,1 (of 24)
        let (le, ge) = inversion_tail_probs(4, 2);
        assert!((le - 9.0 / 24.0).abs() < 1e-15 && (ge - 20.0 / 24.0).abs() < 1e-15);
        let mut v = vec![3.0, 1.0, 2.0];
        assert_eq!(inversions(&mut v), 2);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn partial_correlation_matches_residuals() {
        let z = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0];
        let pc = partial_correlation(&X, &Y, &[z.to_vec()], 0.95).unwrap();
        // residual method
        let resid = |v: &[f64]| -> Vec<f64> {
            let (mz, mv) = (z.iter().sum::<f64>() / 8.0, v.iter().sum::<f64>() / 8.0);
            let b = z.iter().zip(v).map(|(a, c)| (a - mz) * (c - mv)).sum::<f64>()
                / z.iter().map(|a| (a - mz).powi(2)).sum::<f64>();
            v.iter().zip(&z).map(|(c, a)| (c - mv) - b * (a - mz)).collect()
        };
        let r = pearson(&resid(&X), &resid(&Y), 0.95).unwrap().r;
        assert!((pc.r - r).abs() < 1e-12);
        assert_eq!((pc.n, pc.df), (8, 5));
        assert!(partial_correlation(&X, &Y, &[X.to_vec()], 0.95).is_err()); // collinear
        assert!(partial_correlation(&X[..3], &Y[..3], &[z[..3].to_vec()], 0.95).is_err()); // too few
                                                                                           // missing rows are dropped listwise
        let mut x2 = X.to_vec();
        x2[0] = f64::NAN;
        assert_eq!(partial_correlation(&x2, &Y, &[z.to_vec()], 0.95).unwrap().n, 7);
    }
}
