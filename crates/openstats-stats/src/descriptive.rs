//! Minitab "Display Descriptive Statistics".
//!
//! Definitions (n = number of non-missing values, s = sample SD):
//!
//! * SE mean = s/√n; CV = 100·s/mean; range = max − min.
//! * Q1, Q3 use Minitab's (n+1)p interpolation ([`QuantileMethod::MINITAB`]);
//!   the median is the middle value (average of the two middle values).
//! * Trimmed mean: the k smallest and k largest values are removed, with
//!   k = round(0.05·n) (halves round up), as Minitab's `TrMean`.
//! * Skewness = n/((n−1)(n−2)) Σ((x−x̄)/s)³ and excess kurtosis
//!   = n(n+1)/((n−1)(n−2)(n−3)) Σ((x−x̄)/s)⁴ − 3(n−1)²/((n−2)(n−3)) (the
//!   bias-adjusted forms Minitab reports).
//! * MSSD = Σ(x_{i+1} − x_i)² / (2(n−1)) over consecutive non-missing values.
//! * Mode: the most frequent value(s); when every value is unique there is no
//!   mode (Minitab shows `*`).
//!
//! Quantities that need more observations than available (or are undefined,
//! such as the CV of a zero-mean sample) are `None`.

use crate::error::{invalid, Result};
use crate::quantile::{quantile_sorted, QuantileMethod};
use openstats_math::stats::summary;

/// All Minitab summary statistics of one sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Descriptives {
    /// Non-missing observations.
    pub n: usize,
    /// Missing observations (`NaN`).
    pub n_missing: usize,
    /// All observations, `n + n_missing`.
    pub n_total: usize,
    pub mean: Option<f64>,
    pub se_mean: Option<f64>,
    /// 5 % trimmed mean.
    pub trimmed_mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub variance: Option<f64>,
    /// Coefficient of variation in percent.
    pub coef_var: Option<f64>,
    pub sum: Option<f64>,
    pub sum_of_squares: Option<f64>,
    pub minimum: Option<f64>,
    pub q1: Option<f64>,
    pub median: Option<f64>,
    pub q3: Option<f64>,
    pub maximum: Option<f64>,
    pub range: Option<f64>,
    pub iqr: Option<f64>,
    /// Modes in increasing order (empty when all values are distinct).
    pub modes: Vec<f64>,
    /// Frequency of each mode (0 when there is no mode).
    pub mode_count: usize,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
    /// Mean square successive difference.
    pub mssd: Option<f64>,
}

/// Fraction trimmed from each end by Minitab's trimmed mean.
pub const MINITAB_TRIM: f64 = 0.05;

/// Trimmed mean removing `k = round(fraction·n)` values from each end.
///
/// `fraction` must lie in `[0, 0.5)`. Missing values are ignored.
pub fn trimmed_mean(data: &[f64], fraction: f64) -> Result<Option<f64>> {
    if !(0.0..0.5).contains(&fraction) {
        return invalid(format!("trim fraction {fraction} must be in [0, 0.5)"));
    }
    let v = crate::clean_sorted(data)?;
    Ok(trimmed_mean_sorted(&v, fraction))
}

fn trimmed_mean_sorted(sorted: &[f64], fraction: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let k = (fraction * n as f64 + 0.5).floor() as usize;
    if 2 * k >= n {
        return None;
    }
    summary::mean(&sorted[k..n - k]).ok()
}

/// Mean square successive difference of the non-missing values in order.
pub fn mssd(data: &[f64]) -> Result<Option<f64>> {
    let v = crate::clean(data)?;
    if v.len() < 2 {
        return Ok(None);
    }
    let s = summary::sum(v.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])));
    Ok(Some(s / (2.0 * (v.len() - 1) as f64)))
}

/// Compute every statistic of `data`. `NaN` is missing; ±∞ is an error.
pub fn describe(data: &[f64]) -> Result<Descriptives> {
    let v = crate::clean(data)?;
    let n = v.len();
    let mut d = Descriptives {
        n,
        n_missing: data.len() - n,
        n_total: data.len(),
        mean: None,
        se_mean: None,
        trimmed_mean: None,
        std_dev: None,
        variance: None,
        coef_var: None,
        sum: None,
        sum_of_squares: None,
        minimum: None,
        q1: None,
        median: None,
        q3: None,
        maximum: None,
        range: None,
        iqr: None,
        modes: Vec::new(),
        mode_count: 0,
        skewness: None,
        kurtosis: None,
        mssd: None,
    };
    if n == 0 {
        return Ok(d);
    }
    let nf = n as f64;
    let mut sorted = v.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mean = summary::mean(&v)?;
    d.mean = Some(mean);
    d.sum = Some(summary::sum(v.iter().copied()));
    d.sum_of_squares = Some(summary::sum(v.iter().map(|x| x * x)));
    d.minimum = Some(sorted[0]);
    d.maximum = Some(sorted[n - 1]);
    d.range = Some(sorted[n - 1] - sorted[0]);
    d.median = Some(quantile_sorted(&sorted, 0.5, QuantileMethod::Type7)?);
    d.q1 = Some(quantile_sorted(&sorted, 0.25, QuantileMethod::MINITAB)?);
    d.q3 = Some(quantile_sorted(&sorted, 0.75, QuantileMethod::MINITAB)?);
    d.iqr = Some(d.q3.unwrap() - d.q1.unwrap());
    d.trimmed_mean = trimmed_mean_sorted(&sorted, MINITAB_TRIM);

    // Modes.
    let (mut best, mut i) = (0usize, 0usize);
    let mut counts: Vec<(f64, usize)> = Vec::new();
    while i < n {
        let mut j = i;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        counts.push((sorted[i], j - i));
        best = best.max(j - i);
        i = j;
    }
    if best > 1 {
        d.modes = counts.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
        d.mode_count = best;
    }

    d.mssd = mssd(&v)?;

    if n >= 2 {
        let var = summary::variance(&v)?;
        let sd = var.sqrt();
        d.variance = Some(var);
        d.std_dev = Some(sd);
        d.se_mean = Some(sd / nf.sqrt());
        if mean != 0.0 {
            d.coef_var = Some(100.0 * sd / mean);
        }
        if sd > 0.0 && n >= 3 {
            let m3 = summary::sum(v.iter().map(|x| ((x - mean) / sd).powi(3)));
            d.skewness = Some(nf / ((nf - 1.0) * (nf - 2.0)) * m3);
            if n >= 4 {
                let m4 = summary::sum(v.iter().map(|x| ((x - mean) / sd).powi(4)));
                d.kurtosis = Some(
                    nf * (nf + 1.0) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0)) * m4
                        - 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0)),
                );
            }
        }
    }
    Ok(d)
}

/// Statistics of `values` for each level of a by-variable.
///
/// `groups[i]` is the level of `values[i]`; an empty level label is a missing
/// level and the row is excluded. Levels are returned in increasing label
/// order (text order).
pub fn describe_by(values: &[f64], groups: &[String]) -> Result<Vec<(String, Descriptives)>> {
    if values.len() != groups.len() {
        return invalid(format!("{} values but {} group labels", values.len(), groups.len()));
    }
    let mut levels: Vec<&str> = groups.iter().map(String::as_str).filter(|g| !g.is_empty()).collect();
    levels.sort_unstable();
    levels.dedup();
    levels
        .into_iter()
        .map(|lv| {
            let sub: Vec<f64> = values.iter().zip(groups).filter(|(_, g)| g.as_str() == lv).map(|(v, _)| *v).collect();
            describe(&sub).map(|d| (lv.to_string(), d))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sample() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0, f64::NAN];
        let d = describe(&x).unwrap();
        assert_eq!((d.n, d.n_missing, d.n_total), (8, 1, 9));
        assert_eq!(d.mean, Some(5.0));
        assert!((d.variance.unwrap() - 32.0 / 7.0).abs() < 1e-14);
        assert_eq!(d.median, Some(4.5));
        assert_eq!(d.modes, vec![4.0]);
        assert_eq!(d.mode_count, 3);
        assert_eq!(d.range, Some(7.0));
        // n=8, 5%: round(0.4) = 0 values trimmed
        assert_eq!(d.trimmed_mean, Some(5.0));
        // MSSD = (4+0+0+1+0+4+4)/(2*7)
        assert!((d.mssd.unwrap() - 13.0 / 14.0).abs() < 1e-14);
    }

    #[test]
    fn degenerate_cases() {
        let d = describe(&[]).unwrap();
        assert_eq!(d.n, 0);
        assert!(d.mean.is_none());
        let d = describe(&[f64::NAN, 3.0]).unwrap();
        assert_eq!(d.mean, Some(3.0));
        assert!(d.std_dev.is_none() && d.skewness.is_none() && d.mssd.is_none());
        let d = describe(&[1.0, 2.0, 3.0]).unwrap();
        assert!(d.kurtosis.is_none());
        assert!(d.modes.is_empty());
        let d = describe(&[5.0, 5.0, 5.0, 5.0]).unwrap();
        assert!(d.skewness.is_none() && d.kurtosis.is_none());
        assert_eq!(d.std_dev, Some(0.0));
        let d = describe(&[-1.0, 1.0]).unwrap();
        assert!(d.coef_var.is_none());
        assert!(describe(&[1.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn trimming_rounds_to_nearest() {
        // n = 30 → 0.05·30 = 1.5 → k = 2
        let x: Vec<f64> = (1..=30).map(f64::from).collect();
        assert_eq!(trimmed_mean(&x, 0.05).unwrap(), Some(15.5));
        let mut y = x.clone();
        y[0] = -1000.0;
        y[29] = 1000.0;
        assert_eq!(trimmed_mean(&y, 0.05).unwrap(), Some(15.5));
        assert!(trimmed_mean(&x, 0.5).is_err());
        assert_eq!(trimmed_mean(&[1.0, 2.0], 0.49).unwrap(), None);
    }

    #[test]
    fn by_group() {
        let v = [1.0, 2.0, 3.0, 10.0, 20.0, 99.0];
        let g: Vec<String> = ["b", "b", "b", "a", "a", ""].iter().map(|s| s.to_string()).collect();
        let r = describe_by(&v, &g).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, "a");
        assert_eq!(r[0].1.mean, Some(15.0));
        assert_eq!(r[1].1.n, 3);
        assert!(describe_by(&v, &g[..2]).is_err());
    }
}
