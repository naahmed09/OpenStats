//! Data behind Minitab's "Graphical Summary" and the basic plots that
//! consume it (histogram, ECDF, box plot, normal probability plot).
//!
//! Nothing here draws; every function returns plain numbers for the chart
//! layer (Phase 4).

use crate::descriptive::{describe, Descriptives};
use crate::error::{invalid, Result, StatsError};
use crate::quantile::{quantile_sorted, QuantileMethod};
use openstats_math::dist::{Binomial, ChiSquared, ContinuousDistribution, DiscreteDistribution, Normal, StudentT};

/// How histogram bin edges are chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum Binning {
    /// ⌈log₂ n + 1⌉ equal-width bins over [min, max].
    Sturges,
    /// Width 3.49·s·n^(−1/3).
    Scott,
    /// Width 2·IQR·n^(−1/3) (IQR by Hyndman–Fan type 7); falls back to
    /// Sturges when the IQR is 0.
    FreedmanDiaconis,
    /// Exactly this many equal-width bins over [min, max].
    Count(usize),
    /// Bins of this width with edges at multiples of the width.
    Width(f64),
    /// Explicit, strictly increasing edges.
    Edges(Vec<f64>),
}

/// Histogram with left-closed intervals `[e_i, e_{i+1})`; the last interval is
/// closed on the right so the maximum is counted.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// `counts.len() + 1` strictly increasing edges.
    pub edges: Vec<f64>,
    pub counts: Vec<usize>,
    /// Count / (n · width): integrates to the fraction of data inside the edges.
    pub density: Vec<f64>,
    /// Observations used (non-missing).
    pub n: usize,
    /// Observations outside explicit edges (always 0 for automatic bins).
    pub outside: usize,
}

/// Histogram of the non-missing values.
pub fn histogram(data: &[f64], binning: &Binning) -> Result<Histogram> {
    let v = crate::clean_sorted(data)?;
    let n = v.len();
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "histogram", needed: 1, got: 0 });
    }
    let (lo, hi) = (v[0], v[n - 1]);
    let nf = n as f64;
    let equal = |k: usize| -> Vec<f64> {
        if lo == hi {
            return vec![lo - 0.5, hi + 0.5];
        }
        let w = (hi - lo) / k as f64;
        let mut e: Vec<f64> = (0..=k).map(|i| lo + w * i as f64).collect();
        e[k] = hi;
        e
    };
    let sd = if n >= 2 { openstats_math::stats::summary::std_dev(&v)? } else { 0.0 };
    let by_width = |w: f64| -> Result<Vec<f64>> {
        if !(w.is_finite() && w > 0.0) {
            return invalid(format!("bin width {w} must be positive"));
        }
        let start = (lo / w).floor() * w;
        let k = (((hi - start) / w).ceil() as usize).max(1);
        if k > 10_000_000 {
            return Err(StatsError::TooLarge(format!("{k} bins")));
        }
        let mut e: Vec<f64> = (0..=k).map(|i| start + w * i as f64).collect();
        if *e.last().unwrap() < hi {
            e.push(hi);
        }
        Ok(e)
    };
    let sturges = || equal(((nf.log2() + 1.0).ceil() as usize).max(1));
    let edges: Vec<f64> = match binning {
        Binning::Sturges => sturges(),
        Binning::Scott => {
            if sd == 0.0 {
                equal(1)
            } else {
                by_width(3.49 * sd / nf.cbrt())?
            }
        }
        Binning::FreedmanDiaconis => {
            let iqr =
                quantile_sorted(&v, 0.75, QuantileMethod::Type7)? - quantile_sorted(&v, 0.25, QuantileMethod::Type7)?;
            if iqr > 0.0 {
                by_width(2.0 * iqr / nf.cbrt())?
            } else {
                sturges()
            }
        }
        Binning::Count(k) => {
            if *k == 0 || *k > 10_000_000 {
                return invalid(format!("bin count {k} must be in 1..=10000000"));
            }
            equal(*k)
        }
        Binning::Width(w) => by_width(*w)?,
        Binning::Edges(e) => {
            if e.len() < 2 || e.windows(2).any(|w| w[0] >= w[1]) || e.iter().any(|x| !x.is_finite()) {
                return invalid("bin edges must be finite and strictly increasing, at least two");
            }
            e.clone()
        }
    };
    let k = edges.len() - 1;
    let mut counts = vec![0usize; k];
    let mut outside = 0;
    for &x in &v {
        if x < edges[0] || x > edges[k] {
            outside += 1;
            continue;
        }
        // last edge i with edges[i] <= x, capped so that x == edges[k] falls in the last bin
        let i = edges.partition_point(|&e| e <= x).saturating_sub(1).min(k - 1);
        counts[i] += 1;
    }
    let density = counts.iter().enumerate().map(|(i, &c)| c as f64 / (nf * (edges[i + 1] - edges[i]))).collect();
    Ok(Histogram { edges, counts, density, n, outside })
}

/// Empirical distribution function: right-continuous step function.
#[derive(Debug, Clone, PartialEq)]
pub struct Ecdf {
    /// Distinct sorted values.
    pub x: Vec<f64>,
    /// F(x\[i\]) = proportion of observations ≤ x\[i\].
    pub cum: Vec<f64>,
    pub n: usize,
}

impl Ecdf {
    /// F(t).
    pub fn eval(&self, t: f64) -> f64 {
        match self.x.partition_point(|&v| v <= t) {
            0 => 0.0,
            i => self.cum[i - 1],
        }
    }
}

/// ECDF of the non-missing values.
pub fn ecdf(data: &[f64]) -> Result<Ecdf> {
    let v = crate::clean_sorted(data)?;
    let n = v.len();
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "ECDF", needed: 1, got: 0 });
    }
    let (mut x, mut cum) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && v[j] == v[i] {
            j += 1;
        }
        x.push(v[i]);
        cum.push(j as f64 / n as f64);
        i = j;
    }
    Ok(Ecdf { x, cum, n })
}

/// Box-plot statistics (Minitab: quartiles by (n+1)p, whiskers reach the most
/// extreme values within 1.5·IQR of the quartiles, others are outliers).
#[derive(Debug, Clone, PartialEq)]
pub struct Boxplot {
    pub n: usize,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub mean: f64,
    pub lower_whisker: f64,
    pub upper_whisker: f64,
    /// Outliers below the lower fence, increasing.
    pub low_outliers: Vec<f64>,
    /// Outliers above the upper fence, increasing.
    pub high_outliers: Vec<f64>,
}

/// Box-plot statistics of the non-missing values.
pub fn boxplot(data: &[f64]) -> Result<Boxplot> {
    let v = crate::clean_sorted(data)?;
    let n = v.len();
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "box plot", needed: 1, got: 0 });
    }
    let q1 = quantile_sorted(&v, 0.25, QuantileMethod::MINITAB)?;
    let q3 = quantile_sorted(&v, 0.75, QuantileMethod::MINITAB)?;
    let median = quantile_sorted(&v, 0.5, QuantileMethod::MINITAB)?;
    let (lf, uf) = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1));
    let lower_whisker = v.iter().copied().find(|&x| x >= lf).unwrap_or(q1);
    let upper_whisker = v.iter().rev().copied().find(|&x| x <= uf).unwrap_or(q3);
    Ok(Boxplot {
        n,
        q1,
        median,
        q3,
        mean: openstats_math::stats::summary::mean(&v)?,
        lower_whisker,
        upper_whisker,
        low_outliers: v.iter().copied().filter(|&x| x < lf).collect(),
        high_outliers: v.iter().copied().filter(|&x| x > uf).collect(),
    })
}

/// Plotting position p_i for the i-th order statistic (1-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotPosition {
    /// (i − 0.3)/(n + 0.4): Benard's median rank, Minitab's default.
    Benard,
    /// (i − 3/8)/(n + 1/4).
    Blom,
    /// (i − 0.5)/n.
    Hazen,
    /// i/(n + 1).
    Weibull,
}

impl PlotPosition {
    /// p_i for sample size `n`.
    pub fn p(self, i: usize, n: usize) -> f64 {
        let (i, n) = (i as f64, n as f64);
        match self {
            PlotPosition::Benard => (i - 0.3) / (n + 0.4),
            PlotPosition::Blom => (i - 0.375) / (n + 0.25),
            PlotPosition::Hazen => (i - 0.5) / n,
            PlotPosition::Weibull => i / (n + 1.0),
        }
    }
}

/// One point of a normal probability plot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityPoint {
    /// Order statistic.
    pub x: f64,
    /// Plotting position (cumulative probability).
    pub p: f64,
    /// Standard normal quantile of `p`.
    pub z: f64,
}

/// Normal probability plot points, in increasing order of `x`. Tied values
/// receive consecutive ranks.
pub fn normal_probability_plot(data: &[f64], position: PlotPosition) -> Result<Vec<ProbabilityPoint>> {
    let v = crate::clean_sorted(data)?;
    let n = v.len();
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "probability plot", needed: 1, got: 0 });
    }
    let z = Normal::standard();
    Ok(v.iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = position.p(i + 1, n);
            ProbabilityPoint { x, p, z: z.quantile(p) }
        })
        .collect())
}

/// A two-sided confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
    /// Requested confidence level, e.g. 0.95.
    pub level: f64,
}

fn check_level(level: f64) -> Result<()> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        invalid(format!("confidence level {level} must be in (0, 1)"))
    }
}

/// t interval for the mean.
pub fn ci_mean(data: &[f64], level: f64) -> Result<Interval> {
    check_level(level)?;
    let d = describe(data)?;
    let (mean, se) = match (d.mean, d.se_mean) {
        (Some(m), Some(s)) if d.n >= 2 => (m, s),
        _ => {
            return Err(StatsError::InsufficientData { what: "confidence interval for the mean", needed: 2, got: d.n })
        }
    };
    let t = StudentT::new((d.n - 1) as f64)?.isf((1.0 - level) / 2.0);
    Ok(Interval { lower: mean - t * se, upper: mean + t * se, level })
}

/// χ² interval for the standard deviation (normal theory).
pub fn ci_std_dev(data: &[f64], level: f64) -> Result<Interval> {
    check_level(level)?;
    let d = describe(data)?;
    let var = match d.variance {
        Some(v) if d.n >= 2 => v,
        _ => {
            return Err(StatsError::InsufficientData {
                what: "confidence interval for the standard deviation",
                needed: 2,
                got: d.n,
            })
        }
    };
    let df = (d.n - 1) as f64;
    let chi = ChiSquared::new(df)?;
    let a = (1.0 - level) / 2.0;
    Ok(Interval { lower: (df * var / chi.isf(a)).sqrt(), upper: (df * var / chi.quantile(a)).sqrt(), level })
}

/// Distribution-free interval for the median (order statistics).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MedianInterval {
    pub lower: f64,
    pub upper: f64,
    /// Confidence actually achieved, 1 − 2·P(Bin(n, ½) ≤ l − 1) ≥ the request.
    pub achieved: f64,
    /// 1-based rank of the lower end; the upper end is rank `n + 1 − rank`.
    pub rank: usize,
}

/// Interval `[x_(l), x_(n+1−l)]` with the largest `l` whose achieved
/// confidence is at least `level`. Fails when n is too small to reach `level`.
pub fn ci_median(data: &[f64], level: f64) -> Result<MedianInterval> {
    check_level(level)?;
    let v = crate::clean_sorted(data)?;
    let n = v.len();
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "confidence interval for the median", needed: 1, got: 0 });
    }
    let b = Binomial::new(n as u64, 0.5)?;
    let mut best: Option<(usize, f64)> = None;
    for l in 1..=n.div_ceil(2) {
        let achieved = 1.0 - 2.0 * b.cdf((l - 1) as f64);
        if achieved >= level - 1e-12 {
            best = Some((l, achieved));
        } else {
            break;
        }
    }
    match best {
        Some((l, achieved)) => Ok(MedianInterval { lower: v[l - 1], upper: v[n - l], achieved, rank: l }),
        None => invalid(format!("n = {n} is too small for a {:.1}% confidence interval for the median", level * 100.0)),
    }
}

/// Anderson–Darling normality test with mean and SD estimated from the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AndersonDarling {
    /// A².
    pub a2: f64,
    /// A²·(1 + 0.75/n + 2.25/n²), the statistic the p-value is based on.
    pub a2_adjusted: f64,
    /// p-value from the D'Agostino–Stephens approximation (case 4).
    pub p_value: f64,
    pub n: usize,
}

/// Anderson–Darling test of normality. Needs n ≥ 3 and positive variance.
pub fn anderson_darling(data: &[f64]) -> Result<AndersonDarling> {
    let v = crate::clean_sorted(data)?;
    let n = v.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "Anderson-Darling test", needed: 3, got: n });
    }
    let mean = openstats_math::stats::summary::mean(&v)?;
    let sd = openstats_math::stats::summary::std_dev(&v)?;
    if sd == 0.0 {
        return invalid("Anderson-Darling test needs a positive standard deviation");
    }
    let norm = Normal::standard();
    let ln_floor = |p: f64| p.max(f64::MIN_POSITIVE).ln();
    let nf = n as f64;
    let mut s = 0.0;
    for i in 0..n {
        let zi = (v[i] - mean) / sd;
        let zj = (v[n - 1 - i] - mean) / sd;
        s += (2 * i + 1) as f64 * (ln_floor(norm.cdf(zi)) + ln_floor(norm.sf(zj)));
    }
    let a2 = -nf - s / nf;
    let adj = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));
    let p = if adj >= 0.6 {
        (1.2937 - 5.709 * adj + 0.0186 * adj * adj).exp()
    } else if adj >= 0.34 {
        (0.9177 - 4.279 * adj - 1.38 * adj * adj).exp()
    } else if adj >= 0.2 {
        1.0 - (-8.318 + 42.796 * adj - 59.938 * adj * adj).exp()
    } else {
        1.0 - (-13.436 + 101.14 * adj - 223.73 * adj * adj).exp()
    };
    Ok(AndersonDarling { a2, a2_adjusted: adj, p_value: p.clamp(0.0, 1.0), n })
}

/// Everything Minitab's Graphical Summary reports, plus chart data.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphicalSummary {
    pub descriptives: Descriptives,
    pub anderson_darling: Option<AndersonDarling>,
    pub ci_mean: Option<Interval>,
    pub ci_median: Option<MedianInterval>,
    pub ci_std_dev: Option<Interval>,
    pub histogram: Option<Histogram>,
    pub boxplot: Option<Boxplot>,
    /// Mean and SD of the normal curve overlaid on the histogram.
    pub normal_fit: Option<(f64, f64)>,
}

/// Compute a Graphical Summary. Components that need more data than is
/// available (or are undefined, such as the A–D test of constant data) are
/// `None`; only invalid arguments or infinite data are errors.
pub fn graphical_summary(data: &[f64], level: f64, binning: &Binning) -> Result<GraphicalSummary> {
    check_level(level)?;
    let descriptives = describe(data)?;
    let have = descriptives.n > 0;
    Ok(GraphicalSummary {
        anderson_darling: anderson_darling(data).ok(),
        ci_mean: ci_mean(data, level).ok(),
        ci_median: ci_median(data, level).ok(),
        ci_std_dev: ci_std_dev(data, level).ok(),
        histogram: if have { Some(histogram(data, binning)?) } else { None },
        boxplot: if have { Some(boxplot(data)?) } else { None },
        normal_fit: descriptives.mean.zip(descriptives.std_dev),
        descriptives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts_and_edges() {
        let x = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, f64::NAN];
        let h = histogram(&x, &Binning::Edges(vec![1.0, 3.0, 5.0])).unwrap();
        assert_eq!(h.counts, vec![3, 3]); // [1,3): 1,2,2 ; [3,5]: 3,4,5
        assert_eq!(h.n, 6);
        let h = histogram(&x, &Binning::Count(4)).unwrap();
        assert_eq!(h.counts.iter().sum::<usize>(), 6);
        assert_eq!(h.edges, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(h.counts, vec![1, 2, 1, 2]);
        let h = histogram(&x, &Binning::Width(2.0)).unwrap();
        assert_eq!(h.edges, vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(h.counts, vec![1, 3, 2]);
        let d: f64 = h.density.iter().zip(h.edges.windows(2)).map(|(d, e)| d * (e[1] - e[0])).sum();
        assert!((d - 1.0).abs() < 1e-14);
        // constant data
        let h = histogram(&[7.0, 7.0], &Binning::Sturges).unwrap();
        assert_eq!(h.counts, vec![2]);
        assert!(histogram(&[], &Binning::Sturges).is_err());
        assert!(histogram(&x, &Binning::Edges(vec![2.0, 1.0])).is_err());
        assert!(histogram(&x, &Binning::Width(0.0)).is_err());
        let h = histogram(&x, &Binning::Edges(vec![2.0, 4.0])).unwrap();
        assert_eq!(h.outside, 2);
    }

    #[test]
    fn automatic_rules() {
        let x: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.37).sin() * 10.0).collect();
        let s = histogram(&x, &Binning::Sturges).unwrap();
        assert_eq!(s.counts.len(), 8); // ceil(log2(100)+1)
        for b in [Binning::Scott, Binning::FreedmanDiaconis] {
            let h = histogram(&x, &b).unwrap();
            assert_eq!(h.counts.iter().sum::<usize>(), 100);
        }
    }

    #[test]
    fn ecdf_steps() {
        let e = ecdf(&[3.0, 1.0, 1.0, 2.0, f64::NAN]).unwrap();
        assert_eq!(e.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(e.cum, vec![0.5, 0.75, 1.0]);
        assert_eq!(e.eval(0.0), 0.0);
        assert_eq!(e.eval(1.0), 0.5);
        assert_eq!(e.eval(2.5), 0.75);
        assert_eq!(e.eval(9.0), 1.0);
    }

    #[test]
    fn boxplot_outliers() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0];
        let b = boxplot(&x).unwrap();
        assert_eq!((b.q1, b.median, b.q3), (2.5, 5.0, 7.5));
        assert_eq!(b.upper_whisker, 8.0);
        assert_eq!(b.high_outliers, vec![30.0]);
        assert!(b.low_outliers.is_empty());
        assert_eq!(b.lower_whisker, 1.0);
    }

    #[test]
    fn plot_positions() {
        let pts = normal_probability_plot(&[5.0, 1.0, 3.0], PlotPosition::Benard).unwrap();
        assert_eq!(pts.iter().map(|p| p.x).collect::<Vec<_>>(), vec![1.0, 3.0, 5.0]);
        assert!((pts[0].p - 0.7 / 3.4).abs() < 1e-15);
        assert!(pts[1].z.abs() < 1e-14); // symmetric
        assert!((pts[0].z + pts[2].z).abs() < 1e-14);
        assert!((PlotPosition::Weibull.p(1, 3) - 0.25).abs() < 1e-15);
        assert!((PlotPosition::Hazen.p(1, 4) - 0.125).abs() < 1e-15);
        assert!((PlotPosition::Blom.p(1, 3) - 0.625 / 3.25).abs() < 1e-15);
    }

    #[test]
    fn intervals() {
        let x = [10.0, 12.0, 9.0, 11.0, 13.0, 10.5, 9.5, 12.5];
        let ci = ci_mean(&x, 0.95).unwrap();
        assert!(ci.lower < 11.0 && ci.upper > 11.0);
        let sd = ci_std_dev(&x, 0.95).unwrap();
        let s = describe(&x).unwrap().std_dev.unwrap();
        assert!(sd.lower < s && s < sd.upper);
        // n = 8: P(Bin(8,.5) <= 0) = 1/256 → 99.2%; l = 2: P(<=1)=9/256 → 93.0%
        let m = ci_median(&x, 0.9).unwrap();
        assert_eq!(m.rank, 2);
        assert!((m.achieved - (1.0 - 18.0 / 256.0)).abs() < 1e-12);
        assert_eq!((m.lower, m.upper), (9.5, 12.5));
        assert!(ci_median(&[1.0, 2.0], 0.95).is_err());
        assert!(ci_mean(&[1.0], 0.95).is_err());
        assert!(ci_mean(&x, 1.5).is_err());
    }

    #[test]
    fn anderson_darling_regions() {
        // a normal-quantile sample is very close to normal: large p
        let z = Normal::standard();
        let x: Vec<f64> = (1..=50).map(|i| z.quantile((f64::from(i) - 0.5) / 50.0)).collect();
        let ad = anderson_darling(&x).unwrap();
        assert!(ad.p_value > 0.9, "{ad:?}");
        // exponential-like sample: small p
        let y: Vec<f64> = (1..=50).map(|i| -(1.0 - (f64::from(i) - 0.5) / 50.0).ln()).collect();
        let ad = anderson_darling(&y).unwrap();
        assert!(ad.p_value < 0.005, "{ad:?}");
        assert!(anderson_darling(&[1.0, 1.0, 1.0]).is_err());
        assert!(anderson_darling(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn summary_bundle() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 40.0, f64::NAN];
        let g = graphical_summary(&x, 0.95, &Binning::Sturges).unwrap();
        assert_eq!(g.descriptives.n, 10);
        assert!(g.anderson_darling.is_some() && g.ci_mean.is_some() && g.histogram.is_some());
        let g = graphical_summary(&[], 0.95, &Binning::Sturges).unwrap();
        assert!(g.histogram.is_none() && g.boxplot.is_none() && g.normal_fit.is_none());
        assert!(graphical_summary(&x, 0.0, &Binning::Sturges).is_err());
    }
}
