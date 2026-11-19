//! Time-weighted charts: moving average, EWMA and CUSUM (tabular and V-mask).
//!
//! The statistics are computed on subgroup means (or individual observations,
//! subgroup size 1). With unequal subgroup sizes every point is standardised
//! by its own standard error `σ/√nᵢ`.

use super::charts::{
    estimate_sigma, individuals_sigma_estimate, mean_of, subgroup_stats, ChartPanel, IndividualsSigma, SubgroupSigma,
};
use super::rules::SpecialCauseTests;
use crate::error::{invalid, Result, StatsError};

/// Data and parameters shared by the time-weighted charts.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeWeightedInput {
    /// Individual observations or subgroup means, in time order.
    pub values: Vec<f64>,
    /// Subgroup size of each value (all 1 for individuals).
    pub sizes: Vec<usize>,
    /// Target (process mean).
    pub target: f64,
    /// Process σ of an individual observation.
    pub sigma: f64,
    /// How σ was obtained.
    pub sigma_method: String,
}

impl TimeWeightedInput {
    fn check(&self) -> Result<()> {
        if self.values.is_empty() {
            return Err(StatsError::InsufficientData { what: "time-weighted chart", needed: 1, got: 0 });
        }
        if self.values.len() != self.sizes.len() || self.sizes.contains(&0) {
            return invalid("values and subgroup sizes must have the same length and sizes must be positive");
        }
        if self.values.iter().any(|v| !v.is_finite()) {
            return invalid("time-weighted charts need finite values (no missing values)");
        }
        if !(self.sigma > 0.0 && self.sigma.is_finite()) || !self.target.is_finite() {
            return invalid("the target must be finite and σ positive");
        }
        Ok(())
    }

    /// Standard error of point `i`.
    pub fn se(&self, i: usize) -> f64 {
        self.sigma / (self.sizes[i] as f64).sqrt()
    }

    /// Individual observations; σ from the average moving range unless given; the target defaults to the mean.
    pub fn individuals(data: &[f64], target: Option<f64>, sigma: Option<f64>) -> Result<Self> {
        if data.iter().any(|v| !v.is_finite()) {
            return invalid("time-weighted charts need finite values (no missing values)");
        }
        if data.is_empty() {
            return Err(StatsError::InsufficientData { what: "time-weighted chart", needed: 1, got: 0 });
        }
        let (s, method) = match sigma {
            Some(s) => (s, "known".to_string()),
            None => individuals_sigma_estimate(data, 2, IndividualsSigma::AverageMovingRange)?,
        };
        let inp = TimeWeightedInput {
            values: data.to_vec(),
            sizes: vec![1; data.len()],
            target: target.unwrap_or_else(|| mean_of(data)),
            sigma: s,
            sigma_method: method,
        };
        inp.check()?;
        Ok(inp)
    }

    /// Subgroup data; σ estimated by `method` (default pooled standard deviation) unless given.
    pub fn subgroups(
        groups: &[Vec<f64>],
        target: Option<f64>,
        sigma: Option<f64>,
        method: Option<SubgroupSigma>,
    ) -> Result<Self> {
        if groups.iter().flatten().any(|v| v.is_nan()) {
            return invalid("time-weighted charts need finite values (no missing values)");
        }
        let st = subgroup_stats(groups)?;
        let (s, m) = match sigma {
            Some(s) => (s, "known".to_string()),
            None => estimate_sigma(&st, method.unwrap_or(SubgroupSigma::Pooled))?,
        };
        let total: f64 = st.iter().map(|x| x.n as f64).sum();
        let inp = TimeWeightedInput {
            values: st.iter().map(|x| x.mean).collect(),
            sizes: st.iter().map(|x| x.n).collect(),
            target: target.unwrap_or_else(|| st.iter().map(|x| x.n as f64 * x.mean).sum::<f64>() / total),
            sigma: s,
            sigma_method: m,
        };
        inp.check()?;
        Ok(inp)
    }
}

/// Which smoothing chart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingKind {
    /// Moving average of the given length.
    MovingAverage(usize),
    /// Exponentially weighted moving average with weight λ; `exact` limits use the exact variance of every point.
    Ewma {
        /// Weight of the newest value.
        lambda: f64,
        /// `true`: widening limits; `false`: the asymptotic limits.
        exact: bool,
    },
}

/// A moving-average or EWMA chart.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothingChart {
    /// Chart kind and parameters.
    pub kind: SmoothingKind,
    /// The smoothed statistic with its limits (test 1 only).
    pub panel: ChartPanel,
    /// Target (centre line).
    pub target: f64,
    /// Process σ.
    pub sigma: f64,
    /// How σ was obtained.
    pub sigma_method: String,
    /// Points beyond the limits.
    pub signals: Vec<usize>,
}

fn check_k(k: f64) -> Result<()> {
    if !(k > 0.0 && k.is_finite()) {
        return invalid(format!("k_sigma must be positive (got {k})"));
    }
    Ok(())
}

fn smoothing_panel(
    name: &str,
    inp: &TimeWeightedInput,
    stat: Vec<f64>,
    se: Vec<f64>,
    k: f64,
    kind: SmoothingKind,
    tests: Option<&SpecialCauseTests>,
) -> Result<SmoothingChart> {
    let default = SpecialCauseTests { beyond_sigma: Some(k), ..SpecialCauseTests::none() };
    let tests = tests.unwrap_or(&default);
    let n = stat.len();
    let panel = ChartPanel::build(
        name,
        stat,
        vec![inp.target; n],
        se.iter().map(|e| inp.target + k * e).collect(),
        se.iter().map(|e| inp.target - k * e).collect(),
        se,
        tests,
    )?;
    let signals = panel.flagged();
    Ok(SmoothingChart {
        kind,
        panel,
        target: inp.target,
        sigma: inp.sigma,
        sigma_method: inp.sigma_method.clone(),
        signals,
    })
}

/// Moving-average chart of length `length`. The first `length − 1` points average what is available;
/// the variance of a window is `σ²/m² · Σ 1/nⱼ` for the `m` values it holds.
pub fn moving_average(inp: &TimeWeightedInput, length: usize, k_sigma: f64) -> Result<SmoothingChart> {
    moving_average_with_tests(inp, length, k_sigma, None)
}

/// [`moving_average`] with a chosen set of special-cause tests (default: test 1 at `k_sigma`).
/// Smoothed points are autocorrelated, so run and zone rules signal more often than their
/// nominal false-alarm rates; use them with that in mind.
pub fn moving_average_with_tests(
    inp: &TimeWeightedInput,
    length: usize,
    k_sigma: f64,
    tests: Option<&SpecialCauseTests>,
) -> Result<SmoothingChart> {
    inp.check()?;
    check_k(k_sigma)?;
    if length == 0 {
        return invalid("the moving-average length must be at least 1");
    }
    let n = inp.values.len();
    let mut stat = Vec::with_capacity(n);
    let mut se = Vec::with_capacity(n);
    for i in 0..n {
        let lo = (i + 1).saturating_sub(length);
        let m = (i + 1 - lo) as f64;
        stat.push(inp.values[lo..=i].iter().sum::<f64>() / m);
        let inv: f64 = inp.sizes[lo..=i].iter().map(|&s| 1.0 / s as f64).sum();
        se.push(inp.sigma * inv.sqrt() / m);
    }
    smoothing_panel(
        &format!("Moving Average (length {length})"),
        inp,
        stat,
        se,
        k_sigma,
        SmoothingKind::MovingAverage(length),
        tests,
    )
}

/// EWMA chart: `zᵢ = λxᵢ + (1 − λ)zᵢ₋₁`, `z₀` = target. Exact variance recursion
/// `vᵢ = (1−λ)²vᵢ₋₁ + λ²σ²/nᵢ`, or the asymptotic `λ/(2−λ)·σ²/nᵢ` when `exact` is false.
pub fn ewma(inp: &TimeWeightedInput, lambda: f64, k_sigma: f64, exact: bool) -> Result<SmoothingChart> {
    ewma_with_tests(inp, lambda, k_sigma, exact, None)
}

/// [`ewma`] with a chosen set of special-cause tests (default: test 1 at `k_sigma`; see
/// [`moving_average_with_tests`] for the caveat on run rules).
pub fn ewma_with_tests(
    inp: &TimeWeightedInput,
    lambda: f64,
    k_sigma: f64,
    exact: bool,
    tests: Option<&SpecialCauseTests>,
) -> Result<SmoothingChart> {
    inp.check()?;
    check_k(k_sigma)?;
    if !(lambda > 0.0 && lambda <= 1.0) {
        return invalid(format!("λ must be in (0, 1] (got {lambda})"));
    }
    let n = inp.values.len();
    let (mut z, mut v) = (inp.target, 0.0);
    let (mut stat, mut se) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        z = lambda * inp.values[i] + (1.0 - lambda) * z;
        let s2 = inp.sigma * inp.sigma / inp.sizes[i] as f64;
        v = (1.0 - lambda) * (1.0 - lambda) * v + lambda * lambda * s2;
        stat.push(z);
        se.push(if exact { v.sqrt() } else { (lambda / (2.0 - lambda) * s2).sqrt() });
    }
    smoothing_panel("EWMA", inp, stat, se, k_sigma, SmoothingKind::Ewma { lambda, exact }, tests)
}

// ------------------------------------------------------------------- CUSUM

/// Parameters of a tabular CUSUM, in standard-error units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CusumOptions {
    /// Reference value K (allowance): half the shift to detect, in σ units (0.5).
    pub k: f64,
    /// Decision interval H in σ units (4).
    pub h: f64,
    /// Head start (fast initial response) S₀ for both sides, in σ units (0).
    pub head_start: f64,
}

impl Default for CusumOptions {
    fn default() -> Self {
        CusumOptions { k: 0.5, h: 4.0, head_start: 0.0 }
    }
}

impl CusumOptions {
    fn check(&self) -> Result<()> {
        if !(self.k > 0.0 && self.k.is_finite() && self.h > 0.0 && self.h.is_finite()) {
            return invalid("CUSUM K and H must be positive");
        }
        if !(self.head_start >= 0.0 && self.head_start < self.h) {
            return invalid("the head start must lie in [0, H)");
        }
        Ok(())
    }
}

/// Which side(s) of a tabular CUSUM are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CusumSides {
    /// Both C⁺ and C⁻ (two-sided).
    Both,
    /// Only C⁺ (detects increases).
    UpperOnly,
    /// Only C⁻ (detects decreases).
    LowerOnly,
}

/// Tabular CUSUM chart. All statistics are in standard-error units (`σ/√nᵢ` per point);
/// [`CusumChart::in_data_units`] converts them when the subgroup size is constant.
#[derive(Debug, Clone, PartialEq)]
pub struct CusumChart {
    /// Sides computed (the other side is all zeros with no signals).
    pub sides: CusumSides,
    /// Parameters.
    pub options: CusumOptions,
    /// Plotted values (observations or subgroup means).
    pub values: Vec<f64>,
    /// Target.
    pub target: f64,
    /// Process σ.
    pub sigma: f64,
    /// How σ was obtained.
    pub sigma_method: String,
    /// Standard error of each point.
    pub se: Vec<f64>,
    /// Upper sums `C⁺ᵢ = max(0, zᵢ − K + C⁺ᵢ₋₁)`.
    pub upper: Vec<f64>,
    /// Lower sums (shown positive) `C⁻ᵢ = max(0, −zᵢ − K + C⁻ᵢ₋₁)`.
    pub lower: Vec<f64>,
    /// Number of points since the upper sum was last 0.
    pub n_upper: Vec<usize>,
    /// Number of points since the lower sum was last 0.
    pub n_lower: Vec<usize>,
    /// Points where `C⁺ > H`.
    pub signals_upper: Vec<usize>,
    /// Points where `C⁻ > H`.
    pub signals_lower: Vec<usize>,
}

impl CusumChart {
    /// Sums in the units of the data: `(C⁺, C⁻, H)` — only when all subgroup sizes are equal.
    pub fn in_data_units(&self) -> Result<(Vec<f64>, Vec<f64>, f64)> {
        let s = self.se[0];
        if self.se.iter().any(|&e| (e - s).abs() > 1e-12 * s) {
            return invalid("data units need a constant subgroup size");
        }
        Ok((self.upper.iter().map(|c| c * s).collect(), self.lower.iter().map(|c| c * s).collect(), self.options.h * s))
    }

    /// Points signalling on either side, sorted.
    pub fn signals(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.signals_upper.iter().chain(&self.signals_lower).copied().collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// Tabular CUSUM of `inp`.
pub fn cusum_tabular(inp: &TimeWeightedInput, opts: &CusumOptions) -> Result<CusumChart> {
    cusum_tabular_sided(inp, opts, CusumSides::Both)
}

/// Tabular CUSUM computing one or both sides (fast initial response through `opts.head_start`).
pub fn cusum_tabular_sided(inp: &TimeWeightedInput, opts: &CusumOptions, sides: CusumSides) -> Result<CusumChart> {
    inp.check()?;
    opts.check()?;
    let n = inp.values.len();
    let (mut cp, mut cm) = (
        if sides == CusumSides::LowerOnly { 0.0 } else { opts.head_start },
        if sides == CusumSides::UpperOnly { 0.0 } else { opts.head_start },
    );
    let (mut np, mut nm) = (0usize, 0usize);
    let (use_up, use_lo) = (sides != CusumSides::LowerOnly, sides != CusumSides::UpperOnly);
    let mut out = CusumChart {
        sides,
        options: *opts,
        values: inp.values.clone(),
        target: inp.target,
        sigma: inp.sigma,
        sigma_method: inp.sigma_method.clone(),
        se: (0..n).map(|i| inp.se(i)).collect(),
        upper: Vec::with_capacity(n),
        lower: Vec::with_capacity(n),
        n_upper: Vec::with_capacity(n),
        n_lower: Vec::with_capacity(n),
        signals_upper: Vec::new(),
        signals_lower: Vec::new(),
    };
    for i in 0..n {
        let z = (inp.values[i] - inp.target) / out.se[i];
        cp = if use_up { (cp + z - opts.k).max(0.0) } else { 0.0 };
        cm = if use_lo { (cm - z - opts.k).max(0.0) } else { 0.0 };
        np = if cp > 0.0 { np + 1 } else { 0 };
        nm = if cm > 0.0 { nm + 1 } else { 0 };
        out.upper.push(cp);
        out.lower.push(cm);
        out.n_upper.push(np);
        out.n_lower.push(nm);
        if cp > opts.h {
            out.signals_upper.push(i);
        }
        if cm > opts.h {
            out.signals_lower.push(i);
        }
    }
    Ok(out)
}

/// V-mask CUSUM: cumulative standardised sums `Sᵢ = Σ zⱼ` with a V-shaped mask placed at the latest point.
///
/// With slope `k` per observation (the reference value) and lead distance `d = h/k`, the arms of the
/// mask at time `n` pass through `Sₙ ± k(d + (n − j))` over the earlier points `j`. The process is out of
/// control at time `n` when an earlier cumulative sum `Sⱼ` lies outside the arms (`j = 0` is the origin,
/// `S₀ = 0`). This is algebraically the tabular CUSUM with the same `k`, `h` and no head start, but it is
/// computed here directly from the cumulative sums.
#[derive(Debug, Clone, PartialEq)]
pub struct VMaskChart {
    /// Slope of the arms per observation (σ units) — the reference value K.
    pub k: f64,
    /// Decision interval H (σ units).
    pub h: f64,
    /// Lead distance `d = h/k` (observations).
    pub lead_distance: f64,
    /// Half-angle of the mask when one observation and one σ unit are drawn the same length: `atan(k)` in radians.
    pub half_angle: f64,
    /// Cumulative sums `S₁ … Sₙ`.
    pub sums: Vec<f64>,
    /// Times at which the mask shows an upward shift (an earlier sum below the lower arm).
    pub signals_upper: Vec<usize>,
    /// Times at which the mask shows a downward shift (an earlier sum above the upper arm).
    pub signals_lower: Vec<usize>,
}

impl VMaskChart {
    /// The arms of the mask positioned at point `n` (0-based): for every earlier cumulative sum (index `j`,
    /// with `None`-origin represented as `j = 0` and point `i` as `j = i + 1`) the pair `(upper arm, lower arm)`.
    pub fn arms_at(&self, n: usize) -> Vec<(f64, f64)> {
        let s_n = self.sums[n];
        (0..=n)
            .map(|j| {
                let dist = self.lead_distance + (n + 1 - j) as f64;
                (s_n + self.k * dist, s_n - self.k * dist)
            })
            .collect()
    }
}

/// V-mask CUSUM of `inp` with reference value `k` and decision interval `h` (σ units).
pub fn cusum_vmask(inp: &TimeWeightedInput, k: f64, h: f64) -> Result<VMaskChart> {
    inp.check()?;
    CusumOptions { k, h, head_start: 0.0 }.check()?;
    let n = inp.values.len();
    let mut s = Vec::with_capacity(n + 1);
    s.push(0.0);
    for i in 0..n {
        s.push(s[i] + (inp.values[i] - inp.target) / inp.se(i));
    }
    let d = h / k;
    let (mut up, mut down) = (Vec::new(), Vec::new());
    for t in 1..=n {
        // index j = 0..t−1 are the earlier sums (S₀ … S_{t−1}), at distance t − j observations.
        let below = (0..t).any(|j| s[j] < s[t] - k * (d + (t - j) as f64));
        let above = (0..t).any(|j| s[j] > s[t] + k * (d + (t - j) as f64));
        if below {
            up.push(t - 1);
        }
        if above {
            down.push(t - 1);
        }
    }
    Ok(VMaskChart {
        k,
        h,
        lead_distance: d,
        half_angle: k.atan(),
        sums: s[1..].to_vec(),
        signals_upper: up,
        signals_lower: down,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(values: &[f64], target: f64, sigma: f64) -> TimeWeightedInput {
        TimeWeightedInput {
            values: values.to_vec(),
            sizes: vec![1; values.len()],
            target,
            sigma,
            sigma_method: "known".into(),
        }
    }

    #[test]
    fn cusum_recursion_by_hand() {
        let c =
            cusum_tabular(&input(&[1.0, 1.0, -1.0, 3.0], 0.0, 1.0), &CusumOptions { k: 0.5, h: 2.0, head_start: 0.0 })
                .unwrap();
        assert_eq!(c.upper, vec![0.5, 1.0, 0.0, 2.5]);
        assert_eq!(c.lower, vec![0.0, 0.0, 0.5, 0.0]);
        assert_eq!(c.n_upper, vec![1, 2, 0, 1]);
        assert_eq!(c.signals_upper, vec![3]);
        assert!(c.signals_lower.is_empty());
    }

    #[test]
    fn cusum_head_start() {
        let c =
            cusum_tabular(&input(&[0.0, 0.0], 0.0, 1.0), &CusumOptions { k: 0.5, h: 4.0, head_start: 2.0 }).unwrap();
        assert_eq!(c.upper, vec![1.5, 1.0]);
        assert_eq!(c.lower, vec![1.5, 1.0]);
    }

    #[test]
    fn vmask_matches_tabular_signals() {
        let mut s = 12345u64;
        let data: Vec<f64> = (0..80)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5;
                u * 2.0 + if i >= 40 { 0.9 } else { 0.0 } - if i >= 65 { 2.2 } else { 0.0 }
            })
            .collect();
        let inp = input(&data, 0.0, 1.0);
        let tab = cusum_tabular(&inp, &CusumOptions { k: 0.5, h: 4.0, head_start: 0.0 }).unwrap();
        let vm = cusum_vmask(&inp, 0.5, 4.0).unwrap();
        assert_eq!(vm.signals_upper, tab.signals_upper);
        assert_eq!(vm.signals_lower, tab.signals_lower);
        assert!(!tab.signals_upper.is_empty());
        assert!((vm.half_angle - 0.5f64.atan()).abs() < 1e-15);
        assert_eq!(vm.lead_distance, 8.0);
        let arms = vm.arms_at(3);
        assert_eq!(arms.len(), 4);
    }

    #[test]
    fn ewma_recursion_and_limits() {
        let c = ewma(&input(&[2.0, 2.0, 2.0], 0.0, 1.0), 0.2, 3.0, true).unwrap();
        assert!((c.panel.values[0] - 0.4).abs() < 1e-15);
        assert!((c.panel.values[1] - 0.72).abs() < 1e-15);
        // first point: variance λ²σ²
        assert!((c.panel.ucl[0] - 3.0 * 0.2).abs() < 1e-15);
        // second: (0.8²·0.04 + 0.04) = 0.0656
        assert!((c.panel.ucl[1] - 3.0 * 0.0656f64.sqrt()).abs() < 1e-14);
        let a = ewma(&input(&[2.0, 2.0, 2.0], 0.0, 1.0), 0.2, 3.0, false).unwrap();
        assert!((a.panel.ucl[0] - 3.0 * (0.2f64 / 1.8).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn moving_average_variance() {
        let c = moving_average(&input(&[1.0, 2.0, 3.0, 4.0], 0.0, 2.0), 3, 3.0).unwrap();
        assert_eq!(c.panel.values, vec![1.0, 1.5, 2.0, 3.0]);
        assert!((c.panel.sigma[0] - 2.0).abs() < 1e-15);
        assert!((c.panel.sigma[3] - 2.0 / 3f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let i = input(&[1.0, 2.0], 0.0, 1.0);
        assert!(ewma(&i, 0.0, 3.0, true).is_err());
        assert!(ewma(&i, 1.5, 3.0, true).is_err());
        assert!(moving_average(&i, 0, 3.0).is_err());
        assert!(cusum_tabular(&i, &CusumOptions { k: 0.0, ..CusumOptions::default() }).is_err());
        assert!(cusum_tabular(&i, &CusumOptions { head_start: 5.0, ..CusumOptions::default() }).is_err());
        assert!(TimeWeightedInput::individuals(&[1.0, f64::NAN], None, None).is_err());
    }

    #[test]
    fn unequal_sizes_use_own_standard_error() {
        let inp = TimeWeightedInput {
            values: vec![1.0, 1.0],
            sizes: vec![1, 4],
            target: 0.0,
            sigma: 1.0,
            sigma_method: "known".into(),
        };
        let c = cusum_tabular(&inp, &CusumOptions::default()).unwrap();
        assert_eq!(c.se, vec![1.0, 0.5]);
        assert!(c.in_data_units().is_err());
        assert!((c.upper[1] - (0.5 + (2.0 - 0.5))).abs() < 1e-15);
    }
}
