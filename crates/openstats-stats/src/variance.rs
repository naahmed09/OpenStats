//! Tests for equality of variances: the F-test for two variances, Levene's
//! test, the Brown–Forsythe test and Bonett's test.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::hypothesis::{check_level, Alternative};
use openstats_math::dist::{ChiSquared, ContinuousDistribution, FisherF, Normal};
use openstats_math::stats::{anova_oneway, summary};

/// Result of the F-test for two variances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FTest {
    pub n1: usize,
    pub n2: usize,
    pub var1: f64,
    pub var2: f64,
    /// s₁²/s₂².
    pub ratio: f64,
    /// Null value of σ₁²/σ₂².
    pub ratio0: f64,
    /// F = (s₁²/s₂²)/ratio₀ with (n₁−1, n₂−1) df.
    pub f: f64,
    pub df1: f64,
    pub df2: f64,
    /// Two-sided p = 2·min(P(F ≤ f), P(F ≥ f)) capped at 1.
    pub p_value: f64,
    /// Interval for σ₁²/σ₂² (one-sided for `Less`/`Greater`).
    pub ci_ratio: Interval,
    /// Interval for σ₁/σ₂ (square roots of the above).
    pub ci_sd_ratio: Interval,
}

/// F-test of H₀: σ₁²/σ₂² = `ratio0` (assumes normal data; sensitive to
/// non-normality — see [`levene_test`] and [`bonett_test`]).
pub fn f_test(x: &[f64], y: &[f64], ratio0: f64, alt: Alternative, level: f64) -> Result<FTest> {
    check_level(level)?;
    if !(ratio0 > 0.0 && ratio0.is_finite()) {
        return invalid(format!("the hypothesised variance ratio {ratio0} must be positive"));
    }
    let (vx, vy) = (crate::clean(x)?, crate::clean(y)?);
    for (v, what) in [(&vx, "F-test (first sample)"), (&vy, "F-test (second sample)")] {
        if v.len() < 2 {
            return Err(StatsError::InsufficientData { what, needed: 2, got: v.len() });
        }
    }
    let (var1, var2) = (summary::variance(&vx)?, summary::variance(&vy)?);
    if var2 == 0.0 || var1 == 0.0 {
        return invalid("a sample has zero variance, so the F statistic is undefined");
    }
    let (df1, df2) = ((vx.len() - 1) as f64, (vy.len() - 1) as f64);
    let ratio = var1 / var2;
    let f = ratio / ratio0;
    let d = FisherF::new(df1, df2)?;
    let p_value = match alt {
        Alternative::TwoSided => (2.0 * d.cdf(f).min(d.sf(f))).min(1.0),
        Alternative::Less => d.cdf(f),
        Alternative::Greater => d.sf(f),
    };
    let tail = if alt == Alternative::TwoSided { (1.0 - level) / 2.0 } else { 1.0 - level };
    let lo = ratio / d.isf(tail);
    let hi = ratio / d.quantile(tail);
    let ci_ratio = match alt {
        Alternative::TwoSided => Interval { lower: lo, upper: hi, level },
        Alternative::Less => Interval { lower: 0.0, upper: hi, level },
        Alternative::Greater => Interval { lower: lo, upper: f64::INFINITY, level },
    };
    let ci_sd_ratio = Interval { lower: ci_ratio.lower.sqrt(), upper: ci_ratio.upper.sqrt(), level };
    Ok(FTest { n1: vx.len(), n2: vy.len(), var1, var2, ratio, ratio0, f, df1, df2, p_value, ci_ratio, ci_sd_ratio })
}

/// Centre used by Levene-type tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Center {
    /// Group mean: Levene's original test.
    Mean,
    /// Group median: the Brown–Forsythe test.
    Median,
    /// Mean after cutting `⌊f·n⌋` values from each end of each group,
    /// 0 ≤ f < 0.5.
    TrimmedMean(f64),
}

/// Result of a Levene-type test.
#[derive(Debug, Clone, PartialEq)]
pub struct LeveneTest {
    /// ANOVA F statistic of the absolute deviations from the centres.
    pub statistic: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    /// Centre of each group.
    pub centers: Vec<f64>,
    /// Sample size of each group.
    pub n: Vec<usize>,
}

fn groups_clean(groups: &[Vec<f64>], min_n: usize, what: &'static str) -> Result<Vec<Vec<f64>>> {
    if groups.len() < 2 {
        return invalid(format!("{what} needs at least two groups"));
    }
    let g: Vec<Vec<f64>> = groups.iter().map(|v| crate::clean(v)).collect::<Result<_>>()?;
    for v in &g {
        if v.len() < min_n {
            return Err(StatsError::InsufficientData { what, needed: min_n, got: v.len() });
        }
    }
    Ok(g)
}

fn center_of(sorted: &[f64], c: Center) -> Result<f64> {
    Ok(match c {
        Center::Mean => summary::mean(sorted)?,
        Center::Median => {
            let n = sorted.len();
            if n % 2 == 1 {
                sorted[n / 2]
            } else {
                0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
            }
        }
        Center::TrimmedMean(f) => {
            if !(0.0..0.5).contains(&f) {
                return invalid(format!("trim fraction {f} must be in [0, 0.5)"));
            }
            let k = (f * sorted.len() as f64).floor() as usize;
            summary::mean(&sorted[k..sorted.len() - k])?
        }
    })
}

/// Levene-type test: a one-way ANOVA of |xᵢⱼ − centreᵢ|. Each group needs at
/// least 2 observations; there must be at least two groups.
pub fn levene_test(groups: &[Vec<f64>], center: Center) -> Result<LeveneTest> {
    let g = groups_clean(groups, 2, "Levene's test")?;
    let mut centers = Vec::new();
    let mut z = Vec::new();
    for v in &g {
        let mut s = v.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let c = center_of(&s, center)?;
        centers.push(c);
        z.push(v.iter().map(|x| (x - c).abs()).collect::<Vec<f64>>());
    }
    let a = anova_oneway(&z)?;
    Ok(LeveneTest {
        statistic: a.f,
        df1: a.df_between,
        df2: a.df_within,
        p_value: a.p_value,
        centers,
        n: g.iter().map(Vec::len).collect(),
    })
}

/// Brown–Forsythe test: Levene's test centred at the group medians.
pub fn brown_forsythe_test(groups: &[Vec<f64>]) -> Result<LeveneTest> {
    levene_test(groups, Center::Median)
}

/// Result of Bonett's test.
#[derive(Debug, Clone, PartialEq)]
pub struct BonettTest {
    /// χ² with `df` degrees of freedom (z² for two groups).
    pub statistic: f64,
    pub df: f64,
    pub p_value: f64,
    /// Per group: n, standard deviation, estimated kurtosis γ and the small
    /// sample correction c.
    pub groups: Vec<BonettGroup>,
}

/// Per-group quantities of Bonett's test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BonettGroup {
    pub n: usize,
    pub std_dev: f64,
    pub kurtosis: f64,
    pub correction: f64,
}

/// Bonett's test for equal variances of k ≥ 2 groups (Bonett 2006; the
/// k-group form pools the log-variances with inverse-variance weights).
///
/// For group i with n ≥ 5 observations: γᵢ = nΣ(x − mᵢ)⁴/[Σ(x − x̄)²]² where
/// mᵢ is the mean after trimming ⌊n/(2√(n−4))⌋ values from each end;
/// cᵢ = n/(n − z_{α/2}) with α = 1 − `level`; zᵢ = ln(cᵢsᵢ²) has variance
/// vᵢ = (γᵢ − (n−3)/n)/(n−1). The statistic is Σwᵢ(zᵢ − z̄)² with
/// wᵢ = 1/vᵢ and z̄ the weighted mean, χ² with k − 1 df; for k = 2 this equals
/// ((z₁ − z₂)/√(v₁ + v₂))².
///
/// `level` (the confidence level of the associated intervals, normally 0.95)
/// enters only through the correction c.
pub fn bonett_test(groups: &[Vec<f64>], level: f64) -> Result<BonettTest> {
    check_level(level)?;
    let g = groups_clean(groups, 5, "Bonett's test")?;
    let zq = Normal::standard().isf((1.0 - level) / 2.0);
    let mut info = Vec::new();
    let (mut zs, mut ws) = (Vec::new(), Vec::new());
    for v in &g {
        let n = v.len();
        let nf = n as f64;
        let mut s = v.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let k = (nf / (2.0 * (nf - 4.0).sqrt())).floor() as usize;
        let m = summary::mean(&s[k.min((n - 1) / 2)..n - k.min((n - 1) / 2)])?;
        let mean = summary::mean(&s)?;
        let ssd = summary::sum_squared_deviations(&s)?;
        if ssd == 0.0 {
            return invalid("a group has zero variance, so Bonett's statistic is undefined");
        }
        let gamma = nf * summary::sum(s.iter().map(|x| (x - m).powi(4))) / (ssd * ssd);
        let var = ssd / (nf - 1.0);
        let c = nf / (nf - zq);
        let vi = (gamma - (nf - 3.0) / nf) / (nf - 1.0);
        if vi <= 0.0 || vi.is_nan() || !mean.is_finite() {
            return invalid("the estimated kurtosis is too small for Bonett's test on this data");
        }
        zs.push((c * var).ln());
        ws.push(1.0 / vi);
        info.push(BonettGroup { n, std_dev: var.sqrt(), kurtosis: gamma, correction: c });
    }
    let wsum: f64 = ws.iter().sum();
    let zbar: f64 = zs.iter().zip(&ws).map(|(z, w)| z * w).sum::<f64>() / wsum;
    let stat: f64 = zs.iter().zip(&ws).map(|(z, w)| w * (z - zbar) * (z - zbar)).sum();
    let df = (g.len() - 1) as f64;
    Ok(BonettTest { statistic: stat, df, p_value: ChiSquared::new(df)?.sf(stat), groups: info })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> Vec<Vec<f64>> {
        vec![
            vec![4.1, 5.2, 6.3, 5.5, 4.8, 5.9, 5.1],
            vec![3.0, 7.5, 9.1, 2.2, 6.8, 8.4, 4.4],
            vec![5.0, 5.1, 5.3, 4.9, 5.2, 5.0, 4.8],
        ]
    }

    #[test]
    fn f_test_two_sided_and_one_sided() {
        let x = [4.1, 5.2, 6.3, 5.5, 4.8, 5.9, 5.1];
        let y = [3.0, 7.5, 9.1, 2.2, 6.8, 8.4, 4.4];
        let r = f_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap();
        assert!(r.f < 1.0 && r.p_value < 0.05);
        assert!((r.ratio - r.var1 / r.var2).abs() < 1e-15);
        assert!(r.ci_ratio.lower < r.ratio && r.ratio < r.ci_ratio.upper);
        assert!((r.ci_sd_ratio.lower.powi(2) - r.ci_ratio.lower).abs() < 1e-12);
        let l = f_test(&x, &y, 1.0, Alternative::Less, 0.95).unwrap();
        let gt = f_test(&x, &y, 1.0, Alternative::Greater, 0.95).unwrap();
        assert!((l.p_value + gt.p_value - 1.0).abs() < 1e-12);
        let h = f_test(&x, &y, r.ratio, Alternative::TwoSided, 0.95).unwrap();
        assert!((h.f - 1.0).abs() < 1e-12);
        assert!(f_test(&x, &[1.0], 1.0, Alternative::TwoSided, 0.95).is_err());
        assert!(f_test(&x, &[2.0, 2.0], 1.0, Alternative::TwoSided, 0.95).is_err());
        assert!(f_test(&x, &y, 0.0, Alternative::TwoSided, 0.95).is_err());
    }

    #[test]
    fn levene_variants_differ_and_validate() {
        let m = levene_test(&g(), Center::Mean).unwrap();
        let b = brown_forsythe_test(&g()).unwrap();
        let t = levene_test(&g(), Center::TrimmedMean(0.1)).unwrap();
        assert_eq!((m.df1, m.df2), (2.0, 18.0));
        assert!(m.statistic != b.statistic);
        assert!(m.p_value < 0.05 && b.p_value < 0.1);
        assert!(t.statistic.is_finite());
        assert!(levene_test(&g()[..1], Center::Mean).is_err());
        assert!(levene_test(&[vec![1.0], vec![1.0, 2.0]], Center::Mean).is_err());
        assert!(levene_test(&g(), Center::TrimmedMean(0.5)).is_err());
    }

    #[test]
    fn bonett_two_group_equals_z_squared() {
        let gs = g();
        let r = bonett_test(&gs[..2], 0.95).unwrap();
        assert_eq!(r.df, 1.0);
        assert!(r.p_value > 0.0 && r.p_value < 1.0);
        let r3 = bonett_test(&gs, 0.95).unwrap();
        assert_eq!(r3.df, 2.0);
        assert!(bonett_test(&[vec![1.0, 2.0, 3.0, 4.0], vec![1.0; 5]], 0.95).is_err());
        assert!(bonett_test(&[vec![1.0; 6], vec![1.0, 2.0, 3.0, 4.0, 9.0]], 0.95).is_err());
    }
}
