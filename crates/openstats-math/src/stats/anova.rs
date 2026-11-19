//! One-way analysis of variance.

use super::summary::{mean, sum, sum_squared_deviations};
use crate::dist::{ContinuousDistribution, FisherF};
use crate::error::{MathError, Result};

/// One-way ANOVA table.
#[derive(Debug, Clone, Copy)]
pub struct OneWayAnova {
    pub df_between: f64,
    pub df_within: f64,
    pub ss_between: f64,
    pub ss_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
    pub f: f64,
    /// P(F > f) under the null hypothesis.
    pub p_value: f64,
    /// SS_between / SS_total.
    pub r_squared: f64,
    /// √MS_within.
    pub residual_sd: f64,
}

/// One-way ANOVA over two or more groups (each non-empty, at least one with
/// two or more observations). Sums of squares use the corrected two-pass
/// algorithm, so groups far from zero lose no accuracy.
pub fn anova_oneway(groups: &[Vec<f64>]) -> Result<OneWayAnova> {
    let k = groups.len();
    if k < 2 || groups.iter().any(|g| g.is_empty()) {
        return Err(MathError::InvalidArgument("anova_oneway: need at least two non-empty groups".into()));
    }
    let n: usize = groups.iter().map(Vec::len).sum();
    if n <= k {
        return Err(MathError::InvalidArgument("anova_oneway: need more observations than groups".into()));
    }
    let means: Vec<f64> = groups.iter().map(|g| mean(g)).collect::<Result<_>>()?;
    let grand = mean(&groups.iter().flatten().copied().collect::<Vec<_>>())?;
    let ss_between = sum(groups.iter().zip(&means).map(|(g, m)| g.len() as f64 * (m - grand) * (m - grand)));
    let ss_within = sum(groups.iter().map(|g| sum_squared_deviations(g)).collect::<Result<Vec<_>>>()?);
    let (dfb, dfw) = ((k - 1) as f64, (n - k) as f64);
    let (msb, msw) = (ss_between / dfb, ss_within / dfw);
    let f = msb / msw;
    let p_value = if f.is_nan() { f64::NAN } else { FisherF::new(dfb, dfw)?.sf(f) };
    Ok(OneWayAnova {
        df_between: dfb,
        df_within: dfw,
        ss_between,
        ss_within,
        ms_between: msb,
        ms_within: msw,
        f,
        p_value,
        r_squared: ss_between / (ss_between + ss_within),
        residual_sd: msw.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn textbook_table() {
        // Three groups: means 2, 4, 6 with unit-spread residuals.
        let g = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0], vec![5.0, 6.0, 7.0]];
        let a = anova_oneway(&g).unwrap();
        assert_eq!((a.df_between, a.df_within), (2.0, 6.0));
        assert!((a.ss_between - 24.0).abs() < 1e-13 && (a.ss_within - 6.0).abs() < 1e-13);
        assert!((a.f - 12.0).abs() < 1e-12);
        // P(F(2,6) > 12) = (1 + 2·12/6)^{-3} = 1/125.
        assert!((a.p_value - 0.008).abs() < 1e-12);
        assert!((a.r_squared - 0.8).abs() < 1e-14 && (a.residual_sd - 1.0).abs() < 1e-14);
        assert!(anova_oneway(&g[..1]).is_err());
        assert!(anova_oneway(&[vec![1.0], vec![2.0]]).is_err());
    }
}
