//! Types shared by the hypothesis-test modules.

use crate::error::{invalid, Result};
use crate::graphical::Interval;
use openstats_math::dist::{ContinuousDistribution, Normal, StudentT};

/// Direction of the alternative hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    /// H₁: parameter ≠ null value.
    TwoSided,
    /// H₁: parameter < null value (one-sided *upper* confidence bound).
    Less,
    /// H₁: parameter > null value (one-sided *lower* confidence bound).
    Greater,
}

/// Variance assumption of a two-sample comparison of means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceAssumption {
    /// Equal variances: pooled SD, n₁ + n₂ − 2 df.
    Pooled,
    /// Unequal variances: Welch's test with Satterthwaite df.
    Welch,
}

pub(crate) fn check_level(level: f64) -> Result<()> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        invalid(format!("confidence level {level} must be in (0, 1)"))
    }
}

/// p-value of a t statistic.
pub(crate) fn t_p_value(t: f64, df: f64, alt: Alternative) -> Result<f64> {
    let d = StudentT::new(df)?;
    Ok(match alt {
        Alternative::TwoSided => (2.0 * d.sf(t.abs())).min(1.0),
        Alternative::Less => d.cdf(t),
        Alternative::Greater => d.sf(t),
    })
}

/// p-value of a standard-normal statistic.
pub(crate) fn z_p_value(z: f64, alt: Alternative) -> f64 {
    let n = Normal::standard();
    match alt {
        Alternative::TwoSided => (2.0 * n.sf(z.abs())).min(1.0),
        Alternative::Less => n.cdf(z),
        Alternative::Greater => n.sf(z),
    }
}

/// Confidence interval `est ± q·se` with the t (`df` finite) or normal
/// (`df = ∞`) quantile; one-sided for `Less` / `Greater`.
pub(crate) fn wald_interval(est: f64, se: f64, df: f64, alt: Alternative, level: f64) -> Result<Interval> {
    let tail = match alt {
        Alternative::TwoSided => (1.0 - level) / 2.0,
        _ => 1.0 - level,
    };
    let q = if df.is_finite() { StudentT::new(df)?.isf(tail) } else { Normal::standard().isf(tail) };
    Ok(match alt {
        Alternative::TwoSided => Interval { lower: est - q * se, upper: est + q * se, level },
        Alternative::Less => Interval { lower: f64::NEG_INFINITY, upper: est + q * se, level },
        Alternative::Greater => Interval { lower: est - q * se, upper: f64::INFINITY, level },
    })
}

pub(crate) fn check_alpha(alpha: f64) -> Result<()> {
    if alpha > 0.0 && alpha < 0.5 {
        Ok(())
    } else {
        invalid(format!("significance level {alpha} must be in (0, 0.5)"))
    }
}
