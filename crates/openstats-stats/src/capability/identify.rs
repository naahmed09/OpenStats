//! Individual-distribution identification: fit candidate families and rank
//! them by the Anderson–Darling statistic (smaller is better).

use crate::error::{Result, StatsError};
use crate::graphical::anderson_darling;
use crate::spc::capability::{fit_distribution, Family};
use crate::spc::transform::fit_johnson;
use openstats_math::dist::JohnsonFamily;

fn family_label(f: JohnsonFamily) -> &'static str {
    match f {
        JohnsonFamily::SU => "SU",
        JohnsonFamily::SB => "SB",
        JohnsonFamily::SL => "SL",
    }
}

/// One candidate distribution in an identification table.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionRank {
    /// Name ("Normal", "Weibull", "Johnson SU", …).
    pub name: String,
    /// The fitted family (`None` for the Johnson system).
    pub family: Option<Family>,
    /// Anderson–Darling statistic against the fitted distribution (for Normal
    /// and Johnson, the standard statistic of the (transformed) data).
    pub ad: f64,
    /// Anderson–Darling p-value (only for the normal and Johnson candidates,
    /// where the D'Agostino–Stephens approximation applies).
    pub p_value: Option<f64>,
    /// Maximised log-likelihood (`None` for the Johnson system).
    pub log_likelihood: Option<f64>,
    /// Number of estimated parameters (threshold fixed, not counted).
    pub parameters: usize,
}

/// Fit the normal, lognormal, exponential, Weibull, gamma, logistic and
/// loglogistic families (positive families with the given `threshold`, default 0)
/// and the Johnson system, ordered by Anderson–Darling statistic.
///
/// Families that cannot be fitted (for example because the data are not above
/// the threshold) are left out; an error is returned only when none fits.
pub fn identify_distributions(data: &[f64], threshold: Option<f64>) -> Result<Vec<DistributionRank>> {
    let mut out = Vec::new();
    let valid: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    for fam in [
        Family::Normal,
        Family::Lognormal,
        Family::Exponential,
        Family::Weibull,
        Family::Gamma,
        Family::Logistic,
        Family::Loglogistic,
    ] {
        let Ok(fit) = fit_distribution(data, fam, threshold) else { continue };
        let (ad, p) = if fam == Family::Normal {
            match anderson_darling(&valid) {
                Ok(a) => (a.a2, Some(a.p_value)),
                Err(_) => (fit.ad, None),
            }
        } else {
            (fit.ad, None)
        };
        out.push(DistributionRank {
            name: fam.name().into(),
            family: Some(fam),
            ad,
            p_value: p,
            log_likelihood: Some(fit.log_likelihood),
            parameters: if fam == Family::Exponential { 1 } else { 2 },
        });
    }
    if let Ok(j) = fit_johnson(data) {
        out.push(DistributionRank {
            name: format!("Johnson {}", family_label(j.dist.family)),
            family: None,
            ad: j.a2,
            p_value: Some(j.p_value),
            log_likelihood: None,
            parameters: 4,
        });
    }
    if out.is_empty() {
        return Err(StatsError::InsufficientData { what: "distribution identification", needed: 3, got: valid.len() });
    }
    out.sort_by(|a, b| a.ad.total_cmp(&b.ad));
    Ok(out)
}
