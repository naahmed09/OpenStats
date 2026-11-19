//! Item analysis: Cronbach's alpha.

use super::common::{corr_from_cov, covariance, listwise, mean, names_or_default};
use crate::error::{invalid, Result, StatsError};
use openstats_math::linalg::Cholesky;
use openstats_math::Matrix;

/// Statistics of one item with the item left out of the total.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemRow {
    pub name: String,
    pub adjusted_total_mean: f64,
    pub adjusted_total_sd: f64,
    /// Correlation of the item with the total of the other items.
    pub item_total_correlation: f64,
    /// Squared multiple correlation of the item with the others.
    pub squared_multiple_correlation: f64,
    /// Cronbach's alpha of the remaining items.
    pub alpha_if_omitted: f64,
}

/// Result of [`cronbach_alpha`].
#[derive(Debug, Clone)]
pub struct ItemAnalysis {
    pub n: usize,
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    pub items: Vec<ItemRow>,
    pub alpha: f64,
    pub standardized_alpha: f64,
    pub correlations: Matrix,
    pub covariances: Matrix,
}

fn alpha_of(s: &Matrix) -> f64 {
    let k = s.rows() as f64;
    let tr = s.trace();
    let tot: f64 = s.as_slice().iter().sum();
    k / (k - 1.0) * (1.0 - tr / tot)
}

pub fn cronbach_alpha(cols: &[Vec<f64>], names: &[String]) -> Result<ItemAnalysis> {
    let k = cols.len();
    if k < 2 {
        return invalid("item analysis needs at least two items");
    }
    let c = listwise(cols)?;
    let n = c.rows.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { what: "item analysis", needed: 3, got: n });
    }
    let cov = covariance(&c.cols);
    let corr = corr_from_cov(&cov)?;
    let alpha = alpha_of(&cov);
    let rbar = (0..k).flat_map(|i| (0..i).map(move |j| (i, j))).map(|(i, j)| corr[(i, j)]).sum::<f64>()
        / (k * (k - 1) / 2) as f64;
    let standardized = k as f64 * rbar / (1.0 + (k as f64 - 1.0) * rbar);
    let inv = Cholesky::new(&corr).and_then(|ch| ch.inverse()).ok();
    let nm = names_or_default(names, k, "C");
    let mut items = Vec::with_capacity(k);
    for i in 0..k {
        let total: Vec<f64> = (0..n).map(|r| (0..k).filter(|&j| j != i).map(|j| c.cols[j][r]).sum()).collect();
        let m = mean(&total);
        let sd = (total.iter().map(|t| (t - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
        let mi = mean(&c.cols[i]);
        let sdi = cov[(i, i)].sqrt();
        let r =
            total.iter().zip(&c.cols[i]).map(|(t, x)| (t - m) * (x - mi)).sum::<f64>() / ((n as f64 - 1.0) * sd * sdi);
        let keep: Vec<usize> = (0..k).filter(|&j| j != i).collect();
        let sub = cov.select(&keep, &keep);
        let smc = inv.as_ref().map(|v| 1.0 - 1.0 / v[(i, i)]).unwrap_or(f64::NAN);
        items.push(ItemRow {
            name: nm[i].clone(),
            adjusted_total_mean: m,
            adjusted_total_sd: sd,
            item_total_correlation: r,
            squared_multiple_correlation: smc,
            alpha_if_omitted: if k > 2 { alpha_of(&sub) } else { f64::NAN },
        });
    }
    Ok(ItemAnalysis {
        n,
        n_dropped: cols[0].len() - n,
        rows_used: c.rows,
        items,
        alpha,
        standardized_alpha: standardized,
        correlations: corr,
        covariances: cov,
    })
}
