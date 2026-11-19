//! Hotelling's T² control chart for multivariate processes (individual
//! observations or subgroups of equal size).

use super::rules::TestViolation;
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{Beta, ChiSquared, ContinuousDistribution, FisherF};
use openstats_math::linalg::{Cholesky, Matrix};

/// How the covariance matrix of individual observations is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceEstimate {
    /// The sample covariance matrix.
    Sample,
    /// Half the covariance of successive differences, `S = VᵀV / (2(m − 1))` (robust to a drifting mean).
    SuccessiveDifferences,
}

/// What the chart is compared with.
#[derive(Debug, Clone, PartialEq)]
pub enum HotellingReference {
    /// Phase I: mean and covariance are estimated from the charted data (limits for retrospective analysis).
    Estimate,
    /// Phase II with parameters estimated from `m` earlier samples (individuals) or `m` earlier subgroups.
    Historical {
        /// Mean vector from the historical data.
        mean: Vec<f64>,
        /// Covariance matrix from the historical data.
        covariance: Matrix,
        /// Number of historical observations (or subgroups).
        m: usize,
    },
    /// Phase II with known mean and covariance (χ² limit).
    Known {
        /// Process mean.
        mean: Vec<f64>,
        /// Process covariance.
        covariance: Matrix,
    },
}

/// Options for the T² chart.
#[derive(Debug, Clone, PartialEq)]
pub struct HotellingOptions {
    /// Probability of a point above the limit when in control (0.00135, the one-sided 3σ tail).
    pub alpha: f64,
    /// What the data are compared with.
    pub reference: HotellingReference,
    /// Covariance estimate for individuals under [`HotellingReference::Estimate`].
    pub covariance: CovarianceEstimate,
}

impl Default for HotellingOptions {
    fn default() -> Self {
        HotellingOptions {
            alpha: 0.00135,
            reference: HotellingReference::Estimate,
            covariance: CovarianceEstimate::Sample,
        }
    }
}

/// A Hotelling T² chart.
#[derive(Debug, Clone, PartialEq)]
pub struct HotellingChart {
    /// T² of every point.
    pub t2: Vec<f64>,
    /// Upper control limit (the lower limit is 0).
    pub ucl: f64,
    /// Points above the limit.
    pub violations: Vec<TestViolation>,
    /// Number of variables.
    pub p: usize,
    /// Subgroup size (1 for individuals).
    pub n: usize,
    /// Number of points.
    pub m: usize,
    /// Mean vector used.
    pub mean: Vec<f64>,
    /// Covariance matrix used.
    pub covariance: Matrix,
    /// Description of the limit.
    pub limit_description: String,
    /// α used.
    pub alpha: f64,
}

impl HotellingChart {
    /// Flagged points (0-based).
    pub fn flagged(&self) -> Vec<usize> {
        self.violations.iter().flat_map(|v| v.points.iter().copied()).collect()
    }
}

fn check_alpha(a: f64) -> Result<()> {
    if !(a > 0.0 && a < 1.0) {
        return invalid(format!("alpha must lie in (0, 1) (got {a})"));
    }
    Ok(())
}

fn col_means(rows: &[Vec<f64>], p: usize) -> Vec<f64> {
    (0..p).map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64).collect()
}

fn cov_matrix(rows: &[Vec<f64>], mean: &[f64]) -> Matrix {
    let p = mean.len();
    let mut s = Matrix::zeros(p, p);
    for r in rows {
        for a in 0..p {
            for b in 0..p {
                s[(a, b)] += (r[a] - mean[a]) * (r[b] - mean[b]);
            }
        }
    }
    s.scale(1.0 / (rows.len() as f64 - 1.0))
}

fn check_params(mean: &[f64], cov: &Matrix, p: usize) -> Result<()> {
    if mean.len() != p || cov.rows() != p || cov.cols() != p {
        return invalid(format!("the reference mean and covariance must have {p} variables"));
    }
    if mean.iter().any(|v| !v.is_finite()) || cov.as_slice().iter().any(|v| !v.is_finite()) {
        return invalid("the reference mean and covariance must be finite");
    }
    Ok(())
}

fn quad_forms(points: &[Vec<f64>], mean: &[f64], cov: &Matrix, scale: f64) -> Result<Vec<f64>> {
    let chol = Cholesky::new(cov).map_err(|_| {
        StatsError::InvalidArgument("the covariance matrix is singular or not positive definite".into())
    })?;
    points
        .iter()
        .map(|x| {
            let d: Vec<f64> = x.iter().zip(mean).map(|(a, b)| a - b).collect();
            let s = chol.solve(&d)?;
            Ok(scale * d.iter().zip(&s).map(|(a, b)| a * b).sum::<f64>())
        })
        .collect()
}

fn finish(
    t2: Vec<f64>,
    ucl: f64,
    p: usize,
    n: usize,
    mean: Vec<f64>,
    covariance: Matrix,
    desc: String,
    alpha: f64,
) -> HotellingChart {
    let pts: Vec<usize> = (0..t2.len()).filter(|&i| t2[i] > ucl).collect();
    HotellingChart {
        m: t2.len(),
        t2,
        ucl,
        violations: if pts.is_empty() { vec![] } else { vec![TestViolation { test: 1, points: pts }] },
        p,
        n,
        mean,
        covariance,
        limit_description: desc,
        alpha,
    }
}

fn check_rows(rows: &[Vec<f64>], p: usize) -> Result<()> {
    if rows.iter().any(|r| r.len() != p) {
        return invalid("every observation must have the same number of variables");
    }
    if rows.iter().flatten().any(|v| !v.is_finite()) {
        return invalid("T² charts need finite values (no missing values)");
    }
    Ok(())
}

/// T² chart of individual multivariate observations (`data[i]` holds the p measurements of observation i).
fn individuals_core(data: &[Vec<f64>], opts: &HotellingOptions, min_p: usize) -> Result<HotellingChart> {
    check_alpha(opts.alpha)?;
    let m = data.len();
    let p = data.first().map_or(0, Vec::len);
    if p < min_p {
        return invalid("a T² chart needs at least two variables");
    }
    check_rows(data, p)?;
    let (pf, mf) = (p as f64, m as f64);
    let q = 1.0 - opts.alpha;
    match &opts.reference {
        HotellingReference::Estimate => {
            if m <= p + 1 {
                return Err(StatsError::InsufficientData { what: "Phase I T² chart", needed: p + 2, got: m });
            }
            let mean = col_means(data, p);
            let (cov, ucl, desc) = match opts.covariance {
                CovarianceEstimate::Sample => {
                    let ucl = (mf - 1.0) * (mf - 1.0) / mf * Beta::new(pf / 2.0, (mf - pf - 1.0) / 2.0)?.quantile(q);
                    (cov_matrix(data, &mean), ucl, "Phase I limit (m−1)²/m · Beta(p/2, (m−p−1)/2)".to_string())
                }
                CovarianceEstimate::SuccessiveDifferences => {
                    let mut s = Matrix::zeros(p, p);
                    for w in data.windows(2) {
                        for a in 0..p {
                            for b in 0..p {
                                s[(a, b)] += (w[1][a] - w[0][a]) * (w[1][b] - w[0][b]);
                            }
                        }
                    }
                    let s = s.scale(1.0 / (2.0 * (mf - 1.0)));
                    // Sullivan & Woodall (1996) beta approximation with ν = 2(m−1)²/(3m−4).
                    let nu = 2.0 * (mf - 1.0) * (mf - 1.0) / (3.0 * mf - 4.0);
                    if nu <= pf + 1.0 {
                        return Err(StatsError::InsufficientData {
                            what: "successive-difference T² chart",
                            needed: p + 3,
                            got: m,
                        });
                    }
                    let ucl = (mf - 1.0) * (mf - 1.0) / mf * Beta::new(pf / 2.0, (nu - pf - 1.0) / 2.0)?.quantile(q);
                    (
                        s,
                        ucl,
                        "Phase I limit, successive-difference covariance (Sullivan–Woodall beta approximation)"
                            .to_string(),
                    )
                }
            };
            let t2 = quad_forms(data, &mean, &cov, 1.0)?;
            Ok(finish(t2, ucl, p, 1, mean, cov, desc, opts.alpha))
        }
        HotellingReference::Historical { mean, covariance, m: hist } => {
            check_params(mean, covariance, p)?;
            if *hist <= p {
                return invalid("the historical sample size must exceed the number of variables");
            }
            let h = *hist as f64;
            let ucl = pf * (h + 1.0) * (h - 1.0) / (h * (h - pf)) * FisherF::new(pf, h - pf)?.quantile(q);
            let t2 = quad_forms(data, mean, covariance, 1.0)?;
            Ok(finish(
                t2,
                ucl,
                p,
                1,
                mean.clone(),
                covariance.clone(),
                "Phase II limit p(m+1)(m−1)/(m(m−p)) · F(p, m−p)".into(),
                opts.alpha,
            ))
        }
        HotellingReference::Known { mean, covariance } => {
            check_params(mean, covariance, p)?;
            let ucl = ChiSquared::new(pf)?.quantile(q);
            let t2 = quad_forms(data, mean, covariance, 1.0)?;
            Ok(finish(
                t2,
                ucl,
                p,
                1,
                mean.clone(),
                covariance.clone(),
                "known parameters: χ²(p) limit".into(),
                opts.alpha,
            ))
        }
    }
}

/// T² chart of subgroups: `groups[i]` holds the observations (rows of p values) of subgroup i; all subgroups need the same size n ≥ 2.
fn subgroups_core(groups: &[Vec<Vec<f64>>], opts: &HotellingOptions, min_p: usize) -> Result<HotellingChart> {
    check_alpha(opts.alpha)?;
    let m = groups.len();
    let n = groups.first().map_or(0, Vec::len);
    let p = groups.first().and_then(|g| g.first()).map_or(0, Vec::len);
    if p < min_p {
        return invalid("a T² chart needs at least two variables");
    }
    if n < 2 || groups.iter().any(|g| g.len() != n) {
        return invalid("subgroups must all have the same size, at least 2");
    }
    for g in groups {
        check_rows(g, p)?;
    }
    let (pf, mf, nf) = (p as f64, m as f64, n as f64);
    let q = 1.0 - opts.alpha;
    let means: Vec<Vec<f64>> = groups.iter().map(|g| col_means(g, p)).collect();
    let nu = mf * nf - mf - pf + 1.0;
    match &opts.reference {
        HotellingReference::Estimate => {
            if m < 2 || nu <= 0.0 {
                return Err(StatsError::InsufficientData {
                    what: "Phase I T² chart", needed: p + 1, got: m * (n - 1)
                });
            }
            let grand = col_means(&means, p);
            let mut pooled = Matrix::zeros(p, p);
            for (g, gm) in groups.iter().zip(&means) {
                pooled = pooled.try_add(&cov_matrix(g, gm))?;
            }
            let pooled = pooled.scale(1.0 / mf);
            let ucl = pf * (mf - 1.0) * (nf - 1.0) / nu * FisherF::new(pf, nu)?.quantile(q);
            let t2 = quad_forms(&means, &grand, &pooled, nf)?;
            Ok(finish(
                t2,
                ucl,
                p,
                n,
                grand,
                pooled,
                "Phase I limit p(m−1)(n−1)/(mn−m−p+1) · F(p, mn−m−p+1)".into(),
                opts.alpha,
            ))
        }
        HotellingReference::Historical { mean, covariance, m: hist } => {
            check_params(mean, covariance, p)?;
            let hf = *hist as f64;
            let nu_h = hf * nf - hf - pf + 1.0;
            if nu_h <= 0.0 {
                return invalid("the historical data are too small for the number of variables");
            }
            let ucl = pf * (hf + 1.0) * (nf - 1.0) / nu_h * FisherF::new(pf, nu_h)?.quantile(q);
            let t2 = quad_forms(&means, mean, covariance, nf)?;
            Ok(finish(
                t2,
                ucl,
                p,
                n,
                mean.clone(),
                covariance.clone(),
                "Phase II limit p(m+1)(n−1)/(mn−m−p+1) · F(p, mn−m−p+1)".into(),
                opts.alpha,
            ))
        }
        HotellingReference::Known { mean, covariance } => {
            check_params(mean, covariance, p)?;
            let ucl = ChiSquared::new(pf)?.quantile(q);
            let t2 = quad_forms(&means, mean, covariance, nf)?;
            Ok(finish(
                t2,
                ucl,
                p,
                n,
                mean.clone(),
                covariance.clone(),
                "known parameters: χ²(p) limit".into(),
                opts.alpha,
            ))
        }
    }
}

/// T² chart of individual multivariate observations (`data[i]` holds the p measurements of observation i).
pub fn hotelling_individuals(data: &[Vec<f64>], opts: &HotellingOptions) -> Result<HotellingChart> {
    individuals_core(data, opts, 2)
}

/// T² chart of subgroups: `groups[i]` holds the observations (rows of p values) of subgroup i; all
/// subgroups need the same size n ≥ 2.
pub fn hotelling_subgroups(groups: &[Vec<Vec<f64>>], opts: &HotellingOptions) -> Result<HotellingChart> {
    subgroups_core(groups, opts, 2)
}

impl HotellingChart {
    /// Additive contributions of each variable to each point's T²: `n · dⱼ · (S⁻¹d)ⱼ` with `d = x − mean`.
    /// `points` are the observations (individuals) or the subgroup means. Row i sums to `t2[i]`;
    /// a contribution can be negative when variables are correlated.
    pub fn contributions(&self, points: &[Vec<f64>]) -> Result<Matrix> {
        if points.len() != self.t2.len() || points.iter().any(|x| x.len() != self.p) {
            return invalid("points must match the chart's observations and variables");
        }
        let chol = Cholesky::new(&self.covariance).map_err(|_| {
            StatsError::InvalidArgument("the covariance matrix is singular or not positive definite".into())
        })?;
        let mut out = Matrix::zeros(points.len(), self.p);
        for (i, x) in points.iter().enumerate() {
            let d: Vec<f64> = x.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
            let sd = chol.solve(&d)?;
            for j in 0..self.p {
                out[(i, j)] = self.n as f64 * d[j] * sd[j];
            }
        }
        Ok(out)
    }
}

/// T² chart with its decomposition into variable contributions.
#[derive(Debug, Clone, PartialEq)]
pub struct HotellingDecomposition {
    /// The T² chart.
    pub chart: HotellingChart,
    /// Additive contributions (rows = points, columns = variables); each row sums to T².
    pub contributions: Matrix,
    /// Mason–Young–Tracy unconditional terms: the one-variable T² chart of each variable
    /// (with its own limit at the same α), to see which variable is out by itself.
    pub unconditional: Vec<HotellingChart>,
}

fn reduce_reference(r: &HotellingReference, j: usize) -> Result<HotellingReference> {
    Ok(match r {
        HotellingReference::Estimate => HotellingReference::Estimate,
        HotellingReference::Historical { mean, covariance, m } => HotellingReference::Historical {
            mean: vec![mean[j]],
            covariance: Matrix::from_vec(1, 1, vec![covariance[(j, j)]])?,
            m: *m,
        },
        HotellingReference::Known { mean, covariance } => HotellingReference::Known {
            mean: vec![mean[j]],
            covariance: Matrix::from_vec(1, 1, vec![covariance[(j, j)]])?,
        },
    })
}

/// T² chart of individuals with contributions and per-variable unconditional charts.
pub fn hotelling_decomposition_individuals(
    data: &[Vec<f64>],
    opts: &HotellingOptions,
) -> Result<HotellingDecomposition> {
    let chart = hotelling_individuals(data, opts)?;
    let contributions = chart.contributions(data)?;
    let mut unconditional = Vec::new();
    for j in 0..chart.p {
        let col: Vec<Vec<f64>> = data.iter().map(|r| vec![r[j]]).collect();
        let o = HotellingOptions { reference: reduce_reference(&opts.reference, j)?, ..opts.clone() };
        unconditional.push(individuals_core(&col, &o, 1)?);
    }
    Ok(HotellingDecomposition { chart, contributions, unconditional })
}

/// T² chart of subgroups with contributions (of the subgroup means) and per-variable unconditional charts.
pub fn hotelling_decomposition_subgroups(
    groups: &[Vec<Vec<f64>>],
    opts: &HotellingOptions,
) -> Result<HotellingDecomposition> {
    let chart = hotelling_subgroups(groups, opts)?;
    let means: Vec<Vec<f64>> = groups.iter().map(|g| col_means(g, chart.p)).collect();
    let contributions = chart.contributions(&means)?;
    let mut unconditional = Vec::new();
    for j in 0..chart.p {
        let col: Vec<Vec<Vec<f64>>> = groups.iter().map(|g| g.iter().map(|r| vec![r[j]]).collect()).collect();
        let o = HotellingOptions { reference: reduce_reference(&opts.reference, j)?, ..opts.clone() };
        unconditional.push(subgroups_core(&col, &o, 1)?);
    }
    Ok(HotellingDecomposition { chart, contributions, unconditional })
}

// ------------------------------------------------------ generalized variance |S|

/// Generalized-variance chart: the determinant `|Sᵢ|` of each subgroup's covariance matrix.
///
/// With subgroup size n and p variables, `E|S| = b₁|Σ|` and `Var|S| = b₂|Σ|²` where
/// `b₁ = (n−1)⁻ᵖ ∏ᵢ₌₁ᵖ (n−i)` and `b₂ = (n−1)⁻²ᵖ ∏ᵢ₌₁ᵖ (n−i) [∏ⱼ₌₁ᵖ (n−j+2) − ∏ⱼ₌₁ᵖ (n−j)]`
/// (Montgomery, §11.3). Limits are `b₁|Σ| ± k·√b₂|Σ|`, the lower one truncated at 0. `|Σ|` is the
/// known value or `|S̄|/b₁` from the pooled covariance matrix `S̄`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedVarianceChart {
    /// `|Sᵢ|` of every subgroup.
    pub det: Vec<f64>,
    /// Centre line `b₁|Σ|`.
    pub center: f64,
    /// Upper control limit.
    pub ucl: f64,
    /// Lower control limit.
    pub lcl: f64,
    /// `|Σ|` used.
    pub sigma_det: f64,
    /// `b₁`.
    pub b1: f64,
    /// `b₂`.
    pub b2: f64,
    /// Variables.
    pub p: usize,
    /// Subgroup size.
    pub n: usize,
    /// Points outside the limits (test 1).
    pub violations: Vec<TestViolation>,
}

impl GeneralizedVarianceChart {
    /// Flagged subgroups (0-based).
    pub fn flagged(&self) -> Vec<usize> {
        self.violations.iter().flat_map(|v| v.points.iter().copied()).collect()
    }
}

/// The constants `(b₁, b₂)` of the distribution of `|S|`.
pub fn generalized_variance_constants(n: usize, p: usize) -> Result<(f64, f64)> {
    if p == 0 || n <= p {
        return invalid("the subgroup size must exceed the number of variables");
    }
    let (nf, pf) = (n as f64, p as i32);
    let prod = |shift: f64| -> f64 { (1..=p).map(|i| nf - i as f64 + shift).product() };
    let b1 = prod(0.0) / (nf - 1.0).powi(pf);
    let b2 = prod(0.0) * (prod(2.0) - prod(0.0)) / (nf - 1.0).powi(2 * pf);
    Ok((b1, b2))
}

/// Generalized-variance chart of equal-size subgroups; `known_covariance` is Σ if known.
pub fn generalized_variance(
    groups: &[Vec<Vec<f64>>],
    known_covariance: Option<&Matrix>,
    k_sigma: f64,
) -> Result<GeneralizedVarianceChart> {
    if !(k_sigma > 0.0 && k_sigma.is_finite()) {
        return invalid(format!("k_sigma must be positive (got {k_sigma})"));
    }
    let m = groups.len();
    let n = groups.first().map_or(0, Vec::len);
    let p = groups.first().and_then(|g| g.first()).map_or(0, Vec::len);
    if m < 2 || p < 2 {
        return invalid("a generalized-variance chart needs at least two subgroups and two variables");
    }
    if groups.iter().any(|g| g.len() != n) {
        return invalid("subgroups must all have the same size");
    }
    for g in groups {
        check_rows(g, p)?;
    }
    let (b1, b2) = generalized_variance_constants(n, p)?;
    let mut sum = Matrix::zeros(p, p);
    let mut det = Vec::with_capacity(m);
    for g in groups {
        let s = cov_matrix(g, &col_means(g, p));
        det.push(Cholesky::new(&s).map_or(0.0, |c| c.det()));
        sum = sum.try_add(&s)?;
    }
    let sigma_det = match known_covariance {
        Some(c) => {
            if c.rows() != p || c.cols() != p {
                return invalid(format!("the known covariance must be {p} × {p}"));
            }
            Cholesky::new(c)
                .map_err(|_| StatsError::InvalidArgument("the known covariance is not positive definite".into()))?
                .det()
        }
        None => {
            let sbar = sum.scale(1.0 / m as f64);
            Cholesky::new(&sbar)
                .map_err(|_| StatsError::InvalidArgument("the pooled covariance is singular".into()))?
                .det()
                / b1
        }
    };
    let (center, w) = (b1 * sigma_det, k_sigma * b2.sqrt() * sigma_det);
    let (ucl, lcl) = (center + w, (center - w).max(0.0));
    let bad: Vec<usize> = (0..m).filter(|&i| det[i] > ucl || det[i] < lcl).collect();
    Ok(GeneralizedVarianceChart {
        det,
        center,
        ucl,
        lcl,
        sigma_det,
        b1,
        b2,
        p,
        n,
        violations: if bad.is_empty() { vec![] } else { vec![TestViolation { test: 1, points: bad }] },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<Vec<f64>> {
        let mut s = 99u64;
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5
        };
        (0..30)
            .map(|_| {
                let a = u();
                let b = 0.7 * a + 0.3 * u();
                vec![10.0 + a, 5.0 + b]
            })
            .collect()
    }

    #[test]
    fn t2_sums_to_known_total_for_sample_covariance() {
        // Σ T²ᵢ = (m − 1)·p when the sample covariance is used.
        let d = data();
        let c = hotelling_individuals(&d, &HotellingOptions::default()).unwrap();
        let total: f64 = c.t2.iter().sum();
        assert!((total - 29.0 * 2.0).abs() < 1e-9, "{total}");
        assert!(c.ucl > 0.0);
    }

    #[test]
    fn outlier_is_flagged_and_known_params_use_chi_square() {
        let mut d = data();
        d.push(vec![14.0, 1.0]);
        let o = HotellingOptions {
            reference: HotellingReference::Known {
                mean: vec![10.0, 5.0],
                covariance: Matrix::from_rows(&[vec![0.08, 0.05], vec![0.05, 0.06]]).unwrap(),
            },
            ..HotellingOptions::default()
        };
        let c = hotelling_individuals(&d, &o).unwrap();
        assert_eq!(c.flagged(), vec![30]);
        let chi = ChiSquared::new(2.0).unwrap().quantile(1.0 - 0.00135);
        assert!((c.ucl - chi).abs() < 1e-12);
    }

    #[test]
    fn phase_two_limit_exceeds_phase_one_style_limit_formula() {
        let d = data();
        let hist = HotellingOptions {
            reference: HotellingReference::Historical {
                mean: vec![10.0, 5.0],
                covariance: Matrix::from_rows(&[vec![0.08, 0.05], vec![0.05, 0.06]]).unwrap(),
                m: 30,
            },
            ..HotellingOptions::default()
        };
        let c = hotelling_individuals(&d, &hist).unwrap();
        let f = FisherF::new(2.0, 28.0).unwrap().quantile(1.0 - 0.00135);
        assert!((c.ucl - 2.0 * 31.0 * 29.0 / (30.0 * 28.0) * f).abs() < 1e-12);
    }

    #[test]
    fn successive_differences_run() {
        let c = hotelling_individuals(
            &data(),
            &HotellingOptions { covariance: CovarianceEstimate::SuccessiveDifferences, ..HotellingOptions::default() },
        )
        .unwrap();
        assert_eq!(c.t2.len(), 30);
        assert!(c.limit_description.contains("Sullivan"));
    }

    #[test]
    fn subgroup_chart_and_errors() {
        let d = data();
        let groups: Vec<Vec<Vec<f64>>> = d.chunks(3).map(<[Vec<f64>]>::to_vec).collect();
        let c = hotelling_subgroups(&groups, &HotellingOptions::default()).unwrap();
        assert_eq!((c.m, c.n, c.p), (10, 3, 2));
        let nu = 10.0 * 3.0 - 10.0 - 2.0 + 1.0;
        let f = FisherF::new(2.0, nu).unwrap().quantile(1.0 - 0.00135);
        assert!((c.ucl - 2.0 * 9.0 * 2.0 / nu * f).abs() < 1e-12);
        assert!(hotelling_subgroups(&groups[..1], &HotellingOptions::default()).is_err());
        assert!(hotelling_individuals(&[vec![1.0, 2.0]], &HotellingOptions::default()).is_err());
        assert!(
            hotelling_individuals(&data(), &HotellingOptions { alpha: 1.5, ..HotellingOptions::default() }).is_err()
        );
        let ragged = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(hotelling_individuals(&ragged, &HotellingOptions::default()).is_err());
    }
}
