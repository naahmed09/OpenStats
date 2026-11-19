//! General MANOVA by formula: Wilks' lambda, Pillai's trace, Lawley–Hotelling trace and Roy's largest root.

use super::common::{eig_desc, names_or_default};
use crate::error::{invalid, Result, StatsError};
use crate::model::{build_design, Contrast, Data, DesignOptions};
use openstats_math::dist::{ContinuousDistribution, FisherF};
use openstats_math::linalg::{Cholesky, Svd};
use openstats_math::Matrix;

/// Options of [`manova`].
#[derive(Debug, Clone, Default)]
pub struct ManovaOptions {
    /// Use treatment (reference-cell) coding instead of the default sum-to-zero coding.
    pub treatment_coding: bool,
}

/// One multivariate test statistic with its F approximation.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateTest {
    pub name: &'static str,
    pub statistic: f64,
    pub f: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    /// The F distribution is exact (otherwise an approximation; Roy's is an upper bound).
    pub exact: bool,
}

/// Tests of one model term.
#[derive(Debug, Clone)]
pub struct ManovaTerm {
    pub label: String,
    /// Degrees of freedom of the term.
    pub df: usize,
    /// Hypothesis SSCP matrix (adjusted for the other terms).
    pub hypothesis: Matrix,
    /// Eigenvalues of E⁻¹H, descending (min(p, df) are non-zero).
    pub eigenvalues: Vec<f64>,
    /// Wilks, Lawley–Hotelling, Pillai, Roy.
    pub tests: Vec<MultivariateTest>,
}

/// Result of [`manova`].
#[derive(Debug, Clone)]
pub struct Manova {
    pub responses: Vec<String>,
    pub n: usize,
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    pub error_df: usize,
    pub error_sscp: Matrix,
    /// Correlations of the error (residual) SSCP.
    pub error_correlation: Matrix,
    pub terms: Vec<ManovaTerm>,
}

pub fn manova(
    responses: &[Vec<f64>],
    names: &[String],
    data: &Data,
    rhs: &str,
    opts: &ManovaOptions,
) -> Result<Manova> {
    let p = responses.len();
    if p < 2 {
        return invalid("MANOVA needs at least two response variables");
    }
    let nrows = responses[0].len();
    if responses.iter().any(|r| r.len() != nrows) || data.n_rows() != nrows {
        return invalid("the responses and the model terms have different lengths");
    }
    let mut first = responses[0].clone();
    for i in 0..nrows {
        if responses.iter().any(|r| r[i].is_nan()) {
            first[i] = f64::NAN;
        }
        if responses.iter().any(|r| r[i].is_infinite()) {
            return Err(StatsError::NonFinite);
        }
    }
    let d = data.clone().with_numeric("__y", first)?;
    let dopts = DesignOptions {
        default_contrast: if opts.treatment_coding {
            Contrast::Treatment(crate::model::Reference::First)
        } else {
            Contrast::SumToZero
        },
        ..DesignOptions::default()
    };
    let rhs = rhs.trim();
    if rhs.is_empty() {
        return invalid("the model has no terms");
    }
    let (_, dm) = build_design(&format!("__y ~ {rhs}"), &d, &dopts)?;
    let n = dm.x.rows();
    let rows_used = dm.rows_used.clone();
    let y_cols: Vec<Vec<f64>> = responses.iter().map(|r| rows_used.iter().map(|&i| r[i]).collect()).collect();
    let rank_full = rank_of(&dm.x)?;
    if n < rank_full + p {
        return Err(StatsError::InsufficientData {
            what: "MANOVA error degrees of freedom",
            needed: rank_full + p,
            got: n,
        });
    }
    let v = n - rank_full;
    let e_full = sscp(&dm.x, &y_cols)?;
    let chol = Cholesky::new(&e_full).map_err(|_| {
        StatsError::InvalidArgument("the error SSCP matrix is singular (responses are linearly dependent)".into())
    })?;

    let mut terms = Vec::new();
    for t in &dm.terms {
        let keep: Vec<usize> = (0..dm.x.cols()).filter(|&c| c < t.start || c >= t.start + t.len).collect();
        let xr = dm.x.select(&(0..n).collect::<Vec<_>>(), &keep);
        let q = rank_full - rank_of(&xr)?;
        if q == 0 {
            continue; // aliased term
        }
        let e_red = sscp(&xr, &y_cols)?;
        let h = e_red.try_sub(&e_full)?.symmetrize();
        let eig = hypothesis_eigenvalues(&chol, &h)?;
        let tests = tests_from(&eig, p, q, v)?;
        terms.push(ManovaTerm { label: t.label.clone(), df: q, hypothesis: h, eigenvalues: eig, tests });
    }
    let sd: Vec<f64> = (0..p).map(|i| e_full[(i, i)].sqrt()).collect();
    let error_correlation = Matrix::from_fn(p, p, |i, j| e_full[(i, j)] / (sd[i] * sd[j]));
    Ok(Manova {
        responses: names_or_default(names, p, "C"),
        n,
        n_dropped: nrows - n,
        rows_used,
        error_df: v,
        error_sscp: e_full,
        error_correlation,
        terms,
    })
}

fn rank_of(x: &Matrix) -> Result<usize> {
    if x.cols() == 0 {
        return Ok(0);
    }
    Ok(Svd::new(x)?.rank(None))
}

/// Residual SSCP matrix of the responses after regressing on `x`.
fn sscp(x: &Matrix, y: &[Vec<f64>]) -> Result<Matrix> {
    let n = x.rows();
    let p = y.len();
    let svd = Svd::new(x)?;
    let mut res: Vec<Vec<f64>> = Vec::with_capacity(p);
    for col in y {
        let beta = svd.solve_least_squares(col, None)?;
        let fit = x.matvec(&beta)?;
        res.push((0..n).map(|i| col[i] - fit[i]).collect());
    }
    Ok(Matrix::from_fn(p, p, |a, b| (0..n).map(|i| res[a][i] * res[b][i]).sum()))
}

/// Eigenvalues of E⁻¹H through the symmetric form L⁻¹HL⁻ᵀ.
fn hypothesis_eigenvalues(chol: &Cholesky, h: &Matrix) -> Result<Vec<f64>> {
    let l = chol.l();
    let p = l.rows();
    let mut a = h.clone();
    for c in 0..p {
        for i in 0..p {
            let mut s = a[(i, c)];
            for j in 0..i {
                s -= l[(i, j)] * a[(j, c)];
            }
            a[(i, c)] = s / l[(i, i)];
        }
    }
    let mut b = a.t();
    for c in 0..p {
        for i in 0..p {
            let mut s = b[(i, c)];
            for j in 0..i {
                s -= l[(i, j)] * b[(j, c)];
            }
            b[(i, c)] = s / l[(i, i)];
        }
    }
    let (vals, _) = eig_desc(&b.symmetrize())?;
    Ok(vals.into_iter().map(|v| v.max(0.0)).collect())
}

fn f_p(f: f64, d1: f64, d2: f64) -> Result<f64> {
    if !(d1 > 0.0 && d2 > 0.0) {
        return invalid("the error degrees of freedom are too small for the F approximation");
    }
    Ok(FisherF::new(d1, d2)?.sf(f))
}

fn tests_from(theta: &[f64], p: usize, q: usize, v: usize) -> Result<Vec<MultivariateTest>> {
    let (pf, qf, vf) = (p as f64, q as f64, v as f64);
    let s = p.min(q) as f64;
    let m = ((pf - qf).abs() - 1.0) / 2.0;
    let nn = (vf - pf - 1.0) / 2.0;
    let exact = p.min(q) <= 2;
    let lambda: f64 = theta.iter().map(|t| 1.0 / (1.0 + t)).product();
    let pillai: f64 = theta.iter().map(|t| t / (1.0 + t)).sum();
    let hl: f64 = theta.iter().sum();
    let roy = theta.first().copied().unwrap_or(0.0);

    // Wilks (Rao's F).
    let r = vf - (pf - qf + 1.0) / 2.0;
    let u = (pf * qf - 2.0) / 4.0;
    let t = if pf * pf + qf * qf - 5.0 > 0.0 {
        ((pf * pf * qf * qf - 4.0) / (pf * pf + qf * qf - 5.0)).sqrt()
    } else {
        1.0
    };
    let w_df1 = pf * qf;
    let w_df2 = r * t - 2.0 * u;
    let lt = lambda.powf(1.0 / t);
    let w_f = (1.0 - lt) / lt * w_df2 / w_df1;

    // Pillai.
    let p_df1 = s * (2.0 * m + s + 1.0);
    let p_df2 = s * (2.0 * nn + s + 1.0);
    let p_f = (2.0 * nn + s + 1.0) / (2.0 * m + s + 1.0) * pillai / (s - pillai);

    // Lawley–Hotelling (McKeon's approximation, as used by R's summary.manova; exact when min(p, q) <= 2).
    let h_df1 = s * (2.0 * m + s + 1.0);
    let h_df2 = 2.0 * (s * nn + 1.0);
    let h_f = 2.0 * (s * nn + 1.0) * hl / (s * s * (2.0 * m + s + 1.0));

    // Roy (upper bound).
    let rr = pf.max(qf);
    let r_df1 = rr;
    let r_df2 = vf - rr + qf;
    let r_f = roy * r_df2 / rr;

    Ok(vec![
        MultivariateTest {
            name: "Wilks",
            statistic: lambda,
            f: w_f,
            df1: w_df1,
            df2: w_df2,
            p_value: f_p(w_f, w_df1, w_df2)?,
            exact,
        },
        MultivariateTest {
            name: "Lawley-Hotelling",
            statistic: hl,
            f: h_f,
            df1: h_df1,
            df2: h_df2,
            p_value: f_p(h_f, h_df1, h_df2)?,
            exact,
        },
        MultivariateTest {
            name: "Pillai",
            statistic: pillai,
            f: p_f,
            df1: p_df1,
            df2: p_df2,
            p_value: f_p(p_f, p_df1, p_df2)?,
            exact,
        },
        MultivariateTest {
            name: "Roy",
            statistic: roy,
            f: r_f,
            df1: r_df1,
            df2: r_df2,
            p_value: f_p(r_f, r_df1, r_df2)?,
            exact: p.min(q) == 1,
        },
    ])
}
