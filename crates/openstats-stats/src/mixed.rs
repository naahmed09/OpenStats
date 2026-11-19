//! Linear mixed-effects models with random intercept terms, fitted by REML or
//! ML.
//!
//! The model is `y = Xβ + Σⱼ Zⱼuⱼ + ε` with `uⱼ ~ N(0, σ²ⱼ I)` and
//! `ε ~ N(0, σ²ₑ I)`. Each random term groups the observations by the levels of
//! one or more factors (crossed, nested or interaction groupings); every
//! level has its own random intercept. With `θⱼ = σ²ⱼ/σ²ₑ`, `Λ = diag(√θ)` and
//! `B = I + ΛZᵀZΛ` the criterion needs only `q × q` and `p × p` Cholesky
//! factors (`q` = total number of random-effect levels):
//!
//! `V⁻¹ = I − ZΛB⁻¹ΛZᵀ`, `ln|V_θ| = ln|B|`.
//!
//! The variance parameters are found by Nelder–Mead and BFGS on `sⱼ = √θⱼ`
//! (so zero variances are reachable), their covariance comes from the observed
//! information of the REML criterion, and fixed-effect degrees of freedom use
//! Satterthwaite's approximation (multi-df tests as in `lmerTest`).

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::model::{build_design, Contrast, Data, DesignOptions};
use crate::regression::independent_columns;
use openstats_math::dist::{ContinuousDistribution, FisherF, StudentT};
use openstats_math::linalg::{Cholesky, Matrix, SymmetricEigen};
use openstats_math::optim::{bfgs, nelder_mead, NelderMeadOptions, Options};
use std::collections::BTreeMap;

const MAX_LEVELS: usize = 2500;

/// Estimation criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedMethod {
    /// Restricted maximum likelihood.
    Reml,
    /// Maximum likelihood.
    Ml,
}

/// A random-intercept term.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomSpec {
    /// Display label.
    pub label: String,
    /// Factors whose level combinations define the groups.
    pub factors: Vec<String>,
}

impl RandomSpec {
    /// A random term grouped by `factors`; the label joins them with `*`.
    pub fn new(factors: &[&str]) -> RandomSpec {
        RandomSpec { label: factors.join("*"), factors: factors.iter().map(|s| s.to_string()).collect() }
    }
}

/// Options of a mixed-model fit.
#[derive(Debug, Clone)]
pub struct MixedOptions {
    /// REML (default) or ML.
    pub method: MixedMethod,
    /// Design coding of the fixed factors (sum-to-zero by default).
    pub design: DesignOptions,
    /// Confidence level.
    pub level: f64,
}

impl Default for MixedOptions {
    fn default() -> Self {
        MixedOptions {
            method: MixedMethod::Reml,
            design: DesignOptions { default_contrast: Contrast::SumToZero, ..DesignOptions::default() },
            level: 0.95,
        }
    }
}

/// A fixed-effect coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedEffect {
    /// Column name.
    pub name: String,
    /// Estimate.
    pub estimate: f64,
    /// Standard error.
    pub se: f64,
    /// Satterthwaite degrees of freedom.
    pub df: f64,
    /// t statistic.
    pub t: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Confidence interval.
    pub ci: Interval,
}

/// A test of one fixed term (all its coefficients zero).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTest {
    /// Term label.
    pub term: String,
    /// Numerator degrees of freedom.
    pub num_df: usize,
    /// Denominator (Satterthwaite) degrees of freedom.
    pub den_df: f64,
    /// F statistic.
    pub f: f64,
    /// p-value.
    pub p_value: f64,
}

/// A variance component.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedVariance {
    /// Random term or `Error`.
    pub source: String,
    /// Variance estimate.
    pub variance: f64,
    /// Standard error from the observed information (`None` on the boundary or when it is singular).
    pub se: Option<f64>,
    /// Standard deviation.
    pub std_dev: f64,
    /// Number of levels of the random term (observations for the error).
    pub levels: usize,
}

/// Best linear unbiased predictions of one random term.
#[derive(Debug, Clone, PartialEq)]
pub struct Blups {
    /// Random term label.
    pub term: String,
    /// Level labels (factor levels joined by a space) and predicted effects.
    pub effects: Vec<(String, f64)>,
}

/// A fitted mixed model.
#[derive(Debug, Clone)]
pub struct MixedFit {
    /// Response column.
    pub response: String,
    /// Fixed-effect formula.
    pub formula: String,
    /// Estimation method.
    pub method: MixedMethod,
    /// Observations used.
    pub n: usize,
    /// Fixed-effect coefficients (aliased columns excluded).
    pub fixed: Vec<FixedEffect>,
    /// Type III tests of the fixed terms.
    pub tests: Vec<FixedTest>,
    /// Random-term variances followed by the error variance.
    pub variances: Vec<MixedVariance>,
    /// Log-likelihood of the criterion that was maximised (restricted for REML).
    pub log_likelihood: f64,
    /// Akaike information criterion (`−2ℓ + 2k`, k = number of fixed effects for ML, plus the variances).
    pub aic: f64,
    /// Corrected AIC.
    pub aicc: f64,
    /// Bayesian information criterion (`−2ℓ + k ln n`).
    pub bic: f64,
    /// Number of parameters used in the criteria.
    pub n_params: usize,
    /// Predicted random effects.
    pub blups: Vec<Blups>,
    /// Fixed-effect fitted values `Xβ`.
    pub fitted_marginal: Vec<f64>,
    /// Conditional fitted values `Xβ + Zû`.
    pub fitted: Vec<f64>,
    /// Conditional residuals `y − fitted`.
    pub residuals: Vec<f64>,
    /// Marginal residuals `y − Xβ`.
    pub residuals_marginal: Vec<f64>,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    /// Whether the optimiser reported convergence.
    pub converged: bool,
    /// Confidence level.
    pub level: f64,
}

struct Prep {
    n: usize,
    p: usize,
    q: usize,
    xtx: Matrix,
    xty: Vec<f64>,
    yty: f64,
    zty: Vec<f64>,
    xtz: Matrix,
    ztz: Matrix,
    block: Vec<usize>,
}

struct Eval {
    r2: f64,
    ldb: f64,
    ldx: f64,
    beta: Vec<f64>,
    xvx_inv: Matrix,
}

fn evaluate(pr: &Prep, theta: &[f64]) -> Option<Eval> {
    let (p, q) = (pr.p, pr.q);
    let lam: Vec<f64> = pr.block.iter().map(|&b| theta[b].max(0.0).sqrt()).collect();
    let mut b = Matrix::identity(q);
    for i in 0..q {
        for j in 0..q {
            b[(i, j)] += lam[i] * pr.ztz[(i, j)] * lam[j];
        }
    }
    let chol = Cholesky::new(&b).ok()?;
    // ΛZᵀX (q × p) and Λ Zᵀy.
    let mut lzx = Matrix::zeros(q, p);
    for i in 0..q {
        for j in 0..p {
            lzx[(i, j)] = lam[i] * pr.xtz[(j, i)];
        }
    }
    let lzy: Vec<f64> = (0..q).map(|i| lam[i] * pr.zty[i]).collect();
    let s = chol.solve_mat(&lzx).ok()?;
    let sy = chol.solve(&lzy).ok()?;
    let mut xvx = pr.xtx.clone();
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for k in 0..q {
                acc += lzx[(k, i)] * s[(k, j)];
            }
            xvx[(i, j)] -= acc;
        }
    }
    let mut xvy = pr.xty.clone();
    for i in 0..p {
        let acc: f64 = (0..q).map(|k| lzx[(k, i)] * sy[k]).sum();
        xvy[i] -= acc;
    }
    let yvy = pr.yty - lzy.iter().zip(&sy).map(|(a, c)| a * c).sum::<f64>();
    let cx = Cholesky::new(&xvx.symmetrize()).ok()?;
    let beta = cx.solve(&xvy).ok()?;
    let r2 = yvy - beta.iter().zip(&xvy).map(|(a, c)| a * c).sum::<f64>();
    if r2 <= 0.0 || r2.is_nan() || !r2.is_finite() {
        return None;
    }
    let ldb = chol.log_det();
    let ldx = cx.log_det();
    let xvx_inv = cx.inverse().ok()?;
    Some(Eval { r2, ldb, ldx, beta, xvx_inv })
}

fn profile(pr: &Prep, theta: &[f64], method: MixedMethod) -> f64 {
    match evaluate(pr, theta) {
        None => f64::INFINITY,
        Some(e) => match method {
            MixedMethod::Reml => (pr.n - pr.p) as f64 * e.r2.ln() + e.ldb + e.ldx,
            MixedMethod::Ml => pr.n as f64 * e.r2.ln() + e.ldb,
        },
    }
}

/// The criterion as a function of the variances `φ = (σ²₁, …, σ²ₖ, σ²ₑ)`.
fn crit_full(pr: &Prep, phi: &[f64], method: MixedMethod) -> f64 {
    let k = phi.len() - 1;
    let se = phi[k];
    if se <= 0.0 || se.is_nan() {
        return f64::INFINITY;
    }
    let theta: Vec<f64> = phi[..k].iter().map(|v| v.max(0.0) / se).collect();
    match evaluate(pr, &theta) {
        None => f64::INFINITY,
        Some(e) => match method {
            MixedMethod::Reml => (pr.n - pr.p) as f64 * se.ln() + e.ldb + e.ldx + e.r2 / se,
            MixedMethod::Ml => pr.n as f64 * se.ln() + e.ldb + e.r2 / se,
        },
    }
}

/// `Cov(β̂)` at the variances `φ`.
fn cov_beta(pr: &Prep, phi: &[f64]) -> Option<Matrix> {
    let k = phi.len() - 1;
    let se = phi[k];
    let theta: Vec<f64> = phi[..k].iter().map(|v| v.max(0.0) / se).collect();
    let e = evaluate(pr, &theta)?;
    Some(e.xvx_inv.scale(se))
}

fn quad(m: &Matrix, v: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..v.len() {
        for j in 0..v.len() {
            s += v[i] * m[(i, j)] * v[j];
        }
    }
    s
}

/// Fit the linear mixed model `formula` (fixed effects) with the random terms `random`.
pub fn mixed_model(formula: &str, random: &[RandomSpec], data: &Data, opts: &MixedOptions) -> Result<MixedFit> {
    if random.is_empty() {
        return invalid("a mixed model needs at least one random term");
    }
    let (_, dm) = build_design(formula, data, &opts.design)?;
    let response = formula.split('~').next().unwrap_or("y").trim().to_string();
    let y = dm.y.clone().ok_or_else(|| StatsError::InvalidArgument("the model has no response".into()))?;
    let n = y.len();
    let (kept, _) = independent_columns(&dm.x, 1e-10);
    let p = kept.len();
    if p == 0 || n <= p {
        return Err(StatsError::InsufficientData {
            what: "mixed model (observations must exceed fixed effects)",
            needed: p + 1,
            got: n,
        });
    }
    let all_rows: Vec<usize> = (0..n).collect();
    let x = dm.x.select(&all_rows, &kept);

    // Random-effect groups.
    let mut zidx: Vec<Vec<usize>> = Vec::new();
    let mut keys: Vec<Vec<String>> = Vec::new();
    let mut offsets = Vec::new();
    let mut block = Vec::new();
    let mut q = 0usize;
    for (b, spec) in random.iter().enumerate() {
        if spec.factors.is_empty() {
            return invalid(format!("random term '{}' has no grouping factor", spec.label));
        }
        let mut cols: Vec<Vec<String>> = Vec::new();
        for f in &spec.factors {
            let col = data.factor(f).ok_or_else(|| {
                StatsError::InvalidArgument(format!("random factor '{f}' is not a categorical column"))
            })?;
            cols.push(dm.rows_used.iter().map(|&r| col[r].clone()).collect());
        }
        let mut map: BTreeMap<Vec<String>, usize> = BTreeMap::new();
        let mut kk: Vec<Vec<String>> = Vec::new();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let key: Vec<String> = cols.iter().map(|c| c[i].clone()).collect();
            let next = kk.len();
            let id = *map.entry(key.clone()).or_insert_with(|| {
                kk.push(key);
                next
            });
            ids.push(id);
        }
        if kk.len() < 2 {
            return invalid(format!("random term '{}' has only one level", spec.label));
        }
        offsets.push(q);
        block.extend(std::iter::repeat_n(b, kk.len()));
        q += kk.len();
        zidx.push(ids);
        keys.push(kk.into_iter().map(|k| k.join(" ")).collect());
    }
    if q > MAX_LEVELS {
        return invalid(format!("the random terms have {q} levels in total; the limit is {MAX_LEVELS}"));
    }
    let k_terms = random.len();

    // Cross-products.
    let xtx = x.gram();
    let xty = x.t_matvec(&y)?;
    let yty: f64 = y.iter().map(|v| v * v).sum();
    let mut zty = vec![0.0; q];
    let mut xtz = Matrix::zeros(p, q);
    let mut ztz = Matrix::zeros(q, q);
    for i in 0..n {
        let act: Vec<usize> = (0..k_terms).map(|b| offsets[b] + zidx[b][i]).collect();
        for &a in &act {
            zty[a] += y[i];
            for j in 0..p {
                xtz[(j, a)] += x[(i, j)];
            }
            for &c in &act {
                ztz[(a, c)] += 1.0;
            }
        }
    }
    let pr = Prep { n, p, q, xtx, xty, yty, zty, xtz, ztz, block };

    // Optimise the variance ratios on the square-root scale.
    let method = opts.method;
    let mut best: Option<(f64, Vec<f64>)> = None;
    let mut converged = false;
    for start in [1.0f64, 0.3, 3.0] {
        let x0 = vec![start; k_terms];
        let mut f = |s: &[f64]| {
            let theta: Vec<f64> = s.iter().map(|v| v * v).collect();
            profile(&pr, &theta, method)
        };
        let nm = nelder_mead(&mut f, &x0, &NelderMeadOptions { max_iter: 4000, ..NelderMeadOptions::default() });
        let (xs, fs) = match nm {
            Ok(m) => (m.x, m.f),
            Err(_) => continue,
        };
        let mut g = |s: &[f64]| {
            let theta: Vec<f64> = s.iter().map(|v| v * v).collect();
            profile(&pr, &theta, method)
        };
        let polished = bfgs(&mut g, None, &xs, &Options::default());
        let (xs, fs, conv) = match polished {
            Ok(m) if m.f <= fs => (m.x, m.f, m.converged),
            _ => (xs, fs, true),
        };
        if fs.is_finite() && best.as_ref().is_none_or(|b| fs < b.0 - 1e-10) {
            best = Some((fs, xs));
            converged = conv;
        }
    }
    let (_, s_hat) = best.ok_or_else(|| {
        StatsError::InvalidArgument("the mixed-model criterion could not be evaluated (singular design)".into())
    })?;
    let mut theta: Vec<f64> = s_hat.iter().map(|v| v * v).collect();
    for t in &mut theta {
        if *t < 1e-9 {
            *t = 0.0;
        }
    }
    let ev = evaluate(&pr, &theta).ok_or_else(|| StatsError::InvalidArgument("the fitted model is singular".into()))?;
    let dfe = match method {
        MixedMethod::Reml => (n - p) as f64,
        MixedMethod::Ml => n as f64,
    };
    let sigma2 = ev.r2 / dfe;
    let mut phi: Vec<f64> = theta.iter().map(|t| t * sigma2).collect();
    phi.push(sigma2);

    // Covariance of the variance parameters (observed information, active set only).
    let active: Vec<usize> = (0..phi.len()).filter(|&i| i == phi.len() - 1 || theta[i] > 0.0).collect();
    let a = active.len();
    let f0 = crit_full(&pr, &phi, method);
    let h: Vec<f64> = active.iter().map(|&i| 2e-4 * phi[i].max(1e-12)).collect();
    let shifted = |d: &[(usize, f64)]| -> f64 {
        let mut v = phi.clone();
        for &(i, dv) in d {
            v[active[i]] += dv;
        }
        crit_full(&pr, &v, method)
    };
    let mut hess = Matrix::zeros(a, a);
    for i in 0..a {
        let fp = shifted(&[(i, h[i])]);
        let fm = shifted(&[(i, -h[i])]);
        hess[(i, i)] = (fp - 2.0 * f0 + fm) / (h[i] * h[i]);
        for j in 0..i {
            let fpp = shifted(&[(i, h[i]), (j, h[j])]);
            let fpm = shifted(&[(i, h[i]), (j, -h[j])]);
            let fmp = shifted(&[(i, -h[i]), (j, h[j])]);
            let fmm = shifted(&[(i, -h[i]), (j, -h[j])]);
            let v = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j]);
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
    let cov_phi: Option<Matrix> =
        Cholesky::new(&hess.symmetrize()).ok().and_then(|c| c.inverse().ok()).map(|m| m.scale(2.0));

    // Fixed effects.
    let cb = cov_beta(&pr, &phi)
        .ok_or_else(|| StatsError::InvalidArgument("the fixed-effect covariance is singular".into()))?;
    let fallback_df = (n - p) as f64;
    let df_for = |ell: &[f64]| -> f64 {
        let v0 = quad(&cb, ell);
        let Some(cp) = &cov_phi else { return fallback_df };
        let mut g = vec![0.0; a];
        for i in 0..a {
            let mut up = phi.clone();
            let mut dn = phi.clone();
            up[active[i]] += h[i];
            dn[active[i]] -= h[i];
            match (cov_beta(&pr, &up), cov_beta(&pr, &dn)) {
                (Some(cu), Some(cd)) => g[i] = (quad(&cu, ell) - quad(&cd, ell)) / (2.0 * h[i]),
                _ => return fallback_df,
            }
        }
        let den = quad(cp, &g);
        let df = 2.0 * v0 * v0 / den;
        if den > 0.0 && df.is_finite() && df > 0.0 {
            df
        } else {
            fallback_df
        }
    };
    let alpha = 1.0 - opts.level;
    let mut fixed = Vec::new();
    for (j, &col) in kept.iter().enumerate() {
        let mut ell = vec![0.0; p];
        ell[j] = 1.0;
        let se = cb[(j, j)].max(0.0).sqrt();
        let df = df_for(&ell);
        let t = ev.beta[j] / se;
        let dist = StudentT::new(df)?;
        let pv = 2.0 * dist.sf(t.abs());
        let tq = dist.isf(alpha / 2.0);
        fixed.push(FixedEffect {
            name: dm.names[col].clone(),
            estimate: ev.beta[j],
            se,
            df,
            t,
            p_value: pv.min(1.0),
            ci: Interval { lower: ev.beta[j] - tq * se, upper: ev.beta[j] + tq * se, level: opts.level },
        });
    }
    // Term tests.
    let mut tests = Vec::new();
    for tc in &dm.terms {
        let idx: Vec<usize> =
            kept.iter().enumerate().filter(|(_, &c)| c >= tc.start && c < tc.start + tc.len).map(|(j, _)| j).collect();
        let qd = idx.len();
        if qd == 0 {
            continue;
        }
        let mut lcl = Matrix::zeros(qd, qd);
        for (a_i, &i) in idx.iter().enumerate() {
            for (b_i, &j) in idx.iter().enumerate() {
                lcl[(a_i, b_i)] = cb[(i, j)];
            }
        }
        let eig = SymmetricEigen::new(&lcl.symmetrize())?;
        let lb: Vec<f64> = idx.iter().map(|&i| ev.beta[i]).collect();
        let mut fstat = 0.0;
        let mut e_sum = 0.0;
        for m in 0..qd {
            let d = eig.values[m];
            if d <= 0.0 || d.is_nan() {
                continue;
            }
            let dir: Vec<f64> = (0..qd).map(|i| eig.vectors[(i, m)]).collect();
            let est: f64 = dir.iter().zip(&lb).map(|(a, b)| a * b).sum();
            fstat += est * est / d;
            let mut ell = vec![0.0; p];
            for (a_i, &i) in idx.iter().enumerate() {
                ell[i] = dir[a_i];
            }
            let dfm = df_for(&ell);
            if dfm > 2.0 {
                e_sum += dfm / (dfm - 2.0);
            }
        }
        fstat /= qd as f64;
        let den_df = if e_sum > qd as f64 { 2.0 * e_sum / (e_sum - qd as f64) } else { fallback_df };
        let pv = FisherF::new(qd as f64, den_df)?.sf(fstat);
        tests.push(FixedTest { term: tc.label.clone(), num_df: qd, den_df, f: fstat, p_value: pv });
    }

    // Variance table.
    let mut variances = Vec::new();
    for (b, spec) in random.iter().enumerate() {
        let se =
            active.iter().position(|&i| i == b).and_then(|pos| cov_phi.as_ref().map(|c| c[(pos, pos)].max(0.0).sqrt()));
        let levels = keys[b].len();
        variances.push(MixedVariance {
            source: spec.label.clone(),
            variance: phi[b],
            se,
            std_dev: phi[b].sqrt(),
            levels,
        });
    }
    let epos = active.len() - 1;
    variances.push(MixedVariance {
        source: "Error".into(),
        variance: sigma2,
        se: cov_phi.as_ref().map(|c| c[(epos, epos)].max(0.0).sqrt()),
        std_dev: sigma2.sqrt(),
        levels: n,
    });

    // Information criteria.
    let (m2ll, n_params) = match method {
        MixedMethod::Reml => (f0 + (n - p) as f64 * (2.0 * std::f64::consts::PI).ln(), p + k_terms + 1),
        MixedMethod::Ml => (f0 + n as f64 * (2.0 * std::f64::consts::PI).ln(), p + k_terms + 1),
    };
    let kf = n_params as f64;
    let aic = m2ll + 2.0 * kf;
    let aicc = if (n as f64) - kf - 1.0 > 0.0 { aic + 2.0 * kf * (kf + 1.0) / (n as f64 - kf - 1.0) } else { f64::NAN };
    let bic = m2ll + kf * (n as f64).ln();

    // BLUPs and fitted values.
    let xb = x.matvec(&ev.beta)?;
    let resid_m: Vec<f64> = y.iter().zip(&xb).map(|(a, c)| a - c).collect();
    let lam: Vec<f64> = pr.block.iter().map(|&b| theta[b].sqrt()).collect();
    let mut zr = vec![0.0; q];
    for i in 0..n {
        for b in 0..k_terms {
            zr[offsets[b] + zidx[b][i]] += resid_m[i];
        }
    }
    let mut bm = Matrix::identity(q);
    for i in 0..q {
        for j in 0..q {
            bm[(i, j)] += lam[i] * pr.ztz[(i, j)] * lam[j];
        }
    }
    let rhs: Vec<f64> = (0..q).map(|i| lam[i] * zr[i]).collect();
    let sol = Cholesky::new(&bm)?.solve(&rhs)?;
    let u: Vec<f64> = (0..q).map(|i| lam[i] * sol[i]).collect();
    let mut fitted = xb.clone();
    for i in 0..n {
        for b in 0..k_terms {
            fitted[i] += u[offsets[b] + zidx[b][i]];
        }
    }
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, c)| a - c).collect();
    let blups = random
        .iter()
        .enumerate()
        .map(|(b, spec)| Blups {
            term: spec.label.clone(),
            effects: keys[b].iter().enumerate().map(|(l, k)| (k.clone(), u[offsets[b] + l])).collect(),
        })
        .collect();

    Ok(MixedFit {
        response,
        formula: formula.to_string(),
        method,
        n,
        fixed,
        tests,
        variances,
        log_likelihood: -0.5 * m2ll,
        aic,
        aicc,
        bic,
        n_params,
        blups,
        fitted_marginal: xb,
        fitted,
        residuals,
        residuals_marginal: resid_m,
        rows_used: dm.rows_used.clone(),
        converged,
        level: opts.level,
    })
}

impl crate::report::Reportable for MixedFit {
    fn report_section(&self) -> crate::report::Section {
        use crate::report::{Cell, Column, ColumnKind as K, Section, Table};
        let int = |n: usize| Cell::Int(i64::try_from(n).unwrap_or(i64::MAX));
        let method = match self.method {
            MixedMethod::Reml => "REML",
            MixedMethod::Ml => "ML",
        };
        let mut sec = Section::new("Mixed Effects Model");
        sec.text(format!(
            "Method: {method}. Model: {}. {} observations. Fixed-effect degrees of freedom: Satterthwaite approximation.{}",
            self.formula,
            self.n,
            if self.converged { "" } else { " The optimiser did not report convergence." }
        ));
        let mut t = Table::new(
            "Tests of Fixed Effects",
            vec![
                Column::new("Source", K::Text),
                Column::new("Num DF", K::Integer),
                Column::new("Den DF", K::Fixed(2)),
                Column::new("F-Value", K::Significant(6)),
                Column::new("P-Value", K::PValue),
            ],
        );
        for r in &self.tests {
            t.row(vec![Cell::text(&r.term), int(r.num_df), Cell::Num(r.den_df), Cell::Num(r.f), Cell::Num(r.p_value)]);
        }
        sec.table(t);
        let mut c = Table::new(
            "Fixed Effects",
            vec![
                Column::new("Term", K::Text),
                Column::new("Coef", K::Significant(6)),
                Column::new("SE Coef", K::Significant(6)),
                Column::new("DF", K::Fixed(2)),
                Column::new("T-Value", K::Fixed(2)),
                Column::new("P-Value", K::PValue),
                Column::new(format!("{}% CI lower", (self.level * 100.0).round()), K::Significant(6)),
                Column::new("upper", K::Significant(6)),
            ],
        );
        for f in &self.fixed {
            c.row(vec![
                Cell::text(&f.name),
                Cell::Num(f.estimate),
                Cell::Num(f.se),
                Cell::Num(f.df),
                Cell::Num(f.t),
                Cell::Num(f.p_value),
                Cell::Num(f.ci.lower),
                Cell::Num(f.ci.upper),
            ]);
        }
        sec.table(c);
        let mut v = Table::new(
            "Variance Components",
            vec![
                Column::new("Source", K::Text),
                Column::new("Levels", K::Integer),
                Column::new("Variance", K::Significant(6)),
                Column::new("SE Variance", K::Significant(6)),
                Column::new("StDev", K::Significant(6)),
            ],
        );
        for r in &self.variances {
            v.row(vec![
                Cell::text(&r.source),
                int(r.levels),
                Cell::Num(r.variance),
                Cell::opt(r.se),
                Cell::Num(r.std_dev),
            ]);
        }
        sec.table(v);
        let mut m = Table::new(
            "Model Summary",
            vec![
                Column::new("Log-likelihood", K::Significant(6)),
                Column::new("AIC", K::Significant(6)),
                Column::new("AICc", K::Significant(6)),
                Column::new("BIC", K::Significant(6)),
            ],
        );
        m.row(vec![
            Cell::Num(self.log_likelihood),
            Cell::Num(self.aic),
            Cell::opt(Some(self.aicc)),
            Cell::Num(self.bic),
        ]);
        sec.table(m);
        if self.variances.iter().any(|r| r.variance == 0.0) {
            sec.warn("A variance component is estimated at zero (boundary); its standard error is not available.");
        }
        sec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_one_way_matches_anova_estimators() {
        // 4 groups x 3 observations; random group intercept. For balanced data REML
        // equals the ANOVA estimators when they are positive.
        let g: Vec<String> = (0..12).map(|i| format!("g{}", i / 3)).collect();
        let y = vec![10.0, 11.0, 12.5, 14.0, 15.5, 14.5, 9.0, 8.0, 10.0, 12.0, 13.5, 12.0];
        let d = Data::new().with_factor("g", &g).unwrap().with_numeric("y", y.clone()).unwrap();
        let fit = mixed_model("y ~ 1", &[RandomSpec::new(&["g"])], &d, &MixedOptions::default()).unwrap();
        let means: Vec<f64> = (0..4).map(|k| y[3 * k..3 * k + 3].iter().sum::<f64>() / 3.0).collect();
        let gm = means.iter().sum::<f64>() / 4.0;
        let ssb: f64 = means.iter().map(|m| 3.0 * (m - gm) * (m - gm)).sum();
        let sse: f64 = (0..12).map(|i| (y[i] - means[i / 3]).powi(2)).sum();
        let (msb, mse) = (ssb / 3.0, sse / 8.0);
        assert!((fit.variances[1].variance - mse).abs() < 1e-5, "{:?}", fit.variances);
        assert!((fit.variances[0].variance - (msb - mse) / 3.0).abs() < 1e-4, "{:?}", fit.variances);
        assert!((fit.fixed[0].estimate - gm).abs() < 1e-8);
    }
}
