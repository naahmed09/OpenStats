//! MARS beyond the least-squares case: a logistic-link model for a two-class response, and
//! cross-validated comparison of MARS settings.
//!
//! **Binary MARS** follows the `earth(glm = binomial)` recipe: the basis functions are chosen by
//! the ordinary MARS forward and backward passes on the 0/1 response (least squares, GCV), then
//! the coefficients of the chosen basis are re-estimated by maximum likelihood for the logistic
//! model `logit P(event) = Σ βⱼ Bⱼ(x)` with iteratively reweighted least squares.
//!
//! **Cross-validation** refits the whole model (forward pass, pruning, coefficients) on each
//! training fold and scores the held-out fold, for every combination of interaction degree and
//! GCV penalty requested; the pooled held-out error picks the setting.

use super::mars::{fit_mars, MarsModel, MarsOptions, Term};
use super::{auc, class_fit, rng_for, shuffle, ClassFit};
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ContinuousDistribution, Normal};
use openstats_math::linalg::Svd;
use openstats_math::Matrix;

/// One coefficient of the logistic fit.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitCoefficient {
    /// Basis function text (`Intercept` for the first).
    pub label: String,
    /// Maximum-likelihood coefficient.
    pub coef: f64,
    /// Standard error (`NaN` when the information matrix is singular).
    pub se: f64,
    /// Wald z.
    pub z: f64,
    /// Two-sided p-value.
    pub p: f64,
}

/// A fitted binary MARS model.
#[derive(Debug, Clone)]
pub struct MarsBinaryModel {
    /// The least-squares MARS model that chose the basis (fitted to the 0/1 response).
    pub basis_model: MarsModel,
    /// The two class labels; the event is `levels[1]`.
    pub levels: Vec<String>,
    /// The basis functions with their logistic coefficients.
    pub terms: Vec<Term>,
    /// Coefficient table.
    pub coefficients: Vec<LogitCoefficient>,
    /// Residual deviance.
    pub deviance: f64,
    /// Deviance of the intercept-only model.
    pub null_deviance: f64,
    /// Maximised log-likelihood.
    pub log_lik: f64,
    /// Akaike information criterion `−2 ℓ + 2 M`.
    pub aic: f64,
    /// IRLS iterations used.
    pub iterations: usize,
    /// Whether IRLS converged.
    pub converged: bool,
    /// Observations.
    pub n: usize,
    /// Class codes (0 or 1) of the observations.
    pub codes: Vec<usize>,
    /// Fitted event probabilities.
    pub probabilities: Vec<f64>,
    /// Classification of the training data at the 0.5 cutoff.
    pub fit: ClassFit,
}

impl MarsBinaryModel {
    /// McFadden's pseudo R²: `1 − deviance / null deviance`.
    pub fn pseudo_r2(&self) -> f64 {
        1.0 - self.deviance / self.null_deviance
    }

    /// Event probabilities for observations given as expanded predictor columns.
    pub fn predict_probability(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        if x.len() != self.basis_model.names.len() {
            return invalid(format!("expected {} predictor columns, got {}", self.basis_model.names.len(), x.len()));
        }
        let n = x.first().map_or(0, Vec::len);
        if x.iter().any(|c| c.len() != n) {
            return invalid("predictor columns have different lengths");
        }
        let mut row = vec![0.0; x.len()];
        Ok((0..n)
            .map(|i| {
                for (v, c) in x.iter().enumerate() {
                    row[v] = c[i];
                }
                let eta: f64 = self.terms.iter().map(|t| t.coef * t.eval(&row)).sum();
                1.0 / (1.0 + (-eta).exp())
            })
            .collect())
    }
}

fn clamp_p(p: f64) -> f64 {
    p.clamp(1e-12, 1.0 - 1e-12)
}

fn deviance_of(y: &[f64], mu: &[f64]) -> f64 {
    -2.0 * y.iter().zip(mu).map(|(&y, &m)| y * clamp_p(m).ln() + (1.0 - y) * (1.0 - clamp_p(m)).ln()).sum::<f64>()
}

/// Coefficients, standard errors, deviance, iterations and convergence flag of a logistic IRLS fit.
pub(crate) type IrlsOut = (Vec<f64>, Vec<f64>, f64, usize, bool);

/// Maximum-likelihood logistic coefficients for the columns of `b` (n × m, row-major by column
/// vectors `cols[j][i]`). Returns coefficients, standard errors, deviance, iterations, convergence.
pub(crate) fn irls_logit(cols: &[Vec<f64>], y: &[f64]) -> Result<IrlsOut> {
    let m = cols.len();
    let n = y.len();
    let ybar = y.iter().sum::<f64>() / n as f64;
    let mut beta = vec![0.0; m];
    // Start from the intercept-only logit when the first column is constant 1.
    if m > 0 && cols[0].iter().all(|&v| v == 1.0) {
        beta[0] = (clamp_p(ybar) / (1.0 - clamp_p(ybar))).ln();
    }
    let eta_of =
        |beta: &[f64]| -> Vec<f64> { (0..n).map(|i| (0..m).map(|j| beta[j] * cols[j][i]).sum::<f64>()).collect() };
    let mu_of = |eta: &[f64]| -> Vec<f64> { eta.iter().map(|&e| 1.0 / (1.0 + (-e).exp())).collect() };
    let mut eta = eta_of(&beta);
    let mut mu = mu_of(&eta);
    let mut dev = deviance_of(y, &mu);
    let mut converged = false;
    let mut iters = 0;
    for it in 1..=100 {
        iters = it;
        let w: Vec<f64> = mu.iter().map(|&p| (clamp_p(p) * (1.0 - clamp_p(p))).max(1e-10)).collect();
        let z: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / w[i]).collect();
        let mut a = Matrix::zeros(n, m);
        let mut zb = vec![0.0; n];
        for i in 0..n {
            let s = w[i].sqrt();
            for j in 0..m {
                a[(i, j)] = s * cols[j][i];
            }
            zb[i] = s * z[i];
        }
        let svd = Svd::new(&a)?;
        let target = svd.solve_least_squares(&zb, None)?;
        // Step halving keeps the deviance from increasing (separation, poor starts).
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let cand: Vec<f64> = beta.iter().zip(&target).map(|(b, t)| b + step * (t - b)).collect();
            let e = eta_of(&cand);
            let mu_c = mu_of(&e);
            let d = deviance_of(y, &mu_c);
            if d.is_finite() && d <= dev + 1e-12 * (1.0 + dev.abs()) {
                accepted = Some((cand, e, mu_c, d));
                break;
            }
            step *= 0.5;
        }
        let Some((cand, e, mu_c, d)) = accepted else {
            converged = true; // no improving step: at the optimum to working precision
            break;
        };
        let change = (dev - d).abs() / (d.abs() + 0.1);
        beta = cand;
        eta = e;
        mu = mu_c;
        dev = d;
        if change < 1e-10 {
            converged = true;
            break;
        }
    }
    // Covariance (XᵀWX)⁺ at the solution.
    let w: Vec<f64> = mu.iter().map(|&p| (clamp_p(p) * (1.0 - clamp_p(p))).max(1e-10)).collect();
    let mut a = Matrix::zeros(n, m);
    for i in 0..n {
        let s = w[i].sqrt();
        for j in 0..m {
            a[(i, j)] = s * cols[j][i];
        }
    }
    let svd = Svd::new(&a)?;
    let pinv = svd.pinv(None);
    let se: Vec<f64> = (0..m)
        .map(|j| {
            let v: f64 = (0..n).map(|i| pinv[(j, i)] * pinv[(j, i)]).sum();
            v.sqrt()
        })
        .collect();
    Ok((beta, se, dev, iters, converged))
}

/// Maximum-likelihood logistic fit on a given basis.
#[derive(Debug, Clone)]
pub struct LogitBasisFit {
    /// Coefficients, one per basis function.
    pub coef: Vec<f64>,
    /// Standard errors from the pseudo-inverse of the information matrix.
    pub se: Vec<f64>,
    /// Residual deviance.
    pub deviance: f64,
    /// Deviance of the intercept-only model.
    pub null_deviance: f64,
    /// Fitted event probabilities.
    pub probabilities: Vec<f64>,
    /// IRLS iterations.
    pub iterations: usize,
    /// Whether IRLS converged.
    pub converged: bool,
}

/// Fit the logistic model `logit P(event) = Σ coefⱼ Bⱼ(x)` for the basis functions `basis` by
/// iteratively reweighted least squares. `codes` are 0/1 (1 = event); `x` are the predictor columns.
pub fn logistic_on_basis(basis: &[Term], x: &[Vec<f64>], codes: &[usize]) -> Result<LogitBasisFit> {
    let n = codes.len();
    if basis.is_empty() {
        return invalid("the basis is empty");
    }
    if x.iter().any(|c| c.len() != n) {
        return invalid("predictor columns do not match the response");
    }
    if codes.iter().any(|&c| c > 1) {
        return invalid("class codes must be 0 or 1");
    }
    let ones = codes.iter().filter(|&&c| c == 1).count();
    if ones == 0 || ones == n {
        return invalid("the response has only one class");
    }
    let y: Vec<f64> = codes.iter().map(|&c| c as f64).collect();
    let mut row = vec![0.0; x.len()];
    let mut cols = vec![vec![0.0; n]; basis.len()];
    for i in 0..n {
        for (v, c) in x.iter().enumerate() {
            row[v] = c[i];
        }
        for (j, t) in basis.iter().enumerate() {
            cols[j][i] = t.eval(&row);
        }
    }
    if cols.iter().flatten().any(|v| !v.is_finite()) {
        return invalid("the basis has non-finite values (missing predictors?)");
    }
    let (coef, se, deviance, iterations, converged) = irls_logit(&cols, &y)?;
    let ybar = ones as f64 / n as f64;
    let null_deviance = -2.0 * (ones as f64 * ybar.ln() + (n - ones) as f64 * (1.0 - ybar).ln());
    let probabilities = (0..n)
        .map(|i| {
            let eta: f64 = (0..basis.len()).map(|j| coef[j] * cols[j][i]).sum();
            1.0 / (1.0 + (-eta).exp())
        })
        .collect();
    Ok(LogitBasisFit { coef, se, deviance, null_deviance, probabilities, iterations, converged })
}

/// Fit a binary MARS model. `codes` are 0/1 (1 = event, the second of `levels`).
pub fn fit_mars_binary(
    x: &[Vec<f64>],
    names: &[String],
    codes: &[usize],
    levels: &[String],
    opts: &MarsOptions,
) -> Result<MarsBinaryModel> {
    if levels.len() != 2 {
        return invalid("binary MARS needs exactly two classes");
    }
    if codes.iter().any(|&c| c > 1) {
        return invalid("class codes must be 0 or 1");
    }
    let ones = codes.iter().filter(|&&c| c == 1).count();
    if ones == 0 || ones == codes.len() {
        return invalid("the response has only one class");
    }
    let y: Vec<f64> = codes.iter().map(|&c| c as f64).collect();
    let basis_model = fit_mars(x, names, &y, opts)?;
    let n = y.len();
    let m = basis_model.terms.len();
    let lf = logistic_on_basis(&basis_model.terms, x, codes)?;
    let mut terms = basis_model.terms.clone();
    for (t, b) in terms.iter_mut().zip(&lf.coef) {
        t.coef = *b;
    }
    let coefficients = terms
        .iter()
        .enumerate()
        .map(|(j, t)| {
            let z = lf.coef[j] / lf.se[j];
            LogitCoefficient {
                label: if j == 0 { "Intercept".to_string() } else { basis_model.term_label(t) },
                coef: lf.coef[j],
                se: lf.se[j],
                z,
                p: if z.is_finite() { 2.0 * Normal { mu: 0.0, sigma: 1.0 }.sf(z.abs()) } else { f64::NAN },
            }
        })
        .collect();
    let probs2: Vec<Vec<f64>> = lf.probabilities.iter().map(|&p| vec![1.0 - p, p]).collect();
    let fit = class_fit(codes, &probs2, levels);
    Ok(MarsBinaryModel {
        basis_model,
        levels: levels.to_vec(),
        terms,
        coefficients,
        deviance: lf.deviance,
        null_deviance: lf.null_deviance,
        log_lik: -lf.deviance / 2.0,
        aic: lf.deviance + 2.0 * m as f64,
        iterations: lf.iterations,
        converged: lf.converged,
        n,
        codes: codes.to_vec(),
        probabilities: lf.probabilities,
        fit,
    })
}

// ------------------------------------------------------------------ cross-validation

/// Held-out performance of one MARS setting.
#[derive(Debug, Clone, PartialEq)]
pub struct MarsCvRow {
    /// Highest interaction degree.
    pub degree: usize,
    /// GCV penalty used.
    pub penalty: f64,
    /// Mean number of terms of the fold models.
    pub mean_terms: f64,
    /// Pooled held-out root mean squared error (numeric response) or `NaN`.
    pub rmse: f64,
    /// Pooled held-out R² (numeric response) or `NaN`.
    pub r2: f64,
    /// Held-out mean log-loss (binary response) or `NaN`.
    pub log_loss: f64,
    /// Held-out misclassification rate (binary response) or `NaN`.
    pub error_rate: f64,
    /// Held-out area under the ROC curve (binary response) or `NaN`.
    pub auc: f64,
}

/// Result of [`mars_cross_validate`].
#[derive(Debug, Clone, PartialEq)]
pub struct MarsCv {
    /// One row per (degree, penalty) combination, degrees varying slowest.
    pub rows: Vec<MarsCvRow>,
    /// Index of the best row (smallest held-out RMSE, or log-loss for a binary response).
    pub best: usize,
    /// Folds used.
    pub folds: usize,
    /// Whether the response was binary.
    pub binary: bool,
}

/// Fold of every observation: a seeded shuffle dealt round-robin (fold sizes differ by at most one).
pub fn random_folds(n: usize, folds: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    shuffle(&mut idx, &mut rng_for(seed, 0x6d61_7273));
    let mut fold = vec![0; n];
    for (k, &i) in idx.iter().enumerate() {
        fold[i] = k % folds;
    }
    fold
}

/// Cross-validate MARS over `degrees` × `penalties`. For a numeric response pass `y`; for a binary
/// one pass `codes` (0/1) and `levels`. `opts.max_degree` and `opts.penalty` are overridden by the
/// candidates.
pub fn mars_cross_validate(
    x: &[Vec<f64>],
    names: &[String],
    response: CvResponse<'_>,
    opts: &MarsOptions,
    degrees: &[usize],
    penalties: &[f64],
    folds: usize,
    seed: u64,
) -> Result<MarsCv> {
    let n = match response {
        CvResponse::Numeric(y) => y.len(),
        CvResponse::Binary { codes, .. } => codes.len(),
    };
    if folds < 2 || folds > n {
        return invalid("the number of folds must be between 2 and the number of observations");
    }
    if degrees.is_empty() || penalties.is_empty() {
        return invalid("give at least one degree and one penalty");
    }
    if penalties.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
        return invalid("penalties must be non-negative numbers");
    }
    let fold = random_folds(n, folds, seed);
    let take = |rows: &[usize]| -> Vec<Vec<f64>> { x.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect() };
    let mut rows_out = Vec::new();
    for &degree in degrees {
        for &penalty in penalties {
            let o = MarsOptions { max_degree: degree, penalty: Some(penalty), ..opts.clone() };
            let mut pred = vec![f64::NAN; n];
            let mut terms = 0.0;
            // folds are independent: fit them in parallel, merge in fold order
            let fits = crate::parallel::par_map(folds, |f| -> Result<(Vec<usize>, f64, Vec<f64>)> {
                let train: Vec<usize> = (0..n).filter(|&i| fold[i] != f).collect();
                let held: Vec<usize> = (0..n).filter(|&i| fold[i] == f).collect();
                let (xt, xh) = (take(&train), take(&held));
                match response {
                    CvResponse::Numeric(y) => {
                        let yt: Vec<f64> = train.iter().map(|&i| y[i]).collect();
                        let m = fit_mars(&xt, names, &yt, &o)?;
                        Ok((held, m.terms.len() as f64, m.predict(&xh)?))
                    }
                    CvResponse::Binary { codes, levels } => {
                        let ct: Vec<usize> = train.iter().map(|&i| codes[i]).collect();
                        let m = fit_mars_binary(&xt, names, &ct, levels, &o)?;
                        Ok((held, m.terms.len() as f64, m.predict_probability(&xh)?))
                    }
                }
            });
            for r in fits {
                let (held, t, preds) = r?;
                terms += t;
                for (&i, p) in held.iter().zip(preds) {
                    pred[i] = p;
                }
            }
            let mean_terms = terms / folds as f64;
            let row = match response {
                CvResponse::Numeric(y) => {
                    let mean = y.iter().sum::<f64>() / n as f64;
                    let sse: f64 = y.iter().zip(&pred).map(|(a, b)| (a - b).powi(2)).sum();
                    let sst: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
                    MarsCvRow {
                        degree,
                        penalty,
                        mean_terms,
                        rmse: (sse / n as f64).sqrt(),
                        r2: 1.0 - sse / sst,
                        log_loss: f64::NAN,
                        error_rate: f64::NAN,
                        auc: f64::NAN,
                    }
                }
                CvResponse::Binary { codes, .. } => {
                    let ll = -codes
                        .iter()
                        .zip(&pred)
                        .map(|(&c, &p)| if c == 1 { clamp_p(p).ln() } else { (1.0 - clamp_p(p)).ln() })
                        .sum::<f64>()
                        / n as f64;
                    let err =
                        codes.iter().zip(&pred).filter(|(&c, &p)| usize::from(p > 0.5) != c).count() as f64 / n as f64;
                    let ev: Vec<bool> = codes.iter().map(|&c| c == 1).collect();
                    MarsCvRow {
                        degree,
                        penalty,
                        mean_terms,
                        rmse: f64::NAN,
                        r2: f64::NAN,
                        log_loss: ll,
                        error_rate: err,
                        auc: auc(&pred, &ev).unwrap_or(f64::NAN),
                    }
                }
            };
            rows_out.push(row);
        }
    }
    let binary = matches!(response, CvResponse::Binary { .. });
    let key = |r: &MarsCvRow| if binary { r.log_loss } else { r.rmse };
    let best = (0..rows_out.len())
        .min_by(|&a, &b| key(&rows_out[a]).total_cmp(&key(&rows_out[b])))
        .ok_or(StatsError::InvalidArgument("nothing to compare".into()))?;
    Ok(MarsCv { rows: rows_out, best, folds, binary })
}

/// The response handed to [`mars_cross_validate`].
#[derive(Debug, Clone, Copy)]
pub enum CvResponse<'a> {
    /// A numeric response.
    Numeric(&'a [f64]),
    /// A two-class response coded 0/1 (1 = event).
    Binary {
        /// Class code of every observation.
        codes: &'a [usize],
        /// The two labels.
        levels: &'a [String],
    },
}
