//! Linear and quadratic discriminant analysis with canonical discriminant functions.

use super::common::{eig_desc, fix_signs, levels_of, listwise, names_or_default};
use crate::error::{invalid, Result, StatsError};
use openstats_math::linalg::Cholesky;
use openstats_math::Matrix;

/// Prior probabilities of the groups.
#[derive(Debug, Clone, PartialEq)]
pub enum Priors {
    /// 1 / g for every group.
    Equal,
    /// The group's share of the observations.
    Proportional,
    /// One positive value per group in level order (normalised to sum to 1).
    Given(Vec<f64>),
}

/// Options of [`discriminant`].
#[derive(Debug, Clone)]
pub struct DiscriminantOptions {
    /// Separate covariance matrices per group (quadratic) instead of a pooled one (linear).
    pub quadratic: bool,
    pub priors: Priors,
    /// Leave-one-out cross-validation.
    pub cross_validate: bool,
}

impl Default for DiscriminantOptions {
    fn default() -> Self {
        DiscriminantOptions { quadratic: false, priors: Priors::Equal, cross_validate: true }
    }
}

/// Canonical discriminant functions (linear analysis only).
#[derive(Debug, Clone)]
pub struct CanonicalFunctions {
    /// Eigenvalues of W⁻¹B (between over within), descending; min(g − 1, p) of them.
    pub eigenvalues: Vec<f64>,
    pub proportion: Vec<f64>,
    pub cumulative: Vec<f64>,
    /// p × k coefficients scaled to unit pooled within-group variance.
    pub coefficients: Matrix,
    /// n × k scores of the observations used (centred at the grand mean).
    pub scores: Matrix,
    /// g × k group centroids in canonical space.
    pub centroids: Matrix,
}

/// Result of [`discriminant`].
#[derive(Debug, Clone)]
pub struct Discriminant {
    pub variables: Vec<String>,
    pub groups: Vec<String>,
    pub counts: Vec<usize>,
    pub priors: Vec<f64>,
    pub n: usize,
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    pub quadratic: bool,
    /// g × p group means.
    pub means: Matrix,
    pub pooled_covariance: Matrix,
    pub group_covariances: Vec<Matrix>,
    /// g × p linear discriminant function coefficients (`Σ⁻¹μ`) and constants (`−½μ'Σ⁻¹μ + ln π`); linear only.
    pub coefficients: Matrix,
    pub constants: Vec<f64>,
    /// Squared Mahalanobis distances between group means (pooled covariance).
    pub squared_distances: Matrix,
    /// Group index of each row used and the resubstitution classification.
    pub true_group: Vec<usize>,
    pub predicted: Vec<usize>,
    /// n × g posterior probabilities (resubstitution).
    pub posterior: Matrix,
    /// g × g counts, true group in rows, predicted in columns.
    pub confusion: Vec<Vec<usize>>,
    pub loo_predicted: Option<Vec<usize>>,
    pub loo_posterior: Option<Matrix>,
    pub loo_confusion: Option<Vec<Vec<usize>>>,
    pub canonical: Option<CanonicalFunctions>,
    overall_mean: Vec<f64>,
}

impl Discriminant {
    /// Proportion of resubstitution-correct classifications.
    pub fn accuracy(&self) -> f64 {
        accuracy(&self.confusion)
    }

    pub fn loo_accuracy(&self) -> Option<f64> {
        self.loo_confusion.as_ref().map(|c| accuracy(c))
    }

    /// Classify one observation: (group index, posterior probabilities).
    pub fn classify(&self, x: &[f64]) -> Result<(usize, Vec<f64>)> {
        let p = self.variables.len();
        if x.len() != p {
            return invalid("the observation has the wrong number of variables");
        }
        let model =
            Model::build(&self.means, &self.pooled_covariance, &self.group_covariances, &self.priors, self.quadratic)?;
        Ok(model.classify(x))
    }

    /// Mean of the variables over all observations used.
    pub fn grand_mean(&self) -> &[f64] {
        &self.overall_mean
    }
}

fn accuracy(c: &[Vec<usize>]) -> f64 {
    let total: usize = c.iter().flatten().sum();
    let ok: usize = (0..c.len()).map(|i| c[i][i]).sum();
    if total == 0 {
        0.0
    } else {
        ok as f64 / total as f64
    }
}

struct Model {
    means: Vec<Vec<f64>>,
    log_priors: Vec<f64>,
    kind: Kind,
}

enum Kind {
    Linear { coef: Vec<Vec<f64>>, constant: Vec<f64> },
    Quadratic { chol: Vec<Cholesky>, log_det: Vec<f64> },
}

impl Model {
    fn build(means: &Matrix, pooled: &Matrix, covs: &[Matrix], priors: &[f64], quadratic: bool) -> Result<Model> {
        let g = means.rows();
        let mu: Vec<Vec<f64>> = (0..g).map(|k| means.row(k).to_vec()).collect();
        let log_priors: Vec<f64> = priors.iter().map(|p| p.ln()).collect();
        if quadratic {
            let mut chol = Vec::with_capacity(g);
            let mut log_det = Vec::with_capacity(g);
            for (k, c) in covs.iter().enumerate() {
                let ch = Cholesky::new(c).map_err(|_| singular(k))?;
                log_det.push(ch.log_det());
                chol.push(ch);
            }
            Ok(Model { means: mu, log_priors, kind: Kind::Quadratic { chol, log_det } })
        } else {
            let ch = Cholesky::new(pooled).map_err(|_| {
                StatsError::InvalidArgument(
                    "the pooled covariance matrix is singular (too few observations, or a variable is a linear combination of others)"
                        .into(),
                )
            })?;
            let mut coef = Vec::with_capacity(g);
            let mut constant = Vec::with_capacity(g);
            for k in 0..g {
                let a = ch.solve(&mu[k])?;
                let c = -0.5 * a.iter().zip(&mu[k]).map(|(x, y)| x * y).sum::<f64>() + log_priors[k];
                coef.push(a);
                constant.push(c);
            }
            Ok(Model { means: mu, log_priors, kind: Kind::Linear { coef, constant } })
        }
    }

    fn scores(&self, x: &[f64]) -> Vec<f64> {
        let g = self.means.len();
        match &self.kind {
            Kind::Linear { coef, constant } => {
                (0..g).map(|k| coef[k].iter().zip(x).map(|(a, b)| a * b).sum::<f64>() + constant[k]).collect()
            }
            Kind::Quadratic { chol, log_det } => (0..g)
                .map(|k| {
                    let d: Vec<f64> = x.iter().zip(&self.means[k]).map(|(a, b)| a - b).collect();
                    let sol = chol[k].solve(&d).unwrap_or_else(|_| vec![f64::NAN; d.len()]);
                    let q: f64 = d.iter().zip(&sol).map(|(a, b)| a * b).sum();
                    -0.5 * log_det[k] - 0.5 * q + self.log_priors[k]
                })
                .collect(),
        }
    }

    fn classify(&self, x: &[f64]) -> (usize, Vec<f64>) {
        let s = self.scores(x);
        let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
        let tot: f64 = e.iter().sum();
        let post: Vec<f64> = e.iter().map(|v| v / tot).collect();
        let mut best = 0;
        for k in 1..s.len() {
            if s[k] > s[best] {
                best = k;
            }
        }
        (best, post)
    }
}

fn singular(k: usize) -> StatsError {
    StatsError::InvalidArgument(format!(
        "the covariance matrix of group {} is singular; use the linear (pooled) analysis or fewer variables",
        k + 1
    ))
}

/// Discriminant analysis. `groups` holds one label per row (empty string = missing).
pub fn discriminant(
    cols: &[Vec<f64>],
    names: &[String],
    groups: &[String],
    opts: &DiscriminantOptions,
) -> Result<Discriminant> {
    let p = cols.len();
    if p == 0 {
        return invalid("no predictors were given");
    }
    if groups.len() != cols[0].len() {
        return invalid("the group column and the predictors have different lengths");
    }
    let c = listwise(cols)?;
    let (level_names, idx) = levels_of(groups);
    let mut rows = Vec::new();
    let mut grp = Vec::new();
    for (pos, &r) in c.rows.iter().enumerate() {
        if let Some(k) = idx[r] {
            rows.push(pos);
            grp.push(k);
        }
    }
    let n = rows.len();
    // Drop levels that disappeared after listwise deletion.
    let mut used: Vec<usize> = (0..level_names.len()).filter(|k| grp.contains(k)).collect();
    used.sort();
    let g = used.len();
    if g < 2 {
        return invalid("discriminant analysis needs at least two groups with data");
    }
    let remap: Vec<usize> = (0..level_names.len()).map(|k| used.iter().position(|&u| u == k).unwrap_or(0)).collect();
    let grp: Vec<usize> = grp.iter().map(|&k| remap[k]).collect();
    let level_names: Vec<String> = used.iter().map(|&k| level_names[k].clone()).collect();
    if n <= g {
        return Err(StatsError::InsufficientData { what: "discriminant analysis", needed: g + 1, got: n });
    }
    let x: Vec<Vec<f64>> = rows.iter().map(|&r| (0..p).map(|v| c.cols[v][r]).collect()).collect();
    let rows_used: Vec<usize> = rows.iter().map(|&r| c.rows[r]).collect();

    let mut counts = vec![0usize; g];
    for &k in &grp {
        counts[k] += 1;
    }
    let mut means = Matrix::zeros(g, p);
    for (i, &k) in grp.iter().enumerate() {
        for v in 0..p {
            means[(k, v)] += x[i][v];
        }
    }
    for k in 0..g {
        for v in 0..p {
            means[(k, v)] /= counts[k] as f64;
        }
    }
    let mut scatter: Vec<Matrix> = (0..g).map(|_| Matrix::zeros(p, p)).collect();
    for (i, &k) in grp.iter().enumerate() {
        for a in 0..p {
            let da = x[i][a] - means[(k, a)];
            for b in a..p {
                let v = da * (x[i][b] - means[(k, b)]);
                scatter[k][(a, b)] += v;
            }
        }
    }
    for s in scatter.iter_mut() {
        for a in 0..p {
            for b in 0..a {
                s[(a, b)] = s[(b, a)];
            }
        }
    }
    let mut within = Matrix::zeros(p, p);
    for s in &scatter {
        within = within.try_add(s)?;
    }
    let dfw = (n - g) as f64;
    let pooled = within.scale(1.0 / dfw);
    let group_covs: Vec<Matrix> = (0..g)
        .map(|k| if counts[k] > 1 { scatter[k].scale(1.0 / (counts[k] as f64 - 1.0)) } else { Matrix::zeros(p, p) })
        .collect();
    if opts.quadratic {
        for (k, &ck) in counts.iter().enumerate() {
            if ck < 2 {
                return invalid(format!(
                    "group '{}' has one observation; a quadratic analysis needs more",
                    level_names[k]
                ));
            }
        }
    }

    let priors = match &opts.priors {
        Priors::Equal => vec![1.0 / g as f64; g],
        Priors::Proportional => counts.iter().map(|&c| c as f64 / n as f64).collect(),
        Priors::Given(v) => {
            if v.len() != g || v.iter().any(|&q| !(q > 0.0) || !q.is_finite()) {
                return invalid(format!("give {g} positive prior probabilities, one per group"));
            }
            let t: f64 = v.iter().sum();
            v.iter().map(|q| q / t).collect()
        }
    };

    let model = Model::build(&means, &pooled, &group_covs, &priors, opts.quadratic)?;
    let mut predicted = Vec::with_capacity(n);
    let mut posterior = Matrix::zeros(n, g);
    for (i, xi) in x.iter().enumerate() {
        let (k, post) = model.classify(xi);
        predicted.push(k);
        for (j, q) in post.iter().enumerate() {
            posterior[(i, j)] = *q;
        }
    }
    let confusion = tabulate(&grp, &predicted, g);

    let (coefficients, constants) = match &model.kind {
        Kind::Linear { coef, constant } => (Matrix::from_fn(g, p, |k, v| coef[k][v]), constant.clone()),
        Kind::Quadratic { .. } => (Matrix::zeros(g, p), vec![0.0; g]),
    };

    // Squared Mahalanobis distances between group means (pooled covariance).
    let squared_distances = match Cholesky::new(&pooled) {
        Ok(ch) => {
            let mut d = Matrix::zeros(g, g);
            for a in 0..g {
                for b in 0..g {
                    let diff: Vec<f64> = (0..p).map(|v| means[(a, v)] - means[(b, v)]).collect();
                    let sol = ch.solve(&diff)?;
                    d[(a, b)] = diff.iter().zip(&sol).map(|(u, w)| u * w).sum();
                }
            }
            d
        }
        Err(_) => Matrix::filled(g, g, f64::NAN),
    };

    let (loo_predicted, loo_posterior, loo_confusion) = if opts.cross_validate {
        let (pr, po) = leave_one_out(&x, &grp, &counts, &means, &scatter, &priors, opts.quadratic, n, g, p)?;
        let cf = tabulate(&grp, &pr, g);
        (Some(pr), Some(po), Some(cf))
    } else {
        (None, None, None)
    };

    let overall_mean: Vec<f64> = (0..p).map(|v| x.iter().map(|r| r[v]).sum::<f64>() / n as f64).collect();
    let canonical = if opts.quadratic {
        None
    } else {
        canonical_functions(&x, &grp, &counts, &means, &within, &overall_mean, g, p, n).ok()
    };

    Ok(Discriminant {
        variables: names_or_default(names, p, "C"),
        groups: level_names,
        counts,
        priors,
        n,
        n_dropped: groups.len() - n,
        rows_used,
        quadratic: opts.quadratic,
        means,
        pooled_covariance: pooled,
        group_covariances: group_covs,
        coefficients,
        constants,
        squared_distances,
        true_group: grp,
        predicted,
        posterior,
        confusion,
        loo_predicted,
        loo_posterior,
        loo_confusion,
        canonical,
        overall_mean,
    })
}

fn tabulate(truth: &[usize], pred: &[usize], g: usize) -> Vec<Vec<usize>> {
    let mut m = vec![vec![0usize; g]; g];
    for (&t, &q) in truth.iter().zip(pred) {
        m[t][q] += 1;
    }
    m
}

#[allow(clippy::too_many_arguments)]
fn leave_one_out(
    x: &[Vec<f64>],
    grp: &[usize],
    counts: &[usize],
    means: &Matrix,
    scatter: &[Matrix],
    priors: &[f64],
    quadratic: bool,
    n: usize,
    g: usize,
    p: usize,
) -> Result<(Vec<usize>, Matrix)> {
    let mut pred = Vec::with_capacity(n);
    let mut post = Matrix::zeros(n, g);
    let mut total_scatter = Matrix::zeros(p, p);
    for s in scatter {
        total_scatter = total_scatter.try_add(s)?;
    }
    for i in 0..n {
        let k = grp[i];
        let nk = counts[k] as f64;
        if counts[k] < 2 {
            return invalid("leave-one-out needs at least two observations in every group");
        }
        let d: Vec<f64> = (0..p).map(|v| x[i][v] - means[(k, v)]).collect();
        let f = nk / (nk - 1.0);
        let mut means_i = means.clone();
        for v in 0..p {
            means_i[(k, v)] = (nk * means[(k, v)] - x[i][v]) / (nk - 1.0);
        }
        let mut sk = scatter[k].clone();
        for a in 0..p {
            for b in 0..p {
                sk[(a, b)] -= f * d[a] * d[b];
            }
        }
        let (pooled_i, covs_i): (Matrix, Vec<Matrix>) = if quadratic {
            let covs = (0..g)
                .map(|j| {
                    if j == k {
                        sk.scale(1.0 / (nk - 2.0).max(1.0))
                    } else {
                        scatter[j].scale(1.0 / (counts[j] as f64 - 1.0))
                    }
                })
                .collect();
            (Matrix::zeros(p, p), covs)
        } else {
            let mut tot = total_scatter.clone();
            for a in 0..p {
                for b in 0..p {
                    tot[(a, b)] -= f * d[a] * d[b];
                }
            }
            (tot.scale(1.0 / (n as f64 - 1.0 - g as f64)), Vec::new())
        };
        if quadratic && counts[k] < 3 {
            return invalid("leave-one-out for the quadratic analysis needs at least three observations per group");
        }
        let model = Model::build(&means_i, &pooled_i, &covs_i, priors, quadratic).map_err(|e| {
            StatsError::InvalidArgument(format!("leave-one-out failed after removing observation {}: {e}", i + 1))
        })?;
        let (best, pr) = model.classify(&x[i]);
        pred.push(best);
        for (j, q) in pr.iter().enumerate() {
            post[(i, j)] = *q;
        }
    }
    Ok((pred, post))
}

#[allow(clippy::too_many_arguments)]
fn canonical_functions(
    x: &[Vec<f64>],
    grp: &[usize],
    counts: &[usize],
    means: &Matrix,
    within: &Matrix,
    overall: &[f64],
    g: usize,
    p: usize,
    n: usize,
) -> Result<CanonicalFunctions> {
    let k = (g - 1).min(p);
    let mut between = Matrix::zeros(p, p);
    for j in 0..g {
        for a in 0..p {
            for b in 0..p {
                between[(a, b)] += counts[j] as f64 * (means[(j, a)] - overall[a]) * (means[(j, b)] - overall[b]);
            }
        }
    }
    let ch = Cholesky::new(within)?;
    let l = ch.l().clone();
    // M = L⁻¹ B L⁻ᵀ
    let linv_b = solve_lower(&l, &between)?;
    let m = solve_lower(&l, &linv_b.t())?.symmetrize();
    let (vals, vecs) = eig_desc(&m)?;
    let scale = ((n - g) as f64).sqrt();
    let mut coef = Matrix::zeros(p, k);
    for j in 0..k {
        // a = L⁻ᵀ v
        let v = vecs.col(j);
        let a = solve_upper_t(&l, &v)?;
        for i in 0..p {
            coef[(i, j)] = a[i] * scale;
        }
    }
    fix_signs(&mut coef);
    let eigenvalues: Vec<f64> = vals.iter().take(k).map(|v| v.max(0.0)).collect();
    let total: f64 = eigenvalues.iter().sum();
    let proportion: Vec<f64> = eigenvalues.iter().map(|v| if total > 0.0 { v / total } else { 0.0 }).collect();
    let mut cumulative = Vec::with_capacity(k);
    let mut acc = 0.0;
    for q in &proportion {
        acc += q;
        cumulative.push(acc);
    }
    let scores = Matrix::from_fn(x.len(), k, |i, j| (0..p).map(|v| (x[i][v] - overall[v]) * coef[(v, j)]).sum());
    let centroids = Matrix::from_fn(g, k, |gi, j| (0..p).map(|v| (means[(gi, v)] - overall[v]) * coef[(v, j)]).sum());
    let _ = grp;
    Ok(CanonicalFunctions { eigenvalues, proportion, cumulative, coefficients: coef, scores, centroids })
}

/// Solve L X = B for lower-triangular L.
fn solve_lower(l: &Matrix, b: &Matrix) -> Result<Matrix> {
    let n = l.rows();
    let mut x = b.clone();
    for c in 0..b.cols() {
        for i in 0..n {
            let mut v = x[(i, c)];
            for j in 0..i {
                v -= l[(i, j)] * x[(j, c)];
            }
            x[(i, c)] = v / l[(i, i)];
        }
    }
    Ok(x)
}

/// Solve Lᵀ x = v.
fn solve_upper_t(l: &Matrix, v: &[f64]) -> Result<Vec<f64>> {
    let n = l.rows();
    let mut x = v.to_vec();
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= l[(j, i)] * x[j];
        }
        x[i] = s / l[(i, i)];
    }
    Ok(x)
}
