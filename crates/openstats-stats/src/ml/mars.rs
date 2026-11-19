//! Multivariate adaptive regression splines (Friedman 1991).
//!
//! The forward pass greedily adds pairs of hinge functions `max(0, x − t)` / `max(0, t − x)`
//! multiplied into an existing basis function (up to a maximum degree of interaction), choosing
//! the variable, knot and parent that reduce the residual sum of squares most; candidate
//! residual reductions are computed by orthogonalisation against the current basis. The
//! backward pass deletes the least useful term one at a time and the subset with the smallest
//! generalised cross-validation (GCV) score is kept.

use crate::error::{invalid, Result, StatsError};
use openstats_math::linalg::{Cholesky, Svd};
use openstats_math::Matrix;

/// One factor of a basis function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Factor {
    /// Predictor index.
    pub var: usize,
    /// Knot (unused for a linear factor).
    pub knot: f64,
    /// `+1`: `max(0, x − knot)`, `−1`: `max(0, knot − x)`, `0`: `x`.
    pub sign: i8,
}

impl Factor {
    /// Value of the factor at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        match self.sign {
            1 => (x - self.knot).max(0.0),
            -1 => (self.knot - x).max(0.0),
            _ => x,
        }
    }
}

/// A basis function: a product of factors (empty = the intercept).
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// Factors multiplied together.
    pub factors: Vec<Factor>,
    /// Coefficient in the final model.
    pub coef: f64,
}

impl Term {
    /// Value at one observation (`x[v]` is the value of predictor `v`).
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.factors.iter().map(|f| f.eval(x[f.var])).product()
    }

    /// Degree of interaction.
    pub fn degree(&self) -> usize {
        self.factors.len()
    }
}

/// Fitting options.
#[derive(Debug, Clone, PartialEq)]
pub struct MarsOptions {
    /// Largest number of terms (including the intercept) the forward pass may create
    /// (`None`: `min(200, max(20, 2p) + 1)`).
    pub max_terms: Option<usize>,
    /// Highest degree of interaction (1 = additive).
    pub max_degree: usize,
    /// Cost per knot in the GCV (`None`: 2 for additive models, 3 with interactions).
    pub penalty: Option<f64>,
    /// Minimum number of observations between knots (`None`: Friedman's rule).
    pub min_span: Option<usize>,
    /// Observations excluded at each end before the first and after the last knot
    /// (`None`: Friedman's rule).
    pub end_span: Option<usize>,
    /// The forward pass stops when a step improves R² by less than this.
    pub threshold: f64,
    /// Also consider linear (hinge-free) terms.
    pub allow_linear: bool,
    /// Run the backward pass (otherwise keep the forward model).
    pub prune: bool,
}

impl Default for MarsOptions {
    fn default() -> Self {
        MarsOptions {
            max_terms: None,
            max_degree: 1,
            penalty: None,
            min_span: None,
            end_span: None,
            threshold: 1e-3,
            allow_linear: true,
            prune: true,
        }
    }
}

/// One size of the backward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneRow {
    /// Terms in the model (including the intercept).
    pub terms: usize,
    /// Residual sum of squares.
    pub rss: f64,
    /// Generalised cross-validation score.
    pub gcv: f64,
    /// `true` for the chosen model.
    pub selected: bool,
}

/// A fitted MARS model.
#[derive(Debug, Clone)]
pub struct MarsModel {
    /// Predictor names.
    pub names: Vec<String>,
    /// Terms; the first is the intercept.
    pub terms: Vec<Term>,
    /// Observations.
    pub n: usize,
    /// Residual sum of squares.
    pub rss: f64,
    /// Total sum of squares.
    pub tss: f64,
    /// GCV of the chosen model.
    pub gcv: f64,
    /// GCV of the intercept-only model.
    pub gcv_null: f64,
    /// Penalty used.
    pub penalty: f64,
    /// Terms created by the forward pass.
    pub forward_terms: usize,
    /// Backward-pass path (empty when `prune` is off).
    pub path: Vec<PruneRow>,
    /// Variable importance, scaled so the largest is 100: `(predictor, GCV importance, RSS importance)`.
    pub importance: Vec<(usize, f64, f64)>,
    /// Observed response.
    pub y: Vec<f64>,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Minimum span and end span used.
    pub spans: (usize, usize),
}

impl MarsModel {
    /// R².
    pub fn r2(&self) -> f64 {
        1.0 - self.rss / self.tss
    }

    /// Generalised R²: `1 − GCV/GCV(intercept only)`.
    pub fn grsq(&self) -> f64 {
        1.0 - self.gcv / self.gcv_null
    }

    /// Root mean squared error.
    pub fn rmse(&self) -> f64 {
        (self.rss / self.n as f64).sqrt()
    }

    /// Predictions for observations given as predictor columns (`NaN` if a used predictor is missing).
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        if x.len() != self.names.len() {
            return invalid(format!("expected {} predictor columns, got {}", self.names.len(), x.len()));
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
                self.terms.iter().map(|t| t.coef * t.eval(&row)).sum()
            })
            .collect())
    }

    /// Text of a term, e.g. `max(0, x1 - 2.5) * max(0, 4 - x2)`.
    pub fn term_label(&self, t: &Term) -> String {
        if t.factors.is_empty() {
            return "1".into();
        }
        t.factors
            .iter()
            .map(|f| {
                let name = &self.names[f.var];
                let k = crate::ml::tree::fmt_num(f.knot);
                match f.sign {
                    1 => format!("max(0, {name} - {k})"),
                    -1 => format!("max(0, {k} - {name})"),
                    _ => name.clone(),
                }
            })
            .collect::<Vec<_>>()
            .join(" * ")
    }
}

/// Residual sum of squares of the least-squares fit on the chosen columns (SVD, minimum norm).
fn svd_rss(cols: &[Vec<f64>], keep: &[usize], y: &[f64]) -> Result<f64> {
    let sel: Vec<Vec<f64>> = keep.iter().map(|&j| cols[j].clone()).collect();
    let a = Matrix::from_columns(&sel)?;
    let beta = Svd::new(&a)?.solve_least_squares(y, None)?;
    Ok((0..y.len())
        .map(|i| {
            let f: f64 = keep.iter().enumerate().map(|(q, &j)| beta[q] * cols[j][i]).sum();
            (y[i] - f).powi(2)
        })
        .sum())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Effective number of parameters of a model with `m` terms.
fn eff_params(m: usize, penalty: f64) -> f64 {
    m as f64 + penalty * (m as f64 - 1.0) / 2.0
}

fn gcv_of(rss: f64, n: usize, m: usize, penalty: f64) -> f64 {
    let nf = n as f64;
    let c = eff_params(m, penalty);
    if c >= nf {
        return f64::INFINITY;
    }
    (rss / nf) / (1.0 - c / nf).powi(2)
}

/// Orthonormal basis grown by Gram–Schmidt.
struct Ortho {
    q: Vec<Vec<f64>>,
}

impl Ortho {
    fn project_out(&self, v: &mut [f64]) {
        for _ in 0..2 {
            for q in &self.q {
                let c = dot(q, v);
                for (a, b) in v.iter_mut().zip(q) {
                    *a -= c * b;
                }
            }
        }
    }

    fn add(&mut self, col: &[f64]) -> bool {
        let n0 = dot(col, col).sqrt();
        let mut v = col.to_vec();
        self.project_out(&mut v);
        let n1 = dot(&v, &v).sqrt();
        if n1 <= 1e-9 * n0.max(1e-300) || n1 == 0.0 {
            return false;
        }
        for a in &mut v {
            *a /= n1;
        }
        self.q.push(v);
        true
    }
}

fn ls_fit(cols: &[&[f64]], y: &[f64]) -> Result<(Vec<f64>, f64)> {
    let n = y.len();
    let m = cols.len();
    let mut a = Matrix::zeros(n, m);
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            a[(i, j)] = c[i];
        }
    }
    let svd = Svd::new(&a)?;
    let beta = svd.solve_least_squares(y, None)?;
    let mut rss = 0.0;
    for i in 0..n {
        let f: f64 = (0..m).map(|j| a[(i, j)] * beta[j]).sum();
        rss += (y[i] - f).powi(2);
    }
    Ok((beta, rss))
}

fn basis_column(factors: &[Factor], x: &[Vec<f64>], n: usize) -> Vec<f64> {
    (0..n).map(|i| factors.iter().map(|f| f.eval(x[f.var][i])).product()).collect()
}

/// Fit a MARS model to numeric predictors (columns, all finite) and a numeric response.
pub fn fit_mars(x: &[Vec<f64>], names: &[String], y: &[f64], opts: &MarsOptions) -> Result<MarsModel> {
    let p = x.len();
    let n = y.len();
    if p == 0 {
        return invalid("MARS needs at least one predictor");
    }
    if names.len() != p || x.iter().any(|c| c.len() != n) {
        return invalid("predictor names and columns do not match the response");
    }
    if n < 8 {
        return Err(StatsError::InsufficientData { what: "MARS", needed: 8, got: n });
    }
    if x.iter().flatten().chain(y.iter()).any(|v| !v.is_finite()) {
        return invalid("MARS needs finite values (drop rows with missing data first)");
    }
    if opts.max_degree == 0 || opts.max_degree > 5 {
        return invalid("the maximum degree of interaction must be between 1 and 5");
    }
    if !(opts.threshold >= 0.0 && opts.threshold < 1.0) {
        return invalid("the forward-pass threshold must be in [0, 1)");
    }
    let penalty = opts.penalty.unwrap_or(if opts.max_degree > 1 { 3.0 } else { 2.0 });
    if !(penalty >= 0.0 && penalty.is_finite()) {
        return invalid("the penalty must be a non-negative number");
    }
    let max_terms = opts.max_terms.unwrap_or_else(|| (2 * p).max(20).saturating_add(1).min(200));
    if max_terms < 3 {
        return invalid("the maximum number of terms must be at least 3");
    }
    let nf = n as f64;
    let min_span = opts.min_span.unwrap_or_else(|| {
        let l = -(-(1.0f64 - 0.05).ln() / (nf * p as f64)).log2() / 2.5;
        (l.floor() as usize).max(1)
    });
    let end_span = opts.end_span.unwrap_or_else(|| ((3.0 - (0.05 / p as f64).log2()).ceil() as usize).max(1));
    if min_span == 0 {
        return invalid("the minimum span must be at least 1");
    }
    let mean = y.iter().sum::<f64>() / nf;
    let tss: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    if tss <= 0.0 {
        return invalid("the response is constant");
    }

    // ------------------------------------------------------------------ forward pass
    let mut terms: Vec<Vec<Factor>> = vec![Vec::new()];
    let mut cols: Vec<Vec<f64>> = vec![vec![1.0; n]];
    let mut ortho = Ortho { q: Vec::new() };
    ortho.add(&cols[0]);
    let residual = |o: &Ortho| -> Vec<f64> {
        let mut r = y.to_vec();
        o.project_out(&mut r);
        r
    };
    let mut r = residual(&ortho);
    let mut rss = dot(&r, &r);
    loop {
        if terms.len() + 1 > max_terms || rss / tss < opts.threshold {
            break;
        }
        // best candidate: (gain, parent, var, knot, is_pair)
        let mut best: Option<(f64, usize, usize, f64, bool)> = None;
        for (m, parent) in terms.iter().enumerate() {
            if parent.len() >= opts.max_degree {
                continue;
            }
            let bm = &cols[m];
            for v in 0..p {
                if parent.iter().any(|f| f.var == v) {
                    continue;
                }
                let mut xs: Vec<f64> = (0..n).filter(|&i| bm[i] > 0.0).map(|i| x[v][i]).collect();
                xs.sort_by(f64::total_cmp);
                let nm = xs.len();
                let mut cand: Vec<f64> = Vec::new();
                if nm > 2 * end_span {
                    let mut i = end_span;
                    while i < nm - end_span {
                        if cand.last() != Some(&xs[i]) {
                            cand.push(xs[i]);
                        }
                        i += min_span;
                    }
                }
                if terms.len() + 2 <= max_terms {
                    for &t in &cand {
                        let mut a: Vec<f64> = (0..n).map(|i| bm[i] * (x[v][i] - t).max(0.0)).collect();
                        let mut b: Vec<f64> = (0..n).map(|i| bm[i] * (t - x[v][i]).max(0.0)).collect();
                        ortho.project_out(&mut a);
                        ortho.project_out(&mut b);
                        let na = dot(&a, &a).sqrt();
                        let mut gain = 0.0;
                        let mut u1: Option<Vec<f64>> = None;
                        if na > 1e-9 {
                            let u: Vec<f64> = a.iter().map(|v| v / na).collect();
                            gain += dot(&u, &r).powi(2);
                            u1 = Some(u);
                        }
                        if let Some(u) = &u1 {
                            let c = dot(u, &b);
                            for (bb, uu) in b.iter_mut().zip(u) {
                                *bb -= c * uu;
                            }
                        }
                        let nb = dot(&b, &b).sqrt();
                        if nb > 1e-9 {
                            let u: Vec<f64> = b.iter().map(|v| v / nb).collect();
                            gain += dot(&u, &r).powi(2);
                        }
                        if gain > 0.0 && best.as_ref().is_none_or(|bst| gain > bst.0) {
                            best = Some((gain, m, v, t, true));
                        }
                    }
                }
                if opts.allow_linear && nm > 0 {
                    let mut c: Vec<f64> = (0..n).map(|i| bm[i] * x[v][i]).collect();
                    ortho.project_out(&mut c);
                    let nc = dot(&c, &c).sqrt();
                    if nc > 1e-9 {
                        let gain = (dot(&c, &r) / nc).powi(2);
                        if gain > 0.0 && best.as_ref().is_none_or(|bst| gain > bst.0) {
                            best = Some((gain, m, v, f64::NAN, false));
                        }
                    }
                }
            }
        }
        let Some((gain, m, v, t, pair)) = best else { break };
        if gain / tss < opts.threshold {
            break;
        }
        let parent = terms[m].clone();
        let mut add = |sign: i8, knot: f64| {
            let mut f = parent.clone();
            f.push(Factor { var: v, knot, sign });
            let col = basis_column(&f, x, n);
            ortho.add(&col);
            terms.push(f);
            cols.push(col);
        };
        if pair {
            add(1, t);
            add(-1, t);
        } else {
            add(0, 0.0);
        }
        r = residual(&ortho);
        rss = dot(&r, &r);
    }
    let forward_terms = terms.len();

    // ------------------------------------------------------------------ backward pass
    let mut keep: Vec<usize> = (0..terms.len()).collect();
    let mut path: Vec<(Vec<usize>, f64, f64)> = Vec::new();
    loop {
        let m = keep.len();
        let mut g = Matrix::zeros(m, m);
        let mut xty = vec![0.0; m];
        for (a, &ja) in keep.iter().enumerate() {
            xty[a] = dot(&cols[ja], y);
            for (b, &jb) in keep.iter().enumerate().take(a + 1) {
                let v = dot(&cols[ja], &cols[jb]);
                g[(a, b)] = v;
                g[(b, a)] = v;
            }
        }
        // Plain normal equations. Hinge pairs on one variable are exactly collinear with the
        // intercept, so a badly conditioned set is solved by SVD least squares instead.
        let diag: Vec<f64> = (0..m).map(|i| g[(i, i)]).collect();
        let normal = Cholesky::new(&g).ok().and_then(|c| {
            let inv = c.inverse().ok()?;
            let ok = (0..m).all(|i| inv[(i, i)] * diag[i] < 1e9);
            let beta = c.solve(&xty).ok()?;
            ok.then_some((inv, beta))
        });
        let mut rss_m = 0.0;
        let mut incr: Vec<f64> = Vec::new();
        if let Some((inv, beta)) = &normal {
            for i in 0..n {
                let f: f64 = keep.iter().enumerate().map(|(a, &j)| beta[a] * cols[j][i]).sum();
                rss_m += (y[i] - f).powi(2);
            }
            incr = (1..m).map(|j| beta[j] * beta[j] / inv[(j, j)]).collect();
        } else {
            rss_m = svd_rss(&cols, &keep, y)?;
            if m > 1 && opts.prune {
                for j in 1..m {
                    let mut k2 = keep.clone();
                    k2.remove(j);
                    incr.push((svd_rss(&cols, &k2, y)? - rss_m).max(0.0));
                }
            }
        }
        path.push((keep.clone(), rss_m, gcv_of(rss_m, n, m, penalty)));
        if m == 1 || !opts.prune {
            break;
        }
        // Ties (exactly redundant terms) go to the earliest term: increases within 1e-7 of the
        // total sum of squares of the smallest count as tied.
        let least = incr.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst = 1 + incr.iter().position(|&d| d <= least + 1e-7 * tss).unwrap_or(0);
        keep.remove(worst);
    }
    let chosen = if opts.prune {
        let mut bi = 0;
        for (i, row) in path.iter().enumerate() {
            if row.2 < path[bi].2 || (row.2 == path[bi].2 && row.0.len() < path[bi].0.len()) {
                bi = i;
            }
        }
        bi
    } else {
        0
    };
    let final_keep = path[chosen].0.clone();
    let colrefs: Vec<&[f64]> = final_keep.iter().map(|&j| cols[j].as_slice()).collect();
    let (beta, rss_f) = ls_fit(&colrefs, y)?;
    let m = final_keep.len();
    let gcv = gcv_of(rss_f, n, m, penalty);
    let gcv_null = gcv_of(tss, n, 1, penalty);
    let fitted: Vec<f64> = (0..n).map(|i| colrefs.iter().zip(&beta).map(|(c, b)| c[i] * b).sum()).collect();
    let final_terms: Vec<Term> =
        final_keep.iter().zip(&beta).map(|(&j, &c)| Term { factors: terms[j].clone(), coef: c }).collect();

    // importance: GCV and RSS increase when every term using the variable is removed
    let mut imp = Vec::new();
    for v in 0..p {
        if !final_terms.iter().any(|t| t.factors.iter().any(|f| f.var == v)) {
            continue;
        }
        let rest: Vec<&[f64]> =
            final_keep.iter().filter(|&&j| !terms[j].iter().any(|f| f.var == v)).map(|&j| cols[j].as_slice()).collect();
        let (_, rss_wo) = ls_fit(&rest, y)?;
        let g_wo = gcv_of(rss_wo, n, rest.len(), penalty);
        imp.push((v, (g_wo - gcv).max(0.0), (rss_wo - rss_f).max(0.0)));
    }
    let (mg, mr) = imp.iter().fold((0.0f64, 0.0f64), |a, b| (a.0.max(b.1), a.1.max(b.2)));
    for e in &mut imp {
        e.1 = if mg > 0.0 { 100.0 * e.1 / mg } else { 0.0 };
        e.2 = if mr > 0.0 { 100.0 * e.2 / mr } else { 0.0 };
    }
    imp.sort_by(|a, b| b.1.total_cmp(&a.1));

    let path_rows: Vec<PruneRow> = if opts.prune {
        path.iter()
            .enumerate()
            .map(|(i, (k, r, g))| PruneRow { terms: k.len(), rss: *r, gcv: *g, selected: i == chosen })
            .collect()
    } else {
        Vec::new()
    };
    Ok(MarsModel {
        names: names.to_vec(),
        terms: final_terms,
        n,
        rss: rss_f,
        tss,
        gcv,
        gcv_null,
        penalty,
        forward_terms,
        path: path_rows,
        importance: imp,
        y: y.to_vec(),
        fitted,
        spans: (min_span, end_span),
    })
}
