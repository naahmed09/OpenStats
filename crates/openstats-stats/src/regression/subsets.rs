//! Best-subsets regression by Furnival–Wilson leaps and bounds.

use super::fit::{fit_xy, independent_columns, RegressionOptions};
use crate::error::{invalid, Result};
use openstats_math::linalg::Matrix;

/// Options of a best-subsets search.
#[derive(Debug, Clone, PartialEq)]
pub struct SubsetOptions {
    /// Include a constant in every model.
    pub intercept: bool,
    /// Smallest number of predictors reported (at least the forced count).
    pub min_size: usize,
    /// Largest number of predictors reported (`None` = all).
    pub max_size: Option<usize>,
    /// Models reported per size (Minitab's default is 2).
    pub n_best: usize,
    /// Names of predictors present in every model.
    pub force_in: Vec<String>,
}

impl Default for SubsetOptions {
    fn default() -> Self {
        SubsetOptions { intercept: true, min_size: 1, max_size: None, n_best: 2, force_in: Vec::new() }
    }
}

/// One reported model.
#[derive(Debug, Clone, PartialEq)]
pub struct SubsetModel {
    /// Indices into the usable predictors (original column order).
    pub vars: Vec<usize>,
    /// Names of the predictors.
    pub names: Vec<String>,
    /// Number of predictors.
    pub size: usize,
    /// R².
    pub r2: f64,
    /// Adjusted R².
    pub r2_adj: f64,
    /// Predicted R².
    pub r2_pred: f64,
    /// Mallows' Cp (error mean square taken from the full model).
    pub cp: f64,
    /// Residual standard error.
    pub s: f64,
    /// PRESS.
    pub press: f64,
    /// AICc.
    pub aicc: f64,
    /// BIC.
    pub bic: f64,
    /// Error sum of squares.
    pub sse: f64,
}

/// Result of a best-subsets search.
#[derive(Debug, Clone, PartialEq)]
pub struct BestSubsets {
    /// Models ordered by size, then by decreasing R².
    pub models: Vec<SubsetModel>,
    /// Predictors excluded because they are linear combinations of earlier ones.
    pub excluded: Vec<String>,
    /// Number of observations.
    pub n: usize,
}

struct Ctx {
    m: usize,
    min_size: usize,
    max_size: usize,
    n_best: usize,
    /// Best (rss, vars) per size, ascending by rss; index = size.
    best: Vec<Vec<(f64, Vec<usize>)>>,
}

fn sweep(a: &mut [Vec<f64>], k: usize) -> bool {
    let d = a[k][k];
    if d.is_nan() || d <= 0.0 || !d.is_finite() {
        return false;
    }
    let dim = a.len();
    for j in 0..dim {
        a[k][j] /= d;
    }
    let rowk = a[k].clone();
    for (i, row) in a.iter_mut().enumerate() {
        if i == k {
            continue;
        }
        let b = row[k];
        for j in 0..dim {
            row[j] -= b * rowk[j];
        }
        row[k] = -b / d;
    }
    a[k][k] = 1.0 / d;
    true
}

impl Ctx {
    fn record(&mut self, rss: f64, vars: &[usize]) {
        let size = vars.len();
        if size < self.min_size || size > self.max_size || size == 0 {
            return;
        }
        let v = &mut self.best[size];
        let pos = v.partition_point(|(r, _)| *r <= rss);
        if pos < self.n_best {
            v.insert(pos, (rss, vars.to_vec()));
            v.truncate(self.n_best);
        }
    }

    fn worth_exploring(&self, bound: f64, cur: usize, rest: usize) -> bool {
        let lo = (cur + 1).max(self.min_size);
        let hi = (cur + rest).min(self.max_size);
        if lo > hi {
            return false;
        }
        (lo..=hi).any(|s| {
            let v = &self.best[s];
            v.len() < self.n_best || bound < v[self.n_best - 1].0 * (1.0 + 1e-9)
        })
    }

    fn search(&mut self, a: &[Vec<f64>], inc: &mut Vec<usize>, next: usize) {
        let y = self.m;
        let rest = self.m - next;
        if rest == 0 || inc.len() >= self.max_size {
            return;
        }
        // Lower bound: RSS with every remaining variable added.
        let mut all = a.to_vec();
        let mut ok = true;
        for j in next..self.m {
            ok &= sweep(&mut all, j);
        }
        let bound = if ok { all[y][y].max(0.0) } else { 0.0 };
        if !self.worth_exploring(bound, inc.len(), rest) {
            return;
        }
        for j in next..self.m {
            let mut child = a.to_vec();
            if !sweep(&mut child, j) {
                continue;
            }
            inc.push(j);
            self.record(child[y][y].max(0.0), inc);
            self.search(&child, inc, j + 1);
            inc.pop();
        }
    }
}

/// Best-subsets regression of `y` on the named predictor columns.
pub fn best_subsets(
    columns: &[(&str, &[f64])],
    y: &[f64],
    opts: &SubsetOptions,
    ropts: &RegressionOptions,
) -> Result<BestSubsets> {
    let n = y.len();
    if columns.is_empty() {
        return invalid("best subsets needs at least one predictor");
    }
    if opts.n_best == 0 {
        return invalid("n_best must be at least 1");
    }
    if columns.iter().any(|(_, c)| c.len() != n) {
        return invalid("predictor columns and the response differ in length");
    }
    if y.iter().chain(columns.iter().flat_map(|(_, c)| c.iter())).any(|v| !v.is_finite()) {
        return Err(crate::error::StatsError::NonFinite);
    }
    // Drop aliased predictors.
    let cols: Vec<Vec<f64>> = {
        let mut cs = Vec::new();
        if opts.intercept {
            cs.push(vec![1.0; n]);
        }
        cs.extend(columns.iter().map(|(_, c)| c.to_vec()));
        cs
    };
    let (kept_all, dropped) = independent_columns(&Matrix::from_columns(&cols)?, ropts.alias_tolerance);
    let off = usize::from(opts.intercept);
    let usable: Vec<usize> = kept_all.into_iter().filter(|&j| j >= off).map(|j| j - off).collect();
    let excluded: Vec<String> =
        dropped.into_iter().filter(|&j| j >= off).map(|j| columns[j - off].0.to_string()).collect();
    let m = usable.len();
    if m == 0 {
        return invalid("no usable predictors");
    }
    // Forced variables (positions among the usable).
    let mut forced = Vec::new();
    for name in &opts.force_in {
        match usable.iter().position(|&u| columns[u].0 == name) {
            Some(p) => forced.push(p),
            None => return invalid(format!("force_in predictor '{name}' is not available")),
        }
    }
    forced.sort_unstable();
    forced.dedup();
    let free: Vec<usize> = (0..m).filter(|p| !forced.contains(p)).collect();
    let nforce = forced.len();
    if n <= m + usize::from(opts.intercept) {
        return Err(crate::error::StatsError::InsufficientData {
            what: "best subsets (observations must exceed predictors + constant)",
            needed: m + usize::from(opts.intercept) + 1,
            got: n,
        });
    }
    // Scaled, centred sweep matrix: variables are forced first, then free, then y.
    let order: Vec<usize> = forced.iter().chain(free.iter()).copied().collect();
    let mut z: Vec<Vec<f64>> = order.iter().map(|&p| columns[usable[p]].1.to_vec()).collect();
    z.push(y.to_vec());
    for c in &mut z {
        if opts.intercept {
            let mean = c.iter().sum::<f64>() / n as f64;
            c.iter_mut().for_each(|v| *v -= mean);
        }
        let nrm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm > 0.0 {
            c.iter_mut().for_each(|v| *v /= nrm);
        }
    }
    let dim = m + 1;
    let mut a = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..=i {
            let v: f64 = z[i].iter().zip(&z[j]).map(|(p, q)| p * q).sum();
            a[i][j] = v;
            a[j][i] = v;
        }
    }
    let max_size = opts.max_size.unwrap_or(m).min(m);
    let min_size = opts.min_size.max(nforce).max(1);
    if min_size > max_size {
        return invalid("min_size exceeds max_size");
    }
    let mut ctx = Ctx { m, min_size, max_size, n_best: opts.n_best, best: vec![Vec::new(); m + 1] };
    // Sweep the forced variables first; they are in every subset.
    let mut base = a.clone();
    for k in 0..nforce {
        if !sweep(&mut base, k) {
            return invalid("forced predictors are collinear");
        }
    }
    let mut inc: Vec<usize> = (0..nforce).collect();
    if nforce > 0 {
        ctx.record(base[m][m].max(0.0), &inc);
    }
    ctx.search(&base, &mut inc, nforce);

    // Final statistics by QR, in original variable numbering.
    let full: Vec<(&str, &[f64])> = usable.iter().map(|&u| (columns[u].0, columns[u].1)).collect();
    let full_fit = fit_xy(&full, y, opts.intercept, ropts)?;
    let mse_full = full_fit.mse;
    let mut models = Vec::new();
    for size in 1..=m {
        for (_, vars) in &ctx.best[size] {
            let mut orig: Vec<usize> = vars.iter().map(|&v| order[v]).collect();
            orig.sort_unstable();
            let sub: Vec<(&str, &[f64])> = orig.iter().map(|&p| full[p]).collect();
            let f = fit_xy(&sub, y, opts.intercept, ropts)?;
            let p = f.n_coef() as f64;
            models.push(SubsetModel {
                names: orig.iter().map(|&p| full[p].0.to_string()).collect(),
                vars: orig.iter().map(|&p| usable[p]).collect(),
                size,
                r2: f.r2,
                r2_adj: f.r2_adj,
                r2_pred: f.r2_pred,
                cp: f.sse / mse_full - (n as f64 - 2.0 * p),
                s: f.s,
                press: f.press,
                aicc: f.aicc(),
                bic: f.bic(),
                sse: f.sse,
            });
        }
    }
    Ok(BestSubsets { models, excluded, n })
}
