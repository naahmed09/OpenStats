//! Stepwise regression (forward, backward, bidirectional) over model terms.

use super::fit::{fit_design, RegressionFit, RegressionOptions};
use crate::error::{invalid, Result, StatsError};
use crate::model::{build_design, Data, DesignMatrix, DesignOptions};
use std::collections::HashSet;

/// Search direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMethod {
    /// Start from the forced terms and only add.
    Forward,
    /// Start from all terms and only remove.
    Backward,
    /// Start from the forced terms; add, then re-check removals.
    Stepwise,
}

/// Rule deciding whether a term enters or leaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepCriterion {
    /// F-test p-values: a term enters when its p-value is ≤ `alpha_enter`
    /// and leaves when its p-value is > `alpha_remove`.
    PValue {
        /// Entry threshold.
        alpha_enter: f64,
        /// Removal threshold.
        alpha_remove: f64,
    },
    /// Akaike information criterion.
    Aic,
    /// Corrected Akaike information criterion.
    Aicc,
    /// Bayesian information criterion.
    Bic,
}

impl StepCriterion {
    /// Minitab's defaults: stepwise 0.15/0.15, forward 0.25, backward 0.10.
    pub fn default_for(method: StepMethod) -> StepCriterion {
        match method {
            StepMethod::Stepwise => StepCriterion::PValue { alpha_enter: 0.15, alpha_remove: 0.15 },
            StepMethod::Forward => StepCriterion::PValue { alpha_enter: 0.25, alpha_remove: 0.25 },
            StepMethod::Backward => StepCriterion::PValue { alpha_enter: 0.10, alpha_remove: 0.10 },
        }
    }
}

/// Options of a stepwise search.
#[derive(Debug, Clone, PartialEq)]
pub struct StepwiseOptions {
    /// Search direction.
    pub method: StepMethod,
    /// Entry/removal rule.
    pub criterion: StepCriterion,
    /// Labels of terms that are always in the model.
    pub force_in: Vec<String>,
    /// Keep the model hierarchical: a term needs all of its lower-order
    /// margins (`A` and `B` for `A*B`) and a margin cannot leave while a term
    /// containing it remains.
    pub hierarchical: bool,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl StepwiseOptions {
    /// Defaults for `method`.
    pub fn new(method: StepMethod) -> StepwiseOptions {
        StepwiseOptions {
            method,
            criterion: StepCriterion::default_for(method),
            force_in: Vec::new(),
            hierarchical: true,
            max_steps: 1000,
        }
    }
}

/// What happened in a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepAction {
    /// A term entered.
    Added,
    /// A term left.
    Removed,
}

/// One step of the search.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// Step number from 1.
    pub number: usize,
    /// Added or removed.
    pub action: StepAction,
    /// Term label.
    pub term: String,
    /// The p-value (p-value criterion) or the information criterion after the step.
    pub value: f64,
    /// Number of terms in the model after the step.
    pub n_terms: usize,
    /// R² after the step.
    pub r2: f64,
    /// Adjusted R² after the step.
    pub r2_adj: f64,
    /// Residual standard error after the step.
    pub s: f64,
}

/// Result of a stepwise search.
#[derive(Debug, Clone)]
pub struct StepwiseResult {
    /// The steps taken.
    pub steps: Vec<Step>,
    /// Labels of the terms in the final model.
    pub selected: Vec<String>,
    /// The final fit.
    pub fit: RegressionFit,
}

fn components(label: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut cur) = (0i32, String::new());
    for ch in label.chars() {
        match ch {
            '(' => {
                depth += 1;
                cur.push(ch);
            }
            ')' => {
                depth -= 1;
                cur.push(ch);
            }
            '*' if depth == 0 => out.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

/// Design containing the constant (if any) and the listed terms (in model order).
pub(crate) fn sub_design(dm: &DesignMatrix, terms: &[usize]) -> Result<DesignMatrix> {
    let mut cols: Vec<usize> = Vec::new();
    if dm.intercept {
        cols.push(0);
    }
    let mut new_terms = Vec::new();
    let mut sorted = terms.to_vec();
    sorted.sort_unstable();
    for &t in &sorted {
        let tc = &dm.terms[t];
        let mut nt = tc.clone();
        nt.start = cols.len();
        new_terms.push(nt);
        cols.extend(tc.start..tc.start + tc.len);
    }
    if cols.is_empty() {
        return invalid("the model has neither a constant nor terms");
    }
    let rows: Vec<usize> = (0..dm.x.rows()).collect();
    Ok(DesignMatrix {
        x: dm.x.select(&rows, &cols),
        names: cols.iter().map(|&c| dm.names[c].clone()).collect(),
        terms: new_terms,
        intercept: dm.intercept,
        y: dm.y.clone(),
        offset: dm.offset.clone(),
        rows_used: dm.rows_used.clone(),
        n_dropped: dm.n_dropped,
    })
}

struct Search<'a> {
    dm: &'a DesignMatrix,
    ropts: &'a RegressionOptions,
    comps: Vec<Vec<String>>,
}

impl Search<'_> {
    fn fit(&self, terms: &[usize]) -> Result<RegressionFit> {
        fit_design(&sub_design(self.dm, terms)?, self.ropts)
    }

    fn can_add(&self, t: usize, cur: &[usize], hier: bool) -> bool {
        if !hier {
            return true;
        }
        let c = &self.comps[t];
        if c.len() < 2 {
            return true;
        }
        // Every proper sub-product of the term's factors present in the model universe must be in.
        (0..self.dm.terms.len()).all(|o| {
            o == t
                || !(self.comps[o].len() < c.len() && self.comps[o].iter().all(|a| c.contains(a)))
                || cur.contains(&o)
        })
    }

    fn can_remove(&self, t: usize, cur: &[usize], hier: bool) -> bool {
        if !hier {
            return true;
        }
        let c = &self.comps[t];
        !cur.iter().any(|&o| o != t && self.comps[o].len() > c.len() && c.iter().all(|a| self.comps[o].contains(a)))
    }
}

fn ic_value(fit: &RegressionFit, crit: StepCriterion) -> f64 {
    let v = match crit {
        StepCriterion::Aic => fit.aic(),
        StepCriterion::Aicc => fit.aicc(),
        StepCriterion::Bic => fit.bic(),
        StepCriterion::PValue { .. } => f64::NAN,
    };
    if v.is_nan() {
        f64::INFINITY
    } else {
        v
    }
}

fn term_p(fit: &RegressionFit, label: &str) -> Option<f64> {
    fit.anova.iter().find(|r| r.source == label).map(|r| r.p_value.unwrap_or(0.0))
}

/// Run a stepwise search on a prepared full design (all candidate terms).
pub fn stepwise_design(dm: &DesignMatrix, opts: &StepwiseOptions, ropts: &RegressionOptions) -> Result<StepwiseResult> {
    if let StepCriterion::PValue { alpha_enter, alpha_remove } = opts.criterion {
        for a in [alpha_enter, alpha_remove] {
            if !(a > 0.0 && a < 1.0) {
                return invalid(format!("alpha {a} must be in (0, 1)"));
            }
        }
        if alpha_remove < alpha_enter {
            return invalid("alpha to remove must be at least alpha to enter (otherwise terms can cycle)");
        }
    }
    let nt = dm.terms.len();
    if nt == 0 {
        return invalid("the model has no terms to select from");
    }
    let mut forced = Vec::new();
    for lab in &opts.force_in {
        match dm.terms.iter().position(|t| &t.label == lab) {
            Some(i) => forced.push(i),
            None => return invalid(format!("force_in term '{lab}' is not in the model")),
        }
    }
    let s = Search { dm, ropts, comps: dm.terms.iter().map(|t| components(&t.label)).collect() };
    let mut cur: Vec<usize> = match opts.method {
        StepMethod::Backward => (0..nt).collect(),
        _ => forced.clone(),
    };
    cur.sort_unstable();
    let can_try_add = opts.method != StepMethod::Backward;
    let can_try_remove = opts.method != StepMethod::Forward;
    let mut steps: Vec<Step> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    seen.insert(cur.clone());
    let mut cur_fit = s.fit(&cur).ok();
    if cur_fit.is_none() && opts.method == StepMethod::Backward {
        return Err(StatsError::InvalidArgument(
            "the full model cannot be fitted (too many coefficients for the data)".into(),
        ));
    }

    while steps.len() < opts.max_steps {
        let mut acted = false;
        // Try adding.
        if can_try_add {
            let mut best: Option<(usize, f64, RegressionFit)> = None;
            for t in 0..nt {
                if cur.contains(&t) || !s.can_add(t, &cur, opts.hierarchical) {
                    continue;
                }
                let mut cand = cur.clone();
                cand.push(t);
                let Ok(f) = s.fit(&cand) else { continue };
                let key = match opts.criterion {
                    StepCriterion::PValue { .. } => match term_p(&f, &dm.terms[t].label) {
                        Some(p) => p,
                        None => continue,
                    },
                    c => ic_value(&f, c),
                };
                if best.as_ref().is_none_or(|b| key < b.1) {
                    best = Some((t, key, f));
                }
            }
            if let Some((t, key, f)) = best {
                let ok = match opts.criterion {
                    StepCriterion::PValue { alpha_enter, .. } => key <= alpha_enter,
                    c => cur_fit.as_ref().is_none_or(|cf| key < ic_value(cf, c)),
                };
                if ok {
                    cur.push(t);
                    cur.sort_unstable();
                    if seen.insert(cur.clone()) {
                        steps.push(Step {
                            number: steps.len() + 1,
                            action: StepAction::Added,
                            term: dm.terms[t].label.clone(),
                            value: key,
                            n_terms: cur.len(),
                            r2: f.r2,
                            r2_adj: f.r2_adj,
                            s: f.s,
                        });
                        cur_fit = Some(f);
                        acted = true;
                    } else {
                        cur.retain(|&x| x != t);
                    }
                }
            }
        }
        // Try removing.
        if can_try_remove && !cur.is_empty() && (!acted || opts.method == StepMethod::Stepwise) {
            let mut worst: Option<(usize, f64, RegressionFit)> = None;
            if let Some(cf) = &cur_fit {
                for &t in &cur {
                    if forced.contains(&t) || !s.can_remove(t, &cur, opts.hierarchical) {
                        continue;
                    }
                    let cand: Vec<usize> = cur.iter().copied().filter(|&x| x != t).collect();
                    if cand.is_empty() && !dm.intercept {
                        continue;
                    }
                    let Ok(f) = s.fit(&cand) else { continue };
                    let key = match opts.criterion {
                        StepCriterion::PValue { .. } => match term_p(cf, &dm.terms[t].label) {
                            Some(p) => -p,
                            None => continue,
                        },
                        c => ic_value(&f, c),
                    };
                    if worst.as_ref().is_none_or(|b| key < b.1) {
                        worst = Some((t, key, f));
                    }
                }
            }
            if let Some((t, key, f)) = worst {
                let ok = match opts.criterion {
                    StepCriterion::PValue { alpha_remove, .. } => -key > alpha_remove,
                    c => cur_fit.as_ref().is_none_or(|cf| key < ic_value(cf, c)),
                };
                if ok {
                    let mut cand = cur.clone();
                    cand.retain(|&x| x != t);
                    if seen.insert(cand.clone()) {
                        cur = cand;
                        let value = if matches!(opts.criterion, StepCriterion::PValue { .. }) { -key } else { key };
                        steps.push(Step {
                            number: steps.len() + 1,
                            action: StepAction::Removed,
                            term: dm.terms[t].label.clone(),
                            value,
                            n_terms: cur.len(),
                            r2: f.r2,
                            r2_adj: f.r2_adj,
                            s: f.s,
                        });
                        cur_fit = Some(f);
                        acted = true;
                    }
                }
            }
        }
        if !acted {
            break;
        }
    }
    let fit = match cur_fit {
        Some(f) => f,
        None => s.fit(&cur)?,
    };
    let selected = cur.iter().map(|&t| dm.terms[t].label.clone()).collect();
    Ok(StepwiseResult { steps, selected, fit })
}

/// Parse `formula` (its terms are the candidates), build the design and search.
pub fn stepwise(
    formula: &str,
    data: &Data,
    design: &DesignOptions,
    opts: &StepwiseOptions,
    ropts: &RegressionOptions,
) -> Result<StepwiseResult> {
    let (_, dm) = build_design(formula, data, design)?;
    stepwise_design(&dm, opts, ropts)
}

#[cfg(test)]
mod tests {
    use super::components;

    #[test]
    fn components_split_on_top_level_stars_only() {
        assert_eq!(components("A*B*C"), vec!["A", "B", "C"]);
        assert_eq!(components("I(a*b)*C"), vec!["I(a*b)", "C"]);
        assert_eq!(components("A(B)"), vec!["A(B)"]);
    }
}
