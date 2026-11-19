//! Stepwise term selection for generalized linear models (forward,
//! backward and bidirectional) with likelihood-ratio entry/removal tests.

use super::{fit_glm, GlmFit, GlmOptions};
use crate::error::{invalid, Result};
use crate::model::{build_design, Data, DesignMatrix, DesignOptions};
use crate::regression::{StepAction, StepMethod};
use openstats_math::dist::{ChiSquared, ContinuousDistribution, FisherF};
use openstats_math::linalg::Matrix;

/// Options of a GLM stepwise search.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmStepwiseOptions {
    /// Search direction.
    pub method: StepMethod,
    /// A term enters when its likelihood-ratio p-value is ≤ this.
    pub alpha_enter: f64,
    /// A term leaves when its likelihood-ratio p-value is > this.
    pub alpha_remove: f64,
    /// Labels of terms that are always in the model.
    pub force_in: Vec<String>,
    /// Require every lower-order margin of an interaction to be present.
    pub hierarchical: bool,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl GlmStepwiseOptions {
    /// Minitab's defaults for `method` (0.15/0.15, 0.25 forward, 0.10 backward).
    pub fn new(method: StepMethod) -> GlmStepwiseOptions {
        let (e, r) = match method {
            StepMethod::Stepwise => (0.15, 0.15),
            StepMethod::Forward => (0.25, 0.25),
            StepMethod::Backward => (0.10, 0.10),
        };
        GlmStepwiseOptions {
            method,
            alpha_enter: e,
            alpha_remove: r,
            force_in: Vec::new(),
            hierarchical: true,
            max_steps: 1000,
        }
    }
}

/// One step of the search.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmStep {
    /// Step number from 1.
    pub number: usize,
    /// Added or removed.
    pub action: StepAction,
    /// Term label.
    pub term: String,
    /// Likelihood-ratio p-value that triggered the step.
    pub p_value: f64,
    /// Terms in the model after the step.
    pub n_terms: usize,
    /// Deviance after the step.
    pub deviance: f64,
    /// AIC after the step (when defined).
    pub aic: Option<f64>,
}

/// Result of a stepwise search.
#[derive(Debug, Clone)]
pub struct GlmStepwiseResult {
    /// Steps taken.
    pub steps: Vec<GlmStep>,
    /// Labels of the selected terms.
    pub selected: Vec<String>,
    /// The final fit.
    pub fit: GlmFit,
}

fn subset(dm: &DesignMatrix, sel: &[usize]) -> DesignMatrix {
    let mut cols: Vec<usize> = Vec::new();
    if dm.intercept {
        cols.push(0);
    }
    let mut terms = Vec::new();
    for &t in sel {
        let tc = &dm.terms[t];
        let start = cols.len();
        cols.extend(tc.start..tc.start + tc.len);
        let mut nt = tc.clone();
        nt.start = start;
        terms.push(nt);
    }
    let rows: Vec<usize> = (0..dm.x.rows()).collect();
    let x: Matrix = dm.x.select(&rows, &cols);
    DesignMatrix {
        x,
        names: cols.iter().map(|&j| dm.names[j].clone()).collect(),
        terms,
        intercept: dm.intercept,
        y: dm.y.clone(),
        offset: dm.offset.clone(),
        rows_used: dm.rows_used.clone(),
        n_dropped: dm.n_dropped,
    }
}

fn components(label: &str) -> Vec<String> {
    label.split(['*', ':']).map(|s| s.trim().to_string()).collect()
}

fn lr_p(small: &GlmFit, big: &GlmFit) -> Result<f64> {
    let df = big.df_model.saturating_sub(small.df_model);
    if df == 0 {
        return Ok(1.0);
    }
    let delta = (small.deviance - big.deviance).max(0.0);
    if big.family == super::Family::Gamma || big.family == super::Family::Gaussian {
        let f = (delta / df as f64) / big.dispersion;
        Ok(FisherF::new(df as f64, big.df_error as f64)?.sf(f))
    } else {
        Ok(ChiSquared::new(df as f64)?.sf(delta))
    }
}

/// Stepwise selection of the terms of `formula` (all candidate terms are
/// those the formula expands to).
pub fn glm_stepwise(
    formula: &str,
    data: &Data,
    design: &DesignOptions,
    opts: &GlmOptions,
    so: &GlmStepwiseOptions,
) -> Result<GlmStepwiseResult> {
    let (_spec, dm) = build_design(formula, data, design)?;
    let mut o = opts.clone();
    if let Some(t) = &opts.trials {
        if t.len() != data.n_rows() {
            return invalid("one trials value per worksheet row is required");
        }
        o.trials = Some(dm.rows_used.iter().map(|&i| t[i]).collect());
    }
    let labels: Vec<String> = dm.terms.iter().map(|t| t.label.clone()).collect();
    let nt = labels.len();
    if nt == 0 {
        return invalid("the model has no terms to select from");
    }
    let mut forced = Vec::new();
    for f in &so.force_in {
        match labels.iter().position(|l| l == f) {
            Some(i) => forced.push(i),
            None => return invalid(format!("forced term '{f}' is not in the model")),
        }
    }
    let fit_sel = |sel: &[usize]| fit_glm(&subset(&dm, sel), &o);
    let allowed_add = |sel: &[usize], cand: usize| -> bool {
        if !so.hierarchical {
            return true;
        }
        let comps = components(&labels[cand]);
        if comps.len() < 2 {
            return true;
        }
        // Every proper margin (single factors) must be in the model.
        comps.iter().all(|c| sel.iter().any(|&s| labels[s] == *c))
    };
    let allowed_remove = |sel: &[usize], cand: usize| -> bool {
        if !so.hierarchical {
            return true;
        }
        let me = &labels[cand];
        !sel.iter().any(|&s| s != cand && components(&labels[s]).len() > 1 && components(&labels[s]).contains(me))
    };

    let mut current: Vec<usize> = match so.method {
        StepMethod::Backward => (0..nt).collect(),
        _ => forced.clone(),
    };
    current.sort_unstable();
    let mut steps: Vec<GlmStep> = Vec::new();
    let mut cur_fit = fit_sel(&current)?;
    while steps.len() < so.max_steps {
        let mut changed = false;
        if so.method != StepMethod::Backward {
            let mut best: Option<(usize, f64, GlmFit)> = None;
            for cand in 0..nt {
                if current.contains(&cand) || !allowed_add(&current, cand) {
                    continue;
                }
                let mut sel = current.clone();
                sel.push(cand);
                sel.sort_unstable();
                let Ok(f) = fit_sel(&sel) else { continue };
                let p = lr_p(&cur_fit, &f)?;
                if p <= so.alpha_enter && best.as_ref().is_none_or(|b| p < b.1) {
                    best = Some((cand, p, f));
                }
            }
            if let Some((cand, p, f)) = best {
                current.push(cand);
                current.sort_unstable();
                steps.push(GlmStep {
                    number: steps.len() + 1,
                    action: StepAction::Added,
                    term: labels[cand].clone(),
                    p_value: p,
                    n_terms: current.len(),
                    deviance: f.deviance,
                    aic: f.aic,
                });
                cur_fit = f;
                changed = true;
            }
        }
        if so.method != StepMethod::Forward && !current.is_empty() {
            let mut worst: Option<(usize, f64, GlmFit)> = None;
            for &cand in &current {
                if forced.contains(&cand) || !allowed_remove(&current, cand) {
                    continue;
                }
                let sel: Vec<usize> = current.iter().copied().filter(|&c| c != cand).collect();
                let Ok(f) = fit_sel(&sel) else { continue };
                let p = lr_p(&f, &cur_fit)?;
                if p > so.alpha_remove && worst.as_ref().is_none_or(|w| p > w.1) {
                    worst = Some((cand, p, f));
                }
            }
            if let Some((cand, p, f)) = worst {
                // Do not undo the step just taken.
                let undo = steps.last().is_some_and(|s| s.action == StepAction::Added && s.term == labels[cand]);
                if !undo {
                    current.retain(|&c| c != cand);
                    steps.push(GlmStep {
                        number: steps.len() + 1,
                        action: StepAction::Removed,
                        term: labels[cand].clone(),
                        p_value: p,
                        n_terms: current.len(),
                        deviance: f.deviance,
                        aic: f.aic,
                    });
                    cur_fit = f;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(GlmStepwiseResult { selected: current.iter().map(|&i| labels[i].clone()).collect(), steps, fit: cur_fit })
}
