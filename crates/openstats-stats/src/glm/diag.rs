//! Binary-response diagnostics: Hosmer–Lemeshow, classification table, ROC
//! curve and rank-association measures. Observations may carry several
//! trials; each trial counts as one case.

use super::{Family, GlmFit};
use crate::error::{invalid, Result};
use openstats_math::dist::{ChiSquared, ContinuousDistribution};

fn binomial_parts(fit: &GlmFit) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if fit.family != Family::Binomial {
        return invalid("this diagnostic needs a binomial (binary) model");
    }
    Ok((fit.events(), fit.trials.clone(), fit.fitted.clone()))
}

/// One group of the Hosmer–Lemeshow test.
#[derive(Debug, Clone, PartialEq)]
pub struct HlGroup {
    /// Cases (trials) in the group.
    pub n: f64,
    /// Observed events.
    pub observed: f64,
    /// Expected events.
    pub expected: f64,
}

/// Hosmer–Lemeshow goodness-of-fit test.
#[derive(Debug, Clone, PartialEq)]
pub struct HosmerLemeshow {
    /// The groups, by increasing predicted probability.
    pub groups: Vec<HlGroup>,
    /// χ² statistic.
    pub chi_sq: f64,
    /// Degrees of freedom (groups − 2).
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

/// Hosmer–Lemeshow test with (about) `g` groups of equal size by predicted
/// probability; rows with identical predictions are kept in one group.
pub fn hosmer_lemeshow(fit: &GlmFit, g: usize) -> Result<HosmerLemeshow> {
    let (k, m, mu) = binomial_parts(fit)?;
    if g < 3 {
        return invalid("the Hosmer–Lemeshow test needs at least 3 groups");
    }
    let n = k.len();
    let total: f64 = m.iter().sum();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mu[a].total_cmp(&mu[b]));
    let mut groups: Vec<HlGroup> = Vec::new();
    let mut cur = HlGroup { n: 0.0, observed: 0.0, expected: 0.0 };
    let mut cum = 0.0;
    for (pos, &i) in order.iter().enumerate() {
        cur.n += m[i];
        cur.observed += k[i];
        cur.expected += m[i] * mu[i];
        cum += m[i];
        let target = total * (groups.len() + 1) as f64 / g as f64;
        let last = pos + 1 == n;
        let tie_next = !last && mu[order[pos + 1]] == mu[i];
        if last || (cum >= target - 1e-9 && !tie_next && groups.len() + 1 < g) {
            groups.push(std::mem::replace(&mut cur, HlGroup { n: 0.0, observed: 0.0, expected: 0.0 }));
        }
    }
    if groups.len() < 3 {
        return invalid("too few distinct predicted probabilities for the Hosmer–Lemeshow test");
    }
    let chi_sq: f64 = groups
        .iter()
        .map(|gr| {
            let pbar = gr.expected / gr.n;
            let v = gr.n * pbar * (1.0 - pbar);
            if v > 0.0 {
                (gr.observed - gr.expected).powi(2) / v
            } else {
                0.0
            }
        })
        .sum();
    let df = groups.len() - 2;
    Ok(HosmerLemeshow { chi_sq, df, p_value: ChiSquared::new(df as f64)?.sf(chi_sq), groups })
}

/// Observed versus predicted classification at a probability cutoff.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    /// Cutoff: predicted probability ≥ cutoff means "event".
    pub cutoff: f64,
    /// Events predicted as events.
    pub true_pos: f64,
    /// Events predicted as non-events.
    pub false_neg: f64,
    /// Non-events predicted as events.
    pub false_pos: f64,
    /// Non-events predicted as non-events.
    pub true_neg: f64,
}

impl Classification {
    /// Proportion of events correctly classified.
    pub fn sensitivity(&self) -> f64 {
        self.true_pos / (self.true_pos + self.false_neg)
    }
    /// Proportion of non-events correctly classified.
    pub fn specificity(&self) -> f64 {
        self.true_neg / (self.true_neg + self.false_pos)
    }
    /// Proportion classified correctly.
    pub fn accuracy(&self) -> f64 {
        (self.true_pos + self.true_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg)
    }
    /// Proportion of predicted events that are events.
    pub fn ppv(&self) -> f64 {
        self.true_pos / (self.true_pos + self.false_pos)
    }
    /// Proportion of predicted non-events that are non-events.
    pub fn npv(&self) -> f64 {
        self.true_neg / (self.true_neg + self.false_neg)
    }
}

/// Classification table at `cutoff`.
pub fn classification_table(fit: &GlmFit, cutoff: f64) -> Result<Classification> {
    let (k, m, mu) = binomial_parts(fit)?;
    if !(cutoff > 0.0 && cutoff < 1.0) {
        return invalid("the cutoff must be in (0, 1)");
    }
    let mut c = Classification { cutoff, true_pos: 0.0, false_neg: 0.0, false_pos: 0.0, true_neg: 0.0 };
    for i in 0..k.len() {
        if mu[i] >= cutoff {
            c.true_pos += k[i];
            c.false_pos += m[i] - k[i];
        } else {
            c.false_neg += k[i];
            c.true_neg += m[i] - k[i];
        }
    }
    Ok(c)
}

/// Receiver operating characteristic curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Roc {
    /// Cutoffs, decreasing (the first is above every prediction).
    pub thresholds: Vec<f64>,
    /// False-positive rate (1 − specificity) at each cutoff.
    pub fpr: Vec<f64>,
    /// True-positive rate (sensitivity) at each cutoff.
    pub tpr: Vec<f64>,
    /// Area under the curve.
    pub auc: f64,
}

/// ROC curve of the fitted probabilities.
pub fn roc_curve(fit: &GlmFit) -> Result<Roc> {
    let (k, m, mu) = binomial_parts(fit)?;
    let pos: f64 = k.iter().sum();
    let neg: f64 = m.iter().sum::<f64>() - pos;
    if pos == 0.0 || neg == 0.0 {
        return invalid("the ROC curve needs both events and non-events");
    }
    let mut order: Vec<usize> = (0..k.len()).collect();
    order.sort_by(|&a, &b| mu[b].total_cmp(&mu[a]));
    let mut thresholds = vec![f64::INFINITY];
    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let (mut tp, mut fp) = (0.0, 0.0);
    let mut i = 0;
    while i < order.len() {
        let v = mu[order[i]];
        while i < order.len() && mu[order[i]] == v {
            tp += k[order[i]];
            fp += m[order[i]] - k[order[i]];
            i += 1;
        }
        thresholds.push(v);
        fpr.push(fp / neg);
        tpr.push(tp / pos);
    }
    let auc = (1..fpr.len()).map(|j| (fpr[j] - fpr[j - 1]) * (tpr[j] + tpr[j - 1]) / 2.0).sum();
    Ok(Roc { thresholds, fpr, tpr, auc })
}

/// Pair counts and rank-correlation measures between predicted
/// probabilities and the response (Minitab's "Measures of Association").
#[derive(Debug, Clone, PartialEq)]
pub struct Association {
    /// Concordant pairs.
    pub concordant: f64,
    /// Discordant pairs.
    pub discordant: f64,
    /// Tied pairs.
    pub ties: f64,
    /// Number of event/non-event pairs.
    pub pairs: f64,
    /// Somers' D = (C − D) / pairs.
    pub somers_d: f64,
    /// Goodman–Kruskal γ = (C − D) / (C + D).
    pub gamma: f64,
    /// Kendall's τ-a = (C − D) / (N(N − 1)/2).
    pub tau_a: f64,
}

/// Measures of association of the fitted probabilities with the response.
pub fn association(fit: &GlmFit) -> Result<Association> {
    let (k, m, mu) = binomial_parts(fit)?;
    let pos: f64 = k.iter().sum();
    let total: f64 = m.iter().sum();
    let neg = total - pos;
    if pos == 0.0 || neg == 0.0 {
        return invalid("the association measures need both events and non-events");
    }
    let mut order: Vec<usize> = (0..k.len()).collect();
    order.sort_by(|&a, &b| mu[a].total_cmp(&mu[b]));
    let (mut conc, mut tie) = (0.0, 0.0);
    let mut neg_below = 0.0;
    let mut i = 0;
    while i < order.len() {
        let v = mu[order[i]];
        let (mut p_here, mut n_here) = (0.0, 0.0);
        while i < order.len() && mu[order[i]] == v {
            p_here += k[order[i]];
            n_here += m[order[i]] - k[order[i]];
            i += 1;
        }
        conc += p_here * neg_below;
        tie += p_here * n_here;
        neg_below += n_here;
    }
    let pairs = pos * neg;
    let disc = pairs - conc - tie;
    Ok(Association {
        concordant: conc,
        discordant: disc,
        ties: tie,
        pairs,
        somers_d: (conc - disc) / pairs,
        gamma: if conc + disc > 0.0 { (conc - disc) / (conc + disc) } else { f64::NAN },
        tau_a: (conc - disc) / (total * (total - 1.0) / 2.0),
    })
}
