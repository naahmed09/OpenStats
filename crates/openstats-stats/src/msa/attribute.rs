//! Attribute agreement analysis.
//!
//! Appraisers classify each sample on several trials into categories; an
//! optional known standard gives the correct category of every sample.
//!
//! * **Agreement**: a sample *matches* for an appraiser when all of that
//!   appraiser's trials agree (within appraiser), agree with the standard (vs
//!   standard), or when every appraiser's every trial agrees (between
//!   appraisers; all appraisers vs standard). Percentages carry exact
//!   (Clopper–Pearson) confidence intervals.
//! * **Fleiss' κ** (Fleiss 1971) for several raters with `m` ratings per sample:
//!   `P̄ = mean((Σⱼ nᵢⱼ² − m)/(m(m − 1)))`, `Pₑ = Σⱼ pⱼ²`, `κ = (P̄ − Pₑ)/(1 − Pₑ)`;
//!   per category `κⱼ = 1 − Σᵢ nᵢⱼ(m − nᵢⱼ)/(N m (m − 1) pⱼ qⱼ)`. The reported
//!   standard error and `Z = κ/SE` are those under H₀: κ = 0, with the one-sided
//!   p-value `P(Z > z)`. *Within appraiser*, the raters are the trials;
//!   *between appraisers*, every appraiser-trial rating of a sample is a rater.
//! * **Cohen's κ** (Cohen 1960) for two raters on the pooled pairs: the
//!   appraiser's rating on each trial against the standard (or against another
//!   appraiser's rating of the same trial). SE under H₀ from Fleiss, Cohen &
//!   Everitt (1969); the confidence interval uses the non-null large-sample
//!   variance of Fleiss (1981).
//! * **Effectiveness** = matched/inspected (vs standard); with a designated
//!   defect category, the **miss rate** is the share of assessments of
//!   defective samples (per the standard) rated non-defective, and the **false
//!   alarm rate** the share of assessments of non-defective samples rated
//!   defective; **mixed** counts the samples an appraiser did not rate the same
//!   on every trial.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::hypothesis::Alternative;
use crate::proportion::clopper_pearson;
use crate::report::{ChartKind, ChartSpec, Series};
use openstats_math::special::erf::{norm_cdf, norm_ppf, norm_sf};
use std::collections::BTreeMap;

/// A balanced attribute-agreement design.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeStudy {
    appraisers: Vec<String>,
    samples: Vec<String>,
    categories: Vec<String>,
    trials: usize,
    /// `ratings[a][s][t]` is a category index.
    ratings: Vec<Vec<Vec<usize>>>,
    /// Standard category of each sample.
    standard: Option<Vec<usize>>,
}

/// Options of an attribute agreement analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeOptions {
    /// Confidence level (0.95).
    pub level: f64,
    /// The category that denotes a defect, for miss and false-alarm rates.
    pub defect_category: Option<String>,
}

impl Default for AttributeOptions {
    fn default() -> Self {
        AttributeOptions { level: 0.95, defect_category: None }
    }
}

/// A kappa statistic of a category (or overall) with its test.
#[derive(Debug, Clone, PartialEq)]
pub struct KappaRow {
    /// Category label, or "Overall".
    pub category: String,
    /// κ (`NaN` when the raters used one category throughout and κ is undefined).
    pub kappa: f64,
    /// Standard error under H₀: κ = 0.
    pub se: f64,
    /// `κ/SE`.
    pub z: f64,
    /// One-sided p-value `P(Z > z)`.
    pub p: f64,
}

/// Cohen's κ of two raters with its tests and interval.
#[derive(Debug, Clone, PartialEq)]
pub struct CohenKappa {
    /// Number of rating pairs.
    pub n: usize,
    /// Observed proportion of agreement.
    pub p_observed: f64,
    /// Chance-expected proportion of agreement.
    pub p_expected: f64,
    /// κ (`NaN` when undefined).
    pub kappa: f64,
    /// Standard error under H₀.
    pub se_null: f64,
    /// `κ/SE₀`.
    pub z: f64,
    /// One-sided p-value `P(Z > z)`.
    pub p: f64,
    /// Non-null standard error.
    pub se: f64,
    /// Confidence interval `κ ± z·SE`.
    pub ci: Interval,
}

/// Matched samples out of inspected, with an exact interval.
#[derive(Debug, Clone, PartialEq)]
pub struct Agreement {
    /// Row label (an appraiser, or "All appraisers").
    pub label: String,
    /// Samples inspected.
    pub inspected: usize,
    /// Samples matched.
    pub matched: usize,
    /// `100·matched/inspected`.
    pub percent: f64,
    /// Exact confidence interval in percent.
    pub ci: Interval,
}

/// Misses and false alarms of one appraiser against the standard.
#[derive(Debug, Clone, PartialEq)]
pub struct Disagreement {
    /// Appraiser.
    pub appraiser: String,
    /// Assessments of defective samples.
    pub defect_assessments: usize,
    /// Of those, rated non-defective.
    pub misses: usize,
    /// `100·misses/defect_assessments` (`None` without defective samples).
    pub miss_rate: Option<f64>,
    /// Assessments of non-defective samples.
    pub good_assessments: usize,
    /// Of those, rated defective.
    pub false_alarms: usize,
    /// `100·false_alarms/good_assessments` (`None` without non-defective samples).
    pub false_alarm_rate: Option<f64>,
    /// Samples not rated identically on all trials.
    pub mixed: usize,
}

/// Within-appraiser results.
#[derive(Debug, Clone, PartialEq)]
pub struct WithinAppraiser {
    /// Percent of samples with all trials in agreement.
    pub agreement: Agreement,
    /// Fleiss' κ across the trials (`None` with a single trial).
    pub kappa: Option<Vec<KappaRow>>,
}

/// Appraiser-versus-standard results.
#[derive(Debug, Clone, PartialEq)]
pub struct VsStandard {
    /// Percent of samples with all trials equal to the standard.
    pub agreement: Agreement,
    /// Cohen's κ of the pooled trials against the standard.
    pub kappa: CohenKappa,
    /// Cohen's κ per category (category versus all others).
    pub kappa_by_category: Vec<(String, CohenKappa)>,
    /// Misses and false alarms (with a defect category).
    pub disagreement: Option<Disagreement>,
}

/// Between-appraiser results.
#[derive(Debug, Clone, PartialEq)]
pub struct BetweenAppraisers {
    /// Samples on which every appraiser agreed on every trial.
    pub agreement: Agreement,
    /// Fleiss' κ over all appraiser-trial ratings.
    pub kappa: Vec<KappaRow>,
}

/// Cohen's κ between two appraisers.
#[derive(Debug, Clone, PartialEq)]
pub struct PairKappa {
    /// First appraiser.
    pub a: String,
    /// Second appraiser.
    pub b: String,
    /// κ of the pooled trial pairs.
    pub kappa: CohenKappa,
}

/// Result of an attribute agreement analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeAgreement {
    /// Appraisers.
    pub appraisers: Vec<String>,
    /// Number of samples.
    pub samples: usize,
    /// Trials per appraiser and sample.
    pub trials: usize,
    /// Categories in sorted order.
    pub categories: Vec<String>,
    /// Confidence level.
    pub level: f64,
    /// Within appraisers, in appraiser order.
    pub within: Vec<WithinAppraiser>,
    /// Each appraiser versus the standard (with a standard).
    pub vs_standard: Option<Vec<VsStandard>>,
    /// Between appraisers.
    pub between: BetweenAppraisers,
    /// All appraisers versus the standard: agreement and pooled Cohen's κ.
    pub all_vs_standard: Option<(Agreement, CohenKappa)>,
    /// Cohen's κ for every pair of appraisers.
    pub pairs: Vec<PairKappa>,
}

impl AttributeStudy {
    /// Build a study from long-format columns: one row per rating, with the
    /// sample, the appraiser and the rating; the trial number is the order of
    /// a sample's rows within an appraiser. `standard` (optional) is the
    /// correct category on every row. The design must be balanced.
    pub fn from_long(
        sample: &[&str],
        appraiser: &[&str],
        rating: &[&str],
        standard: Option<&[&str]>,
    ) -> Result<AttributeStudy> {
        let n = sample.len();
        if appraiser.len() != n || rating.len() != n || standard.is_some_and(|s| s.len() != n) {
            return invalid("all columns must have the same length");
        }
        if n == 0 {
            return Err(StatsError::InsufficientData { what: "attribute agreement", needed: 1, got: 0 });
        }
        let mut appraisers: Vec<String> = Vec::new();
        let mut samples: Vec<String> = Vec::new();
        for i in 0..n {
            if !appraisers.iter().any(|a| a == appraiser[i]) {
                appraisers.push(appraiser[i].to_string());
            }
            if !samples.iter().any(|s| s == sample[i]) {
                samples.push(sample[i].to_string());
            }
        }
        let mut cats: Vec<String> = rating.iter().map(|s| s.to_string()).collect();
        if let Some(st) = standard {
            cats.extend(st.iter().map(|s| s.to_string()));
        }
        cats.sort();
        cats.dedup();
        let cat_ix: BTreeMap<&str, usize> = cats.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
        let mut cells: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); samples.len()]; appraisers.len()];
        let mut std_of: Vec<Option<usize>> = vec![None; samples.len()];
        for i in 0..n {
            let a = appraisers.iter().position(|x| x == appraiser[i]).unwrap_or(0);
            let s = samples.iter().position(|x| x == sample[i]).unwrap_or(0);
            cells[a][s].push(cat_ix[rating[i]]);
            if let Some(st) = standard {
                let c = cat_ix[st[i]];
                match std_of[s] {
                    None => std_of[s] = Some(c),
                    Some(prev) if prev != c => {
                        return invalid(format!("sample '{}' has more than one standard value", samples[s]))
                    }
                    _ => {}
                }
            }
        }
        let trials = cells[0][0].len();
        if trials == 0 {
            return invalid("every appraiser must rate every sample");
        }
        if cells.iter().flatten().any(|c| c.len() != trials) {
            return invalid("the design must be balanced: every appraiser rates every sample the same number of times");
        }
        Ok(AttributeStudy {
            appraisers,
            samples,
            categories: cats,
            trials,
            ratings: cells,
            standard: if standard.is_some() {
                Some(std_of.into_iter().map(|c| c.unwrap_or(0)).collect())
            } else {
                None
            },
        })
    }

    /// Number of appraisers.
    pub fn appraiser_count(&self) -> usize {
        self.appraisers.len()
    }

    /// Run the analysis.
    pub fn analyze(&self, opts: &AttributeOptions) -> Result<AttributeAgreement> {
        if !(opts.level > 0.0 && opts.level < 1.0) {
            return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
        }
        let ns = self.samples.len();
        let t = self.trials;
        let k = self.categories.len();
        let defect = match &opts.defect_category {
            Some(d) => match self.categories.iter().position(|c| c == d) {
                Some(i) => Some(i),
                None => return invalid(format!("defect category '{d}' does not occur in the data")),
            },
            None => None,
        };
        if defect.is_some() && self.standard.is_none() {
            return invalid("misses and false alarms need a standard");
        }
        let agreement = |label: &str, matched: usize, inspected: usize| -> Result<Agreement> {
            let ci = clopper_pearson(matched as u64, inspected as u64, Alternative::TwoSided, opts.level)?;
            Ok(Agreement {
                label: label.to_string(),
                inspected,
                matched,
                percent: 100.0 * matched as f64 / inspected as f64,
                ci: Interval { lower: 100.0 * ci.lower, upper: 100.0 * ci.upper, level: opts.level },
            })
        };
        let all_equal = |v: &[usize]| v.iter().all(|&x| x == v[0]);
        // Within appraisers.
        let mut within = Vec::new();
        for (a, name) in self.appraisers.iter().enumerate() {
            let matched = (0..ns).filter(|&s| all_equal(&self.ratings[a][s])).count();
            let kappa = if t >= 2 {
                let counts: Vec<Vec<usize>> = (0..ns).map(|s| tally(&self.ratings[a][s], k)).collect();
                Some(fleiss(&counts, &self.categories)?)
            } else {
                None
            };
            within.push(WithinAppraiser { agreement: agreement(name, matched, ns)?, kappa });
        }
        // Against the standard.
        let vs_standard = match &self.standard {
            None => None,
            Some(std) => {
                let mut out = Vec::new();
                for (a, name) in self.appraisers.iter().enumerate() {
                    let matched = (0..ns).filter(|&s| self.ratings[a][s].iter().all(|&r| r == std[s])).count();
                    let (mut x, mut y) = (Vec::new(), Vec::new());
                    for s in 0..ns {
                        for &r in &self.ratings[a][s] {
                            x.push(r);
                            y.push(std[s]);
                        }
                    }
                    let kappa = cohen(&x, &y, k, opts.level)?;
                    let mut by_cat = Vec::new();
                    for (c, label) in self.categories.iter().enumerate() {
                        let bx: Vec<usize> = x.iter().map(|&v| usize::from(v == c)).collect();
                        let by: Vec<usize> = y.iter().map(|&v| usize::from(v == c)).collect();
                        if by.contains(&1) {
                            by_cat.push((label.clone(), cohen(&bx, &by, 2, opts.level)?));
                        }
                    }
                    let disagreement = defect.map(|d| {
                        let (mut dn, mut dm, mut gn, mut gf) = (0, 0, 0, 0);
                        for s in 0..ns {
                            for &r in &self.ratings[a][s] {
                                if std[s] == d {
                                    dn += 1;
                                    dm += usize::from(r != d);
                                } else {
                                    gn += 1;
                                    gf += usize::from(r == d);
                                }
                            }
                        }
                        Disagreement {
                            appraiser: name.clone(),
                            defect_assessments: dn,
                            misses: dm,
                            miss_rate: (dn > 0).then(|| 100.0 * dm as f64 / dn as f64),
                            good_assessments: gn,
                            false_alarms: gf,
                            false_alarm_rate: (gn > 0).then(|| 100.0 * gf as f64 / gn as f64),
                            mixed: (0..ns).filter(|&s| !all_equal(&self.ratings[a][s])).count(),
                        }
                    });
                    out.push(VsStandard {
                        agreement: agreement(name, matched, ns)?,
                        kappa,
                        kappa_by_category: by_cat,
                        disagreement,
                    });
                }
                Some(out)
            }
        };
        // Between appraisers.
        let pooled: Vec<Vec<usize>> =
            (0..ns).map(|s| self.ratings.iter().flat_map(|r| r[s].iter().copied()).collect()).collect();
        let b_matched = pooled.iter().filter(|v| all_equal(v)).count();
        let b_counts: Vec<Vec<usize>> = pooled.iter().map(|v| tally(v, k)).collect();
        let between = BetweenAppraisers {
            agreement: agreement("All appraisers", b_matched, ns)?,
            kappa: fleiss(&b_counts, &self.categories)?,
        };
        let all_vs_standard = match &self.standard {
            None => None,
            Some(std) => {
                let matched = (0..ns).filter(|&s| pooled[s].iter().all(|&r| r == std[s])).count();
                let (mut x, mut y) = (Vec::new(), Vec::new());
                for s in 0..ns {
                    for &r in &pooled[s] {
                        x.push(r);
                        y.push(std[s]);
                    }
                }
                Some((agreement("All appraisers", matched, ns)?, cohen(&x, &y, k, opts.level)?))
            }
        };
        // Appraiser pairs, trial by trial.
        let mut pairs = Vec::new();
        for a in 0..self.appraisers.len() {
            for b in a + 1..self.appraisers.len() {
                let (mut x, mut y) = (Vec::new(), Vec::new());
                for s in 0..ns {
                    for tr in 0..t {
                        x.push(self.ratings[a][s][tr]);
                        y.push(self.ratings[b][s][tr]);
                    }
                }
                pairs.push(PairKappa {
                    a: self.appraisers[a].clone(),
                    b: self.appraisers[b].clone(),
                    kappa: cohen(&x, &y, k, opts.level)?,
                });
            }
        }
        Ok(AttributeAgreement {
            appraisers: self.appraisers.clone(),
            samples: ns,
            trials: t,
            categories: self.categories.clone(),
            level: opts.level,
            within,
            vs_standard,
            between,
            all_vs_standard,
            pairs,
        })
    }
}

fn tally(v: &[usize], k: usize) -> Vec<usize> {
    let mut c = vec![0; k];
    for &x in v {
        c[x] += 1;
    }
    c
}

/// Fleiss' κ for subjects rated by `m` raters; `counts[i][j]` = raters putting
/// subject `i` in category `j` (equal row sums). Categories nobody used are skipped.
pub(crate) fn fleiss(counts: &[Vec<usize>], labels: &[String]) -> Result<Vec<KappaRow>> {
    let n = counts.len();
    let m = counts[0].iter().sum::<usize>();
    if n < 2 || m < 2 {
        return invalid("Fleiss' kappa needs at least 2 subjects and 2 ratings per subject");
    }
    let nf = n as f64;
    let mf = m as f64;
    let k = labels.len();
    let p: Vec<f64> = (0..k).map(|j| counts.iter().map(|r| r[j]).sum::<usize>() as f64 / (nf * mf)).collect();
    let pbar =
        counts.iter().map(|r| (r.iter().map(|&c| (c * c) as f64).sum::<f64>() - mf) / (mf * (mf - 1.0))).sum::<f64>()
            / nf;
    let pe: f64 = p.iter().map(|x| x * x).sum();
    let mut rows = Vec::new();
    let denom = (nf * mf * (mf - 1.0)).sqrt();
    // Overall.
    let sum_pq: f64 = p.iter().map(|x| x * (1.0 - x)).sum();
    let kappa = if pe < 1.0 { (pbar - pe) / (1.0 - pe) } else { f64::NAN };
    let inner: f64 = p.iter().map(|&x| x * (1.0 - x) * ((1.0 - x) - x)).sum();
    let se = if sum_pq > 0.0 {
        (2.0_f64).sqrt() / (sum_pq * denom) * (sum_pq * sum_pq - inner).max(0.0).sqrt()
    } else {
        f64::NAN
    };
    rows.push(kappa_row("Overall", kappa, se));
    for j in 0..k {
        if p[j] <= 0.0 {
            continue;
        }
        let q = 1.0 - p[j];
        let kj = if q > 0.0 {
            1.0 - counts.iter().map(|r| (r[j] * (m - r[j])) as f64).sum::<f64>() / (nf * mf * (mf - 1.0) * p[j] * q)
        } else {
            f64::NAN
        };
        let sej = (2.0 / (nf * mf * (mf - 1.0))).sqrt();
        rows.push(kappa_row(&labels[j], kj, sej));
    }
    Ok(rows)
}

fn kappa_row(label: &str, kappa: f64, se: f64) -> KappaRow {
    let z = kappa / se;
    KappaRow { category: label.to_string(), kappa, se, z, p: if z.is_nan() { f64::NAN } else { norm_sf(z) } }
}

/// Cohen's κ of two raters' paired ratings with `k` categories.
pub(crate) fn cohen(x: &[usize], y: &[usize], k: usize, level: f64) -> Result<CohenKappa> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "Cohen's kappa", needed: 2, got: n });
    }
    let nf = n as f64;
    let mut pij = vec![vec![0.0; k]; k];
    for (&a, &b) in x.iter().zip(y) {
        pij[a][b] += 1.0 / nf;
    }
    let row: Vec<f64> = (0..k).map(|i| pij[i].iter().sum()).collect();
    let col: Vec<f64> = (0..k).map(|j| (0..k).map(|i| pij[i][j]).sum()).collect();
    let po: f64 = (0..k).map(|i| pij[i][i]).sum();
    let pe: f64 = (0..k).map(|i| row[i] * col[i]).sum();
    let z975 = norm_ppf(1.0 - (1.0 - level) / 2.0);
    if pe >= 1.0 - 1e-15 {
        return Ok(CohenKappa {
            n,
            p_observed: po,
            p_expected: pe,
            kappa: f64::NAN,
            se_null: f64::NAN,
            z: f64::NAN,
            p: f64::NAN,
            se: f64::NAN,
            ci: Interval { lower: f64::NAN, upper: f64::NAN, level },
        });
    }
    let kappa = (po - pe) / (1.0 - pe);
    let t: f64 = (0..k).map(|i| row[i] * col[i] * (row[i] + col[i])).sum();
    let var0 = (pe + pe * pe - t).max(0.0) / (nf * (1.0 - pe) * (1.0 - pe));
    let se0 = var0.sqrt();
    let a: f64 = (0..k).map(|i| pij[i][i] * (1.0 - (row[i] + col[i]) * (1.0 - kappa)).powi(2)).sum();
    let mut b = 0.0;
    for i in 0..k {
        for j in 0..k {
            if i != j {
                b += pij[i][j] * (col[i] + row[j]).powi(2);
            }
        }
    }
    let c = (kappa - pe * (1.0 - kappa)).powi(2);
    let var = (a + (1.0 - kappa).powi(2) * b - c).max(0.0) / (nf * (1.0 - pe) * (1.0 - pe));
    let se = var.sqrt();
    let z = if se0 > 0.0 { kappa / se0 } else { f64::NAN };
    Ok(CohenKappa {
        n,
        p_observed: po,
        p_expected: pe,
        kappa,
        se_null: se0,
        z,
        p: if z.is_nan() { f64::NAN } else { 1.0 - norm_cdf(z) },
        se,
        ci: Interval { lower: kappa - z975 * se, upper: kappa + z975 * se, level },
    })
}

impl AttributeAgreement {
    /// Percent-agreement interval plots: within appraisers and, with a standard, against it.
    pub fn charts(&self) -> Vec<ChartSpec> {
        let plot = |title: &str, rows: Vec<&Agreement>| -> ChartSpec {
            let pos: Vec<f64> = (1..=rows.len()).map(|i| i as f64).collect();
            ChartSpec {
                kind: ChartKind::IntervalPlot,
                title: title.into(),
                x_label: "Percent".into(),
                y_label: "Appraiser".into(),
                categories: rows.iter().map(|r| r.label.clone()).collect(),
                series: vec![
                    Series { name: "estimate".into(), x: rows.iter().map(|r| r.percent).collect(), y: pos.clone() },
                    Series { name: "lower".into(), x: rows.iter().map(|r| r.ci.lower).collect(), y: pos.clone() },
                    Series { name: "upper".into(), x: rows.iter().map(|r| r.ci.upper).collect(), y: pos },
                ],
                reference_lines: Vec::new(),
            }
        };
        let mut v =
            vec![plot("Assessment Agreement: Within Appraisers", self.within.iter().map(|w| &w.agreement).collect())];
        if let Some(vs) = &self.vs_standard {
            v.push(plot("Assessment Agreement: Appraiser vs Standard", vs.iter().map(|w| &w.agreement).collect()));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fleiss_textbook_example() {
        // Fleiss' kappa worked example: 10 subjects, 14 raters, 5 categories → κ ≈ 0.210.
        let counts: Vec<Vec<usize>> = vec![
            vec![0, 0, 0, 0, 14],
            vec![0, 2, 6, 4, 2],
            vec![0, 0, 3, 5, 6],
            vec![0, 3, 9, 2, 0],
            vec![2, 2, 8, 1, 1],
            vec![7, 7, 0, 0, 0],
            vec![3, 2, 6, 3, 0],
            vec![2, 5, 3, 2, 2],
            vec![6, 5, 2, 1, 0],
            vec![0, 2, 2, 3, 7],
        ];
        let labels: Vec<String> = (1..=5).map(|i| i.to_string()).collect();
        let r = fleiss(&counts, &labels).unwrap();
        assert!((r[0].kappa - 0.2099).abs() < 5e-4, "{}", r[0].kappa);
        // Category κ's from the same example: category 1 is 0.2 – 0.3 range; they are all finite.
        assert!(r.iter().all(|x| x.kappa.is_finite() && x.se > 0.0));
        assert!(fleiss(&counts[..1], &labels).is_err());
    }

    #[test]
    fn cohen_textbook_example() {
        // 50 subjects: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no → po = 0.7, pe = 0.5, κ = 0.4.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (a, b, n) in [(0, 0, 20), (0, 1, 5), (1, 0, 10), (1, 1, 15)] {
            for _ in 0..n {
                x.push(a);
                y.push(b);
            }
        }
        let c = cohen(&x, &y, 2, 0.95).unwrap();
        assert!(
            (c.kappa - 0.4).abs() < 1e-12 && (c.p_observed - 0.7).abs() < 1e-12 && (c.p_expected - 0.5).abs() < 1e-12
        );
        assert!(c.ci.lower < 0.4 && 0.4 < c.ci.upper && c.p < 0.01);
        // The same raters on constant ratings have κ undefined.
        let k = cohen(&[0, 0, 0], &[0, 0, 0], 2, 0.95).unwrap();
        assert!(k.kappa.is_nan() && k.ci.lower.is_nan());
    }
}
