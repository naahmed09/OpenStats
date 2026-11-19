//! Multiple comparisons of means: Tukey (Tukey–Kramer), Fisher's LSD,
//! Dunnett's comparisons with a control and Hsu's multiple comparisons with
//! the best (MCB), together with the grouping-letter tables.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::special::dunnett::Dunnett;
use openstats_math::special::tukey::{ptukey, qtukey};

/// Group means with the error estimate they are compared with.
#[derive(Debug, Clone, PartialEq)]
pub struct Means {
    /// Level labels.
    pub labels: Vec<String>,
    /// Number of observations per level.
    pub n: Vec<f64>,
    /// Level means (least-squares means for a model with other terms).
    pub mean: Vec<f64>,
    /// Mean square error.
    pub mse: f64,
    /// Error degrees of freedom.
    pub df: f64,
}

impl Means {
    /// Validate and build.
    pub fn new(labels: Vec<String>, n: Vec<f64>, mean: Vec<f64>, mse: f64, df: f64) -> Result<Means> {
        let k = labels.len();
        if k < 2 || n.len() != k || mean.len() != k {
            return invalid("multiple comparisons need at least two levels with matching labels, sizes and means");
        }
        if n.iter().any(|&x| x.is_nan() || x <= 0.0) || mean.iter().any(|x| !x.is_finite()) {
            return invalid("level sizes must be positive and means finite");
        }
        if !(mse.is_finite() && mse > 0.0) || df.is_nan() || df < 1.0 {
            return invalid("multiple comparisons need MSE > 0 and error df ≥ 1");
        }
        Ok(Means { labels, n, mean, mse, df })
    }

    fn k(&self) -> usize {
        self.labels.len()
    }

    fn se(&self, i: usize, j: usize) -> f64 {
        (self.mse * (1.0 / self.n[i] + 1.0 / self.n[j])).sqrt()
    }
}

/// Which family of comparisons a result belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Tukey's honestly significant difference (Tukey–Kramer for unequal sizes).
    Tukey,
    /// Fisher's least significant difference (no multiplicity adjustment).
    Fisher,
    /// Dunnett's comparisons with a control.
    Dunnett,
}

/// Direction of Dunnett's comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    /// Two-sided intervals and tests.
    TwoSided,
    /// `μᵢ > μ_control` (lower confidence bounds).
    Greater,
    /// `μᵢ < μ_control` (upper confidence bounds).
    Less,
}

/// Which end of the scale is "best" in Hsu's MCB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Best {
    /// The largest mean is best.
    Largest,
    /// The smallest mean is best.
    Smallest,
}

/// One difference of means.
#[derive(Debug, Clone, PartialEq)]
pub struct Difference {
    /// Index of the level on the left (`level − reference`).
    pub level: usize,
    /// Index of the reference level.
    pub reference: usize,
    /// Difference of the sample means.
    pub diff: f64,
    /// Standard error of the difference.
    pub se: f64,
    /// t value of the difference.
    pub t: f64,
    /// p-value adjusted for the multiplicity of the method.
    pub p_adjusted: f64,
    /// Interval for `μ_level − μ_reference` (simultaneous for Tukey and
    /// Dunnett, individual for Fisher; one-sided Dunnett intervals have an
    /// infinite far end).
    pub interval: Interval,
}

/// One row of a grouping-information table.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupingRow {
    /// Level index.
    pub index: usize,
    /// Level label.
    pub label: String,
    /// Number of observations.
    pub n: f64,
    /// Mean.
    pub mean: f64,
    /// Grouping letters; levels sharing a letter are not significantly different.
    pub letters: String,
}

/// Result of Tukey, Fisher or Dunnett comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparisons {
    /// Method.
    pub method: Method,
    /// Direction (always two-sided for Tukey and Fisher).
    pub alternative: Alternative,
    /// Family confidence level (simultaneous for Tukey and Dunnett).
    pub confidence: f64,
    /// Confidence level of each individual interval implied by the critical value.
    pub individual_confidence: f64,
    /// Critical value (in t units: `q/√2` for Tukey).
    pub critical: f64,
    /// Control level (Dunnett only).
    pub control: Option<usize>,
    /// The differences: all pairs `j − i` (`i < j`) for Tukey and Fisher, every level against the control for Dunnett.
    pub differences: Vec<Difference>,
    /// Grouping information, levels sorted by decreasing mean.
    pub grouping: Vec<GroupingRow>,
    /// The levels and error estimate that were compared.
    pub means: Means,
}

fn check_alpha(alpha: f64) -> Result<()> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        invalid(format!("significance level {alpha} must be in (0, 1)"))
    }
}

/// Grouping letters from a significance matrix (Piepho-style connected runs
/// over the levels sorted by decreasing mean).
fn letters(means: &Means, significant: &dyn Fn(usize, usize) -> bool) -> Result<Vec<GroupingRow>> {
    let k = means.k();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| means.mean[b].total_cmp(&means.mean[a]).then(a.cmp(&b)));
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for s in 0..k {
        let mut e = s;
        while e + 1 < k && (s..=e).all(|i| !significant(order[i], order[e + 1])) {
            e += 1;
        }
        if runs.last().is_none_or(|r| r.1 < e) {
            runs.push((s, e));
        }
    }
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    if runs.len() > ALPHABET.len() {
        return Err(StatsError::TooLarge(format!("{} grouping letters are needed", runs.len())));
    }
    let mut out: Vec<GroupingRow> = order
        .iter()
        .map(|&i| GroupingRow {
            index: i,
            label: means.labels[i].clone(),
            n: means.n[i],
            mean: means.mean[i],
            letters: String::new(),
        })
        .collect();
    for (r, &(s, e)) in runs.iter().enumerate() {
        for row in &mut out[s..=e] {
            row.letters.push(char::from(ALPHABET[r]));
        }
    }
    Ok(out)
}

fn pair_list(k: usize) -> Vec<(usize, usize)> {
    (0..k).flat_map(|i| (i + 1..k).map(move |j| (i, j))).collect()
}

/// Tukey (Tukey–Kramer) simultaneous comparisons of all pairs at family
/// confidence `1 − alpha`.
pub fn tukey(means: &Means, alpha: f64) -> Result<Comparisons> {
    check_alpha(alpha)?;
    let k = means.k() as f64;
    let crit = qtukey(1.0 - alpha, k, means.df)? / std::f64::consts::SQRT_2;
    let mut differences = Vec::new();
    for (i, j) in pair_list(means.k()) {
        let diff = means.mean[j] - means.mean[i];
        let se = means.se(i, j);
        let t = diff / se;
        let p = (1.0 - ptukey(t.abs() * std::f64::consts::SQRT_2, k, means.df)?).clamp(0.0, 1.0);
        differences.push(Difference {
            level: j,
            reference: i,
            diff,
            se,
            t,
            p_adjusted: p,
            interval: Interval { lower: diff - crit * se, upper: diff + crit * se, level: 1.0 - alpha },
        });
    }
    let individual = 1.0 - 2.0 * StudentT::new(means.df)?.sf(crit);
    let sig = |a: usize, b: usize| {
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        differences.iter().find(|d| d.reference == i && d.level == j).is_some_and(|d| d.p_adjusted < alpha)
    };
    let grouping = letters(means, &sig)?;
    Ok(Comparisons {
        method: Method::Tukey,
        alternative: Alternative::TwoSided,
        confidence: 1.0 - alpha,
        individual_confidence: individual,
        critical: crit,
        control: None,
        differences,
        grouping,
        means: means.clone(),
    })
}

/// Fisher's least-significant-difference comparisons of all pairs; every
/// interval and p-value is individual (not adjusted for multiplicity).
pub fn fisher(means: &Means, alpha: f64) -> Result<Comparisons> {
    check_alpha(alpha)?;
    let t_dist = StudentT::new(means.df)?;
    let crit = t_dist.isf(alpha / 2.0);
    let mut differences = Vec::new();
    for (i, j) in pair_list(means.k()) {
        let diff = means.mean[j] - means.mean[i];
        let se = means.se(i, j);
        let t = diff / se;
        differences.push(Difference {
            level: j,
            reference: i,
            diff,
            se,
            t,
            p_adjusted: (2.0 * t_dist.sf(t.abs())).min(1.0),
            interval: Interval { lower: diff - crit * se, upper: diff + crit * se, level: 1.0 - alpha },
        });
    }
    let sig = |a: usize, b: usize| {
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        differences.iter().find(|d| d.reference == i && d.level == j).is_some_and(|d| d.p_adjusted < alpha)
    };
    let grouping = letters(means, &sig)?;
    Ok(Comparisons {
        method: Method::Fisher,
        alternative: Alternative::TwoSided,
        confidence: 1.0 - alpha,
        individual_confidence: 1.0 - alpha,
        critical: crit,
        control: None,
        differences,
        grouping,
        means: means.clone(),
    })
}

/// Dunnett's simultaneous comparisons of every level with level `control`.
pub fn dunnett(means: &Means, control: usize, alternative: Alternative, alpha: f64) -> Result<Comparisons> {
    check_alpha(alpha)?;
    let k = means.k();
    if control >= k {
        return invalid(format!("control level {control} is out of range (0..{k})"));
    }
    let others: Vec<usize> = (0..k).filter(|&i| i != control).collect();
    let sizes: Vec<f64> = others.iter().map(|&i| means.n[i]).collect();
    let dist = Dunnett::from_sizes(means.n[control], &sizes, means.df)?;
    let crit = match alternative {
        Alternative::TwoSided => dist.quantile_two_sided(1.0 - alpha)?,
        _ => dist.quantile_one_sided(1.0 - alpha)?,
    };
    let mut differences = Vec::new();
    for &i in &others {
        let diff = means.mean[i] - means.mean[control];
        let se = means.se(i, control);
        let t = diff / se;
        let (p, interval) = match alternative {
            Alternative::TwoSided => (
                1.0 - dist.cdf_two_sided(t.abs()),
                Interval { lower: diff - crit * se, upper: diff + crit * se, level: 1.0 - alpha },
            ),
            Alternative::Greater => (
                1.0 - dist.cdf_one_sided(t),
                Interval { lower: diff - crit * se, upper: f64::INFINITY, level: 1.0 - alpha },
            ),
            Alternative::Less => (
                1.0 - dist.cdf_one_sided(-t),
                Interval { lower: f64::NEG_INFINITY, upper: diff + crit * se, level: 1.0 - alpha },
            ),
        };
        differences.push(Difference {
            level: i,
            reference: control,
            diff,
            se,
            t,
            p_adjusted: p.clamp(0.0, 1.0),
            interval,
        });
    }
    let individual = match alternative {
        Alternative::TwoSided => 1.0 - 2.0 * StudentT::new(means.df)?.sf(crit),
        _ => 1.0 - StudentT::new(means.df)?.sf(crit),
    };
    // Control and the levels not different from it share "A"; the others get "B".
    let mut grouping: Vec<GroupingRow> = {
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| means.mean[b].total_cmp(&means.mean[a]).then(a.cmp(&b)));
        order
            .into_iter()
            .map(|i| {
                let differs = differences.iter().any(|d| d.level == i && d.p_adjusted < alpha);
                GroupingRow {
                    index: i,
                    label: means.labels[i].clone(),
                    n: means.n[i],
                    mean: means.mean[i],
                    letters: if differs { "B".into() } else { "A".into() },
                }
            })
            .collect()
    };
    if grouping.iter().all(|g| g.letters == "B") {
        for g in &mut grouping {
            g.letters = "A".into();
        }
    }
    Ok(Comparisons {
        method: Method::Dunnett,
        alternative,
        confidence: 1.0 - alpha,
        individual_confidence: individual,
        critical: crit,
        control: Some(control),
        differences,
        grouping,
        means: means.clone(),
    })
}

/// One level of Hsu's multiple comparisons with the best.
#[derive(Debug, Clone, PartialEq)]
pub struct McbRow {
    /// Level index.
    pub index: usize,
    /// Level label.
    pub label: String,
    /// Lower end of the constrained interval for `μᵢ − best of the others` (in the "best is largest" sense; negated for [`Best::Smallest`]).
    pub lower: f64,
    /// Sample difference from the best of the other levels.
    pub center: f64,
    /// Upper end of the constrained interval.
    pub upper: f64,
    /// Dunnett one-sided critical value used for this level.
    pub critical: f64,
    /// `false` when the level is significantly worse than the best (upper end of the constrained interval is 0).
    pub possibly_best: bool,
    /// `true` when the level is significantly better than every other (lower end above 0).
    pub significantly_best: bool,
}

/// Result of Hsu's multiple comparisons with the best.
#[derive(Debug, Clone, PartialEq)]
pub struct McbResult {
    /// Which end of the scale is best.
    pub best: Best,
    /// Family confidence level.
    pub confidence: f64,
    /// One row per level.
    pub rows: Vec<McbRow>,
}

/// Hsu's MCB: simultaneous one-sided intervals for `μᵢ − max_{j≠i} μⱼ`
/// (or the minimum), constrained to contain 0 at an end as Hsu (1984)
/// prescribes.
pub fn hsu_mcb(means: &Means, best: Best, alpha: f64) -> Result<McbResult> {
    check_alpha(alpha)?;
    let k = means.k();
    let sgn = if best == Best::Largest { 1.0 } else { -1.0 };
    let mut rows = Vec::with_capacity(k);
    for i in 0..k {
        let others: Vec<usize> = (0..k).filter(|&j| j != i).collect();
        let sizes: Vec<f64> = others.iter().map(|&j| means.n[j]).collect();
        let crit = Dunnett::from_sizes(means.n[i], &sizes, means.df)?.quantile_one_sided(1.0 - alpha)?;
        let mut lo = f64::INFINITY;
        let mut up = f64::INFINITY;
        for &j in &others {
            let d = sgn * (means.mean[i] - means.mean[j]);
            let se = means.se(i, j);
            lo = lo.min(d - crit * se);
            up = up.min(d + crit * se);
        }
        let jb = others
            .iter()
            .copied()
            .max_by(|&a, &b| (sgn * means.mean[a]).total_cmp(&(sgn * means.mean[b])).then(b.cmp(&a)))
            .unwrap_or(i);
        let center = sgn * (means.mean[i] - means.mean[jb]);
        rows.push(McbRow {
            index: i,
            label: means.labels[i].clone(),
            lower: lo.min(0.0),
            center,
            upper: up.max(0.0),
            critical: crit,
            possibly_best: up > 0.0,
            significantly_best: lo > 0.0,
        });
    }
    Ok(McbResult { best, confidence: 1.0 - alpha, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn means() -> Means {
        Means::new(
            ["a", "b", "c", "d"].map(String::from).to_vec(),
            vec![5.0; 4],
            vec![10.0, 10.5, 14.0, 14.2],
            1.0,
            16.0,
        )
        .unwrap()
    }

    #[test]
    fn grouping_letters_connect_runs() {
        let t = tukey(&means(), 0.05).unwrap();
        let by: Vec<(&str, &str)> = t.grouping.iter().map(|g| (g.label.as_str(), g.letters.as_str())).collect();
        assert_eq!(by, vec![("d", "A"), ("c", "A"), ("b", "B"), ("a", "B")]);
        let f = fisher(&means(), 0.05).unwrap();
        assert_eq!(f.grouping.len(), 4);
        assert!(f.individual_confidence == 0.95);
        assert!(t.individual_confidence > 0.98 && t.individual_confidence < 0.995);
    }

    #[test]
    fn dunnett_and_mcb_basics() {
        let d = dunnett(&means(), 0, Alternative::TwoSided, 0.05).unwrap();
        assert_eq!(d.differences.len(), 3);
        assert!(d.differences[0].p_adjusted > 0.5 && d.differences[1].p_adjusted < 0.01);
        assert_eq!(d.grouping.iter().find(|g| g.index == 0).unwrap().letters, "A");
        assert!(dunnett(&means(), 9, Alternative::Less, 0.05).is_err());
        let g = dunnett(&means(), 0, Alternative::Greater, 0.05).unwrap();
        assert!(g.differences[0].interval.upper.is_infinite());
        let m = hsu_mcb(&means(), Best::Largest, 0.05).unwrap().rows;
        assert!(m[3].possibly_best && m[2].possibly_best && !m[0].possibly_best);
        assert!(m.iter().all(|r| r.lower <= 0.0 || r.significantly_best));
        let s = hsu_mcb(&means(), Best::Smallest, 0.05).unwrap().rows;
        assert!(s[0].possibly_best && !s[3].possibly_best);
    }

    #[test]
    fn invalid_inputs() {
        assert!(Means::new(vec!["a".into()], vec![1.0], vec![1.0], 1.0, 2.0).is_err());
        assert!(Means::new(vec!["a".into(), "b".into()], vec![1.0, 2.0], vec![1.0, 2.0], 0.0, 2.0).is_err());
        assert!(tukey(&means(), 1.5).is_err());
    }
}
