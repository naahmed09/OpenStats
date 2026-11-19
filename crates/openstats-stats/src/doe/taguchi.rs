//! Taguchi orthogonal arrays, signal-to-noise ratios and robust-design analysis.
//!
//! Arrays available: L4 (2³), L8 (2⁷), L9 (3⁴), L12 (2¹¹), L16 (2¹⁵), L18
//! (2¹ 3⁷), L25 (5⁶), L27 (3¹³), L32 (2³¹) and L81 (3⁴⁰). L4, L8, L16, L32
//! are the linear arrays of the GF(2) construction; L9, L25, L27, L81 the
//! GF(p) arrays (each column a non-zero coefficient vector, first non-zero
//! entry 1); L12 comes from a Hadamard matrix; L18 is tabulated. There is no
//! standard "L11" array (the 11-column 12-run array is L12). Arrays with
//! four-level or mixed 2/4 columns (L16 4⁵, L32 2¹ 4⁹, L36, L50, L64) are not
//! provided.

use super::hadamard::{hadamard, two_level_columns};
use crate::error::{invalid, Result};
use crate::report::{ChartKind, ChartSpec, Series};
use openstats_math::dist::{ContinuousDistribution, FisherF};

/// An orthogonal array: `data[run][column]` with levels 0-based.
#[derive(Debug, Clone, PartialEq)]
pub struct OrthogonalArray {
    /// Name ("L9").
    pub name: String,
    /// Number of levels of each column.
    pub levels: Vec<usize>,
    /// Entries (run-major).
    pub data: Vec<Vec<usize>>,
}

/// Names of the arrays that can be built.
pub fn available_arrays() -> &'static [&'static str] {
    &["L4", "L8", "L9", "L12", "L16", "L18", "L25", "L27", "L32", "L81", "L16(4^5)", "L64(4^21)"]
}

fn gf2_array(m: usize, name: &str) -> OrthogonalArray {
    let runs = 1usize << m;
    let cols = runs - 1;
    // Column c (1-based): parity of the run bits selected by c; bit 0 of c selects the slowest-changing run bit.
    let data = (0..runs)
        .map(|r| {
            (1..=cols)
                .map(|c| {
                    let mut parity = 0usize;
                    for i in 0..m {
                        if c >> i & 1 == 1 {
                            parity ^= r >> (m - 1 - i) & 1;
                        }
                    }
                    parity
                })
                .collect()
        })
        .collect();
    OrthogonalArray { name: name.into(), levels: vec![2; cols], data }
}

fn gfp_array(p: usize, m: usize, name: &str) -> OrthogonalArray {
    let runs = p.pow(m as u32);
    // Normalised coefficient vectors (first non-zero entry 1), in lexicographic order.
    let mut vecs: Vec<Vec<usize>> = Vec::new();
    for v in 1..runs {
        let digits: Vec<usize> = (0..m).map(|i| v / p.pow((m - 1 - i) as u32) % p).collect();
        if digits.iter().find(|&&d| d != 0) == Some(&1) {
            vecs.push(digits);
        }
    }
    let data = (0..runs)
        .map(|r| {
            let x: Vec<usize> = (0..m).map(|i| r / p.pow((m - 1 - i) as u32) % p).collect();
            vecs.iter().map(|c| c.iter().zip(&x).map(|(a, b)| a * b).sum::<usize>() % p).collect()
        })
        .collect();
    OrthogonalArray { name: name.into(), levels: vec![p; vecs.len()], data }
}

fn gf4_mul(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    const EXP: [usize; 3] = [1, 2, 3];
    EXP[(a - 1 + b - 1) % 3]
}

/// Orthogonal array with 4-level columns from GF(4): `4^m` runs and `(4^m − 1)/3` columns.
fn gf4_array(m: usize, name: &str) -> OrthogonalArray {
    let runs = 4usize.pow(m as u32);
    let digit = |v: usize, i: usize| v / 4usize.pow((m - 1 - i) as u32) % 4;
    let vecs: Vec<usize> = (1..runs).filter(|&v| (0..m).map(|i| digit(v, i)).find(|&d| d != 0) == Some(1)).collect();
    let data = (0..runs)
        .map(|r| vecs.iter().map(|&c| (0..m).fold(0usize, |acc, i| acc ^ gf4_mul(digit(c, i), digit(r, i)))).collect())
        .collect();
    OrthogonalArray { name: name.into(), levels: vec![4; vecs.len()], data }
}

const L18: &str = "\
1 1 1 1 1 1 1 1
1 1 2 2 2 2 2 2
1 1 3 3 3 3 3 3
1 2 1 1 2 2 3 3
1 2 2 2 3 3 1 1
1 2 3 3 1 1 2 2
1 3 1 2 1 3 2 3
1 3 2 3 2 1 3 1
1 3 3 1 3 2 1 2
2 1 1 3 3 2 2 1
2 1 2 1 1 3 3 2
2 1 3 2 2 1 1 3
2 2 1 2 3 1 3 2
2 2 2 3 1 2 1 3
2 2 3 1 2 3 2 1
2 3 1 3 2 3 1 2
2 3 2 1 3 1 2 3
2 3 3 2 1 2 3 1";

/// Build an orthogonal array by name.
pub fn orthogonal_array(name: &str) -> Result<OrthogonalArray> {
    let n = name.trim().to_uppercase();
    Ok(match n.as_str() {
        "L4" => gf2_array(2, "L4"),
        "L8" => gf2_array(3, "L8"),
        "L16" => gf2_array(4, "L16"),
        "L32" => gf2_array(5, "L32"),
        "L9" => gfp_array(3, 2, "L9"),
        "L27" => gfp_array(3, 3, "L27"),
        "L81" => gfp_array(3, 4, "L81"),
        "L25" => gfp_array(5, 2, "L25"),
        "L16(4^5)" => gf4_array(2, "L16(4^5)"),
        "L64(4^21)" => gf4_array(3, "L64(4^21)"),
        "L12" => {
            let h = hadamard(12).ok_or_else(|| crate::error::StatsError::InvalidArgument("no L12".into()))?;
            let cols = two_level_columns(&h);
            OrthogonalArray {
                name: "L12".into(),
                levels: vec![2; 11],
                data: cols.iter().map(|r| r.iter().map(|&v| usize::from(v > 0)).collect()).collect(),
            }
        }
        "L18" => {
            let data: Vec<Vec<usize>> = L18
                .lines()
                .map(|l| l.split_whitespace().map(|v| v.parse::<usize>().unwrap_or(1) - 1).collect())
                .collect();
            let mut levels = vec![3; 8];
            levels[0] = 2;
            OrthogonalArray { name: "L18".into(), levels, data }
        }
        _ => return invalid(format!("unknown array '{name}'; available: {}", available_arrays().join(", "))),
    })
}

impl OrthogonalArray {
    /// Number of runs.
    pub fn runs(&self) -> usize {
        self.data.len()
    }

    /// Number of columns.
    pub fn columns(&self) -> usize {
        self.levels.len()
    }

    /// Whether every pair of columns contains each level combination equally often (strength 2).
    pub fn is_orthogonal(&self) -> bool {
        let n = self.runs();
        for a in 0..self.columns() {
            for b in a + 1..self.columns() {
                let (la, lb) = (self.levels[a], self.levels[b]);
                if n % (la * lb) != 0 {
                    return false;
                }
                let mut cnt = vec![0usize; la * lb];
                for r in &self.data {
                    cnt[r[a] * lb + r[b]] += 1;
                }
                if cnt.iter().any(|&c| c != n / (la * lb)) {
                    return false;
                }
            }
        }
        true
    }
}

/// A Taguchi design: factors assigned to the columns of an array.
#[derive(Debug, Clone, PartialEq)]
pub struct TaguchiDesign {
    /// The array.
    pub array: OrthogonalArray,
    /// Factor names.
    pub factors: Vec<String>,
    /// Array column of each factor.
    pub columns: Vec<usize>,
    /// Levels (1-based) of each factor in each run.
    pub levels: Vec<Vec<usize>>,
    /// Number of levels of each factor.
    pub n_levels: Vec<usize>,
}

impl TaguchiDesign {
    /// Assign each `(name, levels)` factor to the first unused column with that many levels.
    pub fn new(array: OrthogonalArray, factors: &[(&str, usize)]) -> Result<TaguchiDesign> {
        if factors.is_empty() {
            return invalid("at least one factor is needed");
        }
        let mut used = vec![false; array.columns()];
        let mut cols = Vec::new();
        for (i, (name, lv)) in factors.iter().enumerate() {
            if name.is_empty() || factors[..i].iter().any(|f| f.0 == *name) {
                return invalid("factor names must be non-empty and distinct");
            }
            match (0..array.columns()).find(|&c| !used[c] && array.levels[c] == *lv) {
                Some(c) => {
                    used[c] = true;
                    cols.push(c);
                }
                None => {
                    return invalid(format!("{} has no free {lv}-level column left for factor '{name}'", array.name))
                }
            }
        }
        let levels = array.data.iter().map(|r| cols.iter().map(|&c| r[c] + 1).collect()).collect();
        Ok(TaguchiDesign {
            factors: factors.iter().map(|f| f.0.to_string()).collect(),
            n_levels: factors.iter().map(|f| f.1).collect(),
            columns: cols,
            levels,
            array,
        })
    }

    /// Number of runs.
    pub fn runs(&self) -> usize {
        self.levels.len()
    }
}

/// Signal-to-noise ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalToNoise {
    /// `−10 log₁₀(Σyᵢ²/n)`.
    SmallerIsBetter,
    /// `−10 log₁₀(Σ(1/yᵢ²)/n)`; needs y > 0.
    LargerIsBetter,
    /// `10 log₁₀(ȳ²/s²)`; needs ≥ 2 replicates.
    NominalIsBest,
    /// `−10 log₁₀(s²)`; needs ≥ 2 replicates (for a target reached by an adjustment factor).
    NominalIsBestVariance,
}

impl SignalToNoise {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            SignalToNoise::SmallerIsBetter => "Smaller is better",
            SignalToNoise::LargerIsBetter => "Larger is better",
            SignalToNoise::NominalIsBest => "Nominal is best (10 log(ybar^2/s^2))",
            SignalToNoise::NominalIsBestVariance => "Nominal is best (-10 log(s^2))",
        }
    }

    /// The ratio of one run's replicates.
    pub fn ratio(self, y: &[f64]) -> Result<f64> {
        let n = y.len() as f64;
        if y.is_empty() || y.iter().any(|v| !v.is_finite()) {
            return invalid("every run needs finite responses");
        }
        let mean = y.iter().sum::<f64>() / n;
        let var = || -> Result<f64> {
            if y.len() < 2 {
                return invalid("this signal-to-noise ratio needs at least 2 replicates per run");
            }
            let v = y.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / (n - 1.0);
            if v <= 0.0 {
                return invalid("a run has zero variance; the signal-to-noise ratio is infinite");
            }
            Ok(v)
        };
        Ok(match self {
            SignalToNoise::SmallerIsBetter => {
                let s = y.iter().map(|v| v * v).sum::<f64>() / n;
                if s <= 0.0 {
                    return invalid("all responses of a run are zero");
                }
                -10.0 * s.log10()
            }
            SignalToNoise::LargerIsBetter => {
                if y.iter().any(|&v| v <= 0.0) {
                    return invalid("larger-is-better needs positive responses");
                }
                -10.0 * (y.iter().map(|v| 1.0 / (v * v)).sum::<f64>() / n).log10()
            }
            SignalToNoise::NominalIsBest => {
                if mean == 0.0 {
                    return invalid("a run has a zero mean; ybar^2/s^2 is zero");
                }
                10.0 * (mean * mean / var()?).log10()
            }
            SignalToNoise::NominalIsBestVariance => -10.0 * var()?.log10(),
        })
    }
}

/// Mean response and S/N of every level of a factor.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseTable {
    /// Factor.
    pub factor: String,
    /// Mean of the statistic at each level.
    pub level_means: Vec<f64>,
    /// Largest − smallest level mean.
    pub delta: f64,
    /// Rank of delta (1 = largest).
    pub rank: usize,
}

/// One row of an ANOVA of run statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct TaguchiAnovaRow {
    /// Source.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sum of squares.
    pub ss: f64,
    /// Mean square.
    pub ms: Option<f64>,
    /// F.
    pub f: Option<f64>,
    /// p-value.
    pub p: Option<f64>,
    /// Percent of the total sum of squares.
    pub percent: f64,
}

/// Result of a Taguchi analysis.
#[derive(Debug, Clone)]
pub struct TaguchiAnalysis {
    /// The design.
    pub design: TaguchiDesign,
    /// S/N type.
    pub sn_type: SignalToNoise,
    /// Per-run mean.
    pub run_mean: Vec<f64>,
    /// Per-run standard deviation (`None` with one replicate).
    pub run_sd: Vec<Option<f64>>,
    /// Per-run signal-to-noise ratio.
    pub run_sn: Vec<f64>,
    /// Response table for S/N ratios.
    pub sn_table: Vec<ResponseTable>,
    /// Response table for means.
    pub mean_table: Vec<ResponseTable>,
    /// ANOVA of the S/N ratios.
    pub sn_anova: Vec<TaguchiAnovaRow>,
    /// ANOVA of the means.
    pub mean_anova: Vec<TaguchiAnovaRow>,
    /// Level (1-based) of each factor with the largest S/N (the S/N-optimal setting).
    pub best_levels: Vec<usize>,
    /// Predicted S/N at the best levels (additive model).
    pub predicted_sn: f64,
    /// Predicted mean at the best-S/N levels (additive model).
    pub predicted_mean: f64,
}

fn tables(design: &TaguchiDesign, stat: &[f64]) -> (Vec<ResponseTable>, Vec<Vec<f64>>) {
    let mut tabs: Vec<ResponseTable> = Vec::new();
    let mut lm = Vec::new();
    for (f, name) in design.factors.iter().enumerate() {
        let l = design.n_levels[f];
        let means: Vec<f64> = (1..=l)
            .map(|lv| {
                let v: Vec<f64> = (0..design.runs()).filter(|&r| design.levels[r][f] == lv).map(|r| stat[r]).collect();
                v.iter().sum::<f64>() / v.len() as f64
            })
            .collect();
        let delta = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - means.iter().cloned().fold(f64::INFINITY, f64::min);
        lm.push(means.clone());
        tabs.push(ResponseTable { factor: name.clone(), level_means: means, delta, rank: 0 });
    }
    let deltas: Vec<f64> = tabs.iter().map(|t| t.delta).collect();
    for t in tabs.iter_mut() {
        t.rank = 1 + deltas.iter().filter(|&&d| d > t.delta).count();
    }
    (tabs, lm)
}

fn anova(design: &TaguchiDesign, stat: &[f64]) -> Result<Vec<TaguchiAnovaRow>> {
    let n = stat.len();
    let grand = stat.iter().sum::<f64>() / n as f64;
    let sst: f64 = stat.iter().map(|v| (v - grand) * (v - grand)).sum();
    let mut rows = Vec::new();
    let mut ss_model = 0.0;
    let mut df_model = 0;
    for (f, name) in design.factors.iter().enumerate() {
        let l = design.n_levels[f];
        let mut ss = 0.0;
        for lv in 1..=l {
            let v: Vec<f64> = (0..n).filter(|&r| design.levels[r][f] == lv).map(|r| stat[r]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            ss += v.len() as f64 * (m - grand) * (m - grand);
        }
        ss_model += ss;
        df_model += l - 1;
        rows.push((name.clone(), l - 1, ss));
    }
    let df_err = (n - 1).saturating_sub(df_model);
    let sse = (sst - ss_model).max(0.0);
    let mse = (df_err > 0).then(|| sse / df_err as f64);
    let mut out = Vec::new();
    for (name, df, ss) in rows {
        let ms = ss / df as f64;
        let f = mse.and_then(|m| (m > 0.0).then(|| ms / m));
        let p = f.and_then(|f| FisherF::new(df as f64, df_err as f64).ok().map(|d| d.sf(f)));
        out.push(TaguchiAnovaRow {
            source: name,
            df,
            ss,
            ms: Some(ms),
            f,
            p,
            percent: if sst > 0.0 { 100.0 * ss / sst } else { 0.0 },
        });
    }
    if df_err > 0 {
        out.push(TaguchiAnovaRow {
            source: "Residual Error".into(),
            df: df_err,
            ss: sse,
            ms: mse,
            f: None,
            p: None,
            percent: if sst > 0.0 { 100.0 * sse / sst } else { 0.0 },
        });
    }
    out.push(TaguchiAnovaRow {
        source: "Total".into(),
        df: n - 1,
        ss: sst,
        ms: None,
        f: None,
        p: None,
        percent: 100.0,
    });
    Ok(out)
}

/// Analyse a Taguchi experiment. `responses[run]` holds the replicates (noise-factor
/// conditions) of that run; every run needs the same number for the S/N types that use a variance.
pub fn analyze_taguchi(design: &TaguchiDesign, responses: &[Vec<f64>], sn: SignalToNoise) -> Result<TaguchiAnalysis> {
    if responses.len() != design.runs() {
        return invalid(format!("{} response rows for {} runs", responses.len(), design.runs()));
    }
    let mut mean = Vec::new();
    let mut sd = Vec::new();
    let mut ratio = Vec::new();
    for y in responses {
        let r = sn.ratio(y)?;
        let m = y.iter().sum::<f64>() / y.len() as f64;
        mean.push(m);
        sd.push(
            (y.len() > 1).then(|| (y.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (y.len() - 1) as f64).sqrt()),
        );
        ratio.push(r);
    }
    let (sn_table, sn_lv) = tables(design, &ratio);
    let (mean_table, mean_lv) = tables(design, &mean);
    let best: Vec<usize> = sn_lv
        .iter()
        .map(|l| 1 + l.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0)
        .collect();
    let g_sn = ratio.iter().sum::<f64>() / ratio.len() as f64;
    let g_mean = mean.iter().sum::<f64>() / mean.len() as f64;
    let predicted_sn = g_sn + best.iter().enumerate().map(|(f, &l)| sn_lv[f][l - 1] - g_sn).sum::<f64>();
    let predicted_mean = g_mean + best.iter().enumerate().map(|(f, &l)| mean_lv[f][l - 1] - g_mean).sum::<f64>();
    Ok(TaguchiAnalysis {
        sn_anova: anova(design, &ratio)?,
        mean_anova: anova(design, &mean)?,
        design: design.clone(),
        sn_type: sn,
        run_mean: mean,
        run_sd: sd,
        run_sn: ratio,
        sn_table,
        mean_table,
        best_levels: best,
        predicted_sn,
        predicted_mean,
    })
}

impl TaguchiAnalysis {
    /// Main-effects charts of the S/N ratios and of the means (one chart each, a series per factor).
    pub fn charts(&self) -> Vec<ChartSpec> {
        let mk = |title: &str, tabs: &[ResponseTable], grand: f64, y: &str| ChartSpec {
            kind: ChartKind::Line,
            title: title.into(),
            x_label: "Level".into(),
            y_label: y.into(),
            categories: Vec::new(),
            series: tabs
                .iter()
                .map(|t| Series {
                    name: t.factor.clone(),
                    x: (1..=t.level_means.len()).map(|i| i as f64).collect(),
                    y: t.level_means.clone(),
                })
                .collect(),
            reference_lines: vec![crate::report::ReferenceLine {
                axis: crate::report::Axis::Y,
                value: grand,
                label: "Mean".into(),
            }],
        };
        let g_sn = self.run_sn.iter().sum::<f64>() / self.run_sn.len() as f64;
        let g_m = self.run_mean.iter().sum::<f64>() / self.run_mean.len() as f64;
        vec![
            mk("Main Effects Plot for SN Ratios", &self.sn_table, g_sn, "Mean of SN ratios"),
            mk("Main Effects Plot for Means", &self.mean_table, g_m, "Mean of means"),
        ]
    }
}
