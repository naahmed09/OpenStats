//! Process capability for attribute data: binomial (defectives) and Poisson (defects).
//!
//! *Binomial capability* pools the samples: the overall proportion defective `p̄ = ΣD/Σn` with a
//! Clopper–Pearson interval, parts per million defective, and the benchmark `Z = Φ⁻¹(1 − p̄)`
//! (intervals map the proportion's limits through the same function). Pearson's χ² tests whether
//! the samples share one proportion.
//!
//! *Poisson capability* pools defects over units: `DPU = ΣD/ΣU` with the Garwood interval, the
//! smallest and largest sample DPU, the χ² test for a common DPU, and the benchmark
//! `Z = Φ⁻¹(e^(−DPU))` (the proportion of defect-free units).

use crate::error::{invalid, Result, StatsError};
use crate::hypothesis::Alternative;
use crate::proportion::clopper_pearson;
use crate::report::{Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Section, Series, Table};
use openstats_math::dist::{ChiSquared, ContinuousDistribution};
use openstats_math::special::erf::norm_ppf;

fn z_of(p: f64) -> f64 {
    if p <= 0.0 {
        f64::INFINITY
    } else if p >= 1.0 {
        f64::NEG_INFINITY
    } else {
        norm_ppf(1.0 - p)
    }
}

fn check_level(level: f64) -> Result<()> {
    if !(level > 0.0 && level < 1.0) {
        return invalid("the confidence level must be strictly between 0 and 1");
    }
    Ok(())
}

fn whole(name: &str, v: f64) -> Result<()> {
    if v < 0.0 || v.fract() != 0.0 {
        return invalid(format!("{name} must be a non-negative whole number (got {v})"));
    }
    Ok(())
}

/// Result of a binomial capability analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct BinomialCapability {
    /// Defectives per sample.
    pub defectives: Vec<f64>,
    /// Sample sizes.
    pub sizes: Vec<f64>,
    /// Pooled proportion defective.
    pub p_bar: f64,
    /// Confidence interval of the proportion (Clopper–Pearson on the pooled counts).
    pub p_interval: (f64, f64),
    /// Confidence level.
    pub level: f64,
    /// Target proportion defective, when given.
    pub target: Option<f64>,
    /// Benchmark Z.
    pub z: f64,
    /// Interval of Z.
    pub z_interval: (f64, f64),
    /// Pearson χ² for a common proportion (`None` with a single sample or `p̄` of 0 or 1).
    pub chi_square: Option<(f64, f64, f64)>,
}

/// Binomial capability from defectives and sample sizes (rows with a missing value are dropped).
pub fn binomial_capability(
    defectives: &[f64],
    sizes: &[f64],
    level: f64,
    target: Option<f64>,
) -> Result<BinomialCapability> {
    check_level(level)?;
    if defectives.len() != sizes.len() {
        return invalid("defectives and sample sizes must have the same length");
    }
    if let Some(t) = target {
        if !(0.0..=1.0).contains(&t) {
            return invalid("the target proportion defective must be in [0, 1]");
        }
    }
    let (mut d, mut n) = (Vec::new(), Vec::new());
    for (&x, &m) in defectives.iter().zip(sizes) {
        if x.is_nan() || m.is_nan() {
            continue;
        }
        if x.is_infinite() || m.is_infinite() {
            return Err(StatsError::NonFinite);
        }
        whole("the number of defectives", x)?;
        whole("the sample size", m)?;
        if m < 1.0 {
            return invalid("every sample size must be at least 1");
        }
        if x > m {
            return invalid(format!("{x} defectives in a sample of {m}"));
        }
        d.push(x);
        n.push(m);
    }
    if d.is_empty() {
        return Err(StatsError::InsufficientData { what: "binomial capability", needed: 1, got: 0 });
    }
    let (td, tn): (f64, f64) = (d.iter().sum(), n.iter().sum());
    let p_bar = td / tn;
    let ci = clopper_pearson(td as u64, tn as u64, Alternative::TwoSided, level)?;
    let chi_square = if d.len() >= 2 && p_bar > 0.0 && p_bar < 1.0 {
        let chi: f64 = d.iter().zip(&n).map(|(x, m)| (x - m * p_bar).powi(2) / (m * p_bar * (1.0 - p_bar))).sum();
        let df = (d.len() - 1) as f64;
        Some((chi, df, ChiSquared::new(df)?.sf(chi)))
    } else {
        None
    };
    Ok(BinomialCapability {
        defectives: d,
        sizes: n,
        p_bar,
        p_interval: (ci.lower, ci.upper),
        level,
        target,
        z: z_of(p_bar),
        z_interval: (z_of(ci.upper), z_of(ci.lower)),
        chi_square,
    })
}

fn line_chart(title: &str, xl: &str, yl: &str, series: Vec<Series>, refs: Vec<ReferenceLine>) -> ChartSpec {
    ChartSpec {
        kind: ChartKind::Line,
        title: title.into(),
        x_label: xl.into(),
        y_label: yl.into(),
        categories: Vec::new(),
        series,
        reference_lines: refs,
    }
}

impl BinomialCapability {
    /// Session section: summary statistics, the equality test and two charts.
    pub fn section(&self, name: &str) -> Section {
        let mut sec = Section::new(format!("Binomial Process Capability Analysis of {name}"));
        let pct = self.level * 100.0;
        let mut t = Table::new(
            format!("Summary Statistics ({pct:.1}% confidence)"),
            vec![
                Column::new("Statistic", ColumnKind::Text),
                Column::new("Estimate", ColumnKind::Significant(6)),
                Column::new("Lower CI", ColumnKind::Significant(6)),
                Column::new("Upper CI", ColumnKind::Significant(6)),
            ],
        );
        t.row(vec![
            Cell::text("%Defective"),
            Cell::Num(100.0 * self.p_bar),
            Cell::Num(100.0 * self.p_interval.0),
            Cell::Num(100.0 * self.p_interval.1),
        ]);
        t.row(vec![
            Cell::text("PPM Defective"),
            Cell::Num(1e6 * self.p_bar),
            Cell::Num(1e6 * self.p_interval.0),
            Cell::Num(1e6 * self.p_interval.1),
        ]);
        t.row(vec![
            Cell::text("Process Z"),
            Cell::Num(self.z),
            Cell::Num(self.z_interval.0),
            Cell::Num(self.z_interval.1),
        ]);
        if let Some(tg) = self.target {
            t.row(vec![Cell::text("Target %Defective"), Cell::Num(100.0 * tg), Cell::Missing, Cell::Missing]);
        }
        sec.table(t);
        let total_n: f64 = self.sizes.iter().sum();
        let total_d: f64 = self.defectives.iter().sum();
        sec.text(format!(
            "{} samples, {total_n} items inspected, {total_d} defective. Process Z = inverse normal of (1 − proportion defective).",
            self.sizes.len()
        ));
        match self.chi_square {
            Some((chi, df, p)) => {
                let mut c = Table::new(
                    "Test of a Common Proportion Defective",
                    vec![
                        Column::new("Chi-Square", ColumnKind::Significant(6)),
                        Column::new("DF", ColumnKind::Integer),
                        Column::new("P-Value", ColumnKind::PValue),
                    ],
                );
                c.row(vec![Cell::Num(chi), Cell::Int(df as i64), Cell::Num(p)]);
                sec.table(c);
                if p < 0.05 {
                    sec.warn(
                        "The proportions differ between samples (p < 0.05); the pooled capability may be misleading.",
                    );
                }
            }
            None => {
                sec.text("The test of a common proportion needs at least two samples and a proportion strictly between 0 and 1.");
            }
        }
        let k = self.sizes.len();
        let x: Vec<f64> = (1..=k).map(|i| i as f64).collect();
        let p_i: Vec<f64> = self.defectives.iter().zip(&self.sizes).map(|(d, n)| 100.0 * d / n).collect();
        let ucl: Vec<f64> = self
            .sizes
            .iter()
            .map(|n| 100.0 * (self.p_bar + 3.0 * (self.p_bar * (1.0 - self.p_bar) / n).sqrt()).min(1.0))
            .collect();
        let lcl: Vec<f64> = self
            .sizes
            .iter()
            .map(|n| 100.0 * (self.p_bar - 3.0 * (self.p_bar * (1.0 - self.p_bar) / n).sqrt()).max(0.0))
            .collect();
        let mut refs = vec![ReferenceLine { axis: Axis::Y, value: 100.0 * self.p_bar, label: "P-bar".into() }];
        if let Some(tg) = self.target {
            refs.push(ReferenceLine { axis: Axis::Y, value: 100.0 * tg, label: "Target".into() });
        }
        sec.chart(line_chart(
            "%Defective by Sample",
            "Sample",
            "%Defective",
            vec![
                Series { name: "%Defective".into(), x: x.clone(), y: p_i },
                Series { name: "UCL".into(), x: x.clone(), y: ucl },
                Series { name: "LCL".into(), x: x.clone(), y: lcl },
            ],
            refs,
        ));
        let (mut cd, mut cn) = (0.0, 0.0);
        let cum: Vec<f64> = self
            .defectives
            .iter()
            .zip(&self.sizes)
            .map(|(d, n)| {
                cd += d;
                cn += n;
                100.0 * cd / cn
            })
            .collect();
        sec.chart(line_chart(
            "Cumulative %Defective",
            "Sample",
            "Cumulative %Defective",
            vec![Series { name: "Cumulative %Defective".into(), x, y: cum }],
            vec![ReferenceLine { axis: Axis::Y, value: 100.0 * self.p_bar, label: "P-bar".into() }],
        ));
        sec
    }
}

/// Result of a Poisson capability analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonCapability {
    /// Defects per sample.
    pub defects: Vec<f64>,
    /// Units inspected per sample.
    pub units: Vec<f64>,
    /// Pooled defects per unit.
    pub dpu: f64,
    /// Garwood interval of the DPU.
    pub dpu_interval: (f64, f64),
    /// Confidence level.
    pub level: f64,
    /// Smallest sample DPU.
    pub min_dpu: f64,
    /// Largest sample DPU.
    pub max_dpu: f64,
    /// Target DPU, when given.
    pub target: Option<f64>,
    /// Benchmark Z from the proportion of defect-free units.
    pub z: f64,
    /// Interval of Z.
    pub z_interval: (f64, f64),
    /// Pearson χ² for a common DPU: statistic, df, p-value (`None` with one sample or no defects).
    pub chi_square: Option<(f64, f64, f64)>,
}

/// Poisson capability from defect counts and the number of units inspected in each sample
/// (rows with a missing value are dropped).
pub fn poisson_capability(
    defects: &[f64],
    units: &[f64],
    level: f64,
    target: Option<f64>,
) -> Result<PoissonCapability> {
    check_level(level)?;
    if defects.len() != units.len() {
        return invalid("defects and units must have the same length");
    }
    if let Some(t) = target {
        if t < 0.0 || !t.is_finite() {
            return invalid("the target DPU must be non-negative");
        }
    }
    let (mut d, mut u) = (Vec::new(), Vec::new());
    for (&x, &m) in defects.iter().zip(units) {
        if x.is_nan() || m.is_nan() {
            continue;
        }
        if x.is_infinite() || m.is_infinite() {
            return Err(StatsError::NonFinite);
        }
        whole("the number of defects", x)?;
        if m <= 0.0 {
            return invalid("every number of units must be positive");
        }
        d.push(x);
        u.push(m);
    }
    if d.is_empty() {
        return Err(StatsError::InsufficientData { what: "Poisson capability", needed: 1, got: 0 });
    }
    let (td, tu): (f64, f64) = (d.iter().sum(), u.iter().sum());
    let dpu = td / tu;
    let a = 1.0 - level;
    let lo = if td == 0.0 { 0.0 } else { ChiSquared::new(2.0 * td)?.quantile(a / 2.0) / 2.0 / tu };
    let hi = ChiSquared::new(2.0 * td + 2.0)?.quantile(1.0 - a / 2.0) / 2.0 / tu;
    let rates: Vec<f64> = d.iter().zip(&u).map(|(x, m)| x / m).collect();
    let z_of_dpu = |r: f64| z_of(1.0 - (-r).exp());
    let chi_square = if d.len() >= 2 && dpu > 0.0 {
        let chi: f64 = d.iter().zip(&u).map(|(x, m)| (x - m * dpu).powi(2) / (m * dpu)).sum();
        let df = (d.len() - 1) as f64;
        Some((chi, df, ChiSquared::new(df)?.sf(chi)))
    } else {
        None
    };
    Ok(PoissonCapability {
        defects: d,
        units: u,
        dpu,
        dpu_interval: (lo, hi),
        level,
        min_dpu: rates.iter().copied().fold(f64::INFINITY, f64::min),
        max_dpu: rates.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        target,
        z: z_of_dpu(dpu),
        z_interval: (z_of_dpu(hi), z_of_dpu(lo)),
        chi_square,
    })
}

impl PoissonCapability {
    /// Session section: summary statistics, the equality test and two charts.
    pub fn section(&self, name: &str) -> Section {
        let mut sec = Section::new(format!("Poisson Process Capability Analysis of {name}"));
        let pct = self.level * 100.0;
        let mut t = Table::new(
            format!("Summary Statistics ({pct:.1}% confidence)"),
            vec![
                Column::new("Statistic", ColumnKind::Text),
                Column::new("Estimate", ColumnKind::Significant(6)),
                Column::new("Lower CI", ColumnKind::Significant(6)),
                Column::new("Upper CI", ColumnKind::Significant(6)),
            ],
        );
        t.row(vec![
            Cell::text("Mean DPU"),
            Cell::Num(self.dpu),
            Cell::Num(self.dpu_interval.0),
            Cell::Num(self.dpu_interval.1),
        ]);
        t.row(vec![Cell::text("Minimum DPU"), Cell::Num(self.min_dpu), Cell::Missing, Cell::Missing]);
        t.row(vec![Cell::text("Maximum DPU"), Cell::Num(self.max_dpu), Cell::Missing, Cell::Missing]);
        if let Some(tg) = self.target {
            t.row(vec![Cell::text("Target DPU"), Cell::Num(tg), Cell::Missing, Cell::Missing]);
        }
        t.row(vec![
            Cell::text("Process Z"),
            Cell::Num(self.z),
            Cell::Num(self.z_interval.0),
            Cell::Num(self.z_interval.1),
        ]);
        sec.table(t);
        sec.text(format!(
            "{} samples, {} units inspected, {} defects. Process Z = inverse normal of the proportion of defect-free units, exp(−DPU).",
            self.units.len(),
            self.units.iter().sum::<f64>(),
            self.defects.iter().sum::<f64>()
        ));
        match self.chi_square {
            Some((chi, df, p)) => {
                let mut c = Table::new(
                    "Test of a Common DPU",
                    vec![
                        Column::new("Chi-Square", ColumnKind::Significant(6)),
                        Column::new("DF", ColumnKind::Integer),
                        Column::new("P-Value", ColumnKind::PValue),
                    ],
                );
                c.row(vec![Cell::Num(chi), Cell::Int(df as i64), Cell::Num(p)]);
                sec.table(c);
                if p < 0.05 {
                    sec.warn("The DPU differs between samples (p < 0.05); the pooled capability may be misleading.");
                }
            }
            None => {
                sec.text("The test of a common DPU needs at least two samples and at least one defect.");
            }
        }
        let k = self.units.len();
        let x: Vec<f64> = (1..=k).map(|i| i as f64).collect();
        let r: Vec<f64> = self.defects.iter().zip(&self.units).map(|(d, u)| d / u).collect();
        let ucl: Vec<f64> = self.units.iter().map(|u| self.dpu + 3.0 * (self.dpu / u).sqrt()).collect();
        let lcl: Vec<f64> = self.units.iter().map(|u| (self.dpu - 3.0 * (self.dpu / u).sqrt()).max(0.0)).collect();
        let mut refs = vec![ReferenceLine { axis: Axis::Y, value: self.dpu, label: "U-bar".into() }];
        if let Some(tg) = self.target {
            refs.push(ReferenceLine { axis: Axis::Y, value: tg, label: "Target".into() });
        }
        sec.chart(line_chart(
            "DPU by Sample",
            "Sample",
            "DPU",
            vec![
                Series { name: "DPU".into(), x: x.clone(), y: r },
                Series { name: "UCL".into(), x: x.clone(), y: ucl },
                Series { name: "LCL".into(), x: x.clone(), y: lcl },
            ],
            refs,
        ));
        let (mut cd, mut cu) = (0.0, 0.0);
        let cum: Vec<f64> = self
            .defects
            .iter()
            .zip(&self.units)
            .map(|(d, u)| {
                cd += d;
                cu += u;
                cd / cu
            })
            .collect();
        sec.chart(line_chart(
            "Cumulative DPU",
            "Sample",
            "Cumulative DPU",
            vec![Series { name: "Cumulative DPU".into(), x, y: cum }],
            vec![ReferenceLine { axis: Axis::Y, value: self.dpu, label: "U-bar".into() }],
        ));
        sec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_pooled_values() {
        let c = binomial_capability(&[2.0, 3.0, 1.0, 4.0], &[100.0, 100.0, 100.0, 100.0], 0.95, None).unwrap();
        assert!((c.p_bar - 0.025).abs() < 1e-15);
        assert!(c.p_interval.0 < 0.025 && c.p_interval.1 > 0.025);
        assert!((c.z - norm_ppf(1.0 - 0.025)).abs() < 1e-12);
        assert!(c.z_interval.0 < c.z && c.z < c.z_interval.1);
        let (chi, df, p) = c.chi_square.unwrap();
        assert_eq!(df, 3.0);
        assert!((chi - 5.0 / 2.4375).abs() < 1e-12);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn poisson_pooled_values() {
        let c = poisson_capability(&[3.0, 5.0, 2.0], &[10.0, 10.0, 10.0], 0.95, Some(0.2)).unwrap();
        assert!((c.dpu - 1.0 / 3.0).abs() < 1e-15);
        assert!((c.min_dpu - 0.2).abs() < 1e-15 && (c.max_dpu - 0.5).abs() < 1e-15);
        assert!(c.dpu_interval.0 < c.dpu && c.dpu < c.dpu_interval.1);
    }

    #[test]
    fn zero_defects_and_errors() {
        let c = binomial_capability(&[0.0, 0.0], &[50.0, 50.0], 0.9, None).unwrap();
        assert_eq!(c.z, f64::INFINITY);
        assert!(c.chi_square.is_none());
        assert!(binomial_capability(&[5.0], &[3.0], 0.95, None).is_err());
        assert!(binomial_capability(&[1.5], &[3.0], 0.95, None).is_err());
        assert!(poisson_capability(&[1.0], &[0.0], 0.95, None).is_err());
        assert!(poisson_capability(&[1.0], &[1.0], 1.5, None).is_err());
    }
}
