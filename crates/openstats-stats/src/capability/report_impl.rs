//! `Reportable` implementations of the capability extensions.

use super::ci::CapabilityIntervals;
use super::empirical::EmpiricalCapability;
use super::extras::{ExtendedSixpack, RunningAverage, SubgroupCapability};
use super::identify::DistributionRank;
use crate::report::{Cell, ChartSpec, Column, ColumnKind, Reportable, Section, Table};
use crate::spc::capability::{NormalCapability, ObservedPerformance};

const T: ColumnKind = ColumnKind::Text;
const S: ColumnKind = ColumnKind::Fixed(4);
const I: ColumnKind = ColumnKind::Integer;
const P: ColumnKind = ColumnKind::PValue;

fn opt(x: Option<f64>) -> Cell {
    x.map_or(Cell::Missing, Cell::Num)
}

fn observed_table(o: &ObservedPerformance) -> Table {
    let mut t = Table::new(
        "Observed Performance",
        vec![Column::new("Item", T), Column::new("Count", I), Column::new("PPM", ColumnKind::Fixed(2))],
    );
    t.row(vec![Cell::text("< LSL"), Cell::Int(o.n_below as i64), Cell::Num(o.ppm_below)]);
    t.row(vec![Cell::text("> USL"), Cell::Int(o.n_above as i64), Cell::Num(o.ppm_above)]);
    t.row(vec![Cell::text("Total"), Cell::Int((o.n_below + o.n_above) as i64), Cell::Num(o.ppm_total)]);
    t
}

fn ci_rows(t: &mut Table, name: &str, est: Option<f64>, ci: Option<&crate::graphical::Interval>) {
    if let Some(i) = ci {
        t.row(vec![Cell::text(name), opt(est), Cell::Num(i.lower), Cell::Num(i.upper)]);
    }
}

impl CapabilityIntervals {
    /// The intervals as a table, given the point estimates they belong to.
    pub fn table(&self, c: &NormalCapability) -> Table {
        let mut t = Table::new(
            format!("Capability Indices with {:.0}% Confidence Intervals", self.level * 100.0),
            vec![Column::new("Index", T), Column::new("Estimate", S), Column::new("Lower", S), Column::new("Upper", S)],
        );
        ci_rows(&mut t, "Cp", c.cp, self.cp.as_ref());
        ci_rows(&mut t, "CPL", c.cpl, self.cpl.as_ref());
        ci_rows(&mut t, "CPU", c.cpu, self.cpu.as_ref());
        ci_rows(&mut t, "Cpk", Some(c.cpk), self.cpk.as_ref());
        ci_rows(&mut t, "Pp", c.pp, self.pp.as_ref());
        ci_rows(&mut t, "PPL", c.ppl, self.ppl.as_ref());
        ci_rows(&mut t, "PPU", c.ppu, self.ppu.as_ref());
        ci_rows(&mut t, "Ppk", Some(c.ppk), Some(&self.ppk));
        ci_rows(&mut t, "Cpm", c.cpm, self.cpm.as_ref());
        t.footnotes.push(
            "Cp, Pp: chi-square. One-sided CPL, CPU, PPL, PPU: exact non-central t. Cpk, Ppk: Bissell's approximation. Cpm: Boyles' approximation."
                .into(),
        );
        t
    }
}

impl Reportable for EmpiricalCapability {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Process Capability Analysis (Empirical Percentiles)");
        for n in &self.notes {
            sec.warn(n.clone());
        }
        let mut p = Table::new("Process Data", vec![Column::new("Item", T), Column::new("Value", S)]);
        if let Some(l) = self.specs.lsl {
            p.row(vec![Cell::text("LSL"), Cell::Num(l)]);
        }
        if let Some(u) = self.specs.usl {
            p.row(vec![Cell::text("USL"), Cell::Num(u)]);
        }
        p.row(vec![Cell::text("N"), Cell::Int(self.n as i64)]);
        p.row(vec![Cell::text("Median"), Cell::Num(self.median)]);
        p.row(vec![Cell::text("0.135 percentile"), Cell::Num(self.x_low)]);
        p.row(vec![Cell::text("99.865 percentile"), Cell::Num(self.x_high)]);
        sec.table(p);
        let mut o = Table::new("Overall Capability", vec![Column::new("Index", T), Column::new("Value", S)]);
        o.row(vec![Cell::text("Pp"), opt(self.pp)]);
        o.row(vec![Cell::text("PPL"), opt(self.ppl)]);
        o.row(vec![Cell::text("PPU"), opt(self.ppu)]);
        o.row(vec![Cell::text("Ppk"), Cell::Num(self.ppk)]);
        sec.table(o);
        sec.table(observed_table(&self.observed));
        sec
    }
}

/// A report section for a distribution-identification table.
pub fn ranking_section(ranks: &[DistributionRank]) -> Section {
    let mut sec = Section::new("Distribution Identification");
    let mut t = Table::new(
        "Goodness of Fit",
        vec![
            Column::new("Distribution", T),
            Column::new("AD", S),
            Column::new("P", P),
            Column::new("Log-likelihood", S),
            Column::new("Parameters", I),
        ],
    );
    for r in ranks {
        t.row(vec![
            Cell::text(r.name.clone()),
            Cell::Num(r.ad),
            opt(r.p_value),
            opt(r.log_likelihood),
            Cell::Int(r.parameters as i64),
        ]);
    }
    sec.table(t);
    sec
}

fn subgroup_table(v: &[SubgroupCapability]) -> Table {
    let mut t = Table::new(
        "Capability by Subgroup",
        vec![
            Column::new("Subgroup", I),
            Column::new("N", I),
            Column::new("Mean", S),
            Column::new("StDev", S),
            Column::new("Pp", S),
            Column::new("Ppk", S),
        ],
    );
    for s in v {
        t.row(vec![
            Cell::Int(s.index as i64),
            Cell::Int(s.n as i64),
            Cell::Num(s.mean),
            Cell::Num(s.sd),
            opt(s.pp),
            opt(s.ppk),
        ]);
    }
    t
}

impl Reportable for RunningAverage {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Running Average");
        sec.chart(self.chart_spec());
        sec
    }
}

impl Reportable for ExtendedSixpack {
    fn report_section(&self) -> Section {
        let mut sec = self.sixpack.report_section();
        sec.table(subgroup_table(&self.by_subgroup));
        let charts: Vec<ChartSpec> = self.charts();
        // The first six panels were already added by the standard sixpack section.
        for c in charts.into_iter().skip(6) {
            sec.chart(c);
        }
        sec
    }
}

/// A section with the capability indices and their confidence intervals.
pub fn intervals_section(c: &NormalCapability, ci: &CapabilityIntervals) -> Section {
    let mut sec = Section::new("Capability Indices: Confidence Intervals");
    if ci.within_df.is_none() {
        sec.text("Intervals for the within-σ indices (Cp, CPL, CPU, Cpk) need the degrees of freedom of the within estimate and were not computed.");
    }
    sec.table(ci.table(c));
    sec
}
