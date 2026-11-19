//! [`Reportable`] implementations for control charts and capability results.

use super::capability::{
    DistributionCapability, ExpectedPerformance, NormalCapability, ObservedPerformance, TransformedCapability,
};
use super::charts::{AttributeChart, AttributeKind, ChartPanel, VariableChart, VariableChartKind};
use super::constants::{constants, MAX_SIZE, MIN_SIZE};
use super::rules::describe;
use super::sixpack::CapabilitySixpack;
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

pub(super) const S: ColumnKind = ColumnKind::Significant(6);
pub(super) const I: ColumnKind = ColumnKind::Integer;
pub(super) const T: ColumnKind = ColumnKind::Text;

fn range_text(v: &[f64]) -> Cell {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let (lo, hi) = finite.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    if finite.is_empty() {
        Cell::Missing
    } else if lo == hi {
        Cell::Num(lo)
    } else {
        Cell::text(format!("{lo:.6} – {hi:.6}"))
    }
}

fn panel_row(t: &mut Table, p: &ChartPanel) {
    t.row(vec![Cell::text(&p.name), range_text(&p.center), range_text(&p.ucl), range_text(&p.lcl)]);
}

pub(super) fn limits_table(panels: &[&ChartPanel]) -> Table {
    let mut t = Table::new(
        "Control Limits",
        vec![Column::new("Chart", T), Column::new("Center line", T), Column::new("UCL", T), Column::new("LCL", T)],
    );
    for p in panels {
        panel_row(&mut t, p);
    }
    t
}

pub(super) fn add_tests(sec: &mut Section, p: &ChartPanel) {
    if p.violations.is_empty() {
        sec.text(format!("{}: no points failed the requested tests.", p.name));
    }
    for v in &p.violations {
        let pts: Vec<String> = v.points.iter().map(|i| (i + 1).to_string()).collect();
        sec.text(format!(
            "{}: TEST {}. {}. Test failed at points: {}",
            p.name,
            v.test,
            describe(&p.tests, v.test),
            pts.join(", ")
        ));
    }
}

impl Reportable for VariableChart {
    fn report_section(&self) -> Section {
        let title = match self.kind {
            VariableChartKind::IMr => "I-MR Chart",
            VariableChartKind::XbarR => "Xbar-R Chart",
            VariableChartKind::XbarS => "Xbar-S Chart",
        };
        let mut sec = Section::new(title);
        sec.table(limits_table(&[&self.location, &self.dispersion]));
        sec.text(format!(
            "Process mean = {:.6}; process sigma = {:.6} ({}).",
            self.mean, self.sigma, self.sigma_method
        ));
        add_tests(&mut sec, &self.location);
        add_tests(&mut sec, &self.dispersion);
        let x = if self.kind == VariableChartKind::IMr { "Observation" } else { "Sample" };
        sec.chart(
            self.location
                .chart_spec(&format!("{} of {}", title.split('-').next().unwrap_or(title), self.location.name), x),
        );
        sec.chart(self.dispersion.chart_spec(&format!("{} of {}", "Dispersion", self.dispersion.name), x));
        sec
    }
}

impl Reportable for AttributeChart {
    fn report_section(&self) -> Section {
        let title = match self.kind {
            AttributeKind::P => "P Chart",
            AttributeKind::Np => "NP Chart",
            AttributeKind::C => "C Chart",
            AttributeKind::U => "U Chart",
            AttributeKind::LaneyP => "Laney P' Chart",
            AttributeKind::LaneyU => "Laney U' Chart",
        };
        let mut sec = Section::new(title);
        sec.table(limits_table(&[&self.panel]));
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(title, "Sample"));
        sec
    }
}

fn opt(x: Option<f64>) -> Cell {
    Cell::opt(x)
}

fn perf_table(title: &str, p: &ExpectedPerformance) -> Table {
    let mut t = Table::new(title, vec![Column::new("Measure", T), Column::new("Value", S)]);
    t.row(vec![Cell::text("Z.Bench"), Cell::Num(p.z_bench)]);
    t.row(vec![Cell::text("Z.LSL"), opt(p.z_lsl)]);
    t.row(vec![Cell::text("Z.USL"), opt(p.z_usl)]);
    t.row(vec![Cell::text("PPM < LSL"), Cell::Num(p.ppm_below)]);
    t.row(vec![Cell::text("PPM > USL"), Cell::Num(p.ppm_above)]);
    t.row(vec![Cell::text("PPM Total"), Cell::Num(p.ppm_total)]);
    t
}

fn observed_table(o: &ObservedPerformance) -> Table {
    let mut t = Table::new(
        "Observed Performance",
        vec![Column::new("Measure", T), Column::new("Count", I), Column::new("PPM", S)],
    );
    t.row(vec![Cell::text("< LSL"), Cell::Int(o.n_below as i64), Cell::Num(o.ppm_below)]);
    t.row(vec![Cell::text("> USL"), Cell::Int(o.n_above as i64), Cell::Num(o.ppm_above)]);
    t.row(vec![Cell::text("Total"), Cell::Int((o.n_below + o.n_above) as i64), Cell::Num(o.ppm_total)]);
    t
}

fn specs_rows(t: &mut Table, s: &super::capability::Specs) {
    t.row(vec![Cell::text("LSL"), opt(s.lsl)]);
    t.row(vec![Cell::text("Target"), opt(s.target)]);
    t.row(vec![Cell::text("USL"), opt(s.usl)]);
}

fn normal_sections(c: &NormalCapability, sec: &mut Section) {
    let mut data = Table::new("Process Data", vec![Column::new("Item", T), Column::new("Value", S)]);
    specs_rows(&mut data, &c.specs);
    data.row(vec![Cell::text("Sample Mean"), Cell::Num(c.mean)]);
    data.row(vec![Cell::text("Sample N"), Cell::Int(c.n as i64)]);
    data.row(vec![Cell::text("StDev (Within)"), Cell::Num(c.sd_within)]);
    data.row(vec![Cell::text("StDev (Overall)"), Cell::Num(c.sd_overall)]);
    data.footnotes.push(format!("Within StDev estimated by {}.", c.within_method));
    sec.table(data);
    let mut w = Table::new("Potential (Within) Capability", vec![Column::new("Index", T), Column::new("Value", S)]);
    w.row(vec![Cell::text("Cp"), opt(c.cp)]);
    w.row(vec![Cell::text("CPL"), opt(c.cpl)]);
    w.row(vec![Cell::text("CPU"), opt(c.cpu)]);
    w.row(vec![Cell::text("Cpk"), Cell::Num(c.cpk)]);
    sec.table(w);
    let mut o = Table::new(
        "Overall Capability",
        vec![Column::new("Index", T), Column::new("Value", S), Column::new("CI lower", S), Column::new("CI upper", S)],
    );
    let (pl, pu) = c.pp_ci.map_or((None, None), |i| (Some(i.lower), Some(i.upper)));
    let (kl, ku) = c.ppk_ci.map_or((None, None), |i| (Some(i.lower), Some(i.upper)));
    o.row(vec![Cell::text("Pp"), opt(c.pp), opt(pl), opt(pu)]);
    o.row(vec![Cell::text("PPL"), opt(c.ppl), Cell::Missing, Cell::Missing]);
    o.row(vec![Cell::text("PPU"), opt(c.ppu), Cell::Missing, Cell::Missing]);
    o.row(vec![Cell::text("Ppk"), Cell::Num(c.ppk), opt(kl), opt(ku)]);
    o.row(vec![Cell::text("Cpm"), opt(c.cpm), Cell::Missing, Cell::Missing]);
    sec.table(o);
    sec.table(observed_table(&c.observed));
    sec.table(perf_table("Expected Performance (Within)", &c.expected_within));
    sec.table(perf_table("Expected Performance (Overall)", &c.expected_overall));
}

impl Reportable for NormalCapability {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Process Capability Analysis (Normal)");
        normal_sections(self, &mut sec);
        sec
    }
}

impl Reportable for TransformedCapability {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Process Capability Analysis (Transformed)");
        sec.text(format!("{}.", self.transformation.describe()));
        sec.text(format!(
            "Anderson-Darling test of the transformed data: A² = {:.4}, P = {}.",
            self.ad.a2,
            crate::report::format_p_value(self.ad.p_value)
        ));
        sec.text("All statistics below are on the transformed scale; observed PPM is unchanged by an increasing transformation.");
        for n in &self.notes {
            sec.warn(n.clone());
        }
        normal_sections(&self.normal, &mut sec);
        sec
    }
}

impl Reportable for DistributionCapability {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(format!("Process Capability Analysis ({})", self.fit.family.name()));
        let mut d = Table::new("Fitted Distribution", vec![Column::new("Item", T), Column::new("Value", S)]);
        let (n1, n2) = self.fit.family.parameter_labels();
        d.row(vec![Cell::text(n1), Cell::Num(self.fit.param1)]);
        if !self.fit.param2.is_nan() {
            d.row(vec![Cell::text(n2), Cell::Num(self.fit.param2)]);
        }
        if self.fit.family.uses_threshold() {
            d.row(vec![Cell::text("Threshold"), Cell::Num(self.fit.threshold)]);
        }
        d.row(vec![Cell::text("Log-likelihood"), Cell::Num(self.fit.log_likelihood)]);
        d.row(vec![Cell::text("Anderson-Darling"), Cell::Num(self.fit.ad)]);
        d.row(vec![Cell::text("N"), Cell::Int(self.fit.n as i64)]);
        sec.table(d);
        let mut p = Table::new("Process Data", vec![Column::new("Item", T), Column::new("Value", S)]);
        specs_rows(&mut p, &self.specs);
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
        sec.table(perf_table("Expected Performance (Overall)", &self.expected));
        sec
    }
}

impl Reportable for CapabilitySixpack {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Capability Sixpack");
        sec.table(limits_table(&[&self.charts.location, &self.charts.dispersion]));
        add_tests(&mut sec, &self.charts.location);
        add_tests(&mut sec, &self.charts.dispersion);
        sec.text(format!(
            "Anderson-Darling normality test: A² = {:.4}, P = {}.",
            self.ad.a2,
            crate::report::format_p_value(self.ad.p_value)
        ));
        normal_sections(&self.capability, &mut sec);
        for c in self.charts() {
            sec.chart(c);
        }
        sec
    }
}

/// A section tabulating the control-chart constants for subgroup sizes `from..=to`.
pub fn constants_section(from: usize, to: usize) -> crate::error::Result<Section> {
    if from < MIN_SIZE || to > MAX_SIZE || from > to {
        return crate::error::invalid(format!("constants are available for sizes {MIN_SIZE}..={MAX_SIZE}"));
    }
    let heads = ["n", "d2", "d3", "c4", "A2", "A3", "B3", "B4", "D3", "D4"];
    let mut cols = vec![Column::new(heads[0], I)];
    cols.extend(heads[1..].iter().map(|h| Column::new(*h, ColumnKind::Fixed(4))));
    let mut t = Table::new("Control Chart Constants", cols);
    for n in from..=to {
        let c = constants(n)?;
        t.row(vec![
            Cell::Int(n as i64),
            Cell::Num(c.d2),
            Cell::Num(c.d3),
            Cell::Num(c.c4),
            Cell::Num(c.a2),
            Cell::Num(c.a3),
            Cell::Num(c.b3),
            Cell::Num(c.b4),
            Cell::Num(c.big_d3),
            Cell::Num(c.big_d4),
        ]);
    }
    let mut sec = Section::new("SPC Constants");
    sec.table(t);
    Ok(sec)
}
