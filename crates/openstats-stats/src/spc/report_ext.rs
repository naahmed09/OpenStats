//! [`Reportable`] implementations for the Phase 2g charts.

use super::attributes::{LaneyChart, RareEventChart, RareEventKind};
use super::charts::VariableChartKind;
use super::multivariate::{GeneralizedVarianceChart, HotellingChart, HotellingDecomposition};
use super::report_impl::{add_tests, limits_table, I, S, T};
use super::stages::{
    Stage, StagedAttributeChart, StagedCusumChart, StagedHotellingChart, StagedLaneyChart, StagedRareEventChart,
    StagedSmoothingChart, StagedVariableChart,
};
use super::time_weighted::{CusumChart, SmoothingChart, SmoothingKind, VMaskChart};
use super::variables_ext::{BetweenWithinChart, BoxCoxChart, WithinChart, ZmrChart, ZoneChart};
use crate::report::{Axis, Cell, ChartKind, ChartSpec, Column, ReferenceLine, Reportable, Section, Series, Table};

fn stage_table(stages: &[Stage]) -> Table {
    let mut t =
        Table::new("Stages", vec![Column::new("Stage", T), Column::new("First point", I), Column::new("Points", I)]);
    for s in stages {
        t.row(vec![Cell::text(&s.label), Cell::Int(s.start as i64 + 1), Cell::Int(s.len as i64)]);
    }
    t
}

impl Reportable for StagedVariableChart {
    fn report_section(&self) -> Section {
        let title = match self.kind {
            VariableChartKind::IMr => "Staged I-MR Chart",
            VariableChartKind::XbarR => "Staged Xbar-R Chart",
            VariableChartKind::XbarS => "Staged Xbar-S Chart",
        };
        let mut sec = Section::new(title);
        sec.table(stage_table(&self.stages));
        let mut t = Table::new(
            "Process Estimates by Stage",
            vec![Column::new("Stage", T), Column::new("Mean", S), Column::new("Sigma", S), Column::new("Method", T)],
        );
        for (s, c) in self.stages.iter().zip(&self.charts) {
            t.row(vec![Cell::text(&s.label), Cell::Num(c.mean), Cell::Num(c.sigma), Cell::text(&c.sigma_method)]);
        }
        sec.table(t);
        add_tests(&mut sec, &self.location);
        add_tests(&mut sec, &self.dispersion);
        sec.chart(self.location.chart_spec(&format!("{title}: {}", self.location.name), "Observation"));
        sec.chart(self.dispersion.chart_spec(&format!("{title}: {}", self.dispersion.name), "Observation"));
        sec
    }
}

impl Reportable for StagedAttributeChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(format!("Staged {:?} Chart", self.kind));
        sec.table(stage_table(&self.stages));
        let mut t = Table::new("Centre Line by Stage", vec![Column::new("Stage", T), Column::new("Center line", S)]);
        for (s, c) in self.stages.iter().zip(&self.charts) {
            t.row(vec![Cell::text(&s.label), Cell::Num(c.center)]);
        }
        sec.table(t);
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(&format!("Staged {:?} Chart", self.kind), "Sample"));
        sec
    }
}

impl Reportable for BetweenWithinChart {
    fn report_section(&self) -> Section {
        let (title, w) = match self.within_chart {
            WithinChart::Range => ("I-MR-R (Between/Within) Chart", "R"),
            WithinChart::StDev => ("I-MR-S (Between/Within) Chart", "S"),
        };
        let mut sec = Section::new(title);
        sec.table(limits_table(&[&self.means, &self.moving_range, &self.within]));
        let mut t = Table::new("Standard Deviations", vec![Column::new("Component", T), Column::new("Value", S)]);
        t.row(vec![Cell::text("Within"), Cell::Num(self.sigma_within)]);
        t.row(vec![Cell::text("Between"), Cell::Num(self.sigma_between)]);
        t.row(vec![Cell::text("Between/Within"), Cell::Num(self.sigma_total)]);
        t.footnotes.push(format!("Within σ estimated from {w}̄ and the unbiasing constants; between σ from the moving range of the subgroup means."));
        sec.table(t);
        add_tests(&mut sec, &self.means);
        add_tests(&mut sec, &self.moving_range);
        add_tests(&mut sec, &self.within);
        sec.chart(self.means.chart_spec(&format!("{title}: Subgroup Means"), "Sample"));
        sec.chart(self.moving_range.chart_spec(&format!("{title}: Moving Range"), "Sample"));
        sec.chart(self.within.chart_spec(&format!("{title}: Within"), "Sample"));
        sec
    }
}

impl Reportable for ZoneChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Zone Chart");
        let mut t = Table::new("Zone Weights", vec![Column::new("Zone", T), Column::new("Weight", S)]);
        for (name, w) in ["Within 1σ", "1σ to 2σ", "2σ to 3σ", "Beyond 3σ"].iter().zip(self.options.weights) {
            t.row(vec![Cell::text(*name), Cell::Num(w)]);
        }
        t.footnotes.push(format!("Signal when the cumulative score reaches {}.", self.options.signal_at));
        sec.table(t);
        if self.signals.is_empty() {
            sec.text("No points reached the signal level.");
        } else {
            let pts: Vec<String> = self.signals.iter().map(|i| (i + 1).to_string()).collect();
            sec.text(format!("Cumulative score reached the signal level at points: {}", pts.join(", ")));
        }
        let x: Vec<f64> = (1..=self.scores.len()).map(|i| i as f64).collect();
        sec.chart(ChartSpec {
            kind: ChartKind::Line,
            title: "Zone Chart".into(),
            x_label: "Observation".into(),
            y_label: "Cumulative score".into(),
            categories: Vec::new(),
            series: vec![Series { name: "Score".into(), x, y: self.scores.clone() }],
            reference_lines: vec![ReferenceLine {
                axis: Axis::Y,
                value: self.options.signal_at,
                label: "Signal".into(),
            }],
        });
        sec
    }
}

impl Reportable for ZmrChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Z-MR Chart");
        sec.table(limits_table(&[&self.chart.location, &self.chart.dispersion]));
        let mut t = Table::new(
            "Part Standardisation",
            vec![Column::new("Part", T), Column::new("Target", S), Column::new("Sigma", S)],
        );
        for ((p, m), s) in self.parts.iter().zip(&self.targets).zip(&self.sigmas) {
            t.row(vec![Cell::text(p), Cell::Num(*m), Cell::Num(*s)]);
        }
        sec.table(t);
        add_tests(&mut sec, &self.chart.location);
        add_tests(&mut sec, &self.chart.dispersion);
        sec.chart(self.chart.location.chart_spec("Z-MR Chart: Standardized Value", "Observation"));
        sec.chart(self.chart.dispersion.chart_spec("Z-MR Chart: Moving Range", "Observation"));
        sec
    }
}

impl Reportable for BoxCoxChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Box-Cox Transformed Control Chart");
        sec.text(format!(
            "Box-Cox transformation with λ = {:.6} (optimal λ = {:.6}).",
            self.transform.lambda, self.transform.optimal_lambda
        ));
        sec.table(limits_table(&[&self.chart.location, &self.chart.dispersion]));
        let mut t = Table::new(
            "Limits in Original Units",
            vec![
                Column::new("Statistic", T),
                Column::new("Center line", S),
                Column::new("UCL", S),
                Column::new("LCL", S),
            ],
        );
        t.row(vec![
            Cell::text("Center / limits (first point)"),
            Cell::Num(self.center[0]),
            Cell::Num(self.ucl[0]),
            Cell::Num(self.lcl[0]),
        ]);
        t.footnotes.push("Limits outside the range of the transformation are shown as missing (*).".into());
        sec.table(t);
        add_tests(&mut sec, &self.chart.location);
        add_tests(&mut sec, &self.chart.dispersion);
        sec.chart(self.chart.location.chart_spec("Transformed Chart (transformed units)", "Observation"));
        sec.chart(self.chart.dispersion.chart_spec("Transformed Dispersion Chart", "Observation"));
        sec
    }
}

impl Reportable for LaneyChart {
    fn report_section(&self) -> Section {
        let title = match self.chart.kind {
            super::charts::AttributeKind::LaneyP => "Laney P' Chart",
            _ => "Laney U' Chart",
        };
        let mut sec = Section::new(title);
        sec.table(limits_table(&[&self.chart.panel]));
        sec.text(format!("σ_z = {:.6}: limits are the ordinary limits widened by this factor.", self.sigma_z));
        add_tests(&mut sec, &self.chart.panel);
        sec.chart(self.chart.panel.chart_spec(title, "Sample"));
        sec
    }
}

impl Reportable for RareEventChart {
    fn report_section(&self) -> Section {
        let (title, label) = match self.kind {
            RareEventKind::G => ("G Chart", self.model.as_str()),
            RareEventKind::T => ("T Chart", self.model.as_str()),
        };
        let mut sec = Section::new(title);
        sec.table(limits_table(&[&self.panel]));
        let mut t = Table::new("Fitted Model", vec![Column::new("Item", T), Column::new("Value", S)]);
        t.row(vec![Cell::text("Model"), Cell::text(label)]);
        match self.kind {
            RareEventKind::G => t.row(vec![Cell::text("Event probability p"), Cell::Num(self.param1)]),
            RareEventKind::T => {
                t.row(vec![Cell::text("Shape"), Cell::Num(self.param1)]);
                t.row(vec![Cell::text("Scale"), Cell::Num(self.param2)])
            }
        };
        t.footnotes.push(format!(
            "Center line is the median; limits are the {:.5} and {:.5} quantiles of the fitted distribution.",
            self.tail_probability,
            1.0 - self.tail_probability
        ));
        sec.table(t);
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(title, "Observation"));
        sec
    }
}

fn smoothing_title(k: SmoothingKind) -> String {
    match k {
        SmoothingKind::MovingAverage(l) => format!("Moving Average Chart (length {l})"),
        SmoothingKind::Ewma { lambda, exact } => {
            format!("EWMA Chart (λ = {lambda}, {} limits)", if exact { "exact" } else { "asymptotic" })
        }
    }
}

impl Reportable for SmoothingChart {
    fn report_section(&self) -> Section {
        let title = smoothing_title(self.kind);
        let mut sec = Section::new(&title);
        sec.table(limits_table(&[&self.panel]));
        sec.text(format!("Target = {:.6}; σ = {:.6} ({}).", self.target, self.sigma, self.sigma_method));
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(&title, "Sample"));
        sec
    }
}

fn cusum_spec(upper: &[f64], lower: &[f64], h: f64, title: &str) -> ChartSpec {
    let x: Vec<f64> = (1..=upper.len()).map(|i| i as f64).collect();
    ChartSpec {
        kind: ChartKind::Line,
        title: title.into(),
        x_label: "Sample".into(),
        y_label: "Cumulative sum".into(),
        categories: Vec::new(),
        series: vec![
            Series { name: "Upper CUSUM".into(), x: x.clone(), y: upper.to_vec() },
            Series { name: "Lower CUSUM".into(), x, y: lower.iter().map(|v| -v).collect() },
        ],
        reference_lines: vec![
            ReferenceLine { axis: Axis::Y, value: h, label: "UCL".into() },
            ReferenceLine { axis: Axis::Y, value: -h, label: "LCL".into() },
            ReferenceLine { axis: Axis::Y, value: 0.0, label: "Center".into() },
        ],
    }
}

fn point_list(v: &[usize]) -> String {
    v.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join(", ")
}

impl Reportable for CusumChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("CUSUM Chart");
        let mut t = Table::new("CUSUM Parameters", vec![Column::new("Item", T), Column::new("Value", S)]);
        t.row(vec![Cell::text("Target"), Cell::Num(self.target)]);
        t.row(vec![Cell::text("Sigma"), Cell::Num(self.sigma)]);
        t.row(vec![Cell::text("K (σ units)"), Cell::Num(self.options.k)]);
        t.row(vec![Cell::text("H (σ units)"), Cell::Num(self.options.h)]);
        t.row(vec![Cell::text("Head start (σ units)"), Cell::Num(self.options.head_start)]);
        t.footnotes.push(format!("σ estimated by {}. Sums are in standard-error units.", self.sigma_method));
        sec.table(t);
        if self.signals_upper.is_empty() && self.signals_lower.is_empty() {
            sec.text("No points are out of control.");
        }
        if !self.signals_upper.is_empty() {
            sec.text(format!("Upper CUSUM beyond H at points: {}", point_list(&self.signals_upper)));
        }
        if !self.signals_lower.is_empty() {
            sec.text(format!("Lower CUSUM beyond H at points: {}", point_list(&self.signals_lower)));
        }
        sec.chart(cusum_spec(&self.upper, &self.lower, self.options.h, "CUSUM Chart"));
        sec
    }
}

impl Reportable for VMaskChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("CUSUM Chart (V-Mask)");
        let mut t = Table::new("V-Mask Parameters", vec![Column::new("Item", T), Column::new("Value", S)]);
        t.row(vec![Cell::text("Slope k (σ units per observation)"), Cell::Num(self.k)]);
        t.row(vec![Cell::text("H (σ units)"), Cell::Num(self.h)]);
        t.row(vec![Cell::text("Lead distance d = h/k"), Cell::Num(self.lead_distance)]);
        t.row(vec![Cell::text("Half-angle atan(k) (radians)"), Cell::Num(self.half_angle)]);
        sec.table(t);
        if self.signals_upper.is_empty() && self.signals_lower.is_empty() {
            sec.text("No points are out of control.");
        }
        if !self.signals_upper.is_empty() {
            sec.text(format!("Upward shift signalled at points: {}", point_list(&self.signals_upper)));
        }
        if !self.signals_lower.is_empty() {
            sec.text(format!("Downward shift signalled at points: {}", point_list(&self.signals_lower)));
        }
        let x: Vec<f64> = (1..=self.sums.len()).map(|i| i as f64).collect();
        sec.chart(ChartSpec {
            kind: ChartKind::Line,
            title: "V-Mask CUSUM".into(),
            x_label: "Sample".into(),
            y_label: "Cumulative sum".into(),
            categories: Vec::new(),
            series: vec![Series { name: "Cumulative sum".into(), x, y: self.sums.clone() }],
            reference_lines: Vec::new(),
        });
        sec
    }
}

impl Reportable for HotellingChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Hotelling T-Squared Chart");
        let mut t = Table::new("T² Control Limit", vec![Column::new("Item", T), Column::new("Value", S)]);
        t.row(vec![Cell::text("Variables"), Cell::Int(self.p as i64)]);
        t.row(vec![Cell::text("Points"), Cell::Int(self.m as i64)]);
        t.row(vec![Cell::text("Subgroup size"), Cell::Int(self.n as i64)]);
        t.row(vec![Cell::text("Alpha"), Cell::Num(self.alpha)]);
        t.row(vec![Cell::text("UCL"), Cell::Num(self.ucl)]);
        t.footnotes.push(self.limit_description.clone());
        sec.table(t);
        if self.violations.is_empty() {
            sec.text("No points failed the test (T² above the upper control limit).");
        } else {
            for v in &self.violations {
                sec.text(format!("TEST 1. T² above the upper control limit at points: {}", point_list(&v.points)));
            }
        }
        let x: Vec<f64> = (1..=self.t2.len()).map(|i| i as f64).collect();
        let flagged = self.flagged();
        sec.chart(ChartSpec {
            kind: ChartKind::Line,
            title: "Hotelling T-Squared Chart".into(),
            x_label: "Sample".into(),
            y_label: "T-squared".into(),
            categories: Vec::new(),
            series: vec![
                Series { name: "T-squared".into(), x: x.clone(), y: self.t2.clone() },
                Series { name: "UCL".into(), x, y: vec![self.ucl; self.t2.len()] },
                Series {
                    name: "Flagged".into(),
                    x: flagged.iter().map(|&i| (i + 1) as f64).collect(),
                    y: flagged.iter().map(|&i| self.t2[i]).collect(),
                },
            ],
            reference_lines: Vec::new(),
        });
        sec
    }
}

// ------------------------------------------------------------ Phase 2h additions

impl Reportable for StagedSmoothingChart {
    fn report_section(&self) -> Section {
        let title = match self.charts[0].kind {
            SmoothingKind::MovingAverage(_) => "Staged Moving Average Chart",
            SmoothingKind::Ewma { .. } => "Staged EWMA Chart",
        };
        let mut sec = Section::new(title);
        sec.table(stage_table(&self.stages));
        let mut t = Table::new(
            "Estimates by Stage",
            vec![Column::new("Stage", T), Column::new("Target", S), Column::new("Sigma", S)],
        );
        for (s, c) in self.stages.iter().zip(&self.charts) {
            t.row(vec![Cell::text(&s.label), Cell::Num(c.target), Cell::Num(c.sigma)]);
        }
        sec.table(t);
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(title, "Sample"));
        sec
    }
}

impl Reportable for StagedCusumChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Staged CUSUM Chart");
        sec.table(stage_table(&self.stages));
        if self.signals_upper.is_empty() && self.signals_lower.is_empty() {
            sec.text("No points are out of control.");
        }
        if !self.signals_upper.is_empty() {
            sec.text(format!("Upper CUSUM beyond H at points: {}", point_list(&self.signals_upper)));
        }
        if !self.signals_lower.is_empty() {
            sec.text(format!("Lower CUSUM beyond H at points: {}", point_list(&self.signals_lower)));
        }
        sec.chart(cusum_spec(&self.upper, &self.lower, self.charts[0].options.h, "Staged CUSUM Chart"));
        sec
    }
}

impl Reportable for StagedLaneyChart {
    fn report_section(&self) -> Section {
        let title = if self.kind == super::charts::AttributeKind::LaneyP {
            "Staged Laney P' Chart"
        } else {
            "Staged Laney U' Chart"
        };
        let mut sec = Section::new(title);
        sec.table(stage_table(&self.stages));
        let mut t = Table::new("Sigma_z by Stage", vec![Column::new("Stage", T), Column::new("Sigma_z", S)]);
        for (s, c) in self.stages.iter().zip(&self.charts) {
            t.row(vec![Cell::text(&s.label), Cell::Num(c.sigma_z)]);
        }
        sec.table(t);
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(title, "Sample"));
        sec
    }
}

impl Reportable for StagedRareEventChart {
    fn report_section(&self) -> Section {
        let title = match self.charts[0].kind {
            RareEventKind::G => "Staged G Chart",
            RareEventKind::T => "Staged T Chart",
        };
        let mut sec = Section::new(title);
        sec.table(stage_table(&self.stages));
        sec.table(limits_table(&[&self.panel]));
        add_tests(&mut sec, &self.panel);
        sec.chart(self.panel.chart_spec(title, "Observation"));
        sec
    }
}

impl Reportable for StagedHotellingChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Staged Hotelling T-Squared Chart");
        sec.table(stage_table(&self.stages));
        let mut t = Table::new("T² Limit by Stage", vec![Column::new("Stage", T), Column::new("UCL", S)]);
        for (s, c) in self.stages.iter().zip(&self.charts) {
            t.row(vec![Cell::text(&s.label), Cell::Num(c.ucl)]);
        }
        sec.table(t);
        if self.flagged.is_empty() {
            sec.text("No points failed the test (T² above the upper control limit).");
        } else {
            sec.text(format!("TEST 1. T² above the upper control limit at points: {}", point_list(&self.flagged)));
        }
        let x: Vec<f64> = (1..=self.t2.len()).map(|i| i as f64).collect();
        sec.chart(ChartSpec {
            kind: ChartKind::Line,
            title: "Staged Hotelling T-Squared Chart".into(),
            x_label: "Sample".into(),
            y_label: "T-squared".into(),
            categories: Vec::new(),
            series: vec![
                Series { name: "T-squared".into(), x: x.clone(), y: self.t2.clone() },
                Series { name: "UCL".into(), x, y: self.ucl.clone() },
            ],
            reference_lines: Vec::new(),
        });
        sec
    }
}

impl Reportable for HotellingDecomposition {
    fn report_section(&self) -> Section {
        let mut sec = self.chart.report_section();
        sec.heading = "Hotelling T-Squared Chart with Variable Contributions".into();
        let p = self.chart.p;
        let flagged = self.chart.flagged();
        if !flagged.is_empty() {
            let mut cols = vec![Column::new("Point", I), Column::new("T²", S)];
            cols.extend((1..=p).map(|j| Column::new(format!("Var {j}"), S)));
            let mut t = Table::new("Contributions to T² of Out-of-Control Points", cols);
            for &i in &flagged {
                let mut r = vec![Cell::Int(i as i64 + 1), Cell::Num(self.chart.t2[i])];
                r.extend((0..p).map(|j| Cell::Num(self.contributions[(i, j)])));
                t.row(r);
            }
            t.footnotes.push("Contributions n·dⱼ(S⁻¹d)ⱼ add up to T²; a contribution can be negative.".into());
            sec.table(t);
        }
        let mut u = Table::new(
            "Unconditional (One-Variable) T² Charts",
            vec![Column::new("Variable", I), Column::new("UCL", S), Column::new("Points above UCL", T)],
        );
        for (j, c) in self.unconditional.iter().enumerate() {
            let pts = c.flagged();
            u.row(vec![
                Cell::Int(j as i64 + 1),
                Cell::Num(c.ucl),
                Cell::text(if pts.is_empty() { "none".to_string() } else { point_list(&pts) }),
            ]);
        }
        sec.table(u);
        sec
    }
}

impl Reportable for GeneralizedVarianceChart {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Generalized Variance Chart");
        let mut t = Table::new("Generalized Variance Limits", vec![Column::new("Item", T), Column::new("Value", S)]);
        t.row(vec![Cell::text("Variables"), Cell::Int(self.p as i64)]);
        t.row(vec![Cell::text("Subgroup size"), Cell::Int(self.n as i64)]);
        t.row(vec![Cell::text("|Sigma|"), Cell::Num(self.sigma_det)]);
        t.row(vec![Cell::text("UCL"), Cell::Num(self.ucl)]);
        t.row(vec![Cell::text("Center line"), Cell::Num(self.center)]);
        t.row(vec![Cell::text("LCL"), Cell::Num(self.lcl)]);
        t.footnotes
            .push(format!("b1 = {:.6}, b2 = {:.6}; limits are b1|Sigma| ± k·sqrt(b2)|Sigma|.", self.b1, self.b2));
        sec.table(t);
        if self.violations.is_empty() {
            sec.text("No points failed the test (|S| outside the control limits).");
        } else {
            for v in &self.violations {
                sec.text(format!("TEST 1. |S| outside the control limits at points: {}", point_list(&v.points)));
            }
        }
        let m = self.det.len();
        let x: Vec<f64> = (1..=m).map(|i| i as f64).collect();
        let flagged = self.flagged();
        sec.chart(ChartSpec {
            kind: ChartKind::Line,
            title: "Generalized Variance Chart".into(),
            x_label: "Sample".into(),
            y_label: "|S|".into(),
            categories: Vec::new(),
            series: vec![
                Series { name: "|S|".into(), x: x.clone(), y: self.det.clone() },
                Series { name: "UCL".into(), x: x.clone(), y: vec![self.ucl; m] },
                Series { name: "Center".into(), x: x.clone(), y: vec![self.center; m] },
                Series { name: "LCL".into(), x, y: vec![self.lcl; m] },
                Series {
                    name: "Flagged".into(),
                    x: flagged.iter().map(|&i| (i + 1) as f64).collect(),
                    y: flagged.iter().map(|&i| self.det[i]).collect(),
                },
            ],
            reference_lines: Vec::new(),
        });
        sec
    }
}
