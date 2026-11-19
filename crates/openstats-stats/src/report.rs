//! The unified session-report model.
//!
//! Every procedure in `openstats-stats` can describe its result as a
//! [`SessionReport`] (through the [`Reportable`] trait, implemented in
//! [`crate::reportable`]). A report is plain data — sections made of text,
//! formatted [`Table`]s and [`ChartSpec`]s, plus footnotes and warnings — so
//! the same object drives
//!
//! * the Minitab-style monospaced *Session window* ([`SessionReport::render_text`]),
//! * lossless JSON for the UI and the script engine
//!   ([`SessionReport::to_json`] / [`SessionReport::from_json`]), and
//! * native chart rendering from the attached [`ChartSpec`]s.
//!
//! Numbers are stored unformatted ([`Cell::Num`]); the column's
//! [`ColumnKind`] decides how they are shown, which keeps decimal points
//! aligned (every cell of a fixed-point column uses the same number of
//! decimals) and p-values in the Minitab style (`0.000` below 0.0005).

use crate::error::{invalid, Result};
use std::fmt::Write as _;

/// A whole session-window output: a title and a list of sections.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionReport {
    /// Title printed (and underlined) at the top.
    pub title: String,
    /// Sections in output order.
    pub sections: Vec<Section>,
    /// Warnings that concern the whole report.
    pub warnings: Vec<String>,
}

/// A titled group of blocks, e.g. "One-Sample T: Weight".
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Section {
    /// Section heading.
    pub heading: String,
    /// Content in output order.
    pub blocks: Vec<Block>,
    /// Warnings that concern this section.
    pub warnings: Vec<String>,
}

/// One piece of section content.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// A paragraph (may contain line breaks).
    Text(String),
    /// A formatted table.
    Table(Table),
    /// A chart specification.
    Chart(ChartSpec),
}

/// How the cells of a column are formatted and aligned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnKind {
    /// Left-aligned text.
    Text,
    /// Right-aligned integers.
    Integer,
    /// Fixed-point with this many decimals (decimal points align).
    Fixed(u8),
    /// At least this many significant digits for every cell; the column uses
    /// the largest number of decimals any cell needs so decimals align.
    Significant(u8),
    /// p-values: three decimals, `0.000` when below 0.0005.
    PValue,
}

/// A column header with its formatting rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Header text.
    pub header: String,
    /// Formatting rule.
    pub kind: ColumnKind,
}

/// One table cell.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    /// Text.
    Text(String),
    /// Integer.
    Int(i64),
    /// Real number (NaN is shown as the missing value `*`).
    Num(f64),
    /// Missing value, shown `*`.
    Missing,
}

impl Cell {
    /// `Num(x)` for `Some(x)`, [`Cell::Missing`] for `None`.
    pub fn opt(x: Option<f64>) -> Cell {
        x.map_or(Cell::Missing, Cell::Num)
    }
    /// Text cell from anything string-like.
    pub fn text(s: impl Into<String>) -> Cell {
        Cell::Text(s.into())
    }
}

/// A rectangular table with footnotes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table {
    /// Caption printed above the table (may be empty).
    pub title: String,
    /// Column definitions.
    pub columns: Vec<Column>,
    /// Rows; each has exactly one cell per column.
    pub rows: Vec<Vec<Cell>>,
    /// Footnotes printed below the table.
    pub footnotes: Vec<String>,
}

/// Kind of a chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartKind {
    /// Bars from `x` = bin edges (series `x` has one more entry than `y`) .
    Histogram,
    /// Box-and-whisker plot: one series per box, `y` = (min whisker, Q1, median, Q3, max whisker).
    Boxplot,
    /// Normal probability plot: `x` = data, `y` = normal scores / percents.
    ProbabilityPlot,
    /// Points.
    Scatter,
    /// Connected points.
    Line,
    /// Bars at categories (`ChartSpec::categories`) with heights `y`.
    Bar,
    /// Interval plot: `x` = estimate, `y` = group position, with lower/upper
    /// in series named "lower" / "upper".
    IntervalPlot,
    /// Contour plot: one series per contour level (named by the level); each
    /// series is a set of polylines whose `x`/`y` are the plane coordinates and
    /// in which a `NaN` pair separates disconnected segments.
    Contour,
    /// Response surface: one series per grid row; the series name is the row's
    /// y-coordinate, `x` the grid x-coordinates and `y` the response (z) values.
    Surface,
    /// Pie chart: `categories` are the slices and `series[0].y` their sizes.
    Pie,
    /// Empirical CDF step plots: one series per group, `x` = the observations.
    Ecdf,
    /// Individual value plot: one series per group, `y` = the observations (name = label).
    IndividualValue,
    /// Filled areas under the series (each series drawn in order, back to front).
    Area,
    /// Bubble plot: `series[0]` holds the points (`x`, `y`); a series named `Size` holds the
    /// bubble sizes in `y`.
    Bubble,
    /// Heat map: laid out like [`ChartKind::Surface`] (one series per row, named by the row's
    /// y value, `x` = column positions, `y` = cell values).
    Heatmap,
    /// Dotplot: one series per group, `x` = the observations; dots are stacked in bins.
    Dotplot,
    /// Matrix plot: `categories` are the variable names; the series named `"i|j"` hold the
    /// scatter of variable `j` (x) against variable `i` (y).
    Matrix,
    /// Marginal plot: `series[0]` holds the scatter (`x`, `y`); histograms of x and y are drawn
    /// in the margins.
    Marginal,
    /// Stacked bars at categories: series are stacked in order.
    StackedBar,
    /// Probability plot of raw data: `series[0].x` = the observations and the series name is
    /// `Normal`, `Lognormal` or `Weibull`.
    DataProbability,
    /// Three-dimensional scatter: `series[0]` has `x`, `y` and (as `categories` is empty) the
    /// z values in a series named `Z` with the same length.
    Scatter3D,
    /// Connected points whose `x` values are Excel serial days; the x axis shows calendar ticks.
    TimeLine,
    /// Pareto chart: `categories` and `series[0].y` (counts); bars are sorted and a cumulative-percent line is drawn.
    Pareto,
}

/// Axis of a [`ReferenceLine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// A vertical line at an x value.
    X,
    /// A horizontal line at a y value.
    Y,
}

/// A data series of a chart.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    /// Legend name.
    pub name: String,
    /// x values.
    pub x: Vec<f64>,
    /// y values.
    pub y: Vec<f64>,
}

/// A labelled reference line (target, specification limit, α, …).
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceLine {
    /// Axis the line is perpendicular to.
    pub axis: Axis,
    /// Position on that axis.
    pub value: f64,
    /// Label.
    pub label: String,
}

/// Declarative chart: enough for any front end to draw it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartSpec {
    /// Chart kind.
    pub kind: ChartKind,
    /// Chart title.
    pub title: String,
    /// x-axis label.
    pub x_label: String,
    /// y-axis label.
    pub y_label: String,
    /// Category labels for [`ChartKind::Bar`] (empty otherwise).
    pub categories: Vec<String>,
    /// Data series.
    pub series: Vec<Series>,
    /// Reference lines.
    pub reference_lines: Vec<ReferenceLine>,
}

/// Anything that can describe its result as a report section.
pub trait Reportable {
    /// The section describing this result.
    fn report_section(&self) -> Section;

    /// A one-section report titled `title`.
    fn to_report(&self, title: &str) -> SessionReport {
        let mut r = SessionReport::new(title);
        r.push(self.report_section());
        r
    }
}

impl Column {
    /// Column with a header and kind.
    pub fn new(header: impl Into<String>, kind: ColumnKind) -> Column {
        Column { header: header.into(), kind }
    }
}

impl Table {
    /// Empty table with the given caption and columns.
    pub fn new(title: impl Into<String>, columns: Vec<Column>) -> Table {
        Table { title: title.into(), columns, rows: Vec::new(), footnotes: Vec::new() }
    }

    /// Append a row; errors unless it has one cell per column.
    pub fn push_row(&mut self, row: Vec<Cell>) -> Result<()> {
        if row.len() != self.columns.len() {
            return invalid(format!("table row has {} cells but {} columns", row.len(), self.columns.len()));
        }
        self.rows.push(row);
        Ok(())
    }

    /// Append a row built by code that knows the width (cannot fail for
    /// internally constructed tables; a wrong width is padded / truncated
    /// with missing values so rendering never panics).
    pub(crate) fn row(&mut self, mut row: Vec<Cell>) {
        row.resize(self.columns.len(), Cell::Missing);
        self.rows.push(row);
    }

    /// Check that all rows are rectangular.
    pub fn validate(&self) -> Result<()> {
        for (i, r) in self.rows.iter().enumerate() {
            if r.len() != self.columns.len() {
                return invalid(format!("row {i} has {} cells, expected {}", r.len(), self.columns.len()));
            }
        }
        Ok(())
    }

    /// The formatted text of every cell, column by column.
    pub fn formatted_columns(&self) -> Vec<Vec<String>> {
        (0..self.columns.len())
            .map(|c| {
                let cells: Vec<&Cell> = self.rows.iter().map(|r| r.get(c).unwrap_or(&Cell::Missing)).collect();
                format_column(self.columns[c].kind, &cells)
            })
            .collect()
    }

    /// Monospaced rendering: caption, header, rule, rows, footnotes.
    pub fn render_text(&self) -> String {
        let cols = self.formatted_columns();
        let widths: Vec<usize> = self
            .columns
            .iter()
            .zip(&cols)
            .map(|(c, cells)| {
                cells.iter().map(|s| s.chars().count()).chain([c.header.chars().count()]).max().unwrap_or(0)
            })
            .collect();
        let mut out = String::new();
        if !self.title.is_empty() {
            let _ = writeln!(out, "{}", self.title);
            out.push('\n');
        }
        let line = |out: &mut String, items: Vec<(String, bool)>| {
            let mut s = String::new();
            for (i, (text, left)) in items.into_iter().enumerate() {
                if i > 0 {
                    s.push_str("  ");
                }
                let pad = widths[i].saturating_sub(text.chars().count());
                if left {
                    s.push_str(&text);
                    s.push_str(&" ".repeat(pad));
                } else {
                    s.push_str(&" ".repeat(pad));
                    s.push_str(&text);
                }
            }
            out.push_str(s.trim_end());
            out.push('\n');
        };
        line(&mut out, self.columns.iter().map(|c| (c.header.clone(), matches!(c.kind, ColumnKind::Text))).collect());
        let total: usize = widths.iter().sum::<usize>() + 2 * widths.len().saturating_sub(1);
        let _ = writeln!(out, "{}", "-".repeat(total));
        for r in 0..self.rows.len() {
            line(
                &mut out,
                self.columns
                    .iter()
                    .enumerate()
                    .map(|(c, col)| (cols[c][r].clone(), matches!(col.kind, ColumnKind::Text)))
                    .collect(),
            );
        }
        for (i, f) in self.footnotes.iter().enumerate() {
            if i == 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "{f}");
        }
        out
    }
}

impl Section {
    /// Empty section with a heading.
    pub fn new(heading: impl Into<String>) -> Section {
        Section { heading: heading.into(), blocks: Vec::new(), warnings: Vec::new() }
    }
    /// Append a text paragraph.
    pub fn text(&mut self, s: impl Into<String>) -> &mut Self {
        self.blocks.push(Block::Text(s.into()));
        self
    }
    /// Append a table.
    pub fn table(&mut self, t: Table) -> &mut Self {
        self.blocks.push(Block::Table(t));
        self
    }
    /// Append a chart.
    pub fn chart(&mut self, c: ChartSpec) -> &mut Self {
        self.blocks.push(Block::Chart(c));
        self
    }
    /// Append a warning.
    pub fn warn(&mut self, s: impl Into<String>) -> &mut Self {
        self.warnings.push(s.into());
        self
    }
}

impl SessionReport {
    /// Empty report.
    pub fn new(title: impl Into<String>) -> SessionReport {
        SessionReport { title: title.into(), sections: Vec::new(), warnings: Vec::new() }
    }

    /// Append a section.
    pub fn push(&mut self, s: Section) -> &mut Self {
        self.sections.push(s);
        self
    }

    /// Append every section of `other` (and its warnings).
    pub fn extend(&mut self, other: SessionReport) -> &mut Self {
        self.sections.extend(other.sections);
        self.warnings.extend(other.warnings);
        self
    }

    /// All charts in the report, in order.
    pub fn charts(&self) -> Vec<&ChartSpec> {
        self.sections
            .iter()
            .flat_map(|s| s.blocks.iter())
            .filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None })
            .collect()
    }

    /// All tables in the report, in order.
    pub fn tables(&self) -> Vec<&Table> {
        self.sections
            .iter()
            .flat_map(|s| s.blocks.iter())
            .filter_map(|b| if let Block::Table(t) = b { Some(t) } else { None })
            .collect()
    }

    /// Validate every table.
    pub fn validate(&self) -> Result<()> {
        self.tables().into_iter().try_for_each(Table::validate)
    }

    /// Minitab-style session text.
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        if !self.title.is_empty() {
            let _ = writeln!(out, "{}", self.title);
            let _ = writeln!(out, "{}", "=".repeat(self.title.chars().count()));
            out.push('\n');
        }
        for s in &self.sections {
            if !s.heading.is_empty() {
                let _ = writeln!(out, "{}", s.heading);
                let _ = writeln!(out, "{}", "-".repeat(s.heading.chars().count()));
                out.push('\n');
            }
            for b in &s.blocks {
                match b {
                    Block::Text(t) => {
                        let _ = writeln!(out, "{t}");
                    }
                    Block::Table(t) => out.push_str(&t.render_text()),
                    Block::Chart(c) => {
                        let _ = writeln!(out, "[Chart: {} ({:?}), {} series]", c.title, c.kind, c.series.len());
                    }
                }
                out.push('\n');
            }
            for w in &s.warnings {
                let _ = writeln!(out, "* WARNING * {w}");
            }
            if !s.warnings.is_empty() {
                out.push('\n');
            }
        }
        for w in &self.warnings {
            let _ = writeln!(out, "* WARNING * {w}");
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

fn decimals_for_sig(x: f64, sig: u8) -> usize {
    if x == 0.0 || !x.is_finite() {
        return sig.saturating_sub(1) as usize;
    }
    let mag = x.abs().log10().floor() as i64;
    (i64::from(sig) - 1 - mag).clamp(0, 15) as usize
}

fn fmt_special(x: f64) -> Option<String> {
    if x.is_nan() {
        Some("*".to_string())
    } else if x == f64::INFINITY {
        Some("∞".to_string())
    } else if x == f64::NEG_INFINITY {
        Some("-∞".to_string())
    } else {
        None
    }
}

/// Format a p-value the Minitab way.
pub fn format_p_value(p: f64) -> String {
    if let Some(s) = fmt_special(p) {
        return s;
    }
    if p < 0.0005 {
        "0.000".to_string()
    } else {
        format!("{p:.3}")
    }
}

fn format_column(kind: ColumnKind, cells: &[&Cell]) -> Vec<String> {
    let sig_decimals = match kind {
        ColumnKind::Significant(s) => cells
            .iter()
            .filter_map(|c| if let Cell::Num(x) = c { Some(decimals_for_sig(*x, s)) } else { None })
            .max()
            .unwrap_or(0),
        _ => 0,
    };
    cells
        .iter()
        .map(|c| match c {
            Cell::Missing => "*".to_string(),
            Cell::Text(s) => s.clone(),
            Cell::Int(i) => i.to_string(),
            Cell::Num(x) => {
                if let Some(s) = fmt_special(*x) {
                    return s;
                }
                match kind {
                    ColumnKind::PValue => format_p_value(*x),
                    ColumnKind::Fixed(d) => format!("{x:.*}", d as usize),
                    ColumnKind::Significant(_) => format!("{x:.*}", sig_decimals),
                    ColumnKind::Integer => format!("{x:.0}"),
                    ColumnKind::Text => format!("{x}"),
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// JSON (hand-written, lossless for f64)
// ---------------------------------------------------------------------------

/// Minimal JSON value used by the (de)serialiser.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn esc(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A real number: finite values as JSON numbers, others as the strings
/// `"NaN"`, `"Infinity"`, `"-Infinity"`.
fn num(x: f64, out: &mut String) {
    if x.is_nan() {
        out.push_str("\"NaN\"");
    } else if x == f64::INFINITY {
        out.push_str("\"Infinity\"");
    } else if x == f64::NEG_INFINITY {
        out.push_str("\"-Infinity\"");
    } else {
        let _ = write!(out, "{x:?}");
    }
}

fn arr<T>(items: &[T], out: &mut String, mut f: impl FnMut(&T, &mut String)) {
    out.push('[');
    for (i, it) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        f(it, out);
    }
    out.push(']');
}

fn strs(items: &[String], out: &mut String) {
    arr(items, out, |s, o| esc(s, o));
}

fn kind_name(k: ColumnKind) -> String {
    match k {
        ColumnKind::Text => "text".into(),
        ColumnKind::Integer => "integer".into(),
        ColumnKind::Fixed(d) => format!("fixed:{d}"),
        ColumnKind::Significant(d) => format!("sig:{d}"),
        ColumnKind::PValue => "pvalue".into(),
    }
}

fn chart_kind_name(k: ChartKind) -> &'static str {
    match k {
        ChartKind::Histogram => "histogram",
        ChartKind::Boxplot => "boxplot",
        ChartKind::ProbabilityPlot => "probability_plot",
        ChartKind::Scatter => "scatter",
        ChartKind::Line => "line",
        ChartKind::Bar => "bar",
        ChartKind::IntervalPlot => "interval_plot",
        ChartKind::Contour => "contour",
        ChartKind::Surface => "surface",
        ChartKind::Pie => "pie",
        ChartKind::Ecdf => "ecdf",
        ChartKind::IndividualValue => "individual_value",
        ChartKind::Area => "area",
        ChartKind::Bubble => "bubble",
        ChartKind::Heatmap => "heatmap",
        ChartKind::Dotplot => "dotplot",
        ChartKind::Matrix => "matrix",
        ChartKind::Marginal => "marginal",
        ChartKind::StackedBar => "stacked_bar",
        ChartKind::DataProbability => "data_probability",
        ChartKind::Scatter3D => "scatter3d",
        ChartKind::TimeLine => "timeline",
        ChartKind::Pareto => "pareto",
    }
}

impl SessionReport {
    /// Serialise to compact JSON. Finite numbers round-trip exactly.
    pub fn to_json(&self) -> String {
        let mut o = String::new();
        o.push_str("{\"title\":");
        esc(&self.title, &mut o);
        o.push_str(",\"warnings\":");
        strs(&self.warnings, &mut o);
        o.push_str(",\"sections\":");
        arr(&self.sections, &mut o, |s, o| {
            o.push_str("{\"heading\":");
            esc(&s.heading, o);
            o.push_str(",\"warnings\":");
            strs(&s.warnings, o);
            o.push_str(",\"blocks\":");
            arr(&s.blocks, o, |b, o| match b {
                Block::Text(t) => {
                    o.push_str("{\"type\":\"text\",\"text\":");
                    esc(t, o);
                    o.push('}');
                }
                Block::Table(t) => {
                    o.push_str("{\"type\":\"table\",\"title\":");
                    esc(&t.title, o);
                    o.push_str(",\"columns\":");
                    arr(&t.columns, o, |c, o| {
                        o.push_str("{\"header\":");
                        esc(&c.header, o);
                        o.push_str(",\"kind\":");
                        esc(&kind_name(c.kind), o);
                        o.push('}');
                    });
                    o.push_str(",\"rows\":");
                    arr(&t.rows, o, |r, o| {
                        arr(r, o, |c, o| match c {
                            Cell::Missing => o.push_str("null"),
                            Cell::Text(s) => esc(s, o),
                            Cell::Int(i) => {
                                let _ = write!(o, "{{\"i\":{i}}}");
                            }
                            Cell::Num(x) => {
                                o.push_str("{\"n\":");
                                num(*x, o);
                                o.push('}');
                            }
                        });
                    });
                    o.push_str(",\"footnotes\":");
                    strs(&t.footnotes, o);
                    o.push('}');
                }
                Block::Chart(c) => {
                    o.push_str("{\"type\":\"chart\",\"kind\":");
                    esc(chart_kind_name(c.kind), o);
                    o.push_str(",\"title\":");
                    esc(&c.title, o);
                    o.push_str(",\"x_label\":");
                    esc(&c.x_label, o);
                    o.push_str(",\"y_label\":");
                    esc(&c.y_label, o);
                    o.push_str(",\"categories\":");
                    strs(&c.categories, o);
                    o.push_str(",\"series\":");
                    arr(&c.series, o, |s, o| {
                        o.push_str("{\"name\":");
                        esc(&s.name, o);
                        o.push_str(",\"x\":");
                        arr(&s.x, o, |v, o| num(*v, o));
                        o.push_str(",\"y\":");
                        arr(&s.y, o, |v, o| num(*v, o));
                        o.push('}');
                    });
                    o.push_str(",\"reference_lines\":");
                    arr(&c.reference_lines, o, |l, o| {
                        o.push_str("{\"axis\":");
                        esc(if l.axis == Axis::X { "x" } else { "y" }, o);
                        o.push_str(",\"value\":");
                        num(l.value, o);
                        o.push_str(",\"label\":");
                        esc(&l.label, o);
                        o.push('}');
                    });
                    o.push('}');
                }
            });
            o.push('}');
        });
        o.push('}');
        o
    }

    /// Parse a report produced by [`SessionReport::to_json`].
    pub fn from_json(text: &str) -> Result<SessionReport> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return invalid("trailing characters after JSON value");
        }
        decode_report(&v)
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<()> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            invalid(format!("JSON: expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn lit(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            invalid(format!("JSON: bad literal at byte {}", self.i))
        }
    }
    fn value(&mut self) -> Result<Json> {
        self.ws();
        match self.s.get(self.i) {
            None => invalid("JSON: unexpected end"),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    v.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(v));
                        }
                        _ => return invalid(format!("JSON: expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(v));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    v.push((k, self.value()?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(v));
                        }
                        _ => return invalid(format!("JSON: expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(_) => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[st..self.i])
                    .ok()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(Json::Num)
                    .ok_or_else(|| crate::error::StatsError::InvalidArgument(format!("JSON: bad number at byte {st}")))
            }
        }
    }
    fn hex4(&mut self) -> Result<u32> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .and_then(|b| std::str::from_utf8(b).ok())
            .and_then(|t| u32::from_str_radix(t, 16).ok());
        match h {
            Some(v) => {
                self.i += 4;
                Ok(v)
            }
            None => invalid("JSON: bad \\u escape"),
        }
    }
    fn string(&mut self) -> Result<String> {
        if self.s.get(self.i) != Some(&b'"') {
            return invalid(format!("JSON: expected string at byte {}", self.i));
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                return invalid("JSON: unterminated string");
            };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return invalid("JSON: bad escape");
                    };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return invalid("JSON: lone surrogate");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return invalid("JSON: bad surrogate pair");
                                }
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            match char::from_u32(cp) {
                                Some(ch) => ch,
                                None => return invalid("JSON: invalid code point"),
                            }
                        }
                        _ => return invalid("JSON: bad escape"),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).or_else(|_| invalid("JSON: invalid UTF-8"))
    }
}

fn get<'a>(o: &'a [(String, Json)], k: &str) -> Result<&'a Json> {
    o.iter()
        .find(|(n, _)| n == k)
        .map(|(_, v)| v)
        .ok_or_else(|| crate::error::StatsError::InvalidArgument(format!("report JSON: missing field '{k}'")))
}

fn as_obj(v: &Json) -> Result<&[(String, Json)]> {
    if let Json::Obj(o) = v {
        Ok(o)
    } else {
        invalid("report JSON: expected object")
    }
}

fn as_arr(v: &Json) -> Result<&[Json]> {
    if let Json::Arr(a) = v {
        Ok(a)
    } else {
        invalid("report JSON: expected array")
    }
}

fn as_str(v: &Json) -> Result<String> {
    if let Json::Str(s) = v {
        Ok(s.clone())
    } else {
        invalid("report JSON: expected string")
    }
}

fn as_strs(v: &Json) -> Result<Vec<String>> {
    as_arr(v)?.iter().map(as_str).collect()
}

fn as_num(v: &Json) -> Result<f64> {
    match v {
        Json::Num(x) => Ok(*x),
        Json::Str(s) => match s.as_str() {
            "NaN" => Ok(f64::NAN),
            "Infinity" => Ok(f64::INFINITY),
            "-Infinity" => Ok(f64::NEG_INFINITY),
            _ => invalid("report JSON: bad number string"),
        },
        _ => invalid("report JSON: expected number"),
    }
}

fn parse_kind(s: &str) -> Result<ColumnKind> {
    let bad = || crate::error::StatsError::InvalidArgument(format!("report JSON: bad column kind '{s}'"));
    Ok(match s {
        "text" => ColumnKind::Text,
        "integer" => ColumnKind::Integer,
        "pvalue" => ColumnKind::PValue,
        _ => {
            let (a, b) = s.split_once(':').ok_or_else(bad)?;
            let d: u8 = b.parse().map_err(|_| bad())?;
            match a {
                "fixed" => ColumnKind::Fixed(d),
                "sig" => ColumnKind::Significant(d),
                _ => return Err(bad()),
            }
        }
    })
}

fn parse_chart_kind(s: &str) -> Result<ChartKind> {
    Ok(match s {
        "histogram" => ChartKind::Histogram,
        "boxplot" => ChartKind::Boxplot,
        "probability_plot" => ChartKind::ProbabilityPlot,
        "scatter" => ChartKind::Scatter,
        "line" => ChartKind::Line,
        "bar" => ChartKind::Bar,
        "interval_plot" => ChartKind::IntervalPlot,
        "contour" => ChartKind::Contour,
        "surface" => ChartKind::Surface,
        "pie" => ChartKind::Pie,
        "ecdf" => ChartKind::Ecdf,
        "individual_value" => ChartKind::IndividualValue,
        "area" => ChartKind::Area,
        "bubble" => ChartKind::Bubble,
        "heatmap" => ChartKind::Heatmap,
        "dotplot" => ChartKind::Dotplot,
        "matrix" => ChartKind::Matrix,
        "marginal" => ChartKind::Marginal,
        "stacked_bar" => ChartKind::StackedBar,
        "data_probability" => ChartKind::DataProbability,
        "scatter3d" => ChartKind::Scatter3D,
        "timeline" => ChartKind::TimeLine,
        "pareto" => ChartKind::Pareto,
        _ => return invalid(format!("report JSON: bad chart kind '{s}'")),
    })
}

fn decode_cell(v: &Json) -> Result<Cell> {
    Ok(match v {
        Json::Null => Cell::Missing,
        Json::Str(s) => Cell::Text(s.clone()),
        Json::Obj(o) => {
            if let Ok(i) = get(o, "i") {
                let x = as_num(i)?;
                if x.fract() != 0.0 || x.abs() > 9.0e15 {
                    return invalid("report JSON: bad integer cell");
                }
                Cell::Int(x as i64)
            } else {
                Cell::Num(as_num(get(o, "n")?)?)
            }
        }
        _ => return invalid("report JSON: bad cell"),
    })
}

fn decode_block(v: &Json) -> Result<Block> {
    let o = as_obj(v)?;
    match as_str(get(o, "type")?)?.as_str() {
        "text" => Ok(Block::Text(as_str(get(o, "text")?)?)),
        "table" => {
            let columns = as_arr(get(o, "columns")?)?
                .iter()
                .map(|c| {
                    let co = as_obj(c)?;
                    Ok(Column { header: as_str(get(co, "header")?)?, kind: parse_kind(&as_str(get(co, "kind")?)?)? })
                })
                .collect::<Result<Vec<_>>>()?;
            let rows = as_arr(get(o, "rows")?)?
                .iter()
                .map(|r| as_arr(r)?.iter().map(decode_cell).collect::<Result<Vec<_>>>())
                .collect::<Result<Vec<_>>>()?;
            let t =
                Table { title: as_str(get(o, "title")?)?, columns, rows, footnotes: as_strs(get(o, "footnotes")?)? };
            t.validate()?;
            Ok(Block::Table(t))
        }
        "chart" => {
            let series = as_arr(get(o, "series")?)?
                .iter()
                .map(|s| {
                    let so = as_obj(s)?;
                    Ok(Series {
                        name: as_str(get(so, "name")?)?,
                        x: as_arr(get(so, "x")?)?.iter().map(as_num).collect::<Result<_>>()?,
                        y: as_arr(get(so, "y")?)?.iter().map(as_num).collect::<Result<_>>()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let reference_lines = as_arr(get(o, "reference_lines")?)?
                .iter()
                .map(|l| {
                    let lo = as_obj(l)?;
                    let axis = match as_str(get(lo, "axis")?)?.as_str() {
                        "x" => Axis::X,
                        "y" => Axis::Y,
                        _ => return invalid("report JSON: bad axis"),
                    };
                    Ok(ReferenceLine { axis, value: as_num(get(lo, "value")?)?, label: as_str(get(lo, "label")?)? })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Block::Chart(ChartSpec {
                kind: parse_chart_kind(&as_str(get(o, "kind")?)?)?,
                title: as_str(get(o, "title")?)?,
                x_label: as_str(get(o, "x_label")?)?,
                y_label: as_str(get(o, "y_label")?)?,
                categories: as_strs(get(o, "categories")?)?,
                series,
                reference_lines,
            }))
        }
        other => invalid(format!("report JSON: unknown block type '{other}'")),
    }
}

fn decode_report(v: &Json) -> Result<SessionReport> {
    let o = as_obj(v)?;
    let sections = as_arr(get(o, "sections")?)?
        .iter()
        .map(|s| {
            let so = as_obj(s)?;
            Ok(Section {
                heading: as_str(get(so, "heading")?)?,
                warnings: as_strs(get(so, "warnings")?)?,
                blocks: as_arr(get(so, "blocks")?)?.iter().map(decode_block).collect::<Result<_>>()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SessionReport { title: as_str(get(o, "title")?)?, sections, warnings: as_strs(get(o, "warnings")?)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SessionReport {
        let mut t = Table::new(
            "Descriptive Statistics",
            vec![
                Column::new("Variable", ColumnKind::Text),
                Column::new("N", ColumnKind::Integer),
                Column::new("Mean", ColumnKind::Significant(4)),
                Column::new("P", ColumnKind::PValue),
            ],
        );
        t.push_row(vec![Cell::text("Weight"), Cell::Int(12), Cell::Num(72.5), Cell::Num(0.04321)]).unwrap();
        t.push_row(vec![Cell::text("Height"), Cell::Int(1200), Cell::Num(1.8234), Cell::Num(0.0001)]).unwrap();
        t.push_row(vec![Cell::text("Missing"), Cell::Int(0), Cell::Missing, Cell::Num(f64::NAN)]).unwrap();
        t.footnotes.push("P-values are two-sided.".into());
        let mut s = Section::new("Summary \"quoted\" é ∞\n2nd line");
        s.text("Hello\tworld").table(t).warn("Small sample");
        s.chart(ChartSpec {
            kind: ChartKind::Bar,
            title: "Bars".into(),
            x_label: "x".into(),
            y_label: "y".into(),
            categories: vec!["a".into(), "𝔹".into()],
            series: vec![Series { name: "s".into(), x: vec![0.1, f64::INFINITY], y: vec![-0.0, f64::NAN] }],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: 1e-300, label: "α".into() }],
        });
        let mut r = SessionReport::new("Session");
        r.push(s);
        r.warnings.push("global".into());
        r
    }

    #[test]
    fn decimal_alignment_and_pvalues() {
        let r = sample();
        let t = r.tables()[0];
        let cols = t.formatted_columns();
        // 4 significant digits: 1.8234 needs 3 decimals, so all Mean cells use 3.
        assert_eq!(cols[2], vec!["72.500", "1.823", "*"]);
        assert_eq!(cols[3], vec!["0.043", "0.000", "*"]);
        let text = t.render_text();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "Descriptive Statistics");
        assert_eq!(lines[2], "Variable     N    Mean      P");
        assert_eq!(lines[4], "Weight      12  72.500  0.043");
        assert_eq!(lines[5], "Height    1200   1.823  0.000");
        assert_eq!(lines[6], "Missing      0       *      *");
        assert!(text.contains("P-values are two-sided."));
        // Decimal points line up.
        assert_eq!(lines[4].find('.'), lines[5].rfind('.').map(|_| lines[4].find('.').unwrap()));
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let r = sample();
        let json = r.to_json();
        let back = SessionReport::from_json(&json).unwrap();
        // NaN != NaN, so compare through a second serialisation and spot checks.
        assert_eq!(back.to_json(), json);
        assert_eq!(back.sections[0].heading, r.sections[0].heading);
        assert_eq!(back.charts()[0].series[0].x[1], f64::INFINITY);
        assert!(back.charts()[0].series[0].y[1].is_nan());
        assert_eq!(back.charts()[0].series[0].y[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(back.charts()[0].reference_lines[0].value, 1e-300);
        assert_eq!(back.tables()[0].rows[1][2], Cell::Num(1.8234));
        assert_eq!(back.warnings, vec!["global".to_string()]);
    }

    #[test]
    fn json_parser_accepts_escapes_and_rejects_garbage() {
        let j = r#"{"title":"aé𝔹
","warnings":[],"sections":[]}"#;
        let r = SessionReport::from_json(j).unwrap();
        assert_eq!(
            r.title,
            "aé𝔹
"
        );
        assert!(SessionReport::from_json("{").is_err());
        assert!(SessionReport::from_json("{\"title\":1}").is_err());
        assert!(SessionReport::from_json(&format!("{} x", sample().to_json())).is_err());
        assert!(SessionReport::from_json(r#"{"title":"\ud835","warnings":[],"sections":[]}"#).is_err());
    }

    #[test]
    fn rows_must_be_rectangular() {
        let mut t = Table::new("", vec![Column::new("a", ColumnKind::Integer)]);
        assert!(t.push_row(vec![Cell::Int(1), Cell::Int(2)]).is_err());
        t.rows.push(vec![]);
        assert!(t.validate().is_err());
        let mut r = SessionReport::new("x");
        let mut s = Section::new("s");
        s.table(t);
        r.push(s);
        assert!(r.validate().is_err());
    }

    #[test]
    fn session_text_contains_warnings_and_chart_stub() {
        let txt = sample().render_text();
        assert!(txt.starts_with("Session\n=======\n"));
        assert!(txt.contains("* WARNING * Small sample"));
        assert!(txt.contains("[Chart: Bars (Bar), 1 series]"));
        assert!(txt.trim_end().ends_with("* WARNING * global"));
    }

    #[test]
    fn p_value_format() {
        assert_eq!(format_p_value(0.0004999), "0.000");
        assert_eq!(format_p_value(0.0005), "0.001");
        assert_eq!(format_p_value(1.0), "1.000");
    }
}
