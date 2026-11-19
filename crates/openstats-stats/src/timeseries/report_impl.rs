//! [`Reportable`] implementations and chart builders for the time-series results.

use super::acf::{AcfResult, PacfResult};
use super::arima::{ArimaFit, ArimaMethod};
use super::decompose::{Decomposition, DecompositionModel, TrendKind};
use super::smoothing::{Seasonality, SmoothingFit};
use super::unitroot::{AdfTest, AutoArima, CrossCorrelation, Deterministic, KpssTest};
use super::{Accuracy, Forecast};
use crate::report::{
    Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Reportable, Section, Series, Table,
};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn table(title: &str, cols: &[(&str, ColumnKind)], rows: Vec<Vec<Cell>>) -> Table {
    let mut t = Table::new(title, cols.iter().map(|(h, k)| Column::new(*h, *k)).collect());
    for r in rows {
        t.row(r);
    }
    t
}

fn text(s: &str) -> Cell {
    Cell::Text(s.to_string())
}

fn xs(n: usize) -> Vec<f64> {
    (1..=n).map(|i| i as f64).collect()
}

/// A series over periods `first, first + 1, …` without its missing values (a line chart cannot
/// draw them, and the renderer only takes finite coordinates).
fn finite_series(name: &str, first: usize, y: &[f64]) -> Series {
    let (mut x, mut v) = (Vec::new(), Vec::new());
    for (i, &a) in y.iter().enumerate() {
        if a.is_finite() {
            x.push((first + i) as f64);
            v.push(a);
        }
    }
    Series { name: name.into(), x, y: v }
}

fn accuracy_table(a: &Accuracy) -> Table {
    table(
        "Accuracy Measures",
        &[("MAPE", S), ("MAD", S), ("MSD", S)],
        vec![vec![Cell::Num(a.mape), Cell::Num(a.mad), Cell::Num(a.msd)]],
    )
}

/// The forecast table: period, forecast and the prediction limits.
pub fn forecast_table(f: &Forecast) -> Table {
    let rows = (0..f.point.len())
        .map(|i| {
            vec![
                Cell::Int((f.origin + i + 2) as i64),
                Cell::Num(f.point[i]),
                Cell::Num(f.lower[i]),
                Cell::Num(f.upper[i]),
            ]
        })
        .collect();
    table(
        &format!("Forecasts ({:.0}% prediction limits)", 100.0 * f.level),
        &[("Period", I), ("Forecast", S), ("Lower", S), ("Upper", S)],
        rows,
    )
}

/// Time series plot of the data, the fits and, if given, the forecasts with their limits.
pub fn series_chart(title: &str, y: &[f64], fitted: Option<&[f64]>, fc: Option<&Forecast>) -> ChartSpec {
    let mut series = vec![finite_series("Actual", 1, y)];
    if let Some(f) = fitted {
        series.push(finite_series("Fits", 1, f));
    }
    if let Some(f) = fc {
        let x: Vec<f64> = (0..f.point.len()).map(|i| (f.origin + i + 2) as f64).collect();
        series.push(Series { name: "Forecast".into(), x: x.clone(), y: f.point.clone() });
        series.push(Series { name: "Lower".into(), x: x.clone(), y: f.lower.clone() });
        series.push(Series { name: "Upper".into(), x, y: f.upper.clone() });
    }
    ChartSpec {
        kind: ChartKind::Line,
        title: title.into(),
        x_label: "Time".into(),
        y_label: "Value".into(),
        categories: vec![],
        series,
        reference_lines: vec![],
    }
}

fn corr_chart(title: &str, values: &[f64], lower: &[f64], upper: &[f64]) -> ChartSpec {
    let k = values.len();
    let mut series = vec![Series { name: "Correlation".into(), x: xs(k), y: values.to_vec() }];
    if lower.len() == k {
        series.push(Series { name: "Lower limit".into(), x: xs(k), y: lower.to_vec() });
        series.push(Series { name: "Upper limit".into(), x: xs(k), y: upper.to_vec() });
    }
    ChartSpec {
        kind: ChartKind::Bar,
        title: title.into(),
        x_label: "Lag".into(),
        y_label: "Correlation".into(),
        categories: (1..=k).map(|i| i.to_string()).collect(),
        series,
        reference_lines: vec![ReferenceLine { axis: Axis::Y, value: 0.0, label: "0".into() }],
    }
}

impl AcfResult {
    /// The ACF bar chart with confidence limits.
    pub fn chart(&self) -> ChartSpec {
        corr_chart("Autocorrelation Function", &self.acf, &self.lower, &self.upper)
    }
}

impl PacfResult {
    /// The PACF bar chart with confidence limits.
    pub fn chart(&self) -> ChartSpec {
        let k = self.pacf.len();
        corr_chart("Partial Autocorrelation Function", &self.pacf, &vec![self.lower; k], &vec![self.upper; k])
    }
}

impl Reportable for AcfResult {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Autocorrelation Function");
        let rows = (0..self.acf.len())
            .map(|i| {
                vec![Cell::Int(i as i64 + 1), Cell::Num(self.acf[i]), Cell::Num(self.t[i]), Cell::Num(self.lbq[i])]
            })
            .collect();
        s.table(table("Autocorrelations", &[("Lag", I), ("ACF", S), ("T", S), ("LBQ", S)], rows));
        s.text(format!("{:.0}% limits use Bartlett's standard errors (series length {}).", 100.0 * self.level, self.n));
        s.chart(self.chart());
        s
    }
}

impl Reportable for PacfResult {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Partial Autocorrelation Function");
        let rows = (0..self.pacf.len()).map(|i| vec![Cell::Int(i as i64 + 1), Cell::Num(self.pacf[i])]).collect();
        s.table(table("Partial Autocorrelations", &[("Lag", I), ("PACF", S)], rows));
        s.text(format!("{:.0}% limits are ±{:.4} (1/√n).", 100.0 * self.level, self.upper));
        s.chart(self.chart());
        s
    }
}

fn smoothing_rows(f: &SmoothingFit) -> Vec<Vec<Cell>> {
    let mut rows = vec![vec![text("Alpha (level)"), Cell::Num(f.alpha)]];
    if let Some(b) = f.beta {
        rows.push(vec![text("Gamma (trend)"), Cell::Num(b)]);
    }
    if let Some(g) = f.gamma {
        rows.push(vec![text("Delta (seasonal)"), Cell::Num(g)]);
    }
    rows
}

impl Reportable for SmoothingFit {
    fn report_section(&self) -> Section {
        let mut s = Section::new(self.spec.name());
        s.table(table("Smoothing Constants", &[("Constant", T), ("Value", S)], smoothing_rows(self)));
        s.table(accuracy_table(&self.accuracy));
        if self.spec.seasonal != Seasonality::None {
            let rows = (0..self.final_seasonal.len())
                .map(|i| {
                    vec![
                        Cell::Int(i as i64 + 1),
                        Cell::Num(self.initial.seasonal[i]),
                        Cell::Num(self.final_seasonal[i]),
                    ]
                })
                .collect();
            s.table(table("Seasonal Terms (initial and final)", &[("Season", I), ("Initial", S), ("Final", S)], rows));
        }
        s.text(format!("Sum of squared one-step errors = {:.6}.", self.sse));
        s.chart(series_chart(&format!("Smoothing Plot for {}", "series"), &self.y, Some(&self.fitted), None));
        s
    }
}

impl SmoothingFit {
    /// The report with a forecast table and plot appended.
    pub fn report_with_forecast(&self, fc: &Forecast) -> Section {
        let mut s = self.report_section();
        s.blocks.retain(|b| !matches!(b, crate::report::Block::Chart(_)));
        s.table(forecast_table(fc));
        s.chart(series_chart("Smoothing Plot with Forecasts", &self.y, Some(&self.fitted), Some(fc)));
        s
    }
}

impl Reportable for ArimaFit {
    fn report_section(&self) -> Section {
        let method = match self.method {
            ArimaMethod::Css => "conditional least squares",
            ArimaMethod::Ml => "exact maximum likelihood",
            ArimaMethod::CssMl => "conditional least squares then maximum likelihood",
        };
        let mut s = Section::new(format!("{} Model", self.spec.name()));
        s.text(format!(
            "Estimation by {method}. Series length {}; {} observations after differencing.",
            self.y.len(),
            self.n_used
        ));
        let rows = self
            .coefs
            .iter()
            .map(|c| vec![text(&c.label), Cell::Num(c.coef), Cell::opt(c.se), Cell::opt(c.t), Cell::opt(c.p)])
            .collect();
        s.table(table(
            "Final Estimates of Parameters",
            &[("Type", T), ("Coef", S), ("SE Coef", S), ("T", S), ("P", P)],
            rows,
        ));
        if self.coefs.is_empty() {
            s.text("The model has no parameters other than the variance.");
        }
        s.text(format!(
            "Number of observations: {}   Sum of squares: {:.6}   MS: {:.6}   Log likelihood: {:.4}",
            self.n_used, self.ssr, self.sigma2, self.loglik
        ));
        s.table(table(
            "Information Criteria",
            &[("AIC", S), ("AICc", S), ("BIC", S)],
            vec![vec![Cell::Num(self.aic), Cell::Num(self.aicc), Cell::Num(self.bic)]],
        ));
        if !self.ljung_box.is_empty() {
            let rows = self
                .ljung_box
                .iter()
                .map(|r| vec![Cell::Int(r.lag as i64), Cell::Num(r.q), Cell::Int(r.df as i64), Cell::opt(r.p)])
                .collect();
            s.table(table(
                "Modified Box-Pierce (Ljung-Box) Chi-Square Statistic",
                &[("Lag", I), ("Chi-Square", S), ("DF", I), ("P", P)],
                rows,
            ));
        }
        s.table(accuracy_table(&self.accuracy));
        if !self.converged {
            s.warn("The optimiser did not report convergence; treat the estimates with caution.");
        }
        let resid: Vec<f64> = self.residuals.iter().copied().filter(|v| !v.is_nan()).collect();
        if !self.residual_acf.is_empty() {
            let k = self.residual_acf.len();
            let band = 1.959_964 / (resid.len() as f64).sqrt();
            s.chart(corr_chart("ACF of Residuals", &self.residual_acf, &vec![-band; k], &vec![band; k]));
        }
        s.chart(series_chart("Time Series Plot with Fits", &self.y, Some(&self.fitted), None));
        s
    }
}

impl ArimaFit {
    /// The report with a forecast table and plot appended.
    pub fn report_with_forecast(&self, fc: &Forecast) -> Section {
        let mut s = self.report_section();
        s.blocks.retain(|b| !matches!(b, crate::report::Block::Chart(c) if c.title.starts_with("Time Series")));
        s.table(forecast_table(fc));
        s.chart(series_chart("Time Series Plot with Forecasts", &self.y, Some(&self.fitted), Some(fc)));
        s
    }
}

impl Reportable for Decomposition {
    fn report_section(&self) -> Section {
        let mult = self.opts.model == DecompositionModel::Multiplicative;
        let mut s =
            Section::new(format!("{} Time Series Decomposition", if mult { "Multiplicative" } else { "Additive" }));
        let trend = match self.opts.trend {
            TrendKind::Constant => "constant (overall mean)".to_string(),
            TrendKind::Linear => format!("linear: Yt = {:.6} + {:.6}·t", self.coef[0], self.coef[1]),
            TrendKind::Quadratic => {
                format!("quadratic: Yt = {:.6} + {:.6}·t + {:.6}·t²", self.coef[0], self.coef[1], self.coef[2])
            }
            TrendKind::Exponential => {
                format!("exponential growth: Yt = {:.6}·exp({:.6}·t)", self.coef[0].exp(), self.coef[1])
            }
            TrendKind::MovingAverage => format!("centred moving average of length {}", self.opts.period),
        };
        s.text(format!("Trend: {trend}."));
        if self.opts.period > 1 {
            let rows =
                (0..self.indices.len()).map(|i| vec![Cell::Int(i as i64 + 1), Cell::Num(self.indices[i])]).collect();
            s.table(table("Seasonal Indices", &[("Period", I), ("Index", S)], rows));
        }
        s.table(accuracy_table(&self.accuracy));
        s.chart(series_chart("Component Analysis: Original and Fits", &self.y, Some(&self.fitted), None));
        s.chart(self.component_chart("Trend", &self.trend));
        if self.opts.period > 1 {
            s.chart(ChartSpec {
                kind: ChartKind::Bar,
                title: "Seasonal Indices".into(),
                x_label: "Period".into(),
                y_label: "Index".into(),
                categories: (1..=self.indices.len()).map(|i| i.to_string()).collect(),
                series: vec![Series { name: "Index".into(), x: xs(self.indices.len()), y: self.indices.clone() }],
                reference_lines: vec![ReferenceLine {
                    axis: Axis::Y,
                    value: if mult { 1.0 } else { 0.0 },
                    label: "".into(),
                }],
            });
        }
        s.chart(self.component_chart("Residuals", &self.residuals));
        s
    }
}

impl Decomposition {
    fn component_chart(&self, title: &str, v: &[f64]) -> ChartSpec {
        ChartSpec {
            kind: ChartKind::Line,
            title: title.into(),
            x_label: "Time".into(),
            y_label: title.into(),
            categories: vec![],
            series: vec![finite_series(title, 1, v)],
            reference_lines: vec![],
        }
    }

    /// The report with a forecast table and plot appended.
    pub fn report_with_forecast(&self, fc: &Forecast) -> Section {
        let mut s = self.report_section();
        s.table(forecast_table(fc));
        s.chart(series_chart("Decomposition Forecasts", &self.y, Some(&self.fitted), Some(fc)));
        s
    }
}

impl Reportable for AdfTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Augmented Dickey-Fuller Test");
        let det = match self.deterministic {
            Deterministic::None => "no constant",
            Deterministic::Constant => "constant",
            Deterministic::Trend => "constant and trend",
        };
        s.table(table(
            "Test",
            &[("N", I), ("Lags", I), ("Observations", I), ("Gamma", S), ("SE", S), ("Test statistic", S)],
            vec![vec![
                Cell::Int(self.n as i64),
                Cell::Int(self.lags as i64),
                Cell::Int(self.nobs as i64),
                Cell::Num(self.gamma),
                Cell::Num(self.se),
                Cell::Num(self.statistic),
            ]],
        ));
        let rows = ["1%", "5%", "10%"]
            .iter()
            .enumerate()
            .map(|(i, l)| {
                vec![
                    text(l),
                    Cell::Num(self.critical[i]),
                    text(if self.rejects(i) { "Reject unit root" } else { "Cannot reject" }),
                ]
            })
            .collect();
        s.table(table("Critical Values", &[("Level", T), ("Critical value", S), ("Decision", T)], rows));
        s.text(format!(
            "Regression with {det}; H0: the series has a unit root (critical values: MacKinnon response surface)."
        ));
        s
    }
}

impl Reportable for KpssTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("KPSS Stationarity Test");
        s.table(table(
            "Test",
            &[("N", I), ("Lags", I), ("Test statistic", S)],
            vec![vec![Cell::Int(self.n as i64), Cell::Int(self.lags as i64), Cell::Num(self.statistic)]],
        ));
        let rows = ["10%", "5%", "2.5%", "1%"]
            .iter()
            .enumerate()
            .map(|(i, l)| {
                vec![
                    text(l),
                    Cell::Num(self.critical[i]),
                    text(if self.rejects(i) { "Reject stationarity" } else { "Cannot reject" }),
                ]
            })
            .collect();
        s.table(table("Critical Values", &[("Level", T), ("Critical value", S), ("Decision", T)], rows));
        s.text(format!("H0: the series is {}-stationary.", if self.trend { "trend" } else { "level" }));
        s
    }
}

impl Reportable for CrossCorrelation {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Cross Correlation Function");
        let rows = (0..self.lags.len()).map(|i| vec![Cell::Int(self.lags[i]), Cell::Num(self.ccf[i])]).collect();
        s.table(table("Cross Correlations", &[("Lag", I), ("CCF", S)], rows));
        s.text(format!(
            "Approximate limits are \u{b1}{:.4} (series length {}). Lag k pairs x(t) with y(t+k).",
            self.limit, self.n
        ));
        let k = self.ccf.len();
        s.chart(ChartSpec {
            kind: ChartKind::Bar,
            title: "Cross Correlation Function".into(),
            x_label: "Lag".into(),
            y_label: "Correlation".into(),
            categories: self.lags.iter().map(|l| l.to_string()).collect(),
            series: vec![
                Series { name: "Correlation".into(), x: xs(k), y: self.ccf.clone() },
                Series { name: "Lower limit".into(), x: xs(k), y: vec![-self.limit; k] },
                Series { name: "Upper limit".into(), x: xs(k), y: vec![self.limit; k] },
            ],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: 0.0, label: "0".into() }],
        });
        s
    }
}

impl Reportable for AutoArima {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Automatic ARIMA Order Search");
        let rows = self
            .ranked
            .iter()
            .map(|f| {
                vec![
                    text(&format!("ARIMA({}, {}, {})", f.spec.p, f.spec.d, f.spec.q)),
                    Cell::Num(f.aicc),
                    Cell::Num(f.aic),
                    Cell::Num(f.bic),
                    Cell::Num(f.sigma2),
                ]
            })
            .collect();
        s.table(table(
            "Candidate Models (best first)",
            &[("Model", T), ("AICc", S), ("AIC", S), ("BIC", S), ("Sigma squared", S)],
            rows,
        ));
        s.text(format!("Differencing order d = {}.", self.d));
        if let Some(best) = self.ranked.first() {
            s.blocks.extend(best.report_section().blocks);
        }
        s
    }
}
