//! Time series (Stat ▸ Time Series) and the multi-response optimiser (Stat ▸ DOE ▸ Response
//! Optimizer).
//!
//! A series is one numeric column in time order; trailing missing values are ignored, any other
//! missing value is an error (the methods need a complete series). Stored fits and residuals have
//! the length of the series; stored forecasts and their limits have one value per forecast period.

use super::analyses::add;
use super::analysis::{col_title, complete_rows, expand, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::doe::{
    design_from_columns, fit_responses, model_summary_section, optimize_responses, parse_goal, ModelOrder,
    OptimizeOptions, PointType, ResponseGoal,
};
use openstats_stats::report::{Block, Reportable, Section};
use openstats_stats::timeseries::{
    acf, adf_test, auto_arima, cross_correlation, decompose, fit_arima, fit_smoothing, kpss_test, pacf, AdfOptions,
    ArimaMethod, ArimaOptions, ArimaSpec, DecomposeOptions, DecompositionModel, Deterministic, Forecast, Seasonality,
    SmoothingOptions, SmoothingSpec, TrendKind,
};

/// The selected series without trailing missing values.
fn series(ws: &Worksheet, v: &Values) -> Res<(String, Vec<f64>)> {
    let c = v.var("series").ok_or("select a series")?;
    let mut y = numeric(ws, c)?;
    while y.last().is_some_and(|x| x.is_nan()) {
        y.pop();
    }
    if let Some(i) = y.iter().position(|x| x.is_nan()) {
        return Err(format!(
            "{} has a missing value in row {}; time series methods need a complete series",
            col_title(ws, c),
            i + 1
        )
        .into());
    }
    Ok((col_title(ws, c), y))
}

fn col(name: &str, v: Vec<f64>) -> Column {
    Column::numeric(name, v)
}

fn store_forecast(out: &mut Vec<Column>, fc: &Forecast) {
    out.push(col("FORE", fc.point.clone()));
    out.push(col("LOWER", fc.lower.clone()));
    out.push(col("UPPER", fc.upper.clone()));
}

fn keep_charts(s: &mut Section, keep: bool) {
    if !keep {
        s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    }
}

fn horizon(v: &Values) -> usize {
    v.count("forecasts")
}

// ----------------------------------------------------------------------------- ACF / PACF

fn lags_of(v: &Values) -> Option<usize> {
    let l = v.count("lags");
    (l > 0).then_some(l)
}

fn acf_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let r = acf(&y, lags_of(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Autocorrelation Function: {name}");
    keep_charts(&mut s, v.flag("plot"));
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(col("ACF", r.acf.clone()));
        stored.push(col("TSTAT", r.t.clone()));
        stored.push(col("LBQ", r.lbq.clone()));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn pacf_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let r = pacf(&y, lags_of(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Partial Autocorrelation Function: {name}");
    keep_charts(&mut s, v.flag("plot"));
    let stored = if v.flag("store") { vec![col("PACF", r.pacf.clone())] } else { Vec::new() };
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------- smoothing

fn weight(v: &Values, name: &str) -> Option<f64> {
    let x = v.num(name);
    x.is_finite().then_some(x)
}

fn smooth(ws: &Worksheet, v: &Values, kind: u8) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let spec = match kind {
        0 => SmoothingSpec::single(),
        1 => SmoothingSpec::double(),
        _ => {
            let seasonal = if v.choice("seasonal") == 0 { Seasonality::Multiplicative } else { Seasonality::Additive };
            SmoothingSpec { trend: v.flag("trend"), seasonal, period: v.count("period") }
        }
    };
    let o = SmoothingOptions { alpha: weight(v, "alpha"), beta: weight(v, "beta"), gamma: weight(v, "gamma") };
    let fit = fit_smoothing(&y, &spec, &o)?;
    let h = horizon(v);
    let fc = if h > 0 { Some(fit.forecast(h, level(v)?)?) } else { None };
    let mut s = match &fc {
        Some(f) => fit.report_with_forecast(f),
        None => fit.report_section(),
    };
    s.heading = format!("{} for {name}", spec.name());
    keep_charts(&mut s, v.flag("plot"));
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        stored.push(col("FITS", fit.fitted.clone()));
    }
    if v.flag("store_resid") {
        stored.push(col("RESI", fit.residuals.clone()));
    }
    if v.flag("store_forecast") {
        if let Some(f) = &fc {
            store_forecast(&mut stored, f);
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn ses_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    smooth(ws, v, 0)
}

fn des_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    smooth(ws, v, 1)
}

fn winters_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    smooth(ws, v, 2)
}

// ----------------------------------------------------------------------------- ARIMA

fn arima_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let spec = ArimaSpec::new(v.count("p"), v.count("d"), v.count("q"))
        .seasonal(v.count("sp"), v.count("sd"), v.count("sq"), v.count("period"))
        .with_constant(v.flag("constant"));
    let lags = v.count("lags");
    let o = ArimaOptions {
        method: match v.choice("method") {
            0 => ArimaMethod::CssMl,
            1 => ArimaMethod::Ml,
            _ => ArimaMethod::Css,
        },
        lb_lags: (lags > 0).then_some(lags),
        ..Default::default()
    };
    let fit = fit_arima(&y, &spec, &o)?;
    let h = horizon(v);
    let fc = if h > 0 { Some(fit.forecast(h, level(v)?)?) } else { None };
    let mut s = match &fc {
        Some(f) => fit.report_with_forecast(f),
        None => fit.report_section(),
    };
    s.heading = format!("{} for {name}", spec.name());
    let (acf_plot, series_plot) = (v.flag("acf_plot"), v.flag("plot"));
    s.blocks.retain(|b| match b {
        Block::Chart(c) => {
            (acf_plot || !c.title.starts_with("ACF")) && (series_plot || !c.title.starts_with("Time Series"))
        }
        _ => true,
    });
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        stored.push(col("FITS", fit.fitted.clone()));
    }
    if v.flag("store_resid") {
        stored.push(col("RESI", fit.residuals.clone()));
    }
    if v.flag("store_forecast") {
        if let Some(f) = &fc {
            store_forecast(&mut stored, f);
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------- decomposition

fn decomp_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let o = DecomposeOptions {
        model: if v.choice("model") == 0 { DecompositionModel::Multiplicative } else { DecompositionModel::Additive },
        trend: match v.choice("trend") {
            0 => TrendKind::Linear,
            1 => TrendKind::Quadratic,
            2 => TrendKind::Exponential,
            3 => TrendKind::Constant,
            _ => TrendKind::MovingAverage,
        },
        period: v.count("period"),
        first_season: v.count("first").max(1),
    };
    let d = decompose(&y, &o)?;
    let h = horizon(v);
    let fc = if h > 0 { Some(d.forecast(h, level(v)?)?) } else { None };
    let mut s = match &fc {
        Some(f) => d.report_with_forecast(f),
        None => d.report_section(),
    };
    s.heading = format!("{} for {name}", s.heading);
    keep_charts(&mut s, v.flag("plot"));
    let mut stored = Vec::new();
    for (flag, label, vals) in [
        ("store_trend", "TREND", &d.trend),
        ("store_detrend", "DETR", &d.detrended),
        ("store_seasonal", "SEAS", &d.seasonal),
        ("store_deseason", "DESEAS", &d.deseasonalized),
        ("store_fits", "FITS", &d.fitted),
        ("store_resid", "RESI", &d.residuals),
    ] {
        if v.flag(flag) {
            stored.push(col(label, vals.clone()));
        }
    }
    if v.flag("store_forecast") {
        if let Some(f) = &fc {
            store_forecast(&mut stored, f);
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------- response optimizer

/// `"a b; c d"` → `[[a, b], [c, d]]`, with `*` or an empty entry as `None`.
pub(super) fn parse_groups(text: &str, what: &str) -> Res<Vec<Vec<Option<f64>>>> {
    let mut out = Vec::new();
    for part in text.split(';') {
        let items: Vec<Option<f64>> = part
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| {
                if t == "*" {
                    Ok(None)
                } else {
                    t.parse::<f64>().map(Some).map_err(|_| format!("'{t}' is not a number in the {what}"))
                }
            })
            .collect::<Result<_, String>>()?;
        out.push(items);
    }
    Ok(out)
}

fn optimizer_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let fcols = v.vars("factors").to_vec();
    let rcols = v.vars("responses").to_vec();
    if fcols.is_empty() || rcols.is_empty() {
        return Err("select the factor columns and at least one response".into());
    }
    if fcols.iter().any(|c| rcols.contains(c)) {
        return Err("a column cannot be both a factor and a response".into());
    }
    let all: Vec<Vec<f64>> = fcols.iter().chain(&rcols).map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let refs: Vec<&[f64]> = all.iter().map(Vec::as_slice).collect();
    let (rows, data) = complete_rows(&refs);
    let n_total = all.iter().map(Vec::len).max().unwrap_or(0);
    let dropped = n_total - rows.len();
    let names: Vec<String> = fcols.iter().map(|&c| col_title(ws, c)).collect();
    let design = design_from_columns(&names, &data[..fcols.len()])?;
    let responses: Vec<(String, Vec<f64>)> =
        rcols.iter().enumerate().map(|(i, &c)| (col_title(ws, c), data[fcols.len() + i].clone())).collect();
    let order = match v.choice("model") {
        1 => ModelOrder::Linear,
        2 => ModelOrder::Interactions,
        3 => ModelOrder::Quadratic,
        _ => {
            if design.point_type.contains(&PointType::Axial) {
                ModelOrder::Quadratic
            } else {
                ModelOrder::Interactions
            }
        }
    };
    let fits = fit_responses(&design, &responses, order)?;
    // Goals and importances.
    let goal_text = v.text("goals").trim();
    if goal_text.is_empty() {
        return Err("enter a goal for each response, e.g. \"max 70 90; min 5 20\"".into());
    }
    let goal_parts: Vec<&str> = goal_text.split(';').map(str::trim).filter(|s| !s.is_empty()).collect();
    if goal_parts.len() != rcols.len() {
        return Err(format!(
            "{} goals were entered for {} responses; separate the goals with ';'",
            goal_parts.len(),
            rcols.len()
        )
        .into());
    }
    let imp_text = v.text("importance").trim();
    let importance: Vec<f64> = if imp_text.is_empty() {
        vec![1.0; rcols.len()]
    } else {
        let g = parse_groups(imp_text, "importance list")?;
        let flat: Vec<f64> = g.into_iter().flatten().map(|x| x.unwrap_or(1.0)).collect();
        if flat.len() != rcols.len() {
            return Err(format!("{} importance values were entered for {} responses", flat.len(), rcols.len()).into());
        }
        flat
    };
    if importance.iter().any(|&w| !(w > 0.0 && w.is_finite())) {
        return Err("importance values must be positive".into());
    }
    let mut goals = Vec::new();
    for (i, f) in fits.iter().enumerate() {
        let goal = parse_goal(goal_parts[i]).map_err(|e| format!("{}: {e}", f.name))?;
        goals.push(ResponseGoal { name: f.name.clone(), model: f.model.clone(), goal, importance: importance[i] });
    }
    // Optional restricted ranges and fixed settings, in natural units.
    let k = fcols.len();
    let mut opts = OptimizeOptions {
        starts: v.count("starts").max(1),
        top: v.count("top").max(1),
        seed: v.count("seed") as u64,
        ..Default::default()
    };
    let range_text = v.text("ranges").trim();
    if !range_text.is_empty() {
        let mut g = parse_groups(range_text, "ranges")?;
        for p in &mut g {
            if p.len() == 1 && p[0].is_none() {
                p.push(None); // a lone "*" is the full range
            }
        }
        if g.len() != k || g.iter().any(|p| p.len() != 2) {
            return Err(format!(
                "ranges need one \"low high\" pair per factor ({k}), separated by ';' (use * for the full range)"
            )
            .into());
        }
        let mut r = Vec::new();
        for (j, p) in g.iter().enumerate() {
            let f = &design.factors[j];
            let lo = p[0].map_or(-1.0, |x| f.to_coded(x));
            let hi = p[1].map_or(1.0, |x| f.to_coded(x));
            if lo < -1.0 - 1e-9 || hi > 1.0 + 1e-9 || lo >= hi {
                return Err(format!(
                    "the range for {} must lie within {} to {} with low < high",
                    f.name, f.low, f.high
                )
                .into());
            }
            r.push((lo, hi));
        }
        opts.ranges = Some(r);
    }
    let fixed_text = v.text("fixed").trim();
    if !fixed_text.is_empty() {
        let g = parse_groups(fixed_text, "fixed settings")?;
        let flat: Vec<Option<f64>> = g.into_iter().flatten().collect();
        if flat.len() != k {
            return Err(format!("fixed settings need one value (or *) per factor ({k})").into());
        }
        opts.fixed = flat;
    }
    let result = optimize_responses(&goals, &design.factors, &opts)?;
    let mut s = result.report_section();
    s.heading = format!(
        "Response Optimization: {} versus {}",
        rcols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", "),
        names.join(", ")
    );
    let model_section = model_summary_section(&fits);
    // Model summary first, then the optimisation.
    let mut blocks = model_section.blocks;
    blocks.extend(s.blocks);
    s.blocks = blocks;
    s.warnings.extend(model_section.warnings);
    if dropped > 0 {
        s.warn(format!("{dropped} rows with missing values were left out."));
    }
    keep_charts(&mut s, v.flag("plot"));
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        for (g, f) in goals.iter().zip(&fits) {
            let vals: Vec<f64> = design.coded.iter().map(|x| f.model.predict(x)).collect();
            stored.push(expand(&format!("FITS_{}", g.name), n_total, &rows, &vals));
        }
    }
    if v.flag("store_d") {
        let vals: Vec<f64> = design
            .coded
            .iter()
            .map(|x| openstats_stats::doe::composite_desirability(&goals, &design.factors, x).composite)
            .collect();
        stored.push(expand("COMPD", n_total, &rows, &vals));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------- registration

fn base(spec: DialogSpec) -> DialogSpec {
    spec.vars("series", "Series", VarKind::Numeric, 1, 1)
}

fn forecast_fields(spec: DialogSpec, default_h: f64) -> DialogSpec {
    use Page::Options;
    spec.integer(Options, "forecasts", "Number of forecasts (0 = none)", default_h, 0.0, 1000.0).number(
        Options,
        "level",
        "Prediction interval level (%)",
        95.0,
        0.001,
        99.999,
    )
}

fn storage_fields(spec: DialogSpec) -> DialogSpec {
    use Page::Storage;
    spec.check(Storage, "store_fits", "Fits (one-step predictions)", false)
        .check(Storage, "store_resid", "Residuals", false)
        .check(Storage, "store_forecast", "Forecasts and prediction limits", false)
}

// -------------------------------------------------------------------- unit roots and CCF

fn two_series(ws: &Worksheet, v: &Values) -> Res<(String, Vec<f64>, Vec<f64>)> {
    let (a, b) =
        (v.var("series").ok_or("select the first series")?, v.var("series2").ok_or("select the second series")?);
    let (mut x, mut y) = (numeric(ws, a)?, numeric(ws, b)?);
    for z in [&mut x, &mut y] {
        while z.last().is_some_and(|v| v.is_nan()) {
            z.pop();
        }
    }
    if x.len() != y.len() {
        return Err("the two series have different lengths".into());
    }
    if let Some(i) = x.iter().zip(&y).position(|(p, q)| p.is_nan() || q.is_nan()) {
        return Err(format!("row {} has a missing value; the cross-correlation needs complete series", i + 1).into());
    }
    Ok((format!("{} and {}", col_title(ws, a), col_title(ws, b)), x, y))
}

fn adf_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let det = match v.choice("det") {
        0 => Deterministic::None,
        2 => Deterministic::Trend,
        _ => Deterministic::Constant,
    };
    let l = v.num("lags");
    let lags = (l.is_finite() && l >= 0.0).then_some(l as usize);
    let mut s = adf_test(&y, &AdfOptions { deterministic: det, lags })?.report_section();
    s.heading = format!("Augmented Dickey-Fuller Test: {name}");
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

fn kpss_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let mut s = kpss_test(&y, v.choice("null") == 1, lags_of(v))?.report_section();
    s.heading = format!("KPSS Test: {name}");
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

fn ccf_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, x, y) = two_series(ws, v)?;
    let lags = if v.count("lags") > 0 {
        v.count("lags")
    } else {
        ((x.len() as f64).sqrt() as usize).clamp(1, 24).min(x.len() - 1)
    };
    let r = cross_correlation(&x, &y, lags, level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Cross Correlation: {name}");
    keep_charts(&mut s, v.flag("plot"));
    let stored = if v.flag("store") { vec![col("CCF", r.ccf.clone())] } else { Vec::new() };
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn auto_arima_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, y) = series(ws, v)?;
    let d = if v.choice("d") >= 3 { None } else { Some(v.choice("d")) };
    let r = auto_arima(&y, v.count("max_p"), v.count("max_q"), d, v.flag("constant"), &ArimaOptions::default())?;
    let mut s = r.report_section();
    s.heading = format!("Automatic ARIMA for {name}");
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

/// Register the time-series and response-optimizer dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options, Storage};
    let menu = |last: &'static str| ["Stat", "Time Series", last];

    add(
        r,
        base(DialogSpec::new("stat.ts.adf", "Augmented Dickey-Fuller Test", "Test a series for a unit root."))
            .choice(Options, "det", "Regression includes", &["No constant", "Constant", "Constant and trend"], 1)
            .optional_number(Options, "lags", "Lagged differences (blank = choose by AIC)"),
        &menu("Augmented Dickey-Fuller..."),
        Runner::Report(adf_run),
    );
    add(
        r,
        base(DialogSpec::new("stat.ts.kpss", "KPSS Test", "Test a series for level or trend stationarity."))
            .choice(Options, "null", "Stationary around", &["A level", "A linear trend"], 0)
            .integer(Options, "lags", "Truncation lag (0 = automatic)", 0.0, 0.0, 1000.0),
        &menu("KPSS Stationarity Test..."),
        Runner::Report(kpss_run),
    );
    add(
        r,
        DialogSpec::new("stat.ts.ccf", "Cross Correlation", "Cross-correlation function of two series with limits.")
            .vars("series", "First series", VarKind::Numeric, 1, 1)
            .vars("series2", "Second series", VarKind::Numeric, 1, 1)
            .integer(Options, "lags", "Number of lags (0 = automatic)", 0.0, 0.0, 1000.0)
            .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
            .check(Graphs, "plot", "Cross-correlation chart", true)
            .check(Storage, "store", "Store the cross-correlations", false),
        &menu("Cross Correlation..."),
        Runner::Report(ccf_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ts.autoarima",
            "Automatic ARIMA",
            "Search ARIMA(p,d,q) orders and rank the models by AICc.",
        ))
        .integer(Options, "max_p", "Maximum AR order", 2.0, 0.0, 5.0)
        .integer(Options, "max_q", "Maximum MA order", 2.0, 0.0, 5.0)
        .choice(Options, "d", "Differences", &["0", "1", "2", "Choose with KPSS"], 3)
        .check(Options, "constant", "Include a constant", true),
        &menu("Automatic ARIMA..."),
        Runner::Report(auto_arima_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ts.acf",
            "Autocorrelation",
            "Sample autocorrelations with Bartlett standard errors, t statistics and Ljung-Box Q.",
        ))
        .integer(Options, "lags", "Number of lags (0 = automatic)", 0.0, 0.0, 1000.0)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Graphs, "plot", "Autocorrelation chart", true)
        .check(Storage, "store", "Store ACF, t and LBQ", false),
        &menu("Autocorrelation..."),
        Runner::Report(acf_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ts.pacf",
            "Partial Autocorrelation",
            "Sample partial autocorrelations by the Durbin-Levinson recursion.",
        ))
        .integer(Options, "lags", "Number of lags (0 = automatic)", 0.0, 0.0, 1000.0)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Graphs, "plot", "Partial autocorrelation chart", true)
        .check(Storage, "store", "Store PACF", false),
        &menu("Partial Autocorrelation..."),
        Runner::Report(pacf_run),
    );
    let smoothing_common = |spec: DialogSpec| {
        storage_fields(forecast_fields(spec, 6.0)).check(Graphs, "plot", "Plot of data, fits and forecasts", true)
    };
    add(
        r,
        smoothing_common(
            base(DialogSpec::new(
                "stat.ts.ses",
                "Single Exponential Smoothing",
                "Smooth a series with one weight on the latest observation. Leave the weight blank to choose it by least squares.",
            ))
            .optional_number(Options, "alpha", "Level weight α between 0 and 1 (blank = optimal)"),
        ),
        &menu("Single Exp Smoothing..."),
        Runner::Report(ses_run),
    );
    add(
        r,
        smoothing_common(
            base(DialogSpec::new(
                "stat.ts.des",
                "Double Exponential Smoothing",
                "Holt's linear method: a smoothed level and trend. Blank weights are chosen by least squares.",
            ))
            .optional_number(Options, "alpha", "Level weight α (blank = optimal)")
            .optional_number(Options, "beta", "Trend weight β (blank = optimal)"),
        ),
        &menu("Double Exp Smoothing..."),
        Runner::Report(des_run),
    );
    add(
        r,
        smoothing_common(
            base(DialogSpec::new(
                "stat.ts.winters",
                "Winters' Method",
                "Holt-Winters seasonal smoothing with additive or multiplicative seasonal terms. Blank weights are chosen by least squares.",
            ))
            .integer(Main, "period", "Seasonal length", 12.0, 2.0, 366.0)
            .choice(Main, "seasonal", "Seasonal model", &["Multiplicative", "Additive"], 0)
            .check(Options, "trend", "Include a trend component", true)
            .optional_number(Options, "alpha", "Level weight α (blank = optimal)")
            .optional_number(Options, "beta", "Trend weight β (blank = optimal)")
            .optional_number(Options, "gamma", "Seasonal weight γ (blank = optimal)"),
        ),
        &menu("Winters' Method..."),
        Runner::Report(winters_run),
    );
    add(
        r,
        storage_fields(forecast_fields(
            base(DialogSpec::new(
                "stat.ts.arima",
                "ARIMA",
                "Seasonal ARIMA(p,d,q)(P,D,Q) by conditional least squares, exact maximum likelihood or both, with residual diagnostics and forecasts.",
            ))
            .integer(Main, "p", "Autoregressive order p", 1.0, 0.0, 10.0)
            .integer(Main, "d", "Differences d", 0.0, 0.0, 3.0)
            .integer(Main, "q", "Moving average order q", 0.0, 0.0, 10.0)
            .integer(Main, "sp", "Seasonal autoregressive order P", 0.0, 0.0, 4.0)
            .integer(Main, "sd", "Seasonal differences D", 0.0, 0.0, 2.0)
            .integer(Main, "sq", "Seasonal moving average order Q", 0.0, 0.0, 4.0)
            .integer(Main, "period", "Seasonal period (used when P, D or Q > 0)", 12.0, 2.0, 366.0)
            .check(Options, "constant", "Include constant term", true)
            .choice(
                Options,
                "method",
                "Estimation",
                &["Conditional least squares, then maximum likelihood", "Maximum likelihood", "Conditional least squares"],
                0,
            )
            .integer(Options, "lags", "Residual lags for the Ljung-Box test (0 = automatic)", 0.0, 0.0, 200.0),
            6.0,
        ))
        .check(Graphs, "plot", "Time series plot with fits and forecasts", true)
        .check(Graphs, "acf_plot", "ACF of residuals", true),
        &menu("ARIMA..."),
        Runner::Report(arima_run),
    );
    add(
        r,
        storage_fields(forecast_fields(
            base(DialogSpec::new(
                "stat.ts.decomp",
                "Time Series Decomposition",
                "Split a series into trend, seasonal and irregular components (additive or multiplicative) and forecast from them.",
            ))
            .integer(Main, "period", "Seasonal length (1 = none)", 12.0, 1.0, 366.0)
            .choice(Main, "model", "Model type", &["Multiplicative", "Additive"], 0)
            .choice(
                Main,
                "trend",
                "Trend component",
                &["Linear", "Quadratic", "Exponential growth", "Constant (mean)", "Centred moving average"],
                0,
            )
            .integer(Options, "first", "Season of the first observation", 1.0, 1.0, 366.0),
            0.0,
        ))
        .check(Graphs, "plot", "Component charts", true)
        .check(Storage, "store_trend", "Trend", false)
        .check(Storage, "store_detrend", "Detrended data", false)
        .check(Storage, "store_seasonal", "Seasonal component", false)
        .check(Storage, "store_deseason", "Deseasonalized data", false),
        &menu("Decomposition..."),
        Runner::Report(decomp_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.optimizer",
            "Response Optimizer",
            "Fit a model to each response of a designed experiment and find the factor settings with the best composite desirability (Derringer-Suich). Goals: \"max LOWER TARGET [SHAPE]\", \"min TARGET UPPER [SHAPE]\", \"target LOWER TARGET UPPER [SHAPE_LOW [SHAPE_HIGH]]\", one per response separated by ';'.",
        )
        .vars("factors", "Factors (design columns in natural units)", VarKind::Numeric, 1, 15)
        .vars("responses", "Responses", VarKind::Numeric, 1, 10)
        .text(Main, "goals", "Goals, one per response (e.g. max 70 90; min 5 20)", "")
        .text(Main, "importance", "Importance of each response (blank = all 1)", "")
        .choice(
            Options,
            "model",
            "Model for every response",
            &["Automatic", "Linear", "Linear and interactions", "Full quadratic"],
            0,
        )
        .text(Options, "ranges", "Restrict factor ranges: \"low high\" per factor, ';' separated (* = full range)", "")
        .text(Options, "fixed", "Hold factors at: one value per factor (* = free)", "")
        .integer(Options, "starts", "Random starting points", 40.0, 1.0, 1000.0)
        .integer(Options, "top", "Number of solutions to report", 3.0, 1.0, 20.0)
        .integer(Options, "seed", "Random seed", 1.0, 0.0, 4.0e9)
        .check(Graphs, "plot", "Optimization plot", true)
        .check(Storage, "store_fits", "Fitted values of each response", false)
        .check(Storage, "store_d", "Composite desirability of each run", false),
        &["Stat", "DOE", "Response Optimizer..."],
        Runner::Report(optimizer_run),
    );
}
