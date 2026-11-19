//! Carried Data/Calc/time-series dialogs: Replace Missing Values, Unpivot Columns, Lag, Differences,
//! Moving Average, Trend Analysis, matrix operations and Project Properties.

use super::analyses::add;
use super::analysis::{col_title, labels, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use crate::grid::GridController;
use openstats_core::column::{Column, ColumnData};
use openstats_core::{StoredMatrix, Worksheet};
use openstats_math::linalg::{Cholesky, Lu, Matrix, SymmetricEigen};
use openstats_math::stats::{ols, R2Kind};
use openstats_stats::report::{
    Cell, ChartKind, ChartSpec, Column as TCol, ColumnKind as TKind, Section, Series, Table,
};
use openstats_stats::timeseries::accuracy;

fn stored(section: Section, cols: Vec<Column>) -> Res<Output> {
    Ok(Output { section, stored: cols, models: Vec::new() })
}

fn note(title: &str, text: String, cols: Vec<Column>) -> Res<Output> {
    let mut s = Section::new(title);
    s.text(text);
    stored(s, cols)
}

// ------------------------------------------------------------------------- replace missing

fn replace_missing_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let how = v.choice("how");
    let fill = v.num("value");
    if how == 0 && !fill.is_finite() {
        return Err("enter the value that replaces the missing values".into());
    }
    let mut out = Vec::new();
    let mut total = 0usize;
    for &c in v.vars("vars") {
        let x = numeric(ws, c)?;
        let present: Vec<f64> = x.iter().copied().filter(|a| !a.is_nan()).collect();
        if present.is_empty() {
            return Err(DialogError(format!("{} has no values to replace missing values from", col_title(ws, c))));
        }
        let mut y = x.clone();
        match how {
            0 => y.iter_mut().filter(|a| a.is_nan()).for_each(|a| *a = fill),
            1 => {
                let m = present.iter().sum::<f64>() / present.len() as f64;
                y.iter_mut().filter(|a| a.is_nan()).for_each(|a| *a = m);
            }
            2 => {
                let mut s = present.clone();
                s.sort_by(|a, b| a.total_cmp(b));
                let n = s.len();
                let m = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
                y.iter_mut().filter(|a| a.is_nan()).for_each(|a| *a = m);
            }
            3 => {
                for i in 1..y.len() {
                    if y[i].is_nan() {
                        y[i] = y[i - 1];
                    }
                }
            }
            4 => {
                for i in (0..y.len().saturating_sub(1)).rev() {
                    if y[i].is_nan() {
                        y[i] = y[i + 1];
                    }
                }
            }
            _ => {
                let known: Vec<usize> = (0..x.len()).filter(|&i| !x[i].is_nan()).collect();
                for i in 0..x.len() {
                    if !x[i].is_nan() {
                        continue;
                    }
                    let lo = known.iter().rev().find(|&&k| k < i);
                    let hi = known.iter().find(|&&k| k > i);
                    if let (Some(&a), Some(&b)) = (lo, hi) {
                        y[i] = x[a] + (x[b] - x[a]) * (i - a) as f64 / (b - a) as f64;
                    }
                }
            }
        }
        total += y.iter().zip(&x).filter(|(a, b)| !a.is_nan() && b.is_nan()).count();
        out.push(Column::numeric(format!("{}_filled", col_title(ws, c)), y));
    }
    note("Replace Missing Values", format!("Replaced {total} missing values in {} columns.", out.len()), out)
}

// ------------------------------------------------------------------------- unpivot

fn unpivot_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let vars = v.vars("vars");
    let ids = v.vars("ids");
    if vars.iter().any(|c| ids.contains(c)) {
        return Err("a column cannot be both unpivoted and kept as an identifier".into());
    }
    let n = vars.iter().chain(ids).map(|&c| ws.columns.get(c).map_or(0, Column::len)).max().unwrap_or(0);
    let k = vars.len();
    let rows = n * k;
    // Row index into the source for each stacked row: variable-major.
    let src: Vec<usize> = (0..k).flat_map(|_| 0..n).collect();
    let mut cols = Vec::new();
    for &c in ids {
        let col = ws.column(c)?;
        let mut data = col.data.clone();
        if data.len() < n {
            let mut padded = ColumnData::missing(col.kind(), 0);
            padded.extend(&data)?;
            for _ in data.len()..n {
                padded.push(&openstats_core::Value::Missing)?;
            }
            data = padded;
        }
        cols.push(Column::new(col_title(ws, c), data.take(&src)));
    }
    let vname = v.text("var_name").trim();
    let vname = if vname.is_empty() { "Variable" } else { vname };
    let nname = v.text("val_name").trim();
    let nname = if nname.is_empty() { "Value" } else { nname };
    let names: Vec<String> = vars.iter().flat_map(|&c| std::iter::repeat_n(col_title(ws, c), n)).collect();
    cols.push(Column::text(vname, names));
    let all_numeric = vars.iter().all(|&c| ws.columns.get(c).is_some_and(|c| !matches!(c.data, ColumnData::Text(_))));
    if all_numeric {
        let mut y = Vec::with_capacity(rows);
        for &c in vars {
            let mut x = numeric(ws, c)?;
            x.resize(n, f64::NAN);
            y.extend(x);
        }
        cols.push(Column::numeric(nname, y));
    } else {
        let mut y = Vec::with_capacity(rows);
        for &c in vars {
            let mut x = labels(ws, c)?;
            x.resize(n, String::new());
            y.extend(x);
        }
        cols.push(Column::text(nname, y));
    }
    note("Unpivot Columns", format!("Unpivoted {k} columns into {rows} rows."), cols)
}

// ------------------------------------------------------------------------- lag, difference, MA

fn lag_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let lag = v.count("lag").max(1);
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        let x = numeric(ws, c)?;
        let y: Vec<f64> = (0..x.len()).map(|i| if i >= lag { x[i - lag] } else { f64::NAN }).collect();
        out.push(Column::numeric(format!("{}_lag{lag}", col_title(ws, c)), y));
    }
    note("Lag", format!("Lagged {} columns by {lag}.", out.len()), out)
}

fn difference_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let lag = v.count("lag").max(1);
    let times = v.count("times").max(1);
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        let mut x = numeric(ws, c)?;
        for _ in 0..times {
            let prev = x.clone();
            x = (0..prev.len()).map(|i| if i >= lag { prev[i] - prev[i - lag] } else { f64::NAN }).collect();
        }
        out.push(Column::numeric(format!("{}_diff{lag}", col_title(ws, c)), x));
    }
    note("Differences", format!("Differenced {} columns (lag {lag}, {times} time(s)).", out.len()), out)
}

fn moving_average_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let len = v.count("length");
    if len < 1 {
        return Err("the length of the moving average must be at least 1".into());
    }
    let centered = v.flag("centered");
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        let x = numeric(ws, c)?;
        let n = x.len();
        let mut y = vec![f64::NAN; n];
        for (i, slot) in y.iter_mut().enumerate() {
            // Trailing window ends at i; the centred window is shifted back by the half-length.
            let end = if centered { i + len / 2 } else { i };
            if end >= n || end + 1 < len {
                continue;
            }
            let w = &x[end + 1 - len..=end];
            if w.iter().all(|a| !a.is_nan()) {
                *slot = if centered && len % 2 == 0 {
                    // Even length: a 2 × m average keeps the window centred on the point.
                    let lo = end + 1 - len;
                    if lo == 0 {
                        continue;
                    }
                    let w2 = &x[lo - 1..=end - 1];
                    if w2.iter().any(|a| a.is_nan()) {
                        continue;
                    }
                    (w.iter().sum::<f64>() + w2.iter().sum::<f64>()) / (2 * len) as f64
                } else {
                    w.iter().sum::<f64>() / len as f64
                };
            }
        }
        out.push(Column::numeric(format!("{}_MA{len}", col_title(ws, c)), y));
    }
    note("Moving Average", format!("Computed {len}-point moving averages for {} columns.", out.len()), out)
}

// ------------------------------------------------------------------------- trend analysis

fn trend_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("y").ok_or("choose the series")?;
    let y = numeric(ws, c)?;
    if y.iter().any(|a| !a.is_finite()) {
        return Err("the series has missing values; trend analysis needs a complete series".into());
    }
    let n = y.len();
    let model = v.choice("model");
    let t: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let design = |tt: &[f64]| -> Matrix {
        Matrix::from_fn(tt.len(), if model == 1 { 3 } else { 2 }, |i, j| match j {
            0 => 1.0,
            1 => tt[i],
            _ => tt[i] * tt[i],
        })
    };
    let work: Vec<f64> = if model == 2 {
        if y.iter().any(|&a| a <= 0.0) {
            return Err("the exponential growth model needs every value to be positive".into());
        }
        y.iter().map(|a| a.ln()).collect()
    } else {
        y.clone()
    };
    let fit = ols(&design(&t), &work, R2Kind::Centered).map_err(|e| DialogError(e.to_string()))?;
    let back = |a: f64| if model == 2 { a.exp() } else { a };
    let fitted: Vec<f64> = fit.fitted.iter().map(|&a| back(a)).collect();
    let b = &fit.coef;
    let eq = match model {
        0 => format!("Yt = {:.6} + {:.6} × t", b[0], b[1]),
        1 => format!("Yt = {:.6} + {:.6} × t + {:.6} × t²", b[0], b[1], b[2]),
        _ => format!("Yt = {:.6} × (e^({:.6} × t))", b[0].exp(), b[1]),
    };
    let h = v.count("forecasts");
    let tf: Vec<f64> = (n + 1..=n + h).map(|i| i as f64).collect();
    let forecasts: Vec<f64> = if h > 0 {
        let xf = design(&tf);
        (0..h).map(|i| back((0..xf.cols()).map(|j| xf[(i, j)] * b[j]).sum())).collect()
    } else {
        Vec::new()
    };
    let acc = accuracy(&y, &fitted);
    let name = col_title(ws, c);
    let mut s = Section::new(format!("Trend Analysis for {name}"));
    s.text(format!("Fitted trend equation: {eq}"));
    let mut t1 = Table::new(
        "Accuracy Measures",
        vec![TCol::new("Measure", TKind::Text), TCol::new("Value", TKind::Significant(6))],
    );
    t1.push_row(vec![Cell::text("MAPE"), Cell::Num(acc.mape)])?;
    t1.push_row(vec![Cell::text("MAD"), Cell::Num(acc.mad)])?;
    t1.push_row(vec![Cell::text("MSD"), Cell::Num(acc.msd)])?;
    s.table(t1);
    if h > 0 {
        let mut t2 = Table::new(
            "Forecasts",
            vec![TCol::new("Period", TKind::Integer), TCol::new("Forecast", TKind::Significant(6))],
        );
        for (i, f) in forecasts.iter().enumerate() {
            t2.push_row(vec![Cell::Int((n + 1 + i) as i64), Cell::Num(*f)])?;
        }
        s.table(t2);
    }
    let mut series = vec![
        Series { name: name.clone(), x: t.clone(), y: y.clone() },
        Series { name: "Fits".into(), x: t.clone(), y: fitted.clone() },
    ];
    if h > 0 {
        series.push(Series { name: "Forecasts".into(), x: tf, y: forecasts.clone() });
    }
    s.chart(ChartSpec {
        kind: ChartKind::Line,
        title: format!("Trend Analysis Plot for {name}"),
        x_label: "Index".into(),
        y_label: name.clone(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    });
    let mut cols = Vec::new();
    if v.flag("store_fits") {
        cols.push(Column::numeric(format!("{name}_fits"), fitted.clone()));
    }
    if v.flag("store_resid") {
        cols.push(Column::numeric(format!("{name}_resid"), y.iter().zip(&fitted).map(|(a, f)| a - f).collect()));
    }
    if v.flag("store_fc") && h > 0 {
        cols.push(Column::numeric(format!("{name}_fcst"), forecasts));
    }
    stored(s, cols)
}

// ------------------------------------------------------------------------- matrices

const OPS: [&str; 10] = [
    "Transpose",
    "Invert",
    "Determinant (into constant)",
    "Trace (into constant)",
    "Multiply A × B",
    "Add A + B",
    "Subtract A − B",
    "Cholesky factor",
    "Eigenvalues and vectors (symmetric)",
    "Columns to matrix / matrix to columns",
];

fn get_matrix(ws: &Worksheet, k: usize, what: &str) -> Res<Matrix> {
    let m = ws.matrices.get(&k).ok_or_else(|| DialogError(format!("{what} M{k} does not exist")))?;
    Matrix::from_vec(m.rows, m.cols, m.data.clone()).map_err(|e| DialogError(e.to_string()))
}

fn to_stored(m: &Matrix) -> StoredMatrix {
    StoredMatrix { rows: m.rows(), cols: m.cols(), data: m.as_slice().to_vec() }
}

fn matrix_run(g: &mut GridController, v: &Values) -> Res<String> {
    let (ka, kb, kr) = (v.count("a"), v.count("b"), v.count("result"));
    if ka == 0 || kr == 0 {
        return Err("matrices are numbered from M1".into());
    }
    let op = v.choice("op");
    let ws = g.worksheet().clone();
    let me = |e: openstats_math::error::MathError| DialogError(e.to_string());
    let msg;
    match op {
        0 => {
            let r = get_matrix(&ws, ka, "matrix")?.t();
            g.set_matrix(kr, Some(to_stored(&r)))?;
            msg = format!("M{kr} = transpose of M{ka} ({} × {}).", r.rows(), r.cols());
        }
        1 => {
            let a = get_matrix(&ws, ka, "matrix")?;
            if !a.is_square() {
                return Err(DialogError(format!("M{ka} is not square; only square matrices can be inverted")));
            }
            let lu = Lu::new(&a).map_err(me)?;
            if lu.is_singular() {
                return Err(DialogError(format!("M{ka} is singular and cannot be inverted")));
            }
            let r = lu.inverse().map_err(me)?;
            g.set_matrix(kr, Some(to_stored(&r)))?;
            msg = format!("M{kr} = inverse of M{ka}.");
        }
        2 | 3 => {
            let a = get_matrix(&ws, ka, "matrix")?;
            if !a.is_square() {
                return Err(DialogError(format!("M{ka} is not square")));
            }
            let (val, what) =
                if op == 2 { (Lu::new(&a).map_err(me)?.det(), "determinant") } else { (a.trace(), "trace") };
            g.set_constant(kr, val)?;
            msg = format!("K{kr} = {what} of M{ka} = {val}.");
        }
        4..=6 => {
            let a = get_matrix(&ws, ka, "matrix")?;
            let b = get_matrix(&ws, kb.max(1), "matrix")?;
            let (r, what) = match op {
                4 => (a.matmul(&b).map_err(me)?, "product"),
                5 => (a.try_add(&b).map_err(me)?, "sum"),
                _ => (a.try_sub(&b).map_err(me)?, "difference"),
            };
            g.set_matrix(kr, Some(to_stored(&r)))?;
            msg = format!("M{kr} = {what} of M{ka} and M{} ({} × {}).", kb.max(1), r.rows(), r.cols());
        }
        7 => {
            let a = get_matrix(&ws, ka, "matrix")?;
            if !a.is_symmetric(1e-9 * a.norm_max().max(1.0)) {
                return Err(DialogError(format!("M{ka} is not symmetric")));
            }
            let ch = Cholesky::new(&a).map_err(|_| DialogError(format!("M{ka} is not positive definite")))?;
            g.set_matrix(kr, Some(to_stored(ch.l())))?;
            msg = format!("M{kr} = lower Cholesky factor of M{ka}.");
        }
        8 => {
            let a = get_matrix(&ws, ka, "matrix")?;
            if !a.is_symmetric(1e-9 * a.norm_max().max(1.0)) {
                return Err(DialogError(format!("M{ka} is not symmetric")));
            }
            let e = SymmetricEigen::new(&a).map_err(me)?;
            let vals = Matrix::column(&e.values);
            g.set_matrix(kr, Some(to_stored(&vals)))?;
            g.set_matrix(kr + 1, Some(to_stored(&e.vectors)))?;
            msg = format!("M{kr} = eigenvalues of M{ka}; M{} = eigenvectors (columns).", kr + 1);
        }
        _ => {
            let cols = v.vars("cols");
            if cols.is_empty() {
                // Matrix to columns.
                let a = get_matrix(&ws, ka, "matrix")?;
                let out: Vec<Column> =
                    (0..a.cols()).map(|j| Column::numeric(format!("M{ka}_{}", j + 1), a.col(j))).collect();
                let n = out.len();
                g.store_columns(out)?;
                msg = format!("Stored M{ka} in {n} columns.");
            } else {
                let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(&ws, c)).collect::<Res<_>>()?;
                let a = Matrix::from_columns(&data).map_err(me)?;
                g.set_matrix(kr, Some(to_stored(&a)))?;
                msg = format!("M{kr} = {} columns copied ({} × {}).", cols.len(), a.rows(), a.cols());
            }
        }
    }
    Ok(format!("Matrix Operations: {msg}"))
}

// ------------------------------------------------------------------------- project properties

fn properties_run(g: &mut GridController, v: &Values) -> Res<String> {
    let title = v.text("title").trim().to_string();
    let wb = g.wb_mut();
    if !title.is_empty() {
        wb.title = title;
    }
    for (key, field) in [("author", "author"), ("description", "description")] {
        let t = v.text(field).trim().to_string();
        if t.is_empty() {
            wb.metadata.remove(key);
        } else {
            wb.metadata.insert(key.to_string(), t);
        }
    }
    Ok(format!("Project Properties: saved (title '{}').", wb.title))
}

// ------------------------------------------------------------------------- registry

/// Register the dialogs of this module.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Main, Options, Storage};
    use VarKind::{Any, Numeric};
    add(
        r,
        DialogSpec::new(
            "data.replace_missing",
            "Replace Missing Values",
            "Fill the missing values of numeric columns.",
        )
        .vars("vars", "Columns", Numeric, 1, 50)
        .choice(
            Main,
            "how",
            "Replace with",
            &["A value", "Column mean", "Column median", "Previous value", "Next value", "Linear interpolation"],
            1,
        )
        .optional_number(Main, "value", "Value (when replacing with a value)"),
        &["Data", "Replace Missing Values..."],
        Runner::Report(replace_missing_run),
    );
    add(
        r,
        DialogSpec::new(
            "data.unpivot",
            "Unpivot Columns",
            "Turn several columns into one column of values with a column of their names.",
        )
        .vars("vars", "Columns to unpivot", Any, 1, 50)
        .vars("ids", "Columns to keep with each row", Any, 0, 20)
        .text(Options, "var_name", "Name of the column of variable names", "Variable")
        .text(Options, "val_name", "Name of the column of values", "Value"),
        &["Data", "Unpivot Columns..."],
        Runner::Report(unpivot_run),
    );
    add(
        r,
        DialogSpec::new("stat.ts.lag", "Lag", "Shift a series down so each row holds an earlier value.")
            .vars("vars", "Series", Numeric, 1, 50)
            .integer(Main, "lag", "Lag", 1.0, 1.0, 10000.0),
        &["Stat", "Time Series", "Lag..."],
        Runner::Report(lag_run),
    );
    add(
        r,
        DialogSpec::new("stat.ts.diff", "Differences", "Differences between values a lag apart.")
            .vars("vars", "Series", Numeric, 1, 50)
            .integer(Main, "lag", "Lag", 1.0, 1.0, 10000.0)
            .integer(Main, "times", "Times to difference", 1.0, 1.0, 10.0),
        &["Stat", "Time Series", "Differences..."],
        Runner::Report(difference_run),
    );
    add(
        r,
        DialogSpec::new("stat.ts.ma", "Moving Average", "Moving averages of a series.")
            .vars("vars", "Series", Numeric, 1, 50)
            .integer(Main, "length", "MA length", 3.0, 1.0, 10000.0)
            .check(Options, "centered", "Center the moving averages", false),
        &["Stat", "Time Series", "Moving Average..."],
        Runner::Report(moving_average_run),
    );
    add(
        r,
        DialogSpec::new("stat.ts.trend", "Trend Analysis", "Fit a trend to a series and forecast it.")
            .vars("y", "Series", Numeric, 1, 1)
            .choice(Main, "model", "Model type", &["Linear", "Quadratic", "Exponential growth"], 0)
            .integer(Options, "forecasts", "Number of forecasts", 0.0, 0.0, 1000.0)
            .check(Storage, "store_fits", "Fits", false)
            .check(Storage, "store_resid", "Residuals", false)
            .check(Storage, "store_fc", "Forecasts", false),
        &["Stat", "Time Series", "Trend Analysis..."],
        Runner::Report(trend_run),
    );
    add(
        r,
        DialogSpec::new("calc.matrix", "Matrix Operations", "Operate on stored matrices M1, M2, …")
            .choice(Main, "op", "Operation", &OPS, 0)
            .integer(Main, "a", "Matrix A (M number)", 1.0, 1.0, 999.0)
            .integer(Main, "b", "Matrix B (M number, for A and B)", 2.0, 1.0, 999.0)
            .integer(Main, "result", "Store the result in M or K number", 3.0, 1.0, 999.0)
            .vars("cols", "Columns to copy into a matrix (blank: matrix A to columns)", Numeric, 0, 100),
        &["Calc", "Matrices", "Matrix Operations..."],
        Runner::Mutate(matrix_run),
    );
    add(
        r,
        DialogSpec::new("file.properties", "Project Properties", "Title, author and description of the project.")
            .text(Main, "title", "Title", "")
            .text(Main, "author", "Author", "")
            .text(Main, "description", "Description", ""),
        &["File", "Project Properties..."],
        Runner::Mutate(properties_run),
    );
}
