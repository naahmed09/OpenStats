//! Data and Calc menus: stack, unstack, transpose, copy, subset, split, merge, recode, rank,
//! concatenate, change type, extract date parts, text functions, display; random data,
//! probability distributions, column and row statistics, patterned data, indicator variables.

use super::analyses::add;
use super::analysis::{col_title, labels, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::{Column, ColumnKind};
use openstats_core::ops::change_type::{change_type, ChangeTypeOptions};
use openstats_core::ops::concatenate::concatenate;
use openstats_core::ops::extract::{extract_date_part, text_op, DatePart, TextOp};
use openstats_core::ops::merge::{merge_by_keys, merge_side_by_side, JoinKind};
use openstats_core::ops::patterned;
use openstats_core::ops::rank::{rank, TieMethod};
use openstats_core::ops::recode::{indicator_columns, recode, RecodeRule, Unmatched};
use openstats_core::ops::split::split_worksheet;
use openstats_core::ops::stack::{stack_columns, Subscripts};
use openstats_core::ops::subset::{copy_columns, subset_worksheet, RowSelector};
use openstats_core::ops::transpose::transpose;
use openstats_core::ops::unstack::{unstack, UnstackOptions};
use openstats_core::value::Value;
use openstats_core::Worksheet;
use openstats_stats::report::{Cell, Column as TCol, ColumnKind as TKind, Section, Table};

use crate::grid::GridController;

fn stored(section: Section, cols: Vec<Column>) -> Res<Output> {
    Ok(Output { section, stored: cols, models: Vec::new() })
}

fn note(title: &str, text: String, cols: Vec<Column>) -> Res<Output> {
    let mut s = Section::new(title);
    s.text(text);
    stored(s, cols)
}

fn numbers(text: &str, what: &str) -> Res<Vec<f64>> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite())
                .ok_or_else(|| DialogError(format!("'{t}' in the {what} box is not a number")))
        })
        .collect()
}

// ------------------------------------------------------------------------- row selection

const SELECTORS: [&str; 6] = [
    "All rows",
    "Row range",
    "Rows where the column has one of the values",
    "Rows where the column is between two values",
    "Rows where the column is missing",
    "Rows where the column is not missing",
];

fn selector_fields(spec: DialogSpec) -> DialogSpec {
    use Page::Options;
    spec.choice(Options, "select", "Rows to use", &SELECTORS, 0)
        .integer(Options, "from_row", "First row", 1.0, 1.0, 1e9)
        .integer(Options, "to_row", "Last row", 1.0, 1.0, 1e9)
        .vars("sel_col", "Column for the row condition", VarKind::Any, 0, 1)
        .text(Options, "sel_values", "Values (separated by semicolons)", "")
        .optional_number(Options, "lo", "Lower limit")
        .optional_number(Options, "hi", "Upper limit")
        .check(Options, "exclude", "Exclude the selected rows instead", false)
}

fn selector(ws: &Worksheet, v: &Values) -> Res<RowSelector> {
    let need_col = || v.var("sel_col").ok_or_else(|| DialogError("choose the column for the row condition".into()));
    Ok(match v.choice("select") {
        0 => RowSelector::All,
        1 => {
            let (a, b) = (v.count("from_row"), v.count("to_row"));
            if a == 0 || b < a {
                return Err("the last row must not be before the first row".into());
            }
            RowSelector::Range(a - 1, b)
        }
        2 => {
            let c = need_col()?;
            let numeric_col = ws.column(c)?.kind() != ColumnKind::Text;
            let vals = v
                .text("sel_values")
                .split(';')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(|t| {
                    if numeric_col {
                        t.parse::<f64>().map(Value::Numeric).map_err(|_| DialogError(format!("'{t}' is not a number")))
                    } else {
                        Ok(Value::Text(t.to_string()))
                    }
                })
                .collect::<Res<Vec<Value>>>()?;
            if vals.is_empty() {
                return Err("enter at least one value".into());
            }
            RowSelector::ValuesIn { col: c, values: vals }
        }
        3 => {
            let c = need_col()?;
            let (lo, hi) = (v.num("lo"), v.num("hi"));
            if !(lo.is_finite() && hi.is_finite()) || lo > hi {
                return Err("enter a lower limit that is not above the upper limit".into());
            }
            RowSelector::Between { col: c, lo, hi }
        }
        4 => RowSelector::Missing(need_col()?),
        _ => RowSelector::NotMissing(need_col()?),
    })
}

// ------------------------------------------------------------------------- data commands

fn stack_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let subs = match v.choice("subs") {
        0 => Subscripts::None,
        1 => Subscripts::Names,
        _ => Subscripts::Numbers,
    };
    let name = v.text("name").trim();
    let name = if name.is_empty() { "Stacked" } else { name };
    let mut out = stack_columns(ws, cols, subs, name)?;
    if subs != Subscripts::None && !v.text("subs_name").trim().is_empty() {
        if let Some(c) = out.get_mut(1) {
            c.props.name = v.text("subs_name").trim().to_string();
        }
    }
    let n = out.first().map_or(0, Column::len);
    note("Stack Columns", format!("Stacked {} columns into {n} rows.", cols.len()), out)
}

fn unstack_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let out =
        unstack(ws, v.vars("vars"), v.vars("by"), UnstackOptions { include_missing_group: v.flag("include_missing") })?;
    note("Unstack Columns", format!("Created {} columns.", out.len()), out)
}

fn transpose_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let out = transpose(ws, v.vars("vars"), v.flag("labels"), v.var("names_from"))?;
    note("Transpose Columns", format!("Transposed {} columns into {} columns.", v.vars("vars").len(), out.len()), out)
}

fn copy_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let out = copy_columns(ws, v.vars("vars"), &selector(ws, v)?, v.flag("exclude"))?;
    let n = out.iter().map(Column::len).max().unwrap_or(0);
    note("Copy Columns", format!("Copied {} columns ({n} rows).", out.len()), out)
}

fn subset_run(g: &mut GridController, v: &Values) -> Res<String> {
    let name = v.text("name").trim().to_string();
    let name = if name.is_empty() { "Subset".to_string() } else { name };
    let sheet = subset_worksheet(g.worksheet(), &selector(g.worksheet(), v)?, v.flag("exclude"), &name)?;
    let rows = sheet.n_rows();
    g.add_worksheets(vec![sheet])?;
    Ok(format!("Subset Worksheet: created '{name}' with {rows} rows."))
}

fn split_run(g: &mut GridController, v: &Values) -> Res<String> {
    let sheets = split_worksheet(g.worksheet(), v.vars("by"), v.flag("include_missing"))?;
    let n = sheets.len();
    g.add_worksheets(sheets)?;
    Ok(format!("Split Worksheet: created {n} worksheets."))
}

fn merge_run(g: &mut GridController, v: &Values) -> Res<String> {
    let other = v.text("other").trim().to_string();
    let right = g
        .workbook()
        .worksheets
        .iter()
        .find(|w| w.name.eq_ignore_ascii_case(&other))
        .ok_or_else(|| DialogError(format!("there is no worksheet named '{other}'")))?
        .clone();
    let name = v.text("name").trim().to_string();
    let name = if name.is_empty() { "Merged".to_string() } else { name };
    let left = g.worksheet().clone();
    let merged = if v.choice("how") == 0 {
        merge_side_by_side(&left, &right, &name)?
    } else {
        let lk = v.vars("keys").to_vec();
        let rk: Vec<usize> = numbers(v.text("other_keys"), "key columns of the other worksheet")?
            .into_iter()
            .map(|x| if x >= 1.0 { x as usize - 1 } else { usize::MAX })
            .collect();
        if rk.contains(&usize::MAX) {
            return Err("key columns of the other worksheet are numbered from 1".into());
        }
        let join = match v.choice("how") {
            1 => JoinKind::Inner,
            2 => JoinKind::Left,
            _ => JoinKind::Outer,
        };
        merge_by_keys(&left, &right, &lk, &rk, join, &name)?
    };
    let rows = merged.n_rows();
    g.add_worksheets(vec![merged])?;
    Ok(format!("Merge Worksheets: created '{name}' with {rows} rows."))
}

/// `a=b; lo:hi=b; *=b` rules.
fn parse_rules(text: &str, numeric_in: bool, out_numeric: bool) -> Res<Vec<RecodeRule>> {
    let value = |s: &str, numeric_target: bool| -> Res<Value> {
        let s = s.trim();
        if s == "*" {
            Ok(Value::Missing)
        } else if numeric_target {
            s.parse::<f64>().map(Value::Numeric).map_err(|_| DialogError(format!("'{s}' is not a number")))
        } else {
            Ok(Value::Text(s.to_string()))
        }
    };
    let mut rules = Vec::new();
    for part in text.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let (from, to) = part.split_once('=').ok_or_else(|| DialogError(format!("'{part}' needs an '=' (old=new)")))?;
        let to = value(to, out_numeric)?;
        let from = from.trim();
        if from == "*" {
            rules.push(RecodeRule::Missing { to });
        } else if let Some((lo, hi)) = from.split_once(':').filter(|_| numeric_in) {
            let (lo, hi) = match (lo.trim().parse::<f64>(), hi.trim().parse::<f64>()) {
                (Ok(a), Ok(b)) if a <= b => (a, b),
                _ => return Err(format!("'{from}' is not a range lo:hi").into()),
            };
            rules.push(RecodeRule::Range { lo, hi, lo_inclusive: true, hi_inclusive: true, to });
        } else {
            rules.push(RecodeRule::Value { from: value(from, numeric_in)?, to });
        }
    }
    if rules.is_empty() {
        return Err("enter at least one rule, such as 1=low; 2:5=mid; *=0".into());
    }
    Ok(rules)
}

fn code_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let out_numeric = v.choice("to") == 0;
    let unmatched = match v.choice("unmatched") {
        0 => Unmatched::KeepOriginal,
        _ => Unmatched::Missing,
    };
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        let col = ws.column(c)?;
        let rules = parse_rules(v.text("rules"), col.kind() != ColumnKind::Text, out_numeric)?;
        let kind = if out_numeric { ColumnKind::Numeric } else { ColumnKind::Text };
        let data = recode(&col.data, &rules, kind, &unmatched, v.flag("case"))?;
        out.push(Column::new(format!("{}_coded", col_title(ws, c)), data));
    }
    note("Code", format!("Recoded {} columns.", out.len()), out)
}

fn rank_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let ties = match v.choice("ties") {
        0 => TieMethod::Average,
        1 => TieMethod::Min,
        2 => TieMethod::Max,
        3 => TieMethod::Ordinal,
        _ => TieMethod::Dense,
    };
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        let r = rank(&numeric(ws, c)?, ties, v.choice("order") == 1);
        out.push(Column::numeric(format!("Rank_{}", col_title(ws, c)), r));
    }
    note("Rank", format!("Ranked {} columns.", out.len()), out)
}

fn concat_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let name = v.text("name").trim();
    let name = if name.is_empty() { "Combined" } else { name };
    let col = concatenate(ws, v.vars("vars"), v.text("sep"), v.flag("skip_missing"), v.flag("trim"), name)?;
    note("Concatenate", format!("Combined {} columns.", v.vars("vars").len()), vec![col])
}

fn type_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let to = match v.choice("to") {
        0 => ColumnKind::Numeric,
        1 => ColumnKind::Text,
        _ => ColumnKind::DateTime,
    };
    let opts = ChangeTypeOptions {
        strict: v.flag("strict"),
        decimal: if v.choice("decimal") == 1 { ',' } else { '.' },
        ..Default::default()
    };
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        let data = change_type(&ws.column(c)?.data, to, &opts)?;
        out.push(Column::new(format!("{}_{}", col_title(ws, c), to.name()), data));
    }
    note("Change Data Type", format!("Converted {} columns to {}.", out.len(), to.name()), out)
}

const PARTS: [(&str, DatePart); 12] = [
    ("Year", DatePart::Year),
    ("Quarter", DatePart::Quarter),
    ("Month", DatePart::Month),
    ("Day", DatePart::Day),
    ("Hour", DatePart::Hour),
    ("Minute", DatePart::Minute),
    ("Second", DatePart::Second),
    ("Weekday (1-7)", DatePart::Weekday),
    ("Day of year", DatePart::DayOfYear),
    ("ISO week", DatePart::IsoWeek),
    ("Month name", DatePart::MonthName),
    ("Weekday name", DatePart::WeekdayName),
];

fn extract_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (label, part) = PARTS.get(v.choice("part")).copied().ok_or("choose the part to extract")?;
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        out.push(Column::new(
            format!("{}_{}", col_title(ws, c), label.split(' ').next().unwrap_or(label)),
            extract_date_part(&ws.column(c)?.data, part)?,
        ));
    }
    note("Extract from Date/Time", format!("Extracted '{label}' from {} columns.", out.len()), out)
}

fn text_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let n = v.count("n");
    let op = match v.choice("op") {
        0 => TextOp::Left(n),
        1 => TextOp::Right(n),
        2 => TextOp::Mid { start: v.count("start").max(1), len: n },
        3 => TextOp::Trim,
        4 => TextOp::Upper,
        5 => TextOp::Lower,
        6 => TextOp::Proper,
        _ => TextOp::Substitute { from: v.text("from").to_string(), to: v.text("to").to_string() },
    };
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        out.push(Column::new(format!("{}_text", col_title(ws, c)), text_op(&ws.column(c)?.data, &op)?));
    }
    note("Text Functions", format!("Transformed {} columns.", out.len()), out)
}

fn display_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let rows = cols.iter().map(|&c| ws.columns.get(c).map_or(0, Column::len)).max().unwrap_or(0);
    let shown = rows.min(5000);
    let heads: Vec<TCol> = cols
        .iter()
        .map(|&c| {
            let kind = if ws.columns[c].kind() == ColumnKind::Numeric { TKind::Significant(6) } else { TKind::Text };
            TCol::new(col_title(ws, c), kind)
        })
        .collect();
    let mut t = Table::new("Data Display", heads);
    for i in 0..shown {
        let row: Vec<Cell> = cols
            .iter()
            .map(|&c| {
                let col = &ws.columns[c];
                if i >= col.len() || col.data.is_missing(i) {
                    Cell::Missing
                } else if col.kind() == ColumnKind::Numeric {
                    Cell::Num(col.data.get(i).as_f64())
                } else {
                    Cell::text(col.display(i))
                }
            })
            .collect();
        t.push_row(row)?;
    }
    let mut s = Section::new("Data Display");
    s.table(t);
    if shown < rows {
        s.warn(format!("Only the first {shown} of {rows} rows are shown."));
    }
    stored(s, Vec::new())
}

// ------------------------------------------------------------------------- calc commands

const DISTS: [(&str, &str); 24] = [
    ("Normal (mean, stdev)", "normal"),
    ("t (df)", "t"),
    ("Chi-square (df)", "chisquare"),
    ("F (df1, df2)", "f"),
    ("Uniform (lower, upper)", "uniform"),
    ("Exponential (scale, threshold)", "exponential"),
    ("Weibull (shape, scale, threshold)", "weibull"),
    ("Lognormal (location, scale, threshold)", "lognormal"),
    ("Gamma (shape, scale, threshold)", "gamma"),
    ("Beta (a, b)", "beta"),
    ("Logistic (location, scale)", "logistic"),
    ("Loglogistic (location, scale, threshold)", "loglogistic"),
    ("Cauchy (location, scale)", "cauchy"),
    ("Laplace (location, scale)", "laplace"),
    ("Triangular (lower, mode, upper)", "triangular"),
    ("Smallest extreme value (location, scale)", "sev"),
    ("Largest extreme value (location, scale)", "lev"),
    ("Binomial (trials, p)", "binomial"),
    ("Poisson (mean)", "poisson"),
    ("Geometric (p)", "geometric"),
    ("Negative binomial (successes, p)", "negativebinomial"),
    ("Hypergeometric (population, events, sample)", "hypergeometric"),
    ("Integer (lower, upper)", "integer"),
    ("Half-normal (scale, threshold)", "halfnormal"),
];

fn dist_labels() -> Vec<&'static str> {
    DISTS.iter().map(|d| d.0).collect()
}

fn dist_call(v: &Values) -> Res<(String, String)> {
    let (_, name) = DISTS.get(v.choice("dist")).ok_or("choose a distribution")?;
    let params = numbers(v.text("params"), "parameters")?;
    let list: String = params.iter().map(|p| format!(", {p}")).collect();
    Ok(((*name).to_string(), list))
}

/// Target column (1-based, 0 = next free) and its name.
fn target_column(g: &GridController, v: &Values) -> usize {
    match v.count("target") {
        0 => g.worksheet().next_free_column(),
        t => t - 1,
    }
}

fn random_run(g: &mut GridController, v: &Values) -> Res<String> {
    let (name, params) = dist_call(v)?;
    let n = v.count("n");
    if n == 0 {
        return Err("enter the number of rows of data to generate".into());
    }
    let seed = v.count("seed");
    let expr = if seed > 0 {
        format!("RANDOMSEED({seed}, {n}, \"{name}\"{params})")
    } else {
        format!("RANDOM({n}, \"{name}\"{params})")
    };
    let col = target_column(g, v);
    g.calculate(col, &expr)?;
    let label = v.text("store").trim().to_string();
    if !label.is_empty() {
        g.rename_column(col, &label)?;
    }
    Ok(format!("Random Data: C{} = {expr}", col + 1))
}

const PROB: [(&str, &str); 4] = [
    ("Probability density (or mass)", "PDF"),
    ("Cumulative probability", "CDF"),
    ("Survival (upper tail) probability", "SFCDF"),
    ("Inverse cumulative probability", "INVCDF"),
];

fn prob_run(g: &mut GridController, v: &Values) -> Res<String> {
    let (name, params) = dist_call(v)?;
    let (_, func) = PROB[v.choice("mode").min(3)];
    let input = match v.var("input") {
        Some(c) => format!("C{}", c + 1),
        None => {
            let k = v.num("constant");
            if !k.is_finite() {
                return Err("choose a column of values or enter a constant".into());
            }
            format!("{k}")
        }
    };
    let expr = format!("{func}({input}, \"{name}\"{params})");
    let col = target_column(g, v);
    g.calculate(col, &expr)?;
    let label = v.text("store").trim().to_string();
    if !label.is_empty() {
        g.rename_column(col, &label)?;
    }
    Ok(format!("Probability Distributions: C{} = {expr}", col + 1))
}

fn mean_sd(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let sd = if x.len() > 1 { (x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0)).sqrt() } else { f64::NAN };
    (m, sd)
}

fn median(x: &mut [f64]) -> f64 {
    x.sort_by(|a, b| a.total_cmp(b));
    let n = x.len();
    if n % 2 == 1 {
        x[n / 2]
    } else {
        (x[n / 2 - 1] + x[n / 2]) / 2.0
    }
}

fn colstats_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let heads = ["Variable", "N", "N*", "Sum", "Mean", "StDev", "Minimum", "Median", "Maximum", "SSQ"];
    let kinds = [
        TKind::Text,
        TKind::Integer,
        TKind::Integer,
        TKind::Significant(6),
        TKind::Significant(6),
        TKind::Significant(6),
        TKind::Significant(6),
        TKind::Significant(6),
        TKind::Significant(6),
        TKind::Significant(6),
    ];
    let mut t = Table::new("Column Statistics", heads.iter().zip(kinds).map(|(h, k)| TCol::new(*h, k)).collect());
    for &c in v.vars("vars") {
        let all = numeric(ws, c)?;
        let mut x: Vec<f64> = all.iter().copied().filter(|a| !a.is_nan()).collect();
        let miss = all.len() - x.len();
        if x.is_empty() {
            t.push_row(vec![
                Cell::text(col_title(ws, c)),
                Cell::Int(0),
                Cell::Int(miss as i64),
                Cell::Missing,
                Cell::Missing,
                Cell::Missing,
                Cell::Missing,
                Cell::Missing,
                Cell::Missing,
                Cell::Missing,
            ])?;
            continue;
        }
        let (m, sd) = mean_sd(&x);
        let sum: f64 = x.iter().sum();
        let ssq: f64 = x.iter().map(|a| a * a).sum();
        let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &a| (l.min(a), h.max(a)));
        let n = x.len();
        let med = median(&mut x);
        t.push_row(vec![
            Cell::text(col_title(ws, c)),
            Cell::Int(n as i64),
            Cell::Int(miss as i64),
            Cell::Num(sum),
            Cell::Num(m),
            if sd.is_nan() { Cell::Missing } else { Cell::Num(sd) },
            Cell::Num(lo),
            Cell::Num(med),
            Cell::Num(hi),
            Cell::Num(ssq),
        ])?;
    }
    let mut s = Section::new("Column Statistics");
    s.table(t);
    stored(s, Vec::new())
}

const ROW_STATS: [&str; 8] =
    ["Sum", "Mean", "Standard deviation", "Minimum", "Maximum", "Median", "N nonmissing", "N missing"];

fn rowstats_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols: Vec<Vec<f64>> = v.vars("vars").iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let n = cols.iter().map(Vec::len).max().unwrap_or(0);
    let which = v.choice("stat").min(7);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut x: Vec<f64> = cols.iter().filter_map(|c| c.get(i).copied()).filter(|a| !a.is_nan()).collect();
        let k = x.len();
        let miss = cols.len() - k;
        out.push(match which {
            6 => k as f64,
            7 => miss as f64,
            _ if k == 0 => f64::NAN,
            0 => x.iter().sum(),
            1 => mean_sd(&x).0,
            2 => mean_sd(&x).1,
            3 => x.iter().copied().fold(f64::INFINITY, f64::min),
            4 => x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            _ => median(&mut x),
        });
    }
    let name = v.text("store").trim();
    let name = if name.is_empty() { format!("Row_{}", ROW_STATS[which].replace(' ', "_")) } else { name.to_string() };
    note(
        "Row Statistics",
        format!("{} across {} columns.", ROW_STATS[which], cols.len()),
        vec![Column::numeric(name, out)],
    )
}

fn patterned_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let (from, to, step) = (v.num("from"), v.num("to"), v.num("step"));
    if !(from.is_finite() && to.is_finite() && step.is_finite()) || step == 0.0 {
        return Err("enter a first value, a last value and a nonzero step".into());
    }
    let data = patterned::numbers(from, to, step, v.count("each").max(1), v.count("list").max(1))?;
    let name = v.text("store").trim();
    let name = if name.is_empty() { "Pattern" } else { name };
    let n = data.len();
    note("Make Patterned Data", format!("Created {n} values."), vec![Column::new(name, data)])
}

fn indicator_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut out = Vec::new();
    for &c in v.vars("vars") {
        out.extend(indicator_columns(ws.column(c)?));
    }
    note("Make Indicator Variables", format!("Created {} indicator columns.", out.len()), out)
}

fn sheet_title(ws: &Worksheet) -> String {
    ws.name.clone()
}

/// Register the Data and Calc dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Main, Options};
    use VarKind::{Any, Numeric};
    let _ = (labels as fn(&Worksheet, usize) -> Res<Vec<String>>, sheet_title as fn(&Worksheet) -> String);

    add(
        r,
        DialogSpec::new(
            "data.stack",
            "Stack Columns",
            "Stack several columns into one, with an optional subscript column.",
        )
        .vars("vars", "Columns to stack", Any, 2, 100)
        .text(Main, "name", "Name of the stacked column", "Stacked")
        .choice(Options, "subs", "Subscripts", &["None", "Column names (text)", "Numbers 1, 2, ..."], 1)
        .text(Options, "subs_name", "Name of the subscript column", "Subscripts"),
        &["Data", "Stack", "Columns..."],
        Runner::Report(stack_run),
    );
    add(
        r,
        DialogSpec::new(
            "data.unstack",
            "Unstack Columns",
            "Split columns into one column per level of the subscript columns.",
        )
        .vars("vars", "Columns to unstack", Any, 1, 50)
        .vars("by", "Subscripts", Any, 1, 4)
        .check(Options, "include_missing", "Treat missing subscripts as a group", false),
        &["Data", "Unstack", "Columns..."],
        Runner::Report(unstack_run),
    );
    add(
        r,
        DialogSpec::new("data.transpose", "Transpose Columns", "Turn columns into rows.")
            .vars("vars", "Columns to transpose", Any, 1, 200)
            .check(Options, "labels", "Add a column of the original column names", true)
            .vars("names_from", "Column holding the new column names (optional)", Any, 0, 1),
        &["Data", "Transpose Columns..."],
        Runner::Report(transpose_run),
    );
    add(
        r,
        selector_fields(
            DialogSpec::new("data.copy", "Copy Columns to Columns", "Copy chosen columns, optionally only some rows.")
                .vars("vars", "Columns to copy", Any, 1, 200),
        ),
        &["Data", "Copy", "Columns to Columns..."],
        Runner::Report(copy_run),
    );
    add(
        r,
        selector_fields(
            DialogSpec::new("data.subset", "Subset Worksheet", "Create a new worksheet with some of the rows.").text(
                Main,
                "name",
                "Name of the new worksheet",
                "Subset",
            ),
        ),
        &["Data", "Subset Worksheet..."],
        Runner::Mutate(subset_run),
    );
    add(
        r,
        DialogSpec::new("data.split", "Split Worksheet", "Create one worksheet for each level of the chosen columns.")
            .vars("by", "Split by", Any, 1, 4)
            .check(Options, "include_missing", "Keep rows with a missing value as a group", false),
        &["Data", "Split Worksheet..."],
        Runner::Mutate(split_run),
    );
    add(
        r,
        DialogSpec::new(
            "data.merge",
            "Merge Worksheets",
            "Put another worksheet beside this one, or join the two on key columns.",
        )
        .text(Main, "other", "Name of the other worksheet", "")
        .choice(
            Main,
            "how",
            "Merge",
            &["Side by side", "By keys: matches only", "By keys: keep this worksheet's rows", "By keys: keep all rows"],
            0,
        )
        .vars("keys", "Key columns in this worksheet", Any, 0, 5)
        .text(Options, "other_keys", "Key column numbers in the other worksheet (1, 2, ...)", "")
        .text(Main, "name", "Name of the merged worksheet", "Merged"),
        &["Data", "Merge Worksheets..."],
        Runner::Mutate(merge_run),
    );
    add(
        r,
        DialogSpec::new(
            "data.code",
            "Code (Recode Values)",
            "Recode values with rules such as 1=low; 2:5=mid; *=0 (old=new, lo:hi ranges, * for missing).",
        )
        .vars("vars", "Columns to recode", Any, 1, 50)
        .text(Main, "rules", "Rules", "")
        .choice(Main, "to", "Store the result as", &["Numeric", "Text"], 0)
        .choice(Options, "unmatched", "Values that match no rule", &["Keep the original", "Make missing"], 0)
        .check(Options, "case", "Match text case", false),
        &["Data", "Code", "Recode Values..."],
        Runner::Report(code_run),
    );
    add(
        r,
        DialogSpec::new("data.rank", "Rank", "Replace values by their ranks.")
            .vars("vars", "Columns", Numeric, 1, 50)
            .choice(
                Options,
                "ties",
                "Ties",
                &["Average rank", "Lowest rank", "Highest rank", "In order of appearance", "Dense (no gaps)"],
                0,
            )
            .choice(Options, "order", "Rank 1 is the", &["Smallest value", "Largest value"], 0),
        &["Data", "Rank..."],
        Runner::Report(rank_run),
    );
    add(
        r,
        DialogSpec::new("data.concatenate", "Concatenate", "Join the text of several columns row by row.")
            .vars("vars", "Columns", Any, 2, 50)
            .text(Main, "sep", "Separator", " ")
            .text(Main, "name", "Name of the new column", "Combined")
            .check(Options, "skip_missing", "Skip missing values and their separators", false)
            .check(Options, "trim", "Trim spaces around each piece", true),
        &["Data", "Concatenate..."],
        Runner::Report(concat_run),
    );
    add(
        r,
        DialogSpec::new("data.change_type", "Change Data Type", "Convert columns between numeric, text and date/time.")
            .vars("vars", "Columns", Any, 1, 50)
            .choice(Main, "to", "Convert to", &["Numeric", "Text", "Date/time"], 0)
            .check(Options, "strict", "Stop at the first value that cannot be converted", false)
            .choice(Options, "decimal", "Decimal separator of text numbers", &["Period", "Comma"], 0),
        &["Data", "Change Data Type..."],
        Runner::Report(type_run),
    );
    add(
        r,
        DialogSpec::new("data.extract", "Extract from Date/Time", "Pull out the year, month, weekday and other parts.")
            .vars("vars", "Date/time columns", Any, 1, 50)
            .choice(Main, "part", "Part", &PARTS.map(|p| p.0), 0),
        &["Data", "Extract from Date/Time..."],
        Runner::Report(extract_run),
    );
    add(
        r,
        DialogSpec::new(
            "data.text",
            "Text Functions",
            "Left, right, middle, trim, case and substitute on text columns.",
        )
        .vars("vars", "Text columns", Any, 1, 50)
        .choice(
            Main,
            "op",
            "Function",
            &["Left", "Right", "Middle", "Trim", "Upper case", "Lower case", "Proper case", "Substitute"],
            0,
        )
        .integer(Main, "n", "Number of characters", 1.0, 0.0, 1e6)
        .integer(Options, "start", "Start (middle)", 1.0, 1.0, 1e6)
        .text(Options, "from", "Replace this text (substitute)", "")
        .text(Options, "to", "With this text", ""),
        &["Data", "Text Functions..."],
        Runner::Report(text_run),
    );
    add(
        r,
        DialogSpec::new("data.display", "Display Data", "Show columns in the Session window.")
            .vars("vars", "Columns", Any, 1, 100),
        &["Data", "Display Data..."],
        Runner::Report(display_run),
    );

    let dl = dist_labels();
    add(
        r,
        DialogSpec::new("calc.random", "Random Data", "Generate random values from a distribution into a column.")
            .integer(Main, "n", "Number of rows of data", 100.0, 1.0, 1e8)
            .choice(Main, "dist", "Distribution", &dl, 0)
            .text(Main, "params", "Parameters (numbers in the order shown; blank = defaults)", "")
            .text(Main, "store", "Store in column named", "")
            .integer(Options, "target", "Column number to store in (0 = next free column)", 0.0, 0.0, 4000.0)
            .integer(Options, "seed", "Base of the random sequence (0 = from the clock)", 0.0, 0.0, 1e12),
        &["Calc", "Random Data..."],
        Runner::Mutate(random_run),
    );
    add(
        r,
        DialogSpec::new(
            "calc.prob",
            "Probability Distributions",
            "Density, cumulative or inverse cumulative probabilities for a column of values or one constant.",
        )
        .choice(Main, "mode", "Compute", &PROB.map(|p| p.0), 1)
        .choice(Main, "dist", "Distribution", &dl, 0)
        .text(Main, "params", "Parameters (numbers in the order shown; blank = defaults)", "")
        .vars("input", "Input column", Numeric, 0, 1)
        .optional_number(Main, "constant", "or an input constant")
        .text(Main, "store", "Store in column named", "")
        .integer(Options, "target", "Column number to store in (0 = next free column)", 0.0, 0.0, 4000.0),
        &["Calc", "Probability Distributions..."],
        Runner::Mutate(prob_run),
    );
    add(
        r,
        DialogSpec::new(
            "calc.colstats",
            "Column Statistics",
            "N, sum, mean, standard deviation, quartile-free summary of columns.",
        )
        .vars("vars", "Columns", Numeric, 1, 200),
        &["Calc", "Column Statistics..."],
        Runner::Report(colstats_run),
    );
    add(
        r,
        DialogSpec::new("calc.rowstats", "Row Statistics", "A statistic across the chosen columns, row by row.")
            .vars("vars", "Columns", Numeric, 1, 200)
            .choice(Main, "stat", "Statistic", &ROW_STATS, 0)
            .text(Main, "store", "Store in column named", ""),
        &["Calc", "Row Statistics..."],
        Runner::Report(rowstats_run),
    );
    add(
        r,
        DialogSpec::new(
            "calc.patterned",
            "Make Patterned Data",
            "A simple numeric pattern: first, last, step, repeats of each value and of the whole list.",
        )
        .number(Main, "from", "From first value", 1.0, -1e15, 1e15)
        .number(Main, "to", "To last value", 10.0, -1e15, 1e15)
        .number(Main, "step", "In steps of", 1.0, -1e15, 1e15)
        .integer(Main, "each", "Number of times to list each value", 1.0, 1.0, 1e6)
        .integer(Main, "list", "Number of times to list the whole sequence", 1.0, 1.0, 1e6)
        .text(Main, "store", "Store in column named", "Pattern"),
        &["Calc", "Make Patterned Data..."],
        Runner::Report(patterned_run),
    );
    add(
        r,
        DialogSpec::new("calc.indicator", "Make Indicator Variables", "One 0/1 column per level of a column.")
            .vars("vars", "Columns", Any, 1, 20),
        &["Calc", "Make Indicator Variables..."],
        Runner::Report(indicator_run),
    );
}
