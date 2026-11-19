//! Commands: the statements of a script that are not control flow or `LET`.
//!
//! A command word may be abbreviated to its first four letters (`DESCRIBE` = `DESC`); short
//! words (`SET`, `N`) must be written in full. Subcommands follow after `;` and the command ends
//! with `.`; subcommand words need three letters.
//!
//! | Group | Commands |
//! |---|---|
//! | Data | `SET`, `READ`, `NAME`, `ERASE`, `COPY`, `SORT`, `PRINT` |
//! | Column statistics into constants | `MEAN`, `SUM`, `STDEV`, `MEDIAN`, `MINIMUM`, `MAXIMUM`, `RANGE`, `N`, `NMISS`, `COUNT`, `SSQ` |
//! | Statistics | `DESCRIBE`, `CORRELATION`, `REGRESS`, `TTEST`, `ONEWAY` |
//! | Predictive analytics | `CART`, `FOREST`, `BOOST` |
//! | Matrices | `TRANSPOSE`, `MULTIPLY`, `INVERT`, `COPY` between columns and matrices |
//! | Session | `ECHO`, `NOECHO`, `TEXT` |

use crate::ast::*;
use crate::error::ScriptError;
use crate::machine::{Env, Machine, Resolved};
use crate::Result;
use openstats_core::column::{Column, ColumnData};
use openstats_core::{Edit, StoredMatrix, Value};
use openstats_stats::anova::{one_way_stacked, OneWayOptions};
use openstats_stats::correlation::{kendall, pearson, spearman};
use openstats_stats::descriptive::describe;
use openstats_stats::hypothesis::Alternative;
use openstats_stats::ml::{
    fit_cart, fit_forest, fit_gbdt, CartOptions, Criterion, Dataset, ForestOptions, GbdtLoss, GbdtOptions, Predictions,
    Response,
};
use openstats_stats::regression::{fit_xy, RegressionOptions};
use openstats_stats::report::{Block, Cell, Column as RCol, ColumnKind, Reportable, Section, Table};
use openstats_stats::ttest::one_sample_t;

/// Every command word, for abbreviation and for the editor's highlighter.
pub const COMMANDS: &[&str] = &[
    "SET",
    "READ",
    "NAME",
    "ERASE",
    "COPY",
    "SORT",
    "PRINT",
    "MEAN",
    "SUM",
    "STDEV",
    "MEDIAN",
    "MINIMUM",
    "MAXIMUM",
    "RANGE",
    "N",
    "NMISS",
    "COUNT",
    "SSQ",
    "DESCRIBE",
    "CORRELATION",
    "REGRESS",
    "TTEST",
    "ONEWAY",
    "CART",
    "FOREST",
    "BOOST",
    "TRANSPOSE",
    "MULTIPLY",
    "INVERT",
    "ECHO",
    "NOECHO",
    "TEXT",
    "STACK",
    "UNSTACK",
    "OUTFILE",
    "NOOUTFILE",
    "EXECUTE",
];

/// The full command word for what was typed, if it is one.
pub fn canonical(word: &str) -> Option<&'static str> {
    let up = word.to_ascii_uppercase();
    if let Some(c) = COMMANDS.iter().find(|c| **c == up) {
        return Some(c);
    }
    if up.len() >= 4 {
        let hits: Vec<&&str> = COMMANDS.iter().filter(|c| c.starts_with(&up)).collect();
        if hits.len() == 1 {
            return Some(hits[0]);
        }
    }
    None
}

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(ScriptError::runtime(0, msg))
}

fn sub<'a>(c: &'a Command, name: &str) -> Option<&'a Sub> {
    c.subs.iter().find(|s| {
        let n = s.name.to_ascii_uppercase();
        n == name || (n.len() >= 3 && name.starts_with(&n))
    })
}

fn flatten_name(a: &Arg) -> String {
    match a {
        Arg::Word(w) | Arg::Quoted(w) => w.clone(),
        Arg::Num(x) => x.to_string(),
        Arg::Text(s) => s.clone(),
        Arg::Missing => "*".into(),
        Arg::Range(a, b) => format!("{}:{}", flatten_name(a), flatten_name(b)),
        Arg::Paren(e) => format!("({e})"),
    }
}

/// Columns named by `args`: words, quoted names and ranges (`C1-C5`), in order.
fn columns(m: &Machine, env: &Env, args: &[Arg]) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for a in args {
        match a {
            Arg::Range(lo, hi) => {
                let (l, h) = (one_col(m, env, lo)?, one_col(m, env, hi)?);
                if l > h {
                    return err(format!("the range {} is backwards", flatten_name(a)));
                }
                out.extend(l..=h);
            }
            _ => out.push(one_col(m, env, a)?),
        }
    }
    Ok(out)
}

fn one_col(m: &Machine, env: &Env, a: &Arg) -> Result<usize> {
    let r = match a {
        Arg::Word(w) => m.resolve(env, w)?,
        Arg::Quoted(q) => m.resolve_quoted(env, q),
        _ => return err(format!("`{}` is not a column", flatten_name(a))),
    };
    match r {
        Resolved::Col(c) => Ok(c),
        Resolved::Unknown(n) => err(format!("there is no column `{n}`")),
        _ => err(format!("`{}` is not a column", flatten_name(a))),
    }
}

fn number(m: &Machine, env: &Env, a: &Arg, what: &str) -> Result<f64> {
    match a {
        Arg::Num(x) => Ok(*x),
        Arg::Paren(e) => m.eval_number(env, e, what),
        Arg::Word(w) => m.eval_number(env, &Expr::Word(w.clone()), what),
        _ => err(format!("{what} must be a number")),
    }
}

/// The constant named by `a`, as its number `k`.
fn const_target(m: &Machine, env: &Env, a: &Arg) -> Result<usize> {
    let name = match a {
        Arg::Word(w) => w,
        _ => return err("a constant such as K1 is expected here"),
    };
    match m.resolve(env, name)? {
        Resolved::Const(k) => Ok(k),
        _ => err(format!("`{name}` is not a constant")),
    }
}

fn matrix_target(m: &Machine, env: &Env, a: &Arg) -> Result<usize> {
    let name = match a {
        Arg::Word(w) => w,
        _ => return err("a matrix such as M1 is expected here"),
    };
    match m.resolve(env, name)? {
        Resolved::Mat(k) => Ok(k),
        _ => err(format!("`{name}` is not a matrix")),
    }
}

fn is_mat(m: &Machine, env: &Env, a: &Arg) -> bool {
    matches!(a, Arg::Word(w) if matches!(m.resolve(env, w), Ok(Resolved::Mat(_))))
}

fn col_values(env: &Env, c: usize) -> Result<Vec<f64>> {
    let ws = env.wb.active_sheet().map_err(|e| ScriptError::runtime(0, e.to_string()))?;
    let col =
        ws.column(c).map_err(|_| ScriptError::runtime(0, format!("column C{} is empty or does not exist", c + 1)))?;
    match &col.data {
        ColumnData::Numeric(v) => Ok(v.clone()),
        ColumnData::Text(_) => err(format!("{} holds text; a number column is needed", title(env, c))),
        _ => Ok((0..col.len()).map(|i| col.data.get(i).as_f64()).collect()),
    }
}

fn col_labels(env: &Env, c: usize) -> Result<Vec<String>> {
    let ws = env.wb.active_sheet().map_err(|e| ScriptError::runtime(0, e.to_string()))?;
    let col =
        ws.column(c).map_err(|_| ScriptError::runtime(0, format!("column C{} is empty or does not exist", c + 1)))?;
    Ok(match &col.data {
        ColumnData::Text(v) => v.clone(),
        _ => (0..col.len()).map(|i| if col.data.is_missing(i) { String::new() } else { col.display(i) }).collect(),
    })
}

fn is_text(env: &Env, c: usize) -> bool {
    env.wb.active_sheet().ok().and_then(|w| w.columns.get(c)).is_some_and(|k| matches!(k.data, ColumnData::Text(_)))
}

fn title(env: &Env, c: usize) -> String {
    match env.wb.active_sheet().ok().and_then(|w| w.columns.get(c)) {
        Some(k) if !k.name().is_empty() => k.name().to_string(),
        _ => format!("C{}", c + 1),
    }
}

fn stat_err<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| ScriptError::runtime(0, e.to_string()))
}

/// Rows where every listed series is present.
fn complete(series: &[&[f64]]) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = series.iter().map(|s| s.len()).max().unwrap_or(0);
    let rows: Vec<usize> = (0..n).filter(|&i| series.iter().all(|s| s.get(i).is_some_and(|x| !x.is_nan()))).collect();
    let data = series.iter().map(|s| rows.iter().map(|&i| s[i]).collect()).collect();
    (rows, data)
}

fn spread(name: &str, n: usize, rows: &[usize], values: &[f64]) -> Column {
    let mut v = vec![f64::NAN; n];
    for (&r, &x) in rows.iter().zip(values) {
        v[r] = x;
    }
    Column::numeric(name, v)
}

fn put_numeric(
    m: &mut Machine,
    env: &mut Env,
    target: Option<&Sub>,
    c: usize,
    n: usize,
    rows: &[usize],
    v: &[f64],
) -> Result<()> {
    if let Some(s) = target {
        if let Some(a) = s.args.first() {
            let idx = one_col(m, env, a)?;
            m.write_column(env, idx, spread("", n, rows, v).data, None)?;
        }
    }
    let _ = c;
    Ok(())
}

fn section_text(m: &mut Machine, heading: &str, lines: &[String]) {
    let mut s = Section::new(heading);
    for l in lines {
        s.text(l.clone());
    }
    m.push_section(s);
}

/// Run one command.
pub(crate) fn run_command(m: &mut Machine, env: &mut Env, c: &Command, line: usize) -> Result<()> {
    let Some(word) = canonical(&c.name) else {
        return Err(ScriptError::runtime(line, format!("`{}` is not a command OpenStats knows", c.name)));
    };
    let r = match word {
        "SET" => set(m, env, c),
        "READ" => read(m, env, c),
        "NAME" => name(m, env, c),
        "ERASE" => erase(m, env, c),
        "COPY" => copy(m, env, c),
        "SORT" => sort(m, env, c),
        "PRINT" => print(m, env, c),
        "MEAN" | "SUM" | "STDEV" | "MEDIAN" | "MINIMUM" | "MAXIMUM" | "RANGE" | "N" | "NMISS" | "COUNT" | "SSQ" => {
            column_stat(m, env, c, word)
        }
        "DESCRIBE" => describe_cmd(m, env, c),
        "CORRELATION" => correlation(m, env, c),
        "REGRESS" => regress(m, env, c),
        "TTEST" => ttest(m, env, c),
        "ONEWAY" => oneway(m, env, c),
        "CART" | "FOREST" | "BOOST" => model(m, env, c, word),
        "TRANSPOSE" => transpose(m, env, c),
        "MULTIPLY" => multiply(m, env, c),
        "INVERT" => invert(m, env, c),
        "ECHO" => {
            m.echo = true;
            Ok(())
        }
        "NOECHO" => {
            m.echo = false;
            Ok(())
        }
        "STACK" => stack_cmd(m, env, c),
        "UNSTACK" => unstack_cmd(m, env, c),
        "OUTFILE" => {
            let Some(a) = c.args.first() else { return Err(ScriptError::runtime(line, "OUTFILE needs a file name")) };
            let name = flatten_name(a);
            m.open_outfile(&name);
            Ok(())
        }
        "NOOUTFILE" => {
            m.outfile = None;
            Ok(())
        }
        "EXECUTE" => err("EXECUTE needs the name of a stored script"),
        "TEXT" => {
            let lines: Vec<String> = c.args.iter().map(flatten_name).collect();
            section_text(m, "", &[lines.join(" ")]);
            Ok(())
        }
        _ => err(format!("`{}` is not a command OpenStats knows", c.name)),
    };
    r.map_err(|mut e| {
        if e.line == 0 {
            e.line = line;
        }
        e
    })
}

/// Split the arguments at the word `INTO` (any case): `(before, after)`.
fn split_into(args: &[Arg]) -> Result<(&[Arg], &[Arg])> {
    let at = args.iter().position(|a| matches!(a, Arg::Word(w) if w.eq_ignore_ascii_case("INTO")));
    match at {
        Some(i) => Ok((&args[..i], &args[i + 1..])),
        None => err("write the source columns, then INTO, then the output columns"),
    }
}

/// `STACK C1 C2 C3 INTO C10 [; SUBSCRIPTS C11.]` (a single block per source column), or
/// `STACK C1 C2 C3 C4 INTO C10 C11` to stack blocks of columns: the number of sources must be a
/// multiple of the number of outputs, and each output receives every k-th source in turn.
fn stack_cmd(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    use openstats_core::ops::stack::{stack_blocks, Subscripts};
    let sheet = m.sheet(env);
    let (src, dst) = split_into(&c.args)?;
    let srcs = columns(m, env, src)?;
    let dsts = columns(m, env, dst)?;
    if srcs.is_empty() || dsts.is_empty() || srcs.len() % dsts.len() != 0 {
        return err("STACK needs the number of source columns to be a multiple of the number of output columns");
    }
    let w = dsts.len();
    let blocks: Vec<Vec<usize>> = srcs.chunks(w).map(<[usize]>::to_vec).collect();
    let sub_col = match sub(c, "SUBSCRIPTS") {
        Some(s) => match s.args.first() {
            Some(a) => Some(one_col(m, env, a)?),
            None => return err("SUBSCRIPTS needs a column"),
        },
        None => None,
    };
    let subs = if sub_col.is_some() { Subscripts::Numbers } else { Subscripts::None };
    let names: Vec<String> = dsts
        .iter()
        .map(|&d| {
            env.wb.sheet(sheet).ok().and_then(|w| w.columns.get(d)).map_or(String::new(), |c| c.name().to_string())
        })
        .collect();
    let ws = env.wb.sheet(sheet).map_err(|e| ScriptError::runtime(0, e.to_string()))?.clone();
    let out = stack_blocks(&ws, &blocks, subs, &names).map_err(|e| ScriptError::runtime(0, e.to_string()))?;
    for (d, col) in dsts.iter().zip(out.iter()) {
        m.write_column(env, *d, col.data.clone(), None)?;
    }
    if let (Some(sc), Some(col)) = (sub_col, out.get(w)) {
        m.write_column(env, sc, col.data.clone(), None)?;
    }
    Ok(())
}

/// `UNSTACK C1 INTO C10 C11 C12; SUBSCRIPTS C2.`: one output column per level of the
/// subscripts column (for each data column, levels in value order).
fn unstack_cmd(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    use openstats_core::ops::unstack::{unstack, UnstackOptions};
    let sheet = m.sheet(env);
    let (src, dst) = split_into(&c.args)?;
    let data = columns(m, env, src)?;
    let outs = columns(m, env, dst)?;
    let by = match sub(c, "SUBSCRIPTS").and_then(|s| s.args.first()) {
        Some(a) => one_col(m, env, a)?,
        None => return err("UNSTACK needs ; SUBSCRIPTS followed by a column"),
    };
    let ws = env.wb.sheet(sheet).map_err(|e| ScriptError::runtime(0, e.to_string()))?.clone();
    let cols =
        unstack(&ws, &data, &[by], UnstackOptions::default()).map_err(|e| ScriptError::runtime(0, e.to_string()))?;
    if outs.len() < cols.len() {
        return err(format!("UNSTACK makes {} columns but only {} output columns were given", cols.len(), outs.len()));
    }
    for (o, col) in outs.iter().zip(cols.iter()) {
        m.write_column(env, *o, col.data.clone(), Some(col.name()))?;
    }
    Ok(())
}

// ----------------------------------------------------------------------------- data entry

fn datum_value(d: &Datum) -> Value {
    match d {
        Datum::Num(x) => Value::Numeric(*x),
        Datum::Text(s) => Value::Text(s.clone()),
        Datum::Missing => Value::Missing,
    }
}

fn build_column(values: &[Value]) -> ColumnData {
    if values.iter().any(|v| matches!(v, Value::Text(_))) {
        ColumnData::Text(
            values
                .iter()
                .map(|v| match v {
                    Value::Text(s) => s.clone(),
                    Value::Numeric(x) if x.is_finite() => x.to_string(),
                    _ => String::new(),
                })
                .collect(),
        )
    } else {
        ColumnData::Numeric(values.iter().map(|v| if let Value::Numeric(x) = v { *x } else { f64::NAN }).collect())
    }
}

fn set(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let cols = columns(m, env, &c.args)?;
    if cols.len() != 1 {
        return err("SET fills one column: SET C1, then the values, then END");
    }
    let values: Vec<Value> = c.data.iter().flat_map(|l| l.items.iter().map(datum_value)).collect();
    if values.is_empty() {
        return err("SET needs data lines before END");
    }
    m.write_column(env, cols[0], build_column(&values), None)
}

fn read(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let cols = columns(m, env, &c.args)?;
    if cols.is_empty() {
        return err("READ needs the columns to fill, as in READ C1-C3");
    }
    let mut data: Vec<Vec<Value>> = vec![Vec::new(); cols.len()];
    for l in &c.data {
        if l.items.len() != cols.len() {
            return Err(ScriptError::runtime(
                l.line,
                format!("this data line has {} values but READ fills {} columns", l.items.len(), cols.len()),
            ));
        }
        for (k, d) in l.items.iter().enumerate() {
            data[k].push(datum_value(d));
        }
    }
    if data[0].is_empty() {
        return err("READ needs data lines before END");
    }
    for (k, &col) in cols.iter().enumerate() {
        m.write_column(env, col, build_column(&data[k]), None)?;
    }
    Ok(())
}

fn name(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    if c.args.is_empty() || c.args.len() % 2 != 0 {
        return err("NAME takes pairs: NAME C1 'height' C2 'weight'");
    }
    let sheet = m.sheet(env);
    for pair in c.args.chunks(2) {
        let label = match &pair[1] {
            Arg::Quoted(s) | Arg::Text(s) | Arg::Word(s) => s.clone(),
            _ => return err("a column name must be text"),
        };
        match &pair[0] {
            Arg::Word(w) => match m.resolve(env, w)? {
                Resolved::Col(i) => {
                    let n = env.wb.sheet(sheet).map_or(0, |w| w.n_cols());
                    if i >= n {
                        m.write_column(env, i, ColumnData::Numeric(Vec::new()), None)?;
                    }
                    let e = stat_err(Edit::rename_column(env.wb, sheet, i, &label))?;
                    m.edit(env, "Script: name column", e)?;
                }
                _ => return err(format!("only columns can be named; `{w}` is not one")),
            },
            _ => return err("NAME needs a column such as C1 before each name"),
        }
    }
    Ok(())
}

fn erase(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let sheet = m.sheet(env);
    for a in &c.args {
        let list: Vec<&Arg> = match a {
            Arg::Range(..) => Vec::new(),
            _ => vec![a],
        };
        if let Arg::Range(..) = a {
            for col in columns(m, env, std::slice::from_ref(a))? {
                erase_col(m, env, col)?;
            }
            continue;
        }
        for a in list {
            let w = match a {
                Arg::Word(w) | Arg::Quoted(w) => w.clone(),
                _ => return err("ERASE takes columns, constants and matrices"),
            };
            let r = if matches!(a, Arg::Quoted(_)) { m.resolve_quoted(env, &w) } else { m.resolve(env, &w)? };
            match r {
                Resolved::Col(i) => erase_col(m, env, i)?,
                Resolved::Const(k) => {
                    m.edit(env, "Script: erase constant", Edit::SetConstant { sheet, k, value: None })?
                }
                Resolved::Mat(k) => {
                    m.edit(env, "Script: erase matrix", Edit::SetMatrix { sheet, m: k, matrix: None })?
                }
                Resolved::Str(n) => {
                    m.strings.remove(&n);
                }
                _ => return err(format!("`{w}` is not something ERASE can remove")),
            }
        }
    }
    Ok(())
}

fn erase_col(m: &mut Machine, env: &mut Env, i: usize) -> Result<()> {
    let sheet = m.sheet(env);
    if i < env.wb.sheet(sheet).map_or(0, |w| w.n_cols()) {
        m.edit(
            env,
            "Script: erase column",
            Edit::ReplaceColumn { sheet, col: i, column: Column::numeric("", Vec::new()) },
        )?;
    }
    Ok(())
}

// ----------------------------------------------------------------------------- copy, sort

fn use_rows(m: &Machine, env: &Env, c: &Command, n: usize) -> Result<Vec<usize>> {
    let Some(s) = sub(c, "USE") else { return Ok((0..n).collect()) };
    let mut rows = Vec::new();
    for a in &s.args {
        match a {
            Arg::Range(lo, hi) => {
                let (l, h) = (number(m, env, lo, "a row number")?, number(m, env, hi, "a row number")?);
                if l < 1.0 || h < l {
                    return err("USE needs row numbers like 3:10");
                }
                rows.extend((l as usize - 1)..(h as usize).min(n));
            }
            _ => {
                let r = number(m, env, a, "a row number")?;
                if r < 1.0 {
                    return err("row numbers start at 1");
                }
                if (r as usize) <= n {
                    rows.push(r as usize - 1);
                }
            }
        }
    }
    Ok(rows)
}

fn pick(data: &ColumnData, rows: &[usize]) -> ColumnData {
    match data {
        ColumnData::Numeric(v) => {
            ColumnData::Numeric(rows.iter().map(|&i| v.get(i).copied().unwrap_or(f64::NAN)).collect())
        }
        ColumnData::Text(v) => ColumnData::Text(rows.iter().map(|&i| v.get(i).cloned().unwrap_or_default()).collect()),
        ColumnData::DateTime(v) => {
            ColumnData::DateTime(rows.iter().map(|&i| v.get(i).copied().unwrap_or(i64::MIN)).collect())
        }
    }
}

fn copy(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let sheet = m.sheet(env);
    let mats: Vec<bool> = c.args.iter().map(|a| is_mat(m, env, a)).collect();
    if mats.iter().any(|&b| b) {
        return copy_matrix(m, env, c, &mats);
    }
    let all = columns(m, env, &c.args)?;
    if all.len() < 2 || all.len() % 2 != 0 {
        return err("COPY needs as many output columns as input columns, as in COPY C1 C2");
    }
    let k = all.len() / 2;
    let (ins, outs) = all.split_at(k);
    let n = ins
        .iter()
        .map(|&i| env.wb.sheet(sheet).ok().and_then(|w| w.columns.get(i)).map_or(0, Column::len))
        .max()
        .unwrap_or(0);
    let rows = use_rows(m, env, c, n)?;
    let mut copies = Vec::new();
    for (&i, &o) in ins.iter().zip(outs) {
        let src = env
            .wb
            .sheet(sheet)
            .ok()
            .and_then(|w| w.columns.get(i))
            .cloned()
            .ok_or_else(|| ScriptError::runtime(0, format!("column C{} does not exist", i + 1)))?;
        copies.push((o, pick(&src.data, &rows), src.name().to_string()));
    }
    for (o, data, nm) in copies {
        let keep = env.wb.sheet(sheet).ok().and_then(|w| w.columns.get(o)).map(|c| c.name().to_string());
        let _ = (keep, nm);
        m.write_column(env, o, data, None)?;
    }
    Ok(())
}

fn copy_matrix(m: &mut Machine, env: &mut Env, c: &Command, mats: &[bool]) -> Result<()> {
    let sheet = m.sheet(env);
    if c.args.len() < 2 {
        return err("COPY needs a source and a destination");
    }
    let last = c.args.len() - 1;
    if mats[last] && !mats[..last].iter().any(|&b| b) {
        let cols = columns(m, env, &c.args[..last])?;
        let target = matrix_target(m, env, &c.args[last])?;
        let data: Vec<Vec<f64>> = cols.iter().map(|&i| col_values(env, i)).collect::<Result<_>>()?;
        let rows = data.iter().map(Vec::len).max().unwrap_or(0);
        if rows == 0 {
            return err("the columns are empty");
        }
        let mut flat = vec![f64::NAN; rows * cols.len()];
        for (j, col) in data.iter().enumerate() {
            for (i, &x) in col.iter().enumerate() {
                flat[i * cols.len() + j] = x;
            }
        }
        let matrix = StoredMatrix { rows, cols: cols.len(), data: flat };
        return m.edit(env, "Script: copy to matrix", Edit::SetMatrix { sheet, m: target, matrix: Some(matrix) });
    }
    if mats[0] && !mats[1..].iter().any(|&b| b) {
        let src = matrix_target(m, env, &c.args[0])?;
        let mat = m.matrix(env, src)?;
        let outs = columns(m, env, &c.args[1..])?;
        if outs.len() != mat.cols {
            return err(format!("matrix M{src} has {} columns, so COPY needs {} output columns", mat.cols, mat.cols));
        }
        for (j, &o) in outs.iter().enumerate() {
            let v: Vec<f64> = (0..mat.rows).map(|i| mat.data[i * mat.cols + j]).collect();
            m.write_column(env, o, ColumnData::Numeric(v), None)?;
        }
        return Ok(());
    }
    if c.args.len() == 2 && mats[0] && mats[1] {
        let (a, b) = (matrix_target(m, env, &c.args[0])?, matrix_target(m, env, &c.args[1])?);
        let mat = m.matrix(env, a)?;
        return m.edit(env, "Script: copy matrix", Edit::SetMatrix { sheet, m: b, matrix: Some(mat) });
    }
    err("COPY moves columns to columns, columns to one matrix, or a matrix to columns")
}

fn sort(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let sheet = m.sheet(env);
    let all = columns(m, env, &c.args)?;
    if all.len() < 2 || all.len() % 2 != 0 {
        return err("SORT needs input columns followed by the same number of output columns");
    }
    let k = all.len() / 2;
    let (ins, outs) = all.split_at(k);
    let by_args = sub(c, "BY").map(|s| s.args.clone()).unwrap_or_default();
    let by = if by_args.is_empty() { vec![ins[0]] } else { columns(m, env, &by_args)? };
    let desc: Vec<usize> = match sub(c, "DESCENDING") {
        Some(s) => columns(m, env, &s.args)?,
        None => Vec::new(),
    };
    let cols: Vec<Column> = ins
        .iter()
        .map(|&i| {
            env.wb
                .sheet(sheet)
                .ok()
                .and_then(|w| w.columns.get(i))
                .cloned()
                .ok_or_else(|| ScriptError::runtime(0, format!("column C{} does not exist", i + 1)))
        })
        .collect::<Result<_>>()?;
    let n = cols.iter().map(Column::len).max().unwrap_or(0);
    let mut keys = Vec::new();
    for &b in &by {
        let Some(pos) = ins.iter().position(|&i| i == b) else {
            return err(format!("the BY column C{} must be one of the columns being sorted", b + 1));
        };
        keys.push((pos, desc.contains(&b)));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        for &(pos, d) in &keys {
            let (x, y) = (cols[pos].get(a), cols[pos].get(b));
            let ord = cmp_values(&x, &y, d);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
    for (col, &o) in cols.iter().zip(outs) {
        m.write_column(env, o, pick(&col.data, &order), None)?;
    }
    Ok(())
}

/// Missing values sort last in either direction.
fn cmp_values(a: &Value, b: &Value, desc: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let miss = |v: &Value| {
        matches!(v, Value::Missing)
            || matches!(v, Value::Numeric(x) if x.is_nan())
            || matches!(v, Value::Text(s) if s.is_empty())
    };
    match (miss(a), miss(b)) {
        (true, true) => return Equal,
        (true, false) => return Greater,
        (false, true) => return Less,
        _ => {}
    }
    let o = match (a, b) {
        (Value::Text(x), Value::Text(y)) => x.cmp(y),
        _ => a.as_f64().partial_cmp(&b.as_f64()).unwrap_or(Equal),
    };
    if desc {
        o.reverse()
    } else {
        o
    }
}

// ----------------------------------------------------------------------------- output

fn print(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let mut cols: Vec<usize> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    let flush = |m: &mut Machine, env: &Env, cols: &mut Vec<usize>| -> Result<()> {
        if cols.is_empty() {
            return Ok(());
        }
        let ws = env.wb.active_sheet().map_err(|e| ScriptError::runtime(0, e.to_string()))?;
        let n = cols.iter().map(|&i| ws.columns.get(i).map_or(0, Column::len)).max().unwrap_or(0);
        let heads: Vec<RCol> = cols
            .iter()
            .map(|&i| {
                let kind = if ws.columns.get(i).is_some_and(|k| matches!(k.data, ColumnData::Text(_))) {
                    ColumnKind::Text
                } else {
                    ColumnKind::Significant(6)
                };
                RCol::new(title(env, i), kind)
            })
            .collect();
        let mut t = Table::new("Data Display", heads);
        for r in 0..n {
            let row: Vec<Cell> = cols
                .iter()
                .map(|&i| match ws.columns.get(i).map(|k| k.get(r)) {
                    Some(Value::Numeric(x)) if x.is_nan() => Cell::Missing,
                    Some(Value::Numeric(x)) => Cell::Num(x),
                    Some(Value::Text(s)) if !s.is_empty() => Cell::Text(s),
                    Some(Value::DateTime(_)) => Cell::Text(ws.columns[i].display(r)),
                    _ => Cell::Missing,
                })
                .collect();
            stat_err(t.push_row(row))?;
        }
        let mut s = Section::new("Data Display");
        s.table(t);
        m.push_section(s);
        cols.clear();
        Ok(())
    };
    for a in &c.args {
        match a {
            Arg::Word(w) => match m.resolve(env, w)? {
                Resolved::Col(i) => cols.push(i),
                Resolved::Const(k) => {
                    flush(m, env, &mut cols)?;
                    let v = env.wb.active_sheet().ok().and_then(|w| w.constants.get(&k)).cloned();
                    lines.push(match v {
                        Some(Value::Numeric(x)) => format!("{w}  {}", fmt(x)),
                        Some(Value::Text(s)) => format!("{w}  {s}"),
                        _ => format!("{w}  *"),
                    });
                }
                Resolved::Mat(k) => {
                    flush(m, env, &mut cols)?;
                    let mat = m.matrix(env, k)?;
                    let heads: Vec<RCol> =
                        (0..mat.cols).map(|j| RCol::new(format!("C{}", j + 1), ColumnKind::Significant(6))).collect();
                    let mut t = Table::new(format!("Matrix {w}"), heads);
                    for i in 0..mat.rows {
                        stat_err(t.push_row((0..mat.cols).map(|j| Cell::Num(mat.data[i * mat.cols + j])).collect()))?;
                    }
                    let mut s = Section::new("Data Display");
                    s.table(t);
                    m.push_section(s);
                }
                Resolved::Str(n) => {
                    flush(m, env, &mut cols)?;
                    lines.push(format!("{w}  {}", m.strings.get(&n).cloned().unwrap_or_default()));
                }
                Resolved::Lit(v) => lines.push(match v {
                    Value::Numeric(x) => fmt(x),
                    Value::Text(s) => s,
                    _ => "*".into(),
                }),
                Resolved::Unknown(n) => return err(format!("`{n}` is not a column, constant or matrix")),
            },
            Arg::Quoted(q) => match m.resolve_quoted(env, q) {
                Resolved::Col(i) => cols.push(i),
                _ => return err(format!("there is no column named '{q}'")),
            },
            Arg::Range(..) => cols.extend(columns(m, env, std::slice::from_ref(a))?),
            Arg::Num(x) => lines.push(fmt(*x)),
            Arg::Text(s) => lines.push(s.clone()),
            _ => lines.push(flatten_name(a)),
        }
    }
    flush(m, env, &mut cols)?;
    if !lines.is_empty() {
        let mut s = Section::new("Data Display");
        for l in lines {
            s.text(l);
        }
        m.push_section(s);
    }
    Ok(())
}

fn fmt(x: f64) -> String {
    if x.is_nan() {
        "*".into()
    } else if x == x.trunc() && x.abs() < 1e15 {
        format!("{x:.0}")
    } else {
        format!("{x:.6}").trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

// ----------------------------------------------------------------------------- statistics

fn column_stat(m: &mut Machine, env: &mut Env, c: &Command, word: &str) -> Result<()> {
    if c.args.is_empty() || c.args.len() > 2 {
        return err(format!("{word} takes a column and, optionally, a constant to store the result in"));
    }
    let col = one_col(m, env, &c.args[0])?;
    let data = col_values(env, col)?;
    let d = stat_err(describe(&data))?;
    let value = match word {
        "MEAN" => d.mean,
        "SUM" => d.sum,
        "STDEV" => d.std_dev,
        "MEDIAN" => d.median,
        "MINIMUM" => d.minimum,
        "MAXIMUM" => d.maximum,
        "RANGE" => d.range,
        "SSQ" => d.sum_of_squares,
        "N" | "COUNT" => Some(d.n as f64),
        _ => Some(d.n_missing as f64),
    };
    let Some(x) = value else {
        return err(format!("{word} of {} cannot be computed from {} value(s)", title(env, col), d.n));
    };
    let label = word.to_ascii_lowercase();
    match c.args.get(1) {
        Some(a) => {
            let k = const_target(m, env, a)?;
            m.set_const(env, k, Value::Numeric(x))
        }
        None => {
            section_text(m, "", &[format!("{} of {} = {}", capitalize(&label), title(env, col), fmt(x))]);
            Ok(())
        }
    }
}

fn capitalize(s: &str) -> String {
    let mut ch = s.chars();
    ch.next().map_or_else(String::new, |f| f.to_uppercase().collect::<String>() + ch.as_str())
}

fn describe_cmd(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let cols = columns(m, env, &c.args)?;
    if cols.is_empty() {
        return err("DESCRIBE needs columns, as in DESCRIBE C1-C3");
    }
    for col in cols {
        let data = col_values(env, col)?;
        let d = stat_err(describe(&data))?;
        let mut s = d.report_section();
        s.heading = format!("Descriptive Statistics: {}", title(env, col));
        m.push_section(s);
    }
    Ok(())
}

fn correlation(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let cols = columns(m, env, &c.args)?;
    if cols.len() < 2 {
        return err("CORRELATION needs at least two columns");
    }
    let kind = if sub(c, "SPEARMAN").is_some() {
        1
    } else if sub(c, "KENDALL").is_some() {
        2
    } else {
        0
    };
    let level = match sub(c, "CONFIDENCE").and_then(|s| s.args.first()) {
        Some(a) => number(m, env, a, "the confidence level")? / 100.0,
        None => 0.95,
    };
    for i in 0..cols.len() {
        for j in (i + 1)..cols.len() {
            let (x, y) = (col_values(env, cols[i])?, col_values(env, cols[j])?);
            let mut s = match kind {
                1 => stat_err(spearman(&x, &y))?.report_section(),
                2 => stat_err(kendall(&x, &y))?.report_section(),
                _ => stat_err(pearson(&x, &y, level))?.report_section(),
            };
            let label = ["Correlation", "Spearman Rho", "Kendall Tau"][kind];
            s.heading = format!("{label}: {} versus {}", title(env, cols[i]), title(env, cols[j]));
            m.push_section(s);
        }
    }
    Ok(())
}

fn regress(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    // Classic: REGRESS C1 k C2 … Ck   or   REGRESS C1 C2-C5.
    // Modern:  REGRESS;  RESPONSE C1;  CONTINUOUS C2 C3.   (also RESPONSES / PREDICTORS)
    let named = |names: &[&str]| c.subs.iter().find(|s| names.iter().any(|n| s.name.eq_ignore_ascii_case(n)));
    let (y, preds) = if c.args.is_empty() {
        let Some(resp) = named(&["RESPONSE", "RESPONSES"]) else {
            return err("REGRESS needs a response and predictors");
        };
        let Some(first) = resp.args.first() else { return err("the RESPONSE subcommand needs a column") };
        let y = one_col(m, env, first)?;
        let Some(cont) = named(&["CONTINUOUS", "PREDICTORS", "PREDICTOR"]) else {
            return err("REGRESS needs predictors: add a CONTINUOUS subcommand");
        };
        (y, columns(m, env, &cont.args)?)
    } else {
        let y = one_col(m, env, &c.args[0])?;
        let mut rest = &c.args[1..];
        if let Some(Arg::Num(k)) = rest.first() {
            let k = *k as usize;
            rest = &rest[1..];
            let got = columns(m, env, rest)?.len();
            if got != k {
                return err(format!("REGRESS says {k} predictor(s) but lists {got}"));
            }
        }
        (y, columns(m, env, rest)?)
    };
    if preds.is_empty() {
        return err("REGRESS needs at least one predictor");
    }
    if preds.contains(&y) {
        return err("the response cannot also be a predictor");
    }
    let ycol = col_values(env, y)?;
    let xcols: Vec<Vec<f64>> = preds.iter().map(|&p| col_values(env, p)).collect::<Result<_>>()?;
    let mut all: Vec<&[f64]> = vec![&ycol];
    all.extend(xcols.iter().map(Vec::as_slice));
    let (rows, data) = complete(&all);
    let names: Vec<String> = preds.iter().map(|&p| title(env, p)).collect();
    let cols: Vec<(&str, &[f64])> = names.iter().zip(&data[1..]).map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    let level = match sub(c, "CONFIDENCE").and_then(|s| s.args.first()) {
        Some(a) => number(m, env, a, "the confidence level")? / 100.0,
        None => 0.95,
    };
    let o = RegressionOptions { level, ..Default::default() };
    let fit = stat_err(fit_xy(&cols, &data[0], sub(c, "NOCONSTANT").is_none(), &o))?;
    let mut s = fit.report_section();
    s.heading = format!("Regression Analysis: {} versus {}", title(env, y), names.join(", "));
    s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    m.push_section(s);
    let n = ycol.len();
    put_numeric(m, env, sub(c, "FITS"), y, n, &rows, &fit.diagnostics.fitted)?;
    put_numeric(m, env, sub(c, "RESIDUALS"), y, n, &rows, &fit.diagnostics.residuals)?;
    put_numeric(m, env, sub(c, "SRESIDUALS"), y, n, &rows, &fit.diagnostics.standardized)?;
    if let Some(a) = sub(c, "COEFFICIENTS").and_then(|s| s.args.first()) {
        let idx = one_col(m, env, a)?;
        let v: Vec<f64> = fit.coefficients.iter().map(|k| k.estimate).collect();
        m.write_column(env, idx, ColumnData::Numeric(v), None)?;
    }
    if let Some(a) = sub(c, "MSE").and_then(|s| s.args.first()) {
        let k = const_target(m, env, a)?;
        m.set_const(env, k, Value::Numeric(fit.mse))?;
    }
    if let Some(a) = sub(c, "RSQUARED").and_then(|s| s.args.first()) {
        let k = const_target(m, env, a)?;
        m.set_const(env, k, Value::Numeric(fit.r2))?;
    }
    Ok(())
}

fn ttest(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let (mu, rest) = match c.args.first() {
        Some(Arg::Num(x)) => (*x, &c.args[1..]),
        Some(Arg::Paren(e)) => (m.eval_number(env, e, "the test mean")?, &c.args[1..]),
        _ => (0.0, &c.args[..]),
    };
    let cols = columns(m, env, rest)?;
    if cols.len() != 1 {
        return err("TTEST takes a test mean and one column: TTEST 5 C1");
    }
    let alt = match sub(c, "ALTERNATIVE").and_then(|s| s.args.first()) {
        Some(a) => match number(m, env, a, "the alternative")? as i64 {
            -1 => Alternative::Less,
            1 => Alternative::Greater,
            0 => Alternative::TwoSided,
            _ => return err("ALTERNATIVE is -1, 0 or 1"),
        },
        None => Alternative::TwoSided,
    };
    let level = match sub(c, "CONFIDENCE").and_then(|s| s.args.first()) {
        Some(a) => number(m, env, a, "the confidence level")? / 100.0,
        None => 0.95,
    };
    let data = col_values(env, cols[0])?;
    let t = stat_err(one_sample_t(&data, mu, alt, level))?;
    let mut s = t.report_section();
    s.heading = format!("One-Sample T: {}", title(env, cols[0]));
    m.push_section(s);
    Ok(())
}

fn oneway(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    let cols = columns(m, env, &c.args)?;
    if cols.len() != 2 {
        return err("ONEWAY takes a response column and a factor column: ONEWAY C1 C2");
    }
    let (y, f) = (col_values(env, cols[0])?, col_labels(env, cols[1])?);
    let level = match sub(c, "CONFIDENCE").and_then(|s| s.args.first()) {
        Some(a) => number(m, env, a, "the confidence level")? / 100.0,
        None => 0.95,
    };
    let a = stat_err(one_way_stacked(&y, &f, &OneWayOptions { level }))?;
    let mut s = a.report_section();
    s.heading = format!("One-Way ANOVA: {} versus {}", title(env, cols[0]), title(env, cols[1]));
    m.push_section(s);
    Ok(())
}

// ----------------------------------------------------------------------------- predictive models

fn model(m: &mut Machine, env: &mut Env, c: &Command, word: &str) -> Result<()> {
    let Some(first) = c.args.first() else { return err(format!("{word} needs a response and predictors")) };
    let y = one_col(m, env, first)?;
    let preds = columns(m, env, &c.args[1..])?;
    if preds.is_empty() {
        return err(format!("{word} needs at least one predictor"));
    }
    if preds.contains(&y) {
        return err("the response cannot also be a predictor");
    }
    let class = if sub(c, "CLASSIFICATION").is_some() {
        true
    } else if sub(c, "REGRESSION").is_some() {
        false
    } else {
        is_text(env, y)
    };
    let opt = |m: &Machine, name: &str| -> Result<Option<f64>> {
        match sub(c, name).and_then(|s| s.args.first()) {
            Some(a) => Ok(Some(number(m, env, a, &name.to_ascii_lowercase())?)),
            None => Ok(None),
        }
    };
    let n_total = env.wb.active_sheet().ok().and_then(|w| w.columns.get(y)).map_or(0, Column::len);
    let (rows, resp) = if class {
        let all = col_labels(env, y)?;
        let rows: Vec<usize> = (0..all.len()).filter(|&i| !all[i].is_empty()).collect();
        let kept: Vec<String> = rows.iter().map(|&i| all[i].clone()).collect();
        (rows, stat_err(Response::from_labels(&kept))?)
    } else {
        let all = col_values(env, y)?;
        let rows: Vec<usize> = (0..all.len()).filter(|&i| !all[i].is_nan()).collect();
        let kept: Vec<f64> = rows.iter().map(|&i| all[i]).collect();
        (rows, stat_err(Response::numeric(kept))?)
    };
    if rows.len() < 4 {
        return err(format!("{} has only {} usable values; at least 4 are needed", title(env, y), rows.len()));
    }
    let mut ds = Dataset::new();
    let mut names = Vec::new();
    for &p in &preds {
        let nm = title(env, p);
        if is_text(env, p) {
            let all = col_labels(env, p)?;
            let vals: Vec<String> = rows.iter().map(|&i| all.get(i).cloned().unwrap_or_default()).collect();
            ds = stat_err(ds.with_categorical(&nm, &vals))?;
        } else {
            let all = col_values(env, p)?;
            let vals: Vec<f64> = rows.iter().map(|&i| all.get(i).copied().unwrap_or(f64::NAN)).collect();
            ds = stat_err(ds.with_numeric(&nm, vals))?;
        }
        names.push(nm);
    }
    let seed = opt(m, "SEED")?.map_or(1, |s| s.max(0.0) as u64);
    let crit = class.then(|| if sub(c, "ENTROPY").is_some() { Criterion::Entropy } else { Criterion::Gini });
    let kind;
    let mut stored: Vec<Column> = Vec::new();
    let heading_of = |k: &str| format!("{k}: {} versus {}", title(env, y), names.join(", "));
    let section = match word {
        "CART" => {
            let mut o = CartOptions { criterion: crit, seed, ..Default::default() };
            if let Some(d) = opt(m, "DEPTH")? {
                o.max_depth = (d as usize).max(1);
            }
            if let Some(l) = opt(m, "LEAF")? {
                o.min_leaf = (l as usize).max(1);
            }
            if let Some(f) = opt(m, "FOLDS")? {
                o.cv_folds = f as usize;
            }
            let model = stat_err(fit_cart(&ds, &resp, &o))?;
            kind = if class { "Classification Tree" } else { "Regression Tree" };
            predictions(&mut stored, &resp, &model.levels, &model.fitted, n_total, &rows, "");
            if sub(c, "NODE").is_some() {
                let node: Vec<f64> = model.terminal_node.iter().map(|&k| k as f64).collect();
                stored.push(spread("NODE", n_total, &rows, &node));
            }
            model.report_section()
        }
        "FOREST" => {
            let mut o = ForestOptions { criterion: crit, seed, ..Default::default() };
            if let Some(t) = opt(m, "TREES")? {
                o.n_trees = (t as usize).max(1);
            }
            if let Some(d) = opt(m, "DEPTH")? {
                o.max_depth = (d as usize).max(1);
            }
            if let Some(l) = opt(m, "LEAF")? {
                o.min_leaf = Some((l as usize).max(1));
            }
            let model = stat_err(fit_forest(&ds, &resp, &o))?;
            kind = if class { "Random Forest Classification" } else { "Random Forest Regression" };
            predictions(&mut stored, &resp, &model.levels, &model.oob, n_total, &rows, "OOB_");
            model.report_section()
        }
        _ => {
            let mut o = GbdtOptions {
                loss: Some(if class { GbdtLoss::Deviance } else { GbdtLoss::SquaredError }),
                seed,
                ..Default::default()
            };
            if let Some(t) = opt(m, "TREES")? {
                o.n_trees = (t as usize).max(1);
            }
            if let Some(r) = opt(m, "RATE")? {
                o.learning_rate = r;
            }
            if let Some(d) = opt(m, "DEPTH")? {
                o.max_depth = (d as usize).max(1);
            }
            let model = stat_err(fit_gbdt(&ds, &resp, &o))?;
            kind = if class { "Gradient Boosting Classification" } else { "Gradient Boosting Regression" };
            predictions(&mut stored, &resp, &model.levels, &model.fitted, n_total, &rows, "");
            model.report_section()
        }
    };
    let mut section = section;
    section.heading = heading_of(kind);
    section.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    m.push_section(section);
    // FITS / RESIDUALS / PROBABILITIES … name the columns that receive the stored results.
    for (sname, wanted) in [("FITS", ["FITS", "PRED"]), ("RESIDUALS", ["RESI", "RESI"]), ("NODE", ["NODE", "NODE"])] {
        if let Some(a) = sub(c, sname).and_then(|s| s.args.first()) {
            let target = one_col(m, env, a)?;
            let hit = stored.iter().find(|k| {
                let n = k.name().trim_start_matches("OOB_");
                n == wanted[0] || n == wanted[1]
            });
            match hit {
                Some(col) => m.write_column(env, target, col.data.clone(), None)?,
                None => return err(format!("{word} has nothing to store for {sname}")),
            }
        }
    }
    Ok(())
}

fn predictions(
    out: &mut Vec<Column>,
    resp: &Response,
    levels: &[String],
    pred: &Predictions,
    n: usize,
    rows: &[usize],
    prefix: &str,
) {
    match resp {
        Response::Numeric(y) => {
            let fitted: Vec<f64> = pred.values.clone();
            out.push(spread(&format!("{prefix}FITS"), n, rows, &fitted));
            let r: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
            out.push(spread(&format!("{prefix}RESI"), n, rows, &r));
        }
        Response::Class { .. } => {
            let mut v = vec![String::new(); n];
            for (&r, &k) in rows.iter().zip(&pred.classes) {
                v[r] = levels[k].clone();
            }
            out.push(Column::text(format!("{prefix}PRED"), v));
        }
    }
}

// ----------------------------------------------------------------------------- matrices

fn transpose(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    if c.args.len() != 2 {
        return err("TRANSPOSE takes two matrices: TRANSPOSE M1 M2");
    }
    let (a, b) = (matrix_target(m, env, &c.args[0])?, matrix_target(m, env, &c.args[1])?);
    let x = m.matrix(env, a)?;
    let mut d = vec![0.0; x.data.len()];
    for i in 0..x.rows {
        for j in 0..x.cols {
            d[j * x.rows + i] = x.data[i * x.cols + j];
        }
    }
    let sheet = m.sheet(env);
    m.edit(
        env,
        "Script: transpose",
        Edit::SetMatrix { sheet, m: b, matrix: Some(StoredMatrix { rows: x.cols, cols: x.rows, data: d }) },
    )
}

fn multiply(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    if c.args.len() != 3 {
        return err("MULTIPLY takes three matrices: MULTIPLY M1 M2 M3");
    }
    let (a, b, o) =
        (matrix_target(m, env, &c.args[0])?, matrix_target(m, env, &c.args[1])?, matrix_target(m, env, &c.args[2])?);
    let (x, y) = (m.matrix(env, a)?, m.matrix(env, b)?);
    if x.cols != y.rows {
        return err(format!(
            "M{a} is {} by {} and M{b} is {} by {}; they cannot be multiplied",
            x.rows, x.cols, y.rows, y.cols
        ));
    }
    let mut d = vec![0.0; x.rows * y.cols];
    for i in 0..x.rows {
        for j in 0..y.cols {
            d[i * y.cols + j] = (0..x.cols).map(|k| x.data[i * x.cols + k] * y.data[k * y.cols + j]).sum();
        }
    }
    let sheet = m.sheet(env);
    m.edit(
        env,
        "Script: multiply",
        Edit::SetMatrix { sheet, m: o, matrix: Some(StoredMatrix { rows: x.rows, cols: y.cols, data: d }) },
    )
}

fn invert(m: &mut Machine, env: &mut Env, c: &Command) -> Result<()> {
    if c.args.len() != 2 {
        return err("INVERT takes two matrices: INVERT M1 M2");
    }
    let (a, o) = (matrix_target(m, env, &c.args[0])?, matrix_target(m, env, &c.args[1])?);
    let x = m.matrix(env, a)?;
    if x.rows != x.cols {
        return err(format!("M{a} is {} by {}; only square matrices can be inverted", x.rows, x.cols));
    }
    let n = x.rows;
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut r: Vec<f64> = x.data[i * n..(i + 1) * n].to_vec();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        let p = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(col);
        if aug[p][col].abs() < 1e-12 {
            return err(format!("M{a} is singular and cannot be inverted"));
        }
        aug.swap(col, p);
        let piv = aug[col][col];
        for v in aug[col].iter_mut() {
            *v /= piv;
        }
        for r in 0..n {
            if r != col {
                let f = aug[r][col];
                if f != 0.0 {
                    let pr = aug[col].clone();
                    for (v, p) in aug[r].iter_mut().zip(&pr) {
                        *v -= f * p;
                    }
                }
            }
        }
    }
    let d: Vec<f64> = aug.iter().flat_map(|r| r[n..].iter().copied()).collect();
    let sheet = m.sheet(env);
    m.edit(
        env,
        "Script: invert",
        Edit::SetMatrix { sheet, m: o, matrix: Some(StoredMatrix { rows: n, cols: n, data: d }) },
    )
}
