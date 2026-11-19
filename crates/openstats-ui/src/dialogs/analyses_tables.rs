//! Tables, exploratory data analysis, bootstrap, simulation and sampling dialogs (Phase 5l).

use super::analyses::add;
use super::analysis::{col_title, labels, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use crate::grid::GridController;
use openstats_core::column::Column;
use openstats_core::{Workbook, Worksheet};
use openstats_stats::bootstrap::{bootstrap, BootstrapOptions, IntervalMethod};
use openstats_stats::descriptive::{describe, trimmed_mean};
use openstats_stats::eda::{letter_values, median_polish, resistant_line, rootogram, symmetry_points};
use openstats_stats::graphical::{histogram, Binning};
use openstats_stats::quantile::{quantile, QuantileMethod};
use openstats_stats::report::{
    Axis, Cell, ChartKind, ChartSpec, Column as TCol, ColumnKind as TKind, ReferenceLine, Section, Series, Table,
};

fn chart(kind: ChartKind, title: String, xl: &str, yl: &str, cats: Vec<String>, series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind,
        title,
        x_label: xl.into(),
        y_label: yl.into(),
        categories: cats,
        series,
        reference_lines: Vec::new(),
    }
}

fn hist_chart(data: &[f64], title: String, xl: &str) -> Res<ChartSpec> {
    let h = histogram(data, &Binning::Sturges)?;
    Ok(chart(
        ChartKind::Histogram,
        title,
        xl,
        "Frequency",
        Vec::new(),
        vec![Series { name: xl.into(), x: h.edges.clone(), y: h.counts.iter().map(|&c| c as f64).collect() }],
    ))
}

fn out(section: Section, stored: Vec<Column>) -> Res<Output> {
    Ok(Output { section, stored, models: Vec::new() })
}

fn tcols(heads: &[(&str, TKind)]) -> Vec<TCol> {
    heads.iter().map(|(h, k)| TCol::new(*h, *k)).collect()
}

/// Distinct non-empty labels of a column in their natural order (numeric order for numbers).
pub(super) fn levels(ws: &Worksheet, c: usize) -> Res<Vec<String>> {
    let lab = labels(ws, c)?;
    let numeric_col = !matches!(ws.column(c)?.data, openstats_core::column::ColumnData::Text(_));
    let nums = if numeric_col { numeric(ws, c).ok() } else { None };
    let mut seen: Vec<(String, f64)> = Vec::new();
    for (i, l) in lab.iter().enumerate() {
        if l.is_empty() || seen.iter().any(|(s, _)| s == l) {
            continue;
        }
        seen.push((l.clone(), nums.as_ref().map_or(0.0, |n| n[i])));
    }
    if nums.is_some() {
        seen.sort_by(|a, b| a.1.total_cmp(&b.1));
    } else {
        seen.sort_by(|a, b| a.0.cmp(&b.0));
    }
    Ok(seen.into_iter().map(|s| s.0).collect())
}

// ------------------------------------------------------------------------- tally

fn tally_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (show_c, show_p, show_cc, show_cp) =
        (v.flag("counts"), v.flag("percents"), v.flag("cum_counts"), v.flag("cum_percents"));
    if !(show_c || show_p || show_cc || show_cp) {
        return Err("choose at least one statistic to display".into());
    }
    let mut s = Section::new("Tally for Discrete Variables");
    for &c in v.vars("vars") {
        let lab = labels(ws, c)?;
        let lv = levels(ws, c)?;
        let n = lab.iter().filter(|l| !l.is_empty()).count();
        let miss = lab.len() - n;
        let mut heads = vec![(col_title(ws, c), TKind::Text)];
        if show_c {
            heads.push(("Count".into(), TKind::Integer));
        }
        if show_p {
            heads.push(("Percent".into(), TKind::Fixed(2)));
        }
        if show_cc {
            heads.push(("CumCnt".into(), TKind::Integer));
        }
        if show_cp {
            heads.push(("CumPct".into(), TKind::Fixed(2)));
        }
        let mut t = Table::new(
            format!("Tally for {}", col_title(ws, c)),
            heads.iter().map(|(h, k)| TCol::new(h.as_str(), *k)).collect(),
        );
        let mut cum = 0usize;
        for l in &lv {
            let cnt = lab.iter().filter(|x| *x == l).count();
            cum += cnt;
            let mut row = vec![Cell::text(l.clone())];
            if show_c {
                row.push(Cell::Int(cnt as i64));
            }
            if show_p {
                row.push(Cell::Num(100.0 * cnt as f64 / n as f64));
            }
            if show_cc {
                row.push(Cell::Int(cum as i64));
            }
            if show_cp {
                row.push(Cell::Num(100.0 * cum as f64 / n as f64));
            }
            t.push_row(row)?;
        }
        let mut total = vec![Cell::text("N")];
        if show_c {
            total.push(Cell::Int(n as i64));
        }
        total.resize(heads.len(), Cell::Missing);
        t.push_row(total)?;
        s.table(t);
        if miss > 0 {
            s.text(format!("{}: {miss} missing values are not tallied.", col_title(ws, c)));
        }
    }
    out(s, Vec::new())
}

// ------------------------------------------------------------------------- descriptive table

const STATS: [(&str, &str); 7] = [
    ("stat_n", "N"),
    ("stat_mean", "Mean"),
    ("stat_median", "Median"),
    ("stat_sd", "StDev"),
    ("stat_min", "Minimum"),
    ("stat_max", "Maximum"),
    ("stat_sum", "Sum"),
];

fn cell_stat(name: &str, x: &[f64]) -> Cell {
    let ok: Vec<f64> = x.iter().copied().filter(|a| !a.is_nan()).collect();
    if ok.is_empty() {
        return if name == "N" { Cell::Int(0) } else { Cell::Missing };
    }
    let n = ok.len() as f64;
    let mean = ok.iter().sum::<f64>() / n;
    match name {
        "N" => Cell::Int(ok.len() as i64),
        "Mean" => Cell::Num(mean),
        "Median" => quantile(&ok, 0.5, QuantileMethod::Type6).map_or(Cell::Missing, Cell::Num),
        "StDev" => {
            if ok.len() < 2 {
                Cell::Missing
            } else {
                Cell::Num((ok.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt())
            }
        }
        "Minimum" => Cell::Num(ok.iter().copied().fold(f64::INFINITY, f64::min)),
        "Maximum" => Cell::Num(ok.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        _ => Cell::Num(ok.iter().sum()),
    }
}

fn cell_table_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let rows_v = v.vars("rows");
    let col_v = v.var("cols");
    let assoc = v.vars("assoc");
    let chosen: Vec<&str> = STATS.iter().filter(|(k, _)| v.flag(k)).map(|(_, l)| *l).collect();
    if !assoc.is_empty() && chosen.is_empty() {
        return Err("choose at least one statistic for the summary variables".into());
    }
    let row_labels: Vec<Vec<String>> = rows_v.iter().map(|&c| labels(ws, c)).collect::<Res<_>>()?;
    let col_labels: Option<Vec<String>> = col_v.map(|c| labels(ws, c)).transpose()?;
    let n = row_labels[0].len();
    let keep: Vec<usize> = (0..n)
        .filter(|&i| {
            row_labels.iter().all(|l| !l[i].is_empty()) && col_labels.as_ref().is_none_or(|l| !l[i].is_empty())
        })
        .collect();
    let dropped = n - keep.len();
    // Row combinations in sorted level order.
    let lv: Vec<Vec<String>> = rows_v.iter().map(|&c| levels(ws, c)).collect::<Res<_>>()?;
    let mut combos: Vec<Vec<String>> = vec![Vec::new()];
    for l in &lv {
        combos =
            combos.into_iter().flat_map(|p| l.iter().map(move |x| [p.clone(), vec![x.clone()]].concat())).collect();
    }
    combos.retain(|c| keep.iter().any(|&i| (0..c.len()).all(|k| row_labels[k][i] == c[k])));
    let col_levels: Vec<Option<String>> = match col_v {
        Some(c) => levels(ws, c)?.into_iter().map(Some).chain(std::iter::once(None)).collect(),
        None => vec![None],
    };
    let data: Vec<Vec<f64>> = assoc.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let mut heads: Vec<(String, TKind)> = rows_v.iter().map(|&c| (col_title(ws, c), TKind::Text)).collect();
    if let Some(c) = col_v {
        heads.push((col_title(ws, c), TKind::Text));
    }
    heads.push(("Count".into(), TKind::Integer));
    for &c in assoc {
        for s in &chosen {
            heads.push((
                format!("{} {}", col_title(ws, c), s),
                if *s == "N" { TKind::Integer } else { TKind::Significant(6) },
            ));
        }
    }
    let mut t = Table::new("Descriptive Statistics", heads.iter().map(|(h, k)| TCol::new(h.as_str(), *k)).collect());
    let mut emit = |rowkey: Option<&Vec<String>>, ck: &Option<String>| -> Res<()> {
        let sel: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&i| {
                rowkey.is_none_or(|r| (0..r.len()).all(|k| row_labels[k][i] == r[k]))
                    && match (ck, &col_labels) {
                        (Some(c), Some(l)) => &l[i] == c,
                        _ => true,
                    }
            })
            .collect();
        let mut row: Vec<Cell> =
            (0..rows_v.len()).map(|k| Cell::text(rowkey.map_or_else(|| "All".to_string(), |r| r[k].clone()))).collect();
        if col_v.is_some() {
            row.push(Cell::text(ck.clone().unwrap_or_else(|| "All".into())));
        }
        row.push(Cell::Int(sel.len() as i64));
        for d in &data {
            let x: Vec<f64> = sel.iter().map(|&i| d[i]).collect();
            for s in &chosen {
                row.push(cell_stat(s, &x));
            }
        }
        t.push_row(row)?;
        Ok(())
    };
    for combo in &combos {
        for ck in &col_levels {
            emit(Some(combo), ck)?;
        }
    }
    for ck in &col_levels {
        emit(None, ck)?;
    }
    let mut s = Section::new("Descriptive Statistics Table");
    s.table(t);
    if dropped > 0 {
        s.text(format!("{dropped} rows with a missing classification were excluded."));
    }
    out(s, Vec::new())
}

// ------------------------------------------------------------------------- Pareto

fn pareto_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let defect = v.var("defects").ok_or("choose the defects column")?;
    let lab = labels(ws, defect)?;
    let (cats, counts): (Vec<String>, Vec<f64>) = match v.var("counts") {
        Some(c) => {
            let cnt = numeric(ws, c)?;
            let mut agg: Vec<(String, f64)> = Vec::new();
            for (l, x) in lab.iter().zip(&cnt) {
                if l.is_empty() || x.is_nan() {
                    continue;
                }
                if x < &0.0 {
                    return Err("counts must not be negative".into());
                }
                match agg.iter_mut().find(|a| &a.0 == l) {
                    Some(a) => a.1 += x,
                    None => agg.push((l.clone(), *x)),
                }
            }
            agg.into_iter().unzip()
        }
        None => levels(ws, defect)?
            .into_iter()
            .map(|l| {
                let n = lab.iter().filter(|x| **x == l).count() as f64;
                (l, n)
            })
            .unzip(),
    };
    let total: f64 = counts.iter().sum();
    if cats.is_empty() || total <= 0.0 {
        return Err("there are no defects to chart".into());
    }
    let mut order: Vec<usize> = (0..cats.len()).collect();
    order.sort_by(|&a, &b| counts[b].total_cmp(&counts[a]));
    let mut t = Table::new(
        "Pareto Table",
        tcols(&[
            ("Defect", TKind::Text),
            ("Count", TKind::Significant(6)),
            ("Percent", TKind::Fixed(2)),
            ("Cum %", TKind::Fixed(2)),
        ]),
    );
    let mut cum = 0.0;
    for &i in &order {
        cum += counts[i];
        t.push_row(vec![
            Cell::text(cats[i].clone()),
            Cell::Num(counts[i]),
            Cell::Num(100.0 * counts[i] / total),
            Cell::Num(100.0 * cum / total),
        ])?;
    }
    let title = format!("Pareto Chart of {}", col_title(ws, defect));
    let mut s = Section::new("Pareto Chart");
    s.table(t);
    s.chart(chart(
        ChartKind::Pareto,
        title,
        &col_title(ws, defect),
        "Count",
        cats.clone(),
        vec![Series { name: "Count".into(), x: Vec::new(), y: counts }],
    ));
    out(s, Vec::new())
}

// ------------------------------------------------------------------------- EDA

fn letter_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut s = Section::new("Letter Values");
    for &c in v.vars("vars") {
        let x = numeric(ws, c)?;
        let lv = letter_values(&x)?;
        let n = x.iter().filter(|a| !a.is_nan()).count();
        let mut t = Table::new(
            format!("Letter Values for {} (N = {n})", col_title(ws, c)),
            tcols(&[
                ("Letter", TKind::Text),
                ("Depth", TKind::Significant(6)),
                ("Lower", TKind::Significant(6)),
                ("Upper", TKind::Significant(6)),
                ("Mid", TKind::Significant(6)),
                ("Spread", TKind::Significant(6)),
            ]),
        );
        for l in &lv {
            let med = l.label == "M";
            t.push_row(vec![
                Cell::text(l.label.clone()),
                Cell::Num(l.depth),
                if med { Cell::Missing } else { Cell::Num(l.lower) },
                if med { Cell::Missing } else { Cell::Num(l.upper) },
                Cell::Num(l.mid),
                if med { Cell::Missing } else { Cell::Num(l.spread) },
            ])?;
        }
        s.table(t);
    }
    out(s, Vec::new())
}

fn polish_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = numeric(ws, v.var("y").ok_or("choose the response column")?)?;
    let (rc, cc) = (v.var("row").ok_or("choose the row factor")?, v.var("col").ok_or("choose the column factor")?);
    let (rl, cl) = (labels(ws, rc)?, labels(ws, cc)?);
    let (rlv, clv) = (levels(ws, rc)?, levels(ws, cc)?);
    let (nr, nc) = (rlv.len(), clv.len());
    let mut table = vec![f64::NAN; nr * nc];
    for i in 0..y.len() {
        if rl[i].is_empty() || cl[i].is_empty() || y[i].is_nan() {
            continue;
        }
        let (a, b) =
            (rlv.iter().position(|l| *l == rl[i]).unwrap_or(0), clv.iter().position(|l| *l == cl[i]).unwrap_or(0));
        if !table[a * nc + b].is_nan() {
            return Err(DialogError(format!(
                "more than one value for row {} and column {}; median polish needs exactly one per cell",
                rlv[a], clv[b]
            )));
        }
        table[a * nc + b] = y[i];
    }
    if table.iter().any(|a| a.is_nan()) {
        return Err("every combination of the two factors needs exactly one value".into());
    }
    let iters = v.count("iterations").max(1);
    let mp = median_polish(&table, nr, nc, iters, 0.01)?;
    let mut s = Section::new("Median Polish");
    s.text(format!("Overall (common) effect: {:.6}", mp.overall));
    let mut heads = vec![("", TKind::Text)];
    let names: Vec<String> = clv.clone();
    for n in &names {
        heads.push((n.as_str(), TKind::Significant(6)));
    }
    heads.push(("Row effect", TKind::Significant(6)));
    let mut t = Table::new("Residuals and effects", tcols(&heads));
    for i in 0..nr {
        let mut row = vec![Cell::text(rlv[i].clone())];
        row.extend((0..nc).map(|j| Cell::Num(mp.residuals[i * nc + j])));
        row.push(Cell::Num(mp.row[i]));
        t.push_row(row)?;
    }
    let mut last = vec![Cell::text("Column effect")];
    last.extend(mp.col.iter().map(|&c| Cell::Num(c)));
    last.push(Cell::Missing);
    t.push_row(last)?;
    s.table(t);
    if !mp.converged {
        s.warn(format!("The sweeps had not converged after {} iterations.", mp.iterations));
    }
    let mut res = vec![f64::NAN; y.len()];
    let mut fit = vec![f64::NAN; y.len()];
    for i in 0..y.len() {
        if rl[i].is_empty() || cl[i].is_empty() || y[i].is_nan() {
            continue;
        }
        let (a, b) =
            (rlv.iter().position(|l| *l == rl[i]).unwrap_or(0), clv.iter().position(|l| *l == cl[i]).unwrap_or(0));
        res[i] = mp.residuals[a * nc + b];
        fit[i] = y[i] - res[i];
    }
    let mut stored = Vec::new();
    if v.flag("store_res") {
        stored.push(Column::numeric("RESI", res));
    }
    if v.flag("store_fit") {
        stored.push(Column::numeric("FITS", fit));
    }
    out(s, stored)
}

fn resistant_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (yc, xc) = (v.var("y").ok_or("choose the response")?, v.var("x").ok_or("choose the predictor")?);
    let (y, x) = (numeric(ws, yc)?, numeric(ws, xc)?);
    let fit = resistant_line(&x, &y, v.count("iterations").max(1))?;
    let (yn, xn) = (col_title(ws, yc), col_title(ws, xc));
    let mut s = Section::new("Resistant Line");
    s.text(format!("The resistant line is {yn} = {:.6} + {:.6} {xn}.", fit.intercept, fit.slope));
    let mut t = Table::new(
        "Group medians",
        tcols(&[("Group", TKind::Text), ("Median x", TKind::Significant(6)), ("Median y", TKind::Significant(6))]),
    );
    for (i, g) in ["Left", "Middle", "Right"].iter().enumerate() {
        t.push_row(vec![Cell::text(*g), Cell::Num(fit.x_medians[i]), Cell::Num(fit.y_medians[i])])?;
    }
    s.table(t);
    let mut pts: Vec<(f64, f64)> =
        x.iter().zip(&y).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (lo, hi) = (pts[0].0, pts[pts.len() - 1].0);
    s.chart(chart(
        ChartKind::Scatter,
        format!("Resistant Line of {yn} vs {xn}"),
        &xn,
        &yn,
        Vec::new(),
        vec![
            Series { name: yn.clone(), x: pts.iter().map(|p| p.0).collect(), y: pts.iter().map(|p| p.1).collect() },
            Series {
                name: "Fit".into(),
                x: vec![lo, hi],
                y: vec![fit.intercept + fit.slope * lo, fit.intercept + fit.slope * hi],
            },
        ],
    ));
    let mut stored = Vec::new();
    if v.flag("store_fit") {
        stored.push(Column::numeric("FITS", fit.fitted.clone()));
    }
    if v.flag("store_res") {
        stored.push(Column::numeric("RESI", fit.residuals.clone()));
    }
    out(s, stored)
}

fn rootogram_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("x").ok_or("choose a column")?;
    let x = numeric(ws, c)?;
    let bins = rootogram(&x, v.count("bins"))?;
    let name = col_title(ws, c);
    let mut t = Table::new(
        format!("Rootogram of {name}"),
        tcols(&[
            ("Lower", TKind::Significant(6)),
            ("Upper", TKind::Significant(6)),
            ("Observed", TKind::Integer),
            ("Expected", TKind::Fixed(3)),
            ("Root obs", TKind::Fixed(3)),
            ("Root exp", TKind::Fixed(3)),
            ("Hanging", TKind::Fixed(3)),
            ("Double root", TKind::Fixed(3)),
        ]),
    );
    for b in &bins {
        t.push_row(vec![
            Cell::Num(b.lower),
            Cell::Num(b.upper),
            Cell::Int(b.observed as i64),
            Cell::Num(b.expected),
            Cell::Num(b.root_observed),
            Cell::Num(b.root_expected),
            Cell::Num(b.hanging),
            Cell::Num(b.double_root),
        ])?;
    }
    let mut s = Section::new("Rootogram");
    s.table(t);
    let mids: Vec<f64> = bins.iter().map(|b| (b.lower + b.upper) / 2.0).collect();
    let mut ch = chart(
        ChartKind::Line,
        format!("Double-Root Residuals of {name} (normal fit)"),
        &name,
        "Double-root residual",
        Vec::new(),
        vec![Series { name: "DRR".into(), x: mids, y: bins.iter().map(|b| b.double_root).collect() }],
    );
    ch.reference_lines = vec![
        ReferenceLine { axis: Axis::Y, value: 0.0, label: "0".into() },
        ReferenceLine { axis: Axis::Y, value: 2.0, label: "+2".into() },
        ReferenceLine { axis: Axis::Y, value: -2.0, label: "-2".into() },
    ];
    s.chart(ch);
    out(s, Vec::new())
}

fn symmetry_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("x").ok_or("choose a column")?;
    let pts = symmetry_points(&numeric(ws, c)?)?;
    let name = col_title(ws, c);
    let top = pts.iter().map(|p| p.0.max(p.1)).fold(0.0, f64::max);
    let mut s = Section::new("Symmetry Plot");
    s.chart(chart(
        ChartKind::Scatter,
        format!("Symmetry Plot of {name}"),
        "Distance below the median",
        "Distance above the median",
        Vec::new(),
        vec![
            Series { name: name.clone(), x: pts.iter().map(|p| p.0).collect(), y: pts.iter().map(|p| p.1).collect() },
            Series { name: "Fit".into(), x: vec![0.0, top], y: vec![0.0, top] },
        ],
    ));
    out(s, Vec::new())
}

// ------------------------------------------------------------------------- bootstrap

const BOOT_STATS: [&str; 11] = [
    "Mean",
    "Median",
    "Standard deviation",
    "Variance",
    "First quartile",
    "Third quartile",
    "5% trimmed mean",
    "Correlation (two columns)",
    "Difference of means (x - y)",
    "Difference of medians (x - y)",
    "Ratio of standard deviations (x / y)",
];

fn med_of(x: &[f64]) -> f64 {
    quantile(x, 0.5, QuantileMethod::Type6).unwrap_or(f64::NAN)
}

fn sd_of(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn one_stat(kind: usize, x: &[f64]) -> f64 {
    match kind {
        0 => x.iter().sum::<f64>() / x.len() as f64,
        1 => med_of(x),
        2 => sd_of(x),
        3 => sd_of(x).powi(2),
        4 => quantile(x, 0.25, QuantileMethod::Type6).unwrap_or(f64::NAN),
        5 => quantile(x, 0.75, QuantileMethod::Type6).unwrap_or(f64::NAN),
        _ => trimmed_mean(x, 0.05).ok().flatten().unwrap_or(f64::NAN),
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

fn bootstrap_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let kind = v.choice("stat").min(BOOT_STATS.len() - 1);
    let xc = v.var("x").ok_or("choose the first column")?;
    let yc = v.var("y");
    let two = kind >= 7;
    if two && yc.is_none() {
        return Err("this statistic needs a second column".into());
    }
    let opts = BootstrapOptions {
        resamples: v.count("resamples"),
        level: v.num("level") / 100.0,
        seed: v.count("seed") as u64,
    };
    let xall = numeric(ws, xc)?;
    let name;
    let result = if kind == 7 {
        let yall = numeric(ws, yc.unwrap_or(xc))?;
        let (x, y): (Vec<f64>, Vec<f64>) =
            xall.iter().zip(&yall).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).unzip();
        name = format!("Correlation of {} and {}", col_title(ws, xc), col_title(ws, yc.unwrap_or(xc)));
        bootstrap(
            &[x.len()],
            |s| {
                let a: Vec<f64> = s[0].iter().map(|&i| x[i]).collect();
                let b: Vec<f64> = s[0].iter().map(|&i| y[i]).collect();
                pearson(&a, &b)
            },
            &opts,
        )?
    } else if two {
        let x: Vec<f64> = xall.iter().copied().filter(|a| a.is_finite()).collect();
        let y: Vec<f64> = numeric(ws, yc.unwrap_or(xc))?.into_iter().filter(|a| a.is_finite()).collect();
        name = format!("{} of {} and {}", BOOT_STATS[kind], col_title(ws, xc), col_title(ws, yc.unwrap_or(xc)));
        bootstrap(
            &[x.len(), y.len()],
            |s| {
                let a: Vec<f64> = s[0].iter().map(|&i| x[i]).collect();
                let b: Vec<f64> = s[1].iter().map(|&i| y[i]).collect();
                match kind {
                    8 => one_stat(0, &a) - one_stat(0, &b),
                    9 => med_of(&a) - med_of(&b),
                    _ => sd_of(&a) / sd_of(&b),
                }
            },
            &opts,
        )?
    } else {
        let x: Vec<f64> = xall.iter().copied().filter(|a| a.is_finite()).collect();
        name = format!("{} of {}", BOOT_STATS[kind], col_title(ws, xc));
        bootstrap(&[x.len()], |s| one_stat(kind, &s[0].iter().map(|&i| x[i]).collect::<Vec<_>>()), &opts)?
    };
    let mut s = Section::new(format!("Bootstrap: {name}"));
    let mut t = Table::new(
        "Bootstrap summary",
        tcols(&[
            ("Statistic", TKind::Text),
            ("Observed", TKind::Significant(6)),
            ("Bias", TKind::Significant(6)),
            ("SE", TKind::Significant(6)),
            ("Resamples", TKind::Integer),
        ]),
    );
    t.push_row(vec![
        Cell::text(BOOT_STATS[kind]),
        Cell::Num(result.estimate),
        Cell::Num(result.bias),
        Cell::Num(result.std_error),
        Cell::Int(result.replicates.len() as i64),
    ])?;
    s.table(t);
    let mut ti = Table::new(
        format!("{:.0}% confidence intervals", opts.level * 100.0),
        tcols(&[("Method", TKind::Text), ("Lower", TKind::Significant(6)), ("Upper", TKind::Significant(6))]),
    );
    for iv in &result.intervals {
        let m = match iv.method {
            IntervalMethod::Percentile => "Percentile",
            IntervalMethod::Basic => "Basic",
            IntervalMethod::Normal => "Normal",
            IntervalMethod::Bca => "BCa",
        };
        ti.push_row(vec![Cell::text(m), Cell::Num(iv.lower), Cell::Num(iv.upper)])?;
    }
    s.table(ti);
    if result.bca_constants.is_none() {
        s.text("The BCa interval is not shown: the bias correction or the acceleration is undefined for these data.");
    }
    if result.dropped > 0 {
        s.warn(format!("{} resamples gave an undefined statistic and were discarded.", result.dropped));
    }
    if v.flag("hist") {
        s.chart(hist_chart(
            &result.replicates,
            format!("Histogram of Bootstrap Replicates: {name}"),
            BOOT_STATS[kind],
        )?);
    }
    let stored =
        if v.flag("store") { vec![Column::numeric("BootReps", result.replicates.clone())] } else { Vec::new() };
    out(s, stored)
}

// ------------------------------------------------------------------------- simulation

fn parse_inputs(text: &str) -> Res<Vec<(String, String, Vec<f64>)>> {
    let mut v = Vec::new();
    for part in text.split([';', '\n']).map(str::trim).filter(|p| !p.is_empty()) {
        let (name, rest) = part
            .split_once(['=', '~'])
            .ok_or_else(|| DialogError(format!("'{part}': write each input as name = distribution(parameters)")))?;
        let (name, rest) = (name.trim(), rest.trim());
        let open = rest.find('(').ok_or_else(|| DialogError(format!("'{part}': missing the parameter list")))?;
        if !rest.ends_with(')') {
            return Err(DialogError(format!("'{part}': missing the closing parenthesis")));
        }
        if name.is_empty()
            || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            || name.starts_with(|c: char| c.is_ascii_digit())
        {
            return Err(DialogError(format!("'{name}' is not a valid input name (letters, digits and _)")));
        }
        let dist = rest[..open].trim().to_ascii_lowercase();
        let params: Vec<f64> = rest[open + 1..rest.len() - 1]
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<f64>().map_err(|_| DialogError(format!("'{p}' in '{part}' is not a number"))))
            .collect::<Res<_>>()?;
        if v.iter().any(|(n, _, _): &(String, String, Vec<f64>)| n.eq_ignore_ascii_case(name)) {
            return Err(DialogError(format!("the input '{name}' is defined twice")));
        }
        v.push((name.to_string(), dist, params));
    }
    if v.is_empty() {
        return Err("describe at least one random input".into());
    }
    Ok(v)
}

fn substitute(expr: &str, names: &[String]) -> String {
    let mut out = String::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let tok: String = chars[start..i].iter().collect();
            // A name followed by "(" is a function, not an input.
            let is_call = chars.get(i) == Some(&'(');
            match names.iter().position(|n| n.eq_ignore_ascii_case(&tok)) {
                Some(k) if !is_call => out.push_str(&format!("C{}", k + 1)),
                _ => out.push_str(&tok),
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn simulate_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let inputs = parse_inputs(v.text("inputs"))?;
    let n = v.count("n");
    if !(2..=1_000_000).contains(&n) {
        return Err("the number of simulated rows must be between 2 and 1,000,000".into());
    }
    let expr = v.text("expr").trim().to_string();
    if expr.is_empty() {
        return Err("enter the equation for the output".into());
    }
    let seed = v.count("seed").max(1);
    let mut g = GridController::new(Workbook::new(), 400.0, 200.0);
    for (i, (_, dist, params)) in inputs.iter().enumerate() {
        let list: String = params.iter().map(|p| format!(", {p}")).collect();
        let e = format!("RANDOMSEED({}, {n}, \"{dist}\"{list})", seed + i);
        g.calculate(i, &e).map_err(|e| DialogError(format!("input {}: {e}", inputs[i].0)))?;
    }
    let names: Vec<String> = inputs.iter().map(|i| i.0.clone()).collect();
    let k = inputs.len();
    g.calculate(k, &substitute(&expr, &names)).map_err(|e| DialogError(format!("equation: {e}")))?;
    let ws = g.worksheet();
    let y = numeric(ws, k)?;
    if y.iter().filter(|a| a.is_finite()).count() < 2 {
        return Err("the equation produced no usable values".into());
    }
    let ok: Vec<f64> = y.iter().copied().filter(|a| a.is_finite()).collect();
    let d = describe(&ok)?;
    let name = if v.text("name").trim().is_empty() { "Output".to_string() } else { v.text("name").trim().to_string() };
    let mut s = Section::new(format!("Simulation of {name}"));
    s.text(format!("{name} = {expr}   ({} simulated rows, seed {seed})", ok.len()));
    let mut t = Table::new(
        "Summary of the simulated output",
        tcols(&[
            ("N", TKind::Integer),
            ("Mean", TKind::Significant(6)),
            ("StDev", TKind::Significant(6)),
            ("Minimum", TKind::Significant(6)),
            ("Q1", TKind::Significant(6)),
            ("Median", TKind::Significant(6)),
            ("Q3", TKind::Significant(6)),
            ("Maximum", TKind::Significant(6)),
        ]),
    );
    let num = |o: Option<f64>| o.map_or(Cell::Missing, Cell::Num);
    t.push_row(vec![
        Cell::Int(ok.len() as i64),
        num(d.mean),
        num(d.std_dev),
        num(d.minimum),
        num(d.q1),
        num(d.median),
        num(d.q3),
        num(d.maximum),
    ])?;
    s.table(t);
    let mut tp = Table::new("Percentiles", tcols(&[("Percent", TKind::Fixed(1)), ("Value", TKind::Significant(6))]));
    for p in [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0] {
        tp.push_row(vec![Cell::Num(p), Cell::Num(quantile(&ok, p / 100.0, QuantileMethod::Type6)?)])?;
    }
    s.table(tp);
    let (lo, hi) = (v.num("lower"), v.num("upper"));
    if lo.is_finite() || hi.is_finite() {
        let mut tl = Table::new(
            "Probability of exceeding limits",
            tcols(&[("Event", TKind::Text), ("Probability", TKind::Fixed(5))]),
        );
        let n = ok.len() as f64;
        if lo.is_finite() {
            tl.push_row(vec![
                Cell::text(format!("{name} < {lo}")),
                Cell::Num(ok.iter().filter(|&&a| a < lo).count() as f64 / n),
            ])?;
        }
        if hi.is_finite() {
            tl.push_row(vec![
                Cell::text(format!("{name} > {hi}")),
                Cell::Num(ok.iter().filter(|&&a| a > hi).count() as f64 / n),
            ])?;
        }
        s.table(tl);
    }
    s.chart(hist_chart(&ok, format!("Histogram of Simulated {name}"), &name)?);
    let mut stored = Vec::new();
    if v.flag("store") {
        for (i, (n, _, _)) in inputs.iter().enumerate() {
            stored.push(Column::numeric(n.clone(), numeric(ws, i)?));
        }
        stored.push(Column::numeric(name, y));
    }
    out(s, stored)
}

// ------------------------------------------------------------------------- sample from columns

fn sample_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let n = v.count("n");
    if n == 0 {
        return Err("enter the number of rows to sample".into());
    }
    let len = cols.iter().map(|&c| ws.columns.get(c).map_or(0, Column::len)).max().unwrap_or(0);
    let pool: Vec<usize> = (0..len).collect();
    let replace = v.flag("replace");
    if !replace && n > pool.len() {
        return Err(DialogError(format!(
            "cannot sample {n} rows without replacement from {} rows; choose fewer rows or sample with replacement",
            pool.len()
        )));
    }
    let mut rng = openstats_math::random::Pcg64::new(u128::from(v.count("seed").max(1) as u64), 0xDA3E_39CB_94B9_5BDB);
    use openstats_math::random::Rng;
    let picks: Vec<usize> = if replace {
        (0..n).map(|_| rng.below(pool.len() as u64) as usize).collect()
    } else {
        let mut p = pool.clone();
        for i in 0..n {
            let j = i + rng.below((p.len() - i) as u64) as usize;
            p.swap(i, j);
        }
        p.truncate(n);
        p
    };
    let mut stored = Vec::new();
    for &c in cols {
        let col = ws.column(c)?;
        let idx: Vec<usize> = picks.iter().map(|&i| i.min(col.len().saturating_sub(1))).collect();
        let mut data = col.data.take(&idx);
        // Rows beyond a shorter column are missing, not repeats of its last row.
        for (k, &i) in picks.iter().enumerate() {
            if i >= col.len() {
                data.set(k, &openstats_core::Value::Missing)?;
            }
        }
        stored.push(Column::new(format!("{}_sample", col_title(ws, c)), data));
    }
    let mut s = Section::new("Sample From Columns");
    s.text(format!(
        "Sampled {n} rows {} replacement from {} rows (the same rows are taken from every column).",
        if replace { "with" } else { "without" },
        pool.len()
    ));
    out(s, stored)
}

// ------------------------------------------------------------------------- registry

/// Register the dialogs of this module.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options, Storage};
    use VarKind::{Any, Numeric};
    add(
        r,
        DialogSpec::new("stat.tables.tally", "Tally Individual Variables", "Counts and percents of each value.")
            .vars("vars", "Variables", Any, 1, 50)
            .check(Options, "counts", "Counts", true)
            .check(Options, "percents", "Percents", false)
            .check(Options, "cum_counts", "Cumulative counts", false)
            .check(Options, "cum_percents", "Cumulative percents", false),
        &["Stat", "Tables", "Tally Individual Variables..."],
        Runner::Report(tally_run),
    );
    let mut cell = DialogSpec::new(
        "stat.tables.cells",
        "Descriptive Statistics Table",
        "Counts and summary statistics in the cells of a table.",
    )
    .vars("rows", "Row variables", Any, 1, 3)
    .vars("cols", "Column variable", Any, 0, 1)
    .vars("assoc", "Summary variables", Numeric, 0, 10);
    for (k, l) in STATS {
        cell = cell.check(Options, k, l, k == "stat_mean" || k == "stat_n");
    }
    add(r, cell, &["Stat", "Tables", "Descriptive Statistics..."], Runner::Report(cell_table_run));
    add(
        r,
        DialogSpec::new("quality.pareto", "Pareto Chart", "Defects in decreasing order with the cumulative percent.")
            .vars("defects", "Defects or categories", Any, 1, 1)
            .vars("counts", "Counts (blank: count the rows)", Numeric, 0, 1),
        &["Stat", "Quality Tools", "Pareto Chart..."],
        Runner::Report(pareto_run),
    );
    add(
        r,
        DialogSpec::new("stat.eda.letter", "Letter Values", "Median, hinges, eighths and the extremes.").vars(
            "vars",
            "Variables",
            Numeric,
            1,
            20,
        ),
        &["Stat", "EDA", "Letter Values..."],
        Runner::Report(letter_run),
    );
    add(
        r,
        DialogSpec::new("stat.eda.polish", "Median Polish", "Additive fit of a two-way table by medians.")
            .vars("y", "Response", Numeric, 1, 1)
            .vars("row", "Row factor", Any, 1, 1)
            .vars("col", "Column factor", Any, 1, 1)
            .integer(Options, "iterations", "Maximum iterations", 10.0, 1.0, 100.0)
            .check(Storage, "store_res", "Residuals", false)
            .check(Storage, "store_fit", "Fits", false),
        &["Stat", "EDA", "Median Polish..."],
        Runner::Report(polish_run),
    );
    add(
        r,
        DialogSpec::new("stat.eda.resistant", "Resistant Line", "Tukey's three-group resistant line.")
            .vars("y", "Response", Numeric, 1, 1)
            .vars("x", "Predictor", Numeric, 1, 1)
            .integer(Options, "iterations", "Maximum iterations", 10.0, 1.0, 100.0)
            .check(Storage, "store_fit", "Fits", false)
            .check(Storage, "store_res", "Residuals", false),
        &["Stat", "EDA", "Resistant Line..."],
        Runner::Report(resistant_run),
    );
    add(
        r,
        DialogSpec::new("stat.eda.rootogram", "Rootogram", "Observed and expected counts on the square-root scale.")
            .vars("x", "Variable", Numeric, 1, 1)
            .integer(Options, "bins", "Number of bins (0 = automatic)", 0.0, 0.0, 200.0),
        &["Stat", "EDA", "Rootogram..."],
        Runner::Report(rootogram_run),
    );
    add(
        r,
        DialogSpec::new("stat.eda.symmetry", "Symmetry Plot", "Distances above and below the median.")
            .vars("x", "Variable", Numeric, 1, 1),
        &["Stat", "EDA", "Symmetry Plot..."],
        Runner::Report(symmetry_run),
    );
    add(
        r,
        DialogSpec::new("stat.resample.bootstrap", "Bootstrap", "Bootstrap standard error and confidence intervals.")
            .choice(Main, "stat", "Statistic", &BOOT_STATS, 0)
            .vars("x", "First column", Numeric, 1, 1)
            .vars("y", "Second column (when needed)", Numeric, 0, 1)
            .integer(Options, "resamples", "Number of resamples", 2000.0, 20.0, 1_000_000.0)
            .number(Options, "level", "Confidence level (%)", 95.0, 50.0, 99.99)
            .integer(Options, "seed", "Random seed", 1.0, 1.0, 1e9)
            .check(Graphs, "hist", "Histogram of the replicates", true)
            .check(Storage, "store", "Store the replicates", false),
        &["Stat", "Resampling", "Bootstrap..."],
        Runner::Report(bootstrap_run),
    );
    add(
        r,
        DialogSpec::new(
            "calc.simulate",
            "Monte Carlo Simulation",
            "Simulate an output from random inputs and an equation.",
        )
        .text(Main, "inputs", "Inputs, e.g. X = normal(10, 2); Y = uniform(0, 1)", "X = normal(0, 1)")
        .text(Main, "expr", "Equation for the output", "X")
        .text(Main, "name", "Name of the output", "Output")
        .integer(Options, "n", "Number of simulated rows", 10000.0, 2.0, 1_000_000.0)
        .integer(Options, "seed", "Random seed", 1.0, 1.0, 1e9)
        .optional_number(Options, "lower", "Report P(output < this value)")
        .optional_number(Options, "upper", "Report P(output > this value)")
        .check(Storage, "store", "Store the simulated inputs and output", false),
        &["Calc", "Monte Carlo Simulation..."],
        Runner::Report(simulate_run),
    );
    add(
        r,
        DialogSpec::new("calc.sample", "Sample From Columns", "Draw rows at random from columns.")
            .vars("vars", "Columns", Any, 1, 50)
            .integer(Main, "n", "Number of rows to sample", 10.0, 1.0, 10_000_000.0)
            .integer(Options, "seed", "Random seed", 1.0, 1.0, 1e9)
            .check(Options, "replace", "Sample with replacement", false),
        &["Calc", "Random Data", "Sample From Columns..."],
        Runner::Report(sample_run),
    );
}
