//! More of Stat ▸ Basic Statistics: proportions, Poisson rates, variances, outliers,
//! equivalence, tolerance intervals, cross-tabulation, goodness of fit, covariance and partial
//! correlation. Every test accepts either summarised data or worksheet columns; rows with a
//! missing value are left out.

use super::analyses::{add, out};
use super::analysis::{col_title, labels, level, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::Worksheet;
use openstats_stats::correlation::{covariance_matrix, partial_correlation, MissingData};
use openstats_stats::crosstab::{
    association, chi_square_gof, fisher_exact, likelihood_ratio_chi_square, linear_by_linear, mcnemar,
    pearson_chi_square, Crosstab, GofExpected,
};
use openstats_stats::equivalence::{tost_one_sample, tost_paired, tost_two_sample};
use openstats_stats::hypothesis::{Alternative, VarianceAssumption};
use openstats_stats::outlier::{dixon_q_test, generalized_esd, grubbs_test, OutlierSide};
use openstats_stats::poisson::{one_sample_poisson, poisson_totals, two_sample_poisson};
use openstats_stats::proportion::{one_proportion, two_proportions};
use openstats_stats::report::{Cell, Column as RCol, ColumnKind as RKind, Reportable, Section, Table};
use openstats_stats::tolerance::{nonparametric_tolerance, normal_tolerance, NormalMethod, ToleranceSides};
use openstats_stats::variance::{bonett_test, f_test, levene_test, Center};

const ALTS: [&str; 3] = ["not equal", "less than", "greater than"];

fn alt(v: &Values) -> Alternative {
    match v.choice("alt") {
        1 => Alternative::Less,
        2 => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

fn with_level(s: DialogSpec) -> DialogSpec {
    s.number(Page::Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
}

fn push(t: &mut Table, row: Vec<Cell>) -> Res<()> {
    t.push_row(row).map_err(DialogError::from)
}

fn sort_levels(l: &mut [String]) {
    let nums: Option<Vec<f64>> = l.iter().map(|s| s.trim().parse::<f64>().ok()).collect();
    if nums.is_some() {
        l.sort_by(|a, b| a.trim().parse::<f64>().unwrap_or(0.0).total_cmp(&b.trim().parse::<f64>().unwrap_or(0.0)));
    } else {
        l.sort();
    }
}

/// Number of events and of trials in a column; the event is `event`, or the last level in
/// sorted order when blank.
fn column_counts(ws: &Worksheet, c: usize, event: &str) -> Res<(u64, u64, String)> {
    let all: Vec<String> = labels(ws, c)?.into_iter().filter(|s| !s.is_empty()).collect();
    if all.is_empty() {
        return Err(format!("{} has no values", col_title(ws, c)).into());
    }
    let mut lv = all.clone();
    lv.sort();
    lv.dedup();
    sort_levels(&mut lv);
    let ev = if event.trim().is_empty() { lv.last().cloned().unwrap_or_default() } else { event.trim().to_string() };
    if !lv.contains(&ev) {
        return Err(format!("{} has no value {ev}", col_title(ws, c)).into());
    }
    Ok((all.iter().filter(|s| **s == ev).count() as u64, all.len() as u64, ev))
}

/// Split a response column by the levels of a factor column (rows with either missing dropped).
fn split_groups(ws: &Worksheet, resp: usize, fac: usize) -> Res<(Vec<String>, Vec<Vec<f64>>)> {
    let y = numeric(ws, resp)?;
    let f = labels(ws, fac)?;
    let mut lv: Vec<String> = f.iter().filter(|s| !s.is_empty()).cloned().collect();
    lv.sort();
    lv.dedup();
    sort_levels(&mut lv);
    let groups = lv
        .iter()
        .map(|l| (0..y.len().min(f.len())).filter(|&i| &f[i] == l && !y[i].is_nan()).map(|i| y[i]).collect())
        .collect();
    Ok((lv, groups))
}

fn two_samples(ws: &Worksheet, v: &Values) -> Res<(String, String, Vec<f64>, Vec<f64>)> {
    if v.choice("mode") == 1 {
        let (r, s) =
            (v.var("resp").ok_or("select the response column")?, v.var("samples").ok_or("select the samples column")?);
        let (lv, g) = split_groups(ws, r, s)?;
        if lv.len() != 2 {
            return Err(format!("{} has {} levels; exactly 2 are needed", col_title(ws, s), lv.len()).into());
        }
        let mut it = g.into_iter();
        let (a, b) = (it.next().unwrap_or_default(), it.next().unwrap_or_default());
        Ok((lv[0].clone(), lv[1].clone(), a, b))
    } else {
        let (a, b) =
            (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
        Ok((col_title(ws, a), col_title(ws, b), numeric(ws, a)?, numeric(ws, b)?))
    }
}

fn append(s: &mut Section, other: Section) {
    s.text(other.heading.clone());
    s.blocks.extend(other.blocks);
    s.warnings.extend(other.warnings);
}

// ------------------------------------------------------------------------------ proportions

fn prop1(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (x, n, name) = if v.choice("mode") == 0 {
        (v.count("events") as u64, v.count("trials") as u64, "Summarized data".to_string())
    } else {
        let c = v.var("var").ok_or("select a column")?;
        let (x, n, ev) = column_counts(ws, c, v.text("event"))?;
        (x, n, format!("{} (event: {ev})", col_title(ws, c)))
    };
    let r = one_proportion(x, n, v.num("p0"), alt(v), level(v)?, v.flag("cc"))?;
    let mut s = r.report_section();
    s.heading = format!("Test and CI for One Proportion: {name}");
    out(s)
}

fn prop2(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (a, b, name) = match v.choice("mode") {
        0 => (
            (v.count("events1") as u64, v.count("trials1") as u64),
            (v.count("events2") as u64, v.count("trials2") as u64),
            "Summarized data".to_string(),
        ),
        1 => {
            let (c1, c2) =
                (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
            let ev = v.text("event").to_string();
            // the same event level for both samples: the given one, or the largest level overall
            let ev = if ev.trim().is_empty() {
                let mut lv: Vec<String> =
                    labels(ws, c1)?.into_iter().chain(labels(ws, c2)?).filter(|s| !s.is_empty()).collect();
                lv.sort();
                lv.dedup();
                sort_levels(&mut lv);
                lv.last().cloned().ok_or("the columns have no values")?
            } else {
                ev
            };
            let (x1, n1, _) = column_counts(ws, c1, &ev).or_else(|_| count_missing_event(ws, c1))?;
            let (x2, n2, _) = column_counts(ws, c2, &ev).or_else(|_| count_missing_event(ws, c2))?;
            ((x1, n1), (x2, n2), format!("{}, {} (event: {ev})", col_title(ws, c1), col_title(ws, c2)))
        }
        _ => {
            let (r, s) = (
                v.var("resp").ok_or("select the response column")?,
                v.var("samples").ok_or("select the samples column")?,
            );
            let (resp, fac) = (labels(ws, r)?, labels(ws, s)?);
            let mut lv: Vec<String> = fac.iter().filter(|x| !x.is_empty()).cloned().collect();
            lv.sort();
            lv.dedup();
            if lv.len() != 2 {
                return Err(format!("{} has {} levels; exactly 2 are needed", col_title(ws, s), lv.len()).into());
            }
            let mut rl: Vec<String> = resp.iter().filter(|x| !x.is_empty()).cloned().collect();
            rl.sort();
            rl.dedup();
            sort_levels(&mut rl);
            let ev = if v.text("event").trim().is_empty() {
                rl.last().cloned().ok_or("the response has no values")?
            } else {
                v.text("event").trim().to_string()
            };
            let mut cnt = [(0u64, 0u64); 2];
            for i in 0..resp.len().min(fac.len()) {
                if resp[i].is_empty() || fac[i].is_empty() {
                    continue;
                }
                let g = usize::from(fac[i] == lv[1]);
                cnt[g].1 += 1;
                if resp[i] == ev {
                    cnt[g].0 += 1;
                }
            }
            (
                cnt[0],
                cnt[1],
                format!("{} by {} (event: {ev}; samples {} and {})", col_title(ws, r), col_title(ws, s), lv[0], lv[1]),
            )
        }
    };
    let r = two_proportions(a.0, a.1, b.0, b.1, v.num("diff"), alt(v), level(v)?, v.flag("cc"))?;
    let mut s = r.report_section();
    s.heading = format!("Test and CI for Two Proportions: {name}");
    out(s)
}

/// A sample that never shows the event level has zero events.
fn count_missing_event(ws: &Worksheet, c: usize) -> Res<(u64, u64, String)> {
    let n = labels(ws, c)?.iter().filter(|s| !s.is_empty()).count() as u64;
    if n == 0 {
        return Err(format!("{} has no values", col_title(ws, c)).into());
    }
    Ok((0, n, String::new()))
}

// ---------------------------------------------------------------------------------- Poisson

fn poisson1(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (total, exposure, name) = if v.choice("mode") == 0 {
        (v.count("total") as u64, v.num("length"), "Summarized data".to_string())
    } else {
        let c = v.var("var").ok_or("select a column")?;
        let (t, e) = poisson_totals(&numeric(ws, c)?, v.num("length"))?;
        (t, e, col_title(ws, c))
    };
    let r = one_sample_poisson(total, exposure, v.num("rate0"), alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Test and CI for One-Sample Poisson Rate: {name}");
    out(s)
}

fn poisson2(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (a, b, name) = if v.choice("mode") == 0 {
        (
            (v.count("total1") as u64, v.num("length1")),
            (v.count("total2") as u64, v.num("length2")),
            "Summarized data".to_string(),
        )
    } else {
        let (c1, c2) =
            (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
        let a = poisson_totals(&numeric(ws, c1)?, v.num("length1"))?;
        let b = poisson_totals(&numeric(ws, c2)?, v.num("length2"))?;
        (a, b, format!("{}, {}", col_title(ws, c1), col_title(ws, c2)))
    };
    let r = two_sample_poisson(a.0, a.1, b.0, b.1, v.num("diff"), alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Test and CI for Two-Sample Poisson Rates: {name}");
    out(s)
}

// ----------------------------------------------------------------------------------- variances

fn var2(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (n1, n2, x, y) = two_samples(ws, v)?;
    let f = f_test(&x, &y, v.num("ratio"), alt(v), level(v)?)?;
    let mut s = f.report_section();
    s.heading = format!("Test and CI for Two Variances: {n1}, {n2}");
    append(&mut s, levene_test(&[x, y], Center::Median)?.report_section());
    out(s)
}

fn eqvar(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (r, f) =
        (v.var("resp").ok_or("select the response column")?, v.var("samples").ok_or("select the factor column")?);
    let (lv, groups) = split_groups(ws, r, f)?;
    let keep: Vec<Vec<f64>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
    if keep.len() < 2 {
        return Err("at least two groups with data are needed".into());
    }
    let _ = lv;
    let mut s = bonett_test(&keep, level(v)?)?.report_section();
    s.heading = format!("Test for Equal Variances: {} versus {}", col_title(ws, r), col_title(ws, f));
    append(&mut s, levene_test(&keep, Center::Median)?.report_section());
    out(s)
}

// -------------------------------------------------------------------------------------- outliers

fn outlier(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let x = numeric(ws, c)?;
    let side = match v.choice("side") {
        1 => OutlierSide::Larger,
        2 => OutlierSide::Smaller,
        _ => OutlierSide::Either,
    };
    let mut s = match v.choice("method") {
        0 => grubbs_test(&x, side)?.report_section(),
        1 => dixon_q_test(&x, side)?.report_section(),
        _ => generalized_esd(&x, v.count("max").max(1), v.num("alpha"))?.report_section(),
    };
    s.heading = format!("Outlier Test: {}", col_title(ws, c));
    out(s)
}

// ------------------------------------------------------------------------------------ equivalence

fn equivalence(ws: &Worksheet, v: &Values) -> Res<Output> {
    let a = v.var("first").ok_or("select the first sample")?;
    let x = numeric(ws, a)?;
    let (lo, hi, alpha) = (v.num("lower"), v.num("upper"), v.num("alpha"));
    let (t, name) = match v.choice("kind") {
        0 => (tost_one_sample(&x, v.num("target"), lo, hi, alpha)?, col_title(ws, a)),
        kind => {
            let b = v.var("second").ok_or("select the second sample")?;
            let y = numeric(ws, b)?;
            let name = format!("{} and {}", col_title(ws, a), col_title(ws, b));
            if kind == 1 {
                (tost_paired(&x, &y, lo, hi, alpha)?, name)
            } else {
                let var = if v.flag("pooled") { VarianceAssumption::Pooled } else { VarianceAssumption::Welch };
                (tost_two_sample(&x, &y, lo, hi, var, alpha)?, name)
            }
        }
    };
    let mut s = t.report_section();
    s.heading = format!("Equivalence Test: {name}");
    out(s)
}

// ------------------------------------------------------------------------------------- tolerance

fn tolerance(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let x = numeric(ws, c)?;
    let (cov, conf) = (v.num("coverage") / 100.0, v.num("confidence") / 100.0);
    let sides = match v.choice("sides") {
        1 => ToleranceSides::Upper,
        2 => ToleranceSides::Lower,
        _ => ToleranceSides::TwoSided,
    };
    let mut s = match v.choice("method") {
        0 => normal_tolerance(&x, cov, conf, sides, NormalMethod::Exact)?.report_section(),
        1 => normal_tolerance(&x, cov, conf, sides, NormalMethod::Howe)?.report_section(),
        _ => nonparametric_tolerance(&x, cov, conf, sides)?.report_section(),
    };
    s.heading = format!("Tolerance Interval: {}", col_title(ws, c));
    out(s)
}

// ------------------------------------------------------------------------------------ crosstab

fn matrix_table(title: &str, ct: &Crosstab, cell: impl Fn(usize, usize) -> Cell, kind: RKind) -> Table {
    let mut cols = vec![RCol::new("", RKind::Text)];
    cols.extend(ct.col_labels.iter().map(|l| RCol::new(l.clone(), kind)));
    let mut t = Table::new(title, cols);
    for (i, l) in ct.row_labels.iter().enumerate() {
        let mut row = vec![Cell::text(l.clone())];
        row.extend((0..ct.col_labels.len()).map(|j| cell(i, j)));
        let _ = t.push_row(row);
    }
    t
}

fn crosstab(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (ct, name) = if v.choice("mode") == 0 {
        let (r, c) =
            (v.var("rowvar").ok_or("select the row variable")?, v.var("colvar").ok_or("select the column variable")?);
        (
            Crosstab::from_labels(&labels(ws, r)?, &labels(ws, c)?)?,
            format!("{} by {}", col_title(ws, r), col_title(ws, c)),
        )
    } else {
        let cols = v.vars("counts");
        if cols.len() < 2 {
            return Err("select at least two columns of counts".into());
        }
        let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
        let n = data.iter().map(Vec::len).max().unwrap_or(0);
        let rows: Vec<usize> = (0..n).filter(|&i| data.iter().all(|d| d.get(i).is_some_and(|x| !x.is_nan()))).collect();
        if rows.is_empty() {
            return Err("the count columns have no complete rows".into());
        }
        let mut counts = Vec::new();
        for &i in &rows {
            let mut r = Vec::new();
            for d in &data {
                if d[i] < 0.0 || d[i].fract() != 0.0 {
                    return Err("counts must be non-negative whole numbers".into());
                }
                r.push(d[i] as u64);
            }
            counts.push(r);
        }
        let mut t = Crosstab::from_counts(counts)?;
        t.col_labels = cols.iter().map(|&c| col_title(ws, c)).collect();
        (t, "Table of counts".to_string())
    };
    let mut s = ct.report_section();
    s.heading = format!("Tabulated Statistics: {name}");
    let pct = ct.percentages();
    if v.flag("row_pct") {
        s.table(matrix_table("Row percentages", &ct, |i, j| Cell::Num(pct.row[i][j]), RKind::Fixed(2)));
    }
    if v.flag("col_pct") {
        s.table(matrix_table("Column percentages", &ct, |i, j| Cell::Num(pct.col[i][j]), RKind::Fixed(2)));
    }
    if v.flag("tot_pct") {
        s.table(matrix_table("Total percentages", &ct, |i, j| Cell::Num(pct.total[i][j]), RKind::Fixed(2)));
    }
    if v.flag("expected") {
        let e = ct.expected();
        s.table(matrix_table("Expected counts", &ct, |i, j| Cell::Num(e[i][j]), RKind::Fixed(3)));
    }
    if v.flag("residuals") {
        let e = ct.pearson_residuals();
        s.table(matrix_table("Standardized (Pearson) residuals", &ct, |i, j| Cell::Num(e[i][j]), RKind::Fixed(3)));
        let e = ct.adjusted_residuals();
        s.table(matrix_table("Adjusted residuals", &ct, |i, j| Cell::Num(e[i][j]), RKind::Fixed(3)));
    }
    if v.flag("chi") {
        append(&mut s, pearson_chi_square(&ct)?.report_section());
    }
    if v.flag("lr") {
        let r = likelihood_ratio_chi_square(&ct)?;
        let mut t = Table::new(
            "Likelihood-Ratio Chi-Square Test",
            vec![
                RCol::new("Chi-Square", RKind::Significant(6)),
                RCol::new("DF", RKind::Integer),
                RCol::new("P-Value", RKind::PValue),
            ],
        );
        push(&mut t, vec![Cell::Num(r.statistic), Cell::Int(r.df as i64), Cell::Num(r.p_value)])?;
        s.table(t);
    }
    if v.flag("trend") {
        let r = linear_by_linear(&ct)?;
        let mut t = Table::new(
            "Mantel-Haenszel Linear-by-Linear Association",
            vec![
                RCol::new("Chi-Square", RKind::Significant(6)),
                RCol::new("DF", RKind::Integer),
                RCol::new("P-Value", RKind::PValue),
            ],
        );
        push(&mut t, vec![Cell::Num(r.statistic), Cell::Int(1), Cell::Num(r.p_value)])?;
        s.table(t);
    }
    if v.flag("fisher") {
        append(&mut s, fisher_exact(&ct)?.report_section());
    }
    if v.flag("mcnemar") {
        append(&mut s, mcnemar(&ct)?.report_section());
    }
    if v.flag("assoc") {
        let a = association(&ct)?;
        let mut t = Table::new(
            "Measures of Association",
            vec![
                RCol::new("Phi", RKind::Fixed(4)),
                RCol::new("Cramer's V", RKind::Fixed(4)),
                RCol::new("Contingency Coefficient", RKind::Fixed(4)),
            ],
        );
        push(&mut t, vec![Cell::Num(a.phi), Cell::Num(a.cramers_v), Cell::Num(a.contingency_coefficient)])?;
        s.table(t);
    }
    out(s)
}

fn gof(ws: &Worksheet, v: &Values) -> Res<Output> {
    let o = v.var("observed").ok_or("select the observed counts column")?;
    let obs_all = numeric(ws, o)?;
    let names = match v.var("names") {
        Some(c) => Some(labels(ws, c)?),
        None => None,
    };
    let expc = if v.choice("expected") == 0 {
        None
    } else {
        Some(numeric(ws, v.var("expcol").ok_or("select the column of expected proportions or counts")?)?)
    };
    let rows: Vec<usize> = (0..obs_all.len())
        .filter(|&i| !obs_all[i].is_nan() && expc.as_ref().is_none_or(|e| e.get(i).is_some_and(|x| !x.is_nan())))
        .collect();
    let obs: Vec<f64> = rows.iter().map(|&i| obs_all[i]).collect();
    let lab: Vec<String> = rows
        .iter()
        .map(|&i| {
            names
                .as_ref()
                .and_then(|n| n.get(i))
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| (i + 1).to_string())
        })
        .collect();
    let exp = match (v.choice("expected"), expc) {
        (1, Some(e)) => GofExpected::Proportions(rows.iter().map(|&i| e[i]).collect()),
        (2, Some(e)) => GofExpected::Counts(rows.iter().map(|&i| e[i]).collect()),
        _ => GofExpected::Equal,
    };
    let r = chi_square_gof(&lab, &obs, &exp, v.count("params"))?;
    let mut s = r.report_section();
    s.heading = format!("Chi-Square Goodness-of-Fit Test for Observed Counts in Variable: {}", col_title(ws, o));
    out(s)
}

// ------------------------------------------------------------------------------ covariance etc.

fn covariance(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let names: Vec<String> = cols.iter().map(|&c| col_title(ws, c)).collect();
    let mode = if v.choice("missing") == 0 { MissingData::Listwise } else { MissingData::Pairwise };
    let cm = covariance_matrix(&data, mode)?;
    let mut s = Section::new(format!("Covariances: {}", names.join(", ")));
    let mut head = vec![RCol::new("", RKind::Text)];
    head.extend(names.iter().map(|n| RCol::new(n.clone(), RKind::Significant(6))));
    let mut t = Table::new("Covariances", head);
    for (i, n) in names.iter().enumerate() {
        let mut row = vec![Cell::text(n.clone())];
        row.extend(cm.values[i].iter().map(|&x| Cell::Num(x)));
        push(&mut t, row)?;
    }
    s.table(t);
    if v.choice("missing") == 1 {
        let mut t = Table::new(
            "Observations per pair",
            vec![RCol::new("", RKind::Text)]
                .into_iter()
                .chain(names.iter().map(|n| RCol::new(n.clone(), RKind::Integer)))
                .collect(),
        );
        for (i, n) in names.iter().enumerate() {
            let mut row = vec![Cell::text(n.clone())];
            row.extend(cm.n[i].iter().map(|&k| Cell::Int(k as i64)));
            push(&mut t, row)?;
        }
        s.table(t);
    }
    out(s)
}

fn partial(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let controls = v.vars("controls");
    if cols.iter().any(|c| controls.contains(c)) {
        return Err("a variable cannot also be a control variable".into());
    }
    let ctrl: Vec<Vec<f64>> = controls.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let names: Vec<String> = cols.iter().map(|&c| col_title(ws, c)).collect();
    let cnames: Vec<String> = controls.iter().map(|&c| col_title(ws, c)).collect();
    let mut s =
        Section::new(format!("Partial Correlations: {} controlling for {}", names.join(", "), cnames.join(", ")));
    let mut t = Table::new(
        "Partial correlation",
        vec![
            RCol::new("Pair", RKind::Text),
            RCol::new("N", RKind::Integer),
            RCol::new("DF", RKind::Integer),
            RCol::new("Partial correlation", RKind::Fixed(4)),
            RCol::new("Lower", RKind::Fixed(4)),
            RCol::new("Upper", RKind::Fixed(4)),
            RCol::new("P-Value", RKind::PValue),
        ],
    );
    let lvl = level(v)?;
    for i in 0..cols.len() {
        for j in 0..i {
            let r = partial_correlation(&numeric(ws, cols[j])?, &numeric(ws, cols[i])?, &ctrl, lvl)?;
            push(
                &mut t,
                vec![
                    Cell::text(format!("{} * {}", names[j], names[i])),
                    Cell::Int(r.n as i64),
                    Cell::Int(r.df as i64),
                    Cell::Num(r.r),
                    Cell::opt(r.ci.map(|c| c.lower)),
                    Cell::opt(r.ci.map(|c| c.upper)),
                    Cell::Num(r.p_value),
                ],
            )?;
        }
    }
    s.table(t);
    out(s)
}

// ------------------------------------------------------------------------------- registration

/// Register the additional basic-statistics dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options};
    use VarKind::{Any, Numeric};
    let _ = Graphs;
    let path = |last: &'static str| ["Stat", "Basic Statistics", last];
    let _ = Main;

    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.prop1",
                "1 Proportion",
                "Test and confidence interval for one proportion, exact and normal approximation.",
            )
            .choice(Page::Main, "mode", "Data are", &["Summarized data", "A column of outcomes"], 0)
            .integer(Page::Main, "events", "Number of events", 0.0, 0.0, 1.0e12)
            .integer(Page::Main, "trials", "Number of trials", 0.0, 0.0, 1.0e12)
            .vars("var", "Column of outcomes", Any, 0, 1)
            .text(Page::Main, "event", "Event (blank = last value in sorted order)", "")
            .number(Options, "p0", "Hypothesized proportion", 0.5, 1e-9, 0.999999999)
            .choice(Options, "alt", "Alternative", &ALTS, 0)
            .check(Options, "cc", "Use continuity correction for the normal test", false),
        ),
        &path("1 Proportion..."),
        Runner::Report(prop1),
    );
    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.prop2",
                "2 Proportions",
                "Compare two proportions: pooled z test, Fisher's exact p-value and an interval for the difference.",
            )
            .choice(
                Page::Main,
                "mode",
                "Data are",
                &["Summarized data", "Each sample in its own column", "Outcomes in one column, samples in another"],
                0,
            )
            .integer(Page::Main, "events1", "Sample 1 events", 0.0, 0.0, 1.0e12)
            .integer(Page::Main, "trials1", "Sample 1 trials", 0.0, 0.0, 1.0e12)
            .integer(Page::Main, "events2", "Sample 2 events", 0.0, 0.0, 1.0e12)
            .integer(Page::Main, "trials2", "Sample 2 trials", 0.0, 0.0, 1.0e12)
            .vars("first", "Sample 1 column", Any, 0, 1)
            .vars("second", "Sample 2 column", Any, 0, 1)
            .vars("resp", "Outcome column", Any, 0, 1)
            .vars("samples", "Samples column", Any, 0, 1)
            .text(Page::Main, "event", "Event (blank = last value in sorted order)", "")
            .number(Options, "diff", "Hypothesized difference", 0.0, -1.0, 1.0)
            .choice(Options, "alt", "Alternative", &ALTS, 0)
            .check(Options, "cc", "Use continuity correction for the normal test", false),
        ),
        &path("2 Proportions..."),
        Runner::Report(prop2),
    );
    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.poisson1",
                "1-Sample Poisson Rate",
                "Exact and normal tests and intervals for a Poisson rate.",
            )
            .choice(Page::Main, "mode", "Data are", &["Summarized data", "A column of counts"], 0)
            .integer(Page::Main, "total", "Total number of occurrences", 0.0, 0.0, 1.0e12)
            .vars("var", "Column of counts", Numeric, 0, 1)
            .number(Page::Main, "length", "Length of observation (per count)", 1.0, 1e-12, 1.0e12)
            .number(Options, "rate0", "Hypothesized rate", 1.0, 1e-12, 1.0e12)
            .choice(Options, "alt", "Alternative", &ALTS, 0),
        ),
        &path("1-Sample Poisson Rate..."),
        Runner::Report(poisson1),
    );
    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.poisson2",
                "2-Sample Poisson Rate",
                "Compare two Poisson rates: z test, exact conditional test and rate-ratio interval.",
            )
            .choice(Page::Main, "mode", "Data are", &["Summarized data", "Each sample in its own column"], 0)
            .integer(Page::Main, "total1", "Sample 1 occurrences", 0.0, 0.0, 1.0e12)
            .number(Page::Main, "length1", "Sample 1 length of observation", 1.0, 1e-12, 1.0e12)
            .integer(Page::Main, "total2", "Sample 2 occurrences", 0.0, 0.0, 1.0e12)
            .number(Page::Main, "length2", "Sample 2 length of observation", 1.0, 1e-12, 1.0e12)
            .vars("first", "Sample 1 counts", Numeric, 0, 1)
            .vars("second", "Sample 2 counts", Numeric, 0, 1)
            .number(Options, "diff", "Hypothesized difference", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .choice(Options, "alt", "Alternative", &ALTS, 0),
        ),
        &path("2-Sample Poisson Rate..."),
        Runner::Report(poisson2),
    );
    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.var2",
                "2 Variances",
                "F test for the ratio of two variances together with Levene's test.",
            )
            .choice(
                Page::Main,
                "mode",
                "Data are",
                &["Each sample in its own column", "Response in one column, samples in another"],
                0,
            )
            .vars("first", "Sample 1", Numeric, 0, 1)
            .vars("second", "Sample 2", Numeric, 0, 1)
            .vars("resp", "Response", Numeric, 0, 1)
            .vars("samples", "Samples", Any, 0, 1)
            .number(Options, "ratio", "Hypothesized ratio of variances", 1.0, 1e-9, 1.0e9)
            .choice(Options, "alt", "Alternative", &ALTS, 0),
        ),
        &path("2 Variances..."),
        Runner::Report(var2),
    );
    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.eqvar",
                "Test for Equal Variances",
                "Bonett's and Levene's tests that several groups have equal variances.",
            )
            .vars("resp", "Response", Numeric, 1, 1)
            .vars("samples", "Factor", Any, 1, 1),
        ),
        &path("Test for Equal Variances..."),
        Runner::Report(eqvar),
    );
    add(
        r,
        DialogSpec::new(
            "stat.basic.outlier",
            "Outlier Test",
            "Grubbs, Dixon or generalized ESD test for outliers in a column.",
        )
        .vars("var", "Variable", Numeric, 1, 1)
        .choice(Options, "method", "Test", &["Grubbs", "Dixon's Q", "Generalized ESD (several outliers)"], 0)
        .choice(
            Options,
            "side",
            "Look for outliers at (Grubbs, Dixon)",
            &["either end", "the high end", "the low end"],
            0,
        )
        .integer(Options, "max", "Maximum number of outliers (ESD)", 3.0, 1.0, 1000.0)
        .number(Options, "alpha", "Significance level (ESD)", 0.05, 1e-6, 0.5),
        &path("Outlier Test..."),
        Runner::Report(outlier),
    );
    add(
        r,
        DialogSpec::new(
            "stat.basic.equiv",
            "Equivalence Test",
            "Two one-sided tests (TOST) that a mean or mean difference lies within equivalence limits.",
        )
        .choice(Page::Main, "kind", "Test", &["1-sample against a target", "Paired", "2-sample"], 0)
        .vars("first", "Sample (or first sample)", Numeric, 1, 1)
        .vars("second", "Second sample", Numeric, 0, 1)
        .number(Options, "target", "Target (1-sample)", 0.0, f64::NEG_INFINITY, f64::INFINITY)
        .number(Options, "lower", "Lower equivalence limit (difference)", -1.0, f64::NEG_INFINITY, f64::INFINITY)
        .number(Options, "upper", "Upper equivalence limit (difference)", 1.0, f64::NEG_INFINITY, f64::INFINITY)
        .number(Options, "alpha", "Significance level of each one-sided test", 0.05, 1e-6, 0.4999)
        .check(Options, "pooled", "Assume equal variances (2-sample)", false),
        &path("Equivalence Test..."),
        Runner::Report(equivalence),
    );
    add(
        r,
        DialogSpec::new(
            "stat.basic.tolerance",
            "Tolerance Interval",
            "An interval that covers a stated proportion of the population with a stated confidence.",
        )
        .vars("var", "Variable", Numeric, 1, 1)
        .number(Options, "coverage", "Minimum percent of the population covered", 99.0, 0.001, 99.999)
        .number(Options, "confidence", "Confidence level (%)", 95.0, 0.001, 99.999)
        .choice(Options, "sides", "Interval", &["two-sided", "upper bound", "lower bound"], 0)
        .choice(
            Options,
            "method",
            "Method",
            &["Normal (exact factor)", "Normal (Howe's approximation)", "Nonparametric (order statistics)"],
            0,
        ),
        &path("Tolerance Interval..."),
        Runner::Report(tolerance),
    );
    add(
        r,
        DialogSpec::new(
            "stat.basic.crosstab",
            "Cross Tabulation and Chi-Square",
            "Counts, percentages and tests of association for two categorical variables or a table of counts.",
        )
        .choice(
            Page::Main,
            "mode",
            "Data are",
            &["Two columns of categories", "A table of counts (one column per table column)"],
            0,
        )
        .vars("rowvar", "Rows", Any, 0, 1)
        .vars("colvar", "Columns", Any, 0, 1)
        .vars("counts", "Columns containing the table", Numeric, 0, 50)
        .check(Options, "chi", "Pearson chi-square test", true)
        .check(Options, "lr", "Likelihood-ratio chi-square test", false)
        .check(Options, "trend", "Mantel-Haenszel linear-by-linear test", false)
        .check(Options, "fisher", "Fisher's exact test", false)
        .check(Options, "mcnemar", "McNemar's test (paired 2 x 2)", false)
        .check(Options, "assoc", "Phi, Cramer's V, contingency coefficient", false)
        .check(Options, "row_pct", "Row percentages", false)
        .check(Options, "col_pct", "Column percentages", false)
        .check(Options, "tot_pct", "Total percentages", false)
        .check(Options, "expected", "Expected counts", false)
        .check(Options, "residuals", "Residuals", false),
        &path("Cross Tabulation and Chi-Square..."),
        Runner::Report(crosstab),
    );
    add(
        r,
        DialogSpec::new(
            "stat.basic.gof",
            "Chi-Square Goodness-of-Fit Test",
            "Compare observed category counts with equal, hypothesized or historical proportions.",
        )
        .vars("observed", "Observed counts", Numeric, 1, 1)
        .vars("names", "Category names (optional)", Any, 0, 1)
        .choice(
            Options,
            "expected",
            "Test",
            &["Equal proportions", "Specific proportions in a column", "Proportional to historical counts in a column"],
            0,
        )
        .vars("expcol", "Expected column", Numeric, 0, 1)
        .integer(Options, "params", "Parameters estimated from the data", 0.0, 0.0, 100.0),
        &path("Chi-Square Goodness-of-Fit Test..."),
        Runner::Report(gof),
    );
    add(
        r,
        DialogSpec::new("stat.basic.covariance", "Covariance", "Covariance matrix of numeric columns.")
            .vars("vars", "Variables", Numeric, 2, 50)
            .choice(
                Options,
                "missing",
                "Missing values",
                &["Delete rows with any missing value", "Use all available pairs"],
                0,
            ),
        &path("Covariance..."),
        Runner::Report(covariance),
    );
    add(
        r,
        with_level(
            DialogSpec::new(
                "stat.basic.partial",
                "Partial Correlation",
                "Correlation of each pair of variables after removing the linear effect of control variables.",
            )
            .vars("vars", "Variables", Numeric, 2, 50)
            .vars("controls", "Control variables", Numeric, 1, 50),
        ),
        &path("Partial Correlation..."),
        Runner::Report(partial),
    );
}
