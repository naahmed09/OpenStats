//! Nonparametric tests (Stat ▸ Nonparametrics).
//!
//! Two-sample and one-sample tests take columns; k-sample tests take a response column and a
//! factor column (any type) that names the group of each row. Rows with a missing response are
//! ignored. A paired analysis is a one-sample analysis of the differences: choose a second column
//! and the test uses `first − second`.

use super::analyses::{add, out};
use super::analysis::{col_title, labels, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::Worksheet;
use openstats_stats::hypothesis::Alternative;
use openstats_stats::nonparametric::{
    ansari_bradley, friedman, kruskal_wallis, ks_two_sample, mann_kendall, mann_whitney, mood_median, mood_scale,
    normal_scores, runs_test, sign_test, wilcoxon_signed_rank, SignRankMethod,
};
use openstats_stats::report::Reportable;

const ALTS: [&str; 3] = ["not equal", "less than", "greater than"];
const METHODS: [&str; 3] = ["Automatic (exact when possible)", "Exact", "Normal approximation"];

fn alt(v: &Values) -> Alternative {
    match v.choice("alt") {
        1 => Alternative::Less,
        2 => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

fn method(v: &Values) -> SignRankMethod {
    match v.choice("method") {
        1 => SignRankMethod::Exact,
        2 => SignRankMethod::Normal,
        _ => SignRankMethod::Auto,
    }
}

/// The values to test and their title: one column, or `first − second` row by row.
fn one_or_paired(ws: &Worksheet, v: &Values) -> Res<(String, Vec<f64>)> {
    let a = v.var("var").ok_or("select a column")?;
    let x = numeric(ws, a)?;
    match v.var("minus") {
        None => Ok((col_title(ws, a), x)),
        Some(b) => {
            let y = numeric(ws, b)?;
            if x.len() != y.len() {
                return Err(format!("{} and {} have different lengths", col_title(ws, a), col_title(ws, b)).into());
            }
            let d = x.iter().zip(&y).map(|(p, q)| p - q).collect();
            Ok((format!("{} - {}", col_title(ws, a), col_title(ws, b)), d))
        }
    }
}

/// Groups of `response` by the levels of `factor`, in order of first appearance.
fn grouped(ws: &Worksheet, response: usize, factor: usize) -> Res<(Vec<String>, Vec<Vec<f64>>)> {
    let y = numeric(ws, response)?;
    let f = labels(ws, factor)?;
    if y.len() != f.len() {
        return Err("the response and the factor columns have different lengths".into());
    }
    let mut names: Vec<String> = Vec::new();
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for (i, level) in f.iter().enumerate() {
        if level.is_empty() || y[i].is_nan() {
            continue;
        }
        match names.iter().position(|n| n == level) {
            Some(k) => groups[k].push(y[i]),
            None => {
                names.push(level.clone());
                groups.push(vec![y[i]]);
            }
        }
    }
    if groups.len() < 2 {
        return Err(format!("{} has fewer than two levels with data", col_title(ws, factor)).into());
    }
    Ok((names, groups))
}

fn sign_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, d) = one_or_paired(ws, v)?;
    let r = sign_test(&d, v.num("median"), alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Sign Test for Median: {name}");
    out(s)
}

fn wilcoxon_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, d) = one_or_paired(ws, v)?;
    let r = wilcoxon_signed_rank(&d, v.num("median"), alt(v), level(v)?, method(v), v.flag("correction"))?;
    let mut s = r.report_section();
    s.heading = format!("Wilcoxon Signed Rank Test: {name}");
    out(s)
}

fn mann_whitney_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (a, b) = (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
    let r = mann_whitney(&numeric(ws, a)?, &numeric(ws, b)?, alt(v), level(v)?, method(v), v.flag("correction"))?;
    let mut s = r.report_section();
    s.heading = format!("Mann-Whitney: {} vs {}", col_title(ws, a), col_title(ws, b));
    out(s)
}

fn kruskal_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, f) = (v.var("response").ok_or("select a response")?, v.var("factor").ok_or("select a factor")?);
    let (names, groups) = grouped(ws, y, f)?;
    let r = kruskal_wallis(&groups)?;
    let mut s = r.section_with_labels(&names);
    s.heading = format!("Kruskal-Wallis Test: {} versus {}", col_title(ws, y), col_title(ws, f));
    out(s)
}

fn mood_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, f) = (v.var("response").ok_or("select a response")?, v.var("factor").ok_or("select a factor")?);
    let (names, groups) = grouped(ws, y, f)?;
    let r = mood_median(&groups)?;
    let mut s = r.section_with_labels(&names);
    s.heading = format!("Mood Median Test: {} versus {}", col_title(ws, y), col_title(ws, f));
    out(s)
}

fn two_columns(ws: &Worksheet, v: &Values) -> Res<(String, Vec<f64>, Vec<f64>)> {
    let (a, b) = (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
    Ok((format!("{} vs {}", col_title(ws, a), col_title(ws, b)), numeric(ws, a)?, numeric(ws, b)?))
}

fn ks_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, x, y) = two_columns(ws, v)?;
    let mut s = ks_two_sample(&x, &y, alt(v))?.report_section();
    s.heading = format!("Two-Sample Kolmogorov-Smirnov: {name}");
    out(s)
}

fn ansari_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, x, y) = two_columns(ws, v)?;
    let mut s = ansari_bradley(&x, &y)?.report_section();
    s.heading = format!("Ansari-Bradley Test: {name}");
    out(s)
}

fn mood_scale_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (name, x, y) = two_columns(ws, v)?;
    let mut s = mood_scale(&x, &y)?.report_section();
    s.heading = format!("Mood's Scale Test: {name}");
    out(s)
}

fn normal_scores_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, f) = (v.var("response").ok_or("select a response")?, v.var("factor").ok_or("select a factor")?);
    let (names, groups) = grouped(ws, y, f)?;
    let pairs: Vec<(&str, &[f64])> = names.iter().map(String::as_str).zip(groups.iter().map(Vec::as_slice)).collect();
    let mut s = normal_scores(&pairs)?.report_section();
    s.heading = format!("Normal Scores Test: {} versus {}", col_title(ws, y), col_title(ws, f));
    out(s)
}

fn mann_kendall_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let a = v.var("var").ok_or("select a column")?;
    let mut s = mann_kendall(&numeric(ws, a)?, alt(v))?.report_section();
    s.heading = format!("Mann-Kendall Trend Test: {}", col_title(ws, a));
    out(s)
}

fn friedman_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let (t, b) = (v.var("treatment").ok_or("select a treatment")?, v.var("block").ok_or("select a block")?);
    let (resp, tl, bl) = (numeric(ws, y)?, labels(ws, t)?, labels(ws, b)?);
    if resp.len() != tl.len() || resp.len() != bl.len() {
        return Err("the response, treatment and block columns have different lengths".into());
    }
    let (mut treatments, mut blocks): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for i in 0..resp.len() {
        if tl[i].is_empty() || bl[i].is_empty() {
            continue;
        }
        if !treatments.contains(&tl[i]) {
            treatments.push(tl[i].clone());
        }
        if !blocks.contains(&bl[i]) {
            blocks.push(bl[i].clone());
        }
    }
    if treatments.len() < 2 || blocks.len() < 2 {
        return Err("the Friedman test needs at least two treatments and two blocks".into());
    }
    let mut cell: Vec<Vec<Option<f64>>> = vec![vec![None; treatments.len()]; blocks.len()];
    for i in 0..resp.len() {
        if tl[i].is_empty() || bl[i].is_empty() {
            continue;
        }
        let (r, c) = (
            blocks.iter().position(|x| *x == bl[i]).unwrap_or(0),
            treatments.iter().position(|x| *x == tl[i]).unwrap_or(0),
        );
        if cell[r][c].is_some() {
            return Err(format!("block {} has more than one observation for treatment {}", bl[i], tl[i]).into());
        }
        cell[r][c] = Some(resp[i]);
    }
    let mut rows = Vec::new();
    for (r, row) in cell.iter().enumerate() {
        match row.iter().map(|x| x.ok_or(())).collect::<Result<Vec<f64>, ()>>() {
            Ok(v) => rows.push(v),
            Err(()) => {
                let c = row.iter().position(Option::is_none).unwrap_or(0);
                return Err(format!("block {} has no observation for treatment {}", blocks[r], treatments[c]).into());
            }
        }
    }
    let r = friedman(&rows)?;
    let mut s = r.section_with_labels(&treatments);
    s.heading =
        format!("Friedman Test: {} versus {} blocked by {}", col_title(ws, y), col_title(ws, t), col_title(ws, b));
    out(s)
}

fn runs_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let x = numeric(ws, c)?;
    let k = match v.choice("about") {
        0 => {
            let ok: Vec<f64> = x.iter().copied().filter(|a| !a.is_nan()).collect();
            if ok.is_empty() {
                return Err("the column has no data".into());
            }
            ok.iter().sum::<f64>() / ok.len() as f64
        }
        1 => openstats_stats::quantile::median(&x)?,
        _ => v.num("k"),
    };
    let r = runs_test(&x, k)?;
    let mut s = r.report_section();
    s.heading = format!("Runs Test: {}", col_title(ws, c));
    out(s)
}

pub(crate) fn register(r: &mut Registry) {
    use Page::Options;
    let menu = |last: &'static str| ["Stat", "Nonparametrics", last];
    let level_field = |s: DialogSpec| s.number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999);
    let one_sample = |s: DialogSpec| {
        level_field(
            s.vars("var", "Variable", VarKind::Numeric, 1, 1)
                .vars("minus", "Subtract this column for paired data (optional)", VarKind::Numeric, 0, 1)
                .number(Options, "median", "Test median", 0.0, f64::NEG_INFINITY, f64::INFINITY)
                .choice(Options, "alt", "Alternative", &ALTS, 0),
        )
    };
    add(
        r,
        one_sample(DialogSpec::new(
            "stat.nonpar.sign",
            "1-Sample Sign",
            "Sign test of the median with an exact binomial p-value and a confidence interval for the median.",
        )),
        &menu("1-Sample Sign..."),
        Runner::Report(sign_run),
    );
    add(
        r,
        one_sample(DialogSpec::new(
            "stat.nonpar.wilcoxon",
            "1-Sample Wilcoxon",
            "Wilcoxon signed-rank test with the Hodges-Lehmann estimate and interval.",
        ))
        .choice(Options, "method", "P-value method", &METHODS, 0)
        .check(Options, "correction", "Continuity correction (normal approximation)", false),
        &menu("1-Sample Wilcoxon..."),
        Runner::Report(wilcoxon_run),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.nonpar.mannwhitney",
                "Mann-Whitney",
                "Compare two populations with the Mann-Whitney (rank-sum) test and estimate the difference.",
            )
            .vars("first", "First sample", VarKind::Numeric, 1, 1)
            .vars("second", "Second sample", VarKind::Numeric, 1, 1)
            .choice(Options, "alt", "Alternative", &ALTS, 0)
            .choice(Options, "method", "P-value method", &METHODS, 0)
            .check(Options, "correction", "Continuity correction (normal approximation)", false),
        ),
        &menu("Mann-Whitney..."),
        Runner::Report(mann_whitney_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.ks2",
            "2-Sample Kolmogorov-Smirnov",
            "Compare two distributions with the two-sample Kolmogorov-Smirnov test.",
        )
        .vars("first", "First sample", VarKind::Numeric, 1, 1)
        .vars("second", "Second sample", VarKind::Numeric, 1, 1)
        .choice(Options, "alt", "Alternative", &ALTS, 0),
        &menu("2-Sample Kolmogorov-Smirnov..."),
        Runner::Report(ks_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.ansari",
            "Ansari-Bradley",
            "Test whether two samples have equal scale (dispersion).",
        )
        .vars("first", "First sample", VarKind::Numeric, 1, 1)
        .vars("second", "Second sample", VarKind::Numeric, 1, 1),
        &menu("Ansari-Bradley..."),
        Runner::Report(ansari_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.moodscale",
            "Mood Scale",
            "Test whether two samples have equal scale with Mood's test.",
        )
        .vars("first", "First sample", VarKind::Numeric, 1, 1)
        .vars("second", "Second sample", VarKind::Numeric, 1, 1),
        &menu("Mood Scale..."),
        Runner::Report(mood_scale_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.normalscores",
            "Normal Scores",
            "van der Waerden normal-scores test for several populations.",
        )
        .vars("response", "Response", VarKind::Numeric, 1, 1)
        .vars("factor", "Factor", VarKind::Any, 1, 1),
        &menu("Normal Scores..."),
        Runner::Report(normal_scores_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.mannkendall",
            "Mann-Kendall Trend",
            "Test a time-ordered series for a monotone trend; reports Sen's slope.",
        )
        .vars("var", "Series", VarKind::Numeric, 1, 1)
        .choice(Options, "alt", "Alternative", &ALTS, 0),
        &menu("Mann-Kendall Trend..."),
        Runner::Report(mann_kendall_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.kruskal",
            "Kruskal-Wallis",
            "Test whether several populations have the same median using ranks.",
        )
        .vars("response", "Response", VarKind::Numeric, 1, 1)
        .vars("factor", "Factor", VarKind::Any, 1, 1),
        &menu("Kruskal-Wallis..."),
        Runner::Report(kruskal_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.mood",
            "Mood's Median Test",
            "Test whether several populations have the same median using counts above and below the overall median.",
        )
        .vars("response", "Response", VarKind::Numeric, 1, 1)
        .vars("factor", "Factor", VarKind::Any, 1, 1),
        &menu("Mood's Median Test..."),
        Runner::Report(mood_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.friedman",
            "Friedman",
            "Rank test for treatment differences in a randomized block design with one observation per cell.",
        )
        .vars("response", "Response", VarKind::Numeric, 1, 1)
        .vars("treatment", "Treatment", VarKind::Any, 1, 1)
        .vars("block", "Blocks", VarKind::Any, 1, 1),
        &menu("Friedman..."),
        Runner::Report(friedman_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.nonpar.runs",
            "Runs Test",
            "Test whether the order of the observations is random about a reference value.",
        )
        .vars("var", "Variable", VarKind::Numeric, 1, 1)
        .choice(Options, "about", "Test for runs above and below", &["Mean", "Median", "Value K"], 0)
        .number(Options, "k", "Value K", 0.0, f64::NEG_INFINITY, f64::INFINITY),
        &menu("Runs Test..."),
        Runner::Report(runs_run),
    );
}
