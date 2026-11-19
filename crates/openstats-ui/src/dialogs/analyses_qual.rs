//! Quality tools: control charts (variables, attributes, rare events, time-weighted,
//! multivariate), non-normal / empirical capability, the capability sixpack, distribution
//! identification and the measurement-system studies. Each dialog wraps a routine of
//! `openstats_stats::spc`, `::capability` or `::msa`.

use super::analyses::{add, out};
use super::analysis::{col_title, labels, level, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::capability::{
    capability_empirical, capability_intervals, extended_sixpack_individuals, extended_sixpack_subgroups,
    identify_distributions, intervals_section, pooled_within_df, ranking_section, EmpiricalOptions,
};
use openstats_stats::msa::{
    AttributeOptions, AttributeStudy, GageOptions, LinearityBiasOptions, LinearityBiasStudy, NestedStudy, Tolerance,
};
use openstats_stats::report::{Reportable, Section};
use openstats_stats::spc::attributes::{
    c_chart_with, g_chart, laney_p, laney_u, np_chart_with, p_chart_with, t_chart_with, u_chart_with, LimitMethod,
    TimeModel,
};
use openstats_stats::spc::capability::{
    capability_distribution, capability_transformed_individuals, capability_transformed_subgroups, CapabilityOptions,
    Family, Specs,
};
use openstats_stats::spc::charts::{i_mr, xbar_r, xbar_s, ChartOptions, ChartPanel, IndividualsSigma, SubgroupSigma};
use openstats_stats::spc::multivariate::generalized_variance;
use openstats_stats::spc::rules::SpecialCauseTests;
use openstats_stats::spc::sixpack::{capability_sixpack_individuals, capability_sixpack_subgroups, SixpackOptions};
use openstats_stats::spc::stages::{staged_c, staged_i_mr, staged_np, staged_p, staged_u, staged_xbar};
use openstats_stats::spc::time_weighted::{
    cusum_tabular_sided, cusum_vmask, ewma, moving_average, CusumOptions, CusumSides, TimeWeightedInput,
};
use openstats_stats::spc::transform::{fit_johnson, BoxCox, LambdaChoice, Transformation};
use openstats_stats::spc::variables_ext::{
    boxcox_i_mr, boxcox_xbar, i_mr_r_s, z_mr, zone_chart_individuals, zone_chart_subgroups, ShortRunSigma, WithinChart,
    ZoneOptions,
};

const VAR_RULES: [&str; 4] =
    ["Test 1 only", "All 8 tests (Nelson)", "Western Electric", "Western Electric with supplementary tests"];
const ATTR_RULES: [&str; 2] = ["Test 1 only", "Tests 1-4"];

// ----------------------------------------------------------------------------- shared helpers

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Individuals,
    Subgroup,
    Attribute,
}

/// Option fields shared by the control-chart dialogs, other than the sigma multiple `k`.
pub(super) fn ctrl_extra(spec: DialogSpec, kind: Kind) -> DialogSpec {
    use Page::{Options, Storage};
    let mut s = match kind {
        Kind::Attribute => spec.choice(Options, "rules", "Tests for special causes", &ATTR_RULES, 0),
        _ => spec.choice(Options, "rules", "Tests for special causes", &VAR_RULES, 0),
    };
    match kind {
        Kind::Individuals => {
            s = s
                .choice(
                    Options,
                    "sigma_est",
                    "Estimate sigma from",
                    &["Average moving range", "Median moving range", "Square root of MSSD"],
                    0,
                )
                .integer(Options, "span", "Moving range length", 2.0, 2.0, 50.0);
        }
        Kind::Subgroup => {
            s = s.choice(
                Options,
                "sigma_est",
                "Estimate sigma from",
                &["Default (R-bar or S-bar)", "R-bar", "S-bar", "Pooled standard deviation"],
                0,
            );
        }
        Kind::Attribute => {}
    }
    if kind != Kind::Attribute {
        s = s.optional_number(Options, "known_center", "Historical mean (blank = estimate)").optional_number(
            Options,
            "known_sigma",
            "Historical standard deviation (blank = estimate)",
        );
    }
    s.vars("stages", "Stage column (separate limits per stage)", VarKind::Any, 0, 1).check(
        Storage,
        "store_chart",
        "Center line, limits and flagged points",
        false,
    )
}

fn ctrl_full(spec: DialogSpec, kind: Kind) -> DialogSpec {
    ctrl_extra(spec.number(Page::Options, "k", "Sigma limit position", 3.0, 0.5, 10.0), kind)
}

pub(super) fn chart_options_ext(v: &Values, kind: Kind) -> Res<ChartOptions> {
    let mut o = ChartOptions::default();
    let k = v.num("k");
    if k.is_finite() {
        o.k_sigma = k;
    }
    o.tests = match (v.choice("rules"), kind) {
        (1, Kind::Attribute) => SpecialCauseTests::first_four(),
        (1, _) => SpecialCauseTests::nelson(),
        (2, Kind::Attribute) => SpecialCauseTests::first_four(),
        (2, _) => SpecialCauseTests::western_electric(),
        (3, Kind::Attribute) => SpecialCauseTests::first_four(),
        (3, _) => SpecialCauseTests::western_electric_supplementary(),
        _ if v.flag("all_tests") => SpecialCauseTests::nelson(),
        _ => SpecialCauseTests::test1_only(),
    };
    o.tests.beyond_sigma = o.tests.beyond_sigma.map(|_| o.k_sigma);
    o.dispersion_tests = SpecialCauseTests::test1_only();
    o.dispersion_tests.beyond_sigma = Some(o.k_sigma);
    let c = v.num("known_center");
    if c.is_finite() {
        o.known_center = Some(c);
    }
    let s = v.num("known_sigma");
    if s.is_finite() {
        o.known_sigma = Some(s);
    }
    if kind == Kind::Individuals {
        o.individuals_sigma = match v.choice("sigma_est") {
            1 => IndividualsSigma::MedianMovingRange,
            2 => IndividualsSigma::Mssd,
            _ => IndividualsSigma::AverageMovingRange,
        };
        let span = v.count("span");
        if span >= 2 {
            o.span = span;
        }
    }
    if kind == Kind::Subgroup {
        o.subgroup_sigma = match v.choice("sigma_est") {
            1 => Some(SubgroupSigma::RBar),
            2 => Some(SubgroupSigma::SBar),
            3 => Some(SubgroupSigma::Pooled),
            _ => None,
        };
    }
    Ok(o)
}

/// The stored columns of a chart panel: center line, limits and 1/0 for flagged points.
fn panel_cols(suffix: &str, p: &ChartPanel) -> Vec<Column> {
    let mut flag = vec![0.0; p.values.len()];
    for i in p.flagged() {
        if i < flag.len() {
            flag[i] = 1.0;
        }
    }
    vec![
        Column::numeric(format!("CENTER{suffix}"), p.center.clone()),
        Column::numeric(format!("UCL{suffix}"), p.ucl.clone()),
        Column::numeric(format!("LCL{suffix}"), p.lcl.clone()),
        Column::numeric(format!("FLAGGED{suffix}"), flag),
    ]
}

fn finish(s: Section, v: &Values, cols: impl FnOnce() -> Vec<Column>) -> Res<Output> {
    Ok(Output { section: s, stored: if v.flag("store_chart") { cols() } else { Vec::new() }, models: Vec::new() })
}

/// Stage labels aligned with `rows` (one label per chart point), when a stage column is chosen.
fn stage_labels(ws: &Worksheet, v: &Values, rows: &[usize]) -> Res<Option<Vec<String>>> {
    let Some(c) = v.var("stages") else { return Ok(None) };
    let all = labels(ws, c)?;
    rows.iter()
        .map(|&r| {
            all.get(r).cloned().ok_or_else(|| DialogError(format!("{} is shorter than the data", col_title(ws, c))))
        })
        .collect::<Res<Vec<_>>>()
        .map(Some)
}

/// Subgroups of the data column: by a subgroup-id column, or consecutive blocks of `size`.
/// Missing values are dropped; the second result is the first worksheet row of each subgroup.
fn subgroups(ws: &Worksheet, v: &Values, c: usize) -> Res<(Vec<Vec<f64>>, Vec<usize>)> {
    let data = numeric(ws, c)?;
    let mut groups: Vec<Vec<f64>> = Vec::new();
    let mut first: Vec<usize> = Vec::new();
    if let Some(id) = v.var("ids") {
        let ids = labels(ws, id)?;
        if ids.len() != data.len() {
            return Err("the subgroup column and the data column have different lengths".into());
        }
        let mut names: Vec<String> = Vec::new();
        for (i, (x, g)) in data.iter().zip(&ids).enumerate() {
            if x.is_nan() {
                continue;
            }
            let k = names.iter().position(|n| n == g).unwrap_or_else(|| {
                names.push(g.clone());
                groups.push(Vec::new());
                first.push(i);
                names.len() - 1
            });
            groups[k].push(*x);
        }
    } else {
        let size = v.count("size").max(1);
        for (b, chunk) in data.chunks(size).enumerate() {
            let g: Vec<f64> = chunk.iter().copied().filter(|x| !x.is_nan()).collect();
            if !g.is_empty() {
                groups.push(g);
                first.push(b * size);
            }
        }
    }
    if groups.is_empty() {
        return Err(format!("{} has no data", col_title(ws, c)).into());
    }
    Ok((groups, first))
}

fn with_subgroup_fields(spec: DialogSpec) -> DialogSpec {
    spec.integer(Page::Main, "size", "Subgroup size (1 = individuals)", 1.0, 1.0, 1000.0).vars(
        "ids",
        "Subgroup identifiers (column; overrides the size)",
        VarKind::Any,
        0,
        1,
    )
}

// -------------------------------------------------------------------------- variables charts

pub(super) fn imr_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let data = numeric(ws, c)?;
    let o = chart_options_ext(v, Kind::Individuals)?;
    let rows: Vec<usize> = (0..data.len()).collect();
    if let Some(st) = stage_labels(ws, v, &rows)? {
        let chart = staged_i_mr(&data, &st, &o)?;
        let mut s = chart.report_section();
        s.heading = format!("I-MR Chart of {} by stage", col_title(ws, c));
        return finish(s, v, || {
            let mut cols = panel_cols("", &chart.location);
            cols.extend(panel_cols("_MR", &chart.dispersion));
            cols
        });
    }
    let chart = i_mr(&data, &o)?;
    let mut s = chart.report_section();
    s.heading = format!("I-MR Chart of {}", col_title(ws, c));
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.location);
        cols.extend(panel_cols("_MR", &chart.dispersion));
        cols
    })
}

pub(super) fn xbar_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let (groups, first) = subgroups(ws, v, c)?;
    let o = chart_options_ext(v, Kind::Subgroup)?;
    let range = v.choice("sigma") == 0;
    let name = if range { "R" } else { "S" };
    if let Some(st) = stage_labels(ws, v, &first)? {
        let chart = staged_xbar(&groups, &st, range, &o)?;
        let mut s = chart.report_section();
        s.heading = format!("Xbar-{name} Chart of {} by stage", col_title(ws, c));
        return finish(s, v, || {
            let mut cols = panel_cols("", &chart.location);
            cols.extend(panel_cols(&format!("_{name}"), &chart.dispersion));
            cols
        });
    }
    let chart = if range { xbar_r(&groups, &o)? } else { xbar_s(&groups, &o)? };
    let mut s = chart.report_section();
    s.heading = format!("Xbar-{name} Chart of {}", col_title(ws, c));
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.location);
        cols.extend(panel_cols(&format!("_{name}"), &chart.dispersion));
        cols
    })
}

fn between_within(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let (groups, _) = subgroups(ws, v, c)?;
    let o = chart_options_ext(v, Kind::Subgroup)?;
    let w = if v.choice("within") == 0 { WithinChart::Range } else { WithinChart::StDev };
    let chart = i_mr_r_s(&groups, w, &o)?;
    let mut s = chart.report_section();
    s.heading = format!("I-MR-{} Chart of {}", if v.choice("within") == 0 { "R" } else { "S" }, col_title(ws, c));
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.means);
        cols.extend(panel_cols("_MR", &chart.moving_range));
        cols.extend(panel_cols("_W", &chart.within));
        cols
    })
}

fn zone_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let o = chart_options_ext(v, Kind::Subgroup)?;
    let z = ZoneOptions { signal_at: v.num("signal_at"), ..Default::default() };
    let size = v.count("size").max(1);
    let chart = if size <= 1 && v.var("ids").is_none() {
        let mut oi = chart_options_ext(v, Kind::Individuals)?;
        oi.tests = o.tests.clone();
        zone_chart_individuals(&numeric(ws, c)?, &oi, &z)?
    } else {
        let (groups, _) = subgroups(ws, v, c)?;
        zone_chart_subgroups(&groups, &o, &z)?
    };
    let mut s = chart.report_section();
    s.heading = format!("Zone Chart of {}", col_title(ws, c));
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.panel);
        cols.push(Column::numeric("ZONESCORE", chart.scores.clone()));
        cols
    })
}

fn zmr_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let p = v.var("parts").ok_or("select the part column")?;
    let data = numeric(ws, c)?;
    let parts = labels(ws, p)?;
    if parts.len() != data.len() {
        return Err("the part column and the data column have different lengths".into());
    }
    let sig = if v.choice("sigma_by") == 0 { ShortRunSigma::Pooled } else { ShortRunSigma::ByPart };
    let o = chart_options_ext(v, Kind::Individuals)?;
    let chart = z_mr(&data, &parts, None, &sig, &o)?;
    let mut s = chart.report_section();
    s.heading = format!("Z-MR Chart of {} by {}", col_title(ws, c), col_title(ws, p));
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.chart.location);
        cols.extend(panel_cols("_MR", &chart.chart.dispersion));
        cols.push(Column::numeric("ZSCORE", chart.z.clone()));
        cols
    })
}

fn boxcox_chart(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let choice = lambda_choice(v);
    let size = v.count("size").max(1);
    if size <= 1 && v.var("ids").is_none() {
        let o = chart_options_ext(v, Kind::Individuals)?;
        let chart = boxcox_i_mr(&numeric(ws, c)?, choice, &o)?;
        let mut s = chart.report_section();
        s.heading = format!("Box-Cox Transformation I-MR Chart of {}", col_title(ws, c));
        return finish(s, v, || {
            let mut cols = panel_cols("", &chart.chart.location);
            cols.extend(panel_cols("_MR", &chart.chart.dispersion));
            cols
        });
    }
    let (groups, _) = subgroups(ws, v, c)?;
    let o = chart_options_ext(v, Kind::Subgroup)?;
    let chart = boxcox_xbar(&groups, v.choice("sigma") == 0, choice, &o)?;
    let mut s = chart.report_section();
    s.heading = format!("Box-Cox Transformation Xbar Chart of {}", col_title(ws, c));
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.chart.location);
        cols.extend(panel_cols("_D", &chart.chart.dispersion));
        cols
    })
}

fn lambda_choice(v: &Values) -> LambdaChoice {
    let l = v.num("lambda");
    if l.is_finite() {
        LambdaChoice::Fixed(l)
    } else if v.flag("round_lambda") {
        LambdaChoice::Rounded
    } else {
        LambdaChoice::Optimal
    }
}

// -------------------------------------------------------------------------- attribute charts

/// Counts, sizes and the worksheet rows that have both (a constant size fills the sizes).
type AttrData = (Vec<f64>, Vec<f64>, Vec<usize>, String);

fn attribute_data(ws: &Worksheet, v: &Values, need_sizes: bool) -> Res<AttrData> {
    let c = v.var("var").ok_or("select a column")?;
    let counts = numeric(ws, c)?;
    let (sizes, label): (Vec<f64>, String) = match v.var("sizes") {
        Some(sc) => (numeric(ws, sc)?, col_title(ws, sc)),
        None if need_sizes => {
            let n = v.num("size");
            if !(n.is_finite() && n > 0.0) {
                return Err("give the subgroup size, or choose a column of sizes".into());
            }
            (vec![n; counts.len()], format!("{n}"))
        }
        None => (vec![1.0; counts.len()], String::new()),
    };
    if sizes.len() != counts.len() {
        return Err("the counts and the sizes have different lengths".into());
    }
    let rows: Vec<usize> = (0..counts.len()).filter(|&i| !counts[i].is_nan() && !sizes[i].is_nan()).collect();
    if rows.is_empty() {
        return Err(format!("{} has no complete rows", col_title(ws, c)).into());
    }
    let cc = rows.iter().map(|&i| counts[i]).collect();
    let ss = rows.iter().map(|&i| sizes[i]).collect();
    Ok((cc, ss, rows, label))
}

fn limit_method(v: &Values) -> LimitMethod {
    if v.choice("limits") == 1 {
        LimitMethod::Exact
    } else {
        LimitMethod::Normal
    }
}

fn attr_title(ws: &Worksheet, v: &Values, name: &str) -> String {
    format!("{name} Chart of {}", v.var("var").map_or_else(String::new, |c| col_title(ws, c)))
}

fn p_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (d, n, rows, _) = attribute_data(ws, v, true)?;
    let o = chart_options_ext(v, Kind::Attribute)?;
    if let Some(st) = stage_labels(ws, v, &rows)? {
        let chart = staged_p(&d, &n, &st, &o)?;
        let mut s = chart.report_section();
        s.heading = attr_title(ws, v, "P") + " by stage";
        return finish(s, v, || panel_cols("", &chart.panel));
    }
    let chart = p_chart_with(&d, &n, &o, limit_method(v))?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "P");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn np_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (d, _, rows, _) = attribute_data(ws, v, false)?;
    let size = v.num("size");
    let o = chart_options_ext(v, Kind::Attribute)?;
    if let Some(st) = stage_labels(ws, v, &rows)? {
        let chart = staged_np(&d, size, &st, &o)?;
        let mut s = chart.report_section();
        s.heading = attr_title(ws, v, "NP") + " by stage";
        return finish(s, v, || panel_cols("", &chart.panel));
    }
    let chart = np_chart_with(&d, size, &o, limit_method(v))?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "NP");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn c_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (d, _, rows, _) = attribute_data(ws, v, false)?;
    let o = chart_options_ext(v, Kind::Attribute)?;
    if let Some(st) = stage_labels(ws, v, &rows)? {
        let chart = staged_c(&d, &st, &o)?;
        let mut s = chart.report_section();
        s.heading = attr_title(ws, v, "C") + " by stage";
        return finish(s, v, || panel_cols("", &chart.panel));
    }
    let chart = c_chart_with(&d, &o, limit_method(v))?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "C");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn u_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (d, n, rows, _) = attribute_data(ws, v, true)?;
    let o = chart_options_ext(v, Kind::Attribute)?;
    if let Some(st) = stage_labels(ws, v, &rows)? {
        let chart = staged_u(&d, &n, &st, &o)?;
        let mut s = chart.report_section();
        s.heading = attr_title(ws, v, "U") + " by stage";
        return finish(s, v, || panel_cols("", &chart.panel));
    }
    let chart = u_chart_with(&d, &n, &o, limit_method(v))?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "U");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn laney_p_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (d, n, _, _) = attribute_data(ws, v, true)?;
    let chart = laney_p(&d, &n, &chart_options_ext(v, Kind::Attribute)?)?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "Laney P'");
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.chart.panel);
        cols.push(Column::numeric("ZSCORE", chart.z.clone()));
        cols
    })
}

fn laney_u_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (d, n, _, _) = attribute_data(ws, v, true)?;
    let chart = laney_u(&d, &n, &chart_options_ext(v, Kind::Attribute)?)?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "Laney U'");
    finish(s, v, || {
        let mut cols = panel_cols("", &chart.chart.panel);
        cols.push(Column::numeric("ZSCORE", chart.z.clone()));
        cols
    })
}

fn g_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let data: Vec<f64> = numeric(ws, c)?.into_iter().filter(|x| !x.is_nan()).collect();
    let chart = g_chart(&data, &chart_options_ext(v, Kind::Attribute)?)?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "G");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn t_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let data: Vec<f64> = numeric(ws, c)?.into_iter().filter(|x| !x.is_nan()).collect();
    let model = if v.choice("model") == 1 { TimeModel::Exponential } else { TimeModel::Weibull };
    let chart = t_chart_with(&data, &chart_options_ext(v, Kind::Attribute)?, model)?;
    let mut s = chart.report_section();
    s.heading = attr_title(ws, v, "T");
    finish(s, v, || panel_cols("", &chart.panel))
}

// ----------------------------------------------------------------------- time-weighted charts

fn tw_input(ws: &Worksheet, v: &Values) -> Res<(TimeWeightedInput, String)> {
    let c = v.var("var").ok_or("select a column")?;
    let target = Some(v.num("target")).filter(|x| x.is_finite());
    let sigma = Some(v.num("sigma_known")).filter(|x| x.is_finite());
    let size = v.count("size").max(1);
    let inp = if size <= 1 && v.var("ids").is_none() {
        let data = numeric(ws, c)?;
        TimeWeightedInput::individuals(&data, target, sigma)?
    } else {
        let (groups, _) = subgroups(ws, v, c)?;
        let method = match v.choice("sigma_est") {
            1 => Some(SubgroupSigma::RBar),
            2 => Some(SubgroupSigma::SBar),
            _ => None,
        };
        TimeWeightedInput::subgroups(&groups, target, sigma, method)?
    };
    Ok((inp, col_title(ws, c)))
}

fn tw_fields(spec: DialogSpec) -> DialogSpec {
    with_subgroup_fields(spec)
        .optional_number(Page::Options, "target", "Target (blank = estimate)")
        .optional_number(Page::Options, "sigma_known", "Historical standard deviation (blank = estimate)")
        .choice(Page::Options, "sigma_est", "Subgroup sigma from", &["Pooled standard deviation", "R-bar", "S-bar"], 0)
        .check(Page::Storage, "store_chart", "Chart values, limits and signals", false)
}

fn ma_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (inp, name) = tw_input(ws, v)?;
    let chart = moving_average(&inp, v.count("length").max(1), v.num("k"))?;
    let mut s = chart.report_section();
    s.heading = format!("Moving Average Chart of {name}");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn ewma_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (inp, name) = tw_input(ws, v)?;
    let chart = ewma(&inp, v.num("lambda_w"), v.num("k"), !v.flag("asymptotic"))?;
    let mut s = chart.report_section();
    s.heading = format!("EWMA Chart of {name}");
    finish(s, v, || panel_cols("", &chart.panel))
}

fn cusum_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (inp, name) = tw_input(ws, v)?;
    if v.choice("form") == 1 {
        let chart = cusum_vmask(&inp, v.num("kk"), v.num("h"))?;
        let mut s = chart.report_section();
        s.heading = format!("CUSUM Chart (V-mask) of {name}");
        return finish(s, v, || vec![Column::numeric("CUSUMSUM", chart.sums.clone())]);
    }
    let sides = match v.choice("sides") {
        1 => CusumSides::UpperOnly,
        2 => CusumSides::LowerOnly,
        _ => CusumSides::Both,
    };
    let o = CusumOptions { k: v.num("kk"), h: v.num("h"), head_start: v.num("head_start") };
    let chart = cusum_tabular_sided(&inp, &o, sides)?;
    let mut s = chart.report_section();
    s.heading = format!("CUSUM Chart of {name}");
    finish(s, v, || {
        let mut flag = vec![0.0; chart.values.len()];
        for i in chart.signals() {
            if i < flag.len() {
                flag[i] = 1.0;
            }
        }
        vec![
            Column::numeric("CUSUMUPPER", chart.upper.clone()),
            Column::numeric("CUSUMLOWER", chart.lower.clone()),
            Column::numeric("CUSUMFLAG", flag),
        ]
    })
}

fn genvar_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars").to_vec();
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let (_, kept) = super::analysis::complete_rows(&refs);
    let n = kept.first().map_or(0, Vec::len);
    let rows: Vec<Vec<f64>> = (0..n).map(|i| kept.iter().map(|c| c[i]).collect()).collect();
    let size = v.count("size");
    if size < 2 {
        return Err("generalized variance needs a subgroup size of at least 2".into());
    }
    let groups: Vec<Vec<Vec<f64>>> = rows.chunks_exact(size).map(<[Vec<f64>]>::to_vec).collect();
    if groups.len() < 2 {
        return Err(format!("{n} complete rows make fewer than two subgroups of {size}").into());
    }
    let chart = generalized_variance(&groups, None, v.num("k"))?;
    let mut s = chart.report_section();
    s.heading = format!(
        "Generalized Variance Chart of {}",
        cols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    );
    finish(s, v, || {
        let mut flag = vec![0.0; chart.det.len()];
        for i in chart.flagged() {
            if i < flag.len() {
                flag[i] = 1.0;
            }
        }
        vec![Column::numeric("GENVAR", chart.det.clone()), Column::numeric("GENVARFLAG", flag)]
    })
}

// ----------------------------------------------------------------------------------- capability

fn specs_of(v: &Values) -> Res<Specs> {
    let specs = Specs {
        lsl: Some(v.num("lsl")).filter(|x| x.is_finite()),
        target: Some(v.num("target")).filter(|x| x.is_finite()),
        usl: Some(v.num("usl")).filter(|x| x.is_finite()),
    };
    if specs.lsl.is_none() && specs.usl.is_none() {
        return Err("enter a lower or an upper specification limit".into());
    }
    Ok(specs)
}

fn spec_fields(spec: DialogSpec) -> DialogSpec {
    spec.optional_number(Page::Main, "lsl", "Lower spec")
        .optional_number(Page::Main, "usl", "Upper spec")
        .optional_number(Page::Main, "target", "Target (optional)")
}

const FAMILIES: [(&str, Option<Family>); 9] = [
    ("Box-Cox transformation", None),
    ("Johnson transformation", None),
    ("Weibull", Some(Family::Weibull)),
    ("Lognormal", Some(Family::Lognormal)),
    ("Gamma", Some(Family::Gamma)),
    ("Exponential", Some(Family::Exponential)),
    ("Logistic", Some(Family::Logistic)),
    ("Loglogistic", Some(Family::Loglogistic)),
    ("Normal", Some(Family::Normal)),
];

fn nonnormal_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let specs = specs_of(v)?;
    let pick = v.choice("method");
    let name = FAMILIES.get(pick).map_or("Box-Cox transformation", |f| f.0);
    let o = CapabilityOptions { level: level(v)?, ..Default::default() };
    let data = numeric(ws, c)?;
    let valid: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let mut s = if let Some(Some(family)) = FAMILIES.get(pick).map(|f| f.1) {
        let thr = Some(v.num("threshold")).filter(|x| x.is_finite());
        capability_distribution(&valid, family, thr, &specs)?.report_section()
    } else {
        let t = if pick == 1 {
            Transformation::Johnson(fit_johnson(&valid)?)
        } else {
            Transformation::BoxCox(BoxCox::fit(&valid, lambda_choice(v))?)
        };
        let size = v.count("size").max(1);
        if size <= 1 {
            capability_transformed_individuals(&data, &t, &specs, &o)?.report_section()
        } else {
            let (groups, _) = subgroups(ws, v, c)?;
            capability_transformed_subgroups(&groups, &t, &specs, &o)?.report_section()
        }
    };
    s.heading = format!("Process Capability of {} ({name})", col_title(ws, c));
    out(s)
}

fn empirical_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let specs = specs_of(v)?;
    let valid: Vec<f64> = numeric(ws, c)?.into_iter().filter(|x| !x.is_nan()).collect();
    let o = EmpiricalOptions { allow_extremes: v.flag("extremes"), ..Default::default() };
    let mut s = capability_empirical(&valid, &specs, &o)?.report_section();
    s.heading = format!("Empirical Process Capability of {}", col_title(ws, c));
    out(s)
}

fn sixpack_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let specs = specs_of(v)?;
    let mut o = SixpackOptions {
        capability: CapabilityOptions { level: level(v)?, ..Default::default() },
        ..Default::default()
    };
    o.chart = chart_options_ext(v, Kind::Subgroup)?;
    o.recent = v.count("recent").max(1);
    let size = v.count("size").max(1);
    let extended = v.flag("extended");
    let mut s = if size <= 1 && v.var("ids").is_none() {
        let data = numeric(ws, c)?;
        let mut oi = o.clone();
        oi.chart = chart_options_ext(v, Kind::Individuals)?;
        if extended {
            extended_sixpack_individuals(&data, &specs, &oi, v.count("block").max(2))?.report_section()
        } else {
            capability_sixpack_individuals(&data, &specs, &oi)?.report_section()
        }
    } else {
        let (groups, _) = subgroups(ws, v, c)?;
        if extended {
            extended_sixpack_subgroups(&groups, &specs, &o)?.report_section()
        } else {
            capability_sixpack_subgroups(&groups, &specs, &o)?.report_section()
        }
    };
    s.heading = format!("Process Capability Sixpack of {}", col_title(ws, c));
    out(s)
}

pub(super) fn capability_intervals_section(
    r: &openstats_stats::spc::capability::NormalCapability,
    groups: Option<&[Vec<f64>]>,
    level: f64,
) -> Res<Section> {
    let df = match groups {
        Some(g) if r.within_method.contains("pooled") => Some(pooled_within_df(g)?),
        _ => None,
    };
    let ci = capability_intervals(r, df, level)?;
    Ok(intervals_section(r, &ci))
}

fn distid_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let thr = Some(v.num("threshold")).filter(|x| x.is_finite());
    let valid: Vec<f64> = numeric(ws, c)?.into_iter().filter(|x| !x.is_nan()).collect();
    let ranks = identify_distributions(&valid, thr)?;
    let mut s = ranking_section(&ranks);
    s.heading = format!("Individual Distribution Identification of {}", col_title(ws, c));
    out(s)
}

// -------------------------------------------------------------------------------------- MSA

fn strs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

fn nested_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (m, p, o) = (
        v.var("measurement").ok_or("select the measurement column")?,
        v.var("part").ok_or("select the part column")?,
        v.var("operator").ok_or("select the operator column")?,
    );
    if m == p || m == o || p == o {
        return Err("measurement, part and operator must be three different columns".into());
    }
    let study = NestedStudy::from_stacked(&numeric(ws, m)?, &labels(ws, p)?, &labels(ws, o)?)?;
    let tol = v.num("tolerance");
    let opts = GageOptions {
        study_sigma: v.num("sigma"),
        tolerance: tol.is_finite().then_some(Tolerance::Width(tol)),
        ..Default::default()
    };
    let mut s = study.anova(&opts)?.report_section();
    s.heading = format!("Gage R&R Study (Nested): {}", col_title(ws, m));
    out(s)
}

fn linearity_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (r, m) = (
        v.var("reference").ok_or("select the reference column")?,
        v.var("measured").ok_or("select the measurement column")?,
    );
    if r == m {
        return Err("reference and measurement must be different columns".into());
    }
    let study = LinearityBiasStudy::new(&numeric(ws, r)?, &numeric(ws, m)?)?;
    let o = LinearityBiasOptions { process_variation: v.num("variation"), level: level(v)? };
    let mut s = study.analyze(&o)?.report_section();
    s.heading = format!("Gage Linearity and Bias Study: {}", col_title(ws, m));
    out(s)
}

fn attribute_agreement(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (sm, ap, rt) = (
        v.var("sample").ok_or("select the sample column")?,
        v.var("appraiser").ok_or("select the appraiser column")?,
        v.var("rating").ok_or("select the rating column")?,
    );
    let (sample, app, rating) = (labels(ws, sm)?, labels(ws, ap)?, labels(ws, rt)?);
    let standard = v.var("standard").map(|c| labels(ws, c)).transpose()?;
    let study = AttributeStudy::from_long(
        &strs(&sample),
        &strs(&app),
        &strs(&rating),
        standard.as_deref().map(strs).as_deref(),
    )?;
    let defect = v.text("defect").trim();
    let o = AttributeOptions { level: level(v)?, defect_category: (!defect.is_empty()).then(|| defect.to_string()) };
    let mut s = study.analyze(&o)?.report_section();
    s.heading = format!("Attribute Agreement Analysis for {}", col_title(ws, rt));
    out(s)
}

// ------------------------------------------------------------------------------ registration

/// Register the quality-tool dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options};
    use VarKind::{Any, Numeric};
    let _ = Graphs;
    let level_field = |s: DialogSpec| s.number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999);
    let cc = |last: &'static str| ["Stat", "Control Charts", last];

    add(
        r,
        ctrl_full(
            DialogSpec::new(
                "stat.spc.between_within",
                "I-MR-R/S (Between/Within) Chart",
                "Separates variation between subgroups from variation within them.",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .choice(Main, "within", "Within-subgroup chart", &["R (range)", "S (standard deviation)"], 0),
            Kind::Subgroup,
        )
        .integer(Main, "size", "Subgroup size", 5.0, 2.0, 1000.0)
        .vars("ids", "Subgroup identifiers (column; overrides the size)", Any, 0, 1),
        &["Stat", "Control Charts", "Variables Charts for Subgroups", "I-MR-R/S (Between/Within)..."],
        Runner::Report(between_within),
    );
    add(
        r,
        with_subgroup_fields(ctrl_full(
            DialogSpec::new(
                "stat.spc.zone",
                "Zone Chart",
                "Cumulative zone scores: 0, 2, 4 and 8 points for the zones either side of the center line.",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .number(Main, "signal_at", "Signal when the score reaches", 8.0, 1.0, 100.0),
            Kind::Subgroup,
        )),
        &["Stat", "Control Charts", "Variables Charts for Subgroups", "Zone..."],
        Runner::Report(zone_run),
    );
    add(
        r,
        ctrl_full(
            DialogSpec::new(
                "stat.spc.zmr",
                "Z-MR Chart",
                "Short-run individuals chart: observations are standardised within each part.",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .vars("parts", "Part numbers", Any, 1, 1)
            .choice(
                Main,
                "sigma_by",
                "Standard deviation",
                &["Pooled over all parts", "Separate for each part"],
                0,
            ),
            Kind::Individuals,
        ),
        &["Stat", "Control Charts", "Variables Charts for Individuals", "Z-MR..."],
        Runner::Report(zmr_run),
    );
    add(
        r,
        with_subgroup_fields(ctrl_full(
            DialogSpec::new(
                "stat.spc.boxcox",
                "Box-Cox Transformation Chart",
                "Control chart of Box-Cox transformed data, with limits shown in the original units.",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .optional_number(Options, "lambda", "Lambda (blank = optimal)")
            .check(Options, "round_lambda", "Round the optimal lambda", false)
            .choice(Options, "sigma", "Subgroup dispersion chart", &["R", "S"], 0),
            Kind::Subgroup,
        )),
        &cc("Box-Cox Transformation..."),
        Runner::Report(boxcox_chart),
    );

    let counts = |id: &str, title: &str, help: &str, label: &str, size_label: Option<&str>| {
        let mut s = DialogSpec::new(id, title, help).vars("var", label, Numeric, 1, 1);
        if let Some(l) = size_label {
            s = s.vars("sizes", "Subgroup sizes (column)", Numeric, 0, 1).number(Main, "size", l, 50.0, 1e-9, 1.0e12);
        }
        s
    };
    let method = |s: DialogSpec| s.choice(Options, "limits", "Control limits", &["Normal approximation", "Exact"], 0);
    let attr = ["Stat", "Control Charts", "Attributes Charts", ""];
    let at = |last: &'static str| {
        let mut p = attr;
        p[3] = last;
        p
    };
    add(
        r,
        method(ctrl_full(
            counts(
                "stat.spc.p",
                "P Chart",
                "Proportion defective per subgroup.",
                "Number of defectives",
                Some("Subgroup size (constant)"),
            ),
            Kind::Attribute,
        )),
        &at("P..."),
        Runner::Report(p_run),
    );
    add(
        r,
        method(ctrl_full(
            counts(
                "stat.spc.np",
                "NP Chart",
                "Number defective per subgroup of a constant size.",
                "Number of defectives",
                Some("Subgroup size (constant)"),
            ),
            Kind::Attribute,
        )),
        &at("NP..."),
        Runner::Report(np_run),
    );
    add(
        r,
        method(ctrl_full(
            counts("stat.spc.c", "C Chart", "Number of defects per unit.", "Number of defects", None),
            Kind::Attribute,
        )),
        &at("C..."),
        Runner::Report(c_run),
    );
    add(
        r,
        method(ctrl_full(
            counts(
                "stat.spc.u",
                "U Chart",
                "Defects per unit of varying size.",
                "Number of defects",
                Some("Subgroup size (constant)"),
            ),
            Kind::Attribute,
        )),
        &at("U..."),
        Runner::Report(u_run),
    );
    add(
        r,
        ctrl_full(
            counts(
                "stat.spc.laney_p",
                "Laney P' Chart",
                "P chart adjusted for overdispersion.",
                "Number of defectives",
                Some("Subgroup size (constant)"),
            ),
            Kind::Attribute,
        ),
        &at("Laney P'..."),
        Runner::Report(laney_p_run),
    );
    add(
        r,
        ctrl_full(
            counts(
                "stat.spc.laney_u",
                "Laney U' Chart",
                "U chart adjusted for overdispersion.",
                "Number of defects",
                Some("Subgroup size (constant)"),
            ),
            Kind::Attribute,
        ),
        &at("Laney U'..."),
        Runner::Report(laney_u_run),
    );
    add(
        r,
        ctrl_full(
            DialogSpec::new("stat.spc.g", "G Chart", "Number of opportunities (or units) between rare events.").vars(
                "var",
                "Counts between events",
                Numeric,
                1,
                1,
            ),
            Kind::Attribute,
        ),
        &["Stat", "Control Charts", "Rare Event Charts", "G..."],
        Runner::Report(g_run),
    );
    add(
        r,
        ctrl_full(
            DialogSpec::new("stat.spc.t", "T Chart", "Time between rare events.")
                .vars("var", "Times between events", Numeric, 1, 1)
                .choice(Options, "model", "Distribution", &["Weibull", "Exponential"], 0),
            Kind::Attribute,
        ),
        &["Stat", "Control Charts", "Rare Event Charts", "T..."],
        Runner::Report(t_run),
    );

    let tw = ["Stat", "Control Charts", "Time-Weighted Charts", ""];
    let twp = |last: &'static str| {
        let mut p = tw;
        p[3] = last;
        p
    };
    add(
        r,
        tw_fields(
            DialogSpec::new(
                "stat.spc.ma",
                "Moving Average Chart",
                "Moving average of the last observations or subgroup means.",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .integer(Main, "length", "Length of the moving average", 3.0, 1.0, 1000.0)
            .number(Options, "k", "Sigma limit position", 3.0, 0.5, 10.0),
        ),
        &twp("Moving Average..."),
        Runner::Report(ma_run),
    );
    add(
        r,
        tw_fields(
            DialogSpec::new("stat.spc.ewma", "EWMA Chart", "Exponentially weighted moving average.")
                .vars("var", "Observations", Numeric, 1, 1)
                .number(Main, "lambda_w", "Weight (lambda)", 0.2, 1e-6, 1.0)
                .number(Options, "k", "Sigma limit position", 3.0, 0.5, 10.0)
                .check(Options, "asymptotic", "Use asymptotic limits", false),
        ),
        &twp("EWMA..."),
        Runner::Report(ewma_run),
    );
    add(
        r,
        tw_fields(
            DialogSpec::new("stat.spc.cusum", "CUSUM Chart", "Tabular or V-mask cumulative sum chart.")
                .vars("var", "Observations", Numeric, 1, 1)
                .choice(Main, "form", "Form", &["Tabular", "V-mask"], 0)
                .number(Options, "kk", "Allowance K (standard errors)", 0.5, 1e-6, 10.0)
                .number(Options, "h", "Decision interval H (standard errors)", 4.0, 1e-6, 100.0)
                .number(Options, "head_start", "Head start (fast initial response)", 0.0, 0.0, 100.0)
                .choice(Options, "sides", "Sides", &["Both", "Upper only", "Lower only"], 0),
        ),
        &twp("CUSUM..."),
        Runner::Report(cusum_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.spc.genvar",
            "Generalized Variance Chart",
            "Determinant of each subgroup's covariance matrix, for several related variables.",
        )
        .vars("vars", "Variables", Numeric, 2, 20)
        .integer(Main, "size", "Subgroup size", 5.0, 2.0, 1000.0)
        .number(Options, "k", "Sigma limit position", 3.0, 0.5, 10.0)
        .check(Page::Storage, "store_chart", "Determinants and flagged points", false),
        &cc("Generalized Variance..."),
        Runner::Report(genvar_run),
    );

    // ------------------------------------------------------------------ capability and MSA
    add(
        r,
        with_subgroup_fields(level_field(spec_fields(
            DialogSpec::new(
                "stat.quality.capability_nonnormal",
                "Capability Analysis (Nonnormal)",
                "Capability from a transformation (Box-Cox, Johnson) or a fitted non-normal distribution.",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .choice(Main, "method", "Fit", &FAMILIES.map(|f| f.0), 0)
            .optional_number(Options, "threshold", "Threshold (distributions with a threshold; blank = 0)")
            .optional_number(Options, "lambda", "Box-Cox lambda (blank = optimal)")
            .check(Options, "round_lambda", "Round the optimal lambda", false),
        ))),
        &["Stat", "Quality Tools", "Capability Analysis", "Nonnormal..."],
        Runner::Report(nonnormal_run),
    );
    add(
        r,
        spec_fields(
            DialogSpec::new(
                "stat.quality.capability_empirical",
                "Capability Analysis (Empirical)",
                "Pp and Ppk from the sample's own 0.135% and 99.865% percentiles (needs a large sample).",
            )
            .vars("var", "Observations", Numeric, 1, 1)
            .check(Options, "extremes", "Use the sample extremes when the sample is too small", false),
        ),
        &["Stat", "Quality Tools", "Capability Analysis", "Empirical..."],
        Runner::Report(empirical_run),
    );
    add(
        r,
        with_subgroup_fields(level_field(ctrl_full(
            spec_fields(
                DialogSpec::new(
                    "stat.quality.sixpack",
                    "Capability Sixpack",
                    "Control chart, capability histogram, probability plot and capability plot on one page.",
                )
                .vars("var", "Observations", Numeric, 1, 1)
                .integer(Options, "recent", "Last observations or subgroups shown", 25.0, 5.0, 1000.0)
                .check(Options, "extended", "Add capability by subgroup and running average", false)
                .integer(Options, "block", "Block size for individuals (extended)", 5.0, 2.0, 1000.0),
            ),
            Kind::Subgroup,
        ))),
        &["Stat", "Quality Tools", "Capability Sixpack", "Normal..."],
        Runner::Report(sixpack_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.quality.distid",
            "Individual Distribution Identification",
            "Fit candidate distributions and rank them by Anderson-Darling goodness of fit.",
        )
        .vars("var", "Observations", Numeric, 1, 1)
        .optional_number(Options, "threshold", "Threshold (blank = 0)"),
        &["Stat", "Quality Tools", "Individual Distribution Identification..."],
        Runner::Report(distid_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.quality.gage_nested",
            "Gage R&R Study (Nested)",
            "Repeatability and reproducibility when each operator measures different parts.",
        )
        .vars("measurement", "Measurement data", Numeric, 1, 1)
        .vars("part", "Part numbers", Any, 1, 1)
        .vars("operator", "Operators", Any, 1, 1)
        .number(Options, "sigma", "Study variation (number of standard deviations)", 6.0, 1.0, 10.0)
        .optional_number(Options, "tolerance", "Process tolerance (USL - LSL)"),
        &["Stat", "Quality Tools", "Gage Study", "Gage R&R Study (Nested)..."],
        Runner::Report(nested_run),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.quality.gage_linearity",
                "Gage Linearity and Bias Study",
                "Bias of a gage across its range from measurements of parts with known reference values.",
            )
            .vars("reference", "Reference values", Numeric, 1, 1)
            .vars("measured", "Measurements", Numeric, 1, 1)
            .number(Main, "variation", "Process variation", 6.0, 1e-12, 1e12),
        ),
        &["Stat", "Quality Tools", "Gage Study", "Gage Linearity and Bias Study..."],
        Runner::Report(linearity_run),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.quality.attribute_agreement",
                "Attribute Agreement Analysis",
                "Agreement within and between appraisers, and with a known standard (kappa statistics).",
            )
            .vars("sample", "Samples", Any, 1, 1)
            .vars("appraiser", "Appraisers", Any, 1, 1)
            .vars("rating", "Ratings", Any, 1, 1)
            .vars("standard", "Known standard (optional)", Any, 0, 1)
            .text(Options, "defect", "Defect category for miss and false-alarm rates (optional)", ""),
        ),
        &["Stat", "Quality Tools", "Attribute Agreement Analysis..."],
        Runner::Report(attribute_agreement),
    );
}
