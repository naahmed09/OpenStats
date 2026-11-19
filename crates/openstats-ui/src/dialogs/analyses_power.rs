//! Stat ▸ Power and Sample Size. One dialog per design of `openstats_stats::power`. Every dialog has
//! three boxes (sample sizes, power values, effects); exactly one is left blank and is solved for.
//! The other two may hold several values (separated by spaces or commas), in which case every
//! combination is solved and the results come back in one table with one power curve per sample
//! size.

use super::analyses::{add, out};
use super::analysis::{DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values};
use openstats_stats::hypothesis::Alternative;
use openstats_stats::power::{
    Direction, DiscreteMethod, Factorial, FactorialDesign, OnePoisson, OneProportion, OneSampleT, OneSampleZ,
    OneVariance, OneWayAnova, PairedT, PowerModel, PowerSolution, Tost, TostDesign, TwoPoisson, TwoProportions,
    TwoSampleT, TwoVariances,
};
use openstats_stats::report::{
    Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Section, Series, Table,
};
use openstats_stats::reportable::power_section;

const ALTS: [&str; 3] = ["not equal", "less than", "greater than"];
const METHODS: [&str; 2] = ["Normal approximation", "Exact"];
const DIRECTIONS: [&str; 2] = ["Above the null value", "Below the null value"];
/// The largest number of combinations a single run will solve.
const MAX_COMBINATIONS: usize = 400;
/// The largest number of power curves drawn.
const MAX_CURVES: usize = 8;

fn alt(v: &Values) -> Alternative {
    match v.choice("alt") {
        1 => Alternative::Less,
        2 => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

fn method(v: &Values) -> DiscreteMethod {
    if v.choice("method") == 1 {
        DiscreteMethod::Exact
    } else {
        DiscreteMethod::NormalApproximation
    }
}

fn direction(v: &Values) -> Direction {
    if v.choice("direction") == 1 {
        Direction::Decrease
    } else {
        Direction::Increase
    }
}

fn alpha(v: &Values) -> f64 {
    v.num("alpha")
}

/// Values in a box: numbers separated by spaces, commas or semicolons.
fn parse_list(text: &str, what: &str) -> Res<Vec<f64>> {
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

fn sample_sizes(text: &str) -> Res<Vec<usize>> {
    parse_list(text, "sample sizes")?
        .into_iter()
        .map(|x| {
            if x >= 1.0 && x.fract() == 0.0 && x <= 1.0e9 {
                Ok(x as usize)
            } else {
                Err(DialogError(format!("sample size {x} must be a whole number of at least 1")))
            }
        })
        .collect()
}

fn powers(text: &str) -> Res<Vec<f64>> {
    let p = parse_list(text, "power values")?;
    if let Some(bad) = p.iter().find(|&&x| !(x > 0.0 && x < 1.0)) {
        return Err(format!("power {bad} must be between 0 and 1").into());
    }
    Ok(p)
}

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

/// Solve the problem described by the three boxes and report it.
fn solve<M: PowerModel>(m: &M, v: &Values) -> Res<Output> {
    let ns = sample_sizes(v.text("ns"))?;
    let ps = powers(v.text("powers"))?;
    let es = parse_list(v.text("effects"), "effects")?;
    let blank = [ns.is_empty(), ps.is_empty(), es.is_empty()];
    if blank.iter().filter(|&&b| b).count() != 1 {
        return Err(
            "leave exactly one of the sample size, power and effect boxes blank: it is the value that is solved for"
                .into(),
        );
    }
    let mut sols: Vec<PowerSolution> = Vec::new();
    let combos = [ns.len().max(1), ps.len().max(1), es.len().max(1)].iter().product::<usize>();
    if combos > MAX_COMBINATIONS {
        return Err(format!("{combos} combinations is too many; at most {MAX_COMBINATIONS} are solved at once").into());
    }
    let pick = |len: usize| if len == 0 { vec![None] } else { (0..len).map(Some).collect() };
    for i in pick(ns.len()) {
        for j in pick(ps.len()) {
            for k in pick(es.len()) {
                let sol = m.solve(i.map(|i| ns[i]), j.map(|j| ps[j]), k.map(|k| es[k]))?;
                sols.push(sol);
            }
        }
    }
    if sols.len() == 1 {
        return out(power_section(m, &sols[0]));
    }
    out(multi_section(m, &sols)?)
}

fn multi_section<M: PowerModel>(m: &M, sols: &[PowerSolution]) -> Res<Section> {
    let mut s = Section::new(format!("Power and Sample Size: {}", m.name()));
    let mut p =
        Table::new("Method", vec![Column::new("Parameter", ColumnKind::Text), Column::new("Value", ColumnKind::Text)]);
    for (k, val) in m.parameters() {
        p.push_row(vec![Cell::text(k), Cell::text(val)]).map_err(DialogError::from)?;
    }
    s.table(p);
    let mut r = Table::new(
        "Results",
        vec![
            Column::new(m.effect_label(), ColumnKind::Significant(6)),
            Column::new(m.sample_size_label(), ColumnKind::Integer),
            Column::new("Target Power", ColumnKind::Fixed(4)),
            Column::new("Actual Power", ColumnKind::Fixed(4)),
        ],
    );
    for sol in sols {
        r.push_row(vec![
            Cell::Num(sol.effect),
            int(sol.sample_size),
            Cell::opt(sol.target_power),
            Cell::Num(sol.power),
        ])
        .map_err(DialogError::from)?;
    }
    r.footnotes.push(
        "A solved sample size is the smallest whose power reaches the target; the actual power is the power at that sample size."
            .into(),
    );
    s.table(r);

    // One power curve per distinct sample size, over the effects solved or given.
    let (lo, _) = m.effect_range();
    let far = sols.iter().map(|x| x.effect).fold(lo, |a, b| if (b - lo).abs() > (a - lo).abs() { b } else { a });
    let span = far - lo;
    if span != 0.0 && span.is_finite() {
        let xs: Vec<f64> = (0..=40).map(|i| lo + span * 1.6 * f64::from(i) / 40.0).collect();
        let mut sizes: Vec<usize> = sols.iter().map(|x| x.sample_size).collect();
        sizes.sort_unstable();
        sizes.dedup();
        let series: Vec<Series> = sizes
            .iter()
            .take(MAX_CURVES)
            .filter_map(|&n| {
                m.power_curve(n, &xs).ok().map(|ys| Series {
                    name: format!("{} = {n}", m.sample_size_label()),
                    x: xs.clone(),
                    y: ys,
                })
            })
            .collect();
        if !series.is_empty() {
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: format!("Power Curve for {}", m.name()),
                x_label: m.effect_label().into(),
                y_label: "Power".into(),
                categories: Vec::new(),
                series,
                reference_lines: vec![ReferenceLine { axis: Axis::Y, value: m.alpha(), label: "alpha".into() }],
            });
        }
        if sizes.len() > MAX_CURVES {
            s.warn(format!("Only the first {MAX_CURVES} sample sizes are drawn on the power curve."));
        }
    }
    Ok(s)
}

// ----------------------------------------------------------------------------------- runners

fn z1(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&OneSampleZ { sigma: v.num("sigma"), alpha: alpha(v), alternative: alt(v) }, v)
}

fn t1(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&OneSampleT { sigma: v.num("sigma"), alpha: alpha(v), alternative: alt(v) }, v)
}

fn paired(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&PairedT { sigma_diff: v.num("sigma"), alpha: alpha(v), alternative: alt(v) }, v)
}

fn t2(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&TwoSampleT { sigma: v.num("sigma"), alpha: alpha(v), alternative: alt(v) }, v)
}

fn prop1(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    let mut m = OneProportion::new(v.num("p0"), alpha(v), alt(v), method(v));
    m.direction = direction(v);
    solve(&m, v)
}

fn prop2(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    let mut m = TwoProportions::new(v.num("p2"), alpha(v), alt(v), method(v));
    m.direction = direction(v);
    solve(&m, v)
}

fn poisson1(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    let mut m = OnePoisson::new(v.num("rate0"), alpha(v), alt(v), method(v));
    m.length = v.num("length");
    m.direction = direction(v);
    solve(&m, v)
}

fn poisson2(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    let mut m = TwoPoisson::new(v.num("rate2"), alpha(v), alt(v), method(v));
    m.length = v.num("length");
    m.direction = direction(v);
    solve(&m, v)
}

fn var1(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&OneVariance { alpha: alpha(v), alternative: alt(v), direction: direction(v) }, v)
}

fn var2(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&TwoVariances { alpha: alpha(v), alternative: alt(v), direction: direction(v) }, v)
}

fn anova(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    solve(&OneWayAnova { groups: v.count("groups"), sigma: v.num("sigma"), alpha: alpha(v) }, v)
}

fn factorial(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    let factors = v.count("factors");
    let design = match v.choice("design") {
        0 => FactorialDesign::full(factors)?,
        1 => FactorialDesign::fractional(factors, v.count("generators"))?,
        _ => FactorialDesign::plackett_burman(v.count("runs"), factors)?,
    };
    let mut m = Factorial::new(design, v.num("sigma"), alpha(v));
    m.center_points = v.count("center");
    m.blocks = v.count("blocks").max(1);
    m.omitted_terms = v.count("omitted");
    solve(&m, v)
}

fn tost(_: &openstats_core::Worksheet, v: &Values) -> Res<Output> {
    let design = match v.choice("design") {
        1 => TostDesign::Paired,
        2 => TostDesign::TwoSample,
        _ => TostDesign::OneSample,
    };
    let mut m = Tost::new(design, v.num("lower"), v.num("upper"), v.num("sigma"), alpha(v));
    m.direction = direction(v);
    solve(&m, v)
}

// ------------------------------------------------------------------------------- registration

/// The three boxes every power dialog has, with the label of the effect.
fn boxes(spec: DialogSpec, n_label: &str, effect_label: &str) -> DialogSpec {
    spec.text(Page::Main, "ns", n_label, "").text(Page::Main, "powers", "Power values", "0.9").text(
        Page::Main,
        "effects",
        effect_label,
        "1",
    )
}

fn with_alpha(spec: DialogSpec) -> DialogSpec {
    spec.number(Page::Options, "alpha", "Significance level", 0.05, 1e-9, 0.5)
}

fn with_alt(spec: DialogSpec) -> DialogSpec {
    spec.choice(Page::Options, "alt", "Alternative hypothesis", &ALTS, 0)
}

fn with_direction(spec: DialogSpec) -> DialogSpec {
    spec.choice(Page::Options, "direction", "When solving for the effect, search", &DIRECTIONS, 0)
}

fn with_method(spec: DialogSpec) -> DialogSpec {
    spec.choice(Page::Options, "method", "Power of the", &METHODS, 0)
}

/// Register the power and sample-size dialogs.
pub(crate) fn register(r: &mut Registry) {
    let path = |last: &'static str| ["Stat", "Power and Sample Size", last];
    let help = "Leave one of the three boxes blank; it is solved for. The other two may hold several values.";

    add(
        r,
        with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.z1", "Power and Sample Size for 1-Sample Z", help),
                "Sample sizes",
                "Differences (mean − null value)",
            )
            .number(Page::Main, "sigma", "Standard deviation (known)", 1.0, 1e-12, 1e12),
        )),
        &path("1-Sample Z..."),
        Runner::Report(z1),
    );
    add(
        r,
        with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.t1", "Power and Sample Size for 1-Sample t", help),
                "Sample sizes",
                "Differences (mean − null value)",
            )
            .number(Page::Main, "sigma", "Standard deviation", 1.0, 1e-12, 1e12),
        )),
        &path("1-Sample t..."),
        Runner::Report(t1),
    );
    add(
        r,
        with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.paired", "Power and Sample Size for Paired t", help),
                "Sample sizes (pairs)",
                "Differences (mean of the paired differences)",
            )
            .number(Page::Main, "sigma", "Standard deviation of the differences", 1.0, 1e-12, 1e12),
        )),
        &path("Paired t..."),
        Runner::Report(paired),
    );
    add(
        r,
        with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.t2", "Power and Sample Size for 2-Sample t", help),
                "Sample sizes (per group)",
                "Differences (mean 1 − mean 2)",
            )
            .number(Page::Main, "sigma", "Standard deviation", 1.0, 1e-12, 1e12),
        )),
        &path("2-Sample t..."),
        Runner::Report(t2),
    );
    add(
        r,
        with_direction(with_method(with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.prop1", "Power and Sample Size for 1 Proportion", help),
                "Sample sizes",
                "Alternative values of p",
            )
            .number(Page::Main, "p0", "Hypothesized p", 0.5, 1e-9, 1.0 - 1e-9),
        )))),
        &path("1 Proportion..."),
        Runner::Report(prop1),
    );
    add(
        r,
        with_direction(with_method(with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.prop2", "Power and Sample Size for 2 Proportions", help),
                "Sample sizes (per group)",
                "Proportion 1 values",
            )
            .number(Page::Main, "p2", "Proportion 2 (baseline)", 0.5, 1e-9, 1.0 - 1e-9),
        )))),
        &path("2 Proportions..."),
        Runner::Report(prop2),
    );
    add(
        r,
        with_direction(with_method(with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.poisson1", "Power and Sample Size for 1-Sample Poisson Rate", help),
                "Sample sizes (observation units)",
                "Alternative rates",
            )
            .number(Page::Main, "rate0", "Hypothesized rate", 1.0, 1e-12, 1e12)
            .number(Page::Main, "length", "Length of each observation unit", 1.0, 1e-12, 1e12),
        )))),
        &path("1-Sample Poisson Rate..."),
        Runner::Report(poisson1),
    );
    add(
        r,
        with_direction(with_method(with_alt(with_alpha(
            boxes(
                DialogSpec::new("stat.power.poisson2", "Power and Sample Size for 2-Sample Poisson Rate", help),
                "Sample sizes (observation units per group)",
                "Rate 1 values",
            )
            .number(Page::Main, "rate2", "Rate 2 (baseline)", 1.0, 1e-12, 1e12)
            .number(Page::Main, "length", "Length of each observation unit", 1.0, 1e-12, 1e12),
        )))),
        &path("2-Sample Poisson Rate..."),
        Runner::Report(poisson2),
    );
    add(
        r,
        with_direction(with_alt(with_alpha(boxes(
            DialogSpec::new("stat.power.var1", "Power and Sample Size for 1 Variance", help),
            "Sample sizes",
            "Variance ratios (true variance / hypothesized variance)",
        )))),
        &path("1 Variance..."),
        Runner::Report(var1),
    );
    add(
        r,
        with_direction(with_alt(with_alpha(boxes(
            DialogSpec::new("stat.power.var2", "Power and Sample Size for 2 Variances", help),
            "Sample sizes (per group)",
            "Variance ratios (variance 1 / variance 2)",
        )))),
        &path("2 Variances..."),
        Runner::Report(var2),
    );
    add(
        r,
        with_alpha(
            boxes(
                DialogSpec::new("stat.power.anova", "Power and Sample Size for One-Way ANOVA", help),
                "Sample sizes (per group)",
                "Maximum differences between means",
            )
            .integer(Page::Main, "groups", "Number of levels", 3.0, 2.0, 1000.0)
            .number(Page::Main, "sigma", "Standard deviation", 1.0, 1e-12, 1e12),
        ),
        &path("One-Way ANOVA..."),
        Runner::Report(anova),
    );
    add(
        r,
        with_alpha(
            boxes(
                DialogSpec::new("stat.power.factorial", "Power and Sample Size for 2-Level Factorial Design", help),
                "Numbers of replicates",
                "Effects (mean at high level − mean at low level)",
            )
            .choice(Page::Main, "design", "Design", &["Full factorial", "Fractional factorial", "Plackett-Burman"], 0)
            .integer(Page::Main, "factors", "Number of factors", 3.0, 2.0, 99.0)
            .integer(Page::Main, "generators", "Number of generators (fractional)", 1.0, 1.0, 10.0)
            .integer(Page::Main, "runs", "Number of runs (Plackett-Burman)", 12.0, 8.0, 100.0)
            .number(Page::Main, "sigma", "Standard deviation", 1.0, 1e-12, 1e12)
            .integer(Page::Options, "center", "Number of center points", 0.0, 0.0, 1000.0)
            .integer(Page::Options, "blocks", "Number of blocks", 1.0, 1.0, 100.0)
            .integer(Page::Options, "omitted", "Terms omitted from the model", 0.0, 0.0, 1000.0),
        ),
        &path("2-Level Factorial Design..."),
        Runner::Report(factorial),
    );
    add(
        r,
        with_direction(with_alpha(
            boxes(
                DialogSpec::new("stat.power.tost", "Power and Sample Size for Equivalence Test (TOST)", help),
                "Sample sizes (per group)",
                "Differences (true mean difference)",
            )
            .choice(Page::Main, "design", "Design", &["1-sample", "Paired", "2-sample"], 2)
            .number(Page::Main, "lower", "Lower equivalence limit", -1.0, -1e12, 1e12)
            .number(Page::Main, "upper", "Upper equivalence limit", 1.0, -1e12, 1e12)
            .number(Page::Main, "sigma", "Standard deviation", 1.0, 1e-12, 1e12),
        )),
        &path("Equivalence Test (TOST)..."),
        Runner::Report(tost),
    );
}
